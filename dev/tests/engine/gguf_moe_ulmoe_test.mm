// gguf-moe-ulmoe: a ULMoE W2 block (routed experts in the ULMoE planes, ops::UlmoeMoeExperts;
// shared expert and router as GGUF) through ops::MoE, on synthetic planes (random code words and meta words, fixed or
// random fp16 anchors; the byte-major code layout the loader restages to), decode at 1 and 4 lanes and an 8-row-tile
// prefill chunk, on the int8 path (--moe-int8) and the bf16 path:
// - int8, against an integer CPU reference fed the GPU's own int8 rows: the down outputs of every routed tile must be
//   bitwise equal (exact int32 sums, then float(tot) x anchor x row scale in fp32 as the kernels do), on the NAX tile
//   and the <= 2-row GEMV alike; the gather's int8 rows and the swiglu's int8 intermediate (CPU silu with std::exp)
//   are counted against a CPU recomputation (reported; the kernels' silu and division are not the CPU's);
// - bf16: down outputs against fp64 sums over the GPU's bf16 intermediate, within 2 bf16 ulps of the row's largest;
// - both: the block output against an fp64 reference of the block (ULMoE weights decoded from the format definition,
//   bf16 inputs, the shared expert's reference values, the GPU's routing), as relative RMS error of the experts'
//   contribution, bounded by 3e-2 (int8) and 1e-2 (bf16).
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/MoE.hpp"
#include "GgufFormatReference.hpp"
#include "MoeFixture.hpp"
#include "metal/abi/MoE.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::ops::MoE;
using splash::ops::MoeConfig;
using splash::ops::MoeExpertTile;
using splash::ops::MoeGgufTile;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using namespace gguf_reference;
using namespace moe_fixture;

constexpr uint32_t kResidual = 128;

// One ULMoE projection's planes on the host: N columns over K inputs + kResidual.
struct Planes {
  uint32_t N = 0, K = 0, G = 0, E = 0;
  std::vector<uint32_t> codes;   // [E][G][N][4]
  std::vector<uint32_t> meta;    // [E][G][N]
  std::vector<float> anchors;    // [N]
  // (q - z_j) s_j of input kk (0..63) of macro group g, column n of expert e, and the macro scale m.
  int qs(uint32_t e, uint32_t g, uint32_t n, uint32_t kk) const {
    const uint64_t at = (uint64_t(e) * G + g) * N + n;
    const uint32_t word = codes[at * 4 + kk / 16], mt = meta[at];
    const uint32_t i = kk % 16, c = i >> 2, b = i & 3u, j = kk / 8;
    const int q = int((word >> (8 * b + 2 * c)) & 3u);
    const int s = int(2u + 2u * (mt & 1u) + ((mt >> (16 + 2 * j)) & 3u)), z = int((mt >> (8 + j)) & 1u) + 1;
    return (q - z) * s;
  }
  int m(uint32_t e, uint32_t g, uint32_t n) const { return int((meta[(uint64_t(e) * G + g) * N + n] >> 1) & 0x7Fu); }
  // The stored input of macro-group input k: inputs past K repeat the first kResidual.
  uint32_t input(uint32_t g, uint32_t kk) const { const uint32_t k = g * 64 + kk; return k < K ? k : k - K; }
};

Planes makePlanes(uint32_t E, uint32_t N, uint32_t K, bool fixed) {
  Planes p;
  p.E = E; p.N = N; p.K = K; p.G = (K + kResidual) / 64;
  const uint64_t cells = uint64_t(E) * p.G * N;
  p.codes.resize(cells * 4);
  p.meta.resize(cells);
  for (uint32_t &w : p.codes) w = uint32_t(rng());
  for (uint32_t &w : p.meta) w = uint32_t(rng());
  p.anchors.resize(N);
  std::normal_distribution<float> normal(0.0f, 1.0f / 4096.0f);
  for (float &a : p.anchors) a = fixed ? 1.0f / 4096.0f : float(_Float16(normal(rng)));
  return p;
}

splash::ops::UlmoeExpertProjection upload(MetalBackend &backend, const Planes &p) {
  return {moe_fixture::upload(backend, p.codes.data(), p.codes.size() * 4, "ulmoe-codes"),
          moe_fixture::upload(backend, p.meta.data(), p.meta.size() * 4, "ulmoe-meta"),
          moe_fixture::upload(backend, p.anchors.data(), p.anchors.size() * 4, "ulmoe-anchors"), kResidual};
}

struct Block {
  MoeShape shape;
  Tensor router, sharedGate;
  std::array<Tensor, 3> shared;
  Planes gateUp, down;
  std::vector<uint32_t> order;
  splash::ops::BlockMoeWeights weights;
};

Block makeBlock(MetalBackend &backend, MoeShape shape, std::array<Fmt, 3> sharedFormats, bool fixed) {
  Block b;
  const uint32_t H = shape.hiddenSize, I = shape.expertIntermediateSize;
  shape.expertFormat = QUANT_FMT_ULMOE_W2;
  b.shape = shape;
  b.router = floating(backend, shape.experts, H, 0.05f);
  if (shape.sharedGate) b.sharedGate = floating(backend, 1, H, 0.05f);
  const uint32_t n[3] = {I, I, H}, k[3] = {H, H, I};
  for (int p = 0; p < 3; ++p) b.shared[p] = quantized(backend, sharedFormats[p], n[p], k[p]);
  b.gateUp = makePlanes(shape.experts, 2 * I, H, fixed);
  b.down = makePlanes(shape.experts, H, I, fixed);
  b.order.resize(H);
  for (uint32_t i = 0; i < H; ++i) b.order[i] = i;
  std::shuffle(b.order.begin(), b.order.end(), rng);
  splash::ops::BlockMoeWeights w;
  w.router = b.router.segment;
  w.sharedScalarGate = b.sharedGate.segment;
  w.gate.shared = b.shared[0].segment;
  w.up.shared = b.shared[1].segment;
  w.down.shared = b.shared[2].segment;
  w.slotTable = splash::model::residentExpertSlots(backend, shape.experts);
  w.slots = shape.experts;
  w.ulmoe.gateUp = upload(backend, b.gateUp);
  w.ulmoe.down = upload(backend, b.down);
  w.ulmoe.inputOrder = moe_fixture::upload(backend, b.order.data(), b.order.size() * 4, "ulmoe-order");
  b.weights = w;
  return b;
}

float bf(float v) { return float(__bf16(v)); }
float bfBits(uint16_t bits) { const uint32_t v = uint32_t(bits) << 16; float f; std::memcpy(&f, &v, 4); return f; }
int8_t quant(float v, float scale) { return int8_t(std::clamp(std::rint(v / scale), -127.0f, 127.0f)); }

struct Result {
  uint64_t gatherCodes = 0, gatherDiffer = 0, interCodes = 0, interDiffer = 0, interBeyond1 = 0;
  uint64_t downCompared = 0, downDiffer = 0, downBeyond = 0, naxTiles = 0, gemvTiles = 0;
  double err2 = 0, ref2 = 0;
};

Result run(MetalBackend &backend, const Block &blk, Buffers &b, const MoePlan &plan, bool int8) {
  allocate(backend, b, plan);
  std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
  CommandGraph graph;
  MoE::add(graph, b.moe, blk.weights, plan);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  Result r;
  const MoeShape &shape = blk.shape;
  const uint32_t rows = plan.rows(), H = shape.hiddenSize, I = shape.expertIntermediateSize, E = shape.experts;
  const uint32_t routes = shape.routesPerToken();
  const auto &scratch = b.moe.scratch;
  const uint32_t tiles = *static_cast<const uint32_t *>(scratch.tileCount.contents());
  const auto *descriptors = static_cast<const MoeTileDescriptor *>(scratch.tileDescriptors.contents());
  const auto *groupedRoutes = static_cast<const uint32_t *>(scratch.groupedRoutes.contents());
  const auto *gin = static_cast<const uint8_t *>(scratch.groupedInput.contents());
  const auto *inter = static_cast<const uint8_t *>(scratch.expertIntermediate.contents());
  const auto *outs = static_cast<const uint16_t *>(scratch.expertOutput.contents());
  const auto *selected = static_cast<const uint32_t *>(scratch.selectedExperts.contents());
  const auto *weights = static_cast<const float *>(scratch.routingWeights.contents());
  const auto *routeRows = static_cast<const uint32_t *>(scratch.routeRows.contents());
  const auto *output = static_cast<const uint16_t *>(b.moe.output.contents());
  for (uint32_t t = 0; t < tiles; ++t) {
    const MoeTileDescriptor tile = descriptors[t];
    if (tile.expert >= E) continue;
    (tile.rows <= 2 ? r.gemvTiles : r.naxTiles) += 1;
    const uint32_t e = tile.expert;
    for (uint32_t row = 0; row < tile.rows; ++row) {
      const uint32_t g = t * 8 + row, token = groupedRoutes[g] / routes;
      const float *x = b.input.data() + uint64_t(token) * H;
      // Down outputs from the GPU's intermediate rows.
      std::vector<double> h(I);
      float hs = 1.0f;
      if (int8) {
        const auto *q = reinterpret_cast<const int8_t *>(inter + uint64_t(t) * 16 * I) + uint64_t(row) * I;
        hs = reinterpret_cast<const float *>(inter + uint64_t(t) * 16 * I + 8 * I)[row];
        for (uint32_t c = 0; c < I; ++c) h[c] = q[c];
      } else {
        const auto *q = reinterpret_cast<const uint16_t *>(inter) + uint64_t(g) * I;
        for (uint32_t c = 0; c < I; ++c) h[c] = bfBits(q[c]);
      }
      for (uint32_t n = 0; n < H; ++n) {
        const float got = bfBits(outs[uint64_t(g) * H + n]);
        float want;
        if (int8) {
          int64_t tot = 0;
          for (uint32_t gg = 0; gg < blk.down.G; ++gg) {
            int64_t acc = 0;
            for (uint32_t kk = 0; kk < 64; ++kk) acc += int64_t(blk.down.qs(e, gg, n, kk)) * int64_t(h[blk.down.input(gg, kk)]);
            tot += acc * blk.down.m(e, gg, n);
          }
          want = bf(float(tot) * blk.down.anchors[n] * hs);
        } else {
          double tot = 0;
          for (uint32_t gg = 0; gg < blk.down.G; ++gg) {
            double acc = 0;
            for (uint32_t kk = 0; kk < 64; ++kk) acc += double(blk.down.qs(e, gg, n, kk)) * h[blk.down.input(gg, kk)];
            tot += acc * blk.down.m(e, gg, n);
          }
          want = bf(float(tot * blk.down.anchors[n]));
        }
        ++r.downCompared;
        if (got != want) ++r.downDiffer;
        if (int8 && got != want) ++r.downBeyond;
      }
      if (!int8) {
        float big = 0;
        for (uint32_t n = 0; n < H; ++n) big = std::max(big, std::fabs(bfBits(outs[uint64_t(g) * H + n])));
        const float ulp = big > 0 ? 2 * std::ldexp(1.0f, std::ilogb(big) - 7) : 0.0f;
        for (uint32_t n = 0; n < H; ++n) {
          double tot = 0;
          for (uint32_t gg = 0; gg < blk.down.G; ++gg) {
            double acc = 0;
            for (uint32_t kk = 0; kk < 64; ++kk) acc += double(blk.down.qs(e, gg, n, kk)) * h[blk.down.input(gg, kk)];
            tot += acc * blk.down.m(e, gg, n);
          }
          if (std::fabs(bfBits(outs[uint64_t(g) * H + n]) - tot * blk.down.anchors[n]) > ulp) ++r.downBeyond;
        }
      }
      if (!int8) continue;
      // The gather's int8 row and the intermediate, recomputed from the bf16 input and the GPU's int8 row.
      std::vector<float> xp(H);
      float amax = 0;
      for (uint32_t k = 0; k < H; ++k) { xp[k] = x[blk.order[k]]; amax = std::max(amax, std::fabs(xp[k])); }
      const float xs = amax > 0 ? amax / 127.0f : 1.0f;
      const auto *a8 = reinterpret_cast<const int8_t *>(gin + uint64_t(t) * 16 * H) + uint64_t(row) * H;
      const float as = reinterpret_cast<const float *>(gin + uint64_t(t) * 16 * H + 8 * H)[row];
      for (uint32_t k = 0; k < H; ++k) { ++r.gatherCodes; if (a8[k] != quant(xp[k], xs)) ++r.gatherDiffer; }
      std::vector<float> raw(2 * I);
      for (uint32_t n = 0; n < 2 * I; ++n) {
        int64_t tot = 0;
        for (uint32_t gg = 0; gg < blk.gateUp.G; ++gg) {
          int64_t acc = 0;
          for (uint32_t kk = 0; kk < 64; ++kk) acc += int64_t(blk.gateUp.qs(e, gg, n, kk)) * a8[blk.gateUp.input(gg, kk)];
          tot += acc * blk.gateUp.m(e, gg, n);
        }
        raw[n] = bf(float(tot) * blk.gateUp.anchors[n] * as);
      }
      std::vector<float> hv(I);
      float hmax = 0;
      for (uint32_t c = 0; c < I; ++c) {
        hv[c] = bf(raw[I + c] * (raw[c] / (1.0f + std::exp(-raw[c]))));
        hmax = std::max(hmax, std::fabs(hv[c]));
      }
      const float hsc = hmax > 0 ? hmax / 127.0f : 1.0f;
      const auto *q = reinterpret_cast<const int8_t *>(inter + uint64_t(t) * 16 * I) + uint64_t(row) * I;
      for (uint32_t c = 0; c < I; ++c) {
        ++r.interCodes;
        const int d = std::abs(int(q[c]) - int(quant(hv[c], hsc)));
        if (d) ++r.interDiffer;
        if (d > 1) ++r.interBeyond1;
      }
    }
  }
  // The block output against fp64 (ULMoE weights from the format definition, bf16 inputs, the GPU's routing).
  for (uint32_t token = 0; token < rows; ++token) {
    const float *x = b.input.data() + uint64_t(token) * H;
    std::vector<double> sum(H, 0.0);
    for (uint32_t s = 0; s < routes; ++s) {
      const uint32_t route = token * routes + s, e = selected[route];
      if (routeRows[route] == ~0u) continue;
      const double w = weights[route];
      std::vector<double> h(I);
      if (e < E) {
        std::vector<double> gu(2 * I);
        for (uint32_t n = 0; n < 2 * I; ++n) {
          double tot = 0;
          for (uint32_t gg = 0; gg < blk.gateUp.G; ++gg) {
            double acc = 0;
            for (uint32_t kk = 0; kk < 64; ++kk)
              acc += double(blk.gateUp.qs(e, gg, n, kk)) * x[blk.order[blk.gateUp.input(gg, kk)]];
            tot += acc * blk.gateUp.m(e, gg, n);
          }
          gu[n] = tot * blk.gateUp.anchors[n];
        }
        for (uint32_t c = 0; c < I; ++c) h[c] = gu[I + c] * gu[c] / (1.0 + std::exp(-gu[c]));
        for (uint32_t n = 0; n < H; ++n) {
          double tot = 0;
          for (uint32_t gg = 0; gg < blk.down.G; ++gg) {
            double acc = 0;
            for (uint32_t kk = 0; kk < 64; ++kk) acc += double(blk.down.qs(e, gg, n, kk)) * h[blk.down.input(gg, kk)];
            tot += acc * blk.down.m(e, gg, n);
          }
          sum[n] += w * tot * blk.down.anchors[n];
        }
      } else {
        for (uint32_t c = 0; c < I; ++c) {
          double gv = 0, uv = 0;
          for (uint32_t k = 0; k < H; ++k) {
            gv += double(blk.shared[0].row(c)[k]) * x[k];
            uv += double(blk.shared[1].row(c)[k]) * x[k];
          }
          h[c] = uv * gv / (1.0 + std::exp(-gv));
        }
        for (uint32_t n = 0; n < H; ++n) {
          double tot = 0;
          for (uint32_t c = 0; c < I; ++c) tot += double(blk.shared[2].row(n)[c]) * h[c];
          sum[n] += w * tot;
        }
      }
    }
    for (uint32_t n = 0; n < H; ++n) {
      const double got = bfBits(output[uint64_t(token) * H + n]) - double(b.residual[uint64_t(token) * H + n]);
      r.err2 += (got - sum[n]) * (got - sum[n]);
      r.ref2 += sum[n] * sum[n];
    }
  }
  return r;
}

int caseOf(MetalBackend &backend, MoeShape shape, const char *shapeName, std::array<Fmt, 3> shared, bool fixed) {
  int failures = 0;
  const Block blk = makeBlock(backend, shape, shared, fixed);
  Buffers b;
  const uint32_t maxRows = 263;
  b.input = activations(uint64_t{maxRows} * shape.hiddenSize);
  b.residual = activations(uint64_t{maxRows} * shape.hiddenSize);
  for (float &v : b.residual) v = bf16(0.1f * v);
  b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
  b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
  b.moe.output = zeros(backend, uint64_t{maxRows} * shape.hiddenSize * 2, "moe-output");
  for (const bool int8 : {true, false})
    for (const uint32_t step : {1u, 4u, 0u}) {
      MoeConfig config{.expertTile = MoeExpertTile::M8, .ggufTile = MoeGgufTile::Staged};
      config.ulmoeInt8 = int8;
      const MoePlan plan = step ? MoE::decodePlan(blk.shape, step, config) : MoE::prefillPlan(blk.shape, 37, config);
      const Result r = run(backend, blk, b, plan, int8);
      const double rel = std::sqrt(r.err2 / std::max(r.ref2, 1e-300));
      const bool ok = (int8 ? r.downDiffer == 0 : r.downBeyond == 0) && rel < (int8 ? 3e-2 : 1e-2);
      failures += !ok;
      printf("%s %-6s shared %s/%s/%s %-4s %s %3u rows: tiles nax %llu gemv %llu | down %llu/%llu differ, %llu beyond | "
             "gather %llu/%llu differ | intermediate %llu/%llu differ, %llu by >1 | block rel rms %.3e %s\n",
             shapeName, fixed ? "fixed" : "random", fmtName(shared[0]), fmtName(shared[1]), fmtName(shared[2]),
             int8 ? "int8" : "bf16", step ? "decode " : "prefill", plan.rows(), (unsigned long long)r.naxTiles,
             (unsigned long long)r.gemvTiles, (unsigned long long)r.downDiffer, (unsigned long long)r.downCompared,
             (unsigned long long)r.downBeyond, (unsigned long long)r.gatherDiffer, (unsigned long long)r.gatherCodes,
             (unsigned long long)r.interDiffer, (unsigned long long)r.interCodes, (unsigned long long)r.interBeyond1, rel,
             ok ? "ok" : "FAIL");
      fflush(stdout);
    }
  return failures;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf_moe_ulmoe_test <metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const MoeShape moe35{.hiddenSize = 2048, .experts = 256, .expertsPerToken = 8, .expertIntermediateSize = 512,
                           .router = splash::ops::MoeRouter::Softmax, .sharedSlices = 1, .sharedGate = true};
      const MoeShape small{.hiddenSize = 1024, .experts = 16, .expertsPerToken = 4, .expertIntermediateSize = 512,
                           .router = splash::ops::MoeRouter::Softmax, .sharedSlices = 1, .sharedGate = true};
      int failures = 0;
      for (const bool fixed : {true, false}) {
        failures += caseOf(backend, moe35, "35B", {Q80, Q80, Q80}, fixed);
        failures += caseOf(backend, small, "16e", {Q80, Q6K, IQ4XS}, fixed);
        if (fixed) failures += caseOf(backend, moe35, "35B", {Q5K, Q5K, Q6K}, fixed);
      }
      if (failures) {
        std::cerr << "gguf_moe_ulmoe_test: " << failures << " failures\n";
        return 1;
      }
      std::cout << "gguf_moe_ulmoe_test: PASS\n";
    } catch (const std::exception &error) {
      std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
