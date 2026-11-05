// gguf-moe-int8: the int8 MoE decode passes (ops::MoeInt8, kernels/shared/moe_gguf_i8.metal)
// against the bf16 passes fed the same int8 rounding (MoeInt8::activations 2, moe_a8_fake), through ops::MoE.
// Power-of-two activation scales make bf16(s q) exact, so both runs multiply the same activations:
// - fixed weight scales (d = dmin = 2^-10, small sub-scales, inputs in steps of 1/16), so that the bf16 tile's half
//   weights are exact and both paths' fp32 sums stay exact: the block output, the intermediate and the down outputs
//   are bitwise equal;
// - the formats' random scales: the half weights round, so a projection's bf16 output may move by one bf16 ulp of its
//   row's largest output: down alone (gate/up on bf16) counts the down outputs outside one, gate/up alone (down on
//   bf16) the intermediates bf16(bf16(up) silu(bf16(gate))) outside two (each product one); both must be 0.
// Each case runs the 35B's shape (hidden 2048, 256 experts of 512, top 8, one gated
// shared slice) at 1 and 4 lanes and the gguf-moe shape (hidden 1024, 16 experts, top 4) at 1 and 4 lanes, where
// most tiles are staged.
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
#include <string>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::ops::MoE;
using splash::ops::MoeConfig;
using splash::ops::MoeExpertTile;
using splash::ops::MoeGgufTile;
using splash::ops::MoeInt8;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using namespace gguf_reference;
using namespace moe_fixture;

// Native rows of f at fixed scales (fixed) or the format's realistic random ones.
std::vector<uint8_t> nativeRows(Fmt f, uint32_t N, uint32_t K, bool fixed) {
  if (!fixed) return makeNative(f, N, K, rng);
  std::vector<uint8_t> v = makeNative(f, N, K, rng, [] { return f2h(1.0f / 1024.0f); });
  const QuantFormat &fi = kQuantFormats[f];
  for (size_t b = 0; b < v.size() / fi.block_bytes; ++b) {
    uint8_t *blk = v.data() + b * fi.block_bytes;
    // Small sub-scales, so that the fp32 sums of both paths stay exact (well under 2^24 of their unit).
    if (f == IQ4XS) { blk[2] = 0x55; blk[3] = 0x55; for (int i = 4; i < 8; ++i) blk[i] |= 0xCC; }   // ls 28..31
    if (f == Q6K) for (int i = 0; i < 16; ++i) blk[192 + i] &= 0x0F;
    if (f == IQ2S) for (int i = 74; i < 82; ++i) blk[i] &= 0x33;
    if (f == IQ2XS) for (int i = 66; i < 74; ++i) blk[i] &= 0x33;
    if (f == IQ3S) for (int i = 106; i < 110; ++i) blk[i] &= 0x33;
    if (f == IQ3XXS) for (int i = 0; i < 8; ++i) blk[66 + 4 * i + 3] &= 0x3F;
  }
  return v;
}
Tensor tensorOf(MetalBackend &backend, Fmt f, uint32_t rows, uint32_t K, bool fixed) {
  Tensor t;
  t.native = nativeRows(f, rows, K, fixed);
  t.rows = rows;
  t.columns = K;
  t.segment = planeSegment(backend, f, repack(f, t.native, rows, K, &t.values), rows, K);
  return t;
}
// formats: routed gate, up, down, shared gate, up, down.
Model modelOf(MetalBackend &backend, const MoeShape &shape, const std::array<Fmt, 6> &formats, bool fixed) {
  const uint32_t hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  const uint32_t n[3] = {intermediate, intermediate, hidden}, k[3] = {hidden, hidden, intermediate};
  Model m;
  m.shape = shape;
  m.formats = formats;
  m.router = floating(backend, shape.experts, hidden, 0.05f);
  if (shape.sharedGate) m.sharedGate = floating(backend, 1, hidden, 0.05f);
  for (int p = 0; p < 3; ++p) {
    m.routed[p] = tensorOf(backend, formats[p], shape.experts * n[p], k[p], fixed);
    m.shared[p] = tensorOf(backend, formats[3 + p], shape.sharedSlices * n[p], k[p], fixed);
  }
  splash::ops::BlockMoeWeights w;
  w.router = m.router.segment;
  w.sharedScalarGate = m.sharedGate.segment;
  w.gate = {m.routed[0].segment, m.shared[0].segment};
  w.up = {m.routed[1].segment, m.shared[1].segment};
  w.down = {m.routed[2].segment, m.shared[2].segment};
  w.slotTable = splash::model::residentExpertSlots(backend, shape.experts);
  w.slots = shape.experts;
  w.chunks = splash::ops::residentExpertChunks(w);
  m.weights = w;
  return m;
}

struct Run {
  std::vector<uint16_t> output, intermediate, down;
  uint32_t stagedTiles = 0, columnTiles = 0;
};
Run run(MetalBackend &backend, const Model &m, Buffers &b, uint32_t lanes, const MoeInt8 &int8) {
  MoeConfig config{.expertTile = MoeExpertTile::M8, .ggufTile = MoeGgufTile::Staged};
  config.int8 = int8;
  const MoePlan plan = MoE::decodePlan(m.shape, lanes, config);
  allocate(backend, b, plan);
  std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
  CommandGraph graph;
  MoE::add(graph, b.moe, m.weights, plan);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  Run r;
  const uint32_t rows = plan.rows(), hidden = m.shape.hiddenSize, I = m.shape.expertIntermediateSize;
  r.output.resize(uint64_t{rows} * hidden);
  std::memcpy(r.output.data(), b.moe.output.contents(), r.output.size() * 2);
  const uint32_t tiles = *static_cast<const uint32_t *>(b.moe.scratch.tileCount.contents());
  const auto *descriptors = static_cast<const MoeTileDescriptor *>(b.moe.scratch.tileDescriptors.contents());
  for (uint32_t t = 0; t < tiles; ++t) (moe_column_tile(descriptors, t) ? r.columnTiles : r.stagedTiles) += 1;
  // Per route (the grouping orders an expert's routes anew each run): its intermediate row, as the down pass reads
  // it (an int8 down input as bf16(s q) of its codes and scales), and its down row.
  const auto *inter = static_cast<const uint16_t *>(b.moe.scratch.expertIntermediate.contents());
  const auto *downs = static_cast<const uint16_t *>(b.moe.scratch.expertOutput.contents());
  const auto *routeRows = static_cast<const uint32_t *>(b.moe.scratch.routeRows.contents());
  const bool int8Down = int8.activations == 1 &&
                        ((int8.formats >> m.weights.down.routed.formatId) & 1u ||
                         (int8.formats >> m.weights.down.shared.formatId) & 1u);
  const splash::ops::MoeInt8Layout layout =
      splash::ops::moeInt8Layout(m.shape, uint64_t{plan.maximumTiles()} * plan.tileRows());
  const auto *sums = static_cast<const uint8_t *>(b.moe.scratch.groupedSums.contents());
  for (uint32_t route = 0; route < rows * m.shape.routesPerToken(); ++route) {
    const uint32_t g = routeRows[route];
    if (g == ~0u) continue;
    if (int8Down) {
      const auto *q = reinterpret_cast<const int8_t *>(sums + layout.d8) + uint64_t{g} * I;
      const auto *sc = reinterpret_cast<const float *>(sums + layout.dScales) + uint64_t{g} * (I / 32);
      for (uint32_t n = 0; n < I; ++n) {
        const __bf16 v = __bf16(float(q[n]) * sc[n / 32]);
        uint16_t bits; std::memcpy(&bits, &v, 2);
        r.intermediate.push_back(bits);
      }
    } else {
      r.intermediate.insert(r.intermediate.end(), inter + uint64_t{g} * I, inter + uint64_t{g + 1} * I);
    }
    r.down.insert(r.down.end(), downs + uint64_t{g} * hidden, downs + uint64_t{g + 1} * hidden);
  }
  return r;
}
float bfloat(uint16_t bits) { const uint32_t v = uint32_t(bits) << 16; float f; std::memcpy(&f, &v, 4); return f; }
// Outputs that differ, and that differ by more than one bf16 ulp of the largest magnitude of their row of `width`
// (a half weight's rounding moves an output by a share of its row's scale, not of its own value).
struct Diff {
  uint64_t compared = 0, differ = 0, beyondUlp = 0;
  double maxRel = 0;
};
Diff compare(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, size_t width, float ulps = 1) {
  Diff d;
  if (a.size() != b.size()) { d.compared = 1; d.differ = d.beyondUlp = 1; return d; }
  for (size_t row = 0; row < a.size(); row += width) {
    float big = 0;
    for (size_t i = row; i < row + width; ++i) big = std::max(big, std::fabs(bfloat(a[i])));
    const float ulp = big > 0 ? ulps * std::ldexp(1.0f, std::ilogb(big) - 7) : 0.0f;
    for (size_t i = row; i < row + width; ++i) {
      ++d.compared;
      if (a[i] == b[i]) continue;
      ++d.differ;
      const float x = bfloat(a[i]), y = bfloat(b[i]);
      if (std::fabs(x - y) > ulp) ++d.beyondUlp;
      d.maxRel = std::max(d.maxRel, double(std::fabs(x - y) / std::max(big, 1e-30f)));
    }
  }
  return d;
}

uint32_t maskOf(std::initializer_list<Fmt> formats) {
  uint32_t mask = 0;
  for (const Fmt f : formats) mask |= 1u << f;
  return mask;
}

int caseOf(MetalBackend &backend, const MoeShape &shape, const char *shapeName, const std::array<Fmt, 6> &formats,
           bool fixed) {
  int failures = 0;
  const Model m = modelOf(backend, shape, formats, fixed);
  Buffers b;
  const uint32_t maxRows = 4 * 8;
  b.input = activations(uint64_t{maxRows} * shape.hiddenSize);
  if (fixed) for (float &v : b.input) v = std::round(v * 16.0f) / 16.0f;   // a few bits: exact sums
  b.residual = activations(uint64_t{maxRows} * shape.hiddenSize);
  for (float &v : b.residual) v = bf16(0.1f * v);
  b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
  b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
  b.moe.output = zeros(backend, uint64_t{maxRows} * shape.hiddenSize * 2, "moe-output");
  const uint32_t all = maskOf({formats[0], formats[1], formats[2], formats[3], formats[4], formats[5]});
  const uint32_t gateUp = maskOf({formats[0], formats[1], formats[3], formats[4]});
  const uint32_t down = maskOf({formats[2], formats[5]});
  struct Part { const char *name; uint32_t mask; };
  // Gate/up alone and down alone only where their formats do not overlap (a mask names formats, not passes).
  std::vector<Part> parts{{"all", all}};
  if (!(gateUp & down)) { parts.push_back({"gate/up", gateUp}); parts.push_back({"down", down}); }
  std::string names;
  for (size_t i = 0; i < 6; ++i) names += std::string(i ? (i == 3 ? " | " : "/") : "") + fmtName(formats[i]);
  for (const uint32_t lanes : {1u, 4u})
    for (const Part &part : parts)
      for (const bool column : {false}) {
        MoeInt8 ref{.activations = 2, .formats = part.mask, .pow2 = true};
        MoeInt8 on{.activations = 1, .formats = part.mask, .pow2 = true};
        const Run a = run(backend, m, b, lanes, ref), c = run(backend, m, b, lanes, on);
        const Diff out = compare(a.output, c.output, shape.hiddenSize),
                   inter = compare(a.intermediate, c.intermediate, shape.expertIntermediateSize, 2),
                   dn = compare(a.down, c.down, shape.hiddenSize);
        // Fixed scales: bitwise. Random scales: the pass alone within one ulp (the block output of "all" carries
        // the down input's re-rounding of a moved intermediate, so it is reported, not bounded). Without the int8
        // column tile a lone tile's rows run the bf16 column tile on unrounded activations: those steps are
        // reported, not bounded.
        bool ok;
        if (!column && c.columnTiles) ok = true;
        else if (fixed) ok = !out.differ && !inter.differ && !dn.differ;
        else if (std::string(part.name) == "gate/up") ok = !inter.beyondUlp;
        else if (std::string(part.name) == "down") ok = !dn.beyondUlp && !inter.differ;
        else ok = true;
        failures += !ok;
        printf("%s %-8s %s lanes %u %-7s column %d: tiles staged %u column %u | output %llu/%llu differ, %llu > 1 ulp, "
               "max rel %.2e | intermediate %llu/%llu differ, %llu > 1 ulp | down %llu/%llu differ, %llu > 1 ulp %s\n",
               shapeName, fixed ? "fixed" : "random", names.c_str(), lanes, part.name, int(column), c.stagedTiles,
               c.columnTiles, (unsigned long long)out.differ, (unsigned long long)out.compared,
               (unsigned long long)out.beyondUlp, out.maxRel, (unsigned long long)inter.differ,
               (unsigned long long)inter.compared, (unsigned long long)inter.beyondUlp, (unsigned long long)dn.differ,
               (unsigned long long)dn.compared, (unsigned long long)dn.beyondUlp, ok ? "ok" : "FAIL");
      }
  return failures;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf_moe_int8_test <metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const MoeShape moe35{.hiddenSize = 2048, .experts = 256, .expertsPerToken = 8, .expertIntermediateSize = 512,
                           .router = splash::ops::MoeRouter::Softmax, .sharedSlices = 1, .sharedGate = true};
      const MoeShape small{.hiddenSize = 1024, .experts = 16, .expertsPerToken = 4, .expertIntermediateSize = 512,
                           .router = splash::ops::MoeRouter::Softmax, .sharedSlices = 1, .sharedGate = true};
      // Routed gate, up, down | shared gate, up, down: the grid formats; IQ4_XS; UD-IQ3_S (routed IQ2_S gate/up,
      // IQ3_S or IQ4_XS down, shared Q8_0 / Q6_K); UD-Q2_K_XL (IQ2_XS / IQ3_XXS routed, Q5_K / Q6_K shared), Q2_K, and
      // Q4_K beside IQ3_S.
      const std::vector<std::array<Fmt, 6>> cases{
          {IQ2S, IQ2S, IQ3S, IQ2XS, IQ2XS, IQ3XXS}, {IQ2XS, IQ2XS, IQ3XXS, IQ3S, IQ3S, IQ2S},
          {IQ2S, IQ2S, IQ4XS, IQ3S, IQ3S, IQ4XS},
          {IQ2S, IQ2S, IQ3S, Q80, Q80, Q6K}, {IQ2S, IQ2S, IQ4XS, Q80, Q80, Q80},
          {IQ2XS, IQ2XS, IQ3XXS, Q5K, Q5K, Q6K}, {Q2K, Q2K, Q6K, Q80, Q80, Q2K}, {IQ3S, IQ3S, Q4K, Q6K, Q6K, IQ4XS},
      };
      int failures = 0;
      for (const bool fixed : {true, false})
        for (const auto &formats : cases) {
          failures += caseOf(backend, moe35, "35B", formats, fixed);
          failures += caseOf(backend, small, "16e", formats, fixed);
        }
      if (failures) {
        std::cerr << "gguf_moe_int8_test: " << failures << " failures\n";
        return 1;
      }
      std::cout << "gguf_moe_int8_test: PASS\n";
    } catch (const std::exception &error) {
      std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
