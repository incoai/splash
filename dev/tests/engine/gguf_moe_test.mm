// GGUF sparse MoE (ops::MoE over GgufMoeWeights) and the fp32 projection of
// GGUF float tensors, against fp64 references over GGML's dequantized weights
// (GgufFormatReference.hpp), through the production dispatch code.
// - Float projection: every row count a decode or prefill dispatch takes (with
//   ragged 32-row tails), the router and alpha/beta widths, bf16 and fp32
//   destinations at a column offset of a wider row; neighbours stay untouched.
// - Float segments of a fused GGUF projection (Q4Linear): the quantized
//   segments' outputs unchanged, the padding past the segments unwritten.
// - MoE: every GGUF plan (the staged 8- and 64-row tiles and the Apple9
//   register tile, whatever GPU runs the test; decode steps and prefill
//   chunks) for all 8 formats, gate, up and down in three formats and the
//   shared expert in three more; routes and weights against the fp64 router,
//   each pass inside the fp64 interval its numerics allow, the block's output,
//   and a row's output bitwise equal at every lane count and chunk of one
//   tile.
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/ExecutionPlans.hpp"
#include "../../../runtime/ops/Linear.hpp"
#include "../../../runtime/ops/MoE.hpp"
#include "GgufFormatReference.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using splash::DeviceCapabilities;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::FloatOutput;
using splash::ops::GgufExpertProjection;
using splash::ops::GgufMoeWeights;
using splash::ops::GgufSegment;
using splash::ops::MoE;
using splash::ops::MoeBuffers;
using splash::ops::MoeConfig;
using splash::ops::MoeExpertSimdgroups;
using splash::ops::MoeExpertTile;
using splash::ops::MoeGgufTile;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using splash::ops::MoeWeights;
using splash::ops::LinearEpilogue;
using splash::ops::LinearMatrix;
using splash::ops::LinearPhase;
using splash::ops::LinearPlan;
using splash::ops::LinearScratch;
using splash::ops::LinearScratchSize;
using splash::ops::Q4Linear;
using splash::ops::Q4Projection;
using splash::ops::QuantFamily;
using namespace gguf_reference;

// K = 1024 on the hidden side (16 spans, four 256-input coefficient units)
// and 512 on the intermediate side, as the 35B's experts.
constexpr uint32_t kHidden = 1024;
constexpr uint32_t kIntermediate = 512;
constexpr uint32_t kExperts = 16;
constexpr uint32_t kTopK = 4;
constexpr uint32_t kRoutes = kTopK + 1;
constexpr uint32_t kMaximumRows = 263;

std::mt19937 rng(0x35);

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition) fail(message);
}

float bf16(double value) { return float(__bf16(float(value))); }
double silu(double value) { return value / (1.0 + std::exp(-value)); }
// Where silu has its minimum (silu' = 0).
constexpr double kSiluArgmin = -1.2784645427610738;

MetalBuffer upload(MetalBackend &backend, const void *data, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

MetalBuffer zeros(MetalBackend &backend, uint64_t bytes, const char *label) {
  if (!bytes) return {};
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// A GGUF tensor [rows, K]: its segment and GGML's fp32 values of it.
struct Tensor {
  GgufSegment segment;
  std::vector<float> values;
  uint32_t rows = 0, columns = 0;
  [[nodiscard]] const float *row(uint64_t n) const { return values.data() + n * columns; }
};

Tensor quantized(MetalBackend &backend, Fmt f, uint32_t rows, uint32_t K) {
  // Scales in realistic per-format ranges (as the dense GGUF checks).
  std::uniform_real_distribution<float> dk(0.0005f, 0.004f), dx(0.00002f, 0.00015f), d6(0.00002f, 0.0001f),
      d3s(0.0001f, 0.0005f);
  std::uniform_real_distribution<float> &d = f == IQ4XS ? dx : f == Q6K ? d6 : f == IQ3S ? d3s : dk;
  const std::vector<uint8_t> native = makeNative(f, rows, K, rng, [&] { return f2h(d(rng)); });
  Tensor t;
  t.rows = rows;
  t.columns = K;
  const Packed planes = repack(f, native, rows, K, &t.values);
  const QuantFormat &layout = kQuantFormats[f];
  GgufSegment &s = t.segment;
  s.plane0 = upload(backend, planes.w0.data(), planes.w0.size(), "gguf-plane0");
  if (layout.plane1_bytes) s.plane1 = upload(backend, planes.w1.data(), planes.w1.size(), "gguf-plane1");
  s.meta = upload(backend, planes.meta.data(), planes.meta.size(), "gguf-meta");
  s.type = layout.ggml_type;
  s.outputSize = rows;
  s.inputSize = K;
  s.p0 = layout.plane0_bytes;
  s.p1 = layout.plane1_bytes;
  s.metaBytes = layout.meta_bytes;
  s.metaGroups = layout.meta_groups;
  s.formatId = f;
  s.format = layout.name;
  return t;
}

Tensor floating(MetalBackend &backend, uint32_t rows, uint32_t K, float scale) {
  std::normal_distribution<float> normal(0.0f, scale);
  Tensor t;
  t.rows = rows;
  t.columns = K;
  t.values.resize(uint64_t{rows} * K);
  for (float &v : t.values) v = normal(rng);
  GgufSegment &s = t.segment;
  s.plane0 = upload(backend, t.values.data(), t.values.size() * sizeof(float), "gguf-floats");
  s.type = GGUF_TYPE_F32;
  s.outputSize = rows;
  s.inputSize = K;
  s.formatId = GGUF_FMT_COUNT;
  s.format = "f32";
  return t;
}

// bf16 rows in [-1, 1].
std::vector<float> activations(uint64_t elements) {
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::vector<float> values(elements);
  for (float &v : values) v = bf16(unit(rng));
  return values;
}

MetalBuffer bfloatBuffer(MetalBackend &backend, const std::vector<float> &values, const char *label) {
  std::vector<__bf16> bits(values.begin(), values.end());
  return upload(backend, bits.data(), bits.size() * 2, label);
}

// The fp64 dot product of a bf16 row with a weight row and the magnitudes its
// error bounds scale with.
struct Dot {
  double value = 0, magnitude = 0, inputs = 0, largest = 0;
};
Dot dot(const float *x, const float *w, uint32_t K) {
  Dot d;
  for (uint32_t k = 0; k < K; ++k) {
    d.value += double(x[k]) * w[k];
    d.magnitude += std::fabs(double(x[k]) * w[k]);
    d.inputs += std::fabs(x[k]);
    d.largest = std::max(d.largest, double(std::fabs(w[k])));
  }
  return d;
}

// fp32 accumulation of K exact products: at most K u sum |x w| (u = 2^-24).
double floatBound(const Dot &d, uint32_t K) { return std::ldexp(d.magnitude * K, -24) + 1e-30; }

// The fp64 interval a quantized projection's result lies in: the register
// tile is exact up to fp32 accumulation (2^-16 sum|x| max|w|, the dense
// register bound); staging rounds every weight once to half (2^-11 relative,
// 2^-25 absolute below the normal range).
double projectionBound(const Dot &d, bool staged) {
  const double accumulation = std::ldexp(d.inputs * d.largest, -16);
  return staged ? accumulation + std::ldexp(d.magnitude, -11) + std::ldexp(d.inputs, -25) : accumulation;
}

bool inside(float got, double exact, double bound) {
  return got >= bf16(exact - bound) && got <= bf16(exact + bound);
}

// ---------------------------------------------------------------- float projection
int floatProjection(MetalBackend &backend) {
  int failures = 0;
  for (const uint32_t K : {512u, 2048u})
    for (const uint32_t N : {64u, 256u}) {
      const Tensor w = floating(backend, N, K, 0.05f);
      double worst = 0;
      for (const uint32_t rows : {1u, 7u, 8u, 16u, 24u, 32u, 33u, 263u, 2048u}) {
        if (rows == 2048 && (K != 2048 || N != 256)) continue;   // the router's prefill width
        const std::vector<float> x = activations(uint64_t{rows} * K);
        const MetalBuffer input = bfloatBuffer(backend, x, "float-input");
        for (const FloatOutput type : {FloatOutput::BFloat16, FloatOutput::Float32}) {
          // A destination row of N + 96 columns at offset 32, one row more than
          // the dispatch writes; the sentinel must survive outside the tile.
          const uint32_t stride = N + 96, offset = 32;
          const uint64_t element = type == FloatOutput::Float32 ? 4 : 2;
          MetalBuffer output = backend.allocateBuffer(uint64_t{rows + 1} * stride * element, BufferStorage::Shared,
                                                      "float-output");
          std::memset(output.contents(), 0x7F, output.sizeBytes());
          CommandGraph graph;
          splash::ops::addGgufFloat(graph, input, w.segment, output, rows, stride, offset, type);
          static_cast<void>(backend.submitCommand(graph.dispatches()));
          size_t outside = 0, touched = 0;
          for (uint32_t r = 0; r <= rows; ++r)
            for (uint32_t c = 0; c < stride; ++c) {
              const uint64_t at = uint64_t{r} * stride + c;
              const bool live = r < rows && c >= offset && c < offset + N;
              const uint8_t *bytes = static_cast<const uint8_t *>(output.contents()) + at * element;
              if (!live) {
                for (uint64_t b = 0; b < element; ++b) touched += bytes[b] != 0x7F;
                continue;
              }
              const Dot d = dot(x.data() + uint64_t{r} * K, w.row(c - offset), K);
              const double bound = floatBound(d, K);
              float got;
              if (type == FloatOutput::Float32) {
                std::memcpy(&got, bytes, 4);
                outside += !(std::fabs(got - d.value) <= bound);
              } else {
                __bf16 half;
                std::memcpy(&half, bytes, 2);
                got = float(half);
                outside += !inside(got, d.value, bound);
              }
              worst = std::max(worst, std::fabs(got - d.value) / (d.magnitude + 1e-30));
            }
          if (outside || touched) {
            printf("  float K=%u N=%u rows=%u %s: %zu outputs outside the fp32 bound, %zu bytes written outside FAIL\n",
                   K, N, rows, type == FloatOutput::Float32 ? "f32" : "bf16", outside, touched);
            ++failures;
          }
        }
      }
      printf("float K=%u N=%u: rows 1-2048, bf16/f32 at a column offset: worst |error| / sum|x w| %.2e %s\n", K, N,
             worst, failures ? "FAIL" : "ok");
    }
  return failures;
}

// A fused projection with a float segment, as the 35B's GDN input (qkv | z |
// alpha-beta): the fp32 kernel writes the float segment's columns, the
// quantized kernels the others bit for bit as without it, and the padding
// past the last segment stays unwritten; decode through both GGUF decode
// families (register, staged) at one to four lanes, and prefill.
int floatSegments(MetalBackend &backend) {
  constexpr uint32_t K = 1024, N = 768, kFloatColumn = 512, kCovered = 576;
  const Tensor q80 = quantized(backend, Q80, 256, K), q4k = quantized(backend, Q4K, 256, K);
  const Tensor gates = floating(backend, 64, K, 0.05f);
  const auto at = [](GgufSegment s, uint32_t offset) { s.columnOffset = offset; return s; };
  Q4Projection full, quantizedOnly;
  full.outputSize = quantizedOnly.outputSize = N;
  full.inputSize = quantizedOnly.inputSize = K;
  full.gguf = {at(q80.segment, 0), at(q4k.segment, 256), at(gates.segment, kFloatColumn)};
  quantizedOnly.gguf = {at(q80.segment, 0), at(q4k.segment, 256)};
  int failures = 0;
  for (const uint32_t family : {9u, 10u}) {
    DeviceCapabilities device = backend.capabilities();
    device.appleGpuFamily = family;
    const Q4Linear linear(device);
    const auto check = [&](uint32_t rows, uint32_t storage, const std::string &label,
                           const std::function<void(CommandGraph &, MetalBuffer, const Q4Projection &, MetalBuffer)> &add) {
      const std::vector<float> x = activations(uint64_t{storage} * K);
      const MetalBuffer input = bfloatBuffer(backend, x, "segments-input");
      MetalBuffer y = zeros(backend, uint64_t{storage} * N * 2, "segments-output"),
                  reference = zeros(backend, uint64_t{storage} * N * 2, "segments-reference");
      std::memset(y.contents(), 0x7F, y.sizeBytes());
      std::memset(reference.contents(), 0x7F, reference.sizeBytes());
      CommandGraph graph;
      add(graph, input, full, y);
      add(graph, input, quantizedOnly, reference);
      static_cast<void>(backend.submitCommand(graph.dispatches()));
      const auto *got = static_cast<const uint16_t *>(y.contents()), *want = static_cast<const uint16_t *>(reference.contents());
      size_t differ = 0, outside = 0, written = 0;
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < N; ++c) {
          const uint64_t i = uint64_t{r} * N + c;
          if (c < kFloatColumn) differ += got[i] != want[i];
          else if (c >= kCovered) written += got[i] != 0x7F7F;
          else {
            const Dot d = dot(x.data() + uint64_t{r} * K, gates.row(c - kFloatColumn), K);
            __bf16 value;
            std::memcpy(&value, got + i, 2);
            outside += !inside(float(value), d.value, floatBound(d, K));
          }
        }
      if (differ || outside || written) {
        printf("  float segment %s family %u: %zu quantized outputs differ, %zu float outputs outside the fp32 bound, "
               "%zu padding outputs written FAIL\n", label.c_str(), family, differ, outside, written);
        ++failures;
      }
    };
    // The scratch of the plan, whose staged tiles may split K and hold more rows than the step.
    const auto scratchFor = [&](const LinearPlan &plan) {
      const LinearScratchSize size = plan.scratchSize();
      return LinearScratch{zeros(backend, std::max<uint64_t>(size.input, 16), "scratch-input"),
                           zeros(backend, std::max<uint64_t>(size.sums, 16), "scratch-sums"),
                           zeros(backend, std::max<uint64_t>(size.partials, 16), "scratch-partials"),
                           zeros(backend, std::max<uint64_t>(size.counters, 16), "scratch-counters")};
    };
    const LinearMatrix matrix{N, K};
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const LinearPlan plan = linear.plan({matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::None, QuantFamily::Gguf},
                                          full);
      const LinearScratch scratch = scratchFor(plan);
      check(lanes * 8, plan.storageRows(), "decode B" + std::to_string(lanes),
            [&](CommandGraph &graph, MetalBuffer input, const Q4Projection &p, MetalBuffer output) {
              splash::ops::Q4DispatchStats stats;
              static_cast<void>(linear.addDecodeBatch(graph, input, p, output, matrix, lanes, stats, scratch));
            });
    }
    for (const uint32_t rows : {1u, 24u, 33u, 263u}) {
      const LinearPlan plan =
          linear.plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None, QuantFamily::Gguf}, full);
      const LinearScratch scratch = scratchFor(plan);
      check(rows, plan.storageRows(), "prefill rows=" + std::to_string(rows),
            [&](CommandGraph &graph, MetalBuffer input, const Q4Projection &p, MetalBuffer output) {
              linear.addPrefill(graph, input, p, output, {}, matrix, rows, scratch);
            });
    }
  }
  printf("float segment: fused Q8_0|Q4_K|F32 decode B1-4 (register, staged) and prefill 1/24/33/263 rows %s\n",
         failures ? "FAIL" : "ok");
  return failures;
}

// ---------------------------------------------------------------- MoE
struct Model {
  Tensor router, sharedGate;
  std::array<Tensor, 3> routed;   // gate, up, down: experts * N rows
  std::array<Tensor, 3> shared;
  MoeWeights weights;
  std::array<Fmt, 6> formats{};
};

// Gate, up and down in formats f, f + 1, f + 2 and the shared expert's in
// f + 3, f + 4, f + 5.
Model makeModel(MetalBackend &backend, int f) {
  Model m;
  for (int i = 0; i < 6; ++i) m.formats[i] = Fmt((f + i) % FMT_COUNT);
  m.router = floating(backend, kExperts, kHidden, 0.05f);
  m.sharedGate = floating(backend, 1, kHidden, 0.05f);
  const uint32_t n[3] = {kIntermediate, kIntermediate, kHidden}, k[3] = {kHidden, kHidden, kIntermediate};
  for (int p = 0; p < 3; ++p) {
    m.routed[p] = quantized(backend, m.formats[p], kExperts * n[p], k[p]);
    m.shared[p] = quantized(backend, m.formats[3 + p], n[p], k[p]);
  }
  GgufMoeWeights gguf;
  gguf.router = m.router.segment;
  gguf.sharedExpertGate = m.sharedGate.segment;
  gguf.gate = {m.routed[0].segment, m.shared[0].segment};
  gguf.up = {m.routed[1].segment, m.shared[1].segment};
  gguf.down = {m.routed[2].segment, m.shared[2].segment};
  m.weights.gguf = gguf;
  return m;
}

struct Buffers {
  MoeBuffers moe;
  std::vector<float> input, residual;
};

void allocate(MetalBackend &backend, Buffers &b, const MoePlan &plan) {
  const auto &w = plan.workspace();
  MoeBuffers &m = b.moe;
  m.selectedExperts = zeros(backend, w.selectedExpertsBytes, "selected");
  m.routingWeights = zeros(backend, w.routingWeightsBytes, "routing");
  m.tileDescriptors = zeros(backend, w.tileDescriptorsBytes, "tiles");
  m.tileCount = zeros(backend, w.tileCountBytes, "tile-count");
  m.groupedRoutes = zeros(backend, w.groupedRoutesBytes, "grouped-routes");
  m.routeRows = zeros(backend, w.routeRowsBytes, "route-rows");
  m.groupedInput = zeros(backend, w.groupedInputBytes, "grouped-input");
  m.expertIntermediate = zeros(backend, w.expertIntermediateBytes, "intermediate");
  m.expertOutput = zeros(backend, w.expertOutputBytes, "expert-output");
  m.groupedSums = zeros(backend, w.groupedSumsBytes, "grouped-sums");
}

// The fp64 gate and up products of (row, expert), shared by every plan.
struct GateUp {
  std::vector<Dot> gate, up;
};

struct Stats {
  size_t outputs = 0, gateUpFlips = 0, downFlips = 0;
  // The worst |error| relative to the magnitudes its products sum (sum |x w|).
  double gateUpWorst = 0, downWorst = 0;
};

// One plan's run checked against the fp64 model; returns its output rows.
std::vector<uint16_t> runPlan(MetalBackend &backend, const Model &m, Buffers &b, const MoePlan &plan, bool staged,
                              std::map<std::pair<uint32_t, uint32_t>, GateUp> &products, Stats &stats,
                              const std::string &label) {
  const uint32_t rows = plan.rows();
  allocate(backend, b, plan);
  std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
  CommandGraph graph;
  MoE::add(graph, b.moe, m.weights, plan);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const auto *selected = static_cast<const uint32_t *>(b.moe.selectedExperts.contents());
  const auto *routing = static_cast<const float *>(b.moe.routingWeights.contents());
  const auto *routeRows = static_cast<const uint32_t *>(b.moe.routeRows.contents());
  const auto *intermediate = static_cast<const __bf16 *>(b.moe.expertIntermediate.contents());
  const auto *down = static_cast<const __bf16 *>(b.moe.expertOutput.contents());
  const auto *output = static_cast<const __bf16 *>(b.moe.output.contents());
  const uint32_t tileRows = plan.tileRows(), tiles = *static_cast<const uint32_t *>(b.moe.tileCount.contents());
  require(tiles <= plan.maximumTiles(), label + ": tile count exceeds its bound");
  for (uint32_t r = 0; r < rows; ++r) {
    const float *x = b.input.data() + uint64_t{r} * kHidden;
    // Routes: the fp64 top-k by descending score, ascending id; the GPU's fp32
    // scores may reorder only experts closer than their error bounds.
    std::array<double, kExperts> score{}, bound{};
    for (uint32_t e = 0; e < kExperts; ++e) {
      const Dot d = dot(x, m.router.row(e), kHidden);
      score[e] = d.value;
      bound[e] = floatBound(d, kHidden);
    }
    std::array<uint32_t, kExperts> order;
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t c) {
      return score[a] != score[c] ? score[a] > score[c] : a < c;
    });
    for (uint32_t rank = 0; rank < kTopK; ++rank) {
      const uint32_t got = selected[r * kRoutes + rank], want = order[rank];
      require(got == want || std::fabs(score[got] - score[want]) <= bound[got] + bound[want],
              label + ": row " + std::to_string(r) + " routes expert " + std::to_string(got) + " at rank " +
                  std::to_string(rank) + ", fp64 ranks " + std::to_string(want));
    }
    // fp32 weights: a softmax weight moves by at most twice the largest score
    // error of the routes (relative), a sigmoid by its argument's error.
    double denominator = 0, scoreError = 0;
    for (uint32_t rank = 0; rank < kTopK; ++rank) {
      denominator += std::exp(score[order[rank]] - score[order[0]]);
      scoreError = std::max(scoreError, bound[order[rank]]);
    }
    for (uint32_t rank = 0; rank < kTopK; ++rank) {
      const double want = std::exp(score[order[rank]] - score[order[0]]) / denominator;
      require(std::fabs(routing[r * kRoutes + rank] - want) <= want * (2 * scoreError + std::ldexp(1.0, -18)),
              label + ": routing weight of row " + std::to_string(r) + " differs from the fp64 softmax");
    }
    const Dot gate = dot(x, m.sharedGate.row(0), kHidden);
    const double sharedWeight = 1.0 / (1.0 + std::exp(-gate.value));
    require(selected[r * kRoutes + kTopK] == kExperts &&
                std::fabs(routing[r * kRoutes + kTopK] - sharedWeight) <=
                    sharedWeight * (floatBound(gate, kHidden) + std::ldexp(1.0, -18)),
            label + ": shared expert route of row " + std::to_string(r) + " is wrong");

    // Each route: silu(gate) * up from the fp64 products, then down over the
    // GPU's own intermediate, then the weighted sum over the GPU's outputs.
    std::vector<double> expected(kHidden), magnitude(kHidden);
    for (uint32_t c = 0; c < kHidden; ++c) expected[c] = b.residual[uint64_t{r} * kHidden + c];
    for (uint32_t slot = 0; slot < kRoutes; ++slot) {
      const uint32_t route = r * kRoutes + slot, expert = selected[route], grouped = routeRows[route];
      require(grouped < tiles * tileRows, label + ": grouped row out of range");
      const bool isShared = expert == kExperts;
      GateUp &gu = products[{r, expert}];
      if (gu.gate.empty())
        for (uint32_t n = 0; n < kIntermediate; ++n) {
          const uint64_t row = isShared ? n : uint64_t{expert} * kIntermediate + n;
          gu.gate.push_back(dot(x, (isShared ? m.shared[0] : m.routed[0]).row(row), kHidden));
          gu.up.push_back(dot(x, (isShared ? m.shared[1] : m.routed[1]).row(row), kHidden));
        }
      std::vector<float> h(kIntermediate);
      for (uint32_t n = 0; n < kIntermediate; ++n) {
        const Dot &g = gu.gate[n], &u = gu.up[n];
        const double eg = projectionBound(g, staged), eu = projectionBound(u, staged);
        // bf16(u) * silu(bf16(g)) rounded to bf16: silu over the gate's
        // interval (its minimum inside if the interval holds it) times the up
        // interval.
        const double g0 = bf16(g.value - eg), g1 = bf16(g.value + eg);
        double s0 = std::min(silu(g0), silu(g1)), s1 = std::max(silu(g0), silu(g1));
        if (g0 < kSiluArgmin && g1 > kSiluArgmin) s0 = silu(kSiluArgmin);
        double lo = 1e300, hi = -1e300;
        for (const double sc : {s0, s1})
          for (const double uc : {double(bf16(u.value - eu)), double(bf16(u.value + eu))}) {
            lo = std::min(lo, uc * sc);
            hi = std::max(hi, uc * sc);
          }
        const double slack = std::ldexp(std::max(std::fabs(lo), std::fabs(hi)), -8) + 1e-7;
        const float got = float(intermediate[uint64_t{grouped} * kIntermediate + n]);
        h[n] = got;
        const double exact = double(bf16(u.value)) * silu(bf16(g.value));
        if (!(got >= bf16(lo - slack) && got <= bf16(hi + slack)))
          fail(label + ": gate/up of row " + std::to_string(r) + " expert " + std::to_string(expert) + " column " +
               std::to_string(n) + ": " + std::to_string(got) + " outside [" + std::to_string(lo) + ", " +
               std::to_string(hi) + "]");
        stats.gateUpFlips += got != bf16(exact);
        // First order in the two products' magnitudes: |silu(g)| sum|x w_u| + |u silu'(g)| sum|x w_g|.
        const double sigma = 1.0 / (1.0 + std::exp(-g.value));
        const double scale = std::fabs(silu(g.value)) * u.magnitude +
                             std::fabs(u.value * sigma * (1.0 + g.value * (1.0 - sigma))) * g.magnitude;
        stats.gateUpWorst = std::max(stats.gateUpWorst, std::fabs(got - exact) / scale);
      }
      for (uint32_t n = 0; n < kHidden; ++n) {
        const uint64_t row = isShared ? n : uint64_t{expert} * kHidden + n;
        const Dot y = dot(h.data(), (isShared ? m.shared[2] : m.routed[2]).row(row), kIntermediate);
        const double e = projectionBound(y, staged);
        const float got = float(down[uint64_t{grouped} * kHidden + n]);
        if (!inside(got, y.value, e))
          fail(label + ": down of row " + std::to_string(r) + " expert " + std::to_string(expert) + " column " +
               std::to_string(n) + ": " + std::to_string(got) + " outside fp64 " + std::to_string(y.value) + " +- " +
               std::to_string(e));
        stats.downFlips += got != bf16(y.value);
        stats.downWorst = std::max(stats.downWorst, std::fabs(got - y.value) / y.magnitude);
        ++stats.outputs;
        const double weight = routing[route];
        expected[n] += weight * got;
        magnitude[n] += std::fabs(weight * got);
      }
    }
    for (uint32_t c = 0; c < kHidden; ++c) {
      const double e = std::ldexp(magnitude[c] + std::fabs(b.residual[uint64_t{r} * kHidden + c]), -21);
      const float got = float(output[uint64_t{r} * kHidden + c]);
      require(inside(got, expected[c], e), label + ": output of row " + std::to_string(r) + " column " +
                                               std::to_string(c) + " differs from the weighted expert sum");
    }
  }
  std::vector<uint16_t> result(uint64_t{rows} * kHidden);
  std::memcpy(result.data(), output, result.size() * 2);
  return result;
}

int moe(MetalBackend &backend) {
  int failures = 0;
  Buffers b;
  b.input = activations(uint64_t{kMaximumRows} * kHidden);
  b.residual = activations(uint64_t{kMaximumRows} * kHidden);
  for (float &v : b.residual) v = bf16(0.1f * v);
  b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
  b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
  b.moe.output = zeros(backend, uint64_t{kMaximumRows} * kHidden * 2, "moe-output");
  const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate, QuantFamily::Gguf};
  for (int f = 0; f < FMT_COUNT; ++f) {
    const Model m = makeModel(backend, f);
    std::map<std::pair<uint32_t, uint32_t>, GateUp> products;
    std::string formats;
    for (Fmt format : m.formats) formats += std::string(formats.empty() ? "" : "/") + fmtName(format);
    for (const MoeGgufTile tile : {MoeGgufTile::Staged, MoeGgufTile::Register}) {
      Stats stats;
      std::vector<uint16_t> widest;
      for (uint32_t lanes = 4; lanes >= 1; --lanes) {
        const MoePlan plan = MoE::decodePlan(
            shape, lanes, MoeConfig{MoeExpertTile::M8, splash::ops::kMoeRouteWideRows, MoeExpertSimdgroups::Eight, tile});
        const std::string label = formats + (tile == MoeGgufTile::Register ? " register" : " staged") + " decode B" +
                                  std::to_string(lanes);
        const std::vector<uint16_t> rows = runPlan(backend, m, b, plan, tile == MoeGgufTile::Staged, products, stats, label);
        // A row's result depends on its own routes only, not on the lanes it
        // is batched with (the tile rows of one expert are independent).
        if (widest.empty()) widest = rows;
        else if (!std::equal(rows.begin(), rows.end(), widest.begin())) {
          printf("  %s: rows differ from the four-lane dispatch FAIL\n", label.c_str());
          ++failures;
        }
      }
      // Prefill chunks on the same 8-row tiles (ExecutionPlans::moePrefill).
      for (const uint32_t chunk : {kMaximumRows, 27u, 9u}) {
        const MoePlan plan = MoE::prefillPlan(
            shape, chunk, MoeConfig{MoeExpertTile::M8, splash::ops::kMoeRouteWideRows, MoeExpertSimdgroups::Eight, tile});
        const std::string label = formats + (tile == MoeGgufTile::Register ? " register" : " staged") + " prefill rows=" +
                                  std::to_string(chunk);
        const std::vector<uint16_t> rows = runPlan(backend, m, b, plan, tile == MoeGgufTile::Staged, products, stats, label);
        if (!std::equal(rows.begin(), rows.begin() + std::min(rows.size(), widest.size()), widest.begin())) {
          printf("  %s: rows differ from the four-lane dispatch FAIL\n", label.c_str());
          ++failures;
        }
      }
      printf("%-20s %s decode B1-4, prefill 263/27/9: gate/up %.2f%% and down %.2f%% of outputs differ from bf16(fp64), "
             "errors at most %.1e/%.1e of sum|x w| %s\n",
             formats.c_str(), tile == MoeGgufTile::Register ? "register" : "staged  ",
             100.0 * stats.gateUpFlips / (stats.outputs / 2), 100.0 * stats.downFlips / stats.outputs,
             stats.gateUpWorst, stats.downWorst, "ok");
    }
    Stats stats;
    std::vector<uint16_t> widest;
    for (const uint32_t rows : {kMaximumRows, 33u, 9u}) {
      const MoePlan plan = MoE::prefillPlan(shape, rows);
      const std::string label = formats + " prefill rows=" + std::to_string(rows);
      const std::vector<uint16_t> result = runPlan(backend, m, b, plan, true, products, stats, label);
      if (widest.empty()) widest = result;
      else if (!std::equal(result.begin(), result.end(), widest.begin())) {
        printf("  %s: rows differ from the %u-row chunk FAIL\n", label.c_str(), kMaximumRows);
        ++failures;
      }
    }
    printf("%-20s staged   prefill 263/33/9 (64-row tiles): gate/up %.2f%% and down %.2f%% of outputs differ from bf16(fp64), "
           "errors at most %.1e/%.1e of sum|x w| ok\n",
           formats.c_str(), 100.0 * stats.gateUpFlips / (stats.outputs / 2), 100.0 * stats.downFlips / stats.outputs,
           stats.gateUpWorst, stats.downWorst);
  }
  return failures;
}

// ---------------------------------------------------------------- timing
// time [rounds]: one MoE layer at the 35B shape (hidden 2048, 256 experts, top 8, intermediate 512), GGUF Q4_K
// gate/up, Q5_K down and a Q8_0 shared expert against the affine Q4 layer, both on the device's plans (and the
// other GGUF tile): a 2048-row prefill chunk, decode B1-B4 and prefill chunks of their rows; medians of GPU time
// per layer. Every row routes to 8 experts of a pool of 24 per request lane (decode, short chunks) or of all 256
// (the long chunk), identically for both formats: expert e scores 4 x[e], so the first 256 inputs pick the routes.
// The weights exceed the system cache, so each layer streams its experts from DRAM.
int timing(MetalBackend &backend, uint32_t rounds) {
  constexpr uint32_t H = 2048, I = 512, E = 256, kRowsMax = 2048;
  const MoeShape affineShape{H, E, 8, I}, ggufShape{H, E, 8, I, QuantFamily::Gguf};
  std::mt19937 local(9);
  // Affine: random Q4 slabs with finite scales, the router and shared gate of the routing fixture.
  const auto affineExperts = [&](uint32_t experts, uint32_t n, uint32_t k) {
    const uint64_t elements = uint64_t{n} * k, stride = elements / 2 + elements / 16;
    MetalBuffer packed = backend.allocateBuffer(experts * stride, BufferStorage::Shared, "affine-experts");
    auto *bytes = static_cast<uint8_t *>(packed.contents());
    for (uint64_t e = 0; e < experts; ++e) {
      uint8_t *slab = bytes + e * stride;
      for (uint64_t i = 0; i < elements / 2; ++i) slab[i] = uint8_t(local());
      auto *parameters = reinterpret_cast<__bf16 *>(slab + elements / 2);
      for (uint64_t i = 0; i < elements / 32; ++i) parameters[i] = __bf16(i < elements / 64 ? 0.01f : -0.05f);
    }
    return splash::ops::ExpertQ4Projection{packed, experts, n, k, stride};
  };
  const auto affineRouter = [&](bool routes) {
    const uint64_t elements = uint64_t{256} * H;
    MetalBuffer weights = zeros(backend, elements, "router-weights"), scales = zeros(backend, elements / 32, "router-scales"),
                biases = zeros(backend, elements / 32, "router-biases");
    auto *w = static_cast<uint8_t *>(weights.contents());
    auto *sc = static_cast<__bf16 *>(scales.contents());
    for (uint32_t e = 0; e < 256; ++e) {
      if (!routes) break;
      w[(uint64_t(e / 64) * 256 + e) * 64 + e % 64] = 1;
      sc[(e / 64) * 256 + e] = __bf16(4.0f);
    }
    return splash::ops::Q8Projection{weights, scales, biases, 256, H};
  };
  MoeWeights affine;
  affine.router = affineRouter(true);
  affine.sharedExpertGate = affineRouter(false);
  affine.expertGate = affineExperts(E, I, H);
  affine.expertUp = affineExperts(E, I, H);
  affine.expertDown = affineExperts(E, H, I);
  affine.sharedGate = affineExperts(1, I, H);
  affine.sharedUp = affineExperts(1, I, H);
  affine.sharedDown = affineExperts(1, H, I);
  // GGUF: the same routing in an F32 router, experts in the 35B UD-Q4_K_M formats.
  const auto planes = [&](Fmt f, uint32_t rows, uint32_t k) {
    std::uniform_real_distribution<float> d(0.0005f, 0.004f);
    const std::vector<uint8_t> native = makeNative(f, rows, k, local, [&] { return f2h(d(local)); });
    const Packed packed = repack(f, native, rows, k, nullptr);
    const QuantFormat &layout = kQuantFormats[f];
    GgufSegment s;
    s.plane0 = upload(backend, packed.w0.data(), packed.w0.size(), "gguf-plane0");
    if (layout.plane1_bytes) s.plane1 = upload(backend, packed.w1.data(), packed.w1.size(), "gguf-plane1");
    s.meta = upload(backend, packed.meta.data(), packed.meta.size(), "gguf-meta");
    s.type = layout.ggml_type;
    s.outputSize = rows;
    s.inputSize = k;
    s.p0 = layout.plane0_bytes;
    s.p1 = layout.plane1_bytes;
    s.metaBytes = layout.meta_bytes;
    s.metaGroups = layout.meta_groups;
    s.formatId = f;
    s.format = layout.name;
    return s;
  };
  std::vector<float> router(uint64_t{E} * H, 0.0f), sharedGate(H, 0.0f);
  for (uint32_t e = 0; e < E; ++e) router[uint64_t{e} * H + e] = 4.0f;
  GgufSegment routerSegment, sharedGateSegment;
  routerSegment.plane0 = upload(backend, router.data(), router.size() * 4, "router");
  sharedGateSegment.plane0 = upload(backend, sharedGate.data(), sharedGate.size() * 4, "shared-gate");
  for (GgufSegment *s : {&routerSegment, &sharedGateSegment}) {
    s->type = GGUF_TYPE_F32;
    s->inputSize = H;
    s->formatId = GGUF_FMT_COUNT;
  }
  routerSegment.outputSize = E;
  sharedGateSegment.outputSize = 1;
  MoeWeights gguf;
  gguf.gguf = GgufMoeWeights{routerSegment, sharedGateSegment,
                             {planes(Q4K, E * I, H), planes(Q80, I, H)},
                             {planes(Q4K, E * I, H), planes(Q80, I, H)},
                             {planes(Q5K, E * H, I), planes(Q80, H, I)}};
  // Rows route to 8 of a pool of 24 experts per lane of 8 rows (decode) or of all 256 (prefill).
  const auto input = [&](uint32_t rows, uint32_t pool) {
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<float> x(uint64_t{rows} * H);
    std::vector<uint32_t> experts(E);
    std::iota(experts.begin(), experts.end(), 0u);
    for (uint32_t r = 0; r < rows; ++r) {
      if (r % 8 == 0) std::shuffle(experts.begin(), experts.end(), local);
      std::vector<uint32_t> candidates(experts.begin(), experts.begin() + pool);
      std::shuffle(candidates.begin(), candidates.end(), local);
      for (uint32_t k = 0; k < H; ++k) x[uint64_t{r} * H + k] = bf16(k < E ? 0.05f * unit(local) : unit(local));
      for (uint32_t rank = 0; rank < 8; ++rank) x[uint64_t{r} * H + candidates[rank]] = bf16(1.0f - 0.05f * rank);
    }
    return x;
  };
  Buffers b;
  b.moe.residual = zeros(backend, uint64_t{kRowsMax} * H * 2, "residual");
  b.moe.output = zeros(backend, uint64_t{kRowsMax} * H * 2, "output");
  const splash::ops::ExecutionPlans plans(backend.capabilities());
  const auto time = [&](const MoeWeights &weights, const MoePlan &plan) {
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b.moe, weights, plan);
    std::vector<double> samples;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double seconds = backend.submitCommand(graph.dispatches()).gpuSeconds;
      if (i) samples.push_back(seconds * 1e3);   // the first round warms the pipelines
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  const MoeGgufTile device = splash::ops::moeGgufTile(backend.capabilities().appleGpuFamily);
  const MoeGgufTile other = device == MoeGgufTile::Register ? MoeGgufTile::Staged : MoeGgufTile::Register;
  printf("%s, GPU family %u, %u cores: median GPU ms per MoE layer of %u rounds\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, rounds);
  // Prefill chunks first: they also bring the GPU clocks up for the short decode layers.
  b.moe.input = bfloatBuffer(backend, input(kRowsMax, E), "input");
  const double affinePrefill = time(affine, plans.moePrefill(affineShape, kRowsMax));
  printf("  prefill %u rows: affine %.3f  gguf %.3f\n", kRowsMax, affinePrefill, time(gguf, plans.moePrefill(ggufShape, kRowsMax)));
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    b.moe.input = bfloatBuffer(backend, input(lanes * 8, 24), "input");
    MoeConfig config = plans.moeDecode(ggufShape, lanes).config();
    config.ggufTile = other;
    const double a = time(affine, plans.moeDecode(affineShape, lanes)), g = time(gguf, plans.moeDecode(ggufShape, lanes));
    const double o = time(gguf, MoE::decodePlan(ggufShape, lanes, config));
    const double ap = time(affine, plans.moePrefill(affineShape, lanes * 8)), gp = time(gguf, plans.moePrefill(ggufShape, lanes * 8));
    printf("  decode B%u: affine %.3f  gguf %s %.3f  gguf %s %.3f  | prefill chunk of %u rows: affine %.3f  gguf %.3f\n", lanes, a,
           device == MoeGgufTile::Register ? "register" : "staged", g, other == MoeGgufTile::Register ? "register" : "staged", o,
           lanes * 8, ap, gp);
  }
  // Where the time goes: every dispatch replayed as its own command.
  for (const auto &[label, weights, plan] : {std::tuple{"affine prefill", &affine, plans.moePrefill(affineShape, kRowsMax)},
                                             std::tuple{"gguf prefill", &gguf, plans.moePrefill(ggufShape, kRowsMax)},
                                             std::tuple{"affine B1", &affine, plans.moeDecode(affineShape, 1)},
                                             std::tuple{"gguf B1", &gguf, plans.moeDecode(ggufShape, 1)},
                                             std::tuple{"affine B4", &affine, plans.moeDecode(affineShape, 4)},
                                             std::tuple{"gguf B4", &gguf, plans.moeDecode(ggufShape, 4)}}) {
    b.moe.input = bfloatBuffer(backend, input(plan.rows(), plan.rows() == kRowsMax ? E : 24), "input");
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b.moe, *weights, plan);
    backend.setDispatchProfiling(true);
    for (uint32_t i = 0; i < rounds; ++i) static_cast<void>(backend.submitCommand(graph.dispatches()));
    backend.setDispatchProfiling(false);
    std::map<std::string, double> spent;
    for (const auto &t : backend.takeDispatchProfile()) spent[t.pipelineName] += t.gpuSeconds * 1e3 / rounds;
    printf("  %s:", label);
    for (const auto &[name, ms] : spent) printf(" %s %.3f", name.c_str(), ms);
    printf("\n");
  }
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2 || (argc > 2 && std::string(argv[2]) != "time")) {
      std::cerr << "usage: gguf_moe_test <metallib> [time [rounds]]\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      if (argc > 2) return timing(backend, argc > 3 ? std::stoul(argv[3]) : 20);
      const int failures = floatProjection(backend) + floatSegments(backend) + moe(backend);
      if (failures) {
        std::cerr << "gguf_moe_test: " << failures << " failures\n";
        return 1;
      }
      std::cout << "gguf_moe_test: PASS\n";
    } catch (const std::exception &error) {
      std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
