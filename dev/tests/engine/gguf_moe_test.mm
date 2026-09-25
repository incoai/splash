// GGUF sparse MoE (ops::MoE over BlockMoeWeights) and the fp32 projection of
// GGUF float tensors, against fp64 references over GGML's dequantized weights
// (GgufFormatReference.hpp), through the production dispatch code.
// - Float projection, both tiles (fp32 simdgroup MMA, and the neural
//   accelerator on three exact bf16 parts per weight): every row count a
//   decode or prefill dispatch takes (with ragged tile tails), the router and
//   alpha/beta widths, bf16 and fp32 destinations at a column offset of a
//   wider row; neighbours stay untouched.
// - Float segments of a fused GGUF projection (Linear), the 35B's and the
//   27B's F32 alpha/beta widths, on either float tile: the quantized
//   segments' outputs unchanged, the padding past the segments unwritten.
// - MoE: every GGUF plan (the staged 8- and 32-row tiles and the Apple9
//   register tile, whatever GPU runs the test; decode steps and prefill
//   chunks) for every format, gate, up and down in three formats and the
//   shared expert in three more; routes and weights against the fp64 router,
//   each pass inside the fp64 interval its numerics allow, the block's output,
//   and a row's output bitwise equal at every lane count and chunk of one
//   tile.
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
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
#include <vector>

namespace {

using splash::DeviceCapabilities;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::FloatOutput;
using splash::ops::FloatTile;
using splash::ops::BlockExpertProjection;
using splash::ops::BlockMoeWeights;
using splash::ops::QuantizedSegment;
using splash::ops::MoE;
using splash::ops::MoeBuffers;
using splash::ops::MoeConfig;
using splash::ops::MoeScratchField;
using splash::ops::kMoeScratchFields;
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
using splash::ops::Linear;
using splash::ops::PreparedInput;
using splash::ops::Projection;
using splash::ops::WeightLayout;
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

// The segments of a repacked [rows, K] tensor in format f and of [rows, K] floats.
QuantizedSegment planeSegment(MetalBackend &backend, Fmt f, const Packed &planes, uint32_t rows, uint32_t K) {
  return QuantizedSegment::planes(
      f, rows, K, upload(backend, planes.w0.data(), planes.w0.size(), "gguf-plane0"),
      kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1.data(), planes.w1.size(), "gguf-plane1")
                                    : MetalBuffer{},
      upload(backend, planes.meta.data(), planes.meta.size(), "gguf-meta"));
}
QuantizedSegment floatSegment(MetalBackend &backend, const std::vector<float> &values, uint32_t rows,
                              uint32_t K) {
  return QuantizedSegment::floats(rows, K, upload(backend, values.data(), values.size() * sizeof(float),
                                                  "gguf-floats"));
}

// A GGUF tensor [rows, K]: its segment and GGML's fp32 values of it.
struct Tensor {
  QuantizedSegment segment;
  std::vector<float> values;
  uint32_t rows = 0, columns = 0;
  [[nodiscard]] const float *row(uint64_t n) const { return values.data() + n * columns; }
};

Tensor quantized(MetalBackend &backend, Fmt f, uint32_t rows, uint32_t K) {
  const std::vector<uint8_t> native = makeNative(f, rows, K, rng);
  Tensor t;
  t.rows = rows;
  t.columns = K;
  t.segment = planeSegment(backend, f, repack(f, native, rows, K, &t.values), rows, K);
  return t;
}

Tensor floating(MetalBackend &backend, uint32_t rows, uint32_t K, float scale) {
  std::normal_distribution<float> normal(0.0f, scale);
  Tensor t;
  t.rows = rows;
  t.columns = K;
  t.values.resize(uint64_t{rows} * K);
  for (float &v : t.values) v = normal(rng);
  t.segment = floatSegment(backend, t.values, rows, K);
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

// fp32 accumulation of n exact products: at most n u sum |x w| (u = 2^-24).
// The simdgroup float tile adds K products, the neural accelerator tile the
// 3 K products of the weights' bf16 parts.
double floatBound(const Dot &d, uint32_t K, FloatTile tile = FloatTile::Simdgroup) {
  return std::ldexp(d.magnitude * K * (tile == FloatTile::NeuralAccelerator ? 3 : 1), -24) + 1e-30;
}

bool inside(float got, double exact, double bound) {
  return got >= bf16(exact - bound) && got <= bf16(exact + bound);
}

// ---------------------------------------------------------------- float projection
// The input holds exactly the dispatch's rows, so a tile reading past them
// would fail shader validation.
int floatProjection(MetalBackend &backend) {
  int failures = 0;
  for (const uint32_t K : {512u, 2048u})
    for (const uint32_t N : {16u, 64u, 256u}) {
      const Tensor w = floating(backend, N, K, 0.05f);
      for (const FloatTile tile : {FloatTile::Simdgroup, FloatTile::NeuralAccelerator}) {
        const bool accelerator = tile == FloatTile::NeuralAccelerator;
        double worst = 0, sum = 0;
        size_t outputs = 0;
        int tileFailures = 0;
        for (const uint32_t rows : {1u, 7u, 8u, 16u, 24u, 32u, 33u, 80u, 263u, 2048u}) {
          if (rows == 2048 && (K != 2048 || N != 256)) continue;   // the router's prefill width
          if (accelerator && rows < 16) continue;
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
            splash::ops::addGgufFloat(graph, input, w.segment, output, rows, stride, offset, type, tile);
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
                const double bound = floatBound(d, K, tile);
                float got;
                if (type == FloatOutput::Float32) {
                  std::memcpy(&got, bytes, 4);
                  outside += !(std::fabs(got - d.value) <= bound);
                  const double error = std::fabs(got - d.value) / (d.magnitude + 1e-30);
                  worst = std::max(worst, error);
                  sum += error;
                  ++outputs;
                } else {
                  __bf16 half;
                  std::memcpy(&half, bytes, 2);
                  got = float(half);
                  outside += !inside(got, d.value, bound);
                }
              }
            if (outside || touched) {
              printf("  float %s K=%u N=%u rows=%u %s: %zu outputs outside the fp32 bound, %zu bytes written outside "
                     "FAIL\n", accelerator ? "accelerator" : "simdgroup", K, N, rows,
                     type == FloatOutput::Float32 ? "f32" : "bf16", outside, touched);
              ++tileFailures;
            }
          }
        }
        printf("float %-11s K=%u N=%3u: rows %u-2048, bf16/f32 at a column offset: |error| / sum|x w| mean %.2e "
               "worst %.2e %s\n", accelerator ? "accelerator" : "simdgroup", K, N, accelerator ? 16u : 1u,
               sum / outputs, worst, tileFailures ? "FAIL" : "ok");
        failures += tileFailures;
      }
    }
  return failures;
}

// A fused projection with a float segment of `floatColumns` columns, as the
// GDN input (qkv | z | alpha-beta) of a GGUF storing alpha/beta as F32 (64
// columns on the 35B, 96 on the 27B): the float projection writes the float
// segment's columns, the quantized kernels the others bit for bit as without
// it, and the padding past the last segment stays unwritten; decode through
// both GGUF decode families (register, staged) at one to four lanes, and
// prefill, where a one-core device takes the neural accelerator float tile
// from 16 rows.
int floatSegments(MetalBackend &backend, uint32_t floatColumns) {
  constexpr uint32_t K = 1024, N = 768, kFloatColumn = 512;
  const uint32_t covered = kFloatColumn + floatColumns;
  const Tensor q80 = quantized(backend, Q80, 256, K), q4k = quantized(backend, Q4K, 256, K);
  const Tensor gates = floating(backend, floatColumns, K, 0.05f);
  const auto at = [](QuantizedSegment s, uint32_t offset) { s.columnOffset = offset; return s; };
  Projection full(N, K, splash::ops::BlockWeights{{at(q80.segment, 0),
      at(q4k.segment, 256), at(gates.segment, kFloatColumn)}});
  Projection quantizedOnly(N, K, splash::ops::BlockWeights{{at(q80.segment, 0), at(q4k.segment, 256)}});
  int failures = 0;
  for (const auto [family, cores] : {std::pair{9u, 0u}, std::pair{10u, 0u}, std::pair{10u, 1u}}) {
    DeviceCapabilities device = backend.capabilities();
    device.appleGpuFamily = family;
    if (cores) device.gpuCoreCount = cores;
    const Linear linear(device);
    const auto check = [&](uint32_t rows, uint32_t storage, const std::string &label,
                           const std::function<void(CommandGraph &, MetalBuffer, const Projection &, MetalBuffer)> &add) {
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
          else if (c >= covered) written += got[i] != 0x7F7F;
          else {
            const Dot d = dot(x.data() + uint64_t{r} * K, gates.row(c - kFloatColumn), K);
            __bf16 value;
            std::memcpy(&value, got + i, 2);
            outside += !inside(float(value), d.value, floatBound(d, K, linear.ggufFloatTile(rows, floatColumns)));
          }
        }
      if (differ || outside || written) {
        printf("  float segment %s family %u, %u cores, %u float columns: %zu quantized outputs differ, %zu float "
               "outputs outside the fp32 bound, %zu padding outputs written FAIL\n", label.c_str(), family,
               device.gpuCoreCount, floatColumns, differ, outside, written);
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
      const LinearPlan plan = linear.plan({matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::None, WeightLayout::Block32},
                                          full);
      const LinearScratch scratch = scratchFor(plan);
      check(lanes * 8, plan.storageRows(), "decode B" + std::to_string(lanes),
            [&](CommandGraph &graph, MetalBuffer input, const Projection &p, MetalBuffer output) {
              splash::ops::LinearDispatchStats stats;
              static_cast<void>(linear.addDecodeBatch(graph, input, p, output, lanes, stats, scratch));
            });
    }
    for (const uint32_t rows : {1u, 24u, 33u, 263u}) {
      const LinearPlan plan =
          linear.plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None, WeightLayout::Block32}, full);
      const LinearScratch scratch = scratchFor(plan);
      check(rows, plan.storageRows(), "prefill rows=" + std::to_string(rows),
            [&](CommandGraph &graph, MetalBuffer input, const Projection &p, MetalBuffer output) {
              linear.addPrefill(graph, input, p, output, {}, rows, scratch);
            });
    }
  }
  printf("float segment: fused Q8_0|Q4_K|F32 (%u float columns) decode B1-4 (register, staged) and prefill "
         "1/24/33/263 rows, both float tiles %s\n", floatColumns, failures ? "FAIL" : "ok");
  return failures;
}

// A projection of float segments alone runs no quantized kernel, so it leaves
// the scratch table as it found it: an Apple9 register plan chained after it
// on the same input (as the draft chains its head and selector) prepares its
// own table and writes what it writes alone, bit for bit.
int floatOnlyChain(MetalBackend &backend) {
  constexpr uint32_t K = 1024, N = 256;
  const Tensor gates = floating(backend, N, K, 0.05f), q4k = quantized(backend, Q4K, N, K);
  const Projection floats(N, K, splash::ops::BlockWeights{{gates.segment}});
  const Projection blocks(N, K, splash::ops::BlockWeights{{q4k.segment}});
  DeviceCapabilities device = backend.capabilities();
  device.appleGpuFamily = 9;
  const Linear linear(device);
  int failures = 0;
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const uint64_t rows = lanes * 8, bytes = rows * N * 2;
    const LinearScratchSize size = linear.decodePlan(blocks, lanes).scratchSize();
    const LinearScratch scratch{zeros(backend, size.input, "chain-table"), zeros(backend, size.sums, "chain-sums"),
                                zeros(backend, size.partials, "chain-partials"),
                                zeros(backend, size.counters, "chain-counters")};
    const MetalBuffer input = bfloatBuffer(backend, activations(rows * K), "chain-input");
    const MetalBuffer scores = zeros(backend, bytes, "chain-scores"), chained = zeros(backend, bytes, "chain-output"),
                      alone = zeros(backend, bytes, "chain-reference");
    splash::ops::LinearDispatchStats stats;
    CommandGraph graph, reference;
    const PreparedInput prepared = linear.addDecodeBatch(graph, input, floats, scores, lanes, stats, scratch);
    static_cast<void>(linear.addDecodeBatch(graph, input, blocks, chained, lanes, stats, scratch, prepared));
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    static_cast<void>(linear.addDecodeBatch(reference, input, blocks, alone, lanes, stats, scratch));
    static_cast<void>(backend.submitCommand(reference.dispatches()));
    if (std::memcmp(chained.contents(), alone.contents(), bytes)) {
      printf("  float-only projection B%u: a register plan chained after it read another table FAIL\n", lanes);
      ++failures;
    }
  }
  printf("float-only projection: a register plan chained after it prepares its own table, B1-4 %s\n",
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
  BlockMoeWeights gguf;
  gguf.router = m.router.segment;
  gguf.sharedScalarGate = m.sharedGate.segment;
  gguf.gate = {m.routed[0].segment, m.shared[0].segment};
  gguf.up = {m.routed[1].segment, m.shared[1].segment};
  gguf.down = {m.routed[2].segment, m.shared[2].segment};
  m.weights = gguf;
  return m;
}

struct Buffers {
  MoeBuffers moe;
  std::vector<float> input, residual;
};

void allocate(MetalBackend &backend, Buffers &b, const MoePlan &plan) {
  const auto &w = plan.workspace();
  for (const MoeScratchField &field : kMoeScratchFields)
    b.moe.scratch.*field.buffer = zeros(backend, w.*field.bytes, "moe-scratch");
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
  const auto *selected = static_cast<const uint32_t *>(b.moe.scratch.selectedExperts.contents());
  const auto *routing = static_cast<const float *>(b.moe.scratch.routingWeights.contents());
  const auto *routeRows = static_cast<const uint32_t *>(b.moe.scratch.routeRows.contents());
  const auto *intermediate = static_cast<const __bf16 *>(b.moe.scratch.expertIntermediate.contents());
  const auto *down = static_cast<const __bf16 *>(b.moe.scratch.expertOutput.contents());
  const auto *output = static_cast<const __bf16 *>(b.moe.output.contents());
  const uint32_t tileRows = plan.tileRows(), tiles = *static_cast<const uint32_t *>(b.moe.scratch.tileCount.contents());
  require(tiles <= plan.maximumTiles(), label + ": tile count exceeds its bound");
  for (uint32_t r = 0; r < rows; ++r) {
    const float *x = b.input.data() + uint64_t{r} * kHidden;
    // Routes: the fp64 top-k by descending score, ascending id; the GPU's fp32
    // scores may reorder only experts closer than their error bounds.
    std::array<double, kExperts> score{}, bound{};
    for (uint32_t e = 0; e < kExperts; ++e) {
      const Dot d = dot(x, m.router.row(e), kHidden);
      score[e] = d.value;
      bound[e] = floatBound(d, kHidden, plan.configuration().ggufRouterTile);
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
  const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate, WeightLayout::Block32};
  for (int f = 0; f < FMT_COUNT; ++f) {
    const Model m = makeModel(backend, f);
    std::map<std::pair<uint32_t, uint32_t>, GateUp> products;
    std::string formats;
    for (Fmt format : m.formats) formats += std::string(formats.empty() ? "" : "/") + fmtName(format);
    for (const MoeGgufTile tile : {MoeGgufTile::Staged, MoeGgufTile::Register}) {
      const int before = failures;
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
             stats.gateUpWorst, stats.downWorst, failures > before ? "FAIL" : "ok");
    }
    // The 32-row tiles, whose 16- and 32-row matmuls both run at 263 rows,
    // with the router on each float tile: a row's result is the same in every
    // chunk on one tile (either tile's scores of a row depend on that row
    // alone).
    for (const FloatTile router : {FloatTile::Simdgroup, FloatTile::NeuralAccelerator}) {
      const int before = failures;
      Stats stats;
      std::vector<uint16_t> widest;
      for (const uint32_t rows : {kMaximumRows, 33u, 16u}) {
        MoeConfig config{MoeExpertTile::M32};
        config.ggufRouterTile = router;
        const MoePlan plan = MoE::prefillPlan(shape, rows, config);
        const std::string label = formats + " prefill rows=" + std::to_string(rows) +
                                  (router == FloatTile::NeuralAccelerator ? " (accelerator router)" : "");
        const std::vector<uint16_t> result = runPlan(backend, m, b, plan, true, products, stats, label);
        if (widest.empty()) widest = result;
        else if (!std::equal(result.begin(), result.end(), widest.begin())) {
          printf("  %s: rows differ from the %u-row chunk FAIL\n", label.c_str(), kMaximumRows);
          ++failures;
        }
      }
      printf("%-20s staged   prefill 263/33/16 (32-row tiles, %s router): gate/up %.2f%% and down %.2f%% of outputs "
             "differ from bf16(fp64), errors at most %.1e/%.1e of sum|x w| %s\n",
             formats.c_str(), router == FloatTile::NeuralAccelerator ? "accelerator" : "simdgroup",
             100.0 * stats.gateUpFlips / (stats.outputs / 2), 100.0 * stats.downFlips / stats.outputs, stats.gateUpWorst,
             stats.downWorst, failures > before ? "FAIL" : "ok");
    }
  }
  return failures;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf_moe_test <metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const int failures = floatProjection(backend) + floatSegments(backend, 64) + floatSegments(backend, 96) +
                           floatOnlyChain(backend) + moe(backend);
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
