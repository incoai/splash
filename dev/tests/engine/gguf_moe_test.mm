// GGUF sparse MoE (ops::MoE over BlockMoeWeights) and the fp32 projection of
// GGUF float tensors, against fp64 references over GGML's dequantized weights
// (GgufFormatReference.hpp), through the production dispatch code.
// - Float projection, both tiles (fp32 simdgroup MMA, and the neural
//   accelerator on three exact bf16 parts per weight): every row count a
//   decode or prefill dispatch takes (with ragged tile tails), the router and
//   alpha/beta widths at K of whole and partial 64-input steps, bf16 and fp32
//   destinations at a column offset of a wider row; neighbours stay
//   untouched. BF16 weights (a router a GGUF stores so) give the outputs of
//   the F32 tensor of their values bit for bit. Two projections of one input
//   (the QSA indexer's query and key) share a dispatch when they share a tile
//   and each gives the outputs of its own dispatch bit for bit.
// - Float segments of a fused GGUF projection (Linear), the 35B's and the
//   27B's F32 alpha/beta widths, on either float tile: the quantized
//   segments' outputs unchanged, the padding past the segments unwritten.
// - MoE: every GGUF plan (the staged 8- and 32-row tiles and the Apple9
//   register tile, whatever GPU runs the test; decode steps and prefill
//   chunks) for every format the tile takes (the register tile all but F16,
//   in rows it takes: quant_register_tensor), gate, up and down in three of
//   those formats and the shared expert in three more, and on the staged
//   tiles, which run gate and up in one pass, also with up in the gate's
//   formats; routes and weights against the fp64 router, each pass inside the
//   fp64 interval its numerics allow, the block's output, a row's output
//   bitwise equal at every lane count and chunk of one tile where its routes
//   run the same tile kinds (the staged expert passes run an expert's only
//   tile of at most MOE_GGUF_COLUMN_ROWS live rows on the column tile; the
//   rows so compared are counted) and at every repeat of a run (a set's
//   first plan also on fresh scratch, where every row must compare), the
//   output of the live tiles' grids bitwise that of every tile's with two
//   gate/up passes, and the grouping: tiles as each expert's routes fill
//   them, every route's grouped row naming it back, the padding a tile's
//   matmul reads marked empty and the rows past it unwritten. Concentrated and
//   skewed router loads on the staged 32-row tiles of one format, at ragged
//   row counts. A verify step that routes its lanes' verify rows alone.
// Streamed experts (model::ExpertStore) run in expert_store_test.mm.
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/Linear.hpp"
#include "../../../runtime/ops/MoE.hpp"
#include "GgufFormatReference.hpp"
#include "MoeExtents.hpp"
#include "MoeFixture.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::DeviceCapabilities;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::FloatOutput;
using splash::ops::FloatProjection;
using splash::ops::FloatTile;
using splash::ops::BlockExpertProjection;
using splash::ops::BlockMoeWeights;
using splash::ops::QuantizedSegment;
using splash::ops::MoE;
using splash::ops::MoeConfig;
using splash::ops::MoeScratch;
using splash::ops::MoeExpertTile;
using splash::ops::MoeGgufTile;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using splash::ops::LinearEpilogue;
using splash::ops::LinearMatrix;
using splash::ops::LinearPhase;
using splash::ops::LinearPlan;
using splash::ops::LinearScratch;
using splash::ops::LinearScratchSize;
using splash::ops::Linear;
using splash::ops::PreparedInput;
using splash::ops::Projection;
using namespace gguf_reference;
using namespace moe_fixture;

// K = 1024 on the hidden side (16 spans, four 256-input coefficient units)
// and 512 on the intermediate side, as the 35B's experts.
// gguf-moe-e512 builds the test at Qwen3.8-Flash-Next's routing: 512
// experts, top 10, a 640-wide intermediate.
#ifndef GGUF_MOE_HIDDEN
#define GGUF_MOE_HIDDEN 1024
#define GGUF_MOE_INTERMEDIATE 512
#define GGUF_MOE_EXPERTS 16
#define GGUF_MOE_TOP_K 4
#endif
constexpr uint32_t kHidden = GGUF_MOE_HIDDEN;
constexpr uint32_t kIntermediate = GGUF_MOE_INTERMEDIATE;
constexpr uint32_t kExperts = GGUF_MOE_EXPERTS;
constexpr uint32_t kTopK = GGUF_MOE_TOP_K;
constexpr uint32_t kRoutes = kTopK + 1;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition) fail(message);
}

double silu(double value) { return value / (1.0 + std::exp(-value)); }
// Where silu has its minimum (silu' = 0).
constexpr double kSiluArgmin = -1.2784645427610738;

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
// would fail shader validation. The simdgroup tile reads K in steps of 64
// inputs, zeros past a row's end: K = 96 and 2080 end in part of a step, and
// K = 96 leaves six of its eight K parts without one.
int floatProjection(MetalBackend &backend) {
  int failures = 0;
  for (const uint32_t K : {96u, 512u, 2048u, 2080u})
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
            static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
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

// BF16 weights, as some GGUFs store the MoE router: on both tiles, at every
// row count and destination floatProjection takes, the outputs of the F32
// tensor of their values bit for bit, the bytes outside the tile untouched.
int bfloat16Weights(MetalBackend &backend) {
  int failures = 0;
  for (const uint32_t K : {512u, 2048u})
    for (const uint32_t N : {16u, 256u}) {
      std::normal_distribution<float> normal(0.0f, 0.05f);
      std::vector<float> values(uint64_t{N} * K);
      for (float &v : values) v = bf16(normal(rng));
      const std::vector<__bf16> bits(values.begin(), values.end());
      const QuantizedSegment f32 = floatSegment(backend, values, N, K);
      const QuantizedSegment b16 = QuantizedSegment::floats(
          N, K, upload(backend, bits.data(), bits.size() * 2, "gguf-bf16"), QuantizedSegment::kBFloat16);
      for (const FloatTile tile : {FloatTile::Simdgroup, FloatTile::NeuralAccelerator}) {
        const bool accelerator = tile == FloatTile::NeuralAccelerator;
        int tileFailures = 0;
        for (const uint32_t rows : {1u, 7u, 16u, 33u, 263u, 2048u}) {
          if (rows == 2048 && (K != 2048 || N != 256)) continue;
          if (accelerator && rows < 16) continue;
          const MetalBuffer input = bfloatBuffer(backend, activations(uint64_t{rows} * K), "float-input");
          for (const FloatOutput type : {FloatOutput::BFloat16, FloatOutput::Float32}) {
            const uint32_t stride = N + 96, offset = 32;
            const uint64_t bytes = uint64_t{rows + 1} * stride * (type == FloatOutput::Float32 ? 4 : 2);
            const auto run = [&](const QuantizedSegment &weights) {
              MetalBuffer output = backend.allocateBuffer(bytes, BufferStorage::Shared, "float-output");
              std::memset(output.contents(), 0x7F, bytes);
              CommandGraph graph;
              splash::ops::addGgufFloat(graph, input, weights, output, rows, stride, offset, type, tile);
              static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
              return output;
            };
            const MetalBuffer want = run(f32), got = run(b16);
            if (std::memcmp(want.contents(), got.contents(), bytes)) {
              printf("  float %s K=%u N=%u rows=%u %s: BF16 weights' outputs differ from the F32 tensor's FAIL\n",
                     accelerator ? "accelerator" : "simdgroup", K, N, rows,
                     type == FloatOutput::Float32 ? "f32" : "bf16");
              ++tileFailures;
            }
          }
        }
        printf("float %-11s K=%u N=%3u: BF16 weights give the F32 tensor's outputs bit for bit, bf16/f32 %s\n",
               accelerator ? "accelerator" : "simdgroup", K, N, tileFailures ? "FAIL" : "ok");
        failures += tileFailures;
      }
    }
  return failures;
}

// Two float projections of one input, each into a destination of its own
// stride and offset and on a tile of its own, as the QSA indexer projects its
// query and key: at decode and prefill row counts and bf16 and fp32
// destinations, one dispatch when they share a value type and a tile and two
// otherwise, each projection's outputs and the bytes around them those of its
// own dispatch bit for bit.
int floatPairs(MetalBackend &backend) {
  constexpr uint32_t K = 512, kQueryStride = 96, kKeyStride = 40;
  constexpr std::array<FloatTile, 2> kTiles{FloatTile::Simdgroup, FloatTile::NeuralAccelerator};
  const auto name = [](FloatTile tile) { return tile == FloatTile::NeuralAccelerator ? "accelerator" : "simdgroup"; };
  const Tensor query = floating(backend, 64, K, 0.05f), key = floating(backend, 24, K, 0.05f);
  const std::vector<__bf16> bits(key.values.begin(), key.values.end());
  const QuantizedSegment bfloatKey = QuantizedSegment::floats(
      24, K, upload(backend, bits.data(), bits.size() * 2, "gguf-bf16"), QuantizedSegment::kBFloat16);
  int failures = 0;
  for (const uint32_t rows : {1u, 16u, 33u, 263u}) {
    const MetalBuffer input = bfloatBuffer(backend, activations(uint64_t{rows} * K), "pair-input");
    for (const FloatTile queryTile : kTiles)
      for (const FloatTile keyTile : kTiles) {
        if (rows < 16 && (queryTile != FloatTile::Simdgroup || keyTile != FloatTile::Simdgroup)) continue;
        for (const FloatOutput type : {FloatOutput::BFloat16, FloatOutput::Float32})
          for (const QuantizedSegment *second : {&key.segment, &bfloatKey}) {
            // One row more than the dispatch writes, whose sentinel must survive.
            const auto destination = [&](uint32_t stride) {
              MetalBuffer output = backend.allocateBuffer(
                  uint64_t{rows + 1} * stride * (type == FloatOutput::Float32 ? 4 : 2), BufferStorage::Shared,
                  "pair-output");
              std::memset(output.contents(), 0x7F, output.sizeBytes());
              return output;
            };
            const std::array<MetalBuffer, 4> outputs{destination(kQueryStride), destination(kKeyStride),
                                                     destination(kQueryStride), destination(kKeyStride)};
            const std::array<FloatProjection, 2> pair{
                FloatProjection{&query.segment, outputs[0], kQueryStride, 32, queryTile},
                FloatProjection{second, outputs[1], kKeyStride, 8, keyTile}};
            CommandGraph together, apart;
            splash::ops::addGgufFloat(together, input, pair, rows, type);
            splash::ops::addGgufFloat(apart, input, query.segment, outputs[2], rows, kQueryStride, 32, type,
                                      queryTile);
            splash::ops::addGgufFloat(apart, input, *second, outputs[3], rows, kKeyStride, 8, type, keyTile);
            const size_t dispatches = together.dispatches().size();
            for (const CommandGraph *graph : {&together, &apart})
              static_cast<void>(backend.submitCommandAsync(graph->dispatches()).wait());
            if (dispatches != (second == &key.segment && queryTile == keyTile ? 1u : 2u) ||
                std::memcmp(outputs[0].contents(), outputs[2].contents(), outputs[0].sizeBytes()) ||
                std::memcmp(outputs[1].contents(), outputs[3].contents(), outputs[1].sizeBytes())) {
              printf("  float pair rows=%u %s, query %s, %s key %s: %zu dispatches, outputs %s FAIL\n", rows,
                     type == FloatOutput::Float32 ? "f32" : "bf16", name(queryTile),
                     second == &key.segment ? "F32" : "BF16", name(keyTile), dispatches,
                     "differ from each projection's own dispatch");
              ++failures;
            }
          }
      }
  }
  printf("float pairs: one dispatch of one value type and tile, two otherwise, each projection's outputs as alone %s\n",
         failures ? "FAIL" : "ok");
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
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
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
      const LinearPlan plan = linear.plan({matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::None}, full);
      const LinearScratch scratch = scratchFor(plan);
      check(lanes * 8, plan.storageRows(), "decode B" + std::to_string(lanes),
            [&](CommandGraph &graph, MetalBuffer input, const Projection &p, MetalBuffer output) {
              static_cast<void>(linear.add(graph, {.input = input, .output = output, .scratch = scratch}, p,
                                           linear.decodePlan(p, lanes)));
            });
    }
    for (const uint32_t rows : {1u, 24u, 33u, 263u}) {
      const LinearPlan plan = linear.plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None}, full);
      const LinearScratch scratch = scratchFor(plan);
      check(rows, plan.storageRows(), "prefill rows=" + std::to_string(rows),
            [&](CommandGraph &graph, MetalBuffer input, const Projection &p, MetalBuffer output) {
              linear.addPrefill(graph, input, p, output, rows, scratch);
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
    const LinearPlan plan = linear.decodePlan(blocks, lanes);
    const LinearScratchSize size = plan.scratchSize();
    const LinearScratch scratch{zeros(backend, size.input, "chain-table"), zeros(backend, size.sums, "chain-sums"),
                                zeros(backend, size.partials, "chain-partials"),
                                zeros(backend, size.counters, "chain-counters")};
    const MetalBuffer input = bfloatBuffer(backend, activations(rows * K), "chain-input");
    const MetalBuffer scores = zeros(backend, bytes, "chain-scores"), chained = zeros(backend, bytes, "chain-output"),
                      alone = zeros(backend, bytes, "chain-reference");
    CommandGraph graph, reference;
    const PreparedInput prepared =
        linear.add(graph, {.input = input, .output = scores, .scratch = scratch}, floats, linear.decodePlan(floats, lanes));
    static_cast<void>(
        linear.add(graph, {.input = input, .output = chained, .scratch = scratch, .prepared = prepared}, blocks, plan));
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    static_cast<void>(linear.add(reference, {.input = input, .output = alone, .scratch = scratch}, blocks, plan));
    static_cast<void>(backend.submitCommandAsync(reference.dispatches()).wait());
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
// The fp64 gate and up products of (row, expert), shared by every plan.
struct GateUp {
  std::vector<Dot> gate, up;
};

struct Stats {
  size_t outputs = 0, gateUpFlips = 0, downFlips = 0;
  // The worst |error| relative to the magnitudes its products sum (sum |x w|).
  double gateUpWorst = 0, downWorst = 0;
};

// The dispatches of `graph` on the grids of their bounds, every tile a step
// could fill, with a gate/up pass split into the gate and up passes
// (moe_expert_gguf_m<rows>_a and _g), the gate parked in expertOutput; the
// passes' parameters live in `passes`.
std::vector<ComputeDispatch> fullGridsTwoPasses(const CommandGraph &graph, const MoeScratch &scratch,
                                                std::deque<MoeGgufExpertParams> &passes) {
  std::vector<ComputeDispatch> dispatches;
  for (ComputeDispatch dispatch : graph.dispatches()) {
    dispatch.indirectThreadgroups = {};
    const size_t merged = dispatch.pipelineName.rfind("_gate_up");
    if (merged == std::string::npos) {
      dispatches.push_back(std::move(dispatch));
      continue;
    }
    // Input, tiles, tile count, gate's and up's routed and shared planes,
    // output, slot table; a pass binds one projection's planes, its output,
    // the gate and the slot table.
    for (const bool up : {false, true}) {
      ComputeDispatch pass = dispatch;
      pass.pipelineName = dispatch.pipelineName.substr(0, merged) + (up ? "_g" : "_a");
      pass.buffers.resize(3);
      for (uint32_t plane = 0; plane < 6; ++plane)
        pass.buffers.push_back({3 + plane, dispatch.buffers[(up ? 9 : 3) + plane].buffer});
      pass.buffers.push_back({9, up ? dispatch.buffers[15].buffer : scratch.expertOutput});
      pass.buffers.push_back({10, scratch.expertOutput});
      pass.buffers.push_back({11, dispatch.buffers[16].buffer});
      // The pass's own parameters: the passes' sizes and this pass's formats.
      MoeGgufGateUpParams both;
      std::memcpy(&both, dispatch.bytes[0].data, sizeof(both));
      const MoeGgufExpertParams &params = passes.emplace_back(MoeGgufExpertParams{
          both.input_size, both.output_size, both.experts, up ? both.up_routed_format : both.gate_routed_format,
          up ? both.up_shared_format : both.gate_shared_format});
      pass.bytes[0] = {12, &params, sizeof(params)};
      pass.threadgroups.x = dispatch.threadgroups.x * GGUF_STAGED_COLUMNS / GGUF_TILE_COLUMNS;
      dispatches.push_back(std::move(pass));
    }
  }
  return dispatches;
}

// The byte every grouped route and grouped input holds before a run.
constexpr uint8_t kCanary = 0xa5;

bool untouched(const void *data, uint64_t bytes) {
  const auto *begin = static_cast<const uint8_t *>(data);
  return std::all_of(begin, begin + bytes, [](uint8_t byte) { return byte == kCanary; });
}

// The grouping of a run whose grouped routes and input held kCanary: as many
// tiles as each expert's routes fill, every route at a grouped row that names
// it back in a tile of its expert, the padding rows a tile's matmul reads
// (moe_matmul_rows: the smallest of 8, 16 and the tile's rows that holds its
// live rows) marked empty, and the rows past them written neither by the
// grouping nor by the gather, but where the router's scores, which come first
// in the grouped input, overwrote them; on the staged tile, whose passes run
// their column tiles (columnTile) over the live rows alone, the gather writes
// no padding row of those.
void checkGrouping(const MoeScratch &scratch, const MoePlan &plan, const std::string &label) {
  const uint32_t rows = plan.rows(), tileRows = plan.tileRows();
  const auto *selected = static_cast<const uint32_t *>(scratch.selectedExperts.contents());
  const auto *routeRows = static_cast<const uint32_t *>(scratch.routeRows.contents());
  const auto *groupedRoutes = static_cast<const uint32_t *>(scratch.groupedRoutes.contents());
  const auto *groupedInput = static_cast<const uint8_t *>(scratch.groupedInput.contents());
  const auto *tiles = static_cast<const MoeTileDescriptor *>(scratch.tileDescriptors.contents());
  const uint32_t tileCount = *static_cast<const uint32_t *>(scratch.tileCount.contents());
  std::array<uint32_t, kExperts + 1> routes{};
  for (uint32_t route = 0; route < rows * kRoutes; ++route) {
    require(selected[route] <= kExperts, label + ": invalid selected expert");
    ++routes[selected[route]];
    const uint32_t grouped = routeRows[route];
    require(grouped < tileCount * tileRows && groupedRoutes[grouped] == route &&
                tiles[grouped / tileRows].expert == selected[route],
            label + ": grouped route mapping does not round-trip");
  }
  uint32_t occupied = 0;
  for (const uint32_t count : routes) occupied += (count + tileRows - 1) / tileRows;
  require(tileCount == occupied, label + ": tile count differs from per-expert occupancy");
  const uint64_t rowBytes = uint64_t{kHidden} * 2,
                 scoreBytes = uint64_t{rows} * MoeShape{kHidden, kExperts, kTopK, kIntermediate}.routerWidth() * 4;
  for (uint32_t tile = 0; tile < tileCount; ++tile) {
    const uint32_t live = tiles[tile].rows;
    require(live > 0 && live <= tileRows, label + ": invalid tile live row count");
    const uint32_t matmul = std::min(tileRows, live <= 8 ? 8u : live <= 16 ? 16u : tileRows);
    const uint32_t read =
        plan.configuration().ggufTile == MoeGgufTile::Staged && moe_column_tile(tiles, tile) ? live : matmul;
    for (uint32_t row = live; row < matmul; ++row)
      require(groupedRoutes[tile * tileRows + row] == UINT32_MAX,
              label + ": padding a tile's matmul reads is not marked empty");
    for (uint32_t row = matmul; row < tileRows; ++row)
      require(untouched(groupedRoutes + uint64_t{tile} * tileRows + row, sizeof(uint32_t)),
              label + ": the grouping wrote past a tile's matmul rows");
    for (uint32_t row = read; row < tileRows; ++row) {
      const uint64_t grouped = uint64_t{tile} * tileRows + row;
      require(grouped * rowBytes < scoreBytes || untouched(groupedInput + grouped * rowBytes, rowBytes),
              label + ": the gather wrote past the rows a tile's passes read");
    }
  }
}

// A run's output rows and, per route, whether its expert tile ran the column tile (columnTile, on the staged expert
// passes).
struct MoeRun {
  std::vector<uint16_t> output;
  std::vector<bool> column;
};
std::vector<bool> columnRoutes(const MoeScratch &scratch, const MoePlan &plan) {
  const uint32_t routes = plan.rows() * kRoutes, tileRows = plan.tileRows();
  const auto *routeRows = static_cast<const uint32_t *>(scratch.routeRows.contents());
  const auto *tiles = static_cast<const MoeTileDescriptor *>(scratch.tileDescriptors.contents());
  std::vector<bool> column(routes);
  if (plan.configuration().ggufTile == MoeGgufTile::Staged)
    for (uint32_t route = 0; route < routes; ++route)
      column[route] = moe_column_tile(tiles, routeRows[route] / tileRows);
  return column;
}
// The first `rows` rows of two runs: rows whose routes ran the same tile kinds are bitwise equal, as a row's output
// depends on its own routes only and a route's on its tile kind only (the rows of one expert tile are independent);
// of the `compared` rows that ran them, `differ` counts the rows that are not. Rows whose routes ran other tile kinds,
// which lie within fp64 in both runs (runPlan), are counted in `acrossKinds`.
struct RowComparison {
  uint32_t compared = 0, differ = 0, acrossKinds = 0;
};
RowComparison compareRows(const MoeRun &a, const MoeRun &b, uint32_t rows) {
  RowComparison c;
  for (uint32_t r = 0; r < rows; ++r) {
    const auto routes = [&](const MoeRun &run) { return run.column.begin() + uint64_t{r} * kRoutes; };
    if (!std::equal(routes(a), routes(a) + kRoutes, routes(b))) {
      ++c.acrossKinds;
      continue;
    }
    const auto row = [&](const MoeRun &run) { return run.output.begin() + uint64_t{r} * kHidden; };
    ++c.compared;
    c.differ += !std::equal(row(a), row(a) + kHidden, row(b));
  }
  return c;
}
// The first `rows` rows of a prefill chunk against the widest chunk and against the chunk before it, which holds more
// rows than it but whose routes run more of its own tile kinds than the widest's do (a lone row's routes run the
// column tile, a wide chunk's mostly the staged tile, so the widest chunk can compare none of a narrow chunk's rows).
RowComparison compareChunks(const MoeRun &run, const MoeRun &widest, const MoeRun &previous, uint32_t rows,
                            const std::string &label) {
  RowComparison sum;
  for (const MoeRun *other : {&widest, &previous}) {
    if (other == &previous && previous.output.size() == widest.output.size()) continue;
    const RowComparison c = compareRows(run, *other, rows);
    if (c.differ)
      printf("  %s: %u rows differ from the %u-row chunk FAIL\n", label.c_str(), c.differ,
             uint32_t(other->output.size() / kHidden));
    sum.compared += c.compared;
    sum.differ += c.differ;
    sum.acrossKinds += c.acrossKinds;
  }
  return sum;
}
// A set's first run against its partner of the same tile kinds by construction: the same plan run again on fresh
// scratch, whose grouping orders each expert's routes anew but routes every row as before, so each route runs the
// tile kind it ran and every row must compare bitwise. The set's runs of other rows compare only the rows whose
// routes happen to run the same kinds, which a shape need not have (a 512-expert block's narrow chunks run the column
// tile where its widest runs staged); this pair keeps the comparison from testing nothing. Returns the failures.
int compareRepeat(const MoeRun &first, const MoeRun &again, const std::string &label) {
  const uint32_t rows = uint32_t(first.output.size() / kHidden);
  const RowComparison c =
      again.output.size() == first.output.size() ? compareRows(again, first, rows) : RowComparison{};
  if (c.compared == rows && !c.differ) return 0;
  printf("  %s: a repeat of the run compared %u of its %u rows bitwise, %u differ FAIL\n", label.c_str(), c.compared,
         rows, c.differ);
  return 1;
}

// One plan's run checked against the fp64 model, gate and up in one pass on the staged tiles; returns its output rows
// and the tile kinds of its routes.
MoeRun runPlan(MetalBackend &backend, const Model &m, Buffers &b, const MoePlan &plan, bool staged,
               std::map<std::pair<uint32_t, uint32_t>, GateUp> &products, Stats &stats, const std::string &label) {
  const uint32_t rows = plan.rows();
  allocate(backend, b, plan);
  std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
  for (const MetalBuffer *buffer : {&b.moe.scratch.groupedRoutes, &b.moe.scratch.groupedInput})
    std::memset(buffer->contents(), kCanary, buffer->sizeBytes());
  CommandGraph graph;
  MoE::add(graph, b.moe, m.weights, plan);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *selected = static_cast<const uint32_t *>(b.moe.scratch.selectedExperts.contents());
  const auto *routing = static_cast<const float *>(b.moe.scratch.routingWeights.contents());
  const auto *routeRows = static_cast<const uint32_t *>(b.moe.scratch.routeRows.contents());
  const auto *intermediate = static_cast<const __bf16 *>(b.moe.scratch.expertIntermediate.contents());
  const auto *down = static_cast<const __bf16 *>(b.moe.scratch.expertOutput.contents());
  const auto *output = static_cast<const __bf16 *>(b.moe.output.contents());
  const uint32_t tiles = *static_cast<const uint32_t *>(b.moe.scratch.tileCount.contents());
  require(tiles <= plan.maximumTiles(), label + ": tile count exceeds its bound");
  checkGrouping(b.moe.scratch, plan, label);
  // Each expert pass runs the live tiles of its column tiles; gate and up run
  // in one pass (_gate_up, _gate_up_mixed where their formats differ) or as
  // the gate and up passes (_a, then _g).
  uint32_t gateUpPasses = 0, upPasses = 0;
  for (const ComputeDispatch &dispatch : graph.dispatches()) {
    const uint32_t *grid = nullptr;
    if (const MetalBuffer &buffer = dispatch.indirectThreadgroups.buffer)
      grid = static_cast<const uint32_t *>(buffer.contents()) + dispatch.indirectThreadgroups.offsetBytes / 4;
    require(dispatch.pipelineName.starts_with("moe_expert_gguf") == (grid != nullptr) &&
                (!grid || (grid[0] == dispatch.threadgroups.x && grid[1] == tiles && grid[2] == 1 &&
                           dispatch.threadgroups.y == plan.maximumTiles())),
            label + ": " + dispatch.pipelineName + " runs another grid than its live tiles");
    gateUpPasses += dispatch.pipelineName.ends_with("_gate_up") || dispatch.pipelineName.ends_with("_gate_up_mixed");
    upPasses += dispatch.pipelineName.ends_with("_g");
  }
  require(gateUpPasses == staged && upPasses == !staged,
          label + (staged ? ": gate and up run in two passes" : ": gate and up run in one pass"));
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
  MoeRun result{std::vector<uint16_t>(uint64_t{rows} * kHidden), columnRoutes(b.moe.scratch, plan)};
  std::memcpy(result.output.data(), output, result.output.size() * 2);
  // Bitwise the output of the same dispatches again, whose grouping orders each expert's routes anew (its atomics),
  // and of every tile's threadgroups and two gate/up passes.
  for (int repeat = 0; repeat < 2; ++repeat) {
    std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    require(!std::memcmp(result.output.data(), output, result.output.size() * 2),
            label + ": the output differs from a repeat of the run");
  }
  std::deque<MoeGgufExpertParams> passes;
  const std::vector<ComputeDispatch> reference = fullGridsTwoPasses(graph, b.moe.scratch, passes);
  std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
  static_cast<void>(backend.submitCommandAsync(reference).wait());
  require(!std::memcmp(result.output.data(), output, result.output.size() * 2),
          label + ": the output differs from the full grids' and two gate/up passes'");
  return result;
}

// Each buffer a register decode plan and a staged 32-row prefill plan reach,
// at its extent and one element short, and each weight: the planes of every
// expert tensor, whose rows fill whole tiles, in units of a row's group of
// 32 inputs or meta unit (metal/abi/QuantFormat.h), and the shared scalar
// gate's fp32 weights.
void bufferExtents(MetalBackend &backend) {
  const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate};
  const Model m = makeModel(backend, shape, 0, false);
  Buffers b;
  b.moe.input = zeros(backend, uint64_t{kMaximumRows} * kHidden * 2, "moe-input");
  b.moe.residual = zeros(backend, uint64_t{kMaximumRows} * kHidden * 2, "moe-residual");
  b.moe.output = zeros(backend, uint64_t{kMaximumRows} * kHidden * 2, "moe-output");
  for (const MoePlan &plan :
       {MoE::decodePlan(shape, 4, MoeConfig{.expertTile = MoeExpertTile::M8, .ggufTile = MoeGgufTile::Register}),
        MoE::decodePlan(shape, 1, MoeConfig{.expertTile = MoeExpertTile::M8, .ggufTile = MoeGgufTile::Staged}),
        MoE::prefillPlan(shape, 33, MoeConfig{MoeExpertTile::M32})}) {
    allocate(backend, b, plan);
    splash::test::requireMoeExtents(backend, b.moe, m.weights, plan);
  }
  const MoePlan plan = MoE::prefillPlan(shape, 33, MoeConfig{MoeExpertTile::M32});
  allocate(backend, b, plan);
  const BlockMoeWeights &blocks = m.weights;
  // A weight buffer of the block that `with` puts in its place.
  const auto requireWeightExtent = [&](const MetalBuffer &buffer, uint64_t bytes, uint64_t element,
                                       const std::string &name, const auto &with) {
    splash::test::requireExtent(backend, buffer, bytes, element, name, [&](CommandGraph &graph, const MetalBuffer &view) {
      BlockMoeWeights changed = blocks;
      with(changed, view);
      MoE::add(graph, b.moe, changed, plan);
    });
  };
  for (const auto &[member, name] : {std::pair{&BlockMoeWeights::gate, "gate"}, std::pair{&BlockMoeWeights::up, "up"},
                                     std::pair{&BlockMoeWeights::down, "down"}})
    for (const auto &[segment, kind] : {std::pair{&BlockExpertProjection::routed, "MoE expert "},
                                        std::pair{&BlockExpertProjection::shared, "MoE shared "}}) {
      const QuantizedSegment &tensor = blocks.*member.*segment;
      const QuantFormat &format = kQuantFormats[tensor.formatId];
      const uint64_t groups = uint64_t{tensor.outputSize} * (tensor.inputSize / 32);
      for (const auto &[plane, planeName, unitBytes, units] :
           {std::tuple{&QuantizedSegment::plane0, " plane0", format.plane0_bytes, groups},
            std::tuple{&QuantizedSegment::plane1, " plane1", format.plane1_bytes, groups},
            std::tuple{&QuantizedSegment::meta, " meta", format.meta_bytes, groups / format.meta_groups}}) {
        if (!unitBytes) continue;
        requireWeightExtent(tensor.*plane, units * unitBytes, unitBytes, std::string(kind) + name + planeName,
                            [&](BlockMoeWeights &changed, const MetalBuffer &view) {
                              (changed.*member.*segment).*plane = view;
                            });
      }
    }
  requireWeightExtent(blocks.sharedScalarGate.plane0, uint64_t{kHidden} * 4, 4, "MoE shared scalar gate weight",
                      [](BlockMoeWeights &changed, const MetalBuffer &view) { changed.sharedScalarGate.plane0 = view; });
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
  const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate};
  // A model's prefill on the staged 32-row tiles, whose live-row matmuls
  // (moe_live_rows) differ by chunk, with the router on each float tile: a
  // row's result is the same in every chunk on one tile (either tile's scores
  // of a row depend on that row alone).
  const auto prefill32 = [&](const Model &m, std::map<std::pair<uint32_t, uint32_t>, GateUp> &products,
                             const std::string &formats) {
    for (const FloatTile router : {FloatTile::Simdgroup, FloatTile::NeuralAccelerator}) {
      const int before = failures;
      Stats stats;
      MoeRun widest, previous;
      uint32_t compared = 0, acrossKinds = 0;
      for (const uint32_t rows : {kMaximumRows, 33u, 16u}) {
        MoeConfig config{MoeExpertTile::M32};
        config.ggufRouterTile = router;
        const MoePlan plan = MoE::prefillPlan(shape, rows, config);
        const std::string label = formats + " prefill rows=" + std::to_string(rows) +
                                  (router == FloatTile::NeuralAccelerator ? " (accelerator router)" : "");
        const MoeRun result = runPlan(backend, m, b, plan, true, products, stats, label);
        if (widest.output.empty()) {
          Stats repeat;
          failures += compareRepeat(result, runPlan(backend, m, b, plan, true, products, repeat, label + " repeat"),
                                    label);
          widest = previous = result;
          continue;
        }
        const RowComparison c = compareChunks(result, widest, previous, rows, label);
        compared += c.compared;
        acrossKinds += c.acrossKinds;
        failures += c.differ > 0;
        previous = result;
      }
      printf("%-20s staged   prefill 263/33/16 (32-row tiles, %s router): gate/up %.2f%% and down %.2f%% of outputs "
             "differ from bf16(fp64), errors at most %.1e/%.1e of sum|x w|, %u rows compared bitwise, %u across tile "
             "kinds %s\n",
             formats.c_str(), router == FloatTile::NeuralAccelerator ? "accelerator" : "simdgroup",
             100.0 * stats.gateUpFlips / (stats.outputs / 2), 100.0 * stats.downFlips / stats.outputs, stats.gateUpWorst,
             stats.downWorst, compared, acrossKinds, failures > before ? "FAIL" : "ok");
    }
  };
  // Each tile runs the models of the formats it takes (makeModel).
  for (const MoeGgufTile tile : {MoeGgufTile::Staged, MoeGgufTile::Register})
    for (const Fmt f : tileFormats(tile))
      for (const bool upInGateFormats : {false, true}) {
        if (upInGateFormats && tile == MoeGgufTile::Register) continue;   // two passes whatever the formats
        const Model m = makeModel(backend, shape, f, upInGateFormats, tile);
        std::map<std::pair<uint32_t, uint32_t>, GateUp> products;
        const std::string formats = formatNames(m);
        const int before = failures;
        Stats stats;
        MoeRun widest;
        uint32_t compared = 0, acrossKinds = 0;
        // The rows of a run that differ from the four-lane dispatch's where their routes ran the same tile kinds.
        const auto compareWidest = [&](const MoeRun &run, const std::string &label) {
          const RowComparison c =
              compareRows(run, widest, uint32_t(std::min(run.output.size(), widest.output.size()) / kHidden));
          compared += c.compared;
          acrossKinds += c.acrossKinds;
          if (c.differ) {
            printf("  %s: %u rows differ from the four-lane dispatch FAIL\n", label.c_str(), c.differ);
            ++failures;
          }
        };
        for (uint32_t lanes = 4; lanes >= 1; --lanes) {
          const MoePlan plan = MoE::decodePlan(
              shape, lanes,
              MoeConfig{.expertTile = MoeExpertTile::M8, .ggufTile = tile});
          const std::string label = formats + (tile == MoeGgufTile::Register ? " register" : " staged") + " decode B" +
                                    std::to_string(lanes);
          const MoeRun run =
              runPlan(backend, m, b, plan, tile == MoeGgufTile::Staged, products, stats, label);
          // A row's result depends on its own routes and the tile kinds they
          // run only, not on the rows it is batched with (the tile rows of one
          // expert are independent).
          if (widest.output.empty()) {
            Stats repeat;
            failures += compareRepeat(
                run, runPlan(backend, m, b, plan, tile == MoeGgufTile::Staged, products, repeat, label + " repeat"),
                label);
            widest = run;
          } else {
            compareWidest(run, label);
          }
        }
        // Prefill chunks on the same 8-row tiles (ExecutionPlans::moePrefill).
        for (const uint32_t chunk : {kMaximumRows, 27u, 9u, 8u}) {
          const MoePlan plan = MoE::prefillPlan(
              shape, chunk,
              MoeConfig{.expertTile = MoeExpertTile::M8, .ggufTile = tile});
          const std::string label = formats + (tile == MoeGgufTile::Register ? " register" : " staged") + " prefill rows=" +
                                    std::to_string(chunk);
          compareWidest(runPlan(backend, m, b, plan, tile == MoeGgufTile::Staged, products, stats, label),
                        label);
        }
        printf("%-20s %s decode B1-4, prefill 263/27/9/8: gate/up %.2f%% and down %.2f%% of outputs differ from bf16(fp64), "
               "errors at most %.1e/%.1e of sum|x w|, %u rows compared bitwise, %u across tile kinds %s\n",
               formats.c_str(), tile == MoeGgufTile::Register ? "register" : "staged  ",
               100.0 * stats.gateUpFlips / (stats.outputs / 2), 100.0 * stats.downFlips / stats.outputs,
               stats.gateUpWorst, stats.downWorst, compared, acrossKinds, failures > before ? "FAIL" : "ok");
        if (tile == MoeGgufTile::Staged) prefill32(m, products, formats);
      }
  return failures;
}

// Concentrated and skewed loads: a router whose expert e scores 4 x[e], and
// rows whose first kExperts inputs rank the experts, the first at 1, the next
// expert at 0.5. When experts outnumber hidden columns, paired experts read
// the same column with opposite signs; the remaining inputs stay below 0.5.
// Every row (Concentrated) or
// six of every eight (Skewed) route to experts 0 and 1 first, the others to
// expert row % kExperts and the next. On the staged 32-row tiles of the
// Q4_K gate/up model, 1, 12 and 48 rows leave ragged tiles of 8- and 16-row
// matmuls; at 256 rows the concentrated load fills whole tiles and the skewed
// one leaves 16-row tails. A row's result is the same at every row count.
int routingLoads(MetalBackend &backend) {
  enum class Load { Concentrated, Skewed };
  const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate};
  Model m = makeModel(backend, shape, Q4K, true);
  m.router.values.assign(uint64_t{kExperts} * kHidden, 0.0f);
  static_assert(kExperts <= 2 * kHidden, "the routing fixture holds two experts per input column");
  const auto column = [](uint32_t expert) { return expert % kHidden; };
  const auto sign = [](uint32_t expert) { return expert < kHidden ? 1.0f : -1.0f; };
  for (uint32_t e = 0; e < kExperts; ++e)
    m.router.values[uint64_t{e} * kHidden + column(e)] = 4.0f * sign(e);
  m.router.segment = floatSegment(backend, m.router.values, kExperts, kHidden);
  m.weights.router = m.router.segment;
  const std::string formats = formatNames(m);
  int failures = 0;
  for (const Load load : {Load::Concentrated, Load::Skewed}) {
    const std::string name = load == Load::Concentrated ? "concentrated" : "skewed";
    Buffers b;
    b.input = activations(uint64_t{kMaximumRows} * kHidden);
    b.residual = activations(uint64_t{kMaximumRows} * kHidden);
    for (float &v : b.residual) v = bf16(0.1f * v);
    for (uint32_t r = 0; r < kMaximumRows; ++r) {
      const uint32_t first = load == Load::Concentrated || r % 8 < 6 ? 0 : r % kExperts;
      const uint32_t next = (first + 1) % kExperts;
      for (uint32_t c = 0; c < std::min(kExperts, kHidden); ++c)
        b.input[uint64_t{r} * kHidden + c] = bf16(0.01 * c * std::min(1.0, 16.0 / kExperts));
      b.input[uint64_t{r} * kHidden + column(first)] = sign(first);
      b.input[uint64_t{r} * kHidden + column(next)] = bf16(0.5f * sign(next));
    }
    b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
    b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
    b.moe.output = zeros(backend, uint64_t{kMaximumRows} * kHidden * 2, "moe-output");
    std::map<std::pair<uint32_t, uint32_t>, GateUp> products;
    Stats stats;
    MoeRun widest, previous;
    uint32_t compared = 0, acrossKinds = 0;
    const int before = failures;
    for (const uint32_t rows : {256u, 48u, 12u, 1u}) {
      const MoePlan plan = MoE::prefillPlan(shape, rows, MoeConfig{MoeExpertTile::M32});
      const std::string label = formats + " " + name + " prefill rows=" + std::to_string(rows);
      const MoeRun result = runPlan(backend, m, b, plan, true, products, stats, label);
      const auto *selected = static_cast<const uint32_t *>(b.moe.scratch.selectedExperts.contents());
      for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t first = load == Load::Concentrated || r % 8 < 6 ? 0 : r % kExperts;
        require(selected[r * kRoutes] == first && selected[r * kRoutes + 1] == (first + 1) % kExperts,
                label + ": fixture routes to its intended first two experts");
      }
      if (widest.output.empty()) {
        Stats repeat;
        failures +=
            compareRepeat(result, runPlan(backend, m, b, plan, true, products, repeat, label + " repeat"), label);
        widest = previous = result;
        continue;
      }
      const RowComparison c = compareChunks(result, widest, previous, rows, label);
      compared += c.compared;
      acrossKinds += c.acrossKinds;
      failures += c.differ > 0;
      previous = result;
    }
    printf("%-20s staged   prefill 256/48/12/1 (32-row tiles, %s routing): gate/up %.2f%% and down %.2f%% of "
           "outputs differ from bf16(fp64), errors at most %.1e/%.1e of sum|x w|, %u rows compared bitwise, %u across "
           "tile kinds %s\n",
           formats.c_str(), name.c_str(), 100.0 * stats.gateUpFlips / (stats.outputs / 2),
           100.0 * stats.downFlips / stats.outputs, stats.gateUpWorst, stats.downWorst, compared, acrossKinds,
           failures > before ? "FAIL" : "ok");
  }
  return failures;
}

} // namespace

// The kernels read each routed expert from the slot its table names: the
// experts in slots permuted by sigma, with each expert's router row moved to
// match (expert e routes as e' did where sigma(e) = e'), give the identity
// table's output bit for bit, in decode and in prefill, on the staged tile
// and on the register tile (Apple9's, which binds the table on its own).
int permutedSlots(MetalBackend &backend) {
  const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate};
  const auto sigma = [](uint32_t expert) { return (expert * 7 + 3) % kExperts; };
  int failures = 0;
  for (const MoeGgufTile tile : {MoeGgufTile::Staged, MoeGgufTile::Register}) {
    const Model m = makeModel(backend, shape, tileFormats(tile).front(), false, tile);
    Model permuted = m;
    MetalBuffer table = backend.allocateBuffer(uint64_t{kExperts} * sizeof(uint32_t), BufferStorage::Shared,
                                               "permuted-slots");
    std::vector<float> router(m.router.values.size());
    for (uint32_t expert = 0; expert < kExperts; ++expert) {
      static_cast<uint32_t *>(table.contents())[expert] = sigma(expert);
      std::copy_n(m.router.row(sigma(expert)), kHidden, router.begin() + uint64_t{expert} * kHidden);
    }
    permuted.weights.slotTable = table;
    permuted.weights.router = floatSegment(backend, router, kExperts, kHidden);
    Buffers b;
    b.input = activations(uint64_t{kMaximumRows} * kHidden);
    b.residual = activations(uint64_t{kMaximumRows} * kHidden);
    b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
    b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
    b.moe.output = zeros(backend, uint64_t{kMaximumRows} * kHidden * 2, "moe-output");
    const auto run = [&](const Model &model, const MoePlan &plan) {
      allocate(backend, b, plan);
      std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
      CommandGraph graph;
      MoE::add(graph, b.moe, model.weights, plan);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      const auto *output = static_cast<const uint16_t *>(b.moe.output.contents());
      return std::vector<uint16_t>(output, output + uint64_t{plan.rows()} * kHidden);
    };
    // The register tile prefills on the 8-row tiles of its decode
    // (ExecutionPlans::moePrefill), the staged one here on 32-row tiles.
    const MoeExpertTile prefillTile = tile == MoeGgufTile::Register ? MoeExpertTile::M8 : MoeExpertTile::M32;
    int tileFailures = 0;
    for (const MoePlan &plan :
         {MoE::decodePlan(shape, 4, MoeConfig{.expertTile = MoeExpertTile::M8, .ggufTile = tile}),
          MoE::prefillPlan(shape, kMaximumRows, MoeConfig{.expertTile = prefillTile, .ggufTile = tile})})
      if (run(m, plan) != run(permuted, plan)) {
        printf("  permuted slots, %u rows: the output differs from the identity table's FAIL\n", plan.rows());
        ++tileFailures;
      }
    printf("%-20s %s permuted slot table, decode B4 and prefill %u: %s\n", formatNames(m).c_str(),
           tile == MoeGgufTile::Register ? "register" : "staged  ", kMaximumRows, tileFailures ? "FAIL" : "ok");
    failures += tileFailures;
  }
  return failures;
}

// A verify step's grouping routes each lane's verify rows alone
// (MoeBuffers::verifyRows). With every row verified, its output and tile kinds
// are bitwise the all-rows grouping's. Over four lanes that verify 8, 5, 2 and
// 1 rows, a routed route of a row past its lane's rows stays out of every tile
// with the route row ~0u, every other route round-trips and the tiles hold
// exactly those routes; a verified row's output is bitwise the all-rows run's
// where its routes ran the same tile kinds (a row's output depends on its own
// routes only), and a row past them stays finite (its residual and shared
// expert). On the staged and the register tile. How many staged rows keep
// their tile kinds depends on the inputs, which the cases before this one draw
// from the shared generator, down to none of the 16; so only the register
// tile, whose kinds cannot change, must compare a row, and the run with every
// row verified covers the staged tile.
int verifiedRows(MetalBackend &backend) {
  const MoeShape shape{kHidden, kExperts, kTopK, kIntermediate};
  constexpr std::array<uint32_t, 4> kVerify{8, 5, 2, 1};
  int failures = 0;
  for (const MoeGgufTile tile : {MoeGgufTile::Staged, MoeGgufTile::Register}) {
    const Model m = makeModel(backend, shape, tileFormats(tile).front(), false, tile);
    const MoePlan plan = MoE::decodePlan(shape, 4, MoeConfig{.expertTile = MoeExpertTile::M8, .ggufTile = tile});
    const uint32_t rows = plan.rows(), tileRows = plan.tileRows();
    Buffers b;
    b.input = activations(uint64_t{kMaximumRows} * kHidden);
    b.residual = activations(uint64_t{kMaximumRows} * kHidden);
    b.moe.input = bfloatBuffer(backend, b.input, "moe-input");
    b.moe.residual = bfloatBuffer(backend, b.residual, "moe-residual");
    b.moe.output = zeros(backend, uint64_t{kMaximumRows} * kHidden * 2, "moe-output");
    MetalBuffer verify = backend.allocateBuffer(kVerify.size() * sizeof(uint32_t), BufferStorage::Shared,
                                                "verify-rows");
    std::copy(kVerify.begin(), kVerify.end(), static_cast<uint32_t *>(verify.contents()));
    MetalBuffer verifyAll = backend.allocateBuffer(kVerify.size() * sizeof(uint32_t), BufferStorage::Shared,
                                                   "verify-all-rows");
    std::fill_n(static_cast<uint32_t *>(verifyAll.contents()), kVerify.size(), SPLASH_TARGET_VERIFY_ROWS);
    const auto unverified = [&](uint32_t route) {
      const uint32_t row = route / kRoutes;
      return route % kRoutes < kTopK && row % SPLASH_TARGET_VERIFY_ROWS >= kVerify[row / SPLASH_TARGET_VERIFY_ROWS];
    };
    // A run's output rows and, per route, its tile kind: column or not on the
    // staged passes, UINT32_MAX where the run left it out.
    const auto run = [&](const MetalBuffer &verifyRows) {
      allocate(backend, b, plan);
      std::memset(b.moe.output.contents(), 0, b.moe.output.sizeBytes());
      b.moe.verifyRows = verifyRows;
      CommandGraph graph;
      MoE::add(graph, b.moe, m.weights, plan);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      const auto *output = static_cast<const uint16_t *>(b.moe.output.contents());
      const auto *routeRows = static_cast<const uint32_t *>(b.moe.scratch.routeRows.contents());
      const auto *tiles = static_cast<const MoeTileDescriptor *>(b.moe.scratch.tileDescriptors.contents());
      std::vector<uint32_t> kinds(uint64_t{rows} * kRoutes);
      for (uint32_t route = 0; route < kinds.size(); ++route)
        kinds[route] = routeRows[route] == UINT32_MAX ? UINT32_MAX
                       : tile == MoeGgufTile::Staged ? uint32_t(moe_column_tile(tiles, routeRows[route] / tileRows))
                                                     : 0;
      return std::pair{std::vector<uint16_t>(output, output + uint64_t{rows} * kHidden), kinds};
    };
    const auto [allOutput, allKinds] = run({});
    const auto [fullOutput, fullKinds] = run(verifyAll);
    const auto [output, kinds] = run(verify);
    int tileFailures = 0;
    const auto fail = [&](const std::string &what) {
      printf("  %s FAIL\n", what.c_str());
      ++tileFailures;
    };
    const bool fullBitwise = fullOutput == allOutput && fullKinds == allKinds;
    if (!fullBitwise) fail("every row verified is not bitwise the all-rows run");

    const MoeScratch &scratch = b.moe.scratch;
    const auto *selected = static_cast<const uint32_t *>(scratch.selectedExperts.contents());
    const auto *routeRows = static_cast<const uint32_t *>(scratch.routeRows.contents());
    const auto *groupedRoutes = static_cast<const uint32_t *>(scratch.groupedRoutes.contents());
    const auto *tiles = static_cast<const MoeTileDescriptor *>(scratch.tileDescriptors.contents());
    const uint32_t tileCount = *static_cast<const uint32_t *>(scratch.tileCount.contents());
    std::array<uint32_t, kExperts + 1> routes{};
    for (uint32_t route = 0; route < rows * kRoutes; ++route) {
      if (unverified(route)) {
        if (routeRows[route] != UINT32_MAX) fail("a route past its lane's verify rows was grouped");
        continue;
      }
      ++routes[selected[route]];
      const uint32_t grouped = routeRows[route];
      if (grouped >= tileCount * tileRows || groupedRoutes[grouped] != route ||
          tiles[grouped / tileRows].expert != selected[route])
        fail("a verified route's grouped row does not round-trip");
    }
    uint32_t occupied = 0;
    for (const uint32_t count : routes) occupied += (count + tileRows - 1) / tileRows;
    if (tileCount != occupied) fail("the tiles hold other routes than the verified ones");

    uint32_t compared = 0, acrossKinds = 0;
    for (uint32_t row = 0; row < rows; ++row) {
      const auto at = [&](const std::vector<uint16_t> &values) { return values.begin() + uint64_t{row} * kHidden; };
      if (row % SPLASH_TARGET_VERIFY_ROWS >= kVerify[row / SPLASH_TARGET_VERIFY_ROWS]) {
        if (!std::all_of(at(output), at(output) + kHidden, [](uint16_t v) { return (v & 0x7F80) != 0x7F80; }))
          fail("a row past its lane's verify rows is not finite");
        continue;
      }
      if (!std::equal(kinds.begin() + uint64_t{row} * kRoutes, kinds.begin() + uint64_t{row + 1} * kRoutes,
                      allKinds.begin() + uint64_t{row} * kRoutes)) {
        ++acrossKinds;
        continue;
      }
      ++compared;
      if (!std::equal(at(output), at(output) + kHidden, at(allOutput)))
        fail("verified row " + std::to_string(row) + " differs from the all-rows run");
    }
    if (!compared && tile == MoeGgufTile::Register) fail("no verified row ran its all-rows tile kinds");
    printf("%-20s %s every row verified: %s; verify rows 8/5/2/1: %u rows bitwise, %u across tile kinds: %s\n",
           formatNames(m).c_str(), tile == MoeGgufTile::Register ? "register" : "staged  ",
           fullBitwise ? "bitwise" : "differs", compared, acrossKinds, tileFailures ? "FAIL" : "ok");
    failures += tileFailures;
  }
  return failures;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf_moe_test <metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      bufferExtents(backend);
      const int failures = floatProjection(backend) + bfloat16Weights(backend) + floatPairs(backend) +
                           floatSegments(backend, 64) + floatSegments(backend, 96) + floatOnlyChain(backend) +
                           moe(backend) + routingLoads(backend) + permutedSlots(backend) + verifiedRows(backend);
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
