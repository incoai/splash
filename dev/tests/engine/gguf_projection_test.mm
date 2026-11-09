// GGUF projections through ops::Linear (runtime/ops/LinearGguf.cpp) in every format a tile takes (the register tile
// all but F16, quant_register_format), against fp64 references over GGML's dequantized weights
// (GgufFormatReference.hpp). Each check forces its tile with Linear::plan(workload, config), so every GPU runs both
// decode tiles:
// - decode: the Apple9 register tile and the staged tile at one to four lanes (three on the staged 32-row tile over
//   NaN padding rows), every K split, each epilogue, dense inputs and sparse ones past half's range; a lane's rows
//   equal a one-lane projection of them bitwise, as does one lane on the staged 16-row tile over a NaN padding lane
//   (LinearConfig::sixteenRows); the plain epilogue into fp32 (the logits) holds the values its bf16 output rounds,
//   bit for bit, each within fp64 before rounding;
// - fused: three segments of different formats in one projection equal the projections of each segment alone, on the
//   register tile at K 4096, so that formats of 256-input units split sixteen ways too, and one lane of them on the
//   16-row tile the 8-row tile's;
// - both on the staged tile's spread walk too (LinearConfig::spread), over partitions from 72 steps down to 9, an odd
//   count, whose tiles' walks wrap past the partition's end;
// - gate/up: each format's gate with the next format the tile takes as up, and in one pass
//   (LinearConfig::oneGateUpPass) bitwise the two passes at the same K split, one lane on the 16-row tile too;
// - prefill: 128-row tiles with each epilogue, whose simdgroups past the chunk write nothing, and chunks of up to 64
//   rows on the decode tiles equal to them bitwise; fused segments at their column offsets;
// - int8 decode: rotated PQ2_0 projections on the int8 decode tiles (LinearConfig::decodeInt8) within the fp32
//   rounding of the exact products of their int8 codes, at every lane count, split, walk and pass mode, the
//   production plans in int8 activations, and a non-finite input's row alone not finite;
// - int8 prefill (af4g64, af5g64, where the family runs it): the products of the inputs rounded per row and 64-input
//   group, also with an outlier input per row, within a bound of the bf16 tile's error, a chunk's last rows at their
//   own height and a view of leading inputs bitwise, and a non-finite input's row alone not finite;
// - split visibility: two projections that share the split scratch, at every pair of K splits either tile's policy
//   picks for 8-80 cores (the register tile's for Flash-Next's hyper-connection pair too), independent of poisoned
//   partials;
// - narrow inputs: Q8_0, MXFP4 and Q4_0 at K 320 and 640, whole 64-input spans but no multiple of 256 (the
//   hyper-connection projections of Qwen3.8-Flash-Next take 320 inputs in Q8_0), decoded on both tiles and walks at
//   every K split each tile takes there, and prefilled; and Q8_0 decoded at K 192, three of the register tile's
//   64-input coefficient units, which two partitions split into one and two.
// Every run leaves the padding columns past its segments, the guard bands past its buffers and its counters as they
// were. The token gather (ops::Embedding) of every format's rows as stored is checked here too.
#include "GgufFormatReference.hpp"
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "model/SparseRowStore.hpp"
#include "ops/Embedding.hpp"
#include "ops/KernelCalibration.hpp"
#include "ops/KernelSelection.hpp"
#include "ops/Linear.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace gguf_reference;
using splash::metal::BytesBinding;
using splash::metal::CommandGraph;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

namespace {

constexpr uint32_t kLaneRows = 8, kMaximumLanes = 4, kMaximumRows = kLaneRows * kMaximumLanes;
// Every K split a plan takes: the powers of two up to the largest.
constexpr uint32_t kSplits[] = {1, 2, 4, 8, 16};
static_assert(kSplits[std::size(kSplits) - 1] == LinearConfig::kMaximumSplits);
// Padding columns past a matrix's segments, which no kernel writes: four 64-column tiles.
constexpr uint32_t kPadding = 256;
// Sparse inputs: one entry per 256 inputs (the register tile's widest coefficient unit) of this magnitude, above
// half's largest finite value (65504), so a tile that converted its inputs to half would overflow.
constexpr float kSparseMagnitude = 1e5f;
constexpr uint16_t kNaN = 0x7FC0;         // bf16 quiet NaN: the input of padding rows
constexpr uint16_t kUnwritten = 0xFFFF;   // bf16 NaN no kernel writes: the output before a run
constexpr uint8_t kGuardByte = 0xA5;
constexpr uint64_t kGuardBytes = 256;
// Residuals of this magnitude exceed the projections' (tens), as a model's residual stream does, so an epilogue that
// mishandles them moves outputs by more than their bf16 rounding.
constexpr float kResidualMagnitude = 64;
// Split partials before a run: NaN, or a large finite value that stays finite in a sum and shows as a wrong value.
constexpr uint32_t kPoisonNaN = 0x7FC00000u, kPoisonFinite = 0x7E800000u;

std::mt19937 rng(42);
int failures = 0;
int sectionFailures = 0;

// Names one failure; the first few of a section are printed.
void fail(const std::string &what) {
  constexpr int kShown = 8;
  if (sectionFailures++ < kShown) std::cout << "  FAIL " << what << '\n';
  ++failures;
}
void section(const std::string &what) {
  std::cout << what << (sectionFailures ? " FAIL (" + std::to_string(sectionFailures) + " failures)" : " ok") << '\n';
  sectionFailures = 0;
}

const char *tileName(LinearTile tile) { return tile == LinearTile::GgufRegister ? "register" : "staged"; }
const char *epilogueName(LinearEpilogue e) {
  switch (e) {
    case LinearEpilogue::None: return "plain";
    case LinearEpilogue::Residual: return "residual";
    case LinearEpilogue::GateUp: return "gate/up";
    case LinearEpilogue::UpWithGate: return "up-with-gate";
  }
  return "?";
}

// ---------------------------------------------------------------- weights
// A GGUF tensor [N, K]: its native rows and the segment of its repacked planes.
struct Tensor {
  Fmt format;
  uint32_t N, K;
  std::vector<uint8_t> native;
  QuantizedSegment segment;
};

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = test::sharedBuffer(backend, bytes.size());
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}

Tensor tensor(MetalBackend &backend, Fmt f, uint32_t N, uint32_t K) {
  Tensor t{f, N, K, makeNative(f, N, K, rng), {}};
  const Packed planes = repack(f, t.native, N, K, nullptr);
  t.segment = QuantizedSegment::planes(f, N, K, upload(backend, planes.w0),
                                       kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1) : MetalBuffer{},
                                       upload(backend, planes.meta));
  return t;
}

// A projection of `columns` destination columns whose leading columns are the parts' segments in order.
Projection projection(const std::vector<const Tensor *> &parts, uint32_t columns) {
  BlockWeights blocks;
  uint32_t offset = 0;
  for (const Tensor *t : parts) {
    blocks.segments.push_back(t->segment);
    blocks.segments.back().columnOffset = offset;
    offset += t->N;
  }
  return Projection(columns, parts.front()->K, std::move(blocks));
}
uint32_t segmentColumns(const std::vector<const Tensor *> &parts) {
  uint32_t columns = 0;
  for (const Tensor *t : parts) columns += t->N;
  return columns;
}

// ---------------------------------------------------------------- fp64 reference
// bf16 activations of `rows` rows: dense in [-1, 1], or sparse: +-kSparseMagnitude once per 256 inputs.
enum class Inputs { Dense, Sparse };
std::vector<float> activations(Inputs kind, uint32_t rows, uint32_t width) {
  std::uniform_real_distribution<float> unit(-1.f, 1.f);
  std::vector<float> x(uint64_t{rows} * width, 0.f);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t k = 0; k < width; ++k) {
      float &v = x[uint64_t{r} * width + k];
      if (kind == Inputs::Dense) v = unit(rng);
      else if (k % 256 == (r * 31 + k / 256 * 97) % 256) v = (r + k / 256) % 2 ? -kSparseMagnitude : kSparseMagnitude;
      v = bf16ToFloat(floatToBf16(v));
    }
  return x;
}
std::vector<float> residuals(uint32_t rows, uint32_t width) {
  std::vector<float> r = activations(Inputs::Dense, rows, width);
  for (float &v : r) v = bf16ToFloat(floatToBf16(v * kResidualMagnitude));
  return r;
}

// dots[r * columns + c]: the fp64 products of input row r with the weight row of destination column c (the
// parts' columns in order; the padding columns hold none).
std::vector<Dot> products(const std::vector<const Tensor *> &parts, uint32_t columns, const std::vector<float> &x,
                          uint32_t rows) {
  std::vector<Dot> dots(uint64_t{rows} * columns);
  Dot *out = dots.data();
  const float *input = x.data();
  uint32_t offset = 0;
  for (const Tensor *t : parts) {
    const Fmt f = t->format;
    const uint32_t K = t->K, at = offset;
    const uint8_t *native = t->native.data();
    dispatch_apply(t->N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t n) {
      std::vector<float> w(K);
      rowValues(f, native + n * rowBytes(f, K), K, w.data());
      for (uint32_t r = 0; r < rows; ++r)
        out[uint64_t{r} * columns + at + n] = dot(input + uint64_t{r} * K, w.data(), K);
    });
    offset += t->N;
  }
  return dots;
}

// The values a bf16 output may take: [bf16(lo), bf16(hi)] of the fp32 value's interval.
struct Interval {
  double lo, hi;
};
float bf16(double v) { return bf16ToFloat(floatToBf16(float(v))); }
// Nine significant digits, which tell apart the values near zero that %f prints alike.
std::string digits(double v) {
  char text[32];
  std::snprintf(text, sizeof text, "%.9g", v);
  return text;
}
bool inside(uint16_t got, Interval i) {
  const float value = bf16ToFloat(got);
  return std::isfinite(value) && value >= bf16(i.lo) && value <= bf16(i.hi);
}
Interval projected(const Dot &d, bool staged) {
  const double e = projectionBound(d, staged);
  return {d.value - e, d.value + e};
}
// The interval of output i (row * columns + column) before rounding.
using Want = std::function<Interval(uint64_t)>;
// The products' intervals on the staged or the register tile.
Want within(const std::vector<Dot> &dots, bool staged) {
  return [&dots, staged](uint64_t i) { return projected(dots[i], staged); };
}
// fl(p + r): rounding is monotonic, so the sum's interval is the rounded sums of the ends.
Interval withResidual(Interval p, float residual) {
  return {float(float(p.lo) + residual), float(float(p.hi) + residual)};
}
// The relative error of the kernels' silu_gate(g) = g / (1 + fast::exp2(-log2(e) g)): fast::exp2(t) is within
// 3 + floor(2|t|) ulp (Metal Shading Language, fast-math accuracy); t = fl(fl(log2 e) g) is within 2u relative,
// which moves exp2 by 2 ln2 |t| u; 1 + e rounds once and the fast-math division is within 2.5 ulp. An ulp is at
// most 2u (u = 2^-24), and e / (1 + e) < 1 carries exp2's relative error to silu at most once.
double siluError(double gate) {
  const double t = std::fabs(gate) * 1.4426950408889634;
  return (2 * (3 + std::floor(2 * t)) + 2 * std::log(2.0) * t + 1 + 2 * 2.5) * 0x1p-24;
}
double silu(double g) { return g / (1 + std::exp(-g)); }
// fl(bf16(up) * silu_gate(gate)): the corners of bf16(up)'s interval times silu's. The fast-math division is only
// bounded for divisors up to 2^126, which 1 + exp2(t) exceeds from t = 126, and the kernels' fp32 flushes values below
// its normal range to zero: there either factor, and a product below that range, may be zero.
Interval withGate(Interval up, float gate) {
  constexpr double kSmallestNormal = 0x1p-126, kLargestDivisorExponent = 126;
  const double s = silu(gate), e = siluError(gate) * std::fabs(s);
  const bool silentGate = -gate * 1.4426950408889634 >= kLargestDivisorExponent || std::fabs(s) + e < kSmallestNormal;
  double lo = INFINITY, hi = -INFINITY;
  for (const double u : {double(bf16(up.lo)), double(bf16(up.hi))})
    for (const double g : {s - e, s + e, silentGate ? 0.0 : s}) {
      lo = std::min(lo, u * g);
      hi = std::max(hi, u * g);
    }
  if (std::min(std::fabs(lo), std::fabs(hi)) < kSmallestNormal) return {std::min(lo, 0.0), std::max(hi, 0.0)};
  return {lo, hi};
}

// ---------------------------------------------------------------- running a plan
// A buffer of `bytes` followed by a guard band that no dispatch may write.
struct Guarded {
  MetalBuffer backing, view;
  uint64_t bytes = 0;
  Guarded(MetalBackend &backend, uint64_t size, uint8_t fill) : bytes(size) {
    if (!size) return;
    backing = test::sharedBuffer(backend, size + kGuardBytes);
    std::memset(backing.contents(), fill, size);
    std::memset(static_cast<uint8_t *>(backing.contents()) + size, kGuardByte, kGuardBytes);
    view = backend.view(backing, 0, size);
  }
  [[nodiscard]] bool intact() const {
    const auto *guard = static_cast<const uint8_t *>(backing.contents()) + bytes;
    return !bytes || std::all_of(guard, guard + kGuardBytes, [](uint8_t b) { return b == kGuardByte; });
  }
  [[nodiscard]] std::vector<uint16_t> halves() const {
    const auto *v = static_cast<const uint16_t *>(view.contents());
    return bytes ? std::vector<uint16_t>(v, v + bytes / 2) : std::vector<uint16_t>{};
  }
};
Guarded bfloats(MetalBackend &backend, const std::vector<uint16_t> &values) {
  Guarded g(backend, values.size() * 2, 0);
  if (g.bytes) std::memcpy(g.view.contents(), values.data(), g.bytes);
  return g;
}

// The bf16 rows [0, storage) of an input: its rows below `active`, NaN past them.
std::vector<uint16_t> storageRows(const std::vector<float> &x, uint32_t width, uint32_t active, uint32_t storage) {
  std::vector<uint16_t> rows(uint64_t{storage} * width, kNaN);
  for (uint64_t i = 0; i < uint64_t{active} * width; ++i) rows[i] = floatToBf16(x[i]);
  return rows;
}

// A plan's split scratch, as LinearPlan::scratchSize sizes it (the largest of several plans that share it), and a
// rotated projection's rotated rows, which start NaN (bytes 0xFF: NaN bf16 values, int8 codes -1 and NaN scales),
// as the padding rows of a plan's storage that no rotation writes are read.
struct Scratch {
  Guarded table, sums, partials, counters, rotated;
  Scratch(MetalBackend &backend, LinearScratchSize size)
      : table(backend, size.input, 0), sums(backend, size.sums, 0), partials(backend, size.partials, 0),
        counters(backend, size.counters, 0), rotated(backend, size.rotated, 0xFF) {}
  [[nodiscard]] LinearScratch bindings() const {
    return {table.view, sums.view, partials.view, counters.view, rotated.view};
  }
  void poison(uint32_t bits) const {
    if (partials.bytes) std::fill_n(static_cast<uint32_t *>(partials.view.contents()), partials.bytes / 4, bits);
  }
  [[nodiscard]] bool intact() const {
    const auto *count = static_cast<const uint32_t *>(counters.view.contents());
    return table.intact() && sums.intact() && partials.intact() && counters.intact() && rotated.intact() &&
           std::all_of(count, count + counters.bytes / 4, [](uint32_t c) { return c == 0; });
  }
};
// The scratch of `plan` over `p`: the plan's, and a rotated projection's rotated rows (Linear::decodeScratchSize).
LinearScratchSize scratchSize(const Linear &linear, const LinearPlan &plan, const Projection &p) {
  LinearScratchSize size = plan.scratchSize();
  if (p.rotation) size.rotated = linear.decodeScratchSize(p.shape()).rotated;
  return size;
}
// One projection's buffers for a plan: its input and auxiliary rows (the residual, or the gate an up-with-gate
// prefill reads), and an output and gate scratch that start unwritten.
struct Operands {
  Guarded input, aux, output, gate;
  Operands(MetalBackend &backend, const LinearPlan &plan, const std::vector<uint16_t> &x,
           const std::vector<uint16_t> &auxiliary)
      : input(bfloats(backend, x)), aux(bfloats(backend, auxiliary)),
        output(backend, uint64_t{plan.storageRows()} * plan.workload().matrix.outputSize * 2, 0xFF),
        gate(backend, plan.gateScratchBytes(), 0xFF) {
    if (plan.workload().epilogue == LinearEpilogue::UpWithGate) gate = bfloats(backend, auxiliary);
  }
  [[nodiscard]] LinearBuffers bindings(const LinearPlan &plan, const Scratch &scratch) const {
    const bool residual = plan.workload().epilogue == LinearEpilogue::Residual;
    return {.input = input.view, .output = output.view, .residual = residual ? aux.view : MetalBuffer{},
            .gateScratch = gate.view, .scratch = scratch.bindings()};
  }
};

// The output rows [0, storage) of a run, for gate/up the gate its up pass read, and the pipelines it dispatched.
struct Outcome {
  std::vector<uint16_t> output, gate;
  std::vector<std::string> pipelines;
};

// Runs one projection with `plan` over `x` (and `aux`), its split partials first set to `poison`; checks that a spread
// plan's tiles bind the spread walk, and what every run must leave as it was: guards, counters and the padding columns
// past `covered`.
Outcome run(MetalBackend &backend, const Linear &linear, const LinearPlan &plan, const Projection &p,
            const Projection *gate, const std::vector<uint16_t> &x, const std::vector<uint16_t> &aux, uint32_t poison,
            uint32_t covered, const std::string &label) {
  const Scratch scratch(backend, scratchSize(linear, plan, p));
  const Operands o(backend, plan, x, aux);
  scratch.poison(poison);
  CommandGraph graph;
  static_cast<void>(linear.add(graph, o.bindings(plan, scratch), p, plan, gate));
  // A spread plan's dispatches past a rotated projection's rotation are staged decode tiles, each binding its
  // parameters with spread set.
  if (plan.configuration().spread)
    for (const ComputeDispatch &dispatch : graph.dispatches()) {
      if (dispatch.pipelineName.starts_with("gguf_rotate")) continue;
      const BytesBinding &params = dispatch.bytes.front();
      uint32_t spread = 0;
      if (params.sizeBytes == sizeof(GgufDecodeParams))
        spread = static_cast<const GgufDecodeParams *>(params.data)->spread;
      else if (params.sizeBytes == sizeof(GgufDecodeFusedParams))
        spread = static_cast<const GgufDecodeFusedParams *>(params.data)->spread;
      if (spread != 1) fail(label + ": " + dispatch.pipelineName + " walks K in lockstep");
    }
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  if (!scratch.intact() || !o.input.intact() || !o.aux.intact() || !o.output.intact() || !o.gate.intact())
    fail(label + ": a counter is not reset or a write past a buffer");
  Outcome out{o.output.halves(),
              plan.workload().epilogue == LinearEpilogue::GateUp ? o.gate.halves() : std::vector<uint16_t>{}, {}};
  for (const ComputeDispatch &dispatch : graph.dispatches()) out.pipelines.push_back(dispatch.pipelineName);
  const uint32_t columns = plan.workload().matrix.outputSize;
  for (const std::vector<uint16_t> *rows : {&out.output, &out.gate})
    for (uint64_t i = 0; i < rows->size(); ++i)
      if (i % columns >= covered && (*rows)[i] != kUnwritten) {
        fail(label + ": writes padding column " + std::to_string(i % columns));
        break;
      }
  return out;
}

// Every output of rows [0, rows) and the covered columns within its interval: the projection's (`want`), plus the
// residual, or times silu of the gate the up pass read (a gate/up plan's gate scratch, itself a projection: `gate`).
void checkValues(const Outcome &out, const Want &want, const Want &gate, const std::vector<uint16_t> &aux,
                 const LinearWorkload &w, uint32_t rows, uint32_t covered, const std::string &label) {
  const uint32_t columns = w.matrix.outputSize;
  uint64_t outside = 0;
  std::string first;
  const auto check = [&](const char *what, uint64_t i, uint16_t got, Interval want) {
    if (inside(got, want) || outside++) return;
    first = std::string(what) + " of row " + std::to_string(i / columns) + " column " + std::to_string(i % columns) +
            " is " + digits(bf16ToFloat(got)) + ", not in [" + digits(bf16(want.lo)) + ", " + digits(bf16(want.hi)) +
            "]";
  };
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < covered; ++c) {
      const uint64_t i = uint64_t{r} * columns + c;
      const Interval p = want(i);
      switch (w.epilogue) {
        case LinearEpilogue::None: check("the output", i, out.output[i], p); break;
        case LinearEpilogue::Residual:
          check("the output", i, out.output[i], withResidual(p, bf16ToFloat(aux[i])));
          break;
        case LinearEpilogue::UpWithGate: check("the output", i, out.output[i], withGate(p, bf16ToFloat(aux[i]))); break;
        case LinearEpilogue::GateUp:
          check("the gate", i, out.gate[i], gate(i));
          check("the output", i, out.output[i], withGate(p, bf16ToFloat(out.gate[i])));
          break;
      }
    }
  if (outside) fail(label + ": " + std::to_string(outside) + " values outside the fp64 bound; " + first);
}

// The fp32 destination of `plan` (the logits) over the input rows `x` of `bf16`, the run of its bf16 plan: every
// value of the active rows and covered columns rounds to that run's output bit for bit and lies within its interval
// before the rounding; the padding columns, guards and counters stay as they were.
void floatOutput(MetalBackend &backend, const Linear &linear, const LinearPlan &plan, const Projection &p,
                 const std::vector<uint16_t> &x, const Outcome &bf16, const Want &want, uint32_t covered,
                 const std::string &label) {
  const uint32_t columns = plan.workload().matrix.outputSize;
  const Scratch scratch(backend, scratchSize(linear, plan, p));
  const Guarded input = bfloats(backend, x);
  const Guarded output(backend, uint64_t{plan.storageRows()} * columns * sizeof(float), 0xFF);
  scratch.poison(kPoisonNaN);
  CommandGraph graph;
  static_cast<void>(linear.add(graph, {.input = input.view, .output = output.view, .scratch = scratch.bindings()},
                               p, plan));
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  if (!scratch.intact() || !input.intact() || !output.intact())
    fail(label + " fp32: a counter is not reset or a write past a buffer");
  const auto *values = static_cast<const uint32_t *>(output.view.contents());
  uint64_t wrong = 0;
  for (uint64_t i = 0; i < uint64_t{plan.workload().rows} * columns; ++i) {
    const float value = std::bit_cast<float>(values[i]);
    if (i % columns >= covered) {
      wrong += values[i] != 0xFFFFFFFFu;
      continue;
    }
    const Interval bound = want(i);
    wrong += floatToBf16(value) != bf16.output[i] || !(value >= bound.lo && value <= bound.hi);
  }
  if (wrong) fail(label + " fp32: " + std::to_string(wrong) + " values differ from the bf16 run or their interval");
}

bool sameRows(const std::vector<uint16_t> &a, uint64_t aRow, const std::vector<uint16_t> &b, uint64_t bRow,
              uint32_t rows, uint32_t columns) {
  return std::equal(a.begin() + aRow * columns, a.begin() + (aRow + rows) * columns, b.begin() + bRow * columns);
}

// The configuration of `tile` for a decode workload, or of the staged tile for a prefill chunk of up to 64 rows; the
// staged tile walks K spread or in lockstep.
LinearConfig config(LinearTile tile, const LinearWorkload &w, uint32_t splits, bool spread = false) {
  return {.tile = w.phase == LinearPhase::Prefill ? LinearTile::GgufStaged : tile, .splits = splits, .spread = spread};
}
// The formats whose weights `tile` takes, in id order: the register tile takes those of quant_register_format.
std::vector<Fmt> tileFormats(LinearTile tile) {
  std::vector<Fmt> formats;
  for (int f = 0; f < FMT_COUNT; ++f)
    if (tile != LinearTile::GgufRegister || quant_register_format(uint32_t(f))) formats.push_back(Fmt(f));
  return formats;
}
// Whether `tile` splits K inputs in `format` `splits` ways: the register tile into whole coefficient units
// (quant_register_unit), the staged tile into whole 32-input groups.
bool takesSplits(LinearTile tile, Fmt format, uint32_t K, uint32_t splits) {
  return tile == LinearTile::GgufRegister ? K / quant_register_unit(format) >= splits : (K / 32) % splits == 0;
}
// The tile and its walk in labels.
std::string walkName(LinearTile tile, bool spread) { return std::string(tileName(tile)) + (spread ? " spread" : ""); }
// K of the decode and fused sections: 2048, or for the spread walk 2304, whose 72 steps make partitions of 72, 36, 18
// and 9 steps: tiles start inside them and wrap past their ends, and at 9 the walk wraps from an even step to an even
// one, so the stages alternate per step walked, not by the step's parity.
uint32_t decodeInputs(bool spread) { return spread ? 2304 : 2048; }
LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  return {matrix, lanes * kLaneRows, LinearPhase::Decode, epilogue};
}

// ---------------------------------------------------------------- decode
// One tile and walk on single tensors [512, K] in each of `formats` (gate in the format three further on of those):
// at every lane count, K split the tile takes and epilogue, on dense and sparse inputs, every output within fp64 and
// each lane's rows equal to a one-lane projection of them, on the staged tile's 16-row tile too.
void decodeTile(MetalBackend &backend, const Linear &linear, LinearTile tile, bool spread, uint32_t K,
                const std::vector<Fmt> &formats) {
  constexpr uint32_t N = 512, columns = N + kPadding;
  const bool staged = tile == LinearTile::GgufStaged;
  uint32_t mostSplits = 1;
  for (size_t i = 0; i < formats.size(); ++i) {
    const Tensor w = tensor(backend, formats[i], N, K), g = tensor(backend, formats[(i + 3) % formats.size()], N, K);
    const Projection up = projection({&w}, columns), gate = projection({&g}, columns);
    for (const Inputs inputs : {Inputs::Dense, Inputs::Sparse}) {
      const std::vector<float> x = activations(inputs, kMaximumRows, K);
      const std::vector<float> residual = residuals(kMaximumRows, columns);
      const std::vector<Dot> dots = products({&w}, columns, x, kMaximumRows);
      const std::vector<Dot> gateDots = products({&g}, columns, x, kMaximumRows);
      for (const LinearEpilogue epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
        for (const uint32_t splits : kSplits) {
          if (!takesSplits(tile, w.format, K, splits) || !takesSplits(tile, g.format, K, splits)) continue;
          mostSplits = std::max(mostSplits, splits);
          const std::string what = std::string(fmtName(w.format)) +
                                   (inputs == Inputs::Sparse ? " sparse " : " dense ") + epilogueName(epilogue) +
                                   " S=" + std::to_string(splits);
          const Projection *gated = epilogue == LinearEpilogue::GateUp ? &gate : nullptr;
          std::vector<Outcome> lanesAlone;
          for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
            const std::vector<float> rows(x.begin() + uint64_t{lane} * kLaneRows * K,
                                          x.begin() + uint64_t{lane + 1} * kLaneRows * K);
            const std::vector<float> aux(residual.begin() + uint64_t{lane} * kLaneRows * columns,
                                         residual.begin() + uint64_t{lane + 1} * kLaneRows * columns);
            const LinearWorkload one = decode({columns, K}, 1, epilogue);
            const LinearPlan plan = Linear::plan(one, config(tile, one, splits, spread), FloatOutput::BFloat16);
            lanesAlone.push_back(run(backend, linear, plan, up, gated, storageRows(rows, K, kLaneRows, kLaneRows),
                                     storageRows(aux, columns, kLaneRows, kLaneRows), kPoisonNaN, N,
                                     what + " lane " + std::to_string(lane) + " alone"));
          }
          // The first lane on the 16-row tile, over a padding lane of NaN rows: the 8-row tile's bits, and the fp32
          // plain outputs rounding to them.
          if (staged) {
            const LinearWorkload one = decode({columns, K}, 1, epilogue);
            LinearConfig sixteen = config(tile, one, splits, spread);
            sixteen.sixteenRows = true;
            const std::string label = what + " lane 0 on the 16-row tile";
            const std::vector<uint16_t> rows = storageRows(x, K, kLaneRows, 2 * kLaneRows);
            const Outcome out = run(backend, linear, Linear::plan(one, sixteen, FloatOutput::BFloat16), up, gated, rows,
                                    storageRows(residual, columns, kLaneRows, 2 * kLaneRows), kPoisonFinite, N, label);
            if (!sameRows(out.output, 0, lanesAlone[0].output, 0, kLaneRows, columns))
              fail(label + ": differs from the 8-row tile");
            if (epilogue == LinearEpilogue::None)
              floatOutput(backend, linear, Linear::plan(one, sixteen, FloatOutput::Float32), up, rows, out,
                          within(dots, staged), N, label);
          }
          for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
            const LinearWorkload wl = decode({columns, K}, lanes, epilogue);
            const LinearPlan plan = Linear::plan(wl, config(tile, wl, splits, spread), FloatOutput::BFloat16);
            const uint32_t storage = plan.storageRows(), rows = wl.rows;
            const std::string label = what + " L=" + std::to_string(lanes);
            const std::vector<uint16_t> aux = storageRows(residual, columns, rows, storage);
            const Outcome out = run(backend, linear, plan, up, gated, storageRows(x, K, rows, storage), aux,
                                    kPoisonFinite, N, label);
            checkValues(out, within(dots, staged), within(gateDots, staged), aux, wl, rows, N, label);
            if (epilogue == LinearEpilogue::None)
              floatOutput(backend, linear, Linear::plan(wl, config(tile, wl, splits, spread), FloatOutput::Float32), up,
                          storageRows(x, K, rows, storage), out, within(dots, staged), N, label);
            for (uint32_t lane = 0; lane < lanes; ++lane)
              if (!sameRows(out.output, lane * kLaneRows, lanesAlone[lane].output, 0, kLaneRows, columns))
                fail(label + ": lane " + std::to_string(lane) + " differs from its one-lane projection");
          }
        }
    }
  }
  section(walkName(tile, spread) + " decode: " + std::to_string(formats.size()) + " formats, K " + std::to_string(K) +
          ", 1-4 lanes, S 1-" + std::to_string(mostSplits) +
          ", plain/residual/gate-up, dense and sparse inputs within fp64, lanes equal to one-lane projections" +
          (staged ? " (one lane on the 16-row tile too)" : "") + ", fp32 plain outputs rounding to bf16's");
}

// One fused projection of three segments of different formats ([512 | 256 | 256, K]) on one tile and walk, each format
// it takes with the formats three and five further on of those: at every lane count and K split the tile takes in
// all three, within fp64 and equal to the projections of each segment alone; one lane on the staged 16-row tile equal
// to the 8-row tile. The register tile's K holds a 256-input unit per partition at the most splits, so that every
// triple takes every split.
void fusedDecode(MetalBackend &backend, const Linear &linear, LinearTile tile, bool spread) {
  const uint32_t K = tile == LinearTile::GgufRegister ? 256 * LinearConfig::kMaximumSplits : decodeInputs(spread);
  const std::vector<Fmt> formats = tileFormats(tile);
  uint32_t mostSplits = 1;
  for (size_t i = 0; i < formats.size(); ++i) {
    const Tensor a = tensor(backend, formats[i], 512, K),
                 b = tensor(backend, formats[(i + 3) % formats.size()], 256, K),
                 c = tensor(backend, formats[(i + 5) % formats.size()], 256, K);
    const std::vector<const Tensor *> parts{&a, &b, &c};
    const uint32_t covered = segmentColumns(parts), columns = covered + kPadding;
    const Projection fused = projection(parts, columns);
    const std::vector<float> x = activations(Inputs::Dense, kMaximumRows, K);
    const std::vector<Dot> dots = products(parts, columns, x, kMaximumRows);
    const std::string formats = std::string(fmtName(a.format)) + "|" + fmtName(b.format) + "|" + fmtName(c.format);
    for (const uint32_t splits : kSplits) {
      if (!std::all_of(parts.begin(), parts.end(),
                       [&](const Tensor *part) { return takesSplits(tile, part->format, K, splits); }))
        continue;
      mostSplits = std::max(mostSplits, splits);
      for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
        const LinearWorkload wl = decode({columns, K}, lanes, LinearEpilogue::None);
        const LinearPlan plan = Linear::plan(wl, config(tile, wl, splits, spread), FloatOutput::BFloat16);
        const uint32_t storage = plan.storageRows(), rows = wl.rows;
        const std::string label = formats + " S=" + std::to_string(splits) + " L=" + std::to_string(lanes);
        const Outcome out = run(backend, linear, plan, fused, nullptr, storageRows(x, K, rows, storage), {},
                                kPoisonNaN, covered, label);
        checkValues(out, within(dots, tile == LinearTile::GgufStaged), {}, {}, wl, rows, covered, label);
        // One lane on the 16-row tile, over a padding lane of NaN rows: the 8-row tile's bits.
        if (lanes == 1 && tile == LinearTile::GgufStaged) {
          LinearConfig sixteen = config(tile, wl, splits, spread);
          sixteen.sixteenRows = true;
          const Outcome wide = run(backend, linear, Linear::plan(wl, sixteen, FloatOutput::BFloat16), fused, nullptr,
                                   storageRows(x, K, rows, 2 * kLaneRows), {}, kPoisonFinite, covered,
                                   label + " on the 16-row tile");
          if (!sameRows(wide.output, 0, out.output, 0, rows, columns))
            fail(label + ": the 16-row tile differs from the 8-row tile");
        }
        uint32_t offset = 0;
        for (const Tensor *part : parts) {
          const Projection alone = projection({part}, part->N);
          const LinearWorkload one = decode({part->N, K}, lanes, LinearEpilogue::None);
          const Outcome single =
              run(backend, linear, Linear::plan(one, config(tile, one, splits, spread), FloatOutput::BFloat16), alone,
                  nullptr, storageRows(x, K, rows, storage), {}, kPoisonNaN, part->N, label + " alone");
          for (uint32_t r = 0; r < rows; ++r)
            if (!std::equal(single.output.begin() + uint64_t{r} * part->N, single.output.begin() + (r + 1) * part->N,
                            out.output.begin() + uint64_t{r} * columns + offset)) {
              fail(label + ": the " + fmtName(part->format) + " segment differs from its projection alone");
              break;
            }
          offset += part->N;
        }
      }
    }
  }
  section(walkName(tile, spread) + " fused: three segments in " + std::to_string(formats.size()) +
          " format triples, K " + std::to_string(K) + ", 1-4 lanes, S 1-" + std::to_string(mostSplits) +
          " within fp64 and equal to each segment's projection" +
          (tile == LinearTile::GgufStaged ? ", one lane on the 16-row tile equal to the 8-row tile" : ""));
}

// Gate/up on one tile for the gate of each format it takes with the up of the next one ([256, 1024]), so that every
// such format runs as both, and four pairs at [1024, 1024], with the K splits by lanes a decode step of these widths
// takes on large GPUs.
void gateUpPairs(MetalBackend &backend, const Linear &linear, LinearTile tile) {
  constexpr uint32_t K = 1024;
  constexpr uint32_t kSplitsByLanes[kMaximumLanes] = {1, 2, 4, 4};
  const auto pair = [&](Fmt gf, Fmt uf, uint32_t N, uint32_t lanes) {
    const uint32_t columns = N + kPadding;
    const Tensor g = tensor(backend, gf, N, K), u = tensor(backend, uf, N, K);
    const Projection gate = projection({&g}, columns), up = projection({&u}, columns);
    const LinearWorkload wl = decode({columns, K}, lanes, LinearEpilogue::GateUp);
    const LinearPlan plan = Linear::plan(wl, config(tile, wl, kSplitsByLanes[lanes - 1]), FloatOutput::BFloat16);
    const std::vector<float> x = activations(Inputs::Dense, wl.rows, K);
    const std::string label = std::string(fmtName(gf)) + " gate " + fmtName(uf) + " up N=" + std::to_string(N) +
                              " L=" + std::to_string(lanes);
    const Outcome out = run(backend, linear, plan, up, &gate, storageRows(x, K, wl.rows, plan.storageRows()), {},
                            kPoisonNaN, N, label);
    const bool staged = tile == LinearTile::GgufStaged;
    checkValues(out, within(products({&u}, columns, x, wl.rows), staged),
                within(products({&g}, columns, x, wl.rows), staged), {}, wl, wl.rows, N, label);
  };
  const std::vector<Fmt> formats = tileFormats(tile);
  for (size_t i = 0; i < formats.size(); ++i)
    for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
      pair(formats[i], formats[(i + 1) % formats.size()], 256, lanes);
  const std::array<std::array<Fmt, 2>, kMaximumLanes> wide{{{IQ4XS, Q4K}, {Q5K, Q5K}, {Q4K, IQ4XS}, {Q3K, Q6K}}};
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) pair(wide[lanes - 1][0], wide[lanes - 1][1], 1024, lanes);
  section(std::string(tileName(tile)) + " gate/up: " + std::to_string(formats.size()) +
          " gate and up format pairs at N 256 and 4 at N 1024, 1-4 lanes, within fp64");
}

// Gate/up in one pass (LinearConfig::oneGateUpPass) on the staged tile, gate and up in each format ([768, K]), and in
// Q4_K and MLX 4-bit at the 27B's [17408, 5120]: at every lane count, K split the tile takes and walk, and for one
// lane on the 16-row tile too, bitwise the output of the gate and up passes, whatever the partials held before.
void oneGateUpPass(MetalBackend &backend, const Linear &linear, bool spread) {
  uint32_t mostSplits = 1;
  const auto pairs = [&](Fmt f, uint32_t N, uint32_t K, uint32_t columns, std::span<const uint32_t> lanes) {
    const Tensor g = tensor(backend, f, N, K), u = tensor(backend, f, N, K);
    const Projection gate = projection({&g}, columns), up = projection({&u}, columns);
    const std::vector<float> x = activations(Inputs::Dense, kMaximumRows, K);
    for (const uint32_t splits : kSplits) {
      if (!takesSplits(LinearTile::GgufStaged, f, K, splits)) continue;
      mostSplits = std::max(mostSplits, splits);
      for (const uint32_t l : lanes) {
        const LinearWorkload wl = decode({columns, K}, l, LinearEpilogue::GateUp);
        LinearConfig c = config(LinearTile::GgufStaged, wl, splits, spread);
        const LinearPlan twoPasses = Linear::plan(wl, c, FloatOutput::BFloat16);
        c.oneGateUpPass = true;
        const LinearPlan onePass = Linear::plan(wl, c, FloatOutput::BFloat16);
        const std::string label = std::string(fmtName(f)) + " " + std::to_string(N) + "x" + std::to_string(K) +
                                  " S=" + std::to_string(splits) + " L=" + std::to_string(l);
        const std::vector<uint16_t> rows = storageRows(x, K, wl.rows, onePass.storageRows());
        const Outcome two = run(backend, linear, twoPasses, up, &gate, rows, {}, kPoisonNaN, N, label + " two passes");
        const Outcome one = run(backend, linear, onePass, up, &gate, rows, {}, kPoisonFinite, N, label);
        if (!sameRows(one.output, 0, two.output, 0, wl.rows, columns)) fail(label + ": differs from the two passes");
        // One lane on the 16-row tile, over a padding lane of NaN rows.
        if (l > 1) continue;
        c.sixteenRows = true;
        const LinearPlan sixteen = Linear::plan(wl, c, FloatOutput::BFloat16);
        const Outcome wide = run(backend, linear, sixteen, up, &gate, storageRows(x, K, wl.rows, sixteen.storageRows()),
                                 {}, kPoisonFinite, N, label + " on the 16-row tile");
        if (!sameRows(wide.output, 0, two.output, 0, wl.rows, columns))
          fail(label + ": the 16-row tile differs from the two passes");
      }
    }
  };
  const uint32_t K = decodeInputs(spread);
  constexpr uint32_t kAllLanes[] = {1, 2, 3, 4}, kEndLanes[] = {1, kMaximumLanes};
  for (int fi = 0; fi < FMT_COUNT; ++fi) pairs(Fmt(fi), 768, K, 768 + kPadding, kAllLanes);
  for (const Fmt f : {Q4K, Fmt(GGUF_FMT_AF4G64)}) pairs(f, 17408, 5120, 17408, kEndLanes);
  section(walkName(LinearTile::GgufStaged, spread) + " one-pass gate/up: " + std::to_string(FMT_COUNT) +
          " formats at 768x" + std::to_string(K) + ", 1-4 lanes, and Q4_K and MLX 4-bit at 17408x5120, 1 and 4 lanes, "
          "S 1-" + std::to_string(mostSplits) + ", one lane on the 16-row tile too, bitwise the two passes");
}

// ---------------------------------------------------------------- prefill
// 168- and 136-row chunks on the 128-row tiles: the second tile holds 40 or 8 rows, so its last two or three 32-row
// simdgroups skip their matmuls, though not the barriers of the stage they share, and leave the rows from 192 or 160
// unwritten. Chunks of up to 32 rows run the decode tiles, whose rows equal the 128-row tiles' bitwise without K
// splits and lie within fp64 with them; chunks of 33-64 rows run the 64-row decode tile, alike, and also at eight
// partitions, which the split law takes for some shapes, where K takes them. Every chunk's last 1 and 4 rows, on its
// tile and splits at their own height, equal its rows bitwise. Each of `formats` ([1024, K]) at each epilogue, and a
// fused projection at its segments' column offsets.
void prefill(MetalBackend &backend, const Linear &linear, uint32_t K, const std::vector<Fmt> &formats) {
  constexpr uint32_t kChunks[] = {168, 136}, kSimdgroupRows = 32;
  // The split chunks' partitions: the most up to four that take whole 32-input groups.
  const uint32_t splitChunk = std::gcd(K / 32, 4u);
  constexpr LinearConfig kTiles{.tile = LinearTile::GgufPrefill};
  const auto chunks = [&](const Projection &p, const std::vector<const Tensor *> &parts, LinearEpilogue epilogue,
                          const std::string &what) {
    const uint32_t covered = segmentColumns(parts), columns = p.outputSize;
    const auto workload = [&](uint32_t rows) {
      return LinearWorkload{{columns, K}, rows, LinearPhase::Prefill, epilogue};
    };
    // Both chunks take two tiles.
    const uint32_t storage = Linear::plan(workload(kChunks[0]), kTiles, FloatOutput::BFloat16).storageRows();
    const std::vector<float> x = activations(Inputs::Dense, storage, K);
    // The residual, or the gate of the up-with-gate epilogue.
    const std::vector<float> auxiliary = epilogue == LinearEpilogue::Residual
        ? residuals(storage, columns) : activations(Inputs::Dense, storage, columns);
    const std::vector<uint16_t> aux = storageRows(auxiliary, columns, storage, storage);
    const std::vector<Dot> dots = products(parts, columns, x, kChunks[0]);
    const std::string label = what + " " + epilogueName(epilogue);
    // A chunk's last rows on its plan's tile and splits at their own rows'
    // height, as a prefill's last layer runs the rows something reads
    // (Linear::prefillPlan of a chunk's rows), write the chunk's bits.
    const auto lastRows = [&](const LinearPlan &plan, const Outcome &out, const std::string &name) {
      const uint32_t chunk = plan.workload().rows;
      for (const uint32_t rows : {1u, 4u}) {
        const LinearPlan part = Linear::plan(workload(rows), plan.configuration(), FloatOutput::BFloat16);
        const auto last = [&](const std::vector<float> &values, uint32_t width) {
          const auto end = values.begin() + uint64_t{chunk} * width;
          return storageRows(std::vector<float>(end - uint64_t{rows} * width, end), width, rows, part.storageRows());
        };
        const std::string own = name + ", its last " + std::to_string(rows);
        const Outcome alone =
            run(backend, linear, part, p, nullptr, last(x, K), last(auxiliary, columns), kPoisonNaN, covered, own);
        if (!sameRows(alone.output, 0, out.output, chunk - rows, rows, columns)) fail(own + ": differ from the chunk's");
      }
    };
    std::vector<uint16_t> tileOutput;  // the first chunk's output on the 128-row tiles
    for (const uint32_t chunk : kChunks) {
      const LinearWorkload w = workload(chunk);
      const std::string name = label + " " + std::to_string(chunk) + " rows";
      const LinearPlan tiles = Linear::plan(w, kTiles, FloatOutput::BFloat16);
      const Outcome whole =
          run(backend, linear, tiles, p, nullptr, storageRows(x, K, chunk, storage), aux, kPoisonNaN, covered, name);
      checkValues(whole, within(dots, true), {}, aux, w, chunk, covered, name);
      lastRows(tiles, whole, name);
      const uint32_t written = (chunk + kSimdgroupRows - 1) / kSimdgroupRows * kSimdgroupRows;
      if (!std::all_of(whole.output.begin() + uint64_t{written} * columns, whole.output.end(),
                       [](uint16_t v) { return v == kUnwritten; }))
        fail(name + ": simdgroups past the chunk write their rows");
      if (tileOutput.empty()) tileOutput = whole.output;
    }
    for (const uint32_t rows : {8u, 16u, 24u, 32u, 33u, 40u, 48u, 57u, 64u})
      for (const uint32_t splits : {1u, splitChunk, 8u}) {
        if (splits == 8 && (rows <= 32 || (K / 32) % 8))
          continue;
        const LinearWorkload chunk{{columns, K}, rows, LinearPhase::Prefill, epilogue};
        const LinearPlan plan =
            Linear::plan(chunk, config(LinearTile::GgufStaged, chunk, splits), FloatOutput::BFloat16);
        const std::string name = label + " chunk of " + std::to_string(rows) + " rows S=" + std::to_string(splits);
        const std::vector<uint16_t> chunkAux(aux.begin(), aux.begin() + uint64_t{plan.storageRows()} * columns);
        const Outcome out = run(backend, linear, plan, p, nullptr, storageRows(x, K, rows, plan.storageRows()),
                                chunkAux, kPoisonNaN, covered, name);
        checkValues(out, within(dots, true), {}, chunkAux, chunk, rows, covered, name);
        lastRows(plan, out, name);
        if (splits == 1 && !sameRows(out.output, 0, tileOutput, 0, rows, columns))
          fail(name + ": differs from the 128-row tiles");
        // A fused chunk runs its segments in one dispatch, each segment's columns bitwise its own chunk's at the
        // same split.
        if (parts.size() > 1) {
          if (out.pipelines.size() != 1)
            fail(name + ": ran " + std::to_string(out.pipelines.size()) + " dispatches, not one");
          uint32_t offset = 0;
          for (const Tensor *t : parts) {
            const LinearWorkload alone{{t->N, K}, rows, LinearPhase::Prefill, epilogue};
            const LinearPlan own = Linear::plan(alone, config(LinearTile::GgufStaged, alone, splits),
                                                FloatOutput::BFloat16);
            const Outcome segment = run(backend, linear, own, projection({t}, t->N), nullptr,
                                        storageRows(x, K, rows, own.storageRows()), chunkAux, kPoisonNaN, t->N,
                                        name + " " + fmtName(t->format) + " alone");
            for (uint32_t r = 0; r < rows; ++r)
              if (!std::equal(segment.output.begin() + uint64_t{r} * t->N,
                              segment.output.begin() + uint64_t{r + 1} * t->N,
                              out.output.begin() + uint64_t{r} * columns + offset))
                fail(name + ": " + fmtName(t->format) + "'s columns differ from its own chunk's");
            offset += t->N;
          }
        }
      }
  };
  for (size_t i = 0; i < formats.size(); ++i) {
    const Tensor w = tensor(backend, formats[i], 1024, K);
    const Projection p = projection({&w}, w.N);
    for (const LinearEpilogue e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      chunks(p, {&w}, e, fmtName(w.format));
    const Tensor b = tensor(backend, formats[(i + 3) % formats.size()], 256, K),
                 c = tensor(backend, formats[(i + 5) % formats.size()], 256, K);
    const std::vector<const Tensor *> parts{&w, &b, &c};
    chunks(projection(parts, segmentColumns(parts) + kPadding), parts, LinearEpilogue::None,
           std::string(fmtName(w.format)) + "|" + fmtName(b.format) + "|" + fmtName(c.format));
  }
  section("prefill: " + std::to_string(formats.size()) + " formats, K " + std::to_string(K) +
          ", plain/residual/up-with-gate and fused segments, 128-row tiles over 168- and 136-row chunks and chunks of "
          "8-64 rows (S 1 and " + std::to_string(splitChunk) + (K / 32 % 8 ? "" : ", and 8 past 32 rows") +
          ") within fp64, equal to the 128-row tiles; a fused chunk in one dispatch, bitwise its segments' own; a "
          "chunk's last rows alone bitwise the chunk's");
}

// ---------------------------------------------------------------- int8 decode
// Rotated PQ2_0 projections in int8 activations (LinearConfig::decodeInt8, which selection takes on plans of
// ProjectionCalibration::decodeInt8FromRows rows or more): gguf_rotate_int8 codes each rotation block of the rotated
// input by the int8-activation rule (kernels/common/int8_activations.h; LinearNumerics.hpp's int8Block), and the int8
// tiles (gguf_decode_int8_*) multiply the codes by the weights' codes exactly and fold each PQ2_0 block's int32 sum
// into fp32 as acc * d * s. Every output lies within the rounding of that fold's fp32 arithmetic of the exact sum
// (int8Products, int8Bound), through each epilogue, and the fp32 destination, which rounds to the bf16 run bit for
// bit, takes a code, a scale or a block that is another's, or rounded another way, past it. The inputs' rotation
// blocks take maxima of 2^-3 to 2^3 by row and block, a block of every third row codes two halves by rounding them to
// even (int8Inputs), and the rotated scratch starts NaN, as the padding rows the rotation leaves are then read.
// Single tensors at one to four lanes (one on the 16-row tile: the int8 tiles hold 16 or 32 rows), every K split that
// keeps whole PQ2_0 blocks, both walks; the first lane's rows equal a one-lane projection of them bitwise, and a
// gate/up pair in one pass the two passes; three fused segments; the production plans on [8192, 5120] and
// [5120, 17408] where they take int8 activations, which they never take over unrotated weights. A NaN or an infinity
// in a later block of a row's inputs leaves that row's outputs alone not finite, on the int8 tiles as on the bf16
// tiles. The int8 plans run where the device's row takes int8 activations.

// The bf16 rotated rows H (D x) / 32 the int8 rows quantize (gguf_rotate's, which gguf-rotation checks).
std::vector<float> rotatedRows(const std::vector<float> &x, uint32_t rows, uint32_t K,
                               const std::vector<int8_t> &signs) {
  std::vector<float> out(uint64_t{rows} * K);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t b = 0; b < K; b += GGUF_ROTATION_BLOCK) {
      float *v = out.data() + uint64_t{r} * K + b;
      for (uint32_t i = 0; i < GGUF_ROTATION_BLOCK; ++i) v[i] = x[uint64_t{r} * K + b + i] * float(signs[b + i]);
      rotationButterflies(v);
      for (uint32_t i = 0; i < GGUF_ROTATION_BLOCK; ++i) v[i] = bf16(v[i] * kRotationScale);
    }
  return out;
}
// bf16 inputs of `rows` rows of K inputs whose rotation block b of row r is dense at 2^((r + 3b) mod 7 - 3), so that
// a scale taken from another block shows, but for block (r / 3) mod (K / 1024) of every third row r, which rotates
// to 63.5 at one input and to 0.25 and -0.25 at two (the inputs D (H y) / 32 of that y are bf16 values and its
// butterflies exact): codes 127, 0 and 0, the halves rounded to even.
std::vector<float> int8Inputs(uint32_t rows, uint32_t K, const std::vector<int8_t> &signs) {
  std::vector<float> x = activations(Inputs::Dense, rows, K);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t k = 0; k < K; ++k)
      x[uint64_t{r} * K + k] = std::ldexp(x[uint64_t{r} * K + k], int((r + 3 * (k / GGUF_ROTATION_BLOCK)) % 7) - 3);
  std::vector<double> y(GGUF_ROTATION_BLOCK);
  for (uint32_t r = 0; r < rows; r += 3) {
    const uint32_t first = (r / 3) % (K / GGUF_ROTATION_BLOCK) * GGUF_ROTATION_BLOCK;
    std::fill(y.begin(), y.end(), 0.0);
    y[9] = 63.5, y[300] = 0.25, y[777] = -0.25;
    rotationButterflies(y.data());
    for (uint32_t i = 0; i < GGUF_ROTATION_BLOCK; ++i) {
      const double v = y[i] / 32 * signs[first + i];
      if (double(bf16(v)) != v) throw std::logic_error("an int8 tie input is no bf16 value");
      x[uint64_t{r} * K + first + i] = float(v);
    }
  }
  return x;
}
// The int8 tiles' outputs, exactly: the rotated rows' blocks coded by the int8-activation rule, as gguf_rotate_int8
// codes them, and output (r, c) the sum over the row's PQ2_0 blocks of acc * d * s (acc the block's sum of the codes
// times the weights' codes - 1, d its scale, s the row block's), in fp64, where each block's term is exact; and
// sum_b |acc d s|, the magnitude their fp32 arithmetic rounds within.
struct Int8Products {
  std::vector<double> value, magnitude;
};
Int8Products int8Products(const std::vector<const Tensor *> &parts, uint32_t columns, const std::vector<float> &x,
                          uint32_t rows, const std::vector<int8_t> &signs) {
  const uint32_t K = parts.front()->K, blocks = K / GGUF_ROTATION_BLOCK, unit = 32 * kQuantFormats[PQ20].meta_groups;
  const std::vector<float> rotated = rotatedRows(x, rows, K, signs);
  std::vector<float> codes(rotated.size()), scales(uint64_t{rows} * blocks);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t b = 0; b < blocks; ++b) {
      const float *v = rotated.data() + uint64_t{r} * K + b * GGUF_ROTATION_BLOCK;
      float largest = 0;
      for (uint32_t i = 0; i < GGUF_ROTATION_BLOCK; ++i) largest = std::max(largest, std::fabs(v[i]));
      const test::Int8Block block = test::int8Block(largest);
      scales[uint64_t{r} * blocks + b] = block.scale;
      for (uint32_t i = 0; i < GGUF_ROTATION_BLOCK; ++i)
        codes[uint64_t{r} * K + b * GGUF_ROTATION_BLOCK + i] = float(test::int8Code(v[i], block));
    }
  Int8Products out{std::vector<double>(uint64_t{rows} * columns), std::vector<double>(uint64_t{rows} * columns)};
  double *value = out.value.data(), *magnitude = out.magnitude.data();
  const float *code = codes.data(), *scale = scales.data();
  uint32_t offset = 0;
  for (const Tensor *t : parts) {
    const Fmt f = t->format;
    const uint8_t *native = t->native.data();
    const uint32_t at = offset;
    dispatch_apply(t->N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t n) {
      std::vector<float> w(K);
      rowValues(f, native + n * rowBytes(f, K), K, w.data());
      for (uint32_t r = 0; r < rows; ++r) {
        double sum = 0, size = 0;
        for (uint32_t k0 = 0; k0 < K; k0 += unit) {
          // d * acc, exactly: the weights are d (q - 1), the codes integers.
          double block = 0;
          for (uint32_t k = k0; k < k0 + unit; ++k) block += double(code[uint64_t{r} * K + k]) * w[k];
          const double term = block * scale[uint64_t{r} * blocks + k0 / GGUF_ROTATION_BLOCK];
          sum += term;
          size += std::fabs(term);
        }
        value[uint64_t{r} * columns + at + n] = sum;
        magnitude[uint64_t{r} * columns + at + n] = size;
      }
    });
    offset += t->N;
  }
  return out;
}
// The int8 tiles' interval of each output over K inputs in `splits` partitions: the exact sum within
// (K / 128 + splits + 2) roundings of u = 2^-24 of sum_b |acc d s|, which bounds every partial sum: d * s and each
// block's fma round once, and the partitions' partials add in fp32.
Want int8Bound(const Int8Products &p, uint32_t K, uint32_t splits) {
  const double n = K / (32.0 * kQuantFormats[PQ20].meta_groups) + splits + 2, e = n * 0x1p-24 / (1 - n * 0x1p-24);
  return [&p, e](uint64_t i) { return Interval{p.value[i] - e * p.magnitude[i], p.value[i] + e * p.magnitude[i]}; };
}

void decodeInt8(MetalBackend &backend, const Linear &linear) {
  const bool int8Tiles = linear.calibration().projection.decodeInt8FromRows != kNever;
  uint32_t ran = 0, production = 0;
  // A rotated projection of `parts` and its sign buffer.
  const auto rotatedProjection = [&](const std::vector<const Tensor *> &parts, uint32_t columns,
                                     const MetalBuffer &signs) {
    Projection p = projection(parts, columns);
    p.rotation.signs = signs;
    return p;
  };
  const auto signsOf = [&](uint32_t K, std::vector<int8_t> &signs) {
    signs.resize(K);
    for (int8_t &s : signs) s = rng() & 1 ? 1 : -1;
    MetalBuffer buffer = test::sharedBuffer(backend, K);
    std::memcpy(buffer.contents(), signs.data(), K);
    return buffer;
  };
  // run() of an int8 plan, which must dispatch an int8 tile.
  const auto runInt8 = [&](const LinearPlan &plan, const Projection &p, const Projection *gate,
                           const std::vector<uint16_t> &x, const std::vector<uint16_t> &aux, uint32_t poison,
                           uint32_t covered, const std::string &label) {
    const Outcome out = run(backend, linear, plan, p, gate, x, aux, poison, covered, label);
    if (std::none_of(out.pipelines.begin(), out.pipelines.end(),
                     [](const std::string &name) { return name.starts_with("gguf_decode_int8_"); }))
      fail(label + ": runs no int8 tile");
    ++ran;
    return out;
  };
  // Single tensors [512, 3072] (96 steps: partitions of 96 down to 12 steps, walks that wrap past their ends).
  if (int8Tiles) {
    constexpr uint32_t N = 512, K = 3072, columns = N + kPadding;
    std::vector<int8_t> signs;
    const MetalBuffer signBuffer = signsOf(K, signs);
    const Tensor up = tensor(backend, PQ20, N, K), gateTensor = tensor(backend, PQ20, N, K);
    const Projection p = rotatedProjection({&up}, columns, signBuffer),
                     gate = rotatedProjection({&gateTensor}, columns, signBuffer);
    const std::vector<float> x = int8Inputs(kMaximumRows, K, signs);
    const std::vector<float> residual = residuals(kMaximumRows, columns);
    const Int8Products d = int8Products({&up}, columns, x, kMaximumRows, signs),
                       g = int8Products({&gateTensor}, columns, x, kMaximumRows, signs);
    for (const LinearEpilogue epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
      const Projection *gated = epilogue == LinearEpilogue::GateUp ? &gate : nullptr;
      for (const uint32_t splits : kSplits) {
        if (!decodeInt8TakesSplits(K, splits)) continue;
        for (const bool spread : {false, true}) {
          std::vector<uint16_t> oneLane;
          for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
            const LinearWorkload w = decode({columns, K}, lanes, epilogue);
            // One lane on the 16-row tile: the int8 tiles hold 16 or 32 rows.
            LinearConfig c{.tile = LinearTile::GgufStaged, .splits = splits, .spread = spread,
                           .sixteenRows = lanes == 1, .decodeInt8 = true};
            const LinearPlan plan = Linear::plan(w, c, FloatOutput::BFloat16);
            const uint32_t storage = plan.storageRows();
            const std::vector<uint16_t> rows = storageRows(x, K, w.rows, storage),
                                        aux = storageRows(residual, columns, w.rows, storage);
            const std::string label = std::string("int8 pq2_0 ") + epilogueName(epilogue) + " lanes " +
                                      std::to_string(lanes) + " S" + std::to_string(splits) + (spread ? " spread" : "");
            const Outcome out = runInt8(plan, p, gated, rows, aux, kPoisonNaN, N, label);
            checkValues(out, int8Bound(d, K, splits), int8Bound(g, K, splits), aux, w, w.rows, N, label);
            if (lanes == 1) oneLane = out.output;
            if (!sameRows(out.output, 0, oneLane, 0, kLaneRows, columns))
              fail(label + ": the first lane differs from one lane alone");
            if (epilogue == LinearEpilogue::GateUp) {
              c.oneGateUpPass = true;
              const Outcome pair = runInt8(Linear::plan(w, c, FloatOutput::BFloat16), p, &gate, rows, aux,
                                           kPoisonFinite, N, label + " one pass");
              if (!sameRows(pair.output, 0, out.output, 0, w.rows, columns))
                fail(label + ": one pass differs from two");
            }
            if (epilogue == LinearEpilogue::None) {
              floatOutput(backend, linear, Linear::plan(w, c, FloatOutput::Float32), p, rows, out,
                          int8Bound(d, K, splits), N, label);
              ++ran;
            }
          }
        }
      }
    }
  }
  // Three fused segments [256 + 256 + 512, 3072], at every lane count, unsplit and in 8 spread partitions.
  if (int8Tiles) {
    constexpr uint32_t K = 3072;
    std::vector<int8_t> signs;
    const MetalBuffer signBuffer = signsOf(K, signs);
    const Tensor a = tensor(backend, PQ20, 256, K), b = tensor(backend, PQ20, 256, K),
                 c3 = tensor(backend, PQ20, 512, K);
    const std::vector<const Tensor *> parts{&a, &b, &c3};
    const uint32_t covered = segmentColumns(parts), columns = covered + kPadding;
    const Projection p = rotatedProjection(parts, columns, signBuffer);
    const std::vector<float> x = int8Inputs(kMaximumRows, K, signs);
    const Int8Products d = int8Products(parts, columns, x, kMaximumRows, signs);
    for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
      for (const auto [splits, spread] : {std::pair{1u, false}, std::pair{8u, true}}) {
        const LinearWorkload w = decode({columns, K}, lanes, LinearEpilogue::None);
        const LinearConfig c{.tile = LinearTile::GgufStaged, .splits = splits, .spread = spread,
                             .sixteenRows = lanes == 1, .decodeInt8 = true};
        const LinearPlan plan = Linear::plan(w, c, FloatOutput::BFloat16);
        const std::string label = "int8 fused lanes " + std::to_string(lanes) + " S" + std::to_string(splits);
        const Outcome out =
            runInt8(plan, p, nullptr, storageRows(x, K, w.rows, plan.storageRows()), {}, kPoisonNaN, covered, label);
        checkValues(out, int8Bound(d, K, splits), {}, {}, w, w.rows, covered, label);
      }
  }
  // The production plans: Bonsai's GDN input and down shapes, those of them in int8 activations.
  for (const auto [N, K, epilogue] : {std::tuple{8192u, 5120u, LinearEpilogue::None},
                                      std::tuple{5120u, 17408u, LinearEpilogue::Residual}}) {
    std::vector<int8_t> signs;
    const MetalBuffer signBuffer = signsOf(K, signs);
    const Tensor t = tensor(backend, PQ20, N, K);
    const Projection p = rotatedProjection({&t}, N, signBuffer), plain = projection({&t}, N);
    const std::vector<float> x = int8Inputs(kMaximumRows, K, signs);
    const std::vector<float> residual = residuals(kMaximumRows, N);
    const Int8Products d = int8Tiles ? int8Products({&t}, N, x, kMaximumRows, signs) : Int8Products{};
    for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
      const LinearWorkload w = decode({N, K}, lanes, epilogue);
      if (linear.plan(w, plain).configuration().decodeInt8)
        fail("an unrotated PQ2_0 projection takes int8 activations");
      const LinearPlan plan = linear.plan(w, p);
      const LinearConfig c = plan.configuration();
      if (!c.decodeInt8) continue;
      const std::string label = "int8 production " + std::to_string(N) + "x" + std::to_string(K) + " lanes " +
                                std::to_string(lanes) + " S" + std::to_string(c.splits);
      if (!int8Tiles) {
        fail(label + ": a device row of no int8 activations took them");
        continue;
      }
      const uint32_t storage = plan.storageRows();
      const std::vector<uint16_t> aux = storageRows(residual, N, w.rows, storage);
      const Outcome out = runInt8(plan, p, nullptr, storageRows(x, K, w.rows, storage), aux, kPoisonNaN, N, label);
      checkValues(out, int8Bound(d, K, c.splits), {}, aux, w, w.rows, N, label);
      ++production;
    }
  }
  if (!int8Tiles) {
    section("int8 decode (PQ2_0): int8 plans not run (the device row takes none), none among the production plans");
    return;
  }
  // A NaN in one row's inputs and an infinity in another lane's, in blocks past the first, leave those rows' outputs
  // not finite, on the int8 tiles (kernels/common/int8_activations.h) as on the bf16 tiles, and every other row's as
  // they were.
  {
    constexpr uint32_t N = 512, K = 3072, kNaNRow = 3, kInfiniteRow = 18;
    std::vector<int8_t> signs;
    const MetalBuffer signBuffer = signsOf(K, signs);
    const Tensor t = tensor(backend, PQ20, N, K);
    const Projection p = rotatedProjection({&t}, N, signBuffer);
    const LinearWorkload w = decode({N, K}, kMaximumLanes, LinearEpilogue::Residual);
    const std::vector<float> x = activations(Inputs::Dense, w.rows, K), residual = residuals(w.rows, N);
    std::vector<float> poisoned = x;
    poisoned[uint64_t{kNaNRow} * K + 2 * GGUF_ROTATION_BLOCK + 5] = std::numeric_limits<float>::quiet_NaN();
    poisoned[uint64_t{kInfiniteRow} * K + GGUF_ROTATION_BLOCK + 700] = std::numeric_limits<float>::infinity();
    for (const bool int8 : {true, false}) {
      const LinearPlan plan =
          Linear::plan(w, {.tile = LinearTile::GgufStaged, .decodeInt8 = int8}, FloatOutput::BFloat16);
      const uint32_t storage = plan.storageRows();
      const std::vector<uint16_t> aux = storageRows(residual, N, w.rows, storage);
      const std::string label = std::string(int8 ? "int8" : "bf16") + " tiles, non-finite inputs";
      const Outcome clean =
          run(backend, linear, plan, p, nullptr, storageRows(x, K, w.rows, storage), aux, kPoisonNaN, N, label);
      const Outcome hit =
          run(backend, linear, plan, p, nullptr, storageRows(poisoned, K, w.rows, storage), aux, kPoisonNaN, N, label);
      for (uint32_t r = 0; r < w.rows; ++r) {
        const auto row = [&](const Outcome &o) { return std::span(o.output).subspan(uint64_t{r} * N, N); };
        const bool bad = r == kNaNRow || r == kInfiniteRow;
        if (bad ? !std::ranges::none_of(row(hit), [](uint16_t v) { return std::isfinite(bf16ToFloat(v)); })
                : !std::ranges::equal(row(hit), row(clean)))
          fail(label + ": row " + std::to_string(r) + (bad ? " has finite outputs" : " changed"));
      }
    }
  }
  section("int8 decode (PQ2_0): " + std::to_string(ran) +
          " plans (single, gate/up in one and two passes, residual, fp32, fused; 1-4 lanes, every split, both walks), "
          "every value within the fp32 rounding of the exact int8 products, lanes and pass modes bitwise, fp32 "
          "rounding to bf16's; " + std::to_string(production) + " production plans in int8 activations within it; a "
          "non-finite input leaves its row alone not finite");
}

// ---------------------------------------------------------------- int8 prefill
// LinearTile::GgufPrefillInt8 over af4g64 and af5g64, where the device's family runs it. Its outputs are the products
// of the inputs rounded to int8 per row and 64-input group (scaled by the group's largest magnitude), within the
// error of the bf16 tile's on the exact inputs, on dense inputs and on dense inputs with one input of 1000 per row (a
// model's outlier channel), which a scale per row would let zero the rest of its row. On dense inputs the RMS error
// it adds over the bf16 tile's, relative to the outputs, stays within kBound. Outputs are finite and rows past the
// chunk unwritten, at each epilogue, chunks of 168, 136 and 65 rows ending inside a row tile, one of 600 rows whose
// walk ends on a partial block of row tiles, also over 5 column tiles, whose GEMM grid, 16 threadgroups wide, ends in
// one past its tiles, the full budget and fused segments, of one format and of both. A
// chunk's last 1 and 4 rows at their own height, as a prefill's last layer runs the rows something reads
// (Linear::prefillPlan of a chunk's rows), write the chunk's bits. A fused projection quantizes its rows once and
// runs a GEMM dispatch per segment, and a SwiGLU's up takes its gate's codes (LinearInput::PrefillInt8), bit for bit.
// A view of leading inputs gives its inputs repacked bit for bit. A non-finite input leaves its row's outputs not
// finite, as the bf16 tile's, and no other row's changed. Unlike the int8 decode test's exact model, these bounds are
// RMS ones against the bf16 tile's own error: this tile's per-group affine folds lie outside projectionBound's model
// of exact products accumulated in fp32.
void prefillInt8(MetalBackend &backend, const Linear &linear) {
  if (!linear.calibration().projection.prefillInt8) {
    section("int8 prefill: skipped, the family does not run it");
    return;
  }
  constexpr LinearConfig kTile{.tile = LinearTile::GgufPrefill}, kInt8{.tile = LinearTile::GgufPrefillInt8};
  constexpr double kBound = 1e-2;
  // The tile computes the products of the rounded inputs: its error from them stays within this multiple of the bf16
  // tile's error on the exact inputs (1.01x measured, 20-core M5 Pro).
  constexpr double kRoundedBound = 1.25;
  constexpr float kOutlier = 1000;
  double worst = 0, worstRounded = 0;
  const auto chunk = [&](const Projection &p, const std::vector<const Tensor *> &parts, LinearEpilogue epilogue,
                         uint32_t rows, bool outliers, const std::string &what) {
    const uint32_t K = p.inputSize, columns = p.outputSize, covered = segmentColumns(parts);
    const LinearWorkload w{{columns, K}, rows, LinearPhase::Prefill, epilogue};
    const LinearPlan plan = Linear::plan(w, kInt8, FloatOutput::BFloat16);
    const uint32_t storage = plan.storageRows();
    std::vector<float> x = activations(Inputs::Dense, storage, K);
    if (outliers)
      for (uint32_t r = 0; r < storage; ++r) x[uint64_t{r} * K + (r * 97 + 13) % K] = r % 2 ? -kOutlier : kOutlier;
    const std::vector<float> auxiliary = epilogue == LinearEpilogue::Residual
        ? residuals(storage, columns) : activations(Inputs::Dense, storage, columns);
    const std::vector<uint16_t> aux = storageRows(auxiliary, columns, storage, storage);
    const std::vector<Dot> dots = products(parts, columns, x, rows);
    // The inputs as the codes of the int8-activation rule (test::int8Block) stand for them.
    std::vector<float> rounded = x;
    for (uint64_t at = 0; at < uint64_t{rows} * K; at += GGUF_PREFILL_INT8_GROUP) {
      float most = 0;
      for (uint32_t k = 0; k < GGUF_PREFILL_INT8_GROUP; ++k) most = std::max(most, std::fabs(x[at + k]));
      const test::Int8Block block = test::int8Block(most);
      for (uint32_t k = 0; k < GGUF_PREFILL_INT8_GROUP; ++k)
        rounded[at + k] = float(test::int8Code(x[at + k], block)) * block.scale;
    }
    const std::vector<Dot> roundedDots = products(parts, columns, rounded, rows);
    const std::string name = what + " " + epilogueName(epilogue) + " " + std::to_string(rows) + " rows K " +
                             std::to_string(K) + (outliers ? " outliers" : "");
    const Outcome out = run(backend, linear, plan, p, nullptr, storageRows(x, K, rows, storage), aux, kPoisonNaN,
                            covered, name);
    const Outcome base = run(backend, linear, Linear::plan(w, kTile, FloatOutput::BFloat16), p, nullptr,
                             storageRows(x, K, rows, storage), aux, kPoisonNaN, covered, name + " bf16 tile");
    // One quantization of the chunk's rows, then a GEMM dispatch per segment.
    const auto dispatches = [&](bool gemm) {
      return std::count_if(out.pipelines.begin(), out.pipelines.end(), [&](const std::string &pipeline) {
        return pipeline.starts_with("gguf_prefill_int8_") && (pipeline != "gguf_prefill_int8_quantize") == gemm;
      });
    };
    if (dispatches(false) != 1 || size_t(dispatches(true)) != parts.size())
      fail(name + ": ran " + std::to_string(dispatches(false)) + " quantizations and " +
           std::to_string(dispatches(true)) + " GEMM dispatches for " + std::to_string(parts.size()) + " segments");
    // The output of products `d` through the epilogue.
    const auto output = [&](double d, double a) {
      if (epilogue == LinearEpilogue::Residual) return d + a;
      return epilogue == LinearEpilogue::UpWithGate ? d * silu(a) : d;
    };
    double err2 = 0, base2 = 0, ref2 = 0, rounded2 = 0;
    uint64_t nonfinite = 0;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < covered; ++c) {
        const uint64_t i = uint64_t{r} * columns + c;
        const double got = bf16ToFloat(out.output[i]), tile = bf16ToFloat(base.output[i]), a = bf16ToFloat(aux[i]),
                     want = output(dots[i].value, a),
                     scale = epilogue == LinearEpilogue::UpWithGate ? want : dots[i].value;
        if (!std::isfinite(got)) {
          ++nonfinite;
          continue;
        }
        err2 += (got - want) * (got - want);
        rounded2 += (got - output(roundedDots[i].value, a)) * (got - output(roundedDots[i].value, a));
        base2 += (tile - want) * (tile - want);
        ref2 += scale * scale;
      }
    const double added = std::sqrt(std::max(err2 - base2, 0.0) / std::max(ref2, 1e-30)),
                 fromRounded = std::sqrt(rounded2 / std::max(base2, 1e-30));
    if (!outliers) worst = std::max(worst, added);
    worstRounded = std::max(worstRounded, fromRounded);
    if (nonfinite) fail(name + ": " + std::to_string(nonfinite) + " non-finite outputs");
    if (!outliers && added > kBound) fail(name + ": adds relative RMS error " + std::to_string(added));
    if (fromRounded > kRoundedBound)
      fail(name + ": differs from the products of its inputs rounded per row and group by " +
           std::to_string(fromRounded) + "x the bf16 tile's error");
    if (!std::all_of(out.output.begin() + uint64_t{rows} * columns, out.output.end(),
                     [](uint16_t v) { return v == kUnwritten; }))
      fail(name + ": rows past the chunk written");
    for (const uint32_t last : {1u, 4u}) {
      const LinearPlan part =
          Linear::plan({w.matrix, last, LinearPhase::Prefill, epilogue}, kInt8, FloatOutput::BFloat16);
      const auto tail = [&](const std::vector<float> &values, uint32_t width) {
        const auto end = values.begin() + uint64_t{rows} * width;
        return storageRows(std::vector<float>(end - uint64_t{last} * width, end), width, last, part.storageRows());
      };
      const std::string own = name + ", its last " + std::to_string(last);
      const Outcome alone =
          run(backend, linear, part, p, nullptr, tail(x, K), tail(auxiliary, columns), kPoisonNaN, covered, own);
      if (!sameRows(alone.output, 0, out.output, rows - last, last, columns)) fail(own + ": differ from the chunk's");
    }
  };
  for (const Fmt f : {Fmt(GGUF_FMT_AF4G64), Fmt(GGUF_FMT_AF5G64)}) {
    for (const uint32_t K : {1024u, 5120u}) {
      const Tensor t = tensor(backend, f, 1024, K);
      const Projection p = projection({&t}, t.N);
      for (const LinearEpilogue e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
        for (const uint32_t rows : {168u, 136u, 65u})
          for (const bool outliers : {false, true}) chunk(p, {&t}, e, rows, outliers, fmtName(f));
      // 600 rows: 19 row tiles, whose walk takes a block of 16 down each column tile, then one of 3.
      static_assert(GGUF_PREFILL_INT8_WALK == 16 && GGUF_PREFILL_INT8_ROWS == 32,
                    "600 rows end on a partial walk block");
      chunk(p, {&t}, LinearEpilogue::Residual, 600, false, fmtName(f));
      // 5 column tiles by 19 row tiles: 95 tiles on a grid of 6 runs of 16.
      const Tensor narrow = tensor(backend, f, 5 * GGUF_TILE_COLUMNS, K);
      chunk(projection({&narrow}, narrow.N), {&narrow}, LinearEpilogue::Residual, 600, false,
            std::string(fmtName(f)) + " 5 column tiles");
      chunk(p, {&t}, LinearEpilogue::None, SPLASH_PREFILL_TOKEN_BUDGET, false, fmtName(f));
      const Tensor b = tensor(backend, f, 256, K), c = tensor(backend, f, 512, K);
      const std::vector<const Tensor *> parts{&t, &b, &c};
      chunk(projection(parts, segmentColumns(parts) + kPadding), parts, LinearEpilogue::None, 168, false,
            std::string(fmtName(f)) + " fused");
    }
    // A view of 512 of 1024 inputs against those inputs repacked, over two plane tiles of rows.
    constexpr uint32_t N = 128, K = 1024, kLeading = 512, kChunk = 168;
    const Tensor whole = tensor(backend, f, N, K);
    std::vector<uint8_t> leading;
    for (uint32_t row = 0; row < N; ++row) {
      const auto begin = whole.native.begin() + uint64_t{row} * rowBytes(f, K);
      leading.insert(leading.end(), begin, begin + rowBytes(f, kLeading));
    }
    const Packed planes = repack(f, leading, N, kLeading, nullptr);
    const Projection compact(
        N, kLeading,
        BlockWeights{{QuantizedSegment::planes(f, N, kLeading, upload(backend, planes.w0),
                                               kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1)
                                                                             : MetalBuffer{},
                                               upload(backend, planes.meta))}});
    const Projection view = Projection(N, K, BlockWeights{{whole.segment}}).leadingInputs(kLeading);
    const LinearPlan plan = Linear::plan({{N, kLeading}, kChunk, LinearPhase::Prefill, LinearEpilogue::Residual},
                                         kInt8, FloatOutput::BFloat16);
    const uint32_t storage = plan.storageRows();
    const std::vector<uint16_t> x = storageRows(activations(Inputs::Dense, storage, kLeading), kLeading, kChunk,
                                                storage);
    const std::vector<uint16_t> aux = storageRows(residuals(storage, N), N, storage, storage);
    const std::string label = std::string(fmtName(f)) + " int8 leading inputs";
    if (run(backend, linear, plan, view, nullptr, x, aux, kPoisonNaN, N, label).output !=
        run(backend, linear, plan, compact, nullptr, x, aux, kPoisonNaN, N, label).output)
      fail(label + ": differs from the repacked leading inputs");
  }
  // af4g64 and af5g64 segments fused, as a mixed 4/5-bit MLX quantization does.
  for (const uint32_t K : {1024u, 5120u}) {
    const Tensor a = tensor(backend, Fmt(GGUF_FMT_AF4G64), 512, K), b = tensor(backend, Fmt(GGUF_FMT_AF5G64), 256, K),
                 c = tensor(backend, Fmt(GGUF_FMT_AF4G64), 256, K);
    const std::vector<const Tensor *> parts{&a, &b, &c};
    for (const bool outliers : {false, true})
      chunk(projection(parts, segmentColumns(parts) + kPadding), parts, LinearEpilogue::None, 168, outliers,
            "af4g64+af5g64 fused");
  }
  // A SwiGLU's up after its gate over the same rows takes the codes and scales the gate's plan left in the scratch
  // (LinearInput::PrefillInt8): it quantizes nothing itself and writes the bits of its plan run alone.
  {
    constexpr uint32_t N = 256, K = 1024, kRows = 168;
    const Tensor gateTensor = tensor(backend, Fmt(GGUF_FMT_AF4G64), N, K),
                 upTensor = tensor(backend, Fmt(GGUF_FMT_AF5G64), N, K);
    const Projection gate = projection({&gateTensor}, N), up = projection({&upTensor}, N);
    const LinearPlan gatePlan =
        Linear::plan({{N, K}, kRows, LinearPhase::Prefill, LinearEpilogue::None}, kInt8, FloatOutput::BFloat16);
    const LinearPlan upPlan =
        Linear::plan({{N, K}, kRows, LinearPhase::Prefill, LinearEpilogue::UpWithGate}, kInt8, FloatOutput::BFloat16);
    const uint32_t storage = gatePlan.storageRows();
    const std::vector<uint16_t> x = storageRows(activations(Inputs::Dense, storage, K), K, kRows, storage);
    LinearScratchSize size = gatePlan.scratchSize();
    size.include(upPlan.scratchSize());
    const Scratch scratch(backend, size);
    const Guarded input = bfloats(backend, x), gateOutput(backend, uint64_t{storage} * N * 2, 0xFF),
                  upOutput(backend, uint64_t{storage} * N * 2, 0xFF);
    CommandGraph graph;
    const PreparedInput codes =
        linear.add(graph, {.input = input.view, .output = gateOutput.view, .scratch = scratch.bindings()}, gate,
                   gatePlan);
    const size_t gateDispatches = graph.dispatches().size();
    static_cast<void>(linear.add(graph,
                                 {.input = input.view, .output = upOutput.view, .gateScratch = gateOutput.view,
                                  .scratch = scratch.bindings(), .prepared = codes},
                                 up, upPlan));
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    if (codes.layout != LinearInput::PrefillInt8 ||
        std::any_of(graph.dispatches().begin() + gateDispatches, graph.dispatches().end(),
                    [](const ComputeDispatch &d) { return d.pipelineName == "gguf_prefill_int8_quantize"; }))
      fail("int8 SwiGLU: the up projection quantized its gate's rows again");
    const Outcome alone = run(backend, linear, upPlan, up, nullptr, x, gateOutput.halves(), kPoisonNaN, N,
                              "int8 SwiGLU up alone");
    if (!sameRows(upOutput.halves(), 0, alone.output, 0, kRows, N))
      fail("int8 SwiGLU: the up projection on its gate's codes differs from its own");
  }
  // A NaN in one row's inputs and an infinity in another's leave those rows' outputs not finite, on this tile as on
  // the bf16 tile, and every other row's as they were.
  for (const Fmt f : {Fmt(GGUF_FMT_AF4G64), Fmt(GGUF_FMT_AF5G64)}) {
    constexpr uint32_t K = 1024, kRows = 136, kNaNRow = 3, kInfiniteRow = 70;
    const Tensor t = tensor(backend, f, 256, K);
    const Projection p = projection({&t}, t.N);
    for (const LinearConfig config : {kInt8, kTile}) {
      const LinearWorkload w{{t.N, K}, kRows, LinearPhase::Prefill, LinearEpilogue::Residual};
      const LinearPlan plan = Linear::plan(w, config, FloatOutput::BFloat16);
      const uint32_t storage = plan.storageRows();
      std::vector<float> x = activations(Inputs::Dense, storage, K);
      const std::vector<uint16_t> aux = storageRows(residuals(storage, t.N), t.N, storage, storage);
      const std::string label =
          std::string(fmtName(f)) + (config == kInt8 ? " int8" : " bf16 tile") + " non-finite inputs";
      const Outcome clean =
          run(backend, linear, plan, p, nullptr, storageRows(x, K, kRows, storage), aux, kPoisonNaN, t.N, label);
      x[uint64_t{kNaNRow} * K + 5] = std::numeric_limits<float>::quiet_NaN();
      x[uint64_t{kInfiniteRow} * K + 700] = std::numeric_limits<float>::infinity();
      const Outcome poisoned =
          run(backend, linear, plan, p, nullptr, storageRows(x, K, kRows, storage), aux, kPoisonNaN, t.N, label);
      for (uint32_t r = 0; r < kRows; ++r) {
        const auto row = [&](const Outcome &o) { return std::span(o.output).subspan(uint64_t{r} * t.N, t.N); };
        const bool hit = r == kNaNRow || r == kInfiniteRow;
        if (hit ? !std::ranges::none_of(row(poisoned), [](uint16_t v) { return std::isfinite(bf16ToFloat(v)); })
                : !std::ranges::equal(row(poisoned), row(clean)))
          fail(label + ": row " + std::to_string(r) + (hit ? " has finite outputs" : " changed"));
      }
    }
  }
  section("int8 prefill: af4g64, af5g64 and both fused, K 1024 and 5120, plain/residual/up-with-gate over 168-, "
          "136- and 65-row chunks with and without outliers, 600 rows and the " +
          std::to_string(SPLASH_PREFILL_TOKEN_BUDGET) + "-row budget: worst added relative RMS " +
          std::to_string(worst) + ", worst error from the inputs rounded per group " + std::to_string(worstRounded) +
          "x the bf16 tile's; a fused projection quantized once and a GEMM a segment; last rows, leading inputs and "
          "an up on its gate's codes bitwise; a non-finite input leaves its row alone not finite");
}

// ---------------------------------------------------------------- leading inputs
// A view of the leading inputs of a projection's rows (Projection::leadingInputs), as the ANE FFN split runs down's GPU
// share: its residual prefill (gguf_prefill_<format>_r_leading_inputs) equals bit for bit the residual prefill of
// those inputs repacked on their own, in every format, over two plane tiles of rows.
void leadingInputs(MetalBackend &backend, const Linear &linear) {
  constexpr uint32_t N = 512, K = 1024, kLeading = 512, kChunk = 168;
  for (int fi = 0; fi < FMT_COUNT; ++fi) {
    const Fmt f = Fmt(fi);
    const Tensor whole = tensor(backend, f, N, K);
    std::vector<uint8_t> leading;
    for (uint32_t row = 0; row < N; ++row) {
      const auto begin = whole.native.begin() + uint64_t{row} * rowBytes(f, K);
      leading.insert(leading.end(), begin, begin + rowBytes(f, kLeading));
    }
    const Packed planes = repack(f, leading, N, kLeading, nullptr);
    const Projection compact(
        N, kLeading,
        BlockWeights{{QuantizedSegment::planes(f, N, kLeading, upload(backend, planes.w0),
                                               kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1)
                                                                             : MetalBuffer{},
                                               upload(backend, planes.meta))}});
    const Projection view = Projection(N, K, BlockWeights{{whole.segment}}).leadingInputs(kLeading);
    const LinearWorkload w{{N, kLeading}, kChunk, LinearPhase::Prefill, LinearEpilogue::Residual};
    const LinearPlan plan = Linear::plan(w, {.tile = LinearTile::GgufPrefill}, FloatOutput::BFloat16);
    const uint32_t storage = plan.storageRows();
    const std::vector<uint16_t> x = storageRows(activations(Inputs::Dense, storage, kLeading), kLeading, kChunk,
                                                storage);
    const std::vector<uint16_t> aux = storageRows(residuals(storage, N), N, storage, storage);
    const std::string label = std::string(fmtName(fi)) + " leading inputs";
    const Outcome expected = run(backend, linear, plan, compact, nullptr, x, aux, kPoisonNaN, N, label);
    const Outcome got = run(backend, linear, plan, view, nullptr, x, aux, kPoisonNaN, N, label);
    if (got.output != expected.output) fail(label + ": differs from the repacked leading inputs");
  }
  section("leading inputs: " + std::to_string(FMT_COUNT) + " formats' residual prefill over a view of 512 of 1024 "
          "inputs equal to them repacked");
}

// ---------------------------------------------------------------- split visibility
// Two projections of a decode step run back to back and share one split scratch, as every GGUF projection of a step
// does, at each pair of K splits the tile's policy (Apple9: register, Apple10: staged) picks for 8-80 cores where it
// plans the tile for both (Apple10 registers the 5120-column grids at one lane from 54 cores). The partials are
// overwritten before each projection, with a large finite value and in another run with NaN, so a
// partial read before its writer published it changes the output. Every output must lie within fp64, as the
// unsplit outputs must, and not depend on the poison or the run; every counter must return to zero.
struct SplitOperand {
  LinearMatrix matrix;
  LinearEpilogue epilogue;
  std::vector<Tensor> weights;  // the projection's, then a gate/up projection's gate
  std::vector<float> x, residual;
  std::vector<Dot> dots, gateDots;
};
SplitOperand splitOperand(MetalBackend &backend, Fmt f, LinearMatrix matrix, LinearEpilogue epilogue) {
  SplitOperand o{matrix, epilogue, {}, activations(Inputs::Dense, kMaximumRows, matrix.inputSize),
                 residuals(kMaximumRows, matrix.outputSize), {}, {}};
  for (uint32_t i = 0; i < (epilogue == LinearEpilogue::GateUp ? 2u : 1u); ++i)
    o.weights.push_back(tensor(backend, f, matrix.outputSize, matrix.inputSize));
  o.dots = products({&o.weights[0]}, matrix.outputSize, o.x, kMaximumRows);
  if (o.weights.size() > 1) o.gateDots = products({&o.weights[1]}, matrix.outputSize, o.x, kMaximumRows);
  return o;
}

void splitVisibility(MetalBackend &backend, const Linear &linear, LinearTile tile,
                     const std::array<const SplitOperand *, 2> &pair, uint32_t lanes) {
  const uint32_t family = tile == LinearTile::GgufRegister ? 9 : 10;
  std::array<std::vector<Projection>, 2> weights;  // each operand's projection and gate
  std::array<LinearWorkload, 2> workloads;
  for (uint32_t i = 0; i < 2; ++i) {
    for (const Tensor &t : pair[i]->weights) weights[i].push_back(projection({&t}, pair[i]->matrix.outputSize));
    workloads[i] = decode(pair[i]->matrix, lanes, pair[i]->epilogue);
  }
  const auto gateOf = [&](uint32_t i) { return weights[i].size() > 1 ? &weights[i][1] : nullptr; };
  std::set<std::array<uint32_t, 2>> splitPairs;
  for (uint32_t cores = 8; cores <= 80; ++cores) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = cores;
    const Linear policy(device, Int8Activations::On);
    const LinearConfig a = policy.plan(workloads[0], weights[0][0]).configuration();
    const LinearConfig b = policy.plan(workloads[1], weights[1][0]).configuration();
    if (a.tile == tile && b.tile == tile && std::max(a.splits, b.splits) > 1) splitPairs.insert({a.splits, b.splits});
  }
  const auto matrix = [](LinearMatrix m) { return std::to_string(m.outputSize) + "x" + std::to_string(m.inputSize); };
  const std::string shape = std::string(tileName(tile)) + " " + std::to_string(lanes) + " lanes, " +
                            matrix(pair[0]->matrix) + " then " + matrix(pair[1]->matrix);
  if (splitPairs.empty()) fail(shape + ": the policy splits neither projection");
  std::string pairs;
  for (const auto &splits : splitPairs) pairs += " " + std::to_string(splits[0]) + "/" + std::to_string(splits[1]);
  std::cout << "  " << shape << ": splits" << pairs << '\n';
  const auto plan = [&](uint32_t i, uint32_t splits) {
    return Linear::plan(workloads[i], config(tile, workloads[i], splits), FloatOutput::BFloat16);
  };
  LinearScratchSize size = plan(0, 1).scratchSize();
  size.include(plan(1, 1).scratchSize());
  for (const auto &splits : splitPairs)
    for (uint32_t i = 0; i < 2; ++i) size.include(plan(i, splits[i]).scratchSize());
  const Scratch scratch(backend, size);
  MetalBuffer poison = test::sharedBuffer(backend, size.partials);
  std::vector<Operands> operands;
  std::vector<std::vector<uint16_t>> auxiliary;
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t rows = workloads[i].rows, n = pair[i]->matrix.outputSize;
    auxiliary.push_back(storageRows(pair[i]->residual, n, rows, rows));
    operands.emplace_back(backend, plan(i, 1), storageRows(pair[i]->x, pair[i]->matrix.inputSize, rows, rows),
                          auxiliary.back());
  }
  const auto check = [&](const std::array<uint32_t, 2> &splits, const std::string &what) {
    for (uint32_t i = 0; i < 2; ++i) {
      const Outcome out{operands[i].output.halves(), operands[i].gate.halves(), {}};
      const bool staged = tile == LinearTile::GgufStaged;
      checkValues(out, within(pair[i]->dots, staged), within(pair[i]->gateDots, staged), auxiliary[i], workloads[i],
                  workloads[i].rows, pair[i]->matrix.outputSize,
                  what + " projection " + std::to_string(i) + " S=" + std::to_string(splits[i]));
    }
    if (!scratch.intact()) fail(what + ": a counter is not reset or a write past the scratch");
  };
  CommandGraph unsplit;
  for (uint32_t i = 0; i < 2; ++i)
    static_cast<void>(linear.add(unsplit, operands[i].bindings(plan(i, 1), scratch), weights[i][0], plan(i, 1),
                                 gateOf(i)));
  static_cast<void>(backend.submitCommandAsync(unsplit.dispatches()).wait());
  check({1, 1}, shape + " unsplit");
  for (const auto &splits : splitPairs) {
    const std::string what = shape + " splits " + std::to_string(splits[0]) + "/" + std::to_string(splits[1]);
    CommandGraph graph;
    for (uint32_t i = 0; i < 2; ++i) {
      // A test kernel's copy poisons the partials in dispatch order.
      graph.add("test_copy_u32", {poison, scratch.partials.view}, uint32_t(size.partials / 4),
                {(size.partials / 4 + 255) / 256, 1, 1}, {256, 1, 1});
      static_cast<void>(linear.add(graph, operands[i].bindings(plan(i, splits[i]), scratch), weights[i][0],
                                   plan(i, splits[i]), gateOf(i)));
    }
    std::array<std::vector<uint16_t>, 2> first;
    std::array<bool, 2> varies{};
    for (const uint32_t bits : {kPoisonFinite, kPoisonNaN, kPoisonFinite}) {
      std::fill_n(static_cast<uint32_t *>(poison.contents()), size.partials / 4, bits);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      for (uint32_t i = 0; i < 2; ++i) {
        const std::vector<uint16_t> output = operands[i].output.halves();
        if (first[i].empty()) first[i] = output;
        else varies[i] = varies[i] || output != first[i];
      }
    }
    for (uint32_t i = 0; i < 2; ++i)
      if (varies[i]) fail(what + ": projection " + std::to_string(i) + " depends on the poison or run");
    check(splits, what);
  }
}

} // namespace

// ---------------------------------------------------------------- token gather
// ops::Embedding over the native rows of every format (kernels/shared/embedding.metal): each gathered value
// is the bf16 rounding of GGML's fp32 value, for tokens in any order, repeated, and the vocabulary's first and last;
// and over BF16 and F16 rows of random bits with the top exponent bit clear (finite and below 2, subnormals
// included), each value the bf16 rounding of the stored one, also in rows of Flash-Next's 160 PLE values, which
// are no whole native block.
void tokenGather(MetalBackend &backend) {
  constexpr uint32_t kVocabulary = 64, kHidden = 1024;
  const std::vector<uint32_t> tokens{kVocabulary - 1, 0, 17, 17, 42, 3, kVocabulary - 1, 29, 8};
  for (int fi = 0; fi < FMT_COUNT; ++fi) {
    const Fmt f = Fmt(fi);
    const std::vector<uint8_t> native = makeNative(f, kVocabulary, kHidden, rng);
    const EmbeddingWeights table(kVocabulary, kHidden, StoredRows(upload(backend, native), f));
    const Guarded ids(backend, tokens.size() * sizeof(uint32_t), 0), output(backend, tokens.size() * kHidden * 2, 0xFF);
    std::memcpy(ids.view.contents(), tokens.data(), ids.bytes);
    CommandGraph graph;
    Embedding::add(graph, ids.view, table, output.view, uint32_t(tokens.size()));
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    if (!output.intact()) fail(std::string(fmtName(f)) + " gather writes past its output");
    const std::vector<uint16_t> got = output.halves();
    std::vector<float> values(kHidden);
    size_t differ = 0;
    for (size_t r = 0; r < tokens.size(); ++r) {
      rowValues(f, native.data() + size_t(tokens[r]) * rowBytes(f, kHidden), kHidden, values.data());
      for (uint32_t k = 0; k < kHidden; ++k) differ += got[r * kHidden + k] != floatToBf16(values[k]);
    }
    if (differ) fail(std::string(fmtName(f)) + " gather: " + std::to_string(differ) + " values differ from bf16(GGML)");
  }
  for (const auto &[format, hidden] : {std::pair{BF16, kHidden}, std::pair{BF16, 160u}, std::pair{F16, kHidden},
                                      std::pair{F16, 160u}}) {
    std::vector<uint8_t> rows(size_t{kVocabulary} * hidden * 2);
    for (auto &b : rows) b = uint8_t(rng());
    for (size_t i = 1; i < rows.size(); i += 2) rows[i] &= 0xBF;   // finite: the top exponent bit clear
    const EmbeddingWeights table(kVocabulary, hidden, StoredRows(upload(backend, rows), format));
    const std::string name = std::string(fmtName(format)) + " at " + std::to_string(hidden) + " values";
    const Guarded ids(backend, tokens.size() * sizeof(uint32_t), 0), output(backend, tokens.size() * hidden * 2, 0xFF);
    std::memcpy(ids.view.contents(), tokens.data(), ids.bytes);
    CommandGraph graph;
    Embedding::add(graph, ids.view, table, output.view, uint32_t(tokens.size()));
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    if (!output.intact()) fail(name + " gather writes past its output");
    const std::vector<uint16_t> got = output.halves();
    size_t differ = 0;
    for (size_t r = 0; r < tokens.size(); ++r)
      for (uint32_t k = 0; k < hidden; ++k) {
        uint16_t stored;
        std::memcpy(&stored, rows.data() + (size_t(tokens[r]) * hidden + k) * 2, 2);
        differ += got[r * hidden + k] != (format == BF16 ? stored : floatToBf16(h2f(stored)));
      }
    if (differ) fail(name + " gather: " + std::to_string(differ) + " values differ from bf16 of the table's");
  }
  section("token gather: " + std::to_string(FMT_COUNT) +
          " formats, each value bf16 of GGML's fp32 value, and bf16 and f16 rows of random bits at 1024 and 160 "
          "values, each value bf16 of the stored one");
}

// A streamed table's gather (Qwen4ExpTarget::addPleGather): a SparseRowStore
// stages each distinct row a step reads once, with each row id's slot, and the
// same gather over the staging as a table gives, byte for byte, what it gives
// over the whole table, for every format, ids repeated and in any
// order, the table's first and last rows, and rows read or cached; and so at
// Qwen3.8-Flash-Next's PLE rows of 160 values in the formats its GGUFs hold
// them in, IQ4_NL (90-byte rows), Q8_0 (170-byte rows), BF16 and F16 (320-byte
// rows).
void stagedGather(MetalBackend &backend) {
  constexpr uint32_t kRows = 4096, kCount = 640;
  // Rows read from memory, as from the table's file.
  class MemoryRows final : public model::RowReader {
  public:
    explicit MemoryRows(const std::vector<uint8_t> &bytes) : bytes_(bytes) {}
    void read(uint64_t offset, std::span<uint8_t> out) override {
      std::memcpy(out.data(), bytes_.data() + offset, out.size());
    }
    void checkUnchanged() override {}
    [[nodiscard]] uint64_t size() const override { return bytes_.size(); }

  private:
    const std::vector<uint8_t> &bytes_;
  };
  const auto check = [&](const Fmt f, const uint32_t hidden) {
    const std::string name = std::string(fmtName(f)) + " at " + std::to_string(hidden) + " values";
    const uint32_t bytesPerRow = uint32_t(rowBytes(f, hidden));
    const std::vector<uint8_t> native = makeNative(f, kRows, hidden, rng);
    model::SparseRowStore store({"table", kRows, bytesPerRow, 0}, std::make_unique<MemoryRows>(native),
                                {model::SparseRowStore::cacheBytesFor(kRows / 2, bytesPerRow) +
                                     model::SparseRowStore::workingBytes(bytesPerRow, 1024, 0),
                                 std::make_shared<model::SsdReaders>(2), 1024});
    const EmbeddingWeights table(kRows, hidden, StoredRows(upload(backend, native), f));
    for (int step = 0; step < 2; ++step) {
      std::vector<uint32_t> ids(kCount);
      for (uint32_t &id : ids) id = uint32_t(rng() % (step ? kRows : 64)); // the second step repeats rows
      ids[0] = 0;
      ids[1] = kRows - 1;
      ids[2] = ids[1];
      const Guarded indices(backend, kCount * 4, 0), slots(backend, kCount * 4, 0),
          staging(backend, uint64_t{kCount} * bytesPerRow, 0x5A),
          resident(backend, uint64_t{kCount} * hidden * 2, 0xFF), staged(backend, uint64_t{kCount} * hidden * 2, 0xFF);
      std::memcpy(indices.view.contents(), ids.data(), indices.bytes);
      store.stage(ids, {static_cast<uint8_t *>(staging.view.contents()), staging.bytes},
                  {static_cast<uint32_t *>(slots.view.contents()), kCount}, model::SparseRowUse::Prefill);
      CommandGraph graph;
      Embedding::add(graph, indices.view, table, resident.view, kCount);
      Embedding::add(graph, slots.view, EmbeddingWeights(kCount, hidden, StoredRows(staging.view, f)), staged.view,
                     kCount);
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      if (!staged.intact() || !resident.intact()) fail(name + " staged gather writes past its output");
      if (resident.halves() != staged.halves()) fail(name + " staged gather differs from the table's");
    }
  };
  for (int fi = 0; fi < FMT_COUNT; ++fi) check(Fmt(fi), 512);
  for (const Fmt f : {IQ4NL, Q80, BF16, F16}) {
    if (rowBytes(f, 160) != (f == IQ4NL ? 90u : f == Q80 ? 170u : 320u))
      fail(std::string(fmtName(f)) + " PLE rows have another width");
    check(f, 160);
  }
  section("staged gather: " + std::to_string(FMT_COUNT) +
          " formats, and IQ4_NL, Q8_0, BF16 and F16 PLE rows of 160 values, byte-equal to the table's gather of "
          "the same rows");
}

// Each buffer the token gathers reach, at its extent and one element short:
// the rows' token ids and bf16 output rows, every token's native blocks (of
// Q8_0, PQ2_0, MLX's af4g64 and nvfp4, BF16 and F16), BF16 and F16 rows of
// Flash-Next's 160 PLE values, and the rotation signs of a rotated PQ2_0
// table.
void gatherExtents(MetalBackend &backend) {
  constexpr uint32_t kVocabulary = 64, kHidden = 1024, kRows = 9;
  const MetalBuffer tokens = test::sharedBuffer(backend, kRows * 4),
                    output = test::sharedBuffer(backend, uint64_t{kRows} * kHidden * 2);
  // A buffer of the table `with` returns with it.
  const auto requireTableExtent = [&](const MetalBuffer &buffer, uint64_t bytes, uint64_t element, const char *what,
                                      const auto &with) {
    test::requireExtent(backend, buffer, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &view) {
      Embedding::add(graph, tokens, with(view), output, kRows);
    });
  };
  std::vector<EmbeddingWeights> tables;
  for (const Fmt f : {Q80, PQ20, Fmt(GGUF_FMT_AF4G64), NVFP4, BF16, F16}) {
    const QuantFormat &format = kQuantFormats[f];
    const uint64_t rowBytes = uint64_t{kHidden} / format.block_elements * format.block_bytes;
    EmbeddingWeights table(kVocabulary, kHidden, StoredRows(test::sharedBuffer(backend, kVocabulary * rowBytes), f));
    if (f == PQ20) table.rotation.signs = test::sharedBuffer(backend, kHidden);
    requireTableExtent(table.rows().rows, kVocabulary * rowBytes, format.block_bytes, "token table",
                       [&](const MetalBuffer &rows) {
                         EmbeddingWeights changed(kVocabulary, kHidden, StoredRows(rows, f));
                         changed.rotation = table.rotation;
                         return changed;
                       });
    if (table.rotation)
      requireTableExtent(table.rotation.signs, kHidden, 1, "embedding rotation sign", [&](const MetalBuffer &signs) {
        EmbeddingWeights changed = table;
        changed.rotation.signs = signs;
        return changed;
      });
    tables.push_back(table);
  }
  for (const Fmt f : {BF16, F16})
    requireTableExtent(test::sharedBuffer(backend, kVocabulary * 320), kVocabulary * 320, 2, "token table",
                       [&](const MetalBuffer &rows) { return EmbeddingWeights(kVocabulary, 160, StoredRows(rows, f)); });
  for (const EmbeddingWeights &table : tables) {
    test::requireExtent(backend, tokens, kRows * 4, 4, "embedding token", [&](CommandGraph &graph, const MetalBuffer &view) {
      Embedding::add(graph, view, table, output, kRows);
    });
    test::requireExtent(backend, output, uint64_t{kRows} * kHidden * 2, 2, "embedding output",
                        [&](CommandGraph &graph, const MetalBuffer &view) { Embedding::add(graph, tokens, table, view, kRows); });
  }
  section("token gather extents: every buffer of native, 160-value BF16 and F16, and rotated tables at its "
          "extent and refused one element short");
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf-projection <production-and-test.metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const Linear linear(backend.capabilities(), Int8Activations::On);
      for (const LinearTile tile : {LinearTile::GgufRegister, LinearTile::GgufStaged}) {
        decodeTile(backend, linear, tile, false, decodeInputs(false), tileFormats(tile));
        fusedDecode(backend, linear, tile, false);
        gateUpPairs(backend, linear, tile);
      }
      decodeTile(backend, linear, LinearTile::GgufStaged, true, decodeInputs(true), tileFormats(LinearTile::GgufStaged));
      fusedDecode(backend, linear, LinearTile::GgufStaged, true);
      for (const bool spread : {false, true}) oneGateUpPass(backend, linear, spread);
      prefill(backend, linear, 1024, tileFormats(LinearTile::GgufStaged));
      decodeInt8(backend, linear);
      prefillInt8(backend, linear);
      leadingInputs(backend, linear);
      // The 27B out_proj then down, and gdn_in then gate/up: K 6144, 17408 and 5120.
      const SplitOperand out = splitOperand(backend, Q4K, {5120, 6144}, LinearEpilogue::Residual);
      const SplitOperand down = splitOperand(backend, Q6K, {5120, 17408}, LinearEpilogue::Residual);
      const SplitOperand gdn = splitOperand(backend, IQ4XS, {12288, 5120}, LinearEpilogue::None);
      const SplitOperand gateUp = splitOperand(backend, Q4K, {17408, 5120}, LinearEpilogue::GateUp);
      // Flash-Next's hyper-connection down then up projections, Q8_0: K 10240 (sixteen partitions from 14 cores on the
      // register tile) and 320.
      const SplitOperand hyperDown = splitOperand(backend, Q80, {320, 10240}, LinearEpilogue::None);
      const SplitOperand hyperUp = splitOperand(backend, Q80, {10240, 320}, LinearEpilogue::None);
      for (const LinearTile tile : {LinearTile::GgufRegister, LinearTile::GgufStaged})
        for (const uint32_t lanes : {1u, kMaximumLanes}) {
          splitVisibility(backend, linear, tile, {&out, &down}, lanes);
          splitVisibility(backend, linear, tile, {&gdn, &gateUp}, lanes);
          if (tile == LinearTile::GgufRegister) splitVisibility(backend, linear, tile, {&hyperDown, &hyperUp}, lanes);
        }
      section("split visibility: both tiles, 1 and 4 lanes, every split pair the policy picks for 8-80 cores, and the "
              "register tile's at Flash-Next's hyper-connection pair, independent of poisoned partials and within fp64");
      // Narrow inputs, in formats of 32-input blocks: K 320 and 640, and on the decode tiles 192.
      const std::vector<Fmt> narrow{Q80, MXFP4, Q40};
      for (const uint32_t K : {320u, 640u}) {
        for (const LinearTile tile : {LinearTile::GgufRegister, LinearTile::GgufStaged})
          decodeTile(backend, linear, tile, false, K, narrow);
        decodeTile(backend, linear, LinearTile::GgufStaged, true, K, narrow);
        prefill(backend, linear, K, narrow);
      }
      for (const LinearTile tile : {LinearTile::GgufRegister, LinearTile::GgufStaged})
        decodeTile(backend, linear, tile, false, 192, {Q80});
      tokenGather(backend);
      stagedGather(backend);
      gatherExtents(backend);
    } catch (const std::exception &e) {
      std::cerr << "gguf-projection: FAIL: " << e.what() << '\n';
      return 1;
    }
    std::cout << "gguf-projection: " << (failures ? "FAIL (" + std::to_string(failures) + " failures)" : "PASS")
              << '\n';
    return failures ? 1 : 0;
  }
}
