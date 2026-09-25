// GGUF projections through ops::Linear (runtime/ops/LinearGguf.cpp) in every format, against fp64 references over
// GGML's dequantized weights (GgufFormatReference.hpp). Each check forces its tile with Linear::plan(workload,
// config), so every GPU runs both decode tiles:
// - decode: the Apple9 register tile and the staged tile at one to four lanes (three on the staged 32-row tile over
//   NaN padding rows), every K split, each epilogue, dense inputs and sparse ones past half's range; a lane's rows
//   equal a one-lane projection of them bitwise; the plain epilogue into fp32 (the logits) holds the values its
//   bf16 output rounds, bit for bit, each within fp64 before rounding;
// - fused: three segments of different formats in one projection equal the projections of each segment alone;
// - gate/up: every gate and up format pair;
// - prefill: 128-row tiles with each epilogue, whose simdgroups past the chunk write nothing, and chunks of up to 32
//   rows on the decode tiles equal to them bitwise; fused segments at their column offsets;
// - split visibility: two projections that share the split scratch, at every pair of K splits either tile's policy
//   picks for 8-80 cores, independent of poisoned partials.
// Every run leaves the padding columns past its segments, the guard bands past its buffers and its counters as they
// were. The token gather (ops::Embedding) of every embedding format's native rows is checked here too.
#include "GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Embedding.hpp"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace gguf_reference;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::tuning::bf16ToFloat;
using splash::ops::tuning::floatToBf16;

namespace {

constexpr uint32_t kLaneRows = 8, kMaximumLanes = 4, kMaximumRows = kLaneRows * kMaximumLanes;
// Every K split a plan takes: the powers of two up to the largest.
constexpr uint32_t kSplits[] = {1, 2, 4, 8};
static_assert(kSplits[std::size(kSplits) - 1] == LinearConfig::kMaximumSplits);
// A matrix's columns are a multiple of 256, so the fewest padding columns past its segments.
constexpr uint32_t kPadding = 256;
// Sparse inputs: one entry per 256 inputs (the register tile's coefficient unit) of this magnitude, above half's
// largest finite value (65504), so a tile that converted its inputs to half would overflow.
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
  MetalBuffer buffer = backend.allocateBuffer(bytes.size());
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
    backing = backend.allocateBuffer(size + kGuardBytes);
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

// A plan's split scratch, as LinearPlan::scratchSize sizes it (the largest of several plans that share it).
struct Scratch {
  Guarded table, sums, partials, counters;
  Scratch(MetalBackend &backend, LinearScratchSize size)
      : table(backend, size.input, 0), sums(backend, size.sums, 0), partials(backend, size.partials, 0),
        counters(backend, size.counters, 0) {}
  [[nodiscard]] LinearScratch bindings() const { return {table.view, sums.view, partials.view, counters.view}; }
  void poison(uint32_t bits) const {
    if (partials.bytes) std::fill_n(static_cast<uint32_t *>(partials.view.contents()), partials.bytes / 4, bits);
  }
  [[nodiscard]] bool intact() const {
    const auto *count = static_cast<const uint32_t *>(counters.view.contents());
    return table.intact() && sums.intact() && partials.intact() && counters.intact() &&
           std::all_of(count, count + counters.bytes / 4, [](uint32_t c) { return c == 0; });
  }
};
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

// The output rows [0, storage) of a run and, for gate/up, the gate its up pass read.
struct Outcome {
  std::vector<uint16_t> output, gate;
};

// Runs one projection with `plan` over `x` (and `aux`), its split partials first set to `poison`; checks what every
// run must leave as it was: guards, counters and the padding columns past `covered`.
Outcome run(MetalBackend &backend, const Linear &linear, const LinearPlan &plan, const Projection &p,
            const Projection *gate, const std::vector<uint16_t> &x, const std::vector<uint16_t> &aux, uint32_t poison,
            uint32_t covered, const std::string &label) {
  const Scratch scratch(backend, plan.scratchSize());
  const Operands o(backend, plan, x, aux);
  scratch.poison(poison);
  CommandGraph graph;
  static_cast<void>(linear.add(graph, o.bindings(plan, scratch), p, plan, gate));
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  if (!scratch.intact() || !o.input.intact() || !o.aux.intact() || !o.output.intact() || !o.gate.intact())
    fail(label + ": a counter is not reset or a write past a buffer");
  Outcome out{o.output.halves(), plan.workload().epilogue == LinearEpilogue::GateUp ? o.gate.halves()
                                                                                   : std::vector<uint16_t>{}};
  const uint32_t columns = plan.workload().matrix.outputSize;
  for (const std::vector<uint16_t> *rows : {&out.output, &out.gate})
    for (uint64_t i = 0; i < rows->size(); ++i)
      if (i % columns >= covered && (*rows)[i] != kUnwritten) {
        fail(label + ": writes padding column " + std::to_string(i % columns));
        break;
      }
  return out;
}

// Every output of rows [0, rows) and the covered columns within its fp64 interval: the projection's, plus the
// residual, or times silu of the gate the up pass read (a gate/up plan's gate scratch, itself a projection).
void checkValues(const Outcome &out, const std::vector<Dot> &dots, const std::vector<Dot> &gateDots,
                 const std::vector<uint16_t> &aux, const LinearWorkload &w, uint32_t rows, uint32_t covered,
                 bool staged, const std::string &label) {
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
      const Interval p = projected(dots[i], staged);
      switch (w.epilogue) {
        case LinearEpilogue::None: check("the output", i, out.output[i], p); break;
        case LinearEpilogue::Residual:
          check("the output", i, out.output[i], withResidual(p, bf16ToFloat(aux[i])));
          break;
        case LinearEpilogue::UpWithGate: check("the output", i, out.output[i], withGate(p, bf16ToFloat(aux[i]))); break;
        case LinearEpilogue::GateUp:
          check("the gate", i, out.gate[i], projected(gateDots[i], staged));
          check("the output", i, out.output[i], withGate(p, bf16ToFloat(out.gate[i])));
          break;
      }
    }
  if (outside) fail(label + ": " + std::to_string(outside) + " values outside the fp64 bound; " + first);
}

// The fp32 destination of `plan` (the logits) over the input rows `x` of `bf16`, the run of its bf16 plan: every
// value of the active rows and covered columns rounds to that run's output bit for bit and lies within fp64 before
// the rounding; the padding columns, guards and counters stay as they were.
void floatOutput(MetalBackend &backend, const Linear &linear, const LinearPlan &plan, const Projection &p,
                 const std::vector<uint16_t> &x, const Outcome &bf16, const std::vector<Dot> &dots, uint32_t covered,
                 bool staged, const std::string &label) {
  const uint32_t columns = plan.workload().matrix.outputSize;
  const Scratch scratch(backend, plan.scratchSize());
  const Guarded input = bfloats(backend, x);
  const Guarded output(backend, uint64_t{plan.storageRows()} * columns * sizeof(float), 0xFF);
  scratch.poison(kPoisonNaN);
  CommandGraph graph;
  static_cast<void>(linear.add(graph, {.input = input.view, .output = output.view, .scratch = scratch.bindings()},
                               p, plan));
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  if (!scratch.intact() || !input.intact() || !output.intact())
    fail(label + " fp32: a counter is not reset or a write past a buffer");
  const auto *values = static_cast<const uint32_t *>(output.view.contents());
  uint64_t wrong = 0;
  for (uint64_t i = 0; i < uint64_t{plan.workload().rows} * columns; ++i) {
    const float value = std::bit_cast<float>(values[i]);
    wrong += i % columns >= covered ? values[i] != 0xFFFFFFFFu
                                    : floatToBf16(value) != bf16.output[i] ||
                                          !(std::fabs(value - dots[i].value) <= projectionBound(dots[i], staged));
  }
  if (wrong) fail(label + " fp32: " + std::to_string(wrong) + " values differ from the bf16 run or fp64");
}

bool sameRows(const std::vector<uint16_t> &a, uint64_t aRow, const std::vector<uint16_t> &b, uint64_t bRow,
              uint32_t rows, uint32_t columns) {
  return std::equal(a.begin() + aRow * columns, a.begin() + (aRow + rows) * columns, b.begin() + bRow * columns);
}

// The configuration of `tile` for a decode workload (its full column grid) or a prefill chunk.
LinearConfig config(LinearTile tile, const LinearWorkload &w, uint32_t splits) {
  if (w.phase == LinearPhase::Prefill) return {LinearTile::GgufStaged, 0, LinearSimdgroups::Two, splits};
  const uint32_t groups = w.matrix.outputSize / 64;
  return tile == LinearTile::GgufRegister ? LinearConfig{tile, groups, LinearSimdgroups::Four, splits}
                                           : LinearConfig{tile, groups, LinearSimdgroups::Two, splits};
}
LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  return {matrix, lanes * kLaneRows, LinearPhase::Decode, epilogue, WeightLayout::Block32};
}

// ---------------------------------------------------------------- decode
// One tile on single tensors [512, 2048] in every format (gate in the format three further on): at every lane count,
// K split and epilogue, on dense and sparse inputs, every output within fp64 and each lane's rows equal to a one-lane
// projection of them.
void decodeTile(MetalBackend &backend, const Linear &linear, LinearTile tile) {
  constexpr uint32_t N = 512, K = 2048, columns = N + kPadding;
  const bool staged = tile == LinearTile::GgufStaged;
  for (int fi = 0; fi < FMT_COUNT; ++fi) {
    const Tensor w = tensor(backend, Fmt(fi), N, K), g = tensor(backend, Fmt((fi + 3) % FMT_COUNT), N, K);
    const Projection up = projection({&w}, columns), gate = projection({&g}, columns);
    for (const Inputs inputs : {Inputs::Dense, Inputs::Sparse}) {
      const std::vector<float> x = activations(inputs, kMaximumRows, K);
      const std::vector<float> residual = residuals(kMaximumRows, columns);
      const std::vector<Dot> dots = products({&w}, columns, x, kMaximumRows);
      const std::vector<Dot> gateDots = products({&g}, columns, x, kMaximumRows);
      for (const LinearEpilogue epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
        for (const uint32_t splits : kSplits) {
          const std::string what = std::string(fmtName(fi)) + (inputs == Inputs::Sparse ? " sparse " : " dense ") +
                                   epilogueName(epilogue) + " S=" + std::to_string(splits);
          const Projection *gated = epilogue == LinearEpilogue::GateUp ? &gate : nullptr;
          std::vector<Outcome> lanesAlone;
          for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
            const std::vector<float> rows(x.begin() + uint64_t{lane} * kLaneRows * K,
                                          x.begin() + uint64_t{lane + 1} * kLaneRows * K);
            const std::vector<float> aux(residual.begin() + uint64_t{lane} * kLaneRows * columns,
                                         residual.begin() + uint64_t{lane + 1} * kLaneRows * columns);
            const LinearWorkload one = decode({columns, K}, 1, epilogue);
            const LinearPlan plan = Linear::plan(one, config(tile, one, splits));
            lanesAlone.push_back(run(backend, linear, plan, up, gated, storageRows(rows, K, kLaneRows, kLaneRows),
                                     storageRows(aux, columns, kLaneRows, kLaneRows), kPoisonNaN, N,
                                     what + " lane " + std::to_string(lane) + " alone"));
          }
          for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
            const LinearWorkload wl = decode({columns, K}, lanes, epilogue);
            const LinearPlan plan = Linear::plan(wl, config(tile, wl, splits));
            const uint32_t storage = plan.storageRows(), rows = wl.rows;
            const std::string label = what + " L=" + std::to_string(lanes);
            const std::vector<uint16_t> aux = storageRows(residual, columns, rows, storage);
            const Outcome out = run(backend, linear, plan, up, gated, storageRows(x, K, rows, storage), aux,
                                    kPoisonFinite, N, label);
            checkValues(out, dots, gateDots, aux, wl, rows, N, staged, label);
            if (epilogue == LinearEpilogue::None)
              floatOutput(backend, linear, Linear::plan(wl, config(tile, wl, splits), FloatOutput::Float32), up,
                          storageRows(x, K, rows, storage), out, dots, N, staged, label);
            for (uint32_t lane = 0; lane < lanes; ++lane)
              if (!sameRows(out.output, lane * kLaneRows, lanesAlone[lane].output, 0, kLaneRows, columns))
                fail(label + ": lane " + std::to_string(lane) + " differs from its one-lane projection");
          }
        }
    }
  }
  section(std::string(tileName(tile)) + " decode: " + std::to_string(FMT_COUNT) + " formats, 1-4 lanes, S 1-8, plain/residual/gate-up, dense and "
          "sparse inputs within fp64, lanes equal to one-lane projections, fp32 plain outputs rounding to bf16's");
}

// One fused projection of three segments of different formats ([512 | 256 | 256, 2048]): at every lane count and K
// split, within fp64 and equal to the projections of each segment alone.
void fusedDecode(MetalBackend &backend, const Linear &linear, LinearTile tile) {
  constexpr uint32_t K = 2048;
  for (int fi = 0; fi < FMT_COUNT; ++fi) {
    const Tensor a = tensor(backend, Fmt(fi), 512, K), b = tensor(backend, Fmt((fi + 3) % FMT_COUNT), 256, K),
                 c = tensor(backend, Fmt((fi + 5) % FMT_COUNT), 256, K);
    const std::vector<const Tensor *> parts{&a, &b, &c};
    const uint32_t covered = segmentColumns(parts), columns = covered + kPadding;
    const Projection fused = projection(parts, columns);
    const std::vector<float> x = activations(Inputs::Dense, kMaximumRows, K);
    const std::vector<Dot> dots = products(parts, columns, x, kMaximumRows);
    const std::string formats = std::string(fmtName(a.format)) + "|" + fmtName(b.format) + "|" + fmtName(c.format);
    for (const uint32_t splits : kSplits)
      for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
        const LinearWorkload wl = decode({columns, K}, lanes, LinearEpilogue::None);
        const LinearPlan plan = Linear::plan(wl, config(tile, wl, splits));
        const uint32_t storage = plan.storageRows(), rows = wl.rows;
        const std::string label = formats + " S=" + std::to_string(splits) + " L=" + std::to_string(lanes);
        const Outcome out = run(backend, linear, plan, fused, nullptr, storageRows(x, K, rows, storage), {},
                                kPoisonNaN, covered, label);
        checkValues(out, dots, {}, {}, wl, rows, covered, tile == LinearTile::GgufStaged, label);
        uint32_t offset = 0;
        for (const Tensor *part : parts) {
          const Projection alone = projection({part}, part->N);
          const LinearWorkload one = decode({part->N, K}, lanes, LinearEpilogue::None);
          const Outcome single = run(backend, linear, Linear::plan(one, config(tile, one, splits)), alone, nullptr,
                                     storageRows(x, K, rows, storage), {}, kPoisonNaN, part->N, label + " alone");
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
  section(std::string(tileName(tile)) + " fused: three segments in " + std::to_string(FMT_COUNT) + " format triples, 1-4 lanes, S 1-8 within fp64 "
          "and equal to each segment's projection");
}

// Gate/up on one tile for every gate and up format pair ([256, 1024]) and four pairs at [1024, 1024], with the K
// splits by lanes a decode step of these widths takes on large GPUs.
void gateUpPairs(MetalBackend &backend, const Linear &linear, LinearTile tile) {
  constexpr uint32_t K = 1024;
  constexpr uint32_t kSplitsByLanes[kMaximumLanes] = {1, 2, 4, 4};
  const auto pair = [&](Fmt gf, Fmt uf, uint32_t N, uint32_t lanes) {
    const uint32_t columns = N + kPadding;
    const Tensor g = tensor(backend, gf, N, K), u = tensor(backend, uf, N, K);
    const Projection gate = projection({&g}, columns), up = projection({&u}, columns);
    const LinearWorkload wl = decode({columns, K}, lanes, LinearEpilogue::GateUp);
    const LinearPlan plan = Linear::plan(wl, config(tile, wl, kSplitsByLanes[lanes - 1]));
    const std::vector<float> x = activations(Inputs::Dense, wl.rows, K);
    const std::string label = std::string(fmtName(gf)) + " gate " + fmtName(uf) + " up N=" + std::to_string(N) +
                              " L=" + std::to_string(lanes);
    const Outcome out = run(backend, linear, plan, up, &gate, storageRows(x, K, wl.rows, plan.storageRows()), {},
                            kPoisonNaN, N, label);
    checkValues(out, products({&u}, columns, x, wl.rows), products({&g}, columns, x, wl.rows), {}, wl, wl.rows, N,
                tile == LinearTile::GgufStaged, label);
  };
  for (int gf = 0; gf < FMT_COUNT; ++gf)
    for (int uf = 0; uf < FMT_COUNT; ++uf)
      for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) pair(Fmt(gf), Fmt(uf), 256, lanes);
  const std::array<std::array<Fmt, 2>, kMaximumLanes> wide{{{IQ4XS, Q4K}, {Q5K, Q5K}, {Q4K, IQ4XS}, {Q3K, Q6K}}};
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) pair(wide[lanes - 1][0], wide[lanes - 1][1], 1024, lanes);
  section(std::string(tileName(tile)) + " gate/up: " + std::to_string(FMT_COUNT * FMT_COUNT) + " gate and up format pairs at N 256 and 4 at N 1024, 1-4 lanes, "
          "within fp64");
}

// ---------------------------------------------------------------- prefill
// A 168-row chunk on the 128-row tiles: the second tile holds 40 rows, so its last two 32-row simdgroups skip their
// matmuls and leave the rows from 192 unwritten. Chunks of up to 32 rows run the decode tiles, whose rows equal the
// 128-row tiles' bitwise without K splits and lie within fp64 with them. Every format ([1024, 1024]) at each
// epilogue, and a fused projection at its segments' column offsets.
void prefill(MetalBackend &backend, const Linear &linear) {
  constexpr uint32_t K = 1024, kChunk = 168, kSimdgroupRows = 32, kSplitChunk = 4;
  const auto chunks = [&](const Projection &p, const std::vector<const Tensor *> &parts, LinearEpilogue epilogue,
                          const std::string &what) {
    const uint32_t covered = segmentColumns(parts), columns = p.outputSize;
    const LinearWorkload tiles{{columns, K}, kChunk, LinearPhase::Prefill, epilogue, WeightLayout::Block32};
    const LinearPlan tilePlan = Linear::plan(tiles, {LinearTile::GgufStaged, 0, LinearSimdgroups::Four, 1});
    const uint32_t storage = tilePlan.storageRows();
    const std::vector<float> x = activations(Inputs::Dense, storage, K);
    // The residual, or the gate of the up-with-gate epilogue.
    const std::vector<float> auxiliary = epilogue == LinearEpilogue::Residual
        ? residuals(storage, columns) : activations(Inputs::Dense, storage, columns);
    const std::vector<uint16_t> aux = storageRows(auxiliary, columns, storage, storage);
    const std::vector<Dot> dots = products(parts, columns, x, kChunk);
    const std::string label = what + " " + epilogueName(epilogue);
    const Outcome whole = run(backend, linear, tilePlan, p, nullptr, storageRows(x, K, kChunk, storage), aux,
                              kPoisonNaN, covered, label + " 168 rows");
    checkValues(whole, dots, {}, aux, tiles, kChunk, covered, true, label + " 168 rows");
    const uint32_t written = (kChunk + kSimdgroupRows - 1) / kSimdgroupRows * kSimdgroupRows;
    if (!std::all_of(whole.output.begin() + uint64_t{written} * columns, whole.output.end(),
                     [](uint16_t v) { return v == kUnwritten; }))
      fail(label + ": simdgroups past the 168-row chunk write their rows");
    for (const uint32_t rows : {8u, 16u, 24u, 32u})
      for (const uint32_t splits : {1u, kSplitChunk}) {
        const LinearWorkload chunk{{columns, K}, rows, LinearPhase::Prefill, epilogue, WeightLayout::Block32};
        const LinearPlan plan = Linear::plan(chunk, config(LinearTile::GgufStaged, chunk, splits));
        const std::string name = label + " chunk of " + std::to_string(rows) + " rows S=" + std::to_string(splits);
        const std::vector<uint16_t> chunkAux(aux.begin(), aux.begin() + uint64_t{plan.storageRows()} * columns);
        const Outcome out = run(backend, linear, plan, p, nullptr, storageRows(x, K, rows, plan.storageRows()),
                                chunkAux, kPoisonNaN, covered, name);
        checkValues(out, dots, {}, chunkAux, chunk, rows, covered, true, name);
        if (splits == 1 && !sameRows(out.output, 0, whole.output, 0, rows, columns))
          fail(name + ": differs from the 128-row tiles");
      }
  };
  for (int fi = 0; fi < FMT_COUNT; ++fi) {
    const Tensor w = tensor(backend, Fmt(fi), 1024, K);
    const Projection p = projection({&w}, w.N);
    for (const LinearEpilogue e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      chunks(p, {&w}, e, fmtName(fi));
    const Tensor b = tensor(backend, Fmt((fi + 3) % FMT_COUNT), 256, K), c = tensor(backend, Fmt((fi + 5) % FMT_COUNT),
                                                                                   256, K);
    const std::vector<const Tensor *> parts{&w, &b, &c};
    chunks(projection(parts, segmentColumns(parts) + kPadding), parts, LinearEpilogue::None,
           std::string(fmtName(fi)) + "|" + fmtName(b.format) + "|" + fmtName(c.format));
  }
  section("prefill: " + std::to_string(FMT_COUNT) + " formats plain/residual/up-with-gate and fused segments, 128-row tiles over a 168-row chunk and "
          "chunks of 8-32 rows (S 1 and 4) within fp64, equal to the 128-row tiles");
}

// ---------------------------------------------------------------- split visibility
// Two projections of a decode step run back to back and share one split scratch, as every GGUF projection of a step
// does, at each pair of K splits the tile's policy (Apple9: register, Apple10: staged) picks for 8-80 cores. The
// partials are overwritten before each projection, with a large finite value and in another run with NaN, so a
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
    const Linear policy(device);
    const LinearConfig a = policy.plan(workloads[0], weights[0][0]).configuration();
    const LinearConfig b = policy.plan(workloads[1], weights[1][0]).configuration();
    if (a.tile != tile || b.tile != tile) fail("GPU family " + std::to_string(family) + " plans another tile");
    if (std::max(a.splits, b.splits) > 1) splitPairs.insert({a.splits, b.splits});
  }
  const auto matrix = [](LinearMatrix m) { return std::to_string(m.outputSize) + "x" + std::to_string(m.inputSize); };
  const std::string shape = std::string(tileName(tile)) + " " + std::to_string(lanes) + " lanes, " +
                            matrix(pair[0]->matrix) + " then " + matrix(pair[1]->matrix);
  if (splitPairs.empty()) fail(shape + ": the policy splits neither projection");
  std::string pairs;
  for (const auto &splits : splitPairs) pairs += " " + std::to_string(splits[0]) + "/" + std::to_string(splits[1]);
  std::cout << "  " << shape << ": splits" << pairs << '\n';
  const auto plan = [&](uint32_t i, uint32_t splits) {
    return Linear::plan(workloads[i], config(tile, workloads[i], splits));
  };
  LinearScratchSize size = plan(0, 1).scratchSize();
  size.include(plan(1, 1).scratchSize());
  for (const auto &splits : splitPairs)
    for (uint32_t i = 0; i < 2; ++i) size.include(plan(i, splits[i]).scratchSize());
  const Scratch scratch(backend, size);
  MetalBuffer poison = backend.allocateBuffer(size.partials);
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
      const Outcome out{operands[i].output.halves(), operands[i].gate.halves()};
      checkValues(out, pair[i]->dots, pair[i]->gateDots, auxiliary[i], workloads[i], workloads[i].rows,
                  pair[i]->matrix.outputSize, tile == LinearTile::GgufStaged,
                  what + " projection " + std::to_string(i) + " S=" + std::to_string(splits[i]));
    }
    if (!scratch.intact()) fail(what + ": a counter is not reset or a write past the scratch");
  };
  CommandGraph unsplit;
  for (uint32_t i = 0; i < 2; ++i)
    static_cast<void>(linear.add(unsplit, operands[i].bindings(plan(i, 1), scratch), weights[i][0], plan(i, 1),
                                 gateOf(i)));
  static_cast<void>(backend.submitCommand(unsplit.dispatches()));
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
      static_cast<void>(backend.submitCommand(graph.dispatches()));
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
// ops::Embedding over the native rows of every embedding format (kernels/shared/embedding.metal): each gathered value
// is the bf16 rounding of GGML's fp32 value, for tokens in any order, repeated, and the vocabulary's first and last.
void tokenGather(MetalBackend &backend) {
  constexpr uint32_t kVocabulary = 64, kHidden = 1024;
  const std::vector<uint32_t> tokens{kVocabulary - 1, 0, 17, 17, 42, 3, kVocabulary - 1, 29, 8};
  int formats = 0;
  for (int fi = 0; fi < FMT_COUNT; ++fi) {
    if (!gguf_embedding_format(fi)) continue;
    ++formats;
    const Fmt f = Fmt(fi);
    const std::vector<uint8_t> native = makeNative(f, kVocabulary, kHidden, rng);
    const EmbeddingWeights table(kVocabulary, kHidden, NativeRows(upload(backend, native), f));
    const Guarded ids(backend, tokens.size() * sizeof(uint32_t), 0), output(backend, tokens.size() * kHidden * 2, 0xFF);
    std::memcpy(ids.view.contents(), tokens.data(), ids.bytes);
    CommandGraph graph;
    Embedding::add(graph, ids.view, table, output.view, uint32_t(tokens.size()));
    static_cast<void>(backend.submitCommand(graph.dispatches()));
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
  section("token gather: " + std::to_string(formats) + " embedding formats, each value bf16 of GGML's fp32 value");
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf-projection <production-and-test.metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const Linear linear(backend.capabilities());
      for (const LinearTile tile : {LinearTile::GgufRegister, LinearTile::GgufStaged}) {
        decodeTile(backend, linear, tile);
        fusedDecode(backend, linear, tile);
        gateUpPairs(backend, linear, tile);
      }
      prefill(backend, linear);
      // The 27B out_proj then down, and gdn_in then gate/up: K 6144, 17408 and 5120.
      const SplitOperand out = splitOperand(backend, Q4K, {5120, 6144}, LinearEpilogue::Residual);
      const SplitOperand down = splitOperand(backend, Q6K, {5120, 17408}, LinearEpilogue::Residual);
      const SplitOperand gdn = splitOperand(backend, IQ4XS, {12288, 5120}, LinearEpilogue::None);
      const SplitOperand gateUp = splitOperand(backend, Q4K, {17408, 5120}, LinearEpilogue::GateUp);
      for (const LinearTile tile : {LinearTile::GgufRegister, LinearTile::GgufStaged})
        for (const uint32_t lanes : {1u, kMaximumLanes}) {
          splitVisibility(backend, linear, tile, {&out, &down}, lanes);
          splitVisibility(backend, linear, tile, {&gdn, &gateUp}, lanes);
        }
      section("split visibility: both tiles, 1 and 4 lanes, every split pair the policy picks for 8-80 cores, "
              "independent of poisoned partials and within fp64");
      tokenGather(backend);
    } catch (const std::exception &e) {
      std::cerr << "gguf-projection: FAIL: " << e.what() << '\n';
      return 1;
    }
    std::cout << "gguf-projection: " << (failures ? "FAIL (" + std::to_string(failures) + " failures)" : "PASS")
              << '\n';
    return failures ? 1 : 0;
  }
}
