// LoRA updates of projections (ops::ProjectionLora) through ops::Linear, and the adapter that attaches them
// (model/LoraAdapter.hpp).
// - update: every plan of an updated projection writes what the same plan writes without the update, plus B (A x):
//   each output within the two runs' bf16 rounding and the rounding of t = A x of the fp64 sum, and the outputs
//   together on a line of slope one through the fp64 corrections. On the staged decode tile at every K split and
//   epilogue, whose tile adds the update to its sums; on the register tile and on the prefill tiles, bf16 and int8,
//   whose sums a kernel after them corrects, a chunk whose last tile of rows is not whole included; a fused
//   projection of three tensors, the middle one without an update; and a view of a tensor's leading rows.
// - zero: an update whose B is zero changes no bit of the staged decode tile's rows.
// - SwiGLU: Linear::addSwiGlu of a gate and an up with updates writes (up + B t) silu(gate + B t) of the plain plans'
//   sums at a decode step's rows and at a chunk's, and its down's rows the residual projection's of those rows plus
//   down's own update.
// - adapter: a PEFT checkpoint's modules become the projections' updates with the adapter's scale, each product
//   B A within bf16 rounding of the checkpoint's, in F32, F16 and BF16, at a rank padded to whole steps; the
//   weights' fingerprint and allocation count it; and a tensor that is no module's, a module without its B, of
//   another shape or rank, or of a layer that has no such projection is refused by name.
// Every run leaves the guard bands past its buffers as they were.
#include "GgufFormatReference.hpp"
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestCheckpoint.hpp"
#include "TestFiles.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "model/LoraAdapter.hpp"
#include "ops/Linear.hpp"
#include "ops/Lora.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace gguf_reference;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::ulpBf16;

namespace {

constexpr uint16_t kNaN = 0x7FC0;       // bf16 quiet NaN: the input of padding rows
constexpr uint16_t kUnwritten = 0xFFFF; // bf16 NaN no kernel writes: the output before a run
constexpr uint8_t kGuardByte = 0xA5;
constexpr uint64_t kGuardBytes = 256;
// Residuals of the projections' own magnitude, so that a sum's rounding stays below its correction.
constexpr float kResidualMagnitude = 4;

std::mt19937 rng(7);
int failures = 0;
int sectionFailures = 0;

void fail(const std::string &what) {
  constexpr int kShown = 8;
  if (sectionFailures++ < kShown) std::cout << "  FAIL " << what << '\n';
  ++failures;
}
void section(const std::string &what) {
  std::cout << what << (sectionFailures ? " FAIL (" + std::to_string(sectionFailures) + " failures)" : " ok") << '\n';
  sectionFailures = 0;
}

// ---------------------------------------------------------------- weights
MetalBuffer upload(MetalBackend &backend, const void *bytes, uint64_t size) {
  MetalBuffer buffer = test::sharedBuffer(backend, size);
  std::memcpy(buffer.contents(), bytes, size);
  return buffer;
}

QuantizedSegment tensor(MetalBackend &backend, Fmt f, uint32_t N, uint32_t K) {
  const std::vector<uint8_t> native = makeNative(f, N, K, rng);
  const Packed planes = repack(f, native, N, K, nullptr);
  return QuantizedSegment::planes(
      f, N, K, upload(backend, planes.w0.data(), planes.w0.size()),
      kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1.data(), planes.w1.size()) : MetalBuffer{},
      upload(backend, planes.meta.data(), planes.meta.size()));
}

Projection projection(std::vector<QuantizedSegment> segments) {
  BlockWeights blocks;
  uint32_t offset = 0;
  for (QuantizedSegment &s : segments) {
    s.columnOffset = offset;
    offset += s.outputSize;
    blocks.segments.push_back(std::move(s));
  }
  const uint32_t inputSize = blocks.segments.front().inputSize;
  return Projection(offset, inputSize, std::move(blocks));
}

// An update's values beside it: a[slot][j][k] and b[segment][n][j], bf16 values.
struct Update final {
  uint32_t rank = 0;
  std::vector<std::vector<float>> a, b;
};

std::vector<float> bf16Values(uint64_t count, float magnitude) {
  std::uniform_real_distribution<float> unit(-magnitude, magnitude);
  std::vector<float> values(count);
  for (float &v : values) v = bf16ToFloat(floatToBf16(unit(rng)));
  return values;
}
MetalBuffer bf16Buffer(MetalBackend &backend, const std::vector<float> &values) {
  std::vector<uint16_t> bits(values.size());
  for (size_t i = 0; i < values.size(); ++i) bits[i] = floatToBf16(values[i]);
  return upload(backend, bits.data(), bits.size() * 2);
}

// Attaches an update of `rank` to `p`: a module on each segment `updated` names, of B scaled by `scale`.
Update attach(MetalBackend &backend, Projection &p, uint32_t rank, const std::vector<bool> &updated,
              float scale = 0.5f) {
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  Update update{rank, {}, {}};
  ProjectionLora lora;
  lora.rank = rank;
  lora.seed = 3;
  lora.segments.resize(segments.size());
  std::vector<float> stacked;
  for (size_t index = 0; index < segments.size(); ++index) {
    update.b.emplace_back();
    if (!updated[index]) continue;
    // A of a magnitude that leaves t near one at dense inputs of a thousand values.
    update.a.push_back(bf16Values(uint64_t{rank} * p.inputSize, 0.1f));
    stacked.insert(stacked.end(), update.a.back().begin(), update.a.back().end());
    update.b.back() = bf16Values(uint64_t{segments[index].outputSize} * rank, scale);
    lora.segments[index] = {bf16Buffer(backend, update.b.back()), lora.slots++};
  }
  lora.a = QuantizedSegment::rows(GGUF_FMT_BF16, lora.slots * rank, p.inputSize, bf16Buffer(backend, stacked));
  p.lora = std::move(lora);
  return update;
}

// ---------------------------------------------------------------- reference
std::vector<float> activations(uint32_t rows, uint32_t width, float magnitude = 1.0f) {
  return bf16Values(uint64_t{rows} * width, magnitude);
}

// The fp64 correction B (A x) of each output and the bound of what the kernels' t, rounded to bf16 from an fp32 sum,
// moves it by.
struct Corrections final {
  std::vector<double> value, bound;
};
Corrections corrections(const Projection &p, const Update &update, const std::vector<float> &x, uint32_t rows) {
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  const uint32_t K = p.inputSize, columns = p.outputSize, rank = update.rank;
  Corrections result{std::vector<double>(uint64_t{rows} * columns, 0.0),
                     std::vector<double>(uint64_t{rows} * columns, 0.0)};
  for (size_t index = 0; index < segments.size(); ++index) {
    if (update.b[index].empty()) continue;
    const std::vector<float> &a = update.a[p.lora.segments[index].slot], &b = update.b[index];
    for (uint32_t r = 0; r < rows; ++r) {
      std::vector<double> t(rank);
      for (uint32_t j = 0; j < rank; ++j) {
        double sum = 0;
        for (uint32_t k = 0; k < K; ++k) sum += double(a[uint64_t{j} * K + k]) * x[uint64_t{r} * K + k];
        t[j] = bf16ToFloat(floatToBf16(float(sum)));
      }
      for (uint32_t n = 0; n < segments[index].outputSize; ++n) {
        double sum = 0, bound = 0;
        for (uint32_t j = 0; j < rank; ++j) {
          const double weight = b[uint64_t{n} * rank + j];
          sum += weight * t[j];
          bound += std::fabs(weight) * (ulpBf16(float(t[j])) + 1e-3 * std::fabs(t[j]));
        }
        const uint64_t at = uint64_t{r} * columns + segments[index].columnOffset + n;
        result.value[at] = sum;
        result.bound[at] = bound;
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------- running a plan
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
Guarded rowsOf(MetalBackend &backend, const std::vector<float> &x, uint32_t width, uint32_t active,
               uint32_t storage) {
  std::vector<uint16_t> rows(uint64_t{storage} * width, kNaN);
  for (uint64_t i = 0; i < uint64_t{active} * width; ++i) rows[i] = floatToBf16(x[i]);
  Guarded g(backend, rows.size() * 2, 0);
  std::memcpy(g.view.contents(), rows.data(), g.bytes);
  return g;
}

struct Scratch {
  Guarded table, sums, partials, counters, rotated, loraDown, loraRows;
  Scratch(MetalBackend &backend, LinearScratchSize size)
      : table(backend, size.input, 0), sums(backend, size.sums, 0), partials(backend, size.partials, 0),
        counters(backend, size.counters, 0), rotated(backend, size.rotated, 0xFF),
        loraDown(backend, size.loraDown, 0), loraRows(backend, size.loraRows, 0xFF) {}
  [[nodiscard]] LinearScratch bindings() const {
    return {table.view, sums.view, partials.view, counters.view, rotated.view, loraDown.view, loraRows.view};
  }
  [[nodiscard]] bool intact() const {
    return table.intact() && sums.intact() && partials.intact() && counters.intact() && rotated.intact() &&
           loraDown.intact() && loraRows.intact();
  }
};
// The scratch of `plan` over `p`: the plan's, and what the projection's shape adds for `rows` rows.
LinearScratchSize scratchSize(const Linear &linear, const LinearPlan &plan, const Projection &p, uint32_t rows) {
  LinearScratchSize size = plan.scratchSize();
  const LinearScratchSize shape = linear.scratchSize(p.shape(), rows);
  size.rotated = shape.rotated;
  size.loraDown = shape.loraDown;
  size.loraRows = shape.loraRows;
  return size;
}

// The rows [0, storage) `plan` writes of `x`, with `residual` under a residual plan.
std::vector<uint16_t> run(MetalBackend &backend, const Linear &linear, const LinearPlan &plan, const Projection &p,
                          const std::vector<float> &x, const std::vector<float> &residual,
                          const std::string &label) {
  const LinearWorkload w = plan.workload();
  const uint32_t storage = plan.storageRows();
  const Scratch scratch(backend, scratchSize(linear, plan, p, w.rows));
  const Guarded input = rowsOf(backend, x, p.inputSize, w.rows, storage);
  const Guarded aux = rowsOf(backend, residual, p.outputSize, residual.empty() ? 0 : w.rows,
                             residual.empty() ? 0 : storage);
  const Guarded output(backend, uint64_t{storage} * p.outputSize * 2, 0xFF);
  CommandGraph graph;
  static_cast<void>(linear.add(graph,
                               {.input = input.view, .output = output.view, .residual = aux.view,
                                .scratch = scratch.bindings()},
                               p, plan));
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  if (!scratch.intact() || !input.intact() || !aux.intact() || !output.intact())
    fail(label + ": a write past a buffer");
  return output.halves();
}

// Each output of the updated run within its bound of the plain run's plus the correction, and all of them on a line
// of slope one through the corrections.
void checkUpdate(const std::vector<uint16_t> &plain, const std::vector<uint16_t> &updated, const Corrections &want,
                 const std::vector<float> &residual, uint32_t rows, uint32_t columns, const std::string &label) {
  double product = 0, square = 0;
  for (uint64_t i = 0; i < uint64_t{rows} * columns; ++i) {
    const float before = bf16ToFloat(plain[i]), after = bf16ToFloat(updated[i]);
    const double moved = double(after) - before;
    // The two runs' rounding, the sums' own before a residual joined them, and the rounding of t.
    const double bound = ulpBf16(before) + ulpBf16(after) +
                         (residual.empty() ? 0.0 : double(ulpBf16(std::fabs(before) + std::fabs(residual[i])))) +
                         want.bound[i];
    if (!(std::fabs(moved - want.value[i]) <= bound)) {
      fail(label + ": output " + std::to_string(i) + " moved by " + std::to_string(moved) + ", its correction is " +
           std::to_string(want.value[i]) + " within " + std::to_string(bound));
    }
    product += moved * want.value[i];
    square += want.value[i] * want.value[i];
  }
  const double slope = square ? product / square : 1.0;
  if (!(std::fabs(slope - 1.0) <= 0.01)) fail(label + ": the outputs follow the corrections at slope " + std::to_string(slope));
}

// A copy of `p` without its update.
Projection withoutUpdate(Projection p) {
  p.lora = {};
  return p;
}

// One plan of `p` against the same plan of it without its update.
void checkPlan(MetalBackend &backend, const Linear &linear, const LinearPlan &plan, const Projection &p,
               const Update &update, const std::string &label) {
  const LinearWorkload w = plan.workload();
  const std::vector<float> x = activations(w.rows, p.inputSize);
  std::vector<float> residual;
  if (w.epilogue == LinearEpilogue::Residual) residual = activations(w.rows, p.outputSize, kResidualMagnitude);
  const std::vector<uint16_t> plain = run(backend, linear, plan, withoutUpdate(p), x, residual, label);
  const std::vector<uint16_t> updated = run(backend, linear, plan, p, x, residual, label);
  checkUpdate(plain, updated, corrections(p, update, x, w.rows), residual, w.rows, p.outputSize, label);
}

LinearConfig config(LinearTile tile, uint32_t splits) {
  LinearConfig c;
  c.tile = tile;
  c.splits = splits;
  return c;
}

void updates(MetalBackend &backend, const Linear &linear) {
  constexpr uint32_t K = 1024, N = 192;
  const Fmt affine = Fmt(GGUF_FMT_AF4G64);
  for (const Fmt f : {Q4K, affine}) {
    Projection p = projection({tensor(backend, f, N, K)});
    const Update update = attach(backend, p, 32, {true});
    for (const LinearEpilogue epilogue : {LinearEpilogue::None, LinearEpilogue::Residual}) {
      const std::string kind = std::string(fmtName(f)) + (epilogue == LinearEpilogue::None ? " plain" : " residual");
      for (const uint32_t rows : {8u, 24u, 64u})
        for (const uint32_t splits : {1u, 4u})
          checkPlan(backend, linear, Linear::plan({{N, K}, rows, epilogue}, config(LinearTile::GgufStaged, splits),
                                                  FloatOutput::BFloat16),
                    p, update, "staged " + kind + " rows " + std::to_string(rows) + " splits " + std::to_string(splits));
      if (quant_register_tensor(f, K))
        checkPlan(backend, linear,
                  Linear::plan({{N, K}, 24, epilogue}, config(LinearTile::GgufRegister, 1), FloatOutput::BFloat16), p,
                  update, "register " + kind);
      // A chunk whose last tile of rows, of the prefill tiles and of the correction's, is not whole.
      checkPlan(backend, linear,
                Linear::plan({{N, K}, 200, epilogue}, config(LinearTile::GgufPrefill, 1), FloatOutput::BFloat16), p,
                update, "prefill " + kind);
      // The device's own plans: of a decode step's rows and of a chunk's, the int8 prefill tile where it runs it.
      for (const uint32_t rows : {16u, 200u})
        checkPlan(backend, linear, linear.plan(p, rows, epilogue), p, update,
                  "device plan " + kind + " rows " + std::to_string(rows));
    }
  }
  section("update: one tensor, Q4_K and af4g64, plain and residual, on the staged tile at 8-64 rows and 1 and 4 "
          "splits, the register tile, the prefill tile at 200 rows and the device's plans");

  // Three tensors of one input, the middle one without a module, at rank 64: a fused projection, which takes no
  // epilogue.
  {
    Projection p = projection({tensor(backend, Q4K, 128, K), tensor(backend, Q4K, 64, K), tensor(backend, Q4K, 64, K)});
    const Update update = attach(backend, p, 64, {true, false, true});
    for (const LinearTile tile : {LinearTile::GgufStaged, LinearTile::GgufRegister})
      checkPlan(backend, linear,
                Linear::plan({{p.outputSize, K}, 16, LinearEpilogue::None}, config(tile, 1), FloatOutput::BFloat16),
                p, update, std::string(tile == LinearTile::GgufStaged ? "staged" : "register") + " fused");
    checkPlan(backend, linear,
              Linear::plan({{p.outputSize, K}, 150, LinearEpilogue::None}, config(LinearTile::GgufPrefill, 1),
                           FloatOutput::BFloat16),
              p, update, "prefill fused");
  }
  section("update: three tensors fused, the middle one without a module, rank 64, on both decode tiles and the "
          "prefill tile");

  // A view of a tensor's leading rows keeps the update over them.
  {
    Projection p = projection({tensor(backend, affine, 256, K)});
    Update update = attach(backend, p, 32, {true});
    const Projection view = p.rows(backend, {{0, 128}});
    update.b.front().resize(uint64_t{128} * update.rank);
    for (const uint32_t rows : {16u, 200u})
      checkPlan(backend, linear, linear.plan(view, rows, LinearEpilogue::None), view, update,
                "leading rows, rows " + std::to_string(rows));
    if (p.leadingInputs(512).lora) fail("a view of leading inputs keeps an update of all the inputs");
  }
  section("update: a view of a tensor's leading rows keeps the update over them, a view of leading inputs none");
}

void zeroUpdate(MetalBackend &backend, const Linear &linear) {
  constexpr uint32_t K = 1024, N = 192;
  Projection p = projection({tensor(backend, Q4K, N, K)});
  static_cast<void>(attach(backend, p, 32, {true}, 0.0f));
  for (const LinearEpilogue epilogue : {LinearEpilogue::None, LinearEpilogue::Residual})
    for (const uint32_t splits : {1u, 4u}) {
      const LinearPlan plan =
          Linear::plan({{N, K}, 24, epilogue}, config(LinearTile::GgufStaged, splits), FloatOutput::BFloat16);
      const std::vector<float> x = activations(24, K);
      std::vector<float> residual;
      if (epilogue == LinearEpilogue::Residual) residual = activations(24, N, kResidualMagnitude);
      if (run(backend, linear, plan, p, x, residual, "zero") !=
          run(backend, linear, plan, withoutUpdate(p), x, residual, "zero"))
        fail("an update of zeros changes the staged tile's rows at splits " + std::to_string(splits));
    }
  section("zero: an update whose B is zero changes no bit of the staged decode tile's rows");
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

void swiGlu(MetalBackend &backend, const Linear &linear) {
  constexpr uint32_t hidden = 1024, intermediate = 256;
  const Fmt affine = Fmt(GGUF_FMT_AF4G64);
  Projection gate = projection({tensor(backend, affine, intermediate, hidden)});
  Projection up = projection({tensor(backend, affine, intermediate, hidden)});
  Projection down = projection({tensor(backend, affine, hidden, intermediate)});
  // Updates small enough to leave the gate in the range where silu is far from its floor.
  const Update gateUpdate = attach(backend, gate, 32, {true}, 0.25f), upUpdate = attach(backend, up, 32, {true});
  const Update downUpdate = attach(backend, down, 32, {true});
  const Projection plainGate = withoutUpdate(gate), plainUp = withoutUpdate(up), plainDown = withoutUpdate(down);
  for (const uint32_t rows : {16u, 200u}) {
    const std::string label = "SwiGLU rows " + std::to_string(rows);
    // Storage of whole prefill tiles holds every plan's rows.
    const uint32_t storage = (rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS;
    const std::vector<float> x = activations(rows, hidden), residual = activations(rows, hidden, kResidualMagnitude);
    LinearScratchSize size;
    for (const Projection *p : {&gate, &up, &down}) size.include(linear.scratchSize(p->shape(), rows));
    const Scratch scratch(backend, size);
    const Guarded normalized = rowsOf(backend, x, hidden, rows, storage);
    const Guarded residualRows = rowsOf(backend, residual, hidden, rows, storage);
    const uint64_t wide = uint64_t{storage} * intermediate * 2, narrow = uint64_t{storage} * hidden * 2;
    const Guarded gateScratch(backend, wide, 0xFF), intermediateRows(backend, wide, 0xFF), output(backend, narrow, 0xFF);
    const PrefillFfnBuffers buffers{normalized.view, gateScratch.view, intermediateRows.view, scratch.bindings()};
    CommandGraph graph;
    linear.addSwiGlu(graph, {&gate, &up, &down}, buffers, residualRows.view, output.view, rows);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    if (!scratch.intact() || !gateScratch.intact() || !intermediateRows.intact() || !output.intact())
      fail(label + ": a write past a buffer");
    const std::vector<uint16_t> h = intermediateRows.halves(), y = output.halves();
    // The plain plans' sums of gate and up.
    const std::vector<uint16_t> g0 = run(backend, linear, linear.plan(plainGate, rows), plainGate, x, {}, label),
                                u0 = run(backend, linear, linear.plan(plainUp, rows), plainUp, x, {}, label);
    const Corrections dg = corrections(gate, gateUpdate, x, rows), du = corrections(up, upUpdate, x, rows);
    std::vector<float> intermediateValues(uint64_t{rows} * intermediate);
    for (uint64_t i = 0; i < uint64_t{rows} * intermediate; ++i) {
      const double g = bf16ToFloat(g0[i]) + dg.value[i], u = bf16ToFloat(u0[i]) + du.value[i];
      // The sums' rounding and t's move gate and up; silu's slope is at most 1.1.
      const double gBound = ulpBf16(bf16ToFloat(g0[i])) + dg.bound[i], uBound = ulpBf16(bf16ToFloat(u0[i])) + du.bound[i];
      const double want = u * silu(g);
      const double bound = std::fabs(silu(g)) * uBound + 1.1 * (std::fabs(u) + uBound) * gBound +
                           ulpBf16(float(want)) + 1e-3 * std::fabs(want);
      const float got = bf16ToFloat(h[i]);
      if (!(std::fabs(got - want) <= bound))
        fail(label + ": intermediate " + std::to_string(i) + " is " + std::to_string(got) + ", not " +
             std::to_string(want) + " within " + std::to_string(bound));
      intermediateValues[i] = got;
    }
    // Down over the rows the pair wrote: the residual projection's without its update, plus the update.
    const std::vector<uint16_t> y0 =
        run(backend, linear, linear.plan(plainDown, rows, LinearEpilogue::Residual), plainDown, intermediateValues,
            residual, label);
    checkUpdate(y0, std::vector<uint16_t>(y.begin(), y.begin() + y0.size()),
                corrections(down, downUpdate, intermediateValues, rows), residual, rows, hidden, label + " down");
    for (uint64_t i = uint64_t{rows} * intermediate; i < h.size(); ++i)
      if (h[i] != kUnwritten && rows > kMaximumDecodeTileRows) {
        fail(label + ": writes intermediate rows past the chunk's");
        break;
      }
  }
  section("SwiGLU: a gate, up and down with updates at a decode step's rows and a chunk's");
}

// ---------------------------------------------------------------- the adapter
std::vector<uint8_t> floats(const std::vector<float> &values, const std::string &dtype) {
  std::vector<uint8_t> bytes;
  for (const float value : values) {
    if (dtype == "F32") {
      const auto *raw = reinterpret_cast<const uint8_t *>(&value);
      bytes.insert(bytes.end(), raw, raw + 4);
    } else {
      const uint16_t bits = dtype == "F16" ? std::bit_cast<uint16_t>(static_cast<_Float16>(value)) : floatToBf16(value);
      const auto *raw = reinterpret_cast<const uint8_t *>(&bits);
      bytes.insert(bytes.end(), raw, raw + 2);
    }
  }
  return bytes;
}

// A projection of float tensors, which an adapter's modules update like any other.
Projection floatProjection(MetalBackend &backend, std::initializer_list<uint32_t> outputs, uint32_t inputSize) {
  std::vector<QuantizedSegment> segments;
  for (const uint32_t outputSize : outputs)
    segments.push_back(QuantizedSegment::floats(outputSize, inputSize,
                                                test::sharedBuffer(backend, uint64_t{outputSize} * inputSize * 4)));
  return projection(std::move(segments));
}

struct Module final {
  std::string name, dtype;
  uint32_t outputSize, inputSize;
  std::vector<float> a, b;
};
Module module(const std::string &name, const std::string &dtype, uint32_t outputSize, uint32_t inputSize,
              uint32_t rank) {
  std::uniform_real_distribution<float> unit(-1.f, 1.f);
  Module m{name, dtype, outputSize, inputSize, std::vector<float>(uint64_t{rank} * inputSize),
           std::vector<float>(uint64_t{outputSize} * rank)};
  for (float &v : m.a) v = unit(rng) * 0.05f;
  for (float &v : m.b) v = unit(rng) * 3.0f;
  return m;
}
std::vector<test::SyntheticTensor> tensors(const std::vector<Module> &modules, uint32_t rank) {
  std::vector<test::SyntheticTensor> result;
  for (const Module &m : modules) {
    const std::string stem = "base_model.model.model.language_model.layers." + m.name;
    result.push_back({stem + ".lora_A.weight", m.dtype, {rank, m.inputSize}, floats(m.a, m.dtype)});
    result.push_back({stem + ".lora_B.weight", m.dtype, {m.outputSize, rank}, floats(m.b, m.dtype)});
  }
  return result;
}

// Two layers: an attention with a dense FFN, and a dense FFN whose mixer is no attention.
struct Layers final {
  Projection input, output, gate, up, down, gate1, up1, down1;
  explicit Layers(MetalBackend &backend)
      : input(floatProjection(backend, {128, 64, 64}, 256)), output(floatProjection(backend, {256}, 128)),
        gate(floatProjection(backend, {512}, 256)), up(floatProjection(backend, {512}, 256)),
        down(floatProjection(backend, {256}, 512)), gate1(floatProjection(backend, {512}, 256)),
        up1(floatProjection(backend, {512}, 256)), down1(floatProjection(backend, {256}, 512)) {}
  [[nodiscard]] std::vector<model::LoraLayerTargets> targets() {
    return {{&input, &output, &gate, &up, &down}, {nullptr, nullptr, &gate1, &up1, &down1}};
  }
};

void adapter(MetalBackend &backend) {
  constexpr uint32_t rank = 8;
  const model::LoraAdapterLayout layout{rank, 2.0f};
  const std::vector<Module> modules{module("0.self_attn.q_proj", "F32", 128, 256, rank),
                                    module("0.self_attn.v_proj", "F16", 64, 256, rank),
                                    module("0.self_attn.o_proj", "BF16", 256, 128, rank),
                                    module("0.mlp.down_proj", "F32", 256, 512, rank),
                                    module("1.mlp.gate_proj", "F32", 512, 256, rank)};
  const auto attached = [&](const std::vector<test::SyntheticTensor> &shard, Layers &layers,
                            model::QwenTargetWeightsBase &weights) {
    const test::TemporaryDirectory directory("splash-lora-adapter");
    test::writeSyntheticShard(directory.path() / "adapter_model.safetensors", shard);
    const std::vector<model::LoraLayerTargets> targets = layers.targets();
    model::attachLoraAdapter(backend, targets, directory.path(), layout, weights);
    return model::loraAdapterBytes(directory.path(), layout);
  };
  Layers layers(backend);
  model::QwenTargetWeightsBase weights;
  weights.manifestFingerprintSha256 = "target";
  const uint64_t bytes = attached(tensors(modules, rank), layers, weights);
  test::require(weights.actualAllocatedBytes >= bytes && weights.actualAllocatedBytes < bytes + (1u << 16),
                "the weights do not count the adapter's allocation");
  test::require(weights.manifestFingerprintSha256 != "target", "the weights' fingerprint does not cover the adapter");
  // The modules' updates: at whole steps of rank, q and v in the packed input's slots, k without one.
  const ProjectionLora &input = layers.input.lora;
  if (input.rank != GGUF_STAGED_STEP || input.slots != 2 || input.segments.size() != 3 || !input.segments[0].b ||
      input.segments[1].b || !input.segments[2].b || input.segments[0].slot != 0 || input.segments[2].slot != 1 ||
      input.a.outputSize != 2 * GGUF_STAGED_STEP || input.a.inputSize != 256)
    fail("the packed input's update is not q's and v's in two slots");
  if (layers.up.lora || layers.gate.lora || layers.up1.lora || layers.down1.lora || !layers.gate1.lora)
    fail("a projection without a module has an update, or one with a module none");
  // Each product B A is the checkpoint's times the scale; B's largest value lies in [0.5, 1), and the rows and
  // columns past the adapter's rank are zero.
  const auto check = [&](const Module &m, const Projection &p, size_t segment) {
    const ProjectionLora &update = p.lora;
    if (!update || !update.segments[segment].b) return fail(m.name + ": has no update");
    const auto *a = static_cast<const uint16_t *>(update.a.plane0.contents()) +
                    uint64_t{update.segments[segment].slot} * update.rank * m.inputSize;
    const auto *b = static_cast<const uint16_t *>(update.segments[segment].b.contents());
    // The values the checkpoint holds, as its dtype rounds them.
    const auto stored = [&](float value) {
      return m.dtype == "F32" ? value
                              : m.dtype == "F16" ? float(static_cast<_Float16>(value)) : bf16ToFloat(floatToBf16(value));
    };
    float peak = 0;
    bool padded = true;
    for (uint32_t n = 0; n < m.outputSize; ++n)
      for (uint32_t j = 0; j < update.rank; ++j) {
        const float value = bf16ToFloat(b[uint64_t{n} * update.rank + j]);
        peak = std::max(peak, std::fabs(value));
        padded = padded && (j < rank || value == 0);
      }
    for (uint32_t j = rank; j < update.rank; ++j)
      for (uint32_t k = 0; k < m.inputSize; ++k) padded = padded && a[uint64_t{j} * m.inputSize + k] == 0;
    if (!(peak >= 0.5f && peak < 1.0f)) fail(m.name + ": B's largest value is " + std::to_string(peak));
    if (!padded) fail(m.name + ": the update past the adapter's rank is not zero");
    double worst = 0;
    for (uint32_t n = 0; n < m.outputSize; n += 37)
      for (uint32_t k = 0; k < m.inputSize; k += 11) {
        double got = 0, want = 0, magnitude = 0;
        for (uint32_t j = 0; j < rank; ++j) {
          got += double(bf16ToFloat(b[uint64_t{n} * update.rank + j])) * bf16ToFloat(a[uint64_t{j} * m.inputSize + k]);
          const double term = double(stored(m.b[uint64_t{n} * rank + j])) * stored(m.a[uint64_t{j} * m.inputSize + k]);
          want += layout.scale * term;
          magnitude += layout.scale * std::fabs(term);
        }
        // Each factor rounds to bf16's eight bits.
        worst = std::max(worst, std::fabs(got - want) / (magnitude + 1e-30));
      }
    if (!(worst <= 1.0 / 64)) fail(m.name + ": B A is off the checkpoint's by " + std::to_string(worst));
  };
  check(modules[0], layers.input, 0);
  check(modules[1], layers.input, 2);
  check(modules[2], layers.output, 0);
  check(modules[3], layers.down, 0);
  check(modules[4], layers.gate1, 0);
  // Another value is another fingerprint.
  {
    std::vector<Module> changed = modules;
    changed[0].b[5] += 1.0f;
    Layers other(backend);
    model::QwenTargetWeightsBase otherWeights;
    otherWeights.manifestFingerprintSha256 = "target";
    static_cast<void>(attached(tensors(changed, rank), other, otherWeights));
    test::require(otherWeights.manifestFingerprintSha256 != weights.manifestFingerprintSha256,
                  "adapters of other values share a fingerprint");
  }
  section("adapter: modules in F32, F16 and BF16 become their projections' updates at the adapter's scale");

  const auto refused = [&](std::vector<test::SyntheticTensor> shard, std::string_view error,
                           const std::string &message) {
    Layers fresh(backend);
    model::QwenTargetWeightsBase unused;
    test::rejects([&] { static_cast<void>(attached(shard, fresh, unused)); }, error, message);
  };
  const std::vector<test::SyntheticTensor> good = tensors(modules, rank);
  {
    std::vector<test::SyntheticTensor> shard = good;
    shard.push_back({"base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_magnitude_vector", "F32",
                     {128}, {}});
    refused(shard, "is not a module's A or B", "a tensor that is no module's A or B was accepted");
    shard.back().name = "base_model.model.model.embed_tokens.lora_A.weight";
    refused(shard, "names no module Splash applies", "a module of no layer was accepted");
    shard.back().name = "base_model.model.model.language_model.layers.0.linear_attn.in_proj.lora_A.weight";
    refused(shard, "names no module Splash applies", "a module of another projection was accepted");
  }
  {
    std::vector<test::SyntheticTensor> shard = good;
    shard.pop_back();
    refused(shard, "has no B: layer 1 mlp.gate_proj", "a module without its B was accepted");
  }
  refused(tensors({module("1.self_attn.q_proj", "F32", 128, 256, rank)}, rank),
          "updates a projection the target does not have: layer 1 self_attn.q_proj",
          "an attention module of a layer without an attention was accepted");
  refused(tensors({module("2.mlp.up_proj", "F32", 512, 256, rank)}, rank),
          "updates a projection the target does not have: layer 2 mlp.up_proj",
          "a module of a layer the target does not have was accepted");
  refused(tensors({module("0.mlp.up_proj", "F32", 256, 256, rank)}, rank), "does not match its projection of 512 x 256",
          "a module of another shape was accepted");
  refused(tensors({module("0.mlp.up_proj", "F32", 512, 256, 16)}, 16), "not a float matrix of the adapter's rank",
          "a module of another rank than the adapter's was accepted");
  {
    std::vector<test::SyntheticTensor> shard = good;
    shard.front().shape = {shard.front().shape[0] * shard.front().shape[1]};
    refused(shard, "not a float matrix of the adapter's rank", "a module that is no matrix was accepted");
  }
  section("adapter: a tensor that is no module's, a module without its B, of another shape or rank, or of a layer "
          "without its projection is refused by name");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: lora-projection <production-and-test.metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const Linear linear(backend.capabilities(), Int8Activations::On);
      updates(backend, linear);
      zeroUpdate(backend, linear);
      swiGlu(backend, linear);
      adapter(backend);
    } catch (const std::exception &e) {
      std::cerr << "lora-projection: FAIL: " << e.what() << '\n';
      return 1;
    }
    std::cout << "lora-projection: " << (failures ? "FAIL (" + std::to_string(failures) + " failures)" : "PASS")
              << '\n';
    return failures ? 1 : 0;
  }
}
