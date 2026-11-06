// GGUF projections: plan validation and dispatch (kernels/shared/gguf_linear.metal,
// kernels/decode/linear_gguf_sgmatrix.metal); KernelSelection.cpp picks the plans.
#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/BufferExtent.hpp"
#include "ops/KernelSelection.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace splash::ops {
namespace {

// The segments one fused decode dispatch runs.
constexpr size_t kFusedSegments = std::extent_v<decltype(GgufDecodeFusedParams::cols)>;

std::string decodeKernel(const char *format, uint32_t rows, char epilogue) {
  return std::string("gguf_decode_") + format + "_m" + std::to_string(rows) + "_" + epilogue;
}
std::string prefillKernel(const char *format, char epilogue) {
  return std::string("gguf_prefill_") + format + "_" + epilogue;
}

// The projection is the plan's matrix, and each of its segments (which tile
// its leading columns) fills whole column tiles of its kernels: 64 columns
// for a quantized segment, whose planes hold its tiles, 8 for a float one
// (addGgufFloat; F32 alpha/beta are 96 columns on the 27B); the columns past
// the last segment are padding no kernel writes. A fused projection keeps the
// layout's sizes: the 35B GGUF's packed GDN row is the layout's 12544
// columns, its qkv|z|alpha-beta segments 12352.
void requireSegments(const Projection &p, LinearMatrix matrix) {
  if (p.outputSize != matrix.outputSize || p.inputSize != matrix.inputSize)
    throw std::invalid_argument("block projection does not match plan");
  for (const QuantizedSegment &s : p.blocks().segments) {
    if (s.outputSize % (s.isFloat() ? 8u : GGUF_TILE_COLUMNS))
      throw std::invalid_argument("block segments do not fill whole column tiles");
    if (!s.isFloat()) requireSegmentPlanes(s, "projection", p.planeInputs());
  }
}

// Columns of the segments, which the fused kernels' grids cover.
uint32_t segmentColumns(const Projection &p) {
  uint32_t columns = 0;
  for (const QuantizedSegment &s : p.blocks().segments) columns += s.outputSize;
  return columns;
}

// The kernel name suffix of an epilogue and the buffer it reads besides the
// input (the output for a plain projection, which reads none): 'r' adds the
// residual, 'g' multiplies silu(gate) from the gate scratch.
char epilogueSuffix(LinearEpilogue epilogue) noexcept {
  return epilogue == LinearEpilogue::None ? 'a' : epilogue == LinearEpilogue::Residual ? 'r' : 'g';
}
const metal::MetalBuffer &epilogueInput(const LinearBuffers &b, LinearEpilogue epilogue) noexcept {
  return epilogue == LinearEpilogue::Residual ? b.residual
       : epilogue == LinearEpilogue::None     ? b.output
                                              : b.gateScratch;
}

// The parameters and planes of a fused decode dispatch, which runs every
// segment of a fused projection (qkv|z|ab, q|k|v) in one dispatch, over
// `order`, the segments in dispatch order. Slots past the last segment take
// no columns and bind its planes again.
GgufDecodeFusedParams fusedSegments(const LinearPlan &plan, std::span<const QuantizedSegment *const> order,
                                    std::vector<metal::MetalBuffer> &bindings) {
  const LinearWorkload w = plan.workload();
  if (order.size() > kFusedSegments || w.epilogue != LinearEpilogue::None)
    throw std::invalid_argument("a fused block projection takes at most three segments and no epilogue");
  const LinearConfig config = plan.configuration();
  GgufDecodeFusedParams params{w.matrix.inputSize, config.splits, w.matrix.outputSize,
                               {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, config.spread};
  for (size_t i = 0; i < kFusedSegments; ++i) {
    const QuantizedSegment &s = *order[std::min(i, order.size() - 1)];
    if (i < order.size()) {
      params.cols[i] = s.outputSize;
      params.fmt[i] = s.formatId;
      params.offset[i] = s.columnOffset;
    }
    bindings.insert(bindings.end(), {s.plane0, s.plane1Slot(), s.meta});
  }
  return params;
}

// A single tensor's decode dispatches through `tensor(segment, epilogue
// suffix, output, epilogue input)`: gate/up as a gate pass into the gate
// scratch and an up pass whose epilogue applies silu(gate) to the bf16 up
// value, any other epilogue in one dispatch.
template <class Tensor>
void addDecodeTensor(const LinearBuffers &b, LinearEpilogue epilogue, const QuantizedSegment &segment,
                     const Projection *gate, const Tensor &tensor) {
  if (epilogue != LinearEpilogue::GateUp) {
    tensor(segment, epilogueSuffix(epilogue), b.output, epilogueInput(b, epilogue));
    return;
  }
  const std::vector<QuantizedSegment> &gates = gate->blocks().segments;
  if (gates.size() != 1) throw std::invalid_argument("block gate/up takes single tensors");
  tensor(gates.front(), epilogueSuffix(LinearEpilogue::None), b.gateScratch, b.gateScratch);
  tensor(segment, epilogueSuffix(LinearEpilogue::UpWithGate), b.output, b.gateScratch);
}

} // namespace

void LinearPlan::requireConfiguration() const {
  const uint32_t k = workload_.matrix.inputSize;
  const LinearConfig &c = config_;
  const bool decode = workload_.phase == LinearPhase::Decode;
  if (c.spread && c.tile != LinearTile::GgufStaged)
    throw std::invalid_argument("only the staged block tile spreads its walks");
  if (c.oneGateUpPass && (c.tile != LinearTile::GgufStaged || !decode || workload_.epilogue != LinearEpilogue::GateUp))
    throw std::invalid_argument("only a staged block gate/up decode runs gate and up in one pass");
  if (c.sixteenRows && (c.tile != LinearTile::GgufStaged || !decode || workload_.rows != SPLASH_TARGET_VERIFY_ROWS))
    throw std::invalid_argument("only a staged one-lane block decode runs the 16-row tile");
  if (c.tile == LinearTile::GgufRegister) {
    // Split boundaries fall on coefficient units, of 64 inputs at the fewest
    // (quant_register_unit); addGgufRegister checks each segment's.
    if (!decode || !c.validSplits() || k / 64 < c.splits)
      throw std::invalid_argument("the register block decode tile takes a K unit per split");
    return;
  }
  if (c.tile == LinearTile::GgufPrefill) {
    if (decode || c.splits != 1) throw std::invalid_argument("invalid block prefill configuration");
    return;
  }
  if (!c.validSplits() || (k / 32) % c.splits)
    throw std::invalid_argument("staged block splits take whole 32-input groups");
  // Prefill chunks of up to a decode batch run the staged tile and split K as
  // in decode; longer ones, up to kMaximumStagedPrefillRows, its 64-row tile.
  if (!decode && workload_.rows > kMaximumStagedPrefillRows)
    throw std::invalid_argument("the staged block tile runs prefill chunks of up to 64 rows");
}

uint32_t LinearPlan::storageRows() const noexcept {
  if (config_.tile == LinearTile::GgufRegister) return workload_.rows;
  if (config_.tile == LinearTile::GgufPrefill)
    return (workload_.rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS;
  return config_.sixteenRows ? 16 : stagedTileRows(workload_.rows);
}

LinearScratchSize LinearPlan::scratchSize() const noexcept {
  const auto [n, k] = workload_.matrix;
  // Register tile: the Table16 table and sums, and split, [lane][split][row]
  // [column] partials and one counter per 64-column tile, which covers every
  // lane.
  if (config_.tile == LinearTile::GgufRegister) {
    const uint64_t rows = workload_.rows;
    return {tableBytes(k, rows), tableSumsBytes(LinearInput::Table16, k, rows),
            config_.splits > 1 ? config_.splits * rows * n * sizeof(float) : 0,
            config_.splits > 1 ? uint64_t{n / GGUF_TILE_COLUMNS} * sizeof(uint32_t) : 0};
  }
  // Staged split-K: [split][row][column] fp32 partials over the tile's rows,
  // a one-pass gate/up's of both tensors, and one counter per column tile (a
  // tile covers every row of the dispatch): 64 columns, 32 in one pass. The
  // prefill tile never splits.
  const uint64_t columns = config_.oneGateUpPass ? 2 * uint64_t{n} : n;
  const uint32_t tile = config_.oneGateUpPass ? GGUF_STAGED_COLUMNS : GGUF_TILE_COLUMNS;
  return config_.splits > 1
      ? LinearScratchSize{0, 0, uint64_t{config_.splits} * storageRows() * columns * sizeof(float),
                          uint64_t{n / tile} * sizeof(uint32_t)}
      : LinearScratchSize{};
}

LinearScratchSize Linear::ggufDecodeScratchSize(LinearWorkload w, bool registerFormats) const {
  // Every format the register tile takes K inputs in has one register plan,
  // whose partitions keep whole coefficient units of each (kRegisterTiers);
  // a projection of segments the tile does not take has none. One lane on the
  // 16-row tile runs unsplit, without partials.
  LinearScratchSize size;
  for (const LinearConfig &config : decodeConfigurations(calibration_, w, registerFormats))
    size.include(LinearPlan(w, config).scratchSize());
  return size;
}

void Linear::addGguf(metal::CommandGraph &graph, const LinearBuffers &b,
                       const Projection &p, const LinearPlan &plan,
                       const Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const uint32_t k = w.matrix.inputSize;
  requireSegments(p, w.matrix);
  if (gate) requireSegments(*gate, w.matrix);
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  // The fused kernels have no fp32 instance.
  if (plan.destination() == FloatOutput::Float32 && segments.size() > 1)
    throw std::invalid_argument("an fp32 destination takes a single-tensor block projection");
  if (std::any_of(segments.begin(), segments.end(), [](const QuantizedSegment &s) { return s.isFloat(); })) {
    // The quantized segments, which precede the float ones, run the plan's tiles.
    addGgufFloatSegments(graph, b, p, plan);
    BlockWeights weights{segments};
    std::erase_if(weights.segments, [](const QuantizedSegment &s) { return s.isFloat(); });
    if (!weights.segments.empty()) {
      Projection quantized(p.outputSize, p.inputSize, std::move(weights));
      quantized.rotation = p.rotation;
      addGguf(graph, b, quantized, plan, gate);
    }
    return;
  }
  if (p.rotation) {
    // Weights stored for rotated inputs (InputRotation): the quantized
    // segments, and a gate/up pair's gate too, read H (D x) from the scratch,
    // rotated once. Rows past the workload's are padding the tiles discard.
    if (gate && !gate->rotation.signs.sameView(p.rotation.signs))
      throw std::invalid_argument("a rotated gate/up pair takes one rotation");
    if (k % GGUF_ROTATION_BLOCK)
      throw std::invalid_argument("a rotated projection takes whole rotation blocks");
    requireBytes(p.rotation.signs, k, "rotation sign");
    graph.add("gguf_rotate", {b.input, p.rotation.signs, b.scratch.rotated}, GgufRotationParams{k},
              {k / GGUF_ROTATION_BLOCK, w.rows, 1}, {GGUF_ROTATION_THREADS, 1, 1});
    LinearBuffers rotated = b;
    rotated.input = b.scratch.rotated;
    rotated.prepared = {};
    Projection plain = p;
    plain.rotation = {};
    addGguf(graph, rotated, plain, plan, gate);
    return;
  }
  switch (plan.configuration().tile) {
  case LinearTile::GgufRegister: addGgufRegister(graph, b, p, plan, gate); break;
  case LinearTile::GgufStaged: addGgufStaged(graph, b, p, plan, gate); break;
  case LinearTile::GgufPrefill: addGgufPrefill(graph, b, p, plan); break;
  }
}

// The staged decode tiles, for decode and prefill chunks of up to
// ProjectionCalibration::stagedPrefillRows rows: every row of the plan's
// storage in each threadgroup's tile, grid (64-column tiles, K splits). A
// projection runs one dispatch, a fused one's segments together
// (fusedSegments), in decode as in prefill; decode runs gate/up as two passes,
// or, where selectProjection sets LinearConfig::oneGateUpPass, as one pass over
// 32-column threadgroups of one simdgroup each, and a prefill chunk with an
// epilogue runs one dispatch per segment.
void Linear::addGgufStaged(metal::CommandGraph &graph, const LinearBuffers &b,
                             const Projection &p, const LinearPlan &plan,
                             const Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  const uint32_t rows = plan.storageRows(), splits = config.splits;
  const auto [partials, counters] = splitScratch(b, splits);
  const auto tensor = [&](const QuantizedSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(kernelInstance(decodeKernel(s.name(), rows, epilogue), plan.destination()),
              {b.input, s.plane0, s.plane1Slot(), s.meta, output, partials, counters, aux},
              GgufDecodeParams{k, splits, n, s.columnOffset, config.spread},
              {s.outputSize / GGUF_TILE_COLUMNS, splits, 1}, {GGUF_STAGED_THREADS, 1, 1});
  };
  // A fused projection's segments in one dispatch (gguf_decode_fused_m*),
  // in tile order: segments with the most bytes per tile first, so their
  // threadgroups do not form the tail.
  const auto fused = [&] {
    std::vector<const QuantizedSegment *> order;
    for (const QuantizedSegment &s : segments) order.push_back(&s);
    const auto bitsPerWeight = [](const QuantizedSegment &s) {
      const QuantFormat &f = s.format();
      return (f.plane0_bytes + f.plane1_bytes) * 8.0 / 32.0 + f.meta_bytes * 8.0 / (32.0 * f.meta_groups);
    };
    std::stable_sort(order.begin(), order.end(), [&](const QuantizedSegment *a, const QuantizedSegment *c) {
      return bitsPerWeight(*a) > bitsPerWeight(*c);
    });
    std::vector<metal::MetalBuffer> bindings{b.input};
    const GgufDecodeFusedParams params = fusedSegments(plan, order, bindings);
    bindings.insert(bindings.end(), {b.output, partials, counters});
    graph.add("gguf_decode_fused_m" + std::to_string(rows), std::move(bindings), params,
              {segmentColumns(p) / GGUF_TILE_COLUMNS, splits, 1}, {GGUF_STAGED_THREADS, 1, 1});
  };
  if (w.phase == LinearPhase::Prefill) {
    // A fused projection's chunk runs its segments in one dispatch, as decode
    // does: a dispatch costs about 0.045 ms however narrow, and at 8 to 32
    // rows the 27B's and 35B's fused inputs take 16-56% less time in one
    // dispatch than in one per segment (32-core M5 Max).
    if (segments.size() > 1 && w.epilogue == LinearEpilogue::None) {
      fused();
      return;
    }
    for (const QuantizedSegment &s : segments)
      tensor(s, epilogueSuffix(w.epilogue), b.output, epilogueInput(b, w.epilogue));
    return;
  }
  if (config.oneGateUpPass) {
    const std::vector<QuantizedSegment> &gates = gate->blocks().segments;
    if (segments.size() != 1 || gates.size() != 1 || gates.front().formatId != segments.front().formatId)
      throw std::invalid_argument("a one-pass block gate/up takes single tensors of one format");
    const QuantizedSegment &g = gates.front(), &u = segments.front();
    graph.add(std::string("gguf_decode_") + u.name() + "_m" + std::to_string(rows) + "_gate_up",
              {b.input, g.plane0, g.plane1Slot(), g.meta, u.plane0, u.plane1Slot(), u.meta, b.output, partials,
               counters},
              GgufDecodeParams{k, splits, n, u.columnOffset, config.spread},
              {u.outputSize / GGUF_STAGED_COLUMNS, splits, 1}, {GGUF_GATE_UP_THREADS, 1, 1});
    return;
  }
  if (segments.size() == 1) {
    addDecodeTensor(b, w.epilogue, segments.front(), gate, tensor);
    return;
  }
  fused();
}

// The prefill tile, which runs prefill chunks of more than
// ProjectionCalibration::stagedPrefillRows rows over 128-row tiles: a plain projection in one
// dispatch per format, over the column tiles of its segments of that format
// (a fused projection's segments otherwise run as small grids of their own),
// a projection with an epilogue in one dispatch per segment. Rows past the
// chunk's stay inside the budget-sized prefill buffers, and the simdgroups of
// a tile that only hold them skip their matmuls.
void Linear::addGgufPrefill(metal::CommandGraph &graph, const LinearBuffers &b,
                            const Projection &p, const LinearPlan &plan) const {
  const LinearWorkload w = plan.workload();
  const auto [n, k] = w.matrix;
  const char epilogue = epilogueSuffix(w.epilogue);
  const uint32_t rowTiles = plan.storageRows() / GGUF_PREFILL_ROWS;
  if (w.epilogue == LinearEpilogue::None && !p.planeInputs()) {
    const std::vector<QuantizedSegment> &segments = p.blocks().segments;
    std::vector<uint32_t> formats;
    for (const QuantizedSegment &s : segments)
      if (std::find(formats.begin(), formats.end(), s.formatId) == formats.end()) formats.push_back(s.formatId);
    for (const uint32_t format : formats) {
      std::vector<const QuantizedSegment *> group;
      for (const QuantizedSegment &s : segments)
        if (s.formatId == format) group.push_back(&s);
      if (group.size() > kFusedSegments) throw std::invalid_argument("a block prefill takes at most three segments");
      GgufPrefillSegmentsParams params{k, w.rows, n, {0, 0, 0}, {0, 0, 0}};
      std::vector<metal::MetalBuffer> bindings{b.input};
      uint32_t columns = 0;
      for (size_t i = 0; i < kFusedSegments; ++i) {
        // Slots past the group's last segment bind its planes, which no tile reads.
        const QuantizedSegment &s = *group[std::min(i, group.size() - 1)];
        if (i < group.size()) {
          params.cols[i] = s.outputSize;
          params.offset[i] = s.columnOffset;
          columns += s.outputSize;
        }
        bindings.insert(bindings.end(), {s.plane0, s.plane1Slot(), s.meta});
      }
      bindings.push_back(b.output);
      graph.add(prefillKernel(group.front()->name(), epilogue), std::move(bindings), params,
                {rowTiles, columns / GGUF_TILE_COLUMNS, 1}, {GGUF_PREFILL_THREADS, 1, 1});
    }
    return;
  }
  for (const QuantizedSegment &s : p.blocks().segments) {
    std::vector<metal::MetalBuffer> bindings{b.input, s.plane0, s.plane1Slot(), s.meta, b.output};
    if (w.epilogue != LinearEpilogue::None) bindings.push_back(epilogueInput(b, w.epilogue));
    const GgufPrefillParams params{k, w.rows, n, s.columnOffset};
    const metal::DispatchSize groups{rowTiles, s.outputSize / GGUF_TILE_COLUMNS, 1};
    if (p.planeInputs())
      graph.add(leadingInputsInstance(prefillKernel(s.name(), epilogue)), std::move(bindings),
                GgufPrefillLeadingParams{params, p.planeInputs()}, groups, {GGUF_PREFILL_THREADS, 1, 1});
    else
      graph.add(prefillKernel(s.name(), epilogue), std::move(bindings), params, groups, {GGUF_PREFILL_THREADS, 1, 1});
  }
}

// All lanes in each threadgroup, decode only: single tensors and gate/up
// through addDecodeTensor, fused projections through fusedSegments.
void Linear::addGgufRegister(metal::CommandGraph &graph, const LinearBuffers &b,
                             const Projection &p, const LinearPlan &plan,
                             const Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  for (const Projection *q : {&p, gate})
    if (q)
      for (const QuantizedSegment &s : q->blocks().segments) {
        if (!quant_register_tensor(s.formatId, s.inputSize))
          throw std::invalid_argument(std::string("the register tile does not take ") + s.name() + " weights of " +
                                      std::to_string(s.inputSize) + " inputs");
        // Each K partition takes whole coefficient units of the segment's format.
        if (s.inputSize / quant_register_unit(s.formatId) < config.splits)
          throw std::invalid_argument(std::string("the register tile takes a coefficient unit per split: ") +
                                      std::to_string(config.splits) + " splits of " + s.name() + " weights of " +
                                      std::to_string(s.inputSize) + " inputs");
      }
  if (b.prepared.layout != LinearInput::Table16 || !b.prepared.source.sameView(b.input))
    graph.add("decode_linear_gguf_prepare", {b.input, b.scratch.input, b.scratch.sums}, k,
              {k / 32, lanes, 1}, {128, 1, 1});
  const metal::DispatchSize grid{segmentColumns(p) / GGUF_TILE_COLUMNS, config.splits, 1};
  const std::string suffix = "_l" + std::to_string(lanes);
  const auto [partials, counters] = splitScratch(b, config.splits);
  if (segments.size() > 1) {
    std::vector<const QuantizedSegment *> order;
    for (const QuantizedSegment &s : segments) order.push_back(&s);
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, b.scratch.sums};
    const GgufDecodeFusedParams params = fusedSegments(plan, order, bindings);
    bindings.insert(bindings.end(), {b.output, partials, counters});
    graph.add("gguf_decode_sg_fused" + suffix, std::move(bindings), params, grid, {GGUF_REGISTER_THREADS, 1, 1});
    return;
  }
  const auto tensor = [&](const QuantizedSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(kernelInstance(std::string("gguf_decode_sg_") + s.name() + suffix + "_" + epilogue, plan.destination()),
              {b.scratch.input, b.scratch.sums, s.plane0, s.plane1Slot(), s.meta, output, partials, counters, aux},
              GgufDecodeParams{k, config.splits, n, s.columnOffset, 0}, grid, {GGUF_REGISTER_THREADS, 1, 1});
  };
  addDecodeTensor(b, w.epilogue, segments.front(), gate, tensor);
}

// Float segments of a projection (QuantizedSegment::isFloat): a float projection
// over the step's rows, whatever the rows of the plan's tiles.
void Linear::addGgufFloatSegments(metal::CommandGraph &graph, const LinearBuffers &b,
                                    const Projection &p, const LinearPlan &plan) const {
  const LinearWorkload w = plan.workload();
  if (w.epilogue != LinearEpilogue::None) throw std::invalid_argument("float segments take no epilogue");
  for (const QuantizedSegment &s : p.blocks().segments)
    if (s.isFloat())
      addGgufFloat(graph, b.input, s, b.output, w.rows, w.matrix.outputSize, s.columnOffset, plan.destination(),
                   ggufFloatTile(w.rows, s.outputSize, s.formatId));
}

FloatTile Linear::ggufFloatTile(uint32_t rows, uint32_t outputSize, uint32_t formatId) const noexcept {
  return selectFloatTile(calibration_, rows, outputSize, formatId);
}

void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, const QuantizedSegment &weights,
                  metal::MetalBuffer output, uint32_t rows, uint32_t outStride, uint32_t outOffset,
                  FloatOutput type, FloatTile tile) {
  const FloatProjection projection{&weights, std::move(output), outStride, outOffset, tile};
  addGgufFloat(graph, std::move(input), {&projection, 1}, rows, type);
}

// Simdgroup: 8 columns of 32 rows per threadgroup of 8 simdgroups. Neural
// accelerator: 32 columns of 64 rows per threadgroup of 4 simdgroups, at least
// 16 rows and K a multiple of 32 (kernels/shared/gguf_float.metal). Every
// projection is checked before any dispatch is added; a dispatch's slots past
// its last projection bind that one's buffers again and take no columns.
void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, std::span<const FloatProjection> projections,
                  uint32_t rows, FloatOutput type) {
  const uint32_t k = projections.empty() ? 0 : projections.front().weights->inputSize;
  if (!rows || !k || k % 8) throw std::invalid_argument("invalid float projection");
  for (const FloatProjection &p : projections) {
    const QuantizedSegment &w = *p.weights;
    const uint32_t n = w.outputSize;
    if (!w.isFloat() || w.inputSize != k || !n || n % 8 || p.outOffset + uint64_t{n} > p.outStride ||
        (p.tile == FloatTile::NeuralAccelerator && (rows < 16 || k % 32)))
      throw std::invalid_argument("invalid float projection");
    // The rows of a quantized format write fp32, the MoE router's scores.
    if (w.quantizedRows() && type != FloatOutput::Float32)
      throw std::invalid_argument("a float projection of quantized rows writes fp32");
    requireBytes(w.plane0, uint64_t{n} * QuantizedSegment::rowBytes(w.formatId, k), "float projection weight");
    requireBytes(p.output, rowBytes(rows, p.outStride, uint64_t{p.outOffset} + n, elementBytes(type)),
                 "float projection output");
  }
  requireBytes(input, uint64_t{rows} * k * 2, "float projection input");
  for (size_t first = 0; first < projections.size();) {
    const FloatProjection &lead = projections[first];
    size_t count = 1;
    while (first + count < projections.size() && count < GGUF_FLOAT_PROJECTIONS &&
           projections[first + count].weights->formatId == lead.weights->formatId &&
           projections[first + count].tile == lead.tile)
      ++count;
    const bool accelerator = lead.tile == FloatTile::NeuralAccelerator;
    const uint32_t columns = accelerator ? 32 : 8, tileRows = accelerator ? 64 : 32;
    GgufFloatParams params{rows, k, {}, {}, {}};
    std::vector<metal::MetalBuffer> bindings{input};
    uint32_t tiles = 0;
    for (uint32_t slot = 0; slot < GGUF_FLOAT_PROJECTIONS; ++slot) {
      const FloatProjection &p = projections[first + std::min<size_t>(slot, count - 1)];
      bindings.insert(bindings.end(), {p.weights->plane0, p.output});
      if (slot >= count) continue;
      params.output_size[slot] = p.weights->outputSize;
      params.out_stride[slot] = p.outStride;
      params.out_offset[slot] = p.outOffset;
      tiles += (p.weights->outputSize + columns - 1) / columns;
    }
    graph.add(std::string(accelerator ? "gguf_float_na_" : "gguf_float_") + lead.weights->name() + "_" +
                  (type == FloatOutput::Float32 ? "f32" : "bf16"),
              std::move(bindings), params, {tiles, (rows + tileRows - 1) / tileRows, 1},
              {accelerator ? 128u : 256u, 1, 1});
    first += count;
  }
}

void requireSegmentPlanes(const QuantizedSegment &segment, std::string_view what, uint32_t planeInputs) {
  // Each plane ends at its unit of the last row's last group the segment
  // reads, or meta unit: tile row (rows - 1) % T of that group of its tile,
  // whose rows hold the groups of planeInputs inputs.
  if (segment.layout != QuantizedSegment::Layout::Planes)
    throw std::invalid_argument(std::string(what) + " takes planes");
  const QuantFormat &format = segment.format();
  const uint32_t groups = segment.inputSize / 32, units = groups / format.meta_groups;
  const uint32_t rowGroups = (planeInputs ? planeInputs : segment.inputSize) / 32,
                 rowUnits = rowGroups / format.meta_groups;
  const auto planeBytes = [&](uint32_t blocks, uint32_t rowBlocks, uint32_t unitBytes) {
    return (quant_tile_index(segment.outputSize - 1, blocks - 1, rowBlocks) + 1) * unitBytes;
  };
  const std::string name(what);
  requireBytes(segment.plane0, planeBytes(groups, rowGroups, format.plane0_bytes), name + " plane0");
  if (format.plane1_bytes)
    requireBytes(segment.plane1, planeBytes(groups, rowGroups, format.plane1_bytes), name + " plane1");
  requireBytes(segment.meta, planeBytes(units, rowUnits, format.meta_bytes), name + " meta");
}

QuantizedSegment QuantizedSegment::planes(uint32_t formatId, uint32_t outputSize, uint32_t inputSize,
                                          metal::MetalBuffer plane0, metal::MetalBuffer plane1,
                                          metal::MetalBuffer meta) {
  if (formatId >= GGUF_FMT_COUNT) throw std::invalid_argument("unknown GGUF segment format");
  return {std::move(plane0), std::move(plane1), std::move(meta), outputSize, inputSize, 0, formatId, Layout::Planes};
}
QuantizedSegment QuantizedSegment::rows(uint32_t formatId, uint32_t outputSize, uint32_t inputSize,
                                        metal::MetalBuffer rows) {
  if (!rowBytes(formatId, inputSize))
    throw std::invalid_argument("rows take F32 or BF16 values or whole native blocks of a quantized GGUF format");
  return {std::move(rows), {}, {}, outputSize, inputSize, 0, formatId, Layout::Rows};
}
// F32 values each their own, any other format's rows as stored; none of
// F16, which the float projection does not run.
uint64_t QuantizedSegment::rowBytes(uint32_t formatId, uint32_t inputSize) noexcept {
  if (formatId == kFloat32) return uint64_t{inputSize} * sizeof(float);
  return formatId == GGUF_FMT_F16 ? 0 : quant_stored_row_bytes(formatId, inputSize);
}
const QuantFormat &QuantizedSegment::format() const noexcept { return kQuantFormats[formatId]; }
const char *QuantizedSegment::name() const noexcept { return formatId == kFloat32 ? "f32" : format().name; }

} // namespace splash::ops
