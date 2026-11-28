#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/BufferExtent.hpp"
#include "ops/KernelSelection.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

// Matrices of whole 64-row tiles (GGUF_TILE_COLUMNS) of whole 64-input spans,
// whose K is a multiple of a 64-input span (or of its format's superblock), and
// up to a prefill chunk's rows. A gate/up pair runs as one plan on the decode
// tiles alone (LinearEpilogue).
void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 64 || !w.matrix.inputSize || w.matrix.inputSize % 64)
    throw std::invalid_argument("invalid linear matrix");
  if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
      (w.epilogue == LinearEpilogue::GateUp && w.rows > kMaximumDecodeTileRows))
    throw std::invalid_argument("invalid linear workload");
}

// LinearScratch::rotated bytes of `rows` bf16 rows of `width` inputs, which
// also hold the int8 rows' byte per input and fp32 scale per rotation block
// (gguf_rotate_int8).
constexpr uint64_t rotatedBytes(uint32_t width, uint64_t rows) noexcept { return uint64_t{width} * rows * 2; }

// Throws if `p` is a view of the leading inputs of wider weight rows
// (Projection::leadingInputs) that `plan` has no kernel instance for: only the
// prefill residual tiles of quantized segments (GgufPrefill's
// leadingInputsInstance, GgufPrefillInt8's plane stride), unrotated, read one.
void requireLeadingInputs(const LinearPlan &plan, const Projection &p) {
  if (!p.planeInputs()) return;
  const LinearWorkload w = plan.workload();
  const LinearTile tile = plan.configuration().tile;
  if (w.epilogue != LinearEpilogue::Residual || (tile != LinearTile::GgufPrefill && tile != LinearTile::GgufPrefillInt8) ||
      p.rotation)
    throw std::invalid_argument("a view of leading inputs runs only the quantized prefill residual tiles");
}

// Throws unless a view of `p`'s leading inputs can stand for it (takesPlaneViews).
void requirePlaneViews(const Projection &p) {
  if (p.takesPlaneViews()) return;
  throw std::invalid_argument("a view of leading inputs takes one unrotated quantized tensor, not a view");
}

// The workload a plan of `w` is selected for: that of one of the tensors of a
// projection side by side (ProjectionShape::planOutputs), else its own.
LinearWorkload selectedWorkload(LinearWorkload w, uint32_t planOutputs) noexcept {
  if (planOutputs) w.matrix.outputSize = planOutputs;
  return w;
}

} // namespace

std::optional<Projection> Projection::sideBySide(std::span<const Projection *const> projections) {
  // The fused dispatches run up to three segments (GgufDecodeFusedParams).
  constexpr size_t kMostTensors = std::extent_v<decltype(GgufDecodeFusedParams::cols)>;
  if (projections.size() < 2 || projections.size() > kMostTensors)
    throw std::invalid_argument("projections side by side are two or three");
  if (std::any_of(projections.begin(), projections.end(),
                  [](const Projection *p) { return p->blocks().segments.size() != 1; }))
    return std::nullopt;
  const Projection &first = *projections.front();
  const QuantizedSegment &lead = first.blocks().segments.front();
  BlockWeights weights;
  uint32_t columns = 0;
  for (const Projection *p : projections) {
    const QuantizedSegment &tensor = p->blocks().segments.front();
    if (p->outputSize != first.outputSize || p->inputSize != first.inputSize || tensor.outputSize != p->outputSize ||
        tensor.formatId != lead.formatId || tensor.layout != lead.layout || p->destination != first.destination ||
        p->rotation || p->planeInputs_ || p->planOutputs_)
      return std::nullopt;
    QuantizedSegment segment = tensor;
    segment.columnOffset = columns;
    columns += segment.outputSize;
    weights.segments.push_back(std::move(segment));
  }
  Projection result(columns, first.inputSize, std::move(weights));
  result.destination = first.destination;
  result.planOutputs_ = first.outputSize;
  return result;
}

bool Projection::takesPlaneViews() const noexcept {
  const std::vector<QuantizedSegment> &segments = weights_.segments;
  return !planeInputs_ && segments.size() == 1 && !segments.front().isFloat() && !rotation;
}

// Every plane holds its rows in tiles of QUANT_TILE_ROWS rows, each tile's
// units in order, and float rows lie in row order, so the rows of whole tiles
// are one range of each plane.
Projection Projection::rows(const metal::MetalBackend &backend, std::initializer_list<RowRange> ranges) const {
  if (weights_.segments.size() != 1 || planeInputs_)
    throw std::invalid_argument("a view of rows takes a projection of one tensor, not a view of its inputs");
  const QuantizedSegment &tensor = weights_.segments.front();
  // Rows `range` of a plane of `rowBytes` bytes a row.
  const auto view = [&](const metal::MetalBuffer &plane, uint64_t rowBytes, RowRange range) {
    return rowBytes ? backend.view(plane, range.first * rowBytes, range.count * rowBytes) : metal::MetalBuffer{};
  };
  BlockWeights weights;
  uint32_t columns = 0;
  for (const RowRange range : ranges) {
    if (range.first % QUANT_TILE_ROWS || range.count % QUANT_TILE_ROWS ||
        uint64_t{range.first} + range.count > tensor.outputSize)
      throw std::invalid_argument("a view of rows takes whole QUANT_TILE_ROWS tiles of the tensor's rows");
    if (!range.count) continue;
    QuantizedSegment segment;
    if (tensor.isFloat()) {
      segment = QuantizedSegment::rows(tensor.formatId, range.count, inputSize,
                                       view(tensor.plane0, QuantizedSegment::rowBytes(tensor.formatId, inputSize),
                                            range));
    } else {
      const QuantFormat &format = tensor.format();
      const uint64_t groups = inputSize / 32;
      segment = QuantizedSegment::planes(tensor.formatId, range.count, inputSize,
                                         view(tensor.plane0, groups * format.plane0_bytes, range),
                                         view(tensor.plane1, groups * format.plane1_bytes, range),
                                         view(tensor.meta, groups / format.meta_groups * format.meta_bytes, range));
    }
    segment.columnOffset = columns;
    columns += range.count;
    weights.segments.push_back(std::move(segment));
  }
  Projection result(columns, inputSize, std::move(weights));
  result.destination = destination;
  result.rotation = rotation;
  return result;
}

Projection Projection::leadingInputs(uint32_t inputs) const {
  requirePlaneViews(*this);
  const QuantizedSegment &segment = weights_.segments.front();
  if (!inputs || inputs > inputSize || inputs % (32 * segment.format().meta_groups))
    throw std::invalid_argument("a view of leading inputs takes whole quant groups and meta units of the projection's");
  Projection result(outputSize, inputs,
                    BlockWeights{{QuantizedSegment::planes(segment.formatId, outputSize, inputs, segment.plane0,
                                                           segment.plane1, segment.meta)}});
  result.planeInputs_ = inputSize;
  return result;
}

// Table16 holds its sums per table tile (metal/abi/Gguf.h).
uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept {
  return layout == LinearInput::Table16 ? tableTiles(static_cast<uint32_t>(rows)) * table16_sums_per_tile(width) *
                                              sizeof(float)
                                        : 0;
}

void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows) {
  if (layout == LinearInput::Plain || !rows || width % 64)
    throw std::invalid_argument("invalid linear table geometry");
  requireBytes(scratch.input, tableBytes(width, uint64_t{tableTiles(rows)} * GGUF_TABLE16_ROWS), "linear table");
  requireBytes(scratch.sums, tableSumsBytes(layout, width, rows), "linear table sums");
}

const char *tableSuffix(LinearInput layout) noexcept {
  return layout == LinearInput::Table16 ? "_table16" : "";
}

LinearInput LinearPlan::input() const noexcept {
  if (rotated_) return LinearInput::Plain;
  return config_.tile == LinearTile::GgufRegister ? LinearInput::Table16 : LinearInput::Plain;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  // Gate/up runs as a gate pass into the gate scratch and an up-with-gate
  // pass, but in one pass (LinearConfig::oneGateUpPass).
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
                      (workload_.epilogue == LinearEpilogue::GateUp && !config_.oneGateUpPass);
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config, FloatOutput destination)
    : workload_(w), config_(config), destination_(destination) {
  validate(w);
  // Only the decode tiles have fp32 instances, of their plain projection.
  if (destination == FloatOutput::Float32 &&
      (config.tile == LinearTile::GgufPrefill || config.tile == LinearTile::GgufPrefillInt8 ||
       w.epilogue != LinearEpilogue::None))
    throw std::invalid_argument("an fp32 destination takes a plain projection on the decode tiles");
  // Kernel names follow the segment formats (LinearGguf.cpp).
  requireConfiguration();
}

Linear::Linear(const DeviceCapabilities &device, Int8Activations int8Activations, StepKind step,
               std::shared_ptr<Int8PathsTaken> taken, bool splitFree) noexcept
    : calibration_(KernelCalibration::of(device)), int8Activations_(int8Activations), step_(step),
      taken_(std::move(taken)), splitFree_(splitFree) {}

uint32_t Linear::storageRows(uint32_t rows, ProjectionShape shape) const {
  // The None and Residual plans share a tile; a gate/up pair of one format
  // counts both tensors' columns in one pass and may stage where they do not.
  // Past the decode tiles a pair runs as its tensors' own plans.
  UnitFormats unit = unitFormats(shape);
  uint32_t storage = 0;
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::GateUp}) {
    if (epilogue == LinearEpilogue::GateUp && rows > kMaximumDecodeTileRows) continue;
    const LinearWorkload w{{shape.outputSize, shape.inputSize}, rows, epilogue};
    validate(w);
    unit.oneFormat = epilogue == LinearEpilogue::GateUp;
    const LinearWorkload selected = selectedWorkload(w, shape.planOutputs);
    storage = std::max(storage,
                       LinearPlan(w, selectProjection(calibration_, int8Activations_, step_, selected, unit)).storageRows());
  }
  return storage;
}

LinearPlan Linear::plan(LinearWorkload workload) const {
  workload.splitFree = workload.splitFree || splitFree_;
  validate(workload);
  return LinearPlan(workload, selectProjection(calibration_, int8Activations_, step_, workload, {}));
}
LinearPlan Linear::plan(LinearWorkload workload, LinearConfig config, FloatOutput destination) {
  return LinearPlan(workload, config, destination);
}
LinearPlan Linear::plan(LinearWorkload w, const Projection &p, const Projection *gate) const {
  w.splitFree = w.splitFree || splitFree_;
  validate(w);
  const LinearWorkload selected = selectedWorkload(w, p.planOutputs());
  validate(selected);
  const std::array<const Projection *, 2> projections{&p, gate};
  // A gate/up pair's 16-row tile follows its tensors' own formats.
  std::array<UnitFormats, 2> tensors{};
  const bool pair = w.epilogue == LinearEpilogue::GateUp && gate;
  if (pair) tensors = {unitFormats(std::span(&projections[0], 1)), unitFormats(std::span(&projections[1], 1))};
  LinearPlan plan(w,
                  selectProjection(calibration_, int8Activations_, step_, selected, unitFormats(projections),
                                   pair ? std::span<const UnitFormats>(tensors) : std::span<const UnitFormats>{}),
                  p.destination);
  plan.rotated_ = static_cast<bool>(p.rotation);
  if (const LinearConfig &config = plan.configuration(); taken_ && config.tile == LinearTile::GgufPrefillInt8)
    Int8PathsTaken::mark(taken_->prefillTile);
  else if (taken_ && config.decodeInt8)
    Int8PathsTaken::mark(step_ == StepKind::Decode ? taken_->decodeTiles : taken_->prefillDecodeTiles);
  return plan;
}

LinearPlan Linear::plan(const Projection &p, uint32_t rows, LinearEpilogue epilogue, const Projection *gate,
                        uint32_t chunkRows) const {
  const LinearWorkload w{{p.outputSize, p.inputSize}, rows, epilogue};
  if (!chunkRows || chunkRows == rows) return plan(w, p, gate);
  if (rows > chunkRows) throw std::invalid_argument("a plan of a chunk's rows takes rows of its chunk");
  const LinearPlan chunk = plan({w.matrix, chunkRows, epilogue}, p, gate);
  LinearPlan part(w, chunk.configuration(), chunk.destination());
  part.rotated_ = chunk.rotated_;
  return part;
}

LinearScratchSize Linear::scratchSize(ProjectionShape shape, uint32_t rows) const {
  if (!rows || rows > SPLASH_PREFILL_TOKEN_BUDGET) throw std::invalid_argument("invalid linear scratch rows");
  LinearScratchSize bound;
  // Split scratch and tables grow with the rows of the decode tiles, each
  // count of rows any tile the device may run it on; the prefill tiles never
  // split (LinearGguf.cpp).
  const LinearMatrix matrix{shape.outputSize, shape.inputSize};
  for (uint32_t count = 1; count <= std::min(rows, kMaximumStagedRows); ++count)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
      if (epilogue != LinearEpilogue::GateUp || count <= kMaximumDecodeTileRows)
        bound.include(ggufDecodeScratchSize({matrix, count, epilogue}, unitFormats(shape), shape.planOutputs));
  // A plan stores at most its rows' tiles: the decode tiles' up to their rows
  // (a decode step's kMaximumStagedRows, a chunk's kMaximumDecodeTileRows),
  // the staged tile's or the matrix tile's fragments, the prefill tiles'
  // whole 128-row tiles.
  static_assert(SPLASH_PREFILL_TOKEN_BUDGET % GGUF_PREFILL_ROWS == 0, "the prefill tiles cover the budget exactly");
  const uint32_t storage = rows <= (step_ == StepKind::Decode ? kMaximumStagedRows : kMaximumDecodeTileRows)
                               ? std::max({rows, stagedStorageRows(rows, true), matrixStorageRows(rows)})
                               : (rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS;
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, storage);
  // The most rows' plan, whose codes and scales the int8 prefill tile holds
  // where it takes the shape's formats.
  if (rows > kMaximumDecodeTileRows) {
    const LinearWorkload most{matrix, rows, LinearEpilogue::None};
    bound.include(LinearPlan(most, selectProjection(calibration_, int8Activations_, step_,
                                                    selectedWorkload(most, shape.planOutputs), unitFormats(shape)))
                      .scratchSize());
  }
  return bound;
}

std::array<metal::MetalBuffer, 2> Linear::splitScratch(const LinearBuffers &buffers, uint32_t splits) {
  if (splits > 1) return {buffers.scratch.partials, buffers.scratch.counters};
  return {buffers.output, buffers.output};
}

PreparedInput Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Projection &p, const LinearPlan &selected, const Projection *gate) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  if ((w.epilogue == LinearEpilogue::GateUp) != (gate != nullptr))
    throw std::invalid_argument("a gate/up plan takes a gate projection and no other plan does");
  requireLeadingInputs(selected, p);
  if (gate) requireLeadingInputs(selected, *gate);
  const uint64_t rows = selected.storageRows();
  requireBytes(b.input, rows * k * 2, "projection input");
  requireBytes(b.output, rows * n * elementBytes(selected.destination()), "projection output");
  if (w.epilogue == LinearEpilogue::Residual) requireBytes(b.residual, rows * n * 2, "projection residual");
  requireBytes(b.gateScratch, selected.gateScratchBytes(), "projection gate scratch");
  const LinearScratchSize scratch = selected.scratchSize();
  requireBytes(b.scratch.input, scratch.input, "projection scratch table");
  requireBytes(b.scratch.sums, scratch.sums, "projection scratch sums");
  requireBytes(b.scratch.partials, scratch.partials, "projection partials");
  requireBytes(b.scratch.counters, scratch.counters, "projection counters");
  if (p.rotation) requireBytes(b.scratch.rotated, rotatedBytes(k, rows), "projection rotated input");
  addGguf(graph, b, p, selected, gate);
  // A rotated projection's plan prepares its table, if any, from the rotated
  // rows, which no other plan reads. The int8 prefill tile leaves its codes
  // and scales of the input in the table, for a next plan over the same rows.
  if (p.rotation) return {};
  if (selected.configuration().tile == LinearTile::GgufPrefillInt8) return {b.input, LinearInput::PrefillInt8};
  // Only quantized segments run the plan's tile: float segments alone leave
  // the scratch table as it was.
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  const bool tiled =
      std::any_of(segments.begin(), segments.end(), [](const QuantizedSegment &s) { return !s.isFloat(); });
  return tiled && selected.input() != LinearInput::Plain ? PreparedInput{b.input, selected.input()} : b.prepared;
}

void Linear::addProjection(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                           metal::MetalBuffer output, uint32_t rows, LinearScratch scratch, uint32_t chunkRows) const {
  add(graph, {.input = input, .output = output, .scratch = scratch}, p,
      plan(p, rows, LinearEpilogue::None, nullptr, chunkRows));
}
void Linear::addResidualProjection(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                   metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows,
                                   LinearScratch scratch, uint32_t chunkRows) const {
  add(graph, {.input = input, .output = output, .residual = residual, .scratch = scratch}, p,
      plan(p, rows, LinearEpilogue::Residual, nullptr, chunkRows));
}
// A chunk's rows decide whether its plans run the pair as one, so that a plan
// of some of its rows computes them as the chunk's does.
LinearInput Linear::swiGluInput(const SwiGluProjections &ffn, uint32_t rows, uint32_t chunkRows) const {
  if (!splitFree_ && (chunkRows ? chunkRows : rows) <= kMaximumDecodeTileRows)
    return plan(*ffn.up, rows, LinearEpilogue::GateUp, ffn.gate, chunkRows).input();
  return plan(*ffn.gate, rows, LinearEpilogue::None, nullptr, chunkRows).input();
}
void Linear::addSwiGlu(metal::CommandGraph &graph, const SwiGluProjections &ffn, const PrefillFfnBuffers &b,
                       metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, uint32_t chunkRows,
                       PreparedInput normalized) const {
  // A split-free SwiGLU runs gate then up at any rows, the plans a chunk past
  // the decode tiles runs, so that a row's output is the same bits in any chunk.
  if (!splitFree_ && (chunkRows ? chunkRows : rows) <= kMaximumDecodeTileRows) {
    add(graph,
        {.input = b.normalized, .output = b.intermediate, .gateScratch = b.gateScratch, .scratch = b.scratch,
         .prepared = normalized},
        *ffn.up, plan(*ffn.up, rows, LinearEpilogue::GateUp, ffn.gate, chunkRows), ffn.gate);
  } else {
    normalized = add(graph,
                     {.input = b.normalized, .output = b.gateScratch, .scratch = b.scratch, .prepared = normalized},
                     *ffn.gate, plan(*ffn.gate, rows, LinearEpilogue::None, nullptr, chunkRows));
    add(graph,
        {.input = b.normalized, .output = b.intermediate, .gateScratch = b.gateScratch, .scratch = b.scratch,
         .prepared = normalized},
        *ffn.up, plan(*ffn.up, rows, LinearEpilogue::UpWithGate, nullptr, chunkRows));
  }
  addResidualProjection(graph, b.intermediate, *ffn.down, residual, output, rows, b.scratch, chunkRows);
}

} // namespace splash::ops
