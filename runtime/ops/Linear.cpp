#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/BufferExtent.hpp"
#include "ops/KernelSelection.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

// Matrices of whole 64-row tiles (GGUF_TILE_COLUMNS) of whole 64-input spans,
// whose K is a multiple of a 64-input span (or of its format's superblock).
void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 64 || !w.matrix.inputSize || w.matrix.inputSize % 64)
    throw std::invalid_argument("invalid linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid linear prefill workload");
  } else {
    if (!w.rows || w.rows % SPLASH_TARGET_VERIFY_ROWS ||
        w.rows > SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid linear decode workload");
  }
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid linear decode batch width");
  return {matrix, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
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
  if (w.phase != LinearPhase::Prefill || w.epilogue != LinearEpilogue::Residual ||
      (tile != LinearTile::GgufPrefill && tile != LinearTile::GgufPrefillInt8) || p.rotation)
    throw std::invalid_argument("a view of leading inputs runs only the quantized prefill residual tiles");
}

// Throws unless views of `p`'s planes can stand for it (takesPlaneViews).
void requirePlaneViews(const Projection &p) {
  if (p.takesPlaneViews()) return;
  throw std::invalid_argument("views of a projection's planes take one unrotated quantized tensor, not a view");
}

} // namespace

bool Projection::takesPlaneViews() const noexcept {
  const std::vector<QuantizedSegment> &segments = weights_.segments;
  return !planeInputs_ && segments.size() == 1 && !segments.front().isFloat() && !rotation;
}

// Every plane holds its rows in tiles of QUANT_TILE_ROWS rows, each tile's
// units in order, so the leading rows' tiles lead it.
Projection Projection::leadingRows(const metal::MetalBackend &backend, uint32_t rows) const {
  requirePlaneViews(*this);
  if (!rows || rows % QUANT_TILE_ROWS || rows > outputSize)
    throw std::invalid_argument("a view of leading rows takes whole plane tiles of the projection's rows");
  // The leading rows of a plane of `rowBytes` bytes per row.
  const auto view = [&](const metal::MetalBuffer &plane, uint64_t rowBytes) {
    return rowBytes ? backend.view(plane, 0, rows * rowBytes) : metal::MetalBuffer{};
  };
  const QuantizedSegment &segment = weights_.segments.front();
  const QuantFormat &format = segment.format();
  const uint64_t groups = inputSize / 32;
  return Projection(rows, inputSize,
                    BlockWeights{{QuantizedSegment::planes(segment.formatId, rows, inputSize,
                                                           view(segment.plane0, groups * format.plane0_bytes),
                                                           view(segment.plane1, groups * format.plane1_bytes),
                                                           view(segment.meta,
                                                                groups / format.meta_groups * format.meta_bytes))}});
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

// Table16 holds its sums per eight-row tile (metal/abi/Gguf.h), a lane's rows.
uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept {
  return layout == LinearInput::Table16
             ? rows / SPLASH_TARGET_VERIFY_ROWS * table16_sums_per_tile(width) * sizeof(float)
             : 0;
}

void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows) {
  if (layout == LinearInput::Plain || !rows || rows % SPLASH_TARGET_VERIFY_ROWS || width % 64)
    throw std::invalid_argument("invalid linear table geometry");
  requireBytes(scratch.input, tableBytes(width, rows), "linear table");
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
  if (destination == FloatOutput::Float32 &&
      (w.phase != LinearPhase::Decode || w.epilogue != LinearEpilogue::None))
    throw std::invalid_argument("an fp32 destination takes a plain decode projection");
  // Kernel names follow the segment formats (LinearGguf.cpp).
  requireConfiguration();
}

Linear::Linear(const DeviceCapabilities &device, Int8Activations int8Activations) noexcept
    : calibration_(KernelCalibration::of(device)), int8Activations_(int8Activations) {}

uint32_t Linear::decodeStorageRows(uint32_t rows, ProjectionShape shape) const {
  // The None and Residual plans share a tile; a gate/up pair of one format
  // counts both tensors' columns in one pass and may stage where they do not.
  UnitFormats unit = unitFormats(shape);
  uint32_t storage = 0;
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::GateUp}) {
    const LinearWorkload w{{shape.outputSize, shape.inputSize}, rows, LinearPhase::Decode, epilogue};
    validate(w);
    unit.oneFormat = epilogue == LinearEpilogue::GateUp;
    storage = std::max(storage, LinearPlan(w, selectProjection(calibration_, int8Activations_, w, unit)).storageRows());
  }
  return storage;
}

LinearPlan Linear::plan(LinearWorkload workload) const {
  validate(workload);
  return LinearPlan(workload, selectProjection(calibration_, int8Activations_, workload, {}));
}
LinearPlan Linear::plan(LinearWorkload workload, LinearConfig config, FloatOutput destination) {
  return LinearPlan(workload, config, destination);
}
LinearPlan Linear::plan(LinearWorkload w, const Projection &p, const Projection *gate) const {
  validate(w);
  const std::array<const Projection *, 2> projections{&p, gate};
  // A gate/up pair's 16-row tile follows its tensors' own formats.
  std::array<UnitFormats, 2> tensors{};
  const bool pair = w.epilogue == LinearEpilogue::GateUp && gate;
  if (pair) tensors = {unitFormats(std::span(&projections[0], 1)), unitFormats(std::span(&projections[1], 1))};
  LinearPlan plan(w,
                  selectProjection(calibration_, int8Activations_, w, unitFormats(projections),
                                   pair ? std::span<const UnitFormats>(tensors) : std::span<const UnitFormats>{}),
                  p.destination);
  plan.rotated_ = static_cast<bool>(p.rotation);
  return plan;
}

LinearPlan Linear::decodePlan(const Projection &p, uint32_t lanes, LinearEpilogue epilogue,
                              const Projection *gate) const {
  return plan(decode({p.outputSize, p.inputSize}, lanes, epilogue), p, gate);
}
LinearPlan Linear::prefillPlan(const Projection &p, uint32_t rows, LinearEpilogue epilogue,
                               uint32_t chunkRows) const {
  const LinearWorkload w{{p.outputSize, p.inputSize}, rows, LinearPhase::Prefill, epilogue};
  if (!chunkRows || chunkRows == rows) return plan(w, p);
  if (rows > chunkRows) throw std::invalid_argument("a prefill plan takes rows of its chunk");
  const LinearPlan chunk = plan({w.matrix, chunkRows, LinearPhase::Prefill, epilogue}, p);
  LinearPlan part(w, chunk.configuration(), chunk.destination());
  part.rotated_ = chunk.rotated_;
  return part;
}

LinearScratchSize Linear::decodeScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
      bound.include(
          ggufDecodeScratchSize(decode({shape.outputSize, shape.inputSize}, lanes, epilogue), shape.registerFormats));
  // Decode plans store at most every lane's rows.
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, kMaximumDecodeTileRows);
  return bound;
}

LinearScratchSize Linear::prefillScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  // Prefill plans take split scratch only in chunks that run the staged tile,
  // of up to kMaximumStagedPrefillRows rows (LinearGguf.cpp).
  for (uint32_t rows = 1; rows <= kMaximumStagedPrefillRows; ++rows)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      bound.include(plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Prefill, epilogue}).scratchSize());
  // Every prefill plan stores at most the token budget.
  static_assert(SPLASH_PREFILL_TOKEN_BUDGET % GGUF_PREFILL_ROWS == 0, "the prefill tiles cover the budget exactly");
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, SPLASH_PREFILL_TOKEN_BUDGET);
  // A full chunk's, which the int8 prefill tile holds where it takes the
  // shape's formats.
  const LinearWorkload chunk{
      {shape.outputSize, shape.inputSize}, SPLASH_PREFILL_TOKEN_BUDGET, LinearPhase::Prefill, LinearEpilogue::None};
  validate(chunk);
  bound.include(
      LinearPlan(chunk, selectProjection(calibration_, int8Activations_, chunk, unitFormats(shape))).scratchSize());
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

void Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                        metal::MetalBuffer output, uint32_t rows, LinearScratch scratch, uint32_t chunkRows) const {
  add(graph, {.input = input, .output = output, .scratch = scratch}, p,
      prefillPlan(p, rows, LinearEpilogue::None, chunkRows));
}
void Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows,
                                LinearScratch scratch, uint32_t chunkRows) const {
  add(graph, {.input = input, .output = output, .residual = residual, .scratch = scratch}, p,
      prefillPlan(p, rows, LinearEpilogue::Residual, chunkRows));
}
void Linear::addPrefillSwiGlu(metal::CommandGraph &graph, const SwiGluProjections &ffn,
                              const PrefillFfnBuffers &b, metal::MetalBuffer residual, metal::MetalBuffer output,
                              uint32_t rows, uint32_t chunkRows) const {
  const PreparedInput normalized = add(graph, {.input = b.normalized, .output = b.gateScratch, .scratch = b.scratch},
                                       *ffn.gate, prefillPlan(*ffn.gate, rows, LinearEpilogue::None, chunkRows));
  add(graph,
      {.input = b.normalized, .output = b.intermediate, .gateScratch = b.gateScratch, .scratch = b.scratch,
       .prepared = normalized},
      *ffn.up, prefillPlan(*ffn.up, rows, LinearEpilogue::UpWithGate, chunkRows));
  addPrefillResidual(graph, b.intermediate, *ffn.down, residual, output, rows, b.scratch, chunkRows);
}

} // namespace splash::ops
