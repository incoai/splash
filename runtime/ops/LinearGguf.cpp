// GGUF projections: plan validation and dispatch (kernels/shared/gguf_linear.metal,
// kernels/prefill/gguf_prefill_int8.metal, kernels/decode/linear_gguf_sgmatrix.metal,
// kernels/decode/linear_gguf_column.metal); KernelSelection.cpp picks the plans.
#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/BufferExtent.hpp"
#include "ops/KernelSelection.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

// The segments one fused decode dispatch runs.
constexpr size_t kFusedSegments = std::extent_v<decltype(GgufDecodeFusedParams::cols)>;
static_assert(QUANT_TILE_ROWS % GGUF_TILE_COLUMNS == 0, "an int8 prefill column tile lies inside one plane tile");

// A staged decode kernel of a format at `rows` rows: gguf_decode_<format>_m<rows>_<suffix>, in int8 activations
// (LinearConfig::decodeInt8) gguf_decode_int8_<format>_m<rows>_<suffix>.
std::string decodeKernel(bool int8, const char *format, uint32_t rows, std::string_view suffix) {
  return std::string(int8 ? "gguf_decode_int8_" : "gguf_decode_") + format + "_m" + std::to_string(rows) + "_" +
         std::string(suffix);
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
  // The gate pass runs the tile's kernel of the gate's format, which F32 or
  // BF16 rows have none of: no source instantiates gguf_decode_f32_m8_a.
  if (gates.front().isFloat()) throw std::invalid_argument("float segments take no epilogue");
  tensor(gates.front(), epilogueSuffix(LinearEpilogue::None), b.gateScratch, b.gateScratch);
  tensor(segment, epilogueSuffix(LinearEpilogue::UpWithGate), b.output, b.gateScratch);
}

// The gate and up tensors of a one-pass gate/up: single tensors of one format.
std::pair<const QuantizedSegment &, const QuantizedSegment &> onePassTensors(const Projection &up,
                                                                             const Projection &gate) {
  const std::vector<QuantizedSegment> &ups = up.blocks().segments, &gates = gate.blocks().segments;
  if (ups.size() != 1 || gates.size() != 1 || gates.front().formatId != ups.front().formatId)
    throw std::invalid_argument("a one-pass block gate/up takes single tensors of one format");
  return {gates.front(), ups.front()};
}

} // namespace

void LinearPlan::requireConfiguration() const {
  const uint32_t k = workload_.matrix.inputSize;
  const LinearConfig &c = config_;
  const bool decodeTiles = workload_.rows <= kMaximumDecodeTileRows;
  if (c.spread && c.tile != LinearTile::GgufStaged)
    throw std::invalid_argument("only the staged block tile spreads its walks");
  if (c.oneGateUpPass && ((c.tile != LinearTile::GgufStaged && c.tile != LinearTile::GgufColumn) ||
                          workload_.rows > kMaximumPairRows || workload_.epilogue != LinearEpilogue::GateUp))
    throw std::invalid_argument("only a staged or column block gate/up of up to 32 rows runs gate and up in one pass");
  if (c.sixteenRows && (c.tile != LinearTile::GgufStaged || workload_.rows > 8))
    throw std::invalid_argument("only a staged block plan of at most 8 rows runs the 16-row tile");
  if (c.decodeInt8 && (c.tile != LinearTile::GgufStaged || workload_.rows > kMaximumInt8Rows ||
                       !decodeInt8TakesSplits(k, c.splits)))
    throw std::invalid_argument("only a staged block plan of up to 32 rows of whole PQ2_0 blocks per split runs "
                                "int8 activations");
  if (c.tile == LinearTile::GgufColumn) {
    // The column tile reads its rows alone, a gate/up pair's both tensors in
    // one pass, and splits K inside its threadgroup alone.
    if (workload_.rows > GGUF_COLUMN_ROWS || c.splits != 1 || workload_.epilogue == LinearEpilogue::UpWithGate ||
        c.oneGateUpPass != (workload_.epilogue == LinearEpilogue::GateUp))
      throw std::invalid_argument("the column block tile takes up to GGUF_COLUMN_ROWS rows, unsplit, no up-with-gate");
    return;
  }
  if (c.tile == LinearTile::GgufRegister) {
    // Split boundaries fall on coefficient units, of 64 inputs at the fewest
    // (quant_register_unit); addGgufRegister checks each segment's.
    if (workload_.rows > kMaximumRegisterRows || !c.validSplits() || k / 64 < c.splits)
      throw std::invalid_argument("the register block tile takes up to four table tiles' rows and a K unit per split");
    return;
  }
  if (c.tile == LinearTile::GgufPrefill || c.tile == LinearTile::GgufPrefillInt8) {
    if (c.splits != 1) throw std::invalid_argument("invalid block prefill configuration");
    return;
  }
  if (!c.validSplits() || (k / 32) % c.splits)
    throw std::invalid_argument("staged block splits take whole 32-input groups");
  if (!decodeTiles) throw std::invalid_argument("the staged block tile runs plans of up to 64 rows");
}

uint32_t LinearPlan::storageRows() const noexcept {
  if (config_.tile == LinearTile::GgufColumn) return workload_.rows;
  if (config_.tile == LinearTile::GgufRegister) return tableTiles(workload_.rows) * GGUF_TABLE16_ROWS;
  if (config_.tile == LinearTile::GgufPrefill || config_.tile == LinearTile::GgufPrefillInt8)
    return (workload_.rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS;
  return stagedStorageRows(workload_.rows, config_.sixteenRows, config_.decodeInt8);
}

LinearScratchSize LinearPlan::scratchSize() const noexcept {
  const auto [n, k] = workload_.matrix;
  // The column tile sums K's parts in its threadgroup alone.
  if (config_.tile == LinearTile::GgufColumn) return {};
  // Register tile: the Table16 table and sums, and split, [lane][split][row]
  // [column] partials and one counter per 64-column tile, which covers every
  // lane.
  if (config_.tile == LinearTile::GgufRegister) {
    const uint64_t rows = storageRows();
    return {tableBytes(k, rows), tableSumsBytes(LinearInput::Table16, k, rows),
            config_.splits > 1 ? config_.splits * rows * n * sizeof(float) : 0,
            config_.splits > 1 ? uint64_t{n / GGUF_TILE_COLUMNS} * sizeof(uint32_t) : 0};
  }
  // Int8 prefill: the codes and scales of the stored rows.
  if (config_.tile == LinearTile::GgufPrefillInt8)
    return {gguf_prefill_int8_table_bytes(k, storageRows()), gguf_prefill_int8_scale_bytes(k, storageRows()), 0, 0};
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
  if (plan.configuration().decodeInt8 &&
      !unitFormats(std::array<const Projection *, 2>{&p, gate}).decodeInt8Takes)
    throw std::invalid_argument("only a rotated PQ2_0 projection takes int8 activations");
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
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
    LinearBuffers rotated = b;
    rotated.input = b.scratch.rotated;
    rotated.prepared = {};
    Projection plain = p;
    plain.rotation = {};
    if (plan.configuration().decodeInt8) {
      // In int8 activations the rotation writes int8 rows and their scales for the plan's storage rows
      // (gguf_rotate_int8), which the staged plan's int8 tiles read.
      const uint32_t rows = plan.storageRows();
      requireBytes(b.scratch.rotated, gguf_rotation_int8_bytes(k, rows), "projection int8 rows");
      graph.add("gguf_rotate_int8", {b.input, p.rotation.signs, b.scratch.rotated}, GgufRotationInt8Params{k, rows},
                {k / GGUF_ROTATION_BLOCK, w.rows, 1}, {GGUF_ROTATION_THREADS, 1, 1});
      addGgufStaged(graph, rotated, plain, plan, gate);
      return;
    }
    graph.add("gguf_rotate", {b.input, p.rotation.signs, b.scratch.rotated}, GgufRotationParams{k},
              {k / GGUF_ROTATION_BLOCK, w.rows, 1}, {GGUF_ROTATION_THREADS, 1, 1});
    addGguf(graph, rotated, plain, plan, gate);
    return;
  }
  switch (plan.configuration().tile) {
  case LinearTile::GgufRegister: addGgufRegister(graph, b, p, plan, gate); break;
  case LinearTile::GgufStaged: addGgufStaged(graph, b, p, plan, gate); break;
  case LinearTile::GgufPrefill: addGgufPrefill(graph, b, p, plan); break;
  case LinearTile::GgufPrefillInt8: addGgufPrefillInt8(graph, b, p, plan); break;
  case LinearTile::GgufColumn: addGgufColumn(graph, b, p, plan, gate); break;
  }
}

// The staged tiles, for plans of up to kMaximumDecodeTileRows rows: every row
// of the plan's storage in each threadgroup's tile, grid
// (64-column tiles, K splits). A gate/up runs as two passes, or, where
// selectProjection sets LinearConfig::oneGateUpPass, as one pass over
// 32-column threadgroups of one simdgroup each. In int8 activations
// (LinearConfig::decodeInt8) the same dispatches run the int8 tiles of the
// segments' one format over the int8 rows of b.input, each K split holding
// whole blocks of it.
void Linear::addGgufStaged(metal::CommandGraph &graph, const LinearBuffers &b,
                             const Projection &p, const LinearPlan &plan,
                             const Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  const uint32_t rows = plan.storageRows(), splits = config.splits;
  const auto [partials, counters] = splitScratch(b, splits);
  if (config.decodeInt8)
    for (const Projection *q : {&p, gate})
      if (q)
        for (const QuantizedSegment &s : q->blocks().segments)
          if (s.formatId != segments.front().formatId || (k / 32) % (splits * s.format().meta_groups))
            throw std::invalid_argument("the int8 tiles take segments of one format, whole blocks of it per split");
  const auto tensor = [&](const QuantizedSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(kernelInstance(decodeKernel(config.decodeInt8, s.name(), rows, std::string_view(&epilogue, 1)),
                             plan.destination()),
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
    if (config.decodeInt8) {
      // The int8 tiles take segments of one format, each format a kernel of its own.
      graph.add(std::string("gguf_decode_int8_") + segments.front().name() + "_fused_m" + std::to_string(rows),
                std::move(bindings), params, {segmentColumns(p) / GGUF_TILE_COLUMNS, splits, 1},
                {GGUF_STAGED_THREADS, 1, 1});
      return;
    }
    graph.add("gguf_decode_fused_m" + std::to_string(rows), std::move(bindings), params,
              {segmentColumns(p) / GGUF_TILE_COLUMNS, splits, 1}, {GGUF_STAGED_THREADS, 1, 1});
    // The pipeline decodes the segments' formats alone, each once in
    // ascending order and the last repeated to three, so projections of one
    // format set share it (gguf_linear.metal, fused_formats).
    std::array<uint32_t, kFusedSegments> formats{};
    for (size_t i = 0; i < order.size(); ++i) formats[i] = order[i]->formatId;
    std::sort(formats.begin(), formats.begin() + order.size());
    const auto distinct = std::unique(formats.begin(), formats.begin() + order.size());
    std::fill(distinct, formats.end(), distinct[-1]);
    graph.specialize(formats);
  };
  if (config.oneGateUpPass) {
    const auto [g, u] = onePassTensors(p, *gate);
    graph.add(decodeKernel(config.decodeInt8, u.name(), rows, "gate_up"),
              {b.input, g.plane0, g.plane1Slot(), g.meta, u.plane0, u.plane1Slot(), u.meta, b.output, partials,
               counters},
              GgufDecodeParams{k, splits, n, u.columnOffset, config.spread},
              {u.outputSize / GGUF_STAGED_COLUMNS, splits, 1}, {GGUF_GATE_UP_THREADS, 1, 1});
    return;
  }
  // A fused projection's segments run in one dispatch where they take no
  // epilogue: a dispatch costs about 0.045 ms however narrow, and at 8 to 32
  // rows the 27B's and 35B's fused inputs take 16-56% less time in one
  // dispatch than in one per segment (32-core M5 Max). Each segment with an
  // epilogue runs its own, and so does each of fp32 rows (the draft's
  // vocabulary head over a subset of its rows, DFlashDraft): the fused
  // kernels write bf16.
  if (segments.size() > 1 && w.epilogue == LinearEpilogue::None && plan.destination() == FloatOutput::BFloat16) {
    fused();
    return;
  }
  if (w.epilogue == LinearEpilogue::GateUp && segments.size() != 1)
    throw std::invalid_argument("block gate/up takes single tensors");
  for (const QuantizedSegment &s : segments) addDecodeTensor(b, w.epilogue, s, gate, tensor);
}

// The prefill tile, which runs plans of more than kMaximumDecodeTileRows rows
// over 128-row tiles: a plain projection in one
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

// The int8 prefill tile: the chunk's rows quantized into the scratch's codes
// and scales, unless it holds them from the plan before over the same rows
// (b.prepared: a SwiGLU's gate for its up), then a GEMM dispatch per segment
// over its column tiles and the chunk's row tiles, on a grid 16 threadgroups
// wide (kernels/prefill/gguf_prefill_int8.metal); a view of leading inputs
// reads its planes' rows at their own stride.
void Linear::addGgufPrefillInt8(metal::CommandGraph &graph, const LinearBuffers &b, const Projection &p,
                                const LinearPlan &plan) const {
  const LinearWorkload w = plan.workload();
  const auto [n, k] = w.matrix;
  const uint32_t storage = plan.storageRows(),
                 rowTiles = (w.rows + GGUF_PREFILL_INT8_ROWS - 1) / GGUF_PREFILL_INT8_ROWS;
  for (const QuantizedSegment &s : p.blocks().segments)
    if (!quant_prefill_int8_format(s.formatId))
      throw std::invalid_argument(std::string("the int8 prefill tile does not take ") + s.name() + " weights");
  if (b.prepared.layout != LinearInput::PrefillInt8 || !b.prepared.source.sameView(b.input))
    addGgufPrefillInt8Quantize(graph, b.input, b.scratch.input, b.scratch.sums, k, w.rows, storage);
  const std::string epilogue(1, epilogueSuffix(w.epilogue));
  for (const QuantizedSegment &s : p.blocks().segments) {
    const uint32_t columnTiles = s.outputSize / GGUF_TILE_COLUMNS;
    graph.add(std::string("gguf_prefill_int8_") + s.name() + "_" + epilogue,
              {b.scratch.input, b.scratch.sums, s.plane0, s.plane1Slot(), s.meta, b.output,
               epilogueInput(b, w.epilogue)},
              GgufPrefillInt8Params{k, w.rows, storage, n, s.columnOffset, p.planeInputs() ? p.planeInputs() : k,
                                    columnTiles},
              {GGUF_PREFILL_INT8_WALK, (columnTiles * rowTiles + GGUF_PREFILL_INT8_WALK - 1) / GGUF_PREFILL_INT8_WALK, 1},
              {GGUF_PREFILL_INT8_THREADS, 1, 1});
  }
}

// Every table tile of the plan's rows in each threadgroup: single tensors and
// gate/up through addDecodeTensor, fused projections through fusedSegments,
// but into fp32 a dispatch per segment.
void Linear::addGgufRegister(metal::CommandGraph &graph, const LinearBuffers &b,
                             const Projection &p, const LinearPlan &plan,
                             const Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
  const uint32_t lanes = tableTiles(w.rows);
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
  const std::string suffix = "_l" + std::to_string(lanes);
  const auto [partials, counters] = splitScratch(b, config.splits);
  // The fused kernel writes bf16, as the staged tile's: fp32 rows of several
  // tensors run each tensor's own fp32 dispatch.
  if (segments.size() > 1 && plan.destination() == FloatOutput::BFloat16) {
    std::vector<const QuantizedSegment *> order;
    for (const QuantizedSegment &s : segments) order.push_back(&s);
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, b.scratch.sums};
    const GgufDecodeFusedParams params = fusedSegments(plan, order, bindings);
    bindings.insert(bindings.end(), {b.output, partials, counters});
    graph.add("gguf_decode_sg_fused" + suffix, std::move(bindings), params,
              {segmentColumns(p) / GGUF_TILE_COLUMNS, config.splits, 1}, {GGUF_REGISTER_THREADS, 1, 1});
    return;
  }
  const auto tensor = [&](const QuantizedSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(kernelInstance(std::string("gguf_decode_sg_") + s.name() + suffix + "_" + epilogue, plan.destination()),
              {b.scratch.input, b.scratch.sums, s.plane0, s.plane1Slot(), s.meta, output, partials, counters, aux},
              GgufDecodeParams{k, config.splits, n, s.columnOffset, 0},
              {s.outputSize / GGUF_TILE_COLUMNS, config.splits, 1}, {GGUF_REGISTER_THREADS, 1, 1});
  };
  if (segments.size() == 1) {
    addDecodeTensor(b, w.epilogue, segments.front(), gate, tensor);
    return;
  }
  for (const QuantizedSegment &s : segments) tensor(s, epilogueSuffix(w.epilogue), b.output, b.output);
}

// One dispatch per tensor, a gate/up pair's both in one (onePassTensors), each
// threadgroup 32 of the tensor's columns over the plan's rows.
void Linear::addGgufColumn(metal::CommandGraph &graph, const LinearBuffers &b, const Projection &p,
                           const LinearPlan &plan, const Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const auto add = [&](std::string name, std::vector<metal::MetalBuffer> bindings, const QuantizedSegment &s) {
    graph.add(std::move(name), std::move(bindings),
              GgufColumnParams{w.rows, w.matrix.inputSize, w.matrix.outputSize, s.columnOffset, s.formatId},
              {s.outputSize / 32, 1, 1}, {GGUF_COLUMN_THREADS, 1, 1});
  };
  if (gate) {
    const auto [g, u] = onePassTensors(p, *gate);
    add("gguf_column_gate_up", {b.input, g.plane0, g.plane1Slot(), g.meta, b.output, u.plane0, u.plane1Slot(), u.meta},
        u);
    return;
  }
  for (const QuantizedSegment &s : p.blocks().segments) {
    if (w.epilogue == LinearEpilogue::Residual) {
      add("gguf_column_r", {b.input, s.plane0, s.plane1Slot(), s.meta, b.output, b.residual}, s);
      continue;
    }
    add(kernelInstance("gguf_column_a", plan.destination()), {b.input, s.plane0, s.plane1Slot(), s.meta, b.output}, s);
  }
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

// A row's GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS groups of 64 inputs a
// threadgroup, a simdgroup a group.
void addGgufPrefillInt8Quantize(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                                const metal::MetalBuffer &codes, const metal::MetalBuffer &scales, uint32_t inputSize,
                                uint32_t rows, uint32_t scaleStride) {
  const uint32_t groups = inputSize / GGUF_PREFILL_INT8_GROUP;
  graph.add("gguf_prefill_int8_quantize", {input, codes, scales},
            GgufPrefillInt8QuantizeParams{inputSize, rows, scaleStride},
            {(groups + GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS - 1) / GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS, rows, 1},
            {GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS * 32, 1, 1});
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
