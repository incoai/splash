// GGUF projections: plan policy and dispatch (kernels/shared/gguf_linear.metal,
// kernels/decode/linear_gguf_sgmatrix.metal).
#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/BufferExtent.hpp"

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

// The staged decode tile that holds `rows` rows: MPP computes 16-row
// fragments, so the tile holds 8, 16 or 32 rows and a three-lane step runs
// the 32-row tile over four lanes of storage.
constexpr uint32_t stagedTileRows(uint32_t rows) noexcept { return rows <= 8 ? 8 : rows <= 16 ? 16 : 32; }

std::string decodeKernel(const char *format, uint32_t rows, char epilogue) {
  return std::string("gguf_decode_") + format + "_m" + std::to_string(rows) + "_" + epilogue;
}
std::string prefillKernel(const char *format, char epilogue) {
  return std::string("gguf_prefill_") + format + "_" + epilogue;
}

// K splits of a decode tile, one rule for both tiles. A tier asks for more
// partitions while the grid holds fewer than `threadgroups` threadgroups per
// core and each partition would still keep `inputs` inputs; the split count
// doubles, up to the maximum, while some tier asks. Decode K is a multiple of 256,
// so eight partitions always hold whole 32-input groups. The rule ignores the
// batch width: bounds that depended on it did not pay on either family.
struct SplitTier {
  uint32_t threadgroups;
  uint32_t inputs;
};
uint32_t decodeSplits(uint32_t n, uint32_t k, uint32_t cores, std::span<const SplitTier> tiers) {
  const uint64_t grid = n / GGUF_TILE_COLUMNS;
  uint32_t splits = 1;
  const auto asks = [&](const SplitTier &t) {
    return grid * splits < uint64_t{t.threadgroups} * cores && k / (2 * splits) >= t.inputs;
  };
  while (splits < LinearConfig::kMaximumSplits && std::any_of(tiers.begin(), tiers.end(), asks)) splits *= 2;
  return splits;
}

// Apple9 register tile (128 threads). A core holds four of its threadgroups
// at once, so the tile's time steps with every four threadgroups per core.
// Below one wave a core must fill it, down to one 256-input coefficient unit
// per partition; below eight waves more threadgroups shrink the last wave's
// tail while partitions of 1024 inputs amortize the partial sums.
constexpr SplitTier kRegisterTiers[] = {{4, 256}, {32, 1024}};

// Staged tile (64 threads): one fitted tier, six threadgroups per core with
// 512 inputs per partition. Six is not a residency (12-17 of these
// threadgroups run at once per core on the M5 Pro): past it a core's memory
// and neural accelerator are busy and more partitions only add reduction.
// Over the 27B and 35B dense shapes, all formats, one to four lanes, on the
// 16- and 20-core M5 Pro and 10-, 30- and 40-core GPUs emulated by width:
// 3.6% over the fastest split of each shape in total and 36% at worst on a
// 15-us shape (the register tiers in threads per core: 6.6%).
constexpr SplitTier kStagedTiers[] = {{6, 512}};

// The staged tile's tiers on a family. Apple9 cores take as many of its
// threadgroups as of the register tile's, and its tiers: on a 40-core M3 Max
// over the 27B and 35B dense shapes at one to four lanes they come within
// 0-9% of each shape's fastest split (20% at one lane on 2048 x 512, a
// 0.012 ms projection), where the fitted tier above is 13-27% slower on
// 17408 x 5120 and 12288 x 5120 and 50% on 2048 x 512.
std::span<const SplitTier> stagedTiers(GpuFamilyClass family) noexcept {
  if (family == GpuFamilyClass::Apple9) return kRegisterTiers;
  return kStagedTiers;
}

// The decode tile configurations over an n x k matrix on `cores` cores.
LinearConfig registerDecode(uint32_t n, uint32_t k, uint32_t cores) {
  return {.tile = LinearTile::GgufRegister, .splits = decodeSplits(n, k, cores, kRegisterTiers)};
}
LinearConfig stagedDecode(uint32_t n, uint32_t k, uint32_t cores, GpuFamilyClass family) {
  return {.tile = LinearTile::GgufStaged, .splits = decodeSplits(n, k, cores, stagedTiers(family))};
}

// Whether the staged decode tiles on `cores` cores of `family` spread their
// walks and run a gate/up pair of one format in one pass (Linear::ggufBaseline):
// Apple10 and later from 16 cores.
constexpr bool spreadsStagedDecode(GpuFamilyClass family, uint32_t cores) noexcept {
  return family == GpuFamilyClass::Apple10 && cores >= 16;
}
// A gate/up pair of n x k tensors in one pass, its K split as one projection
// of the pair's 2n columns.
LinearConfig oneGateUpPassDecode(uint32_t n, uint32_t k, uint32_t cores, GpuFamilyClass family) {
  return {.tile = LinearTile::GgufStaged, .splits = decodeSplits(2 * n, k, cores, stagedTiers(family)),
          .spread = true, .oneGateUpPass = true};
}
// Whether a gate/up plan's projections, up then gate, are single tensors of
// one format, which the one-pass kernels take.
bool oneFormat(std::span<const Projection *const> projections) {
  if (projections.size() != 2 || !projections[0] || !projections[1]) return false;
  const std::vector<QuantizedSegment> &up = projections[0]->blocks().segments, &gate = projections[1]->blocks().segments;
  return up.size() == 1 && gate.size() == 1 && !up.front().isFloat() && up.front().formatId == gate.front().formatId;
}

// Whether Apple9 decodes a plan's projections (a gate/up plan's two) on the
// staged tile: where it holds the lanes' rows unpadded, so the plan binds the
// register tile's rows, and every quantized segment is in a format of
// apple9StagesFormat, or in Q2_K from two lanes. On a 40-core M3 Max at
// 17408x5120 and 5120x17408, best split of each tile, the staged tile takes
// 6-21% less time on those formats at one lane, 2-11% at two and 2-9% at
// four, and on Q2_K 24-33% less at two and four lanes but 5-8% more at one.
// Three lanes pad its 24 rows to 32, where it takes 16-47% more for every
// format; the other formats keep the register tile, 6-44% faster at one lane.
bool apple9Stages(LinearWorkload w, std::span<const Projection *const> projections) {
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  bool quantized = false;
  for (const Projection *p : projections) {
    if (!p) continue;
    for (const QuantizedSegment &s : p->blocks().segments) {
      if (s.isFloat()) continue;
      if (!apple9StagesFormat(s.formatId) && !(s.formatId == GGUF_FMT_Q2K && lanes >= 2)) return false;
      quantized = true;
    }
  }
  return quantized && stagedTileRows(w.rows) == w.rows;
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
  if (c.tile == LinearTile::GgufRegister) {
    // Split boundaries fall on 256-input coefficient units.
    if (!decode || !c.validSplits() || k / 256 < c.splits)
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
  // in decode.
  if (!decode && workload_.rows > kMaximumDecodeTileRows)
    throw std::invalid_argument("the staged block tile runs prefill chunks of up to 32 rows");
}

uint32_t LinearPlan::storageRows() const noexcept {
  if (config_.tile == LinearTile::GgufRegister) return workload_.rows;
  if (config_.tile == LinearTile::GgufPrefill)
    return (workload_.rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS;
  return stagedTileRows(workload_.rows);
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

LinearConfig Linear::ggufBaseline(LinearWorkload w, std::span<const Projection *const> projections) const {
  const auto [n, k] = w.matrix;
  // Prefill: 128-row tiles. A chunk of up to 32 rows runs the staged tile
  // of its rows (8, 16 or 32, two simdgroups) with the decode split rule:
  // the same half stage and matmul rows, so its outputs equal the prefill
  // tile's up to the K split's fp32 reassociation (gguf-projection checks it),
  // and each simdgroup streams its own 32 columns instead of four 8-row
  // simdgroups sharing a stage. Unsplit, on a 17408 x 5120 Q4_K projection
  // that is 1.8-2.9x faster on a 16-core M5 Pro (its neural accelerator pads
  // 8 rows to 16) and 1.1-2.8x on a 40-core M3 Max; the 27B down and output
  // projections split in two gain 24-37% more on the 16-core M5 Pro.
  if (w.phase == LinearPhase::Prefill)
    return w.rows <= kMaximumDecodeTileRows
        ? LinearConfig{.tile = LinearTile::GgufStaged,
                       .splits = decodeSplits(n, k, gpuCores_, stagedTiers(family_))}
        : LinearConfig{.tile = LinearTile::GgufPrefill};
  // Apple9 runs matrix operations on the FP32 pipe, so the exact register
  // kernel beats staging but for the projections apple9Stages names.
  if (family_ == GpuFamilyClass::Apple9 && !apple9Stages(w, projections)) return registerDecode(n, k, gpuCores_);
  // Decode tiles walking K in lockstep all fetch the same slice of the input
  // at each step, just written by the operator before and in no cache yet,
  // and the fetch sits on every tile's critical path; starting each column
  // tile at its own step lets most tiles find their slice brought in by
  // others. On Apple10 and later the 27B decodes 4-10% faster end to end on a
  // 40-core M5 Max and 4-8% on a 20-core M5 Pro (8-bit weights, whose tiles
  // stream twice the bytes, 1% slower at one and two lanes and 1-2% faster
  // at three and four on the M5 Max); a 12-core M6 and Apple9 gain nothing,
  // lockstep as fast or faster. 16 cores lies between the measured 12 and 20.
  //
  // There a gate/up pair of one format also runs in one pass, each simdgroup
  // staging both tensors' columns, so each serial step of a chain (dequantize,
  // barrier, matmul) serves both tensors: a 40-core M5 Max core saturates at
  // about 16 chains and the 27B pair puts 13.6 on each, so the steps' latency,
  // not DRAM, bounds it. The 27B pair alone takes 8-9% less time in MLX 4-bit
  // and Q4_K at one and two lanes, the 27B decode cycle 2.4-3.2% less in MLX
  // 4-bit and 0.6-2.0% in UD-Q4_K_M (1.5-3.7% and 0-1.3% on a 20-core M5
  // Pro); 8-bit, whose pair streams near DRAM's rate either way, takes 0.1-0.6%
  // more. A 12-core M6 holds 45 chains of the pair per core, saturated, where
  // one pass took 2-11% more.
  LinearConfig config = stagedDecode(n, k, gpuCores_, family_);
  config.spread = spreadsStagedDecode(family_, gpuCores_);
  if (config.spread && w.epilogue == LinearEpilogue::GateUp && oneFormat(projections))
    return oneGateUpPassDecode(n, k, gpuCores_, family_);
  return config;
}

LinearScratchSize Linear::ggufDecodeScratchSize(LinearWorkload w) const {
  const auto [n, k] = w.matrix;
  LinearScratchSize size = LinearPlan(w, ggufBaseline(w)).scratchSize();
  if (family_ == GpuFamilyClass::Apple9) size.include(LinearPlan(w, stagedDecode(n, k, gpuCores_, family_)).scratchSize());
  if (w.epilogue == LinearEpilogue::GateUp && spreadsStagedDecode(family_, gpuCores_))
    size.include(LinearPlan(w, oneGateUpPassDecode(n, k, gpuCores_, family_)).scratchSize());
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

// The staged decode tiles, for decode and prefill chunks of up to 32 rows:
// every row of the plan's storage in each threadgroup's tile, grid (64-column
// tiles, K splits). Decode runs one dispatch per projection (addDecodeTensor,
// fusedSegments), gate/up as two passes, or, where ggufBaseline sets
// LinearConfig::oneGateUpPass, as one pass over 32-column threadgroups of one
// simdgroup each. Prefill chunks run one dispatch per segment.
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
  if (w.phase == LinearPhase::Prefill) {
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
  // Dispatch order is tile order: segments with the most bytes per tile
  // first, so their threadgroups do not form the tail.
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
}

// The prefill tile, which runs prefill chunks of more than
// kMaximumDecodeTileRows rows over 128-row tiles: a plain projection in one
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
                   ggufFloatTile(w.rows, s.outputSize));
}

// The neural accelerator tile needs one (Apple9's matrix operations share the
// FP32 pipe, where three bf16 matmuls cost three fp32 ones) and a grid of its
// 64-row by 32-column tiles of at least three threadgroups per two cores. A
// tile runs K / 32 dependent steps, while the fp32 kernel spreads fewer rows
// over 8-column tiles with 16 K partitions each and finishes first below that
// grid: at the 35B's K = 2048 the two take the same time at 1.4-1.6 tiles per
// core for both of its float projections on the 16- and 20-core M5 Pro.
FloatTile Linear::ggufFloatTile(uint32_t rows, uint32_t outputSize) const noexcept {
  const uint64_t tiles = uint64_t{(rows + 63) / 64} * ((outputSize + 31) / 32);
  return family_ == GpuFamilyClass::Apple10 && rows >= 16 && 2 * tiles >= uint64_t{3} * gpuCores_
             ? FloatTile::NeuralAccelerator
             : FloatTile::Simdgroup;
}

// Simdgroup: 8 columns of 32 rows per threadgroup of 16 simdgroups. Neural
// accelerator: 32 columns of 64 rows per threadgroup of 4 simdgroups, at least
// 16 rows and K a multiple of 32 (kernels/shared/gguf_float.metal).
void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, const QuantizedSegment &weights,
                  metal::MetalBuffer output, uint32_t rows, uint32_t outStride, uint32_t outOffset,
                  FloatOutput type, FloatTile tile) {
  const uint32_t n = weights.outputSize, k = weights.inputSize;
  const bool accelerator = tile == FloatTile::NeuralAccelerator;
  if (!weights.isFloat() || !rows || !n || n % 8 || !k || k % 8 || outOffset + uint64_t{n} > outStride ||
      (accelerator && (rows < 16 || k % 32)))
    throw std::invalid_argument("invalid float projection");
  requireBytes(weights.plane0, uint64_t{n} * k * sizeof(float), "float projection weight");
  requireBytes(input, uint64_t{rows} * k * 2, "float projection input");
  requireBytes(output, rowBytes(rows, outStride, uint64_t{outOffset} + n, elementBytes(type)), "float projection output");
  const std::string kernel = std::string(accelerator ? "gguf_float_na_" : "gguf_float_") +
                             (type == FloatOutput::Float32 ? "f32" : "bf16");
  const metal::DispatchSize grid = accelerator ? metal::DispatchSize{(n + 31) / 32, (rows + 63) / 64, 1}
                                               : metal::DispatchSize{n / 8, (rows + 31) / 32, 1};
  graph.add(kernel, {std::move(input), weights.plane0, std::move(output)},
            GgufFloatParams{rows, k, n, outStride, outOffset}, grid, {accelerator ? 128u : 512u, 1, 1});
}

void requireSegmentPlanes(const QuantizedSegment &segment, std::string_view what, uint32_t planeInputs) {
  // Each plane ends at its unit of the last row's last group the segment
  // reads, or meta unit: tile row (rows - 1) % T of that group of its tile,
  // whose rows hold the groups of planeInputs inputs.
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

bool apple9StagesFormat(uint32_t format) noexcept {
  switch (format) {
  case GGUF_FMT_IQ3XXS: case GGUF_FMT_IQ2XXS: case GGUF_FMT_IQ2XS: case GGUF_FMT_IQ2S:
  case GGUF_FMT_IQ1S: case GGUF_FMT_IQ1M: return true;
  default: return false;
  }
}

QuantizedSegment QuantizedSegment::planes(uint32_t formatId, uint32_t outputSize, uint32_t inputSize,
                                          metal::MetalBuffer plane0, metal::MetalBuffer plane1,
                                          metal::MetalBuffer meta) {
  if (formatId >= GGUF_FMT_COUNT) throw std::invalid_argument("unknown GGUF segment format");
  return {std::move(plane0), std::move(plane1), std::move(meta), outputSize, inputSize, 0, formatId};
}
const QuantFormat &QuantizedSegment::format() const noexcept { return kQuantFormats[formatId]; }
const char *QuantizedSegment::name() const noexcept { return isFloat() ? "f32" : format().name; }

} // namespace splash::ops
