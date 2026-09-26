// GGUF projections: plan policy and dispatch (kernels/shared/gguf_linear.metal,
// kernels/decode/linear_gguf_sgmatrix.metal).
#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace splash::ops {
namespace {

// The staged decode tiles hold at most a full decode batch; prefill chunks of
// up to this many rows run them (Linear::ggufBaseline).
constexpr uint32_t kMaximumDecodeTileRows = SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS;
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

// Apple9 register tile (128 threads). Four of its threadgroups are resident
// on a core at once: on a 40-core M3 Max its time steps every four per core
// (Q4_K, K = 8192, one lane, ms: 3 per core 0.156, 4 0.157, 5 0.220, 7 0.281,
// 8 0.286; the same steps at two to four lanes and for Q8_0). Below one wave
// a core must fill it, down to one 256-input coefficient unit per partition;
// below eight waves more threadgroups shrink the last wave's tail while
// partitions of 1024 inputs amortize the partial sums (flat from eight to 32
// waves). Over every 27B and 35B projection kind at one to four lanes and
// 10-80 cores emulated by width, the decode step's projections run 0.95%
// slower than the fastest split of each shape on average and 2.3% at worst
// (sixteen threadgroups per core with two units per partition: 2.8%, 7.8%).
constexpr SplitTier kRegisterTiers[] = {{4, 256}, {32, 1024}};

// Staged tile (64 threads): one fitted tier, six threadgroups per core with
// 512 inputs per partition. Six is not a residency (12-17 of these
// threadgroups run at once per core on the M5 Pro): past it a core's memory
// and neural accelerator are busy and more partitions only add reduction.
// Over the 27B and 35B dense shapes, all formats, one to four lanes, on the
// 16- and 20-core M5 Pro and 10-, 30- and 40-core GPUs emulated by width:
// 3.6% over the fastest split of each shape in total and 36% at worst on a
// 15-us shape (the register tiers in threads per core: 6.6%; the previous 32
// per core with 1024 inputs and unsplit fused and gate/up kernels: 6.4%).
constexpr SplitTier kStagedTiers[] = {{6, 512}};

// The staged tile's tiers on a family. Apple9 cores take as many of its
// threadgroups as of the register tile's, and its tiers: on a 40-core M3 Max
// over the 27B and 35B dense shapes at one to four lanes they come within
// 0-9% of each shape's fastest split (20% at one lane on 2048 x 512, a
// 0.012 ms projection), where the fitted tier above is 13-27% slower on
// 17408 x 5120 and 12288 x 5120 and 50% on 2048 x 512.
std::span<const SplitTier> stagedTiers(uint32_t appleGpuFamily) noexcept {
  if (appleGpuFamily == 9) return kRegisterTiers;
  return kStagedTiers;
}

// The decode tile configurations over an n x k matrix on `cores` cores.
LinearConfig registerDecode(uint32_t n, uint32_t k, uint32_t cores) {
  return {LinearTile::GgufRegister, n / GGUF_TILE_COLUMNS, LinearSimdgroups::Four,
          decodeSplits(n, k, cores, kRegisterTiers)};
}
LinearConfig stagedDecode(uint32_t n, uint32_t k, uint32_t cores, uint32_t appleGpuFamily) {
  return {LinearTile::GgufStaged, n / GGUF_TILE_COLUMNS, LinearSimdgroups::Two,
          decodeSplits(n, k, cores, stagedTiers(appleGpuFamily))};
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
// for a quantized segment, 8 for a float one (addGgufFloat; F32 alpha/beta
// are 96 columns on the 27B); the columns past the last segment are padding
// no kernel writes. A fused projection keeps the layout's sizes: the 35B
// GGUF's packed GDN row is 12544 columns (the affine layout's), its
// qkv|z|alpha-beta segments 12352.
void requireSegments(const Projection &p, LinearMatrix matrix) {
  if (p.outputSize != matrix.outputSize || p.inputSize != matrix.inputSize)
    throw std::invalid_argument("block projection does not match plan");
  for (const QuantizedSegment &s : p.blocks().segments)
    if (s.outputSize % (s.isFloat() ? 8u : GGUF_TILE_COLUMNS))
      throw std::invalid_argument("block segments do not fill whole column tiles");
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
  GgufDecodeFusedParams params{w.matrix.inputSize, plan.configuration().splits, w.matrix.outputSize,
                               {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
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

void LinearPlan::requireBlockConfiguration() const {
  const auto [n, k] = workload_.matrix;
  const LinearConfig &c = config_;
  if (c.tile == LinearTile::GgufRegister) {
    // Split boundaries fall on 256-input coefficient units.
    if (workload_.phase != LinearPhase::Decode || c.groups != n / tileColumns() ||
        c.simdgroups != LinearSimdgroups::Four || !c.validSplits() || k / 256 < c.splits)
      throw std::invalid_argument("the register block decode tile takes the full column grid and a K unit per split");
    return;
  }
  if (!c.validSplits() || (k / 32) % c.splits)
    throw std::invalid_argument("staged block splits take whole 32-input groups");
  if (workload_.phase == LinearPhase::Prefill) {
    // Four simdgroups: 128-row prefill tiles. Two: the decode tiles, which
    // split K as in decode.
    const bool decodeTile = c.simdgroups == LinearSimdgroups::Two && workload_.rows <= kMaximumDecodeTileRows;
    if (c.groups || (c.simdgroups != LinearSimdgroups::Four && !decodeTile) || (c.splits > 1 && !decodeTile))
      throw std::invalid_argument("invalid block prefill configuration");
  } else if (c.groups != n / tileColumns() || c.simdgroups != LinearSimdgroups::Two) {
    throw std::invalid_argument("the staged block decode tile takes the full column grid");
  }
}

uint32_t LinearPlan::blockStorageRows() const noexcept {
  if (config_.tile == LinearTile::GgufRegister) return workload_.rows;
  return config_.simdgroups == LinearSimdgroups::Four
      ? (workload_.rows + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS * GGUF_PREFILL_ROWS
      : stagedTileRows(workload_.rows);
}

LinearScratchSize LinearPlan::blockScratchSize() const noexcept {
  const auto [n, k] = workload_.matrix;
  // Register tile: the Table16 table and sums, [lane][split][row][column]
  // partials and one counter per 64-column tile, which covers every lane.
  // Every binding exists even without splits.
  if (config_.tile == LinearTile::GgufRegister) {
    const uint64_t rows = workload_.rows;
    return {tableBytes(k, rows), tableSumsBytes(LinearInput::Table16, k, rows),
            config_.splits > 1 ? config_.splits * rows * n * sizeof(float) : sizeof(float),
            config_.splits > 1 ? uint64_t{n / tileColumns()} * sizeof(uint32_t) : sizeof(uint32_t)};
  }
  // Staged split-K: [split][row][column] fp32 partials over the tile's rows
  // and one counter per 64-column tile (a tile covers every row of the
  // dispatch).
  return config_.splits > 1
      ? LinearScratchSize{0, 0, uint64_t{config_.splits} * storageRows() * n * sizeof(float),
                          uint64_t{n / tileColumns()} * sizeof(uint32_t)}
      : LinearScratchSize{};
}

LinearScratchSize Linear::prefillScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  for (uint32_t rows = 1; rows <= kMaximumDecodeTileRows; ++rows)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      bound.include(
          plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Prefill, epilogue, shape.layout}).scratchSize());
  return bound;
}

LinearConfig Linear::ggufBaseline(LinearWorkload w, std::span<const Projection *const> projections) const {
  const auto [n, k] = w.matrix;
  // Prefill: 128-row tiles. A chunk of up to 32 rows runs the decode tile
  // of its rows (8, 16 or 32, two simdgroups) with the decode split rule:
  // the same half stage and matmul rows, so its outputs equal the prefill
  // tile's up to the K split's fp32 reassociation (gguf-projection full),
  // and each simdgroup streams its own 32 columns instead of four 8-row
  // simdgroups sharing a stage. Unsplit, on a 17408 x 5120 Q4_K projection
  // that is 1.8-2.9x faster on a 16-core M5 Pro (its neural accelerator pads
  // 8 rows to 16) and 1.1-2.8x on a 40-core M3 Max; the 27B down and output
  // projections split in two gain 24-37% more on the 16-core M5 Pro.
  if (w.phase == LinearPhase::Prefill)
    return w.rows <= kMaximumDecodeTileRows
        ? LinearConfig{LinearTile::GgufStaged, 0, LinearSimdgroups::Two,
                       decodeSplits(n, k, gpuCores_, stagedTiers(appleGpuFamily_))}
        : LinearConfig{LinearTile::GgufStaged, 0, LinearSimdgroups::Four};
  // Apple9 runs matrix operations on the FP32 pipe, so the exact register
  // kernel beats staging but for the projections apple9Stages names.
  if (appleGpuFamily_ == 9 && !apple9Stages(w, projections)) return registerDecode(n, k, gpuCores_);
  return stagedDecode(n, k, gpuCores_, appleGpuFamily_);
}

LinearScratchSize Linear::ggufDecodeScratchSize(LinearWorkload w) const {
  const auto [n, k] = w.matrix;
  LinearScratchSize size = LinearPlan(w, baseline(w)).scratchSize();
  if (appleGpuFamily_ == 9) size.include(LinearPlan(w, stagedDecode(n, k, gpuCores_, appleGpuFamily_)).scratchSize());
  return size;
}

void Linear::addGguf(metal::CommandGraph &graph, const LinearBuffers &b,
                       const Projection &p, const LinearPlan &plan,
                       const Projection *gate, LinearDispatchStats *stats) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
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
    if (!weights.segments.empty())
      addGguf(graph, b, Projection(p.outputSize, p.inputSize, std::move(weights)), plan, gate, stats);
    return;
  }
  if (config.tile == LinearTile::GgufRegister) {
    addGgufRegister(graph, b, p, plan, gate);
  } else if (config.simdgroups == LinearSimdgroups::Two) {
    addGgufStaged(graph, b, p, plan, gate);
  } else {
    // Prefill chunks of more than kMaximumDecodeTileRows rows: one dispatch
    // per segment over 128-row tiles; rows past w.rows stay inside the
    // budget-sized prefill buffers, and the simdgroups of a tile that only
    // hold them skip their matmuls.
    const char epilogue = epilogueSuffix(w.epilogue);
    for (const QuantizedSegment &s : segments) {
      std::vector<metal::MetalBuffer> bindings{b.input, s.plane0, s.plane1Slot(), s.meta, b.output};
      if (w.epilogue != LinearEpilogue::None) bindings.push_back(epilogueInput(b, w.epilogue));
      graph.add(prefillKernel(s.name(), epilogue), std::move(bindings),
                GgufPrefillParams{s.outputSize, k, w.rows, n, s.columnOffset},
                {plan.storageRows() / GGUF_PREFILL_ROWS, s.outputSize / GGUF_TILE_COLUMNS, 1},
                {GGUF_PREFILL_THREADS, 1, 1});
    }
  }
  if (stats && w.phase == LinearPhase::Decode) account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1);
}

// The staged decode tiles, for decode and prefill chunks of up to 32 rows:
// every row of the plan's storage in each threadgroup's tile, grid (64-column
// tiles, K splits). Decode runs one dispatch per projection (addDecodeTensor,
// fusedSegments); its two gate/up passes were within -4..+2% of the fused
// gate/up kernel they replaced on the 27B gate/up at 10-40 cores. Prefill
// chunks run one dispatch per segment.
void Linear::addGgufStaged(metal::CommandGraph &graph, const LinearBuffers &b,
                             const Projection &p, const LinearPlan &plan,
                             const Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  const uint32_t rows = plan.storageRows(), splits = config.splits;
  // One partition never touches the partials and counters: the output stands in.
  const metal::MetalBuffer partials = splits > 1 ? b.scratch.partials : b.output;
  const metal::MetalBuffer counters = splits > 1 ? b.scratch.counters : b.output;
  const auto tensor = [&](const QuantizedSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(kernelInstance(decodeKernel(s.name(), rows, epilogue), plan.destination()),
              {b.input, s.plane0, s.plane1Slot(), s.meta, output, partials, counters, aux},
              GgufDecodeParams{k, splits, n, s.columnOffset}, {s.outputSize / GGUF_TILE_COLUMNS, splits, 1},
              {GGUF_STAGED_THREADS, 1, 1});
  };
  if (w.phase == LinearPhase::Prefill) {
    for (const QuantizedSegment &s : segments)
      tensor(s, epilogueSuffix(w.epilogue), b.output, epilogueInput(b, w.epilogue));
    return;
  }
  if (segments.size() == 1) {
    addDecodeTensor(b, w.epilogue, segments.front(), gate, tensor);
    return;
  }
  // Dispatch order is tile order: segments with the most bytes per tile
  // first, so their threadgroups do not form the tail (alpha/beta are Q8_0).
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
  if (segments.size() > 1) {
    std::vector<const QuantizedSegment *> order;
    for (const QuantizedSegment &s : segments) order.push_back(&s);
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, b.scratch.sums};
    const GgufDecodeFusedParams params = fusedSegments(plan, order, bindings);
    bindings.insert(bindings.end(), {b.output, b.scratch.partials, b.scratch.counters});
    graph.add("gguf_decode_sg_fused" + suffix, std::move(bindings), params, grid, {GGUF_REGISTER_THREADS, 1, 1});
    return;
  }
  const auto tensor = [&](const QuantizedSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(kernelInstance(std::string("gguf_decode_sg_") + s.name() + suffix + "_" + epilogue, plan.destination()),
              {b.scratch.input, b.scratch.sums, s.plane0, s.plane1Slot(), s.meta, output, b.scratch.partials,
               b.scratch.counters, aux},
              GgufDecodeParams{k, config.splits, n, s.columnOffset}, grid, {GGUF_REGISTER_THREADS, 1, 1});
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
// tile runs K / 32 dependent steps (~50 us at the 35B's K = 2048), while the
// fp32 kernel spreads fewer rows over 8-column tiles with 16 K partitions
// each and finishes first below that: at K = 2048 its time equals the
// accelerator's at 1.4-1.6 tiles per core for both float projections of the
// 35B on the 16- and 20-core M5 Pro (router, N 256: 170 and 210 rows;
// alpha/beta, N 64: 720 and 850 rows). Above it the accelerator is up to
// 2.2x (router) and 2.3x (alpha/beta) faster at 2048 rows, ms per dispatch
// 0.81 -> 0.36 and 0.20 -> 0.083 on 16 cores.
FloatTile Linear::ggufFloatTile(uint32_t rows, uint32_t outputSize) const noexcept {
  const uint64_t tiles = uint64_t{(rows + 63) / 64} * ((outputSize + 31) / 32);
  return appleGpuFamily_ != 9 && rows >= 16 && 2 * tiles >= uint64_t{3} * gpuCores_ ? FloatTile::NeuralAccelerator
                                                                                     : FloatTile::Simdgroup;
}

// Simdgroup: 8 columns of 32 rows per threadgroup of 16 simdgroups. Neural
// accelerator: 32 columns of 64 rows per threadgroup of 4 simdgroups, at least
// 16 rows and K a multiple of 32 (kernels/shared/gguf_float.metal).
void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, const QuantizedSegment &weights,
                  metal::MetalBuffer output, uint32_t rows, uint32_t outStride, uint32_t outOffset,
                  FloatOutput type, FloatTile tile) {
  const uint32_t n = weights.outputSize, k = weights.inputSize;
  const uint64_t element = elementBytes(type);
  const bool accelerator = tile == FloatTile::NeuralAccelerator;
  if (!weights.isFloat() || !rows || !n || n % 8 || !k || k % 8 || outOffset + uint64_t{n} > outStride ||
      (accelerator && (rows < 16 || k % 32)) || weights.plane0.sizeBytes() < uint64_t{n} * k * sizeof(float) ||
      input.sizeBytes() < uint64_t{rows} * k * 2 ||
      output.sizeBytes() < (uint64_t{rows - 1} * outStride + outOffset + n) * element)
    throw std::invalid_argument("invalid float projection");
  const std::string kernel = std::string(accelerator ? "gguf_float_na_" : "gguf_float_") +
                             (type == FloatOutput::Float32 ? "f32" : "bf16");
  const metal::DispatchSize grid = accelerator ? metal::DispatchSize{(n + 31) / 32, (rows + 63) / 64, 1}
                                               : metal::DispatchSize{n / 8, (rows + 31) / 32, 1};
  graph.add(kernel, {std::move(input), weights.plane0, std::move(output)},
            GgufFloatParams{rows, k, n, outStride, outOffset}, grid, {accelerator ? 128u : 512u, 1, 1});
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
