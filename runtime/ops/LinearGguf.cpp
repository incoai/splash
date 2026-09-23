// GGUF projections: plan policy and dispatch (kernels/shared/gguf_linear.metal,
// kernels/decode/linear_gguf_sgmatrix.metal).
#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::ops {
namespace {

// Decode tiles: 64 output columns per threadgroup, two simdgroups of 32
// columns each. Prefill tiles: 64 columns, four simdgroups of 32 rows.
constexpr uint32_t kDecodeTileColumns = 64;
constexpr uint32_t kDecodeThreads = 64;
constexpr uint32_t kPrefillTileColumns = 64;
constexpr uint32_t kPrefillThreads = 128;
constexpr uint32_t kPrefillRows = 128;

std::string decodeKernel(const char *format, uint32_t rows, char epilogue) {
  return std::string("gguf_decode_") + format + "_m" + std::to_string(rows) + "_" + epilogue;
}
std::string prefillKernel(const std::string &family, const char *format) {
  return family + "_" + format + "_r32_sg4_n64_k64_p1";
}

// K splits of a decode tile, one rule for both tiles. A tier asks for more
// partitions while the grid holds fewer than `threadgroups` threadgroups per
// core and each partition would still keep `inputs` inputs; the split count
// doubles, up to eight, while some tier asks. Decode K is a multiple of 256,
// so eight partitions always hold whole 32-input groups. The rule ignores the
// batch width: bounds that depended on it did not pay on either family.
struct SplitTier {
  uint32_t threadgroups;
  uint32_t inputs;
};
uint32_t decodeSplits(uint32_t n, uint32_t k, uint32_t cores, std::span<const SplitTier> tiers) {
  const uint64_t grid = n / kDecodeTileColumns;
  uint32_t splits = 1;
  const auto asks = [&](const SplitTier &t) {
    return grid * splits < uint64_t{t.threadgroups} * cores && k / (2 * splits) >= t.inputs;
  };
  while (splits < 8 && std::any_of(tiers.begin(), tiers.end(), asks)) splits *= 2;
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
constexpr SplitTier kSimdgroupTiers[] = {{4, 256}, {32, 1024}};

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

const metal::MetalBuffer &plane1(const GgufSegment &s) { return s.plane1 ? s.plane1 : s.meta; }

// The request lanes one decode dispatch fuses, whatever its tile height.
void recordLanes(Q4DispatchStats *stats, uint32_t rows) {
  if (!stats || rows <= SPLASH_TARGET_VERIFY_ROWS) return;
  const uint32_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  stats->fusedSourceOperations += lanes;
  if (lanes == 2) ++stats->m16Dispatches;
  else if (lanes == 3) ++stats->m24Dispatches;
  else ++stats->m32Dispatches;
}

// Segments tile the leading columns of the destination rows; the columns past
// the last segment are padding no kernel writes. The 35B GGUF's packed GDN row
// is 12544 columns (the affine layout's), its qkv|z|alpha-beta segments 12352.
void requireSegments(const Q4Projection &p, LinearMatrix matrix) {
  uint32_t covered = 0;
  for (const GgufSegment &s : p.gguf) {
    if (s.inputSize != matrix.inputSize || s.columnOffset != covered ||
        s.outputSize % kDecodeTileColumns)
      throw std::invalid_argument("GGUF segments do not tile the projection");
    covered += s.outputSize;
  }
  if (covered > matrix.outputSize)
    throw std::invalid_argument("GGUF segments exceed the projection");
}

// Columns of the segments, which the fused kernels' grids cover.
uint32_t segmentColumns(const Q4Projection &p) {
  uint32_t columns = 0;
  for (const GgufSegment &s : p.gguf) columns += s.outputSize;
  return columns;
}

} // namespace

// Apple9 runs matrix operations on the FP32 pipe, so the exact register
// kernel beats staging (scratchpad DESIGN).
LinearTile Q4Linear::ggufDecodeTile() const noexcept {
  return appleGpuFamily_ == 9 ? LinearTile::GgufSimdgroup : LinearTile::GgufStaged;
}

LinearConfig Q4Linear::ggufBaseline(LinearWorkload w) const {
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
    return w.rows <= SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS
        ? LinearConfig{LinearTile::GgufStaged, 0, LinearSimdgroups::Two,
                       decodeSplits(n, k, gpuCores_, kStagedTiers)}
        : LinearConfig{LinearTile::GgufStaged, 0, LinearSimdgroups::Four};
  if (ggufDecodeTile() == LinearTile::GgufSimdgroup)
    return {LinearTile::GgufSimdgroup, n / kDecodeTileColumns, LinearSimdgroups::Four,
            decodeSplits(n, k, gpuCores_, kSimdgroupTiers)};
  return {LinearTile::GgufStaged, n / kDecodeTileColumns, LinearSimdgroups::Two,
          decodeSplits(n, k, gpuCores_, kStagedTiers)};
}

void Q4Linear::addGguf(metal::CommandGraph &graph, const LinearBuffers &b,
                       const Q4Projection &p, const LinearPlan &plan,
                       const Q4Projection *gate, Q4DispatchStats *stats) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  if ((config.tile != LinearTile::GgufStaged && config.tile != LinearTile::GgufSimdgroup) ||
      w.quant != QuantFamily::Gguf)
    throw std::invalid_argument("GGUF projection requires a GGUF plan");
  const auto [n, k] = w.matrix;
  requireSegments(p, w.matrix);
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!gate || gate->gguf.empty()) throw std::invalid_argument("GGUF gate projection is missing");
    requireSegments(*gate, w.matrix);
  } else if (gate) {
    throw std::invalid_argument("unexpected GGUF gate projection");
  }
  if (std::any_of(p.gguf.begin(), p.gguf.end(), [](const GgufSegment &s) { return s.isFloat(); })) {
    // The quantized segments, which precede the float ones, run the plan's tiles.
    addGgufFloatSegments(graph, b, p, plan);
    Q4Projection quantized = p;
    std::erase_if(quantized.gguf, [](const GgufSegment &s) { return s.isFloat(); });
    if (!quantized.gguf.empty()) addGguf(graph, b, quantized, plan, gate, stats);
    return;
  }
  const uint32_t rows = plan.storageRows();
  const auto need = [&](const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
    if (buffer.sizeBytes() < bytes)
      throw std::invalid_argument(std::string("GGUF ") + what + " buffer holds " +
                                  std::to_string(buffer.sizeBytes()) + " bytes, needs " +
                                  std::to_string(bytes) + " (rows " + std::to_string(rows) +
                                  ", n " + std::to_string(n) + ")");
  };
  need(b.input, uint64_t{rows} * k * 2, "input");
  need(b.output, uint64_t{rows} * n * 2, "output");
  if (w.epilogue == LinearEpilogue::Residual) need(b.residual, uint64_t{rows} * n * 2, "residual");
  need(b.gateScratch, plan.gateScratchBytes(), "gate scratch");
  if (config.tile == LinearTile::GgufSimdgroup) {
    addGgufSimdgroup(graph, b, p, plan, gate);
  } else if (config.simdgroups == LinearSimdgroups::Two) {
    const LinearScratchSize scratch = plan.scratchSize();
    need(b.scratch.partials, scratch.partials, "partials");
    need(b.scratch.counters, scratch.counters, "counters");
    addGgufStaged(graph, b, p, plan, gate);
  } else {
    // Prefill chunks of more than 32 rows: one dispatch per segment over
    // 128-row tiles; rows past w.rows stay inside the budget-sized prefill
    // buffers, and the simdgroups of a tile that only hold them skip their
    // matmuls.
    const metal::MetalBuffer aux = w.epilogue == LinearEpilogue::Residual ? b.residual
                                 : w.epilogue == LinearEpilogue::UpWithGate ? b.gateScratch
                                                                             : b.output;
    const char epilogue = w.epilogue == LinearEpilogue::None       ? 'a'
                        : w.epilogue == LinearEpilogue::Residual   ? 'r'
                                                                   : 'g';
    for (const GgufSegment &s : p.gguf) {
      std::vector<metal::MetalBuffer> bindings{b.input, s.plane0, plane1(s), s.meta, b.output};
      if (w.epilogue != LinearEpilogue::None) bindings.push_back(aux);
      graph.add(prefillKernel(std::string("pf") + epilogue, s.format), std::move(bindings),
                GgufPrefillParams{s.outputSize, k, w.rows, n, s.columnOffset},
                {rows / kPrefillRows, s.outputSize / kPrefillTileColumns, 1}, {kPrefillThreads, 1, 1});
    }
  }
  if (w.phase == LinearPhase::Decode) recordLanes(stats, w.rows);
}

// The staged decode tiles, for decode and prefill chunks of up to 32 rows:
// every row of the plan's storage in each threadgroup's tile, grid (64-column
// tiles, K splits). Decode runs one dispatch per projection: single tensors
// their format's kernel, fused projections (qkv|z|ab, q|k|v) every segment
// in one dispatch, gate/up a gate pass into the gate scratch and an up pass
// whose epilogue applies silu(gate) to the bf16 up value, as on Apple9 (the
// fused gate/up kernel it replaces was within -4..+2% on the 27B gate/up at
// 10-40 cores). Prefill chunks run one dispatch per segment.
void Q4Linear::addGgufStaged(metal::CommandGraph &graph, const LinearBuffers &b,
                             const Q4Projection &p, const LinearPlan &plan,
                             const Q4Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
  const uint32_t rows = plan.storageRows(), splits = config.splits;
  // One partition never touches the partials and counters: the output stands in.
  const metal::MetalBuffer partials = splits > 1 ? b.scratch.partials : b.output;
  const metal::MetalBuffer counters = splits > 1 ? b.scratch.counters : b.output;
  const auto tensor = [&](const GgufSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(decodeKernel(s.format, rows, epilogue),
              {b.input, s.plane0, plane1(s), s.meta, output, partials, counters, aux},
              GgufDecodeParams{k, splits, n, s.columnOffset}, {s.outputSize / kDecodeTileColumns, splits, 1},
              {kDecodeThreads, 1, 1});
  };
  if (w.phase == LinearPhase::Prefill) {
    const char epilogue = w.epilogue == LinearEpilogue::None ? 'a' : w.epilogue == LinearEpilogue::Residual ? 'r' : 'g';
    const metal::MetalBuffer aux = w.epilogue == LinearEpilogue::Residual ? b.residual
                                 : w.epilogue == LinearEpilogue::UpWithGate ? b.gateScratch
                                                                             : b.output;
    for (const GgufSegment &s : p.gguf) tensor(s, epilogue, b.output, aux);
    return;
  }
  switch (w.epilogue) {
  case LinearEpilogue::GateUp:
    if (gate->gguf.size() != 1 || p.gguf.size() != 1)
      throw std::invalid_argument("GGUF gate/up requires single tensors");
    tensor(gate->gguf.front(), 'a', b.gateScratch, b.gateScratch);
    tensor(p.gguf.front(), 'g', b.output, b.gateScratch);
    return;
  case LinearEpilogue::Residual:
    if (p.gguf.size() != 1) throw std::invalid_argument("GGUF residual projection requires a single tensor");
    tensor(p.gguf.front(), 'r', b.output, b.residual);
    return;
  case LinearEpilogue::UpWithGate: throw std::invalid_argument("GGUF decode has no up-with-gate projection");
  case LinearEpilogue::None: break;
  }
  if (p.gguf.size() == 1) {
    tensor(p.gguf.front(), 'a', b.output, b.output);
    return;
  }
  if (p.gguf.size() > 3) throw std::invalid_argument("GGUF fused projection needs <= 3 segments");
  // Dispatch order is tile order: segments with the most bytes per tile
  // first, so their threadgroups do not form the tail (alpha/beta are Q8_0).
  std::vector<const GgufSegment *> order;
  for (const GgufSegment &s : p.gguf) order.push_back(&s);
  const auto bitsPerWeight = [](const GgufSegment &s) {
    return (s.p0 + s.p1) * 8.0 / 32.0 + s.metaBytes * 8.0 / (32.0 * s.metaGroups);
  };
  std::stable_sort(order.begin(), order.end(), [&](const GgufSegment *a, const GgufSegment *c) {
    return bitsPerWeight(*a) > bitsPerWeight(*c);
  });
  GgufDecodeFusedParams params{k, splits, n, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  std::vector<metal::MetalBuffer> bindings{b.input};
  for (size_t i = 0; i < 3; ++i) {
    const GgufSegment &s = *order[std::min(i, order.size() - 1)];
    if (i < order.size()) {
      params.cols[i] = s.outputSize;
      params.fmt[i] = s.formatId;
      params.offset[i] = s.columnOffset;
    }
    bindings.insert(bindings.end(), {s.plane0, plane1(s), s.meta});
  }
  bindings.insert(bindings.end(), {b.output, partials, counters});
  graph.add("gguf_decode_fused_m" + std::to_string(rows), std::move(bindings), params,
            {segmentColumns(p) / kDecodeTileColumns, splits, 1}, {kDecodeThreads, 1, 1});
}

// All lanes in each threadgroup. Single tensors run their format's kernel;
// fused projections (qkv|z|ab, q|k|v) run every segment in one dispatch.
// Gate/up runs as a gate pass into the gate scratch and an up pass whose
// epilogue applies silu(gate) to the bf16 up value, as the fused kernels do.
void Q4Linear::addGgufSimdgroup(metal::CommandGraph &graph, const LinearBuffers &b,
                                const Q4Projection &p, const LinearPlan &plan,
                                const Q4Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  const LinearScratchSize size = plan.scratchSize();
  const auto need = [](const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
    if (buffer.sizeBytes() < bytes)
      throw std::invalid_argument(std::string("GGUF simdgroup ") + what + " scratch is below the plan");
  };
  need(b.scratch.input, size.input, "table");
  need(b.scratch.sums, size.sums, "sums");
  need(b.scratch.partials, size.partials, "partials");
  need(b.scratch.counters, size.counters, "counters");
  if (b.prepared.layout != LinearInput::Table16 || !b.prepared.source.sameView(b.input))
    graph.add("decode_linear_gguf_prepare", {b.input, b.scratch.input, b.scratch.sums}, k,
              {k / 32, lanes, 1}, {128, 1, 1});
  const metal::DispatchSize grid{segmentColumns(p) / kDecodeTileColumns, config.splits, 1};
  const std::string suffix = "_l" + std::to_string(lanes);
  if (p.gguf.size() > 1) {
    if (p.gguf.size() > 3 || w.epilogue != LinearEpilogue::None)
      throw std::invalid_argument("GGUF fused projection needs <= 3 segments and no epilogue");
    GgufDecodeFusedParams params{k, config.splits, n, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, b.scratch.sums};
    for (size_t i = 0; i < 3; ++i) {
      const GgufSegment &s = p.gguf[std::min(i, p.gguf.size() - 1)];
      if (i < p.gguf.size()) {
        params.cols[i] = s.outputSize;
        params.fmt[i] = s.formatId;
        params.offset[i] = s.columnOffset;
      }
      bindings.insert(bindings.end(), {s.plane0, plane1(s), s.meta});
    }
    bindings.insert(bindings.end(), {b.output, b.scratch.partials, b.scratch.counters});
    graph.add("decode_linear_gguf_sg_fused" + suffix, std::move(bindings), params, grid, {128, 1, 1});
    return;
  }
  const auto tensor = [&](const GgufSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(std::string("decode_linear_gguf_sg_") + s.format + suffix + "_" + epilogue,
              {b.scratch.input, b.scratch.sums, s.plane0, plane1(s), s.meta, output, b.scratch.partials,
               b.scratch.counters, aux},
              GgufDecodeParams{k, config.splits, n, s.columnOffset}, grid, {128, 1, 1});
  };
  switch (w.epilogue) {
  case LinearEpilogue::None: tensor(p.gguf.front(), 'a', b.output, b.output); break;
  case LinearEpilogue::Residual: tensor(p.gguf.front(), 'r', b.output, b.residual); break;
  case LinearEpilogue::GateUp:
    if (gate->gguf.size() != 1) throw std::invalid_argument("GGUF gate/up requires single tensors");
    tensor(gate->gguf.front(), 'a', b.gateScratch, b.gateScratch);
    tensor(p.gguf.front(), 'g', b.output, b.gateScratch);
    break;
  case LinearEpilogue::UpWithGate: throw std::invalid_argument("GGUF decode has no up-with-gate projection");
  }
}

// Float segments of a projection (GgufSegment::isFloat): a float projection
// over the step's rows, whatever the rows of the plan's tiles.
void Q4Linear::addGgufFloatSegments(metal::CommandGraph &graph, const LinearBuffers &b,
                                    const Q4Projection &p, const LinearPlan &plan) const {
  const LinearWorkload w = plan.workload();
  if (w.epilogue != LinearEpilogue::None) throw std::invalid_argument("GGUF float segments take no epilogue");
  for (const GgufSegment &s : p.gguf)
    if (s.isFloat())
      addGgufFloat(graph, b.input, s, b.output, w.rows, w.matrix.outputSize, s.columnOffset, FloatOutput::BFloat16,
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
FloatTile Q4Linear::ggufFloatTile(uint32_t rows, uint32_t outputSize) const noexcept {
  const uint64_t tiles = uint64_t{(rows + 63) / 64} * ((outputSize + 31) / 32);
  return appleGpuFamily_ != 9 && rows >= 16 && 2 * tiles >= uint64_t{3} * gpuCores_ ? FloatTile::NeuralAccelerator
                                                                                     : FloatTile::Simdgroup;
}

bool GgufSegment::isFloat() const noexcept { return type == GGUF_TYPE_F32; }

// Simdgroup: 8 columns of 32 rows per threadgroup of 16 simdgroups. Neural
// accelerator: 32 columns of 64 rows per threadgroup of 4 simdgroups, at least
// 16 rows and K a multiple of 32 (kernels/shared/gguf_float.metal).
void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, const GgufSegment &weights,
                  metal::MetalBuffer output, uint32_t rows, uint32_t outStride, uint32_t outOffset,
                  FloatOutput type, FloatTile tile) {
  const uint32_t n = weights.outputSize, k = weights.inputSize;
  const uint64_t element = type == FloatOutput::Float32 ? sizeof(float) : sizeof(uint16_t);
  const bool accelerator = tile == FloatTile::NeuralAccelerator;
  if (!weights.isFloat() || !rows || !n || n % 8 || !k || k % 8 || outOffset + uint64_t{n} > outStride ||
      (accelerator && (rows < 16 || k % 32)) || weights.plane0.sizeBytes() < uint64_t{n} * k * sizeof(float) ||
      input.sizeBytes() < uint64_t{rows} * k * 2 ||
      output.sizeBytes() < (uint64_t{rows - 1} * outStride + outOffset + n) * element)
    throw std::invalid_argument("invalid GGUF float projection");
  const std::string kernel = std::string(accelerator ? "gguf_float_na_" : "gguf_float_") +
                             (type == FloatOutput::Float32 ? "f32" : "bf16");
  const metal::DispatchSize grid = accelerator ? metal::DispatchSize{(n + 31) / 32, (rows + 63) / 64, 1}
                                               : metal::DispatchSize{n / 8, (rows + 31) / 32, 1};
  graph.add(kernel, {std::move(input), weights.plane0, std::move(output)},
            GgufFloatParams{rows, k, n, outStride, outOffset}, grid, {accelerator ? 128u : 512u, 1, 1});
}

} // namespace splash::ops
