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
// the 32-row tile over four lanes of storage (a one-lane step may run the
// 16-row tile over two, LinearConfig::sixteenRows); a prefill chunk of 33 to
// 64 rows runs the 64-row tile (Linear::ggufBaseline).
constexpr uint32_t stagedTileRows(uint32_t rows) noexcept {
  return rows <= 8 ? 8 : rows <= 16 ? 16 : rows <= 32 ? 32 : 64;
}

std::string decodeKernel(const char *format, uint32_t rows, char epilogue) {
  return std::string("gguf_decode_") + format + "_m" + std::to_string(rows) + "_" + epilogue;
}
std::string prefillKernel(const char *format, char epilogue) {
  return std::string("gguf_prefill_") + format + "_" + epilogue;
}

// K splits of the register tile, and of Apple9's staged tile, by tiers. A
// tier asks for more partitions, up to its `splits`, while the grid holds
// fewer than `threadgroups` threadgroups per core and each partition would
// still keep `inputs` inputs; the split count doubles, in whole 32-input
// groups per partition, while some tier asks. The tiers ignore the batch
// width: bounds that depended on it did not pay on Apple9. These tiles run
// their matrix operations on the FP32 pipe (Apple9) or as exact bf16
// register operations, without a residency step in their time (a 10-core M4
// over 2 to 40 register threadgroups per core): the Apple10 staged tiles'
// wave law (apple10Staged) does not describe them.
struct SplitTier {
  uint32_t threadgroups;
  uint32_t inputs;
  uint32_t splits = 8;
};
uint32_t decodeSplits(uint32_t n, uint32_t k, uint32_t cores, std::span<const SplitTier> tiers) {
  const uint64_t grid = n / GGUF_TILE_COLUMNS;
  uint32_t splits = 1;
  const auto asks = [&](const SplitTier &t) {
    return splits < t.splits && grid * splits < uint64_t{t.threadgroups} * cores && k / (2 * splits) >= t.inputs;
  };
  while ((k / 32) % (2 * splits) == 0 && std::any_of(tiers.begin(), tiers.end(), asks)) splits *= 2;
  return splits;
}

// Register tile (128 threads). A core holds four of its threadgroups at
// once, so the tile's time steps with every four threadgroups per core.
// Below one wave a core must fill it, down to 256 inputs per partition, whole
// coefficient units in every format (quant_register_unit); below eight waves
// more threadgroups shrink the last wave's tail while partitions of 1024
// inputs amortize the partial sums (both measured on a 40-core M3 Max). Both
// stop at eight partitions. A grid that eight leave below three threadgroups
// per core takes sixteen while partitions keep 512 inputs, a tier measured on
// Apple10 alone (Apple9 takes it unmeasured): Qwen3.8-Flash-Next's
// hyper-connection down projection, Q8_0 320 x 10240 on five column tiles,
// then takes 11-17% less time at one to four lanes on the 20-core M5 Pro and
// 6-20% less on the 32-core M5 Max, and 32 partitions more than 16 on both.
// On the M5 Max, sixteen take 4% less to 10% more 512 columns wide (two
// threadgroups per core at eight), up to 18% more 832 wide (3.25), 4-15% more
// on the same 320 x 10240 in BF16, twice Q8_0's bytes (Flash-Next loads such
// tensors as Q8_0), and 22-38% more on Q4_K 512 x 5120, whose partitions
// would keep 320 inputs. From five threadgroups per core a 10-core M4 takes
// 3-7% less time at one lane on one or two partitions than on the tiers' four
// (MLX 4-bit, 5120 inputs), where at two to four lanes the tiers' are the
// fastest: telling one lane apart needs an Apple9 GPU of more cores.
constexpr SplitTier kRegisterTiers[] = {{4, 256}, {32, 1024}, {3, 512, 16}};

// Apple9's staged tile takes the register tile's tiers of eight partitions
// (the first two): on a 40-core M3 Max over the 27B and 35B dense shapes at
// one to four lanes they come within 0-9% of each shape's fastest split (20%
// at one lane on 2048 x 512, a 0.012 ms projection), where a fitted tier of
// six threadgroups per core is 13-27% slower on 17408 x 5120 and
// 12288 x 5120 and 50% on 2048 x 512.
constexpr std::span<const SplitTier> kApple9StagedTiers = std::span(kRegisterTiers).first(2);

// The Apple10 staged decode tiles' wave law. A tile kernel's threadgroups each
// walk K / 32 / splits steps over a 64-column tile's weights (a one-pass
// gate/up pair's 32 columns of both tensors). A core runs `resident` of them at
// once, so a grid runs in waves of that many per core; it streams its share of
// DRAM from `saturating` of them on, so a grid's first wave of fewer takes as
// long as one of that many, waiting on its chains' latency; a later wave starts
// as the one before drains, which hides part of that latency, and takes as long
// as at least `following`. Measured on the 32-core M5 Max, the 20-core M5 Pro
// and the 12-core M6 (gguf-projection-benchmark over 2 to 40 threadgroups per
// core, MLX 4-bit, Q4_K to Q8_0, IQ3_XXS and IQ2_XS, 2048 and 5120 inputs, one
// to four lanes): time per byte steps up past 10 of the two-simdgroup tile's
// threadgroups per core at one and two lanes on the M5 Max and the M6 (M5 Max,
// one lane: 140 us at 10, 190 at 11 for 1.1 times the bytes), past 8 or 9 on
// the M5 Pro, and past 16 of the pair's at every lane count (229 us at 16, 334
// at 17). The 32-row tile of three and four lanes steps past 9 on the M5 Max, 8
// on the M5 Pro and 11 on the M6, which the law leaves out: it takes 10 at
// every row count. A lone chain of 160 steps takes as long as 5.4 and 9
// threadgroups' bytes at DRAM's rate, and one threadgroup per core past a full
// wave adds 2.7-3.5 of the tile's threadgroups' time (M5 Max and M6, one and
// two lanes) and 6-8.5 of the pair's (M5 Max and M5 Pro). `saturating` and
// `following` are not those measurements but a grid search's fit, with the
// residencies, to the fastest split of every measured case (the 16-row tile
// where one lane takes it): 6 and 10 for the measured 5.4 and 9, 2 and 3 for
// the measured 2.7-3.5 and 6-8.5; they stand in for what the law leaves out,
// such as the 32-row residency and the pair's per-byte advantage
// (apple10Staged). Floors at the measured trailing waves (3 and 7) take as much
// time in total on the same cases but leave 18 of them more than 3% slower than
// the earlier rules, against 5 (Q8_0 and Q6_K split at 12 threadgroups per
// core, three- and four-lane pairs at 10 per pass). On those cases, which
// fitted it (in-sample), the law's choices take 1.1%, 1.6% and 0.05% less time
// than the earlier rules' in total on the M5 Max, the M5 Pro and the M6, 29, 30
// and 3 cases 3-33% less, and 2, 2 and 1 cases 4-12% more (a 34-us pair of four
// threadgroups per core over 2048 inputs, two lanes). The sweeps cover 2048 and
// 5120 inputs and pairs in MLX 4-bit: the down projections' 17408 inputs, pairs
// in other formats and low-bit pairs take the law unmeasured.
struct StagedOccupancy final {
  uint32_t resident, saturating, following;
};
// gguf_decode_<format>_m<rows>_<e> and gguf_decode_fused_m<rows>: two
// simdgroups, 64 columns.
constexpr StagedOccupancy kStagedTile{10, 6, 2};
// gguf_decode_<format>_m<rows>_gate_up: one simdgroup, 32 columns of each
// tensor.
constexpr StagedOccupancy kGateUpPairTile{16, 10, 3};
static_assert(kStagedTile.resident >= kStagedTile.saturating && kGateUpPairTile.resident >= kGateUpPairTile.saturating,
              "a full wave streams at DRAM's rate");
// Staged partitions keep 512 inputs, eight at most.
constexpr uint32_t kStagedPartitionInputs = 512, kStagedMaximumSplits = 8;
// The most bits per weight a format stores, in sixteenths (BF16 and F16).
constexpr uint32_t kMostWeightBits16 = [] {
  uint32_t most = 0;
  for (const QuantFormat &f : kQuantFormats) most = std::max(most, f.block_bytes * 128 / f.block_elements);
  return most;
}();

// The time of a dispatch of `groups` threadgroups of a staged tile, `steps`
// K steps each, on `cores` cores, in bits a core streams at its share of DRAM,
// times the cores: every wave but the last holds `resident` threadgroups on
// each core and the last the rest, each threadgroup streaming the weights of
// a step of 64 columns, of `bits16` sixteenths of a bit each, per step; the
// last wave takes at least `saturating` threadgroups' time when it is the
// grid's only one and `following`'s otherwise. A split grid's threadgroups
// each also store their fp32 partials of `rows` rows, which the last
// partition reads back.
uint64_t stagedTime(uint64_t groups, uint32_t steps, uint32_t splits, uint32_t rows, uint32_t bits16,
                    uint32_t cores, StagedOccupancy tile) {
  const uint64_t perCore = (groups + cores - 1) / cores, waves = (perCore + tile.resident - 1) / tile.resident;
  const uint64_t full = (waves - 1) * tile.resident * cores,
                 floor = uint64_t{waves > 1 ? tile.following : tile.saturating} * cores;
  const uint64_t stepBits = uint64_t{GGUF_TILE_COLUMNS} * 32 * bits16 / 16,
                 partialBits = splits > 1 ? 2 * uint64_t{rows} * GGUF_TILE_COLUMNS * 32 : 0;
  return steps * stepBits * (full + std::max(groups - full, floor)) + groups * partialBits;
}

// How an Apple10 staged decode plan of a gate/up pair may run: in two passes,
// in one, or in either.
enum class Passes : uint8_t { Two, One, Either };

// The Apple10 staged decode configuration of `rows` rows of an n x k
// projection of `bits16` sixteenths of a bit per weight on `cores` cores, a
// gate/up pair's two tensors when `gateUp` in `passes`: the K split, and
// passes, of least time by the wave law; ties keep fewer partitions, then one
// pass. The law tells the pass modes apart only where a last wave falls below
// its floor: on the 32-core M5 Max the 27B gate/up pair puts 17 one-pass
// threadgroups on each core, the 17th of which runs a second wave alone (333
// us against 245 in two passes of 8.5 per core, one lane). Where both modes'
// last waves clear their floors it predicts the same time for both, and the
// tie takes one pass: in 133 of the 152 tied cases of the M5 Max's and the
// M5 Pro's sweeps (one to four lanes) one pass takes up to 22% less time, a
// 20-core M5 Pro's 27B pair (27.2 per core) 8% (395 us against 430), a
// 40-core M5 Max's (13.6) 8-9%; at 20 and 36 one-pass threadgroups per core
// it takes up to 20% more, most at one lane. A 16-core M5 Pro puts 34 (16,
// 16 and 2), where the law takes two passes at every lane count: emulated by
// width on the M5 Max they take 9-12% less time than one unsplit pass at one
// and two lanes but 7% more at three and four, where the 32-row tile's lower
// residency slows them. Below 16 cores a pair runs two passes whatever the
// law (spreadsStagedDecode): a 12-core M6 took 2-11% more in one.
LinearConfig apple10Staged(uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16, uint32_t cores, bool gateUp,
                           Passes passes) {
  const uint32_t tileRows = stagedTileRows(rows);
  LinearConfig best{.tile = LinearTile::GgufStaged};
  uint64_t least = UINT64_MAX;
  for (uint32_t splits = 1; splits <= kStagedMaximumSplits && (k / 32) % splits == 0 &&
                            (splits == 1 || k / splits >= kStagedPartitionInputs);
       splits *= 2) {
    const uint32_t steps = k / 32 / splits;
    const uint64_t tiles = uint64_t{n / GGUF_TILE_COLUMNS} * splits;
    if (gateUp && passes != Passes::Two) {
      const uint64_t time = stagedTime(2 * tiles, steps, splits, tileRows, bits16, cores, kGateUpPairTile);
      if (time < least) least = time, best = {.tile = LinearTile::GgufStaged, .splits = splits, .oneGateUpPass = true};
    }
    if (!gateUp || passes != Passes::One) {
      const uint64_t time = (gateUp ? 2 : 1) * stagedTime(tiles, steps, splits, tileRows, bits16, cores, kStagedTile);
      if (time < least) least = time, best = {.tile = LinearTile::GgufStaged, .splits = splits};
    }
  }
  return best;
}

// The decode tile configurations over an n x k matrix on `cores` cores.
LinearConfig registerDecode(uint32_t n, uint32_t k, uint32_t cores) {
  return {.tile = LinearTile::GgufRegister, .splits = decodeSplits(n, k, cores, kRegisterTiers)};
}
// The staged tile's, of `rows` rows of `bits16` sixteenths of a bit per
// weight (a prefill chunk's, or a decode step's of a gate/up pair when
// `gateUp` in `passes`).
LinearConfig stagedDecode(uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16, uint32_t cores,
                          GpuFamilyClass family, bool gateUp = false, Passes passes = Passes::Two) {
  if (family == GpuFamilyClass::Apple9)
    return {.tile = LinearTile::GgufStaged, .splits = decodeSplits(n, k, cores, kApple9StagedTiers)};
  return apple10Staged(n, k, rows, bits16 ? bits16 : kMostWeightBits16, cores, gateUp, passes);
}
// The bits per weight of a plan's projections in sixteenths (a gate/up
// pair's two together), or without them the shape's.
uint32_t weightBits16(std::span<const Projection *const> projections, ProjectionShape shape) {
  uint32_t sum = 0, count = 0;
  for (const Projection *p : projections)
    if (p) sum += p->shape().weightBits16, ++count;
  return count ? sum / count : shape.weightBits16;
}

// Whether every quantized segment of the projections satisfies `holds`.
template <class Predicate>
bool everyQuantizedSegment(std::span<const Projection *const> projections, Predicate holds) {
  for (const Projection *p : projections)
    if (p)
      for (const QuantizedSegment &s : p->blocks().segments)
        if (!s.isFloat() && !holds(s)) return false;
  return true;
}
// Whether the register tile takes a segment (quant_register_tensor).
bool registerTileTakes(const QuantizedSegment &s) { return quant_register_tensor(s.formatId, s.inputSize); }
// Whether a segment stores at least four bits per weight (quant_four_bits).
bool fourBitsPerWeight(const QuantizedSegment &s) { return quant_four_bits(s.formatId); }

// Apple10 runs decode steps on the register tile on narrow grids: where the
// grid, a threadgroup per 64 columns (a gate/up pair's that may run in one
// pass staged) and lane, holds fewer than one and a half threadgroups per core,
// half that in formats of fewer than four bits per weight. There the staged
// tile's 64 threads per column tile leave its cores' loads too few and the
// register tile's 128 run more of them; its time grows with each lane, where
// the staged tile's matmuls take up to 16 rows at the cost of 8, and with its
// decode, where fewer bits leave the staged tile less to stream. Register
// against staged time on one lane's 8-row tile, best split of each tile:
// - 32-core M5 Max: 13-43% less below the bound (BF16 and Q8_0 320 x 10240,
//   Q4_K and Q5_K 512 x 2560, Q8_0 512 x 2048, Q6_K 1024 x 5120, Q4_K and
//   Q6_K 2048 x 4096); at 2560 x 6144 (1.25 per core) 6-9% less in Q4_K, Q5_K,
//   Q6_K, IQ4_XS, IQ4_NL and MXFP4, BF16 2%, Q8_0 the same, and 2-20% more in
//   Q3_K, IQ3_S, IQ3_XXS, Q2_K, Q2_0, IQ2_XS, IQ2_XXS and IQ1_M, which take
//   7-41% less below theirs (IQ2_XXS and Q3_K 1024 x 5120, IQ3_XXS, Q3_K and
//   Q2_K 512 x 2560, Q3_K 320 x 10240) and as much at 1.0 (Q3_K 2048 x 4096);
// - 20-core M5 Pro: 18-33% less at 320 and 512 columns, 4-9% more past the
//   bound (Q4_K and Q6_K 2560 x 6144, Q4_K 2048 x 4096); in fewer bits 23-33%
//   less below theirs (Q3_K, Q2_K, Q2_0 512 x 2560, Q3_K 320 x 10240), 3% and
//   14% more past (Q3_K and IQ2_XXS 1024 x 5120).
// A one-pass gate/up pair of 1024 x 5120 takes 3% (Q4_K) and 23% (IQ2_XXS)
// less time staged than in the register tile's two passes on the M5 Pro, 11%
// more (Q4_K, 1.0 per core) and 10% less (IQ2_XXS) on the M5 Max; one of
// 512 x 2560 (Q4_K) takes 17% and 25% less in the two.
constexpr bool apple10Registers(uint32_t rows, uint32_t columns, uint32_t cores, bool fourBits) noexcept {
  const uint32_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  return uint64_t{fourBits ? 2u : 4u} * (columns / GGUF_TILE_COLUMNS) * lanes < uint64_t{3} * cores;
}

// Whether the staged decode tiles on `cores` cores of `family` spread their
// walks and may run a gate/up pair of one format in one pass
// (Linear::ggufBaseline): Apple10 and later from 16 cores.
constexpr bool spreadsStagedDecode(GpuFamilyClass family, uint32_t cores) noexcept {
  return family == GpuFamilyClass::Apple10 && cores >= 16;
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
// A segment the register tile cannot take (quant_register_tensor) stages the
// plan whatever its rows.
bool apple9Stages(LinearWorkload w, std::span<const Projection *const> projections) {
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  bool quantized = false, faster = true;
  for (const Projection *p : projections) {
    if (!p) continue;
    for (const QuantizedSegment &s : p->blocks().segments) {
      if (s.isFloat()) continue;
      if (!quant_register_tensor(s.formatId, s.inputSize)) return true;
      faster &= apple9StagesFormat(s.formatId) || (s.formatId == GGUF_FMT_Q2K && lanes >= 2);
      quantized = true;
    }
  }
  return quantized && faster && stagedTileRows(w.rows) == w.rows;
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

LinearConfig Linear::ggufBaseline(LinearWorkload w, std::span<const Projection *const> projections,
                                  ProjectionShape shape) const {
  const auto [n, k] = w.matrix;
  // Prefill: 128-row tiles. A chunk of up to 32 rows runs the staged tile
  // of its rows (8, 16 or 32, two simdgroups) with the decode split law:
  // the same half stage and matmul rows, so its outputs equal the prefill
  // tile's up to the K split's fp32 reassociation (gguf-projection checks it),
  // and each simdgroup streams its own 32 columns instead of four 8-row
  // simdgroups sharing a stage. Unsplit, on a 17408 x 5120 Q4_K projection
  // that is 1.8-2.9x faster on a 16-core M5 Pro (its neural accelerator pads
  // 8 rows to 16) and 1.1-2.8x on a 40-core M3 Max; the 27B down and output
  // projections split in two gain 24-37% more on the 16-core M5 Pro.
  //
  // Past 32 rows the prefill tile takes about as long for 33 rows as for 64:
  // 0.58 ms on that Q4_K projection on the 16-core M5 Pro, against 0.25 ms
  // for the 32-row staged tile. On Apple10 a chunk of up to
  // kMaximumStagedPrefillRows rows runs the staged tile's 64-row height,
  // which reads the weights once: 0.48 ms there, and 6-43% less than the
  // prefill tile over the 27B's Q4_K, MLX 4-bit and Q8_0 projections at the
  // split law's choice (gguf-projection-benchmark prefill=33,48,64). Apple9
  // is unmeasured and keeps the prefill tile. A split-free workload keeps one
  // K pass, whose outputs equal the prefill tile's bit for bit, and the
  // prefill tile past kMaximumDecodeTileRows rows.
  if (w.phase == LinearPhase::Prefill) {
    if (w.splitFree)
      return LinearConfig{.tile = w.rows <= kMaximumDecodeTileRows ? LinearTile::GgufStaged : LinearTile::GgufPrefill};
    const bool staged = w.rows <= kMaximumDecodeTileRows ||
                        (family_ == GpuFamilyClass::Apple10 && w.rows <= kMaximumStagedPrefillRows);
    return staged ? stagedDecode(n, k, w.rows, weightBits16(projections, shape), gpuCores_, family_)
                  : LinearConfig{.tile = LinearTile::GgufPrefill};
  }
  // Apple9 runs matrix operations on the FP32 pipe, so the exact register
  // kernel beats staging but for the projections apple9Stages names; Apple10
  // runs it on narrow grids (apple10Registers).
  const bool spreads = spreadsStagedDecode(family_, gpuCores_), gateUp = w.epilogue == LinearEpilogue::GateUp,
             onePass = spreads && gateUp && oneFormat(projections);
  if (shape.registerFormats && everyQuantizedSegment(projections, registerTileTakes) &&
      (family_ == GpuFamilyClass::Apple9
           ? !apple9Stages(w, projections)
           : apple10Registers(w.rows, onePass ? 2 * n : n, gpuCores_,
                              shape.fourBits && everyQuantizedSegment(projections, fourBitsPerWeight))))
    return registerDecode(n, k, gpuCores_);
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
  // There a gate/up pair of one format may also run in one pass, each
  // simdgroup staging both tensors' columns, where the wave law
  // (apple10Staged) takes it: its threadgroups of one simdgroup reach DRAM's
  // rate from fewer columns per core than two passes' and hold more per wave,
  // and lose a second wave where the pair puts a few more than a wave on each
  // core. On a 40-core M5 Max (13.6 per core) the 27B pair alone took 8-9%
  // less time in one pass in MLX 4-bit and Q4_K at one and two lanes, the 27B
  // decode cycle 2.4-3.2% less in MLX 4-bit and 0.6-2.0% in UD-Q4_K_M; 8-bit,
  // whose pair streams near DRAM's rate either way, 0.1-0.6% more. A 12-core
  // M6 holds 45 of the pair's threadgroups per core, where one pass took 2-11%
  // more: fewer cores run two passes.
  //
  // One lane of a projection whose quantized segments store fewer than four
  // and a half bits per weight together runs unsplit on Apple10's 16-row
  // tile, the second lane of its storage padding: the lane's rows are the
  // bits of the 8-row tile's (gguf-projection checks it), in less time. On
  // the 27B UD-IQ3_XXS's shapes, DRAM-cold and unsplit, a 12-core M6 takes
  // 2-10% less (its fused inputs 8-10%, an IQ3_S down projection 1% more)
  // and a 32-core M5 Max 5-7% less (a one-pass gate/up pair 10%), and the
  // 27B decodes one lane 3.3-3.6% faster on the M6; Flash-Next's fused
  // inputs take 7-9% less on a 20-core M5 Pro, its one-lane cycle about 1%.
  // At 4.5 bits the M6 takes up to 1.6% more (Q4_K, the vocabulary head's).
  // Split, the padding lane's partials double the split's traffic. At the
  // same split the 16-row tile takes 2-8% less time on the M6 at 4 to 20
  // threadgroups per core (IQ3_XXS and IQ2_XS, 5120 inputs, two and four
  // partitions), but on the narrow grids where the law splits K most it
  // takes 4-36% more on a 32-core M5 Max (IQ3_XXS, IQ2_XS and Q2_K in four
  // and eight partitions on the 35B's 2048 x 4096, 2048 x 6144 and
  // 2048 x 16384, Flash-Next's 2560 x 6144 and the 122B's 3072 x 8192, 1 to
  // 1.5 threadgroups per core unsplit; IQ4_XS on the last), against 7% less
  // to 2% more on the 27B's 5120 x 6144 and 5120 x 17408 in four. An unsplit
  // gate/up pair runs it where both its tensors' own plans do
  // (sixteenRowLane).
  LinearConfig config = stagedDecode(n, k, w.rows, weightBits16(projections, shape), gpuCores_, family_, gateUp,
                                     onePass ? Passes::Either : Passes::Two);
  config.spread = spreads;
  config.sixteenRows = sixteenRowLane(w, config.splits, projections, shape);
  return config;
}

bool Linear::sixteenRowLane(LinearWorkload w, uint32_t splits, std::span<const Projection *const> projections,
                            ProjectionShape shape) const {
  // An unsplit gate/up pair where both its tensors' own plans run it: the
  // decode storage of their shapes then holds its rows (decodeStorageRows).
  if (w.epilogue == LinearEpilogue::GateUp)
    return splits == 1 && !projections.empty() &&
           std::all_of(projections.begin(), projections.end(), [&](const Projection *p) {
             return p && ggufBaseline({w.matrix, w.rows}, std::span(&p, 1)).sixteenRows;
           });
  if (!projections.empty()) shape = projections.front()->shape();
  return family_ == GpuFamilyClass::Apple10 && w.rows == SPLASH_TARGET_VERIFY_ROWS && splits == 1 &&
         shape.underFourAndHalfBits();
}

LinearScratchSize Linear::ggufDecodeScratchSize(LinearWorkload w, bool registerFormats) const {
  const auto [n, k] = w.matrix;
  // Every format the register tile takes K inputs in has one register plan,
  // whose partitions keep whole coefficient units of each (kRegisterTiers);
  // a projection of segments the tile does not take has none.
  LinearScratchSize size = LinearPlan(w, ggufBaseline(w, {}, {.registerFormats = registerFormats})).scratchSize();
  // A projection of segments the register tile does not take, or of fewer
  // bits, stages where the device registers; a gate/up pair of one format may
  // run in one pass where the walks spread. Without the formats both take the
  // most bits per weight, whose partials weigh least against the weights: the
  // most K partitions of each pass mode, since the law's split never falls
  // with the bits per weight and never rises with the rows (a split's
  // partials grow with its rows alone, the same in either pass mode). One
  // lane on the 16-row tile runs unsplit, without partials.
  const bool gateUp = w.epilogue == LinearEpilogue::GateUp;
  if (family_ == GpuFamilyClass::Apple9 || apple10Registers(w.rows, n, gpuCores_, true))
    size.include(LinearPlan(w, stagedDecode(n, k, w.rows, 0, gpuCores_, family_, gateUp)).scratchSize());
  if (gateUp && spreadsStagedDecode(family_, gpuCores_))
    size.include(LinearPlan(w, stagedDecode(n, k, w.rows, 0, gpuCores_, family_, true, Passes::One)).scratchSize());
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

// The staged decode tiles, for decode and prefill chunks of up to 32 rows (64
// on Apple10): every row of the plan's storage in each threadgroup's tile,
// grid (64-column tiles, K splits). Decode runs one dispatch per projection
// (addDecodeTensor, fusedSegments), gate/up as two passes, or, where
// ggufBaseline sets LinearConfig::oneGateUpPass, as one pass over 32-column
// threadgroups of one simdgroup each. Prefill chunks run one dispatch per
// segment, but a fused projection's chunk of 33 to 64 rows runs one.
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
    // A fused projection's chunk of 33 to 64 rows runs its segments in one
    // dispatch: a dispatch costs about 0.045 ms however narrow, and one per
    // segment of a 35B GDN layer's qkv|z|ab made its resumes 3-5% slower
    // than the prefill tile's one per format.
    if (segments.size() > 1 && w.rows > kMaximumDecodeTileRows && w.epilogue == LinearEpilogue::None) {
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
// kMaximumDecodeTileRows rows (on Apple10, more than
// kMaximumStagedPrefillRows) over 128-row tiles: a plain projection in one
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

// The neural accelerator tile needs one (Apple9's matrix operations share the
// FP32 pipe, where three bf16 matmuls cost three fp32 ones) and enough of its
// 64-row by 32-column tiles per core. A tile runs K / 32 dependent steps, each
// staging the step's weights in threadgroup memory as bf16 parts and running
// a matmul per part (kernels/shared/gguf_float.metal), while the fp32 kernel
// spreads fewer rows over 8-column tiles with 8 K partitions each and
// finishes first below that grid. The grid it needs grows with that staging:
// 1.5 tiles per core for BF16 values, one part each, 2 for F32 values, three
// parts, and 3 for a quantized format's rows, decoded into three parts.
// Where the two tiles take the same time (gguf-projection-benchmark
// float-<format>, rows in steps of 64) on a 32-core M5 Max and a 12-core M6:
// F32 at 2.25-2.5 and 2.0-2.67 tiles per core for 256 columns of 2048 or 3072
// inputs, but 1.75-2.0 on the M6 and 2.0 and more on the M5 Max for the 64
// and 96 of GDN alpha/beta; quantized rows at 3.5 (8-bit affine) and
// 2.67-4.0 (8-bit affine, Q8_0, 4-bit affine, Q4_K); on a 20-core M5 Pro F32
// at about 2.2 for the 35B's router. Over those points the rule's tile takes
// at most 13% (F32) and 6% (quantized) longer than the faster one, where 1.5
// for every weight took up to 42% and 61% longer. BF16 crosses at 0.67-1.33
// (the M5 Pro: about 1.5 for Flash-Next's router), below its rule, which
// keeps a decode step's 16 to 32 rows on the simdgroup tile on 12 cores (a
// 512-column router: 1.6-1.8x faster there).
FloatTile Linear::ggufFloatTile(uint32_t rows, uint32_t outputSize, uint32_t formatId) const noexcept {
  const uint64_t tiles = uint64_t{(rows + 63) / 64} * ((outputSize + 31) / 32);
  // The grid in half tiles per core.
  const uint32_t halves = formatId == GGUF_FMT_BF16 ? 3 : formatId == QuantizedSegment::kFloat32 ? 4 : 6;
  return family_ == GpuFamilyClass::Apple10 && rows >= 16 && 2 * tiles >= uint64_t{halves} * gpuCores_
             ? FloatTile::NeuralAccelerator
             : FloatTile::Simdgroup;
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
  return {std::move(plane0), std::move(plane1), std::move(meta), outputSize, inputSize, 0, formatId, Layout::Planes};
}
QuantizedSegment QuantizedSegment::rows(uint32_t formatId, uint32_t outputSize, uint32_t inputSize,
                                        metal::MetalBuffer rows) {
  if (!rowBytes(formatId, inputSize))
    throw std::invalid_argument("rows take F32 or BF16 values or whole native blocks of a quantized GGUF format");
  return {std::move(rows), {}, {}, outputSize, inputSize, 0, formatId, Layout::Rows};
}
// F32 and BF16 values each their own; a quantized format's native blocks
// (kernels/common/gguf_embedding_formats.h).
uint64_t QuantizedSegment::rowBytes(uint32_t formatId, uint32_t inputSize) noexcept {
  if (formatId == kFloat32) return uint64_t{inputSize} * sizeof(float);
  if (formatId == GGUF_FMT_BF16) return uint64_t{inputSize} * sizeof(uint16_t);
  if (formatId >= GGUF_FMT_COUNT || formatId == GGUF_FMT_F16) return 0;
  const QuantFormat &f = kQuantFormats[formatId];
  return inputSize % f.block_elements ? 0 : uint64_t{inputSize} / f.block_elements * f.block_bytes;
}
const QuantFormat &QuantizedSegment::format() const noexcept { return kQuantFormats[formatId]; }
const char *QuantizedSegment::name() const noexcept { return formatId == kFloat32 ? "f32" : format().name; }

} // namespace splash::ops
