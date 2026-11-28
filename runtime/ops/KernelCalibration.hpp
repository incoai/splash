#pragma once

// Every measured constant of kernel selection, by GPU class and, where a
// measurement says so, by core range. The rules that read them are in
// KernelSelection.cpp; KernelCalibration::of is the only place that picks a
// row by the device's class.

#include "metal/DeviceCapabilities.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Weights.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>

namespace splash::ops {

// A "from N" threshold nothing reaches (cores, pages, rows), and a register
// boundary every grid is below.
inline constexpr uint32_t kNever = UINT32_MAX;
inline constexpr uint32_t kAlways = UINT32_MAX;

// ---- K-split laws ---------------------------------------------------------

// K splits by tiers. A tier asks for more partitions, up to its `splits`,
// while the grid holds fewer than `threadgroupsPerCore` threadgroups per core
// and each partition would still keep `inputs` inputs; the split count
// doubles, in whole 32-input groups per partition, while some tier asks. Tiers
// ignore the batch width and serve the tiles with no residency step in their
// time: the register tile and Apple9's staged tile.
struct SplitTier final {
  uint32_t threadgroupsPerCore;
  uint32_t inputs;
  uint32_t splits = 8;
};
struct TierLaw final {
  std::span<const SplitTier> tiers;
};

// The Apple10 staged decode tiles' wave law. A threadgroup walks K / 32 /
// splits steps over a 64-column tile (a one-pass gate/up pair's 32 columns of
// both tensors). A core runs `resident` of them at once, so a grid runs in
// waves; from `saturating` per core a wave streams at DRAM's rate, so a first
// wave of fewer takes as long as one of that many; a later wave hides part of
// that latency and takes as long as at least `following`.
struct StagedOccupancy final {
  uint32_t resident, saturating, following;
};
struct WaveLaw final {
  // gguf_decode_<format>_m<rows>_<e> and gguf_decode_fused_m<rows>: two
  // simdgroups, 64 columns.
  StagedOccupancy tile;
  // gguf_decode_<format>_m<rows>_gate_up: one simdgroup, 32 columns of each
  // tensor.
  StagedOccupancy gateUpPair;
  // From wideBits16 sixteenths of a bit per weight `tile`'s first wave
  // saturates from wideSaturating threadgroups per core, each streaming that
  // many more bytes a step; a unit of unknown formats takes `tile`'s floor.
  uint32_t wideBits16, wideSaturating;
  // `tile`'s residency at 16 rows, which decides only one unsplit lane's
  // height (sixteenRowLane).
  uint32_t sixteenRowResident;
  // Partitions keep this many inputs, maximumSplits at most.
  uint32_t partitionInputs;
  uint32_t maximumSplits;
};
using SplitLaw = std::variant<TierLaw, WaveLaw>;

// Register tile against the staged tile for a plan of `tiles` table tiles
// (8 rows each, a decode lane's verify rows where it was measured), one to
// four (kRegisterTableTiles), the most the register tile holds (Linear.hpp,
// kMaximumRegisterRows), by the unit's FormatClass: register where
// 4 * columnTiles * tiles < belowQuarters * cores (in quarter threadgroups
// per core); kAlways where it takes the formats whatever the grid, 0 never.
inline constexpr uint32_t kRegisterTableTiles = 4;
struct RegisterBoundary final {
  std::array<std::array<uint32_t, kRegisterTableTiles>, kFormatClasses> belowQuarters;
};

// The column tile's K splits (decode plans of one or two rows). Its
// threadgroup of GGUF_COLUMN_COLUMNS columns streams in proportion to the
// bytes its lanes load a group, so a grid of g threadgroups of b sixteenths of
// a bit per weight splits while g * b < bits16PerCore * cores, into
// partitions of at least partitionInputs inputs, maximumSplits at most.
struct ColumnLaw final {
  uint32_t bits16PerCore;
  uint32_t partitionInputs;
  uint32_t maximumSplits;
};

// The matrix tile's K splits (decode plans of more rows than the column tile
// holds, on the GPUs whose row has the tile). A threadgroup keeps its
// threads' chunks in flight, eight weights of each of four columns a thread:
// GGUF_COLUMN_THREADS * 32 weights, 64 bytes a sixteenth of a bit per weight
// (matrixBytesInFlight). A grid of g threadgroups splits while g times those
// bytes fall short of what the device's DRAM delivers in latencyNanoseconds
// (KernelCalibration::memoryBytesPerSecond), into partitions of at least
// partitionInputs inputs, maximumSplits at most; a device whose class has no
// rate (DeviceCapabilities::classMemoryBytesPerSecond) runs unsplit.
struct MatrixLaw final {
  uint32_t latencyNanoseconds;
  uint32_t partitionInputs;
  uint32_t maximumSplits;
};

struct ProjectionCalibration final {
  ColumnLaw columnSplits;
  SplitLaw registerSplits;
  SplitLaw stagedSplits;
  RegisterBoundary decodeRegister;
  // From this many cores the staged decode walks spread (LinearConfig::spread).
  uint32_t spreadFromCores;
  // From this many cores a gate/up pair of one format may run in one pass.
  uint32_t onePassFromCores;
  // One unsplit table tile of rows (at most 8) of fewer bits per weight than
  // this, in sixteenths, runs the staged 16-row tile; 0 never.
  uint32_t sixteenRowBelowBits16;
  // The most waves of the 16-row tile one such plan runs in from there, under
  // a wave law (sixteenRowLane); 0 never.
  uint32_t sixteenRowWaves;
  // A staged plan on the decode tiles of a unit the int8 decode tiles take
  // (UnitFormats::decodeInt8Takes) runs them where it stores at least this
  // many rows (a 16-row plan counts 16).
  uint32_t decodeInt8FromRows;
  // Longer prefill chunks of a unit the int8 prefill tile takes
  // (UnitFormats::prefillInt8Takes) run that tile, where the family's int8
  // matmul is faster than its bf16 one.
  bool prefillInt8;
  // The matrix tile's law where the device runs it (Apple10 and Apple11,
  // whose matmul2d operand layout it is written for): there it takes every
  // decode-tile plan of more rows than the column tile holds, and the
  // register and staged tiles none but the int8 decode tiles' (decodeInt8).
  std::optional<MatrixLaw> matrixSplits = std::nullopt;
};

struct FloatTileCalibration final {
  // The device has a neural accelerator per core.
  bool neuralAccelerator;
  // The accelerator tile's fewest rows, and the grid it needs in half tiles
  // per core for BF16 values, F32 values and a quantized format's rows.
  uint32_t minimumRows;
  uint32_t bf16Halves, f32Halves, quantizedRowsHalves;
};

// The dense bf16 projections' tile rule (DenseLinear.cpp), its grids in half
// 64 x 64 output tiles per core: a projection of at least wideInputs inputs
// runs 64 x 128 tiles from wideFromHalves, and one of fewer than
// eightBelowHalves (or of one row tile) runs 64 x 64 tiles on eight
// simdgroups.
struct DenseTileCalibration final {
  uint32_t eightBelowHalves;
  uint32_t wideInputs, wideFromHalves;
};

// A decode step's single-pass tile on a GPU family (AttentionPlanning.hpp
// decodePassNanoseconds and decodePagesPerSplit): its costs, each a
// threadgroup's or a core's, fitted to forced splits over streams of layers
// of distinct K/V.
struct DecodeTileCosts final {
  // The tile's threadgroups the GPU keeps resident, every four cores (its
  // registers' and threadgroup memory's share of a core, which need not be a
  // whole number of threadgroups): past them a split waits for a later wave.
  uint32_t residentPerFourCores = 0;
  // A split's fixed time: the dispatch's launch, its tile's staging and K/V
  // store, and its partials' publish and arrival (ns).
  uint32_t splitNs = 0;
  // A page's time for a threadgroup alone on the GPU, with bf16 queries and
  // with INT8, and a biased page's more (ns).
  uint32_t pageNs = 0;
  uint32_t int8PageNs = 0;
  uint32_t biasNs = 0;
  // What a page takes a threadgroup more for each other of its wave's on its
  // core (ns), and for its wave's share of the DRAM's rate: each threadgroup
  // of the wave reads its page at this share of it (permille).
  uint32_t corePageNs = 0;
  uint32_t dramSharePermille = 0;
  // The last split's combine, a split it combines (ns).
  uint32_t combineNs = 0;
  // The pages a threadgroup reads from which a step's waves lose their
  // synchrony (0: they keep it): past the resident threadgroups, a step of
  // threadgroups of `per` pages each takes per / (per + queuePages) of its
  // time as a queue's (decodePassNanoseconds).
  uint32_t queuePages = 0;
};

struct AttentionCalibration final {
  // A dense tile of a sequence of tiles, a prompt chunk's or a 9- to 16-row
  // lane's, takes one history split per this many pages its rows read.
  uint32_t pagesPerSplit;
  // From this many pages each split of the longest tile or lane reads, a
  // dense split over INT8 pages scores INT8 queries (PagedAttention.cpp).
  uint32_t int8QueriesFromPages;
  // Relative attention's tile shapes (RelativeAttention.cpp): a prompt
  // chunk's direct tiles take directRows rows, and a tile of at most
  // boundedRescaleRows fused rows (3 a row) bounds its running output's
  // rescale by its rows, a taller one rescales every row
  // (relative_attention_tile.h).
  uint32_t directRows;
  uint32_t boundedRescaleRows;
  // A decode step's relative attention (RelativeAttention::decodeSplits):
  // the threadgroups every four cores keep resident of its single pass's
  // tiles, a narrow lane cap's and a verify block's (twice the fused rows),
  // which the step's lanes' history splits share: a split past them waits for
  // a second wave. kNever: no share binds, each lane takes what it repays
  // alone.
  uint32_t decodeThreadgroupsPerFourCores;
  uint32_t decodeWideThreadgroupsPerFourCores;
  // The device's GPU cores (KernelCalibration::cores), which a decode step's
  // relative attention fills (RelativeAttention::decodeSplits); 0 in a row
  // no device made.
  uint32_t cores = 0;
  // The memory rate of the device's class in bytes a second
  // (KernelCalibration::memoryBytesPerSecond); 0 for a class without one.
  uint64_t memoryBytesPerSecond = 0;
};

// A decode step's pass on 8-row expert tiles splits K in two partitions
// (MoeConfig) where its grid holds fewer threadgroups a core than
// `occupancy.saturating`, short of DRAM's rate, or more than `fromWaves`
// waves of its `resident`, up to `waves` waves; between the first two, at
// most `fromWaves` waves at DRAM's rate, and past the last, where its last
// wave is a small share of the pass, it does not. A law of no waves splits no
// pass.
struct MoePartitionLaw final {
  StagedOccupancy occupancy;
  uint32_t fromWaves;
  uint32_t waves;
};

// The MoE expert passes' tiles (selectMoe).
struct MoeCalibration final {
  // The decode passes' K partitions, the device's law (KernelCalibration::of)
  // rather than a row's.
  MoePartitionLaw partitions{};
  // A prefill chunk on the staged expert passes whose routes average at least
  // this many an expert runs 64-row expert tiles (MoeExpertTile::M64); kNever
  // never.
  uint32_t wideTileRoutes;
};

// The DFlash2 draft's decode kernels: the threadgroups every four cores keep
// resident of its top-16 pass, which a step's (lane, proposal position) rows'
// shards share (DraftSelector::shards). kNever: no share binds, each takes
// the most.
struct DraftKernelCalibration final {
  uint32_t topThreadgroupsPerFourCores;
};

// Every measured constant of kernel selection for one device.
struct KernelCalibration final {
  // DeviceCapabilities::plannedGpuCores.
  uint32_t cores;
  ProjectionCalibration projection;
  FloatTileCalibration floatTile;
  DenseTileCalibration denseTile;
  AttentionCalibration attention;
  MoeCalibration moe;
  DraftKernelCalibration draft;
  // The memory rate of the device's class (DeviceCapabilities::memoryBytesPerSecond: a plain read in bytes a
  // second, by GPU family and core count), which the laws take per core over `cores`; 0 for a class without one.
  uint64_t memoryBytesPerSecond = 0;

  [[nodiscard]] static KernelCalibration of(const DeviceCapabilities &device) noexcept;
};

// ---- Projections ----------------------------------------------------------

// Register tile (128 threads, four threadgroups a core). Below one wave K
// splits down to 256 inputs (whole coefficient units, quant_register_unit),
// below eight waves to 1024, shrinking the last wave's tail (40-core M3 Max).
// A grid eight partitions leave under three threadgroups per core takes
// sixteen of 512 inputs: 11-17% less time on Flash-Next's Q8_0 320 x 10240
// (20-core M5 Pro); eight take up to 21% more on a 10-core M4.
inline constexpr SplitTier kRegisterTiers[] = {{4, 256}, {32, 1024}, {3, 512, 16}};
// Apple10 and Apple11 split the register tile below one wave only to two
// threadgroups per core: at 4-16 rows Inkling-2-Mini's k (1024 x 2304, 16
// column tiles) took 15-27% less time on four partitions than on the eight
// four threadgroups ask on a 32-core M5 Max (af4g64, af8g64, Q4_K, Q6_K at 16
// rows; Q8_0 2% more at 4 and 8 rows, 9% less at 16) and 10-11% less on a
// 20-core M5 Pro (af8g64, Q8_0; Q4_K the same), and the DFlash2 draft's
// [576 x 2304] 2-14% less on a 16-core M5 Pro, where Inkling's own grids
// split as before.
inline constexpr SplitTier kApple10RegisterTiers[] = {{2, 256}, {32, 1024}, {3, 512, 16}};

// The column tile splits K only while its threadgroups times their bits per
// weight hold fewer than 4.5 a core (fewer threadgroups than cores in 4-bit
// formats, half as many in 8-bit), into partitions of at least 512 inputs (two
// groups a simdgroup), eight at most: where it splits, every split measured
// takes less time, and where it does not, none measured takes less on both
// M5 GPUs. A [320 x 10240] projection (Qwen3.8-Flash-Next's hyper-connection
// down, 10 threadgroups) on a 16-core M5 Pro took 24-29% less time on two
// partitions in Q4_K and af4g64 (30-34% on four). At one threadgroup a core
// the two GPUs disagree, a core of the 16-core M5 Pro streaming 17.6 GB/s
// against 13.3 on the 32-core M5 Max: a [512 x 5120] projection took 15-24%
// less time on two partitions in Q4_K and af4g64 on the M5 Pro, Inkling-2-
// Mini's k (1024 x 2304) 0-8% more on two on the M5 Max (Q4_K 2%, af4g64 0%);
// so neither splits. Q8_0 at 0.6 threadgroups a core took 17% less on two on
// the M5 Pro (320 x 10240), BF16 at 0.6 17% more, and neither splits.
// Inkling-2-Mini's projections split on no GPU of 10-32 cores, where every
// split measured lost (its k 4-12% more on two partitions on the M5 Pro, at two
// threadgroups a core); its 4-bit k splits in two from 40 cores, its k in every
// format but BF16 from 64, and on 80 its 4-bit k in four and its 4-bit k|v and
// o in two.
inline constexpr ColumnLaw kColumnLaw{.bits16PerCore = 72, .partitionInputs = 512, .maximumSplits = 8};

// The matrix tile's grid keeps the DRAM busy from about 600 ns of its rate in
// flight: on a 12-core M6 (160 GB/s, 96 KB) every projection of Inkling-2-
// Mini of 32 threadgroups or more ran fastest or within 2% of it unsplit at 3
// to 16 rows, its k (32 threadgroups) in IQ3_XXS among them (100 KB in
// flight), where two partitions took 17% more. Partitions keep two groups a
// simdgroup.
inline constexpr MatrixLaw kMatrixLaw{.latencyNanoseconds = 600, .partitionInputs = 512, .maximumSplits = 8};

// Apple9's staged tile takes the register tile's tiers of eight partitions:
// over the 27B and 35B dense shapes they come within 0-9% of each shape's
// fastest split on a 40-core M3 Max, where a fitted tier was up to 50% slower.
inline constexpr std::span<const SplitTier> kApple9StagedTiers = std::span(kRegisterTiers).first(2);

// The wave law: time per byte steps up past 10 of the tile's threadgroups per
// core and 16 of the pair's (`resident`); `saturating` and `following` are a
// grid search's fit to every measured case's fastest split (32-core M5 Max,
// 20-core M5 Pro, 12-core M6). On the 27B's projections its choice is at most
// 65 us a call from the fastest, under the 5% of a decode step that would
// refit it. Staged partitions keep 512 inputs, eight at most. At 16 rows the
// tile steps past 9 (5120-input Q4_K and MLX 4-bit, 16-core M5 Pro and
// 32-core M5 Max), which decides only one lane's height: splits take 10 at
// every height, as a fit per height sped no measured Mac without slowing some
// model.
inline constexpr WaveLaw kApple10StagedLaw{
    .tile = {10, 6, 2},
    .gateUpPair = {16, 10, 3},
    .wideBits16 = 100,
    .wideSaturating = 6,
    .sixteenRowResident = 9,
    .partitionInputs = 512,
    .maximumSplits = 8,
};
static_assert(kApple10StagedLaw.tile.resident >= kApple10StagedLaw.tile.saturating &&
                  kApple10StagedLaw.tile.saturating >= kApple10StagedLaw.wideSaturating &&
                  kApple10StagedLaw.gateUpPair.resident >= kApple10StagedLaw.gateUpPair.saturating,
              "a full wave streams at DRAM's rate, a wider unit's from fewer threadgroups");

// Apple9 decodes on the register tile but for a unit the staged tile holds
// unpadded whose quantized segments are all GridLookup, or Q2_K from two
// lanes: staged, those take 6-21% less time on a 40-core M3 Max (Q2_K 24-33%
// at two and four lanes). Three lanes pad its 24 rows to 32, 16-47% slower.
inline constexpr RegisterBoundary kApple9DecodeRegister{{{
    {0, 0, kAlways, 0},             // GridLookup
    {kAlways, 0, kAlways, 0},       // GridLookupOrQ2K
    {kAlways, kAlways, kAlways, kAlways},  // UnderFourBits
    {kAlways, kAlways, kAlways, kAlways},  // FourBitsOrMore
}}};

// Apple10 decodes on the register tile where the grid, a threadgroup per 64
// columns (a one-pass gate/up pair's too) and lane, holds fewer than 1.5
// threadgroups per core, 0.75 under four bits per weight: there the staged
// tile's 64 threads per column tile issue too few loads. The register tile's
// time grows with lanes and decode: one lane takes 13-43% less below the
// bound, up to 20% more at 1.25 under four bits (32-core M5 Max).
inline constexpr RegisterBoundary kApple10DecodeRegister{{{
    {3, 3, 3, 3},  // GridLookup
    {3, 3, 3, 3},  // GridLookupOrQ2K
    {3, 3, 3, 3},  // UnderFourBits
    {6, 6, 6, 6},  // FourBitsOrMore
}}};

// Decode tiles walking K in lockstep all fetch the same fresh input slice on
// their critical paths; starting each column tile at its own step lets most
// find it brought in by others: the 27B decodes 4-10% faster on a 40-core M5
// Max, a 12-core M6 and Apple9 no faster. 16 lies between the measured 12
// and 20.
inline constexpr uint32_t kApple10SpreadFromCores = 16;

// From as many cores a gate/up pair of one format may run in one pass, each
// simdgroup staging both tensors, where the wave law takes it: the 27B pair
// takes 8-9% less time on a 40-core M5 Max, 2-11% more on a 12-core M6.
inline constexpr uint32_t kApple10OnePassFromCores = 16;

// One lane of a projection whose quantized segments store fewer than 4.5
// bits per weight together runs unsplit on the staged 16-row tile, the second
// lane of its storage padding: the 8-row tile's bits (gguf-projection checks
// it) in less time at every grid, as no measured format there has a
// residency step: IQ2, IQ3 and Q3_K take 3-6% less a wave on a 16-core M5 Pro
// and a 32-core M5 Max, and a 27B UD-IQ3_XXS lane decodes 3.3-3.6% faster on
// a 12-core M6. Split, the padding lane's partials double the traffic. An
// unsplit gate/up pair runs it where both its tensors' own plans do.
inline constexpr uint32_t kApple10SixteenRowBelowBits16 = 72;

// From 4.5 bits one lane runs the 16-row tile where its grid runs in as many
// waves there as on 8 rows, two at most: a second wave for the 10th
// threadgroup per core took 18% more (Q4_K, 16-core M5 Pro), and two is the
// most waves measured end to end (the 27B's gate/up passes, 17 per core).
inline constexpr uint32_t kApple10SixteenRowWaves = 2;

// Prefill: 128-row tiles, past the decode tiles' 64 rows (KernelSelection.cpp:
// the staged tile of a chunk's rows with the decode split law ran a 17408 x
// 5120 Q4_K chunk of up to 32 rows 1.8-2.9x faster than the prefill tile on a
// 16-core M5 Pro, within fp32 reassociation of its outputs, gguf-projection
// checks it). The prefill tile takes as long for 33 rows as for 64, the staged
// 64-row tile 6-43% less on Apple10 and 12-22% less on a 10-core M4; larger
// Apple9 GPUs, unmeasured, take it too, which computes half the prefill
// tile's rows.

// Rotated PQ2_0 units decode in int8 activations on Apple10 and Apple11,
// whose neural accelerators multiply int8 operands, from 16 rows: the 8-row
// int8 matmul takes about twice the 16-row one's time, so one lane's split
// 8-row plans keep bf16.
inline constexpr uint32_t kApple10DecodeInt8FromRows = 16;

// Apple9 GPUs of up to this many cores (the base M3 and M4) take the 10-core
// M4's rows, larger ones the 40-core M3 Max's; the boundary is unmeasured.
inline constexpr uint32_t kApple9SmallGpuCores = 10;

inline constexpr ProjectionCalibration kApple9Projection{
    .columnSplits = kColumnLaw,
    .registerSplits = TierLaw{kRegisterTiers},
    .stagedSplits = TierLaw{kApple9StagedTiers},
    .decodeRegister = kApple9DecodeRegister,
    .spreadFromCores = kNever,
    .onePassFromCores = kNever,
    .sixteenRowBelowBits16 = 0,
    .sixteenRowWaves = 0,
    // Apple9's matrix operations share the FP32 pipe and multiply int8 at
    // 0.9x their bf16 rate (40-core M3 Max, measured): it keeps bf16
    // activations in decode and prefill, end to end unmeasured.
    .decodeInt8FromRows = kNever,
    .prefillInt8 = false,
};
inline constexpr ProjectionCalibration kApple10Projection{
    .columnSplits = kColumnLaw,
    .registerSplits = TierLaw{kApple10RegisterTiers},
    .stagedSplits = kApple10StagedLaw,
    .decodeRegister = kApple10DecodeRegister,
    .spreadFromCores = kApple10SpreadFromCores,
    .onePassFromCores = kApple10OnePassFromCores,
    .sixteenRowBelowBits16 = kApple10SixteenRowBelowBits16,
    .sixteenRowWaves = kApple10SixteenRowWaves,
    .decodeInt8FromRows = kApple10DecodeInt8FromRows,
    // Apple10 and Apple11 multiply int8 faster than bf16, so every chunk past
    // the decode tiles runs the int8 prefill tile where it takes the formats:
    // at least 1.13x less time than the bf16 tile over the 27B's projection
    // shapes in af4g64 and af5g64 at 65-2048 rows (M5 Max, M5 Pro and M6,
    // measured); with Q8_0 projections on it, Inkling-2-Mini's GGUF
    // prefill takes 0.4-1.5% less time at 512-32K tokens and
    // Qwen3.6-35B-A3B's 3.8-4.1% at 512-4K (20-core M5 Pro, measured); with
    // af8g64 ones, its 8-bit MLX pack's 4.4% at 8K, the projections 25%
    // (16-core M5 Pro, measured).
    .prefillInt8 = true,
    .matrixSplits = kMatrixLaw,
};

// From 32 cores (the 32-core M5 Max) the staged tile's first wave of a unit of
// 6.25 bits per weight or more (Q6_K, Q8_0, MLX 6- and 8-bit, FP8) saturates
// from four threadgroups per core. On 28 projections of the 35B, the 27B and
// Inkling-2-Mini in ten formats at 8 to 64 rows (macOS 27.0), four took up to
// 36% less time than six where their splits differ (the 35B's 2048 x 4096 Q8_0
// output projection on four partitions rather than eight, 18-36% less at 16 to
// 64 rows, the same shape in MLX 8-bit 16-32% less at 16 to 64 and in Q6_K
// 12-28% at 16 to 48; its fused Q8_0 attention input on one rather than two,
// 12-36% less at 8 to 32 rows; Inkling's fused Q8_0 attention input, 16-24%
// less at 8 and 16; Q6_K at 5120 x 6144, 11-21% less at 8 to 48), and nowhere
// more. Narrower formats keep six: four took up to 19% more there (the IQ4_NL,
// MXFP4, MLX 4-bit, Q4_K, IQ4_XS and NVFP4 down projections of 5120 x 17408 at
// 8 and 16 rows), and the 12-core M6 keeps six, where four took 25% more on
// Inkling's 3072 x 2304 Q8_0 projections at 56 and 64 rows. The 40-core M5 Max,
// unmeasured, takes the 32-core one's.
inline constexpr uint32_t kApple10LargeGpuCores = 32;
inline constexpr ProjectionCalibration kApple10LargeProjection = [] {
  ProjectionCalibration row = kApple10Projection;
  WaveLaw law = kApple10StagedLaw;
  law.wideSaturating = 4;
  row.stagedSplits = law;
  return row;
}();

// Apple11 takes Apple10's row but for the 16-row lane from 4.5 bits: the M6's
// 16-row tile takes as long a wave as its 8-row tile there (5120-input Q4_K,
// 12-core M6), so its lanes keep 8 rows; 16 took 10% more on the 27B
// UD-IQ3_XXS's Q4_K out-projections.
inline constexpr ProjectionCalibration kApple11Projection = [] {
  ProjectionCalibration row = kApple10Projection;
  row.sixteenRowWaves = 0;
  return row;
}();

// ---- Float projections ----------------------------------------------------

// The neural accelerator tile needs one (Apple9's matrix operations share the
// FP32 pipe) and enough of its 64 x 32 tiles per core, a grid that grows with
// the bf16 parts each K step stages (kernels/shared/gguf_float.metal): 2 tiles
// per core for F32 (three parts), 3 for quantized rows (decoded into three),
// at most 13% slower than the faster kernel at the measured crossings (32-core
// M5 Max, 12-core M6). BF16 crosses at 0.67-1.33 but keeps 1.5, which holds a
// decode step's 16-32 rows on the 1.6-1.8x faster simdgroup tile on 12 cores.
inline constexpr FloatTileCalibration kApple9FloatTile{false, 16, 3, 4, 6};
inline constexpr FloatTileCalibration kApple10FloatTile{true, 16, 3, 4, 6};

// ---- Dense projections ----------------------------------------------------

// Apple10 and Apple11. Measured on the 12-core M6 (dense-linear-benchmark at
// every tower projection and the rows images of 6x8 to 64x64 patches give it):
// from 4096 inputs a grid of 12 half tiles a core runs 64 x 128 tiles (the down
// projection at 4096 rows 2498 us against 3091 on 64 x 64 of four, the merger
// to 2048 at 1024 rows 1272 against 1624); a single row tile runs eight
// simdgroups (down at 64 rows 68.2 against 78.0 on four), and so does a grid of
// fewer than 8 half tiles a core; the rest run four (qkv at 4096 rows 1730
// against 1841 on eight). The pick is the fastest or within 6% of it at every
// shape (qkv at 48 rows 35.4 against 33.4). On Apple10 the two chips measured
// disagree: the 20-core M5 Pro's and the 32-core M5 Max's benchmarks both run
// the down projection about 4% faster on 64 x 64 of four than this row's 64 x
// 128 tile at 1024 and 4096 rows (M5 Pro 378 against 393 us and 1566 against
// 1606), while the M5 Max runs the merger at 1024 rows 8-21% faster on 64 x 128
// (up 973 against 1055 us; to 5120 1045 against 1268), where the M5 Pro is
// split (up 1851 against 1891 on 64 x 128; to 2048 670 against 708 and to 5120
// 2107 against 2155 on 64 x 64); on the M5 Pro the tower's encode is within 1%
// either way (a row without the 64 x 128 tile: 32x32 patches 48.09 ms against
// 47.76 on this row, 64x64 297.2 against 297.4). A rule per device would be a
// block of its own for about a percent of an image's encode.
inline constexpr DenseTileCalibration kApple10DenseTile{8, 4096, 12};
// Apple9 (10-core M4, dense-linear-benchmark at the tower's projections and
// rows): the 64 x 128 tile never won (merger at 1024 rows 13017 us against
// 12743 on 64 x 64 of four; down at 4096 rows 11190 against 10918), so no
// projection runs it; one of a single row tile runs fastest on eight
// simdgroups (the up projection at 48 rows 214.6 us against 244.8 on a
// 64 x 128 tile of eight, down 226.7 against 253.9), every
// other on four (qkv at 4096 rows 8646 against 8805 on eight).
inline constexpr DenseTileCalibration kApple9DenseTile{0, kNever, 0};

// ---- Attention ----------------------------------------------------------

// A chunk of a few hundred rows over a long history otherwise leaves much of
// the GPU idle: one split per 32 pages runs a 256-row chunk's attention after
// 32K-128K tokens 23-26% faster on an M5 Max (Apple10, 12-32 cores measured).
// The neural accelerator multiplies INT8 at twice its bf16 rate, so a split
// of 8 pages or more scores INT8 queries, repaying their quantization once
// per threadgroup: splits of 1-5 pages took up to 31% more, of 9 less (M6,
// 20-core M5 Pro).
// Relative attention's direct tiles of 21 rows (63 fused rows as 64) took a
// 2,048-row chunk's full layer 14-15% less time than tiles of 16 after 32K
// and 1M tokens (20-core M5 Pro). Without the rows' bound in the rescale the
// 21-row tiles took 24-32% less time, and tiles of 24 and 48 fused rows (the
// split's, a decode step's 8- and 16-row lanes') 4-23% more (16-core M5 Pro;
// the same on an M6 and an M5 Max).
// A decode step's lanes share 19 threadgroups every four cores of a narrow
// lane cap's tiles and 14 of a verify block's (4.75 and 3.5 a core). They are
// fitted, not measured residency: rounding the lanes' share at them gives the
// fastest four-lane splits measured on 16, 20 and 32 cores. Four lanes over a
// full layer's 8K tokens (us a layer) ran fastest at 2 splits each on a
// 16-core M5 Pro (8 rows 441, 16 rows 611; 3 took 462 and 681), 3 and 2 on a
// 20-core (361 and 582) and 4-5 and 4 on a 32-core M5 Max (280-294 and 367; 6
// and 5 took 297 and 482). Measured residency differs: a 16-row tile held 64 to
// 79 threadgroups on both M5 Pros, 16 and 20 cores (80 took one lane 30-46%
// longer). One lane keeps its own most (RelativeAttention::decodeMostSplits),
// under its share from 16 cores.
inline constexpr AttentionCalibration kApple10Attention{.pagesPerSplit = 32,
                                                        .int8QueriesFromPages = 8,
                                                        .directRows = 21,
                                                        .boundedRescaleRows = 48,
                                                        .decodeThreadgroupsPerFourCores = 19,
                                                        .decodeWideThreadgroupsPerFourCores = 14};

// No Apple9 GPU scores INT8 queries: on a 10-core M4 they took 8-12% longer,
// its matrix operations sharing the FP32 pipe. Its relative attention runs
// direct tiles of 16 rows, which took a 2,048-row chunk's sliding layer 10%
// less time than tiles of 21 and a full layer's 6% after 2K tokens; and it
// bounds the rescale of tiles of 24 fused rows alone: the bound took them
// 3-15% less time, but 16-row lanes' tiles of 48 rows 7-17% more and direct
// tiles of 16 and of 21 rows 15% more (10-core M4). Larger Apple9 GPUs,
// unmeasured, share that pipe too. A decode step's single pass showed no
// edge of the threadgroups a core keeps resident up to 26 (four lanes of 8
// splits each): more of them hide more of that pipe's latency, so its lanes
// share none and each takes the most a lane's combine repays (steps of more
// than four lanes, past 26 a core, unmeasured). One lane over a full layer's
// 32K tokens ran at 8 to 10 splits within 5% (8 rows 1.81-1.91 ms, 16 rows
// 2.46-2.58; the old 4 of 16 rows took 3.07), four lanes over 8K tokens 7-8%
// faster at 8 splits each than at 2 (8 rows 1.81 ms against 1.95, 16 rows
// 2.47 against 2.64; 10-core M4).
inline constexpr AttentionCalibration kApple9Attention = [] {
  AttentionCalibration row = kApple10Attention;
  row.int8QueriesFromPages = kNever;
  row.directRows = 16;
  row.boundedRescaleRows = 24;
  row.decodeThreadgroupsPerFourCores = kNever;
  row.decodeWideThreadgroupsPerFourCores = kNever;
  return row;
}();

// On a 10-core M4 one split per 64 pages leaves every case within the A/A
// spread (0.7%), where one per 32 takes 1.8% more on the 35B's 512-row chunk
// after 4K tokens. Larger Apple9 GPUs, unmeasured, keep 32.
inline constexpr AttentionCalibration kApple9SmallAttention = [] {
  AttentionCalibration row = kApple9Attention;
  row.pagesPerSplit = 64;
  return row;
}();

// ---- MoE experts ------------------------------------------------------------

// A prefill chunk's GGUF expert passes dequantize an expert's weights once a
// tile, on 32-row tiles for up to 32 of its rows, on 64-row tiles for up to
// 64. The M5 Pro (Apple10, 16 to 31 cores) runs 64-row tiles where a chunk's
// routes average 64 or more an expert from 20 cores, 128 or more on the
// 16-core. One layer on the 20-core M5 Pro
// (gguf-moe-benchmark tiles, random routes, macOS 27.0): at Inkling-2-Mini's
// shape (NVFP4, 8 of 128 experts) they took 5.7% less time than 32-row tiles
// at 1024 rows (64 routes an expert), 5.0% at 1536 and 6.5% at 2048, and 1.0%
// more at 256 (16); at the 35B's (Q4_K/Q5_K/Q8_0, 8 of 256) 1.8% less at 2048
// (64), 1.1% less at 1536 (48) and 0.9% more at 1024 (32). In the prefill
// (decode-profile, ABBA rounds against the 32-row tiles) Inkling's 4K and 32K
// tokens took 3.0-3.5% less GPU time and its 512 the same (32 routes an
// expert: its tiles do not change), the 35B's 4K within 0.3% either way.
// The 16-core M5 Pro gains less. One layer there (the same benchmark): at
// Inkling's shape 0.6% less time at 1024 rows (64 routes an expert), 0.4%
// more at 1536 (96) and 2.3% less at 2048 (128); at the 35B's 3.9% more at
// 2048 (64), 7.3% and 6.9% more at 1024 and 1536. In the prefill
// (decode-profile, base new new base base new, medians, macOS 27.2) Inkling's
// 4K and 32K tokens took 1.5% and 1.2% less GPU time on 64-row tiles (2048-row
// chunks, 128 routes an expert), but the 35B's 4K 2.3% more and its 8K
// 1.5-2.2% more (2048-row chunks, 64 an expert). So the 16-core runs them
// from 128 routes an expert: Inkling-2-Mini's 2048-row chunks, never the
// 35B's or the 122B's (8 of 256 experts: 64 at most).
// 128-row tiles took 19% more time over Inkling's 4K prefill, their passes'
// synchronization outweighing the dequantization they save. The 12-core M6
// keeps 32-row tiles: there 64-row ones took 3-6% more on the 35B's formats at
// 1024 to 2048 rows, though 9-15% less at Inkling's NVFP4 shape at 768 to 2048
// (unmeasured end to end). So does Apple9, where they gained nothing on a
// 10-core M4, and so do the 10-core M5 and the M5 Max, unmeasured.
inline constexpr uint32_t kApple10ProGpuCores = 16;
// From as many cores an M5 Pro runs 64-row tiles from 64 routes an expert.
inline constexpr uint32_t kApple10ProMoeFromCores = 20;
inline constexpr MoeCalibration kApple10ProMoe{.wideTileRoutes = 64};
inline constexpr MoeCalibration kApple10SixteenCoreMoe{.wideTileRoutes = 128};
inline constexpr MoeCalibration kThirtyTwoRowMoe{.wideTileRoutes = kNever};
// The expert passes' K partitions follow the staged tile's wave law (they are
// staged tiles of the same 64 threads a column pair): a pass splits K where its
// grid holds fewer than `saturating` (6) threadgroups a core, which stream
// short of DRAM's rate, so twice the simdgroups stream more, or more than the
// `resident` (10) of one wave, where a threadgroup's walk of half of K ends
// sooner and the last wave's tail with it, up to twelve waves (120 a core),
// past which the tail is too small a share of the pass to pay the
// partitions' sums; between the first two, one wave at DRAM's rate,
// partitions only add their sums. The counts are the law's, of rows routed
// independently: a verify block's rows route alike, to about half the tiles
// at 16 rows. Measured by decode-profile on the 32-core M5 Max (gate/up,
// down: 1 against 2 partitions, the law's counts): Qwen3.6-35B-A3B's gate/up
// at one row, 4.5 a core, 2.5% faster (6% on an earlier run);
// Inkling-2-Mini's passes and the 35B's down at one row, 9 a core, up to 3%
// slower; from 16 to 104 a core 0.6-5.3% faster (Inkling-2-Mini at 16 rows,
// 93 a core, 5.3% and 2.0%; the 35B's down there, 104, 2.6%). On a 16-core
// M5 Pro by decode-profile ABBA: Inkling-2-Mini's passes 0.5-6.1% faster at 4
// and 8 rows (43 and 65 a core), level at one row (18) and at 16 rows (185,
// one partition now); the 35B's within 2.5% either way; at 9 a core (its
// gate/up at one row) two partitions level with one over three rounds (0.99);
// by gguf-moe-benchmark's partitions there, rows routed at random, 1-6% slower
// at 193 and 206 a core. Apple9 and Apple11 run one partition: on a 10-core M4
// and a 12-core M6 two partitions took 0.95-1.06 of one's time, a median of
// 1.00, and on the M6 the 35B's one-row decode cycle took 0.6-1.1% more with
// them (Inkling-2-Mini's at 4 rows 0.7-1.8% less, within its noise).
inline constexpr MoePartitionLaw kApple10MoePartitions{kApple10StagedLaw.tile, 1, 12};
// The M5 Pros split from two waves (20 a core): at one row Inkling-2-Mini's
// down pass, 14.4 a core on 20 cores and 18 on 16, took 2.6-2.7% more in two
// partitions and the 35B's down at the same counts 0-1.6% more (decode-profile
// ABBA, calm), where the M5 Max's passes at 16 a core took up to 3.6% less;
// inside the band, the 35B's gate/up at 7.2 a core (20 cores) took 3.7% more.
// The edge is measured; its cause is unknown (bandwidth a core does not
// explain it: the Pros have more, 14.3-17.2 GB/s against the Max's 12.3-13.3,
// and pay later). By core range, as the rows here: an Apple10 GPU of
// kApple10ProGpuCores to kApple10LargeGpuCores - 1 cores (16-31) is a Pro;
// every other Apple10 GPU takes the M5 Max's law (fromWaves 1). That is
// assumed, not measured, for the 10-core base M5 and for the 64- and 80-core
// M5 Ultra, whose 15 GB/s a core (1.2 TB/s over 80 cores) is the Pros' rather
// than the Max's. 1.7.5: derive the edge per core.
inline constexpr MoePartitionLaw kApple10ProMoePartitions{kApple10StagedLaw.tile, 2, 12};

// ---- The DFlash2 draft --------------------------------------------------------

// Six of the top-16 pass's threadgroups every four cores (draft-selector-sweep,
// Inkling-2-Mini's 200,058 tokens in rows of 200,064): its 256 threads hold a
// sixteen-entry list and a 64-value chunk each, and past about one and a half
// a core a shard's partial and merge cost more than its residency hides. It
// takes 4, 2, 2 and 2 shards at 1, 2, 4 and 8 lanes on a 20-core M5 Pro, the
// best of 2 to 32 at each (top-16 alone 44.9, 79.1, 140.3 and 247.3 us; the
// fixed 8 took 44.7, 91.5, 158.6 and 278.1), and 2 at every lane count on a
// 12-core M6, the best at 1, 4 and 8 lanes (68.4, 184.0, 345.4 us) and 7.7%
// past it at 2 (119.6 against 4 shards' 111.0). Apple9 takes it unmeasured.
inline constexpr DraftKernelCalibration kApple10Draft{.topThreadgroupsPerFourCores = 6};

// ---- The device's row ------------------------------------------------------

// Apple11 takes the rows measured on the 12-core M6 (kApple11Projection and
// Apple10's other rows) at every core count.
inline KernelCalibration KernelCalibration::of(const DeviceCapabilities &device) noexcept {
  const uint32_t cores = device.plannedGpuCores();
  const GpuClass gpu = device.gpuClass();
  AttentionCalibration attention =
      gpu != GpuClass::Apple9 ? kApple10Attention
                              : cores <= kApple9SmallGpuCores ? kApple9SmallAttention : kApple9Attention;
  attention.cores = cores;
  attention.memoryBytesPerSecond = device.memoryBytesPerSecond;
  const bool pro = gpu == GpuClass::Apple10 && cores >= kApple10ProGpuCores && cores < kApple10LargeGpuCores;
  MoeCalibration moe = !pro                              ? kThirtyTwoRowMoe
                       : cores < kApple10ProMoeFromCores ? kApple10SixteenCoreMoe
                                                         : kApple10ProMoe;
  if (gpu == GpuClass::Apple10) moe.partitions = pro ? kApple10ProMoePartitions : kApple10MoePartitions;
  if (gpu != GpuClass::Apple9)
    return {cores,
            gpu == GpuClass::Apple11            ? kApple11Projection
            : cores >= kApple10LargeGpuCores ? kApple10LargeProjection
                                              : kApple10Projection,
            kApple10FloatTile, kApple10DenseTile, attention, moe, kApple10Draft, device.memoryBytesPerSecond};
  return {cores, kApple9Projection, kApple9FloatTile, kApple9DenseTile, attention, moe, kApple10Draft,
          device.memoryBytesPerSecond};
}

} // namespace splash::ops
