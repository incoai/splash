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

struct ProjectionCalibration final {
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

struct AttentionCalibration final {
  // A dense tile of a sequence of tiles, a prompt chunk's or a 9- to 16-row
  // lane's, takes one history split per this many pages its rows read.
  uint32_t pagesPerSplit;
  // From this many pages each split of the longest tile or lane reads, a
  // dense split over INT8 pages scores INT8 queries (PagedAttention.cpp).
  uint32_t int8QueriesFromPages;
  // The device's GPU cores (KernelCalibration::cores), which a decode step's
  // relative attention fills (RelativeAttention::decodeSplits); 0 in a row
  // no device made.
  uint32_t cores = 0;
};

// Every measured constant of kernel selection for one device.
struct KernelCalibration final {
  // DeviceCapabilities::plannedGpuCores.
  uint32_t cores;
  ProjectionCalibration projection;
  FloatTileCalibration floatTile;
  DenseTileCalibration denseTile;
  AttentionCalibration attention;

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
    .registerSplits = TierLaw{kRegisterTiers},
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
    // measured).
    .prefillInt8 = true,
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
inline constexpr AttentionCalibration kApple10Attention{.pagesPerSplit = 32, .int8QueriesFromPages = 8};

// No Apple9 GPU scores INT8 queries: on a 10-core M4 they took 8-12% longer,
// its matrix operations sharing the FP32 pipe. Larger Apple9 GPUs, unmeasured,
// share that pipe too.
inline constexpr AttentionCalibration kApple9Attention = [] {
  AttentionCalibration row = kApple10Attention;
  row.int8QueriesFromPages = kNever;
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

// ---- The device's row ------------------------------------------------------

inline KernelCalibration KernelCalibration::of(const DeviceCapabilities &device) noexcept {
  const uint32_t cores = device.plannedGpuCores();
  const GpuClass gpu = device.gpuClass();
  AttentionCalibration attention =
      gpu != GpuClass::Apple9 ? kApple10Attention
                              : cores <= kApple9SmallGpuCores ? kApple9SmallAttention : kApple9Attention;
  attention.cores = cores;
  if (gpu != GpuClass::Apple9)
    return {cores,
            gpu == GpuClass::Apple11            ? kApple11Projection
            : cores >= kApple10LargeGpuCores ? kApple10LargeProjection
                                              : kApple10Projection,
            kApple10FloatTile, kApple10DenseTile, attention};
  return {cores, kApple9Projection, kApple9FloatTile, kApple9DenseTile, attention};
}

} // namespace splash::ops
