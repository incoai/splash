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

// A "from N cores" threshold no device reaches, and a register boundary
// every grid is below.
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
  // Partitions keep this many inputs, maximumSplits at most.
  uint32_t partitionInputs;
  uint32_t maximumSplits;
};
using SplitLaw = std::variant<TierLaw, WaveLaw>;

// Register tile against the staged tile for a decode step of `lanes` lanes,
// by the unit's FormatClass: register where 4 * columnTiles * lanes <
// belowQuarters * cores (in quarter threadgroups per core); kAlways where it
// takes the formats whatever the grid, 0 never.
struct RegisterBoundary final {
  std::array<std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH>, kFormatClasses> belowQuarters;
};

struct ProjectionCalibration final {
  SplitLaw registerSplits;
  SplitLaw stagedSplits;
  RegisterBoundary decodeRegister;
  // From this many cores the staged decode walks spread (LinearConfig::spread).
  uint32_t spreadFromCores;
  // From this many cores a gate/up pair of one format may run in one pass.
  uint32_t onePassFromCores;
  // One unsplit decode lane of fewer bits per weight than this, in
  // sixteenths, runs the staged 16-row tile; 0 never.
  uint32_t sixteenRowBelowBits16;
  // Prefill chunks of up to this many rows run the staged tile
  // (<= kMaximumStagedPrefillRows); longer ones the prefill tile.
  uint32_t stagedPrefillRows;
};

struct FloatTileCalibration final {
  // The device has a neural accelerator per core.
  bool neuralAccelerator;
  // The accelerator tile's fewest rows, and the grid it needs in half tiles
  // per core for BF16 values, F32 values and a quantized format's rows.
  uint32_t minimumRows;
  uint32_t bf16Halves, f32Halves, quantizedRowsHalves;
};

struct AttentionCalibration final {
  // A dense prefill tile takes one history split per this many pages its
  // rows read (PagedAttention.cpp).
  uint32_t prefillPagesPerSplit;
};

struct MoeCalibration final {
  // The expert tile by the block's FormatClass, in both phases: register
  // where true.
  std::array<bool, kFormatClasses> registerTile;
};

// Every measured constant of kernel selection for one device.
struct KernelCalibration final {
  // DeviceCapabilities::plannedGpuCores.
  uint32_t cores;
  ProjectionCalibration projection;
  FloatTileCalibration floatTile;
  MoeCalibration moe;
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
// refit it. Staged partitions keep 512 inputs, eight at most.
inline constexpr WaveLaw kApple10StagedLaw{{10, 6, 2}, {16, 10, 3}, 512, 8};
static_assert(kApple10StagedLaw.tile.resident >= kApple10StagedLaw.tile.saturating &&
                  kApple10StagedLaw.gateUpPair.resident >= kApple10StagedLaw.gateUpPair.saturating,
              "a full wave streams at DRAM's rate");

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
// bits per weight together runs unsplit on Apple10's 16-row tile, the second
// lane of its storage padding: the 8-row tile's bits (gguf-projection checks
// it) in less time, a 27B UD-IQ3_XXS lane 3.3-3.6% faster (12-core M6), Q4_K
// at 4.5 bits up to 1.6% slower. Split, the padding lane's partials double
// the traffic. An unsplit gate/up pair runs it where both its tensors' own
// plans do.
inline constexpr uint32_t kApple10SixteenRowBelowBits16 = 72;

// Prefill: 128-row tiles. A chunk of up to 32 rows runs the staged tile of its
// rows with the decode split law (the prefill tile's outputs up to fp32
// reassociation, gguf-projection checks it): 1.8-2.9x faster on a 17408 x
// 5120 Q4_K (16-core M5 Pro). The prefill tile takes as long for 33 rows as
// for 64, so up to 64 rows run the staged 64-row tile: 6-43% less time on
// Apple10, 12-22% on a 10-core M4.
// Larger Apple9 GPUs, unmeasured, keep the prefill tile past 32 rows.
inline constexpr uint32_t kApple9StagedPrefillRows = SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS;
inline constexpr uint32_t kApple9SmallStagedPrefillRows = 64;
inline constexpr uint32_t kApple10StagedPrefillRows = 64;

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
    .stagedPrefillRows = kApple9StagedPrefillRows,
};
inline constexpr ProjectionCalibration kApple9SmallProjection = [] {
  ProjectionCalibration row = kApple9Projection;
  row.stagedPrefillRows = kApple9SmallStagedPrefillRows;
  return row;
}();
inline constexpr ProjectionCalibration kApple10Projection{
    .registerSplits = TierLaw{kRegisterTiers},
    .stagedSplits = kApple10StagedLaw,
    .decodeRegister = kApple10DecodeRegister,
    .spreadFromCores = kApple10SpreadFromCores,
    .onePassFromCores = kApple10OnePassFromCores,
    .sixteenRowBelowBits16 = kApple10SixteenRowBelowBits16,
    .stagedPrefillRows = kApple10StagedPrefillRows,
};

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

// ---- MoE ------------------------------------------------------------------

// Apple9 stages experts only in GridLookup formats, in both phases: a
// 35B-shaped layer decodes those 4-21% faster staged on a 40-core M3 Max and
// Q4_K/Q5_K, Q2_K and IQ4_XS 3-34% slower, its matrix operations sharing the
// FP32 pipe. Apple9 of more than 10 cores is unmeasured since the staged
// tile's one-pass gate/up turned the 10-core M4 to staging every format.
inline constexpr MoeCalibration kApple9Moe{{false, true, true, true}};

// Apple9 GPUs of 10 cores or fewer stage experts in every format, in both
// phases: on a 10-core M4 a layer takes 12-45% less time staged at B1-B4 and
// 7-28% less on a 2048-row chunk.
inline constexpr MoeCalibration kApple9SmallMoe{{false, false, false, false}};

// Apple10 stages experts in every format.
inline constexpr MoeCalibration kApple10Moe{{false, false, false, false}};


// ---- Attention ----------------------------------------------------------

// A chunk of a few hundred rows over a long history otherwise leaves much of
// the GPU idle: one split per 32 pages runs a 256-row chunk's attention after
// 32K-128K tokens 23-26% faster on an M5 Max (Apple10, 12-32 cores measured).
inline constexpr AttentionCalibration kAttention{32};

// On a 10-core M4 one split per 64 pages leaves every case within the A/A
// spread (0.7%), where one per 32 takes 1.8% more on the 35B's 512-row chunk
// after 4K tokens. Larger Apple9 GPUs, unmeasured, keep 32.
inline constexpr AttentionCalibration kApple9SmallAttention{64};

// ---- The device's row ------------------------------------------------------

inline KernelCalibration KernelCalibration::of(const DeviceCapabilities &device) noexcept {
  const uint32_t cores = device.plannedGpuCores();
  if (device.gpuClass() == GpuClass::Apple10)
    return {cores, kApple10Projection, kApple10FloatTile, kApple10Moe, kAttention};
  const bool small = cores <= kApple9SmallGpuCores;
  return {cores, small ? kApple9SmallProjection : kApple9Projection, kApple9FloatTile,
          small ? kApple9SmallMoe : kApple9Moe, small ? kApple9SmallAttention : kAttention};
}

} // namespace splash::ops
