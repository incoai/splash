#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

constexpr uint32_t kAffinePrefillTileRows = 32;
constexpr uint32_t kQuantGroup = 64;
static_assert(SPLASH_TARGET_VERIFY_ROWS == 8,
              "Q4 register tiles require eight verify rows per lane");
// The kernels sum their input per block of four quant groups (256 inputs);
// Split128 partitions K in whole blocks.
constexpr uint32_t kInputSumBlock = 4 * kQuantGroup;

bool oneLaneTile(LinearTile tile) noexcept {
  return tile == LinearTile::Paired128 || tile == LinearTile::Paired256;
}
// The decode tiles whose threadgroups stream several column tiles.
bool persistentTile(LinearTile tile) noexcept {
  return tile == LinearTile::N128 || tile == LinearTile::N256 || tile == LinearTile::Paired128 ||
         tile == LinearTile::Paired256;
}
// The tiles of block (GGUF) projections.
bool blockTile(LinearTile tile) noexcept {
  return tile == LinearTile::GgufStaged || tile == LinearTile::GgufPrefill || tile == LinearTile::GgufRegister;
}
// Simdgroups fixed by the kernel instance: the paired N256 tile runs four and
// Split128 the N128 tile's eight.
std::optional<LinearSimdgroups> fixedSimdgroups(LinearTile tile) noexcept {
  switch (tile) {
  case LinearTile::Q4Register:
  case LinearTile::Paired256: return LinearSimdgroups::Four;
  case LinearTile::Split128: return LinearSimdgroups::Eight;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::GgufStaged:
  case LinearTile::GgufPrefill:
  case LinearTile::GgufRegister: return std::nullopt;
  }
  return std::nullopt;
}

void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid linear prefill workload");
  } else {
    if (w.matrix.inputSize % 256 || !w.rows || w.rows % SPLASH_TARGET_VERIFY_ROWS ||
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

// LinearScratch::rotated bytes of `rows` bf16 rows of `width` inputs.
constexpr uint64_t rotatedBytes(uint32_t width, uint64_t rows) noexcept { return uint64_t{width} * rows * 2; }

// The four-simdgroup kernels: every prefill N128 tile, the decode M24 N128
// plain and residual projections, the Q4 register tile, and the one-lane
// Paired256 (plain) tile.
bool supportsFourSimdgroups(LinearWorkload w, LinearTile tile) noexcept {
  if (tile == LinearTile::Q4Register) return w.phase == LinearPhase::Decode;
  // Only the affine paired N256 kernel is instantiated: this tile is used
  // for wide plain projections; residual and gate/up retain their own tiles.
  if (tile == LinearTile::Paired256)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS &&
        w.epilogue == LinearEpilogue::None;
  if (tile != LinearTile::N128) return false;
  return w.phase == LinearPhase::Prefill ||
      (w.rows == 24 && (w.epilogue == LinearEpilogue::None ||
                        w.epilogue == LinearEpilogue::Residual));
}

} // namespace

// Table16 holds its sums per eight-row tile (metal/abi/Gguf.h), a lane's rows.
uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept {
  return layout == LinearInput::Table16
             ? rows / SPLASH_TARGET_VERIFY_ROWS * table16_sums_per_tile(width) * sizeof(float)
       : layout == LinearInput::Table64 ? uint64_t{width} * rows / 16 : 0;
}

void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows) {
  if (layout == LinearInput::Plain || !rows || rows % SPLASH_TARGET_VERIFY_ROWS || width % 64)
    throw std::invalid_argument("invalid linear table geometry");
  requireBytes(scratch.input, tableBytes(width, rows), "linear table");
  requireBytes(scratch.sums, tableSumsBytes(layout, width, rows), "linear table sums");
}

const char *tableSuffix(LinearInput layout) noexcept {
  return layout == LinearInput::Table16 ? "_table16" : layout == LinearInput::Table64 ? "_table64" : "";
}

void requireAffineProjection(const Projection &p, LinearMatrix matrix) {
  if (p.layout() != WeightLayout::Affine64 || p.outputSize != matrix.outputSize ||
      p.inputSize != matrix.inputSize)
    throw std::invalid_argument("affine projection does not match plan");
  requireBytes(p.affine().weights, uint64_t{matrix.outputSize} * matrix.inputSize / 2, "projection weight");
  const uint64_t bytes = uint64_t{matrix.outputSize} * (matrix.inputSize / kQuantGroup) * 2;
  requireBytes(p.affine().scales, bytes, "projection scale");
  requireBytes(p.affine().biases, bytes, "projection bias");
}

uint32_t LinearPlan::storageRows() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockStorageRows();
  if (workload_.phase != LinearPhase::Prefill) return workload_.rows;
  return ((workload_.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows) * kAffinePrefillTileRows;
}
uint32_t LinearPlan::tileColumns() const noexcept {
  switch (config_.tile) {
  case LinearTile::Q4Register: return workload_.epilogue == LinearEpilogue::GateUp ? 32 : 64;
  case LinearTile::GgufStaged:
  case LinearTile::GgufPrefill:
  case LinearTile::GgufRegister: return GGUF_TILE_COLUMNS;
  case LinearTile::N256:
  case LinearTile::Paired256: return 256;
  case LinearTile::N128:
  case LinearTile::Paired128:
  case LinearTile::Split128: return 128;
  }
  return 0;
}
uint32_t LinearPlan::groups() const noexcept {
  return config_.groups ? config_.groups : workload_.matrix.outputSize / tileColumns();
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  switch (config_.tile) {
  case LinearTile::GgufStaged: return GGUF_STAGED_THREADS;
  case LinearTile::GgufPrefill: return GGUF_PREFILL_THREADS;
  case LinearTile::GgufRegister: return GGUF_REGISTER_THREADS;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::Split128:
  case LinearTile::Paired256:
  case LinearTile::Q4Register: return static_cast<uint32_t>(config_.simdgroups) * 32;
  }
  return 0;
}
bool LinearPlan::usesQ4Register() const noexcept { return config_.tile == LinearTile::Q4Register; }
LinearInput LinearPlan::input() const noexcept {
  if (rotated_) return LinearInput::Plain;
  if (config_.tile == LinearTile::GgufRegister) return LinearInput::Table16;
  return usesQ4Register() ? LinearInput::Table64 : LinearInput::Plain;
}
LinearScratchSize LinearPlan::scratchSize() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockScratchSize();
  const auto [n, k] = workload_.matrix;
  // Split128: [split][row][column] fp32 partials over every row of the step
  // and one counter per column tile.
  if (config_.tile == LinearTile::Split128)
    return {0, 0, uint64_t{config_.splits} * workload_.rows * n * sizeof(float),
            uint64_t{n / tileColumns()} * sizeof(uint32_t)};
  if (!usesQ4Register()) return {};
  const uint64_t rows = workload_.rows;
  const uint64_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  // Each row tile owns two fp32 fragment streams per K partition and one
  // completion counter per column tile; a single partition needs neither.
  return {tableBytes(k, workload_.rows), tableSumsBytes(LinearInput::Table64, k, workload_.rows),
          config_.splits > 1 ? config_.splits * 2 * rows * n * sizeof(float) : 0,
          config_.splits > 1 ? lanes * (n / tileColumns()) * sizeof(uint32_t) : 0};
}

uint64_t LinearPlan::sumsBytes() const noexcept {
  return workload_.phase == LinearPhase::Prefill && workload_.weightLayout == WeightLayout::Affine64
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  // GGUF tiles run gate/up as a gate pass and an up-with-gate pass.
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp &&
       (!secondPipeline_.empty() || workload_.weightLayout == WeightLayout::Block32));
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate && workload_.weightLayout == WeightLayout::Affine64
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config, FloatOutput destination)
    : workload_(w), config_(config), destination_(destination) {
  validate(w);
  if (destination == FloatOutput::Float32 &&
      (w.phase != LinearPhase::Decode || w.epilogue != LinearEpilogue::None))
    throw std::invalid_argument("an fp32 destination takes a plain decode projection");
  const bool ggufTile = blockTile(config.tile);
  if (ggufTile != (w.weightLayout == WeightLayout::Block32))
    throw std::invalid_argument("block projections run the GGUF tiles, affine ones the Q4 tiles");
  if (w.phase == LinearPhase::Decode && persistentTile(config.tile)
          ? !config.groups || config.groups > w.matrix.outputSize / tileColumns()
          : config.groups != 0)
    throw std::invalid_argument("a persistent decode tile takes 1 to its column tiles in groups, every other plan 0");
  if (ggufTile) {
    // Kernel names follow the segment formats (LinearGguf.cpp).
    requireBlockConfiguration();
    return;
  }
  const bool splitsK = config.tile == LinearTile::Q4Register || config.tile == LinearTile::Split128;
  if (!splitsK && config.splits != 1)
    throw std::invalid_argument("K splits require the register or Split128 Q4 tile");
  if (config.simdgroups == LinearSimdgroups::Four && !supportsFourSimdgroups(w, config.tile))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (const auto fixed = fixedSimdgroups(config.tile); fixed && config.simdgroups != *fixed)
    throw std::invalid_argument("Q4 tile requires its kernel's simdgroup count");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  if (w.phase == LinearPhase::Prefill) {
    if (oneLaneTile(config.tile) || splitsK)
      throw std::invalid_argument("invalid Q4 prefill configuration");
    if (four) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? "prefill_linear_q4_n128_up_silu_sums_sg4"
          : residual ? "prefill_linear_q4_n128_residual_sg4" : "prefill_linear_q4_n128_sg4";
    } else if (w.epilogue == LinearEpilogue::UpWithGate) {
      if (config.tile != LinearTile::N256)
        throw std::invalid_argument(
            "Q4 fused prefill up requires N256 or four simdgroups");
      pipeline_ = "prefill_linear_q4_n256_up_silu_sums";
    } else if (residual) {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128_residual" : "prefill_linear_q4_n256_residual";
    } else {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128" : "prefill_linear_q4_n256";
    }
    return;
  }
  const uint32_t lane = w.rows / SPLASH_TARGET_VERIFY_ROWS - 1;
  if (oneLaneTile(config.tile) && lane != 0)
    throw std::invalid_argument("paired Q4 tile requires one lane");
  if (usesQ4Register()) {
    const uint32_t groups = w.matrix.inputSize / kQuantGroup;
    if (!config.validSplits() || groups % config.splits)
      throw std::invalid_argument("the Q4 register tile requires whole power-of-two K partitions");
    pipeline_ = w.epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
        residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
    return;
  }
  if (config.tile == LinearTile::Split128) {
    // Every partition holds at least one of the kernel's 256-input blocks.
    if (!config.validSplits() || config.splits < 2 ||
        w.matrix.inputSize / kInputSumBlock < config.splits)
      throw std::invalid_argument("Split128 requires 2, 4 or 8 K partitions of 256-input blocks");
    constexpr std::array plainNames{"decode_linear_q4_n128_split", "decode_linear_q4_n128_split_m16",
        "decode_linear_q4_n128_split_m24", "decode_linear_q4_n128_split_m32"};
    constexpr std::array residualNames{"decode_linear_q4_n128_split_residual",
        "decode_linear_q4_n128_split_residual_m16", "decode_linear_q4_n128_split_residual_m24",
        "decode_linear_q4_n128_split_residual_m32"};
    constexpr std::array upSiluNames{"decode_linear_q4_n128_split_up_silu",
        "decode_linear_q4_n128_split_up_silu_m16", "decode_linear_q4_n128_split_up_silu_m24",
        "decode_linear_q4_n128_split_up_silu_m32"};
    // Gate/up at every lane count: a plain gate pass into the gate scratch,
    // then the up pass whose epilogue applies the SiLU gate.
    pipeline_ = residual ? residualNames[lane] : plainNames[lane];
    if (w.epilogue == LinearEpilogue::GateUp) secondPipeline_ = upSiluNames[lane];
    return;
  }
  if (config.tile == LinearTile::Paired256) {
    pipeline_ = "decode_linear_q4_n256_paired_sg4";
    return;
  }
  if (four) {
    pipeline_ = residual ? "decode_linear_q4_n128_residual_m24_sg4"
                         : "decode_linear_q4_n128_m24_sg4";
    return;
  }
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (config.tile != LinearTile::N256)
      throw std::invalid_argument("Q4 gate/up requires N256");
    constexpr std::array names{"decode_linear_q4_n256_gate_up", "decode_linear_q4_n256_gate_up_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
    if (lane >= 2)
      secondPipeline_ = lane == 2 ? "decode_linear_q4_n256_up_silu_m24"
                                  : "decode_linear_q4_n256_up_silu_m32";
  } else if (residual) {
    if (config.tile == LinearTile::N256)
      throw std::invalid_argument("Q4 decode residual requires N128");
    constexpr std::array names{"decode_linear_q4_n128_residual", "decode_linear_q4_n128_residual_m16",
        "decode_linear_q4_n128_residual_m24", "decode_linear_q4_n128_residual_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
        ? "decode_linear_q4_n128_residual_paired" : names[lane];
  } else if (config.tile == LinearTile::N256) {
    constexpr std::array names{"decode_linear_q4_n256", "decode_linear_q4_n256_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
  } else {
    constexpr std::array names{"decode_linear_q4_n128", "decode_linear_q4_n128_m16",
        "decode_linear_q4_n128_m24", "decode_linear_q4_n128_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
                    ? "decode_linear_q4_n128_paired" : names[lane];
  }
}

namespace {

// Decode groups stream output tiles. Under round-robin group placement, the
// most loaded core sets dispatch latency. Use the full grid for small workloads,
// balanced two-tile groups at intermediate sizes, and one resident wave for
// longer chains; sufficiently large grids balance themselves.
struct DecodeGroupPolicy final {
  // The one-tile grid wins up to this many groups per core.
  uint32_t fullGridGroupsPerCore;
  // Resident groups per core: one wave for this kernel's register footprint.
  uint32_t waveGroupsPerCore;
  // From this many tiles per core the many-wave grid wins again.
  uint32_t manyWaveTilesPerCore;
};
// Resident-wave and full-grid thresholds measured on 16/20-core Apple10 GPUs.
// Gate/up takes the lower of the two devices' limits. Two thresholds are
// derived rather than measured: gate/up's many-wave threshold is N256's, and
// the four-simdgroup thresholds are N128's doubled.
constexpr DecodeGroupPolicy kN128Groups{4, 4, 12}, kN128M16Groups{5, 4, 12},
    kN256Groups{3, 3, 8}, kGateUpGroups{3, 3, 8},
    kFourSimdgroupGroups{8, 8, 24};

// Tiles on the most loaded core when `groups` threadgroups are placed
// round-robin on `cores` and group g streams tiles g, g + groups, ...
uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) noexcept {
  uint32_t worst = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint32_t load = 0;
    for (uint32_t group = core; group < groups; group += cores)
      load += (tiles - group + groups - 1) / groups;
    worst = std::max(worst, load);
  }
  return worst;
}

uint32_t decodeGroups(uint32_t tiles, uint32_t cores,
                      DecodeGroupPolicy policy) noexcept {
  const uint32_t wave = policy.waveGroupsPerCore * cores;
  if (tiles <= policy.fullGridGroupsPerCore * cores ||
      tiles >= policy.manyWaveTilesPerCore * cores)
    return tiles;
  const uint32_t twoTile = (tiles + 1) / 2;
  // Here wave < twoTile <= tiles, so the wave is a valid count (LinearPlan
  // rejects more groups than tiles) whatever the per-core constants are.
  if (twoTile > wave) return wave;
  // The smallest balanced two-tile count keeping three quarters of the
  // full-grid limit resident. A multiple of the core count is always
  // balanced, so the search ends within `cores` steps and below `tiles`.
  const uint32_t balanced = (tiles + cores - 1) / cores;
  uint32_t groups =
      std::max(twoTile, policy.fullGridGroupsPerCore * cores * 3 / 4);
  while (maxCoreTiles(tiles, groups, cores) != balanced) ++groups;
  return groups;
}
// A multi-row N256 decode tile halves the input re-reads of N128 but also
// halves the grid; it pays only while the N256 grid keeps two tiles per core.
constexpr uint32_t kWideDecodeTilesPerCore = 2;
// Apple9 GPUs of up to 32 cores prefill with the Apple10 rule, the
// four-simdgroup N128 tile, which on a 32-core M4 Max is never slower than
// N256 on any prefill shape or row count and 6-10% faster wherever the
// difference is measurable (tune-kernels, benchmark-prefill).
// Larger Apple9 GPUs (40-core class) run N256 for UpWithGate and wherever its
// grid reaches eight threadgroups per core, which amortizes the larger tile;
// N128's four-simdgroup tile is not measured against it on them.
constexpr uint32_t kApple9MeasuredPrefillCores = 32;
constexpr double kApple9WidePrefillGroupsPerCore = 8.0;

// Apple10 and later decode split K across the threadgroups of the Split128
// tile (256 threads) by one rule at every batch width: the largest power of
// two up to LinearConfig::kMaximumSplits whose split grid still fits four
// threadgroups per core (1024 threads, twice the 512-thread occupancy knee),
// with at least one 256-input block per partition. A grid of more than two
// tiles per core keeps one split, the sequential tiles. A split grid that
// only reaches the knee leaves time, and one past four threadgroups per core
// loses to its second wave.
constexpr uint32_t kSplitGroupsPerCore = 4;

uint32_t apple10Splits(LinearMatrix matrix, uint32_t cores) noexcept {
  const uint64_t grid = matrix.outputSize / 128;
  uint32_t splits = 1;
  while (splits < LinearConfig::kMaximumSplits && grid * 2 * splits <= uint64_t{kSplitGroupsPerCore} * cores &&
         2 * splits <= matrix.inputSize / kInputSumBlock)
    splits *= 2;
  return splits;
}

// Apple10 wide plain projections reduce input re-reads with paired N256
// tiles at one resident wave, measured on 16/20-core GPUs. Apple9's register
// tile policy is independent.
constexpr uint32_t kPaired256TilesPerCore = 8;
constexpr uint32_t kPaired256WaveGroupsPerCore = 4;

std::optional<LinearConfig> apple10OneLaneConfig(LinearWorkload w, uint32_t cores) {
  // validate() requires outputSize % 256 == 0, so every tile width divides it.
  const uint32_t n = w.matrix.outputSize;
  const uint32_t tiles256 = n / 256;
  if (w.epilogue == LinearEpilogue::None && tiles256 >= kPaired256TilesPerCore * cores)
    return LinearConfig{LinearTile::Paired256,
                        std::min(tiles256, kPaired256WaveGroupsPerCore * cores),
                        LinearSimdgroups::Four};
  return std::nullopt;
}

} // namespace

Linear::Linear(const DeviceCapabilities &device) noexcept
    : family_(gpuFamilyClass(device.appleGpuFamily)),
      gpuCores_(plannedGpuCores(device)) {}

uint32_t Linear::decodeStorageRows(uint32_t rows, ProjectionShape shape) const {
  return plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Decode, LinearEpilogue::None, shape.layout})
      .storageRows();
}

// The GPU family class selects variants; core count and workload tile counts
// determine parallelism.
LinearConfig Linear::baseline(LinearWorkload w, std::span<const Projection *const> projections) const {
  validate(w);
  if (w.weightLayout == WeightLayout::Block32) return ggufBaseline(w, projections);
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  if (w.phase == LinearPhase::Prefill) {
    if (family_ == GpuFamilyClass::Apple10 || gpuCores_ <= kApple9MeasuredPrefillCores)
      return {LinearTile::N128, 0, LinearSimdgroups::Four};
    const uint32_t rowTiles = (w.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
    const bool wide = double(rowTiles) * tiles256 >=
        kApple9WidePrefillGroupsPerCore * gpuCores_;
    return {w.epilogue == LinearEpilogue::UpWithGate || wide ? LinearTile::N256
                                                              : LinearTile::N128, 0};
  }
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  // Wide plain projections of three or four lanes take the broad-column tiles
  // below on every family, since independent row tiles would repeat the weight
  // stream. Wide is the two-N256-tiles-per-core boundary
  // (kWideDecodeTilesPerCore), not a model dimension.
  const bool widePlain = lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * gpuCores_;
  if (family_ == GpuFamilyClass::Apple9 && !widePlain) {
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    const uint32_t grid = w.matrix.outputSize / columns, groups = w.matrix.inputSize / 64;
    uint32_t splits = 1;
    // Aim for sixteen independent column/K groups per core, retaining at
    // least twelve quant groups per partition to amortize the reduction.
    while (splits < LinearConfig::kMaximumSplits && uint64_t(grid) * splits < 16ULL * gpuCores_ &&
           groups % (2 * splits) == 0 && groups / (2 * splits) >= 12)
      splits *= 2;
    return {LinearTile::Q4Register, 0, LinearSimdgroups::Four, splits};
  }
  if (family_ == GpuFamilyClass::Apple10) {
    if (const uint32_t splits = apple10Splits(w.matrix, gpuCores_); splits > 1)
      return {LinearTile::Split128, 0, LinearSimdgroups::Eight, splits};
    if (lanes == 1)
      if (const auto config = apple10OneLaneConfig(w, gpuCores_)) return *config;
  }
  // Apple9 reaches here only for wide plain projections of three or four
  // lanes, which keep their one-tile grids: the round-robin policy above was
  // measured on Apple10.
  const auto groups = [&](uint32_t tiles, DecodeGroupPolicy policy) {
    return family_ == GpuFamilyClass::Apple10 ? decodeGroups(tiles, gpuCores_, policy)
                                              : tiles;
  };
  if (w.epilogue == LinearEpilogue::GateUp)
    return {LinearTile::N256, groups(tiles256, kGateUpGroups)};
  // Pipelined N128 hides the latency of a single lane's weight stream.
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, kN128Groups)};
  // Every M24 projection that gets here runs four SIMD groups.
  if (lanes == 3)
    return {LinearTile::N128, groups(tiles128, kFourSimdgroupGroups),
            LinearSimdgroups::Four};
  if (widePlain) return {LinearTile::N256, groups(tiles256, kN256Groups)};
  return {LinearTile::N128,
          groups(tiles128, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

LinearPlan Linear::plan(LinearWorkload workload) const {
  return LinearPlan(workload, baseline(workload));
}
LinearPlan Linear::plan(LinearWorkload workload, LinearConfig config, FloatOutput destination) {
  return LinearPlan(workload, config, destination);
}
LinearPlan Linear::plan(LinearWorkload w, const Projection &p, const Projection *gate) const {
  w.weightLayout = p.layout();
  const std::array<const Projection *, 2> projections{&p, gate};
  LinearPlan plan(w, baseline(w, projections), p.destination);
  plan.rotated_ = static_cast<bool>(p.rotation);
  return plan;
}

LinearPlan Linear::decodePlan(const Projection &p, uint32_t lanes, LinearEpilogue epilogue,
                              const Projection *gate) const {
  return plan(decode({p.outputSize, p.inputSize}, lanes, epilogue), p, gate);
}
LinearPlan Linear::prefillPlan(const Projection &p, uint32_t rows, LinearEpilogue epilogue) const {
  return plan({{p.outputSize, p.inputSize}, rows, LinearPhase::Prefill, epilogue}, p);
}

LinearScratchSize Linear::decodeScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
      LinearWorkload w = decode({shape.outputSize, shape.inputSize}, lanes, epilogue);
      w.weightLayout = shape.layout;
      bound.include(shape.layout == WeightLayout::Block32 ? ggufDecodeScratchSize(w) : plan(w).scratchSize());
    }
  // Decode plans store at most every lane's rows.
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, kMaximumDecodeTileRows);
  return bound;
}

LinearScratchSize Linear::prefillScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  // Prefill plans take split scratch only in chunks of up to a decode batch,
  // which a GGUF projection runs on the staged tile (LinearGguf.cpp).
  for (uint32_t rows = 1; rows <= kMaximumDecodeTileRows; ++rows)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      bound.include(plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Prefill, epilogue, shape.layout})
                        .scratchSize());
  // Every prefill plan stores at most the token budget.
  static_assert(SPLASH_PREFILL_TOKEN_BUDGET % GGUF_PREFILL_ROWS == 0, "the prefill tiles cover the budget exactly");
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, SPLASH_PREFILL_TOKEN_BUDGET);
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
  if (p.layout() != w.weightLayout || (gate && gate->layout() != w.weightLayout))
    throw std::invalid_argument("projection layout does not match execution plan");
  if ((w.epilogue == LinearEpilogue::GateUp) != (gate != nullptr))
    throw std::invalid_argument("a gate/up plan takes a gate projection and no other plan does");
  const uint64_t rows = selected.storageRows();
  requireBytes(b.input, rows * k * 2, "projection input");
  requireBytes(b.output, rows * n * elementBytes(selected.destination()), "projection output");
  if (w.epilogue == LinearEpilogue::Residual) requireBytes(b.residual, rows * n * 2, "projection residual");
  requireBytes(b.sums, selected.sumsBytes(), "projection sums");
  requireBytes(b.gateScratch, selected.gateScratchBytes(), "projection gate scratch");
  requireBytes(b.downSums, selected.downSumsBytes(), "projection down sums");
  const LinearScratchSize scratch = selected.scratchSize();
  requireBytes(b.scratch.input, scratch.input, "projection scratch table");
  requireBytes(b.scratch.sums, scratch.sums, "projection scratch sums");
  requireBytes(b.scratch.partials, scratch.partials, "projection partials");
  requireBytes(b.scratch.counters, scratch.counters, "projection counters");
  if (p.layout() == WeightLayout::Block32) {
    if (p.rotation) requireBytes(b.scratch.rotated, rotatedBytes(k, rows), "projection rotated input");
    addGguf(graph, b, p, selected, gate);
    // A rotated projection's plan prepares its table, if any, from the
    // rotated rows, which no other plan reads.
    if (p.rotation) return {};
    // Only quantized segments run the plan's tile: float segments alone
    // leave the scratch table as it was.
    const std::vector<QuantizedSegment> &segments = p.blocks().segments;
    const bool tiled =
        std::any_of(segments.begin(), segments.end(), [](const QuantizedSegment &s) { return !s.isFloat(); });
    return tiled && selected.input() != LinearInput::Plain ? PreparedInput{b.input, selected.input()} : b.prepared;
  }
  requireAffineProjection(p, w.matrix);
  if (gate) requireAffineProjection(*gate, w.matrix);
  const AffineWeights &weights = p.affine();
  if (selected.usesQ4Register()) {
    if (b.prepared.layout != LinearInput::Table64 || !b.prepared.source.sameView(b.input))
      graph.add("decode_linear_q4_prepare", {b.input, b.scratch.input, b.scratch.sums},
                k, {k / 32, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1}, {128, 1, 1});
    const AffineWeights &first = gate ? gate->affine() : weights;
    const auto [partials, counters] = splitScratch(b, selected.configuration().splits);
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, first.weights, first.scales, first.biases,
                                             b.output, b.scratch.sums, partials, counters};
    if (gate) bindings.insert(bindings.end(), {weights.weights, weights.scales, weights.biases});
    else if (w.epilogue == LinearEpilogue::Residual) bindings.push_back(b.residual);
    graph.add(kernelInstance(selected.pipeline(), selected.destination()), std::move(bindings),
        Q4Params{n, k},
        {selected.groups(), selected.configuration().splits, w.rows / SPLASH_TARGET_VERIFY_ROWS},
        {128, 1, 1});
    return {b.input, LinearInput::Table64};
  }
  const auto dispatch = [&](std::string_view name,
      std::initializer_list<metal::MetalBuffer> bindings) {
    if (w.phase == LinearPhase::Prefill)
      graph.add(std::string(name), bindings, Q4Params{n, k},
          {selected.storageRows() / kAffinePrefillTileRows, n / selected.tileColumns(), 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    else {
      // Split128 binds its partials and counters after the sequential
      // kernel's buffers; every other decode tile runs one K split.
      const LinearConfig config = selected.configuration();
      std::vector<metal::MetalBuffer> buffers(bindings);
      if (config.tile == LinearTile::Split128)
        buffers.insert(buffers.end(), {b.scratch.partials, b.scratch.counters});
      const std::string kernel = kernelInstance(name, selected.destination());
      const metal::DispatchSize groups{selected.groups(), config.splits, 1};
      const metal::DispatchSize threads{selected.threadsPerThreadgroup(), 1, 1};
      // The persistent tiles stride over the column tiles by their groups.
      if (persistentTile(config.tile))
        graph.add(kernel, std::move(buffers), Q4PersistentParams{n, k, selected.groups()}, groups, threads);
      else
        graph.add(kernel, std::move(buffers), Q4Params{n, k}, groups, threads);
    }
  };
  const bool prefill = w.phase == LinearPhase::Prefill;
  if (w.epilogue == LinearEpilogue::GateUp) {
    const AffineWeights &g = gate->affine();
    if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases,
                                     b.output, weights.weights, weights.scales, weights.biases});
    else {
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases, b.gateScratch});
      dispatch(selected.secondPipeline(),
               {b.input, weights.weights, weights.scales, weights.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate)
    dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                   b.gateScratch, b.output, b.sums, b.downSums});
  else if (w.epilogue == LinearEpilogue::Residual) {
    if (prefill)
      dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                     b.residual, b.output, b.sums});
    else
      dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                     b.residual, b.output});
  } else if (prefill)
    dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases, b.output, b.sums});
  else dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases, b.output});
  return b.prepared;
}

void Linear::addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer sums,
                            const Projection &consumer, uint32_t rows) const {
  validate({{consumer.outputSize, consumer.inputSize}, rows, LinearPhase::Prefill});
  const uint32_t tiles = (rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
  const uint64_t storageRows = uint64_t{tiles} * kAffinePrefillTileRows;
  requireBytes(input, storageRows * consumer.inputSize * 2, "projection input");
  requireBytes(sums, storageRows * (consumer.inputSize / kQuantGroup) * 4, "projection sums");
  graph.add("prefill_linear_q4_sums32", {input, sums}, consumer.inputSize, {tiles, 1, 1});
}
void Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                        metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                        LinearScratch scratch) const {
  add(graph, {.input = input, .output = output, .sums = sums, .scratch = scratch}, p,
      prefillPlan(p, rows, LinearEpilogue::None));
}
void Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                metal::MetalBuffer residual, metal::MetalBuffer output, metal::MetalBuffer sums,
                                uint32_t rows, LinearScratch scratch) const {
  add(graph, {.input = input, .output = output, .sums = sums, .residual = residual, .scratch = scratch}, p,
      prefillPlan(p, rows, LinearEpilogue::Residual));
}
void Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &up,
                                  metal::MetalBuffer gateScratch, metal::MetalBuffer output,
                                  metal::MetalBuffer sums, metal::MetalBuffer downSums, uint32_t rows,
                                  LinearScratch scratch) const {
  add(graph,
      {.input = input, .output = output, .sums = sums, .gateScratch = gateScratch, .downSums = downSums,
       .scratch = scratch},
      up, prefillPlan(up, rows, LinearEpilogue::UpWithGate));
}

} // namespace splash::ops
