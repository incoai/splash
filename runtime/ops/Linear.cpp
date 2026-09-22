#include "Linear.hpp"

#include "metal/abi/Gguf.h"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Linear.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kPrefillRows = 32;
constexpr uint32_t kQuantGroup = 64;
constexpr uint32_t kMaximumSimdgroupSplits = 8;
static_assert(SPLASH_TARGET_VERIFY_ROWS == 8,
              "simdgroup Q4 tiles require eight verify rows per lane");
// Split tiles hold four K partitions, each a whole number of the kernels'
// four-quant-group (256-input) input-sum blocks.
constexpr uint32_t kSplitPartitions = 4;
constexpr uint32_t kSplitInputBlock = kSplitPartitions * 4 * kQuantGroup;
static_assert(sizeof(LinearMatrix) == 8);

bool splitTile(LinearTile tile) noexcept {
  return tile == LinearTile::Split32 || tile == LinearTile::Split64;
}
bool oneLaneTile(LinearTile tile) noexcept {
  return tile == LinearTile::Paired128 || tile == LinearTile::Paired256 || splitTile(tile);
}
// Simdgroups fixed by the kernel instance: split tiles run four partitions of
// one (N32) or two (N64) simdgroups; the paired N256 tile runs four.
std::optional<LinearSimdgroups> fixedSimdgroups(LinearTile tile) noexcept {
  switch (tile) {
  case LinearTile::Split32:
  case LinearTile::Simdgroup:
  case LinearTile::Paired256: return LinearSimdgroups::Four;
  case LinearTile::Split64: return LinearSimdgroups::Eight;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128: return std::nullopt;
  }
  return std::nullopt;
}

void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid Q4 linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid Q4 prefill workload");
  } else if (w.phase == LinearPhase::Decode) {
    if (w.matrix.inputSize % 256 || !w.rows || w.rows % SPLASH_TARGET_VERIFY_ROWS ||
        w.rows > SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid Q4 decode workload");
  } else {
    throw std::invalid_argument("invalid Q4 linear phase");
  }
  if (w.epilogue != LinearEpilogue::None && w.epilogue != LinearEpilogue::Residual &&
      w.epilogue != LinearEpilogue::GateUp && w.epilogue != LinearEpilogue::UpWithGate)
    throw std::invalid_argument("invalid Q4 linear epilogue");
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes) {
  if (bytes && (!buffer || buffer.sizeBytes() < bytes))
    throw std::invalid_argument("Q4 buffer is below plan requirement");
}

void requireProjection(const Q4Projection &p, LinearMatrix matrix) {
  if (p.outputSize != matrix.outputSize || p.inputSize != matrix.inputSize)
    throw std::invalid_argument("Q4 projection does not match plan");
  requireBytes(p.weights, uint64_t{matrix.outputSize} * matrix.inputSize / 2);
  const uint64_t bytes = uint64_t{matrix.outputSize} * (matrix.inputSize / kQuantGroup) * 2;
  requireBytes(p.scales, bytes);
  requireBytes(p.biases, bytes);
}

void account(Q4DispatchStats &stats, uint32_t lanes, uint32_t count) noexcept {
  if (lanes == 1) return;
  stats.fusedSourceOperations += uint64_t{lanes} * count;
  if (lanes == 2) stats.m16Dispatches += count;
  else if (lanes == 3) stats.m24Dispatches += count;
  else stats.m32Dispatches += count;
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid Q4 decode batch width");
  return {matrix, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
}

// The four-simdgroup kernels: every prefill N128 tile, the decode M24 N128
// plain and residual projections, all matrix row tiles, and the one-lane
// Split32 (plain, residual and gate/up) and Paired256 (plain) tiles.
bool supportsFourSimdgroups(LinearWorkload w, LinearTile tile) noexcept {
  if (tile == LinearTile::Simdgroup) return w.phase == LinearPhase::Decode;
  if (tile == LinearTile::Split32)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS;
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

uint32_t LinearPlan::storageRows() const noexcept {
  return workload_.phase == LinearPhase::Prefill
      ? ((workload_.rows + kPrefillRows - 1) / kPrefillRows) * kPrefillRows : workload_.rows;
}
uint32_t LinearPlan::tileColumns() const noexcept {
  switch (config_.tile) {
  case LinearTile::Simdgroup: return workload_.epilogue == LinearEpilogue::GateUp ? 32 : 64;
  case LinearTile::Split32: return 32;
  case LinearTile::Split64: return 64;
  case LinearTile::N256:
  case LinearTile::Paired256: return 256;
  case LinearTile::N128:
  case LinearTile::Paired128: return 128;
  }
  return 0;
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  return static_cast<uint32_t>(config_.simdgroups) * 32;
}
uint32_t LinearPlan::partialSums() const noexcept {
  return usesSimdgroup() ? config_.splits : splitTile(config_.tile) ? kSplitPartitions : 1;
}
bool LinearPlan::usesSimdgroup() const noexcept { return config_.tile == LinearTile::Simdgroup; }
LinearScratchSize LinearPlan::scratchSize() const noexcept {
  if (!usesSimdgroup()) return {};
  const auto [n, k] = workload_.matrix;
  const uint64_t rows = workload_.rows;
  const uint64_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  // Each row tile owns two fp32 fragment streams per K partition and one
  // completion counter per column tile. Single-partition kernels use neither.
  return {rows * k * sizeof(uint16_t), rows * (k / kQuantGroup) * sizeof(float),
          config_.splits > 1 ? config_.splits * 2 * rows * n * sizeof(float) : sizeof(float),
          config_.splits > 1 ? lanes * (n / tileColumns()) * sizeof(uint32_t) : sizeof(uint32_t)};
}

uint64_t LinearPlan::sumsBytes() const noexcept {
  return workload_.phase == LinearPhase::Prefill
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp && !secondPipeline_.empty());
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config)
    : workload_(w), config_(config) {
  validate(w);
  if (config.tile != LinearTile::Simdgroup && config.splits != 1)
    throw std::invalid_argument("K splits require the simdgroup Q4 tile");
  if (config.tile != LinearTile::N128 && config.tile != LinearTile::N256 &&
      config.tile != LinearTile::Simdgroup && !oneLaneTile(config.tile))
    throw std::invalid_argument("invalid Q4 linear tile");
  if ((config.simdgroups != LinearSimdgroups::Four &&
       config.simdgroups != LinearSimdgroups::Eight) ||
      (config.simdgroups == LinearSimdgroups::Four &&
       !supportsFourSimdgroups(w, config.tile)))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (const auto fixed = fixedSimdgroups(config.tile); fixed && config.simdgroups != *fixed)
    throw std::invalid_argument("Q4 tile requires its kernel's simdgroup count");
  if (w.matrix.outputSize % tileColumns())
    throw std::invalid_argument("Q4 matrix is not divisible by tile columns");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  if (w.phase == LinearPhase::Prefill) {
    if (config.groups || oneLaneTile(config.tile) || usesSimdgroup())
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
  if (!config.groups || config.groups > w.matrix.outputSize / tileColumns())
    throw std::invalid_argument("invalid Q4 decode group count");
  const uint32_t lane = w.rows / SPLASH_TARGET_VERIFY_ROWS - 1;
  if (oneLaneTile(config.tile) && (lane != 0 || w.matrix.outputSize % 256))
    throw std::invalid_argument("paired or split Q4 tile requires one lane and paired columns");
  if (usesSimdgroup()) {
    const uint32_t groups = w.matrix.inputSize / kQuantGroup;
    if (config.groups != w.matrix.outputSize / tileColumns() ||
        !config.splits || config.splits > kMaximumSimdgroupSplits || (config.splits & (config.splits - 1)) ||
        groups % config.splits)
      throw std::invalid_argument("simdgroup Q4 requires full column grid and whole power-of-two K partitions");
    pipeline_ = w.epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
        residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
    return;
  }
  if (splitTile(config.tile)) {
    // Each partition takes a quarter of K in whole 256-input blocks, and the
    // split kernels are dispatched one threadgroup per tile.
    if (w.matrix.inputSize % kSplitInputBlock ||
        config.groups != w.matrix.outputSize / tileColumns())
      throw std::invalid_argument("split Q4 tile requires K % 1024 == 0 and the full grid");
    const bool n32 = config.tile == LinearTile::Split32;
    if (w.epilogue == LinearEpilogue::GateUp) {
      // Only the N32 two-stream split kernel is instantiated.
      if (!n32) throw std::invalid_argument("Q4 split gate/up requires Split32");
      pipeline_ = "decode_linear_q4_n32_split4_gate_up";
    } else if (residual) {
      pipeline_ = n32 ? "decode_linear_q4_n32_split4_residual"
                      : "decode_linear_q4_n64_split4_residual";
    } else {
      pipeline_ = n32 ? "decode_linear_q4_n32_split4" : "decode_linear_q4_n64_split4";
    }
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
// Gate/up uses the conservative limit shared by both devices. Its many-wave
// threshold follows N256; the four-simdgroup threshold scales from N128. Those
// two extrapolations remain unmeasured.
constexpr DecodeGroupPolicy kN128Groups{4, 4, 12}, kN128M16Groups{5, 4, 12},
    kN256Groups{3, 3, 8}, kGateUpGroups{3, 3, 8},
    kFourSimdgroupGroups{8, 8, 24};
// Apple9 retains its measured gate/up clamp. The round-robin policy above was
// measured on Apple10; applying it to Apple9 requires separate calibration.
constexpr double kApple9GateUpGroupsPerCore = 2.25;

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
// Apple9 N256 prefill needs eight threadgroups per core to amortize its larger
// tile. Paired-A/B tuning (tune-kernels) and the per-shape microprofile
// (benchmark-prefill) on a 32-core Apple9 GPU (M4 Max) measured the
// four-simdgroup N128 tile ahead of N256 on every prefill shape and probed
// row count: +6..10% GPU wherever the margin cleared the tuning threshold,
// never behind. Apple9 GPUs at or below that measured core count therefore
// share the Apple10 prefill rule. Larger Apple9 GPUs (40-core class) keep the
// wide-tile rule below; it was sized for them and remains unremeasured there.
constexpr uint32_t kApple9MeasuredPrefillCores = 32;
constexpr double kApple9WidePrefillGroupsPerCore = 8.0;
// Missing core metadata uses one intermediate estimate for all families.
// This is a fallback, not a calibrated optimum. Reported counts always win.
constexpr uint32_t kAssumedGpuCores = 32;

// Apple10 wide plain projections reduce input re-reads with paired N256
// tiles at one resident wave, measured on 16/20-core GPUs. Split-K remains
// an offline candidate: its reassociation reduced speculative acceptance
// on some measured prompts. Apple9's simdgroup policy is independent.
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

Q4Linear::Q4Linear(const DeviceCapabilities &device) noexcept
    : appleGpuFamily_(device.appleGpuFamily),
      gpuCores_(device.gpuCoreCount ? device.gpuCoreCount : kAssumedGpuCores) {}

// GPU family selects variants; core count and workload tile counts determine
// parallelism.
LinearConfig Q4Linear::baseline(LinearWorkload w) const {
  validate(w);
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  if (w.phase == LinearPhase::Prefill) {
    if (appleGpuFamily_ >= 10 || gpuCores_ <= kApple9MeasuredPrefillCores)
      return {LinearTile::N128, 0, LinearSimdgroups::Four};
    const uint32_t rowTiles = (w.rows + kPrefillRows - 1) / kPrefillRows;
    const bool wide = double(rowTiles) * tiles256 >=
        kApple9WidePrefillGroupsPerCore * gpuCores_;
    return {w.epilogue == LinearEpilogue::UpWithGate || wide ? LinearTile::N256
                                                              : LinearTile::N128, 0};
  }
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  // Keep the existing broad-column plain projection path for wider batches:
  // independent row tiles repeat its weight stream. Reuse the existing
  // two-N256-tiles-per-core boundary rather than model-specific dimensions.
  const bool widePlain = lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * gpuCores_;
  if (appleGpuFamily_ == 9 && !widePlain) {
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    const uint32_t grid = w.matrix.outputSize / columns, groups = w.matrix.inputSize / 64;
    uint32_t splits = 1;
    // Aim for sixteen independent column/K groups per core, retaining at
    // least twelve quant groups per partition to amortize the reduction.
    while (splits < kMaximumSimdgroupSplits && uint64_t(grid) * splits < 16ULL * gpuCores_ &&
           groups % (2 * splits) == 0 && groups / (2 * splits) >= 12)
      splits *= 2;
    return {LinearTile::Simdgroup, grid, LinearSimdgroups::Four, splits};
  }
  if (appleGpuFamily_ >= 10 && lanes == 1)
    if (const auto config = apple10OneLaneConfig(w, gpuCores_)) return *config;
  // Apple9 keeps its one-tile grids (see kApple9GateUpGroupsPerCore).
  const auto groups = [&](uint32_t tiles, DecodeGroupPolicy policy) {
    return appleGpuFamily_ >= 10 ? decodeGroups(tiles, gpuCores_, policy)
                                 : tiles;
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (appleGpuFamily_ < 10) {
      const auto resident = static_cast<uint32_t>(
          std::max(1L, std::lround(kApple9GateUpGroupsPerCore * gpuCores_)));
      return {LinearTile::N256, std::min(tiles256, resident)};
    }
    return {LinearTile::N256, groups(tiles256, kGateUpGroups)};
  }
  // Pipelined N128 hides the latency of a single lane's weight stream.
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, kN128Groups)};
  // With at most one N128 tile per core, longer M24 dot products benefit
  // from eight groups. Short K and wider grids retain the four-group path.
  if (appleGpuFamily_ >= 10 && lanes == 3 && tiles128 <= gpuCores_ &&
      w.matrix.inputSize >= 4096)
    return {LinearTile::N128, tiles128, LinearSimdgroups::Eight};
  // M24 plain projections benefit from four SIMD groups on Apple9 too.
  // Apple9 residual projections retain eight groups with compact prefix
  // traversal; Apple10 uses four groups outside the narrow-grid case above.
  if (lanes == 3 && (appleGpuFamily_ >= 10 ||
      (appleGpuFamily_ == 9 && w.epilogue == LinearEpilogue::None)))
    return {LinearTile::N128, groups(tiles128, kFourSimdgroupGroups),
            LinearSimdgroups::Four};
  if (lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * gpuCores_)
    return {LinearTile::N256, groups(tiles256, kN256Groups)};
  return {LinearTile::N128,
          groups(tiles128, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

LinearPlan Q4Linear::plan(LinearWorkload workload) const {
  const auto found = std::lower_bound(choices_.begin(), choices_.end(), workload,
      [](const LinearChoice &choice, LinearWorkload key) { return choice.workload < key; });
  return LinearPlan(workload, found != choices_.end() && found->workload == workload
      ? found->configuration : baseline(workload));
}
LinearPlan Q4Linear::plan(LinearWorkload workload, LinearConfig config) {
  return LinearPlan(workload, config);
}
void Q4Linear::setChoices(std::span<const LinearChoice> choices) {
  std::vector<LinearChoice> pending(choices.begin(), choices.end());
  for (const auto &choice : pending) (void)plan(choice.workload, choice.configuration);
  std::sort(pending.begin(), pending.end(), [](const auto &a, const auto &b) {
    return a.workload < b.workload;
  });
  for (size_t i = 1; i < pending.size(); ++i)
    if (pending[i - 1].workload == pending[i].workload)
      throw std::invalid_argument("duplicate Q4 linear choice");
  choices_ = std::move(pending);
}

std::vector<LinearPlan> Q4Linear::candidates(LinearWorkload w) const {
  std::vector<LinearPlan> result;
  result.reserve(kMaximumCandidates);
  result.push_back(LinearPlan(w, baseline(w)));
  const auto append = [&](LinearConfig config) {
    for (const auto &existing : result)
      if (existing.configuration() == config) return;
    result.push_back(LinearPlan(w, config));
  };
  for (const auto tile : {LinearTile::N128, LinearTile::N256, LinearTile::Paired128}) {
    const uint32_t columns = tile == LinearTile::N256 ? 256 : 128;
    if (w.matrix.outputSize % columns ||
        (tile == LinearTile::Paired128 && (w.phase != LinearPhase::Decode ||
         w.rows != SPLASH_TARGET_VERIFY_ROWS || w.matrix.outputSize % 256)) ||
        (w.epilogue == LinearEpilogue::GateUp && tile != LinearTile::N256) ||
        (w.phase == LinearPhase::Decode && w.epilogue == LinearEpilogue::Residual && tile == LinearTile::N256))
      continue;
    if (w.phase == LinearPhase::Prefill) {
      // The fused up projection has no eight-simdgroup N128 kernel.
      if (tile == LinearTile::N256 || w.epilogue != LinearEpilogue::UpWithGate)
        append({tile, 0});
      if (supportsFourSimdgroups(w, tile))
        append({tile, 0, LinearSimdgroups::Four});
    } else {
      const uint32_t tiles = w.matrix.outputSize / columns;
      // Sample two, three and four groups per core plus the full grid.
      // Always retain the measured baseline above, including its balanced
      // group count. Fixed counts tied to one GPU miss these waves elsewhere.
      for (const uint32_t groups : {2 * gpuCores_, 3 * gpuCores_, 4 * gpuCores_, tiles}) {
        append({tile, std::min(groups, tiles)});
        if (supportsFourSimdgroups(w, tile))
          append({tile, std::min(groups, tiles), LinearSimdgroups::Four});
      }
    }
  }
  if (w.phase == LinearPhase::Decode && appleGpuFamily_ == 9) {
    const uint32_t n = w.matrix.outputSize;
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    for (uint32_t splits = 1; splits <= kMaximumSimdgroupSplits; splits *= 2)
      if ((w.matrix.inputSize / kQuantGroup) % splits == 0)
        append({LinearTile::Simdgroup, n / columns, LinearSimdgroups::Four, splits});
  }
  // One-lane tiles: the split forms at their full grid and the paired N256
  // tile at one resident wave and at its full grid.
  if (w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS) {
    const uint32_t n = w.matrix.outputSize;
    if (w.matrix.inputSize % kSplitInputBlock == 0) {
      append({LinearTile::Split32, n / 32, LinearSimdgroups::Four});
      if (w.epilogue != LinearEpilogue::GateUp)
        append({LinearTile::Split64, n / 64, LinearSimdgroups::Eight});
    }
    if (w.epilogue == LinearEpilogue::None)
      for (const uint32_t groups : {kPaired256WaveGroupsPerCore * gpuCores_, n / 256})
        append({LinearTile::Paired256, std::min(groups, n / 256), LinearSimdgroups::Four});
  }
  return result;
}

LinearScratchSize Q4Linear::decodeScratchSize(LinearWorkload w) const {
  auto size = LinearPlan(w, baseline(w)).scratchSize();
  const auto selected = plan(w).scratchSize();
  size.input = std::max(size.input, selected.input);
  size.sums = std::max(size.sums, selected.sums);
  size.partials = std::max(size.partials, selected.partials);
  size.counters = std::max(size.counters, selected.counters);
  return size;
}


namespace {

// ---- GGUF quantized dispatch (see kernels/shared/gguf_linear.metal). Decode-style
// kernels stage per-simdgroup fp16 tiles for <= 32 rows; prefill uses shared
// 128-row tiles. Narrow projections use split-K with an fp32 reduce.
constexpr uint32_t kKQDecodeTileColumns = 64;
constexpr uint32_t kKQDecodeThreads = 64;
constexpr uint32_t kKQPrefillRows = 128;
constexpr uint32_t kKQPrefillTileColumns = 64;
constexpr uint32_t kKQPrefillThreads = 128;

std::string kqKernel(const char *family, const char *format, uint32_t rows) {
  return std::string(family) + "_" + format + "_m" + std::to_string(rows) + "_c32_sg2_k32_b2_p1";
}
std::string kqPrefillKernel(const char *family, const char *format) {
  return std::string(family) + "_" + format + "_r32_sg4_n64_k64_p1";
}
uint32_t ggufSplits(uint32_t n, uint32_t k, uint32_t rows) {
  if (n <= 1024) return 8;
  if (n <= 6144) return k <= 6144 ? 4 : 8;   // serialized M=8 sweep: 4 splits best for out (K 6144), 8 for down (K 17408)
  if (n <= 12288) return rows <= 16 ? 4 : 2;
  return 1;
}

void addGguf(metal::CommandGraph &graph, const LinearBuffers &b, const Q4Projection &p,
               LinearWorkload w, const Q4Projection *gate, Q4DispatchStats *stats) {
  const auto [n, k] = w.matrix;
  const auto requireSegments = [&](const Q4Projection &proj) {
    uint32_t covered = 0;
    for (const GgufSegment &s : proj.gguf) {
      if (s.inputSize != k || s.columnOffset != covered || s.outputSize % kKQDecodeTileColumns)
        throw std::invalid_argument("GGUF segments do not tile the projection");
      covered += s.outputSize;
    }
    if (covered != n) throw std::invalid_argument("GGUF segments do not cover the projection");
  };
  requireSegments(p);
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!gate || gate->gguf.empty()) throw std::invalid_argument("GGUF gate projection is missing");
    requireSegments(*gate);
  } else if (gate) throw std::invalid_argument("unexpected GGUF gate projection");
  // Rows the kernels compute: 8/16/32 for decode (24 lanes-rows run as a zero-padded 32-row
  // tile), otherwise 128-row prefill tiles. Arena buffers hold 32 decode rows and the full
  // prefill token budget, so the padded rows stay inside every bound buffer.
  // Decode rows are exact (8/16/24/32 kernels); prefill rounds up to 128-row tiles inside
  // the budget-sized prefill buffers, and <= 32 prefill rows use the 32-row decode kernels.
  const bool small = w.rows <= 32;
  const uint32_t rows = w.phase == LinearPhase::Decode ? w.rows
                        : small ? 32u : (w.rows + kKQPrefillRows - 1) / kKQPrefillRows * kKQPrefillRows;
  if (small && rows != 8 && rows != 16 && rows != 24 && rows != 32)
    throw std::invalid_argument("GGUF decode rows must be 8, 16, 24 or 32");
  const auto need = [&](const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
    if (buffer.sizeBytes() < bytes)
      throw std::invalid_argument(std::string("GGUF ") + what + " buffer holds " +
                                  std::to_string(buffer.sizeBytes()) + " bytes, needs " +
                                  std::to_string(bytes) + " (rows " + std::to_string(rows) +
                                  ", workload rows " + std::to_string(w.rows) + ", n " + std::to_string(n) + ")");
  };
  need(b.input, uint64_t{rows} * k * 2, "input");
  need(b.output, uint64_t{rows} * n * 2, "output");
  if (w.epilogue == LinearEpilogue::Residual) need(b.residual, uint64_t{rows} * n * 2, "residual");
  if (w.epilogue == LinearEpilogue::UpWithGate || w.epilogue == LinearEpilogue::GateUp)
    need(b.gateScratch, uint64_t{rows} * n * 2, "gate scratch");
  const metal::MetalBuffer &input = b.input;
  const auto epilogueId = [](LinearEpilogue e) { return e == LinearEpilogue::Residual ? GGUF_EPILOGUE_RESIDUAL
                                                      : e == LinearEpilogue::UpWithGate ? GGUF_EPILOGUE_UP_WITH_GATE : GGUF_EPILOGUE_NONE; };
  const auto plane1 = [](const GgufSegment &s) { return s.plane1 ? s.plane1 : s.meta; };
  if (small) {
    // Decode rows: one dispatch per projection. Fused multi-type segments (qkv|z|ab, q|k|v), gate+up in one kernel,
    // split-K with in-kernel reduction for narrow projections, plain tiles otherwise.
    if (w.epilogue == LinearEpilogue::GateUp) {
      if (gate->gguf.size() != 1 || p.gguf.size() != 1) throw std::invalid_argument("GGUF gate/up must be single tensors");
      const GgufSegment &g = gate->gguf.front(), &u = p.gguf.front();
      graph.add("gguf_gateup_m" + std::to_string(rows), {input, g.plane0, plane1(g), g.meta, u.plane0, plane1(u), u.meta, b.output},
                GgufGateUpParams{k, n, n, g.formatId, u.formatId}, {n / kKQDecodeTileColumns, 1, 1}, {kKQDecodeThreads, 1, 1});
    } else if (p.gguf.size() > 1) {
      if (p.gguf.size() > 3 || w.epilogue != LinearEpilogue::None) throw std::invalid_argument("GGUF fused projection needs <= 3 segments and no epilogue");
      // Dispatch order = tile order: put the segments with the most bytes per tile first so their
      // threadgroups start early instead of forming the tail of the dispatch (alpha/beta are Q8_0).
      std::vector<const GgufSegment *> order;
      for (const GgufSegment &s : p.gguf) order.push_back(&s);
      const auto bitsPerWeight = [](const GgufSegment &s) {
        return (s.p0 + s.p1) * 8.0 / 32.0 + s.metaBytes * 8.0 / (32.0 * s.metaGroups); };
      std::stable_sort(order.begin(), order.end(), [&](const GgufSegment *a, const GgufSegment *c) {
        return bitsPerWeight(*a) > bitsPerWeight(*c); });
      GgufFusedParams fp{k, n, static_cast<uint32_t>(p.gguf.size()), 0, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      std::vector<metal::MetalBuffer> bufs{input};
      for (size_t i = 0; i < 3; ++i) {
        const GgufSegment &s = *order[std::min(i, order.size() - 1)];
        if (i < order.size()) { fp.cols[i] = s.outputSize; fp.fmt[i] = s.formatId; fp.offset[i] = s.columnOffset; }
        bufs.insert(bufs.end(), {s.plane0, plane1(s), s.meta});
      }
      bufs.push_back(b.output);
      graph.add("gguf_fused_m" + std::to_string(rows), std::move(bufs), fp, {n / kKQDecodeTileColumns, 1, 1}, {kKQDecodeThreads, 1, 1});
    } else {
      const GgufSegment &s = p.gguf.front();
      const uint32_t tiles = n / kKQDecodeTileColumns, splits = ggufSplits(n, k, rows);
      const metal::MetalBuffer aux = w.epilogue == LinearEpilogue::Residual ? b.residual
                                   : w.epilogue == LinearEpilogue::UpWithGate ? b.gateScratch : b.output;
      if (splits > 1) {
        need(p.kqPartials, uint64_t{splits} * rows * n * 4, "partials");
        need(p.kqCounters, uint64_t{tiles} * 4, "counters");
        graph.add(std::string("gguf_splitk_") + s.format + "_m" + std::to_string(rows),
                  {input, s.plane0, plane1(s), s.meta, p.kqPartials, p.kqCounters, b.output, aux},
                  GgufSplitParams{n, k, splits, n, 0, epilogueId(w.epilogue), p.ggufPermuteK}, {tiles, splits, 1}, {kKQDecodeThreads, 1, 1});
      } else if (w.epilogue == LinearEpilogue::None) {
        graph.add(kqKernel("sga", s.format, rows), {input, s.plane0, plane1(s), s.meta, b.output},
                  GgufParams{n, k, tiles, n, 0, p.ggufPermuteK}, {tiles, 1, 1}, {kKQDecodeThreads, 1, 1});
      } else {
        graph.add(kqKernel(w.epilogue == LinearEpilogue::Residual ? "sgr" : "sgg", s.format, rows),
                  {input, s.plane0, plane1(s), s.meta, b.output, aux}, GgufParams{n, k, tiles, n, 0, p.ggufPermuteK}, {tiles, 1, 1}, {kKQDecodeThreads, 1, 1});
      }
    }
  } else {
    // Prefill: 128-row shared-B tiles, one dispatch per segment; gate+up as gate then up-with-gate.
    const auto gemm = [&](const Q4Projection &proj, const metal::MetalBuffer &output, uint32_t epilogue, const metal::MetalBuffer &aux) {
      for (const GgufSegment &s : proj.gguf) {
        const uint32_t tiles = s.outputSize / kKQPrefillTileColumns;
        const GgufParams params{s.outputSize, k, 0, n, s.columnOffset, proj.ggufPermuteK};
        if (epilogue == GGUF_EPILOGUE_NONE)
          graph.add(kqPrefillKernel("pfa", s.format), {input, s.plane0, plane1(s), s.meta, output}, params,
                    {rows / kKQPrefillRows, tiles, 1}, {kKQPrefillThreads, 1, 1});
        else
          graph.add(kqPrefillKernel(epilogue == GGUF_EPILOGUE_RESIDUAL ? "pfr" : "pfg", s.format),
                    {input, s.plane0, plane1(s), s.meta, output, aux}, params, {rows / kKQPrefillRows, tiles, 1}, {kKQPrefillThreads, 1, 1});
      }
    };
    switch (w.epilogue) {
    case LinearEpilogue::None: gemm(p, b.output, GGUF_EPILOGUE_NONE, {}); break;
    case LinearEpilogue::Residual: gemm(p, b.output, GGUF_EPILOGUE_RESIDUAL, b.residual); break;
    case LinearEpilogue::UpWithGate: gemm(p, b.output, GGUF_EPILOGUE_UP_WITH_GATE, b.gateScratch); break;
    case LinearEpilogue::GateUp:
      gemm(*gate, b.gateScratch, GGUF_EPILOGUE_NONE, {});
      gemm(p, b.output, GGUF_EPILOGUE_UP_WITH_GATE, b.gateScratch);
      break;
    }
  }
  if (stats && w.phase == LinearPhase::Decode)
    account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1);
}

} // namespace

void Q4Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Q4Projection &p, const LinearPlan &selected, const Q4Projection *gate,
    Q4DispatchStats *stats) const {
  if (!p.gguf.empty()) { addGguf(graph, b, p, selected.workload(), gate, stats); return; }
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  requireProjection(p, w.matrix);
  requireBytes(b.input, uint64_t{selected.storageRows()} * k * 2);
  requireBytes(b.output, uint64_t{selected.storageRows()} * n * 2);
  requireBytes(b.sums, selected.sumsBytes());
  requireBytes(b.gateScratch, selected.gateScratchBytes());
  requireBytes(b.downSums, selected.downSumsBytes());
  if (w.epilogue == LinearEpilogue::Residual)
    requireBytes(b.residual, uint64_t{selected.storageRows()} * n * 2);
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!gate) throw std::invalid_argument("Q4 gate projection is missing");
    requireProjection(*gate, w.matrix);
  } else if (gate) throw std::invalid_argument("unexpected Q4 gate projection");
  if (selected.usesSimdgroup()) {
    const auto size = selected.scratchSize();
    requireBytes(b.scratch.input, size.input);
    requireBytes(b.scratch.sums, size.sums);
    requireBytes(b.scratch.partials, size.partials);
    requireBytes(b.scratch.counters, size.counters);
    if (!b.inputPrepared)
      graph.add("decode_linear_q4_prepare", {b.input, b.scratch.input, b.scratch.sums},
                k, {k / 32, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1}, {128, 1, 1});
    const auto &first = gate ? *gate : p;
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, first.weights,
        first.scales, first.biases, b.output, b.scratch.sums,
        b.scratch.partials, b.scratch.counters};
    if (gate) bindings.insert(bindings.end(), {p.weights, p.scales, p.biases});
    else if (w.epilogue == LinearEpilogue::Residual) bindings.push_back(b.residual);
    graph.add(std::string(selected.pipeline()), std::move(bindings),
        Q4Params{n, k, selected.configuration().splits},
        {selected.configuration().groups, selected.configuration().splits,
         w.rows / SPLASH_TARGET_VERIFY_ROWS}, {128, 1, 1});
    if (stats) account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1);
    return;
  }
  const auto dispatch = [&](std::string_view name,
      std::initializer_list<metal::MetalBuffer> bindings) {
    if (w.phase == LinearPhase::Prefill)
      graph.add(std::string(name), bindings,
          Q4PrefillParams{w.matrix.outputSize, w.matrix.inputSize},
          {selected.storageRows() / kPrefillRows, n / selected.tileColumns(), 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    else {
      const uint32_t groups = selected.configuration().groups;
      graph.add(std::string(name), bindings, Q4Params{n, k, groups}, {groups, 1, 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    }
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales, gate->biases,
          b.output, p.weights, p.scales, p.biases});
    else {
      dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales, gate->biases, b.gateScratch});
      dispatch(selected.secondPipeline(), {b.input, p.weights, p.scales, p.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
        b.gateScratch, b.output, b.sums, b.downSums});
  else if (w.epilogue == LinearEpilogue::Residual) {
    if (w.phase == LinearPhase::Prefill)
      dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.residual, b.output, b.sums});
    else dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.residual, b.output});
  } else if (w.phase == LinearPhase::Prefill)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.output, b.sums});
  else dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.output});
  if (stats && w.phase == LinearPhase::Decode)
    account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, selected.secondPipeline().empty() ? 1 : 2);
}

void Q4Linear::addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input,
    metal::MetalBuffer sums, LinearMatrix matrix, uint32_t rows) const {
  validate({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None});
  const uint32_t tiles = (rows + kPrefillRows - 1) / kPrefillRows;
  const uint64_t storageRows = uint64_t{tiles} * kPrefillRows;
  requireBytes(input, storageRows * matrix.inputSize * 2);
  requireBytes(sums, storageRows * (matrix.inputSize / kQuantGroup) * 4);
  graph.add("prefill_linear_q4_sums32", {input, sums},
      Q4PrefillParams{matrix.outputSize, matrix.inputSize}, {tiles, 1, 1});
}
void Q4Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, metal::MetalBuffer sums,
    LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, {}, {}, {}}, p,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None}));
}
void Q4Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer residual, metal::MetalBuffer output,
    metal::MetalBuffer sums, LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, residual, {}, {}}, p,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::Residual}));
}
void Q4Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &up, metal::MetalBuffer gateScratch, metal::MetalBuffer output,
    metal::MetalBuffer sums, metal::MetalBuffer downSums, LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, {}, gateScratch, downSums}, up,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::UpWithGate}));
}
void Q4Linear::addDecode(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, LinearMatrix matrix, LinearScratch scratch) const {
  add(graph, {input, output, {}, {}, {}, {}, scratch}, p, plan(decode(matrix, 1, LinearEpilogue::None)));
}
void Q4Linear::addDecodeBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, LinearMatrix matrix,
    uint32_t lanes, Q4DispatchStats &stats, LinearScratch scratch, bool inputPrepared) const {
  add(graph, {input, output, {}, {}, {}, {}, scratch, inputPrepared}, p, plan(decode(matrix, lanes, LinearEpilogue::None)), nullptr, &stats);
}
void Q4Linear::addResidualBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer residual, metal::MetalBuffer output,
    LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats, LinearScratch scratch, bool inputPrepared) const {
  add(graph, {input, output, {}, residual, {}, {}, scratch, inputPrepared}, p, plan(decode(matrix, lanes, LinearEpilogue::Residual)), nullptr, &stats);
}
void Q4Linear::addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &gate, const Q4Projection &up, metal::MetalBuffer gateScratch,
    metal::MetalBuffer output, LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats, LinearScratch scratch, bool inputPrepared) const {
  add(graph, {input, output, {}, {}, gateScratch, {}, scratch, inputPrepared}, up, plan(decode(matrix, lanes, LinearEpilogue::GateUp)), &gate, &stats);
}

} // namespace splash::ops
