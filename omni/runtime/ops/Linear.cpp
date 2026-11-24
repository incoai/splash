#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
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
  return tile == LinearTile::Paired128 || tile == LinearTile::Paired256 || tile == LinearTile::LaneSplit128 ||
         tile == LinearTile::Qmv;
}
// The decode tiles whose threadgroups stream several column tiles.
bool persistentTile(LinearTile tile) noexcept {
  return tile == LinearTile::N128 || tile == LinearTile::N256 || tile == LinearTile::Paired128 ||
         tile == LinearTile::Paired256;
}
// Simdgroups fixed by the kernel instance: the paired N256 tile runs four,
// Split128 and LaneSplit128 the N128 tile's eight.
std::optional<LinearSimdgroups> fixedSimdgroups(LinearTile tile) noexcept {
  switch (tile) {
  case LinearTile::Q4Register:
  case LinearTile::Paired256: return LinearSimdgroups::Four;
  case LinearTile::Split128:
  case LinearTile::LaneSplit128:
  case LinearTile::Qmv: return LinearSimdgroups::Eight;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128: return std::nullopt;
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
  if (w.liveRows > w.rows || (w.liveRows && w.phase == LinearPhase::Prefill))
    throw std::invalid_argument("invalid live rows of a linear workload");
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue, uint32_t liveRows = 0) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid linear decode batch width");
  return {matrix, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue, WeightLayout::Affine64, liveRows};
}

// A Qmv plan's folded norm repeats the row's reduction in every
// threadgroup. Up to this many outputs (every Omni layer and predictor-head
// projection) that ties with the norm dispatch it replaces or beats it
// (omni-dq4-bench --qmv, M5 Pro, weights from DRAM: predictor QKV, gate/up
// and head 0.6-1.1 us faster, talker QKV and gate/up within 1.5 us; whole
// ASR steps 4.11 against 4.10 ms) and saves the dispatch's encoding; the ASR
// head's 152064 outputs lose 606 against 586 us.
constexpr uint32_t kQmvFoldedNormOutputs = 16384;

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

// Throws if `p` is a view of the leading inputs of wider weight rows
// (Projection::leadingInputs) that `plan` has no kernel instance for: only the
// prefill residual tiles (N128, N256) read one (leadingInputsInstance).
void requireLeadingInputs(const LinearPlan &plan, const Projection &p) {
  if (!p.planeInputs()) return;
  const LinearWorkload w = plan.workload();
  const LinearTile tile = plan.configuration().tile;
  if (w.phase != LinearPhase::Prefill || w.epilogue != LinearEpilogue::Residual ||
      (tile != LinearTile::N128 && tile != LinearTile::N256))
    throw std::invalid_argument("a view of leading inputs runs only the prefill residual tiles");
}

// Throws unless views of `p`'s planes can stand for it (takesPlaneViews).
void requirePlaneViews(const Projection &p) {
  if (p.takesPlaneViews()) return;
  throw std::invalid_argument("views of a projection's planes take its own planes, not a view");
}

} // namespace

// Every plane holds its rows in tiles of SPLASH_AFFINE_TILE_ROWS rows, each
// tile's groups in order, so the leading rows' tiles lead it.
Projection Projection::leadingRows(const metal::MetalBackend &backend, uint32_t rows) const {
  requirePlaneViews(*this);
  if (!rows || rows % SPLASH_AFFINE_TILE_ROWS || rows > outputSize)
    throw std::invalid_argument("a view of leading rows takes whole plane tiles of the projection's rows");
  // The leading rows of a plane of `rowBytes` bytes per row.
  const auto view = [&](const metal::MetalBuffer &plane, uint64_t rowBytes) {
    return backend.view(plane, 0, rows * rowBytes);
  };
  const uint64_t groups = inputSize / kQuantGroup;
  const AffineWeights &planes = affine();
  return Projection(rows, inputSize,
                    AffineWeights{view(planes.weights, groups * kQuantGroup / 2), view(planes.scales, groups * 2),
                                  view(planes.biases, groups * 2)});
}

Projection Projection::leadingInputs(uint32_t inputs) const {
  requirePlaneViews(*this);
  if (!inputs || inputs > inputSize || inputs % kQuantGroup)
    throw std::invalid_argument("a view of leading inputs takes whole quant groups of the projection's");
  Projection result(outputSize, inputs, affine());
  result.planeInputs_ = inputSize;
  return result;
}

// Table64 holds one fp32 sum per 64 inputs of each row.
uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept {
  return layout == LinearInput::Table64 ? uint64_t{width} * rows / 16 : 0;
}

void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows) {
  if (layout == LinearInput::Plain || !rows || rows % SPLASH_TARGET_VERIFY_ROWS || width % 64)
    throw std::invalid_argument("invalid linear table geometry");
  requireBytes(scratch.input, tableBytes(width, rows), "linear table");
  requireBytes(scratch.sums, tableSumsBytes(layout, width, rows), "linear table sums");
}

const char *tableSuffix(LinearInput layout) noexcept {
  return layout == LinearInput::Table64 ? "_table64" : "";
}

void requireAffineProjection(const Projection &p, LinearMatrix matrix) {
  if (p.layout() != WeightLayout::Affine64 || p.outputSize != matrix.outputSize ||
      p.inputSize != matrix.inputSize)
    throw std::invalid_argument("affine projection does not match plan");
  // Each plane holds a scale, a bias or 64 weights per row and group, in
  // tiles of SPLASH_AFFINE_TILE_ROWS rows, each tile's groups in order. It
  // ends at the last group the projection reads of its last tile: a view of
  // leading inputs leaves the tile's later groups unread.
  const uint64_t groups = matrix.inputSize / kQuantGroup, rowGroups = p.planeInputSize() / kQuantGroup;
  const uint64_t parameters =
      uint64_t{matrix.outputSize} * rowGroups - SPLASH_AFFINE_TILE_ROWS * (rowGroups - groups);
  requireBytes(p.affine().weights, parameters * kQuantGroup / 2, "projection weight");
  requireBytes(p.affine().scales, parameters * 2, "projection scale");
  requireBytes(p.affine().biases, parameters * 2, "projection bias");
}

uint32_t LinearPlan::storageRows() const noexcept {
  if (workload_.phase != LinearPhase::Prefill) return workload_.rows;
  return ((workload_.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows) * kAffinePrefillTileRows;
}
uint32_t LinearPlan::tileColumns() const noexcept {
  switch (config_.tile) {
  case LinearTile::Q4Register: return workload_.epilogue == LinearEpilogue::GateUp ? 32 : 64;
  case LinearTile::N256:
  case LinearTile::Paired256: return 256;
  case LinearTile::N128:
  case LinearTile::Paired128:
  case LinearTile::Split128:
  case LinearTile::LaneSplit128: return 128;
  case LinearTile::Qmv: return 8 * config_.simdgroupOutputs;
  }
  return 0;
}
uint32_t LinearPlan::groups() const noexcept {
  return config_.groups ? config_.groups : workload_.matrix.outputSize / tileColumns();
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  switch (config_.tile) {
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::Split128:
  case LinearTile::LaneSplit128:
  case LinearTile::Qmv:
  case LinearTile::Paired256:
  case LinearTile::Q4Register: return static_cast<uint32_t>(config_.simdgroups) * 32;
  }
  return 0;
}
// The Qmv tile's threadgroups each reduce their one live row's squares
// before the projection (two shared rows take the norm's dispatch:
// apple10QmvConfig).
bool LinearPlan::foldsNorm(uint32_t liveRows) const noexcept { return canFoldNorm() && liveRows == 1; }
bool LinearPlan::usesQ4Register() const noexcept { return config_.tile == LinearTile::Q4Register; }
LinearInput LinearPlan::input() const noexcept {
  return usesQ4Register() ? LinearInput::Table64 : LinearInput::Plain;
}
LinearScratchSize LinearPlan::scratchSize() const noexcept {
  const auto [n, k] = workload_.matrix;
  // Split128: [split][row][column] fp32 partials over every row of the step
  // and one counter per column tile.
  if (config_.tile == LinearTile::Split128)
    return {0, 0, uint64_t{config_.splits} * workload_.rows * n * sizeof(float),
            uint64_t{n / tileColumns()} * sizeof(uint32_t)};
  // LaneSplit128: the same per split, for both projections of gate/up; one
  // partition reads neither.
  if (config_.tile == LinearTile::LaneSplit128) {
    if (config_.splits == 1) return {};
    const uint64_t streams = workload_.epilogue == LinearEpilogue::GateUp ? 2 : 1;
    return {0, 0, streams * config_.splits * workload_.rows * n * sizeof(float),
            uint64_t{n / tileColumns()} * sizeof(uint32_t)};
  }
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
  return workload_.phase == LinearPhase::Prefill
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  // Two-pass gate/up plans run a gate pass and an up-with-gate pass.
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp && !secondPipeline_.empty());
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config, FloatOutput destination)
    : workload_(w), config_(config), destination_(destination) {
  validate(w);
  if (destination == FloatOutput::Float32 &&
      (w.phase != LinearPhase::Decode || w.epilogue != LinearEpilogue::None))
    throw std::invalid_argument("an fp32 destination takes a plain decode projection");
  if (w.phase == LinearPhase::Decode && persistentTile(config.tile)
          ? !config.groups || config.groups > w.matrix.outputSize / tileColumns()
          : config.groups != 0)
    throw std::invalid_argument("a persistent decode tile takes 1 to its column tiles in groups, every other plan 0");
  const bool splitsK = config.tile == LinearTile::Q4Register || config.tile == LinearTile::Split128 ||
                       config.tile == LinearTile::LaneSplit128;
  if (!splitsK && config.splits != 1)
    throw std::invalid_argument("K splits require the register, Split128 or LaneSplit128 Q4 tile");
  if (config.simdgroups == LinearSimdgroups::Four && !supportsFourSimdgroups(w, config.tile))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (const auto fixed = fixedSimdgroups(config.tile); fixed && config.simdgroups != *fixed)
    throw std::invalid_argument("Q4 tile requires its kernel's simdgroup count");
  if ((config.tile == LinearTile::Qmv) !=
      (config.simdgroupOutputs == 2 || config.simdgroupOutputs == 4))
    throw std::invalid_argument("the Qmv tile takes 2 or 4 outputs per simdgroup, every other tile 0");
  if (config.sharedRows && config.tile != LinearTile::Qmv)
    throw std::invalid_argument("only the Qmv tile shares its threadgroups between rows");
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
  if (config.tile == LinearTile::LaneSplit128) {
    // Every partition holds at least one of the kernel's 256-input blocks.
    if (!config.validSplits() || w.matrix.inputSize / kInputSumBlock < config.splits)
      throw std::invalid_argument("LaneSplit128 requires 1, 2, 4 or 8 K partitions of 256-input blocks");
    pipeline_ = w.epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_n128_lane_split_gate_up"
        : residual ? "decode_linear_q4_n128_lane_split_residual" : "decode_linear_q4_n128_lane_split";
    return;
  }
  if (config.tile == LinearTile::Paired256) {
    pipeline_ = "decode_linear_q4_n256_paired_sg4";
    return;
  }
  if (config.tile == LinearTile::Qmv) {
    if (w.liveRows != (config.sharedRows ? 2u : 1u))
      throw std::invalid_argument("the Qmv tile projects one live row, or two sharing its threadgroups");
    const bool gateUp = w.epilogue == LinearEpilogue::GateUp;
    if (config.sharedRows) {
      // Two live rows in each threadgroup, after the norm's dispatch.
      if (config.simdgroupOutputs != (gateUp ? 2u : 4u))
        throw std::invalid_argument("shared Qmv rows take four outputs a simdgroup, two for gate/up");
      pipeline_ = gateUp ? "decode_linear_q4_qmv_gate_up_r2" : residual ? "decode_linear_q4_qmv_residual_r2"
                                                                       : "decode_linear_q4_qmv_r2";
      return;
    }
    const bool two = config.simdgroupOutputs == 2;
    pipeline_ = gateUp ? (two ? "decode_linear_q4_qmv_gate_up_o2" : "decode_linear_q4_qmv_gate_up")
        : residual     ? (two ? "decode_linear_q4_qmv_residual_o2" : "decode_linear_q4_qmv_residual")
                       : (two ? "decode_linear_q4_qmv_o2" : "decode_linear_q4_qmv");
    if (!residual && w.matrix.outputSize <= kQmvFoldedNormOutputs)
      normPipeline_ = gateUp ? (two ? "decode_linear_q4_qmv_gate_up_norm_o2" : "decode_linear_q4_qmv_gate_up_norm")
                             : (two ? "decode_linear_q4_qmv_norm_o2" : "decode_linear_q4_qmv_norm");
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

// Apple10 and later decode two to four lanes with K split across the
// threadgroups of the Split128 tile (256 threads) by one rule (one lane runs
// LaneSplit128, apple10LaneSplitConfig): the largest power of
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

// Apple10 runs one lane's projections of up to one resident wave of N128
// tiles (four threadgroups per core) on LaneSplit128, splitting K by
// apple10Splits' rule except that every partition keeps at least two
// 256-input blocks. On the Omni models' decode shapes (16-core M5 Pro,
// dev/benchmarks/omni_dq4_bench.mm --sweep, weights streamed from DRAM) this
// picks the measured best of 1 to 8 splits on every shape (or one within
// 0.2 us of it), and the paired tile is 4-15% faster than the unpaired
// Split128 tile; gate/up in one dispatch is 18-24% faster than the N256
// gate/up tile and the two-pass Split128 pair. Finer splits lose to the
// reduction once a partition holds a single block; N64 (four simdgroups),
// four-simdgroup N128 and N256 tiles never beat this one.
constexpr uint32_t kLaneSplitMinimumBlocks = 2;

// Apple10 projects a single live row with the Qmv tile: on every Omni
// decode shape (omni-dq4-bench --qmv, M5 Pro, weights from DRAM) it beats
// LaneSplit128 and the paired N256 tile by 4-45% (ASR layer projections
// 22.9/14.5/56.0/33.6 -> 20.4/10.2/53.6/26.5 us, its head 638 -> 586 us),
// and a whole ASR step takes 4.83 -> 4.13 ms. At two live rows it ties or
// loses on the norm projections and from four it loses 2-5x, its fp32 dot
// products growing with the rows while the tiles' matrix operations do
// not, so those keep the tiles. Four outputs per simdgroup, two for the
// gate/up pair, whose four-output simdgroup carries eight code streams: in
// whole ASR steps 4.10 against 4.13 ms (two for the plain and residual
// projections: 4.16 and 4.16).
constexpr uint32_t kQmvOutputs = 4, kQmvGateUpOutputs = 2;

// Two live rows share the Qmv tile's threadgroups (LinearConfig::
// sharedRows): each step's codes load once and are masked once for both
// rows, after the norm's dispatch. On every Omni decode shape (omni-dq4-bench
// --multirow, M5 Pro, weights from DRAM, norm included) it beats the
// LaneSplit128 plan the step ran: talker/ASR QKV 23.7 -> 21.5 us, O 14.4 ->
// 11.4, gate/up 56.7 -> 54.6, down 33.6 -> 28.2, the ASR head 637 -> 598, the
// predictor's (and the 0.6B draft's) QKV 14.4 -> 12.4, O 10.5 -> 6.5,
// gate/up 19.0 -> 16.8, down 13.7 -> 8.6, heads 10.5 -> 8.0. Folding the
// norm into it costs more than the dispatch (talker QKV 25.2, gate/up 64.7),
// and three or four rows cost about twice two (talker O 21.2 and 24.5 us
// against the tile's 14.4), so those keep the tiles.
constexpr uint32_t kQmvSharedRows = 2;

LinearConfig apple10QmvConfig(LinearWorkload w) noexcept {
  const uint32_t outputs = w.epilogue == LinearEpilogue::GateUp ? kQmvGateUpOutputs : kQmvOutputs;
  return LinearConfig{LinearTile::Qmv, 0, LinearSimdgroups::Eight, 1, outputs, w.liveRows == kQmvSharedRows};
}

std::optional<LinearConfig> apple10LaneSplitConfig(LinearMatrix matrix, uint32_t cores) noexcept {
  const uint64_t grid = matrix.outputSize / 128, wave = uint64_t{kSplitGroupsPerCore} * cores;
  if (grid > wave) return std::nullopt;
  uint32_t splits = 1;
  while (splits < LinearConfig::kMaximumSplits && grid * 2 * splits <= wave &&
         2 * splits * kLaneSplitMinimumBlocks <= matrix.inputSize / kInputSumBlock)
    splits *= 2;
  return LinearConfig{LinearTile::LaneSplit128, 0, LinearSimdgroups::Eight, splits};
}

// Apple10 one-lane plain projections reduce input re-reads with paired N256
// tiles at one resident wave: one tile per group while the grid fits the
// wave, several per group from kPaired256TilesPerCore. Apple9's register
// tile policy is independent.
// From eight tiles per core each group streams enough tiles (16/20-core GPUs).
constexpr uint32_t kPaired256TilesPerCore = 8;
// One resident wave; a grid past it needs a second wave, 11-23% slower.
constexpr uint32_t kPaired256WaveGroupsPerCore = 4;
// From two tiles per core one wave takes 0.78-1.04 times as long as the paired
// N128 tile on 12-, 20- and 40-core GPUs; below two it loses on 40 cores.
constexpr uint32_t kPaired256OneWaveTilesPerCore = 2;

std::optional<LinearConfig> apple10OneLaneConfig(LinearWorkload w, uint32_t cores) {
  // validate() requires outputSize % 256 == 0, so every tile width divides it.
  const uint32_t n = w.matrix.outputSize;
  const uint32_t tiles256 = n / 256;
  const bool oneWave = tiles256 >= kPaired256OneWaveTilesPerCore * cores &&
                       tiles256 <= kPaired256WaveGroupsPerCore * cores;
  if (w.epilogue == LinearEpilogue::None && (oneWave || tiles256 >= kPaired256TilesPerCore * cores))
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
LinearConfig Linear::baseline(LinearWorkload w) const {
  validate(w);
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
    if (lanes == 1 && (w.liveRows == 1 || w.liveRows == kQmvSharedRows)) return apple10QmvConfig(w);
    if (lanes == 1) {
      if (const auto config = apple10OneLaneConfig(w, gpuCores_)) return *config;
      if (const auto config = apple10LaneSplitConfig(w.matrix, gpuCores_)) return *config;
    }
    if (const uint32_t splits = apple10Splits(w.matrix, gpuCores_); splits > 1)
      return {LinearTile::Split128, 0, LinearSimdgroups::Eight, splits};
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
LinearPlan Linear::plan(LinearWorkload w, const Projection &p, const Projection *) const {
  w.weightLayout = p.layout();
  return LinearPlan(w, baseline(w), p.destination);
}

LinearPlan Linear::decodePlan(const Projection &p, uint32_t lanes, LinearEpilogue epilogue,
                              const Projection *gate, uint32_t liveRows) const {
  return plan(decode({p.outputSize, p.inputSize}, lanes, epilogue, liveRows), p, gate);
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
      bound.include(plan(w).scratchSize());
    }
  return bound;
}

LinearScratchSize Linear::prefillScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  // The prefill tiles take no split scratch; the bound still covers every
  // chunk of up to a decode batch and epilogue.
  for (uint32_t rows = 1; rows <= kMaximumDecodeTileRows; ++rows)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      bound.include(plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Prefill, epilogue, shape.layout})
                        .scratchSize());
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
  requireLeadingInputs(selected, p);
  if (gate) requireLeadingInputs(selected, *gate);
  const uint64_t rows = selected.storageRows();
  const bool norm = static_cast<bool>(b.normWeight);
  if (norm) {
    if (!selected.canFoldNorm()) throw std::invalid_argument("the plan does not fold a norm");
    if (!b.normRows || b.normRows > rows) throw std::invalid_argument("a folded norm takes 1 to the lane's rows");
    if (w.liveRows && b.normRows != w.liveRows)
      throw std::invalid_argument("a folded norm of a plan of live rows takes those rows");
    requireBytes(b.normWeight, uint64_t{k} * 2, "norm weight");
  }
  // The rows the plan reads and writes: a Qmv plan its live row, every other
  // the lane's rows.
  const uint64_t touched = selected.configuration().tile == LinearTile::Qmv ? w.liveRows : rows;
  requireBytes(b.input, (norm ? b.normRows : touched) * k * 2, "projection input");
  requireBytes(b.output, touched * n * elementBytes(selected.destination()), "projection output");
  if (w.epilogue == LinearEpilogue::Residual) requireBytes(b.residual, touched * n * 2, "projection residual");
  requireBytes(b.sums, selected.sumsBytes(), "projection sums");
  requireBytes(b.gateScratch, selected.gateScratchBytes(), "projection gate scratch");
  requireBytes(b.downSums, selected.downSumsBytes(), "projection down sums");
  const LinearScratchSize scratch = selected.scratchSize();
  requireBytes(b.scratch.input, scratch.input, "projection scratch table");
  requireBytes(b.scratch.sums, scratch.sums, "projection scratch sums");
  requireBytes(b.scratch.partials, scratch.partials, "projection partials");
  requireBytes(b.scratch.counters, scratch.counters, "projection counters");
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
    if (w.phase == LinearPhase::Prefill) {
      const metal::DispatchSize groups{selected.storageRows() / kAffinePrefillTileRows,
                                       n / selected.tileColumns(), 1};
      const metal::DispatchSize threads{selected.threadsPerThreadgroup(), 1, 1};
      if (p.planeInputs())
        graph.add(leadingInputsInstance(name), bindings, Q4PrefillLeadingParams{{n, k}, p.planeInputs()}, groups,
                  threads);
      else
        graph.add(std::string(name), bindings, Q4Params{n, k}, groups, threads);
    } else {
      // Split128 and LaneSplit128 bind their partials and counters after the
      // sequential kernel's buffers; every other decode tile runs one K split.
      const LinearConfig config = selected.configuration();
      std::vector<metal::MetalBuffer> buffers(bindings);
      if (config.tile == LinearTile::Split128 || config.tile == LinearTile::LaneSplit128) {
        const auto [partials, counters] = splitScratch(b, config.splits);
        buffers.insert(buffers.end(), {partials, counters});
      }
      const std::string kernel = kernelInstance(name, selected.destination());
      const metal::DispatchSize groups{selected.groups(), config.splits, 1};
      const metal::DispatchSize threads{selected.threadsPerThreadgroup(), 1, 1};
      if (config.tile == LinearTile::Qmv) {
        // The live rows in every threadgroup; a folded norm's weights
        // follow the buffers.
        if (norm) buffers.push_back(b.normWeight);
        graph.add(kernelInstance(norm ? selected.normPipeline_ : name, selected.destination()), std::move(buffers),
                  Q4Params{n, k}, {1, selected.groups(), 1}, threads);
        return;
      }
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
void Linear::addPrefillSwiGlu(metal::CommandGraph &graph, const SwiGluProjections &ffn,
                              const PrefillFfnBuffers &b, metal::MetalBuffer residual, metal::MetalBuffer output,
                              uint32_t rows) const {
  addPrefill(graph, b.normalized, *ffn.gate, b.gateScratch, b.sums, rows, b.scratch);
  addPrefillUpWithGate(graph, b.normalized, *ffn.up, b.gateScratch, b.intermediate, b.sums, b.downSums, rows,
                       b.scratch);
  addPrefillResidual(graph, b.intermediate, *ffn.down, residual, output, b.downSums, rows, b.scratch);
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
