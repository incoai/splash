#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/VerifyTree.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace splash::ops {

// A lane's rows among a command's packed rows: `rows` rows from row `first`.
// A decode lane's are its verify block, the anchor's row and its proposals';
// a lane selecting the first token after its prompt has the one row it
// selects from. The batched kernels address a lane's rows by these (their
// parameters' per-lane first_row and rows), so lanes of different row counts
// share a dispatch.
struct LaneRows {
  uint32_t first = 0;
  uint32_t rows = 0;
};

// A row of a decode lane's verify block as a tree (metal/abi/VerifyTree.h),
// one per row of a command's packed rows.
using TreeRow = SplashTreeRow;

// Row `row` of a chain, which follows row - 1.
[[nodiscard]] constexpr TreeRow chainRow(uint32_t row) noexcept {
  return {(2u << row) - 1u, static_cast<uint16_t>(row ? row - 1 : SPLASH_TREE_ROOT), static_cast<uint16_t>(row)};
}

// One past the last row any of `lanes` holds.
[[nodiscard]] inline uint64_t rowsEnd(std::span<const LaneRows> lanes) noexcept {
  uint64_t end = 0;
  for (const LaneRows &lane : lanes) end = std::max(end, uint64_t{lane.first} + lane.rows);
  return end;
}

// The most rows of any of `lanes`, which a grid of a lane's rows by lanes
// spans.
[[nodiscard]] inline uint32_t mostRows(std::span<const LaneRows> lanes) noexcept {
  uint32_t most = 0;
  for (const LaneRows &lane : lanes) most = std::max(most, lane.rows);
  return most;
}

// Throws unless `lanes` are one to a batch of lanes, each of at least one row
// and at most `limit`, where a kernel holds that many a lane.
inline void requireLaneRows(std::span<const LaneRows> lanes,
                            uint32_t limit = std::numeric_limits<uint32_t>::max()) {
  if (lanes.empty() || lanes.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("lane rows name 1 to a batch of lanes");
  for (const LaneRows &lane : lanes)
    if (!lane.rows || lane.rows > limit) throw std::invalid_argument("a lane's rows are out of range");
}

// Throws unless `lanes`' rows follow each other in row order without
// overlapping, as a kernel that takes a row's lane to be the last whose first
// row is at most it reads them.
inline void requireRowOrder(std::span<const LaneRows> lanes) {
  for (size_t lane = 1; lane < lanes.size(); ++lane)
    if (lanes[lane].first < uint64_t{lanes[lane - 1].first} + lanes[lane - 1].rows)
      throw std::invalid_argument("lane rows overlap or are out of row order");
}

// The per-lane first_row and rows arrays of a kernel's parameters, zero past
// `lanes`.
inline void copyLaneRows(std::span<const LaneRows> lanes, uint32_t (&first)[SPLASH_MAXIMUM_BATCH_WIDTH],
                         uint32_t (&rows)[SPLASH_MAXIMUM_BATCH_WIDTH]) {
  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    first[lane] = lanes[lane].first;
    rows[lane] = lanes[lane].rows;
  }
}

// The rows a decode cycle verifies of each lane (SPLASH_DECODE_LANE_ROWS).
inline constexpr uint32_t kDecodeLaneRows = SPLASH_DECODE_LANE_ROWS;
static_assert(kDecodeLaneRows <= SPLASH_TARGET_VERIFY_ROWS);

// The most rows of a lane's verify block in a decode step of `rows` rows over
// `lanes` lanes, which the step's proposer chooses on the device within it:
// eight while the step's rows give each lane no more, the rows the GDN
// decode kernels of eight rows hold, which took 13-19% less time than those
// of sixteen over a layer's 8-row chains (DRAM-cold, an M6); else a verify
// block's most.
[[nodiscard]] constexpr uint32_t verifyLaneCap(uint32_t rows, uint32_t lanes) noexcept {
  return rows <= uint64_t{8} * lanes ? 8 : SPLASH_TARGET_VERIFY_ROWS;
}

// The path of the first `rows` rows of a chain, a block's rows a row mask
// names as a tree's accepted path does (metal/abi/VerifyTree.h).
[[nodiscard]] constexpr uint32_t chainPath(uint32_t rows) noexcept {
  return rows >= 32 ? ~uint32_t{0} : (uint32_t{1} << rows) - 1;
}

// Lane rows as a proposer writes them on the device for the kernels after it
// (SplashLaneRows).
static_assert(sizeof(LaneRows) == sizeof(SplashLaneRows));

// A decode step's rows, which its proposer lays out on the device for the
// kernels after it (SplashLaneRows, ops::DraftSelector::addVerifyRows):
// `rows` rows over `lanes` lanes, packed in lane order, each lane's 1 to
// `laneCap` of them (verifyLaneCap). A kernel's grid of a lane's rows spans
// the cap.
struct StepRows final {
  uint32_t lanes = 0;
  uint32_t rows = 0;
  uint32_t laneCap = 0;
};

// Throws unless `step` holds 1 to a batch of lanes of 1 to a verify block's
// rows each, its rows as many as its lanes may hold.
inline void requireStepRows(const StepRows &step) {
  if (!step.lanes || step.lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !step.laneCap ||
      step.laneCap > SPLASH_TARGET_VERIFY_ROWS || step.rows < step.lanes ||
      step.rows > uint64_t{step.lanes} * step.laneCap)
    throw std::invalid_argument("invalid decode step rows");
}

} // namespace splash::ops
