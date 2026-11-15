#pragma once

#include "metal/abi/ExecutionGeometry.h"

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

} // namespace splash::ops
