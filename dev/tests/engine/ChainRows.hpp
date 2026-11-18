#pragma once

#include "ops/LaneRows.hpp"

#include <algorithm>
#include <cstdint>
#include <span>

namespace splash::test {

// Row `row` of a chain, which follows row - 1: the tree rows of a lane whose
// rows the tests and benchmarks lay out as a chain.
[[nodiscard]] constexpr ops::TreeRow chainRow(uint32_t row) noexcept {
  return {(2u << row) - 1u, static_cast<uint16_t>(row ? row - 1 : SPLASH_TREE_ROOT), static_cast<uint16_t>(row)};
}

// The path of the first `rows` rows of a chain, a block's rows a row mask
// names as a tree's accepted path does (metal/abi/VerifyTree.h).
[[nodiscard]] constexpr uint32_t chainPath(uint32_t rows) noexcept {
  return rows >= 32 ? ~uint32_t{0} : (uint32_t{1} << rows) - 1;
}

// One past the last row any of `lanes` holds.
[[nodiscard]] inline uint64_t rowsEnd(std::span<const ops::LaneRows> lanes) noexcept {
  uint64_t end = 0;
  for (const ops::LaneRows &lane : lanes) end = std::max(end, uint64_t{lane.first} + lane.rows);
  return end;
}

// The most rows of any of `lanes`, which a grid of a lane's rows by lanes
// spans.
[[nodiscard]] inline uint32_t mostRows(std::span<const ops::LaneRows> lanes) noexcept {
  uint32_t most = 0;
  for (const ops::LaneRows &lane : lanes) most = std::max(most, lane.rows);
  return most;
}

} // namespace splash::test
