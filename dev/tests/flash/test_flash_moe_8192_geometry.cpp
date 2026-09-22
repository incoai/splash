#include "flash/FlashMoEBuckets.hpp"
#include "metal/abi/FlashMoEBuckets.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using splash::flash::moEBucketJobCapacity;
constexpr uint32_t kExperts = 512;
constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try { function(); }
  catch (const std::invalid_argument &) { return; }
  throw std::runtime_error("invalid production MoE geometry was accepted");
}

void bounds() {
  static_assert(kFlashMoEBucketMaximumRows == 8192);
  static_assert(kFlashMoEBucketMaximumSelections == 10);
  for (uint32_t rows = 1; rows <= 8192; ++rows) {
    for (uint32_t selections = 1; selections <= 10; ++selections) {
      const uint64_t routes = uint64_t{rows} * selections;
      require(routes <= 81920, "route product exceeds its maximum");
      for (uint32_t tile : {8U, 16U, 32U, 64U}) {
        const uint64_t expected = routes / tile + (routes % tile != 0) + 511;
        require(moEBucketJobCapacity(rows, selections, tile) == expected,
                "production GPU job capacity differs from U64 bound");
      }
    }
  }
  require(moEBucketJobCapacity(8192, 10, 8) == 10751 &&
              moEBucketJobCapacity(8192, 10, 16) == 5631 &&
              moEBucketJobCapacity(8192, 10, 32) == 3071 &&
              moEBucketJobCapacity(8192, 10, 64) == 1791,
          "maximum job capacity golden differs");
  require(moEBucketJobCapacity(2048, 10, 8) == 3071 &&
              moEBucketJobCapacity(2048, 10, 16) == 1791 &&
              moEBucketJobCapacity(2048, 10, 32) == 1151 &&
              moEBucketJobCapacity(2048, 10, 64) == 831,
          "existing 2048-row capacity changed");
  for (uint32_t rows : {0U, 8193U, kInvalid})
    rejects([&] { (void)moEBucketJobCapacity(rows, 10, 8); });
  for (uint32_t selections : {0U, 11U, kInvalid})
    rejects([&] { (void)moEBucketJobCapacity(8192, selections, 8); });
  for (uint32_t tile : {0U, 1U, 4U, 7U, 128U, kInvalid})
    rejects([&] { (void)moEBucketJobCapacity(8192, 10, tile); });
}

// CPU extent and job-emission model: no Metal backend is constructed and no
// large activation plane is allocated. Integer maps stay independent of the
// shader's SIMD scans; packed/scattered BF16 accesses are checked as offsets.
void checkRoutes(uint32_t rows, uint32_t selections,
                 const std::vector<int64_t> &ids) {
  const uint32_t routes = rows * selections;
  require(ids.size() == routes, "ID extent is incorrect");
  std::array<uint32_t, kExperts> counts{};
  std::array<uint32_t, kExperts + 1> offsets{};
  std::vector<uint32_t> map, inverse(routes, kInvalid);
  for (uint32_t route = 0; route < routes; ++route) {
    if (ids[route] < 0 || ids[route] >= kExperts) continue;
    ++counts[ids[route]];
    map.push_back(route);
  }
  std::stable_sort(map.begin(), map.end(),
                   [&](uint32_t a, uint32_t b) { return ids[a] < ids[b]; });
  std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
  require(offsets.back() == map.size(), "valid route prefix differs");
  const uint64_t inputBytes = uint64_t{rows} * 2560 * 2;
  const uint64_t packedBytes = uint64_t{routes} * 2560 * 2;
  const uint64_t activationBytes = uint64_t{routes} * 640 * 2;
  for (uint32_t packed = 0; packed < map.size(); ++packed) {
    const uint32_t canonical = map[packed];
    inverse[canonical] = packed;
    require(canonical < routes && canonical / selections < rows,
            "stable route source row exceeds input geometry");
    require((uint64_t{canonical / selections} * 2560 + 2559) * 2 + 2 <= inputBytes &&
                (uint64_t{packed} * 2560 + 2559) * 2 + 2 <= packedBytes &&
                (uint64_t{packed} * 640 + 639) * 2 + 2 <= activationBytes &&
                (uint64_t{canonical} * 2560 + 2559) * 2 + 2 <= packedBytes,
            "input, packed activation, or canonical scatter extent exceeds allocation");
  }
  for (uint32_t route = 0; route < routes; ++route) {
    if (ids[route] < 0 || ids[route] >= kExperts)
      require(inverse[route] == kInvalid, "excluded route has a live inverse");
    else require(map[inverse[route]] == route, "stable map inverse differs");
  }
  for (uint32_t tile : {8U, 16U, 32U, 64U}) {
    const uint32_t capacity = moEBucketJobCapacity(rows, selections, tile);
    std::array<uint32_t, kExperts + 1> jobOffsets{};
    for (uint32_t expert = 0; expert < kExperts; ++expert)
      jobOffsets[expert + 1] = jobOffsets[expert] + counts[expert] / tile +
          (counts[expert] % tile != 0);
    require(jobOffsets.back() <= capacity, "expert fragmentation exceeds GPU launch capacity");
    std::vector<uint32_t> coverage(map.size(), 0);
    for (uint32_t index = 0; index < jobOffsets.back(); ++index) {
      // Match the production job shader's upper-bound search, including the
      // repeated job offsets of empty experts and the concentrated expert 511.
      const auto upper = std::upper_bound(jobOffsets.begin(), jobOffsets.end(), index);
      const uint32_t expert = static_cast<uint32_t>(upper - jobOffsets.begin() - 1);
      require(expert < kExperts, "job emission selects an invalid expert");
      const uint32_t begin = offsets[expert] + (index - jobOffsets[expert]) * tile;
      const uint32_t end = offsets[expert + 1];
      require(begin >= offsets[expert] && begin < end && end <= routes,
              "job emission crosses expert bucket or route bound");
      for (uint32_t row = begin; row < std::min(begin + tile, end); ++row) {
        require(ids[map[row]] == expert, "matrix job crosses expert boundary");
        ++coverage[row];
      }
    }
    require(std::all_of(coverage.begin(), coverage.end(),
                        [](uint32_t hits) { return hits == 1; }),
            "matrix jobs miss or repeat a valid packed row");
  }
}

void distributions() {
  for (uint32_t rows : {1U, 32U, 2048U, 2049U, 8191U, 8192U}) {
    for (uint32_t selections : {1U, 10U}) {
      const uint32_t routes = rows * selections;
      std::vector<int64_t> ids(routes, 511);
      checkRoutes(rows, selections, ids); // Every legal route in one expert.
      for (uint32_t route = 0; route < routes; ++route)
        ids[route] = (route / selections * 73 + route % selections * 53) % 512;
      checkRoutes(rows, selections, ids);
      std::fill(ids.begin(), ids.end(), 511);
      for (uint32_t expert = 0; expert < std::min(511U, routes); ++expert)
        ids[expert] = expert;
      checkRoutes(rows, selections, ids); // 511 singletons plus heavy tail.
      for (uint32_t route = 0; route < routes; ++route)
        ids[route] = route % 3 == 0 ? -1 : route % 3 == 1 ? 512 : 511;
      checkRoutes(rows, selections, ids);
      std::fill(ids.begin(), ids.end(), std::numeric_limits<int64_t>::min());
      checkRoutes(rows, selections, ids);
    }
  }
  require(uint64_t{81920} * 2560 * 2 == 419430400 &&
              uint64_t{81920} * 640 * 2 == 104857600,
          "maximum BF16 allocation extent golden differs");
}

} // namespace

int main() {
  try {
    bounds();
    distributions();
    std::cout << "PASS production MoE geometry through 8192 physical rows; "
                 "81920 routes, concentrated/fragmented M8/16/32/64 jobs, "
                 "stable-map and canonical-scatter extents\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
