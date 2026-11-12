#pragma once

// FastVideo's VSA-H3 tiles of a packed layout, recomputed for the CPU tests (_h3_segment_tile_geometry of the segments
// (L, C, Na, video)): each dense segment in 64-row chunks, then every video row sorted by (cube, position inside its
// cube) as tile_partition_indices sorts them, cubes of 4 x 4 x 4 latent tokens, partial at the grid's edges.
#include "video/ops/Geometry.hpp"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

namespace splash::video::test {

// The rows of each tile, in FastVideo's order (dense chunks, then the video's cubes).
inline std::vector<std::vector<int32_t>> fastVideoTiles(const ops::SequenceLayout &lay) {
  std::vector<std::vector<int32_t>> tiles;
  int32_t cursor = 0;
  for (int32_t segment : {lay.text, lay.cond, lay.audio}) {
    for (int32_t r = 0; r < segment; r += 64) {
      tiles.emplace_back();
      for (int32_t i = r; i < std::min(segment, r + 64); ++i) tiles.back().push_back(cursor + i);
    }
    cursor += segment;
  }
  const int32_t ch = (lay.gridH + 3) / 4, cw = (lay.gridW + 3) / 4;
  std::vector<std::tuple<int32_t, int32_t, int32_t>> rows;
  for (int32_t f = 0; f < lay.frames; ++f)
    for (int32_t y = 0; y < lay.gridH; ++y)
      for (int32_t x = 0; x < lay.gridW; ++x)
        rows.emplace_back(((f / 4) * ch + y / 4) * cw + x / 4, ((f % 4) * 4 + y % 4) * 4 + x % 4,
                          cursor + (f * lay.gridH + y) * lay.gridW + x);
  std::sort(rows.begin(), rows.end());
  for (size_t i = 0; i < rows.size(); ++i) {
    if (i == 0 || std::get<0>(rows[i]) != std::get<0>(rows[i - 1])) tiles.emplace_back();
    tiles.back().push_back(std::get<2>(rows[i]));
  }
  return tiles;
}

} // namespace splash::video::test
