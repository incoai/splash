#pragma once

// The traversal kernels use FlashDenseCacheParams.reserved as a0..4 mode:
// column-fast, row-fast, or rows blocked by2/4/8. Original kernels require0.
// This mapping controls logical work order without exposing die affinity.
inline bool flash_dense_traversal_group_valid(metal::uint2 group,
    uint rowTiles, uint columnTiles, uint mode) {
  if (mode == 0) return group.x < columnTiles && group.y < rowTiles;
  if (mode == 1) return group.x < rowTiles && group.y < columnTiles;
  if (mode > 4) return false;
  const uint width = 1u << (mode - 1);
  return ulong(group.x) < ulong(columnTiles) * width &&
      group.y < (rowTiles + width - 1) / width;
}

inline metal::uint2 flash_dense_traversal_tile(metal::uint2 group, uint mode) {
  if (mode == 0) return group;
  if (mode == 1) return metal::uint2(group.y, group.x);
  const uint log = mode - 1;
  return metal::uint2(group.x >> log,
      (group.y << log) + (group.x & ((1u << log) - 1)));
}
