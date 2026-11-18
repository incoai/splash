#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/target_rope.h"

// Target rows carry (t, h, w) positions; text rows repeat one position.
// YaRN scales the rotated dimensions of the target's queries and keys by its
// attention factor, so their cosines and sines (target_scale). A target row
// holds TargetPairs pairs, those its attention rotates: Qwen3.5's
// SPLASH_TARGET_ROPE_PAIRS, or another geometry's (ops::RoPE::addTables).
#define ROPE_BUILD_TABLES(Name, TargetPairs)                                   \
  kernel void Name(                                                            \
      device const uint *target_positions [[buffer(0)]],                       \
      device const uint *draft_positions [[buffer(1)]],                        \
      device const float *target_inverse_frequencies [[buffer(2)]],            \
      device const float *draft_inverse_frequencies [[buffer(3)]],             \
      device float *target_cosine [[buffer(4)]],                               \
      device float *target_sine [[buffer(5)]],                                 \
      device float *draft_cosine [[buffer(6)]],                                \
      device float *draft_sine [[buffer(7)]],                                  \
      constant RopeTableParams &params [[buffer(8)]],                          \
      uint index [[thread_position_in_grid]],                                  \
      uint grid_size [[threads_per_grid]]) {                                   \
    const uint target_elements = params.target_rows * TargetPairs;             \
    for (uint element = index; element < target_elements;                     \
         element += grid_size) {                                               \
      const uint row = element / TargetPairs;                                  \
      device const uint *position = target_positions + row * 3;                \
      const float2 rope = target_rope_pair(                                    \
          uint3(position[0], position[1], position[2]),                        \
          target_inverse_frequencies, element % TargetPairs,                   \
          params.target_scale);                                                \
      target_cosine[element] = rope.x;                                         \
      target_sine[element] = rope.y;                                           \
    }                                                                          \
    const uint draft_elements = params.draft_rows * SPLASH_DRAFT_ROPE_PAIRS;   \
    for (uint element = index; element < draft_elements;                       \
         element += grid_size) {                                               \
      const uint row = element / SPLASH_DRAFT_ROPE_PAIRS;                      \
      const uint dim = element % SPLASH_DRAFT_ROPE_PAIRS;                      \
      const float angle =                                                      \
          float(draft_positions[row]) * draft_inverse_frequencies[dim];        \
      draft_cosine[element] = cos(angle);                                      \
      draft_sine[element] = sin(angle);                                        \
    }                                                                          \
  }
ROPE_BUILD_TABLES(rope_build_tables, SPLASH_TARGET_ROPE_PAIRS)
// Rows of 64 pairs: whole heads of 128 dimensions.
ROPE_BUILD_TABLES(rope_build_tables_p64, 64u)
#undef ROPE_BUILD_TABLES

// The target rows of a decode step's packed rows (RopeTreeParams), as
// rope_build_tables makes a row at the same position: group (row, lane) of
// TargetPairs threads, a pair each, computes the lane's row at its anchor's
// position plus the row's depth, on every axis alike (a decode row is text),
// into the row's place among the packed rows. A row past its lane's rows, or
// a lane whose rows run past the step's, writes nothing.
#define ROPE_BUILD_TREE_TABLES(Name, TargetPairs)                              \
  kernel void Name(                                                            \
      device const SplashLaneRows *layout [[buffer(0)]],                       \
      device const SplashTreeRow *tree [[buffer(1)]],                          \
      device const float *target_inverse_frequencies [[buffer(2)]],            \
      device float *target_cosine [[buffer(3)]],                               \
      device float *target_sine [[buffer(4)]],                                 \
      constant RopeTreeParams &params [[buffer(5)]],                           \
      uint2 group [[threadgroup_position_in_grid]],                            \
      uint pair [[thread_index_in_threadgroup]]) {                             \
    const uint row = group.x, lane = group.y;                                  \
    if (lane >= params.lanes || pair >= TargetPairs)                           \
      return;                                                                  \
    const SplashLaneRows rows = layout[lane];                                  \
    if (row >= rows.rows || ulong(rows.first) + rows.rows > params.rows)       \
      return;                                                                  \
    const uint packed = rows.first + row;                                      \
    const float2 rope = target_rope_pair(                                      \
        uint3(params.anchors[lane] + tree[packed].depth),                      \
        target_inverse_frequencies, pair, params.target_scale);                \
    target_cosine[ulong(packed) * TargetPairs + pair] = rope.x;                \
    target_sine[ulong(packed) * TargetPairs + pair] = rope.y;                  \
  }
ROPE_BUILD_TREE_TABLES(rope_build_tree_tables, SPLASH_TARGET_ROPE_PAIRS)
ROPE_BUILD_TREE_TABLES(rope_build_tree_tables_p64, 64u)
#undef ROPE_BUILD_TREE_TABLES

// The draft table rows of rope_build_tables for consecutive positions.
kernel void rope_build_draft_range_tables(
    device const float *draft_inverse_frequencies [[buffer(0)]],
    device float *draft_cosine [[buffer(1)]],
    device float *draft_sine [[buffer(2)]],
    constant RopeRangeParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  const uint draft_elements = params.rows * SPLASH_DRAFT_ROPE_PAIRS;
  for (uint element = index; element < draft_elements; element += grid_size) {
    const uint row = element / SPLASH_DRAFT_ROPE_PAIRS;
    const uint dim = element % SPLASH_DRAFT_ROPE_PAIRS;
    const float angle = float(params.start_position + row) *
                        draft_inverse_frequencies[dim];
    draft_cosine[element] = cos(angle);
    draft_sine[element] = sin(angle);
  }
}
