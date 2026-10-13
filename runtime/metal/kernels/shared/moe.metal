#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/moe_expert_slab.h"

// Routes group rows by expert so each tile streams its weights once. Each row
// carries top_k routed experts plus the shared expert (id `experts`), weighted
// by the sigmoid of its scalar gate.
//
// One row per threadgroup: thread e holds expert e's score, the threadgroup
// reduces the shared expert's F32 scalar gate with a fixed partial-sum order,
// and simdgroup 0 orders the experts, descending score then ascending id,
// which is the order the shape's routing and tie-break contract requires.
// Scores and routing weights are fp32. On 40,960 router rows each of real 35B
// prose, code and chat, top-8 sets chosen from bf16-rounded scores differ from
// the fp64 choice on 12-15% of rows (fp32: at most 0.0024%), and bf16 weights
// round 19-20% of the combine's outputs away from bf16 of its exact sum
// (fp32: 0.008%).
inline void moe_route_select(device const float *scores, device bfloat *input,
                             device const float *shared_gate, device uint *selected,
                             device float *routing_weights,
                             constant MoeRouteParams &params, uint group,
                             uint thread_index, uint simd_lane,
                             uint simd_group, threadgroup float *row_scores,
                             threadgroup float *ordered,
                             threadgroup float *scalar_partials) {
  constexpr uint Slots = SPLASH_MOE_EXPERT_SLOTS;
  constexpr uint Simdgroups = Slots / 32;
  constexpr uint ExpertsPerLane = Slots / 32;
  const uint row = group;
  if (row >= params.rows)
    return;
  row_scores[thread_index] =
      thread_index < params.experts
          ? scores[ulong(row) * Slots + thread_index]
          : -numeric_limits<float>::infinity();

  device bfloat *row_input = input + ulong(row) * params.input_size;
  float scalar = 0.0f;
  for (uint dimension = thread_index; dimension < params.input_size;
       dimension += Slots) {
    scalar += float(row_input[dimension]) *
              shared_gate[dimension];
  }
  scalar = simd_sum(scalar);
  if (simd_lane == 0)
    scalar_partials[simd_group] = scalar;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong row_routes = ulong(row) * (params.top_k + 1);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint partial = 0; partial < Simdgroups; ++partial)
      total += scalar_partials[partial];
    selected[row_routes + params.top_k] = params.experts;
    routing_weights[row_routes + params.top_k] = splash_sigmoid(total);
  }
  if (simd_group != 0)
    return;
  // Lane l holds experts l, l + 32, ... Each rank takes the highest remaining
  // score and the lowest expert id among ties.
  float lane_scores[ExpertsPerLane];
  for (uint slot = 0; slot < ExpertsPerLane; ++slot)
    lane_scores[slot] = row_scores[simd_lane + 32 * slot];
  for (uint rank = 0; rank < params.top_k; ++rank) {
    float best = -numeric_limits<float>::infinity();
    uint best_slot = 0;
    for (uint slot = 0; slot < ExpertsPerLane; ++slot) {
      if (lane_scores[slot] > best) {
        best = lane_scores[slot];
        best_slot = slot;
      }
    }
    const float row_best = simd_max(best);
    const uint candidate =
        best == row_best ? simd_lane + 32 * best_slot : 0xFFFFFFFFu;
    const uint winner = simd_min(candidate);
    if (candidate == winner)
      lane_scores[best_slot] = -numeric_limits<float>::infinity();
    if (simd_lane == 0) {
      ordered[rank] = row_best;
      selected[row_routes + rank] = winner;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint rank = simd_lane; rank < params.top_k; rank += 32) {
    float denominator = 0.0f;
    for (uint other = 0; other < params.top_k; ++other) {
      denominator +=
          fast::exp2((ordered[other] - ordered[0]) * 1.44269504089f);
    }
    routing_weights[row_routes + rank] =
        fast::exp2((ordered[rank] - ordered[0]) * 1.44269504089f) /
        denominator;
  }
}

// The GGUF router: fp32 scores (kernels/shared/gguf_float.metal) and the F32
// shared-expert scalar gate.
kernel void moe_route_select_f32(
    device const float *scores [[buffer(0)]],
    device bfloat *input [[buffer(1)]],
    device const float *shared_gate [[buffer(2)]],
    device uint *selected [[buffer(3)]],
    device float *routing_weights [[buffer(4)]],
    constant MoeRouteParams &params [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float row_scores[SPLASH_MOE_EXPERT_SLOTS];
  threadgroup float ordered[SPLASH_MOE_EXPERT_SLOTS];
  threadgroup float scalar_partials[SPLASH_MOE_EXPERT_SLOTS / 32];
  moe_route_select(scores, input, shared_gate, selected,
                   routing_weights, params, group, thread_index, simd_lane,
                   simd_group, row_scores, ordered, scalar_partials);
}

// Sorts one command's routes by expert. Tile t covers grouped rows
// [t * tile_rows, (t + 1) * tile_rows) of a single expert; the rows past
// that expert's last route that its last tile's matmul reads
// (moe_matmul_rows) carry the route ~0u. One threadgroup covers all routes
// and thread e owns expert e's count, offsets and tile descriptors. The
// shared expert's tiles follow the routed tiles and hold every row in order.
// It writes the tile count with the expert passes' grids (MoeTileCount).
// In place (a plan whose rows fill one tile) each expert gets one tile of the
// plan's rows: a route's grouped row is its expert tile's row of the route's
// own row, and the routed tiles' grouped routes go unwritten, as no gather
// reads them.
kernel void moe_group_routes(
    device const uint *selected [[buffer(0)]],
    device MoeTileDescriptor *tiles [[buffer(1)]],
    device MoeTileCount *tile_count [[buffer(2)]],
    device uint *grouped_routes [[buffer(3)]],
    device uint *route_rows [[buffer(4)]],
    constant MoeGroupParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Slots = SPLASH_MOE_EXPERT_SLOTS;
  const uint routes_per_row = params.top_k + 1;
  const uint routes = params.rows * routes_per_row;
  threadgroup atomic_uint counts[Slots];
  threadgroup atomic_uint cursors[Slots];
  threadgroup uint tile_offsets[Slots];
  threadgroup uint simd_totals[Slots / 32];
  threadgroup uint routed_tiles;
  atomic_store_explicit(&counts[thread_index], 0u, memory_order_relaxed);
  atomic_store_explicit(&cursors[thread_index], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint route = thread_index; route < routes; route += Slots) {
    if (route % routes_per_row != params.top_k) {
      atomic_fetch_add_explicit(&counts[selected[route]], 1u,
                                memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint count = atomic_load_explicit(&counts[thread_index], memory_order_relaxed);
  uint expert_tiles = (count + params.tile_rows - 1) / params.tile_rows;
  uint prefix = simd_prefix_exclusive_sum(expert_tiles);
  if (simd_lane == 31)
    simd_totals[simd_group] = prefix + expert_tiles;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint tile_offset = prefix;
  for (uint group = 0; group < simd_group; ++group)
    tile_offset += simd_totals[group];
  tile_offsets[thread_index] = tile_offset;
  if (thread_index == Slots - 1)
    routed_tiles = tile_offset + expert_tiles;
  for (uint tile = 0; tile < expert_tiles; ++tile) {
    tiles[tile_offset + tile] = MoeTileDescriptor{
        thread_index, params.in_place ? params.rows : min(params.tile_rows, count - tile * params.tile_rows)};
  }
  if (expert_tiles && !params.in_place) {
    const uint last = tile_offset + expert_tiles - 1;
    const uint live = count - (expert_tiles - 1) * params.tile_rows;
    for (uint row = tile_offset * params.tile_rows + count;
         row < last * params.tile_rows + moe_matmul_rows(live, params.tile_rows);
         ++row)
      grouped_routes[row] = ~0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint route = thread_index; route < routes; route += Slots) {
    if (route % routes_per_row == params.top_k)
      continue;
    uint expert = selected[route];
    if (params.in_place) {
      route_rows[route] = tile_offsets[expert] * params.tile_rows + route / routes_per_row;
      continue;
    }
    uint slot =
        atomic_fetch_add_explicit(&cursors[expert], 1u, memory_order_relaxed);
    uint row = tile_offsets[expert] * params.tile_rows + slot;
    grouped_routes[row] = route;
    route_rows[route] = row;
  }

  const uint shared_tiles =
      (params.rows + params.tile_rows - 1) / params.tile_rows;
  const uint shared_base = routed_tiles * params.tile_rows;
  const uint shared_rows =
      (shared_tiles - 1) * params.tile_rows +
      moe_matmul_rows(params.rows - (shared_tiles - 1) * params.tile_rows,
                      params.tile_rows);
  for (uint tile = thread_index; tile < shared_tiles; tile += Slots) {
    tiles[routed_tiles + tile] = MoeTileDescriptor{
        params.experts,
        min(params.tile_rows, params.rows - tile * params.tile_rows)};
  }
  for (uint row = thread_index; row < shared_rows; row += Slots) {
    if (row < params.rows) {
      uint route = row * routes_per_row + params.top_k;
      grouped_routes[shared_base + row] = route;
      route_rows[route] = shared_base + row;
    } else {
      grouped_routes[shared_base + row] = ~0u;
    }
  }
  if (thread_index == 0) {
    const uint live = routed_tiles + shared_tiles;
    *tile_count = MoeTileCount{live, {params.gate_up_columns, live, 1},
                               {params.down_columns, live, 1}};
  }
}

// Copies each grouped row's input so every expert tile is a dense matrix,
// over the rows its matmul reads (moe_matmul_rows): padding rows among them
// are zero and their outputs are never read.
kernel void moe_gather_rows(device const bfloat *input [[buffer(0)]],
                            device const uint *grouped_routes [[buffer(1)]],
                            device const MoeTileDescriptor *tiles [[buffer(2)]],
                            device const uint *tile_count [[buffer(3)]],
                            device bfloat *grouped_input [[buffer(4)]],
                            constant MoeGatherParams &params [[buffer(5)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]]) {
  if (group.x >= *tile_count)
    return;
  uint column = group.y * 256 + thread_index;
  const uint rows = moe_matmul_rows(tiles[group.x].rows, params.tile_rows);
  for (uint local = 0; local < rows; ++local) {
    uint row = group.x * params.tile_rows + local;
    uint route = grouped_routes[row];
    grouped_input[ulong(row) * params.input_size + column] =
        route == ~0u
            ? bfloat(0.0f)
            : input[ulong(route / params.routes_per_row) * params.input_size +
                    column];
  }
}

// The grouped rows as Table16 tiles (kernels/common/gguf_sgmatrix.h), the
// input of the Apple9 GGUF register expert tile. Threadgroup (tile, part)
// writes the four 64-input spans [4 part, 4 part + 4) of an 8-row tile, one
// simdgroup per row: on the 40-core M3 Max, 8 us per 35B decode layer at one
// lane and 14 us at four, under two thirds of the time of a threadgroup per
// span pair or per tile. The gather reads each grouped row's route like
// moe_gather_rows (padding rows are zero); the prepare reads grouped rows (the
// down pass's intermediate). Grid (tiles, width / 256), 256 threads.
template <class Source>
inline void moe_table16_tile(Source source, device bfloat *table,
                             device float *sums, uint width, uint2 group,
                             uint simd_lane, uint simd_group) {
  for (uint span = group.y * 4; span < group.y * 4 + 4; ++span) {
    const bfloat2 values = source(simd_group, span * 64 + 2 * simd_lane);
    gguf_sg::write_input(table + ulong(group.x) * width * 8,
                       sums + ulong(group.x) * table16_sums_per_tile(width),
                       width, span, simd_group, simd_lane, values.x, values.y);
  }
}

kernel void moe_gather_table16(device const bfloat *input [[buffer(0)]],
                               device const uint *grouped_routes [[buffer(1)]],
                               device const uint *tile_count [[buffer(2)]],
                               device bfloat *table [[buffer(3)]],
                               device float *sums [[buffer(4)]],
                               constant MoeGatherParams &params [[buffer(5)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (group.x >= *tile_count)
    return;
  moe_table16_tile(
      [&](uint row, uint k) {
        const uint route = grouped_routes[group.x * 8 + row];
        return route == ~0u
                   ? bfloat2(bfloat(0.0f))
                   : *reinterpret_cast<device const bfloat2 *>(
                         input + ulong(route / params.routes_per_row) *
                                     params.input_size + k);
      },
      table, sums, params.input_size, group, simd_lane, simd_group);
}

kernel void moe_prepare_table16(device const bfloat *rows [[buffer(0)]],
                                device const uint *tile_count [[buffer(1)]],
                                device bfloat *table [[buffer(2)]],
                                device float *sums [[buffer(3)]],
                                constant uint &width [[buffer(4)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint simd_lane [[thread_index_in_simdgroup]],
                                uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (group.x >= *tile_count)
    return;
  moe_table16_tile(
      [&](uint row, uint k) {
        return *reinterpret_cast<device const bfloat2 *>(
            rows + (ulong(group.x) * 8 + row) * width + k);
      },
      table, sums, width, group, simd_lane, simd_group);
}

kernel void moe_combine(
    device const bfloat *expert_output [[buffer(0)]],
    device const uint *route_rows [[buffer(1)]],
    device const float *routing_weights [[buffer(2)]],
    device const bfloat *residual [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant MoeCombineParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  uint row = group.x;
  uint dimension = group.y * 256 + thread_index;
  if (row >= params.rows || dimension >= params.hidden_size)
    return;
  float value = float(residual[ulong(row) * params.hidden_size + dimension]);
  ulong route = ulong(row) * params.routes_per_row;
  for (uint slot = 0; slot < params.routes_per_row; ++slot) {
    value += routing_weights[route + slot] *
             float(expert_output[ulong(route_rows[route + slot]) *
                                     params.hidden_size +
                                 dimension]);
  }
  output[ulong(row) * params.hidden_size + dimension] = bfloat(value);
}
