#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/short_convolution_row.h"

// Routes group rows by expert so each tile streams its weights once. Each row
// carries top_k routed experts plus the Shared slices of the shared expert
// (ids `experts` on, moe_shared_slice), weighted by the sigmoid of the
// block's scalar gate (SharedGate) or by 1.
//
// One row per threadgroup: thread e holds expert e's score, the threadgroup
// reduces the shared expert's F32 scalar gate with a fixed partial-sum order,
// and simdgroup 0 orders the experts, descending score then ascending id,
// which is the order the shape's routing and tie-break contract requires.
// Scores and routing weights are fp32: on real 35B router rows, top-8 sets
// chosen from bf16-rounded scores differ from the fp64 choice on 12-15% of
// rows (fp32: at most 0.0024%), and bf16 weights round a fifth of the
// combine's outputs away from bf16 of its exact sum. The router
// (ops::MoeRouter) then weights the top_k experts by the softmax of their
// scores, or by each one's sigmoid over the sum of theirs
// (SigmoidNormalized), summed in rank order.
enum class MoeRouter : uint { Softmax, SigmoidNormalized };
template <uint Slots, MoeRouter Router, uint Shared, bool SharedGate>
inline void moe_route_select(device const float *scores, device bfloat *input,
                             device const float *shared_gate, device uint *selected,
                             device float *routing_weights,
                             constant MoeRouteParams &params, uint group,
                             uint thread_index, uint simd_lane,
                             uint simd_group, threadgroup float *row_scores,
                             threadgroup float *ordered,
                             threadgroup float *scalar_partials) {
  // Slots expert slots, one per thread.
  constexpr uint Simdgroups = Slots / 32;
  constexpr uint ExpertsPerLane = Slots / 32;
  const uint row = group;
  if (row >= params.rows)
    return;
  row_scores[thread_index] =
      thread_index < params.experts
          ? scores[ulong(row) * Slots + thread_index]
          : -numeric_limits<float>::infinity();

  if constexpr (SharedGate) {
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
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong row_routes = ulong(row) * (params.top_k + Shared);
  // Thread s routes shared slice s, the row's route top_k + s.
  if (thread_index < Shared) {
    const uint route = params.top_k + thread_index;
    float total = 0.0f;
    if constexpr (SharedGate)
      for (uint partial = 0; partial < Simdgroups; ++partial)
        total += scalar_partials[partial];
    selected[row_routes + route] = params.experts + thread_index;
    routing_weights[row_routes + route] = SharedGate ? splash_sigmoid(total) : 1.0f;
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
    if constexpr (Router == MoeRouter::Softmax) {
      for (uint other = 0; other < params.top_k; ++other) {
        denominator +=
            fast::exp2((ordered[other] - ordered[0]) * 1.44269504089f);
      }
      routing_weights[row_routes + rank] =
          fast::exp2((ordered[rank] - ordered[0]) * 1.44269504089f) /
          denominator;
    } else {
      for (uint other = 0; other < params.top_k; ++other)
        denominator += splash_sigmoid(ordered[other]);
      routing_weights[row_routes + rank] =
          splash_sigmoid(ordered[rank]) / (denominator + 1e-20f);
    }
  }
}

// The router's fp32 scores (kernels/shared/gguf_float.metal) in rows of
// Slots expert slots and, with a gated shared expert, the F32 shared-expert
// scalar gate, each routing a compiled routing (ops::kCompiledMoeRoutings)
// under its select suffix: the softmax one on SPLASH_MOE_EXPERT_SLOTS slots
// or (_e512) SPLASH_MOE_WIDE_EXPERT_SLOTS, Qwen3.8-Flash-Next's 512 experts,
// and Flint's on SPLASH_MOE_EXPERT_SLOTS, its 256.
#define MOE_ROUTE_SELECT_BODY(Slots, Router, Shared, SharedGate, input, shared_gate) \
    threadgroup float row_scores[Slots];                                       \
    threadgroup float ordered[Slots];                                          \
    threadgroup float scalar_partials[Slots / 32];                             \
    moe_route_select<Slots, Router, Shared, SharedGate>(                       \
        scores, input, shared_gate, selected, routing_weights, params, group,  \
        thread_index, simd_lane, simd_group, row_scores, ordered,              \
        scalar_partials);
#define MOE_ROUTE_SELECT_ENTRY(Name, Slots, Router, Shared)                    \
  kernel void Name(                                                            \
      device const float *scores [[buffer(0)]],                                \
      device bfloat *input [[buffer(1)]],                                      \
      device const float *shared_gate [[buffer(2)]],                           \
      device uint *selected [[buffer(3)]],                                     \
      device float *routing_weights [[buffer(4)]],                             \
      constant MoeRouteParams &params [[buffer(5)]],                           \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    MOE_ROUTE_SELECT_BODY(Slots, Router, Shared, true, input, shared_gate)     \
  }
// A shared expert without the gate, which reads neither the input nor a gate.
#define MOE_ROUTE_SELECT_UNGATED_ENTRY(Name, Slots, Router, Shared)            \
  kernel void Name(                                                            \
      device const float *scores [[buffer(0)]],                                \
      device uint *selected [[buffer(1)]],                                     \
      device float *routing_weights [[buffer(2)]],                             \
      constant MoeRouteParams &params [[buffer(3)]],                           \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    MOE_ROUTE_SELECT_BODY(Slots, Router, Shared, false, nullptr, nullptr)      \
  }
MOE_ROUTE_SELECT_ENTRY(moe_route_select_f32, SPLASH_MOE_EXPERT_SLOTS, MoeRouter::Softmax, 1)
MOE_ROUTE_SELECT_ENTRY(moe_route_select_f32_e512, SPLASH_MOE_WIDE_EXPERT_SLOTS, MoeRouter::Softmax, 1)
MOE_ROUTE_SELECT_UNGATED_ENTRY(moe_route_select_f32_sigmoid_s2, SPLASH_MOE_EXPERT_SLOTS,
                               MoeRouter::SigmoidNormalized, 2)
#undef MOE_ROUTE_SELECT_UNGATED_ENTRY
#undef MOE_ROUTE_SELECT_ENTRY
#undef MOE_ROUTE_SELECT_BODY

// A biased selection (ops::MoeShape::selectionBias) of a sigmoid router
// without a shared expert, on Slots expert slots: thread e ranks expert e by
// sigmoid(score) + bias[e], simdgroup 0 orders them as moe_route_select does
// (descending, the lower id among ties), and the top_k take their sigmoids
// over the sum of theirs plus 1e-20, summed in rank order, times the
// routing's scale.
template <uint Slots>
inline void moe_route_select_biased(device const float *scores, device const float *bias, device uint *selected,
                                    device float *routing_weights, constant MoeBiasedRouteParams &p, uint row,
                                    uint thread_index, uint simd_lane, uint simd_group,
                                    threadgroup float *row_scores, threadgroup float *ordered) {
  constexpr uint ExpertsPerLane = Slots / 32;
  if (row >= p.route.rows) return;
  device const float *row_logits = scores + ulong(row) * Slots;
  row_scores[thread_index] = thread_index < p.route.experts
                                 ? splash_sigmoid(row_logits[thread_index]) + bias[thread_index]
                                 : -numeric_limits<float>::infinity();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group != 0) return;
  const ulong row_routes = ulong(row) * p.route.top_k;
  float lane_scores[ExpertsPerLane];
  for (uint slot = 0; slot < ExpertsPerLane; ++slot) lane_scores[slot] = row_scores[simd_lane + 32 * slot];
  for (uint rank = 0; rank < p.route.top_k; ++rank) {
    float best = -numeric_limits<float>::infinity();
    uint best_slot = 0;
    for (uint slot = 0; slot < ExpertsPerLane; ++slot)
      if (lane_scores[slot] > best) {
        best = lane_scores[slot];
        best_slot = slot;
      }
    const float row_best = simd_max(best);
    const uint candidate = best == row_best ? simd_lane + 32 * best_slot : 0xFFFFFFFFu;
    const uint winner = simd_min(candidate);
    if (candidate == winner) lane_scores[best_slot] = -numeric_limits<float>::infinity();
    if (simd_lane == 0) {
      // The winner's sigmoid weighs it; its bias only ranked it.
      ordered[rank] = splash_sigmoid(row_logits[winner]);
      selected[row_routes + rank] = winner;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint rank = simd_lane; rank < p.route.top_k; rank += 32) {
    float denominator = 0.0f;
    for (uint other = 0; other < p.route.top_k; ++other) denominator += ordered[other];
    routing_weights[row_routes + rank] = ordered[rank] / (denominator + 1e-20f) * p.scale;
  }
}
kernel void moe_route_select_f32_sigmoid_biased_s0(device const float *scores [[buffer(0)]],
                                                   device const float *bias [[buffer(1)]],
                                                   device uint *selected [[buffer(2)]],
                                                   device float *routing_weights [[buffer(3)]],
                                                   constant MoeBiasedRouteParams &p [[buffer(4)]],
                                                   uint group [[threadgroup_position_in_grid]],
                                                   uint thread_index [[thread_index_in_threadgroup]],
                                                   uint simd_lane [[thread_index_in_simdgroup]],
                                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float row_scores[SPLASH_MOE_EXPERT_SLOTS];
  threadgroup float ordered[SPLASH_MOE_EXPERT_SLOTS];
  moe_route_select_biased<SPLASH_MOE_EXPERT_SLOTS>(scores, bias, selected, routing_weights, p, group, thread_index,
                                                   simd_lane, simd_group, row_scores, ordered);
}

// Whether a verified grouping leaves routed route `route` out: its row lies
// in no lane's rows, which no lane verifies (MoeGroupParams).
template <uint Shared, bool Verified>
inline bool moe_unverified_route(device const SplashLaneRows *lane_rows, constant MoeGroupParams &params,
                                 uint route) {
  if (!Verified)
    return false;
  const uint row = route / (params.top_k + Shared);
  for (uint lane = 0; lane < params.lanes; ++lane)
    // Wraps past the rows for a row before the lane's.
    if (row - lane_rows[lane].first < lane_rows[lane].rows)
      return false;
  return true;
}

// Sorts one command's routes by expert into tiles of at most tile_rows routes
// of one expert, a grouped row per route, each tile's rows following the tile
// before it's (MoeTileDescriptor). One threadgroup covers all routes and
// thread e owns expert e's count, tiles and first grouped row. The tiles of
// experts the slot table holds come first, then the shared slices', slice
// after slice, each slice's holding every row in order, then those of
// experts it marks MOE_SLOT_ABSENT. It writes the tile count with the expert
// passes' grids (MoeTileCount) and a streamed block's slot table into
// expert_slots (ops::MoeScratch::expertSlots). Slots threads, one per expert
// slot. A Verified grouping routes its lanes' rows alone (MoeGroupParams).
template <uint Slots, uint Shared, bool Verified>
inline void moe_group_routes_phase(
    device const uint *selected, device MoeTileDescriptor *tiles, device MoeTileCount *tile_count,
    device uint *grouped_routes, device uint *route_rows, device const uint *slots, device uint *expert_slots,
    device const SplashLaneRows *lane_rows,
    constant MoeGroupParams &params, uint thread_index, uint simd_lane, uint simd_group,
    threadgroup atomic_uint *counts, threadgroup atomic_uint *cursors, threadgroup uint *first_rows,
    threadgroup uint4 *simd_totals) {
  const uint routes_per_row = params.top_k + Shared;
  const uint routes = params.rows * routes_per_row;
  // A streamed grouping's slot, loaded before the counting: copied for the
  // passes beside the load, its absent bit parked in first_rows until the
  // counts are in.
  uint slot = 0;
  if (params.streamed && thread_index < params.experts) {
    slot = slots[thread_index];
    expert_slots[thread_index] = slot;
  }
  first_rows[thread_index] = slot & MOE_SLOT_ABSENT;
  atomic_store_explicit(&counts[thread_index], 0u, memory_order_relaxed);
  atomic_store_explicit(&cursors[thread_index], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint route = thread_index; route < routes; route += Slots) {
    if (moe_shared_slice(route % routes_per_row, params.top_k) >= Shared &&
        !moe_unverified_route<Shared, Verified>(lane_rows, params, route)) {
      atomic_fetch_add_explicit(&counts[selected[route]], 1u,
                                memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint count = atomic_load_explicit(&counts[thread_index], memory_order_relaxed);
  const uint expert_tiles = (count + params.tile_rows - 1) / params.tile_rows;
  const bool absent = first_rows[thread_index];
  // The tiles and rows of the experts before this one: the resident
  // experts' (x, z) and the absent ones' (y, w).
  const uint4 own = absent ? uint4(0, expert_tiles, 0, count) : uint4(expert_tiles, 0, count, 0);
  uint4 prefix = simd_prefix_exclusive_sum(own);
  if (simd_lane == 31)
    simd_totals[simd_group] = prefix + own;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint4 total = 0;
  for (uint group = 0; group < Slots / 32; ++group) {
    if (group < simd_group)
      prefix += simd_totals[group];
    total += simd_totals[group];
  }
  const uint shared_tiles =
      (params.rows + params.tile_rows - 1) / params.tile_rows;
  const uint resident = total.x + Shared * shared_tiles;
  // The shared slices' rows follow the resident experts', and the absent
  // experts' theirs.
  const uint shared_first_row = total.z;
  const uint tile_offset = absent ? resident + prefix.y : prefix.x;
  const uint first_row = absent ? shared_first_row + Shared * params.rows + prefix.w : prefix.z;
  first_rows[thread_index] = first_row;
  for (uint tile = 0; tile < expert_tiles; ++tile) {
    tiles[tile_offset + tile] = MoeTileDescriptor{
        thread_index, min(params.tile_rows, count - tile * params.tile_rows),
        first_row + tile * params.tile_rows};
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint route = thread_index; route < routes; route += Slots) {
    if (moe_shared_slice(route % routes_per_row, params.top_k) < Shared)
      continue;
    if (moe_unverified_route<Shared, Verified>(lane_rows, params, route)) {
      route_rows[route] = ~0u;
      continue;
    }
    const uint expert = selected[route];
    const uint row = first_rows[expert] +
                     atomic_fetch_add_explicit(&cursors[expert], 1u, memory_order_relaxed);
    grouped_routes[row] = route;
    route_rows[route] = row;
  }

  // Slice s's tiles and grouped rows follow slice s - 1's.
  for (uint tile = thread_index; tile < shared_tiles; tile += Slots) {
    for (uint slice = 0; slice < Shared; ++slice)
      tiles[total.x + slice * shared_tiles + tile] = MoeTileDescriptor{
          params.experts + slice,
          min(params.tile_rows, params.rows - tile * params.tile_rows),
          shared_first_row + slice * params.rows + tile * params.tile_rows};
  }
  for (uint row = thread_index; row < params.rows; row += Slots) {
    for (uint slice = 0; slice < Shared; ++slice) {
      const uint route = row * routes_per_row + params.top_k + slice;
      const uint grouped = shared_first_row + slice * params.rows + row;
      grouped_routes[grouped] = route;
      route_rows[route] = grouped;
    }
  }
  if (thread_index == 0) {
    const uint live = resident + total.y;
    *tile_count = MoeTileCount{live, resident,
                               {params.gate_up_columns, live, 1}, {params.down_columns, live, 1},
                               {params.gate_up_columns, resident, 1}, {params.down_columns, resident, 1},
                               {params.gate_up_columns, total.y, 1},
                               {params.down_columns, total.y, 1}};
  }
}

#define MOE_GROUP_ROUTES_BODY(Slots, Shared, Verified, lane_rows)             \
    threadgroup atomic_uint counts[Slots];                                     \
    threadgroup atomic_uint cursors[Slots];                                    \
    threadgroup uint first_rows[Slots];                                        \
    threadgroup uint4 simd_totals[Slots / 32];                                 \
    moe_group_routes_phase<Slots, Shared, Verified>(                           \
        selected, tiles, tile_count, grouped_routes, route_rows, slots,        \
        expert_slots, lane_rows, params, thread_index, simd_lane, simd_group,  \
        counts, cursors, first_rows, simd_totals);
#define MOE_GROUP_ROUTES_ENTRY(Name, Slots, Shared)                            \
  kernel void Name(                                                            \
      device const uint *selected [[buffer(0)]],                               \
      device MoeTileDescriptor *tiles [[buffer(1)]],                           \
      device MoeTileCount *tile_count [[buffer(2)]],                           \
      device uint *grouped_routes [[buffer(3)]],                               \
      device uint *route_rows [[buffer(4)]],                                   \
      device const uint *slots [[buffer(5)]],                                  \
      device uint *expert_slots [[buffer(6)]],                                 \
      constant MoeGroupParams &params [[buffer(7)]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    MOE_GROUP_ROUTES_BODY(Slots, Shared, false, nullptr)                       \
  }
// A grouping of a decode step, which routes its lanes' rows alone
// (MoeGroupParams), as the draft selector laid them out.
#define MOE_GROUP_ROUTES_VERIFIED_ENTRY(Name, Slots, Shared)                   \
  kernel void Name(                                                            \
      device const uint *selected [[buffer(0)]],                               \
      device MoeTileDescriptor *tiles [[buffer(1)]],                           \
      device MoeTileCount *tile_count [[buffer(2)]],                           \
      device uint *grouped_routes [[buffer(3)]],                               \
      device uint *route_rows [[buffer(4)]],                                   \
      device const uint *slots [[buffer(5)]],                                  \
      device uint *expert_slots [[buffer(6)]],                                 \
      device const SplashLaneRows *lane_rows [[buffer(7)]],                    \
      constant MoeGroupParams &params [[buffer(8)]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    MOE_GROUP_ROUTES_BODY(Slots, Shared, true, lane_rows)                      \
  }
// Each for the shared slices of a compiled routing, under its slices suffix
// (ops::kCompiledMoeRoutings), on the expert slots of its selection.
MOE_GROUP_ROUTES_ENTRY(moe_group_routes, SPLASH_MOE_EXPERT_SLOTS, 1)
MOE_GROUP_ROUTES_ENTRY(moe_group_routes_e512, SPLASH_MOE_WIDE_EXPERT_SLOTS, 1)
MOE_GROUP_ROUTES_VERIFIED_ENTRY(moe_group_routes_verified, SPLASH_MOE_EXPERT_SLOTS, 1)
MOE_GROUP_ROUTES_VERIFIED_ENTRY(moe_group_routes_verified_e512, SPLASH_MOE_WIDE_EXPERT_SLOTS, 1)
MOE_GROUP_ROUTES_ENTRY(moe_group_routes_s2, SPLASH_MOE_EXPERT_SLOTS, 2)
MOE_GROUP_ROUTES_VERIFIED_ENTRY(moe_group_routes_verified_s2, SPLASH_MOE_EXPERT_SLOTS, 2)
MOE_GROUP_ROUTES_ENTRY(moe_group_routes_s0, SPLASH_MOE_EXPERT_SLOTS, 0)
MOE_GROUP_ROUTES_VERIFIED_ENTRY(moe_group_routes_verified_s0, SPLASH_MOE_EXPERT_SLOTS, 0)
#undef MOE_GROUP_ROUTES_VERIFIED_ENTRY
#undef MOE_GROUP_ROUTES_ENTRY
#undef MOE_GROUP_ROUTES_BODY

// Copies each grouped row's input so every expert tile is a dense matrix of
// its live rows. Threadgroup (tile, part) copies the part's 256 columns of
// the tile's rows, eight rows at once, each thread eight columns (16 bytes,
// rows of whole 256 columns starting 16-byte aligned) of one row.
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
  const MoeTileDescriptor tile = tiles[group.x];
  const uint column = group.y * 256 + (thread_index % 32) * 8;
  for (uint row = tile.first_row + thread_index / 32; row < tile.first_row + tile.rows; row += 8) {
    const uint route = grouped_routes[row];
    *reinterpret_cast<device uint4 *>(grouped_input + ulong(row) * params.input_size + column) =
        *reinterpret_cast<device const uint4 *>(input + ulong(route / params.routes_per_row) * params.input_size +
                                                column);
  }
}

// Row `row`'s combined value at `dimension`: its residual plus each route's
// expert output times its routing weight, in route order, in fp32.
inline float moe_combine_value(device const bfloat *expert_output, device const uint *route_rows,
                               device const float *routing_weights, device const bfloat *residual,
                               constant MoeCombineParams &params, uint row, uint dimension) {
  float value = float(residual[ulong(row) * params.hidden_size + dimension]);
  ulong route = ulong(row) * params.routes_per_row;
  for (uint slot = 0; slot < params.routes_per_row; ++slot) {
    // A route a verified grouping left out (MoeGroupParams) adds nothing.
    const uint grouped = route_rows[route + slot];
    if (grouped == ~0u)
      continue;
    value += routing_weights[route + slot] *
             float(expert_output[ulong(grouped) * params.hidden_size +
                                 dimension]);
  }
  return value;
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
  output[ulong(row) * params.hidden_size + dimension] =
      bfloat(moe_combine_value(expert_output, route_rows, routing_weights, residual, params, row, dimension));
}

// A verify step's combine and the short convolution of its rows
// (MoeCombineConvolutionParams, ops::MoeCombineConvolution), what moe_combine
// and then short_convolution's ROUNDED_SUM of its rows write, in one
// dispatch: threadgroup (channel group, lane) runs the lane's rows of
// MOE_COMBINE_CONVOLUTION_CHANNELS channels, thread (x, y) the lane's row y
// at the group's channel x. Each combines its row's channel as moe_combine
// does into the threadgroup's rows and the tape; then each convolves it along
// its row's path from those rows (and before the lane's anchor its state,
// state_in lane) onto the residual. No combined row reaches memory but the
// tape. A lane whose rows run past the step's writes nothing.
kernel void moe_combine_convolution(
    device const bfloat *expert_output [[buffer(0)]], device const uint *route_rows [[buffer(1)]],
    device const float *routing_weights [[buffer(2)]], device const bfloat *combine_residual [[buffer(3)]],
    device const bfloat *state_in0 [[buffer(4)]], device const bfloat *state_in1 [[buffer(5)]],
    device const bfloat *state_in2 [[buffer(6)]], device const bfloat *state_in3 [[buffer(7)]],
    device const bfloat *state_in4 [[buffer(8)]], device const bfloat *state_in5 [[buffer(9)]],
    device const bfloat *state_in6 [[buffer(10)]], device const bfloat *state_in7 [[buffer(11)]],
    device const float *taps [[buffer(12)]], device const float *residual [[buffer(13)]],
    device bfloat *output [[buffer(14)]], device const SplashLaneRows *layout [[buffer(15)]],
    device const SplashTreeRow *tree [[buffer(16)]], device bfloat *tape [[buffer(17)]],
    constant MoeCombineConvolutionParams &params [[buffer(18)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 position [[thread_position_in_threadgroup]]) {
  constexpr int kStateRows = int(SHORT_CONVOLUTION_STATE_ROWS);
  threadgroup bfloat combined[MOE_COMBINE_CONVOLUTION_ROWS][MOE_COMBINE_CONVOLUTION_CHANNELS];
  constant ShortConvolutionParams &p = params.convolution;
  const uint lane = group.y, row = position.y, c = group.x * MOE_COMBINE_CONVOLUTION_CHANNELS + position.x;
  if (lane >= min(p.sequences, SPLASH_MAXIMUM_BATCH_WIDTH)) return;
  const SplashLaneRows rows = layout[lane];
  if (ulong(rows.first) + rows.rows > p.rows) return;
  const bool live = row < rows.rows && c < p.channels;
  const uint packed = rows.first + row;
  const ulong column = p.state_column + c;
  if (live) {
    const bfloat value = bfloat(moe_combine_value(expert_output, route_rows, routing_weights, combine_residual,
                                                  params.combine, packed, c));
    combined[row][position.x] = value;
    tape[ulong(packed) * p.state_stride + column] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!live) return;
  device const bfloat *state_in = SPLASH_LANE_BINDING(lane, state_in0, state_in1, state_in2, state_in3, state_in4,
                                                      state_in5, state_in6, state_in7);
  const auto at = [&](int t) -> bfloat {
    return t >= 0 ? combined[t][position.x] : state_in[ulong(kStateRows + t) * p.state_stride + column];
  };
  // The convolution's taps are static (p.heads 0): it reads no dynamic tap.
  short_convolution_output(at, taps, state_in, residual, output, p, c, packed, tree[packed].ancestors,
                           int(tree[packed].depth), combined[row][position.x]);
}
