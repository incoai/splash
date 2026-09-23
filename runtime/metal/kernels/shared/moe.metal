#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Routes group rows by expert so each tile streams its weights once. Each row
// carries top_k routed experts plus the shared expert (id `experts`), weighted
// by the sigmoid of its scalar gate. Experts use the StorageN=256 Q4 layout.
//
// Routing first writes 256 fp32 scores per row, then sorts them. Q8 affine
// terms accumulate per K slice, and slices sum in fixed order so both score
// tile shapes produce identical results. The 8-row tile favors short chunks;
// the 32-row tile amortizes weight loads over longer chunks. Measurements
// cover Apple10; Apple9 validation remains outstanding.
constant constexpr uint kMoeRouteSlices = 8;

// Quant groups [x, y) of one K slice; both tiles use this partition.
inline uint2 moe_route_slice(uint slice, uint quant_groups) {
  return uint2(slice * quant_groups / kMoeRouteSlices,
               (slice + 1) * quant_groups / kMoeRouteSlices);
}

// Sum of a staged 64-wide row in a fixed lane order.
inline float moe_route_row_sum(threadgroup const bfloat *row, uint simd_lane) {
  return simd_sum(float(row[simd_lane]) + float(row[simd_lane + 32]));
}

// One quant group's affine term. The explicit fma pins the rounding so both
// tiles evaluate it identically.
inline float moe_route_group_term(float dot, float scale, float sum,
                                  float bias) {
  return fma(dot, scale, sum * bias);
}

// Eight rows by 32 experts per threadgroup. Simdgroup s owns K slice s and
// stages its rows in threadgroup memory, so a ragged tail never reads past
// the last live row; the threadgroup then sums the slice partials in order.
kernel void moe_route_scores_q8_m8(
    device bfloat *input [[buffer(0)]],
    device uint8_t *router_weights [[buffer(1)]],
    device bfloat *router_scales [[buffer(2)]],
    device bfloat *router_biases [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant MoeRouteParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = 8;
  constexpr uint TileN = 32;
  constexpr uint StorageN = 256;
  constexpr uint Simdgroups = 8;
  static_assert(Simdgroups == kMoeRouteSlices, "one simdgroup per K slice");
  threadgroup uint4 staged_storage[Simdgroups * Rows * 64 * 2 / 16];
  threadgroup float partials[Simdgroups * Rows * TileN];
  threadgroup float parameters[Simdgroups * 2 * TileN];
  threadgroup float sums[Simdgroups * Rows];
  const uint row_base = group.x * Rows;
  if (row_base >= params.rows)
    return;
  const uint live_rows = min(Rows, params.rows - row_base);
  const uint expert_origin = group.y * TileN;
  const uint quant_groups = params.input_size / 64;

  threadgroup bfloat *staged =
      reinterpret_cast<threadgroup bfloat *>(staged_storage) +
      simd_group * Rows * 64;
  threadgroup float *slice_parameters = parameters + simd_group * 2 * TileN;
  threadgroup float *slice_sums = sums + simd_group * Rows;
  auto a = tensor(staged, dextents<int, 2>{64, int(Rows)},
                  array<int, 2>{1, 64});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a_slice = a.slice<64, Rows>(0, 0);
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> first_b(
      router_weights + ulong(expert_origin) * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto first_b_slice = first_b.slice<64, TileN>(0, 0);
  auto partial = operation.get_destination_cooperative_tensor<
      decltype(a_slice), decltype(first_b_slice), float>();
  auto accumulated = operation.get_destination_cooperative_tensor<
      decltype(a_slice), decltype(first_b_slice), float>();
  const bool fullyOccupied =
      uint(accumulated.get_capacity()) * 32u == Rows * TileN;
  const auto traversal =
      fullyOccupied ? Q4Traversal::All : q4_traversal(accumulated);
  q4_visit(accumulated, traversal, [&](ushort i) { accumulated[i] = 0.0f; });

  const uint2 slice = moe_route_slice(simd_group, quant_groups);
  const uint stage_row = simd_lane / 4;
  const uint stage_column = (simd_lane % 4) * 16;
  device const bfloat *stage_source =
      input +
      ulong(row_base + min(stage_row, live_rows - 1)) * params.input_size +
      stage_column;
  threadgroup uint4 *stage_destination = reinterpret_cast<threadgroup uint4 *>(
      staged + stage_row * 64 + stage_column);
  for (uint quant_group = slice.x; quant_group < slice.y; ++quant_group) {
    if (stage_row < live_rows) {
      device const uint4 *source = reinterpret_cast<device const uint4 *>(
          stage_source + quant_group * 64);
      stage_destination[0] = source[0];
      stage_destination[1] = source[1];
    } else {
      stage_destination[0] = uint4(0);
      stage_destination[1] = uint4(0);
    }
    const ulong parameter =
        ulong(quant_group) * StorageN + expert_origin + simd_lane;
    slice_parameters[simd_lane] = float(router_scales[parameter]);
    slice_parameters[TileN + simd_lane] = float(router_biases[parameter]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = 0; row < Rows; ++row) {
      const float sum = moe_route_row_sum(staged + row * 64, simd_lane);
      if (simd_lane == 0)
        slice_sums[row] = sum;
    }
    tensor<device uint8_t, dextents<int, 2>, tensor_inline> b(
        router_weights + (ulong(quant_group) * StorageN + expert_origin) * 64,
        dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    operation.run(a_slice, b_slice, partial);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    q4_visit(accumulated, traversal,
             [&](ushort i) __attribute__((always_inline)) {
      auto index = accumulated.get_multidimensional_index(i);
      accumulated[i] = accumulated[i] +
          moe_route_group_term(partial[i], slice_parameters[index[0]],
                               slice_sums[index[1]],
                               slice_parameters[TileN + index[0]]);
    });
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  q4_visit(accumulated, traversal, [&](ushort i) {
    auto index = accumulated.get_multidimensional_index(i);
    partials[(simd_group * Rows + index[1]) * TileN + index[0]] =
        accumulated[i];
  });
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row = thread_index / TileN;
  const uint column = thread_index % TileN;
  float total = 0.0f;
  for (uint s = 0; s < kMoeRouteSlices; ++s)
    total += partials[(s * Rows + row) * TileN + column];
  if (row < live_rows) {
    scores[ulong(row_base + row) * StorageN + expert_origin + column] = total;
  }
}

// 32 rows by 128 experts per threadgroup: all eight simdgroups run each
// quant group's matmul from the staged rows, accumulating the K slices in the
// same order as the 8-row tile.
kernel void moe_route_scores_q8_m32(
    device bfloat *input [[buffer(0)]],
    device uint8_t *router_weights [[buffer(1)]],
    device bfloat *router_scales [[buffer(2)]],
    device bfloat *router_biases [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant MoeRouteParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = 32;
  constexpr uint TileN = 128;
  constexpr uint StorageN = 256;
  constexpr uint Simdgroups = 8;
  threadgroup uint4 staged_storage[Rows * 64 * 2 / 16];
  // Scales/biases and row sums alternate buffers so the next group's stores
  // never race the epilogue reads of the current one.
  threadgroup float parameters[2 * 2 * TileN];
  threadgroup float sums[2 * Rows];
  const uint row_base = group.x * Rows;
  if (row_base >= params.rows)
    return;
  const uint live_rows = min(Rows, params.rows - row_base);
  const uint expert_origin = group.y * TileN;
  const uint quant_groups = params.input_size / 64;

  threadgroup bfloat *staged =
      reinterpret_cast<threadgroup bfloat *>(staged_storage);
  auto a = tensor(staged, dextents<int, 2>{64, int(Rows)},
                  array<int, 2>{1, 64});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a_slice = a.slice<64, Rows>(0, 0);
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> first_b(
      router_weights + ulong(expert_origin) * 64, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto first_b_slice = first_b.slice<64, TileN>(0, 0);
  auto partial = operation.get_destination_cooperative_tensor<
      decltype(a_slice), decltype(first_b_slice), float>();
  auto slice_sum = operation.get_destination_cooperative_tensor<
      decltype(a_slice), decltype(first_b_slice), float>();
  auto total = operation.get_destination_cooperative_tensor<
      decltype(a_slice), decltype(first_b_slice), float>();
  const bool fullyOccupied =
      uint(total.get_capacity()) * (Simdgroups * 32u) == Rows * TileN;
  const auto traversal =
      fullyOccupied ? Q4Traversal::All : q4_traversal(total);
  q4_visit(total, traversal, [&](ushort i) { total[i] = 0.0f; });

  const uint stage_row = thread_index / 8;
  const uint stage_column = (thread_index % 8) * 8;
  device const bfloat *stage_source =
      input +
      ulong(row_base + min(stage_row, live_rows - 1)) * params.input_size +
      stage_column;
  threadgroup uint4 *stage_destination = reinterpret_cast<threadgroup uint4 *>(
      staged + stage_row * 64 + stage_column);
  const bool scale_thread = thread_index < TileN;
  const uint parameter_column = expert_origin + thread_index % TileN;
  for (uint s = 0; s < kMoeRouteSlices; ++s) {
    q4_visit(slice_sum, traversal, [&](ushort i) { slice_sum[i] = 0.0f; });
    const uint2 slice = moe_route_slice(s, quant_groups);
    for (uint quant_group = slice.x; quant_group < slice.y; ++quant_group) {
      const uint buffer = quant_group & 1;
      if (stage_row < live_rows) {
        stage_destination[0] = *reinterpret_cast<device const uint4 *>(
            stage_source + quant_group * 64);
      } else {
        stage_destination[0] = uint4(0);
      }
      threadgroup float *group_parameters = parameters + buffer * 2 * TileN;
      threadgroup float *group_sums = sums + buffer * Rows;
      const ulong parameter = ulong(quant_group) * StorageN + parameter_column;
      group_parameters[thread_index] =
          scale_thread ? float(router_scales[parameter])
                       : float(router_biases[parameter]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint row = simd_group; row < Rows; row += Simdgroups) {
        const float sum = moe_route_row_sum(staged + row * 64, simd_lane);
        if (simd_lane == 0)
          group_sums[row] = sum;
      }
      tensor<device uint8_t, dextents<int, 2>, tensor_inline> b(
          router_weights +
              (ulong(quant_group) * StorageN + expert_origin) * 64,
          dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
      auto b_slice = b.slice<64, TileN>(0, 0);
      operation.run(a_slice, b_slice, partial);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      q4_visit(slice_sum, traversal,
               [&](ushort i) __attribute__((always_inline)) {
        auto index = slice_sum.get_multidimensional_index(i);
        slice_sum[i] = slice_sum[i] +
            moe_route_group_term(partial[i], group_parameters[index[0]],
                                 group_sums[index[1]],
                                 group_parameters[TileN + index[0]]);
      });
    }
    q4_visit(total, traversal,
             [&](ushort i) { total[i] = total[i] + slice_sum[i]; });
  }
  q4_visit(total, traversal, [&](ushort i) {
    auto index = total.get_multidimensional_index(i);
    if (uint(index[1]) < live_rows) {
      scores[ulong(row_base + index[1]) * StorageN + expert_origin +
             index[0]] = total[i];
    }
  });
}

// The shared expert's scalar gate weight of input `dimension`: the Q8 affine
// row of a packed model, or the F32 tensor of a GGUF (ffn_gate_inp_shexp),
// which runs unrounded.
struct MoeSharedGateQ8 {
  device uint8_t *weights;
  device bfloat *scales;
  device bfloat *biases;
};
struct MoeSharedGateF32 {
  device const float *weights;
};
inline float moe_shared_gate_weight(MoeSharedGateQ8 gate, uint dimension) {
  constexpr uint StorageN = 256;
  uint quant_group = dimension / 64;
  uint within_group = dimension % 64;
  ulong weight_index = ulong(quant_group) * StorageN * 64 + within_group;
  ulong parameter = ulong(quant_group) * StorageN;
  return float(gate.weights[weight_index]) * float(gate.scales[parameter]) +
         float(gate.biases[parameter]);
}
inline float moe_shared_gate_weight(MoeSharedGateF32 gate, uint dimension) {
  return gate.weights[dimension];
}

// One row per threadgroup: thread e holds expert e's score, the threadgroup
// reduces the shared expert's scalar gate with a fixed partial-sum order, and
// simdgroup 0 orders the experts, descending score then ascending id, which
// is the order the shape's routing and tie-break contract requires. Scores
// and routing weights are fp32. On 40,960 router rows each of real 35B prose,
// code and chat, top-8 sets chosen from bf16-rounded scores differ from the
// fp64 choice on 12-15% of rows (fp32: at most 0.0024%), and bf16 weights
// round 19-20% of the combine's outputs away from bf16 of its exact sum
// (fp32: 0.008%).
template <class SharedGate>
inline void moe_route_select(device const float *scores, device bfloat *input,
                             SharedGate shared_gate, device uint *selected,
                             device float *routing_weights,
                             constant MoeRouteParams &params, uint group,
                             uint thread_index, uint simd_lane,
                             uint simd_group, threadgroup float *row_scores,
                             threadgroup float *ordered,
                             threadgroup float *scalar_partials) {
  constexpr uint StorageN = 256;
  constexpr uint Simdgroups = StorageN / 32;
  constexpr uint ExpertsPerLane = StorageN / 32;
  const uint row = group;
  if (row >= params.rows)
    return;
  row_scores[thread_index] =
      thread_index < params.experts
          ? scores[ulong(row) * StorageN + thread_index]
          : -numeric_limits<float>::infinity();

  device bfloat *row_input = input + ulong(row) * params.input_size;
  float scalar = 0.0f;
  for (uint dimension = thread_index; dimension < params.input_size;
       dimension += StorageN) {
    scalar += float(row_input[dimension]) *
              moe_shared_gate_weight(shared_gate, dimension);
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
    routing_weights[row_routes + params.top_k] =
        1.0f / (1.0f + fast::exp2(-1.44269504089f * total));
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

kernel void moe_route_select_q8(
    device const float *scores [[buffer(0)]],
    device bfloat *input [[buffer(1)]],
    device uint8_t *shared_weights [[buffer(2)]],
    device bfloat *shared_scales [[buffer(3)]],
    device bfloat *shared_biases [[buffer(4)]],
    device uint *selected [[buffer(5)]],
    device float *routing_weights [[buffer(6)]],
    constant MoeRouteParams &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float row_scores[256];
  threadgroup float ordered[256];
  threadgroup float scalar_partials[8];
  moe_route_select(scores, input,
                   MoeSharedGateQ8{shared_weights, shared_scales, shared_biases},
                   selected, routing_weights, params, group, thread_index,
                   simd_lane, simd_group, row_scores, ordered, scalar_partials);
}

// The GGUF router: fp32 scores (kernels/shared/gguf_float.metal) and the F32
// shared-expert gate.
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
  threadgroup float row_scores[256];
  threadgroup float ordered[256];
  threadgroup float scalar_partials[8];
  moe_route_select(scores, input, MoeSharedGateF32{shared_gate}, selected,
                   routing_weights, params, group, thread_index, simd_lane,
                   simd_group, row_scores, ordered, scalar_partials);
}

// Sorts one command's routes by expert. Tile t covers grouped rows
// [t * tile_rows, (t + 1) * tile_rows) of a single expert; rows past that
// expert's last route carry the route ~0u. One threadgroup covers all routes
// and thread e owns expert e's count, offsets and tile descriptors. The shared
// expert's tiles follow the routed tiles and hold every row in order.
kernel void moe_group_routes(
    device const uint *selected [[buffer(0)]],
    device MoeTileDescriptor *tiles [[buffer(1)]],
    device uint *tile_count [[buffer(2)]],
    device uint *grouped_routes [[buffer(3)]],
    device uint *route_rows [[buffer(4)]],
    constant MoeGroupParams &params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Experts = 256;
  const uint routes_per_row = params.top_k + 1;
  const uint routes = params.rows * routes_per_row;
  threadgroup atomic_uint counts[Experts];
  threadgroup atomic_uint cursors[Experts];
  threadgroup uint tile_offsets[Experts];
  threadgroup uint simd_totals[8];
  threadgroup uint routed_tiles;
  atomic_store_explicit(&counts[thread_index], 0u, memory_order_relaxed);
  atomic_store_explicit(&cursors[thread_index], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint route = thread_index; route < routes; route += Experts) {
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
  if (thread_index == Experts - 1)
    routed_tiles = tile_offset + expert_tiles;
  for (uint tile = 0; tile < expert_tiles; ++tile) {
    tiles[tile_offset + tile] = MoeTileDescriptor{
        thread_index, min(params.tile_rows, count - tile * params.tile_rows)};
  }
  for (uint row = tile_offset * params.tile_rows + count;
       row < (tile_offset + expert_tiles) * params.tile_rows; ++row) {
    grouped_routes[row] = ~0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint route = thread_index; route < routes; route += Experts) {
    if (route % routes_per_row == params.top_k)
      continue;
    uint expert = selected[route];
    uint slot =
        atomic_fetch_add_explicit(&cursors[expert], 1u, memory_order_relaxed);
    uint row = tile_offsets[expert] * params.tile_rows + slot;
    grouped_routes[row] = route;
    route_rows[route] = row;
  }

  const uint shared_tiles =
      (params.rows + params.tile_rows - 1) / params.tile_rows;
  const uint shared_base = routed_tiles * params.tile_rows;
  for (uint tile = thread_index; tile < shared_tiles; tile += Experts) {
    tiles[routed_tiles + tile] = MoeTileDescriptor{
        params.experts,
        min(params.tile_rows, params.rows - tile * params.tile_rows)};
  }
  for (uint row = thread_index; row < shared_tiles * params.tile_rows;
       row += Experts) {
    if (row < params.rows) {
      uint route = row * routes_per_row + params.top_k;
      grouped_routes[shared_base + row] = route;
      route_rows[route] = shared_base + row;
    } else {
      grouped_routes[shared_base + row] = ~0u;
    }
  }
  if (thread_index == 0)
    *tile_count = routed_tiles + shared_tiles;
}

// Copies each grouped row's input so every expert tile is a dense matrix;
// padding rows are zero and their outputs are never read.
kernel void moe_gather_rows(device const bfloat *input [[buffer(0)]],
                            device const uint *grouped_routes [[buffer(1)]],
                            device const uint *tile_count [[buffer(2)]],
                            device bfloat *grouped_input [[buffer(3)]],
                            constant MoeGatherParams &params [[buffer(4)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]]) {
  if (group.x >= *tile_count)
    return;
  uint column = group.y * 256 + thread_index;
  for (uint local = 0; local < params.tile_rows; ++local) {
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
// simdgroup per row: 8 us per 35B decode layer at one lane on the 40-core M3
// Max and 14 us at four, against 15 and 44 us for a threadgroup per span pair
// (most of a decode dispatch's tiles are past the tile count) and 19 and 23 us
// for one per tile. The gather reads each grouped row's route like
// moe_gather_rows (padding rows are zero); the prepare reads grouped rows (the
// down pass's intermediate). Grid (tiles, width / 256), 256 threads.
template <class Source>
inline void moe_table16_tile(Source source, device bfloat *table,
                             device float *sums, uint width, uint2 group,
                             uint simd_lane, uint simd_group) {
  for (uint span = group.y * 4; span < group.y * 4 + 4; ++span) {
    const bfloat2 values = source(simd_group, span * 64 + 2 * simd_lane);
    q16sg::write_input(table + ulong(group.x) * width * 8,
                       sums + ulong(group.x) * q16sg::sums_per_tile(width),
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

// One expert tile: Rows grouped rows times TileN output columns, read through
// the same Q4 tile as the dense projections. The expert's weights stream once
// per tile instead of once per route. Decode plans and the M8 prefill plan
// run this fused gate/up tile; the M32 prefill plan runs the split N256
// passes in prefill/moe.metal. The threadgroup is Simdgroups * 32 threads.
template <ushort Rows, bool GateUp, ushort TileN = 128, ushort Simdgroups = 8>
inline void moe_expert_tile(device bfloat *grouped_input,
                            device const MoeTileDescriptor *tiles,
                            device const uint *tile_count,
                            device uchar *packed_0, device uchar *packed_1,
                            device uchar *shared_0, device uchar *shared_1,
                            device bfloat *output,
                            constant MoeExpertParams &params, uint2 group,
                            threadgroup float *input_sums, uint simd_lane,
                            uint simd_group) {
  if (group.y >= *tile_count)
    return;
  const uint expert = tiles[group.y].expert;
  const MoeQ4Slab slab_0 =
      moe_q4_slab(packed_0, shared_0, expert, params.experts,
                  params.expert_stride_bytes_0, params.output_size,
                  params.input_size);
  MoeQ4Slab slab_1 = slab_0;
  if constexpr (GateUp) {
    slab_1 = moe_q4_slab(packed_1, shared_1, expert, params.experts,
                         params.expert_stride_bytes_1, params.output_size,
                         params.input_size);
  }
  device bfloat *input = grouped_input + ulong(group.y) * Rows * params.input_size;
  device bfloat *tile_output =
      output + ulong(group.y) * Rows * params.output_size;
  if constexpr (Rows == 8) {
    q4_mpp_tile<TileN, GateUp, false, 256, false, Simdgroups>(
        input, slab_0.weights, slab_0.scales, slab_0.biases, tile_output,
        slab_1.weights, slab_1.scales, slab_1.biases, tile_output,
        params.output_size, params.input_size, input_sums, group.x * TileN,
        simd_lane, simd_group);
  } else {
    q4_mpp_tile_batched<Rows, TileN, GateUp, false, 256, false, Simdgroups>(
        input, slab_0.weights, slab_0.scales, slab_0.biases, tile_output,
        slab_1.weights, slab_1.scales, slab_1.biases, tile_output,
        params.output_size, params.input_size, input_sums, group.x * TileN,
        simd_lane, simd_group);
  }
}

kernel void moe_expert_gate_up_q4_m8(
    device bfloat *grouped_input [[buffer(0)]],
    device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]],
    device uchar *gate_packed [[buffer(3)]],
    device uchar *up_packed [[buffer(4)]],
    device uchar *shared_gate [[buffer(5)]],
    device uchar *shared_up [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant MoeExpertParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[64];
  moe_expert_tile<8, true>(grouped_input, tiles, tile_count, gate_packed,
                           up_packed, shared_gate, shared_up, output, params,
                           group, input_sums, simd_lane, simd_group);
}

kernel void moe_expert_gate_up_q4_m32(
    device bfloat *grouped_input [[buffer(0)]],
    device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]],
    device uchar *gate_packed [[buffer(3)]],
    device uchar *up_packed [[buffer(4)]],
    device uchar *shared_gate [[buffer(5)]],
    device uchar *shared_up [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant MoeExpertParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[256];
  moe_expert_tile<32, true>(grouped_input, tiles, tile_count, gate_packed,
                            up_packed, shared_gate, shared_up, output, params,
                            group, input_sums, simd_lane, simd_group);
}

kernel void moe_expert_down_q4_m8(
    device bfloat *grouped_input [[buffer(0)]],
    device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]],
    device uchar *down_packed [[buffer(3)]],
    device uchar *shared_down [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    constant MoeExpertParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[64];
  moe_expert_tile<8, false>(grouped_input, tiles, tile_count, down_packed,
                            down_packed, shared_down, shared_down, output,
                            params, group, input_sums, simd_lane, simd_group);
}

kernel void moe_expert_down_q4_m32(
    device bfloat *grouped_input [[buffer(0)]],
    device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]],
    device uchar *down_packed [[buffer(3)]],
    device uchar *shared_down [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    constant MoeExpertParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[256];
  moe_expert_tile<32, false>(grouped_input, tiles, tile_count, down_packed,
                             down_packed, shared_down, shared_down, output,
                             params, group, input_sums, simd_lane, simd_group);
}

// Four-simdgroup 8-row tiles for Apple9 decode plans: gate/up at N128 and
// down at N256, 128 threads per threadgroup, the same buffers as the m8
// kernels above and a {output_size / TileN, tiles} grid. Family 9 has no
// per-core matrix unit, and a decode expert grid leaves it latency-bound at
// low occupancy, where halving the simdgroups per tile pays. Measured on a
// 40-core Apple9 GPU at the 35B shape (H=2048, E=256, top_k=8, I=512), ms
// per layer at rows 8/16/24/32 against the eight-simdgroup tiles: gate/up
// 0.332/0.551/0.728/0.859 -> 0.314/0.503/0.618/0.699 (1.06x-1.23x), down
// 0.157/0.274/0.363/0.419 -> 0.137/0.226/0.297/0.335 (1.15x-1.25x). Apple10
// stays within noise of the shipped tiles (<= 1.07x) and keeps them. Every
// output element sums its quant groups in the same order at either width or
// simdgroup count, so the outputs are bit-identical to the tiles above.
kernel void moe_expert_gate_up_q4_m8_n128_sg4(
    device bfloat *grouped_input [[buffer(0)]],
    device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]],
    device uchar *gate_packed [[buffer(3)]],
    device uchar *up_packed [[buffer(4)]],
    device uchar *shared_gate [[buffer(5)]],
    device uchar *shared_up [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant MoeExpertParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[64];
  moe_expert_tile<8, true, 128, 4>(grouped_input, tiles, tile_count,
                                   gate_packed, up_packed, shared_gate,
                                   shared_up, output, params, group,
                                   input_sums, simd_lane, simd_group);
}

kernel void moe_expert_down_q4_m8_n256_sg4(
    device bfloat *grouped_input [[buffer(0)]],
    device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]],
    device uchar *down_packed [[buffer(3)]],
    device uchar *shared_down [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    constant MoeExpertParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float input_sums[64];
  moe_expert_tile<8, false, 256, 4>(grouped_input, tiles, tile_count,
                                    down_packed, down_packed, shared_down,
                                    shared_down, output, params, group,
                                    input_sums, simd_lane, simd_group);
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
