#include <metal_stdlib>
#include "metal/abi/FlashMoE.h"

using namespace metal;

// Keep the source's storage-type boundaries even under an optimized build.
// The routing slot order is explicitly canonical: BF16 probability descending,
// ties by ascending ID. MLX argpartition does not specify that order.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline float flash_moe_nan() { return as_type<float>(0x7fc00000u); }

// Actual MLX Metal compiled SwiGLU uses fast exp, with BF16 exp/add/div/sub
// boundaries. The standalone unary/shared-expert sigmoid uses precise exp.
// This is an operator distinction, fixed across every row count and phase.
inline bfloat flash_moe_sigmoid_compiled(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bfloat flash_moe_sigmoid_unary(bfloat source) {
  const bfloat exponent = bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bool flash_moe_routing_geometry(uint rows, uint experts,
                                        uint selections) {
  return rows && rows <= 8192 && experts && experts <= 512 && selections &&
      selections <= 10 && selections <= experts;
}

inline bool flash_moe_pointwise_geometry(uint rows, uint width,
                                          uint selections) {
  return rows && rows <= 8192 && width && width <= 2560 && selections &&
      selections <= 10;
}

template <uint Threads>
inline void flash_moe_route_impl(
    const device bfloat *logits, device long *expert_ids, device bfloat *weights,
    device atomic_uint *diagnostics, constant FlashMoERouteParams &p,
    uint row, uint tid, uint actual_threads, uint simd_group, uint lane,
    threadgroup float *exponentials, threadgroup float *group_peaks,
    threadgroup uint *group_ids, threadgroup uint *group_invalid,
    threadgroup float *shared_float, threadgroup uint *shared_uint) {
  if (!p.rows || p.rows > 8192 || !p.experts || p.experts > 512 ||
      !p.selections || p.selections > 10 || p.selections > p.experts ||
      p.normalize_top_k > 1 || actual_threads != Threads) {
    if (tid == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (row >= p.rows) return;
  constexpr uint Members = 512 / Threads;
  constexpr uint Groups = Threads / 32;
  float local[Members];
  float peak = -INFINITY;
  bool invalid = false;
  for (uint member = 0; member < Members; ++member) {
    const uint expert = tid + member * Threads;
    const float x = expert < p.experts
        ? float(logits[ulong(row) * p.experts + expert]) : -INFINITY;
    local[member] = x;
    peak = metal::max(peak, x);
    if (expert < p.experts && !metal::isfinite(x)) invalid = true;
  }
  const float simd_peak = simd_max(peak);
  const uint simd_invalid = uint(simd_any(invalid));
  if (lane == 0) {
    group_peaks[simd_group] = simd_peak;
    group_invalid[simd_group] = simd_invalid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const float lane_peak = lane < Groups ? group_peaks[lane] : -INFINITY;
    const uint lane_invalid = lane < Groups ? group_invalid[lane] : 0u;
    const float all_peak = simd_max(lane_peak);
    const uint all_invalid = simd_or(lane_invalid);
    if (lane == 0) {
      shared_float[0] = all_peak;
      shared_uint[0] = all_invalid;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (shared_uint[0]) {
    if (tid == 0) {
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
      for (uint slot = 0; slot < p.selections; ++slot) {
        const ulong index = ulong(row) * p.selections + slot;
        expert_ids[index] = -1;
        weights[index] = bfloat(flash_moe_nan());
      }
    }
    return;
  }
  for (uint member = 0; member < Members; ++member) {
    const uint expert = tid + member * Threads;
    if (expert < p.experts)
      exponentials[expert] = metal::exp(local[member] - shared_float[0]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    // Keep EXACT production reduction order. Only peak/search are parallel.
    float total = 0.0f;
    for (uint expert = 0; expert < p.experts; ++expert)
      total += exponentials[expert];
    shared_float[1] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint member = 0; member < Members; ++member) {
    const uint expert = tid + member * Threads;
    local[member] = expert < p.experts
        ? float(bfloat(exponentials[expert] / shared_float[1])) : -1.0f;
  }
  long selected[10];
  float selected_scores[10];
  float selected_sum = 0.0f;
  for (uint slot = 0; slot < p.selections; ++slot) {
    float local_best = -1.0f;
    uint local_id = UINT_MAX;
    for (uint member = 0; member < Members; ++member) {
      const uint expert = tid + member * Threads;
      const float probability = local[member];
      if (expert < p.experts && (probability > local_best ||
          (probability == local_best && expert < local_id))) {
        local_best = probability;
        local_id = expert;
      }
    }
    const float simd_best = simd_max(local_best);
    const uint simd_id = simd_min(local_best == simd_best ? local_id : UINT_MAX);
    float all_best;
    uint all_id;
    if constexpr (Threads == 32) {
      // All lanes agree; no cross-SIMD barrier is necessary for top-k.
      all_best = simd_best;
      all_id = simd_id;
    } else {
      if (lane == 0) {
        group_peaks[simd_group] = simd_best;
        group_ids[simd_group] = simd_id;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_group == 0) {
        const float candidate = lane < Groups ? group_peaks[lane] : -1.0f;
        const uint candidate_id = lane < Groups ? group_ids[lane] : UINT_MAX;
        const float winner = simd_max(candidate);
        const uint winner_id = simd_min(candidate == winner ? candidate_id : UINT_MAX);
        if (lane == 0) {
          shared_float[2] = winner;
          shared_uint[1] = winner_id;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      all_best = shared_float[2];
      all_id = shared_uint[1];
    }
    if (tid == 0) {
      selected[slot] = long(all_id);
      selected_scores[slot] = all_best;
      // Actual MLX Metal row-small reduction stores every sum in BF16.
      selected_sum = float(bfloat(selected_sum) + bfloat(all_best));
    }
    // Each lane owns every write to its local candidates. The next reduction
    // has a barrier before any shared winner is replaced in the 256 route.
    for (uint member = 0; member < Members; ++member)
      if (tid + member * Threads == all_id) local[member] = -1.0f;
  }
  if (tid != 0) return;
  const float denominator = shared_float[1];
  const float normalized_denominator = float(bfloat(selected_sum));
  if (!metal::isfinite(denominator) || denominator <= 0.0f ||
      !metal::isfinite(normalized_denominator) ||
      normalized_denominator <= 0.0f) {
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
    for (uint slot = 0; slot < p.selections; ++slot) {
      const ulong index = ulong(row) * p.selections + slot;
      expert_ids[index] = -1;
      weights[index] = bfloat(flash_moe_nan());
    }
    return;
  }
  for (uint slot = 0; slot < p.selections; ++slot) {
    const ulong index = ulong(row) * p.selections + slot;
    expert_ids[index] = selected[slot];
    weights[index] = bfloat(p.normalize_top_k
        ? selected_scores[slot] / normalized_denominator : selected_scores[slot]);
  }
}

// 0 BF16[rows,experts] logits, 1 I64[rows,selections] IDs,
// 2 BF16[rows,selections] weights, 3 sticky uint32 diagnostics,
// 4 FlashMoERouteParams. Exactly 256 threads and one group per logical row.
// The eight SIMD groups reduce max/ID in parallel, preserving ascending-ID
// ties, the original sequential F32 denominator, and source BF16 score sum.
kernel void flash_moe_route(
    const device bfloat *logits [[buffer(0)]],
    device long *expert_ids [[buffer(1)]],
    device bfloat *weights [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant FlashMoERouteParams &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float exponentials[512];
  threadgroup float group_peaks[8];
  threadgroup uint group_ids[8];
  threadgroup uint group_invalid[8];
  threadgroup float shared_float[3];
  threadgroup uint shared_uint[2];
  flash_moe_route_impl<256>(
      logits, expert_ids, weights, diagnostics, p, row, tid, threads,
      simd_group, lane, exponentials, group_peaks, group_ids, group_invalid,
      shared_float, shared_uint);
}

// 0 BF16 gate, 1 BF16 up, 2 BF16 activated output, 3 sticky diagnostics,
// 4 FlashMoEPointwiseParams. Shapes [rows,selections,width], including the
// shared expert's single selection. Three BF16 storage boundaries match the
// actual MLX GPU compiled-SwiGLU golden oracle.
kernel void flash_moe_silu_multiply(
    const device bfloat *gate [[buffer(0)]],
    const device bfloat *up [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant FlashMoEPointwiseParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
  if (!flash_moe_pointwise_geometry(p.rows, p.width, p.selections) ||
      p.experts || threads.x != 256 || threads.y != 1 || threads.z != 1) {
    if (tid == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint column = group.x * 256 + tid;
  if (column >= p.width || group.y >= p.rows || group.z >= p.selections)
    return;
  const ulong index = (ulong(group.y) * p.selections + group.z) * p.width + column;
  const float gate_value = float(gate[index]);
  const float up_value = float(up[index]);
  const bfloat sigmoid = flash_moe_sigmoid_compiled(gate[index]);
  const bfloat silu = gate[index] * sigmoid;
  const bfloat result = silu * up[index];
  if (!metal::isfinite(gate_value) || !metal::isfinite(up_value) ||
      !metal::isfinite(float(result))) {
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
    output[index] = bfloat(flash_moe_nan());
  } else {
    output[index] = result;
  }
}

// 0 expert BF16[rows,selections,width], 1 I64 IDs, 2 BF16 route weights,
// 3 shared BF16[rows,width], 4 BF16 shared gate[rows,1], 5 BF16 output,
// 6 sticky diagnostics, 7 FlashMoEPointwiseParams.
kernel void flash_moe_combine(
    const device bfloat *expert_down [[buffer(0)]],
    const device long *expert_ids [[buffer(1)]],
    const device bfloat *route_weights [[buffer(2)]],
    const device bfloat *shared_down [[buffer(3)]],
    const device bfloat *shared_gate [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashMoEPointwiseParams &p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
  if (!flash_moe_pointwise_geometry(p.rows, p.width, p.selections) ||
      !flash_moe_routing_geometry(p.rows, p.experts, p.selections) ||
      threads.x != 256 || threads.y != 1) {
    if (tid == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint column = group.x * 256 + tid;
  if (column >= p.width || group.y >= p.rows) return;
  const ulong row = group.y;
  uint error = 0;
  // Actual MLX Metal column-small reduction for K10/W2560. Eight y partials
  // pair slots (0,8), (1,9), then 2..7; every local and merged sum is BF16.
  // Keep this canonical order for every phase, including narrow test widths.
  bfloat partials[8];
  for (uint y = 0; y < 8; ++y) {
    bfloat partial = bfloat(0.0f);
    for (uint slot = y; slot < p.selections; slot += 8) {
      const ulong route = row * p.selections + slot;
      const long expert = expert_ids[route];
      if (expert < 0 || ulong(expert) >= p.experts) error |= 1;
      for (uint previous = 0; previous < slot; ++previous)
        if (expert_ids[row * p.selections + previous] == expert) error |= 1;
      const bfloat weight = route_weights[route];
      const bfloat down = expert_down[route * p.width + column];
      const bfloat weighted = down * weight;
      if (!metal::isfinite(float(weight)) || float(weight) < 0.0f ||
          float(weight) > 1.0f || !metal::isfinite(float(down)) ||
          !metal::isfinite(float(weighted))) error |= 4;
      partial = partial + weighted;
    }
    partials[y] = partial;
  }
  bfloat routed = partials[0];
  for (uint y = 1; y < 8; ++y) routed = routed + partials[y];
  const ulong index = row * p.width + column;
  const bfloat shared = shared_down[index];
  const bfloat raw_gate = shared_gate[row];
  const bfloat scaled_shared = shared * flash_moe_sigmoid_unary(raw_gate);
  const bfloat combined = routed + scaled_shared;
  if (!metal::isfinite(float(shared)) || !metal::isfinite(float(raw_gate)) ||
      !metal::isfinite(float(routed)) ||
      !metal::isfinite(float(scaled_shared)) ||
      !metal::isfinite(float(combined))) error |= 4;
  if (error) {
    atomic_fetch_or_explicit(diagnostics, error, memory_order_relaxed);
    output[index] = bfloat(flash_moe_nan());
  } else {
    output[index] = combined;
  }
}
