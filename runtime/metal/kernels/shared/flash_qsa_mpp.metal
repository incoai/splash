#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashQSAMPP.h"
#include "metal/abi/FlashQSAFast.h"

#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

constant uint qsa_mpp_width = 2051;
inline void qsa_mpp_failure(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}
inline uint qsa_mpp_visible(FlashQSAParams p, uint row) {
  const uint count = p.begin + row + 1;
  return metal::min(count / 4, 512u) * 4 + count % 4;
}
inline uint qsa_mpp_token(device const uint *selected,
                           FlashQSAParams p, uint row, uint slot) {
  const uint completed = (p.begin + row + 1) / 4;
  const uint picked = metal::min(completed, 512u);
  return slot < picked * 4
      ? selected[ulong(row) * 512 + slot / 4] * 4 + slot % 4
      : completed * 4 + slot - picked * 4;
}
inline bfloat qsa_mpp_gate(bfloat attention, bfloat source) {
  const bfloat e = bfloat(precise::exp(metal::abs(float(source))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat tail = bfloat(1.0f) / d;
  const bfloat sigmoid = source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
  return attention * sigmoid;
}

// Each query/KV-head group computes all twelve shared-key query heads in one
// padded sixteen-row MPP tile. Its selected cache rows are read once, instead
// of once per query head. No gathered K/V tensor is materialized.
template <ushort N, ushort DC>
inline void qsa_mpp_scores(
    device const bfloat *queries, device const bfloat *keys,
    device const uint *selected, device float *scores,
    device atomic_uint *diagnostics, constant FlashQSAMPPParams &params,
    uint3 group, uint tid, threadgroup bfloat *aq,
    threadgroup bfloat *bk) {
  const FlashQSAParams p = params.common;
  const uint row = group.x, kv = group.y, slotBegin = group.z * N;
  if (row >= p.rows || kv >= 2) return;
  const uint visible = qsa_mpp_visible(p, row);
  if (slotBegin >= visible) return;
  auto a = tensor(aq, dextents<int, 2>{DC, 16}, array<int, 2>{1, DC});
  auto b = tensor(bk, dextents<int, 2>{DC, N}, array<int, 2>{1, DC});
  auto at = a.template slice<DC, 16>(0, 0);
  auto bt = b.template slice<DC, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      16, N, DC, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(at), decltype(bt), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    if (accumulated.is_valid_element(i)) accumulated[i] = 0.0f;
#pragma unroll
  for (ushort chunk = 0; chunk < 256 / DC; ++chunk) {
    const uint dimension = chunk * DC;
    for (uint i = tid; i < 16u * DC; i += 128) {
      const uint h = i / DC, d = i % DC;
      aq[i] = h < 12 ? queries[(ulong(row) * 24 + kv * 12 + h) * 256 +
                                 dimension + d] : bfloat(0.0f);
    }
    for (uint i = tid; i < uint(N) * DC; i += 128) {
      const uint localSlot = i / DC, d = i % DC;
      const uint slot = slotBegin + localSlot;
      bfloat value = bfloat(0.0f);
      if (slot < visible) {
        const uint token = qsa_mpp_token(selected, p, row, slot);
        if (token >= p.capacity || token > p.begin + row) {
          qsa_mpp_failure(diagnostics, 1u << 9);
        } else {
          value = keys[(ulong(token) * 2 + kv) * 256 + dimension + d];
        }
      }
      bk[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(at), decltype(bt), float>();
    operation.run(at, bt, partial);
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i)
      if (accumulated.is_valid_element(i)) accumulated[i] += partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    if (!accumulated.is_valid_element(i)) continue;
    const auto index = accumulated.get_multidimensional_index(i);
    const uint h = index[1], slot = slotBegin + index[0];
    if (h >= 12 || slot >= visible) continue;
    const float score = accumulated[i] * 0.0625f;
    scores[(ulong(row) * 24 + kv * 12 + h) * qsa_mpp_width + slot] = score;
    if (!isfinite(score)) qsa_mpp_failure(diagnostics, 1u << 8);
  }
}
#define QSA_MPP_SCORE_ENTRY(NAME, N, DC)                                    \
  kernel void NAME(device const bfloat *queries [[buffer(0)]],             \
      device const bfloat *keys [[buffer(1)]],                              \
      device const uint *selected [[buffer(2)]],                           \
      device float *scores [[buffer(3)]],                                  \
      device atomic_uint *diagnostics [[buffer(4)]],                       \
      constant FlashQSAMPPParams &params [[buffer(5)]],                     \
      uint3 group [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_index_in_threadgroup]]) {                          \
    threadgroup bfloat aq[16 * DC], bk[N * DC];                            \
    qsa_mpp_scores<N, DC>(queries, keys, selected, scores, diagnostics,      \
                          params, group, tid, aq, bk);                    \
  }
QSA_MPP_SCORE_ENTRY(flash_qsa_mpp_scores_n64, 64, 128)
QSA_MPP_SCORE_ENTRY(flash_qsa_mpp_scores_n128, 128, 64)
#undef QSA_MPP_SCORE_ENTRY

// Global stable normalization retains the qualified eight-SIMD F32 reduction
// tree and BF16 probability boundary, while MPP owns the QK/PV dot products.
kernel void flash_qsa_mpp_probabilities(
    device const float *scores [[buffer(0)]],
    device bfloat *probabilities [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant FlashQSAMPPParams &params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.x, head = group.y;
  if (row >= params.common.rows || head >= 24) return;
  const uint count = qsa_mpp_visible(params.common, row);
  const ulong base = (ulong(row) * 24 + head) * qsa_mpp_width;
  threadgroup float partial[8];
  threadgroup float state[2];
  float maximum = -INFINITY;
  for (uint slot = tid; slot < count; slot += 256)
    maximum = metal::max(maximum, scores[base + slot]);
  maximum = simd_max(maximum);
  if (lane == 0) partial[simd] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    maximum = partial[0];
    for (uint i = 1; i < 8; ++i) maximum = metal::max(maximum, partial[i]);
    state[0] = maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  for (uint slot = tid; slot < count; slot += 256)
    sum += metal::exp(scores[base + slot] - state[0]);
  sum = simd_sum(sum);
  if (lane == 0) partial[simd] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    sum = 0.0f;
    for (uint i = 0; i < 8; ++i) sum += partial[i];
    state[1] = sum;
    if (!(sum > 0.0f) || !isfinite(sum) || !isfinite(state[0]))
      qsa_mpp_failure(diagnostics, 1u << 8);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint slot = tid; slot < count; slot += 256)
    probabilities[base + slot] = bfloat(
        metal::exp(scores[base + slot] - state[0]) / state[1]);
}

// F32 probabilities can overwrite the score sheet after both reductions
// finish, as all score reads precede the final threadgroup barrier.
kernel void flash_qsa_mpp_probabilities_f32(
    device const float *scores [[buffer(0)]],
    device float *probabilities [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant FlashQSAMPPParams &params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.x, head = group.y;
  if (row >= params.common.rows || head >= 24) return;
  const uint count = qsa_mpp_visible(params.common, row);
  const ulong base = (ulong(row) * 24 + head) * qsa_mpp_width;
  threadgroup float partial[8];
  threadgroup float state[2];
  float maximum = -INFINITY;
  for (uint slot = tid; slot < count; slot += 256)
    maximum = metal::max(maximum, scores[base + slot]);
  maximum = simd_max(maximum);
  if (lane == 0) partial[simd] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    maximum = partial[0];
    for (uint i = 1; i < 8; ++i) maximum = metal::max(maximum, partial[i]);
    state[0] = maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  for (uint slot = tid; slot < count; slot += 256)
    sum += metal::exp(scores[base + slot] - state[0]);
  sum = simd_sum(sum);
  if (lane == 0) partial[simd] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    sum = 0.0f;
    for (uint i = 0; i < 8; ++i) sum += partial[i];
    state[1] = sum;
    if (!(sum > 0.0f) || !isfinite(sum) || !isfinite(state[0]))
      qsa_mpp_failure(diagnostics, 1u << 8);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint slot = tid; slot < count; slot += 256)
    probabilities[base + slot] =
        metal::exp(scores[base + slot] - state[0]) / state[1];
}


// The sixteen query heads reuse each selected V tile. Four groups distribute
// the 256 output dimensions; this supplies rows*8 independent threadgroups.
kernel void flash_qsa_mpp_values_gate(
    device const bfloat *probabilities [[buffer(0)]],
    device const bfloat *values [[buffer(1)]],
    device const uint *selected [[buffer(2)]],
    device const bfloat *qProjection [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashQSAMPPParams &params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const FlashQSAParams p = params.common;
  const uint row = group.x, kv = group.y, dimensionBegin = group.z * 64;
  if (row >= p.rows || kv >= 2 || group.z >= 4) return;
  const uint count = qsa_mpp_visible(p, row);
  threadgroup bfloat ap[16 * 64], bv[64 * 64];
  auto a = tensor(ap, dextents<int, 2>{64, 16}, array<int, 2>{1, 64});
  // B stores output dimension-major, so the descriptor transposes B just as
  // the dense matrix route does. Selected token is the reduction dimension.
  auto b = tensor(bv, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto at = a.slice<64, 16>(0, 0);
  auto bt = b.slice<64, 64>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      16, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto accumulated = operation.get_destination_cooperative_tensor<
      decltype(at), decltype(bt), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    if (accumulated.is_valid_element(i)) accumulated[i] = 0.0f;
  for (uint start = 0; start < count; start += 64) {
    for (uint i = tid; i < 16 * 64; i += 128) {
      const uint h = i / 64, slot = start + i % 64;
      ap[i] = h < 12 && slot < count
          ? probabilities[(ulong(row) * 24 + kv * 12 + h) * qsa_mpp_width + slot]
          : bfloat(0.0f);
    }
    for (uint i = tid; i < 64 * 64; i += 128) {
      const uint d = i / 64, slot = start + i % 64;
      bfloat value = bfloat(0.0f);
      if (slot < count) {
        const uint token = qsa_mpp_token(selected, p, row, slot);
        if (token >= p.capacity || token > p.begin + row)
          qsa_mpp_failure(diagnostics, 1u << 9);
        else
          value = values[(ulong(token) * 2 + kv) * 256 + dimensionBegin + d];
      }
      bv[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto partial = operation.get_destination_cooperative_tensor<
        decltype(at), decltype(bt), float>();
    operation.run(at, bt, partial);
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i)
      if (accumulated.is_valid_element(i)) accumulated[i] += partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    if (!accumulated.is_valid_element(i)) continue;
    const auto index = accumulated.get_multidimensional_index(i);
    const uint h = index[1], d = dimensionBegin + index[0];
    if (h >= 12) continue;
    const uint head = kv * 12 + h;
    const bfloat attention = bfloat(accumulated[i]);
    const bfloat gate = qProjection[ulong(row) * 12288 + head * 512 + 256 + d];
    const bfloat result = qsa_mpp_gate(attention, gate);
    output[(ulong(row) * 24 + head) * 256 + d] = result;
    if (!isfinite(accumulated[i]) || !isfinite(float(result)) ||
        !isfinite(float(gate))) qsa_mpp_failure(diagnostics, 1u << 8);
  }
}
kernel void flash_qsa_mpp_values_f32_gate(
    device const float *probabilities [[buffer(0)]],
    device const bfloat *values [[buffer(1)]],
    device const uint *selected [[buffer(2)]],
    device const bfloat *qProjection [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashQSAMPPParams &params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const FlashQSAParams p = params.common;
  const uint row = group.x, kv = group.y, dimensionBegin = group.z * 64;
  if (row >= p.rows || kv >= 2 || group.z >= 4) return;
  const uint count = qsa_mpp_visible(p, row);
  threadgroup float ap[16 * 64];
  threadgroup bfloat bv[64 * 64];
  auto a = tensor(ap, dextents<int, 2>{64, 16}, array<int, 2>{1, 64});
  // B stores output dimension-major, so the descriptor transposes B just as
  // the dense matrix route does. Selected token is the reduction dimension.
  auto b = tensor(bv, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto at = a.slice<64, 16>(0, 0);
  auto bt = b.slice<64, 64>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      16, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto accumulated = operation.get_destination_cooperative_tensor<
      decltype(at), decltype(bt), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    if (accumulated.is_valid_element(i)) accumulated[i] = 0.0f;
  for (uint start = 0; start < count; start += 64) {
    for (uint i = tid; i < 16 * 64; i += 128) {
      const uint h = i / 64, slot = start + i % 64;
      ap[i] = h < 12 && slot < count
          ? probabilities[(ulong(row) * 24 + kv * 12 + h) * qsa_mpp_width + slot]
          : bfloat(0.0f);
    }
    for (uint i = tid; i < 64 * 64; i += 128) {
      const uint d = i / 64, slot = start + i % 64;
      bfloat value = bfloat(0.0f);
      if (slot < count) {
        const uint token = qsa_mpp_token(selected, p, row, slot);
        if (token >= p.capacity || token > p.begin + row)
          qsa_mpp_failure(diagnostics, 1u << 9);
        else
          value = values[(ulong(token) * 2 + kv) * 256 + dimensionBegin + d];
      }
      bv[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto partial = operation.get_destination_cooperative_tensor<
        decltype(at), decltype(bt), float>();
    operation.run(at, bt, partial);
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i)
      if (accumulated.is_valid_element(i)) accumulated[i] += partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    if (!accumulated.is_valid_element(i)) continue;
    const auto index = accumulated.get_multidimensional_index(i);
    const uint h = index[1], d = dimensionBegin + index[0];
    if (h >= 12) continue;
    const uint head = kv * 12 + h;
    const bfloat attention = bfloat(accumulated[i]);
    const bfloat gate = qProjection[ulong(row) * 12288 + head * 512 + 256 + d];
    const bfloat result = qsa_mpp_gate(attention, gate);
    output[(ulong(row) * 24 + head) * 256 + d] = result;
    if (!isfinite(accumulated[i]) || !isfinite(float(result)) ||
        !isfinite(float(gate))) qsa_mpp_failure(diagnostics, 1u << 8);
  }
}

// Fused online selected-block GQA. One group owns twelve query heads sharing
// a KV head and a token partition. F32 softmax weights remain in threadgroup
// memory and feed mixed F32/BF16 MPP PV; only final partition results leave it.
kernel void flash_qsa_mpp_online_partition(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device float *statistics [[buffer(4)]],
    device float *numerators [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashQSAFastParams &params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const FlashQSAParams p = params.common;
  const uint row = group.x, kv = group.y, partition = group.z;
  if (row >= p.rows || kv >= 2 || partition >= params.partitions) return;
  const uint visible = qsa_mpp_visible(p, row);
  const uint length = (visible + params.partitions - 1) / params.partitions;
  const uint start = metal::min(partition * length, visible);
  const uint stop = metal::min(start + length, visible);
  threadgroup bfloat aq[16 * 64], bk[64 * 64], bv[64 * 64];
  threadgroup float weights[16 * 64];
  threadgroup float maximum[16], sum[16], alpha[16];
  if (tid < 16) {
    maximum[tid] = -INFINITY;
    sum[tid] = 0.0f;
  }
  auto qt = tensor(aq, dextents<int, 2>{64, 16}, array<int, 2>{1, 64});
  auto kt = tensor(bk, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto pt = tensor(weights, dextents<int, 2>{64, 16}, array<int, 2>{1, 64});
  auto vt = tensor(bv, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto q = qt.slice<64, 16>(0, 0);
  auto k = kt.slice<64, 64>(0, 0);
  auto prob = pt.slice<64, 16>(0, 0);
  auto v = vt.slice<64, 64>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      16, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto out0 = operation.get_destination_cooperative_tensor<
      decltype(prob), decltype(v), float>();
  auto out1 = operation.get_destination_cooperative_tensor<
      decltype(prob), decltype(v), float>();
  auto out2 = operation.get_destination_cooperative_tensor<
      decltype(prob), decltype(v), float>();
  auto out3 = operation.get_destination_cooperative_tensor<
      decltype(prob), decltype(v), float>();
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i)
    if (out0.is_valid_element(i))
      out0[i] = out1[i] = out2[i] = out3[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint tokenBegin = start; tokenBegin < stop; tokenBegin += 64) {
    auto scores = operation.get_destination_cooperative_tensor<
        decltype(q), decltype(k), float>();
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      if (scores.is_valid_element(i)) scores[i] = 0.0f;
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      const uint dimension = chunk * 64;
      for (uint i = tid; i < 16 * 64; i += 128) {
        const uint h = i / 64, d = i % 64;
        aq[i] = h < 12
            ? queries[(ulong(row) * 24 + kv * 12 + h) * 256 + dimension + d]
            : bfloat(0.0f);
      }
      for (uint i = tid; i < 64 * 64; i += 128) {
        const uint slot = tokenBegin + i / 64, d = i % 64;
        bfloat value = bfloat(0.0f);
        if (slot < stop) {
          const uint token = qsa_mpp_token(selected, p, row, slot);
          if (token >= p.capacity || token > p.begin + row)
            qsa_mpp_failure(diagnostics, 1u << 9);
          else
            value = keys[(ulong(token) * 2 + kv) * 256 + dimension + d];
        }
        bk[i] = value;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.get_destination_cooperative_tensor<
          decltype(q), decltype(k), float>();
      operation.run(q, k, partial);
#pragma unroll
      for (ushort i = 0; i < scores.get_capacity(); ++i)
        if (scores.is_valid_element(i)) scores[i] += partial[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i) {
      if (!scores.is_valid_element(i)) continue;
      const auto index = scores.get_multidimensional_index(i);
      const uint h = index[1], slot = index[0];
      weights[h * 64 + slot] = h < 12 && tokenBegin + slot < stop
          ? scores[i] * 0.0625f : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Eight SIMD lanes share one head. XOR reductions stay inside each
    // eight-lane sub-group; no full-SIMD head cross-contamination occurs.
    const uint h = tid / 8, hLane = tid % 8;
    float tileMaximum = -INFINITY;
    for (uint slot = hLane; slot < 64; slot += 8)
      tileMaximum = metal::max(tileMaximum, weights[h * 64 + slot]);
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 4));
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 2));
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 1));
    const float newMaximum = metal::max(maximum[h], tileMaximum);
    const float rescale = h < 12 ? metal::exp(maximum[h] - newMaximum) : 0.0f;
    float tileSum = 0.0f;
    for (uint slot = hLane; slot < 64; slot += 8) {
      const float weight = h < 12
          ? metal::exp(weights[h * 64 + slot] - newMaximum) : 0.0f;
      weights[h * 64 + slot] = weight;
      tileSum += weight;
    }
    tileSum += simd_shuffle_xor(tileSum, 4);
    tileSum += simd_shuffle_xor(tileSum, 2);
    tileSum += simd_shuffle_xor(tileSum, 1);
    if (hLane == 0) {
      alpha[h] = rescale;
      maximum[h] = newMaximum;
      sum[h] = sum[h] * rescale + tileSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint dimensionChunk = 0; dimensionChunk < 4; ++dimensionChunk) {
      const uint dimension = dimensionChunk * 64;
      for (uint i = tid; i < 64 * 64; i += 128) {
        const uint d = i / 64, slot = tokenBegin + i % 64;
        bfloat value = bfloat(0.0f);
        if (slot < stop) {
          const uint token = qsa_mpp_token(selected, p, row, slot);
          if (token >= p.capacity || token > p.begin + row)
            qsa_mpp_failure(diagnostics, 1u << 9);
          else
            value = values[(ulong(token) * 2 + kv) * 256 + dimension + d];
        }
        bv[i] = value;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.get_destination_cooperative_tensor<
          decltype(prob), decltype(v), float>();
      operation.run(prob, v, partial);
#pragma unroll
      for (ushort i = 0; i < out0.get_capacity(); ++i) {
        if (!out0.is_valid_element(i)) continue;
        const auto index = out0.get_multidimensional_index(i);
        const float scale = alpha[index[1]];
        if (dimensionChunk == 0) out0[i] = out0[i] * scale + partial[i];
        if (dimensionChunk == 1) out1[i] = out1[i] * scale + partial[i];
        if (dimensionChunk == 2) out2[i] = out2[i] * scale + partial[i];
        if (dimensionChunk == 3) out3[i] = out3[i] * scale + partial[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (tid < 12) {
    const ulong index = (ulong(row) * 24 + kv * 12 + tid) *
                       params.maximum_partitions + partition;
    statistics[index * 2] = maximum[tid];
    statistics[index * 2 + 1] = sum[tid];
    if (stop > start && (!(sum[tid] > 0.0f) || !isfinite(sum[tid]) ||
                         !isfinite(maximum[tid])))
      qsa_mpp_failure(diagnostics, 1u << 8);
  }
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i) {
    if (!out0.is_valid_element(i)) continue;
    const auto coord = out0.get_multidimensional_index(i);
    const uint h = coord[1], d = coord[0];
    if (h >= 12) continue;
    const ulong index = (ulong(row) * 24 + kv * 12 + h) *
                       params.maximum_partitions + partition;
    numerators[index * 256 + d] = out0[i];
    numerators[index * 256 + 64 + d] = out1[i];
    numerators[index * 256 + 128 + d] = out2[i];
    numerators[index * 256 + 192 + d] = out3[i];
    if (!isfinite(out0[i]) || !isfinite(out1[i]) || !isfinite(out2[i]) ||
        !isfinite(out3[i])) qsa_mpp_failure(diagnostics, 1u << 8);
  }
}
#endif
