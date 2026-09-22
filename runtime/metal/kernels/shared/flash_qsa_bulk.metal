// Exact fresh-2K bulk prefill. Preparation/pooling/chronological selection
// are authoritative QSA phases. Attention preserves the original 128-row
// causal windows, K64 F32 probability arithmetic and partition reduction.
// Early windows use SG4; optional temporal SG8 changes thread ownership only.
// Attention math_mode(safe) and scalar reducer math_mode(fast) are deliberate:
// changing the reducer policy changes BF16 outputs near rounding boundaries.
#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashQSAFast.h"
#include "metal/abi/FlashQSARowTiles.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline void bulk_qsa_mpp_failure(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}
inline uint bulk_qsa_mpp_visible(FlashQSAParams p, uint row) {
  const uint count = p.begin + row + 1;
  return metal::min(count / 4, 512u) * 4 + count % 4;
}
inline uint bulk_qsa_mpp_token(device const uint *selected,
                           FlashQSAParams p, uint row, uint slot) {
  const uint completed = (p.begin + row + 1) / 4;
  const uint picked = metal::min(completed, 512u);
  return slot < picked * 4
      ? selected[ulong(row) * 512 + slot / 4] * 4 + slot % 4
      : completed * 4 + slot - picked * 4;
}
inline void bulk_qsa_online_partition(
    device const bfloat *queries,
    device const bfloat *keys,
    device const bfloat *values,
    device const uint *selected,
    device float *statistics,
    device float *numerators,
    device atomic_uint *diagnostics,
    thread const FlashQSAFastParams &params,
    uint3 group,
    uint tid,
    threadgroup bfloat *aq, threadgroup bfloat *bk, threadgroup bfloat *bv,
    threadgroup float *weights, threadgroup float *maximum,
    threadgroup float *sum, threadgroup float *alpha) {
  const FlashQSAParams p = params.common;
  const uint row = group.x, kv = group.y, partition = group.z;
  if (row >= p.rows || kv >= 2 || partition >= params.partitions) return;
  const uint visible = bulk_qsa_mpp_visible(p, row);
  const uint length = (visible + params.partitions - 1) / params.partitions;
  const uint start = metal::min(partition * length, visible);
  const uint stop = metal::min(start + length, visible);
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
          const uint token = bulk_qsa_mpp_token(selected, p, row, slot);
          if (token >= p.capacity || token > p.begin + row)
            bulk_qsa_mpp_failure(diagnostics, 1u << 9);
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
          const uint token = bulk_qsa_mpp_token(selected, p, row, slot);
          if (token >= p.capacity || token > p.begin + row)
            bulk_qsa_mpp_failure(diagnostics, 1u << 9);
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
      bulk_qsa_mpp_failure(diagnostics, 1u << 8);
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
        !isfinite(out3[i])) bulk_qsa_mpp_failure(diagnostics, 1u << 8);
  }
}
inline void bulk_rowtile_error(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}
template <ushort M> union BulkQueryStorage {
  bfloat operand[M * 64];
  float alpha[M];
};
// Dense-only temporal sharing. Each flattened row owns one real query/head;
// common K/V staging covers the union of their original partition windows.
// QK retains four K64 partials; PV and probabilities remain F32/BF16 mixed.
template <ushort M>
inline void bulk_rowtile_online(device const bfloat *queries,
    device const bfloat *keys, device const bfloat *values,
    device const uint *selected, device float *statistics,
    device float *numerators, device atomic_uint *diagnostics,
    thread const FlashQSAFastParams &params, uint3 group, uint tid,
    threadgroup BulkQueryStorage<M> &queryMemory,
    threadgroup bfloat *kvMemory, threadgroup float *weights) {
  const FlashQSAParams p = params.common;
  if (!flash_qsa_row_tiles_geometry(p.begin, p.rows, params.partitions) ||
      params.maximum_partitions < params.partitions ||
      p.capacity < p.begin + p.rows || group.y >= 2 || group.z >= params.partitions) {
    if (!tid) bulk_rowtile_error(diagnostics, 1u << 9);
    return;
  }
  const uint flatBase = group.x * M, kv = group.y, partition = group.z;
  if (flatBase >= p.rows * 12) return;
  const uint firstQuery = flatBase / 12;
  const uint lastQuery = min((flatBase + M - 1) / 12, p.rows - 1);
  const uint firstCount = p.begin + firstQuery + 1, lastCount = p.begin + lastQuery + 1;
  const uint firstLength = (firstCount + params.partitions - 1) / params.partitions;
  const uint lastLength = (lastCount + params.partitions - 1) / params.partitions;
  const uint commonStart = min(partition * firstLength, firstCount);
  const uint commonStop = min(partition * lastLength + lastLength, lastCount);
  // Never silently reuse K/V if the caller's discrete selection is corrupted.
  const uint blockCount = lastCount / 4;
  for (uint i = tid; i < (lastQuery - firstQuery + 1) * blockCount; i += 128) {
    const uint row = firstQuery + i / blockCount, block = i % blockCount;
    if (block < (p.begin + row + 1) / 4 && selected[ulong(row) * 512 + block] != block)
      bulk_rowtile_error(diagnostics, 1u << 9);
  }
  float runningMaximum[M / 16], runningSum[M / 16];
#pragma unroll
  for (uint batch = 0; batch < M / 16; ++batch) {
    runningMaximum[batch] = -INFINITY;
    runningSum[batch] = 0.0f;
  }
  auto qt = tensor(queryMemory.operand, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
  auto kt = tensor(kvMemory, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto pt = tensor(weights, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
  auto vt = tensor(kvMemory, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto q = qt.template slice<64, M>(0, 0);
  auto k = kt.slice<64, 64>(0, 0);
  auto prob = pt.template slice<64, M>(0, 0);
  auto v = vt.slice<64, 64>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      M, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto out0 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out1 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out2 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out3 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i)
    if (out0.is_valid_element(i)) out0[i] = out1[i] = out2[i] = out3[i] = 0.0f;
  for (uint tokenBegin = commonStart; tokenBegin < commonStop; tokenBegin += 64) {
    auto scores = operation.template get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      if (scores.is_valid_element(i)) scores[i] = 0.0f;
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      for (uint i = tid; i < M * 64; i += 128) {
        const uint flat = flatBase + i / 64, row = flat / 12, h = kv * 12 + flat % 12;
        queryMemory.operand[i] = flat < p.rows * 12
            ? queries[(ulong(row) * 24 + h) * 256 + chunk * 64 + i % 64] : bfloat(0.0f);
      }
      for (uint i = tid; i < 64 * 64; i += 128) {
        const uint token = tokenBegin + i / 64;
        kvMemory[i] = token < commonStop
            ? keys[(ulong(token) * 2 + kv) * 256 + chunk * 64 + i % 64] : bfloat(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.template get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
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
      const uint h = index[1], token = tokenBegin + index[0], flat = flatBase + h;
      const uint count = p.begin + flat / 12 + 1;
      const uint length = (count + params.partitions - 1) / params.partitions;
      const uint start = min(partition * length, count), stop = min(start + length, count);
      weights[h * 64 + index[0]] = flat < p.rows * 12 && token >= start && token < stop
          ? scores[i] * 0.0625f : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint batch = 0; batch < M / 16; ++batch) {
      const uint h = batch * 16 + tid / 8, hLane = tid % 8, flat = flatBase + h;
      const bool real = flat < p.rows * 12;
      float tileMaximum = -INFINITY;
      for (uint slot = hLane; slot < 64; slot += 8)
        tileMaximum = max(tileMaximum, weights[h * 64 + slot]);
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 4));
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 2));
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 1));
      const float newMaximum = max(runningMaximum[batch], tileMaximum);
      const float rescale = real && isfinite(newMaximum)
          ? exp(runningMaximum[batch] - newMaximum) : 0.0f;
      float tileSum = 0.0f;
      for (uint slot = hLane; slot < 64; slot += 8) {
        const float weight = real && isfinite(newMaximum) ? exp(weights[h * 64 + slot] - newMaximum) : 0.0f;
        weights[h * 64 + slot] = weight;
        tileSum += weight;
      }
      tileSum += simd_shuffle_xor(tileSum, 4);
      tileSum += simd_shuffle_xor(tileSum, 2);
      tileSum += simd_shuffle_xor(tileSum, 1);
      runningMaximum[batch] = newMaximum;
      runningSum[batch] = runningSum[batch] * rescale + tileSum;
      // QK has finished reading the query staging bank. Reuse it for alpha;
      // max/sum remain per-head SIMD registers until the next token tile.
      if (!hLane) queryMemory.alpha[h] = rescale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      for (uint i = tid; i < 64 * 64; i += 128) {
        const uint token = tokenBegin + i % 64;
        kvMemory[i] = token < commonStop
            ? values[(ulong(token) * 2 + kv) * 256 + chunk * 64 + i / 64] : bfloat(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
      operation.run(prob, v, partial);
#pragma unroll
      for (ushort i = 0; i < out0.get_capacity(); ++i) {
        if (!out0.is_valid_element(i)) continue;
        const auto index = out0.get_multidimensional_index(i);
        const float rescale = queryMemory.alpha[index[1]];
        if (chunk == 0) out0[i] = out0[i] * rescale + partial[i];
        if (chunk == 1) out1[i] = out1[i] * rescale + partial[i];
        if (chunk == 2) out2[i] = out2[i] * rescale + partial[i];
        if (chunk == 3) out3[i] = out3[i] * rescale + partial[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
#pragma unroll
  for (uint batch = 0; batch < M / 16; ++batch) {
    const uint flat = flatBase + batch * 16 + tid / 8;
    if (tid % 8 || flat >= p.rows * 12) continue;
    const uint row = flat / 12, head = kv * 12 + flat % 12;
    const ulong index = (ulong(row) * 24 + head) * params.maximum_partitions + partition;
    statistics[index * 2] = runningMaximum[batch];
    statistics[index * 2 + 1] = runningSum[batch];
    if (!(runningSum[batch] > 0) || !isfinite(runningSum[batch]) || !isfinite(runningMaximum[batch]))
      bulk_rowtile_error(diagnostics, 1u << 8);
  }
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i) {
    if (!out0.is_valid_element(i)) continue;
    const auto coord = out0.get_multidimensional_index(i);
    const uint flat = flatBase + coord[1];
    if (flat >= p.rows * 12) continue;
    const uint row = flat / 12, head = kv * 12 + flat % 12, d = coord[0];
    const ulong index = (ulong(row) * 24 + head) * params.maximum_partitions + partition;
    numerators[index * 256 + d] = out0[i];
    numerators[index * 256 + 64 + d] = out1[i];
    numerators[index * 256 + 128 + d] = out2[i];
    numerators[index * 256 + 192 + d] = out3[i];
    if (!isfinite(out0[i]) || !isfinite(out1[i]) || !isfinite(out2[i]) || !isfinite(out3[i]))
      bulk_rowtile_error(diagnostics, 1u << 8);
  }
}

// The branch is uniform for every lane in a CTA. Sharing these banks preserves
// each original epoch and reserves only max(online,temporal) threadgroup bytes.
// Online K and V share one bank: QK scores/softmax are complete before the first
// V store; the original barriers separate every K/QK and V/PV storage epoch.
struct BulkOnlineScratch {
  bfloat aq[16 * 64], kvMemory[64 * 64];
  float weights[16 * 64];
  float maximum[16], sum[16], alpha[16];
};
struct BulkTemporalScratch {
  BulkQueryStorage<32> queryMemory;
  bfloat kvMemory[64 * 64];
  float weights[32 * 64];
};
union BulkAttentionScratch {
  BulkOnlineScratch online;
  BulkTemporalScratch temporal;
};
static_assert(sizeof(BulkOnlineScratch) == 14528, "online K/V epoch-alias shared bank extent");
static_assert(sizeof(BulkTemporalScratch) == 20480, "original M32 temporal shared bank extent");
static_assert(sizeof(BulkAttentionScratch) == 20480, "bulk shared-bank union extent");

[[max_total_threads_per_threadgroup(128)]]
kernel void flash_qsa_mpp_prefill_bulk_2048(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device float *statistics [[buffer(4)]],
    device float *numerators [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashQSAFastParams &params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (params.common.begin != 0 || params.common.rows != 2048 ||
      params.common.capacity < 2048 || params.maximum_partitions < 4 ||
      params.partitions != 4 || group.y != 0 || group.z != 0 ||
      any(groupSize != uint3(128, 1, 1))) {
    if (!tid) bulk_qsa_mpp_failure(diagnostics, 1u << 9);
    return;
  }
  const uint cta = group.x;
  if (cta >= 7936) return;
  uint window;
  uint3 localGroup;
  uint partitions;
  const bool temporal = cta >= 3328;
  if (cta < 256) {
    window = 0;
    partitions = 1;
    localGroup = uint3(cta % 128, cta / 128, 0);
  } else if (cta < 3328) {
    const uint offset = cta - 256;
    window = 1 + offset / 1024;
    const uint local = offset % 1024;
    partitions = 4;
    localGroup = uint3(local % 128, (local / 128) % 2, local / 256);
  } else {
    const uint offset = cta - 3328;
    window = 4 + offset / 384;
    const uint local = offset % 384;
    partitions = 4;
    localGroup = uint3(local % 48, (local / 48) % 2, local / 96);
  }
  FlashQSAFastParams localParams = params;
  localParams.common.begin = window * 128;
  localParams.common.rows = 128;
  localParams.partitions = partitions;
  const ulong firstRow = ulong(window) * 128;
  const ulong scratchBase = firstRow * 24 * params.maximum_partitions;
  device const bfloat *localQueries = queries + firstRow * 24 * 256;
  device const uint *localSelected = selected + firstRow * 512;
  device float *localStatistics = statistics + scratchBase * 2;
  device float *localNumerators = numerators + scratchBase * 256;
  threadgroup BulkAttentionScratch scratch;
  if (temporal) {
    bulk_rowtile_online<32>(localQueries, keys, values, localSelected,
        localStatistics, localNumerators, diagnostics, localParams, localGroup, tid,
        scratch.temporal.queryMemory, scratch.temporal.kvMemory, scratch.temporal.weights);
  } else {
    bulk_qsa_online_partition(localQueries, keys, values, localSelected,
        localStatistics, localNumerators, diagnostics, localParams, localGroup, tid,
        scratch.online.aq, scratch.online.kvMemory, scratch.online.kvMemory,
        scratch.online.weights, scratch.online.maximum, scratch.online.sum,
        scratch.online.alpha);
  }
}

inline void bulk_temporal_sg8_error(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}
template <ushort M> union BulkTemporalSG8QueryStorage {
  bfloat operand[M * 64];
  float alpha[M];
};
// Dense-only temporal sharing. Each flattened row owns one real query/head;
// common K/V staging covers the union of their original partition windows.
// QK retains four K64 partials; PV and probabilities remain F32/BF16 mixed.
template <ushort M>
inline void bulk_temporal_sg8_online(device const bfloat *queries,
    device const bfloat *keys, device const bfloat *values,
    device const uint *selected, device float *statistics,
    device float *numerators, device atomic_uint *diagnostics,
    thread const FlashQSAFastParams &params, uint3 group, uint tid,
    threadgroup BulkTemporalSG8QueryStorage<M> &queryMemory,
    threadgroup bfloat *kvMemory, threadgroup float *weights) {
  static_assert(M == 32, "Private SG8 uses one 32-head softmax pass");
  const FlashQSAParams p = params.common;
  if (!flash_qsa_row_tiles_geometry(p.begin, p.rows, params.partitions) ||
      params.maximum_partitions < params.partitions ||
      p.capacity < p.begin + p.rows || group.y >= 2 || group.z >= params.partitions) {
    if (!tid) bulk_temporal_sg8_error(diagnostics, 1u << 9);
    return;
  }
  const uint flatBase = group.x * M, kv = group.y, partition = group.z;
  if (flatBase >= p.rows * 12) return;
  const uint firstQuery = flatBase / 12;
  const uint lastQuery = min((flatBase + M - 1) / 12, p.rows - 1);
  const uint firstCount = p.begin + firstQuery + 1, lastCount = p.begin + lastQuery + 1;
  const uint firstLength = (firstCount + params.partitions - 1) / params.partitions;
  const uint lastLength = (lastCount + params.partitions - 1) / params.partitions;
  const uint commonStart = min(partition * firstLength, firstCount);
  const uint commonStop = min(partition * lastLength + lastLength, lastCount);
  // Never silently reuse K/V if the caller's discrete selection is corrupted.
  const uint blockCount = lastCount / 4;
  for (uint i = tid; i < (lastQuery - firstQuery + 1) * blockCount; i += 256) {
    const uint row = firstQuery + i / blockCount, block = i % blockCount;
    if (block < (p.begin + row + 1) / 4 && selected[ulong(row) * 512 + block] != block)
      bulk_temporal_sg8_error(diagnostics, 1u << 9);
  }
  float runningMaximum[M / 32], runningSum[M / 32];
#pragma unroll
  for (uint batch = 0; batch < M / 32; ++batch) {
    runningMaximum[batch] = -INFINITY;
    runningSum[batch] = 0.0f;
  }
  auto qt = tensor(queryMemory.operand, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
  auto kt = tensor(kvMemory, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto pt = tensor(weights, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
  auto vt = tensor(kvMemory, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto q = qt.template slice<64, M>(0, 0);
  auto k = kt.slice<64, 64>(0, 0);
  auto prob = pt.template slice<64, M>(0, 0);
  auto v = vt.slice<64, 64>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      M, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto out0 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out1 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out2 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out3 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i)
    if (out0.is_valid_element(i)) out0[i] = out1[i] = out2[i] = out3[i] = 0.0f;
  for (uint tokenBegin = commonStart; tokenBegin < commonStop; tokenBegin += 64) {
    auto scores = operation.template get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      if (scores.is_valid_element(i)) scores[i] = 0.0f;
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      for (uint i = tid; i < M * 64; i += 256) {
        const uint flat = flatBase + i / 64, row = flat / 12, h = kv * 12 + flat % 12;
        queryMemory.operand[i] = flat < p.rows * 12
            ? queries[(ulong(row) * 24 + h) * 256 + chunk * 64 + i % 64] : bfloat(0.0f);
      }
      for (uint i = tid; i < 64 * 64; i += 256) {
        const uint token = tokenBegin + i / 64;
        kvMemory[i] = token < commonStop
            ? keys[(ulong(token) * 2 + kv) * 256 + chunk * 64 + i % 64] : bfloat(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.template get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
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
      const uint h = index[1], token = tokenBegin + index[0], flat = flatBase + h;
      const uint count = p.begin + flat / 12 + 1;
      const uint length = (count + params.partitions - 1) / params.partitions;
      const uint start = min(partition * length, count), stop = min(start + length, count);
      weights[h * 64 + index[0]] = flat < p.rows * 12 && token >= start && token < stop
          ? scores[i] * 0.0625f : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint batch = 0; batch < M / 32; ++batch) {
      const uint h = batch * 32 + tid / 8, hLane = tid % 8, flat = flatBase + h;
      const bool real = flat < p.rows * 12;
      float tileMaximum = -INFINITY;
      for (uint slot = hLane; slot < 64; slot += 8)
        tileMaximum = max(tileMaximum, weights[h * 64 + slot]);
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 4));
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 2));
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 1));
      const float newMaximum = max(runningMaximum[batch], tileMaximum);
      const float rescale = real && isfinite(newMaximum)
          ? exp(runningMaximum[batch] - newMaximum) : 0.0f;
      float tileSum = 0.0f;
      for (uint slot = hLane; slot < 64; slot += 8) {
        const float weight = real && isfinite(newMaximum) ? exp(weights[h * 64 + slot] - newMaximum) : 0.0f;
        weights[h * 64 + slot] = weight;
        tileSum += weight;
      }
      tileSum += simd_shuffle_xor(tileSum, 4);
      tileSum += simd_shuffle_xor(tileSum, 2);
      tileSum += simd_shuffle_xor(tileSum, 1);
      runningMaximum[batch] = newMaximum;
      runningSum[batch] = runningSum[batch] * rescale + tileSum;
      // QK has finished reading the query staging bank. Reuse it for alpha;
      // max/sum remain per-head SIMD registers until the next token tile.
      if (!hLane) queryMemory.alpha[h] = rescale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      for (uint i = tid; i < 64 * 64; i += 256) {
        const uint token = tokenBegin + i % 64;
        kvMemory[i] = token < commonStop
            ? values[(ulong(token) * 2 + kv) * 256 + chunk * 64 + i / 64] : bfloat(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
      operation.run(prob, v, partial);
#pragma unroll
      for (ushort i = 0; i < out0.get_capacity(); ++i) {
        if (!out0.is_valid_element(i)) continue;
        const auto index = out0.get_multidimensional_index(i);
        const float rescale = queryMemory.alpha[index[1]];
        if (chunk == 0) out0[i] = out0[i] * rescale + partial[i];
        if (chunk == 1) out1[i] = out1[i] * rescale + partial[i];
        if (chunk == 2) out2[i] = out2[i] * rescale + partial[i];
        if (chunk == 3) out3[i] = out3[i] * rescale + partial[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
#pragma unroll
  for (uint batch = 0; batch < M / 32; ++batch) {
    const uint flat = flatBase + batch * 32 + tid / 8;
    if (tid % 8 || flat >= p.rows * 12) continue;
    const uint row = flat / 12, head = kv * 12 + flat % 12;
    const ulong index = (ulong(row) * 24 + head) * params.maximum_partitions + partition;
    statistics[index * 2] = runningMaximum[batch];
    statistics[index * 2 + 1] = runningSum[batch];
    if (!(runningSum[batch] > 0) || !isfinite(runningSum[batch]) || !isfinite(runningMaximum[batch]))
      bulk_temporal_sg8_error(diagnostics, 1u << 8);
  }
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i) {
    if (!out0.is_valid_element(i)) continue;
    const auto coord = out0.get_multidimensional_index(i);
    const uint flat = flatBase + coord[1];
    if (flat >= p.rows * 12) continue;
    const uint row = flat / 12, head = kv * 12 + flat % 12, d = coord[0];
    const ulong index = (ulong(row) * 24 + head) * params.maximum_partitions + partition;
    numerators[index * 256 + d] = out0[i];
    numerators[index * 256 + 64 + d] = out1[i];
    numerators[index * 256 + 128 + d] = out2[i];
    numerators[index * 256 + 192 + d] = out3[i];
    if (!isfinite(out0[i]) || !isfinite(out1[i]) || !isfinite(out2[i]) || !isfinite(out3[i]))
      bulk_temporal_sg8_error(diagnostics, 1u << 8);
  }
}


// Split only because the early SG4 and temporal SG8 routes require different
// physical threadgroup sizes. Both write disjoint windows of the SAME scratch.
[[max_total_threads_per_threadgroup(128)]]
kernel void flash_qsa_mpp_prefill_bulk_early_2048(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device float *statistics [[buffer(4)]],
    device float *numerators [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashQSAFastParams &params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (params.common.begin != 0 || params.common.rows != 2048 ||
      params.common.capacity < 2048 || params.maximum_partitions < 4 ||
      params.partitions != 4 || group.y != 0 || group.z != 0 ||
      any(groupSize != uint3(128, 1, 1))) {
    if (!tid) bulk_qsa_mpp_failure(diagnostics, 1u << 9);
    return;
  }
  const uint cta = group.x;
  if (cta >= 3328) return;
  uint window;
  uint3 localGroup;
  uint partitions;
  if (cta < 256) {
    window = 0;
    partitions = 1;
    localGroup = uint3(cta % 128, cta / 128, 0);
  } else {
    const uint offset = cta - 256;
    window = 1 + offset / 1024;
    const uint local = offset % 1024;
    partitions = 4;
    localGroup = uint3(local % 128, (local / 128) % 2, local / 256);
  }
  FlashQSAFastParams localParams = params;
  localParams.common.begin = window * 128;
  localParams.common.rows = 128;
  localParams.partitions = partitions;
  const ulong firstRow = ulong(window) * 128;
  const ulong scratchBase = firstRow * 24 * params.maximum_partitions;
  threadgroup BulkOnlineScratch scratch;
  bulk_qsa_online_partition(queries + firstRow * 24 * 256, keys, values,
      selected + firstRow * 512, statistics + scratchBase * 2,
      numerators + scratchBase * 256, diagnostics, localParams, localGroup, tid,
      scratch.aq, scratch.kvMemory, scratch.kvMemory, scratch.weights,
      scratch.maximum, scratch.sum, scratch.alpha);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void flash_qsa_mpp_prefill_bulk_temporal_sg8_2048(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device float *statistics [[buffer(4)]],
    device float *numerators [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashQSAFastParams &params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (params.common.begin != 0 || params.common.rows != 2048 ||
      params.common.capacity < 2048 || params.maximum_partitions < 4 ||
      params.partitions != 4 || group.y != 0 || group.z != 0 ||
      any(groupSize != uint3(256, 1, 1))) {
    if (!tid) bulk_temporal_sg8_error(diagnostics, 1u << 9);
    return;
  }
  const uint cta = group.x;
  if (cta >= 4608) return;
  const uint window = 4 + cta / 384;
  const uint local = cta % 384;
  const uint3 localGroup(local % 48, (local / 48) % 2, local / 96);
  FlashQSAFastParams localParams = params;
  localParams.common.begin = window * 128;
  localParams.common.rows = 128;
  localParams.partitions = 4;
  const ulong firstRow = ulong(window) * 128;
  const ulong scratchBase = firstRow * 24 * params.maximum_partitions;
  threadgroup BulkTemporalSG8QueryStorage<32> queryMemory;
  threadgroup bfloat kvMemory[64 * 64];
  threadgroup float weights[32 * 64];
  bulk_temporal_sg8_online<32>(queries + firstRow * 24 * 256, keys, values,
      selected + firstRow * 512, statistics + scratchBase * 2,
      numerators + scratchBase * 256, diagnostics, localParams, localGroup, tid,
      queryMemory, kvMemory, weights);
}

// The original scalar reducer uses contraction/reassociation off. Keep these
// pragmas after the attention helpers so their MPP source policy is unchanged.
#pragma METAL fp math_mode(fast)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline bfloat bulk_qsa_fast_gate(bfloat attention, bfloat source) {
  const bfloat exponential = bfloat(precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  const bfloat sigmoid = source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
  return attention * sigmoid;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void flash_qsa_fast_prefill_bulk_reduce_2048(
    device const float *statistics [[buffer(0)]],
    device const float *numerators [[buffer(1)]],
    device const bfloat *q_projection [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashQSAFastParams &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (params.common.begin != 0 || params.common.rows != 2048 ||
      params.common.capacity < 2048 || params.maximum_partitions < 4 ||
      params.partitions != 4 || group.z != 0 ||
      any(groupSize != uint3(256, 1, 1))) {
    if (!tid) bulk_qsa_mpp_failure(diagnostics, 1u << 9);
    return;
  }
  const uint row = group.x, head = group.y;
  if (row >= 2048 || head >= 24 || tid >= 256)
    return;
  const uint partitions = row < 128 ? 1u : 4u;
  const ulong base = (ulong(row) * 24 + head) * params.maximum_partitions;
  float maximum = -INFINITY;
  for (uint p = 0; p < partitions; ++p)
    maximum = metal::max(maximum, statistics[(base + p) * 2]);
  float sum = 0.0f, value = 0.0f;
  for (uint p = 0; p < partitions; ++p) {
    const float local_sum = statistics[(base + p) * 2 + 1];
    if (local_sum == 0.0f)
      continue;
    const float factor = metal::exp(statistics[(base + p) * 2] - maximum);
    sum += local_sum * factor;
    value += numerators[(base + p) * 256 + tid] * factor;
  }
  const bfloat attention = bfloat(value / sum);
  const bfloat gate = q_projection[ulong(row) * 12288 + head * 512 + 256 + tid];
  const bfloat gated = bulk_qsa_fast_gate(attention, gate);
  output[(ulong(row) * 24 + head) * 256 + tid] = gated;
  if (!(sum > 0.0f) || !isfinite(sum) || !isfinite(float(gated)) || !isfinite(float(gate)))
    bulk_qsa_mpp_failure(diagnostics, (1u << 8));
}

#endif
