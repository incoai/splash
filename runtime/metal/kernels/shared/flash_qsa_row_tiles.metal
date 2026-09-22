#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashQSARowTiles.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline void rowtile_error(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}
template <ushort M> union QueryStorage {
  bfloat operand[M * 64];
  float alpha[M];
};
// Dense-only temporal sharing. Each flattened row owns one real query/head;
// common K/V staging covers the union of their original partition windows.
// QK retains four K64 partials; PV and probabilities remain F32/BF16 mixed.
template <ushort M>
inline void rowtile_online(device const bfloat *queries,
    device const bfloat *keys, device const bfloat *values,
    device const uint *selected, device float *statistics,
    device float *numerators, device atomic_uint *diagnostics,
    constant FlashQSAFastParams &params, uint3 group, uint tid,
    threadgroup QueryStorage<M> &queryMemory,
    threadgroup bfloat *kvMemory, threadgroup float *weights) {
  const FlashQSAParams p = params.common;
  if (!flash_qsa_row_tiles_geometry(p.begin, p.rows, params.partitions) ||
      params.maximum_partitions < params.partitions ||
      p.capacity < p.begin + p.rows || group.y >= 2 || group.z >= params.partitions) {
    if (!tid) rowtile_error(diagnostics, 1u << 9);
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
      rowtile_error(diagnostics, 1u << 9);
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
      rowtile_error(diagnostics, 1u << 8);
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
      rowtile_error(diagnostics, 1u << 8);
  }
}
#define ROWTILE_ENTRY(NAME,M) \
[[max_total_threads_per_threadgroup(128)]] kernel void NAME(device const bfloat *queries [[buffer(0)]], device const bfloat *keys [[buffer(1)]], \
    device const bfloat *values [[buffer(2)]], device const uint *selected [[buffer(3)]], \
    device float *statistics [[buffer(4)]], device float *numerators [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], constant FlashQSAFastParams &params [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
  threadgroup QueryStorage<M> queryMemory; \
  threadgroup bfloat kvMemory[64*64]; threadgroup float weights[M*64]; \
  rowtile_online<M>(queries,keys,values,selected,statistics,numerators,diagnostics,params,group,tid,queryMemory,kvMemory,weights); \
}
ROWTILE_ENTRY(flash_qsa_mpp_prefill_rows_m32,32)
