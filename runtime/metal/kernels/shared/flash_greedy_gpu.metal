#include <metal_stdlib>
#include "metal/abi/FlashGreedyGPU.h"

using namespace metal;

inline uint flash_greedy_gpu_rank(ushort bits) {
  return !(bits & 0x7fffu) ? 0x8000u
      : (bits & 0x8000u) ? uint(ushort(~bits)) : uint(bits ^ 0x8000u);
}

inline bool flash_greedy_gpu_geometry(constant FlashGreedyGPUParams &p) {
  return p.rows && p.rows <= kFlashGreedyGPUMaximumRows &&
      p.vocabulary && p.vocabulary <= 248320 &&
      p.row_stride >= p.vocabulary &&
      p.partitions == (p.vocabulary + kFlashGreedyGPUValuesPerPartition - 1) /
          kFlashGreedyGPUValuesPerPartition && !p.reserved;
}

// This is an integer-only comparator. In particular GPU denormal handling,
// fast-math flags, and SIMD reduction ordering cannot change the chosen ID.
kernel void flash_greedy_gpu_partials(
    const device ushort *logits [[buffer(0)]],
    device FlashGreedyGPURowResult *partials [[buffer(1)]],
    constant FlashGreedyGPUParams &p [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  if (!flash_greedy_gpu_geometry(p) || group.x >= p.partitions || group.y >= p.rows)
    return;
  const uint begin = group.x * kFlashGreedyGPUValuesPerPartition;
  const uint end = min(begin + kFlashGreedyGPUValuesPerPartition, p.vocabulary);
  uint rank = 0, token = UINT_MAX, errors = 0;
  for (uint id = begin + tid; id < end; id += 256) {
    const ushort bits = logits[ulong(group.y) * p.row_stride + id];
    errors |= uint((bits & 0x7f80u) == 0x7f80u);
    const uint candidate = flash_greedy_gpu_rank(bits);
    if (candidate > rank || (candidate == rank && id < token)) {
      rank = candidate; token = id;
    }
  }
  const uint group_rank = simd_max(rank);
  const uint group_token = simd_min(rank == group_rank ? token : UINT_MAX);
  const uint group_errors = simd_or(errors);
  threadgroup uint ranks[8], tokens[8], invalids[8];
  if (!lane) {
    ranks[simd] = group_rank; tokens[simd] = group_token;
    invalids[simd] = group_errors;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!simd) {
    const uint local_rank = lane < 8 ? ranks[lane] : 0;
    const uint best_rank = simd_max(local_rank);
    const uint best_token = simd_min(local_rank == best_rank && lane < 8
        ? tokens[lane] : UINT_MAX);
    const uint all_errors = simd_or(lane < 8 ? invalids[lane] : 0);
    if (!lane)
      partials[ulong(group.y) * p.partitions + group.x] =
          FlashGreedyGPURowResult{best_token, best_rank, all_errors, 0};
  }
}

inline FlashGreedyGPURowResult flash_greedy_gpu_reduce_row(
    const device FlashGreedyGPURowResult *partials,
    constant FlashGreedyGPUParams &p, uint row, uint lane) {
  uint rank = 0, token = UINT_MAX, errors = 0;
  for (uint part = lane; part < p.partitions; part += 32) {
    const auto record = partials[ulong(row) * p.partitions + part];
    errors |= record.errors;
    if (record.rank > rank || (record.rank == rank && record.token < token)) {
      rank = record.rank; token = record.token;
    }
  }
  const uint best_rank = simd_max(rank);
  const uint best_token = simd_min(rank == best_rank ? token : UINT_MAX);
  const uint all_errors = simd_or(errors);
  return FlashGreedyGPURowResult{
      all_errors ? UINT_MAX : best_token, best_rank, all_errors, 0};
}

kernel void flash_greedy_gpu_finish(
    const device FlashGreedyGPURowResult *partials [[buffer(0)]],
    device FlashGreedyGPURowResult *results [[buffer(1)]],
    constant FlashGreedyGPUParams &p [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  if (!flash_greedy_gpu_geometry(p)) return;
  for (uint row = simd; row < p.rows; row += 8) {
    const auto result = flash_greedy_gpu_reduce_row(partials, p, row, lane);
    if (!lane) results[row] = result;
  }
}

kernel void flash_greedy_gpu_prefix(
    const device FlashGreedyGPURowResult *partials [[buffer(0)]],
    const device uint *inputs [[buffer(1)]],
    const device uint *remaining [[buffer(2)]],
    device FlashGreedyGPUPrefixResult *results [[buffer(3)]],
    constant FlashGreedyGPUParams &p [[buffer(4)]],
    uint request [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  if (!flash_greedy_gpu_geometry(p) || !p.lanes || p.lanes > 4 ||
      !p.rows_per_lane || p.rows != p.lanes * p.rows_per_lane ||
      request >= p.lanes) return;
  threadgroup FlashGreedyGPURowResult rows[kFlashGreedyGPUMaximumRows];
  for (uint row = simd; row < p.rows_per_lane; row += 8) {
    const auto result = flash_greedy_gpu_reduce_row(
        partials, p, request * p.rows_per_lane + row, lane);
    if (!lane) rows[row] = result;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid) return;
  device auto &result = results[request];
  result.matched_drafts = 0; result.retained_rows = 0;
  result.finish = kFlashGreedyGPUFinishNone; result.errors = 0;
  for (uint row = 0; row < kFlashGreedyGPUMaximumRows; ++row) {
    result.output[row] = UINT_MAX;
    result.predictions[row] = row < p.rows_per_lane ? rows[row].token : UINT_MAX;
  }
  if (!(p.active_lane_mask & (1u << request))) return;
  const uint offset = request * p.rows_per_lane;
  for (uint row = 0; row < p.rows_per_lane; ++row) {
    result.errors |= rows[row].errors;
    if (inputs[offset + row] >= p.vocabulary)
      result.errors |= kFlashGreedyGPUErrorInputToken;
  }
  const uint quota = remaining[request];
  if (!quota) result.errors |= kFlashGreedyGPUErrorBudget;
  if (result.errors) return;
  const uint drafts = p.rows_per_lane - 1;
  while (result.matched_drafts < drafts &&
      rows[result.matched_drafts].token == inputs[offset + result.matched_drafts + 1])
    ++result.matched_drafts;
  for (uint index = 0; index <= result.matched_drafts &&
      result.retained_rows < quota; ++index) {
    const uint token = index < result.matched_drafts
        ? inputs[offset + index + 1] : rows[index].token;
    result.output[result.retained_rows++] = token;
    if (token == 248044 || token == 248046) {
      result.finish = kFlashGreedyGPUFinishStop; break;
    }
    if (result.retained_rows == quota) {
      result.finish = kFlashGreedyGPUFinishLength; break;
    }
  }
}
