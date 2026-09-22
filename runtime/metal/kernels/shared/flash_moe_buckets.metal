#include <metal_stdlib>
#include "metal/abi/FlashMoEBuckets.h"

using namespace metal;

inline bool flash_moe_bucket_base_geometry(constant FlashMoEBucketParams &p,
                                            uint threads) {
  return p.rows && p.rows <= kFlashMoEBucketMaximumRows && p.selections &&
      p.selections <= kFlashMoEBucketMaximumSelections &&
      p.width == 2560 && p.experts == 512 &&
      p.routes == p.rows * p.selections && !p.reserved && threads == 256;
}

inline bool flash_moe_bucket_pack_geometry(constant FlashMoEBucketParams &p,
                                            uint threads) {
  return flash_moe_bucket_base_geometry(p, threads) &&
      !p.tile_rows && !p.job_capacity;
}

inline bool flash_moe_bucket_job_geometry(constant FlashMoEBucketParams &p,
                                           uint threads) {
  if (!flash_moe_bucket_base_geometry(p, threads) ||
      (p.tile_rows != 8 && p.tile_rows != 16 && p.tile_rows != 32 &&
       p.tile_rows != 64)) return false;
  return p.job_capacity == (p.routes + p.tile_rows - 1) / p.tile_rows + 511;
}

// Each expert group scans the fixed logical route count. Only expert 0 checks
// invalid IDs and duplicate selections, avoiding repeated diagnostic scans.
// 0 IDs I64[R,S], 1 counts U32[512], 2 inverse U32[R*S],
// 3 sticky diagnostics, 4 parameters.
kernel void flash_moe_bucket_histogram(
    const device long *ids [[buffer(0)]],
    device uint *counts [[buffer(1)]],
    device uint *canonical_to_packed [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant FlashMoEBucketParams &p [[buffer(4)]],
    uint expert [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  if (!flash_moe_bucket_pack_geometry(p, threads)) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (expert >= p.experts) return;
  uint total = 0;
  bool invalid = false;
  for (uint route = tid; route < p.routes; route += 256) {
    const long id = ids[route];
    total += uint(id == long(expert));
    if (expert == 0) {
      canonical_to_packed[route] = UINT_MAX;
      if (id < 0 || id >= long(p.experts)) invalid = true;
      else {
        const uint slot = route % p.selections;
        const uint row_begin = route - slot;
        for (uint previous = 0; previous < slot; ++previous)
          if (ids[row_begin + previous] == id) invalid = true;
      }
    }
  }
  const uint group_total = simd_sum(total);
  const uint group_invalid = uint(simd_any(invalid));
  threadgroup uint totals[8];
  threadgroup uint invalids[8];
  if (!lane) {
    totals[simd] = group_total;
    invalids[simd] = group_invalid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    const uint combined = simd_sum(lane < 8 ? totals[lane] : 0u);
    const uint combined_invalid = simd_or(lane < 8 ? invalids[lane] : 0u);
    if (!lane) {
      counts[expert] = combined;
      if (combined_invalid)
        atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
    }
  }
}

// Integer exclusive prefix is small and deterministic. Saturation protects
// every downstream access if a malformed count buffer is supplied directly.
// 0 counts U32[512], 1 offsets U32[513], 2 diagnostics, 3 parameters.
kernel void flash_moe_bucket_prefix(
    const device uint *counts [[buffer(0)]],
    device uint *offsets [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant FlashMoEBucketParams &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if (!flash_moe_bucket_pack_geometry(p, threads)) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (tid) return;
  uint total = 0;
  offsets[0] = 0;
  for (uint expert = 0; expert < p.experts; ++expert) {
    const uint count = counts[expert];
    if (count > p.routes - total) {
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
      // No valid downstream map is possible. Empty every expert range.
      for (uint index = 0; index <= p.experts; ++index) offsets[index] = 0;
      return;
    }
    total += count;
    offsets[expert + 1] = total;
  }
}

// One group per nonempty expert builds a stable map in ascending flat-route
// order. SIMD scans process consecutive 256-route tiles; there are no atomics
// or scheduler-dependent slot assignments. 0 IDs, 1 counts, 2 offsets,
// 3 routeMap U32[R*S], 4 inverse U32[R*S], 5 diagnostics, 6 parameters.
kernel void flash_moe_bucket_stable_map(
    const device long *ids [[buffer(0)]],
    const device uint *counts [[buffer(1)]],
    const device uint *offsets [[buffer(2)]],
    device uint *route_map [[buffer(3)]],
    device uint *canonical_to_packed [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashMoEBucketParams &p [[buffer(6)]],
    uint expert [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  if (!flash_moe_bucket_pack_geometry(p, threads)) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (expert >= p.experts) return;
  const uint count = counts[expert];
  if (!count) return;
  const uint begin = offsets[expert];
  const uint end = offsets[expert + 1];
  if (begin > p.routes || end > p.routes || end < begin || end - begin != count) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  threadgroup uint group_totals[8];
  threadgroup uint group_prefixes[8];
  threadgroup uint tile_total;
  uint previous_total = 0;
  for (uint base = 0; base < p.routes; base += 256) {
    const uint route = base + tid;
    const uint match = uint(route < p.routes && ids[route] == long(expert));
    const uint lane_prefix = simd_prefix_exclusive_sum(match);
    const uint group_total = simd_sum(match);
    if (!lane) group_totals[simd] = group_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
      const uint local_total = lane < 8 ? group_totals[lane] : 0u;
      const uint group_prefix = simd_prefix_exclusive_sum(local_total);
      const uint all_total = simd_sum(local_total);
      if (lane < 8) group_prefixes[lane] = group_prefix;
      if (!lane) tile_total = all_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (match) {
      const uint rank = previous_total + group_prefixes[simd] + lane_prefix;
      if (rank < count) {
        route_map[begin + rank] = route;
        canonical_to_packed[route] = begin + rank;
      }
      else atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    }
    previous_total += tile_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!tid && previous_total != count)
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
}

// Preserve BF16 bits exactly; activation quantization is deliberately absent.
// Also inspect each original hidden row once, so nonfinite values are flagged
// even when that row's IDs are all invalid. 0 BF16[R,2560] input, 1 offsets,
// 2 routeMap (valid tail is filled here), 3 packed BF16[R*S,2560], 4 diagnostic,
// 5 parameters. One 256-thread group per packed row.
kernel void flash_moe_bucket_pack(
    const device ushort *input [[buffer(0)]],
    const device uint *offsets [[buffer(1)]],
    device uint *route_map [[buffer(2)]],
    device ushort *packed [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashMoEBucketParams &p [[buffer(5)]],
    uint packed_row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (!flash_moe_bucket_pack_geometry(p, threads)) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (packed_row >= p.routes) return;
  const uint total = offsets[p.experts];
  bool valid = packed_row < total && total <= p.routes;
  uint route = valid ? route_map[packed_row] : UINT_MAX;
  if (valid && route >= p.routes) {
    valid = false;
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
  }
  if (!tid && (!valid || total > p.routes)) {
    route_map[packed_row] = UINT_MAX;
    if (total > p.routes)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
  }
  const uint source_row = valid ? route / p.selections : 0u;
  bool invalid = false;
  for (uint column = tid; column < p.width; column += 256) {
    const ushort value = valid ? input[ulong(source_row) * p.width + column] : 0;
    packed[ulong(packed_row) * p.width + column] = value;
    if (packed_row < p.rows) {
      const ushort source = input[ulong(packed_row) * p.width + column];
      if ((source & 0x7f80u) == 0x7f80u) invalid = true;
    }
  }
  if (simd_any(invalid) && !lane)
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
}

// 0 counts, 1 offsets, 2 jobOffsets U32[513], 3 jobCount U32[1],
// 4 diagnostics, 5 parameters. No CPU readback of any dynamic count.
kernel void flash_moe_bucket_job_prefix(
    const device uint *counts [[buffer(0)]],
    const device uint *offsets [[buffer(1)]],
    device uint *job_offsets [[buffer(2)]],
    device uint *job_count [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashMoEBucketParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if (!flash_moe_bucket_job_geometry(p, threads)) {
    if (!tid) {
      job_count[0] = 0;
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    }
    return;
  }
  if (tid) return;
  uint total = 0;
  uint routes = 0;
  job_count[0] = 0;
  job_offsets[0] = 0;
  bool invalid = offsets[0] != 0;
  for (uint expert = 0; expert < p.experts; ++expert) {
    const uint count = counts[expert];
    const uint begin = offsets[expert];
    const uint end = offsets[expert + 1];
    if (begin != routes || begin > p.routes || end > p.routes ||
        end < begin || end - begin != count) invalid = true;
    // Never perform potentially overflowing count+M-1 for malformed buffers.
    const uint expert_jobs = count <= p.routes
        ? (count + p.tile_rows - 1) / p.tile_rows : p.job_capacity + 1;
    if (expert_jobs > p.job_capacity - total) invalid = true;
    if (invalid) {
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
      for (uint index = 0; index <= p.experts; ++index) job_offsets[index] = 0;
      return;
    }
    total += expert_jobs;
    routes = end;
    job_offsets[expert + 1] = total;
  }
  job_count[0] = total;
}

// Fixed-capacity GPU emission. An upper-bound search skips repeated offsets
// from empty experts, then maps each active job to its expert/packed row pair.
// 0 offsets, 1 jobOffsets, 2 jobCount, 3 jobs pairs, 4 diag, 5 params.
kernel void flash_moe_bucket_jobs(
    const device uint *offsets [[buffer(0)]],
    const device uint *job_offsets [[buffer(1)]],
    const device uint *job_count [[buffer(2)]],
    device FlashMoEBucketJob *jobs [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashMoEBucketParams &p [[buffer(5)]],
    uint index [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if (!flash_moe_bucket_job_geometry(p, threads)) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (index >= p.job_capacity) return;
  const uint count = job_count[0];
  if (count > p.job_capacity || index >= count) {
    jobs[index] = {UINT_MAX, 0};
    if (count > p.job_capacity && !tid)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  uint lower = 0;
  uint upper = p.experts;
  while (lower < upper) {
    const uint middle = (lower + upper) / 2;
    if (job_offsets[middle] <= index) lower = middle + 1;
    else upper = middle;
  }
  const uint expert = lower ? lower - 1 : UINT_MAX;
  if (expert >= p.experts) {
    jobs[index] = {UINT_MAX, 0};
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint first_job = job_offsets[expert];
  const uint end_job = job_offsets[expert + 1];
  const uint begin = offsets[expert];
  const uint end = offsets[expert + 1];
  if (first_job > index || end_job <= index || end_job > count ||
      begin > p.routes || end > p.routes || end <= begin ||
      index - first_job >= (end - begin + p.tile_rows - 1) / p.tile_rows) {
    jobs[index] = {UINT_MAX, 0};
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  jobs[index] = {expert, begin + (index - first_job) * p.tile_rows};
}
