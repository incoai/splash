#if __METAL_VERSION__ >= 400
#include "metal/kernels/common/flash_moe_direct_a_common.h"

#define FLASH_DIRECT_A_Q4X8_GATE(NAME, M, SG)                                        \
kernel void NAME(                                                         \
    device bfloat *input [[buffer(0)]],                              \
    device const uchar *gw [[buffer(1)]], device const uchar *gs [[buffer(2)]], \
    device const uchar *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]], \
    device const uchar *us [[buffer(5)]], device const uchar *ub [[buffer(6)]], \
    device const uint *offsets [[buffer(7)]],                              \
    device const FlashMoEBucketJob *jobs [[buffer(8)]],                     \
    device const uint *job_count [[buffer(9)]],                            \
    device bfloat *output [[buffer(10)]], device uint *diag [[buffer(11)]], \
    constant FlashMoEBlockedGateParams &p [[buffer(12)]],                   \
    uint3 group [[threadgroup_position_in_grid]],                         \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  alignas(16) threadgroup bfloat g[64 * 64], u[64 * 64];         \
  flash_direct_a_q4x8_gate_up_tile<M, SG>(input, gw, gs, gb, uw, us, ub, offsets, jobs, job_count, \
      output, diag, p, group, threads, tid, g, u);                       \
}
FLASH_DIRECT_A_Q4X8_GATE(flash_moe_direct_a_gate_up_m8_n64, 8, 4)
FLASH_DIRECT_A_Q4X8_GATE(flash_moe_direct_a_gate_up_m16_n64, 16, 4)
FLASH_DIRECT_A_Q4X8_GATE(flash_moe_direct_a_gate_up_m32_n64, 32, 4)
FLASH_DIRECT_A_Q4X8_GATE(flash_moe_direct_a_gate_up_m64_n64_sg8, 64, 8)
#undef FLASH_DIRECT_A_Q4X8_GATE

#define FLASH_DIRECT_A_Q4X8_DOWN(NAME, M, SG)                                        \
kernel void NAME(                                                         \
    device bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]], \
    device const uchar *s [[buffer(2)]], device const uchar *b [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]],                              \
    device const FlashMoEBucketJob *jobs [[buffer(5)]],                     \
    device const uint *job_count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *output [[buffer(8)]], device uint *diag [[buffer(9)]],    \
    constant FlashMoEBlockedDownParams &p [[buffer(10)]],                   \
    uint3 group [[threadgroup_position_in_grid]],                         \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  alignas(16) threadgroup bfloat bstage[64 * 64];               \
  flash_direct_a_q4x8_down_tile<M, SG>(input, w, s, b, offsets, jobs, job_count, map, output, diag, \
      p, group, threads, tid, bstage);                                  \
}
FLASH_DIRECT_A_Q4X8_DOWN(flash_moe_direct_a_down_scatter_m8_n64, 8, 4)
FLASH_DIRECT_A_Q4X8_DOWN(flash_moe_direct_a_down_scatter_m16_n64, 16, 4)
FLASH_DIRECT_A_Q4X8_DOWN(flash_moe_direct_a_down_scatter_m32_n64, 32, 4)
FLASH_DIRECT_A_Q4X8_DOWN(flash_moe_direct_a_down_scatter_m64_n64_sg8, 64, 8)
#undef FLASH_DIRECT_A_Q4X8_DOWN


// Same producer extent and source-row sticky validation as bucket_pack,
// with finite sanitization moved once to the packed device copy. Guard rows
// are initialized on the GPU on every replay, including Private storage.
kernel void flash_moe_direct_a_pack(
    device const ushort *input [[buffer(0)]], device const uint *offsets [[buffer(1)]],
    device uint *route_map [[buffer(2)]], device ushort *packed [[buffer(3)]],
    device atomic_uint *diag [[buffer(4)]], constant FlashMoEBucketParams &p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > 10 || p.width != 2560 || p.experts != 512 ||
      p.routes != p.rows * p.selections || p.tile_rows || p.job_capacity || p.reserved ||
      threads != 256 || row >= p.routes + kFlashMoEDirectAPaddingRows) {
    if (!tid) atomic_fetch_or_explicit(diag, 2u, memory_order_relaxed); return;
  }
  if (row >= p.routes) {
    for (uint k = tid; k < 2560; k += 256) packed[ulong(row) * 2560 + k] = 0;
    return;
  }
  const uint total = offsets[512];
  bool valid = row < total && total <= p.routes;
  uint route = valid ? route_map[row] : UINT_MAX;
  if (valid && route >= p.routes) {
    valid = false;
    if (!tid) atomic_fetch_or_explicit(diag, 2u, memory_order_relaxed);
  }
  if (!tid && (!valid || total > p.routes)) {
    route_map[row] = UINT_MAX;
    if (total > p.routes) atomic_fetch_or_explicit(diag, 2u, memory_order_relaxed);
  }
  const uint source_row = valid ? route / p.selections : 0;
  bool invalid = false;
  for (uint k = tid; k < 2560; k += 256) {
    ushort value = valid ? input[ulong(source_row) * 2560 + k] : 0;
    // This substitutes exactly the old gate/up staged-A nonfinite zero.
    if ((value & 0x7f80u) == 0x7f80u) { value = 0; invalid = true; }
    packed[ulong(row) * 2560 + k] = value;
    if (row < p.rows && (input[ulong(row) * 2560 + k] & 0x7f80u) == 0x7f80u)
      invalid = true;
  }
  if (simd_any(invalid) && !lane)
    atomic_fetch_or_explicit(diag, 4u, memory_order_relaxed);
}

// Down staging formerly sanitized every valid input independently for each
// output tile. Do it once into a padded device operand; ignore dead/dummy rows.
kernel void flash_moe_direct_a_prepare_down(
    device const ushort *input [[buffer(0)]], device const uint *offsets [[buffer(1)]],
    device ushort *output [[buffer(2)]], device atomic_uint *diag [[buffer(3)]],
    constant FlashMoEDirectAPrepareParams &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (!p.routes || p.routes > kFlashMoEBucketMaximumRows * 10 || p.width != 640 ||
      p.padding_rows != kFlashMoEDirectAPaddingRows || p.reserved || threads != 256 ||
      row >= p.routes + p.padding_rows || offsets[512] > p.routes) {
    if (!tid) atomic_fetch_or_explicit(diag, 2u, memory_order_relaxed); return;
  }
  const bool valid = row < offsets[512];
  bool invalid = false;
  for (uint k = tid; k < 640; k += 256) {
    ushort value = valid ? input[ulong(row) * 640 + k] : 0;
    if ((value & 0x7f80u) == 0x7f80u) { value = 0; invalid = true; }
    output[ulong(row) * 640 + k] = value;
  }
  if (simd_any(invalid) && !lane)
    atomic_fetch_or_explicit(diag, 4u, memory_order_relaxed);
}

#endif
