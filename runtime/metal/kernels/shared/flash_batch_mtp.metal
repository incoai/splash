#include <metal_stdlib>
#include "metal/abi/FlashBatchMTP.h"

using namespace metal;

// Select one real final mixed feature row per lane before the shared head.
// Offsets describe compact real spans; no model token or cache is padded.
kernel void flash_batch_mtp_gather_last(
    device const bfloat *mixed [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device bfloat *head_input [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant FlashBatchMTPGatherParams &p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (!p.lanes || p.lanes > 4 || p.width != 2560 || !p.rows || p.rows > 512) {
    if (!index) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (ulong(index) >= ulong(p.lanes) * p.width) return;
  const uint lane = index / p.width, column = index % p.width;
  const uint begin = offsets[lane], end = offsets[lane + 1];
  if (begin >= end || end > p.rows || end - begin > 128) {
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    head_input[index] = bfloat(0.0f);
    return;
  }
  head_input[index] = mixed[ulong(end - 1) * p.width + column];
}
