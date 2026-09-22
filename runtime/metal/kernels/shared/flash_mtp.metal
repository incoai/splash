#include <metal_stdlib>
#include "metal/abi/FlashMTP.h"

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// The trained fuse is a BF16 add of the projected next-token embedding to
// each independently projected, globally normalized previous HC stream.
kernel void flash_mtp_fuse_inputs(
    device const bfloat *projected_embedding [[buffer(0)]],
    device const bfloat *projected_hidden [[buffer(1)]],
    device bfloat *hyper [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant FlashMTPFuseParams &p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (!p.rows || !p.width || p.streams != 4) {
    if (!index)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const ulong count = ulong(p.rows) * p.streams * p.width;
  if (ulong(index) >= count) return;
  const ulong row = ulong(index) / (ulong(p.streams) * p.width);
  const uint column = index % p.width;
  const bfloat value = bfloat(float(projected_hidden[index]) +
      float(projected_embedding[row * p.width + column]));
  hyper[index] = value;
  if (!isfinite(float(value)))
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
}
