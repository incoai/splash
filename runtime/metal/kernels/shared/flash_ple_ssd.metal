#include "metal/abi/FlashPLESSD.h"
#include <metal_stdlib>

using namespace metal;

// These are the original gather8 arithmetic boundaries, including an explicit
// BF16 affine result before multiplication by the checkpoint BF16 scale.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

kernel void flash_ple_ssd_gather(
    device const long *gpu_ids [[buffer(0)]],
    device const long *expected_ids [[buffer(1)]],
    device const uchar *raw_rows [[buffer(2)]],
    device const bfloat *shared_scale [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashPLESSDParams &p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint column [[thread_index_in_threadgroup]]) {
  if (!p.rows || p.heads != 16 || p.head_width != 160 ||
      p.row_stride_bytes != 100 || !p.table_rows ||
      p.table_rows > 0x7ffffffffffffffful) {
    if (column == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (group.y >= p.rows || group.z >= p.heads || column >= p.head_width)
    return;
  const ulong selection = ulong(group.y) * p.heads + group.z;
  const long id = gpu_ids[selection];
  if (id < 0 || ulong(id) >= p.table_rows || id != expected_ids[selection]) {
    if (column == 0)
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
    output[selection * p.head_width + column] =
        bfloat(as_type<float>(0x7fc00000u));
    return;
  }
  const device uchar *row = raw_rows + selection * p.row_stride_bytes;
  const device bfloat *scales = reinterpret_cast<const device bfloat *>(row + 80);
  const device bfloat *biases = reinterpret_cast<const device bfloat *>(row + 90);
  const uint packed = uint(row[column / 2]);
  const uint code = (packed >> ((column % 2) * 4)) & 15;
  const float product = float(code) * float(scales[column / 32]);
  const bfloat affine = bfloat(product + float(biases[column / 32]));
  const bfloat value = bfloat(float(affine) * float(shared_scale[0]));
  if (!metal::isfinite(float(affine)) || !metal::isfinite(float(value)))
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
  output[selection * p.head_width + column] = value;
}
