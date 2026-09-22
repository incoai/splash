#include <metal_stdlib>
#include "metal/abi/FlashForward.h"

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline bfloat flash_forward_sigmoid_bf16(bfloat source) {
  // Installed MLX GPU rounds exp, denominator, division and positive-tail
  // subtraction in BF16. Its CPU functor uses different scalar promotions.
  const bfloat exponential = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

// Canonical BF16 boundaries: down/streams, sigmoid, product. Exact input/output
// aliasing is safe because each thread reads and replaces one independent item.
kernel void flash_forward_hc_silu(
    const device bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant FlashForwardActivationParams &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (!p.rows || !p.width || !p.streams) {
    if (!index) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (ulong(index) >= ulong(p.rows) * p.width) return;
  const bfloat divided = bfloat(float(input[index]) / float(p.streams));
  const bfloat sigmoid = flash_forward_sigmoid_bf16(divided);
  const bfloat result = bfloat(float(divided) * float(sigmoid));
  output[index] = result;
  if (!isfinite(float(result)))
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
}

// Raw-word copies preserve every F32 recurrence and BF16 history bit. Source
// and destination are validated disjoint logical views by the host caller.
kernel void flash_forward_copy_words(
    const device uint *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant FlashForwardCopyParams &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (ulong(index) < p.words) output[index] = input[index];
}
