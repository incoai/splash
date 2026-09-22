#include <metal_stdlib>
#include <metal_simdgroup>
#include "metal/abi/FlashMoEFused.h"

using namespace metal;

// Unary/shared sigmoid has the same fast arithmetic flags and precise exp
// route as the GPU-qualified MoE v3 helper. Affine dots return to safe math.
#pragma METAL fp math_mode(fast)
inline bfloat flash_moe_down_shared_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

inline bool flash_moe_down_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

inline bool flash_moe_down_geometry(FlashMoEDownFusedParams p) {
  return p.rows && p.rows <= 2048 && p.selections && p.selections <= 10 &&
      p.input_size == 640 && p.output_size == 2560 && p.experts == 512 &&
      !p.reserved0 && !p.reserved1 && !p.reserved2;
}

inline bool flash_moe_down_strides(FlashMoEDownFusedParams p) {
  constexpr ulong top = ~ulong(0);
  const ulong wr = p.weight_row_stride_bytes;
  const ulong we = p.weight_expert_stride_bytes;
  const ulong pr = p.parameter_row_stride_bytes;
  const ulong pe = p.parameter_expert_stride_bytes;
  if (wr < 320 || wr % 4 || we % 4 || pr < 20 || pr % 2 || pe % 2 ||
      wr > (top - 320) / 2559 || pr > (top - 20) / 2559) return false;
  const ulong wm = 2559 * wr + 320;
  const ulong pm = 2559 * pr + 20;
  return we >= wm && pe >= pm && we <= (top - wm) / 511 && pe <= (top - pm) / 511;
}

// Every SIMD lane keeps the original ascending k=lane+32*j sequence and both
// qualified F32 fmuladd expressions. The final SIMD reduction rounds to BF16.
inline float flash_moe_down_dot(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    constant FlashMoEDownFusedParams &p, ulong route, long expert,
    uint output_column, uint lane) {
  const device bfloat *x = input + route * 640;
  const device uint *w = reinterpret_cast<const device uint *>(
      weights + ulong(expert) * p.weight_expert_stride_bytes +
      ulong(output_column) * p.weight_row_stride_bytes);
  const ulong coefficient_offset = ulong(expert) * p.parameter_expert_stride_bytes +
      ulong(output_column) * p.parameter_row_stride_bytes;
  const device bfloat *sf = reinterpret_cast<const device bfloat *>(scales + coefficient_offset);
  const device bfloat *bias = reinterpret_cast<const device bfloat *>(biases + coefficient_offset);
  float sum = 0.0f;
  for (uint g = 0; g < 10; ++g) {
    const float scale = float(sf[g]);
    const float addend = float(bias[g]);
    for (uint sub = 0; sub < 2; ++sub) {
      const uint k = g * 64 + sub * 32 + lane;
      const uint code = (w[g * 8 + sub * 4 + lane / 8] >> ((lane % 8) * 4)) & 15u;
      const float coefficient = float(code) * scale + addend;
      sum += float(x[k]) * coefficient;
    }
  }
  return simd_sum(sum);
}

inline uint flash_moe_down_route_error(const device long *ids,
                                      const device bfloat *scores,
                                      ulong row, uint slot, uint selections,
                                      bool check_duplicates = true) {
  const ulong route = row * selections + slot;
  uint error = 0;
  const long expert = ids[route];
  if (expert < 0 || expert >= 512) error |= 5;
  if (check_duplicates)
    for (uint previous = 0; previous < slot; ++previous)
      if (ids[row * selections + previous] == expert) error |= 1;
  const float weight = float(scores[route]);
  if (!flash_moe_down_finite(weight) || weight < 0.0f || weight > 1.0f) error |= 4;
  return error;
}

template <ushort Simds>
inline void flash_moe_down_terms_impl(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    const device long *ids, const device bfloat *scores,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashMoEDownFusedParams &p, uint3 group, uint3 threads,
    uint simd_group, uint lane) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (!flash_moe_down_geometry(p) || !flash_moe_down_strides(p) ||
      threads.x != 32 * Simds || threads.y != 1 || threads.z != 1) {
    if (lane == 0) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = group.x * Simds + simd_group;
  if (n >= 2560 || group.y >= p.rows || group.z >= p.selections) return;
  const ulong route = ulong(group.y) * p.selections + group.z;
  // Duplicate IDs are validated by the final combine, as in the separate
  // chain. Rejecting them here would introduce a spurious numeric-NaN bit.
  const uint route_error = flash_moe_down_route_error(ids, scores, group.y, group.z,
                                                     p.selections, false);
  if (route_error) {
    if (lane == 0) {
      atomic_fetch_or_explicit(diagnostics, route_error, memory_order_relaxed);
      output[route * 2560 + n] = bfloat(as_type<float>(0x7fc00000u));
    }
    return;
  }
  const float sum = flash_moe_down_dot(input, weights, scales, biases, p,
                                       route, ids[route], n, lane);
  if (lane == 0) {
    const bfloat projected = bfloat(sum);
    const bfloat result = projected * scores[route];
    if (!flash_moe_down_finite(sum) || !flash_moe_down_finite(float(projected)) ||
        !flash_moe_down_finite(float(result))) {
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
      output[route * 2560 + n] = bfloat(as_type<float>(0x7fc00000u));
    } else {
      output[route * 2560 + n] = result;
    }
  }
}

template <ushort Simds>
inline void flash_moe_down_combine_impl(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    const device long *ids, const device bfloat *scores,
    const device bfloat *shared_down, const device bfloat *shared_gate,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashMoEDownFusedParams &p, uint3 group, uint3 threads,
    uint simd_group, uint lane) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (!flash_moe_down_geometry(p) || !flash_moe_down_strides(p) ||
      threads.x != 32 * Simds || threads.y != 1 || threads.z != 1) {
    if (lane == 0) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = group.x * Simds + simd_group;
  if (n >= 2560 || group.y >= p.rows) return;
  const ulong row = group.y;
  bfloat partials[8];
  for (uint part = 0; part < 8; ++part) partials[part] = bfloat(0.0f);
  uint error = 0;
  // Compute in source col8 order so no ten-term down scratch is necessary.
  for (uint part = 0; part < 8; ++part) {
    for (uint slot = part; slot < p.selections; slot += 8) {
      const ulong route = row * p.selections + slot;
      const uint route_error = flash_moe_down_route_error(ids, scores, row, slot, p.selections);
      if (route_error) { error |= route_error; continue; }
      const float sum = flash_moe_down_dot(input, weights, scales, biases, p,
                                           route, ids[route], n, lane);
      if (lane == 0) {
        const bfloat projected = bfloat(sum);
        const bfloat weighted = projected * scores[route];
        if (!flash_moe_down_finite(sum) || !flash_moe_down_finite(float(projected)) ||
            !flash_moe_down_finite(float(weighted))) error |= 4;
        partials[part] = partials[part] + weighted;
      }
    }
  }
  if (lane != 0) return;
  bfloat routed = partials[0];
  for (uint part = 1; part < 8; ++part) routed = routed + partials[part];
  const ulong index = row * 2560 + n;
  const bfloat shared = shared_down[index] * flash_moe_down_shared_sigmoid(shared_gate[row]);
  const bfloat result = routed + shared;
  if (!flash_moe_down_finite(float(shared_down[index])) ||
      !flash_moe_down_finite(float(shared_gate[row])) ||
      !flash_moe_down_finite(float(routed)) || !flash_moe_down_finite(float(shared)) ||
      !flash_moe_down_finite(float(result))) error |= 4;
  if (error) {
    atomic_fetch_or_explicit(diagnostics, error, memory_order_relaxed);
    output[index] = bfloat(as_type<float>(0x7fc00000u));
  } else {
    output[index] = result;
  }
}

#define FLASH_MOE_DOWN_TERMS(NAME, SIMDS)                                    \
kernel void NAME(                                                          \
    const device bfloat *input [[buffer(0)]], const device uchar *w [[buffer(1)]], \
    const device uchar *s [[buffer(2)]], const device uchar *b [[buffer(3)]], \
    const device long *ids [[buffer(4)]], const device bfloat *scores [[buffer(5)]], \
    device bfloat *out [[buffer(6)]], device atomic_uint *diag [[buffer(7)]], \
    constant FlashMoEDownFusedParams &p [[buffer(8)]],                      \
    uint3 group [[threadgroup_position_in_grid]],                          \
    uint3 threads [[threads_per_threadgroup]],                             \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  flash_moe_down_terms_impl<SIMDS>(input, w, s, b, ids, scores, out, diag, p, group, threads, sg, lane); \
}

FLASH_MOE_DOWN_TERMS(flash_moe_fused_down_terms, 8)
FLASH_MOE_DOWN_TERMS(flash_moe_fused_down_terms_s2, 2)
FLASH_MOE_DOWN_TERMS(flash_moe_fused_down_terms_s4, 4)
#undef FLASH_MOE_DOWN_TERMS

#define FLASH_MOE_DOWN_COMBINE(NAME, SIMDS)                                  \
kernel void NAME(                                                          \
    const device bfloat *input [[buffer(0)]], const device uchar *w [[buffer(1)]], \
    const device uchar *s [[buffer(2)]], const device uchar *b [[buffer(3)]], \
    const device long *ids [[buffer(4)]], const device bfloat *scores [[buffer(5)]], \
    const device bfloat *shared [[buffer(6)]], const device bfloat *gate [[buffer(7)]], \
    device bfloat *out [[buffer(8)]], device atomic_uint *diag [[buffer(9)]], \
    constant FlashMoEDownFusedParams &p [[buffer(10)]],                     \
    uint3 group [[threadgroup_position_in_grid]],                          \
    uint3 threads [[threads_per_threadgroup]],                             \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  flash_moe_down_combine_impl<SIMDS>(input, w, s, b, ids, scores, shared, gate, out, diag, p, group, threads, sg, lane); \
}

FLASH_MOE_DOWN_COMBINE(flash_moe_fused_down_combine, 8)
FLASH_MOE_DOWN_COMBINE(flash_moe_fused_down_combine_s2, 2)
FLASH_MOE_DOWN_COMBINE(flash_moe_fused_down_combine_s4, 4)
#undef FLASH_MOE_DOWN_COMBINE

kernel void flash_moe_fused_weighted_combine(
    const device bfloat *terms [[buffer(0)]], const device long *ids [[buffer(1)]],
    const device bfloat *shared_down [[buffer(2)]], const device bfloat *shared_gate [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashMoEDownFusedParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (!flash_moe_down_geometry(p) || threads.x != 256 || threads.y != 1) {
    if (tid == 0) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = group.x * 256 + tid;
  if (n >= 2560 || group.y >= p.rows) return;
  const ulong row = group.y;
  uint error = 0;
  bfloat partials[8];
  for (uint part = 0; part < 8; ++part) {
    bfloat partial = bfloat(0.0f);
    for (uint slot = part; slot < p.selections; slot += 8) {
      const ulong route = row * p.selections + slot;
      const long expert = ids[route];
      if (expert < 0 || expert >= 512) error |= 5;
      for (uint previous = 0; previous < slot; ++previous)
        if (ids[row * p.selections + previous] == expert) error |= 1;
      const bfloat term = terms[route * 2560 + n];
      if (!flash_moe_down_finite(float(term))) error |= 4;
      partial = partial + term;
    }
    partials[part] = partial;
  }
  bfloat routed = partials[0];
  for (uint part = 1; part < 8; ++part) routed = routed + partials[part];
  const ulong index = row * 2560 + n;
  const bfloat shared = shared_down[index] * flash_moe_down_shared_sigmoid(shared_gate[row]);
  const bfloat result = routed + shared;
  if (!flash_moe_down_finite(float(shared_down[index])) ||
      !flash_moe_down_finite(float(shared_gate[row])) ||
      !flash_moe_down_finite(float(routed)) || !flash_moe_down_finite(float(shared)) ||
      !flash_moe_down_finite(float(result))) error |= 4;
  if (error) {
    atomic_fetch_or_explicit(diagnostics, error, memory_order_relaxed);
    output[index] = bfloat(as_type<float>(0x7fc00000u));
  } else {
    output[index] = result;
  }
}
