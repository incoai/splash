#include <metal_stdlib>
#include <metal_simdgroup>
#include "metal/abi/FlashMoEFused.h"

// Match the qualified affine kernel's F32 fmuladd contraction and safe math.
// SwiGLU explicitly requests fast exp, matching the separate compiled route.
#pragma METAL fp math_mode(safe)

using namespace metal;

inline bool flash_moe_fused_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

inline bfloat flash_moe_fused_compiled_sigmoid(bfloat source) {
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bool flash_moe_fused_strides(ulong weight_row, ulong weight_expert,
                                    ulong parameter_row, ulong parameter_expert,
                                    bool word_loads) {
  constexpr ulong max_ulong = ~ulong(0);
  if (weight_row < 1280 || parameter_row < 80 || parameter_row % 2 ||
      parameter_expert % 2 || weight_row > (max_ulong - 1280) / 639 ||
      parameter_row > (max_ulong - 80) / 639 ||
      (word_loads && (weight_row % 4 || weight_expert % 4))) return false;
  const ulong weight_matrix = 639 * weight_row + 1280;
  const ulong parameter_matrix = 639 * parameter_row + 80;
  return weight_expert >= weight_matrix && parameter_expert >= parameter_matrix &&
      weight_expert <= (max_ulong - weight_matrix) / 511 &&
      parameter_expert <= (max_ulong - parameter_matrix) / 511;
}

template <ushort Columns, ushort SimdGroups, bool WordLoads>
inline void flash_moe_fused_gate_up(
    const device bfloat *input, const device uchar *gate_weights,
    const device uchar *gate_scales, const device uchar *gate_biases,
    const device uchar *up_weights, const device uchar *up_scales,
    const device uchar *up_biases, const device long *expert_ids,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashMoEFusedParams &p, uint3 group, uint3 threads,
    uint simd_group, uint lane) {
  static_assert(Columns == 1 || Columns == 2 || Columns == 4 || Columns == 8);
  static_assert(SimdGroups == 2 || SimdGroups == 4 || SimdGroups == 8);
  const bool valid = p.rows && p.rows <= 2048 && p.selections &&
      p.selections <= 10 && p.input_size == 2560 && p.output_size == 640 &&
      p.experts == 512 && !p.reserved0 && !p.reserved1 && !p.reserved2 &&
      threads.x == 32 * SimdGroups && threads.y == 1 && threads.z == 1 &&
      flash_moe_fused_strides(p.gate_weight_row_stride_bytes,
                             p.gate_weight_expert_stride_bytes,
                             p.gate_parameter_row_stride_bytes,
                             p.gate_parameter_expert_stride_bytes, WordLoads) &&
      flash_moe_fused_strides(p.up_weight_row_stride_bytes,
                             p.up_weight_expert_stride_bytes,
                             p.up_parameter_row_stride_bytes,
                             p.up_parameter_expert_stride_bytes, WordLoads);
  if (!valid) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint nbase = group.x * (SimdGroups * Columns) + simd_group * Columns;
  if (nbase >= 640 || group.y >= p.rows || group.z >= p.selections) return;
  const ulong route = ulong(group.y) * p.selections + group.z;
  const long expert = expert_ids[route];
  if (expert < 0 || expert >= 512) {
    if (lane == 0) {
      // The separate projection chain marks the bad ID, then SwiGLU marks
      // its NaN inputs. Preserve the complete chain's sticky bits.
      atomic_fetch_or_explicit(diagnostics, 5u, memory_order_relaxed);
      for (ushort column = 0; column < Columns; ++column)
        if (nbase + column < 640)
          output[route * 640 + nbase + column] = bfloat(as_type<float>(0x7fc00000u));
    }
    return;
  }
  const device bfloat *x = input + ulong(group.y) * 2560;
  const device uchar *gw = gate_weights + ulong(expert) * p.gate_weight_expert_stride_bytes +
      ulong(nbase) * p.gate_weight_row_stride_bytes;
  const device uchar *uw = up_weights + ulong(expert) * p.up_weight_expert_stride_bytes +
      ulong(nbase) * p.up_weight_row_stride_bytes;
  const ulong gate_coefficient = ulong(expert) * p.gate_parameter_expert_stride_bytes +
      ulong(nbase) * p.gate_parameter_row_stride_bytes;
  const ulong up_coefficient = ulong(expert) * p.up_parameter_expert_stride_bytes +
      ulong(nbase) * p.up_parameter_row_stride_bytes;
  float gate_sums[Columns];
  float up_sums[Columns];
  for (ushort column = 0; column < Columns; ++column) {
    gate_sums[column] = 0.0f;
    up_sums[column] = 0.0f;
  }
  for (uint g = 0; g < 40; ++g) {
    float gate_sf[Columns], gate_bias[Columns], up_sf[Columns], up_bias[Columns];
    for (ushort column = 0; column < Columns; ++column) {
      if (nbase + column >= 640) continue;
      const ulong goffset = gate_coefficient + ulong(column) * p.gate_parameter_row_stride_bytes;
      const ulong uoffset = up_coefficient + ulong(column) * p.up_parameter_row_stride_bytes;
      gate_sf[column] = float(reinterpret_cast<const device bfloat *>(gate_scales + goffset)[g]);
      gate_bias[column] = float(reinterpret_cast<const device bfloat *>(gate_biases + goffset)[g]);
      up_sf[column] = float(reinterpret_cast<const device bfloat *>(up_scales + uoffset)[g]);
      up_bias[column] = float(reinterpret_cast<const device bfloat *>(up_biases + uoffset)[g]);
    }
    for (ushort subblock = 0; subblock < 2; ++subblock) {
      const uint k = g * 64 + uint(subblock) * 32 + lane;
      const float activation = float(x[k]);
      for (ushort column = 0; column < Columns; ++column) {
        if (nbase + column >= 640) continue;
        const device uchar *gate_row = gw + ulong(column) * p.gate_weight_row_stride_bytes;
        const device uchar *up_row = uw + ulong(column) * p.up_weight_row_stride_bytes;
        uint gate_code, up_code;
        if (WordLoads) {
          const uint word_index = g * 8 + uint(subblock) * 4 + lane / 8;
          const uint shift = (lane % 8) * 4;
          gate_code = (reinterpret_cast<const device uint *>(gate_row)[word_index] >> shift) & 15u;
          up_code = (reinterpret_cast<const device uint *>(up_row)[word_index] >> shift) & 15u;
        } else {
          const uint shift = (lane & 1u) * 4;
          gate_code = (uint(gate_row[k / 2]) >> shift) & 15u;
          up_code = (uint(up_row[k / 2]) >> shift) & 15u;
        }
        const float gate_coefficient_value = float(gate_code) * gate_sf[column] + gate_bias[column];
        const float up_coefficient_value = float(up_code) * up_sf[column] + up_bias[column];
        gate_sums[column] += activation * gate_coefficient_value;
        up_sums[column] += activation * up_coefficient_value;
      }
    }
  }
  for (ushort column = 0; column < Columns; ++column) {
    const float gate_sum = simd_sum(gate_sums[column]);
    const float up_sum = simd_sum(up_sums[column]);
    if (lane == 0 && nbase + column < 640) {
      const bfloat gate = bfloat(gate_sum);
      const bfloat up = bfloat(up_sum);
      // Gate/up must round before sigmoid and multiplication, exactly as the
      // separate projections do; combining all terms in F32 changes output.
      const bfloat sigmoid = flash_moe_fused_compiled_sigmoid(gate);
      const bfloat silu = gate * sigmoid;
      const bfloat result = silu * up;
      if (!flash_moe_fused_finite(gate_sum) || !flash_moe_fused_finite(up_sum) ||
          !flash_moe_fused_finite(float(gate)) || !flash_moe_fused_finite(float(up)) ||
          !flash_moe_fused_finite(float(result))) {
        atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
        output[route * 640 + nbase + column] = bfloat(as_type<float>(0x7fc00000u));
      } else {
        output[route * 640 + nbase + column] = result;
      }
    }
  }
}

// 0 BF16 input, 1 gate Q4, 2 gate SF, 3 gate biases, 4 up Q4, 5 up SF,
// 6 up biases, 7 I64 IDs, 8 BF16 activation, 9 sticky status, 10 parameters.
#define FLASH_MOE_FUSED(NAME, COLS, SIMDS, WORDS)                             \
kernel void NAME(                                                           \
    const device bfloat *input [[buffer(0)]],                               \
    const device uchar *gate_weights [[buffer(1)]],                         \
    const device uchar *gate_scales [[buffer(2)]],                          \
    const device uchar *gate_biases [[buffer(3)]],                          \
    const device uchar *up_weights [[buffer(4)]],                           \
    const device uchar *up_scales [[buffer(5)]],                            \
    const device uchar *up_biases [[buffer(6)]],                            \
    const device long *expert_ids [[buffer(7)]],                            \
    device bfloat *output [[buffer(8)]],                                    \
    device atomic_uint *diagnostics [[buffer(9)]],                          \
    constant FlashMoEFusedParams &p [[buffer(10)]],                         \
    uint3 group [[threadgroup_position_in_grid]],                          \
    uint3 threads [[threads_per_threadgroup]],                             \
    uint simd_group [[simdgroup_index_in_threadgroup]],                     \
    uint lane [[thread_index_in_simdgroup]]) {                             \
  flash_moe_fused_gate_up<COLS, SIMDS, WORDS>(                               \
      input, gate_weights, gate_scales, gate_biases, up_weights, up_scales,   \
      up_biases, expert_ids, output, diagnostics, p, group, threads,         \
      simd_group, lane);                                                   \
}

FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_c1, 1, 8, false)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_c2, 2, 8, false)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_c4, 4, 8, false)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_c8, 8, 8, false)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_u32_c1, 1, 8, true)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_u32_c2, 2, 8, true)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_u32_c4, 4, 8, true)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_u32_c8, 8, 8, true)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_c1_s2, 1, 2, false)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_c1_s4, 1, 4, false)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_u32_c1_s2, 1, 2, true)
FLASH_MOE_FUSED(flash_moe_fused_q4_gate_up_u32_c1_s4, 1, 4, true)
#undef FLASH_MOE_FUSED
