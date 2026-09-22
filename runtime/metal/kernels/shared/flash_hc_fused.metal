#include "metal/abi/FlashHCFused.h"
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline bfloat hc_fused_sigmoid_fast(bfloat x) {
  const bfloat e = bfloat(metal::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline bfloat hc_fused_sigmoid_unary(bfloat x) {
  const bfloat e = bfloat(metal::precise::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline void hc_fused_check(float x, device atomic_uint *diagnostics) {
  if ((as_type<uint>(x) & 0x7f800000u) == 0x7f800000u)
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
}
inline bool hc_fused_format(FlashHCFusedMatrix m) {
  return m.input_size && m.output_size &&
      (m.bits == 4 || m.bits == 5 || m.bits == 6 || m.bits == 8) &&
      (m.group_size == 32 || m.group_size == 64 || m.group_size == 128) &&
      m.input_size % m.group_size == 0 &&
      m.weight_row_stride_bytes >= (ulong(m.input_size) * m.bits + 7) / 8 &&
      m.parameter_row_stride_bytes >= ulong(m.input_size / m.group_size) * 2 &&
      m.parameter_row_stride_bytes % 2 == 0;
}
inline bool hc_fused_workload(FlashHCFusedParams p) {
  return p.rows && p.rows <= 32 && p.width == 2560 && p.streams == 4 &&
      p.lowrank == 320 && p.arithmetic_mode <= 1 && p.has_injection <= 1 &&
      p.write_raw_up <= 1 && (p.simdgroups == 4 || p.simdgroups == 8);
}

template <ushort Bits>
inline uint hc_fused_code(const device uchar *w, uint k, uint dynamic_bits) {
  const uint bits = Bits ? Bits : dynamic_bits;
  const ulong bit = ulong(k) * bits;
  const uint shift = uint(bit & 7);
  const ulong byte = bit >> 3;
  uint code = uint(w[byte]);
  if (shift + bits > 8) code |= uint(w[byte + 1]) << 8;
  return (code >> shift) & ((1u << bits) - 1);
}

// The exact mode keeps k=lane+32*j and reconstructs every F32 coefficient
// before multiplying. Hoisting parameters never distributes the affine bias.
template <ushort Bits, ushort Group>
inline float hc_fused_dot(
    const device bfloat *x, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    FlashHCFusedMatrix m, uint n, uint lane, uint mode) {
  const uint group_size = Group ? Group : m.group_size;
  const device uchar *w = weights + ulong(n) * m.weight_row_stride_bytes;
  const device bfloat *s = reinterpret_cast<const device bfloat *>(
      scales + ulong(n) * m.parameter_row_stride_bytes);
  const device bfloat *b = reinterpret_cast<const device bfloat *>(
      biases + ulong(n) * m.parameter_row_stride_bytes);
  float sum = 0.0f;
  for (uint g = 0; g < m.input_size / group_size; ++g) {
    const float sf = float(s[g]);
    const float bias = float(b[g]);
    float qdot = 0.0f, input_sum = 0.0f;
    for (uint block = 0; block < group_size / 32; ++block) {
      const uint k = g * group_size + block * 32 + lane;
      const float value = float(x[k]);
      const float code = float(hc_fused_code<Bits>(w, k, m.bits));
      if (mode == 0) {
        const float coefficient = code * sf + bias;
        sum += value * coefficient;
      } else {
        qdot += value * code;
        input_sum += value;
      }
    }
    if (mode == 1) {
      const float scaled = sf * qdot;
      const float corrected = bias * input_sum;
      sum += scaled + corrected;
    }
  }
  return sum;
}

template <ushort Bits, ushort Group, ushort SG>
inline void hc_fused_down(
    const device bfloat *normalized,
    const device uchar *dw, const device uchar *ds, const device uchar *db,
    const device uchar *iw, const device uchar *is, const device uchar *ib,
    device bfloat *activated, device bfloat *gates,
    device atomic_uint *diagnostics, FlashHCFusedParams p,
    uint2 grid, uint simd, uint lane) {
  if (!hc_fused_workload(p) || p.simdgroups != SG ||
      !hc_fused_format(p.down) || p.down.bits != Bits || p.down.group_size != Group ||
      p.down.input_size != 10240 || p.down.output_size != 320 ||
      (p.has_injection && (!hc_fused_format(p.injection) ||
                           p.injection.input_size != 10240 || p.injection.output_size != 4))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = grid.x * SG + simd;
  if (grid.y >= p.rows || n >= p.lowrank + p.has_injection * p.streams) return;
  const device bfloat *x = normalized + ulong(grid.y) * p.width * p.streams;
  float partial;
  if (n < p.lowrank) {
    partial = hc_fused_dot<Bits, Group>(x, dw, ds, db, p.down, n, lane, p.arithmetic_mode);
  } else {
    const uint index = n - p.lowrank;
    if (p.injection.bits == Bits && p.injection.group_size == Group)
      partial = hc_fused_dot<Bits, Group>(x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode);
    else
      partial = hc_fused_dot<0, 0>(x, iw, is, ib, p.injection, index, lane, p.arithmetic_mode);
  }
  const float sum = simd_sum(partial);
  if (!lane) {
    // This raw projection boundary remains even though it is register-only.
    const bfloat raw = bfloat(sum);
    hc_fused_check(sum, diagnostics); hc_fused_check(float(raw), diagnostics);
    const bfloat divided = bfloat(float(raw) / float(p.streams));
    if (n < p.lowrank) {
      const bfloat sigmoid = hc_fused_sigmoid_fast(divided);
      const bfloat result = bfloat(float(divided) * float(sigmoid));
      activated[ulong(grid.y) * p.lowrank + n] = result;
      hc_fused_check(float(result), diagnostics);
    } else {
      const bfloat sigmoid = hc_fused_sigmoid_unary(divided);
      const bfloat result = bfloat(2.0f * float(sigmoid));
      gates[ulong(grid.y) * p.streams + n - p.lowrank] = result;
      hc_fused_check(float(result), diagnostics);
    }
  }
}

template <ushort Bits, ushort Group, ushort SG>
inline void hc_fused_up_mix(
    const device bfloat *normalized, const device bfloat *activated,
    const device uchar *weights, const device uchar *scales, const device uchar *biases,
    device bfloat *mixed, device bfloat *raw_debug,
    device atomic_uint *diagnostics, FlashHCFusedParams p,
    uint2 grid, uint simd, uint lane) {
  if (!hc_fused_workload(p) || p.simdgroups != SG || !hc_fused_format(p.up) ||
      p.up.bits != Bits || p.up.group_size != Group ||
      p.up.input_size != 320 || p.up.output_size != 10240) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint h = grid.x * SG + simd;
  if (grid.y >= p.rows || h >= p.width) return;
  const device bfloat *x = activated + ulong(grid.y) * p.lowrank;
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint g = 0; g < p.lowrank / Group; ++g) {
    float sf[4], bias[4], qdots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float input_sum = 0.0f;
    for (uint stream = 0; stream < 4; ++stream) {
      const uint n = stream * p.width + h;
      const device bfloat *s = reinterpret_cast<const device bfloat *>(
          scales + ulong(n) * p.up.parameter_row_stride_bytes);
      const device bfloat *b = reinterpret_cast<const device bfloat *>(
          biases + ulong(n) * p.up.parameter_row_stride_bytes);
      sf[stream] = float(s[g]); bias[stream] = float(b[g]);
    }
    for (uint block = 0; block < Group / 32; ++block) {
      const uint k = g * Group + block * 32 + lane;
      const float value = float(x[k]);
      input_sum += value;
      for (uint stream = 0; stream < 4; ++stream) {
        const uint n = stream * p.width + h;
        const device uchar *w = weights + ulong(n) * p.up.weight_row_stride_bytes;
        const float code = float(hc_fused_code<Bits>(w, k, Bits));
        if (p.arithmetic_mode == 0) {
          const float coefficient = code * sf[stream] + bias[stream];
          sums[stream] += value * coefficient;
        } else {
          qdots[stream] += value * code;
        }
      }
    }
    if (p.arithmetic_mode == 1) {
      for (uint stream = 0; stream < 4; ++stream) {
        const float scaled = sf[stream] * qdots[stream];
        const float corrected = bias[stream] * input_sum;
        sums[stream] += scaled + corrected;
      }
    }
  }
  for (uint stream = 0; stream < 4; ++stream) sums[stream] = simd_sum(sums[stream]);
  if (!lane) {
    bfloat total = bfloat(0.0f);
    for (uint stream = 0; stream < 4; ++stream) {
      const ulong index = (ulong(grid.y) * 4 + stream) * p.width + h;
      const bfloat raw = bfloat(sums[stream]);
      if (p.write_raw_up) raw_debug[index] = raw;
      hc_fused_check(sums[stream], diagnostics); hc_fused_check(float(raw), diagnostics);
      const bfloat gate = hc_fused_sigmoid_unary(raw);
      const bfloat product = bfloat(float(gate) * float(normalized[index]));
      total = bfloat(float(product) + float(total));
    }
    const bfloat result = bfloat(float(total) / float(p.streams));
    mixed[ulong(grid.y) * p.width + h] = result;
    hc_fused_check(float(result), diagnostics);
  }
}

#define HC_FUSED_ENTRIES(BITS, GROUP, SG) \
kernel void flash_hc_fused_down_q##BITS##_g##GROUP##_s##SG( \
    const device bfloat *normalized [[buffer(0)]], \
    const device uchar *dw [[buffer(1)]], const device uchar *ds [[buffer(2)]], \
    const device uchar *db [[buffer(3)]], const device uchar *iw [[buffer(4)]], \
    const device uchar *is [[buffer(5)]], const device uchar *ib [[buffer(6)]], \
    device bfloat *activated [[buffer(7)]], device bfloat *gates [[buffer(8)]], \
    device atomic_uint *diagnostics [[buffer(9)]], \
    constant FlashHCFusedParams &p [[buffer(10)]], \
    uint2 grid [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  hc_fused_down<BITS, GROUP, SG>(normalized, dw, ds, db, iw, is, ib, activated, gates, \
                                 diagnostics, p, grid, simd, lane); \
} \
kernel void flash_hc_fused_up_mix_q##BITS##_g##GROUP##_s##SG( \
    const device bfloat *normalized [[buffer(0)]], const device bfloat *activated [[buffer(1)]], \
    const device uchar *w [[buffer(2)]], const device uchar *s [[buffer(3)]], \
    const device uchar *b [[buffer(4)]], device bfloat *mixed [[buffer(5)]], \
    device bfloat *raw_debug [[buffer(6)]], device atomic_uint *diagnostics [[buffer(7)]], \
    constant FlashHCFusedParams &p [[buffer(8)]], \
    uint2 grid [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  hc_fused_up_mix<BITS, GROUP, SG>(normalized, activated, w, s, b, mixed, raw_debug, \
                                   diagnostics, p, grid, simd, lane); \
}

#define HC_FUSED_GROUPS(BITS, SG) \
HC_FUSED_ENTRIES(BITS, 32, SG) \
HC_FUSED_ENTRIES(BITS, 64, SG) \
HC_FUSED_ENTRIES(BITS, 128, SG)
#define HC_FUSED_BITS(SG) \
HC_FUSED_GROUPS(4, SG) \
HC_FUSED_GROUPS(5, SG) \
HC_FUSED_GROUPS(6, SG) \
HC_FUSED_GROUPS(8, SG)
HC_FUSED_BITS(4)
HC_FUSED_BITS(8)
#undef HC_FUSED_BITS
#undef HC_FUSED_GROUPS
#undef HC_FUSED_ENTRIES

// Each stream owns one group. Four adjacent values/thread and the two SIMD
// reductions match the qualified norm traversal; cached injected BF16 values
// avoid rereading a freshly written hyper plane and make exact in-place safe.
kernel void flash_hc_fused_inject_norm(
    const device bfloat *hyper [[buffer(0)]], const device bfloat *branch [[buffer(1)]],
    const device bfloat *gates [[buffer(2)]], const device uchar *weight [[buffer(3)]],
    device bfloat *updated [[buffer(4)]], device bfloat *normalized [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashHCFusedParams &p [[buffer(7)]],
    uint2 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (!hc_fused_workload(p) || p.norm_is_float > 1 || p.norm_convention > 1 ||
      !isfinite(p.norm_epsilon) || p.norm_epsilon <= 0) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (grid.x >= p.rows || grid.y >= p.streams) return;
  const ulong base = (ulong(grid.x) * p.streams + grid.y) * p.width;
  const bfloat gate = gates[ulong(grid.x) * p.streams + grid.y];
  bfloat values[4];
  float square_sum = 0.0f;
  for (uint element = 0; element < 4; ++element) {
    const uint column = tid * 4 + element;
    const bfloat product = bfloat(float(branch[ulong(grid.x) * p.width + column]) * float(gate));
    values[element] = bfloat(float(hyper[base + column]) + float(product));
    const float value = float(values[element]);
    square_sum += value * value;
  }
  square_sum = simd_sum(square_sum);
  threadgroup float partials[32];
  if (simd == 0) partials[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!lane) partials[simd] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    const float total = simd_sum(partials[lane]);
    if (!lane) partials[0] = metal::precise::rsqrt(total / float(p.width) + p.norm_epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint element = 0; element < 4; ++element) {
    const uint column = tid * 4 + element;
    const ulong offset = ulong(grid.y) * p.width + column;
    const float raw = p.norm_is_float
        ? reinterpret_cast<const device float *>(weight)[offset]
        : float(reinterpret_cast<const device bfloat *>(weight)[offset]);
    const float scale = p.norm_convention == 0 ? 1.0f + raw : raw;
    const float norm = float(values[element]) * partials[0];
    const bfloat result = bfloat(norm * scale);
    updated[base + column] = values[element];
    normalized[base + column] = result;
    hc_fused_check(float(values[element]), diagnostics); hc_fused_check(float(result), diagnostics);
  }
}
