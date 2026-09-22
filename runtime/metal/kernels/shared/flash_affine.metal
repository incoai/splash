#include <metal_stdlib>
#include <metal_simdgroup>
#include "metal/abi/FlashAffine.h"

#pragma METAL fp math_mode(safe)

using namespace metal;

// Native source-layout affine primitives. The packed rows are unsigned LE
// bitstreams, not Splash's Q4 StorageN format. Each SIMD group owns one output
// column and reduces K in F32; eight columns share a 256-thread group. This is
// a correctness control for future measured matrix-tile kernels.

inline uint flash_affine_code(const device uchar *row, uint bits,
                              uint channel) {
  const ulong bit = ulong(channel) * bits;
  const ulong byte = bit >> 3;
  const uint shift = uint(bit & 7);
  uint word = uint(row[byte]);
  if (shift + bits > 8) word |= uint(row[byte + 1]) << 8;
  return (word >> shift) & ((1u << bits) - 1);
}

inline bool flash_affine_geometry(uint k, uint bits, uint group_size,
                                  ulong weight_stride,
                                  ulong parameter_stride) {
  return k && (bits == 4 || bits == 5 || bits == 6 || bits == 8) &&
      (group_size == 32 || group_size == 64 || group_size == 128) &&
      k % group_size == 0 && weight_stride >= (ulong(k) * bits + 7) / 8 &&
      parameter_stride >= ulong(k / group_size) * 2 &&
      parameter_stride % 2 == 0;
}

inline float flash_affine_nan() { return as_type<float>(0x7fc00000u); }
inline bool flash_affine_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

// 0 BF16 input, 1 packed U32-as-bytes, 2 BF16 scales, 3 BF16 biases,
// 4 I64 selected IDs (unused dummy for dense), 5 BF16 output,
// 6 sticky uint32 diagnostics, 7 FlashAffineParams.
kernel void flash_affine_project(
    const device bfloat *input [[buffer(0)]],
    const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashAffineParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = group.x * 8 + simd_group;
  if (!p.rows || !p.selections || !p.experts || !p.output_size ||
      (p.flags & ~3u) || ((p.flags & 2u) && !(p.flags & 1u)) ||
      (!(p.flags & 1u) && (p.experts != 1 || p.selections != 1)) ||
      !flash_affine_geometry(p.input_size, p.bits, p.group_size,
                             p.weight_row_stride_bytes,
                             p.parameter_row_stride_bytes) ||
      p.parameter_expert_stride_bytes % 2) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (n >= p.output_size || group.y >= p.rows || group.z >= p.selections)
    return;
  const ulong route = ulong(group.y) * p.selections + group.z;
  const long expert = (p.flags & 1u) ? expert_ids[route] : 0;
  if (expert < 0 || ulong(expert) >= p.experts) {
    if (lane == 0) {
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      output[route * p.output_size + n] = bfloat(flash_affine_nan());
    }
    return;
  }
  const ulong input_row = (p.flags & 2u) ? route : group.y;
  const device bfloat *x = input + input_row * p.input_size;
  const device uchar *w = weights + ulong(expert) *
      p.weight_expert_stride_bytes + ulong(n) * p.weight_row_stride_bytes;
  const ulong coefficient_offset = ulong(expert) *
      p.parameter_expert_stride_bytes + ulong(n) *
      p.parameter_row_stride_bytes;
  const device bfloat *s = reinterpret_cast<const device bfloat *>(
      scales + coefficient_offset);
  const device bfloat *b = reinterpret_cast<const device bfloat *>(
      biases + coefficient_offset);
  float sum = 0.0f;
  for (uint k = lane; k < p.input_size; k += 32) {
    const uint coefficient = k / p.group_size;
    const float weight = float(flash_affine_code(w, p.bits, k)) *
        float(s[coefficient]) + float(b[coefficient]);
    sum += float(x[k]) * weight;
  }
  sum = simd_sum(sum);
  if (lane == 0) {
    const bfloat result = bfloat(sum);
    if (!flash_affine_finite(sum) || !flash_affine_finite(float(result)))
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
    output[route * p.output_size + n] = result;
  }
}

// 0 BF16 input, 1 BF16 row-major weights[N,K], 2 BF16 output,
// 3 sticky uint32 diagnostics, 4 FlashDenseParams.
kernel void flash_dense_bf16_project(
    const device bfloat *input [[buffer(0)]],
    const device uchar *weights [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant FlashDenseParams &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (!p.rows || !p.input_size || !p.output_size ||
      p.weight_row_stride_bytes < ulong(p.input_size) * 2 ||
      p.weight_row_stride_bytes % 2) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint n = group.x * 8 + simd_group;
  if (n >= p.output_size || group.y >= p.rows) return;
  const device bfloat *w = reinterpret_cast<const device bfloat *>(
      weights + ulong(n) * p.weight_row_stride_bytes);
  const device bfloat *x = input + ulong(group.y) * p.input_size;
  float sum = 0.0f;
  for (uint k = lane; k < p.input_size; k += 32)
    sum += float(x[k]) * float(w[k]);
  sum = simd_sum(sum);
  if (lane == 0) {
    const bfloat result = bfloat(sum);
    if (!flash_affine_finite(sum) || !flash_affine_finite(float(result)))
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
    output[ulong(group.y) * p.output_size + n] = result;
  }
}

// 0 I64 token IDs, 1 packed U32-as-bytes, 2 BF16 scales, 3 BF16 biases,
// 4 BF16 output, 5 sticky uint32 diagnostics, 6 FlashEmbeddingParams.
kernel void flash_affine_embedding(
    const device long *token_ids [[buffer(0)]],
    const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device uchar *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashEmbeddingParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (!p.rows || !p.vocabulary_size ||
      !flash_affine_geometry(p.input_size, p.bits, p.group_size,
                             p.weight_row_stride_bytes,
                             p.parameter_row_stride_bytes)) {
    if (tid == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint k = group.x * 256 + tid;
  if (k >= p.input_size || group.y >= p.rows) return;
  const long token = token_ids[group.y];
  if (token < 0 || ulong(token) >= p.vocabulary_size) {
    if (tid == 0)
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
    output[ulong(group.y) * p.input_size + k] = bfloat(flash_affine_nan());
    return;
  }
  const device uchar *w = weights + ulong(token) * p.weight_row_stride_bytes;
  const ulong coefficient_offset = ulong(token) * p.parameter_row_stride_bytes;
  const device bfloat *s = reinterpret_cast<const device bfloat *>(
      scales + coefficient_offset);
  const device bfloat *b = reinterpret_cast<const device bfloat *>(
      biases + coefficient_offset);
  const uint coefficient = k / p.group_size;
  const float value = float(flash_affine_code(w, p.bits, k)) *
      float(s[coefficient]) + float(b[coefficient]);
  const bfloat result = bfloat(value);
  if (!flash_affine_finite(value) || !flash_affine_finite(float(result)))
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
  output[ulong(group.y) * p.input_size + k] = result;
}

// Measured source-layout routes. The generic control above remains available.
// Hoisting scales/biases and reusing activations does not distribute affine
// correction: every coefficient is still formed in F32 before multiplication.
template <ushort Bits, ushort GroupSize, ushort Columns, bool WordLoads = false>
inline void flash_affine_specialized_project(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    const device long *expert_ids, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashAffineParams &p,
    uint3 group, uint simd_group, uint lane) {
  static_assert(Bits == 4 || Bits == 8);
  static_assert(GroupSize == 64 || GroupSize == 128);
  const uint nbase = group.x * (8 * Columns) + simd_group * Columns;
  const bool shape_ok = p.rows && p.selections && p.experts && p.output_size &&
      p.input_size && p.input_size % GroupSize == 0 &&
      p.bits == Bits && p.group_size == GroupSize &&
      !(p.flags & ~3u) && (!(p.flags & 2u) || (p.flags & 1u)) &&
      ((p.flags & 1u) || (p.experts == 1 && p.selections == 1)) &&
      p.weight_row_stride_bytes >= ulong(p.input_size) * Bits / 8 &&
      p.parameter_row_stride_bytes >= ulong(p.input_size / GroupSize) * 2 &&
      p.parameter_row_stride_bytes % 2 == 0 &&
      p.parameter_expert_stride_bytes % 2 == 0 &&
      (!WordLoads || (p.weight_row_stride_bytes % 4 == 0 &&
                     p.weight_expert_stride_bytes % 4 == 0));
  if (!shape_ok) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (nbase >= p.output_size || group.y >= p.rows || group.z >= p.selections)
    return;
  const ulong route = ulong(group.y) * p.selections + group.z;
  const long expert = (p.flags & 1u) ? expert_ids[route] : 0;
  if (expert < 0 || ulong(expert) >= p.experts) {
    if (lane == 0) {
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      for (ushort c = 0; c < Columns; ++c)
        if (nbase + c < p.output_size)
          output[route * p.output_size + nbase + c] =
              bfloat(as_type<float>(0x7fc00000u));
    }
    return;
  }
  const ulong input_row = (p.flags & 2u) ? route : group.y;
  const device bfloat *x = input + input_row * p.input_size;
  const device uchar *wbase = weights + ulong(expert) *
      p.weight_expert_stride_bytes + ulong(nbase) * p.weight_row_stride_bytes;
  const ulong coefficient_offset = ulong(expert) *
      p.parameter_expert_stride_bytes + ulong(nbase) *
      p.parameter_row_stride_bytes;
  const device uchar *sbase = scales + coefficient_offset;
  const device uchar *bbase = biases + coefficient_offset;
  float sums[Columns];
  for (ushort c = 0; c < Columns; ++c) sums[c] = 0.0f;
  for (uint g = 0; g < p.input_size / GroupSize; ++g) {
    float sf[Columns];
    float bias[Columns];
    for (ushort c = 0; c < Columns; ++c) {
      if (nbase + c < p.output_size) {
        const device bfloat *s = reinterpret_cast<const device bfloat *>(
            sbase + ulong(c) * p.parameter_row_stride_bytes);
        const device bfloat *b = reinterpret_cast<const device bfloat *>(
            bbase + ulong(c) * p.parameter_row_stride_bytes);
        sf[c] = float(s[g]);
        bias[c] = float(b[g]);
      }
    }
    for (ushort subblock = 0; subblock < GroupSize / 32; ++subblock) {
      const uint k = g * GroupSize + uint(subblock) * 32 + lane;
      const float activation = float(x[k]);
      for (ushort c = 0; c < Columns; ++c) {
        if (nbase + c < p.output_size) {
          const device uchar *w = wbase + ulong(c) * p.weight_row_stride_bytes;
          uint code;
          if (WordLoads) {
            const device uint *wu = reinterpret_cast<const device uint *>(w);
            const uint packed = wu[g * (GroupSize * Bits / 32) +
                uint(subblock) * Bits + lane / (32 / Bits)];
            code = (packed >> ((lane % (32 / Bits)) * Bits)) & ((1u << Bits) - 1);
          } else {
            code = Bits == 4
                ? (uint(w[k / 2]) >> ((lane & 1u) * 4)) & 15u
                : uint(w[k]);
          }
          const float coefficient = float(code) * sf[c] + bias[c];
          sums[c] += activation * coefficient;
        }
      }
    }
  }
  for (ushort c = 0; c < Columns; ++c) {
    const float sum = simd_sum(sums[c]);
    if (lane == 0 && nbase + c < p.output_size) {
      const bfloat result = bfloat(sum);
      if (!flash_affine_finite(sum) || !flash_affine_finite(float(result)))
        atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
      output[route * p.output_size + nbase + c] = result;
    }
  }
}

#define FLASH_AFFINE_SPECIALIZED(BITS, GROUP, COLS) \
kernel void flash_affine_q##BITS##_g##GROUP##_c##COLS( \
    const device bfloat *input [[buffer(0)]], \
    const device uchar *weights [[buffer(1)]], \
    const device uchar *scales [[buffer(2)]], \
    const device uchar *biases [[buffer(3)]], \
    const device long *expert_ids [[buffer(4)]], \
    device bfloat *output [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], \
    constant FlashAffineParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  flash_affine_specialized_project<BITS, GROUP, COLS>( \
      input, weights, scales, biases, expert_ids, output, diagnostics, \
      p, group, simd_group, lane); \
}


FLASH_AFFINE_SPECIALIZED(4, 64, 1)
FLASH_AFFINE_SPECIALIZED(8, 64, 2)
#undef FLASH_AFFINE_SPECIALIZED

#define FLASH_AFFINE_WORD_SPECIALIZED(BITS, GROUP, COLS) \
kernel void flash_affine_q##BITS##_g##GROUP##_u32_c##COLS( \
    const device bfloat *input [[buffer(0)]], \
    const device uchar *weights [[buffer(1)]], \
    const device uchar *scales [[buffer(2)]], \
    const device uchar *biases [[buffer(3)]], \
    const device long *expert_ids [[buffer(4)]], \
    device bfloat *output [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], \
    constant FlashAffineParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  flash_affine_specialized_project<BITS, GROUP, COLS, true>( \
      input, weights, scales, biases, expert_ids, output, diagnostics, \
      p, group, simd_group, lane); \
}


FLASH_AFFINE_WORD_SPECIALIZED(4, 64, 1)
#undef FLASH_AFFINE_WORD_SPECIALIZED

// Qualified source-layout dense, HC and shared-gate byte-stream routes. These
// preserve the generic lane K order while hoisting coefficients over K32
// subblocks; cross-byte Q5/Q6 extraction remains in the original LE layout.
template <ushort Bits, ushort GroupSize>
inline void flash_affine_grouped_byte_project(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    const device long *expert_ids, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashAffineParams &p,
    uint3 group, uint simd_group, uint lane) {
  static_assert(Bits == 4 || Bits == 5 || Bits == 6 || Bits == 8);
  static_assert(GroupSize == 32 || GroupSize == 64 || GroupSize == 128);
  const uint n = group.x * 8 + simd_group;
  const bool shape_ok = p.rows && p.selections && p.experts && p.output_size &&
      p.input_size && p.input_size % GroupSize == 0 &&
      p.bits == Bits && p.group_size == GroupSize &&
      !(p.flags & ~3u) && (!(p.flags & 2u) || (p.flags & 1u)) &&
      ((p.flags & 1u) || (p.experts == 1 && p.selections == 1)) &&
      p.weight_row_stride_bytes >= ulong(p.input_size) * Bits / 8 &&
      p.parameter_row_stride_bytes >= ulong(p.input_size / GroupSize) * 2 &&
      p.parameter_row_stride_bytes % 2 == 0 &&
      p.parameter_expert_stride_bytes % 2 == 0;
  if (!shape_ok) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (n >= p.output_size || group.y >= p.rows || group.z >= p.selections)
    return;
  const ulong route = ulong(group.y) * p.selections + group.z;
  const long expert = (p.flags & 1u) ? expert_ids[route] : 0;
  if (expert < 0 || ulong(expert) >= p.experts) {
    if (lane == 0) {
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      output[route * p.output_size + n] = bfloat(as_type<float>(0x7fc00000u));
    }
    return;
  }
  const ulong input_row = (p.flags & 2u) ? route : group.y;
  const device bfloat *x = input + input_row * p.input_size;
  const device uchar *w = weights + ulong(expert) *
      p.weight_expert_stride_bytes + ulong(n) * p.weight_row_stride_bytes;
  const ulong coefficient_offset = ulong(expert) *
      p.parameter_expert_stride_bytes + ulong(n) *
      p.parameter_row_stride_bytes;
  const device bfloat *s = reinterpret_cast<const device bfloat *>(scales + coefficient_offset);
  const device bfloat *b = reinterpret_cast<const device bfloat *>(biases + coefficient_offset);
  const uint lane_byte = lane * Bits / 8;
  const uint lane_shift = lane * Bits % 8;
  float sum = 0.0f;
  for (uint g = 0; g < p.input_size / GroupSize; ++g) {
    const float sf = float(s[g]);
    const float bias = float(b[g]);
    for (ushort subblock = 0; subblock < GroupSize / 32; ++subblock) {
      const uint k = g * GroupSize + uint(subblock) * 32 + lane;
      const ulong byte = ulong(g) * (GroupSize * Bits / 8) +
          uint(subblock) * (4 * Bits) + lane_byte;
      uint word = uint(w[byte]);
      if (lane_shift + Bits > 8) word |= uint(w[byte + 1]) << 8;
      const uint code = (word >> lane_shift) & ((1u << Bits) - 1);
      const float weight = float(code) * sf + bias;
      sum += float(x[k]) * weight;
    }
  }
  sum = simd_sum(sum);
  if (lane == 0) {
    const bfloat result = bfloat(sum);
    if (!flash_affine_finite(sum) || !flash_affine_finite(float(result)))
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
    output[route * p.output_size + n] = result;
  }
}

#define FLASH_AFFINE_GROUPED_BYTE(BITS, GROUP) \
kernel void flash_affine_q##BITS##_g##GROUP##_c1( \
    const device bfloat *input [[buffer(0)]], \
    const device uchar *weights [[buffer(1)]], \
    const device uchar *scales [[buffer(2)]], \
    const device uchar *biases [[buffer(3)]], \
    const device long *expert_ids [[buffer(4)]], \
    device bfloat *output [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], \
    constant FlashAffineParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint simd_group [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  flash_affine_grouped_byte_project<BITS, GROUP>(input, weights, scales, biases, expert_ids, \
      output, diagnostics, p, group, simd_group, lane); \
}


FLASH_AFFINE_GROUPED_BYTE(5, 64)
FLASH_AFFINE_GROUPED_BYTE(6, 64)
FLASH_AFFINE_GROUPED_BYTE(8, 64)
FLASH_AFFINE_GROUPED_BYTE(5, 128)
FLASH_AFFINE_GROUPED_BYTE(8, 128)
#undef FLASH_AFFINE_GROUPED_BYTE

// Keep the previously qualified Q4 expert byte fallback as its own export;
// the measured dense/HC route uses the literal grouped byte helper.
kernel void flash_affine_q4_g64_grouped_c1(
    const device bfloat *input [[buffer(0)]],
    const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashAffineParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  flash_affine_grouped_byte_project<4, 64>(input, weights, scales, biases,
      expert_ids, output, diagnostics, p, group, simd_group, lane);
}
