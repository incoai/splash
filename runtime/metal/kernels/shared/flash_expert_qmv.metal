#include <metal_stdlib>
#include <metal_simdgroup>
#include "metal/abi/FlashAffine.h"

#pragma METAL fp math_mode(safe)

using namespace metal;

// Off-default SPLASH_FLASH_EXPERT_QMV expert F32 coefficient policy. Original LE Q4/group64 weights and
// signed BF16 SF/bias remain unchanged; no BF16 coefficient operand is made.
// Contiguous8/16-value chunks amortize packed/SF/input address work. This
// changes per-lane K assignment and therefore the reduction semantic version.
inline bool expert_qmv_finite(float value) {
  return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}

template <ushort Values, ushort Simds, ushort Columns>
inline void expert_qmv_contig(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    const device long *expert_ids, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashAffineParams &p,
    uint3 group, uint3 threads, uint simd, uint lane) {
  static_assert(Values == 8 || Values == 16);
  static_assert(Simds * Columns == 8);
  const uint nbase = group.x * 8 + simd * Columns;
  const bool matrix = (p.output_size == 640 && p.input_size == 2560) ||
      (p.output_size == 2560 && p.input_size == 640);
  if (!matrix || (!p.rows || p.rows > 16) || p.selections != 10 ||
      p.experts != 512 || p.bits != 4 || p.group_size != 64 ||
      (p.flags != 1 && p.flags != 3) || threads.x != Simds * 32 ||
      threads.y != 1 || threads.z != 1 ||
      p.weight_row_stride_bytes < p.input_size / 2 ||
      p.weight_row_stride_bytes % 4 || p.weight_expert_stride_bytes % 4 ||
      p.parameter_row_stride_bytes < ulong(p.input_size / 64) * 2 ||
      p.parameter_row_stride_bytes % 2 || p.parameter_expert_stride_bytes % 2) {
    if (lane == 0) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (nbase >= p.output_size || group.y >= p.rows || group.z >= 10) return;
  const ulong route = ulong(group.y) * 10 + group.z;
  const long expert = expert_ids[route];
  if (expert < 0 || ulong(expert) >= 512) {
    if (lane == 0) {
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      for (ushort c = 0; c < Columns; ++c)
        if (nbase + c < p.output_size)
          output[route * p.output_size + nbase + c] = bfloat(as_type<float>(0x7fc00000u));
    }
    return;
  }
  const ulong xrow = (p.flags & 2) ? route : group.y;
  const device bfloat *x = input + xrow * p.input_size;
  const ulong wbase = ulong(expert) * p.weight_expert_stride_bytes +
      ulong(nbase) * p.weight_row_stride_bytes;
  const ulong pbase = ulong(expert) * p.parameter_expert_stride_bytes +
      ulong(nbase) * p.parameter_row_stride_bytes;
  float sums[Columns];
  for (ushort c = 0; c < Columns; ++c) sums[c] = 0.0f;
  for (uint block = 0; block < p.input_size; block += Values * 32) {
    const uint origin = block + lane * Values;
    if (origin >= p.input_size) continue;
    float activations[Values];
    for (ushort j = 0; j < Values; ++j) activations[j] = float(x[origin + j]);
    for (ushort c = 0; c < Columns; ++c) {
      if (nbase + c >= p.output_size) continue;
      const device uint *w = reinterpret_cast<const device uint *>(weights +
          wbase + ulong(c) * p.weight_row_stride_bytes);
      const ulong coefficient = pbase + ulong(c) * p.parameter_row_stride_bytes +
          ulong(origin / 64) * 2;
      const float sf = float(*reinterpret_cast<const device bfloat *>(scales + coefficient));
      const float bias = float(*reinterpret_cast<const device bfloat *>(biases + coefficient));
      for (ushort pack = 0; pack < Values / 8; ++pack) {
        const uint codes = w[origin / 8 + pack];
        for (ushort j = 0; j < 8; ++j) {
          const uint q = (codes >> (j * 4)) & 15u;
          const float weight = float(q) * sf + bias;
          sums[c] += activations[pack * 8 + j] * weight;
        }
      }
    }
  }
  for (ushort c = 0; c < Columns; ++c) {
    const float sum = simd_sum(sums[c]);
    if (lane == 0 && nbase + c < p.output_size) {
      const bfloat result = bfloat(sum);
      if (!expert_qmv_finite(sum) || !expert_qmv_finite(float(result)))
        atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
      output[route * p.output_size + nbase + c] = result;
    }
  }
}

#define EXPERT_QMV_CONTIG(VALUES, SIMDS, COLS) \
kernel void flash_expert_qmv_contig_k##VALUES##_sg##SIMDS##_c##COLS( \
    const device bfloat *input [[buffer(0)]], \
    const device uchar *weights [[buffer(1)]], \
    const device uchar *scales [[buffer(2)]], \
    const device uchar *biases [[buffer(3)]], \
    const device long *expert_ids [[buffer(4)]], \
    device bfloat *output [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], \
    constant FlashAffineParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  expert_qmv_contig<VALUES, SIMDS, COLS>(input, weights, scales, biases, \
      expert_ids, output, diagnostics, p, group, threads, simd, lane); \
}

EXPERT_QMV_CONTIG(16, 4, 2)
EXPERT_QMV_CONTIG(8, 2, 4)
#undef EXPERT_QMV_CONTIG
