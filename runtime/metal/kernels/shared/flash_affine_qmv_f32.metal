// Copyright © 2023-2024 Apple Inc.
// Adapted from installed Apple MLX quantized.h load_vector/qdot/qmv variants.
// Source: mlx/backend/metal/kernels/quantized.h, qmv_fast_impl/qmv_impl.
// License copied from the bundled MLX license below.
/*
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <metal_stdlib>
#include <metal_simdgroup>
#include "metal/abi/FlashAffine.h"

#pragma METAL fp math_mode(safe)

using namespace metal;

// Off-default SPLASH_FLASH_QMV_F32 profile: math version "mlx-qmv-f32xsum-v1".
// This intentionally uses MLX's contiguous per-lane packed-vector traversal,
// F32 activation prescaling/XSUM, grouped dot, then scale*dot + bias*activation_sum.
// Every activation is cast to F32 before sums and divisions, avoiding BF16
// chunk-add rounding from the literal original load_vector expressions.
// It is not bit-identical to the native explicit F32 coefficient control.
// Stock MLX output-tail backshift is replaced by guarded columns to avoid
// duplicate stores; guards and traversal remain unchanged from mlx-qmv-v1.
// Q4's uint16 byte assembly supports unaligned row views with identical masks.
namespace splash_mlx_qmv_f32xsum_v1 {

template <typename T, typename U, int values_per_thread, int bits>
inline U mlx_qmv_f32xsum_v1_load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 4.0f;
      x_thread[i + 2] = float(x[i + 2]) / 16.0f;
      x_thread[i + 3] = float(x[i + 3]) / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + float(x[i + 4]) + float(x[i + 5]) +
          float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 8.0f;
      x_thread[i + 2] = float(x[i + 2]) / 64.0f;
      x_thread[i + 3] = float(x[i + 3]) / 2.0f;
      x_thread[i + 4] = float(x[i + 4]) / 16.0f;
      x_thread[i + 5] = float(x[i + 5]) / 128.0f;
      x_thread[i + 6] = float(x[i + 6]) / 4.0f;
      x_thread[i + 7] = float(x[i + 7]) / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 16.0f;
      x_thread[i + 2] = float(x[i + 2]) / 256.0f;
      x_thread[i + 3] = float(x[i + 3]) / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + float(x[i + 4]) + float(x[i + 5]) +
          float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 32.0f;
      x_thread[i + 2] = float(x[i + 2]) / 4.0f;
      x_thread[i + 3] = float(x[i + 3]) / 128.0f;
      x_thread[i + 4] = float(x[i + 4]) / 16.0f;
      x_thread[i + 5] = float(x[i + 5]) / 2.0f;
      x_thread[i + 6] = float(x[i + 6]) / 64.0f;
      x_thread[i + 7] = float(x[i + 7]) / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 64.0f;
      x_thread[i + 2] = float(x[i + 2]) / 16.0f;
      x_thread[i + 3] = float(x[i + 3]) / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += float(x[i]);
      x_thread[i] = float(x[i]);
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U mlx_qmv_f32xsum_v1_load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 4.0f;
      x_thread[i + 2] = float(x[i + 2]) / 16.0f;
      x_thread[i + 3] = float(x[i + 3]) / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + float(x[i + 4]) + float(x[i + 5]) +
          float(x[i + 6]) + float(x[i + 7]);

      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 8.0f;
      x_thread[i + 2] = float(x[i + 2]) / 64.0f;
      x_thread[i + 3] = float(x[i + 3]) / 2.0f;
      x_thread[i + 4] = float(x[i + 4]) / 16.0f;
      x_thread[i + 5] = float(x[i + 5]) / 128.0f;
      x_thread[i + 6] = float(x[i + 6]) / 4.0f;
      x_thread[i + 7] = float(x[i + 7]) / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 16.0f;
      x_thread[i + 2] = float(x[i + 2]) / 256.0f;
      x_thread[i + 3] = float(x[i + 3]) / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + float(x[i + 4]) + float(x[i + 5]) +
          float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 32.0f;
      x_thread[i + 2] = float(x[i + 2]) / 4.0f;
      x_thread[i + 3] = float(x[i + 3]) / 128.0f;
      x_thread[i + 4] = float(x[i + 4]) / 16.0f;
      x_thread[i + 5] = float(x[i + 5]) / 2.0f;
      x_thread[i + 6] = float(x[i + 6]) / 64.0f;
      x_thread[i + 7] = float(x[i + 7]) / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = float(x[i]);
      x_thread[i + 1] = float(x[i + 1]) / 64.0f;
      x_thread[i + 2] = float(x[i + 2]) / 16.0f;
      x_thread[i + 3] = float(x[i + 3]) / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += float(x[i]);
      x_thread[i] = float(x[i]);
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U mlx_qmv_f32xsum_v1_qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      const uint16_t packed = uint16_t(w[2 * i]) |
          (uint16_t(w[2 * i + 1]) << 8);
      accum +=
          (x_thread[4 * i] * (packed & 0x000f) +
           x_thread[4 * i + 1] * (packed & 0x00f0) +
           x_thread[4 * i + 2] * (packed & 0x0f00) +
           x_thread[4 * i + 3] * (packed & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U mlx_qmv_f32xsum_v1_qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < (N / 4); i++) {
      const uint16_t packed = uint16_t(w[2 * i]) |
          (uint16_t(w[2 * i + 1]) << 8);
      accum +=
          (x_thread[4 * i] * (packed & 0x000f) +
           x_thread[4 * i + 1] * (packed & 0x00f0) +
           x_thread[4 * i + 2] * (packed & 0x0f00) +
           x_thread[4 * i + 3] * (packed & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

inline bool finite(float x) {
  return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u;
}

template <ushort Bits>
inline constexpr uint pack_factor_32() {
  return Bits == 5 ? 8 : (Bits == 6 ? 4 : 32 / Bits);
}

// Integer addresses adapt row/expert strides to FlashAffineParams. Traversal
// and floating point expressions below retain the installed MLX qmv order.
template <ushort Bits, ushort GroupSize, ushort Values>
inline void accumulate_chunk(
    const device uchar *weights, const device uchar *scales,
    const device uchar *biases, constant FlashAffineParams &p,
    uint out_row, uint channel, uint expert, const thread float *x_thread,
    float activation_sum, uint remaining, bool tail,
    thread float *result) {
  const ulong coefficient_expert = ulong(expert) *
      p.parameter_expert_stride_bytes;
  const ulong weight_expert = ulong(expert) * p.weight_expert_stride_bytes;
  const uint coefficient = channel / GroupSize;
  for (ushort row = 0; row < 4; ++row) {
    if (out_row + row < p.output_size) {
      const device uchar *wl = weights + weight_expert +
          ulong(out_row + row) * p.weight_row_stride_bytes +
          ulong(channel) * Bits / 8;
      const ulong coefficient_row = coefficient_expert +
          ulong(out_row + row) * p.parameter_row_stride_bytes;
      const device bfloat *sl = reinterpret_cast<const device bfloat *>(
          scales + coefficient_row);
      const device bfloat *bl = reinterpret_cast<const device bfloat *>(
          biases + coefficient_row);
      const float s = float(sl[coefficient]);
      const float b = float(bl[coefficient]);
      if (tail) {
        result[row] += mlx_qmv_f32xsum_v1_qdot_safe<float, Values, Bits>(
            wl, x_thread, s, b, activation_sum, int(remaining));
      } else {
        result[row] += mlx_qmv_f32xsum_v1_qdot<float, Values, Bits>(
            wl, x_thread, s, b, activation_sum);
      }
    }
  }
}

template <ushort Bits, ushort GroupSize, bool Fast>
inline void project_math(
    const device bfloat *x, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashAffineParams &p, uint out_row, uint expert,
    ulong route, uint lane) {
  constexpr uint PacksPerThread = Fast ? 2 : 1;
  constexpr uint Values = pack_factor_32<Bits>() * PacksPerThread;
  constexpr uint Block = Values * 32;
  static_assert(GroupSize % Values == 0);
  thread float x_thread[Values];
  thread float result[4] = {0};

  if (Fast) {
    // qmv_fast_impl: a complete packed block per lane, then four output rows.
    for (uint k = 0; k < p.input_size; k += Block) {
      const uint channel = k + lane * Values;
      const float sum = mlx_qmv_f32xsum_v1_load_vector<bfloat, float, Values, Bits>(
          x + channel, x_thread);
      accumulate_chunk<Bits, GroupSize, Values>(
          weights, scales, biases, p, out_row, channel, expert,
          x_thread, sum, Values, false, result);
    }
  } else {
    // qmv_impl processes all but its final block with the regular helpers.
    // Use subtraction after k<K so tiny K cannot underflow this condition.
    uint k = 0;
    for (; k < p.input_size && p.input_size - k > Block; k += Block) {
      const uint channel = k + lane * Values;
      const float sum = mlx_qmv_f32xsum_v1_load_vector<bfloat, float, Values, Bits>(
          x + channel, x_thread);
      accumulate_chunk<Bits, GroupSize, Values>(
          weights, scales, biases, p, out_row, channel, expert,
          x_thread, sum, Values, false, result);
    }
    const uint lane_start = lane * Values;
    const uint available = p.input_size - k;
    const uint remaining = available > lane_start
        ? min(available - lane_start, Values) : 0;
    if (remaining > 0) {
      const uint channel = k + lane_start;
      const float sum = mlx_qmv_f32xsum_v1_load_vector_safe<
          bfloat, float, Values, Bits>(x + channel, x_thread, int(remaining));
      accumulate_chunk<Bits, GroupSize, Values>(
          weights, scales, biases, p, out_row, channel, expert,
          x_thread, sum, remaining, true, result);
    }
  }

  for (ushort row = 0; row < 4; ++row) {
    if (out_row + row < p.output_size) {
      const float sum = simd_sum(result[row]);
      if (lane == 0) {
        const bfloat value = bfloat(sum);
        if (!finite(sum) || !finite(float(value)))
          atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
        output[route * p.output_size + out_row + row] = value;
      }
    }
  }
}

template <ushort Bits, ushort GroupSize>
inline void project(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    const device long *expert_ids, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashAffineParams &p,
    uint3 group, uint simd_group, uint lane) {
  static_assert(Bits == 4 || Bits == 5 || Bits == 6 || Bits == 8);
  static_assert(GroupSize == 64 || GroupSize == 128);
  if (!p.rows || !p.selections || !p.experts || !p.output_size ||
      !p.input_size || p.bits != Bits || p.group_size != GroupSize ||
      p.input_size % GroupSize || (p.flags & ~3u) ||
      ((p.flags & 2u) && !(p.flags & 1u)) ||
      (!(p.flags & 1u) && (p.experts != 1 || p.selections != 1)) ||
      p.weight_row_stride_bytes < ulong(p.input_size) * Bits / 8 ||
      p.parameter_row_stride_bytes < ulong(p.input_size / GroupSize) * 2 ||
      p.parameter_row_stride_bytes % 2 ||
      p.parameter_expert_stride_bytes % 2) {
    if (lane == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint out_row = group.x * 8 + simd_group * 4;
  if (out_row >= p.output_size || group.y >= p.rows ||
      group.z >= p.selections) return;
  const ulong route = ulong(group.y) * p.selections + group.z;
  const long expert = (p.flags & 1u) ? expert_ids[route] : 0;
  if (expert < 0 || ulong(expert) >= p.experts) {
    if (lane == 0) {
      atomic_fetch_or_explicit(diagnostics, 1u, memory_order_relaxed);
      for (ushort row = 0; row < 4; ++row)
        if (out_row + row < p.output_size)
          output[route * p.output_size + out_row + row] =
              bfloat(as_type<float>(0x7fc00000u));
    }
    return;
  }
  const ulong input_row = (p.flags & 2u) ? route : group.y;
  const device bfloat *x = input + input_row * p.input_size;
  constexpr uint FastBlock = pack_factor_32<Bits>() * 2 * 32;
  if (p.output_size >= 8 && p.input_size % FastBlock == 0) {
    project_math<Bits, GroupSize, true>(
        x, weights, scales, biases, output, diagnostics, p, out_row,
        uint(expert), route, lane);
  } else {
    project_math<Bits, GroupSize, false>(
        x, weights, scales, biases, output, diagnostics, p, out_row,
        uint(expert), route, lane);
  }
}

} // namespace splash_mlx_qmv_f32xsum_v1

// Existing seven-buffer projection ABI; FlashAffineParams remains at7.
// Launch64 threads (two SIMD groups) and {ceil(N/8), rows, selections}.
#define FLASH_AFFINE_MLX_QMV_V1(BITS, GROUP) \
kernel void flash_affine_mlx_qmv_f32xsum_v1_q##BITS##_g##GROUP( \
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
  splash_mlx_qmv_f32xsum_v1::project<BITS, GROUP>(input, weights, scales, biases, \
      expert_ids, output, diagnostics, p, group, simd_group, lane); \
}

FLASH_AFFINE_MLX_QMV_V1(4, 64)
FLASH_AFFINE_MLX_QMV_V1(4, 128)
FLASH_AFFINE_MLX_QMV_V1(5, 64)
FLASH_AFFINE_MLX_QMV_V1(5, 128)
FLASH_AFFINE_MLX_QMV_V1(6, 64)
FLASH_AFFINE_MLX_QMV_V1(6, 128)
FLASH_AFFINE_MLX_QMV_V1(8, 64)
FLASH_AFFINE_MLX_QMV_V1(8, 128)
#undef FLASH_AFFINE_MLX_QMV_V1
