#if defined(SPLASH_INT8_EXPERIMENT) && __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/ExperimentalINT8.h"

#pragma METAL fp math_mode(safe)

using namespace metal;
using namespace mpp::tensor_ops;

// Optional whole-K W8A8 experiment. The host requantizes complete affine Q4
// weights into row-major signed INT8 and one positive FP32 scale per N row.
// Activations have one positive FP32 scale per row. Both code buffers exclude
// -128: K <= 25600 bounds exact INT32 dots by K*127*127 <= 412902400.
//
// Quantization: buffers 0 BF16 input[rows*K], 1 INT8 codes[rows*K],
// 2 FP32 activation scales[rows], 3 uint diagnostics[4], params at 4 (8 bytes).
// row_scale dispatches {rows,1,1}, 256 threads/TG. quantize dispatches
// {ceil(rows*K/256),1,1}, 256 threads/TG. Zero rows use scale 1 and code 0.
//
// Projection: buffers 0 INT8 activations[rows*K], 1 INT8 up weights[N*K],
// 2 FP32 up scales[N], 3 INT8 gate weights[N*K], 4 FP32 gate scales[N],
// 5 FP32 activation scales[rows], 6 BF16 auxiliary[rows*N],
// 7 BF16 output[rows*N], 8 uint diagnostics[4], params at 9 (16 bytes).
// Auxiliary is a residual in mode 1, raw gate projection in mode 3, and an
// unused dummy otherwise. Gate buffers are used only in mode 2.
//
// Projection dispatch: {Xgroups, rows/Rows,1}, 256 threads/TG. Xgroups may be
// N/TileN or a smaller positive persistent grid. Input backing is initialized
// for complete Rows tiles; logical rows must be divisible by the chosen Rows.
// All arenas must hold their declared ranges. Inputs/output do not alias;
// residual/output may alias. Host conversion validates code ranges and backing.
//
// diagnostics[0..2] report N,K,rows from projection group (0,0), thread 0.
// diagnostics[3] is sticky: bit 0 invalid numeric data, bit 1 bad parameters.
// The host clears diagnostics once; every shader failure only atomically ORs
// error bits. The host rejects the graph's results if status becomes nonzero.
// IEEE exponent-bit checks remain explicit under optimization. Projections and
// final outputs are checked after BF16 rounding as well as before rounding.

constant constexpr uint ExperimentalINT8MaxK = 25600;
constant constexpr uint ExperimentalINT8MaxRows = 2048;

__attribute__((always_inline)) inline void experimental_int8_error(
    device uint *diagnostics, uint bits) {
  atomic_fetch_or_explicit(
      reinterpret_cast<device atomic_uint *>(diagnostics + 3), bits,
      memory_order_relaxed);
}

__attribute__((always_inline)) inline bool experimental_int8_finite(float v) {
  return (as_type<uint>(v) & 0x7f800000u) != 0x7f800000u;
}

__attribute__((always_inline)) inline bool experimental_int8_finite(bfloat v) {
  return (as_type<ushort>(v) & 0x7f80u) != 0x7f80u;
}

__attribute__((always_inline)) inline bool experimental_int8_scale(float v) {
  return experimental_int8_finite(v) && v > 0.0f;
}

kernel void experimental_int8_row_scale(
    device const bfloat *input [[buffer(0)]],
    device int8_t *codes [[buffer(1)]],
    device float *scales [[buffer(2)]],
    device uint *diagnostics [[buffer(3)]],
    constant ExperimentalINT8QuantParams &params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  (void)codes;
  if (!params.k || params.k > ExperimentalINT8MaxK || !params.rows ||
      params.rows > ExperimentalINT8MaxRows || row >= params.rows) {
    if (thread_index == 0)
      experimental_int8_error(diagnostics, 2u);
    return;
  }
  threadgroup float peaks[8];
  threadgroup uint invalid[8];
  float peak = 0.0f;
  uint bad = 0;
  for (uint k = thread_index; k < params.k; k += 256) {
    const bfloat input_value = input[ulong(row) * params.k + k];
    if (!experimental_int8_finite(input_value)) {
      bad = 1;
    } else {
      peak = max(peak, abs(float(input_value)));
    }
  }
  peak = simd_max(peak);
  bad = simd_sum(bad);
  if (lane == 0) {
    peaks[simd_group] = peak;
    invalid[simd_group] = bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float row_peak = 0.0f;
    uint row_bad = 0;
    for (uint group = 0; group < 8; ++group) {
      row_peak = max(row_peak, peaks[group]);
      row_bad |= invalid[group];
    }
    const float scale = row_peak > 0.0f ? row_peak / 127.0f : 1.0f;
    if (row_bad || !experimental_int8_scale(scale)) {
      scales[row] = 1.0f;
      experimental_int8_error(diagnostics, 1u);
    } else {
      scales[row] = scale;
    }
  }
}

kernel void experimental_int8_quantize(
    device const bfloat *input [[buffer(0)]],
    device int8_t *codes [[buffer(1)]],
    device const float *scales [[buffer(2)]],
    device uint *diagnostics [[buffer(3)]],
    constant ExperimentalINT8QuantParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (!params.k || params.k > ExperimentalINT8MaxK || !params.rows ||
      params.rows > ExperimentalINT8MaxRows) {
    if (index == 0)
      experimental_int8_error(diagnostics, 2u);
    return;
  }
  if (ulong(index) >= ulong(params.rows) * params.k)
    return;
  const bfloat input_value = input[index];
  const float scale = scales[index / params.k];
  if (!experimental_int8_finite(input_value) || !experimental_int8_scale(scale)) {
    codes[index] = int8_t(0);
    experimental_int8_error(diagnostics, 1u);
    return;
  }
  const float ratio = float(input_value) / scale;
  if (!experimental_int8_finite(ratio)) {
    codes[index] = int8_t(0);
    experimental_int8_error(diagnostics, 1u);
    return;
  }
  codes[index] = int8_t(int(clamp(rint(ratio), -127.0f, 127.0f)));
}

// Parallel activation peaks for the measured small-row profile. Original
// row_scale/quantize remain available for larger rows and other geometries.
// The caller selects Apple family 10, 80 cores, K 5120/K 17408, rows <= 128.
// Shared scratch reserves 128*ceil(25600/512) entries per 4-byte plane. The
// tighter row guard below protects that capacity; all quantization math is
// unchanged.
constant constexpr uint ExperimentalINT8ChunkColumns = 512;
constant constexpr uint ExperimentalINT8ChunkMaxRows = 128;

__attribute__((always_inline)) inline bool
experimental_int8_chunk_params(constant ExperimentalINT8QuantParams &params) {
  return params.k && params.k <= ExperimentalINT8MaxK && params.rows &&
         params.rows <= ExperimentalINT8ChunkMaxRows;
}

// Both passes: input0,codes1,scales2,sticky diagnostics3,peak scratch4,
// invalid scratch5,8-byte ExperimentalINT8QuantParams at6. Scratch layout is
// [row][ceil(K/512)] with no padding. Each stage dispatch is a global fence.
// Both grids={ceil(K/512),rows,1}; TG256/SG32.
kernel void experimental_int8_partial_peak512(
    device const bfloat *input [[buffer(0)]],
    device int8_t *codes [[buffer(1)]], device float *scales [[buffer(2)]],
    device uint *diagnostics [[buffer(3)]],
    device float *partial_peaks [[buffer(4)]],
    device uint *partial_invalid [[buffer(5)]],
    constant ExperimentalINT8QuantParams &params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  (void)codes;
  (void)scales;
  if (!experimental_int8_chunk_params(params) || threads.x != 256 ||
      threads.y != 1 || threads.z != 1 || group.z != 0) {
    if (tid == 0)
      experimental_int8_error(diagnostics, 2u);
    return;
  }
  const uint chunks = (params.k + ExperimentalINT8ChunkColumns - 1) /
                      ExperimentalINT8ChunkColumns;
  if (group.y >= params.rows || group.x >= chunks) {
    if (tid == 0)
      experimental_int8_error(diagnostics, 2u);
    return;
  }
  threadgroup float peaks[8];
  threadgroup uint invalid[8];
  float peak = 0.0f;
  uint bad = 0;
  const uint begin = group.x * ExperimentalINT8ChunkColumns,
             end = min(begin + ExperimentalINT8ChunkColumns, params.k);
  for (uint col = begin + tid; col < end; col += 256) {
    const bfloat value = input[ulong(group.y) * params.k + col];
    if (!experimental_int8_finite(value))
      bad = 1;
    else
      peak = max(peak, abs(float(value)));
  }
  peak = simd_max(peak);
  bad = simd_sum(bad);
  if (lane == 0) {
    peaks[simdgroup] = peak;
    invalid[simdgroup] = bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float chunk_peak = 0.0f;
    uint chunk_bad = 0;
    for (uint i = 0; i < 8; ++i) {
      chunk_peak = max(chunk_peak, peaks[i]);
      chunk_bad |= invalid[i];
    }
    const ulong index = ulong(group.y) * chunks + group.x;
    partial_peaks[index] = chunk_peak;
    partial_invalid[index] = chunk_bad != 0;
    if (chunk_bad)
      experimental_int8_error(diagnostics, 1u);
  }
}

kernel void experimental_int8_reduce_quantize512(
    device const bfloat *input [[buffer(0)]],
    device int8_t *codes [[buffer(1)]], device float *scales [[buffer(2)]],
    device uint *diagnostics [[buffer(3)]],
    device const float *partial_peaks [[buffer(4)]],
    device const uint *partial_invalid [[buffer(5)]],
    constant ExperimentalINT8QuantParams &params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  if (!experimental_int8_chunk_params(params) || threads.x != 256 ||
      threads.y != 1 || threads.z != 1 || group.z != 0) {
    if (tid == 0)
      experimental_int8_error(diagnostics, 2u);
    return;
  }
  if (group.y >= params.rows ||
      group.x >= (params.k + ExperimentalINT8ChunkColumns - 1) /
                     ExperimentalINT8ChunkColumns) {
    if (tid == 0)
      experimental_int8_error(diagnostics, 2u);
    return;
  }
  const uint chunks = (params.k + ExperimentalINT8ChunkColumns - 1) /
                      ExperimentalINT8ChunkColumns;
  // One SIMDgroup reduces the small row list once per quantization TG. A
  // broadcast barrier then gives every thread the exact same scale, without
  // a new row-only global dispatch. Only TG(x=0),thread0 writes scales[row].
  threadgroup float row_peak[1];
  threadgroup uint row_invalid[1];
  if (simdgroup == 0) {
    float peak = 0.0f;
    uint bad = 0;
    for (uint chunk = lane; chunk < chunks; chunk += 32) {
      const ulong i = ulong(group.y) * chunks + chunk;
      const float p = partial_peaks[i];
      const uint invalid = partial_invalid[i];
      if (!experimental_int8_finite(p) || p < 0.0f || invalid)
        bad = 1;
      else
        peak = max(peak, p);
    }
    peak = simd_max(peak);
    bad = simd_sum(bad);
    if (lane == 0) {
      row_peak[0] = peak;
      row_invalid[0] = bad;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float peak = row_peak[0];
  const uint bad = row_invalid[0];
  const float raw_scale = peak > 0.0f ? peak / 127.0f : 1.0f;
  const bool row_bad = bad || !experimental_int8_scale(raw_scale);
  const float scale = row_bad ? 1.0f : raw_scale;
  if (group.x == 0 && tid == 0)
    scales[group.y] = scale;
  if (row_bad && lane == 0)
    experimental_int8_error(diagnostics, 1u);
  const uint begin = group.x * ExperimentalINT8ChunkColumns,
             end = min(begin + ExperimentalINT8ChunkColumns, params.k);
  for (uint column = begin + tid; column < end; column += 256) {
    const ulong index = ulong(group.y) * params.k + column;
    const bfloat input_value = input[index];
    if (!experimental_int8_finite(input_value) ||
        !experimental_int8_scale(scale)) {
      codes[index] = int8_t(0);
      experimental_int8_error(diagnostics, 1u);
      continue;
    }
    const float ratio = float(input_value) / scale;
    if (!experimental_int8_finite(ratio)) {
      codes[index] = int8_t(0);
      experimental_int8_error(diagnostics, 1u);
      continue;
    }
    codes[index] = int8_t(int(clamp(rint(ratio), -127.0f, 127.0f)));
  }
}
__attribute__((always_inline)) inline bool experimental_int8_scaled_dot(
    int dot, int bound, float activation_scale, float weight_scale,
    thread float &rounded) {
  if (dot < -bound || dot > bound ||
      !experimental_int8_scale(activation_scale) ||
      !experimental_int8_scale(weight_scale))
    return false;
  const float product = activation_scale * weight_scale;
  // Positive finite scales can multiply to zero by legitimate FP32 underflow.
  // The resulting zero projection remains valid, including its BF16 rounding.
  if (!experimental_int8_finite(product) || product < 0.0f)
    return false;
  const float value = float(dot) * product;
  if (!experimental_int8_finite(value))
    return false;
  const bfloat projected = bfloat(value);
  if (!experimental_int8_finite(projected))
    return false;
  rounded = float(projected);
  return true;
}

template <bool Fused, class Fragment>
__attribute__((always_inline)) inline void experimental_int8_store(
    const thread Fragment &up, const thread Fragment &gate,
    device const float *up_scales, device const float *gate_scales,
    device const float *activation_scales, device const bfloat *auxiliary,
    device bfloat *output, device uint *diagnostics,
    constant ExperimentalINT8Params &params, uint row_origin,
    uint output_origin) {
  const int bound = int(params.k) * 127 * 127;
#pragma unroll
  for (ushort i = 0; i < up.get_capacity(); ++i) {
    if (!up.is_valid_element(i))
      continue;
    const auto index = up.get_multidimensional_index(i);
    const uint row = row_origin + index[1];
    const uint column = output_origin + index[0];
    const ulong output_index = ulong(row) * params.n + column;
    const float activation_scale = activation_scales[row];
    float up_value = 0.0f;
    float gate_value = 0.0f;
    bool valid = experimental_int8_scaled_dot(
        up[i], bound, activation_scale, up_scales[column], up_value);
    if constexpr (Fused) {
      valid &= experimental_int8_scaled_dot(
          gate[i], bound, activation_scale, gate_scales[column], gate_value);
    }
    if (!valid) {
      experimental_int8_error(diagnostics, 1u);
      continue;
    }
    float value = up_value;
    if constexpr (Fused) {
      value = gate_value /
                  (1.0f + fast::exp2(-1.44269504089f * gate_value)) *
              up_value;
    } else if (params.epilogue == 1 || params.epilogue == 3) {
      const bfloat aux = auxiliary[output_index];
      if (!experimental_int8_finite(aux)) {
        experimental_int8_error(diagnostics, 1u);
        continue;
      }
      if (params.epilogue == 1) {
        value += float(aux);
      } else {
        const float raw_gate = float(aux);
        value = raw_gate /
                    (1.0f + fast::exp2(-1.44269504089f * raw_gate)) *
                up_value;
      }
    }
    const bfloat result = bfloat(value);
    if (!experimental_int8_finite(value) || !experimental_int8_finite(result)) {
      experimental_int8_error(diagnostics, 1u);
      continue;
    }
    output[output_index] = result;
  }
}

template <ushort Rows, ushort TileN>
__attribute__((always_inline)) inline void experimental_int8_project(
    device int8_t *activation_codes, device int8_t *up_weights,
    device const float *up_scales, device int8_t *gate_weights,
    device const float *gate_scales, device const float *activation_scales,
    device const bfloat *auxiliary, device bfloat *output,
    device uint *diagnostics, constant ExperimentalINT8Params &params,
    uint2 group, uint2 groups, uint thread_index) {
  if (group.x == 0 && group.y == 0 && thread_index == 0) {
    diagnostics[0] = params.n;
    diagnostics[1] = params.k;
    diagnostics[2] = params.rows;
  }
  const ulong row_begin = ulong(group.y) * Rows;
  if (!params.n || params.n > 0x7fffffffu || params.n % TileN ||
      !params.k || params.k > ExperimentalINT8MaxK || !params.rows ||
      params.rows > ExperimentalINT8MaxRows || params.rows % Rows ||
      params.epilogue > 3 || !groups.x || groups.x > params.n / TileN ||
      row_begin + Rows > params.rows) {
    if (thread_index == 0)
      experimental_int8_error(diagnostics, 2u);
    return;
  }
  const uint row_origin = uint(row_begin);
  constexpr auto descriptor = matmul2d_descriptor(
      Rows, TileN, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  const int k = int(params.k);
  auto a = tensor(activation_codes + ulong(row_origin) * params.k,
                  dextents<int, 2>{k, Rows}, array<int, 2>{1, k});
  for (uint tile = group.x; tile < params.n / TileN; tile += groups.x) {
    const uint output_origin = tile * TileN;
    auto up = tensor(up_weights + ulong(output_origin) * params.k,
                     dextents<int, 2>{k, TileN}, array<int, 2>{1, k});
    auto up_dot = operation.template get_destination_cooperative_tensor<
        decltype(a), decltype(up), int>();
    operation.run(a, up, up_dot);
    if (params.epilogue == 2) {
      auto gate = tensor(gate_weights + ulong(output_origin) * params.k,
                         dextents<int, 2>{k, TileN}, array<int, 2>{1, k});
      auto gate_dot = operation.template get_destination_cooperative_tensor<
          decltype(a), decltype(gate), int>();
      operation.run(a, gate, gate_dot);
      experimental_int8_store<true>(
          up_dot, gate_dot, up_scales, gate_scales, activation_scales,
          auxiliary, output, diagnostics, params, row_origin, output_origin);
    } else {
      experimental_int8_store<false>(
          up_dot, up_dot, up_scales, gate_scales, activation_scales,
          auxiliary, output, diagnostics, params, row_origin, output_origin);
    }
  }
}

#define EXPERIMENTAL_INT8_PROJECTION(Name, Rows, TileN)                       \
  kernel void Name(                                                        \
      device int8_t *activation_codes [[buffer(0)]],                        \
      device int8_t *up_weights [[buffer(1)]],                              \
      device const float *up_scales [[buffer(2)]],                           \
      device int8_t *gate_weights [[buffer(3)]],                            \
      device const float *gate_scales [[buffer(4)]],                         \
      device const float *activation_scales [[buffer(5)]],                   \
      device const bfloat *auxiliary [[buffer(6)]],                          \
      device bfloat *output [[buffer(7)]],                                  \
      device uint *diagnostics [[buffer(8)]],                              \
      constant ExperimentalINT8Params &params [[buffer(9)]],                \
      uint2 group [[threadgroup_position_in_grid]],                         \
      uint2 groups [[threadgroups_per_grid]],                               \
      uint thread_index [[thread_index_in_threadgroup]]) {                  \
    experimental_int8_project<Rows, TileN>(                                \
        activation_codes, up_weights, up_scales, gate_weights, gate_scales, \
        activation_scales, auxiliary, output, diagnostics, params, group,   \
        groups, thread_index);                                             \
  }

EXPERIMENTAL_INT8_PROJECTION(experimental_int8_decode_m8_n64, 8, 64)
EXPERIMENTAL_INT8_PROJECTION(experimental_int8_decode_m16_n64, 16, 64)
EXPERIMENTAL_INT8_PROJECTION(experimental_int8_decode_m24_n64, 24, 64)
EXPERIMENTAL_INT8_PROJECTION(experimental_int8_decode_m32_n64, 32, 64)
EXPERIMENTAL_INT8_PROJECTION(experimental_int8_prefill_m32_n128, 32, 128)
EXPERIMENTAL_INT8_PROJECTION(experimental_int8_prefill_m128_n128, 128, 128)
EXPERIMENTAL_INT8_PROJECTION(experimental_int8_prefill_m32_n64, 32, 64)
EXPERIMENTAL_INT8_PROJECTION(experimental_int8_prefill_m32_n256, 32, 256)
EXPERIMENTAL_INT8_PROJECTION(experimental_int8_prefill_m256_n64, 256, 64)

#undef EXPERIMENTAL_INT8_PROJECTION
#endif
