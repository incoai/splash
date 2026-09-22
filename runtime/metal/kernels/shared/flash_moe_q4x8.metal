// Opt-in prefill route: identical M8/M16/M32 N64 K64 MPP
// geometry, traversal, BF16 stages and stable route jobs to FlashMoEBlocked.
// Only Q4 staging changes: eight nibbles share one U32 and scale/bias loads.
// FlashMoEBlocked selects this only for validated, U32-aligned Q4/G64 sources
// when SPLASH_FLASH_MOE_Q4X8=1; unaligned sources retain scalar staging.
#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"
#pragma METAL fp math_mode(safe)
#include "metal/kernels/common/flash_affine_mpp_common.h"

using namespace metal;
using namespace mpp::tensor_ops;

#pragma METAL fp math_mode(fast)
inline bfloat flash_q4x8_compiled_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

inline bool flash_q4x8_strides(uint n, uint k, ulong wr, ulong we,
                                   ulong pr, ulong pe) {
  constexpr ulong top = ~ulong(0);
  if (wr < k / 2 || wr % 4 || we % 4 || pr < k / 32 || pr % 2 || pe % 2 ||
      wr > (top - k / 2) / (n - 1) || pr > (top - k / 32) / (n - 1)) return false;
  const ulong wm = ulong(n - 1) * wr + k / 2;
  const ulong pm = ulong(n - 1) * pr + k / 32;
  return we >= wm && pe >= pm && we <= (top - wm) / 511 && pe <= (top - pm) / 511;
}

inline bool flash_q4x8_job(device const uint *offsets,
                               device const FlashMoEBucketJob *jobs,
                               device const uint *job_count, uint job,
                               uint capacity, uint routes,
                               thread uint &expert, thread uint &row_begin,
                               thread uint &row_end, device uint *diagnostics) {
  const uint active = job_count[0];
  if (active > capacity || offsets[512] > routes) {
    flash_mpp_error(diagnostics, 2u); return false;
  }
  if (job >= active) return false;
  const FlashMoEBucketJob selected = jobs[job];
  if (selected.expert >= 512) { flash_mpp_error(diagnostics, 1u); return false; }
  const uint begin = offsets[selected.expert], end = offsets[selected.expert + 1];
  if (begin > end || end > routes || selected.row_begin < begin ||
      selected.row_begin >= end) { flash_mpp_error(diagnostics, 2u); return false; }
  expert = selected.expert; row_begin = selected.row_begin; row_end = end;
  return true;
}

// Exactly the original coefficient arithmetic and BF16 operand boundary.
// One aligned U32 contains eight adjacent Q4 codes, all from one G64 group.
// Source row/expert strides are checked as multiples of four before dispatch.
inline void flash_q4x8_stage8(device const uchar *weights,
                              device const uchar *scales,
                              device const uchar *biases,
                              ulong wr, ulong we, ulong pr, ulong pe,
                              uint expert, uint n, uint k,
                              threadgroup bfloat *destination,
                              device uint *diagnostics) {
  const ulong weight = ulong(expert) * we + ulong(n) * wr + ulong(k / 2);
  const uint codes = *reinterpret_cast<device const uint *>(weights + weight);
  const ulong parameter = ulong(expert) * pe + ulong(n) * pr + ulong(k / 64) * 2;
  const bfloat scale = *reinterpret_cast<device const bfloat *>(scales + parameter);
  const bfloat bias = *reinterpret_cast<device const bfloat *>(biases + parameter);
#pragma unroll
  for (ushort j = 0; j < 8; ++j) {
    const float reconstructed = flash_mpp_dequantize_f32((codes >> (j * 4)) & 15u, scale, bias);
    bfloat value = bfloat(reconstructed);
    if (!flash_mpp_finite(reconstructed) || !flash_mpp_finite(value)) {
      flash_mpp_error(diagnostics, 4u); value = bfloat(0.0f);
    }
    destination[j] = value;
  }
}

template <ushort M, ushort SG = 4>
inline void flash_q4x8_gate_up_tile(
    device const bfloat *input, device const uchar *gate_w,
    device const uchar *gate_s, device const uchar *gate_b,
    device const uchar *up_w, device const uchar *up_s, device const uchar *up_b,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device bfloat *output, device uint *diagnostics,
    constant FlashMoEBlockedGateParams &params, uint3 group, uint3 threads,
    uint tid, threadgroup bfloat *staged_a, threadgroup bfloat *staged_gate,
    threadgroup bfloat *staged_up) {
  const constant FlashMoEFusedParams &p = params.affine;
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections ||
      p.input_size != 2560 || p.output_size != 640 || p.experts != 512 ||
      p.reserved0 || p.reserved1 || p.reserved2 || params.reserved ||
      params.route_capacity != p.rows * p.selections || params.tile_rows != M ||
      params.job_capacity != (params.route_capacity + M - 1) / M + 511 ||
      threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1 || group.x >= 10 ||
      group.y >= params.job_capacity || group.z ||
      !flash_q4x8_strides(640, 2560, p.gate_weight_row_stride_bytes,
          p.gate_weight_expert_stride_bytes, p.gate_parameter_row_stride_bytes,
          p.gate_parameter_expert_stride_bytes) ||
      !flash_q4x8_strides(640, 2560, p.up_weight_row_stride_bytes,
          p.up_weight_expert_stride_bytes, p.up_parameter_row_stride_bytes,
          p.up_parameter_expert_stride_bytes)) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!flash_q4x8_job(offsets, jobs, job_count, group.y, params.job_capacity,
                          params.route_capacity, expert, begin, end, diagnostics)) return;
  constexpr ushort N = 64, BK = 64;
  const uint norigin = group.x * N;
  auto a = tensor(staged_a, dextents<int, 2>{BK, M}, array<int, 2>{1, BK});
  auto g = tensor(staged_gate, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto u = tensor(staged_up, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto a0 = a.template slice<BK, M>(0, 0);
  auto g0 = g.template slice<BK, N>(0, 0);
  auto u0 = u.template slice<BK, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto gate_acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(g0), float>();
  auto up_acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(u0), float>();
#pragma unroll
  for (ushort i = 0; i < gate_acc.get_capacity(); ++i)
    if (gate_acc.is_valid_element(i)) { gate_acc[i] = 0.0f; up_acc[i] = 0.0f; }
  for (uint chunk = 0; chunk < 40; ++chunk) {
    const uint korigin = chunk * BK;
    for (uint i = tid; i < uint(M) * BK; i += uint(SG) * 32) {
      const uint m = i / BK, k = i % BK;
      bfloat value = bfloat(0.0f);
      if (begin + m < end) {
        value = input[ulong(begin + m) * 2560 + korigin + k];
        if (!flash_mpp_finite(value)) { flash_mpp_error(diagnostics, 4u); value = bfloat(0.0f); }
      }
      staged_a[i] = value;
    }
    for (uint i = tid; i < uint(N) * BK / 8; i += uint(SG) * 32) {
      const uint n = norigin + i / (BK / 8), k = korigin + i % (BK / 8) * 8;
      flash_q4x8_stage8(gate_w, gate_s, gate_b,
          p.gate_weight_row_stride_bytes, p.gate_weight_expert_stride_bytes,
          p.gate_parameter_row_stride_bytes, p.gate_parameter_expert_stride_bytes,
          expert, n, k, staged_gate + i * 8, diagnostics);
      flash_q4x8_stage8(up_w, up_s, up_b,
          p.up_weight_row_stride_bytes, p.up_weight_expert_stride_bytes,
          p.up_parameter_row_stride_bytes, p.up_parameter_expert_stride_bytes,
          expert, n, k, staged_up + i * 8, diagnostics);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    operation.run(a0, g0, gate_acc);
    operation.run(a0, u0, up_acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < gate_acc.get_capacity(); ++i) {
    if (!gate_acc.is_valid_element(i)) continue;
    const auto index = gate_acc.get_multidimensional_index(i);
    const uint row = begin + index[1], n = norigin + index[0];
    if (row >= end) continue;
    const bfloat gate = bfloat(gate_acc[i]), up = bfloat(up_acc[i]);
    const bfloat sigmoid = flash_q4x8_compiled_sigmoid(gate);
    const bfloat silu = gate * sigmoid;
    const bfloat value = silu * up;
    if (!flash_mpp_finite(gate_acc[i]) || !flash_mpp_finite(up_acc[i]) ||
        !flash_mpp_finite(gate) || !flash_mpp_finite(up) || !flash_mpp_finite(value))
      flash_mpp_error(diagnostics, 4u);
    output[ulong(row) * 640 + n] = value;
  }
}

template <ushort M, ushort SG = 4>
inline void flash_q4x8_down_tile(
    device const bfloat *input, device const uchar *weights,
    device const uchar *scales, device const uchar *biases,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device const uint *route_map,
    device bfloat *output, device uint *diagnostics,
    constant FlashMoEBlockedDownParams &params, uint3 group, uint3 threads,
    uint tid, threadgroup bfloat *staged_a, threadgroup bfloat *staged_b) {
  const constant FlashMoEDownFusedParams &p = params.affine;
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections ||
      p.input_size != 640 || p.output_size != 2560 || p.experts != 512 ||
      p.reserved0 || p.reserved1 || p.reserved2 || params.reserved ||
      params.route_capacity != p.rows * p.selections || params.tile_rows != M ||
      params.job_capacity != (params.route_capacity + M - 1) / M + 511 ||
      threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1 || group.x >= 40 ||
      group.y >= params.job_capacity || group.z ||
      !flash_q4x8_strides(2560, 640, p.weight_row_stride_bytes,
          p.weight_expert_stride_bytes, p.parameter_row_stride_bytes,
          p.parameter_expert_stride_bytes)) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!flash_q4x8_job(offsets, jobs, job_count, group.y, params.job_capacity,
                          params.route_capacity, expert, begin, end, diagnostics)) return;
  constexpr ushort N = 64, BK = 64;
  const uint norigin = group.x * N;
  auto a = tensor(staged_a, dextents<int, 2>{BK, M}, array<int, 2>{1, BK});
  auto b = tensor(staged_b, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto a0 = a.template slice<BK, M>(0, 0);
  auto b0 = b.template slice<BK, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  for (uint chunk = 0; chunk < 10; ++chunk) {
    const uint korigin = chunk * BK;
    for (uint i = tid; i < uint(M) * BK; i += uint(SG) * 32) {
      const uint m = i / BK, k = i % BK;
      bfloat value = bfloat(0.0f);
      if (begin + m < end) {
        value = input[ulong(begin + m) * 640 + korigin + k];
        if (!flash_mpp_finite(value)) { flash_mpp_error(diagnostics, 4u); value = bfloat(0.0f); }
      }
      staged_a[i] = value;
    }
    for (uint i = tid; i < uint(N) * BK / 8; i += uint(SG) * 32) {
      const uint n = norigin + i / (BK / 8), k = korigin + i % (BK / 8) * 8;
      flash_q4x8_stage8(weights, scales, biases,
          p.weight_row_stride_bytes, p.weight_expert_stride_bytes,
          p.parameter_row_stride_bytes, p.parameter_expert_stride_bytes,
          expert, n, k, staged_b + i * 8, diagnostics);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    operation.run(a0, b0, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    const uint row = begin + index[1], n = norigin + index[0];
    if (row >= end) continue;
    const uint route = route_map[row];
    if (route >= params.route_capacity) { flash_mpp_error(diagnostics, 1u); continue; }
    const bfloat value = bfloat(acc[i]);
    if (!flash_mpp_finite(acc[i]) || !flash_mpp_finite(value)) flash_mpp_error(diagnostics, 4u);
    output[ulong(route) * 2560 + n] = value;
  }
}

#define FLASH_Q4X8_GATE(NAME, M, SG)                                        \
kernel void NAME(                                                         \
    device const bfloat *input [[buffer(0)]],                              \
    device const uchar *gw [[buffer(1)]], device const uchar *gs [[buffer(2)]], \
    device const uchar *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]], \
    device const uchar *us [[buffer(5)]], device const uchar *ub [[buffer(6)]], \
    device const uint *offsets [[buffer(7)]],                              \
    device const FlashMoEBucketJob *jobs [[buffer(8)]],                     \
    device const uint *job_count [[buffer(9)]],                            \
    device bfloat *output [[buffer(10)]], device uint *diag [[buffer(11)]], \
    constant FlashMoEBlockedGateParams &p [[buffer(12)]],                   \
    uint3 group [[threadgroup_position_in_grid]],                         \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  alignas(16) threadgroup bfloat a[M * 64], g[64 * 64], u[64 * 64];         \
  flash_q4x8_gate_up_tile<M, SG>(input, gw, gs, gb, uw, us, ub, offsets, jobs, job_count, \
      output, diag, p, group, threads, tid, a, g, u);                       \
}
FLASH_Q4X8_GATE(flash_moe_q4x8_gate_up_m8_n64, 8, 4)
FLASH_Q4X8_GATE(flash_moe_q4x8_gate_up_m16_n64, 16, 4)
FLASH_Q4X8_GATE(flash_moe_q4x8_gate_up_m32_n64, 32, 4)
FLASH_Q4X8_GATE(flash_moe_q4x8_gate_up_m64_n64_sg8, 64, 8)
#undef FLASH_Q4X8_GATE

#define FLASH_Q4X8_DOWN(NAME, M, SG)                                        \
kernel void NAME(                                                         \
    device const bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]], \
    device const uchar *s [[buffer(2)]], device const uchar *b [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]],                              \
    device const FlashMoEBucketJob *jobs [[buffer(5)]],                     \
    device const uint *job_count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *output [[buffer(8)]], device uint *diag [[buffer(9)]],    \
    constant FlashMoEBlockedDownParams &p [[buffer(10)]],                   \
    uint3 group [[threadgroup_position_in_grid]],                         \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  alignas(16) threadgroup bfloat a[M * 64], bstage[64 * 64];               \
  flash_q4x8_down_tile<M, SG>(input, w, s, b, offsets, jobs, job_count, map, output, diag, \
      p, group, threads, tid, a, bstage);                                  \
}
FLASH_Q4X8_DOWN(flash_moe_q4x8_down_scatter_m8_n64, 8, 4)
FLASH_Q4X8_DOWN(flash_moe_q4x8_down_scatter_m16_n64, 16, 4)
FLASH_Q4X8_DOWN(flash_moe_q4x8_down_scatter_m32_n64, 32, 4)
FLASH_Q4X8_DOWN(flash_moe_q4x8_down_scatter_m64_n64_sg8, 64, 8)
#undef FLASH_Q4X8_DOWN

#endif
