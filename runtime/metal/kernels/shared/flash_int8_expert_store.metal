// Private offline signed INT8 expert candidate: original BF16 activations,
// genuine signed INT8 matrix operands, F32 accumulation, one F32 row scale.
// Opt-in only after separate primitive, model-quality and HTTP qualification.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/kernels/common/flash_moe_direct_a_common.h"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

#pragma METAL fp math_mode(fast)
inline bfloat flash_i8_store_miss_q4x8_compiled_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

inline bool flash_i8_store_miss_q4x8_strides(uint n, uint k, ulong wr, ulong we,
                                   ulong pr, ulong pe) {
  constexpr ulong top = ~ulong(0);
  if (wr < k / 2 || wr % 4 || we % 4 || pr < k / 32 || pr % 2 || pe % 2 ||
      wr > (top - k / 2) / (n - 1) || pr > (top - k / 32) / (n - 1)) return false;
  const ulong wm = ulong(n - 1) * wr + k / 2;
  const ulong pm = ulong(n - 1) * pr + k / 32;
  return we >= wm && pe >= pm && we <= (top - wm) / 511 && pe <= (top - pm) / 511;
}

inline bool flash_i8_store_miss_q4x8_job(device const uint *offsets,
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
inline void flash_i8_store_miss_q4x8_stage8(device const uchar *weights,
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
inline void flash_i8_store_miss_q4x8_gate_up_tile(
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
      !flash_i8_store_miss_q4x8_strides(640, 2560, p.gate_weight_row_stride_bytes,
          p.gate_weight_expert_stride_bytes, p.gate_parameter_row_stride_bytes,
          p.gate_parameter_expert_stride_bytes) ||
      !flash_i8_store_miss_q4x8_strides(640, 2560, p.up_weight_row_stride_bytes,
          p.up_weight_expert_stride_bytes, p.up_parameter_row_stride_bytes,
          p.up_parameter_expert_stride_bytes)) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!flash_i8_store_miss_q4x8_job(offsets, jobs, job_count, group.y, params.job_capacity,
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
      flash_i8_store_miss_q4x8_stage8(gate_w, gate_s, gate_b,
          p.gate_weight_row_stride_bytes, p.gate_weight_expert_stride_bytes,
          p.gate_parameter_row_stride_bytes, p.gate_parameter_expert_stride_bytes,
          expert, n, k, staged_gate + i * 8, diagnostics);
      flash_i8_store_miss_q4x8_stage8(up_w, up_s, up_b,
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
    const bfloat sigmoid = flash_i8_store_miss_q4x8_compiled_sigmoid(gate);
    const bfloat silu = gate * sigmoid;
    const bfloat value = silu * up;
    if (!flash_mpp_finite(gate_acc[i]) || !flash_mpp_finite(up_acc[i]) ||
        !flash_mpp_finite(gate) || !flash_mpp_finite(up) || !flash_mpp_finite(value))
      flash_mpp_error(diagnostics, 4u);
    output[ulong(row) * 640 + n] = value;
  }
}

template <ushort M, ushort SG = 4>
inline void flash_i8_store_miss_q4x8_down_tile(
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
      !flash_i8_store_miss_q4x8_strides(2560, 640, p.weight_row_stride_bytes,
          p.weight_expert_stride_bytes, p.parameter_row_stride_bytes,
          p.parameter_expert_stride_bytes)) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!flash_i8_store_miss_q4x8_job(offsets, jobs, job_count, group.y, params.job_capacity,
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
      flash_i8_store_miss_q4x8_stage8(weights, scales, biases,
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

#pragma METAL fp math_mode(fast)
inline bfloat int8_expert_store_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <ushort M, ushort SG>
inline bool int8_expert_store_job(constant FlashInt8ExpertStoreParams &p,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device uint *diag, uint3 group, uint3 threads, uint tid,
    thread uint &rank, thread uint &begin, thread uint &valid_rows) {
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections ||
      p.route_capacity != p.rows * p.selections ||
      p.job_capacity != (p.route_capacity + M - 1) / M + 511 ||
      p.tile_rows != M || !p.stored_experts || p.stored_experts > 512 ||
      p.scale_group_size || p.reserved || group.y >= p.job_capacity || group.z ||
      threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  const uint active = job_count[0];
  if (active > p.job_capacity || offsets[512] > p.route_capacity) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  if (group.y >= active) return false;
  const auto job = jobs[group.y];
  if (job.expert >= 512) { if (!tid) flash_mpp_error(diag, 1u); return false; }
  const uint first = offsets[job.expert], end = offsets[job.expert + 1];
  if (first > end || end > p.route_capacity || job.row_begin < first || job.row_begin >= end) {
    if (!tid) flash_mpp_error(diag, 2u); return false;
  }
  rank = ranks[job.expert];
  if (rank == UINT_MAX) return false;
  if (rank >= p.stored_experts) { if (!tid) flash_mpp_error(diag, 1u); return false; }
  begin = job.row_begin;
  valid_rows = min(uint(M), end - begin);
  return true;
}

template <ushort M, ushort SG>
inline void int8_expert_store_gate(device bfloat *input, device int8_t *gate,
    device const float *gate_scale, device int8_t *up, device const float *up_scale,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,
    uint3 group, uint3 threads, uint tid) {
  if (group.x >= 10) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!int8_expert_store_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  // Dynamic row bounds make incomplete bucket tiles safe without staging,
  // copying or reading the next expert's input. MPP masks the tail rows.
  auto a = tensor(input + ulong(begin) * 2560,
      dextents<int, 2>{2560, int(valid_rows)}, array<int, 2>{1, 2560});
  auto g = tensor(gate + (ulong(rank) * 640 + column) * 2560,
      dextents<int, 2>{2560, N}, array<int, 2>{1, 2560});
  auto u = tensor(up + (ulong(rank) * 640 + column) * 2560,
      dextents<int, 2>{2560, N}, array<int, 2>{1, 2560});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(u), float>();
  operation.run(a, g, gd); operation.run(a, u, ud);
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint n = column + index[0];
    const float gs = gate_scale[ulong(rank) * 640 + n];
    const float us = up_scale[ulong(rank) * 640 + n];
    const float gf = gd[i] * gs, uf = ud[i] * us;
    const bfloat gv = bfloat(gf), uv = bfloat(uf);
    const bfloat silu = gv * int8_expert_store_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!(gs > 0.0f) || !(us > 0.0f) || !flash_mpp_finite(gs) ||
        !flash_mpp_finite(us) || !flash_mpp_finite(gf) || !flash_mpp_finite(uf) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + index[1]) * 640 + n] = value;
  }
}

template <ushort M, ushort SG>
inline void int8_expert_store_down(device bfloat *input, device int8_t *weights,
    device const float *scales, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device const uint *route_map, device bfloat *output, device uint *diag,
    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads, uint tid) {
  if (group.x >= 40) { if (!tid) flash_mpp_error(diag, 2u); return; }
  uint rank, begin, valid_rows;
  if (!int8_expert_store_job<M, SG>(p, ranks, offsets, jobs, job_count, diag,
      group, threads, tid, rank, begin, valid_rows)) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  auto a = tensor(input + ulong(begin) * 640,
      dextents<int, 2>{640, int(valid_rows)}, array<int, 2>{1, 640});
  auto b = tensor(weights + (ulong(rank) * 2560 + column) * 640,
      dextents<int, 2>{640, N}, array<int, 2>{1, 640});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint route = route_map[begin + index[1]], n = column + index[0];
    if (route >= p.route_capacity) { flash_mpp_error(diag, 1u); continue; }
    const float scale = scales[ulong(rank) * 2560 + n];
    const float result = dot[i] * scale;
    const bfloat value = bfloat(result);
    if (!(scale > 0.0f) || !flash_mpp_finite(scale) || !flash_mpp_finite(result) ||
        !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(route) * 2560 + n] = value;
  }
}

#define INT8_GATE(NAME, M, SG) \
kernel void NAME(device bfloat *a [[buffer(0)]], device int8_t *g [[buffer(1)]], \
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]], \
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], \
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], \
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], \
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  int8_expert_store_gate<M, SG>(a, g, gs, u, us, ranks, offsets, jobs, count, out, diag, p, group, threads, tid); \
}
#define INT8_DOWN(NAME, M, SG) \
kernel void NAME(device bfloat *a [[buffer(0)]], device int8_t *w [[buffer(1)]], \
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], \
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], \
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  int8_expert_store_down<M, SG>(a, w, s, ranks, offsets, jobs, count, map, out, diag, p, group, threads, tid); \
}
INT8_GATE(flash_int8_expert_store_gate_up_m16_n64, 16, 4)
INT8_GATE(flash_int8_expert_store_gate_up_m32_n64, 32, 4)
INT8_GATE(flash_int8_expert_store_gate_up_m64_n64_sg8, 64, 8)
INT8_DOWN(flash_int8_expert_store_down_scatter_m16_n64, 16, 4)
INT8_DOWN(flash_int8_expert_store_down_scatter_m32_n64, 32, 4)
INT8_DOWN(flash_int8_expert_store_down_scatter_m64_n64_sg8, 64, 8)
#undef INT8_GATE
#undef INT8_DOWN

// Preserve the original finite-input behavior once per packed producer,
// without changing any finite BF16 bit. Only the valid packed rows are read.
kernel void flash_int8_expert_store_sanitize(
    device ushort *values [[buffer(0)]], device const uint *offsets [[buffer(1)]],
    device uint *diag [[buffer(2)]],
    constant FlashInt8ExpertStoreSanitizeParams &p [[buffer(3)]],
    uint index [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if (!p.route_capacity || p.route_capacity > 81920 ||
      (p.width != 640 && p.width != 2560) || p.reserved0 || p.reserved1 || threads != 256 ||
      offsets[512] > p.route_capacity) {
    if (!tid) flash_mpp_error(diag, 2u); return;
  }
  if (ulong(index) >= ulong(offsets[512]) * p.width) return;
  const ushort value = values[index];
  if ((value & 0x7f80u) == 0x7f80u) {
    values[index] = 0;
    flash_mpp_error(diag, 4u);
  }
}

#define STORE_GATE_MISS(NAME, M, SG, DIRECT) \
kernel void NAME(device bfloat *a [[buffer(0)]], device const uchar *gw [[buffer(1)]], \
    device const uchar *gs [[buffer(2)]], device const uchar *gb [[buffer(3)]], \
    device const uchar *uw [[buffer(4)]], device const uchar *us [[buffer(5)]], \
    device const uchar *ub [[buffer(6)]], device const uint *offsets [[buffer(7)]], \
    device const FlashMoEBucketJob *jobs [[buffer(8)]], device const uint *count [[buffer(9)]], \
    device bfloat *out [[buffer(10)]], device uint *diag [[buffer(11)]], \
    device const uint *ranks [[buffer(12)]], constant FlashInt8ExpertStoreGateParams &p [[buffer(13)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  if (!p.stored_experts || p.stored_experts > 128 || p.flags != uint(DIRECT) || p.reserved0 || p.reserved1 || \
      group.y >= p.blocked.job_capacity || group.z) { if (!tid) flash_mpp_error(diag, 2u); return; } \
  uint expert, begin, end; \
  if (!flash_i8_store_miss_q4x8_job(offsets, jobs, count, group.y, p.blocked.job_capacity, \
      p.blocked.route_capacity, expert, begin, end, diag)) return; \
  const uint rank = ranks[expert]; \
  if (rank != UINT_MAX) { if (rank >= p.stored_experts && !tid) flash_mpp_error(diag, 2u); return; } \
  alignas(16) threadgroup bfloat astage[M * 64], g[64 * 64], u[64 * 64]; \
  if constexpr (DIRECT) { \
    flash_direct_a_q4x8_gate_up_tile<M, SG>(a, gw, gs, gb, uw, us, ub, offsets, jobs, count, \
        out, diag, p.blocked, group, threads, tid, g, u); \
  } else { \
    flash_i8_store_miss_q4x8_gate_up_tile<M, SG>(a, gw, gs, gb, uw, us, ub, offsets, jobs, count, \
        out, diag, p.blocked, group, threads, tid, astage, g, u); \
  } \
}
#define STORE_DOWN_MISS(NAME, M, SG, DIRECT) \
kernel void NAME(device bfloat *a [[buffer(0)]], device const uchar *w [[buffer(1)]], \
    device const uchar *s [[buffer(2)]], device const uchar *b [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], \
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], \
    device const uint *ranks [[buffer(10)]], constant FlashInt8ExpertStoreDownParams &p [[buffer(11)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  if (!p.stored_experts || p.stored_experts > 128 || p.flags != uint(DIRECT) || p.reserved0 || p.reserved1 || \
      group.y >= p.blocked.job_capacity || group.z) { if (!tid) flash_mpp_error(diag, 2u); return; } \
  uint expert, begin, end; \
  if (!flash_i8_store_miss_q4x8_job(offsets, jobs, count, group.y, p.blocked.job_capacity, \
      p.blocked.route_capacity, expert, begin, end, diag)) return; \
  const uint rank = ranks[expert]; \
  if (rank != UINT_MAX) { if (rank >= p.stored_experts && !tid) flash_mpp_error(diag, 2u); return; } \
  alignas(16) threadgroup bfloat astage[M * 64], bstage[64 * 64]; \
  if constexpr (DIRECT) { \
    flash_direct_a_q4x8_down_tile<M, SG>(a, w, s, b, offsets, jobs, count, map, \
        out, diag, p.blocked, group, threads, tid, bstage); \
  } else { \
    flash_i8_store_miss_q4x8_down_tile<M, SG>(a, w, s, b, offsets, jobs, count, map, \
        out, diag, p.blocked, group, threads, tid, astage, bstage); \
  } \
}
STORE_GATE_MISS(flash_int8_expert_store_gate_up_miss_staged_m16_n64, 16, 4, false)
STORE_GATE_MISS(flash_int8_expert_store_gate_up_miss_staged_m32_n64, 32, 4, false)
STORE_GATE_MISS(flash_int8_expert_store_gate_up_miss_staged_m64_n64_sg8, 64, 8, false)
STORE_DOWN_MISS(flash_int8_expert_store_down_miss_staged_m16_n64, 16, 4, false)
STORE_DOWN_MISS(flash_int8_expert_store_down_miss_staged_m32_n64, 32, 4, false)
STORE_DOWN_MISS(flash_int8_expert_store_down_miss_staged_m64_n64_sg8, 64, 8, false)
STORE_GATE_MISS(flash_int8_expert_store_gate_up_miss_direct_m16_n64, 16, 4, true)
STORE_GATE_MISS(flash_int8_expert_store_gate_up_miss_direct_m32_n64, 32, 4, true)
STORE_GATE_MISS(flash_int8_expert_store_gate_up_miss_direct_m64_n64_sg8, 64, 8, true)
STORE_DOWN_MISS(flash_int8_expert_store_down_miss_direct_m16_n64, 16, 4, true)
STORE_DOWN_MISS(flash_int8_expert_store_down_miss_direct_m32_n64, 32, 4, true)
STORE_DOWN_MISS(flash_int8_expert_store_down_miss_direct_m64_n64_sg8, 64, 8, true)
#undef STORE_GATE_MISS
#undef STORE_DOWN_MISS
#endif
