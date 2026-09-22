// Isolated sparse immutable expert cache. Full cached jobs use whole-K BF16
// tensors; misses/partial jobs preserve FlashMoEBlocked's Q4x8 K64 producer.
// No production Forward or profile selects this candidate.
#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashExpertDenseCache.h"
#include "metal/abi/FlashMoEBuckets.h"
#pragma METAL fp math_mode(safe)
#include "metal/kernels/common/flash_affine_mpp_common.h"

using namespace metal;
using namespace mpp::tensor_ops;

#pragma METAL fp math_mode(fast)
inline bfloat flash_sparse_expert_q4x8_compiled_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

inline bool flash_sparse_expert_q4x8_strides(uint n, uint k, ulong wr, ulong we,
                                   ulong pr, ulong pe) {
  constexpr ulong top = ~ulong(0);
  if (wr < k / 2 || wr % 4 || we % 4 || pr < k / 32 || pr % 2 || pe % 2 ||
      wr > (top - k / 2) / (n - 1) || pr > (top - k / 32) / (n - 1)) return false;
  const ulong wm = ulong(n - 1) * wr + k / 2;
  const ulong pm = ulong(n - 1) * pr + k / 32;
  return we >= wm && pe >= pm && we <= (top - wm) / 511 && pe <= (top - pm) / 511;
}

inline bool flash_sparse_expert_q4x8_job(device const uint *offsets,
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
inline void flash_sparse_expert_q4x8_stage8(device const uchar *weights,
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

template <ushort M>
inline void flash_sparse_expert_q4x8_gate_up_tile(
    device const bfloat *input, device const uchar *gate_w,
    device const uchar *gate_s, device const uchar *gate_b,
    device const uchar *up_w, device const uchar *up_s, device const uchar *up_b,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device bfloat *output, device uint *diagnostics,
    constant FlashMoEBlockedGateParams &params, uint3 group, uint3 threads,
    uint tid, threadgroup bfloat *staged_a, threadgroup bfloat *staged_gate,
    threadgroup bfloat *staged_up) {
  const constant FlashMoEFusedParams &p = params.affine;
  if (!p.rows || p.rows > 8192 || !p.selections || p.selections > 10 ||
      p.input_size != 2560 || p.output_size != 640 || p.experts != 512 ||
      p.reserved0 || p.reserved1 || p.reserved2 || params.reserved ||
      params.route_capacity != p.rows * p.selections || params.tile_rows != M ||
      params.job_capacity != (params.route_capacity + M - 1) / M + 511 ||
      threads.x != 128 || threads.y != 1 || threads.z != 1 || group.x >= 10 ||
      group.y >= params.job_capacity || group.z ||
      !flash_sparse_expert_q4x8_strides(640, 2560, p.gate_weight_row_stride_bytes,
          p.gate_weight_expert_stride_bytes, p.gate_parameter_row_stride_bytes,
          p.gate_parameter_expert_stride_bytes) ||
      !flash_sparse_expert_q4x8_strides(640, 2560, p.up_weight_row_stride_bytes,
          p.up_weight_expert_stride_bytes, p.up_parameter_row_stride_bytes,
          p.up_parameter_expert_stride_bytes)) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!flash_sparse_expert_q4x8_job(offsets, jobs, job_count, group.y, params.job_capacity,
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
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto gate_acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(g0), float>();
  auto up_acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(u0), float>();
#pragma unroll
  for (ushort i = 0; i < gate_acc.get_capacity(); ++i)
    if (gate_acc.is_valid_element(i)) { gate_acc[i] = 0.0f; up_acc[i] = 0.0f; }
  for (uint chunk = 0; chunk < 40; ++chunk) {
    const uint korigin = chunk * BK;
    for (uint i = tid; i < uint(M) * BK; i += 128) {
      const uint m = i / BK, k = i % BK;
      bfloat value = bfloat(0.0f);
      if (begin + m < end) {
        value = input[ulong(begin + m) * 2560 + korigin + k];
        if (!flash_mpp_finite(value)) { flash_mpp_error(diagnostics, 4u); value = bfloat(0.0f); }
      }
      staged_a[i] = value;
    }
    for (uint i = tid; i < uint(N) * BK / 8; i += 128) {
      const uint n = norigin + i / (BK / 8), k = korigin + i % (BK / 8) * 8;
      flash_sparse_expert_q4x8_stage8(gate_w, gate_s, gate_b,
          p.gate_weight_row_stride_bytes, p.gate_weight_expert_stride_bytes,
          p.gate_parameter_row_stride_bytes, p.gate_parameter_expert_stride_bytes,
          expert, n, k, staged_gate + i * 8, diagnostics);
      flash_sparse_expert_q4x8_stage8(up_w, up_s, up_b,
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
    const bfloat sigmoid = flash_sparse_expert_q4x8_compiled_sigmoid(gate);
    const bfloat silu = gate * sigmoid;
    const bfloat value = silu * up;
    if (!flash_mpp_finite(gate_acc[i]) || !flash_mpp_finite(up_acc[i]) ||
        !flash_mpp_finite(gate) || !flash_mpp_finite(up) || !flash_mpp_finite(value))
      flash_mpp_error(diagnostics, 4u);
    output[ulong(row) * 640 + n] = value;
  }
}

template <ushort M>
inline void flash_sparse_expert_q4x8_down_tile(
    device const bfloat *input, device const uchar *weights,
    device const uchar *scales, device const uchar *biases,
    device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *job_count, device const uint *route_map,
    device bfloat *output, device uint *diagnostics,
    constant FlashMoEBlockedDownParams &params, uint3 group, uint3 threads,
    uint tid, threadgroup bfloat *staged_a, threadgroup bfloat *staged_b) {
  const constant FlashMoEDownFusedParams &p = params.affine;
  if (!p.rows || p.rows > 8192 || !p.selections || p.selections > 10 ||
      p.input_size != 640 || p.output_size != 2560 || p.experts != 512 ||
      p.reserved0 || p.reserved1 || p.reserved2 || params.reserved ||
      params.route_capacity != p.rows * p.selections || params.tile_rows != M ||
      params.job_capacity != (params.route_capacity + M - 1) / M + 511 ||
      threads.x != 128 || threads.y != 1 || threads.z != 1 || group.x >= 40 ||
      group.y >= params.job_capacity || group.z ||
      !flash_sparse_expert_q4x8_strides(2560, 640, p.weight_row_stride_bytes,
          p.weight_expert_stride_bytes, p.parameter_row_stride_bytes,
          p.parameter_expert_stride_bytes)) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u); return;
  }
  uint expert = 0, begin = 0, end = 0;
  if (!flash_sparse_expert_q4x8_job(offsets, jobs, job_count, group.y, params.job_capacity,
                          params.route_capacity, expert, begin, end, diagnostics)) return;
  constexpr ushort N = 64, BK = 64;
  const uint norigin = group.x * N;
  auto a = tensor(staged_a, dextents<int, 2>{BK, M}, array<int, 2>{1, BK});
  auto b = tensor(staged_b, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto a0 = a.template slice<BK, M>(0, 0);
  auto b0 = b.template slice<BK, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  for (uint chunk = 0; chunk < 10; ++chunk) {
    const uint korigin = chunk * BK;
    for (uint i = tid; i < uint(M) * BK; i += 128) {
      const uint m = i / BK, k = i % BK;
      bfloat value = bfloat(0.0f);
      if (begin + m < end) {
        value = input[ulong(begin + m) * 640 + korigin + k];
        if (!flash_mpp_finite(value)) { flash_mpp_error(diagnostics, 4u); value = bfloat(0.0f); }
      }
      staged_a[i] = value;
    }
    for (uint i = tid; i < uint(N) * BK / 8; i += 128) {
      const uint n = norigin + i / (BK / 8), k = korigin + i % (BK / 8) * 8;
      flash_sparse_expert_q4x8_stage8(weights, scales, biases,
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


// Startup-only conversion. Each thread shares one packed U32 and BF16 affine
// coefficient pair across eight codes from the same G64 quantization group.
kernel void flash_expert_cache_convert_q4x8(
    device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]],
    device const uchar *b [[buffer(2)]], device const uint *selected [[buffer(3)]],
    device bfloat *output [[buffer(4)]], device uint *diag [[buffer(5)]],
    constant FlashExpertCacheConvertParams &params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  constant auto &p = params.affine;
  if (p.rows != 1 || p.selections != 1 || p.experts != 512 || p.bits != 4 ||
      p.group_size != 64 || p.flags || params.reserved0 || params.reserved1 || params.reserved2 ||
      !params.cached_experts || params.cached_experts > 128 ||
      !((p.output_size == 640 && p.input_size == 2560) ||
        (p.output_size == 2560 && p.input_size == 640)) ||
      !flash_sparse_expert_q4x8_strides(p.output_size, p.input_size, p.weight_row_stride_bytes,
          p.weight_expert_stride_bytes, p.parameter_row_stride_bytes, p.parameter_expert_stride_bytes) ||
      threads.x != 256 || threads.y != 1 || threads.z != 1 || group.z ||
      group.y >= params.cached_experts) {
    if (!tid) flash_mpp_error(diag, 2u); return;
  }
  const uint expert = selected[group.y];
  if (expert >= 512) { if (!tid) flash_mpp_error(diag, 1u); return; }
  const ulong index = (ulong(group.x) * 256 + tid) * 8;
  const ulong matrix = ulong(p.output_size) * p.input_size;
  if (index >= matrix) return;
  const uint n = uint(index / p.input_size), k = uint(index % p.input_size);
  const uint codes = *reinterpret_cast<device const uint *>(w + ulong(expert) * p.weight_expert_stride_bytes +
      ulong(n) * p.weight_row_stride_bytes + k / 2);
  const ulong param = ulong(expert) * p.parameter_expert_stride_bytes +
      ulong(n) * p.parameter_row_stride_bytes + ulong(k / 64) * 2;
  const bfloat scale = *reinterpret_cast<device const bfloat *>(s + param);
  const bfloat bias = *reinterpret_cast<device const bfloat *>(b + param);
#pragma unroll
  for (ushort j = 0; j < 8; ++j) {
    const float reconstructed = flash_mpp_dequantize_f32((codes >> (j * 4)) & 15u, scale, bias);
    const bfloat value = bfloat(reconstructed);
    if (!flash_mpp_finite(reconstructed) || !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(group.y) * matrix + index + j] = value;
  }
}

inline bool flash_expert_cache_rank(device const uint *ranks, uint expert, uint count,
                                    thread uint &rank, device uint *diag) {
  rank = ranks[expert];
  if (rank != UINT_MAX && rank >= count) { flash_mpp_error(diag, 2u); return false; }
  return true;
}
template<ushort M>
inline bool flash_expert_cache_select(device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *jobCount, device const uint *ranks,
    uint job, uint capacity, uint routes, uint count, uint3 group, uint3 threads,
    thread uint &expert, thread uint &begin, thread uint &end, thread uint &rank, device uint *diag) {
  if (!count || count > 128 || group.z || threads.x != 128 || threads.y != 1 || threads.z != 1) {
    flash_mpp_error(diag, 2u); return false;
  }
  if (!flash_sparse_expert_q4x8_job(offsets, jobs, jobCount, job, capacity, routes, expert, begin, end, diag)) return false;
  return flash_expert_cache_rank(ranks, expert, count, rank, diag);
}

// Hit and miss predicates partition each valid job exactly once. Whole-K
// device input tensors are created only for full rows inside this bucket.
template<ushort M>
inline void flash_expert_cache_gate_hit(device bfloat *input, device bfloat *gate,
    device bfloat *up, device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *jobCount,
    device bfloat *output, device uint *diag, constant FlashExpertCacheGateParams &params,
    uint3 group, uint3 threads, uint tid) {
  constant auto &p = params.blocked;
  if (params.reserved0 || params.reserved1 || params.reserved2 || p.reserved ||
      p.tile_rows != M || !p.affine.rows || p.affine.rows > 8192 ||
      !p.affine.selections || p.affine.selections > 10 || p.affine.input_size != 2560 ||
      p.affine.output_size != 640 || p.affine.experts != 512 || p.affine.reserved0 ||
      p.affine.reserved1 || p.affine.reserved2 || p.route_capacity != p.affine.rows * p.affine.selections ||
      p.job_capacity != (p.route_capacity + M - 1) / M + 511 || group.x >= 10 || group.y >= p.job_capacity) {
    if (!tid) flash_mpp_error(diag, 2u); return;
  }
  uint expert, begin, end, rank;
  if (!flash_expert_cache_select<M>(offsets, jobs, jobCount, ranks, group.y, p.job_capacity,
      p.route_capacity, params.cached_experts, group, threads, expert, begin, end, rank, diag)) return;
  if (rank == UINT_MAX || end - begin < M) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  auto a = tensor(input + ulong(begin) * 2560, dextents<int, 2>{2560, M}, array<int, 2>{1, 2560});
  auto g = tensor(gate + (ulong(rank) * 640 + column) * 2560,
      dextents<int, 2>{2560, N}, array<int, 2>{1, 2560});
  auto u = tensor(up + (ulong(rank) * 640 + column) * 2560,
      dextents<int, 2>{2560, N}, array<int, 2>{1, 2560});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto gd = operation.template get_destination_cooperative_tensor<decltype(a), decltype(g), float>();
  auto ud = operation.template get_destination_cooperative_tensor<decltype(a), decltype(u), float>();
  operation.run(a, g, gd); operation.run(a, u, ud);
#pragma unroll
  for (ushort i = 0; i < gd.get_capacity(); ++i) {
    if (!gd.is_valid_element(i)) continue;
    const auto index = gd.get_multidimensional_index(i);
    const bfloat gv = bfloat(gd[i]), uv = bfloat(ud[i]);
    const bfloat silu = gv * flash_sparse_expert_q4x8_compiled_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!flash_mpp_finite(gd[i]) || !flash_mpp_finite(ud[i]) || !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + index[1]) * 640 + column + index[0]] = value;
  }
}

template<ushort M>
inline void flash_expert_cache_down_hit(device bfloat *input, device bfloat *weights,
    device const uint *ranks, device const uint *offsets, device const FlashMoEBucketJob *jobs,
    device const uint *jobCount, device const uint *routeMap, device bfloat *output,
    device uint *diag, constant FlashExpertCacheDownParams &params, uint3 group, uint3 threads, uint tid) {
  constant auto &p = params.blocked;
  if (params.reserved0 || params.reserved1 || params.reserved2 || p.reserved || p.tile_rows != M ||
      !p.affine.rows || p.affine.rows > 8192 || !p.affine.selections || p.affine.selections > 10 ||
      p.affine.input_size != 640 || p.affine.output_size != 2560 || p.affine.experts != 512 ||
      p.affine.reserved0 || p.affine.reserved1 || p.affine.reserved2 ||
      p.route_capacity != p.affine.rows * p.affine.selections ||
      p.job_capacity != (p.route_capacity + M - 1) / M + 511 || group.x >= 40 || group.y >= p.job_capacity) {
    if (!tid) flash_mpp_error(diag, 2u); return;
  }
  uint expert, begin, end, rank;
  if (!flash_expert_cache_select<M>(offsets, jobs, jobCount, ranks, group.y, p.job_capacity,
      p.route_capacity, params.cached_experts, group, threads, expert, begin, end, rank, diag)) return;
  if (rank == UINT_MAX || end - begin < M) return;
  constexpr ushort N = 64;
  const uint column = group.x * N;
  auto a = tensor(input + ulong(begin) * 640, dextents<int, 2>{640, M}, array<int, 2>{1, 640});
  auto b = tensor(weights + (ulong(rank) * 2560 + column) * 640,
      dextents<int, 2>{640, N}, array<int, 2>{1, 640});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const uint route = routeMap[begin + index[1]];
    if (route >= p.route_capacity) { flash_mpp_error(diag, 1u); continue; }
    const bfloat value = bfloat(dot[i]);
    if (!flash_mpp_finite(dot[i]) || !flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(route) * 2560 + column + index[0]] = value;
  }
}

#define FLASH_EXPERT_GATE_HIT(NAME, M) \
kernel void NAME(device bfloat *input [[buffer(0)]], device bfloat *gate [[buffer(1)]], \
    device bfloat *up [[buffer(2)]], device const uint *ranks [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], \
    device const uint *count [[buffer(6)]], device bfloat *output [[buffer(7)]], device uint *diag [[buffer(8)]], \
    constant FlashExpertCacheGateParams &p [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  flash_expert_cache_gate_hit<M>(input, gate, up, ranks, offsets, jobs, count, output, diag, p, group, threads, tid); \
}
#define FLASH_EXPERT_DOWN_HIT(NAME, M) \
kernel void NAME(device bfloat *input [[buffer(0)]], device bfloat *weights [[buffer(1)]], \
    device const uint *ranks [[buffer(2)]], device const uint *offsets [[buffer(3)]], \
    device const FlashMoEBucketJob *jobs [[buffer(4)]], device const uint *count [[buffer(5)]], \
    device const uint *map [[buffer(6)]], device bfloat *output [[buffer(7)]], device uint *diag [[buffer(8)]], \
    constant FlashExpertCacheDownParams &p [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  flash_expert_cache_down_hit<M>(input, weights, ranks, offsets, jobs, count, map, output, diag, p, group, threads, tid); \
}
#define FLASH_EXPERT_GATE_MISS(NAME, M) \
kernel void NAME(device const bfloat *input [[buffer(0)]], device const uchar *gw [[buffer(1)]], \
    device const uchar *gs [[buffer(2)]], device const uchar *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]], \
    device const uchar *us [[buffer(5)]], device const uchar *ub [[buffer(6)]], device const uint *offsets [[buffer(7)]], \
    device const FlashMoEBucketJob *jobs [[buffer(8)]], device const uint *count [[buffer(9)]], \
    device bfloat *output [[buffer(10)]], device uint *diag [[buffer(11)]], device const uint *ranks [[buffer(12)]], \
    constant FlashExpertCacheGateParams &p [[buffer(13)]], uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  if (p.reserved0 || p.reserved1 || p.reserved2) { if (!tid) flash_mpp_error(diag, 2u); return; } \
  uint expert, begin, end, rank; \
  if (!flash_expert_cache_select<M>(offsets, jobs, count, ranks, group.y, p.blocked.job_capacity, \
      p.blocked.route_capacity, p.cached_experts, group, threads, expert, begin, end, rank, diag)) return; \
  if (rank != UINT_MAX && end - begin >= M) return; \
  alignas(16) threadgroup bfloat a[M * 64], g[64 * 64], u[64 * 64]; \
  flash_sparse_expert_q4x8_gate_up_tile<M>(input, gw, gs, gb, uw, us, ub, offsets, jobs, count, output, diag, \
      p.blocked, group, threads, tid, a, g, u); \
}
#define FLASH_EXPERT_DOWN_MISS(NAME, M) \
kernel void NAME(device const bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]], \
    device const uchar *s [[buffer(2)]], device const uchar *b [[buffer(3)]], device const uint *offsets [[buffer(4)]], \
    device const FlashMoEBucketJob *jobs [[buffer(5)]], device const uint *count [[buffer(6)]], \
    device const uint *map [[buffer(7)]], device bfloat *output [[buffer(8)]], device uint *diag [[buffer(9)]], \
    device const uint *ranks [[buffer(10)]], constant FlashExpertCacheDownParams &p [[buffer(11)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  if (p.reserved0 || p.reserved1 || p.reserved2) { if (!tid) flash_mpp_error(diag, 2u); return; } \
  uint expert, begin, end, rank; \
  if (!flash_expert_cache_select<M>(offsets, jobs, count, ranks, group.y, p.blocked.job_capacity, \
      p.blocked.route_capacity, p.cached_experts, group, threads, expert, begin, end, rank, diag)) return; \
  if (rank != UINT_MAX && end - begin >= M) return; \
  alignas(16) threadgroup bfloat a[M * 64], stage[64 * 64]; \
  flash_sparse_expert_q4x8_down_tile<M>(input, w, s, b, offsets, jobs, count, map, output, diag, \
      p.blocked, group, threads, tid, a, stage); \
}
FLASH_EXPERT_GATE_HIT(flash_expert_cache_gate_up_hit_m8_n64, 8)
FLASH_EXPERT_GATE_HIT(flash_expert_cache_gate_up_hit_m16_n64, 16)
FLASH_EXPERT_GATE_HIT(flash_expert_cache_gate_up_hit_m32_n64, 32)
FLASH_EXPERT_DOWN_HIT(flash_expert_cache_down_hit_m8_n64, 8)
FLASH_EXPERT_DOWN_HIT(flash_expert_cache_down_hit_m16_n64, 16)
FLASH_EXPERT_DOWN_HIT(flash_expert_cache_down_hit_m32_n64, 32)
FLASH_EXPERT_GATE_MISS(flash_expert_cache_gate_up_miss_m8_n64, 8)
FLASH_EXPERT_GATE_MISS(flash_expert_cache_gate_up_miss_m16_n64, 16)
FLASH_EXPERT_GATE_MISS(flash_expert_cache_gate_up_miss_m32_n64, 32)
FLASH_EXPERT_DOWN_MISS(flash_expert_cache_down_miss_m8_n64, 8)
FLASH_EXPERT_DOWN_MISS(flash_expert_cache_down_miss_m16_n64, 16)
FLASH_EXPERT_DOWN_MISS(flash_expert_cache_down_miss_m32_n64, 32)
#undef FLASH_EXPERT_GATE_HIT
#undef FLASH_EXPERT_DOWN_HIT
#undef FLASH_EXPERT_GATE_MISS
#undef FLASH_EXPERT_DOWN_MISS
#endif
