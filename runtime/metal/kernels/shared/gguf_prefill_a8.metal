// INT8-activation prefill GEMMs for MLX affine 4- and 5-bit weights in groups of 64 (af4g64, af5g64), Apple10 only.
//
// The bf16 prefill tile (gguf_linear.metal) dequantizes weights to half in threadgroup memory and runs bf16 x half
// matmul2d, about 59 TFLOPS on a 40-core M5 Max. These kernels run int8 x int8 -> int32 on the neural accelerators
// instead: prefill_a8_quantize quantizes each row of the chunk with one scale (and sums each 64-input group of the
// quantized row), and the GEMM feeds the weights' codes straight from the planes into int8 fragments, folding each
// group's affine correction once per 64 inputs in the activations' quantized units:
//
//   y[m][n] = sa[m] * sum_g ( s[n][g] * sum_{k in g} qa[m][k] q[n][k]  +  z[n][g] * sum_{k in g} qa[m][k] )
//
// with w = s * q + z. A port of oMLX's oQ A8 NAX kernel (jundot/omlx, Apache-2.0,
// omlx/custom_kernels/qwen35_prefill/csrc/qwen35_oq_a8_nax.metal, its v8 schedule and row-scale mode) to Splash's
// planes: 16x32x16 fragments fed from registers, no threadgroup staging, two int32 accumulators per simdgroup.
//
// Schedule: in fragment step t (0-3) of a 64-group, lane x (coord.x / 4) supplies four of the 16 inputs it owns,
// positions 16x + 4t .. 16x + 4t + 3. They are 32-group x >> 1, chunk 2 (x & 1) + (t >> 1) of the planes (each
// column's 16 codes of the lane are one aligned uint2 of its 4-bit plane), and within the chunk word the nibbles
// (word >> 4 (t & 1)) & 0x0f0f0f0f: chunk slots {0, 4, 1, 5} or {2, 6, 3, 7} (metal/abi/QuantFormat.h). The
// quantizer writes each 32-group of activations in that order, so both operands load contiguously.
//
// Opt-in (splash serve --prefill-int8, ops::LinearConfig::prefillInt8): int8 activations are lossy, unlike the bf16
// tile's fp32-exact accumulation.
#include "metal/abi/Gguf.h"
#include "metal/abi/GgufPrefillA8.h"
#include "metal/kernels/common/gguf_tile.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// ---------------- activation quantizer: one threadgroup of PREFILL_A8_QUANTIZE_THREADS per row of the chunk.
// The schedule position, within its 32-group, of the input at slot s of the planes: chunk c = s >> 3 is lane half
// c >> 1 and step half c & 1; chunk slot u holds step bit (u >> 1) & 1 and lane input ((u & 1) << 1) | (u >> 2).
inline uint a8_position(uint slot) {
  const uint c = slot >> 3, u = slot & 7;
  return 16 * (c >> 1) + 4 * (2 * (c & 1) + ((u >> 1) & 1)) + (((u & 1) << 1) | (u >> 2));
}

kernel void prefill_a8_quantize(device const bfloat *input [[buffer(0)]], device int8_t *qa [[buffer(1)]],
                                device uchar *sums [[buffer(2)]], constant PrefillA8QuantizeParams &p [[buffer(3)]],
                                uint row [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                                ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort kSimdgroups = PREFILL_A8_QUANTIZE_THREADS / 32;
  if (row >= p.rows) return;
  const uint groups = p.input_size / 64;
  device const bfloat *x = input + ulong(row) * p.input_size;
  device int8_t *q = qa + ulong(row) * p.input_size;
  device float *sa = (device float *)sums;
  device short *ra = (device short *)(sums + ulong(p.storage_rows) * 4);
  threadgroup float partial[kSimdgroups];
  float amax = 0.0f;
  for (uint g = simd_group; g < groups; g += kSimdgroups)
    amax = max(amax, max(fabs(float(x[g * 64 + lane])), fabs(float(x[g * 64 + 32 + lane]))));
  amax = simd_max(amax);
  if (lane == 0) partial[simd_group] = amax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = 0.0f;
#pragma unroll
  for (ushort i = 0; i < kSimdgroups; ++i) row_max = max(row_max, partial[i]);
  const float inv = row_max > 0.0f ? 127.0f / row_max : 0.0f;
  if (simd_group == 0 && lane == 0) sa[row] = row_max > 0.0f ? row_max / 127.0f : 0.0f;
  // Lane l quantizes element l of both 32-groups of the 64-group: slot quant_slot(l).
  const uint position = a8_position(quant_slot(lane));
  for (uint g = simd_group; g < groups; g += kSimdgroups) {
    const int q0 = int(clamp(rint(float(x[g * 64 + lane]) * inv), -127.0f, 127.0f));
    const int q1 = int(clamp(rint(float(x[g * 64 + 32 + lane]) * inv), -127.0f, 127.0f));
    q[g * 64 + position] = int8_t(q0);
    q[g * 64 + 32 + position] = int8_t(q1);
    const int sum = simd_sum(q0 + q1);
    if (lane == 0) ra[ulong(g) * p.storage_rows + row] = short(sum);
  }
}

// ---------------- the GEMM tile.
constant constexpr int kA8FragM = 16, kA8FragN = 32, kA8FragK = 16, kA8TM = 2, kA8Dest = 16;
constant constexpr uint kA8PlaneTile = QUANT_TILE_ROWS;

// This lane's (column, row) in a 16x16 fragment, as MLX's steel/gemm/nax.h BaseNAXFrag::get_coord gives them.
inline short2 a8_coord(ushort lane) {
  const short qid = lane >> 2;
  return short2(((qid & 2) | (lane & 1)) * 4, (qid & 4) | ((lane >> 1) & 3));
}

// Four 5-bit codes: the step's nibbles and the fifth bits of chunk byte `b` at slots {0, 4, 1, 5} (or, shifted by
// two, {2, 6, 3, 7}) moved to bit 4 of each byte.
inline uint a8_fifth(uint b) { return ((b & 1u) | ((b & 0x10u) << 4) | ((b & 2u) << 15) | ((b & 0x20u) << 19)) << 4; }

template <int BITS, int WM, GgufEpilogue Ep>
inline void a8_tile(device const int8_t *qa, device const uchar *sums, device const uchar *w0, device const uchar *w1,
                    device const uchar *meta, device bfloat *output, device const bfloat *aux,
                    constant PrefillA8Params &p, uint2 tid, ushort lane, ushort simd_group) {
  using namespace mpp::tensor_ops;
  constexpr int WN = PREFILL_A8_WN, BM = kA8TM * kA8FragM * WM, BN = kA8FragN * WN;
  const int K = int(p.input_size), M = int(p.rows), groups = K / 64, groups32 = K / 32;
  const int sg_m = simd_group % WM, sg_n = simd_group / WM;
  const int row_base = int(tid.y) * BM + sg_m * (kA8TM * kA8FragM);
  const int col_base = int(tid.x) * BN + sg_n * kA8FragN;
  const short2 coord = a8_coord(lane);
  const int cx = coord.x >> 2;

  constexpr auto desc = matmul2d_descriptor(kA8FragM, kA8FragN, kA8FragK, false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = matmul2d_descriptor(kA8FragM, kA8FragN, kA8FragK, false, true, false,
                                                matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<desc_set, execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  float cf[kA8TM][kA8Dest];
#pragma unroll
  for (int i = 0; i < kA8TM; ++i)
#pragma unroll
    for (int e = 0; e < kA8Dest; ++e) cf[i][e] = 0.0f;

  // The lane's weight column n_lane + 8 (q & 1) + 16 (q >> 1) for right-operand run q, all in one plane tile (a
  // 64-column tile never straddles a QUANT_TILE_ROWS one): plane0 records of 16 bytes, plane1 of 4, a 32-group
  // kA8PlaneTile records further on. The lane reads bytes 8 (x & 1) .. +7 of 32-group 2g + (x >> 1).
  const uint n_lane = uint(col_base + coord.y);
  const ulong record0 = ulong(n_lane / kA8PlaneTile) * uint(groups32) * kA8PlaneTile + n_lane % kA8PlaneTile;
  device const uchar *wl0 = w0 + record0 * 16 + 8 * (cx & 1);
  device const uchar *wl1 = w1 + record0 * 4 + 2 * (cx & 1);
  const int h32 = cx >> 1;
  // Meta (s, z as bf16 in one uint) of the lane's destination columns n_run + 16h + j: 4 consecutive records.
  const uint n_run = uint(col_base + coord.x);
  device const uint *ml = (device const uint *)(meta) +
                          ulong(n_run / kA8PlaneTile) * uint(groups) * kA8PlaneTile + n_run % kA8PlaneTile;
  const int m_base = row_base + coord.y;
  device const float *sa = (device const float *)sums;
  device const short *ra = (device const short *)(sums + ulong(p.storage_rows) * 4);

  for (int g = 0; g < groups; ++g) {
    const ulong g32 = ulong(2 * g + h32) * kA8PlaneTile;
    uint2 wv[4];
    uint hb[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int off = (q & 1) * 8 + (q >> 1) * 16;
      wv[q] = *((device const uint2 *)(wl0 + (g32 + off) * 16));
      if (BITS == 5) hb[q] = *((device const ushort *)(wl1 + (g32 + off) * 4));
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint codes = (wv[q][t >> 1] >> (4 * (t & 1))) & 0x0f0f0f0fu;
        if (BITS == 5) codes |= a8_fifth((hb[q] >> (8 * (t >> 1) + 2 * (t & 1))) & 0xFFu);
        const char4 quad = as_type<char4>(codes);
        const int base = (q >> 1) * 8 + (q & 1) * 4;
        ct_b[base + 0] = quad.x; ct_b[base + 1] = quad.y; ct_b[base + 2] = quad.z; ct_b[base + 3] = quad.w;
      }
#pragma unroll
      for (int hf = 0; hf < kA8TM; ++hf) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int m = min(m_base + hf * kA8FragM + r * 8, M - 1);
          const char4 quad = as_type<char4>(*((device const uint *)(qa + ulong(m) * K + g * 64 + cx * 16 + t * 4)));
          ct_a[r * 4 + 0] = quad.x; ct_a[r * 4 + 1] = quad.y; ct_a[r * 4 + 2] = quad.z; ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) { if (hf == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
        else { if (hf == 0) op.run(ct_a, ct_b, acc0); else op.run(ct_a, ct_b, acc1); }
      }
    }
    // The group's correction: s, z of the lane's two runs of four columns, and the quantized sums of its rows.
    const uint4 mv0 = *((device const uint4 *)(ml + ulong(g) * kA8PlaneTile));
    const uint4 mv1 = *((device const uint4 *)(ml + ulong(g) * kA8PlaneTile + 16));
    float r_g[kA8TM][2];
#pragma unroll
    for (int i = 0; i < kA8TM; ++i)
#pragma unroll
      for (int r = 0; r < 2; ++r) r_g[i][r] = float(ra[ulong(g) * p.storage_rows + min(m_base + i * kA8FragM + r * 8, M - 1)]);
#pragma unroll
    for (int e = 0; e < kA8Dest; ++e) {
      const int r = (e & 7) >> 2;
      const uint mt = (e >> 3) ? mv1[e & 3] : mv0[e & 3];
      const float s = float(as_type<bfloat>(ushort(mt & 0xFFFF))), z = float(as_type<bfloat>(ushort(mt >> 16)));
      cf[0][e] = fma(s, float(acc0[e]), fma(z, r_g[0][r], cf[0][e]));
      cf[1][e] = fma(s, float(acc1[e]), fma(z, r_g[1][r], cf[1][e]));
    }
  }
#pragma unroll
  for (int i = 0; i < kA8TM; ++i)
#pragma unroll
    for (int e = 0; e < kA8Dest; ++e) {
      const int m = m_base + i * kA8FragM + ((e & 7) >> 2) * 8;
      if (m >= M) continue;
      const ulong o = ulong(m) * p.out_stride + p.out_offset + n_run + (e & 3) + 16 * (e >> 3);
      output[o] = gguf_epilogue<Ep>(sa[m] * cf[i][e], aux, o);
    }
}

// grid (column tiles of PREFILL_A8_COLUMNS, row tiles of PREFILL_A8_ROWS(wm)), PREFILL_A8_THREADS(wm) threads.
#define PREFILL_A8_KERNEL(B, WM, f, ep, Ep)                                                                        \
  kernel void gguf_prefill_a8_##f##_##ep(device const int8_t *qa [[buffer(0)]], device const uchar *sums [[buffer(1)]], \
                                         device const uchar *w0 [[buffer(2)]], device const uchar *w1 [[buffer(3)]], \
                                         device const uchar *meta [[buffer(4)]], device bfloat *output [[buffer(5)]],\
                                         device const bfloat *aux [[buffer(6)]],                                     \
                                         constant PrefillA8Params &p [[buffer(7)]],                                  \
                                         uint2 tid [[threadgroup_position_in_grid]],                                 \
                                         ushort lane [[thread_index_in_simdgroup]],                                  \
                                         ushort simd_group [[simdgroup_index_in_threadgroup]]) {                     \
    a8_tile<B, WM, Ep>(qa, sums, w0, w1, meta, output, aux, p, tid, lane, simd_group);                             \
  }
#define PREFILL_A8_FORMAT(B, WM, f)                                                                                \
  PREFILL_A8_KERNEL(B, WM, f, a, EpNone) PREFILL_A8_KERNEL(B, WM, f, r, EpResidual)                              \
  PREFILL_A8_KERNEL(B, WM, f, g, EpUpWithGate)
PREFILL_A8_FORMAT(4, PREFILL_A8_Q4_WM, af4g64)
PREFILL_A8_FORMAT(5, PREFILL_A8_Q5_WM, af5g64)
#undef PREFILL_A8_FORMAT
#undef PREFILL_A8_KERNEL
