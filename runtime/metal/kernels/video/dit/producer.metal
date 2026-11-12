// DiT Hadamard int8 producers: the activation side of every W8A8 linear (Hadamard convention in common/quant.h).
//   dit_producer_norm    RMSNorm + adaLN modulate (per-row modality) + Hadamard(128) + per-token int8, K = 5376
//   dit_producer_plain   Hadamard(128) + per-token int8 of the SwiGLU output h, K = 14336
//   dit_producer_gated   softmax gate (.) attention output -> Hadamard + int8 (out-projection input), K = 7168
//   dit_producer_branch  linear-branch readout of the inner video rows -> Hadamard + int8 (to_out_linear input),
//                        signs 7168..14335 of the 14336-wide sign sequence
// One threadgroup per row; a thread holds 8 consecutive channels of NGP 128-groups (threads = 16 K / 128 / NGP),
// single read of the input row. Output: int8 [row][K] and the row scale amax / 127 / sqrt(128).
#include "metal/abi/video/LinearABI.h"
#include "metal/kernels/video/common/quant.h"

using namespace h3;

//   NORM: y = rmsnorm(x) * w ; m = y * (1 + scale[mod]) + shift[mod]   (fp32, eps = p.eps)
template <bool NORM, int NGP>
kernel void dit_producer(device const bfloat *X [[buffer(0)]], device char *Q [[buffer(1)]],
                         device float *S [[buffer(2)]], device const uchar *mod [[buffer(3)]],
                         device const bfloat *normw [[buffer(4)]], device const bfloat *ada [[buffer(5)]],
                         constant ProducerParams &p [[buffer(6)]], device const char *signs [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort nt [[threads_per_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  if (int(row) >= p.M) return;
  const ushort halves = nt / 16, hh = tid / 16, j = tid & 15, nsg = nt / 32;
  device const bfloat *x = X + ulong(row) * p.K;
  float4 va[NGP], vb[NGP];
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    int c = (hh + i * halves) * 128 + 8 * j;
    va[i] = ld4(x + c);
    vb[i] = ld4(x + c + 4);
  }
  if (NORM) {
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < NGP; ++i) ss += dot(va[i], va[i]) + dot(vb[i], vb[i]);
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (ushort k = 0; k < nsg; ++k) tot += red[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rinv = rsqrt(tot / float(p.K) + p.eps);
    device const bfloat *ad = ada + int(mod[row]) * p.ada_stride;
    device const bfloat *sh = ad + p.shift_off, *sc = ad + p.scale_off;
#pragma unroll
    for (int i = 0; i < NGP; ++i) {
      int c = (hh + i * halves) * 128 + 8 * j;
      va[i] = va[i] * rinv * ld4(normw + c) * (1.0f + ld4(sc + c)) + ld4(sh + c);
      vb[i] = vb[i] * rinv * ld4(normw + c + 4) * (1.0f + ld4(sc + c + 4)) + ld4(sh + c + 4);
    }
  }
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    int c = (hh + i * halves) * 128 + 8 * j;
    va[i] *= ld4s(signs + c);
    vb[i] *= ld4s(signs + c + 4);
    wht128(va[i], vb[i], lane);
  }
  const float post = kRs128;
  float am = 0.0f;
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    float4 m4 = max(fabs(va[i]), fabs(vb[i]));
    am = max(am, max(max(m4.x, m4.y), max(m4.z, m4.w)));
  }
  am = simd_max(am);
  if (lane == 0) red[sg] = am;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  am = red[0];
  for (ushort k = 1; k < nsg; ++k) am = max(am, red[k]);
  const float inv = q8_inv(am);
  device char *q = Q + ulong(row) * p.K;
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    int c = (hh + i * halves) * 128 + 8 * j;
    *(device uint2 *)(q + c) = uint2(q4pack(va[i], inv), q4pack(vb[i], inv));
  }
  if (tid == 0) S[row] = am / kQMax * post;
}

#define PRODUCER_ARGS                                                                                         \
  device const bfloat *, device char *, device float *, device const uchar *, device const bfloat *,          \
      device const bfloat *, constant ProducerParams &, device const char *, uint, ushort, ushort, ushort, ushort
template [[host_name("dit_producer_norm")]] [[kernel]] void
dit_producer<true, H3_PRODUCER_GROUPS_HIDDEN>(PRODUCER_ARGS);
template [[host_name("dit_producer_plain")]] [[kernel]] void dit_producer<false, H3_PRODUCER_GROUPS_FFN>(PRODUCER_ARGS);

// MODE 1: gated attention, rows = all M, group g reads sgate[row][g] * attn[row][128 g ..]
// MODE 2: linear-branch readout, rows = inner video rows, group g reads lin[row][128 g ..] (g < lg)
template <int MODE, int NGP>
kernel void dit_producer_inner(device const bfloat *attn [[buffer(0)]], device const bfloat *lin [[buffer(1)]],
                               device const float *sgate [[buffer(2)]], device char *Q [[buffer(3)]],
                               device float *S [[buffer(4)]], device const char *signs [[buffer(5)]],
                               constant GatedProducerParams &p [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  if (int(row) >= p.M) return;
  const ushort halves = nt / 16, hh = tid / 16, j = tid & 15, nsg = nt / 32;
  float4 va[NGP], vb[NGP];
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    int g = hh + i * halves;
    int c = g * 128 + 8 * j;
    if (MODE == 2) {
      if (g < p.lg) {
        device const bfloat *x = lin + ulong(row) * p.ldl + c;
        va[i] = ld4(x);
        vb[i] = ld4(x + 4);
      } else {
        va[i] = 0.0f;
        vb[i] = 0.0f;
      }
    } else if (g < H3_DIT_HEADS) {
      float s = sgate[ulong(row) * H3_DIT_HEADS + g];
      device const bfloat *x = attn + ulong(row) * H3_DIT_INNER;
      va[i] = ld4(x + c) * s;
      vb[i] = ld4(x + c + 4) * s;
    } else {
      va[i] = 0.0f;
      vb[i] = 0.0f;
    }
    const int cs = c + p.sign_off;
    va[i] *= ld4s(signs + cs);
    vb[i] *= ld4s(signs + cs + 4);
    wht128(va[i], vb[i], lane);
  }
  float am = 0.0f;
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    float4 m4 = max(fabs(va[i]), fabs(vb[i]));
    am = max(am, max(max(m4.x, m4.y), max(m4.z, m4.w)));
  }
  am = simd_max(am);
  if (lane == 0) red[sg] = am;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  am = red[0];
  for (ushort k = 1; k < nsg; ++k) am = max(am, red[k]);
  const float inv = q8_inv(am);
  device char *q = Q + ulong(row) * p.K;
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    int c = (hh + i * halves) * 128 + 8 * j;
    *(device uint2 *)(q + c) = uint2(q4pack(va[i], inv), q4pack(vb[i], inv));
  }
  if (tid == 0) S[row] = am / kQMax * kRs128;
}

#define INNER_ARGS                                                                                       \
  device const bfloat *, device const bfloat *, device const float *, device char *, device float *,     \
      device const char *, constant GatedProducerParams &, uint, ushort, ushort, ushort, ushort
template [[host_name("dit_producer_gated")]] [[kernel]] void
dit_producer_inner<1, H3_PRODUCER_GROUPS_INNER>(INNER_ARGS);
template [[host_name("dit_producer_branch")]] [[kernel]] void
dit_producer_inner<2, H3_PRODUCER_GROUPS_INNER>(INNER_ARGS);
