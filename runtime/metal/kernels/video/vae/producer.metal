// Video VAE producers: [RMSNorm | LayerNorm | none] (fp32) -> fp16, or -> Hadamard(128) + per-token int8.
// One threadgroup per row; a thread holds 8 consecutive channels of NGP 128-groups (threads = 16 K / 128 / NGP).
//   vae_producer_rms_f16   RMSNorm -> fp16 (linears in fp16)
//   vae_producer_rms_i8    RMSNorm -> Hadamard -> int8 (W8A8 linears)
//   vae_producer_ln_f16    LayerNorm -> fp16 (proj_out)
//   vae_producer_i8        fp16 attention output (K = 2048) -> Hadamard -> int8
//   vae_producer_ffn_i8    fp16 SwiGLU output (K = 8192) -> Hadamard -> int8
#include "metal/abi/video/VaeABI.h"
#include "metal/kernels/video/common/quant.h"

using namespace h3;

enum : int { NORM_NONE = 0, NORM_RMS = 1, NORM_LN = 2 };

template <class TI, int NORM, bool HAD, bool I8, int NGP>
kernel void vae_producer(device const TI *X [[buffer(0)]], device char *Q [[buffer(1)]], device float *Sc [[buffer(2)]],
                         device half *Y [[buffer(3)]], device const float *nw [[buffer(4)]],
                         device const float *nb [[buffer(5)]], device const char *signs [[buffer(6)]],
                         constant VaeParams &p [[buffer(7)]], uint row [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[32];
  if (int(row) >= p.M) return;
  const ushort halves = nt / 16, hh = tid / 16, j = tid & 15, nsg = nt / 32;
  device const TI *x = X + ulong(row) * p.K;
  float4 va[NGP], vb[NGP];
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    int c = (hh + i * halves) * 128 + 8 * j;
    va[i] = ld4(x + c);
    vb[i] = ld4(x + c + 4);
  }
  auto block_sum = [&](float s) {
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (ushort k = 0; k < nsg; ++k) tot += red[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return tot;
  };
  if (NORM == NORM_LN) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NGP; ++i) s += (va[i].x + va[i].y + va[i].z + va[i].w) + (vb[i].x + vb[i].y + vb[i].z + vb[i].w);
    const float mu = block_sum(s) / float(p.K);
#pragma unroll
    for (int i = 0; i < NGP; ++i) { va[i] -= mu; vb[i] -= mu; }
  }
  if (NORM != NORM_NONE) {
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < NGP; ++i) ss += dot(va[i], va[i]) + dot(vb[i], vb[i]);
    const float rinv = rsqrt(block_sum(ss) / float(p.K) + p.eps);
#pragma unroll
    for (int i = 0; i < NGP; ++i) {
      int c = (hh + i * halves) * 128 + 8 * j;
      va[i] = va[i] * rinv * ld4(nw + c);
      vb[i] = vb[i] * rinv * ld4(nw + c + 4);
      if (NORM == NORM_LN) {
        va[i] += ld4(nb + c);
        vb[i] += ld4(nb + c + 4);
      }
    }
  }
  if (!I8) {
    device half *y = Y + ulong(row) * p.K;
#pragma unroll
    for (int i = 0; i < NGP; ++i) {
      int c = (hh + i * halves) * 128 + 8 * j;
      *(device half4 *)(y + c) = half4(va[i]);
      *(device half4 *)(y + c + 4) = half4(vb[i]);
    }
    return;
  }
  if (HAD) {
#pragma unroll
    for (int i = 0; i < NGP; ++i) {
      int c = (hh + i * halves) * 128 + 8 * j;
      va[i] *= ld4s(signs + c);
      vb[i] *= ld4s(signs + c + 4);
      wht128(va[i], vb[i], lane);
    }
  }
  const float post = HAD ? kRs128 : 1.0f;
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
  if (tid == 0) Sc[row] = am / kQMax * post;
}

#define PRODUCER_ARGS(TI)                                                                                       \
  device const TI *, device char *, device float *, device half *, device const float *, device const float *, \
      device const char *, constant VaeParams &, uint, ushort, ushort, ushort, ushort
#define PRODUCER(TI, NORM, HAD, I8, NGP, NAME) \
  template [[host_name(NAME)]] [[kernel]] void vae_producer<TI, NORM, HAD, I8, NGP>(PRODUCER_ARGS(TI));
PRODUCER(float, NORM_RMS, false, false, 2, "vae_producer_rms_f16")
PRODUCER(float, NORM_LN, false, false, 2, "vae_producer_ln_f16")
PRODUCER(float, NORM_RMS, true, true, 2, "vae_producer_rms_i8")
PRODUCER(half, NORM_NONE, true, true, 2, "vae_producer_i8")
PRODUCER(half, NORM_NONE, true, true, 4, "vae_producer_ffn_i8")
