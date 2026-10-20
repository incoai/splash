// DiT pieces outside the 50-block loop that run every step (fp32 math, bf16 storage where the reference stores bf16).
//   dit_io_embed  packed residual X [M, 5376] bf16, rows [text L | audio Na | video Nv]:
//                 text rows: the refined text stream (computed once per request by prep);
//                 audio / video rows: audio_patch_proj / video_patch_proj (fp32 weights [K][5376] transposed + bias),
//                 input rounded to bf16 first, output rounded to bf16
//   dit_io_final  final_layer: RMSNorm -> x * (1 + scale[t]) + shift[t] with the bf16 rounding of the MLX reference op
//                 by op -> fp32 heads (video_out [96, 5376] / audio_out [32, 5376] + bias); audio and video rows only
#include <metal_stdlib>
#include "metal/abi/video/DitIoABI.h"

using namespace metal;

// grid (5376 / 256, M), 256 threads
kernel void dit_io_embed(device bfloat *X [[buffer(0)]], device const bfloat *text [[buffer(1)]],
                         device const float *ain [[buffer(2)]], device const float *vin [[buffer(3)]],
                         device const float *WaT [[buffer(4)]], device const float *ba [[buffer(5)]],
                         device const float *WvT [[buffer(6)]], device const float *bv [[buffer(7)]],
                         constant DitIoParams &p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float xin[128];
  const int row = int(tg.y), n = int(tg.x) * 256 + tid;
  if (row < p.L) {  // uniform per threadgroup
    if (n < p.D) X[ulong(row) * p.D + n] = text[ulong(row) * p.D + n];
    return;
  }
  const bool aud = row < p.L + p.Na;
  const int K = aud ? p.Ka : p.Kv;
  device const float *src = aud ? ain + ulong(row - p.L) * p.Ka : vin + ulong(row - p.L - p.Na) * p.Kv;
  if (tid < K) xin[tid] = float(bfloat(src[tid]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (n >= p.D) return;
  device const float *W = aud ? WaT : WvT;  // transposed [K][D] for coalesced reads
  float acc = aud ? ba[n] : bv[n];
  for (int k = 0; k < K; ++k) acc = fma(xin[k], W[k * p.D + n], acc);
  X[ulong(row) * p.D + n] = bfloat(acc);
}

// One threadgroup (256 threads = 8 simdgroups) per audio / video row.
kernel void dit_io_final(device const bfloat *X [[buffer(0)]], device const uchar *mod [[buffer(1)]],
                         device const bfloat *fnw [[buffer(2)]], device const bfloat *fada [[buffer(3)]],
                         device const float *Wv [[buffer(4)]], device const float *bv [[buffer(5)]],
                         device const float *Wa [[buffer(6)]], device const float *ba [[buffer(7)]],
                         device float *vout [[buffer(8)]], device float *aout [[buffer(9)]],
                         constant DitIoParams &p [[buffer(10)]], uint tg [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float y[H3_DIT_HIDDEN];
  threadgroup float red[8];
  const int row = p.L + int(tg);
  device const bfloat *x = X + ulong(row) * p.D;
  float ss = 0.0f;
  for (int k = tid; k < p.D; k += 256) {
    float v = float(x[k]);
    y[k] = v;
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float rinv = rsqrt(tot / float(p.D) + p.eps);
  device const bfloat *sh = fada + ulong(mod[row]) * 2 * p.D, *sc = sh + p.D;
  for (int k = tid; k < p.D; k += 256) {
    float nrm = float(bfloat(y[k] * rinv * float(fnw[k])));             // rms_norm output (bf16)
    float t = float(bfloat(nrm * float(bfloat(1.0f + float(sc[k])))));  // * (1 + scale), bf16
    y[k] = float(bfloat(t + float(sh[k])));                             // + shift, bf16
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool aud = row < p.L + p.Na;
  const int N = aud ? p.Ka : p.Kv;
  device const float *W = aud ? Wa : Wv;
  device const float *b = aud ? ba : bv;
  device float *o = aud ? aout + ulong(row - p.L) * p.Ka : vout + ulong(row - p.L - p.Na) * p.Kv;
  for (int j = sg; j < N; j += 8) {
    device const float *w = W + ulong(j) * p.D;
    float acc = 0.0f;
    for (int k = lane; k < p.D; k += 32) acc = fma(y[k], w[k], acc);
    acc = simd_sum(acc);
    if (lane == 0) o[j] = acc + b[j];
  }
}
