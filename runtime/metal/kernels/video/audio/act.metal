// Audio VAE decoder (BigVGAN) elementwise stages: latent de-normalisation, the anti-aliased SnakeBeta activation
// (Activation1d: UpSample1d(2, 12) -> SnakeBeta -> DownSample1d(2, 12), edge-replicate padding as alias-free-torch),
// the AMP-block average and the zeroing of padding rows. fp32 arithmetic; conv inputs leave as fp16 hi / lo pairs.
#include <metal_stdlib>
#include "metal/abi/video/AudioABI.h"

using namespace metal;

inline void put2(device half *Yh, device half *Yl, long idx, float v) {
  const half h = half(v);
  Yh[idx] = h;
  Yl[idx] = half(v - float(h));
}

// Packed audio rows [2 n][32] (channel-major) -> x * std + mean -> hi / lo padded [2][P + n + P][32], pads zero.
kernel void audio_input(device const float *rows [[buffer(0)]], device const float *mean [[buffer(1)]],
                        device const float *stdv [[buffer(2)]], device half *Yh [[buffer(3)]],
                        device half *Yl [[buffer(4)]], constant AudioActParams &p [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, r = gid.y, Tp = p.T + 2 * H3_AUDIO_PAD;
  if (c >= H3_AUDIO_LATENT || r >= 2 * Tp) return;
  const int b = r / Tp, t = r - b * Tp - H3_AUDIO_PAD;
  float v = 0.0f;
  if (t >= 0 && t < p.T) {
    v = rows[(long(b) * p.T + t) * H3_AUDIO_LATENT + c] * stdv[c];
    v = v + mean[c];
  }
  put2(Yh, Yl, long(r) * H3_AUDIO_LATENT + c, v);
}

// Zero the padding rows of a hi / lo padded buffer [2][P + T + P][C] (before a GEMM writes its valid rows).
kernel void audio_zero_pads(device half *Yh [[buffer(0)]], device half *Yl [[buffer(1)]],
                            constant AudioActParams &p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, y = gid.y;
  if (c >= p.C || y >= 4 * H3_AUDIO_PAD) return;
  const int b = y / (2 * H3_AUDIO_PAD), k = y - b * 2 * H3_AUDIO_PAD;
  const long row = long(b) * (p.T + 2 * H3_AUDIO_PAD) + (k < H3_AUDIO_PAD ? k : p.T + k);
  Yh[row * p.C + c] = 0.0h;
  Yl[row * p.C + c] = 0.0h;
}

// Activation1d(SnakeBeta) on fp32 [2][T][C] -> hi / lo padded [2][P + T + P][C] (pads written as zeros).
//   u[m] = 2 sum_t f[t] x[clamp((m + 15 - t) / 2 - 5)]   (zero-stuffed 2x transposed conv over the edge-padded x,
//                                                       cropped 15 / 15; taps with m + 15 - t even)
//   z[m] = u + b sin(a u)^2,   a = exp(alpha), b = 1 / (exp(beta) + 1e-9)
//   y[j] = sum_t f[t] z[clamp(2 j + t - 5, 0, 2 T - 1)]
// One threadgroup: CT channels x ACT_ROWS output rows; x rows and z values staged in threadgroup memory.
template <int CT>
kernel void audio_act(device const float *X [[buffer(0)]], device half *Yh [[buffer(1)]], device half *Yl [[buffer(2)]],
                      device const float *snake [[buffer(3)]], device const float *filt [[buffer(4)]],
                      constant AudioActParams &p [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
                      ushort lid [[thread_index_in_threadgroup]]) {
  constexpr int NR = 256 / CT, JT = H3_AUDIO_ACT_ROWS, XR = JT + 11, ZR = 2 * JT + 10;
  threadgroup float xs[XR * CT];
  threadgroup float zs[ZR * CT];
  const int c = lid % CT, rg = lid / CT;
  const int ch = int(tg.x) * CT + c, j0 = int(tg.y) * JT, b = tg.z, T = p.T, C = p.C;
  device const float *xb = X + long(b) * T * C;
  for (int i = rg; i < XR; i += NR) xs[i * CT + c] = xb[long(clamp(j0 - 6 + i, 0, T - 1)) * C + ch];
  float f[H3_AUDIO_FILTER];
  for (int t = 0; t < H3_AUDIO_FILTER; ++t) f[t] = filt[t];
  const float a = snake[ch], binv = snake[C + ch];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int mi = rg; mi < ZR; mi += NR) {
    const int mc = clamp(2 * j0 - 5 + mi, 0, 2 * T - 1);
    float s = 0.0f;
    for (int t = (mc + 15) & 1; t < H3_AUDIO_FILTER; t += 2) {
      const int xr = clamp(((mc + 15 - t) >> 1) - 5, 0, T - 1);
      s += f[t] * xs[(xr - (j0 - 6)) * CT + c];
    }
    const float u = 2.0f * s;
    const float sn = precise::sin(a * u);
    zs[mi * CT + c] = u + binv * (sn * sn);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const long base = long(b) * (T + 2 * H3_AUDIO_PAD);
  for (int ji = rg; ji < JT; ji += NR) {
    const int j = j0 + ji;
    if (j >= T) break;
    float o = 0.0f;
    for (int t = 0; t < H3_AUDIO_FILTER; ++t) o += f[t] * zs[(2 * ji + t) * CT + c];
    put2(Yh, Yl, (base + H3_AUDIO_PAD + j) * C + ch, o);
  }
  if (j0 == 0)
    for (int k = rg; k < H3_AUDIO_PAD; k += NR) put2(Yh, Yl, (base + k) * C + ch, 0.0f);
  if (j0 + JT >= T)
    for (int k = rg; k < H3_AUDIO_PAD; k += NR) put2(Yh, Yl, (base + H3_AUDIO_PAD + T + k) * C + ch, 0.0f);
}

template [[host_name("audio_act_32")]] [[kernel]] void audio_act<32>(
    device const float *, device half *, device half *, device const float *, device const float *,
    constant AudioActParams &, uint3, ushort);
template [[host_name("audio_act_16")]] [[kernel]] void audio_act<16>(
    device const float *, device half *, device half *, device const float *, device const float *,
    constant AudioActParams &, uint3, ushort);
template [[host_name("audio_act_8")]] [[kernel]] void audio_act<8>(
    device const float *, device half *, device half *, device const float *, device const float *,
    constant AudioActParams &, uint3, ushort);

// AMP-block average ((b0 + b1) + b2) / 3 of fp32 [2][T][C]: HALF2 -> hi / lo padded (next upsampler input, pads
// zero), else fp32 [2][T][C] (input of activation_post).
template <bool HALF2>
kernel void audio_sum3(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
                       device const float *D [[buffer(2)]], device float *Y [[buffer(3)]],
                       device half *Yh [[buffer(4)]], device half *Yl [[buffer(5)]],
                       constant AudioActParams &p [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, r = gid.y, C = p.C;
  if (c >= C) return;
  if (HALF2) {
    const int Tp = p.T + 2 * H3_AUDIO_PAD;
    if (r >= 2 * Tp) return;
    const int b = r / Tp, t = r - b * Tp - H3_AUDIO_PAD;
    float v = 0.0f;
    if (t >= 0 && t < p.T) {
      const long i = (long(b) * p.T + t) * C + c;
      v = ((A[i] + B[i]) + D[i]) / 3.0f;
    }
    put2(Yh, Yl, long(r) * C + c, v);
  } else {
    if (r >= 2 * p.T) return;
    const long i = long(r) * C + c;
    Y[i] = ((A[i] + B[i]) + D[i]) / 3.0f;
  }
}

template [[host_name("audio_sum3_half")]] [[kernel]] void audio_sum3<true>(
    device const float *, device const float *, device const float *, device float *, device half *, device half *,
    constant AudioActParams &, uint2);
template [[host_name("audio_sum3_f32")]] [[kernel]] void audio_sum3<false>(
    device const float *, device const float *, device const float *, device float *, device half *, device half *,
    constant AudioActParams &, uint2);
