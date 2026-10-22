// Video VAE encoder (MiniMax-H3 AutoencoderKLLegacy encoder, MLX port video_vae.Encoder3d): causal 3D convolutions,
// TIsolatedGroupNorm + SiLU, Downsample3d, quant_conv. fp16 weights and conv inputs, fp32 accumulation, fp32 residual
// stream and statistics.
//
// Conv inputs are fp16 NHWC frames padded to (H + 2) x Wp, Wp = W + 2, stored flat (frame stride FS pixels). A 3 x 3
// conv is 9 shifted GEMMs over a flat row domain (as kernels/preview/conv.metal): stride 1, output pixel (y, x) is
// row r = y Wp + x and tap (dy, dx) reads input row r + dy Wp + dx; stride 2 (Downsample3d), the same r = y Wp + x
// and the tap reads input row 2 r + dy Wp + dx = (2 y + dy) Wp + 2 x + dx (row stride 2 Cin). Columns x >= Wout of
// the domain are garbage and never stored. The causal temporal taps (kd = 0, 1, 2 over frames t s - 2 + kd; zero frames before 0)
// skip whole threadgroups when the source frame is padding.
//   vae_enc_conv3 / vae_enc_down / vae_enc_conv1   3 x 3 stride 1, 3 x 3 stride 2, 1 x 1 (centre of the padded frame)
//   vae_enc_stats      TIsolatedGroupNorm statistics (two passes: mean, then mean squared deviation), per frame / group
//   vae_enc_pad        fp32 -> fp16 padded producer (GroupNorm + SiLU, reflect padding, channel zero-padding)
//   vae_enc_quant      quant_conv
#include "metal/abi/video/VaeEncoderABI.h"
#include "metal/kernels/video/common/tensor.h"

using namespace h3;

// Row-major [rows][k] operand with row stride `stride` elements.
template <class T>
inline dmat<T> mks(device T *p, int k, int rows, int stride) {
  return dmat<T>(p, dextents<int, 2>(k, rows), array<int, 2>{1, stride});
}

template <int TM, int TN, int NSG, int STRIDE, int TAPS>
kernel void vae_enc_conv(device half *X [[buffer(0)]], device half *Wt [[buffer(1)]],
                         device const float *bias [[buffer(2)]], device float *Y [[buffer(3)]],
                         device const float *R [[buffer(4)]], constant VaeEncConvParams &p [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]]) {
  const int n0 = tg.x * TN, m0 = tg.y * TM;
  const int item = int(tg.z), b = item / p.Dout, t = item - b * p.Dout;
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  const long wsz = long(p.CoutPad) * p.Cin;
  auto a0 = mks<half>(X, p.Cin, p.Mrows, STRIDE * p.Cin).template slice<dynamic_extent, TM>(0, m0);
  auto b0 = mk<half>(Wt, p.Cin, p.CoutPad).template slice<dynamic_extent, TN>(0, n0);
  auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  for (int kd = 0; kd < p.KD; ++kd) {
    const int src = p.KD == 1 ? t : t * p.tstride + kd - 2;
    if (src < 0) continue;  // causal zero frame (uniform per threadgroup)
    device half *xs = X + long(b * p.Din + src) * p.FSin * p.Cin;
    for (int tap = 0; tap < TAPS; ++tap) {
      const int off = TAPS == 9 ? (tap / 3) * p.Wp + (tap % 3) : p.Wp + 1;
      auto a = mks<half>(xs + long(off) * p.Cin, p.Cin, p.Mrows, STRIDE * p.Cin).template slice<dynamic_extent, TM>(0, m0);
      auto w = mk<half>(Wt + long(kd * TAPS + tap) * wsz, p.Cin, p.CoutPad).template slice<dynamic_extent, TN>(0, n0);
      op.run(a, w, c);
    }
  }
  const long obase = long(item) * p.Hout * p.Wout;
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i) {
    if (!c.is_valid_element(i)) continue;
    auto q = c.get_multidimensional_index(i);
    const int r = m0 + q[1], col = n0 + q[0];
    if (r >= p.Mv || col >= p.Cout) continue;
    const int y = r / p.Wq, x = r - y * p.Wq;
    if (x >= p.Wout) continue;
    const long o = (obase + long(y) * p.Wout + x) * p.Cout + col;
    float v = c[i] + bias[col];
    if (p.residual) v += R[o];
    Y[o] = v;
  }
}

#define CONV_ARGS                                                                                         \
  device half *, device half *, device const float *, device float *, device const float *,              \
      constant VaeEncConvParams &, uint3
#define ENC_CONV(TM, TN, NSG, STRIDE, TAPS, NAME) \
  template [[host_name(NAME "_" #TM "x" #TN "x" #NSG)]] [[kernel]] void vae_enc_conv<TM, TN, NSG, STRIDE, TAPS>(CONV_ARGS);
#define ENC_TILE(TM, TN, NSG)                     \
  ENC_CONV(TM, TN, NSG, 1, 9, "vae_enc_conv3")    \
  ENC_CONV(TM, TN, NSG, 2, 9, "vae_enc_down")     \
  ENC_CONV(TM, TN, NSG, 1, 1, "vae_enc_conv1")
ENC_TILE(64, 128, 4)
ENC_TILE(32, 128, 4)
ENC_TILE(32, 64, 2)

// grid (32 groups, frames), 256 threads
kernel void vae_enc_stats(device const float *X [[buffer(0)]], device float *stats [[buffer(1)]],
                          constant VaeEncStatsParams &p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  const int g = int(tg.x), cg = p.C / H3_VAE_ENC_GROUPS, n = p.HW * cg;
  device const float *x = X + long(tg.y) * p.HW * p.C + g * cg;
  auto reduce = [&](float v) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) s += red[i];
    return s;
  };
  float s = 0.0f;
  for (int i = tid; i < n; i += 256) {
    const int px = i / cg, ch = i - px * cg;
    s += x[long(px) * p.C + ch];
  }
  const float mean = reduce(s) / float(n);
  float s2 = 0.0f;
  for (int i = tid; i < n; i += 256) {
    const int px = i / cg, ch = i - px * cg;
    const float d = x[long(px) * p.C + ch] - mean;
    s2 += d * d;
  }
  const float var = reduce(s2) / float(n);
  if (tid == 0) {
    stats[(long(tg.y) * H3_VAE_ENC_GROUPS + g) * 2] = mean;
    stats[(long(tg.y) * H3_VAE_ENC_GROUPS + g) * 2 + 1] = rsqrt(var + p.eps);
  }
}

// one thread per (padded pixel, 4 channels); grid (ceil(FS Cdst / 4 / 256), frames), 256 threads
kernel void vae_enc_pad(device const float *X [[buffer(0)]], device half *P [[buffer(1)]],
                        device const float *stats [[buffer(2)]], device const float *gamma [[buffer(3)]],
                        device const float *beta [[buffer(4)]], constant VaeEncPadParams &p [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  const int q = (p.Cdst >> 2), idx = int(tg.x) * 256 + tid;
  if (idx >= p.FS * q) return;
  const int px = idx / q, c0 = (idx - px * q) * 4;
  const int Y = px / p.Wp, Xc = px - Y * p.Wp;
  const long frame = tg.y;
  int sy, sx;
  bool valid;
  if (p.mode == 0) {  // reflect pad 1: padded (Y, X) <- source (Y - 1, X - 1) reflected into [0, H) x [0, W)
    sy = Y - 1;
    sx = Xc - 1;
    sy = sy < 0 ? -sy : (sy >= p.H ? 2 * p.H - 2 - sy : sy);
    sx = sx < 0 ? -sx : (sx >= p.W ? 2 * p.W - 2 - sx : sx);
    valid = Y < p.H + 2;
  } else {  // data at (0, 0), row H / column W reflected (index n -> n - 2)
    sy = Y < p.H ? Y : p.H - 2;
    sx = Xc < p.W ? Xc : p.W - 2;
    valid = Y <= p.H && Xc <= p.W;
  }
  half4 out = half4(0.0h);
  if (valid) {
    device const float *src = X + ((frame * p.H + sy) * p.W + sx) * p.Csrc;
    float v[4];
    for (int k = 0; k < 4; ++k) {
      const int c = c0 + k;
      float x = c < p.Csrc ? src[c] : 0.0f;
      if (p.norm && c < p.Csrc) {
        const int g = c / (p.Csrc / H3_VAE_ENC_GROUPS);
        device const float *st = stats + (frame * H3_VAE_ENC_GROUPS + g) * 2;
        x = (x - st[0]) * st[1];
        x = x * gamma[c] + beta[c];
        x = x / (1.0f + exp(-x));  // SiLU
      }
      v[k] = x;
    }
    out = half4(v[0], v[1], v[2], v[3]);
  }
  *(device half4 *)(P + (frame * p.FS + px) * p.Cdst + c0) = out;
}

// grid (ceil(pixels * 48 / 256)), 256 threads: out[px][o] = b[o] + sum_k W[o][k] in[px][k]
kernel void vae_enc_quant(device const float *X [[buffer(0)]], device const float *W [[buffer(1)]],
                          device const float *bq [[buffer(2)]], device float *Y [[buffer(3)]],
                          constant VaeEncQuantParams &p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const int px = int(gid) / H3_VAE_ENC_MOMENTS, o = int(gid) - px * H3_VAE_ENC_MOMENTS;
  if (px >= p.pixels) return;
  device const float *x = X + long(px) * H3_VAE_ENC_MOMENTS;
  device const float *w = W + o * H3_VAE_ENC_MOMENTS;
  float acc = 0.0f;
  for (int k = 0; k < H3_VAE_ENC_MOMENTS; ++k) acc = fma(x[k], w[k], acc);
  Y[long(px) * H3_VAE_ENC_MOMENTS + o] = acc + bq[o];
}
