// Audio VAE decoder (BigVGAN) convolutions. Implicit GEMM over the padded channels-last layout [2][P + T + P][C]
// (metal/abi/AudioABI.h): a conv with k taps is k row-shifted GEMMs, each split into hi*hi + lo*hi + hi*lo fp16
// matmul2d products accumulated in fp32; GEMM rows are the flat padded rows of both stereo channels (outputs of pad
// rows are dropped). The transposed convs (ups) run per output phase: out[s i0 + phase - upad] = sum_m X[i0 - m]
// W[phase + s m]. Channel counts below 16 (the last stage, its upsampler and conv_post) use the SIMT kernels
// audio_small_*, fp32 throughout.
//   epilogues: E_F32 fp32 [2][Tout][Cout] (+ bias), E_RES (+ bias + residual; may alias), E_HALF2 fp16 hi / lo padded
//   (+ bias), E_WAV (SIMT, Cout 1: clip to [-1, 1] -> waveform [2][Tout]).
#include "metal/abi/video/AudioABI.h"
#include "metal/kernels/video/common/tensor.h"

using namespace h3;

enum : int { A_CONV = 0, A_UPS = 1 };
enum : int { E_F32 = 0, E_RES = 1, E_HALF2 = 2, E_WAV = 3 };

// Output row of a GEMM domain row (input padded layout), or -1 when the row is padding / outside the output.
inline int audio_out_row(int r, int phase, int MODE, constant AudioConvParams &p, thread int &b) {
  const int Tp = p.T + 2 * H3_AUDIO_PAD;
  b = r / Tp;
  if (b >= 2) return -1;
  const int t = r - b * Tp - H3_AUDIO_PAD;
  const int q = MODE == A_CONV ? t : p.stride * t + phase - p.upad;
  if (MODE == A_CONV && (t < 0 || t >= p.T)) return -1;
  return (q >= 0 && q < p.Tout) ? q : -1;
}

template <int TM, int TN, int NSG, int MODE, int EPI>
kernel void audio_conv(device half *Xh [[buffer(0)]], device half *Xl [[buffer(1)]], device half *Wh [[buffer(2)]],
                       device half *Wl [[buffer(3)]], device const float *bias [[buffer(4)]],
                       device float *Y [[buffer(5)]], device const float *R [[buffer(6)]],
                       device half *Yh [[buffer(7)]], device half *Yl [[buffer(8)]],
                       constant AudioConvParams &p [[buffer(9)]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int n0 = tg.x * TN, m0 = tg.y * TM, phase = tg.z;
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  const long wsz = long(p.Cout) * p.Cin;
  auto a0 = mk<half>(Xh, p.Cin, p.Mrows).template slice<dynamic_extent, TM>(0, m0);
  auto b0 = mk<half>(Wh, p.Cin, p.Cout).template slice<dynamic_extent, TN>(0, n0);
  auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  const int ntap = MODE == A_CONV ? p.taps : (p.taps - phase + p.stride - 1) / p.stride;
  for (int j = 0; j < ntap; ++j) {
    const int off = MODE == A_CONV ? j * p.dil - p.pad : -j;
    const int tap = MODE == A_CONV ? j : phase + p.stride * j;
    auto ah = mk<half>(Xh + long(off) * p.Cin, p.Cin, p.Mrows).template slice<dynamic_extent, TM>(0, m0);
    auto al = mk<half>(Xl + long(off) * p.Cin, p.Cin, p.Mrows).template slice<dynamic_extent, TM>(0, m0);
    auto bh = mk<half>(Wh + tap * wsz, p.Cin, p.Cout).template slice<dynamic_extent, TN>(0, n0);
    auto bl = mk<half>(Wl + tap * wsz, p.Cin, p.Cout).template slice<dynamic_extent, TN>(0, n0);
    op.run(ah, bh, c);
    op.run(al, bh, c);
    op.run(ah, bl, c);
  }
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i) {
    if (!c.is_valid_element(i)) continue;
    auto q = c.get_multidimensional_index(i);
    const int col = n0 + q[0];
    int b;
    const int o = audio_out_row(m0 + q[1], phase, MODE, p, b);
    if (o < 0) continue;
    const float v = c[i] + bias[col];
    if (EPI == E_HALF2) {
      const long idx = (long(b) * (p.Tout + 2 * H3_AUDIO_PAD) + H3_AUDIO_PAD + o) * p.Cout + col;
      const half h = half(v);
      Yh[idx] = h;
      Yl[idx] = half(v - float(h));
    } else {
      const long idx = (long(b) * p.Tout + o) * p.Cout + col;
      Y[idx] = EPI == E_RES ? v + R[idx] : v;
    }
  }
}

#define CONV_ARGS                                                                                            \
  device half *, device half *, device half *, device half *, device const float *, device float *,           \
      device const float *, device half *, device half *, constant AudioConvParams &, uint3
#define CONV(TM, TN, NSG, MODE, EPI, NAME) \
  template [[host_name(NAME "_" #TM "x" #TN "x" #NSG)]] [[kernel]] void audio_conv<TM, TN, NSG, MODE, EPI>(CONV_ARGS);
#define CONV_TILE(TM, TN, NSG)                         \
  CONV(TM, TN, NSG, A_CONV, E_F32, "audio_conv_f32")   \
  CONV(TM, TN, NSG, A_CONV, E_RES, "audio_conv_res")   \
  CONV(TM, TN, NSG, A_CONV, E_HALF2, "audio_conv_half") \
  CONV(TM, TN, NSG, A_UPS, E_F32, "audio_ups_f32")
CONV_TILE(64, 128, 4)
CONV_TILE(64, 64, 2)
CONV_TILE(64, 32, 1)
CONV_TILE(64, 16, 1)

// SIMT conv for small channel counts: one thread per GEMM domain row (all COUT outputs), weights hi + lo in fp32 in
// threadgroup memory, activations hi + lo in fp32.
template <int CIN, int COUT, int MAXTAPS, int MODE, int EPI>
kernel void audio_small(device const half *Xh [[buffer(0)]], device const half *Xl [[buffer(1)]],
                        device const half *Wh [[buffer(2)]], device const half *Wl [[buffer(3)]],
                        device const float *bias [[buffer(4)]], device float *Y [[buffer(5)]],
                        device const float *R [[buffer(6)]], constant AudioConvParams &p [[buffer(9)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint3 lid3 [[thread_position_in_threadgroup]],
                        uint3 tpg [[threads_per_threadgroup]]) {
  const int lid = lid3.x, nthreads = tpg.x;
  threadgroup float w[MAXTAPS * COUT * CIN];
  const int nw = p.taps * COUT * CIN;
  for (int i = lid; i < nw; i += nthreads) w[i] = float(Wh[i]) + float(Wl[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int phase = tg.z;
  const int r = int(tg.x) * nthreads + lid;
  if (r >= p.Mrows) return;
  int b;
  const int o = audio_out_row(r, phase, MODE, p, b);
  if (o < 0) return;
  float acc[COUT];
  for (int k = 0; k < COUT; ++k) acc[k] = 0.0f;
  const int ntap = MODE == A_CONV ? p.taps : (p.taps - phase + p.stride - 1) / p.stride;
  for (int j = 0; j < ntap; ++j) {
    const int off = MODE == A_CONV ? j * p.dil - p.pad : -j;
    const int tap = MODE == A_CONV ? j : phase + p.stride * j;
    const long xr = long(r + off) * CIN;
    float x[CIN];
    for (int ci = 0; ci < CIN; ++ci) x[ci] = float(Xh[xr + ci]) + float(Xl[xr + ci]);
    threadgroup const float *wt = w + tap * COUT * CIN;
    for (int k = 0; k < COUT; ++k)
      for (int ci = 0; ci < CIN; ++ci) acc[k] += x[ci] * wt[k * CIN + ci];
  }
  if (EPI == E_WAV) {
    Y[long(b) * p.Tout + o] = clamp(acc[0], -1.0f, 1.0f);
    return;
  }
  const long idx = (long(b) * p.Tout + o) * COUT;
  for (int k = 0; k < COUT; ++k) {
    const float v = acc[k] + bias[k];
    Y[idx + k] = EPI == E_RES ? v + R[idx + k] : v;
  }
}

#define SMALL_ARGS                                                                                            \
  device const half *, device const half *, device const half *, device const half *, device const float *,   \
      device float *, device const float *, constant AudioConvParams &, uint3, uint3, uint3
#define SMALL(CIN, COUT, MAXTAPS, MODE, EPI, NAME) \
  template [[host_name(NAME)]] [[kernel]] void audio_small<CIN, COUT, MAXTAPS, MODE, EPI>(SMALL_ARGS);
SMALL(8, 8, 11, A_CONV, E_F32, "audio_small_f32_8")
SMALL(8, 8, 11, A_CONV, E_RES, "audio_small_res_8")
SMALL(16, 8, 4, A_UPS, E_F32, "audio_small_ups_16x8")
SMALL(8, 1, 7, A_CONV, E_WAV, "audio_small_wav_8")
