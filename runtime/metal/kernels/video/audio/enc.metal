// Audio VAE encoder (DAC, AudioABI.h): conv_in (k7, 1 -> 64, SIMT fp32), the Snake1d producer of the conv inputs
// (fp32 -> fp16 hi / lo padded) and the strided channel-doubling convs as an implicit GEMM over K = s Cin (the
// stride's s input rows of a channel-last layout are contiguous), three fp16 matmul2d products (hi*hi + lo*hi +
// hi*lo) accumulated in fp32, fp32 output [2][Tout][Cout] + bias. The stride-1 convs of the residual units run on
// audio_conv (conv.metal).
#include "metal/abi/video/AudioABI.h"
#include "metal/kernels/video/common/tensor.h"

using namespace h3;

// out[b][t][c] = bias[c] + sum_j x[b][t + j - 3] (Wh + Wl)[j][c], x zero outside [0, n). W [7][64][1].
kernel void audio_enc_in(device const float *X [[buffer(0)]], device const half *Wh [[buffer(1)]],
                         device const half *Wl [[buffer(2)]], device const float *bias [[buffer(3)]],
                         device float *Y [[buffer(4)]], constant AudioActParams &p [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, r = gid.y;
  if (c >= p.C || r >= 2 * p.T) return;
  const int b = r / p.T, t = r - b * p.T;
  device const float *x = X + long(b) * p.n;
  float acc = 0.0f;
  for (int j = 0; j < 7; ++j) {
    const int s = t + j - 3;
    const float v = (s >= 0 && s < p.n) ? x[s] : 0.0f;
    acc += v * (float(Wh[j * p.C + c]) + float(Wl[j * p.C + c]));
  }
  Y[long(r) * p.C + c] = acc + bias[c];
}

// Snake1d: fp32 [2][T][C] -> hi / lo padded [2][P + T + P][C] (pads zero). snake [2][C]: alpha, 1 / (alpha + 1e-9).
kernel void audio_snake(device const float *X [[buffer(0)]], device half *Yh [[buffer(1)]],
                        device half *Yl [[buffer(2)]], device const float *snake [[buffer(3)]],
                        constant AudioActParams &p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, r = gid.y, Tp = p.T + 2 * H3_AUDIO_PAD;
  if (c >= p.C || r >= 2 * Tp) return;
  const int b = r / Tp, t = r - b * Tp - H3_AUDIO_PAD;
  float v = 0.0f;
  if (t >= 0 && t < p.T) {
    const float x = X[(long(b) * p.T + t) * p.C + c];
    const float sn = precise::sin(snake[c] * x);
    v = x + snake[p.C + c] * (sn * sn);
  }
  const long idx = long(r) * p.C + c;
  const half h = half(v);
  Yh[idx] = h;
  Yl[idx] = half(v - float(h));
}

template <int TM, int TN, int NSG>
kernel void audio_down(device half *Xh [[buffer(0)]], device half *Xl [[buffer(1)]], device half *Wh [[buffer(2)]],
                       device half *Wl [[buffer(3)]], device const float *bias [[buffer(4)]],
                       device float *Y [[buffer(5)]], constant AudioConvParams &p [[buffer(9)]],
                       uint3 tg [[threadgroup_position_in_grid]]) {
  const int n0 = tg.x * TN, m0 = tg.y * TM, b = tg.z;
  const int K = p.stride * p.Cin;
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  // Output row o's tap q reads padded rows P - pad + s o + q s .. + s - 1 of channel b: K contiguous values.
  const long base = (long(b) * (p.T + 2 * H3_AUDIO_PAD) + H3_AUDIO_PAD - p.pad) * p.Cin;
  const long wsz = long(p.Cout) * K;
  auto a0 = mk<half>(Xh + base, K, p.Mrows).template slice<dynamic_extent, TM>(0, m0);
  auto b0 = mk<half>(Wh, K, p.Cout).template slice<dynamic_extent, TN>(0, n0);
  auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  for (int q = 0; q < p.taps; ++q) {
    auto ah = mk<half>(Xh + base + long(q) * K, K, p.Mrows).template slice<dynamic_extent, TM>(0, m0);
    auto al = mk<half>(Xl + base + long(q) * K, K, p.Mrows).template slice<dynamic_extent, TM>(0, m0);
    auto bh = mk<half>(Wh + q * wsz, K, p.Cout).template slice<dynamic_extent, TN>(0, n0);
    auto bl = mk<half>(Wl + q * wsz, K, p.Cout).template slice<dynamic_extent, TN>(0, n0);
    op.run(ah, bh, c);
    op.run(al, bh, c);
    op.run(ah, bl, c);
  }
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i) {
    if (!c.is_valid_element(i)) continue;
    auto q = c.get_multidimensional_index(i);
    const int col = n0 + q[0], o = m0 + q[1];
    if (o >= p.Tout) continue;
    Y[(long(b) * p.Tout + o) * p.Cout + col] = c[i] + bias[col];
  }
}

template [[host_name("audio_down_64x128x4")]] [[kernel]] void audio_down<64, 128, 4>(
    device half *, device half *, device half *, device half *, device const float *, device float *,
    constant AudioConvParams &, uint3);
