// TAEH3 preview decoder (madebyollin/taehv, H3 variant) convolutions. Only the inner matmuls are MPP matmul2d
// (f16 x f16 -> f32); conv addressing, epilogues, upsampling, pixel shuffle and the memory-block state are here.
//
// Activation layout: fp16 NHWC frames, each frame zero-padded by one pixel and stored flat:
//   frame f, padded pixel (Y, X) in [0, H+2) x [0, Wp = W+2) -> element ((f FS) + Y Wp + X) C + c.
// A 3x3 conv is 9 shifted GEMMs over the flat row domain r in [0, H Wp): output row r is the padded pixel r + Wp + 1
// and tap (dy, dx) reads input row r + dy Wp + dx, a contiguous [rows][C] block (columns x >= W of the row domain are
// garbage and are written as zeros = the padding of the output frame). Every buffer has one extra frame slot in front
// of frame 0 holding the MemBlock memory (previous frame of the sequence), so the "past" input of a MemBlock is the
// same buffer shifted by one frame. Nearest 2x upsample + 3x3 conv runs as 4 output phases x 2x2 taps on the low-res
// input (weights pre-combined by tools/convert_preview.py).
//   preview_conv_relu / _memory / _residual   3x3 conv + bias + relu (MemBlock conv.0 with x and past; + residual)
//   preview_tgrow                             1x1 conv, output channels split into S frames
//   preview_upsample                          nearest 2x + 3x3 conv (+ relu)
//   preview_conv_out                          3x3 conv 64 -> 12, bias, pixel shuffle, clamp, frame drop -> uint8
#include "metal/abi/video/PreviewABI.h"
#include "metal/kernels/video/common/tensor.h"

using namespace h3;

enum : int { E_RELU = 0, E_RES = 1, E_TGROW = 2, E_UP = 3, E_FIN = 4 };

// NSRC sources (MemBlock conv.0: x and past), NTAP taps: 9 (3x3), 1 (1x1, centre), 4 (polyphase up, z = frame 4 + phase)
template <int TM, int TN, int NSG, int NSRC, int NTAP, int EPI>
kernel void preview_conv(device half *X [[buffer(0)]], device half *Wt [[buffer(1)]], device const float *bias [[buffer(2)]],
                         device half *Y [[buffer(3)]], device half *R [[buffer(4)]], device uchar *O8 [[buffer(5)]],
                         device const int *fmap [[buffer(6)]], constant PreviewConvParams &p [[buffer(7)]],
                         device atomic_uint *bad [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int n0 = tg.x * TN, m0 = tg.y * TM;
  const int frame = (EPI == E_UP) ? int(tg.z) >> 2 : int(tg.z);
  const int phase = (EPI == E_UP) ? int(tg.z) & 3 : 0;
  const int py = phase >> 1, px = phase & 1;
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  device half *xf = X + long(frame) * p.FSin * p.Cin;
  const long wsz = long(p.Cout) * p.Cin;
  device half *wb = Wt + (EPI == E_UP ? long(phase) * NTAP * wsz : 0);
  auto a0 = mk<half>(xf, p.Cin, p.Mrows).template slice<dynamic_extent, TM>(0, m0);
  auto b0 = mk<half>(wb, p.Cin, p.Cout).template slice<dynamic_extent, TN>(0, n0);
  auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  for (int s = 0; s < NSRC; ++s) {
    device half *xs = xf - long(s) * p.FSin * p.Cin;
    for (int t = 0; t < NTAP; ++t) {
      int off;
      if (NTAP == 9) off = (t / 3) * p.Wp + (t % 3);
      else if (NTAP == 1) off = p.Wp + 1;
      else off = (py + (t >> 1)) * p.Wp + (px + (t & 1));
      auto a = mk<half>(xs + long(off) * p.Cin, p.Cin, p.Mrows).template slice<dynamic_extent, TM>(0, m0);
      auto b = mk<half>(wb + long(s * NTAP + t) * wsz, p.Cin, p.Cout).template slice<dynamic_extent, TN>(0, n0);
      op.run(a, b, c);
    }
  }
  const int Mv = p.H * p.Wp;
  const ushort cap = c.get_capacity();
#pragma unroll
  for (ushort i0 = 0; i0 < cap; i0 += 4) {
    if (EPI != E_FIN) {
      bool run4 = (cap % 4 == 0) && c.is_valid_element(i0) && c.is_valid_element(i0 + 3);
      if (run4) {
        auto q0 = c.get_multidimensional_index(i0);
        auto q3 = c.get_multidimensional_index(i0 + 3);
        run4 = q3[1] == q0[1] && q3[0] == q0[0] + 3 && ((n0 + q0[0]) & 3) == 0;
      }
      if (run4) {
        auto q0 = c.get_multidimensional_index(i0);
        const int r = m0 + q0[1], col = n0 + q0[0];
        if (r >= Mv) continue;
        const int y = r / p.Wp, x = r - y * p.Wp;
        float4 v = float4(c[i0], c[i0 + 1], c[i0 + 2], c[i0 + 3]);
        if (EPI == E_RELU) {
          v = x < p.W ? max(v + *(device const float4 *)(bias + col), 0.0f) : 0.0f;
          *(device half4 *)(Y + (long(frame) * p.FSout + r + p.Wp + 1) * p.Cout + col) = half4(v);
        } else if (EPI == E_RES) {
          float4 res = float4(*(device const half4 *)(R + (long(frame) * p.FSres + r + p.Wp + 1) * p.Cout + col));
          v = x < p.W ? max(v + *(device const float4 *)(bias + col) + res, 0.0f) : 0.0f;
          *(device half4 *)(Y + (long(frame) * p.FSout + r + p.Wp + 1) * p.Cout + col) = half4(v);
        } else if (EPI == E_TGROW) {
          const int C = p.Cout / p.S, s = col / C, ci = col - s * C;
          v = x < p.W ? v : 0.0f;
          *(device half4 *)(Y + (long(frame * p.S + s) * p.FSout + r + p.Wp + 1) * C + ci) = half4(v);
        } else {  // E_UP
          if (x >= p.W) continue;
          if (p.relu) v = max(v, 0.0f);
          const long o = long(frame) * p.FSout + long(2 * y + py + 1) * p.WpOut + (2 * x + px + 1);
          *(device half4 *)(Y + o * p.Cout + col) = half4(v);
        }
        continue;
      }
    }
    for (ushort i = i0; i < i0 + 4 && i < cap; ++i) {
      if (!c.is_valid_element(i)) continue;
      auto q = c.get_multidimensional_index(i);
      const int r = m0 + q[1], col = n0 + q[0];
      if (r >= Mv) continue;
      const int y = r / p.Wp, x = r - y * p.Wp;
      float v = c[i];
      if (EPI == E_RELU) {
        v = x < p.W ? max(v + bias[col], 0.0f) : 0.0f;
        Y[(long(frame) * p.FSout + r + p.Wp + 1) * p.Cout + col] = half(v);
      } else if (EPI == E_RES) {
        v = x < p.W ? max(v + bias[col] + float(R[(long(frame) * p.FSres + r + p.Wp + 1) * p.Cout + col]), 0.0f) : 0.0f;
        Y[(long(frame) * p.FSout + r + p.Wp + 1) * p.Cout + col] = half(v);
      } else if (EPI == E_TGROW) {
        const int C = p.Cout / p.S, s = col / C, ci = col - s * C;
        Y[(long(frame * p.S + s) * p.FSout + r + p.Wp + 1) * C + ci] = half(x < p.W ? v : 0.0f);
      } else if (EPI == E_UP) {
        if (x >= p.W) continue;
        if (p.relu) v = max(v, 0.0f);
        const long o = long(frame) * p.FSout + long(2 * y + py + 1) * p.WpOut + (2 * x + px + 1);
        Y[o * p.Cout + col] = half(v);
      } else {  // E_FIN: bias, pixel shuffle (rgb, py, px), clamp, frame drop map -> uint8
        if (x >= p.W || col >= 12) continue;
        const int of = fmap[frame];
        if (of < 0) continue;
        v += bias[col];
        if (!isfinite(v)) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
        v = clamp(v, 0.0f, 1.0f);
        const int rgb = col >> 2, sy = (col >> 1) & 1, sx = col & 1;
        const long pix = (long(of) * p.Hout + 2 * y + sy) * p.Wout + 2 * x + sx;
        O8[pix * 3 + rgb] = uchar(v * 255.0f + 0.5f);
      }
    }
  }
}

#define CONV_ARGS                                                                                           \
  device half *, device half *, device const float *, device half *, device half *, device uchar *,          \
      device const int *, constant PreviewConvParams &, device atomic_uint *, uint3
#define CONV(TM, TN, NSG, NSRC, NTAP, EPI, NAME) \
  template [[host_name(NAME "_" #TM "x" #TN "x" #NSG)]] [[kernel]] void preview_conv<TM, TN, NSG, NSRC, NTAP, EPI>(CONV_ARGS);
#define CONV_LEVEL(TM, TN, NSG)                          \
  CONV(TM, TN, NSG, 1, 9, E_RELU, "preview_conv_relu")   \
  CONV(TM, TN, NSG, 2, 9, E_RELU, "preview_conv_memory") \
  CONV(TM, TN, NSG, 1, 9, E_RES, "preview_conv_residual")
CONV_LEVEL(32, 128, 4)  // 256 channels
CONV_LEVEL(64, 128, 4)  // 128 channels
CONV_LEVEL(32, 64, 2)   // 64 channels
CONV(32, 128, 4, 1, 1, E_TGROW, "preview_tgrow")
CONV(64, 128, 4, 1, 1, E_TGROW, "preview_tgrow")
CONV(64, 128, 4, 1, 4, E_UP, "preview_upsample")
CONV(32, 64, 2, 1, 4, E_UP, "preview_upsample")
CONV(32, 16, 1, 1, 9, E_FIN, "preview_conv_out")
