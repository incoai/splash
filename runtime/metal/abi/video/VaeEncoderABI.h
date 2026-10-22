#pragma once

// Video VAE encoder (kernels/vae/encoder.metal): the causal 3D CNN of the MiniMax-H3 VAE as implicit-GEMM
// convolutions over fp16 padded NHWC frames (MPP matmul2d, f16 x f16 -> f32), TIsolatedGroupNorm statistics and the
// fused norm + SiLU + reflect-pad producer. Conv kernels are instantiated per tile shape and kind
// (vae_enc_conv3_<TM>x<TN>x<NSG>, vae_enc_down_..., vae_enc_conv1_...); engine/DeviceProfile.hpp picks the tiles.
#include "metal/abi/video/Layout.h"

#define H3_VAE_ENC_GROUPS 32
#define H3_VAE_ENC_IN_CHANNELS 32  // conv_in's 3 RGB channels zero-padded (MPP K granularity)
#define H3_VAE_ENC_MOMENTS 48      // mean 24 | logvar 24

// Convolution: input frames fp16 [B * Din][FSin][Cin] (padded rows of width Wp), weights fp16 [KD][taps][CoutPad][Cin],
// output fp32 dense [B * Dout][Hout][Wout][Cout] (+ residual R of the same layout).
struct VaeEncConvParams {
  int32_t Wp, Wq, Mv, Mrows;     // input padded width; output flat width (Wp, also for stride 2); valid GEMM rows
                                 // (Hout * Wq); rows padded to the tile
  int32_t Hout, Wout, Cin, Cout;  // Cin: padded input channels (GEMM K); Cout: stored output channels
  int32_t CoutPad, FSin, Din, Dout;  // FSin: padded frame stride (pixels)
  int32_t KD, tstride, residual, pad0;  // temporal taps (3 causal: frames t * tstride - 2 .. t * tstride; or 1)
};

// GroupNorm statistics over fp32 dense frames [frames][HW][C]: (mean, rstd) per frame and group.
struct VaeEncStatsParams {
  int32_t HW, C, pad0;
  float eps;
};

// Producer: fp32 dense frames [frames][H][W][Csrc] -> fp16 padded frames [frames][FS][Cdst] (channels >= Csrc zero).
// mode 0: reflect pad 1 on every side (data at (1, 1)); mode 1: data at (0, 0), one reflected row / column appended at
// the bottom / right (Downsample3d). norm: (x - mean) rstd w + b, then SiLU.
struct VaeEncPadParams {
  int32_t H, W, Wp, FS;
  int32_t Csrc, Cdst, mode, norm;
};

// quant_conv (1 x 1 x 1, 48 -> 48, fp32) over [pixels][48].
struct VaeEncQuantParams {
  int32_t pixels, pad0, pad1, pad2;
};
