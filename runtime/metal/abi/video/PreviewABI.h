#pragma once

// TAEH3 preview decoder (madebyollin/taehv, H3 variant): implicit-GEMM convolutions over zero-padded fp16 NHWC frames.
// Conv kernels are instantiated per tile shape (preview_<op>_<TM>x<TN>x<NSG>); engine/DeviceProfile.hpp picks the
// shape of each channel level.
#include "metal/abi/video/Layout.h"

#define H3_PREVIEW_CHANNELS_IN 32  // 24 latent channels zero-padded
#define H3_PREVIEW_OUT_CHANNELS 16  // 12 (3 colours x 2 x 2 pixel shuffle) zero-padded

struct PreviewConvParams {
  int32_t H, W, Wp, Mrows;  // input level: GEMM rows = H * Wp, Mrows padded to the tile
  int32_t Cin, Cout, FSin, FSout;
  int32_t FSres, WpOut, S, relu;
  int32_t Hout, Wout;
};

// preview_rgb_to_yuv420
struct PreviewYuvParams {
  int32_t H, W, frames, pad;
};
