#pragma once

// Video VAE ViT decoder: embedding, producers, GEMMs with fused epilogues, attention and the clip finish
// (tile stitch, temporal cross-fade, uint8 frames).
#include "metal/abi/video/Layout.h"

#define H3_VAE_QKV_TM 32   // QKV GEMM tile: 32 x 128, 4 simdgroups
#define H3_VAE_TM 64       // residual / GLU / proj_out GEMM tiles: 64 x 128, 4 simdgroups
#define H3_VAE_TN 128
#define H3_VAE_NSG 4
#define H3_VAE_ATTENTION_NSG 4

struct VaeParams {
  int32_t M, N, K, ldo;   // rows, output columns, reduction, output row stride
  int32_t S, Spad, H, F;  // rows per tile-clip, padded rows (attention), heads, fc1 hidden width
  int32_t swizzle, LH, LW, T;  // tile-order swizzle (M-blocks per group); latent height / width / frames
  int32_t TS;             // tile side (latent)
  float eps, qk_scale_log2;
};

// vae_finish: one temporal clip of NTY x NTX tiles -> uint8 frames (and fp32 overlap frames for the next clip).
struct VaeFinishParams {
  int32_t NTY, NTX, HP, WP;       // tiles, pixel height / width
  int32_t S, has_prev, mode, NF0;  // rows per tile-clip; previous clip present; mode 1 = tail; frames of a clip (17)
  int32_t FPP, FOV, CF, TS;       // frame pre-padding (3), frame overlap (5), frames per sub-chunk (20), tile side
  int32_t ys[H3_VAE_MAX_TILES], xs[H3_VAE_MAX_TILES], ovy[H3_VAE_MAX_TILES], ovx[H3_VAE_MAX_TILES];
  float pm[4], ps[4];             // pixel mean / std
};
