#pragma once

// Video Sparse Attention (VSA) of the FastH3 variants (kernels/video/dit/vsa.metal, video/ops/VsaTable): FastVideo's VSA-H3
// contract on 64-token cubes. Inside the blocks the rows are stored in tile order (video/ops/VsaTable): every video cube
// of 4 x 4 x 4 latent tokens (t, h, w; partial at the grid edges) is a contiguous run of rows, the video tiles first in
// (t, h, w) tile order, then the dense prefix tiles (text and audio, each segment cut into 64-row chunks).
#include "metal/abi/video/AttentionABI.h"

#define H3_VSA_TILE 64          // tokens per full tile (4 x 4 x 4)
#define H3_VSA_MAX_TILES 6144   // tiles per request held in threadgroup memory by dit_vsa_route (24 KB)
#define H3_VSA_ROUTE_THREADS 256

// dit_vsa_permute: dst[i] = src[index[i]], rows of D bf16 (i < rows)
struct VsaPermuteParams {
  int32_t rows, D;
};

// dit_vsa_pool / dit_vsa_route / dit_vsa_attention / dit_vsa_attention_qkfix
//   tiles int2 [T] (start row, rows); video tiles are [0, V), prefix tiles [V, T); the prefix rows are [Nv, M)
//   pooled Q / K / V, compressed output fp32 [H][Tp][128]; scores / softmax fp32 [H][Tp][Tp] (Tp = T rounded up to
//   32, zero pad rows); selection int32 [H][V][topk] (ascending tile indices)
struct VsaParams {
  int32_t M, Spad, H, T;  // rows (tile order), padded rows of the head-major tensors, heads, tiles
  int32_t Tp, pad0, pad1, pad2;  // tiles padded to 32
  int32_t V, Nv, topk, nkb;  // video tiles, video rows (= start of the prefix rows), kept video tiles, key blocks/head
  int32_t nitems, head0, hb, flags;  // attention items (prefix tiles first), head range; flags bit0: QKFIX split
                                     // term, bit1: dense (every query chunk attends to [0, M); no gate term)
  float score_scale;  // 1 / sqrt(128) (routing scores)
  float qk_scale;     // QKFIX attention: log2(e) / sqrt(128)
};
