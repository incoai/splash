#pragma once

// GGUF quantized projection parameters shared by host dispatch code and Metal
// kernels (kernels/shared/gguf_linear.metal).
#include "metal/abi/QuantFormat.h"

struct GgufParams {
  uint32_t output_size;       // columns of this segment
  uint32_t input_size;        // K
  uint32_t persistent_groups; // decode: threadgroups (>= tiles); split-K: splits
  uint32_t out_stride;        // row stride of the destination (0 = output_size)
  uint32_t out_offset;        // first destination column of this segment
};
static_assert(sizeof(GgufParams) == 20, "GGUF parameters are 20 bytes on both sides");

struct GgufReduceParams {
  uint32_t splits;
  uint32_t rows;
  uint32_t cols;
  uint32_t out_stride;
  uint32_t out_offset;
  uint32_t epilogue;
};
static_assert(sizeof(GgufReduceParams) == 24, "GGUF reduce parameters are 24 bytes on both sides");

struct GgufEmbedParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t hidden;
};
static_assert(sizeof(GgufEmbedParams) == 12, "GGUF embedding parameters are 12 bytes on both sides");

struct GgufPermuteParams {
  uint32_t rows;
  uint32_t width;
  uint32_t block;
};
static_assert(sizeof(GgufPermuteParams) == 12, "GGUF permute parameters are 12 bytes on both sides");

// Load-time repack of native GGUF rows into the MDGG0001 planes (gguf_repack):
// one thread per (destination row, 32-wide K group).
struct GgufRepackParams {
  uint32_t rows;                // destination rows (multiple of 256)
  uint32_t input_size;          // K
  uint32_t fmt;                 // GGUF_FMT_*
  uint32_t src_offset;          // first source block, bytes from the source buffer start
  uint32_t src_row_bytes;       // bytes per source row
  uint32_t dst_plane0;          // byte offsets in the image buffer
  uint32_t dst_plane1;
  uint32_t dst_meta;
  uint32_t permute_from_row;    // rows >= this take llama.cpp's tiled head order (0xFFFFFFFF: none)
  uint32_t permute_head_rows;   // rows per head
  uint32_t permute_group_heads; // heads per key group (tiled index = group * this + head)
  uint32_t permute_groups;      // value heads per key head
};
static_assert(sizeof(GgufRepackParams) == 48, "GGUF repack parameters are 48 bytes on both sides");

struct GgufCopyParams {
  uint32_t src_offset;
  uint32_t dst_offset;
  uint32_t bytes;
};
static_assert(sizeof(GgufCopyParams) == 12, "GGUF copy parameters are 12 bytes on both sides");

// Split-K with in-kernel last-arriver reduction (counters: one uint per 64-column tile, zero at rest).
struct GgufSplitParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t splits;
  uint32_t out_stride;
  uint32_t out_offset;
  uint32_t epilogue;
};
static_assert(sizeof(GgufSplitParams) == 24, "GGUF split parameters are 24 bytes on both sides");

// One dispatch over up to three column segments of different formats (fused qkv|z|ab, q|k|v).
struct GgufFusedParams {
  uint32_t input_size;
  uint32_t out_stride;
  uint32_t segments;
  uint32_t reserved;
  uint32_t cols[3];
  uint32_t fmt[3];
  uint32_t offset[3];
};
static_assert(sizeof(GgufFusedParams) == 52, "GGUF fused parameters are 52 bytes on both sides");

// Gate and up projections in one dispatch: output = silu(gate) * up.
struct GgufGateUpParams {
  uint32_t input_size;
  uint32_t output_size;
  uint32_t out_stride;
  uint32_t gate_fmt;
  uint32_t up_fmt;
};
static_assert(sizeof(GgufGateUpParams) == 20, "GGUF gate/up parameters are 20 bytes on both sides");

#define GGUF_EPILOGUE_NONE 0u
#define GGUF_EPILOGUE_RESIDUAL 1u
#define GGUF_EPILOGUE_UP_WITH_GATE 2u
