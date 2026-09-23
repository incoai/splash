#pragma once

// GGUF quantized projection parameters shared by host dispatch code and Metal
// kernels (kernels/shared/gguf_linear.metal).
#include "metal/abi/QuantFormat.h"

struct GgufParams {
  uint32_t output_size;       // columns of this segment
  uint32_t input_size;        // K
  uint32_t persistent_groups; // decode: threadgroups (>= tiles)
  uint32_t out_stride;        // row stride of the destination (0 = output_size)
  uint32_t out_offset;        // first destination column of this segment
};
static_assert(sizeof(GgufParams) == 20, "GGUF parameters are 20 bytes on both sides");

// Prefill tiles (pf kernels): the grid covers whole 128-row tiles of the chunk; the simdgroups of the last tile
// whose rows start past `rows` skip their matmuls.
struct GgufPrefillParams {
  uint32_t output_size; // columns of this segment
  uint32_t input_size;  // K
  uint32_t rows;        // rows of the chunk
  uint32_t out_stride;  // row stride of the destination (0 = output_size)
  uint32_t out_offset;  // first destination column of this segment
};
static_assert(sizeof(GgufPrefillParams) == 20, "GGUF prefill parameters are 20 bytes on both sides");

// Apple9 register decode (kernels/decode/linear_gguf_sgmatrix.metal): one
// tensor per dispatch, over the dispatch's 64-column tiles.
struct GgufSgParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions; 1 = no cross-threadgroup reduction
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column of the tensor
};
static_assert(sizeof(GgufSgParams) == 16, "GGUF register decode parameters are 16 bytes on both sides");

// Apple9 register decode of a fused projection: up to three column segments
// of any formats in one dispatch, tiles in segment order.
struct GgufSgFusedParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions of every segment
  uint32_t out_stride;  // columns of a destination row
  uint32_t cols[3];     // columns per segment; 0 past the last
  uint32_t fmt[3];      // GGUF_FMT_* per segment
  uint32_t offset[3];   // first destination column per segment
};
static_assert(sizeof(GgufSgFusedParams) == 48, "GGUF register fused parameters are 48 bytes on both sides");

struct GgufEmbedParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t hidden;
};
static_assert(sizeof(GgufEmbedParams) == 12, "GGUF embedding parameters are 12 bytes on both sides");

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
