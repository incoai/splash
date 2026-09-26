#pragma once

// GGUF quantized projection parameters and tile geometry shared by host
// dispatch code and Metal kernels: the staged tiles of
// kernels/shared/gguf_linear.metal and kernels/shared/moe_gguf.metal, and the
// Apple9 register tile of kernels/decode/linear_gguf_sgmatrix.metal.
#include "metal/abi/QuantFormat.h"

// Vocabulary: a group is 32 inputs (the image's quantization unit), a span 64
// (Table16's); a plane tile is QUANT_TILE_ROWS rows of the weight image, a
// column tile the GGUF_TILE_COLUMNS output columns of one threadgroup (grid.x
// counts column tiles), and an expert tile the grouped rows of one MoE expert.
#define GGUF_TILE_COLUMNS 64u
// Decode tile of the staged kernels, which also runs prefill chunks of up to
// 32 rows: two simdgroups, each staging its own GGUF_STAGED_COLUMNS columns
// GGUF_STAGED_STEP inputs at a time over 8, 16 or 32 rows.
#define GGUF_STAGED_COLUMNS 32u
#define GGUF_STAGED_STEP 32u
#define GGUF_STAGED_THREADS (GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS * 32u)
// Prefill tile of the staged kernels: GGUF_PREFILL_SIMDGROUPS simdgroups of
// GGUF_PREFILL_SIMDGROUP_ROWS rows share one stage of GGUF_PREFILL_STEP
// inputs of the tile's columns, which all threads dequantize.
#define GGUF_PREFILL_SIMDGROUP_ROWS 32u
#define GGUF_PREFILL_SIMDGROUPS 4u
#define GGUF_PREFILL_STEP 64u
#define GGUF_PREFILL_ROWS (GGUF_PREFILL_SIMDGROUP_ROWS * GGUF_PREFILL_SIMDGROUPS)
#define GGUF_PREFILL_THREADS (GGUF_PREFILL_SIMDGROUPS * 32u)
// Apple9 register tile: four simdgroups of GGUF_REGISTER_COLUMNS columns.
#define GGUF_REGISTER_COLUMNS 16u
#define GGUF_REGISTER_THREADS (GGUF_TILE_COLUMNS / GGUF_REGISTER_COLUMNS * 32u)

// The register tile's activations, Table16 (kernels/common/gguf_sgmatrix.h):
// per eight-row tile of `width` inputs, the X^T table of width * 8 bfloat,
// then the fp32 chain seeds [width / 16][8 rows] and sums [width / 32][8
// rows]. Each 64-input span holds these values, seeds and sums per tile.
#define GGUF_TABLE16_SPAN_INPUTS 64u
#define GGUF_TABLE16_SPAN_VALUES 512u
#define GGUF_TABLE16_SPAN_SEEDS 32u
#define GGUF_TABLE16_SPAN_SUMS 16u
inline constexpr uint64_t table16_sums32_offset(uint32_t width) { return uint64_t(width) / 16 * 8; }
inline constexpr uint64_t table16_sums_per_tile(uint32_t width) {
  return table16_sums32_offset(width) + uint64_t(width) / 32 * 8;
}

// Prefill tiles: the grid covers whole GGUF_PREFILL_ROWS-row tiles of the
// chunk; the simdgroups of the last tile whose rows start past `rows` skip
// their matmuls.
struct GgufPrefillParams {
  uint32_t output_size; // columns of this segment
  uint32_t input_size;  // K
  uint32_t rows;        // rows of the chunk
  uint32_t out_stride;  // row stride of the destination (0 = output_size)
  uint32_t out_offset;  // first destination column of this segment
};
static_assert(sizeof(GgufPrefillParams) == 20, "GGUF prefill parameters are 20 bytes on both sides");

// Decode tiles of both families (the register tile on Apple9, the staged
// tile elsewhere and for prefill chunks of up to 32 rows): one tensor per
// dispatch, over the dispatch's tiles (grid.x) and K partitions (grid.y).
struct GgufDecodeParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions; 1 = no cross-threadgroup reduction
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column of the tensor
};
static_assert(sizeof(GgufDecodeParams) == 16, "GGUF decode parameters are 16 bytes on both sides");

// Decode of a fused projection: up to three column segments of any formats in
// one dispatch, tiles in segment order.
struct GgufDecodeFusedParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions of every segment
  uint32_t out_stride;  // columns of a destination row
  uint32_t cols[3];     // columns per segment; 0 past the last
  uint32_t fmt[3];      // GGUF_FMT_* per segment
  uint32_t offset[3];   // first destination column per segment
};
static_assert(sizeof(GgufDecodeFusedParams) == 48, "GGUF fused decode parameters are 48 bytes on both sides");

struct GgufEmbedParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t hidden;
};
static_assert(sizeof(GgufEmbedParams) == 12, "GGUF embedding parameters are 12 bytes on both sides");
// The formats whose native token rows the embedding kernels gather
// (kernels/shared/embedding.metal, gguf_embed_<kQuantFormats name>).
inline constexpr bool gguf_embedding_format(uint32_t format) {
  return format == GGUF_FMT_Q4K || format == GGUF_FMT_Q5K || format == GGUF_FMT_Q6K || format == GGUF_FMT_Q3K ||
         format == GGUF_FMT_Q2K || format == GGUF_FMT_Q80 || format == GGUF_FMT_Q40 || format == GGUF_FMT_Q41 ||
         format == GGUF_FMT_PQ20;
}

// fp32 projection of a GGUF float tensor (kernels/shared/gguf_float.metal):
// out[r][out_offset + n] = sum_k x[r][k] * W[n][k] for rows r < rows, W as
// stored ([output_size][input_size] floats of ggml type GGUF_TYPE_F32).
#define GGUF_TYPE_F32 0u
struct GgufFloatParams {
  uint32_t rows;
  uint32_t input_size;  // K, a multiple of 8
  uint32_t output_size; // N, a multiple of 8
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column
};
static_assert(sizeof(GgufFloatParams) == 20, "GGUF float parameters are 20 bytes on both sides");

#define GGUF_EPILOGUE_NONE 0u
#define GGUF_EPILOGUE_RESIDUAL 1u
#define GGUF_EPILOGUE_UP_WITH_GATE 2u
