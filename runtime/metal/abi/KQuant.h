#pragma once

// GGUF K-quant projection parameters shared by host dispatch code and Metal
// kernels (kernels/shared/kquant.metal).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct KQParams {
  uint32_t output_size;       // columns of this segment
  uint32_t input_size;        // K
  uint32_t persistent_groups; // decode: threadgroups (>= tiles); split-K: splits
  uint32_t out_stride;        // row stride of the destination (0 = output_size)
  uint32_t out_offset;        // first destination column of this segment
};
static_assert(sizeof(KQParams) == 20, "K-quant parameters are 20 bytes on both sides");

struct KQReduceParams {
  uint32_t splits;
  uint32_t rows;
  uint32_t cols;
  uint32_t out_stride;
  uint32_t out_offset;
  uint32_t epilogue;
};
static_assert(sizeof(KQReduceParams) == 24, "K-quant reduce parameters are 24 bytes on both sides");

struct KQEmbedParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t hidden;
};
static_assert(sizeof(KQEmbedParams) == 12, "K-quant embedding parameters are 12 bytes on both sides");

struct KQPermuteParams {
  uint32_t rows;
  uint32_t width;
  uint32_t block;
};
static_assert(sizeof(KQPermuteParams) == 12, "K-quant permute parameters are 12 bytes on both sides");

// Runtime format ids for kernels that select the dequantizer per tile.
#define KQ_FMT_Q4K 0u
#define KQ_FMT_IQ4XS 1u
#define KQ_FMT_IQ4NL 2u
#define KQ_FMT_Q5K 3u
#define KQ_FMT_Q6K 4u
#define KQ_FMT_Q3K 5u
#define KQ_FMT_Q80 6u
#define KQ_FMT_IQ3S 7u

// Split-K with in-kernel last-arriver reduction (counters: one uint per 64-column tile, zero at rest).
struct KQSplitParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t splits;
  uint32_t out_stride;
  uint32_t out_offset;
  uint32_t epilogue;
};
static_assert(sizeof(KQSplitParams) == 24, "K-quant split parameters are 24 bytes on both sides");

// One dispatch over up to three column segments of different formats (fused qkv|z|ab, q|k|v).
struct KQFusedParams {
  uint32_t input_size;
  uint32_t out_stride;
  uint32_t segments;
  uint32_t reserved;
  uint32_t cols[3];
  uint32_t fmt[3];
  uint32_t offset[3];
};
static_assert(sizeof(KQFusedParams) == 52, "K-quant fused parameters are 52 bytes on both sides");

// Gate and up projections in one dispatch: output = silu(gate) * up.
struct KQGateUpParams {
  uint32_t input_size;
  uint32_t output_size;
  uint32_t out_stride;
  uint32_t gate_fmt;
  uint32_t up_fmt;
};
static_assert(sizeof(KQGateUpParams) == 20, "K-quant gate/up parameters are 20 bytes on both sides");

#define KQ_EPILOGUE_NONE 0u
#define KQ_EPILOGUE_RESIDUAL 1u
#define KQ_EPILOGUE_UP_WITH_GATE 2u
