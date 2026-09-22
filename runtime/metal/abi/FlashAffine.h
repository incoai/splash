#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// All strides are bytes relative to each bound tensor view. The source MLX
// affine layout is expert-major, output-row-major, with an unsigned little
// endian bitstream along K. No Splash StorageN packing is used here.
struct FlashAffineParams {
  uint32_t rows;
  uint32_t selections;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t experts;
  uint32_t bits;
  uint32_t group_size;
  // bit 0: gather expert_ids[row * selections + selection].
  // bit 1: input contains one K-vector for every row/selection.
  uint32_t flags;
  uint64_t weight_row_stride_bytes;
  uint64_t weight_expert_stride_bytes;
  uint64_t parameter_row_stride_bytes;
  uint64_t parameter_expert_stride_bytes;
};

struct FlashDenseParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t reserved;
  uint64_t weight_row_stride_bytes;
};

struct FlashEmbeddingParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t vocabulary_size;
  uint32_t bits;
  uint32_t group_size;
  uint32_t reserved;
  uint64_t weight_row_stride_bytes;
  uint64_t parameter_row_stride_bytes;
};

static_assert(sizeof(FlashAffineParams) == 64,
              "Flash affine parameters must have identical host/shader layouts");
static_assert(sizeof(FlashDenseParams) == 24,
              "Flash dense parameters must have identical host/shader layouts");
static_assert(sizeof(FlashEmbeddingParams) == 40,
              "Flash embedding parameters must have identical host/shader layouts");
