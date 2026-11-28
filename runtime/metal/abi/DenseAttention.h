#pragma once

// Parameter layout of the dense attention (kernels/shared/dense_attention.metal)
// and its host dispatch (ops/DenseAttention.hpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Query rows of one head a dense_attention threadgroup takes, its threads
// (four simdgroups whose softmax gives each row four lanes; a static_assert
// in shared/dense_attention.metal ties the two together), and the keys of
// one key tile, which a threadgroup reads whole.
#define SPLASH_DENSE_ATTENTION_ROWS 32u
#define SPLASH_DENSE_ATTENTION_THREADS 128u
#define SPLASH_DENSE_ATTENTION_KEYS 128u

// Queries and keys hold each head's dimensions padded with zeros to a
// multiple of 16, the K extent a matmul takes; the zeros add nothing to the
// dot products. Values and the output hold the dimensions alone.
inline constexpr uint32_t splash_dense_attention_padded(uint32_t dimensions) {
  return (dimensions + 15u) / 16u * 16u;
}

// The attention of `rows` rows, each over every one of their keys and
// values. Element (head, row, dimension) of a tensor lies head * its head
// stride + row * its row stride + dimension elements from its buffer's
// start.
struct DenseAttentionParams {
  uint32_t rows;
  float scale;
  uint32_t query_head_stride;
  uint32_t key_head_stride;
  uint32_t value_head_stride;
  uint32_t output_head_stride;
  uint32_t query_row_stride;
  uint32_t key_row_stride;
  uint32_t value_row_stride;
  uint32_t output_row_stride;
};

static_assert(sizeof(DenseAttentionParams) == 40,
              "dense attention parameters are 40 bytes on both sides");
