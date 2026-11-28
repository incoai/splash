#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>

namespace splash::ops {

// A bf16 tensor of every head's rows: element (head, row, dimension) lies
// head * headStride + row * rowStride + dimension elements from the start of
// `buffer`. Per-head planes and a projection's rows of every head are both
// such tensors.
struct HeadTensor final {
  metal::MetalBuffer buffer;
  uint32_t headStride = 0;
  uint32_t rowStride = 0;
};

// Queries and keys hold each head's dimensions padded with zeros
// (splash_dense_attention_padded, metal/abi/DenseAttention.h); values and the
// output hold the dimensions alone.
struct DenseAttentionTensors final {
  HeadTensor queries, keys, values, output;
};

class DenseAttention final {
public:
  // The rows of queries a dispatch of `rows` rows reads: whole tiles of 32.
  [[nodiscard]] static uint32_t paddedQueries(uint32_t rows) noexcept;
  // The rows of keys and values it reads: whole tiles of 128, finite past
  // the rows.
  [[nodiscard]] static uint32_t paddedKeys(uint32_t rows) noexcept;
  // softmax(q k^T scale) v of `rows` rows of `heads` heads of `dimensions`,
  // every row over all `rows` keys, written for the rows alone. The kernel
  // takes heads of 72 or 64, Qwen3.5's and Inkling 2's vision towers'.
  // Throws unless every tensor holds what the kernel reaches.
  static void add(metal::CommandGraph &graph, const DenseAttentionTensors &tensors, uint32_t heads,
                  uint32_t dimensions, uint32_t rows, float scale);
};

} // namespace splash::ops
