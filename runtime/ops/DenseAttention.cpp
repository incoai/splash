#include "DenseAttention.hpp"

#include "metal/abi/DenseAttention.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }

// The bytes of `tensor` the kernel reaches over `rows` rows of `width`
// dimensions of every head.
void requireTensor(const HeadTensor &tensor, uint32_t heads, uint32_t rows, uint32_t width, const char *what) {
  const uint64_t last = uint64_t{heads - 1} * tensor.headStride + uint64_t{rows - 1} * tensor.rowStride + width;
  requireBytes(tensor.buffer, last * 2, what);
}

} // namespace

uint32_t DenseAttention::paddedQueries(uint32_t rows) noexcept { return roundUp(rows, SPLASH_DENSE_ATTENTION_ROWS); }

uint32_t DenseAttention::paddedKeys(uint32_t rows) noexcept { return roundUp(rows, SPLASH_DENSE_ATTENTION_KEYS); }

void DenseAttention::add(metal::CommandGraph &graph, const DenseAttentionTensors &tensors, uint32_t heads,
                         uint32_t dimensions, uint32_t rows, float scale) {
  const uint32_t padded = splash_dense_attention_padded(dimensions);
  requireTensor(tensors.queries, heads, paddedQueries(rows), padded, "dense attention query");
  requireTensor(tensors.keys, heads, paddedKeys(rows), padded, "dense attention key");
  requireTensor(tensors.values, heads, paddedKeys(rows), dimensions, "dense attention value");
  requireTensor(tensors.output, heads, rows, dimensions, "dense attention output");
  const DenseAttentionParams params{rows,
                                    scale,
                                    tensors.queries.headStride,
                                    tensors.keys.headStride,
                                    tensors.values.headStride,
                                    tensors.output.headStride,
                                    tensors.queries.rowStride,
                                    tensors.keys.rowStride,
                                    tensors.values.rowStride,
                                    tensors.output.rowStride};
  graph.add("dense_attention_" + std::to_string(dimensions),
            {tensors.queries.buffer, tensors.keys.buffer, tensors.values.buffer, tensors.output.buffer}, params,
            {paddedQueries(rows) / SPLASH_DENSE_ATTENTION_ROWS, heads, 1}, {SPLASH_DENSE_ATTENTION_THREADS, 1, 1});
}

} // namespace splash::ops
