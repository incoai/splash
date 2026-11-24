#include "ops/Embedding.hpp"

#include "metal/abi/Embedding.h"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {

void Embedding::add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    const EmbeddingWeights &table, metal::MetalBuffer output,
                    uint32_t rows, uint32_t tokenStride) {
  if (!rows || !table.outputSize || !table.inputSize)
    throw std::invalid_argument("invalid Q4 embedding shape");
  // The gather reads `rows` token ids and writes `rows` bf16 rows of the table's width.
  const uint32_t stride = std::max(tokenStride, 1u);
  requireBytes(tokens, (uint64_t{rows - 1} * stride + 1) * sizeof(uint32_t), "embedding token");
  requireBytes(output, uint64_t{rows} * table.inputSize * sizeof(uint16_t), "embedding output");
  const uint32_t hiddenGroups = (table.inputSize + 127) / 128;
  const Q4EmbeddingParams params{rows, table.outputSize, stride, 0};
  const AffineWeights &affine = table.affine();
  // Every token's Q4 row with a scale and bias per 64 values.
  const uint64_t parameters = uint64_t{table.outputSize} * (table.inputSize / 64) * 2;
  requireBytes(affine.weights, uint64_t{table.outputSize} * table.inputSize / 2, "token table");
  requireBytes(affine.scales, parameters, "token table scale");
  requireBytes(affine.biases, parameters, "token table bias");
  // One kernel per compiled hidden size (kernels/shared/embedding.metal).
  graph.add("embedding_q4_h" + std::to_string(table.inputSize),
            {std::move(tokens), affine.weights, affine.scales, affine.biases, std::move(output)},
            params, {hiddenGroups, 1, 1});
}

void Embedding::addVerifyInput(metal::CommandGraph &graph,
                               metal::MetalBuffer draftInputTokens,
                               metal::MetalBuffer proposedTokens,
                               metal::MetalBuffer verifyInputTokens,
                               uint32_t vocabulary, uint32_t lanes) {
  if (!vocabulary || !lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid verify input batch");
  // A lane's anchor is row 0 of its draft input rows.
  requireBytes(draftInputTokens, rowBytes(lanes, SPLASH_TARGET_VERIFY_ROWS, 1, sizeof(uint32_t)), "draft input token");
  requireBytes(proposedTokens, uint64_t{lanes} * SPLASH_DRAFT_PROPOSAL_TOKENS * sizeof(uint32_t), "proposed token");
  requireBytes(verifyInputTokens, uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS * sizeof(uint32_t), "verify input token");
  const VerifyInputBatchParams params{vocabulary};
  graph.add("verify_input_tokens",
            {std::move(draftInputTokens), std::move(proposedTokens),
             std::move(verifyInputTokens)},
            params, {uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS, 1, 1},
            {1, 1, 1});
}

} // namespace splash::ops
