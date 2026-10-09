#include "ops/Embedding.hpp"

#include "metal/abi/Embedding.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {

NativeRows::NativeRows(metal::MetalBuffer rows, uint32_t formatId) : rows(std::move(rows)), formatId(formatId) {
  if (!gguf_embedding_format(formatId)) throw std::invalid_argument("unsupported native embedding format");
}
const char *NativeRows::name() const noexcept { return kQuantFormats[formatId].name; }

void Embedding::add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    const EmbeddingWeights &table, metal::MetalBuffer output,
                    uint32_t rows) {
  if (!rows || !table.outputSize || !table.inputSize)
    throw std::invalid_argument("invalid embedding shape");
  // Every gather reads `rows` token ids and writes `rows` bf16 rows of the table's width.
  requireBytes(tokens, uint64_t{rows} * sizeof(uint32_t), "embedding token");
  requireBytes(output, uint64_t{rows} * table.inputSize * sizeof(uint16_t), "embedding output");
  const NativeRows &native = table.blocks();
  // Every token's row of native blocks (kernels/shared/embedding.metal).
  const QuantFormat &format = kQuantFormats[native.formatId];
  if (table.inputSize % format.block_elements)
    throw std::invalid_argument("native token rows take whole blocks");
  requireBytes(native.rows, uint64_t{table.outputSize} * (table.inputSize / format.block_elements) * format.block_bytes,
               "token table");
  const GgufEmbedParams params{rows, table.outputSize, table.inputSize};
  if (table.rotation) {
    // One threadgroup per rotation block of a row, which gathers the block
    // and inverts its rotation in fp32 (kernels/shared/gguf_rotation.metal).
    if (native.formatId != GGUF_FMT_PQ20 || table.inputSize % GGUF_ROTATION_BLOCK)
      throw std::invalid_argument("a rotated token table takes PQ2_0 rows of whole rotation blocks");
    requireBytes(table.rotation.signs, table.inputSize, "embedding rotation sign");
    graph.add("gguf_embed_rotated_pq20", {std::move(tokens), native.rows, table.rotation.signs, std::move(output)},
              params, {table.inputSize / GGUF_ROTATION_BLOCK, rows, 1}, {GGUF_ROTATION_THREADS, 1, 1});
    return;
  }
  graph.add(std::string("gguf_embed_") + native.name(), {std::move(tokens), native.rows, std::move(output)}, params,
            {(rows * table.inputSize + 255) / 256, 1, 1}, {256, 1, 1});
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
