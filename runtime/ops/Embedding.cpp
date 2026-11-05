#include "ops/Embedding.hpp"

#include "metal/abi/Embedding.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {

StoredRows::StoredRows(metal::MetalBuffer rows, uint32_t formatId) : rows(std::move(rows)), formatId(formatId) {
  if (formatId >= GGUF_FMT_COUNT) throw std::invalid_argument("unsupported embedding format");
}
const QuantFormat &StoredRows::format() const noexcept { return kQuantFormats[formatId]; }

void Embedding::add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    const EmbeddingWeights &table, metal::MetalBuffer output,
                    uint32_t rows) {
  if (!rows || !table.outputSize || !table.inputSize)
    throw std::invalid_argument("invalid embedding shape");
  // Every gather reads `rows` token ids and writes `rows` bf16 rows of the table's width.
  requireBytes(tokens, uint64_t{rows} * sizeof(uint32_t), "embedding token");
  requireBytes(output, uint64_t{rows} * table.inputSize * sizeof(uint16_t), "embedding output");
  const StoredRows &stored = table.rows();
  const QuantFormat &format = stored.format();
  // Every token's row as stored, which the gather reads (kernels/shared/embedding.metal).
  const uint64_t rowBytes = quant_stored_row_bytes(stored.formatId, table.inputSize);
  if (!rowBytes) throw std::invalid_argument("token rows take whole blocks of their format");
  requireBytes(stored.rows, uint64_t{table.outputSize} * rowBytes, "token table");
  const GgufEmbedParams params{rows, table.outputSize, table.inputSize};
  if (table.rotation) {
    // One threadgroup per rotation block of a row, which gathers the block
    // and inverts its rotation in fp32 (kernels/shared/gguf_rotation.metal).
    if (stored.formatId != GGUF_FMT_PQ20 || table.inputSize % GGUF_ROTATION_BLOCK)
      throw std::invalid_argument("a rotated token table takes PQ2_0 rows of whole rotation blocks");
    requireBytes(table.rotation.signs, table.inputSize, "embedding rotation sign");
    graph.add("gguf_embed_rotated_pq20", {std::move(tokens), stored.rows, table.rotation.signs, std::move(output)},
              params, {table.inputSize / GGUF_ROTATION_BLOCK, rows, 1}, {GGUF_ROTATION_THREADS, 1, 1});
    return;
  }
  graph.add(std::string("gguf_embed_") + format.name, {std::move(tokens), stored.rows, std::move(output)}, params,
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

void Embedding::addDraftMaskRows(metal::CommandGraph &graph, metal::MetalBuffer mask, metal::MetalBuffer rows,
                                 uint32_t width, uint32_t lanes) {
  if (!width || !lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft mask rows");
  requireBytes(mask, uint64_t{width} * sizeof(uint16_t), "draft mask embedding");
  requireBytes(rows, uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS * width * sizeof(uint16_t), "draft input row");
  const DraftMaskRowsParams params{width};
  graph.add("draft_mask_rows", {std::move(mask), std::move(rows)}, params,
            {(width + 255) / 256, SPLASH_TARGET_VERIFY_ROWS - 1, lanes}, {256, 1, 1});
}

} // namespace splash::ops
