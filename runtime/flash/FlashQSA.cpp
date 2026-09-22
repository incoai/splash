#include "FlashQSA.hpp"

#include "metal/abi/FlashQSA.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {

constexpr uint32_t kMaximumRows = 128;
constexpr uint32_t kMaximumContext = 262144;

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!bytes || !buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash QSA insufficient ") + name);
}

void requireGeometry(uint32_t capacity) {
  if (!capacity || capacity > kMaximumContext)
    throw std::invalid_argument("Flash QSA capacity must be in [1,262144]");
}

std::string normPipeline(const FlashTensor &tensor, uint32_t dimension,
                         const char *prefix) {
  if (tensor.shape.size() != 1 || tensor.shape[0] != dimension ||
      (tensor.dtype != FlashDType::BF16 && tensor.dtype != FlashDType::F32))
    throw std::invalid_argument("Flash QSA norm must be BF16/F32[head dimension]");
  const uint64_t bytes = uint64_t(dimension) *
                         (tensor.dtype == FlashDType::BF16 ? 2 : 4);
  if (tensor.logicalBytes < bytes)
    throw std::invalid_argument("Flash QSA norm logical extent is too small");
  requireBytes(tensor.buffer, bytes, "norm weights");
  return std::string(prefix) +
         (tensor.dtype == FlashDType::BF16 ? "_bf16" : "_f32");
}

uint32_t convention(NormConvention value) {
  switch (value) {
  case NormConvention::OnePlusWeight: return 0;
  case NormConvention::DirectGamma: return 1;
  }
  throw std::invalid_argument("Flash QSA invalid norm convention");
}

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes,
                            const char *label) {
  return backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
}

} // namespace

FlashQSAState allocateQSAState(metal::MetalBackend &backend,
                               uint32_t capacity) {
  requireGeometry(capacity);
  const uint64_t blocks = (uint64_t(capacity) + 3) / 4;
  return {capacity,
          allocate(backend, uint64_t(capacity) * 512 * 2, "flash-qsa-key-cache"),
          allocate(backend, uint64_t(capacity) * 512 * 2, "flash-qsa-value-cache"),
          allocate(backend, uint64_t(capacity) * 128 * 2, "flash-qsa-raw-index-cache"),
          allocate(backend, blocks * 128 * 2, "flash-qsa-pooled-index-cache"),
          allocate(backend, uint64_t(capacity) * 8, "flash-qsa-index-positions")};
}

FlashQSAWorkspace allocateQSAWorkspace(metal::MetalBackend &backend,
                                       uint32_t maximumRows,
                                       uint32_t capacity) {
  requireGeometry(capacity);
  if (!maximumRows || maximumRows > kMaximumRows)
    throw std::invalid_argument("Flash QSA workspace rows must be in [1,128]");
  const uint64_t rows = maximumRows;
  const uint64_t blocks = (uint64_t(capacity) + 3) / 4;
  return {maximumRows, capacity,
          allocate(backend, rows * 6144 * 2, "flash-qsa-queries"),
          allocate(backend, rows * 512 * 2, "flash-qsa-index-queries"),
          allocate(backend, rows * blocks * 4, "flash-qsa-block-scores"),
          allocate(backend, rows * 512 * 4, "flash-qsa-selected-blocks"),
          allocate(backend, rows * 24 * kFlashQSATokenWidth * 4,
                    "flash-qsa-attention-scores"),
          allocate(backend, rows * 24 * kFlashQSATokenWidth * 2,
                    "flash-qsa-probabilities")};
}

void addQSA(metal::CommandGraph &graph, metal::MetalBuffer qProjection,
            metal::MetalBuffer kProjection, metal::MetalBuffer vProjection,
            metal::MetalBuffer indexProjection, const FlashTensor &qNorm,
            const FlashTensor &kNorm, const FlashTensor &indexQNorm,
            const FlashTensor &indexKNorm, FlashQSAState &state,
            FlashQSAWorkspace &workspace, metal::MetalBuffer output,
            metal::MetalBuffer diagnostics, uint32_t begin, uint32_t rows,
            NormConvention qConvention, NormConvention kConvention,
            NormConvention indexQConvention, NormConvention indexKConvention,
            double epsilon, double theta, metal::MetalBuffer positions) {
  requireGeometry(state.capacity);
  if (!rows || rows > workspace.maximumRows || rows > kMaximumRows ||
      workspace.capacity != state.capacity || begin > state.capacity ||
      rows > state.capacity - begin || !std::isfinite(epsilon) ||
      epsilon <= 0 || epsilon > 1 || !std::isfinite(theta) || theta <= 1 ||
      theta > 1e12)
    throw std::invalid_argument("Flash QSA invalid append/workspace geometry");
  const uint64_t n = rows, capacity = state.capacity;
  const uint32_t blocks = (state.capacity + 3) / 4;
  requireBytes(qProjection, n * 12288 * 2, "q/gate projection");
  requireBytes(kProjection, n * 512 * 2, "key projection");
  requireBytes(vProjection, n * 512 * 2, "value projection");
  requireBytes(indexProjection, n * 640 * 2, "index projection");
  requireBytes(output, n * 6144 * 2, "output");
  requireBytes(diagnostics, 4, "diagnostics");
  if (positions)
    requireBytes(positions, n * 8, "positions");
  requireBytes(state.keys, capacity * 512 * 2, "key cache");
  requireBytes(state.values, capacity * 512 * 2, "value cache");
  requireBytes(state.rawIndexKeys, capacity * 128 * 2, "raw index cache");
  requireBytes(state.pooledKeys, uint64_t(blocks) * 128 * 2, "pooled index cache");
  requireBytes(state.indexPositions, capacity * 8, "index positions");
  requireBytes(workspace.queries, n * 6144 * 2, "query scratch");
  requireBytes(workspace.indexQueries, n * 512 * 2, "index query scratch");
  requireBytes(workspace.blockScores, n * blocks * 4, "block score scratch");
  requireBytes(workspace.selectedBlocks, n * 512 * 4, "selection scratch");
  requireBytes(workspace.attentionScores, n * 24 * kFlashQSATokenWidth * 4,
                "attention score scratch");
  requireBytes(workspace.probabilities, n * 24 * kFlashQSATokenWidth * 2,
                "probability scratch");
  for (const auto &input : {qProjection, kProjection, vProjection, indexProjection})
    if (input.sameView(output))
      throw std::invalid_argument("Flash QSA input/output must be distinct");

  const auto qPipeline = normPipeline(qNorm, 256, "flash_qsa_norm_rope");
  const auto kPipeline = normPipeline(kNorm, 256, "flash_qsa_norm_rope");
  const auto iqPipeline = normPipeline(indexQNorm, 128, "flash_qsa_norm_rope");
  const auto ikPipeline = normPipeline(indexKNorm, 128, "flash_qsa_pool_rope");
  FlashQSAParams params{rows, begin, state.capacity, blocks, begin / 4,
                        (begin + rows) / 4 - begin / 4, 0, 0,
                        static_cast<float>(epsilon), static_cast<float>(theta),
                        positions ? 1u : 0u, 0};
  // The dummy position binding is not accessed unless positions_supplied=1.
  const auto positionBinding = positions ? positions : state.indexPositions;
  params.norm_convention = convention(qConvention);
  graph.add(qPipeline, {qProjection, qNorm.buffer, positionBinding,
                         workspace.queries, diagnostics},
            params, {rows, 24, 1});
  params.mode = 1;
  params.norm_convention = convention(kConvention);
  graph.add(kPipeline, {kProjection, kNorm.buffer, positionBinding,
                         state.keys, diagnostics}, params, {rows, 2, 1});
  params.mode = 2;
  params.norm_convention = convention(indexQConvention);
  graph.add(iqPipeline, {indexProjection, indexQNorm.buffer, positionBinding,
                          workspace.indexQueries, diagnostics},
            params, {rows, 4, 1});
  graph.add("flash_qsa_append_aux", {vProjection, indexProjection, positionBinding,
                                     state.values, state.rawIndexKeys,
                                     state.indexPositions, diagnostics},
            params, {rows, 1, 1});
  params.norm_convention = convention(indexKConvention);
  if (params.new_blocks)
    graph.add(ikPipeline, {state.rawIndexKeys, indexKNorm.buffer,
                           state.indexPositions, state.pooledKeys, diagnostics},
              params, {params.new_blocks, 1, 1});
  // Short queries select all completed blocks. The score dispatch does no
  // arithmetic for them, even when a later row in this append is sparse.
  const uint32_t completeBlocks = (begin + rows) / 4;
  if (completeBlocks > kFlashQSABlockBudget)
    graph.add("flash_qsa_index_scores", {workspace.indexQueries, state.pooledKeys,
                                         workspace.blockScores, diagnostics},
              params, {(completeBlocks + 7) / 8, rows, 1});
  graph.add("flash_qsa_select_blocks", {workspace.blockScores,
                                        workspace.selectedBlocks, diagnostics},
            params, {rows, 1, 1});
  graph.add("flash_qsa_main_scores", {workspace.queries, state.keys,
                                      workspace.selectedBlocks,
                                      workspace.attentionScores, diagnostics},
            params, {(kFlashQSATokenWidth + 7) / 8, rows, 24});
  graph.add("flash_qsa_probabilities", {workspace.attentionScores,
                                        workspace.probabilities, diagnostics},
            params, {rows, 24, 1});
  graph.add("flash_qsa_values_gate", {workspace.probabilities, state.values,
                                      workspace.selectedBlocks, qProjection,
                                      output, diagnostics},
            params, {rows, 24, 1});
}

} // namespace splash::flash
