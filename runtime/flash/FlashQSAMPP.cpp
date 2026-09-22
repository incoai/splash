#include "FlashQSAMPP.hpp"
#include "metal/abi/FlashQSAMPP.h"
#include "metal/abi/FlashQSAFast.h"
#include "metal/abi/FlashQSARowTiles.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash {
bool qsaOnlineMPPRowTilesEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_QSA_ROW_TILES");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("SPLASH_FLASH_QSA_ROW_TILES must be 0 or 1");
  }();
  return enabled;
}
const char *qsaOnlineMPPRouteSemantics() {
  if (!qsaOnlineMPPRowTilesEnabled()) return kFlashQSAOnlineMPPRoute;
  static const std::string identifier = std::string(kFlashQSAOnlineMPPRoute) + kFlashQSARowTilesRoute;
  return identifier.c_str();
}
bool qsaOnlineMPPRowTilesGeometry(uint32_t begin, uint32_t rows,
                                  uint32_t partitions) noexcept {
  return flash_qsa_row_tiles_geometry(begin, rows, partitions);
}
namespace {
void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!bytes || !buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash QSA MPP insufficient ") + name);
}
void distinct(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto *ap = a.contents(), *bp = b.contents();
  if (!ap || !bp)
    throw std::invalid_argument("Flash QSA MPP requires shared addressable views");
  const uintptr_t aa = reinterpret_cast<uintptr_t>(ap);
  const uintptr_t bb = reinterpret_cast<uintptr_t>(bp);
  if (aa <= bb ? uint64_t(bb - aa) < a.sizeBytes()
               : uint64_t(aa - bb) < b.sizeBytes())
    throw std::invalid_argument("Flash QSA MPP unsupported overlapping views");
}
}
FlashQSAMPPWorkspace allocateQSAMPPWorkspace(metal::MetalBackend &backend,
                                              uint32_t maximumRows) {
  if (!maximumRows || maximumRows > 128)
    throw std::invalid_argument("Flash QSA MPP workspace rows unsupported");
  const uint64_t elements = uint64_t(maximumRows) * 24 * kFlashQSATokenWidth;
  return {maximumRows,
          backend.allocateBuffer(elements * 4, metal::BufferStorage::Shared,
                                  "flash-qsa-mpp-scores"),
          backend.allocateBuffer(elements * 2, metal::BufferStorage::Shared,
                                  "flash-qsa-mpp-probabilities")};
}
uint64_t qsaOnlineMPPWorkspacePlannedBytes(uint32_t maximumRows,
                                          uint32_t maximumPartitions) {
  if (!maximumRows || maximumRows > 128 || !maximumPartitions ||
      maximumPartitions > 32 || (maximumPartitions & (maximumPartitions - 1)))
    throw std::invalid_argument("Flash QSA online MPP workspace rows/partitions unsupported");
  return uint64_t(maximumRows) * 24 * maximumPartitions * 258 * sizeof(float);
}
FlashQSAFastWorkspace allocateQSAOnlineMPPWorkspace(metal::MetalBackend &backend,
                                                     uint32_t maximumRows,
                                                     uint32_t maximumPartitions) {
  (void)qsaOnlineMPPWorkspacePlannedBytes(maximumRows, maximumPartitions);
  const uint64_t groups = uint64_t(maximumRows) * 24 * maximumPartitions;
  return {maximumRows, maximumPartitions,
          backend.allocateBuffer(groups * 2 * 4, metal::BufferStorage::Shared,
                                  "flash-qsa-online-mpp-partition-statistics"),
          backend.allocateBuffer(groups * 256 * 4, metal::BufferStorage::Shared,
                                  "flash-qsa-online-mpp-partition-values")};
}
uint32_t qsaOnlineMPPRoutePartitions(uint32_t begin, uint32_t rows) noexcept {
  if (!rows || rows > 128) return 0;
  if (rows >= 32) {
    if (begin == 0) return rows >= 64 ? 1 : 0;
    return 4;
  }
  if (rows <= 4 && begin >= 1024) return 32;
  if (rows > 4 && rows <= 12 && begin >= 512) return 16;
  if (rows > 12 && rows <= 16 && begin >= 512) return 8;
  return 0;
}
void addQSAAttentionMPP(metal::CommandGraph &graph,
                        metal::MetalBuffer qProjection,
                        const FlashQSAState &state,
                        const FlashQSAWorkspace &workspace,
                        FlashQSAMPPWorkspace &mppWorkspace,
                        metal::MetalBuffer output,
                        metal::MetalBuffer diagnostics,
                        uint32_t begin, uint32_t rows,
                        uint32_t scoreTile, bool f32Probabilities) {
  if (!rows || rows > 128 || rows > workspace.maximumRows ||
      rows > mppWorkspace.maximumRows || !state.capacity ||
      state.capacity > 262144 || workspace.capacity != state.capacity ||
      begin > state.capacity || rows > state.capacity - begin ||
      (scoreTile != 64 && scoreTile != 128))
    throw std::invalid_argument("Flash QSA MPP attention geometry unsupported");
  requireBytes(qProjection, uint64_t(rows) * 12288 * 2, "query/gate projection");
  requireBytes(workspace.queries, uint64_t(rows) * 6144 * 2, "prepared queries");
  requireBytes(workspace.selectedBlocks, uint64_t(rows) * 512 * 4, "selected blocks");
  requireBytes(state.keys, uint64_t(state.capacity) * 512 * 2, "key cache");
  requireBytes(state.values, uint64_t(state.capacity) * 512 * 2, "value cache");
  requireBytes(mppWorkspace.scores, uint64_t(rows) * 24 * 2051 * 4, "scores");
  requireBytes(mppWorkspace.probabilities, uint64_t(rows) * 24 * 2051 * 2,
                "probabilities");
  requireBytes(output, uint64_t(rows) * 6144 * 2, "output");
  requireBytes(diagnostics, 4, "diagnostics");
  const std::array immutable{qProjection, workspace.queries,
                             workspace.selectedBlocks, state.keys, state.values};
  const std::array writable{mppWorkspace.scores, mppWorkspace.probabilities,
                            output, diagnostics};
  for (const auto &a : immutable)
    for (const auto &b : writable) distinct(a, b);
  for (size_t i = 0; i < writable.size(); ++i)
    for (size_t j = i + 1; j < writable.size(); ++j)
      distinct(writable[i], writable[j]);
  const FlashQSAMPPParams params{
      {rows, begin, state.capacity, (state.capacity + 3) / 4, begin / 4,
       (begin + rows) / 4 - begin / 4, 0, 0, 1e-6f, 1e7f, 0, 0},
      scoreTile, {0, 0, 0}};
  // All queries use the selected complete blocks and a causal incomplete tail.
  // Dispatches cover only the largest visible count in this query chunk.
  const uint32_t largest = std::min(begin + rows, kFlashQSATokenWidth);
  graph.add("flash_qsa_mpp_scores_n" + std::to_string(scoreTile),
            {workspace.queries, state.keys, workspace.selectedBlocks,
             mppWorkspace.scores, diagnostics},
            params, {rows, 2, (largest + scoreTile - 1) / scoreTile}, {128, 1, 1});
  graph.add(f32Probabilities ? "flash_qsa_mpp_probabilities_f32"
                            : "flash_qsa_mpp_probabilities",
            {mppWorkspace.scores,
             f32Probabilities ? mppWorkspace.scores : mppWorkspace.probabilities,
             diagnostics},
            params, {rows, 24, 1});
  graph.add(f32Probabilities ? "flash_qsa_mpp_values_f32_gate"
                            : "flash_qsa_mpp_values_gate",
            {f32Probabilities ? mppWorkspace.scores : mppWorkspace.probabilities,
             state.values, workspace.selectedBlocks,
             qProjection, output, diagnostics},
            params, {rows, 2, 4}, {128, 1, 1});
}
void addQSAAttentionOnlineMPP(metal::CommandGraph &graph,
                              metal::MetalBuffer qProjection,
                              const FlashQSAState &state,
                              const FlashQSAWorkspace &workspace,
                              FlashQSAFastWorkspace &fastWorkspace,
                              metal::MetalBuffer output,
                              metal::MetalBuffer diagnostics,
                              uint32_t begin, uint32_t rows,
                              uint32_t partitions) {
  if (!rows || rows > 128 || rows > workspace.maximumRows ||
      rows > fastWorkspace.maximumRows || !state.capacity ||
      state.capacity > 262144 || workspace.capacity != state.capacity ||
      begin > state.capacity || rows > state.capacity - begin ||
      !fastWorkspace.maximumPartitions || fastWorkspace.maximumPartitions > 32 ||
      (fastWorkspace.maximumPartitions & (fastWorkspace.maximumPartitions - 1)) ||
      (partitions != 1 && partitions != 2 && partitions != 4 && partitions != 8 &&
       partitions != 16 && partitions != 32) ||
      partitions > fastWorkspace.maximumPartitions)
    throw std::invalid_argument("Flash QSA online MPP geometry unsupported");
  requireBytes(qProjection, uint64_t(rows) * 12288 * 2, "query/gate projection");
  requireBytes(workspace.queries, uint64_t(rows) * 6144 * 2, "prepared queries");
  requireBytes(workspace.selectedBlocks, uint64_t(rows) * 512 * 4, "selected blocks");
  requireBytes(state.keys, uint64_t(state.capacity) * 512 * 2, "key cache");
  requireBytes(state.values, uint64_t(state.capacity) * 512 * 2, "value cache");
  const uint64_t groups = uint64_t(rows) * 24 * fastWorkspace.maximumPartitions;
  requireBytes(fastWorkspace.partitionStatistics, groups * 2 * 4, "statistics");
  requireBytes(fastWorkspace.partitionValues, groups * 256 * 4, "numerators");
  requireBytes(output, uint64_t(rows) * 6144 * 2, "output");
  requireBytes(diagnostics, 4, "diagnostics");
  const std::array immutable{qProjection, workspace.queries,
                             workspace.selectedBlocks, state.keys, state.values};
  const std::array writable{fastWorkspace.partitionStatistics,
                            fastWorkspace.partitionValues, output, diagnostics};
  for (const auto &a : immutable)
    for (const auto &b : writable) distinct(a, b);
  for (size_t i = 0; i < writable.size(); ++i)
    for (size_t j = i + 1; j < writable.size(); ++j)
      distinct(writable[i], writable[j]);
  const FlashQSAFastParams params{
      {rows, begin, state.capacity, (state.capacity + 3) / 4, begin / 4,
       (begin + rows) / 4 - begin / 4, 0, 0, 1e-6f, 1e7f, 0, 0},
      0, 0, partitions, fastWorkspace.maximumPartitions};
  const bool temporalRows = qsaOnlineMPPRowTilesEnabled() &&
      qsaOnlineMPPRowTilesGeometry(begin, rows, partitions);
  graph.add(temporalRows ? "flash_qsa_mpp_prefill_rows_m32" : "flash_qsa_mpp_online_partition",
            {workspace.queries, state.keys, state.values, workspace.selectedBlocks,
             fastWorkspace.partitionStatistics, fastWorkspace.partitionValues,
             diagnostics},
            params, {temporalRows ? (uint64_t(rows) * 12 + 31) / 32 : rows, 2, partitions}, {128, 1, 1});
  graph.add("flash_qsa_fast_f32_reduce",
            {fastWorkspace.partitionStatistics, fastWorkspace.partitionValues,
             qProjection, output, diagnostics}, params, {rows, 24, 1});
}
void addQSAOnlineMPP(metal::CommandGraph &graph,
                     const FlashQSAFastInputs &input,
                     FlashQSAState &state, FlashQSAWorkspace &workspace,
                     FlashQSAFastWorkspace &fastWorkspace,
                     uint32_t begin, uint32_t rows,
                     uint32_t partitions,
                     bool fusePreparation) {
  metal::CommandGraph qualified;
  addQSAFast(qualified, input, state, workspace, fastWorkspace, begin, rows,
             FlashQSAFastMode::PartitionedF32Probabilities, std::min(partitions, 8u),
             fusePreparation);
  const auto dispatches = qualified.dispatches();
  if (dispatches.size() < 2 ||
      !dispatches[dispatches.size() - 2].pipelineName.starts_with(
          "flash_qsa_fast_f32_partition_c") ||
      dispatches.back().pipelineName != "flash_qsa_fast_f32_reduce")
    throw std::logic_error("Flash QSA online MPP qualified graph changed");
  for (size_t i = 0; i < dispatches.size() - 2; ++i) {
    const auto &dispatch = dispatches[i];
    if (dispatch.bytes.size() != 1)
      throw std::logic_error("Flash QSA online MPP prefix parameters changed");
    std::vector<metal::MetalBuffer> buffers;
    buffers.reserve(dispatch.buffers.size());
    for (uint32_t b = 0; b < dispatch.buffers.size(); ++b) {
      if (dispatch.buffers[b].index != b)
        throw std::logic_error("Flash QSA online MPP prefix buffer order changed");
      buffers.push_back(dispatch.buffers[b].buffer);
    }
    if (dispatch.bytes[0].sizeBytes == sizeof(FlashQSAFastParams)) {
      FlashQSAFastParams params;
      std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
      graph.add(dispatch.pipelineName, std::move(buffers), params,
                dispatch.threadgroups, dispatch.threadsPerThreadgroup);
    } else if (dispatch.bytes[0].sizeBytes == sizeof(FlashQSAParams)) {
      FlashQSAParams params;
      std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
      graph.add(dispatch.pipelineName, std::move(buffers), params,
                dispatch.threadgroups, dispatch.threadsPerThreadgroup);
    } else {
      throw std::logic_error("Flash QSA online MPP prefix ABI changed");
    }
  }
  addQSAAttentionOnlineMPP(graph, input.qProjection, state, workspace,
                           fastWorkspace, input.output, input.diagnostics,
                           begin, rows, partitions);
}
} // namespace splash::flash
