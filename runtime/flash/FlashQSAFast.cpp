#include "FlashQSAFast.hpp"

#include "metal/abi/FlashQSAFast.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!bytes || !buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash QSA fast insufficient ") + name);
}

uint32_t convention(NormConvention value) {
  switch (value) {
  case NormConvention::OnePlusWeight: return 0;
  case NormConvention::DirectGamma: return 1;
  }
  throw std::invalid_argument("Flash QSA fast invalid norm convention");
}

void copyDispatch(metal::CommandGraph &graph, const metal::ComputeDispatch &dispatch) {
  if (dispatch.bytes.size() != 1 ||
      dispatch.bytes[0].sizeBytes != sizeof(FlashQSAParams))
    throw std::logic_error("Flash QSA fast qualified prefix ABI changed");
  FlashQSAParams params;
  std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
  std::vector<metal::MetalBuffer> buffers;
  buffers.reserve(dispatch.buffers.size());
  for (uint32_t index = 0; index < dispatch.buffers.size(); ++index) {
    if (dispatch.buffers[index].index != index)
      throw std::logic_error("Flash QSA fast qualified buffer order changed");
    buffers.push_back(dispatch.buffers[index].buffer);
  }
  graph.add(dispatch.pipelineName, std::move(buffers), params,
            dispatch.threadgroups, dispatch.threadsPerThreadgroup);
}

FlashQSAFastParams parameters(uint32_t rows, uint32_t begin, uint32_t capacity,
                              uint32_t partitions, uint32_t maximumPartitions) {
  return {{rows, begin, capacity, (capacity + 3) / 4, begin / 4,
            (begin + rows) / 4 - begin / 4, 0, 0, 1e-6f, 1e7f, 0, 0},
          0, 0, partitions, maximumPartitions};
}

} // namespace

FlashQSAFastWorkspace allocateQSAFastWorkspace(metal::MetalBackend &backend,
                                               uint32_t maximumRows,
                                               uint32_t maximumPartitions) {
  if (!maximumRows || maximumRows > 128 || !maximumPartitions ||
      maximumPartitions > 8)
    throw std::invalid_argument("Flash QSA fast workspace rows/partitions unsupported");
  const uint64_t groups = uint64_t(maximumRows) * 24 * maximumPartitions;
  return {maximumRows, maximumPartitions,
          backend.allocateBuffer(groups * 2 * 4, metal::BufferStorage::Shared,
                                  "flash-qsa-fast-partition-statistics"),
          backend.allocateBuffer(groups * 256 * 4, metal::BufferStorage::Shared,
                                  "flash-qsa-fast-partition-values")};
}

void addQSAAttentionFast(metal::CommandGraph &graph,
                         metal::MetalBuffer qProjection,
                         const FlashQSAState &state,
                         const FlashQSAWorkspace &workspace,
                         FlashQSAFastWorkspace &fastWorkspace,
                         metal::MetalBuffer output,
                         metal::MetalBuffer diagnostics, uint32_t begin,
                         uint32_t rows, FlashQSAFastMode mode,
                         uint32_t partitions) {
  if (!rows || rows > 128 || rows > workspace.maximumRows ||
      !state.capacity || state.capacity > 262144 ||
      workspace.capacity != state.capacity || begin > state.capacity ||
      rows > state.capacity - begin)
    throw std::invalid_argument("Flash QSA fast attention geometry unsupported");
  requireBytes(qProjection, uint64_t(rows) * 12288 * 2, "query/gate projection");
  requireBytes(workspace.queries, uint64_t(rows) * 6144 * 2, "prepared queries");
  requireBytes(workspace.selectedBlocks, uint64_t(rows) * 512 * 4, "selected blocks");
  requireBytes(state.keys, uint64_t(state.capacity) * 512 * 2, "key cache");
  requireBytes(state.values, uint64_t(state.capacity) * 512 * 2, "value cache");
  requireBytes(output, uint64_t(rows) * 6144 * 2, "output");
  requireBytes(diagnostics, 4, "diagnostics");
  if (qProjection.sameView(output) || workspace.queries.sameView(output) ||
      state.keys.sameView(output) || state.values.sameView(output))
    throw std::invalid_argument("Flash QSA fast attention input/output aliases");
  if (mode == FlashQSAFastMode::CanonicalBF16Probabilities) {
    const auto params = parameters(rows, begin, state.capacity, 1, 1);
    graph.add("flash_qsa_fast_canonical", {workspace.queries, state.keys, state.values,
                                           workspace.selectedBlocks, qProjection,
                                           output, diagnostics},
              params, {rows, 24, 1});
    return;
  }
  if (mode != FlashQSAFastMode::PartitionedF32Probabilities ||
      (partitions != 1 && partitions != 2 && partitions != 4 && partitions != 8) ||
      partitions > fastWorkspace.maximumPartitions ||
      rows > fastWorkspace.maximumRows)
    throw std::invalid_argument("Flash QSA fast numerical mode/partitions unsupported");
  const uint64_t groups = uint64_t(rows) * 24 * fastWorkspace.maximumPartitions;
  requireBytes(fastWorkspace.partitionStatistics, groups * 2 * 4,
                "partition statistics");
  requireBytes(fastWorkspace.partitionValues, groups * 256 * 4,
                "partition values");
  const auto params = parameters(rows, begin, state.capacity, partitions,
                                  fastWorkspace.maximumPartitions);
  const uint32_t chunkCapacity = (kFlashQSATokenWidth + partitions - 1) / partitions;
  const std::string partitionPipeline =
      "flash_qsa_fast_f32_partition_c" + std::to_string(chunkCapacity);
  graph.add(partitionPipeline, {workspace.queries, state.keys, state.values,
                                            workspace.selectedBlocks,
                                            fastWorkspace.partitionStatistics,
                                            fastWorkspace.partitionValues, diagnostics},
            params, {rows, 24, partitions});
  graph.add("flash_qsa_fast_f32_reduce", {fastWorkspace.partitionStatistics,
                                         fastWorkspace.partitionValues, qProjection,
                                         output, diagnostics}, params, {rows, 24, 1});
}

void addQSAFast(metal::CommandGraph &graph, const FlashQSAFastInputs &input,
                FlashQSAState &state, FlashQSAWorkspace &workspace,
                FlashQSAFastWorkspace &fastWorkspace, uint32_t begin,
                uint32_t rows, FlashQSAFastMode mode, uint32_t partitions,
                bool fusePreparation) {
  if (!input.qNorm || !input.kNorm || !input.indexQNorm || !input.indexKNorm)
    throw std::invalid_argument("Flash QSA fast missing norm tensor");
  // Reuse qualified argument checks and build its authoritative prefix rather
  // than maintaining a second pooling/indexer/selection routing policy.
  metal::CommandGraph qualified;
  addQSA(qualified, input.qProjection, input.kProjection, input.vProjection,
          input.indexProjection, *input.qNorm, *input.kNorm, *input.indexQNorm,
          *input.indexKNorm, state, workspace, input.output, input.diagnostics,
          begin, rows, input.qConvention, input.kConvention,
          input.indexQConvention, input.indexKConvention, input.epsilon,
          input.theta, input.positions);
  const auto dispatches = qualified.dispatches();
  if (dispatches.size() < 8 ||
      dispatches[dispatches.size() - 3].pipelineName != "flash_qsa_main_scores" ||
      dispatches[dispatches.size() - 2].pipelineName != "flash_qsa_probabilities" ||
      dispatches[dispatches.size() - 1].pipelineName != "flash_qsa_values_gate")
    throw std::logic_error("Flash QSA fast qualified attention graph changed");
  uint64_t prefixStart = 0;
  if (fusePreparation) {
    if (dispatches[3].pipelineName != "flash_qsa_append_aux")
      throw std::logic_error("Flash QSA fast qualified preparation graph changed");
    FlashQSAFastParams params = parameters(rows, begin, state.capacity,
                                           partitions, fastWorkspace.maximumPartitions);
    std::memcpy(&params.common, dispatches[0].bytes[0].data, sizeof(FlashQSAParams));
    params.norm_dtype_mask =
        (input.qNorm->dtype == FlashDType::F32 ? 1u : 0u) |
        (input.kNorm->dtype == FlashDType::F32 ? 2u : 0u) |
        (input.indexQNorm->dtype == FlashDType::F32 ? 4u : 0u);
    params.norm_convention_bits = convention(input.qConvention) |
        (convention(input.kConvention) << 2) |
        (convention(input.indexQConvention) << 4);
    const auto positions = input.positions ? input.positions : state.indexPositions;
    graph.add("flash_qsa_fast_prepare", {input.qProjection, input.kProjection,
                                        input.vProjection, input.indexProjection,
                                        input.qNorm->buffer, input.kNorm->buffer,
                                        input.indexQNorm->buffer, positions,
                                        workspace.queries, state.keys,
                                        workspace.indexQueries, state.values,
                                        state.rawIndexKeys, state.indexPositions,
                                        input.diagnostics}, params, {rows, 30, 1});
    prefixStart = 4;
  }
  for (uint64_t index = prefixStart; index < dispatches.size() - 3; ++index)
    copyDispatch(graph, dispatches[index]);
  addQSAAttentionFast(graph, input.qProjection, state, workspace, fastWorkspace,
                      input.output, input.diagnostics, begin, rows, mode, partitions);
}

} // namespace splash::flash
