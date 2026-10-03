#include "ops/DraftAttention.hpp"

#include "metal/abi/DraftAttention.h"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kWindow = SPLASH_DRAFT_SLIDING_WINDOW;
constexpr uint32_t kThreads = metal::CommandGraph::kDefaultThreads;
// Each split leaves a 32-row x (128 + max + sum) fp32 partial behind the
// grouped queries.
constexpr uint32_t kSplits = SPLASH_DRAFT_ATTENTION_SPLITS;
constexpr uint32_t kAttentionRows = 32;
constexpr uint64_t kPartialBytes =
    uint64_t{kAttentionRows} * (128 + 2) * sizeof(float);

// Dispatch width of one surrounding phase per lane: one 256-thread group per
// 256 elements and at least one group per whole-group task.
uint32_t phaseGroups(uint64_t elements, uint32_t tasks = 0) {
  return static_cast<uint32_t>(
      std::max<uint64_t>((elements + kThreads - 1) / kThreads, tasks));
}

void requireBuffer(const metal::MetalBuffer &buffer, uint64_t bytes) {
  if (!buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument("draft attention buffer is below plan size");
}

// The grouped query rows alone, without the split partials behind them.
uint64_t queryRowsBytes(const DraftAttentionPlan &plan) {
  return uint64_t{plan.lanes()} * kRows * plan.shape().attentionSize * 2;
}

// One lane's keys, or values, of one layer: the kernels place a position in
// slot position % kWindow of each KV head's ring.
uint64_t ringBytes(DraftAttentionShape shape) {
  return uint64_t{shape.kvHeads} * kWindow * shape.headDimension * 2;
}

// What the context writers read for `rows` rows: the key and value rows,
// the key norm and the rows' RoPE tables (kernels/common/draft_context_kv.h).
void requireContextInputs(const metal::MetalBuffer &contextKv,
                          const metal::MetalBuffer &keyNorm,
                          const metal::MetalBuffer &ropeCos,
                          const metal::MetalBuffer &ropeSin, uint64_t rows,
                          DraftAttentionShape shape) {
  requireBuffer(contextKv, rows * (shape.qkvSize - shape.attentionSize) * 2);
  requireBuffer(keyNorm, uint64_t{shape.headDimension} * 2);
  const uint64_t ropeBytes = rows * shape.headDimension / 2 * 4;
  requireBuffer(ropeCos, ropeBytes);
  requireBuffer(ropeSin, ropeBytes);
}

void requireLanes(uint32_t lanes) {
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid draft batch width");
}

enum class KernelLayout : uint8_t { Hidden5120, Hidden2048 };

[[nodiscard]] KernelLayout kernelShape(DraftAttentionShape shape) {
  if (shape == DraftAttentionShape{5120, 1280, 6144, 4096, 32, 8, 128})
    return KernelLayout::Hidden5120;
  if (shape == DraftAttentionShape{2048, 512, 6144, 4096, 32, 8, 128})
    return KernelLayout::Hidden2048;
  throw std::invalid_argument("unsupported compiled draft attention shape");
}

} // namespace

DraftAttentionWorkspace DraftAttentionPlan::workspace() const noexcept {
  const uint64_t rows = uint64_t{lanes_} * kRows;
  const uint64_t kvBytes = rows * shape_.kvHeads * shape_.headDimension * 2;
  const uint64_t partialBytes =
      uint64_t{lanes_} * shape_.kvHeads * kSplits * kPartialBytes;
  return {rows * shape_.hiddenSize * 2, rows * shape_.qkvSize * 2,
          rows * shape_.attentionSize * 2 + partialBytes, kvBytes, kvBytes};
}

DraftAttentionPlan DraftAttention::plan(DraftAttentionShape shape,
                                        uint32_t lanes) {
  requireLanes(lanes);
  static_cast<void>(kernelShape(shape));
  return {shape, lanes};
}

void DraftAttention::addConvolution(metal::CommandGraph &graph,
                                    DraftConvolutionBuffers buffers,
                                    const DraftAttentionPlan &plan,
                                    DraftConvolutionStage stage) {
  // Every stage has a case, so a new one fails -Wswitch here.
  uint32_t finish = 0;
  switch (stage) {
  case DraftConvolutionStage::Prepare:
    break;
  case DraftConvolutionStage::Residual:
    finish = 1;
    break;
  }
  const auto shape = plan.shape();
  const auto workspace = plan.workspace();
  const uint32_t groups = phaseGroups(uint64_t{kRows} * shape.hiddenSize);
  const uint32_t lanes = plan.lanes();
  requireBuffer(buffers.input, workspace.convolutionBytes);
  requireBuffer(buffers.output, workspace.convolutionBytes);
  requireBuffer(buffers.residual, workspace.convolutionBytes);
  requireBuffer(buffers.dynamic, uint64_t{lanes} * kRows * shape.dynamicSize * 2);
  requireBuffer(buffers.weights, uint64_t{4} * shape.hiddenSize * 2);
  const KernelLayout kernel = kernelShape(shape);
  const DraftConvBatchParams params{finish};
  graph.add(kernel == KernelLayout::Hidden5120 ? "draft_conv"
                                               : "draft_conv_h2048",
            {std::move(buffers.input), std::move(buffers.dynamic),
             std::move(buffers.weights), std::move(buffers.residual),
             std::move(buffers.output)},
            params, {groups, lanes, 1});
}

void DraftAttention::addPrepare(metal::CommandGraph &graph,
                                DraftPrepareBuffers buffers,
                                const DraftAttentionPlan &plan) {
  const auto shape = plan.shape();
  const auto workspace = plan.workspace();
  // The prepare kernel copies one element of the value rows per thread and
  // normalizes one (row, head) per group.
  const uint32_t groups = phaseGroups(
      uint64_t{kRows} * shape.kvHeads * shape.headDimension,
      kRows * (shape.queryHeads + shape.kvHeads));
  const uint32_t lanes = plan.lanes();
  requireBuffer(buffers.qkv, workspace.qkvBytes);
  requireBuffer(buffers.groupedQueries, queryRowsBytes(plan));
  requireBuffer(buffers.queryKeys, workspace.queryKeysBytes);
  requireBuffer(buffers.queryValues, workspace.queryValuesBytes);
  requireBuffer(buffers.queryNorm, uint64_t{shape.headDimension} * 2);
  requireBuffer(buffers.keyNorm, uint64_t{shape.headDimension} * 2);
  const uint64_t ropeBytes = uint64_t{lanes} * kRows * shape.headDimension / 2 * 4;
  requireBuffer(buffers.ropeCos, ropeBytes);
  requireBuffer(buffers.ropeSin, ropeBytes);
  graph.add("draft_attention_qkv",
            {std::move(buffers.qkv), std::move(buffers.groupedQueries),
             std::move(buffers.queryNorm), std::move(buffers.keyNorm),
             std::move(buffers.ropeCos), std::move(buffers.ropeSin),
             std::move(buffers.queryKeys), std::move(buffers.queryValues)},
            {groups, lanes, 1});
}

void DraftAttention::addDecode(
    metal::CommandGraph &graph, DraftDecodeAttentionBuffers buffers,
    std::span<const uint32_t> cacheLengths, const DraftAttentionPlan &plan) {
  const auto shape = plan.shape();
  const auto workspace = plan.workspace();
  const uint32_t lanes = plan.lanes();
  if (cacheLengths.size() != lanes ||
      buffers.persistentKeys.size() != kMaximumLanes ||
      buffers.persistentValues.size() != kMaximumLanes)
    throw std::invalid_argument("invalid draft attention geometry");
  requireBuffer(buffers.groupedQueries, workspace.groupedQueriesBytes);
  requireBuffer(buffers.queryKeys, workspace.queryKeysBytes);
  requireBuffer(buffers.queryValues, workspace.queryValuesBytes);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (cacheLengths[lane] > SPLASH_MAXIMUM_CONTEXT_TOKENS)
      throw std::invalid_argument("draft attention cache length exceeds limit");
    requireBuffer(buffers.persistentKeys[lane], ringBytes(shape));
    requireBuffer(buffers.persistentValues[lane], ringBytes(shape));
  }
  DraftAttentionBatchParams params{kWindow, lanes, {}};
  std::copy(cacheLengths.begin(), cacheLengths.end(),
            std::begin(params.cache_length));
  std::vector<metal::MetalBuffer> bindings{buffers.groupedQueries};
  bindings.reserve(2 * kMaximumLanes + 3);
  appendLaneBindings(bindings, buffers.persistentKeys,
                     buffers.persistentValues);
  bindings.push_back(buffers.queryKeys);
  bindings.push_back(buffers.queryValues);
  graph.add("draft_attention_bf16_split", std::move(bindings), params,
            {shape.kvHeads, lanes, kSplits});
  graph.add("draft_attention_bf16_reduce", {buffers.groupedQueries}, params,
            {shape.kvHeads, lanes, 1});
}

void DraftAttention::addReorder(metal::CommandGraph &graph,
                                metal::MetalBuffer grouped,
                                metal::MetalBuffer packed,
                                const DraftAttentionPlan &plan) {
  const auto shape = plan.shape();
  const uint32_t groups =
      phaseGroups(uint64_t{kRows} * shape.queryHeads * shape.headDimension);
  const uint32_t lanes = plan.lanes();
  requireBuffer(grouped, queryRowsBytes(plan));
  requireBuffer(packed, queryRowsBytes(plan));
  graph.add("draft_attention_reorder", {std::move(grouped), std::move(packed)},
            {groups, lanes, 1});
}

void DraftAttention::addContextPrefill(
    metal::CommandGraph &graph, metal::MetalBuffer contextKv,
    metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
    metal::MetalBuffer ropeSin, metal::MetalBuffer keys,
    metal::MetalBuffer values, uint32_t tokens, uint32_t startPosition,
    DraftAttentionShape shape) {
  static_cast<void>(kernelShape(shape));
  if (!tokens)
    throw std::invalid_argument("invalid draft context prefill geometry");
  requireContextInputs(contextKv, keyNorm, ropeCos, ropeSin, tokens, shape);
  requireBuffer(keys, ringBytes(shape));
  requireBuffer(values, ringBytes(shape));
  const DraftContextParams params{tokens, startPosition};
  graph.add("prefill_draft_context_kv",
            {std::move(contextKv), std::move(keyNorm), std::move(ropeCos),
             std::move(ropeSin), std::move(keys), std::move(values)},
            params, {uint64_t{tokens} * shape.kvHeads, 1, 1});
}

void DraftAttention::addContextCommit(
    metal::CommandGraph &graph, metal::MetalBuffer contextKv,
    metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
    metal::MetalBuffer ropeSin,
    std::span<const metal::MetalBuffer> persistentKeys,
    std::span<const metal::MetalBuffer> persistentValues,
    metal::MetalBuffer retainedCounts,
    std::span<const uint32_t> startPositions, DraftAttentionShape shape) {
  const auto lanes = static_cast<uint32_t>(startPositions.size());
  requireLanes(lanes);
  static_cast<void>(kernelShape(shape));
  if (persistentKeys.size() != kMaximumLanes ||
      persistentValues.size() != kMaximumLanes)
    throw std::invalid_argument("invalid draft context commit geometry");
  // Each lane commits up to its eight verify rows.
  requireContextInputs(contextKv, keyNorm, ropeCos, ropeSin,
                       uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS, shape);
  requireBuffer(retainedCounts, uint64_t{lanes} * sizeof(uint32_t));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBuffer(persistentKeys[lane], ringBytes(shape));
    requireBuffer(persistentValues[lane], ringBytes(shape));
  }
  DraftContextBatchParams params{};
  std::copy(startPositions.begin(), startPositions.end(),
            std::begin(params.start_position));
  std::vector<metal::MetalBuffer> bindings{
      std::move(contextKv), std::move(keyNorm), std::move(ropeCos),
      std::move(ropeSin)};
  bindings.reserve(2 * kMaximumLanes + 5);
  appendLaneBindings(bindings, persistentKeys, persistentValues);
  bindings.push_back(std::move(retainedCounts));
  graph.add("draft_context_kv_commit", std::move(bindings), params,
            {uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS * shape.kvHeads, 1,
             1});
}

} // namespace splash::ops
