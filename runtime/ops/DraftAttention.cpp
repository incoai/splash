#include "ops/DraftAttention.hpp"

#include "metal/abi/DraftAttention.h"
#include "metal/abi/RoPE.h"
#include "ops/BufferExtent.hpp"
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
// Each split leaves a partial of its attention rows x (head dimensions + max
// + sum) fp32 values behind the grouped queries.
constexpr uint32_t kSplits = SPLASH_DRAFT_ATTENTION_SPLITS;
constexpr uint64_t kPartialBytes = uint64_t{SPLASH_DRAFT_ATTENTION_ROWS} *
                                   (SPLASH_DRAFT_HEAD_DIMENSION + 2) * sizeof(float);

// Dispatch width of one surrounding phase per lane: one 256-thread group per
// 256 elements and at least one group per whole-group task.
uint32_t phaseGroups(uint64_t elements, uint32_t tasks = 0) {
  return static_cast<uint32_t>(
      std::max<uint64_t>((elements + kThreads - 1) / kThreads, tasks));
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
  requireBytes(contextKv, rows * (shape.qkvSize - shape.attentionSize) * 2, "draft context K/V");
  requireBytes(keyNorm, uint64_t{shape.headDimension} * 2, "draft key norm");
  const uint64_t ropeBytes = rows * SPLASH_DRAFT_ROPE_PAIRS * 4;
  requireBytes(ropeCos, ropeBytes, "draft RoPE cosine");
  requireBytes(ropeSin, ropeBytes, "draft RoPE sine");
}

void requireLanes(uint32_t lanes) {
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid draft batch width");
}

enum class KernelLayout : uint8_t { Hidden5120, Hidden2048 };

// The compiled draft (metal/abi/DraftAttention.h) of each hidden size the
// convolution kernels take.
[[nodiscard]] KernelLayout kernelShape(DraftAttentionShape shape) {
  const auto compiled = [](uint32_t hidden) {
    return DraftAttentionShape{hidden,
                               draft_dynamic_width(hidden),
                               SPLASH_DRAFT_QKV_WIDTH,
                               SPLASH_DRAFT_ATTENTION_WIDTH,
                               SPLASH_DRAFT_QUERY_HEADS,
                               SPLASH_DRAFT_KV_HEADS,
                               SPLASH_DRAFT_HEAD_DIMENSION};
  };
  if (shape == compiled(5120))
    return KernelLayout::Hidden5120;
  if (shape == compiled(2048))
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
  requireBytes(buffers.input, workspace.convolutionBytes, "draft convolution input");
  requireBytes(buffers.output, workspace.convolutionBytes, "draft convolution output");
  requireBytes(buffers.residual, workspace.convolutionBytes, "draft convolution residual");
  requireBytes(buffers.dynamic, uint64_t{lanes} * kRows * shape.dynamicSize * 2, "draft dynamic convolution");
  requireBytes(buffers.weights,
               uint64_t{SPLASH_DRAFT_CONVOLUTION_STAGES} * SPLASH_DRAFT_CONVOLUTION_TAPS * shape.hiddenSize * 2,
               "draft convolution weight");
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
  requireBytes(buffers.qkv, workspace.qkvBytes, "draft q/k/v");
  requireBytes(buffers.groupedQueries, queryRowsBytes(plan), "draft grouped queries");
  requireBytes(buffers.queryKeys, workspace.queryKeysBytes, "draft query keys");
  requireBytes(buffers.queryValues, workspace.queryValuesBytes, "draft query values");
  requireBytes(buffers.queryNorm, uint64_t{shape.headDimension} * 2, "draft query norm");
  requireBytes(buffers.keyNorm, uint64_t{shape.headDimension} * 2, "draft key norm");
  const uint64_t ropeBytes = uint64_t{lanes} * kRows * SPLASH_DRAFT_ROPE_PAIRS * 4;
  requireBytes(buffers.ropeCos, ropeBytes, "draft RoPE cosine");
  requireBytes(buffers.ropeSin, ropeBytes, "draft RoPE sine");
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
  requireBytes(buffers.groupedQueries, workspace.groupedQueriesBytes, "draft grouped queries");
  requireBytes(buffers.queryKeys, workspace.queryKeysBytes, "draft query keys");
  requireBytes(buffers.queryValues, workspace.queryValuesBytes, "draft query values");
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (cacheLengths[lane] > SPLASH_MAXIMUM_CONTEXT_TOKENS)
      throw std::invalid_argument("draft attention cache length exceeds limit");
    requireBytes(buffers.persistentKeys[lane], ringBytes(shape), "draft key ring");
    requireBytes(buffers.persistentValues[lane], ringBytes(shape), "draft value ring");
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
                                metal::MetalBuffer rowMajor,
                                const DraftAttentionPlan &plan) {
  const auto shape = plan.shape();
  const uint32_t groups =
      phaseGroups(uint64_t{kRows} * shape.queryHeads * shape.headDimension);
  const uint32_t lanes = plan.lanes();
  requireBytes(grouped, queryRowsBytes(plan), "draft grouped attention");
  requireBytes(rowMajor, queryRowsBytes(plan), "draft attention");
  graph.add("draft_attention_reorder", {std::move(grouped), std::move(rowMajor)},
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
  requireBytes(keys, ringBytes(shape), "draft key ring");
  requireBytes(values, ringBytes(shape), "draft value ring");
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
  requireBytes(retainedCounts, uint64_t{lanes} * sizeof(uint32_t), "draft retained counts");
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBytes(persistentKeys[lane], ringBytes(shape), "draft key ring");
    requireBytes(persistentValues[lane], ringBytes(shape), "draft value ring");
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
