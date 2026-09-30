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
constexpr std::array kConfigurations{DraftAttentionConfiguration{},
                                      DraftAttentionConfiguration{32},
                                      DraftAttentionConfiguration{60},
                                      DraftAttentionConfiguration{80}};
constexpr uint32_t kThreads = metal::CommandGraph::kDefaultThreads;
// The live window tiles of one (lane, KV head) are dealt round-robin to this
// many groups, the last of which also attends the eight current rows. Fixed
// rather than derived from the GPU so the combine order, and with it the
// rounding, is the same on every machine and lane count. Each split leaves a
// 32-row x (128 + max + sum) fp32 partial behind the grouped queries.
constexpr uint32_t kSplits = 4;
constexpr uint32_t kAttentionRows = 32;
constexpr uint64_t kPartialBytes =
    uint64_t{kAttentionRows} * (128 + 2) * sizeof(float);

// Dispatch width of one surrounding phase per lane: the configured persistent
// count, or else one 256-thread group per 256 elements and at least one group
// per whole-group task.
uint32_t phaseGroups(const DraftAttentionPlan &plan, uint64_t elements,
                     uint32_t tasks = 0) {
  if (const uint32_t configured = plan.configuration().groups) return configured;
  const uint64_t groups = std::max<uint64_t>((elements + kThreads - 1) / kThreads, tasks);
  return static_cast<uint32_t>(groups);
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
// slot position % kWindow of each KV head's window.
uint64_t windowBytes(DraftAttentionShape shape) {
  return uint64_t{shape.kvHeads} * kWindow * shape.headDimension * 2;
}

void describeContext(const DraftKvBuffers &context, DraftAttentionShape shape,
                     DraftKvAddresses &addresses) {
  if (context.keyPages.size() != SPLASH_DRAFT_PAGE_COUNT ||
      context.valuePages.size() != SPLASH_DRAFT_PAGE_COUNT)
    throw std::invalid_argument("incomplete draft page table");
  for (size_t i = 0; i < context.keyPages.size(); ++i) {
    requireBuffer(context.keyPages[i],
                  windowBytes(shape) / SPLASH_DRAFT_PAGE_COUNT);
    requireBuffer(context.valuePages[i],
                  windowBytes(shape) / SPLASH_DRAFT_PAGE_COUNT);
    addresses.keys[i] = context.keyPages[i].gpuAddress();
    if (context.valuePages[i].gpuAddress() !=
        addresses.keys[i] + windowBytes(shape) / SPLASH_DRAFT_PAGE_COUNT)
      throw std::invalid_argument("draft page K/V views must be adjacent");
  }
}
void retainContext(metal::CommandGraph &graph,
                   std::span<const DraftKvBuffers> context) {
  for (const auto &lane : context) {
    graph.addIndirectBuffers(lane.keyPages);
    graph.addIndirectBuffers(lane.valuePages);
  }
}

// What the context writers read for `rows` rows: the packed QKV rows, the
// key norm and the rows' RoPE tables (kernels/common/draft_context_kv.h).
void requireContextInputs(const metal::MetalBuffer &contextQkv,
                          const metal::MetalBuffer &keyNorm,
                          const metal::MetalBuffer &ropeCos,
                          const metal::MetalBuffer &ropeSin, uint64_t rows,
                          DraftAttentionShape shape) {
  requireBuffer(contextQkv, rows * shape.qkvSize * 2);
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

std::span<const DraftAttentionConfiguration>
DraftAttention::candidates(DraftAttentionShape shape) {
  static_cast<void>(kernelShape(shape));
  return kConfigurations;
}

DraftAttentionPlan
DraftAttention::plan(DraftAttentionShape shape, uint32_t lanes,
                     DraftAttentionConfiguration configuration) {
  requireLanes(lanes);
  const auto configurations = candidates(shape);
  if (std::find(configurations.begin(), configurations.end(), configuration) ==
      configurations.end())
    throw std::invalid_argument("unsupported draft attention configuration");
  return {shape, lanes, configuration};
}

void DraftAttention::captureTargetHidden(
    metal::CommandGraph &graph, metal::MetalBuffer source,
    metal::MetalBuffer captured, uint32_t rows, uint32_t captureSlot,
    uint32_t sourceStart, uint32_t destinationStart, uint32_t hiddenWidth,
    uint32_t targetWidth) {
  if (!rows || !hiddenWidth || !targetWidth ||
      targetWidth % hiddenWidth != 0 ||
      captureSlot >= targetWidth / hiddenWidth)
    throw std::invalid_argument("invalid target hidden capture");
  const CaptureParams params{rows, captureSlot, sourceStart, destinationStart,
                             hiddenWidth, targetWidth};
  graph.add("capture_target_hidden",
            {std::move(source), std::move(captured)}, params,
            {(hiddenWidth + 127) / 128, 1, 1});
}

void DraftAttention::gatherLastRows(metal::CommandGraph &graph,
                                    metal::MetalBuffer source,
                                    metal::MetalBuffer destination,
                                    uint32_t rows, uint32_t width) {
  if (!rows || !width)
    throw std::invalid_argument("invalid last-row gather");
  const LastHiddenRowsParams params{rows, width};
  graph.add("prefill_gather_last_hidden_rows8",
            {std::move(source), std::move(destination)}, params,
            {(width + 127) / 128, 1, 1});
}

void DraftAttention::addConvolution(metal::CommandGraph &graph,
                                    DraftConvolutionBuffers buffers,
                                    const DraftAttentionPlan &plan,
                                    DraftConvolutionStage stage) {
  if (stage != DraftConvolutionStage::Prepare &&
      stage != DraftConvolutionStage::Residual)
    throw std::invalid_argument("invalid draft convolution stage");
  const auto shape = plan.shape();
  const auto workspace = plan.workspace();
  const uint32_t groups = phaseGroups(plan, uint64_t{kRows} * shape.hiddenSize);
  const uint32_t lanes = plan.lanes();
  requireBuffer(buffers.input, workspace.convolutionBytes);
  requireBuffer(buffers.output, workspace.convolutionBytes);
  requireBuffer(buffers.residual, workspace.convolutionBytes);
  requireBuffer(buffers.dynamic, uint64_t{lanes} * kRows * shape.dynamicSize * 2);
  requireBuffer(buffers.weights, uint64_t{4} * shape.hiddenSize * 2);
  const KernelLayout kernel = kernelShape(shape);
  const DraftConvBatchParams params{
      groups, stage == DraftConvolutionStage::Residual ? 1U : 0U, lanes};
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
  // The prepare kernel runs an element loop over the key/value rows and a
  // task loop with one group per (row, head) normalization.
  const uint32_t groups = phaseGroups(
      plan, uint64_t{kRows} * shape.kvHeads * shape.headDimension,
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
  const DraftQkvBatchParams params{groups, lanes};
  graph.add("draft_attention_qkv",
            {std::move(buffers.qkv), std::move(buffers.groupedQueries),
             std::move(buffers.queryNorm), std::move(buffers.keyNorm),
             std::move(buffers.ropeCos), std::move(buffers.ropeSin),
             std::move(buffers.queryKeys), std::move(buffers.queryValues)},
            params, {groups, lanes, 1});
}

void DraftAttention::addDecode(
    metal::CommandGraph &graph, DraftDecodeAttentionBuffers buffers,
    std::span<const uint32_t> cacheLengths, uint32_t cacheStride,
    const DraftAttentionPlan &plan) {
  const auto shape = plan.shape();
  const auto workspace = plan.workspace();
  const uint32_t lanes = plan.lanes();
  if (cacheLengths.size() != kMaximumLanes || cacheStride != kWindow ||
      buffers.context.size() != kMaximumLanes)
    throw std::invalid_argument("invalid draft attention geometry");
  requireBuffer(buffers.groupedQueries, workspace.groupedQueriesBytes);
  requireBuffer(buffers.queryKeys, workspace.queryKeysBytes);
  requireBuffer(buffers.queryValues, workspace.queryValuesBytes);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (cacheLengths[lane] > SPLASH_MAXIMUM_CONTEXT_TOKENS)
      throw std::invalid_argument("draft attention cache length exceeds limit");
  }
  DraftAttentionBatchParams params{kSplits, lanes, {}, {}};
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane)
    describeContext(buffers.context[lane], shape, params.pages[lane]);
  std::copy(cacheLengths.begin(), cacheLengths.end(),
            std::begin(params.cache_length));
  std::vector<metal::MetalBuffer> bindings{buffers.groupedQueries};
  bindings.push_back(buffers.queryKeys);
  bindings.push_back(buffers.queryValues);
  graph.add("draft_attention_bf16_split", std::move(bindings), params,
            {shape.kvHeads, lanes, kSplits});
  retainContext(graph, buffers.context);
  graph.add("draft_attention_bf16_reduce", {buffers.groupedQueries}, params,
            {shape.kvHeads, lanes, 1});
}

void DraftAttention::addReorder(metal::CommandGraph &graph,
                                metal::MetalBuffer grouped,
                                metal::MetalBuffer packed,
                                const DraftAttentionPlan &plan) {
  const auto shape = plan.shape();
  const uint32_t groups = phaseGroups(
      plan, uint64_t{kRows} * shape.queryHeads * shape.headDimension);
  const uint32_t lanes = plan.lanes();
  requireBuffer(grouped, queryRowsBytes(plan));
  requireBuffer(packed, queryRowsBytes(plan));
  const DraftQkvBatchParams params{groups, lanes};
  graph.add("draft_attention_reorder",
            {std::move(grouped), std::move(packed)}, params,
            {groups, lanes, 1});
}

void DraftAttention::addContextPrefill(
    metal::CommandGraph &graph, metal::MetalBuffer contextQkv,
    metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
    metal::MetalBuffer ropeSin, const DraftKvBuffers &context, uint32_t tokens,
    uint32_t cacheStride, uint32_t startPosition, DraftAttentionShape shape) {
  static_cast<void>(kernelShape(shape));
  if (!tokens || cacheStride != kWindow)
    throw std::invalid_argument("invalid draft context prefill geometry");
  requireContextInputs(contextQkv, keyNorm, ropeCos, ropeSin, tokens, shape);
  DraftContextParams params{tokens, startPosition, {}};
  describeContext(context, shape, params.pages);
  graph.add("prefill_draft_context_kv",
            {std::move(contextQkv), std::move(keyNorm), std::move(ropeCos),
             std::move(ropeSin)},
            params, {uint64_t{tokens} * shape.kvHeads, 1, 1});
  retainContext(graph, std::span(&context, 1));
}

void DraftAttention::addContextCommit(
    metal::CommandGraph &graph, metal::MetalBuffer contextQkv,
    metal::MetalBuffer keyNorm, metal::MetalBuffer ropeCos,
    metal::MetalBuffer ropeSin, std::span<const DraftKvBuffers> context,
    metal::MetalBuffer retainedCounts, std::span<const uint32_t> startPositions,
    uint32_t cacheStride, DraftAttentionShape shape, uint32_t lanes) {
  requireLanes(lanes);
  static_cast<void>(kernelShape(shape));
  if (startPositions.size() != kMaximumLanes || cacheStride != kWindow ||
      context.size() != kMaximumLanes)
    throw std::invalid_argument("invalid draft context commit geometry");
  // Each lane commits up to its eight verify rows.
  requireContextInputs(contextQkv, keyNorm, ropeCos, ropeSin,
                       uint64_t{lanes} * SPLASH_TARGET_VERIFY_ROWS, shape);
  requireBuffer(retainedCounts, uint64_t{lanes} * sizeof(uint32_t));
  DraftContextBatchParams params{lanes, 0, {}, {}};
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane)
    describeContext(context[lane], shape, params.pages[lane]);
  std::copy(startPositions.begin(), startPositions.end(),
            std::begin(params.start_position));
  std::vector<metal::MetalBuffer> bindings{
      std::move(contextQkv), std::move(keyNorm), std::move(ropeCos),
      std::move(ropeSin)};
  bindings.push_back(std::move(retainedCounts));
  graph.add("draft_context_kv_commit", std::move(bindings), params,
            {uint64_t{lanes} * SPLASH_DRAFT_QUERY_ROWS * shape.kvHeads, 1,
             1});
  retainContext(graph, context);
}

} // namespace splash::ops
