#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/PagedAttention.h"
#include "ops/PagedKv.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace splash::kv {

// Host aliases for the layouts shared with Metal; containers require these traits.
using ChunkedPrefillParams = ::SplashChunkedPrefillParams;
using VerifyAttentionParams = ::SplashVerifyAttentionParams;
using PrefillAttentionParams = ::SplashPrefillAttentionParams;

static_assert(std::is_standard_layout_v<ChunkedPrefillParams>);
static_assert(std::is_trivially_copyable_v<ChunkedPrefillParams>);
static_assert(std::is_standard_layout_v<VerifyAttentionParams>);
static_assert(std::is_trivially_copyable_v<VerifyAttentionParams>);
static_assert(std::is_standard_layout_v<PrefillAttentionParams>);
static_assert(std::is_trivially_copyable_v<PrefillAttentionParams>);

inline constexpr uint32_t kVerifyRows = SPLASH_TARGET_VERIFY_ROWS;
inline constexpr uint32_t kVerifyMaximumSplits =
    SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS;
// Rows per KV head (and per query group) of one lane's verify chunk
// staging: one KV block, which holds the lane's verify rows.
inline constexpr uint32_t kVerifyChunkStride = SPLASH_VERIFY_CHUNK_STRIDE;
static_assert(kVerifyChunkStride >= kVerifyRows &&
              kVerifyChunkStride % kPageTokens == 0);
// Verify attention runs one split per kVerifyPagesPerSplit visible Page32
// blocks, at least kVerifySplits and at most the maximum that sizes the
// partial workspace (verifyAttentionSplits).
inline constexpr uint32_t kVerifySplits = 32;
inline constexpr uint32_t kVerifyPagesPerSplit = 16;
static_assert(kVerifySplits <= kVerifyMaximumSplits);

// One lane's verify split count: one split per kVerifyPagesPerSplit
// pages its history and verify rows fill, never fewer than splitBase
// and never more than the maximum the partial workspace is sized for. It
// depends only on the lane's own history, so batching never changes a
// lane's arithmetic.
[[nodiscard]] constexpr uint32_t
verifyAttentionSplits(uint32_t committedTokens,
                      uint32_t splitBase = kVerifySplits) noexcept {
  const uint64_t visible = uint64_t{committedTokens} + kVerifyRows;
  const uint64_t pages = (visible + kPageTokens - 1) / kPageTokens;
  const uint64_t scaled =
      (pages + kVerifyPagesPerSplit - 1) / kVerifyPagesPerSplit;
  return static_cast<uint32_t>(std::min<uint64_t>(
      std::max<uint64_t>(splitBase, scaled), kVerifyMaximumSplits));
}

inline constexpr uint32_t kChunkedPrefillMaximumRows =
    SPLASH_PREFILL_TOKEN_BUDGET;
inline constexpr uint32_t kPrefillAttentionTileRows =
    SPLASH_PREFILL_ATTENTION_TILE_ROWS;

[[nodiscard]] constexpr uint32_t
prefillAttentionTiles(uint32_t rows) noexcept {
  return (rows + kPrefillAttentionTileRows - 1) / kPrefillAttentionTileRows;
}

[[nodiscard]] constexpr uint32_t
chunkedPrefillRequiredPages(const ChunkedPrefillParams &params) noexcept {
  return (params.committed_tokens + params.chunk_tokens + kPageTokens - 1) /
         kPageTokens;
}

[[nodiscard]] constexpr std::string_view
chunkedPrefillValidationError(const ChunkedPrefillParams &params) noexcept {
  if (!params.chunk_tokens || params.chunk_tokens > kChunkedPrefillMaximumRows)
    return "chunk_tokens_out_of_range";
  if (uint64_t{params.committed_tokens} + params.chunk_tokens >
      kMaximumPhysicalTokens)
    return "context_out_of_range";
  if (params.chunk_stride < params.chunk_tokens ||
      params.chunk_stride > kChunkedPrefillMaximumRows ||
      params.chunk_stride % kPageTokens)
    return "chunk_stride_invalid";
  if (params.page_table_entries < chunkedPrefillRequiredPages(params))
    return "page_table_too_short";
  return {};
}

} // namespace splash::kv

namespace splash::ops {

// The optional tuning override accepts exactly 1, 8, 16 or 32. Read it once
// when constructing ExecutionPlans; unset/empty preserves the shipped base.
[[nodiscard]] uint32_t parseVerifySplitCount(std::string_view text);
[[nodiscard]] uint32_t defaultVerifySplitCount();

struct AttentionWorkspace final {
  uint64_t partialsBytes = 0;
  uint64_t statisticsBytes = 0;
};

// Immutable factory-built plans are shared by allocation, measurement and
// encoding. Each prefill uses one split dispatch followed by one reduction.
// Callers cannot replace a dispatch or reduce its scratch bound.
struct PrefillAttentionPlan final {
  const kv::Format format;
  const uint32_t rows;
  const uint32_t splits;
  const AttentionWorkspace workspace;
  const std::string_view splitPipeline;
  const std::string_view reducePipeline;
  const metal::DispatchSize splitGroups;
  const metal::DispatchSize reduceGroups;

private:
  friend class PagedAttention;
  PrefillAttentionPlan(uint32_t rows, uint32_t splits, AttentionWorkspace workspace,
                       std::string_view splitPipeline, std::string_view reducePipeline,
                       metal::DispatchSize splitGroups, metal::DispatchSize reduceGroups,
                       kv::Format format)
      : format(format), rows(rows),
        splits(splits), workspace(workspace),
        splitPipeline(splitPipeline), reducePipeline(reducePipeline),
        splitGroups(splitGroups), reduceGroups(reduceGroups) {}
};

struct VerifyAttentionPlan final {
  const kv::Format format;
  const uint32_t lanes;
  // Each lane's history-scaled split count; splits is their maximum, the
  // split grid and the slot stride of every lane's partials. The workspace
  // covers the maximum split count for every lane regardless of history.
  const std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> laneSplits;
  const uint32_t splits;
  const AttentionWorkspace workspace;
  const std::string_view splitPipeline;
  const std::string_view reducePipeline;
  const metal::DispatchSize splitGroups;
  const metal::DispatchSize reduceGroups;

private:
  friend class PagedAttention;
  const std::string_view storePipeline_;
  const metal::DispatchSize storeGroups_;
  const metal::DispatchSize storeThreads_;
  VerifyAttentionPlan(uint32_t lanes,
                      std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> laneSplits,
                      uint32_t splits, AttentionWorkspace workspace,
                      std::string_view splitPipeline, std::string_view reducePipeline,
                      metal::DispatchSize splitGroups, metal::DispatchSize reduceGroups,
                      std::string_view storePipeline, metal::DispatchSize storeGroups,
                      metal::DispatchSize storeThreads, kv::Format format)
      : format(format), lanes(lanes), laneSplits(laneSplits),
        splits(splits), workspace(workspace),
        splitPipeline(splitPipeline), reducePipeline(reducePipeline),
        splitGroups(splitGroups), reduceGroups(reduceGroups),
        storePipeline_(storePipeline), storeGroups_(storeGroups), storeThreads_(storeThreads) {}
};

struct PagedVerifyBuffers final {
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  metal::MetalBuffer queries;
  metal::MetalBuffer partials;
  metal::MetalBuffer statistics;
  metal::MetalBuffer output;
  std::span<const metal::MetalBuffer> pageTables;
};

// Target attention over paged INT8 or BF16 history. Prefill and verify both
// read the history one Page32 at a time; neither changes cache ownership or
// commit semantics.
class PagedAttention final {
public:
  // The kernels read a sequence's history from its chunk parameters, so a
  // plan depends on the rows only.
  [[nodiscard]] static PrefillAttentionPlan
  prefillPlan(uint32_t rows, uint32_t queryHeads, kv::Layout layout);
  // historyTokens holds each lane's committed tokens before its verify rows,
  // one entry per lane. splitBase is the tuning minimum (1, 8, 16 or 32);
  // history scaling and the maximum workspace bound remain unchanged.
  [[nodiscard]] static VerifyAttentionPlan
  verifyPlan(uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
             std::span<const uint32_t> historyTokens,
             uint32_t splitBase = kv::kVerifySplits);

  // The runtime owns allocation, not the selected kernel's workspace layout.
  // Prefill storage covers every sequence length up to maximumRows; sequences
  // in a packed command reuse it serially. Verify storage covers all lanes at
  // the maximum split count.
  [[nodiscard]] static AttentionWorkspace
  prefillWorkspace(uint32_t maximumRows, uint32_t queryHeads,
                   kv::Layout layout);
  [[nodiscard]] static AttentionWorkspace
  verifyWorkspace(uint32_t lanes, uint32_t queryHeads, kv::Layout layout);

  static void
  addPrefillProjection(metal::CommandGraph &graph, metal::MetalBuffer packed,
                       const NormWeights &queryNorm, const NormWeights &keyNorm,
                       metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                       metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
                       metal::MetalBuffer chunkValues, uint32_t tokens,
                       uint32_t stride, uint32_t queryHeads, kv::Layout layout);
  static void addPrefillGate(metal::CommandGraph &graph,
                             metal::MetalBuffer packed,
                             metal::MetalBuffer attention,
                             metal::MetalBuffer hidden, uint32_t tokens,
                             uint32_t stride, uint32_t queryHeads,
                             kv::Layout layout);
  static void
  addVerifyProjection(metal::CommandGraph &graph, metal::MetalBuffer packed,
                      const NormWeights &queryNorm, const NormWeights &keyNorm,
                      metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                      metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
                      metal::MetalBuffer chunkValues, uint32_t queryHeads,
                      kv::Layout layout, uint32_t lanes);
  // Also writes the out-projection's `input` table into `scratch` when it is
  // not Plain, and throws when `scratch` cannot hold it.
  static PreparedInput addVerifyGate(metal::CommandGraph &graph,
                                     metal::MetalBuffer packed,
                                     metal::MetalBuffer attention,
                                     metal::MetalBuffer hidden,
                                     uint32_t queryHeads, kv::Layout layout,
                                     uint32_t lanes, LinearScratch scratch,
                                     LinearInput input);

  // A lane's parameters; each layer's encoding adds the layer's place in
  // the extents.
  [[nodiscard]] static kv::ChunkedPrefillParams
  prefillParams(uint64_t logicalPosition, uint32_t chunkTokens,
                uint32_t chunkStride, uint32_t pageTableEntries);
  // A verify lane's chunk: its verify rows in kVerifyChunkStride rows of
  // staging, the only parameters addVerify attends.
  [[nodiscard]] static kv::ChunkedPrefillParams
  verifyParams(uint64_t logicalPosition, uint32_t pageTableEntries);

  static void addPrefillStore(metal::CommandGraph &graph, SplashKvLayer layer,
                              metal::MetalBuffer chunkKeys,
                              metal::MetalBuffer chunkValues,
                              metal::MetalBuffer pageTable,
                              const kv::ChunkedPrefillParams &params,
                              kv::Layout layout);
  // Queries and output are [KV head][row][query head in group][dimension] and
  // must not alias. Encode the store before attention; both stay in one
  // compute encoder. The plan owns both dispatch grids and their exact scratch.
  // prefillWorkspace() bounds every legal history for the command's largest
  // sequence.
  static void addPrefill(metal::CommandGraph &graph, SplashKvLayer layer,
                         metal::MetalBuffer queries, metal::MetalBuffer output,
                         metal::MetalBuffer partials,
                         metal::MetalBuffer statistics,
                         metal::MetalBuffer pageTable,
                         const kv::ChunkedPrefillParams &chunk,
                         const PrefillAttentionPlan &plan);
  // Stores each lane's chunk (verifyParams, one per plan lane) and attends
  // its verify rows with the plan's split counts.
  static void addVerify(metal::CommandGraph &graph, SplashKvLayer layer,
                        PagedVerifyBuffers buffers,
                        std::span<const kv::ChunkedPrefillParams> chunks,
                        const VerifyAttentionPlan &plan);
};

} // namespace splash::ops
