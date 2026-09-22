#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/PagedAttention.h"
#include "ops/PagedKv.hpp"
#include "ops/Linear.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace splash::kv {

// Host aliases for the layouts shared with Metal; containers require these traits.
using Q8ChunkedPrefillParams = ::SplashChunkedPrefillParams;
using Q8VerifyAttentionParams = ::SplashQ8VerifyAttentionParams;
using Q8PrefillAttentionParams = ::SplashQ8PrefillAttentionParams;

static_assert(std::is_standard_layout_v<Q8ChunkedPrefillParams>);
static_assert(std::is_trivially_copyable_v<Q8ChunkedPrefillParams>);
static_assert(std::is_standard_layout_v<Q8VerifyAttentionParams>);
static_assert(std::is_trivially_copyable_v<Q8VerifyAttentionParams>);
static_assert(std::is_standard_layout_v<Q8PrefillAttentionParams>);
static_assert(std::is_trivially_copyable_v<Q8PrefillAttentionParams>);

inline constexpr uint32_t kQ8VerifyMaximumRows = SPLASH_TARGET_VERIFY_ROWS;
inline constexpr uint32_t kQ8VerifySplits = SPLASH_VERIFY_ATTENTION_SPLITS;
inline constexpr uint32_t kQ8VerifyMaximumSplits =
    SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS;
inline constexpr uint32_t kQ8VerifyPagesPerSplit =
    SPLASH_VERIFY_ATTENTION_PAGES_PER_SPLIT;
static_assert(kQ8VerifySplits >= 1 && kQ8VerifySplits <= kQ8VerifyMaximumSplits);
static_assert(kQ8VerifyPagesPerSplit >= 1);

// One lane's verify split count: never fewer than the configured base, one
// more split per kQ8VerifyPagesPerSplit visible pages, never more than the
// maximum the partial workspace is sized for. It depends only on the lane's
// own history, so batching never changes a lane's arithmetic.
[[nodiscard]] constexpr uint32_t
q8VerifyAttentionSplits(uint32_t baseSplits, uint32_t committedTokens,
                        uint32_t activeRows) noexcept {
  const uint64_t visible = uint64_t{committedTokens} + activeRows;
  const uint64_t pages = (visible + kPageTokens - 1) / kPageTokens;
  const uint64_t scaled =
      (pages + kQ8VerifyPagesPerSplit - 1) / kQ8VerifyPagesPerSplit;
  return static_cast<uint32_t>(std::min<uint64_t>(
      std::max<uint64_t>(baseSplits, scaled), kQ8VerifyMaximumSplits));
}

// Default parameters; the verify plan sets both split counts for each lane
// before dispatch.
[[nodiscard]] constexpr Q8VerifyAttentionParams
q8VerifyAttentionParams(uint32_t committedTokens, uint32_t activeRows,
                        uint32_t chunkStride, uint32_t pageTableEntries,
                        uint32_t physicalPageCount) noexcept {
  return {committedTokens, activeRows, chunkStride, pageTableEntries,
          physicalPageCount, kQ8VerifySplits, kQ8VerifySplits, 0};
}

[[nodiscard]] constexpr std::string_view q8VerifyAttentionValidationError(
    const Q8VerifyAttentionParams &params) noexcept {
  if (!params.active_rows || params.active_rows > kQ8VerifyMaximumRows)
    return "active_rows_out_of_range";
  if (params.chunk_stride < params.active_rows ||
      params.chunk_stride % kPageTokens)
    return "chunk_stride_invalid";
  const uint64_t visible =
      uint64_t{params.committed_tokens} + params.active_rows;
  if (visible > kMaximumPhysicalTokens)
    return "context_out_of_range";
  const uint64_t requiredPages = (visible + kPageTokens - 1) / kPageTokens;
  if (params.page_table_entries < requiredPages)
    return "page_table_too_short";
  if (!params.physical_page_count)
    return "physical_page_pool_empty";
  if (!params.split_count || params.split_count > kQ8VerifyMaximumSplits)
    return "split_count_invalid";
  if (params.slot_splits < params.split_count ||
      params.slot_splits > kQ8VerifyMaximumSplits)
    return "slot_splits_invalid";
  if (params.reserved2)
    return "reserved_fields_nonzero";
  return {};
}

inline constexpr uint32_t kChunkedPrefillMaximumRows =
    SPLASH_PREFILL_TOKEN_BUDGET;
inline constexpr uint32_t kQ8PrefillAttentionTileRows =
    SPLASH_PREFILL_ATTENTION_TILE_ROWS;

[[nodiscard]] constexpr uint32_t
prefillAttentionTiles(uint32_t rows) noexcept {
  return (rows + kQ8PrefillAttentionTileRows - 1) / kQ8PrefillAttentionTileRows;
}

[[nodiscard]] constexpr uint32_t
chunkedPrefillRequiredPages(const Q8ChunkedPrefillParams &params) noexcept {
  return (params.committed_tokens + params.chunk_tokens + kPageTokens - 1) /
         kPageTokens;
}

[[nodiscard]] constexpr std::string_view
chunkedPrefillValidationError(const Q8ChunkedPrefillParams &params) noexcept {
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
  if (!params.physical_page_count)
    return "physical_page_pool_empty";
  if (params.reserved0 || params.reserved1 || params.reserved2)
    return "reserved_fields_nonzero";
  return {};
}

[[nodiscard]] constexpr bool
chunkedPrefillValid(const Q8ChunkedPrefillParams &params) noexcept {
  return chunkedPrefillValidationError(params).empty();
}

// Call this once when preparing a request lane, not once per attention layer.
// Cache is the sole page-table owner and structurally guarantees
// unique leases; this ABI boundary only has to reject out-of-range indices.
[[nodiscard]] inline bool
chunkedPrefillPageTableInRange(const Q8ChunkedPrefillParams &params,
                               std::span<const uint32_t> pageTable) {
  const uint32_t pages = chunkedPrefillRequiredPages(params);
  if (!chunkedPrefillValid(params) || pageTable.size() < pages)
    return false;
  for (uint32_t logical = 0; logical < pages; ++logical) {
    if (pageTable[logical] >= params.physical_page_count)
      return false;
  }
  return true;
}

} // namespace splash::kv

namespace splash::ops {

struct AttentionWorkspace final {
  uint64_t partialsBytes = 0;
  uint64_t statisticsBytes = 0;
};

// Both kernels use one full-K QK multiply. Only the key-scale placement differs.
enum class AttentionScalePlacement : uint8_t { Softmax = 0, Cooperative = 1 };

// One preserves the shipped row-dependent split count; Two offers additional
// history parallelism with an explicitly larger scratch bound.
enum class PrefillSplitMultiplier : uint32_t { One = 1, Two = 2 };
struct PrefillAttentionConfig final {
  PrefillSplitMultiplier splitMultiplier = PrefillSplitMultiplier::One;
  AttentionScalePlacement scalePlacement = AttentionScalePlacement::Softmax;
  bool operator==(const PrefillAttentionConfig &) const = default;
};

enum class VerifySplitCount : uint32_t {
  One = 1,
  Eight = 8,
  Sixteen = 16,
  ThirtyTwo = 32
};
struct VerifyAttentionConfig final {
  VerifySplitCount splitCount = VerifySplitCount::ThirtyTwo;
  AttentionScalePlacement scalePlacement = AttentionScalePlacement::Softmax;
  bool operator==(const VerifyAttentionConfig &) const = default;
};

// Immutable factory-built plans are shared by allocation, measurement and
// encoding. Each prefill uses one split dispatch followed by one reduction.
// Callers cannot replace a dispatch or reduce its scratch bound.
struct PrefillAttentionPlan final {
  const kv::Format format;
  const PrefillAttentionConfig configuration;
  const uint32_t rows;
  const uint32_t historyTokens;
  const uint32_t splits;
  const AttentionWorkspace workspace;
  const std::string_view splitPipeline;
  const std::string_view reducePipeline;
  const metal::DispatchSize splitGroups;
  const metal::DispatchSize reduceGroups;

  // Policy provenance is irrelevant when its resolved execution is identical.
  [[nodiscard]] bool sameExecutionAs(const PrefillAttentionPlan &other) const noexcept;

private:
  friend class PagedAttention;
  PrefillAttentionPlan(PrefillAttentionConfig configuration, uint32_t rows,
                       uint32_t historyTokens, uint32_t splits,
                       AttentionWorkspace workspace,
                       std::string_view splitPipeline, std::string_view reducePipeline,
                       metal::DispatchSize splitGroups, metal::DispatchSize reduceGroups,
                       kv::Format format)
      : format(format), configuration(configuration), rows(rows), historyTokens(historyTokens),
        splits(splits), workspace(workspace),
        splitPipeline(splitPipeline), reducePipeline(reducePipeline),
        splitGroups(splitGroups), reduceGroups(reduceGroups) {}
};

struct VerifyAttentionPlan final {
  const kv::Format format;
  const VerifyAttentionConfig configuration;
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

  // Compare resolved execution, including each lane's history partition.
  [[nodiscard]] bool sameExecutionAs(const VerifyAttentionPlan &other) const noexcept;

private:
  friend class PagedAttention;
  const std::string_view storePipeline_;
  const metal::DispatchSize storeGroups_;
  const metal::DispatchSize storeThreads_;
  VerifyAttentionPlan(VerifyAttentionConfig configuration, uint32_t lanes,
                      std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> laneSplits,
                      uint32_t splits, AttentionWorkspace workspace,
                      std::string_view splitPipeline, std::string_view reducePipeline,
                      metal::DispatchSize splitGroups, metal::DispatchSize reduceGroups,
                      std::string_view storePipeline, metal::DispatchSize storeGroups,
                      metal::DispatchSize storeThreads, kv::Format format)
      : format(format), configuration(configuration), lanes(lanes), laneSplits(laneSplits),
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

// Q8 target attention over paged history. Prefill and verify both read the
// history one Page32 at a time; neither changes cache ownership or commit
// semantics.
class PagedAttention final {
public:
  [[nodiscard]] static std::span<const PrefillAttentionConfig>
  prefillCandidates() noexcept;
  [[nodiscard]] static std::span<const VerifyAttentionConfig>
  verifyCandidates() noexcept;
  [[nodiscard]] static PrefillAttentionPlan
  prefillPlan(uint32_t rows, uint32_t queryHeads, kv::Layout layout,
              uint32_t historyTokens, PrefillAttentionConfig configuration = {});
  // historyTokens holds each lane's committed tokens before its verify rows,
  // sized to the batch width or to the maximum width with inactive lanes zero.
  [[nodiscard]] static VerifyAttentionPlan
  verifyPlan(uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
             std::span<const uint32_t> historyTokens,
             VerifyAttentionConfig configuration = {});

  // The runtime owns allocation, not the selected kernel's workspace layout.
  // Prefill storage covers every sequence length up to maximumRows; sequences
  // in a packed command reuse it serially. Verify storage covers all lanes at
  // the maximum split count.
  [[nodiscard]] static AttentionWorkspace
  prefillWorkspace(uint32_t maximumRows, uint32_t queryHeads,
                   kv::Layout layout,
                   PrefillAttentionConfig configuration = {});
  [[nodiscard]] static AttentionWorkspace
  verifyWorkspace(uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
                  VerifyAttentionConfig configuration = {});

  static void
  addPrefillProjection(metal::CommandGraph &graph, metal::MetalBuffer packed,
                       metal::MetalBuffer queryNorm, metal::MetalBuffer keyNorm,
                       metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                       metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
                       metal::MetalBuffer chunkValues, uint32_t tokens,
                       uint32_t cacheStride, uint32_t rowStride,
                       uint32_t queryHeads, kv::Layout layout);
  static void addPrefillGate(metal::CommandGraph &graph,
                             metal::MetalBuffer packed,
                             metal::MetalBuffer attention,
                             metal::MetalBuffer hidden, uint32_t tokens,
                             uint32_t cacheStride, uint32_t rowStride,
                             uint32_t queryHeads, kv::Layout layout);
  static void
  addVerifyProjection(metal::CommandGraph &graph, metal::MetalBuffer packed,
                      metal::MetalBuffer queryNorm, metal::MetalBuffer keyNorm,
                      metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                      metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
                      metal::MetalBuffer chunkValues, uint32_t rowsPerLane,
                      uint32_t cacheStride, uint32_t rowStride,
                      uint32_t queryHeads, kv::Layout layout,
                      uint32_t lanes);
  static void addVerifyGate(metal::CommandGraph &graph,
                            metal::MetalBuffer packed,
                            metal::MetalBuffer attention,
                            metal::MetalBuffer hidden, uint32_t rowsPerLane,
                            uint32_t cacheStride, uint32_t rowStride,
                            uint32_t queryHeads, kv::Layout layout,
                            uint32_t lanes, LinearScratch scratch = {});

  [[nodiscard]] static kv::Q8ChunkedPrefillParams
  prefillParams(uint64_t logicalPosition, uint32_t chunkTokens,
                uint32_t chunkStride, std::span<const uint32_t> pageTable,
                uint32_t physicalPageCount);

  static void addPrefillStore(metal::CommandGraph &graph,
                              const kv::LayerStorage &layer,
                              metal::MetalBuffer chunkKeys,
                              metal::MetalBuffer chunkValues,
                              metal::MetalBuffer pageTable,
                              const kv::Q8ChunkedPrefillParams &params,
                              kv::Layout layout);
  // Queries and output are [KV head][row][query head in group][dimension] and
  // must not alias. Encode the store before attention; both stay in one
  // compute encoder. The plan owns both dispatch grids and their exact scratch.
  // prefillWorkspace() bounds every legal history for
  // the command's largest sequence and configuration.
  static void addPrefill(metal::CommandGraph &graph,
                         const kv::LayerStorage &layer,
                         metal::MetalBuffer queries, metal::MetalBuffer output,
                         metal::MetalBuffer partials,
                         metal::MetalBuffer statistics,
                         metal::MetalBuffer pageTable,
                         const kv::Q8ChunkedPrefillParams &chunk,
                         const PrefillAttentionPlan &plan);
  static void
  addVerify(metal::CommandGraph &graph, const kv::LayerStorage &layer,
            PagedVerifyBuffers buffers,
            std::span<const kv::Q8ChunkedPrefillParams> storeParams,
            std::span<const kv::Q8VerifyAttentionParams> attentionParams,
            const VerifyAttentionPlan &plan);
};

} // namespace splash::ops
