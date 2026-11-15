#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/PagedAttention.h"
#include "metal/abi/VerifyTree.h"
#include "ops/KernelCalibration.hpp"
#include "ops/PagedKv.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace splash::kv {

// Host alias for the layout shared with Metal; containers require these traits.
using AttentionParams = ::SplashAttentionParams;
static_assert(std::is_standard_layout_v<AttentionParams>);
static_assert(std::is_trivially_copyable_v<AttentionParams>);

// A sequence's rows of a command: chunk_tokens rows at positions from
// committed_tokens, staged chunk_stride rows per KV head (whole KV blocks),
// whose history and rows the first page_table_entries entries of its page
// table hold.
struct ChunkedPrefillParams final {
  uint32_t committed_tokens = 0;
  uint32_t chunk_tokens = 0;
  uint32_t chunk_stride = 0;
  uint32_t page_table_entries = 0;
};

inline constexpr uint32_t kChunkedPrefillMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
// Attention runs each sequence's rows in tiles of kAttentionTileRows, its
// last holding the rest (a decode lane's rows are one tile), each over at
// most kAttentionMaximumSplits history splits.
inline constexpr uint32_t kAttentionTileRows = SPLASH_ATTENTION_TILE_ROWS;
inline constexpr uint32_t kAttentionMaximumSplits = SPLASH_ATTENTION_MAXIMUM_SPLITS;

[[nodiscard]] constexpr uint32_t attentionTiles(uint32_t rows) noexcept {
  return (rows + kAttentionTileRows - 1) / kAttentionTileRows;
}

// The most pages a tile reads with a window of windowTokens keys before each
// row: its keys span windowTokens + kAttentionTileRows tokens, which may
// straddle a page boundary at either end.
[[nodiscard]] constexpr uint32_t windowAttentionPages(uint32_t windowTokens) noexcept {
  return (windowTokens + kAttentionTileRows - 1 + kPageTokens - 1) / kPageTokens + 1;
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

struct AttentionWorkspace final {
  uint64_t partialsBytes = 0;
  uint64_t statisticsBytes = 0;
};

// The tile splits all of a sequence's tiles take at most (PagedAttention.cpp
// attentionSplits): a workspace of this many, a prompt chunk's, holds any one
// sequence's partials.
inline constexpr uint32_t kAttentionSequenceSplits = 2 * kv::kAttentionMaximumSplits;

// How one attention layer attends, of the variants its geometry is compiled
// for (kCompiledAttention, PagedAttention.cpp): its queries and keys rotated
// by the geometry's RoPE, or unrotated (NoPE); and each row over its whole
// history, or over a window of the windowTokens keys before it (keys
// p - windowTokens to p, a sliding-window layer's). The default is every
// geometry's.
struct AttentionVariant final {
  bool rotated = true;
  uint32_t windowTokens = 0;
  bool operator==(const AttentionVariant &) const = default;
};

// The block lists a sparse plan's tiles attend through (metal/abi/
// PagedAttention.h, PagedAttention::addBlockLists): lists of `stride` entries,
// each list's count in `counts`; a sequence's tile t reads list `list` + t.
// Without a stride every tile attends to every block of its history. Only a
// sparse plan takes lists. With `locations`, an array of the lists' shape,
// each entry's block is read where its location names (SPLASH_ATTENTION_PAGED,
// or a page of locationPages and the block's place in it), as the hot block
// cache of a request whose K/V spilled gives them (model::KvBlockCache).
struct AttentionBlocks final {
  metal::MetalBuffer lists;
  metal::MetalBuffer counts;
  uint32_t stride = 0;
  uint32_t list = 0;
  metal::MetalBuffer locations{};
  metal::MetalBuffer locationPages{};
};

// Where a sequence's attention reads its history: the layer's pages, placed
// in the pool's extents by `kv`, through `pageTable`, and for a sparse plan
// its block lists.
struct AttentionHistory final {
  SplashKvLayer kv{};
  metal::MetalBuffer pageTable;
  AttentionBlocks blocks{};
};

// A layer's staging of its command's sequences (SplashAttentionSequence),
// each after the one before it, a KV head's rows its chunk_stride apart: the
// queries and attention rows [KV head][row][query head in group][dimension],
// keys [KV head][row][dimension] and values [KV head][dimension][row].
struct AttentionStaging final {
  metal::MetalBuffer queries;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer rows;
};

// Which of its sequences' attention rows a layer outputs: every row, or each
// sequence's last row alone, sequence s's as output row s.
enum class AttentionRows : uint8_t { Every, Last };

// The attention of a command's sequences (at most a batch's) for one layer
// variant, planned once a step: each sequence's history splits and the tiles
// of it that run, the plan's kernels, and the dispatches of a split and a
// reduce its sequences share. A dispatch holds sequences while their tiles'
// splits fit the workspace, so no sequence's partition depends on another's.
class AttentionPlan final {
public:
  struct Sequence final {
    uint32_t rows = 0;
    uint32_t historyTokens = 0;
    // Its tiles from this one on run: its first, or the one that holds its
    // last row (PagedAttention::lastTiles).
    uint32_t firstTile = 0;
    uint32_t splits = 0;
    [[nodiscard]] uint32_t tiles() const noexcept { return kv::attentionTiles(rows); }
  };
  struct Dispatch final {
    // The plan's sequences it runs, in row order.
    std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> sequence{};
    uint32_t sequences = 0;
    uint32_t tiles = 0;
    // The most splits of its sequences' tiles, and their splits all told.
    uint32_t splits = 0;
    uint32_t slots = 0;
  };

  const kv::Layout layout;
  const uint32_t queryHeads;
  // Whether the kernels attend through block lists (AttentionBlocks).
  const bool sparse;
  // Whether the reduce adds the layer's learned sinks, which addAttention
  // then takes: its geometry's (kCompiledAttention).
  const bool sinks;
  // The tile splits the workspace holds, which each dispatch's fit.
  const uint32_t workspaceSlots;

  [[nodiscard]] std::span<const Sequence> sequences() const noexcept { return {sequences_.data(), sequenceCount_}; }
  [[nodiscard]] std::span<const Dispatch> dispatches() const noexcept { return {dispatches_.data(), dispatchCount_}; }
  // The workspace its largest dispatch writes.
  [[nodiscard]] AttentionWorkspace workspace() const noexcept;
  // The kernels and grids of its dispatches: a sparse plan's split for lists
  // with locations, or without.
  [[nodiscard]] const std::string &splitPipeline(bool located = false) const noexcept {
    return located ? locatedSplitPipeline_ : splitPipeline_;
  }
  [[nodiscard]] const std::string &reducePipeline() const noexcept { return reducePipeline_; }
  [[nodiscard]] metal::DispatchSize splitGroups(const Dispatch &dispatch) const noexcept;
  [[nodiscard]] metal::DispatchSize reduceGroups(const Dispatch &dispatch) const noexcept;

private:
  friend class PagedAttention;
  AttentionPlan(kv::Layout layout, uint32_t queryHeads, bool sparse, bool sinks, uint32_t workspaceSlots)
      : layout(layout), queryHeads(queryHeads), sparse(sparse), sinks(sinks), workspaceSlots(workspaceSlots) {}
  std::array<Sequence, SPLASH_MAXIMUM_BATCH_WIDTH> sequences_{};
  uint32_t sequenceCount_ = 0;
  std::array<Dispatch, SPLASH_MAXIMUM_BATCH_WIDTH> dispatches_{};
  uint32_t dispatchCount_ = 0;
  std::string splitPipeline_;
  std::string locatedSplitPipeline_;
  std::string reducePipeline_;
};

// Rows whose block masks become block lists (PagedAttention::addBlockLists):
// `rows` rows from row `maskRow` of the masks, the first at position
// `firstPosition`, whose tiles take the lists from `list` on.
struct BlockListRows final {
  uint32_t maskRow = 0;
  uint32_t rows = 0;
  uint32_t firstPosition = 0;
  uint32_t list = 0;
};

// Target attention over paged INT8 or BF16 history, one op for every
// command: each layer stages its sequences' rows (addProjection), stores
// their K/V in their pages (addStore), attends them (addAttention) and gates
// the attention rows into the out-projection's input rows (addGate), each
// once for all its sequences, whether a decode step's lanes of eight rows or
// a prompt chunk's. A tile reads its history one Page32 at a time, or,
// sparse, the blocks of its list a few at a time; neither changes cache
// ownership or commit semantics. A layer's AttentionVariant selects its
// geometry's kernels: the projection's RoPE, and the window of the plans'
// kernels and of their split counts.
class PagedAttention final {
public:
  // The plan of a command's sequences (their chunks, in row order), whose
  // dispatches each fit `capacity`, a workspace. Each sequence's tiles take
  // the history splits its rows and the pages its tiles read of its history
  // decide (attentionSplits, PagedAttention.cpp), with the device's
  // `prefillPagesPerSplit` (AttentionCalibration). With `int8Activations`
  // on, the plan's queries score as INT8 against INT8 pages (the dense
  // kernels' q8q entries) where some tile's splits read at least the
  // device's `int8QueriesFromPages` pages each: a command's sequences share
  // the choice, as a decode step's lanes do.
  [[nodiscard]] static AttentionPlan plan(std::span<const kv::ChunkedPrefillParams> sequences, uint32_t queryHeads,
                                          kv::Layout layout, const AttentionCalibration &calibration,
                                          Int8Activations int8Activations, AttentionWorkspace capacity,
                                          AttentionVariant variant = {});
  // `plan`'s sequences from the tile that holds each one's last row: their
  // kernels and splits, so each tile computes what it does in `plan`.
  [[nodiscard]] static AttentionPlan lastTiles(const AttentionPlan &plan);

  // The workspace of `slots` tile splits of a geometry's tiles. The runtime
  // owns allocation: a prompt chunk's takes kAttentionSequenceSplits, a decode
  // lane's kAttentionMaximumSplits (its one tile at the most splits, or each
  // of its two, past 8 rows, at most 32).
  [[nodiscard]] static AttentionWorkspace workspace(uint32_t slots, uint32_t queryHeads, kv::Layout layout);
  // Whether the compiled geometry of `queryHeads` over a layout's KV heads
  // (kCompiledAttention) computes attention as a target's layout states it:
  // gated or not, rotating `rotaryPairs` pairs. Throws for a geometry with
  // no kernels.
  [[nodiscard]] static bool compiles(uint32_t queryHeads, kv::Layout layout, bool gated,
                                     uint32_t rotaryPairs);

  // A sequence's chunk: `chunkTokens` rows after `logicalPosition`, staged
  // `chunkStride` rows per KV head, its page table `pageTableEntries` long.
  [[nodiscard]] static kv::ChunkedPrefillParams chunkParams(uint64_t logicalPosition, uint32_t chunkTokens,
                                                            uint32_t chunkStride, uint32_t pageTableEntries);

  // The q/k norms and, for a rotated variant, the RoPE of the sequences'
  // packed projection rows (every query head's queries, and a gated
  // geometry's gates, then the K and V heads), staged for the store and
  // attention. An unrotated variant reads no RoPE tables; ropeCos and ropeSin
  // may be empty.
  static void addProjection(metal::CommandGraph &graph, metal::MetalBuffer packed, const NormWeights &queryNorm,
                            const NormWeights &keyNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                            const AttentionStaging &staging, std::span<const kv::ChunkedPrefillParams> sequences,
                            uint32_t queryHeads, kv::Layout layout, AttentionVariant variant = {});
  // Writes the sequences' staged K/V into the pages of `layer`, each through
  // its page table, before attention; both stay in one compute encoder.
  static void addStore(metal::CommandGraph &graph, SplashKvLayer layer, const AttentionStaging &staging,
                       std::span<const metal::MetalBuffer> pageTables,
                       std::span<const kv::ChunkedPrefillParams> sequences, kv::Layout layout);
  // The plan's dispatches over the staged queries into the attention rows,
  // which must not alias, each sequence reading its history where `histories`
  // say. The plan owns the grids; partials and statistics hold its workspace.
  // A dispatch whose sequences read their layer in two placements (`kv`)
  // runs as a dispatch per run of them that reads one.
  // `sinks` holds the layer's learned sinks, an F32 per query head, for a
  // plan whose reduce adds them (its `sinks`). With `tree`, a decode step's
  // tree rows (a SplashTreeRow per row of the command, metal/abi/
  // VerifyTree.h), each sequence is a lane's block of at most 32 rows whose
  // rows attend their paths (SplashAttentionParams::trees); without, every
  // row attends causally.
  static void addAttention(metal::CommandGraph &graph, const AttentionPlan &plan, const AttentionStaging &staging,
                           const metal::MetalBuffer &partials, const metal::MetalBuffer &statistics,
                           std::span<const kv::ChunkedPrefillParams> sequences,
                           std::span<const AttentionHistory> histories, metal::MetalBuffer sinks = {},
                           metal::MetalBuffer tree = {});
  // The attention rows `rows` names as the out-projection's input rows: each
  // multiplied by the sigmoid of its query head's gate in its packed row, or
  // for an ungated geometry gathered as they are, reading no packed row.
  // Also writes the out-projection's `input` table into `scratch` when it is
  // not Plain, and throws when `scratch` cannot hold it.
  static PreparedInput addGate(metal::CommandGraph &graph, metal::MetalBuffer packed,
                               const AttentionStaging &staging, metal::MetalBuffer hidden,
                               std::span<const kv::ChunkedPrefillParams> sequences, AttentionRows rows,
                               uint32_t queryHeads, kv::Layout layout, LinearScratch scratch = {},
                               LinearInput input = LinearInput::Plain);
  // One dispatch of the block lists of up to a batch of sequences of query
  // rows, whose block masks are rows of `mask`, `maskWords` words each
  // (metal/abi/PagedAttention.h) that cover each sequence's history: each
  // sequence's tiles of kAttentionTileRows rows, its tile t's list into list
  // blocks.list + its list + t. A list holds at most blocks.stride entries,
  // which the masks' rows must keep to.
  static void addBlockLists(metal::CommandGraph &graph, const metal::MetalBuffer &mask, uint32_t maskWords,
                            std::span<const BlockListRows> sequences, const AttentionBlocks &blocks);
  // Stages one layer's history for a prefill that reads it apart from the
  // pool (SplashKvStageParams, kernels/shared/kv_stage.metal): copies the K/V
  // of pages [firstPage, firstPage + pages) of poolTable, in the pool's layer
  // poolLayer, to the same pages of stagedTable, in the staging storage's one
  // layer stagedLayer; a pool entry that is SPLASH_KV_PAGE_ABSENT is left to
  // the reads that fill its staging page. Both storages have the layout's
  // heads, dimension and format.
  static void addStageLayer(metal::CommandGraph &graph, SplashKvLayer poolLayer, metal::MetalBuffer poolTable,
                            SplashKvLayer stagedLayer, metal::MetalBuffer stagedTable, uint32_t firstPage,
                            uint32_t pages, kv::Layout layout);
  // The commit of each lane's accepted path after a decode step stored its
  // block's rows at the slots of their rows (SplashKvPathCommitParams): in
  // every attention layer of `layout`, in extents of extentPages pages, the
  // path's row at depth k moves from committed[lane] plus its row to
  // committed[lane] plus k, through the lane's page table, which holds the
  // slots of laneRows rows from committed[lane], the most rows a lane's
  // block held. `paths` holds each lane's path (a row mask, metal/abi/
  // VerifyTree.h), as acceptance wrote it; a chain's moves nothing.
  static void addPathCommit(metal::CommandGraph &graph, kv::Layout layout, uint32_t extentPages,
                            std::span<const metal::MetalBuffer> pageTables, std::span<const uint32_t> committed,
                            uint32_t laneRows, const metal::MetalBuffer &paths);

private:
  // Gathers a plan's sequences into dispatches that fit its workspace.
  static void group(AttentionPlan &plan);
};

} // namespace splash::ops
