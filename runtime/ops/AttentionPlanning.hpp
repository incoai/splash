#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// The planning the attention operators share, which every split and reduce
// of the paged attention (ops::PagedAttention) and of relative attention
// (ops::RelativeAttention) follows: the pages a tile reads, its history
// splits, a command's staging, the dispatches a plan's sequences share, and
// the checks and bindings of their rows, page tables and path commits.
namespace splash::ops {

// A layout's KV format, as the kernels' names hold it.
inline std::string kernelFormat(kv::Layout layout) { return layout.format == kv::Format::BFloat16 ? "bf16" : "q8"; }

// The workspace of `rows` fused rows' partials and statistics.
inline AttentionWorkspace attentionWorkspace(uint64_t rows, uint32_t headDimension) {
  return {rows * headDimension * sizeof(float), rows * 2 * sizeof(float)};
}

// The fused rows of one tile split's partials, of every KV head and part.
inline uint64_t slotRows(uint32_t queryHeads) { return uint64_t{kv::kAttentionTileRows} * queryHeads; }

// The pages a tile that sees visibleTokens reads: with a window of
// windowTokens keys before each row, no more than its window's.
inline uint64_t readPages(uint64_t visibleTokens, uint32_t windowTokens) {
  const uint64_t seen = (visibleTokens + kv::kPageTokens - 1) / kv::kPageTokens;
  return windowTokens ? std::min<uint64_t>(seen, kv::windowAttentionPages(windowTokens)) : seen;
}

// The history splits of each of a sequence's `tiles` query tiles, which read
// `pages` pages. A sequence of one tile (a decode lane's of up to 8 rows)
// takes a split per kTilePagesPerSplit pages, between kSharedSplits and
// kAttentionMaximumSplits. A longer one, a lane's of 9 to 16 rows too, takes
// a whole split per pagesPerSplit pages (AttentionCalibration), at least
// kSharedSplits shared out over its tiles (all a short history takes) and at
// most kSharedSplits, nor over all its tiles more than
// kAttentionSequenceSplits: taking a tile's splits ran a sequence of two to
// four tiles up to 9% slower (on an M6 at 128K tokens). With a window, no
// more than the pages it reads (a split each). It depends on the sequence
// alone, so batching never changes a sequence's partition.
inline constexpr uint32_t kSharedSplits = 32;
inline constexpr uint32_t kTilePagesPerSplit = 16;
inline uint32_t attentionSplits(uint32_t tiles, uint64_t pages, uint32_t windowTokens, uint32_t pagesPerSplit) {
  const uint32_t fewest = std::clamp(kSharedSplits / tiles, 1U, kSharedSplits);
  const uint32_t most = std::clamp(kAttentionSequenceSplits / tiles, fewest, kSharedSplits);
  const uint32_t splits = static_cast<uint32_t>(
      tiles == 1 ? std::clamp<uint64_t>((pages + kTilePagesPerSplit - 1) / kTilePagesPerSplit, kSharedSplits,
                                        kv::kAttentionMaximumSplits)
                 : std::clamp<uint64_t>(pages / pagesPerSplit, fewest, most));
  return windowTokens ? std::min(splits, kv::windowAttentionPages(windowTokens)) : splits;
}

// The command's sequences' bounds and staging: each sequence's first staged
// row, the staged KV-head rows of the sequences before it; their bounds'
// sum and the largest bound, which the per-row kernels' grids span; and the
// last sequence's staged rows, which end the staging its kernels reach.
struct CommandRows final {
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> stageRow{};
  uint64_t bounds = 0;
  uint32_t mostRows = 0;
  uint64_t staged = 0;
  kv::ChunkedPrefillParams last{};
  uint32_t kvHeads = 0;

  // The staging's bytes up to the last sequence's last KV head's first
  // `reach` rows, `width` elements each (queries and attention rows).
  [[nodiscard]] uint64_t rowBytes(uint64_t width, uint32_t reach) const noexcept {
    return (staged - kvHeads * uint64_t{last.chunk_stride} + (kvHeads - 1) * uint64_t{last.chunk_stride} + reach) *
           width * sizeof(uint16_t);
  }
};
inline CommandRows commandRows(std::span<const kv::ChunkedPrefillParams> sequences, kv::Layout layout) {
  if (sequences.empty() || sequences.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid attention sequences");
  CommandRows result;
  for (size_t s = 0; s < sequences.size(); ++s) {
    const std::string_view error = kv::chunkedPrefillValidationError(sequences[s]);
    if (!error.empty()) throw std::invalid_argument(std::string(error));
    result.stageRow[s] = static_cast<uint32_t>(result.staged);
    result.bounds += sequences[s].chunk_tokens;
    result.mostRows = std::max(result.mostRows, sequences[s].chunk_tokens);
    result.staged += uint64_t{layout.kvHeads} * sequences[s].chunk_stride;
  }
  result.last = sequences.back();
  result.kvHeads = layout.kvHeads;
  if (result.staged > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("attention staging exceeds the kernel ABI");
  return result;
}

// Sequence s's rows from row `first` of its layout entry on.
inline SplashAttentionSequence sequenceRows(const kv::ChunkedPrefillParams &chunk, const CommandRows &rows, size_t s,
                                            uint32_t first) {
  return {.committed_tokens = chunk.committed_tokens + first,
          .row_bound = chunk.chunk_tokens - first,
          .layout_row = first,
          .stage_row = rows.stageRow[s] + first,
          .chunk_stride = chunk.chunk_stride,
          .page_table_entries = chunk.page_table_entries,
          .split_count = 0,
          .first_tile = 0,
          .first_slot = 0,
          .block_stride = 0,
          .block_list = 0};
}

// A command's rows (AttentionRows) of its sequences: at least a row, at most
// their bounds' sum, each sequence's layout entry, and a tree row a row.
inline void requireRows(const AttentionRows &rows, const CommandRows &command, size_t sequences) {
  if (!rows.rows || rows.rows > command.bounds)
    throw std::invalid_argument("attention rows exceed their sequences' bounds");
  requireBytes(rows.layout, sequences * sizeof(SplashLaneRows), "attention layout");
  if (rows.tree) requireBytes(rows.tree, uint64_t{rows.rows} * sizeof(SplashTreeRow), "attention tree rows");
}

// A chunk's page table holds the pages of its visible tokens.
inline void requirePageTable(const metal::MetalBuffer &table, const kv::ChunkedPrefillParams &chunk) {
  requireBytes(table, uint64_t{kv::chunkedPrefillRequiredPages(chunk)} * sizeof(SplashKvPage), "page table");
}

// Whether two sequences' histories lie in the same layer of their pools.
inline bool sameLayer(SplashKvLayer a, SplashKvLayer b) {
  return a.extent_pages == b.extent_pages && a.offset == b.offset;
}

// Gathers a plan's sequences (in row order) into `dispatches` while their
// tiles' splits fit `workspaceSlots` tile splits, so that no sequence's
// partition depends on another's; returns the dispatches. Each split of a
// tile takes `splitSlots` slots (a relative attention single pass's two).
inline uint32_t groupDispatches(std::span<const AttentionPlan::Sequence> sequences, uint32_t workspaceSlots,
                                std::array<AttentionPlan::Dispatch, SPLASH_MAXIMUM_BATCH_WIDTH> &dispatches,
                                uint32_t splitSlots = 1) {
  uint32_t count = 0;
  for (uint32_t s = 0; s < sequences.size(); ++s) {
    const AttentionPlan::Sequence &sequence = sequences[s];
    const uint32_t tiles = kv::attentionTiles(sequence.rows) - sequence.firstTile;
    const uint32_t slots = tiles * sequence.splits * splitSlots;
    if (slots > workspaceSlots)
      throw std::invalid_argument("the attention workspace holds no sequence's splits");
    // A sequence that does not fit the last dispatch starts the next.
    if (!count || dispatches[count - 1].slots + slots > workspaceSlots)
      dispatches[count++] = {};
    AttentionPlan::Dispatch *dispatch = &dispatches[count - 1];
    dispatch->sequence[dispatch->sequences++] = s;
    dispatch->tiles += tiles;
    dispatch->splits = std::max(dispatch->splits, sequence.splits);
    dispatch->slots += slots;
  }
  return count;
}

// The commit of each lane's accepted path after a decode step
// (PagedAttention::addPathCommit) by kernel(layout), the operator's commit
// kernel of the layout's pages, which it names once the commit is checked.
inline void addKvPathCommit(metal::CommandGraph &graph, std::string (*kernel)(kv::Layout), kv::Layout layout,
                            uint32_t extentPages, std::span<const metal::MetalBuffer> pageTables,
                            std::span<const uint32_t> committed, uint32_t laneRows, const metal::MetalBuffer &paths) {
  const uint32_t lanes = static_cast<uint32_t>(committed.size());
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || pageTables.size() != lanes || !extentPages || !laneRows ||
      laneRows > SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid KV path commit");
  SplashKvPathCommitParams params{.extent_pages = extentPages, .regions = layout.attentionLayers, .committed_tokens = {}};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint64_t pages = (uint64_t{committed[lane]} + laneRows + kv::kPageTokens - 1) / kv::kPageTokens;
    requireBytes(pageTables[lane], pages * sizeof(SplashKvPage), "page table");
    params.committed_tokens[lane] = committed[lane];
  }
  requireBytes(paths, uint64_t{lanes} * sizeof(uint32_t), "KV path commit paths");
  const auto tables = laneBindings(pageTables);
  std::vector<metal::MetalBuffer> bindings(tables.begin(), tables.end());
  bindings.push_back(paths);
  graph.add(kernel(layout), std::move(bindings), params, {uint64_t{layout.attentionLayers} * layout.kvHeads, lanes, 1},
            {layout.headDimension, 1, 1});
}

} // namespace splash::ops
