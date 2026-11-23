#pragma once

#include "metal/abi/ExecutionGeometry.h"
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

// The planning the attention operators share, which every split and reduce
// of the paged attention (ops::PagedAttention) and of relative attention
// (ops::RelativeAttention) follows: the pages a tile reads, its history
// splits, a command's staging, and the dispatches a plan's sequences share.
namespace splash::ops {

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

} // namespace splash::ops
