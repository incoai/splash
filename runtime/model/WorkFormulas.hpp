#pragma once

// The formula behind a model's CommandWork pairs (model/CommandCost.hpp): the query-key pairs an attention
// layer's rows compute at its reach. Model internals: the engine plans from the counts a model reports, never
// from these.

#include <algorithm>
#include <cstdint>

namespace splash::model {

// How far one attention layer's rows reach back: every key; the last windowTokens, the row's own included
// (QwenHybridLayout::slidingWindow, one more than kv::Layout::windowTokens; learned sinks are logits, not keys);
// or, block-sparse, the whole history up to denseHistoryTokens and past it selectedTokens plus the row's
// incomplete block.
struct AttentionReach final {
  uint32_t windowTokens = 0, denseHistoryTokens = 0, selectedTokens = 0, blockTokens = 0;
};
// The query-key pairs of `rows` rows from `position` at `reach`. The row at position p has p + 1 keys, its own
// included, and attends to all of them, to the last windowTokens of them, or, block-sparse, to all of them while
// its complete blocks of blockTokens hold at most denseHistoryTokens, and past that to selectedTokens of complete
// blocks and its keys after them (kernels/shared/block_indexer.metal); denseHistoryTokens and selectedTokens are
// whole blocks.
[[nodiscard]] constexpr uint64_t attentionPairs(const AttentionReach &reach, uint64_t position,
                                                uint32_t rows) noexcept {
  // The pairs of `count` rows from `first` that each attend to every key, and how many of the `rows` rows have
  // at most `keys` keys.
  const auto causal = [](uint64_t first, uint64_t count) { return count * first + count * (count + 1) / 2; };
  const auto within = [&](uint64_t keys) { return position < keys ? std::min<uint64_t>(rows, keys - position) : 0; };
  if (reach.windowTokens) {
    const uint64_t whole = within(reach.windowTokens);
    return causal(position, whole) + (rows - whole) * reach.windowTokens;
  }
  if (reach.denseHistoryTokens) {
    const uint64_t block = reach.blockTokens;
    // The keys after their last complete block, summed over rows of 1 to `keys` keys.
    const auto tails = [block](uint64_t keys) {
      const uint64_t tail = keys % block;
      return keys / block * (block * (block - 1) / 2) + tail * (tail + 1) / 2;
    };
    const uint64_t whole = within(uint64_t{reach.denseHistoryTokens} + block - 1);
    return causal(position, whole) + (rows - whole) * reach.selectedTokens + tails(position + rows) -
           tails(position + whole);
  }
  return causal(position, rows);
}
// The pairs of `rows` rows from `position` that each attend to every key up to their own.
[[nodiscard]] constexpr uint64_t denseAttentionPairs(uint64_t position, uint32_t rows) noexcept {
  return attentionPairs({}, position, rows);
}

} // namespace splash::model
