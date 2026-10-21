#pragma once

#include "TestKvTier.hpp"
#include "engine/Cache.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::test {

// Publishes the request's committed blocks and returns the deepest one, 0
// while no block is complete.
inline uint64_t publishBlocks(engine::Cache &cache, uint64_t requestId,
                              std::span<const uint32_t> tokens,
                              uint32_t committedTokens,
                              engine::PromptIdentity identity = {}) {
  cache.publishCommittedBlocks(requestId, tokens, committedTokens, identity);
  constexpr uint32_t pageTokens = engine::KvCache::pageTokens;
  return committedTokens < pageTokens
             ? 0
             : cache.blockAt(requestId, committedTokens / pageTokens * pageTokens);
}

// One round of what the engine does between commands to spill, with the
// tier's IO done at once: starts writing the Spill requests' published pages
// through (Cache::writeThrough), lands every write (TestKvTier::complete) and
// moves the eligible pages to the index form (Cache::spillWritten), as far
// as the pool admits. Returns the writes it started.
inline uint32_t spillRound(engine::Cache &cache, TestKvTier &tier) {
  const uint32_t written = cache.writeThrough();
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  static_cast<void>(cache.spillWritten());
  return written;
}

// The blocks the request's next command reads from their copies, by page,
// 0 for a Full page: its page sources, cut as the engine cuts them
// (Cache::cutPageTable).
inline std::vector<uint64_t> indexBlocks(engine::Cache &cache, uint64_t requestId) {
  const engine::PageTableView table = cache.cutPageTable(requestId);
  std::vector<uint64_t> blocks(table.pages.size());
  if (table.sources) {
    for (uint32_t page = 0; page < blocks.size(); ++page)
      blocks[page] = (*table.sources)[page].block;
  }
  return blocks;
}

} // namespace splash::test
