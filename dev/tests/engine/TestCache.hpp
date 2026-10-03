#pragma once

#include "engine/Cache.hpp"

#include <cstdint>
#include <span>

namespace splash::test {

// Publishes the request's committed blocks and returns the deepest one, 0
// while no block is complete.
inline uint64_t publishBlocks(engine::Cache &cache, uint64_t requestId,
                              std::span<const uint32_t> tokens,
                              uint32_t committedTokens,
                              std::span<const ImageSpan> images = {}) {
  cache.publishCommittedBlocks(requestId, tokens, committedTokens, images);
  constexpr uint32_t pageTokens = engine::KvCache::pageTokens;
  return committedTokens < pageTokens
             ? 0
             : cache.blockAt(requestId, committedTokens / pageTokens * pageTokens);
}

} // namespace splash::test
