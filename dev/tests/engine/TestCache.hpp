#pragma once

#include "TestKvTier.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryLedger.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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

// One spill round as the engine runs it between commands, with the tier's IO done at once:
// write through, land every write, spillWritten. Returns the writes it started.
inline uint32_t spillRound(engine::Cache &cache, TestKvTier &tier) {
  const uint32_t written = cache.writeThrough();
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  static_cast<void>(cache.spillWritten(engine::SpilledExtents::KeepRunway));
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

// The query the ledger makes for a claim of `spec` that lacks `need`, with
// no lane in service (MemoryLedger::free).
inline engine::VictimQuery query(const engine::ClaimSpec &spec, const engine::Need &need, bool critical = false,
                                 const engine::RestoredChains *kept = nullptr) {
  return {.rights = engine::takeRights(spec, need,
                                       critical ? engine::MemoryPressure::Critical : engine::MemoryPressure::Normal,
                                       std::nullopt),
          .need = need,
          .kept = kept};
}

// What the ledger takes of the cache for a claim of `spec` that lacks
// `need` (MemoryLedger::free over its first source): the cache's first unit
// for it, taken; Short when it has none.
inline engine::FreeResult takeFirst(engine::Cache &cache, const engine::ClaimSpec &spec, const engine::Need &need) {
  const engine::VictimQuery query = test::query(spec, need);
  const std::optional<engine::Victim> unit = cache.peek(query, nullptr);
  return unit ? cache.take(query, *unit) : engine::FreeResult{};
}

// What the ledger frees for a claim of `spec` that lacks `need`, the cache
// its only source (MemoryLedger::free): nothing, Pending, while what the
// cache's transfers return covers the need, as the engine's claims wait.
inline engine::FreeResult freeFromCache(engine::Cache &cache, const engine::ClaimSpec &spec,
                                        const engine::Need &need) {
  engine::MemoryLedger ledger;
  ledger.addSource(cache);
  ledger.setReturning([&cache] { return cache.returning(); },
                      [&cache](std::span<const uint64_t> requestIds) { return cache.returningFor(requestIds); });
  const engine::ClaimId claim = ledger.open(spec);
  const engine::FreeResult result = ledger.free(claim, need, 0, engine::MemoryPressure::Normal);
  ledger.close(claim);
  return result;
}

// The claim of in-service work that reaches what is in use, as a request's
// KV growth is.
inline constexpr engine::ClaimSpec kInServiceGrowth{.cls = engine::ClaimClass::InService,
                                                    .reach = engine::ReclaimClass::InUse};

// One step of the cache's take for a claim (Cache::take): its units in
// order, past those a claim that cannot wait passes over, until one gives
// something. For a page or a device need that is one eviction or one
// memory reclaim; Short when none gives anything, Pending when a transfer
// in flight holds what it needs. The ledger's order across sources and its
// stop at a pending unit are not part of it (freeFromCache has them).
inline engine::FreeResult takeStep(engine::Cache &cache, const engine::ClaimSpec &spec, const engine::Need &need) {
  const engine::VictimQuery query = test::query(spec, need);
  engine::FreeResult result;
  for (std::optional<engine::Victim> unit = cache.peek(query, nullptr); unit; unit = cache.peek(query, &*unit)) {
    if (!query.rights.waitsForWrites && unit->later)
      continue;
    const engine::FreeResult taken = cache.take(query, *unit);
    if (taken.gave())
      return taken;
    if (taken.outcome == engine::FreeOutcome::Pending)
      result.outcome = engine::FreeOutcome::Pending;
  }
  return result;
}
// In-service work's step for a page (an eviction), for an extent's bytes
// (a memory reclaim), of what is in use too unless `reach` says otherwise,
// or for one state's buffers (the oldest checkpoint, else the oldest state,
// written when the tier takes it), of ordinary states unless `reach` says
// otherwise.
inline engine::FreeResult evictStep(engine::Cache &cache, engine::ReclaimClass reach = engine::ReclaimClass::InUse) {
  return takeStep(cache, {.cls = engine::ClaimClass::InService, .reach = reach}, {.pages = 1});
}
inline engine::FreeResult releaseStep(engine::Cache &cache, engine::ReclaimClass reach = engine::ReclaimClass::InUse) {
  return takeStep(cache, {.cls = engine::ClaimClass::InService, .reach = reach}, {.bytes = 1});
}
inline engine::FreeResult stateStep(engine::Cache &cache, engine::ReclaimClass reach = engine::ReclaimClass::Ordinary) {
  return takeStep(cache, {.cls = engine::ClaimClass::InService, .reach = reach}, {.stateBuffers = 1});
}

} // namespace splash::test
