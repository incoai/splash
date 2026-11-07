#pragma once

// The memory checks the harness runs after every tick when the test's engine
// allocates through FakeMemory: I1, the allocators' counts, and I2, the
// ledger's relations.

#include "FakeMemory.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "engine/Cache.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::test {

// What allocates through FakeMemory, as a check reads it.
struct MemoryHolders final {
  const FakeMemory &memory;
  const TestKvStorage &storage;
  const Executor &model;
  const Weights *weights = nullptr;
  const engine::MemoryGovernor *governor = nullptr;
  // With them, the checks of the cache's transfers and of the engine's
  // requests' claims.
  const engine::Cache *cache = nullptr;
  const engine::Engine *engine = nullptr;
};

// I1: FakeMemory agrees with each allocator's own count and the governor
// charges at least what it holds. I2: the ledger's accounts, limit,
// transfers and claims agree with the engine.
inline void checkMemory(const MemoryHolders &holders) {
  holders.memory.check();
  const auto agree = [&](std::string_view holder, uint64_t counted) {
    if (holders.memory.held(holder) != counted)
      throw std::logic_error("fake memory holds " + std::to_string(holders.memory.held(holder)) + " bytes for " +
                             std::string(holder) + ", its allocator " + std::to_string(counted));
  };
  agree("kv", holders.storage.allocatedBytes());
  agree("states", holders.model.heldStateBytes);
  agree("spilled_reads", std::accumulate(holders.model.spilledReadBytes.begin(), holders.model.spilledReadBytes.end(),
                                         uint64_t{0}, [](uint64_t sum, const auto &entry) { return sum + entry.second; }));
  if (holders.model.experts) {
    const model::PoolState pool = holders.model.experts->state();
    agree(pool.name, pool.currentBytes);
  }
  if (holders.weights)
    agree("weights", uint64_t{holders.weights->held()} * Weights::kImageBytes);
  if (!holders.governor)
    return;
  const engine::MemoryGovernorSnapshot snapshot = holders.governor->snapshot();
  if (snapshot.chargedBytes < holders.memory.total())
    throw std::logic_error("the governor charges less than fake memory holds");
  // The accounts beside the fixed one (the first) stay within what is charged.
  uint64_t accounts = 0;
  for (size_t index = 1; index < snapshot.accounts.size(); ++index)
    accounts += snapshot.accounts[index].currentBytes;
  if (accounts > snapshot.chargedBytes)
    throw std::logic_error("the ledger's accounts hold " + std::to_string(accounts) + " bytes, " +
                           std::to_string(snapshot.chargedBytes) + " are charged");
  // What is charged stays under the limit.
  if (snapshot.chargedBytes > snapshot.limitBytes)
    throw std::logic_error("charged " + std::to_string(snapshot.chargedBytes) + " bytes pass the limit of " +
                           std::to_string(snapshot.limitBytes));
  // Transfers return nothing while none is in flight and no copy is owed.
  if (holders.cache) {
    const engine::Returning back = holders.cache->returning();
    if (!back.transfers && std::ranges::none_of(back.owedPages, [](uint32_t pages) { return pages != 0; }) &&
        snapshot.owed.bytes)
      throw std::logic_error("transfers return " + std::to_string(snapshot.owed.bytes) +
                             " bytes with none in flight and no copy owed");
  }
  // No claim of a request the engine finalized waits for memory, and none
  // of the engine's own once it serves no request.
  if (holders.engine) {
    const std::vector<engine::ActiveRequestSnapshot> served = holders.engine->snapshot().activeRequests;
    for (const auto &[_, claim] : holders.governor->ledger().claims()) {
      const uint64_t id = claim.spec.requestId;
      if (claim.waiting && (id ? std::ranges::none_of(served, [&](const auto &active) { return active.id == id; })
                               : served.empty()))
        throw std::logic_error(id ? "the claim of request " + std::to_string(id) +
                                        ", which the engine finalized, waits for memory"
                                  : std::string("a claim waits for memory with no request left"));
    }
  }
}

} // namespace splash::test
