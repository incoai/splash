#pragma once

#include "TestChecks.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Cache.hpp"
#include "model/SlotFile.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace checkpoint_history_tests {

using namespace splash;
using namespace splash::engine;
using test::require;

class MockState final : public CompositeState {
public:
  explicit MockState(uint64_t bytes, bool offloadable = false)
      : bytes_(bytes), offloadable_(offloadable) {}
  uint64_t bytes() const noexcept override { return bytes_; }
  bool canOffload(bool) const noexcept override { return offloadable_; }

private:
  uint64_t bytes_;
  bool offloadable_;
};

// Two unrelated synthetic histories. The state payload's bytes are mock
// accounting units, not a measurement of model memory. No model is loaded.
struct Fixture {
  test::TestKvStorage storage{17, 100, 1};
  KvPool pool{storage, 8};
  Cache cache;
  std::array<std::vector<uint32_t>, 2> prompts;
  std::array<std::array<uint64_t, 4>, 2> blocks{};

  explicit Fixture(CheckpointHistoryConfig history = {64, 2, 300})
      : cache(pool, nullptr, nullptr, history) {
    for (uint32_t h = 0; h != prompts.size(); ++h) {
      auto &prompt = prompts[h];
      for (uint32_t row = 0; row != 128; ++row)
        prompt.push_back(1000 * (h + 1) + row);
      prompt.push_back(9999);
      const uint64_t request = h + 1;
      cache.beginRequest(request);
      require(cache.ensureTokens(request, 128).granted(), "fixture KV admission failed");
      cache.publishCommittedBlocks(request, prompt, 128, {});
      for (uint32_t b = 0; b != blocks[h].size(); ++b)
        blocks[h][b] = cache.blockAt(request, (b + 1) * KvCache::pageTokens);
      cache.endRequest(request);
    }
  }

  StateCheckpoint publish(uint32_t history, uint32_t block, uint64_t bytes = 100,
                          bool checkpoint = true, bool offloadable = false) {
    const uint64_t id = blocks.at(history).at(block);
    cache.publishCompositeState(id, std::make_shared<MockState>(bytes, offloadable), checkpoint);
    return cache.checkpointState(id);
  }

  CacheLookup lookup(uint32_t history, uint32_t boundary) {
    return cache.lookup(std::span<const uint32_t>(prompts.at(history)).first(boundary + 1), {});
  }

  void history(uint32_t count, uint64_t bytes) const {
    const auto state = cache.snapshot().stateCache;
    require(state.historyEntries == count && state.historyBytes == bytes,
            "retained history count or mock-byte accounting differs");
  }
};

void testDisabledModeAndBoundarySelection() {
  for (const CheckpointHistoryConfig config :
       {CheckpointHistoryConfig{}, {0, 2, 300}, {64, 0, 300}, {64, 2, 0}}) {
    Fixture fixture(config);
    const auto point = fixture.publish(0, 1);
    require(!fixture.cache.retainCheckpointState(point), "disabled history retained a state");
    fixture.history(0, 0);
    require(fixture.cache.retireCheckpointState(point) && !fixture.lookup(0, 64).state,
            "disabled history changed hard retirement");
  }
  Fixture fixture;
  const auto offGrid = fixture.publish(0, 0);
  const auto onGrid = fixture.publish(0, 1);
  const auto ordinary = fixture.publish(0, 3, 100, false);
  const auto before = fixture.cache.snapshot().stateCache;
  require(!fixture.cache.retainCheckpointState(offGrid) &&
              fixture.cache.retainCheckpointState(onGrid) &&
              !fixture.cache.retainCheckpointState(ordinary) &&
              !fixture.cache.retainCheckpointState({fixture.blocks[1][1], 1}),
          "history admitted an off-grid, ordinary, or absent publication");
  fixture.history(1, 100);
  require(fixture.cache.snapshot().stateCache.publications == before.publications &&
              fixture.cache.snapshot().stateCache.entries == before.entries,
          "history manufactured a new capture or publication");
}

void testUnsupportedConfigurationIsRejected() {
  test::TestKvStorage storage{5, 100, 1};
  KvPool pool{storage, 4};
  test::TestKvTier tier;
  const auto budget = std::make_shared<model::DiskBudget>(1000);
  const auto rejects = [](auto &&construct) {
    test::rejects(construct, "checkpoint history",
                  "unsupported history configuration was accepted");
  };
  rejects([&] { Cache cache(pool, nullptr, nullptr, {33, 1, 100}); });
  rejects([&] { Cache cache(pool, &tier, nullptr, {64, 1, 100}); });
  rejects([&] { Cache cache(pool, nullptr, budget, {64, 1, 100}); });
  // Existing disk configurations still construct when the policy is disabled.
  Cache disabled(pool, &tier, budget, {});
}

void testCountBoundIsAggregateAcrossHistoriesAndRequests() {
  Fixture fixture({32, 2, 1000});
  const auto first = fixture.publish(0, 0);
  const auto second = fixture.publish(1, 0, 200);
  const auto third = fixture.publish(0, 1);
  const auto fourth = fixture.publish(1, 1);
  require(fixture.cache.retainCheckpointState(first) && fixture.cache.retainCheckpointState(second),
          "two histories could not share the aggregate allowance");
  for (uint64_t request = 10; request != 14; ++request) {
    fixture.cache.beginRequest(request);
    require(fixture.cache.retainCheckpointState(first) &&
                !fixture.cache.retainCheckpointState(third) &&
                !fixture.cache.retainCheckpointState(fourth),
            "history duplicated a mark or reset the count limit per request");
    fixture.cache.endRequest(request);
    fixture.history(2, 300);
  }
  require(fixture.cache.retireCheckpointState(first) && fixture.cache.retainCheckpointState(fourth),
          "hard retirement did not return the aggregate allowance");
  fixture.history(2, 300);
}

void testByteBoundUsesActualMockPayloadSizes() {
  for (const uint64_t limit : {uint64_t{299}, uint64_t{300}}) {
    Fixture fixture({64, 8, limit});
    const auto first = fixture.publish(0, 1, 100);
    const auto second = fixture.publish(1, 1, 200);
    require(fixture.cache.retainCheckpointState(first), "first payload did not fit byte allowance");
    require(fixture.cache.retainCheckpointState(second) == (limit == 300),
            "history did not enforce its exact aggregate byte allowance");
    fixture.history(limit == 300 ? 2 : 1, limit == 300 ? 300 : 100);
    require(fixture.cache.retireCheckpointState(first) &&
                fixture.cache.retainCheckpointState(second),
            "released byte allowance could not admit a larger state");
    fixture.history(1, 200);
  }
}

void testPublicationGenerationProtectsReplacement() {
  Fixture fixture;
  const auto stale = fixture.publish(0, 1);
  require(fixture.cache.retainCheckpointState(stale) && fixture.cache.retireCheckpointState(stale),
          "first retained publication could not be retired");
  fixture.history(0, 0);
  const auto replacement = fixture.publish(0, 1, 200);
  require(replacement.publication != stale.publication &&
              !fixture.cache.retainCheckpointState(stale) &&
              fixture.cache.retireCheckpointState(stale) &&
              fixture.cache.stateResident(replacement.kvBlock),
          "a stale publication retained or removed its replacement");
  fixture.history(0, 0);
  require(fixture.cache.retainCheckpointState(replacement), "replacement was not eligible");
  fixture.history(1, 200);
}

void testVeryLargeMockPayloadCannotOverflowTheAllowance() {
  Fixture fixture({64, 2, 300});
  const auto large = fixture.publish(0, 1, std::numeric_limits<uint64_t>::max());
  require(!fixture.cache.retainCheckpointState(large),
          "very large mock payload overflowed the retained-byte allowance");
  fixture.history(0, 0);
  require(fixture.cache.retireCheckpointState(large), "large mock payload could not retire");
  const auto normal = fixture.publish(1, 1, 100);
  require(fixture.cache.retainCheckpointState(normal), "rejected payload corrupted the allowance");
  fixture.history(1, 100);
}

void testSharedRestoreLeasesProtectWithoutPromotion() {
  Fixture fixture;
  const auto point = fixture.publish(0, 1);
  auto first = fixture.lookup(0, 64);
  require(first.resumeBoundary() == 64 && !fixture.cache.retainCheckpointState(point),
          "history admitted a newly marked state while a restore held it");
  first = {};
  require(fixture.cache.retainCheckpointState(point), "released checkpoint was not retained");
  first = fixture.lookup(0, 64);
  auto second = fixture.lookup(0, 64);
  require(!fixture.cache.retireCheckpointState(point) &&
              !fixture.cache.reclaimOneState(StateClass::Checkpoint, false),
          "retirement or optional reclaim bypassed shared restore leases");
  first = {};
  require(!fixture.cache.retireCheckpointState(point), "one release discarded a shared state");
  fixture.history(1, 100);
  second = {};
  require(fixture.cache.checkpointState(point.kvBlock) &&
              fixture.cache.reclaimOneState(StateClass::Checkpoint, false),
          "restore permanently pinned or promoted optional history");
  fixture.history(0, 0);
}

void testInvalidationKeepsAccountingUntilFinalLeaseRelease() {
  for (const bool discardPayload : {false, true}) {
    Fixture fixture({64, 1, 100});
    const auto invalid = fixture.publish(0, 1);
    const auto next = fixture.publish(1, 1);
    require(fixture.cache.retainCheckpointState(invalid), "invalidation fixture did not retain");
    auto first = fixture.lookup(0, 64);
    auto second = fixture.lookup(0, 64);
    if (discardPayload)
      fixture.cache.discardState(invalid.kvBlock, first.state->state().get());
    else
      fixture.cache.invalidateState(invalid.kvBlock);
    require(!fixture.lookup(0, 64).state && !fixture.cache.retainCheckpointState(invalid) &&
                !fixture.cache.retainCheckpointState(next),
            "invalid or still-leased payload escaped the aggregate bound");
    fixture.history(1, 100);
    first = {};
    fixture.history(1, 100);
    second = {};
    fixture.history(0, 0);
    require(fixture.cache.retainCheckpointState(next), "last lease failed to release allowance");
    fixture.history(1, 100);
  }
}

void testOrdinaryPromotionAndSharedUsesReleaseHistoryAllowance() {
  for (const bool throughUse : {false, true}) {
    Fixture fixture({64, 1, 100});
    const auto point = fixture.publish(0, 1);
    const auto next = fixture.publish(1, 1);
    require(fixture.cache.retainCheckpointState(point), "promotion fixture did not retain");
    StateUse first, second;
    if (throughUse) {
      first = fixture.cache.useState(point.kvBlock);
      second = fixture.cache.useState(point.kvBlock);
    } else {
      require(fixture.cache.reuseStoredState(point.kvBlock, false), "ordinary reuse failed");
    }
    fixture.history(0, 0);
    require(!fixture.cache.checkpointState(point.kvBlock) &&
                fixture.cache.retireCheckpointState(point) &&
                fixture.cache.stateResident(point.kvBlock) &&
                fixture.cache.retainCheckpointState(next),
            "promotion left a history mark or allowed stale retirement");
    fixture.history(1, 100);
    require(fixture.cache.reclaimOneState(StateClass::Checkpoint, false) &&
                fixture.cache.stateResident(point.kvBlock) &&
                !fixture.cache.reclaimOneState(StateClass::Checkpoint, false),
            "optional reclaim displaced a promoted or shared foreground state");
    first.reset();
    require(fixture.cache.stateResident(point.kvBlock), "one peer release removed a shared state");
    second.reset();
    fixture.history(0, 0);
  }
}

void testRetentionKeepsCheckpointEvictionClassAndRecency() {
  Fixture fixture;
  fixture.publish(0, 0, 100, false);
  const auto older = fixture.publish(0, 1);
  const auto newer = fixture.publish(0, 3);
  require(fixture.cache.retainCheckpointState(older) &&
              fixture.cache.retainCheckpointState(newer) &&
              fixture.cache.retainCheckpointState(older),
          "eviction fixture did not retain");
  require(fixture.cache.reclaimOneState(StateClass::Checkpoint, false) &&
              !fixture.cache.stateResident(older.kvBlock) &&
              fixture.cache.stateResident(newer.kvBlock) &&
              fixture.cache.stateResident(fixture.blocks[0][0]),
          "retention changed recency or displaced an older ordinary state");
  fixture.history(1, 100);
  require(fixture.cache.reclaimOneState(StateClass::Checkpoint, false).made,
          "retained state stopped being evictable");
  fixture.history(0, 0);
  const auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.checkpointEvictions == 2 && snapshot.checkpointRetirements == 0 &&
              snapshot.entries == 1 && snapshot.checkpointEntries == 0,
          "retained reclamation changed class or was counted as rolling retirement");
}

void testOffloadablePayloadIsNotRetained() {
  Fixture fixture;
  const auto point = fixture.publish(0, 1, 100, true, true);
  require(!fixture.cache.retainCheckpointState(point) && fixture.cache.retireCheckpointState(point),
          "RAM-only history retained an offloadable payload");
  fixture.history(0, 0);
}

void testHistoryYieldSkipsExactOwnerAndPreservesNativeFallback() {
  Fixture fixture;
  const auto owner = fixture.publish(0, 1);
  const auto spare = fixture.publish(1, 1);
  require(fixture.cache.retainCheckpointState(owner) && fixture.cache.retainCheckpointState(spare),
          "yield fixture did not retain");
  const auto protectedOwner = [owner](StateCheckpoint candidate) {
    return candidate.kvBlock == owner.kvBlock && candidate.publication == owner.publication;
  };
  require(fixture.cache.reclaimCheckpointHistory(protectedOwner) &&
              fixture.cache.stateResident(owner.kvBlock) &&
              !fixture.cache.stateResident(spare.kvBlock),
          "history yield removed the exact owner or failed to skip it");
  fixture.history(1, 100);
  require(!fixture.cache.reclaimCheckpointHistory(protectedOwner),
          "history yield ignored the last remaining owner");
  const auto before = fixture.cache.snapshot().stateCache;
  require(before.checkpointEvictions == 1 && before.checkpointRetirements == 0 &&
              fixture.cache.reclaimOneState(StateClass::Checkpoint, false).made,
          "history yield changed native counters or permanently pinned an owner");
  fixture.history(0, 0);
}

void testHistoryYieldCannotTakeRestoreLeasesOrInvalidPayloads() {
  for (const bool invalid : {false, true}) {
    Fixture fixture;
    const auto first = fixture.publish(0, 1);
    const auto second = fixture.publish(1, 1);
    require(fixture.cache.retainCheckpointState(first) &&
                fixture.cache.retainCheckpointState(second),
            "lease fixture did not retain");
    auto heldFirst = fixture.lookup(0, 64);
    auto heldSecond = fixture.lookup(1, 64);
    if (invalid)
      fixture.cache.invalidateState(first.kvBlock);
    const auto unprotected = [](StateCheckpoint) { return false; };
    require(!fixture.cache.reclaimCheckpointHistory(unprotected),
            "history yield took a leased or invalid payload");
    fixture.history(2, 200);
    heldSecond = {};
    require(fixture.cache.reclaimCheckpointHistory(unprotected) &&
                fixture.cache.stateResident(first.kvBlock),
            "history yield could not take only the released payload");
    fixture.history(1, 100);
    heldFirst = {};
    if (!invalid)
      require(fixture.cache.reclaimCheckpointHistory(unprotected),
              "released history remained permanently protected");
    fixture.history(0, 0);
  }
}

void testHistoryYieldUsesPublicationGeneration() {
  Fixture fixture;
  const auto stale = fixture.publish(0, 1);
  require(fixture.cache.retireCheckpointState(stale), "stale owner fixture did not retire");
  const auto replacement = fixture.publish(0, 1);
  require(replacement.publication != stale.publication &&
              fixture.cache.retainCheckpointState(replacement),
          "replacement was not retained");
  require(fixture.cache.reclaimCheckpointHistory([stale](StateCheckpoint candidate) {
    return candidate.kvBlock == stale.kvBlock && candidate.publication == stale.publication;
  }) && !fixture.cache.stateResident(replacement.kvBlock),
          "stale ownership shielded a replacement publication");
  fixture.history(0, 0);
}

void testNoHistoryYieldLeavesOrdinaryAndDisabledFallbacksIntact() {
  const auto unprotected = [](StateCheckpoint) { return false; };
  Fixture disabled(CheckpointHistoryConfig{});
  const auto unmarked = disabled.publish(0, 1);
  require(!disabled.cache.retainCheckpointState(unmarked) &&
              !disabled.cache.reclaimCheckpointHistory(unprotected) &&
              disabled.cache.stateResident(unmarked.kvBlock) &&
              disabled.cache.reclaimOneState(StateClass::Checkpoint, false).made,
          "disabled history yield took a state or broke native fallback");
  for (const bool sharedUse : {false, true}) {
    Fixture fixture;
    const auto promoted = fixture.publish(0, 1);
    require(fixture.cache.retainCheckpointState(promoted), "promotion fixture did not retain");
    StateUse owner;
    if (sharedUse)
      owner = fixture.cache.useState(promoted.kvBlock);
    else
      require(fixture.cache.reuseStoredState(promoted.kvBlock, false), "ordinary reuse failed");
    require(!fixture.cache.reclaimCheckpointHistory(unprotected) &&
                fixture.cache.stateResident(promoted.kvBlock),
            "history yield took an ordinary or in-use state");
    fixture.history(0, 0);
    owner.reset();
    require(fixture.cache.reclaimOneState(StateClass::Ordinary, false).made,
            "no eligible history prevented native fallback");
  }
}

void run() {
  testDisabledModeAndBoundarySelection();
  testUnsupportedConfigurationIsRejected();
  testCountBoundIsAggregateAcrossHistoriesAndRequests();
  testByteBoundUsesActualMockPayloadSizes();
  testPublicationGenerationProtectsReplacement();
  testVeryLargeMockPayloadCannotOverflowTheAllowance();
  testSharedRestoreLeasesProtectWithoutPromotion();
  testInvalidationKeepsAccountingUntilFinalLeaseRelease();
  testOrdinaryPromotionAndSharedUsesReleaseHistoryAllowance();
  testRetentionKeepsCheckpointEvictionClassAndRecency();
  testOffloadablePayloadIsNotRetained();
  testHistoryYieldSkipsExactOwnerAndPreservesNativeFallback();
  testHistoryYieldCannotTakeRestoreLeasesOrInvalidPayloads();
  testHistoryYieldUsesPublicationGeneration();
  testNoHistoryYieldLeavesOrdinaryAndDisabledFallbacksIntact();
}

} // namespace checkpoint_history_tests
