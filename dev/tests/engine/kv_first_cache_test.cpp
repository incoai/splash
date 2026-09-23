#include "TestKvPool.hpp"
#include "engine/Cache.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

class TestState final : public CompositeState {
public:
  explicit TestState(uint64_t byteCount) : byteCount_(byteCount) {}
  [[nodiscard]] uint64_t bytes() const noexcept override { return byteCount_; }

private:
  uint64_t byteCount_;
};

struct SharedBudget {
  static constexpr uint64_t capacity = 600;
  uint64_t used = 0;

  bool acquire(uint64_t bytes) {
    if (bytes > capacity - used)
      return false;
    used += bytes;
    return true;
  }

  void release(uint64_t bytes) { used -= bytes; }
};

class BudgetBacking final : public KvBacking {
public:
  explicit BudgetBacking(SharedBudget &budget) : budget_(budget) {}
  ~BudgetBacking() override {
    for (bool resident : resident_) {
      if (resident)
        budget_.release(bytesPerPage());
    }
  }
  uint32_t pageCount() const noexcept override { return resident_.size(); }
  uint64_t bytesPerPage() const noexcept override { return 100; }
  bool isResident(uint32_t page) const override { return resident_.at(page); }
  splash::metal::AllocationResult ensureResident(uint32_t page) override {
    if (isResident(page))
      return true;
    if (!budget_.acquire(bytesPerPage()))
      return false;
    resident_[page] = true;
    return true;
  }
  bool releaseBackingForPage(uint32_t page) override {
    if (!isResident(page))
      return false;
    resident_[page] = false;
    budget_.release(bytesPerPage());
    return true;
  }
  uint32_t extentFirstPage(uint32_t page) const override {
    static_cast<void>(resident_.at(page));
    return page;
  }
  uint32_t extentPageCount(uint32_t page) const override {
    static_cast<void>(resident_.at(page));
    return 1;
  }

private:
  SharedBudget &budget_;
  std::array<bool, 4> resident_{};
};

class BudgetState final : public CompositeState {
public:
  explicit BudgetState(SharedBudget &budget) : budget_(budget) {
    require(budget_.acquire(bytes()), "fixture state exceeded shared budget");
  }
  ~BudgetState() override { budget_.release(bytes()); }
  uint64_t bytes() const noexcept override { return 200; }

private:
  SharedBudget &budget_;
};

CacheNamespace cacheNamespace() {
  CacheNamespace result;
  result.digest.fill(0x5a);
  return result;
}

struct CacheFixture {
  test::TestKvBacking backing{4, 100};
  KvPool pool{backing};
  engine::Cache cache{pool, cacheNamespace()};
  std::vector<uint32_t> prompt;
  std::vector<uint64_t> blocks;

  CacheFixture() {
    for (uint32_t block = 0; block < 4; ++block) {
      for (uint32_t row = 0; row < KvCache::pageTokens; ++row)
        prompt.push_back(1000 + block * 100 + row);
    }
    prompt.push_back(9999);
    cache.beginRequest(1);
    require(cache.ensureTokens(1, 128).granted(),
            "fixture KV pages were not acquired");
    static_cast<void>(cache.publishCommittedBlocks(1, prompt, 128));
    for (uint32_t boundary = 32; boundary <= 128; boundary += 32)
      blocks.push_back(cache.blockAt(1, boundary));
    cache.endRequest(1);
  }

  void publish(uint32_t block, uint64_t bytes = 100) {
    cache.publishCompositeState(blocks.at(block),
                                std::make_shared<TestState>(bytes));
  }

  engine::CacheLookup lookup(uint32_t tokens) {
    return cache.lookup(std::span<const uint32_t>(prompt).first(tokens));
  }
};

void testSchedulingProbeDoesNotChangeCachePolicy() {
  CacheFixture fixture;
  require(fixture.cache.cachedTokens(fixture.prompt) == 0,
          "KV without recurrent state was counted as reusable work");
  fixture.publish(0);
  fixture.publish(3);
  const auto prefix = std::span<const uint32_t>(fixture.prompt).first(33);
  require(fixture.cache.cachedTokens(prefix) == 32 &&
              fixture.cache.cachedTokens(fixture.prompt) == 128 &&
              fixture.cache.snapshot().stateCache.pinned == 0 &&
              fixture.cache.snapshot().lookup.lookups == 0,
          "scheduling probe pinned backing or counted a cache hit");
  require(fixture.cache.reclaimOneState() &&
              fixture.cache.cachedTokens(prefix) == 0 &&
              fixture.cache.cachedTokens(fixture.prompt) == 128,
          "scheduling probe refreshed the oldest state's eviction order");
}

void testValidAdmissionProbePreservesLookupAndAccounting() {
  CacheFixture fixture;
  fixture.publish(0);
  fixture.publish(2);
  const CacheProbe probe = fixture.cache.probe(fixture.prompt);
  const auto before = fixture.cache.snapshot();
  require(probe.cachedTokens() == 96 && before.stateCache.pinned == 0 &&
              before.lookup.lookups == 0,
          "admission probe changed cache ownership or accounting");
  auto lookup = fixture.cache.lookup(fixture.prompt, {}, &probe);
  require(lookup.kvBoundary == 128 && lookup.resumeBoundary() == 96 &&
              lookup.junctionBoundary() == 128,
          "valid admission probe lost the deepest state or KV tail");
  fixture.cache.recordLookup(lookup);
  const auto after = fixture.cache.snapshot();
  require(after.lookup.lookups == 1 && after.lookup.kvHitTokens == 128 &&
              after.lookup.stateHitTokens == 96 &&
              after.lookup.lazyJunctions == 1 &&
              after.stateCache.pinned == 1,
          "admission probe changed lookup accounting or lease ownership");
}

void testProbeFallsBackWhenPromptChanges() {
  CacheFixture fixture;
  fixture.publish(0);
  const CacheProbe probe = fixture.cache.probe(fixture.prompt);
  fixture.prompt.front() += 1;
  const auto lookup = fixture.cache.lookup(fixture.prompt, {}, &probe);
  require(lookup.kvBoundary == 0 && lookup.resumeBoundary() == 0,
          "a same-buffer prompt edit reused a different prompt's cache");
}

void testProbeRechecksFirstMissAndPromptLength() {
  CacheFixture fixture;
  fixture.publish(0);
  fixture.publish(3);
  auto changed = fixture.prompt;
  changed[32] += 1;
  const CacheProbe partial = fixture.cache.probe(changed);
  require(partial.cachedTokens() == 32, "partial probe missed its first page");
  // Restoring the first missed page must reveal the already-cached suffix,
  // even though the matched pages and KV generation did not change.
  auto restored = fixture.cache.lookup(fixture.prompt, {}, &partial);
  require(restored.kvBoundary == 128 && restored.resumeBoundary() == 128,
          "probe hid a prefix after an edit to its first missed page");
  restored.state.reset();

  const CacheProbe full = fixture.cache.probe(fixture.prompt);
  for (size_t size : {size_t{0}, size_t{1}, size_t{32}, size_t{33}}) {
    const auto shorter = std::span<const uint32_t>(fixture.prompt).first(size);
    const auto lookup = fixture.cache.lookup(shorter, {}, &full);
    const uint32_t expected = size == 33 ? 32 : 0;
    require(lookup.kvBoundary == expected && lookup.resumeBoundary() == expected,
            "probe reused pages past a shortened prompt's replay boundary");
  }
  const auto shortPrompt = std::span<const uint32_t>(fixture.prompt).first(33);
  const CacheProbe shortProbe = fixture.cache.probe(shortPrompt);
  const auto longer = fixture.cache.lookup(fixture.prompt, {}, &shortProbe);
  require(longer.kvBoundary == 128 && longer.resumeBoundary() == 128,
          "probe hid cached pages after the prompt grew");
}

void testProbeRechecksStateChanges() {
  CacheFixture fixture;
  fixture.publish(3);
  fixture.publish(0);
  const CacheProbe probe = fixture.cache.probe(fixture.prompt);
  require(probe.cachedTokens() == 128 && fixture.cache.reclaimOneState(),
          "state eviction fixture did not remove the deepest state");
  auto lookup = fixture.cache.lookup(fixture.prompt, {}, &probe);
  require(lookup.kvBoundary == 128 && lookup.resumeBoundary() == 32,
          "admission probe reused an evicted state");
  lookup.state.reset();

  CacheFixture published;
  const CacheProbe cold = published.cache.probe(published.prompt);
  published.publish(2);
  auto newlyPublished = published.cache.lookup(published.prompt, {}, &cold);
  require(cold.cachedTokens() == 0 && newlyPublished.resumeBoundary() == 96,
          "admission probe missed a state published after preview");
}

void testProbeFallsBackWhenKvChanges() {
  CacheFixture fixture;
  fixture.publish(0);
  const CacheProbe probe = fixture.cache.probe(fixture.prompt);
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReuseBacking).madeProgress &&
              fixture.cache.snapshot().kvCache.blocks == 3,
          "KV eviction fixture did not evict the cached tail");
  auto lookup = fixture.cache.lookup(fixture.prompt, {}, &probe);
  require(lookup.kvBoundary == 96 && lookup.resumeBoundary() == 32,
          "admission probe reused an evicted KV block");

  test::TestKvBacking backing{1, 100};
  KvPool pool{backing};
  engine::Cache cache{pool, cacheNamespace()};
  std::vector<uint32_t> prompt(33, 77);
  const CacheProbe cold = cache.probe(prompt);
  require(cold.cachedTokens() == 0, "cold admission probe found cached work");
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 32).granted(), "new KV page was not acquired");
  const uint64_t block = cache.publishCommittedBlocks(1, prompt, 32);
  cache.publishCompositeState(block, std::make_shared<TestState>(100));
  cache.endRequest(1);
  auto newlyCached = cache.lookup(prompt, {}, &cold);
  require(newlyCached.kvBoundary == 32 &&
              newlyCached.resumeBoundary() == 32,
          "cold probe hid a prefix published after preview");
}

void testProbeBindsImageIdentity() {
  test::TestKvBacking backing{1, 100};
  KvPool pool{backing};
  engine::Cache cache{pool, cacheNamespace()};
  std::vector<uint32_t> prompt(33, 77);
  ImageSpan image{0, 32, 1, 1, 101, 202};
  const std::span<const ImageSpan> images(&image, 1);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 32).granted(), "image KV page was not acquired");
  const uint64_t block = cache.publishCommittedBlocks(1, prompt, 32, images);
  cache.publishCompositeState(block, std::make_shared<TestState>(100));
  cache.endRequest(1);
  const CacheProbe probe = cache.probe(prompt, images);
  auto valid = cache.lookup(prompt, images, &probe);
  require(valid.kvBoundary == 32 && valid.resumeBoundary() == 32,
          "matching image probe lost its cached prefix");
  valid.state.reset();
  image.digestLo += 1;
  auto changed = cache.lookup(prompt, images, &probe);
  require(changed.kvBoundary == 0 && changed.resumeBoundary() == 0,
          "image digest change reused a different image's cache");
}

void testProbeCannotCrossCaches() {
  test::TestKvBacking firstBacking{1, 100};
  test::TestKvBacking secondBacking{1, 100};
  KvPool firstPool{firstBacking};
  KvPool secondPool{secondBacking};
  engine::Cache first{firstPool, cacheNamespace()};
  engine::Cache second{secondPool, cacheNamespace()};
  const std::vector<uint32_t> firstPrompt(33, 11);
  const std::vector<uint32_t> secondPrompt(33, 22);
  const auto populate = [](engine::Cache &cache,
                           const std::vector<uint32_t> &prompt) {
    cache.beginRequest(1);
    require(cache.ensureTokens(1, 32).granted(), "KV page was not acquired");
    const uint64_t block = cache.publishCommittedBlocks(1, prompt, 32);
    cache.publishCompositeState(block, std::make_shared<TestState>(100));
    cache.endRequest(1);
  };
  populate(first, firstPrompt);
  populate(second, secondPrompt);
  const CacheProbe probe = first.probe(firstPrompt);
  const auto lookup = second.lookup(firstPrompt, {}, &probe);
  require(lookup.kvBoundary == 0 && lookup.resumeBoundary() == 0,
          "a probe from another cache reused a colliding block id");
}

void testCacheLookupAndOneTokenReplay() {
  CacheFixture fixture;
  fixture.publish(0);
  fixture.publish(2);

  auto full = fixture.cache.lookup(fixture.prompt);
  require(full.kvBoundary == 128 && full.resumeBoundary() == 96 &&
              full.junctionBoundary() == 128 && full.state &&
              full.state->kvBlock() == fixture.blocks[2],
          "KV-first lookup did not coordinate dense KV and sparse state");

  full.state.reset();
  auto exactEdge = fixture.lookup(128);
  require(exactEdge.kvBoundary == 96 && exactEdge.resumeBoundary() == 96 &&
              !exactEdge.junctionBoundary(),
          "exact block-edge prompt did not replay one input token");

  auto shortPrompt = fixture.lookup(32);
  require(shortPrompt.kvBoundary == 0 && shortPrompt.resumeBoundary() == 0,
          "single-block prompt illegally became an exact hit");
}

void testPage31Page32Page33Backoff() {
  CacheFixture fixture;
  fixture.publish(0);
  auto page31 = fixture.lookup(31);
  auto page32 = fixture.lookup(32);
  auto page33 = fixture.lookup(33);
  require(page31.kvBoundary == 0 && page31.resumeBoundary() == 0 &&
              page32.kvBoundary == 0 && page32.resumeBoundary() == 0 &&
              page33.kvBoundary == 32 && page33.resumeBoundary() == 32,
          "Page31/32/33 one-token replay boundary is wrong");
}

void testLazyJunctionMaterialization() {
  CacheFixture fixture;
  fixture.publish(0);
  {
    auto first = fixture.cache.lookup(fixture.prompt);
    require(first.resumeBoundary() == 32 && first.junctionBoundary() == 128,
            "first shared KV lookup did not request lazy materialization");
  }
  fixture.publish(3);
  auto second = fixture.cache.lookup(fixture.prompt);
  require(second.resumeBoundary() == 128 && !second.junctionBoundary(),
          "second request did not resume from the lazy junction");
}

void testByteLruAndPins() {
  CacheFixture fixture;
  fixture.publish(0, 150);
  fixture.publish(1, 150);
  auto pinned = fixture.lookup(33);
  require(pinned.state.has_value(), "state pin failed");
  fixture.publish(2, 150);
  // Keep the only KV leaf state-backed so this assertion isolates state LRU;
  // state-free KV leaves otherwise participate in the same global order.
  fixture.publish(3, 150);
  require(fixture.cache.reclaimCache(1, false) == 150,
          "state LRU did not evict one unpinned entry");
  auto missingMiddle = fixture.lookup(65);
  auto newest = fixture.lookup(97);
  require(missingMiddle.resumeBoundary() == 32 &&
              newest.resumeBoundary() == 96 &&
              fixture.cache.snapshot().stateCache.pinned == 2,
          "state LRU evicted a pinned state or lost pin accounting");
  missingMiddle.state.reset();
  pinned.state.reset();
  newest.state.reset();
  require(fixture.cache.reclaimCache(1, false) == 150,
          "released state pins did not restore LRU eligibility");
}

// The field failure this guards: under host pressure a shrink that no request
// was waiting for discarded the only published state one second after it
// appeared, and the follow-up replayed its whole prompt instead of resuming.
void testSpeculativeReclaimKeepsTheResumePoint() {
  CacheFixture fixture;
  fixture.publish(0, 100);
  fixture.publish(2, 100);
  const uint64_t everything = std::numeric_limits<uint64_t>::max();
  static_cast<void>(fixture.cache.reclaimCache(everything, false, true));
  require(fixture.cache.snapshot().stateCache.entries == 1,
          "an unbounded speculative shrink did not stop at the resume point");
  // The chain the kept publication needs survives with it: its own KV block
  // is not state-free, and every ancestor still has a child.
  auto resumed = fixture.cache.lookup(fixture.prompt);
  require(resumed.resumeBoundary() == 96,
          "the kept publication could not resume the next request");
  resumed.state.reset();
  static_cast<void>(fixture.cache.reclaimCache(everything, false, false));
  require(fixture.cache.snapshot().stateCache.entries == 0,
          "a demanded shrink could not reach the resume point");
}

// A newer disposable checkpoint must not displace a warmed ordinary state
// during speculative reclaim.
void testCheckpointDoesNotOutrankTheResumePoint() {
  CacheFixture fixture;
  fixture.publish(0, 100);
  auto warm = fixture.lookup(33);
  require(warm.resumeBoundary() == 32, "the warm prefix did not resume");
  warm.state.reset();
  fixture.cache.publishCompositeState(fixture.blocks.at(2),
                                      std::make_shared<TestState>(100), true);
  const CacheReclaimResult step =
      fixture.cache.reclaimOne(CacheReclaimMode::ReleaseBacking, true);
  const auto kept = fixture.cache.snapshot().stateCache;
  require(step.madeProgress && kept.entries == 1 && kept.checkpointEntries == 0,
          "a disposable checkpoint outranked the resume point");
  require(fixture.lookup(33).resumeBoundary() == 32,
          "the warm conversation lost its resumable prefix");
}

void testKvEvictionInvalidatesStateFirst() {
  CacheFixture fixture;
  fixture.publish(3);
  require(fixture.cache.reclaimCache(1, false) == 100,
          "state was not reclaimed before its KV block");
  require(fixture.cache.reclaimCache(1, false) != 0 &&
              fixture.cache.lookup(fixture.prompt).kvBoundary == 96,
          "KV leaf eviction did not remove the dependent prefix");

  CacheFixture pinnedFixture;
  pinnedFixture.publish(2);
  auto lease = pinnedFixture.lookup(97);
  require(lease.state.has_value(), "replacement state pin failed");
  static_cast<void>(pinnedFixture.cache.reclaimCache(
      std::numeric_limits<uint64_t>::max(), false));
  require(pinnedFixture.cache.snapshot().kvCache.blocks == 3 &&
              pinnedFixture.cache.snapshot().stateCache.entries == 1,
          "pinned composite state did not protect its KV dependency");
}

void testStatePublicationValidation() {
  CacheFixture fixture;
  fixture.publish(0);
  bool duplicateRejected = false;
  try {
    fixture.publish(0);
  } catch (const std::logic_error &) {
    duplicateRejected = true;
  }
  bool missingKvRejected = false;
  try {
    fixture.cache.publishCompositeState(999, std::make_shared<TestState>(100));
  } catch (const std::invalid_argument &) {
    missingKvRejected = true;
  }
  require(duplicateRejected && missingKvRejected &&
              fixture.cache.snapshot().stateCache.entries == 1,
          "invalid state publication changed the cache");
}

void testDuplicateProbePromotesStateWithoutLookupAccounting() {
  CacheFixture fixture;
  fixture.publish(0, 100);
  fixture.publish(1, 200);
  fixture.publish(3, 300);
  const auto before = fixture.cache.snapshot();
  require(fixture.cache.reuseCompositeState(fixture.blocks[0]),
          "resident publication probe missed an existing state");
  require(!fixture.cache.reuseCompositeState(fixture.blocks[2]),
          "publication probe found a state that was never published");
  const auto probed = fixture.cache.snapshot();
  require(probed.stateCache.hits == before.stateCache.hits &&
              probed.stateCache.misses == before.stateCache.misses &&
              probed.stateCache.deduplicatedPublications == 1,
          "publication probe polluted restore hit/miss accounting");
  require(fixture.cache.reclaimCache(1, false) == 200,
          "publication probe did not promote the existing state in LRU");
}

void testUnifiedRecencyAndPhysicalReclaimAccounting() {
  {
    CacheFixture fixture;
    fixture.publish(0, 150);
    const CacheReclaimResult reclaimed = fixture.cache.reclaimOne();
    require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 100 &&
                fixture.cache.snapshot().kvCache.blocks == 3 &&
                fixture.cache.snapshot().stateCache.entries == 1,
            "global cache order did not select the older state-free KV leaf");
  }

  {
    // A cached state is a private copy, so evicting it frees its whole
    // footprint. With the only leaf state-backed, the state is selected.
    CacheFixture fixture;
    fixture.publish(3, 150);
    const CacheReclaimResult reclaimed = fixture.cache.reclaimOne();
    require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 150 &&
                fixture.cache.snapshot().stateCache.entries == 0 &&
                fixture.cache.snapshot().stateCache.bytes == 0 &&
                fixture.cache.snapshot().kvCache.blocks == 4,
            "state eviction did not report its private bytes as freed");
  }
}

// A request publishes a state on an interior block, then keeps committing
// blocks (its decode tail) before it ends. endRequest stamps the state newer
// than the tail, so the unified LRU evicts the state-free tail leaves first
// and the state only once its own block is the oldest leaf.
void testFinishedRequestLeavesTailKvBeforeItsState() {
  test::TestKvBacking backing{4, 100};
  KvPool pool{backing};
  engine::Cache cache{pool, cacheNamespace()};
  std::vector<uint32_t> prompt;
  for (uint32_t token = 0; token < 129; ++token)
    prompt.push_back(5000 + token);

  cache.beginRequest(1);
  require(cache.ensureTokens(1, 64).granted(),
          "prefix pages were not acquired");
  static_cast<void>(cache.publishCommittedBlocks(1, prompt, 64));
  const uint64_t stateBlock = cache.blockAt(1, 64);
  cache.publishCompositeState(stateBlock, std::make_shared<TestState>(100));
  require(cache.ensureTokens(1, 128).granted(), "tail pages were not acquired");
  static_cast<void>(cache.publishCommittedBlocks(1, prompt, 128));
  cache.endRequest(1);
  require(cache.snapshot().kvCache.blocks == 4 &&
              cache.snapshot().stateCache.entries == 1,
          "tail-before-state fixture geometry changed");

  for (uint32_t remaining : {3U, 2U}) {
    const CacheReclaimResult reclaimed =
        cache.reclaimOne(CacheReclaimMode::ReuseBacking);
    require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 0 &&
                cache.snapshot().kvCache.blocks == remaining &&
                cache.snapshot().stateCache.entries == 1,
            "finished request's state was evicted before its KV tail");
  }
  require(cache.lookup(prompt).resumeBoundary() == 64,
          "state did not survive the eviction of the decode tail");
  const CacheReclaimResult state =
      cache.reclaimOne(CacheReclaimMode::ReuseBacking);
  require(state.madeProgress && state.reclaimedBytes == 100 &&
              cache.snapshot().stateCache.entries == 0 &&
              cache.snapshot().kvCache.blocks == 2,
          "state was not evicted once its block became the oldest leaf");
  const CacheReclaimResult leaf =
      cache.reclaimOne(CacheReclaimMode::ReuseBacking);
  require(leaf.madeProgress && cache.snapshot().kvCache.blocks == 1,
          "state block was not evictable after its state left");
}

void testCheckpointLookupProbeDoesNotPromote() {
  CacheFixture fixture;
  const uint64_t block = fixture.blocks[0];
  fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100),
                                      true);
  const StateCheckpoint checkpoint = fixture.cache.checkpointState(block);
  {
    auto probe = fixture.lookup(33);
    require(probe.resumeBoundary() == 32 &&
                !fixture.cache.retireCheckpointState(checkpoint),
            "admission probe did not pin the candidate checkpoint");
  }
  require(fixture.cache.retireCheckpointState(checkpoint) &&
              fixture.cache.snapshot().stateCache.entries == 0,
          "unconsumed admission probe permanently promoted a checkpoint");
}

void testCheckpointRetirementRespectsUseAndPublicationIdentity() {
  CacheFixture fixture;
  const uint64_t block = fixture.blocks[1];
  const auto publish = [&] {
    fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100),
                                        true);
    return fixture.cache.checkpointState(block);
  };
  const StateCheckpoint first = publish();
  require(fixture.cache.reuseCompositeState(block, true),
          "concurrent progress publication did not deduplicate");
  require(fixture.cache.retireCheckpointState(first) &&
              fixture.cache.snapshot().stateCache.entries == 0,
          "duplicate progress alone made a checkpoint permanent");

  const StateCheckpoint replacement = publish();
  require(fixture.cache.retireCheckpointState(first) &&
              fixture.cache.snapshot().stateCache.entries == 1 &&
              replacement.publication != first.publication,
          "stale checkpoint retirement removed a later publication");
  {
    auto lookup = fixture.lookup(65);
    require(lookup.resumeBoundary() == 64, "checkpoint could not be restored");
    fixture.cache.beginRequest(2);
    fixture.cache.restoreRequest(2, lookup);
    fixture.cache.endRequest(2);
    require(!fixture.cache.retireCheckpointState(replacement) &&
                fixture.cache.snapshot().stateCache.entries == 1,
            "retirement removed a pinned restore checkpoint");
  }
  require(fixture.cache.retireCheckpointState(replacement) &&
              fixture.cache.snapshot().stateCache.entries == 0,
          "restoration made a checkpoint ineligible for rolling retirement");

  const StateCheckpoint shared = publish();
  require(fixture.cache.reuseCompositeState(block),
          "junction could not reuse a checkpoint");
  require(fixture.cache.retireCheckpointState(shared) &&
              fixture.cache.retireCheckpointState({}) &&
              fixture.cache.snapshot().stateCache.entries == 1,
          "junction publication did not preserve its checkpoint");
  const auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.checkpointEntries == 0 && snapshot.checkpointBytes == 0 &&
              snapshot.evictions == 0 && snapshot.checkpointEvictions == 0 &&
              snapshot.checkpointRetirements == 2,
          "retirement or upgrade corrupted checkpoint accounting");
}

void testRestoredCheckpointsKeepTheirEvictionPriority() {
  CacheFixture fixture;
  fixture.publish(0, 150);
  fixture.publish(3, 400);
  fixture.cache.publishCompositeState(fixture.blocks[1],
                                      std::make_shared<TestState>(200), true);
  fixture.cache.publishCompositeState(fixture.blocks[2],
                                      std::make_shared<TestState>(300), true);
  {
    auto lookup = fixture.lookup(65);
    fixture.cache.beginRequest(2);
    fixture.cache.restoreRequest(2, lookup);
    fixture.cache.endRequest(2);
  }

  require(fixture.cache.reclaimOneState() &&
              !fixture.cache.checkpointState(fixture.blocks[2]) &&
              fixture.cache.checkpointState(fixture.blocks[1]),
          "restore did not refresh recency within checkpoint LRU");
  require(fixture.cache.reclaimOneState() &&
              !fixture.cache.checkpointState(fixture.blocks[1]) &&
              fixture.lookup(33).resumeBoundary() == 32 &&
              fixture.cache.snapshot().stateCache.entries == 2,
          "recently restored checkpoint displaced an ordinary state");
  const auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.checkpointEntries == 0 && snapshot.checkpointBytes == 0 &&
              snapshot.checkpointEvictions == 2 &&
              snapshot.checkpointRetirements == 0,
          "pressure eviction was counted as rolling retirement");
}

void testCheckpointReclaimPrecedesOlderKv() {
  CacheFixture fixture;
  fixture.publish(0, 150);
  fixture.cache.publishCompositeState(fixture.blocks[2],
                                      std::make_shared<TestState>(200), true);
  const auto reclaimed = fixture.cache.reclaimOne();
  require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 200 &&
              fixture.cache.snapshot().kvCache.blocks == 4 &&
              fixture.lookup(33).resumeBoundary() == 32,
          "checkpoint reclaim displaced older ordinary state or KV");
  require(fixture.cache.reclaimOne().reclaimedBytes == 100 &&
              fixture.cache.snapshot().kvCache.blocks == 3,
          "ordinary state and KV lost their shared LRU order");
}

void testOptionalReclaimLeavesOrdinaryStateIntact() {
  CacheFixture fixture;
  fixture.publish(0, 150);
  fixture.cache.publishCompositeState(fixture.blocks[2],
                                      std::make_shared<TestState>(200), true);
  require(fixture.cache.reclaimOneState(true) &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 0 &&
              fixture.lookup(33).resumeBoundary() == 32,
          "optional publication failed to recycle a disposable checkpoint");
  require(!fixture.cache.reclaimOneState(true) &&
              fixture.cache.snapshot().stateCache.entries == 1,
          "optional publication displaced ordinary cached state");
}

void testCheckpointPinsAndBoundaryUpgrade() {
  CacheFixture fixture;
  fixture.publish(0, 100);
  fixture.publish(3, 400);
  fixture.cache.publishCompositeState(fixture.blocks[1],
                                      std::make_shared<TestState>(200), true);
  fixture.cache.publishCompositeState(fixture.blocks[2],
                                      std::make_shared<TestState>(300), true);
  const auto checkpoint = fixture.cache.checkpointState(fixture.blocks[1]);
  auto first = fixture.lookup(65);
  auto second = fixture.lookup(97);
  require(fixture.cache.reclaimOneState() &&
              fixture.lookup(33).resumeBoundary() == 0 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 2 &&
              fixture.cache.snapshot().stateCache.checkpointBytes == 500,
          "global reclaim evicted pinned checkpoint state");

  require(fixture.cache.reuseCompositeState(fixture.blocks[1]) &&
              fixture.cache.retireCheckpointState(checkpoint),
          "pinned ordinary boundary could not upgrade its checkpoint");
  first.state.reset();
  second.state.reset();
  require(fixture.cache.reclaimOneState() &&
              fixture.lookup(97).resumeBoundary() == 64 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 0,
          "pinned boundary upgrade corrupted the eviction queues");
  require(fixture.cache.reclaimOneState() &&
              fixture.cache.lookup(fixture.prompt).resumeBoundary() == 64,
          "upgraded checkpoint did not join ordinary LRU");

  require(fixture.cache.reuseCompositeState(fixture.blocks[1], true) &&
              !fixture.cache.checkpointState(fixture.blocks[1]),
          "optional publication downgraded an ordinary boundary");
}

void testCheckpointPressurePreservesHotPrefix() {
  SharedBudget budget;
  BudgetBacking backing(budget);
  KvPool pool(backing);
  engine::Cache cache(pool, cacheNamespace());
  const std::vector<uint32_t> hot(33, 11);
  const std::vector<uint32_t> cold(65, 22);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 32).granted(), "hot KV admission failed");
  const uint64_t hotBlock = cache.publishCommittedBlocks(1, hot, 32);
  cache.publishCompositeState(hotBlock, std::make_shared<BudgetState>(budget));
  cache.endRequest(1);
  {
    auto lookup = cache.lookup(hot);
    require(lookup.resumeBoundary() == 32, "hot prefix did not restore");
    cache.beginRequest(2);
    cache.restoreRequest(2, lookup);
    cache.endRequest(2);
  }

  cache.beginRequest(3);
  require(cache.ensureTokens(3, 32).granted(), "cold KV admission failed");
  const uint64_t coldBlock = cache.publishCommittedBlocks(3, cold, 32);
  cache.publishCompositeState(coldBlock, std::make_shared<BudgetState>(budget),
                              true);
  require(budget.used == SharedBudget::capacity,
          "checkpoint did not fill the shared allocation budget");
  const TokenAdmission denied = cache.ensureTokens(3, 64);
  require(denied.failure == KvPageAcquireFailure::PhysicalCapacity,
          "necessary KV growth did not reach physical capacity");
  const auto reclaimed = cache.reclaimOne(CacheReclaimMode::ReuseBacking);
  require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 200 &&
              cache.ensureTokens(3, 64).granted() &&
              cache.lookup(hot).resumeBoundary() == 32 &&
              cache.lookup(cold).resumeBoundary() == 0 && budget.used == 500,
          "successful checkpoint allocation later displaced the hot prefix");
  cache.endRequest(3);
}

void testLogicalKvPressureStillReclaimsPages() {
  CacheFixture fixture;
  fixture.cache.publishCompositeState(fixture.blocks[0],
                                      std::make_shared<TestState>(200), true);
  fixture.cache.beginRequest(2);
  require(fixture.cache.ensureTokens(2, 32).granted() &&
              fixture.cache.snapshot().kvCache.blocks == 3 &&
              fixture.cache.checkpointState(fixture.blocks[0]),
          "logical page pressure discarded state without freeing a KV page");
  fixture.cache.endRequest(2);
}

} // namespace

int main() {
  try {
    testCheckpointLookupProbeDoesNotPromote();
    testCheckpointRetirementRespectsUseAndPublicationIdentity();
    testRestoredCheckpointsKeepTheirEvictionPriority();
    testCheckpointReclaimPrecedesOlderKv();
    testOptionalReclaimLeavesOrdinaryStateIntact();
    testCheckpointPinsAndBoundaryUpgrade();
    testCheckpointPressurePreservesHotPrefix();
    testLogicalKvPressureStillReclaimsPages();
    testSchedulingProbeDoesNotChangeCachePolicy();
    testValidAdmissionProbePreservesLookupAndAccounting();
    testProbeFallsBackWhenPromptChanges();
    testProbeRechecksFirstMissAndPromptLength();
    testProbeRechecksStateChanges();
    testProbeFallsBackWhenKvChanges();
    testProbeBindsImageIdentity();
    testProbeCannotCrossCaches();
    testCacheLookupAndOneTokenReplay();
    testPage31Page32Page33Backoff();
    testLazyJunctionMaterialization();
    testByteLruAndPins();
    testSpeculativeReclaimKeepsTheResumePoint();
    testCheckpointDoesNotOutrankTheResumePoint();
    testKvEvictionInvalidatesStateFirst();
    testStatePublicationValidation();
    testDuplicateProbePromotesStateWithoutLookupAccounting();
    testUnifiedRecencyAndPhysicalReclaimAccounting();
    testFinishedRequestLeavesTailKvBeforeItsState();
    std::cout << "KV-first cache tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "KV-first cache tests failed: " << error.what() << '\n';
    return 1;
  }
}
