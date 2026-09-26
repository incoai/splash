#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Cache.hpp"

#include <algorithm>
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

struct TransferControl {
  bool ready = false;
  bool success = true;
  uint32_t slots = 0;
  uint32_t capacity = 2;
};

// A write the tier admits while the quota has a slot; its disk copy holds
// that slot until it is dropped.
std::unique_ptr<StateOffload> writeState(const std::shared_ptr<TransferControl> &control);

class TieredState final : public CompositeState {
public:
  TieredState(std::shared_ptr<TransferControl> control, bool disk = false)
      : control_(std::move(control)), disk_(disk) { if (disk_) ++control_->slots; }
  ~TieredState() override { if (disk_) --control_->slots; }
  uint64_t bytes() const noexcept override { return 100; }
  uint64_t residentBytes() const noexcept override { return disk_ ? 0 : bytes(); }
  bool canOffload() const noexcept override { return !disk_; }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override {
    return writeState(control_);
  }
private:
  std::shared_ptr<TransferControl> control_;
  bool disk_;
};

std::unique_ptr<StateOffload> writeState(const std::shared_ptr<TransferControl> &control) {
  if (control->slots >= control->capacity) return {};
  struct Ticket final : StateOffload {
    std::shared_ptr<TransferControl> control;
    std::shared_ptr<const CompositeState> disk;
    bool ready() const noexcept override { return control->ready; }
    bool finish() override { return control->success; }
    const std::shared_ptr<const CompositeState> &state() const noexcept override {
      return disk;
    }
  };
  auto ticket = std::make_unique<Ticket>();
  ticket->control = control;
  ticket->disk = std::make_shared<TieredState>(control, true);
  return ticket;
}

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
  engine::Cache cache;
  std::vector<uint32_t> prompt;
  std::vector<uint64_t> blocks;

  explicit CacheFixture(model::KvTier *tier = nullptr) : cache(pool, cacheNamespace(), tier) {
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
  const auto cachedTokens = [&](std::span<const uint32_t> prompt) {
    return fixture.cache.probe(prompt).cachedTokens();
  };
  require(cachedTokens(fixture.prompt) == 0,
          "KV without recurrent state was counted as reusable work");
  fixture.publish(0);
  fixture.publish(3);
  const auto prefix = std::span<const uint32_t>(fixture.prompt).first(33);
  require(cachedTokens(prefix) == 32 && cachedTokens(fixture.prompt) == 128 &&
              fixture.cache.snapshot().stateCache.pinned == 0 &&
              fixture.cache.snapshot().lookup.lookups == 0,
          "scheduling probe pinned backing or counted a cache hit");
  require(fixture.cache.reclaimOneState() && cachedTokens(prefix) == 0 &&
              cachedTokens(fixture.prompt) == 128,
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
    require(fixture.cache.restoreRequest(2, lookup).granted(), "restore pages were denied");
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
    require(fixture.cache.restoreRequest(2, lookup).granted(), "restore pages were denied");
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
    require(cache.restoreRequest(2, lookup).granted(), "restore pages were denied");
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

void recordUse(CacheFixture &fixture, uint32_t tokens) {
  auto lookup = fixture.lookup(tokens);
  fixture.cache.recordLookup(lookup);
}

void publishReusable(CacheFixture &fixture, uint64_t block,
                     std::shared_ptr<const CompositeState> state) {
  fixture.cache.publishCompositeState(block, std::move(state));
  const auto index = std::find(fixture.blocks.begin(), fixture.blocks.end(), block) - fixture.blocks.begin();
  recordUse(fixture, static_cast<uint32_t>((index + 1) * 32 + 1));
}

void testTieredStateLifecycle() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne().reclaimedBytes == 100, "demotion did not free the state's RAM at once");
  auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.bytes == 0 && snapshot.diskBytes == 100 && snapshot.entries == 1 &&
              snapshot.offloads == 1 && control->slots == 1,
          "demoted entry did not become its disk copy");
  require(!fixture.cache.releasePending() && !fixture.cache.pollTransfers(),
          "unfinished write was reported as pending backing or consumed early");
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TestState>(100), true);
  require(fixture.cache.reclaimOne().reclaimedBytes == 100,
          "pending write prevented disposable checkpoint reclamation");
  control->ready = true;
  require(fixture.cache.pollTransfers() && !fixture.cache.pollTransfers(),
          "completed write was not consumed exactly once");
  snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.bytes == 0 && snapshot.diskBytes == 100 && snapshot.entries == 1,
          "completion changed tier occupancy");
  {
    auto lookup = fixture.lookup(129);
    require(lookup.resumeBoundary() == 128 && !lookup.state->state()->residentBytes(),
            "disk state did not retain matching KV identity");
    require(!fixture.cache.reclaimOne().madeProgress, "pinned restore lost its KV");
  }
  require(fixture.cache.reclaimOne().madeProgress, "disk state permanently protected KV");
  require(fixture.cache.snapshot().stateCache.entries == 0 && control->slots == 0,
          "KV eviction leaked disk state");
}

void testTieredWriteReuseAndFailure() {
  for (bool fail : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
    require(fixture.cache.reclaimOne().reclaimedBytes == 100, "demotion did not free RAM");
    {
      // A hit inside the write window is a disk hit queued behind the write.
      auto lookup = fixture.lookup(129);
      require(lookup.state && !lookup.state->state()->residentBytes(),
              "in-flight copy was not served from disk");
    }
    control->ready = true;
    control->success = !fail;
    require(fixture.cache.pollTransfers(), "write did not finish");
    const auto snapshot = fixture.cache.snapshot().stateCache;
    require(fail ? snapshot.entries == 0 && snapshot.offloadFailures == 1 && control->slots == 0
                 : snapshot.entries == 1 && snapshot.diskBytes == 100 && control->slots == 1,
            "failed or completed write has incorrect lifetime");
  }
}

// A full quota replaces the least recently used copy; disposable checkpoints
// are never written.
void testDiskQuotaReplacesByRecency() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 1;
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control));
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne().madeProgress && fixture.cache.pollTransfers() &&
              control->slots == 1,
          "first write failed");
  // The second state needs the slot: the older copy, the only one of its
  // state, gives way.
  require(fixture.cache.reclaimOne().madeProgress && fixture.cache.pollTransfers(),
          "replacement failed");
  require(fixture.cache.snapshot().stateCache.entries == 1 && control->slots == 1 &&
              fixture.lookup(129).resumeBoundary() == 128 && !fixture.lookup(33).state,
          "the older disk copy was not replaced");
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), true);
  require(fixture.cache.reclaimOneState(true) && fixture.cache.snapshot().stateCache.offloads == 2,
          "checkpoint was not immediately reclaimable or replaced another copy");
}

// A rolling checkpoint uses the tier like any state, into free quota only:
// straight to disk when no cache slot holds it, written under RAM pressure
// while the quota has room and dropped when it has none, retired from both
// tiers with its successor; under KV pressure its leaf is demoted, not dropped.
void testRollingCheckpointsUseTheTier() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const StateWriter write = [&](std::function<void()>) { return writeState(control); };
  require(fixture.cache.publishStateToDisk(fixture.blocks[1], write, true) &&
              fixture.cache.pollTransfers(),
          "checkpoint was refused the tier");
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.checkpointEntries == 1 && stats.checkpointBytes == 0 &&
              stats.diskBytes == 100 && control->slots == 1,
          "disk checkpoint was not accounted as a checkpoint");
  const StateCheckpoint point = fixture.cache.checkpointState(fixture.blocks[1]);
  require(point && fixture.cache.retireCheckpointState(point) && control->slots == 0 &&
              fixture.cache.snapshot().stateCache.entries == 0,
          "retirement left the disk copy behind");
  fixture.cache.publishCompositeState(fixture.blocks[3],
                                      std::make_shared<TieredState>(control), true);
  require(fixture.cache.reclaimOneState(true) && fixture.cache.pollTransfers(),
          "RAM checkpoint was not reclaimed");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.offloads == 2 && stats.bytes == 0 && stats.checkpointEntries == 1 &&
              stats.checkpointBytes == 0 && control->slots == 1,
          "RAM checkpoint was dropped although the quota had room");
  control->capacity = 1;
  fixture.cache.publishCompositeState(fixture.blocks[2],
                                      std::make_shared<TieredState>(control), true);
  require(fixture.cache.reclaimOneState(true) && control->slots == 1 &&
              fixture.cache.snapshot().stateCache.offloads == 2 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 1,
          "a checkpoint replaced another copy");
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReuseBacking).madeProgress &&
              tier.demotions == 1 && fixture.cache.snapshot().kvCache.blocks == 4,
          "the checkpoint's leaf was dropped instead of demoted");
}

void testDiskPromotionAndInvalidation() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const auto block = fixture.blocks[3];
  publishReusable(fixture, block, std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne().madeProgress && fixture.cache.pollTransfers(), "offload failed");
  auto first = fixture.lookup(129);
  auto peer = fixture.lookup(129);
  auto invalid = first.state->state();
  fixture.cache.discardState(block, invalid.get());
  require(!fixture.lookup(129).state, "invalid disk entry admitted another reader");
  require(fixture.cache.probe(fixture.prompt).cachedTokens() == 0,
          "scheduling probe counted an invalid disk state as reusable work");
  first = {};
  fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100));
  fixture.cache.discardState(block, invalid.get());
  peer = {};
  invalid.reset();
  auto replacement = fixture.lookup(129);
  require(replacement.state && replacement.state->state()->residentBytes() == 100 &&
              fixture.cache.snapshot().stateCache.diskBytes == 0 && control->slots == 0,
          "late failed read invalidated a fresh publication or leaked a slot");
}

void testInvalidationDuringOffload() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  auto source = std::make_shared<TieredState>(control);
  publishReusable(fixture, fixture.blocks[3], source);
  require(fixture.cache.reclaimOne().madeProgress, "write not started");
  fixture.cache.discardState(fixture.blocks[3], source.get());
  require(fixture.cache.snapshot().stateCache.entries == 1,
          "a stale handle to the RAM copy discarded the disk copy");
  {
    auto twin = fixture.lookup(129);
    fixture.cache.discardState(fixture.blocks[3], twin.state->state().get());
  }
  control->ready = true;
  require(fixture.cache.pollTransfers() && fixture.cache.snapshot().stateCache.entries == 0 &&
              control->slots == 0, "completed write revived an invalidated disk copy");
}

void testDemotionFreesTheBufferAtOnce() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control));
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState(), "required publication could not recycle a buffer");
  auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.entries == 2 && snapshot.bytes == 100 && snapshot.diskBytes == 100,
          "demotion took a second state or kept the victim's RAM");
  // One write in flight: a second victim inside the window is dropped.
  require(fixture.cache.reclaimOneState(), "second recycle failed");
  snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.entries == 1 && snapshot.bytes == 0 && snapshot.offloads == 1,
          "second victim was written while a write was in flight");
  control->ready = true;
  require(fixture.cache.pollTransfers() && fixture.lookup(33).resumeBoundary() == 32,
          "cold state was not preserved on disk");
}

class PromotionTicket final : public StateRestore {
public:
  uint32_t copies = 0;
  bool available = true;
  bool ready() const noexcept override { return true; }
  bool finish() override { return true; }
  void cancel() noexcept override {}
  std::shared_ptr<const CompositeState> snapshot() override {
    ++copies;
    return available ? std::make_shared<TestState>(100) : nullptr;
  }
};

void testPromotionIdentityAndDenial() {
  for (bool available : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
    require(fixture.cache.reclaimOne().madeProgress && fixture.cache.pollTransfers(), "promotion fixture write failed");
    auto first = fixture.lookup(129);
    auto peer = fixture.lookup(129);
    PromotionTicket ticket;
    ticket.available = available;
    fixture.cache.promoteState(first, ticket);
    if (available) fixture.cache.promoteState(peer, ticket);
    auto current = fixture.lookup(129);
    require(current.state && bool(current.state->state()->residentBytes()) == available,
            "denied promotion lost disk fallback or successful promotion remained on disk");
    require(ticket.copies == 1, "concurrent restore copied an already-promoted payload");
    if (available) {
      fixture.cache.discardState(fixture.blocks[3], first.state->state().get());
      require(fixture.lookup(129).state.has_value(), "late read invalidated a promoted payload");
    }
    const auto stats = fixture.cache.snapshot().stateCache;
    require(stats.promotions == (available ? 1 : 0) &&
                stats.promotionsSkipped == (available ? 0 : 1), "promotion accounting failed");
  }
}

// A republication over a disk-only copy gives the state a RAM copy and keeps
// the disk copy: its next eviction from RAM costs no write.
void testRepublicationKeepsTheDiskCopy() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control));
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne().madeProgress && fixture.cache.pollTransfers(),
          "state was not demoted");
  require(!fixture.cache.reuseCompositeState(fixture.blocks[0]),
          "a disk-only copy stood in for a RAM publication");
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control));
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.diskBytes == 100 && stats.bytes == 200 && control->slots == 1 &&
              fixture.lookup(33).state->state()->residentBytes() == 100,
          "republication dropped the disk copy or did not add the RAM copy");
  require(fixture.cache.reuseCompositeState(fixture.blocks[0]), "RAM copy was not reusable");
  // The other state is used; the republished one is the oldest RAM copy and
  // leaves RAM without a second write.
  recordUse(fixture, 129);
  require(fixture.cache.reclaimOneState() && !fixture.cache.pollTransfers(),
          "re-eviction of a state with a disk copy wrote it again");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.bytes == 100 && stats.diskBytes == 100 && stats.offloads == 1 &&
              fixture.lookup(33).state && !fixture.lookup(33).state->state()->residentBytes(),
          "the disk copy did not serve after the RAM copy left");
}

// A block remembers that it held a reusable state: a lookup that finds the
// KV without a state in either tier reports a lost state. Disk copies and
// dropped checkpoints are not lost states.
void testLostStatesAreCounted() {
  CacheFixture fixture;
  fixture.publish(3);
  require(fixture.cache.reclaimOneState(), "state was not dropped");
  {
    auto lookup = fixture.lookup(129);
    require(lookup.kvBoundary == 128 && !lookup.state && lookup.lostState,
            "dropped state was not recognised");
    fixture.cache.recordLookup(lookup);
  }
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState() && fixture.cache.pollTransfers(), "state was not demoted");
  {
    auto hit = fixture.lookup(129);
    require(hit.state && !hit.lostState, "a disk hit was counted as a lost state");
  }
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TestState>(100), true);
  require(fixture.cache.reclaimOneState(true), "checkpoint was not dropped");
  {
    auto shallow = fixture.lookup(65);
    require(!shallow.state && !shallow.lostState, "a dropped checkpoint counted as a lost state");
  }
  require(fixture.cache.snapshot().lookup.lostStateMisses == 1, "lost states were not counted");
}

// A state restored from disk takes a RAM copy and keeps the disk copy; when
// RAM reclaims it again nothing is written.
void testRestoredStateKeepsItsDiskCopy() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState() && fixture.cache.pollTransfers(), "state was not demoted");
  {
    auto lookup = fixture.lookup(129);
    PromotionTicket ticket;
    fixture.cache.promoteState(lookup, ticket);
    require(ticket.copies == 1, "restored state was not promoted");
  }
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.promotions == 1 && stats.bytes == 100 && stats.diskBytes == 100 && control->slots == 1,
          "promotion dropped the disk copy");
  require(fixture.cache.reclaimOneState() && !fixture.cache.pollTransfers(),
          "re-eviction after a restore wrote the state again");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.bytes == 0 && stats.diskBytes == 100 && stats.offloads == 1 &&
              fixture.lookup(129).state && !fixture.lookup(129).state->state()->residentBytes(),
          "the disk copy did not remain after the RAM copy left");
}

void testDiskCheckpointRamAccounting() {
  for (bool republish : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    fixture.cache.publishCompositeState(fixture.blocks[3],
                                        std::make_shared<TieredState>(control), true);
    const auto point = fixture.cache.checkpointState(fixture.blocks[3]);
    require(fixture.cache.reclaimOneState(true) && fixture.cache.pollTransfers(),
            "checkpoint demotion failed");
    if (republish) {
      fixture.cache.publishCompositeState(fixture.blocks[3],
                                          std::make_shared<TestState>(100), true);
    } else {
      auto lookup = fixture.lookup(129);
      PromotionTicket ticket;
      fixture.cache.promoteState(lookup, ticket);
    }
    auto stats = fixture.cache.snapshot().stateCache;
    require(stats.checkpointEntries == 1 && stats.checkpointBytes == 100 &&
                stats.bytes == 100 && stats.diskBytes == 100,
            "restored checkpoint RAM was not accounted");
    require(fixture.cache.retireCheckpointState(point), "checkpoint retirement failed");
    stats = fixture.cache.snapshot().stateCache;
    require(stats.checkpointBytes == 0 && stats.bytes == 0 && stats.diskBytes == 0,
            "checkpoint retirement underflowed memory accounting");
  }
}

void testOrdinaryPublicationUpgradesDiskCheckpoint() {
  for (bool onDisk : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    const StateWriter write = [&](std::function<void()>) { return writeState(control); };
    require(fixture.cache.publishStateToDisk(fixture.blocks[3], write, true) &&
                fixture.cache.pollTransfers(), "disk checkpoint publication failed");
    const auto point = fixture.cache.checkpointState(fixture.blocks[3]);
    if (onDisk) {
      require(fixture.cache.publishStateToDisk(fixture.blocks[3], write),
              "ordinary disk publication failed");
    } else {
      fixture.cache.publishCompositeState(fixture.blocks[3],
                                          std::make_shared<TestState>(100));
    }
    require(!fixture.cache.checkpointState(fixture.blocks[3]) &&
                fixture.cache.snapshot().stateCache.checkpointEntries == 0 &&
                fixture.cache.snapshot().stateCache.checkpointBytes == 0 &&
                fixture.cache.retireCheckpointState(point) &&
                fixture.lookup(129).state.has_value(),
            "ordinary publication retained a disposable disk checkpoint lifetime");
    require(control->slots == 1 && fixture.cache.snapshot().stateCache.offloads == 1,
            "upgrading a disk checkpoint rewrote its payload");
  }
}

// A state no cache slot can hold is written from its lane straight to disk:
// the entry is the disk copy with the write in flight, a hit inside the
// write window is a disk hit, one write at a time holds the staging buffer,
// a block already on disk is not written twice, and under KV pressure the
// stated leaf is demoted rather than dropped.
void testDiskPublicationLifecycle() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  const StateWriter write = [&](std::function<void()>) { return writeState(control); };
  require(fixture.cache.publishStateToDisk(fixture.blocks[3], write),
          "disk publication was refused");
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 1 && stats.bytes == 0 && stats.diskBytes == 100 &&
              stats.offloads == 1 && stats.publications == 1 && control->slots == 1,
          "disk publication did not become a disk copy with its write in flight");
  {
    auto lookup = fixture.lookup(129);
    require(lookup.resumeBoundary() == 128 && lookup.state &&
                !lookup.state->state()->residentBytes() && !lookup.lostState,
            "the state in flight was not served from disk");
  }
  require(!fixture.cache.publishStateToDisk(fixture.blocks[1], write) &&
              fixture.cache.snapshot().stateCache.entries == 1 && control->slots == 1,
          "a second write started beside the one in flight");
  require(fixture.cache.publishStateToDisk(fixture.blocks[3], write) &&
              fixture.cache.snapshot().stateCache.offloads == 1 &&
              fixture.cache.snapshot().stateCache.deduplicatedPublications == 1,
          "a block already on disk was written again");
  control->ready = true;
  require(fixture.cache.pollTransfers() && !fixture.cache.pollTransfers(),
          "the write was not consumed exactly once");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 1 && stats.diskBytes == 100 && stats.offloadFailures == 0,
          "completion changed tier occupancy");
  require(fixture.cache.publishStateToDisk(fixture.blocks[1], write) && control->slots == 2,
          "the next publication did not follow the finished write");
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReuseBacking).madeProgress &&
              tier.demotions == 1 && fixture.cache.snapshot().kvCache.blocks == 4,
          "the stated leaf was dropped instead of demoted");
}

// A failed write leaves nothing of a state published to disk; the block
// remembers that it had one.
void testDiskPublicationFailure() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->success = false;
  const StateWriter write = [&](std::function<void()>) { return writeState(control); };
  require(fixture.cache.publishStateToDisk(fixture.blocks[3], write) &&
              fixture.cache.pollTransfers(),
          "the failing write did not run");
  const auto stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 0 && stats.offloadFailures == 1 && stats.diskBytes == 0 &&
              control->slots == 0,
          "a failed write left a disk copy behind");
  auto lookup = fixture.lookup(129);
  require(lookup.kvBoundary == 128 && !lookup.state && lookup.lostState,
          "the lost state was not recognised");
}

// A write that fails while a lookup holds its disk copy leaves the block
// nothing; a publication there meanwhile, in RAM or on disk, takes the
// entry over and outlives the reader.
void testFailedWriteUnderALookup() {
  for (bool onDisk : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    const auto block = fixture.blocks[3];
    publishReusable(fixture, block, std::make_shared<TieredState>(control));
    require(fixture.cache.reclaimOneState(), "state was not demoted");
    auto reader = fixture.lookup(129);
    require(reader.state && !reader.state->state()->residentBytes(),
            "the state in flight was not served from disk");
    control->ready = true;
    control->success = false;
    require(fixture.cache.pollTransfers(), "the failing write did not run");
    auto stats = fixture.cache.snapshot().stateCache;
    require(stats.entries == 1 && stats.pinned == 1 && stats.diskBytes == 0 &&
                stats.offloadFailures == 1 && !fixture.lookup(129).state,
            "a failed write under a lookup left a copy behind");
    if (onDisk) {
      control->success = true;
      const StateWriter write = [&](std::function<void()>) { return writeState(control); };
      require(fixture.cache.publishStateToDisk(block, write) && fixture.cache.pollTransfers(),
              "the disk publication at the emptied block failed");
    } else {
      fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100));
    }
    // The reader's read fails as well; it leaves with its lease.
    fixture.cache.discardState(block, reader.state->state().get());
    reader = {};
    stats = fixture.cache.snapshot().stateCache;
    require(stats.entries == 1 && stats.pinned == 0 && stats.bytes == (onDisk ? 0 : 100) &&
                stats.diskBytes == (onDisk ? 100 : 0) && control->slots == (onDisk ? 1 : 0) &&
                fixture.lookup(129).state,
            "the publication did not take over the emptied entry");
  }
}

// Every failed write is counted, also one whose entry left before it
// landed: a checkpoint retired meanwhile, or a state a failed read
// condemned and a new publication replaced.
void testFailedWriteIsCountedAfterItsEntryLeft() {
  for (bool replaced : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    const auto block = fixture.blocks[3];
    if (replaced) {
      publishReusable(fixture, block, std::make_shared<TieredState>(control));
      require(fixture.cache.reclaimOneState(), "state was not demoted");
      {
        auto reader = fixture.lookup(129);
        fixture.cache.discardState(block, reader.state->state().get());
      }
      fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100));
    } else {
      const StateWriter write = [&](std::function<void()>) { return writeState(control); };
      require(fixture.cache.publishStateToDisk(block, write, true) &&
                  fixture.cache.retireCheckpointState(fixture.cache.checkpointState(block)),
              "the checkpoint in flight was not retired");
    }
    control->ready = true;
    control->success = false;
    require(fixture.cache.pollTransfers(), "the failing write did not run");
    const auto stats = fixture.cache.snapshot().stateCache;
    require(stats.offloads == 1 && stats.offloadFailures == 1 &&
                stats.entries == (replaced ? 1 : 0) && control->slots == 0,
            "a failed write went uncounted");
  }
}

// A full quota gives up its oldest copy for a state written from a lane;
// when the disk holds nothing to give, nothing is published.
void testDiskPublicationMakesRoom() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 1;
  publishReusable(fixture, fixture.blocks[1], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState() && fixture.cache.pollTransfers() &&
              control->slots == 1,
          "the older state did not fill the quota");
  const StateWriter write = [&](std::function<void()>) { return writeState(control); };
  require(fixture.cache.publishStateToDisk(fixture.blocks[3], write) &&
              fixture.cache.pollTransfers(),
          "the full quota refused the lane's state");
  const auto stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 1 && stats.offloads == 2 && control->slots == 1 &&
              !fixture.lookup(65).state && fixture.lookup(129).state,
          "the older copy did not make room for the new one");
  control->capacity = 0;
  CacheFixture empty;
  require(!empty.cache.publishStateToDisk(empty.blocks[3], write) &&
              empty.cache.snapshot().stateCache.entries == 0 &&
              !empty.lookup(129).lostState,
          "a state was published without a disk copy");
}

// The quota is one order across both kinds of copies: a write that needs
// room drops redundant copies first, KV or state, then the oldest copy that
// is the only one.
void testDiskReplacementSpansStatesAndKv() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  tier.capacity = 1;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 1;
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState() && fixture.cache.pollTransfers(), "state was not demoted");
  require(fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 1, "leaf was not written");
  tier.complete();
  require(fixture.cache.pollTransfers(), "leaf did not land");
  // Both come back: the block and its state hold RAM and disk copies.
  {
    auto lookup = fixture.lookup(129);
    fixture.cache.beginRequest(2);
    require(fixture.cache.restoreRequest(2, lookup).granted(), "restore was denied");
    tier.complete();
    require(fixture.cache.pollTransfers(), "restore did not land");
    PromotionTicket ticket;
    fixture.cache.promoteState(lookup, ticket);
    require(ticket.copies == 1, "state was not promoted");
    lookup = {};
    fixture.cache.endRequest(2);
  }
  auto stats = fixture.cache.snapshot();
  require(stats.kvTier.diskBlocks == 1 && stats.kvCache.blocks == 4 &&
              stats.stateCache.bytes == 100 && stats.stateCache.diskBytes == 100,
          "restore did not leave both copies of both");
  // A new state at block 2 needs disk room. The oldest RAM copy leaves RAM
  // for nothing first; then the write gives up the redundant KV copy and,
  // when that is not enough, the state copy that is the only one.
  fixture.cache.publishCompositeState(fixture.blocks[2], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState() && fixture.cache.snapshot().stateCache.offloads == 1,
          "the restored state was written again");
  require(fixture.cache.reclaimOneState() && fixture.cache.pollTransfers(), "new state was not written");
  stats = fixture.cache.snapshot();
  require(stats.stateCache.offloads == 2 && stats.stateCache.entries == 1 && control->slots == 1 &&
              stats.kvTier.diskBlocks == 0 && stats.kvCache.blocks == 4 &&
              fixture.lookup(97).state && !fixture.lookup(97).state->state()->residentBytes(),
          "disk replacement did not give up redundant copies before the only ones");
}

// Independent prefixes under random publications, uses and reclaims: the
// RAM contents match a cache without the tier step for step, and every hit
// without the tier is a hit with it. The disk only adds.
void testTierOnlyAddsToTierOff() {
  struct Prefixes {
    test::TestKvBacking backing{4, 100};
    KvPool pool{backing};
    engine::Cache cache{pool, cacheNamespace()};
    std::array<std::vector<uint32_t>, 4> prompts;
    std::array<uint64_t, 4> blocks{};

    Prefixes() {
      for (uint32_t i = 0; i < prompts.size(); ++i) {
        prompts[i].assign(KvCache::pageTokens, 1000 + i);
        cache.beginRequest(i + 1);
        require(cache.ensureTokens(i + 1, KvCache::pageTokens).granted(), "prefix KV failed");
        blocks[i] = cache.publishCommittedBlocks(i + 1, prompts[i], KvCache::pageTokens);
        cache.endRequest(i + 1);
        prompts[i].push_back(9999);
      }
    }
    engine::CacheLookup lookup(uint32_t index) { return cache.lookup(prompts[index]); }
  };
  for (uint32_t seed = 1; seed <= 64; ++seed) {
    Prefixes off, on;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    uint32_t random = seed;
    const auto next = [&] { return (random = random * 1664525u + 1013904223u) >> 8; };
    for (int step = 0; step < 48; ++step) {
      const uint32_t index = next() % 4;
      switch (next() % 4) {
      case 0: { // a request reaching this boundary publishes unless RAM has it
        auto a = off.lookup(index);
        auto b = on.lookup(index);
        require(!a.state || (b.state && b.state->state()->residentBytes()),
                "a RAM state without the tier is not resident with it");
        const bool resident = a.state.has_value();
        a = {};
        b = {};
        if (resident) break;
        off.cache.publishCompositeState(off.blocks[index], std::make_shared<TestState>(100));
        on.cache.publishCompositeState(on.blocks[index], std::make_shared<TieredState>(control));
        break;
      }
      case 1: { // a request uses this prefix
        auto a = off.lookup(index);
        auto b = on.lookup(index);
        require(!a.state || b.state, "a hit without the tier missed with it");
        off.cache.recordLookup(a);
        on.cache.recordLookup(b);
        break;
      }
      default: // memory pressure recycles one buffer on each side
        static_cast<void>(off.cache.reclaimOneState());
        static_cast<void>(on.cache.reclaimOneState());
        static_cast<void>(on.cache.pollTransfers());
      }
      require(on.cache.snapshot().stateCache.bytes == off.cache.snapshot().stateCache.bytes,
              "RAM occupancy diverged from the cache without the tier");
    }
  }
}

// A -> A -> B -> C -> B with room for two states: without a disk tier only A
// is dropped for C and the next B hits. A demotion must not cost a second
// state, or the tier lowers the hit rate it is meant to raise.
void testDemotionCostsNoSecondState() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control)); // A, A
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control)); // B
  require(fixture.cache.reclaimOneState(), "no buffer was recycled for C");            // C needs a buffer
  {
    auto b = fixture.lookup(129);
    require(b.state && b.state->kvBlock() == fixture.blocks[3] && b.state->state()->residentBytes() == 100,
            "the demotion of A evicted B as well");
  }
  control->ready = true;
  require(fixture.cache.pollTransfers() && fixture.lookup(33).state &&
              !fixture.lookup(33).state->state()->residentBytes(),
          "A did not reach the disk tier");
}

void testCancelledRestoreStopsQueuedReads() {
  for (unsigned completed = 0; completed < 4; ++completed) {
    for (unsigned peerBlocks : {0u, 2u, 4u}) {
      test::TestKvTier tier;
      tier.stagingSlots = 1;
      CacheFixture fixture(&tier);
      auto control = std::make_shared<TransferControl>();
      control->ready = true;
      if (peerBlocks == 2) {
        fixture.cache.publishCompositeState(fixture.blocks[1],
                                            std::make_shared<TieredState>(control));
        require(fixture.cache.reclaimOneState() && fixture.cache.pollTransfers(),
                "shared prefix state demotion failed");
      }
      fixture.cache.publishCompositeState(fixture.blocks[3],
                                          std::make_shared<TieredState>(control));
      require(fixture.cache.reclaimOneState() && fixture.cache.pollTransfers(),
              "state demotion failed");
      for (unsigned i = 0; i < 4; ++i) {
        require(fixture.cache.reclaimOne(CacheReclaimMode::ReuseBacking).madeProgress,
                "KV demotion did not start");
        tier.complete();
        require(fixture.cache.pollTransfers(), "KV demotion did not finish");
      }
      auto lookup = fixture.lookup(129);
      fixture.cache.beginRequest(2);
      require(fixture.cache.restoreRequest(2, lookup).granted(), "restore was denied");
      if (peerBlocks) {
        auto peerLookup = fixture.lookup(peerBlocks * KvCache::pageTokens + 1);
        fixture.cache.beginRequest(3);
        require(peerLookup.state && fixture.cache.restoreRequest(3, peerLookup).granted(),
                "peer restore was denied");
      }
      for (unsigned i = 0; i < completed; ++i) {
        tier.complete();
        static_cast<void>(fixture.cache.pollTransfers());
        static_cast<void>(fixture.cache.pollTransfers());
      }
      require(tier.restores == completed + 1, "restore window did not advance");
      lookup = {};
      fixture.cache.endRequest(2);
      for (unsigned i = 0; i < 12; ++i) {
        tier.complete();
        static_cast<void>(fixture.cache.pollTransfers());
      }
      require(tier.restores == std::max(peerBlocks, completed + 1),
              "cancelled restore read unused pages or interrupted a shared restore");
      require(tier.inFlight() == 0 && fixture.cache.snapshot().kvTier.diskBlocks == 4 &&
                  fixture.lookup(129).resumeBoundary() == 128,
              "cancellation stranded transfers or discarded the disk prefix");
      if (peerBlocks) {
        require(fixture.cache.kvRestoreStatus(3) == KvRestoreStatus::None,
                "shared restore did not finish");
        fixture.cache.endRequest(3);
      }
      auto retry = fixture.lookup(129);
      fixture.cache.beginRequest(4);
      require(fixture.cache.restoreRequest(4, retry).granted(), "retry restore was denied");
      for (unsigned i = 0; i < 12; ++i) {
        tier.complete();
        static_cast<void>(fixture.cache.pollTransfers());
      }
      require(fixture.cache.kvRestoreStatus(4) == KvRestoreStatus::None && tier.restores == 4,
              "retry failed or reread already restored pages");
      retry = {};
      fixture.cache.endRequest(4);
    }
  }
}

// The KV tier through the cache. A leaf whose state went to disk is written
// rather than dropped and keeps its page until the copy has landed; the
// prefix then matches through disk, and requests wait for its restore.
void testKvDemotionAndRestoreLifecycle() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  // The state goes first and frees its RAM at once; the leaf under it is
  // then worth keeping.
  require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100 && tier.demotions == 0 &&
              fixture.cache.pollTransfers(),
          "state was not demoted ahead of its leaf");
  const auto step = fixture.cache.reclaimOne(reuse);
  require(step.madeProgress && step.reclaimedBytes == 0 && tier.demotions == 1 &&
              tier.slots == 1 && fixture.pool.freePageCount() == 0,
          "leaf under a disk state was dropped or freed before its copy landed");
  auto stats = fixture.cache.snapshot();
  require(stats.kvTier.pendingPages == 1 && stats.kvCache.blocks == 4 &&
              stats.kvTier.diskBlocks == 1,
          "pending demotion was not accounted");
  require(!fixture.cache.pollTransfers() && !fixture.cache.reclaimOne(reuse).madeProgress,
          "the chain was reclaimed past its demoting leaf");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1,
          "landed copy did not free the page");
  stats = fixture.cache.snapshot();
  require(stats.kvTier.demotions == 1 && stats.kvTier.pendingPages == 0 &&
              stats.kvCache.blocks == 3 && stats.kvTier.diskBlocks == 1 &&
              stats.kvTier.diskBytes == 100 && stats.stateCache.diskBytes == 100,
          "tier accounting after the demotion is off");
  {
    auto lookup = fixture.lookup(129);
    require(lookup.kvBoundary == 128 && lookup.state &&
                lookup.state->kvBlock() == fixture.blocks[3] &&
                !lookup.state->state()->residentBytes(),
            "prefix on disk did not match");
    fixture.cache.beginRequest(2);
    require(fixture.cache.restoreRequest(2, lookup).granted() &&
                fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::Pending &&
                tier.restores == 1 && fixture.pool.freePageCount() == 0 &&
                fixture.cache.pageTable(2).pages.size() == 4,
            "restore did not take a page for the disk-only block");
    // A second request on the same prefix waits for the same restore.
    auto again = fixture.lookup(129);
    fixture.cache.beginRequest(3);
    require(fixture.cache.restoreRequest(3, again).granted() &&
                fixture.cache.kvRestoreStatus(3) == KvRestoreStatus::Pending && tier.restores == 1,
            "a second request started its own restore");
    require(!fixture.cache.pollTransfers(), "restore finished before the tier did");
    tier.complete();
    require(fixture.cache.pollTransfers() &&
                fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::None &&
                fixture.cache.kvRestoreStatus(3) == KvRestoreStatus::None,
            "restore did not complete for both requests");
    fixture.cache.recordLookup(lookup);
    stats = fixture.cache.snapshot();
    require(stats.kvTier.restores == 1,
            "disk KV hit was not counted");
    again = {};
    lookup = {};
    fixture.cache.endRequest(2);
    fixture.cache.endRequest(3);
  }
  // Both tiers hold the block now: its next reclaim costs no write, and the
  // state on disk stays.
  require(fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 1 &&
              fixture.pool.freePageCount() == 1 &&
              fixture.cache.snapshot().kvCache.blocks == 3 &&
              fixture.cache.snapshot().stateCache.diskBytes == 100,
          "a block with a disk copy was written again or lost its state");
}

// Leaves nothing depends on are dropped, never written; a parent whose child
// went to disk is written when its turn comes, so the chain stays whole.
void testTailsDropAndParentsFollowToDisk() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  require(fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 0 &&
              fixture.pool.freePageCount() == 1,
          "a tail was written");
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[2], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100 && fixture.cache.pollTransfers(),
          "state was not demoted first");
  for (uint32_t written = 1; written <= 3; ++written) {
    require(fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == written,
            "a block the disk chain depends on was dropped");
    tier.complete();
    require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1 + written,
            "written block did not free its page");
  }
  const auto stats = fixture.cache.snapshot();
  require(stats.kvCache.blocks == 0 && stats.kvTier.diskBlocks == 3 && tier.slots == 3 &&
              stats.stateCache.diskBytes == 100,
          "chain did not move to disk whole");
  auto lookup = fixture.lookup(129);
  require(lookup.kvBoundary == 96 && lookup.state,
          "disk chain did not match up to its state");
}

// A demotion the ring cannot take right now keeps its leaf: the requester is
// told to wait, the leaf is written when the ring has room. Only a tier that
// can never write again lets the leaf go as without a tier.
// A ring that transfers in flight will free is worth waiting for: the leaf
// stays and the shortfall is Pending. A ring or a file that nothing will
// free is not: the leaf goes, exactly as it would without a tier, unless a
// disk subtree depends on it.
void testRefusedDemotionKeepsTheLeafWhileTransfersLand() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  tier.stagingSlots = 1;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100 && fixture.cache.pollTransfers(),
          "state was not demoted first");
  // One demotion takes the only staging slot and stays in flight.
  require(fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 1 &&
              tier.inFlight() == 1,
          "the first leaf was not written");
  // Its parent is no leaf while the page being written is still resident, so
  // the pass has nothing to give and says so: wait for the transfer.
  const CacheReclaimResult busy = fixture.cache.reclaimOne(reuse);
  require(!busy.madeProgress && busy.pending && tier.demotions == 1 &&
              fixture.cache.snapshot().kvCache.blocks == 4,
          "a leaf was dropped or the wait was not reported while the ring was busy");
  fixture.cache.beginRequest(2);
  require(fixture.cache.ensureTokens(2, 32).failure == KvPageAcquireFailure::Pending,
          "a request was failed while a transfer was landing");
  fixture.cache.endRequest(2);
  tier.complete();
  require(fixture.cache.pollTransfers() && tier.inFlight() == 0, "the demotion did not land");
  require(fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 2,
          "the leaf was not written once the ring had room");
  tier.complete();
  require(fixture.cache.pollTransfers(), "second leaf did not land");
}

void testUnusableTierDropsTheLeafInstead() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  tier.stagingSlots = 0;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100 && fixture.cache.pollTransfers(),
          "state was not demoted first");
  // Nothing is in flight and nothing ever frees the ring: waiting would be
  // waiting for nothing, so the leaf and its disk copy go instead.
  require(fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 0 &&
              fixture.cache.snapshot().kvCache.blocks == 3 &&
              fixture.cache.snapshot().kvTier.demotionsRefused == 1,
          "an unusable tier parked the leaf instead of dropping it");
  fixture.cache.beginRequest(2);
  require(fixture.cache.ensureTokens(2, 32).granted(),
          "a request waited although nothing was in flight");
  fixture.cache.endRequest(2);

  // A leaf a disk subtree depends on stays while that subtree is in use: a
  // lookup holds the state below it. Once the tier takes no writes and the
  // subtree is free, the leaf goes with it, as without a tier, instead of
  // holding its RAM until the server restarts.
  test::TestKvTier second;
  CacheFixture deep(&second);
  deep.cache.publishCompositeState(deep.blocks[3], std::make_shared<TieredState>(control));
  require(deep.cache.reclaimOne(reuse).reclaimedBytes == 100 && deep.cache.pollTransfers(),
          "deep state was not demoted");
  require(deep.cache.reclaimOne(reuse).madeProgress && second.demotions == 1, "leaf was not written");
  second.complete();
  require(deep.cache.pollTransfers(), "leaf did not land");
  second.writableFile = false;
  {
    auto held = deep.lookup(129);
    require(held.state && !deep.cache.reclaimOne(reuse).madeProgress &&
                deep.cache.snapshot().kvCache.blocks == 3 &&
                deep.cache.snapshot().kvTier.diskBlocks == 1,
            "a leaf was dropped under a disk subtree a lookup holds");
  }
  require(deep.cache.reclaimOne(reuse).madeProgress &&
              deep.cache.snapshot().kvCache.blocks == 2 &&
              deep.cache.snapshot().kvTier.diskBlocks == 0 &&
              deep.cache.snapshot().stateCache.entries == 0,
          "an unwritable tier kept the leaf and its disk subtree");
  deep.cache.beginRequest(2);
  require(deep.cache.ensureTokens(2, 64).granted(),
          "the dropped leaf's page did not come back");
  deep.cache.endRequest(2);
}

// The failure seen at 23G: a leaf with disk-only children whose demotion is
// refused must stay, not be erased under its children.
void testParentOfDiskChildrenSurvivesRefusal() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100 && fixture.cache.pollTransfers() &&
              fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 1,
          "leaf was not written");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "leaf did not land");
  // With the ring unusable and nothing in flight the parent still stays: its
  // disk subtree depends on it. The request is told it cannot have the pages
  // rather than told to wait for something that will never happen.
  tier.stagingSlots = 0;
  fixture.cache.beginRequest(2);
  require(fixture.cache.ensureTokens(2, 64).failure == KvPageAcquireFailure::LogicalCapacity &&
              fixture.cache.snapshot().kvCache.blocks == 3 && tier.demotions == 1,
          "the parent of a disk block was dropped, or the request was told to wait");
  tier.stagingSlots = 8;
  require(fixture.cache.ensureTokens(2, 64).failure == KvPageAcquireFailure::Pending &&
              tier.demotions == 2,
          "the parent was not written once the ring had room");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.cache.ensureTokens(2, 64).granted(),
          "pages did not return to the request");
  fixture.cache.endRequest(2);
}

// While one state write is in flight, a pressure pass keeps the next state
// in RAM rather than dropping it; the pass after the write takes it.
void testSecondStateWaitsForTheWrite() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control));
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100, "first state was not written");
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 100 && stats.diskBytes == 100 && stats.offloads == 1,
          "first write did not free its RAM");
  require(!fixture.cache.reclaimOne(reuse).madeProgress,
          "a state was dropped or a leaf under one was taken while a write was in flight");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 100 && stats.evictions == 0,
          "the waiting state did not stay");
  control->ready = true;
  require(fixture.cache.pollTransfers() && fixture.cache.reclaimOne(reuse).reclaimedBytes == 100,
          "the waiting state was not written after the first landed");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 0 && stats.diskBytes == 200 && stats.offloads == 2 &&
              control->slots == 2,
          "second write did not land beside the first");
}

// A refusal ends the scan: the ring is full for every leaf alike, so one
// attempt costs one refusal, not one per cached block.
void testRefusedRingStopsTheScan() {
  struct Prefixes {
    test::TestKvBacking backing{4, 100};
    KvPool pool{backing};
    test::TestKvTier tier;
    engine::Cache cache{pool, cacheNamespace(), &tier};
    std::array<std::vector<uint32_t>, 4> prompts;
    std::array<uint64_t, 4> blocks{};

    Prefixes() {
      for (uint32_t i = 0; i < prompts.size(); ++i) {
        prompts[i].assign(KvCache::pageTokens, 1000 + i);
        cache.beginRequest(i + 1);
        require(cache.ensureTokens(i + 1, KvCache::pageTokens).granted(), "prefix KV failed");
        blocks[i] = cache.publishCommittedBlocks(i + 1, prompts[i], KvCache::pageTokens);
        cache.endRequest(i + 1);
      }
    }
  } p;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  for (uint64_t block : p.blocks) {
    p.cache.publishCompositeState(block, std::make_shared<TieredState>(control));
    require(p.cache.reclaimOneState() && p.cache.pollTransfers(), "state was not demoted");
  }
  // One staging slot: the first leaf takes it and the request waits for that
  // page rather than evicting more.
  p.tier.stagingSlots = 1;
  p.cache.beginRequest(9);
  require(p.cache.ensureTokens(9, 32).failure == KvPageAcquireFailure::Pending &&
              p.tier.demotions == 1 && p.cache.snapshot().kvCache.blocks == 4,
          "the first leaf was not written, or a leaf was dropped");
  // A larger shortfall meets a ring that the transfer in flight holds. Every
  // leaf would answer the same, so the scan asks once and waits.
  require(p.cache.ensureTokens(9, 64).failure == KvPageAcquireFailure::Pending &&
              p.cache.snapshot().kvTier.demotionsRefused == 1 &&
              p.cache.snapshot().kvCache.blocks == 4,
          "a full ring was asked once per leaf, or a leaf was dropped");
  p.tier.complete();
  require(p.cache.pollTransfers() && p.cache.ensureTokens(9, 32).granted(),
          "the page did not return");
  p.cache.endRequest(9);
}

// A shortfall while restores are in flight is pending, not exhausted: the
// restored blocks become leaves with disk copies. Pages held by an active
// request are exhausted for real.
void testRestoresInFlightMakeAShortfallPending() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100 && fixture.cache.pollTransfers() &&
              fixture.cache.reclaimOne(reuse).madeProgress,
          "leaf was not written");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "leaf did not land");
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(2);
  require(fixture.cache.restoreRequest(2, lookup).granted() && fixture.pool.freePageCount() == 0,
          "restore did not take the free page");
  fixture.cache.beginRequest(3);
  require(fixture.cache.ensureTokens(3, 32).failure == KvPageAcquireFailure::Pending,
          "a shortfall during a restore was reported as exhausted");
  tier.complete();
  require(fixture.cache.pollTransfers() &&
              fixture.cache.ensureTokens(3, 32).failure == KvPageAcquireFailure::LogicalCapacity,
          "pages held by an active request were not exhausted");
  lookup = {};
  fixture.cache.endRequest(2);
  require(fixture.cache.ensureTokens(3, 32).granted() && tier.demotions == 1,
          "the restored leaf did not give up its page for nothing");
  fixture.cache.endRequest(3);
}

// A full quota replaces the oldest redundant copy first, then the oldest
// disk-only leaf; a block whose copy was replaced stays in RAM.
void testDiskReplacementOrder() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  struct Prefixes {
    test::TestKvBacking backing{4, 100};
    KvPool pool{backing};
    test::TestKvTier tier;
    engine::Cache cache{pool, cacheNamespace(), &tier};
    std::array<std::vector<uint32_t>, 4> prompts;
    std::array<uint64_t, 4> blocks{};

    Prefixes() {
      tier.capacity = 1;
      for (uint32_t i = 0; i < prompts.size(); ++i) {
        prompts[i].assign(KvCache::pageTokens, 1000 + i);
        cache.beginRequest(i + 1);
        require(cache.ensureTokens(i + 1, KvCache::pageTokens).granted(), "prefix KV failed");
        blocks[i] = cache.publishCommittedBlocks(i + 1, prompts[i], KvCache::pageTokens);
        cache.endRequest(i + 1);
        prompts[i].push_back(9999);
      }
    }
  } p;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  for (uint64_t block : p.blocks)
    p.cache.publishCompositeState(block, std::make_shared<TieredState>(control));
  // The oldest state goes to disk, then the leaf under it.
  const auto demoteNext = [&] {
    require(p.cache.reclaimOne(reuse).reclaimedBytes == 100 && p.cache.pollTransfers(),
            "state was not demoted first");
    require(p.cache.reclaimOne(reuse).madeProgress, "leaf under a disk state was not reclaimed");
  };
  // A goes to disk and comes back: RAM and disk both hold it.
  demoteNext();
  require(p.tier.demotions == 1, "A was not written");
  p.tier.complete();
  require(p.cache.pollTransfers() && p.pool.freePageCount() == 1, "A did not free its page");
  {
    auto lookup = p.cache.lookup(p.prompts[0]);
    p.cache.beginRequest(9);
    require(p.cache.restoreRequest(9, lookup).granted(),
            "A did not restore");
    p.tier.complete();
    require(p.cache.pollTransfers(), "A's restore did not finish");
    lookup = {};
    p.cache.endRequest(9);
  }
  // B needs the one slot: A's redundant copy goes, A stays resident.
  demoteNext();
  require(p.tier.demotions == 2 && p.tier.slots == 1 &&
              p.cache.snapshot().kvCache.blocks == 4 && p.cache.snapshot().kvTier.diskBlocks == 1,
          "B did not replace A's redundant copy");
  p.tier.complete();
  require(p.cache.pollTransfers() && p.pool.freePageCount() == 1, "B did not free its page");
  // C needs the slot: no redundant copy is left, so the oldest disk-only
  // leaf, B, leaves with its state.
  demoteNext();
  require(p.tier.demotions == 3 && p.tier.slots == 1 &&
              p.cache.lookup(p.prompts[1]).kvBoundary == 0 && control->slots == 2,
          "C did not replace the oldest disk-only leaf");
}

// Pages already on their way back gate allocation: the request is told to
// wait instead of the cache demoting more than it needs.
void testPendingPagesGateAllocation() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  fixture.cache.beginRequest(2);
  require(fixture.cache.ensureTokens(2, 32).failure == KvPageAcquireFailure::Pending &&
              tier.demotions == 1 && fixture.cache.snapshot().kvCache.blocks == 4 &&
              fixture.cache.snapshot().kvTier.pendingPages == 1,
          "allocation evicted past the page on its way back");
  require(fixture.cache.ensureTokens(2, 32).failure == KvPageAcquireFailure::Pending &&
              tier.demotions == 1,
          "a retry before the copy landed demoted more");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.cache.ensureTokens(2, 32).granted() &&
              fixture.cache.pageTable(2).pages.size() == 1 &&
              fixture.cache.snapshot().kvCache.blocks == 3,
          "pages did not return to the waiting request");
  fixture.cache.endRequest(2);
}

// A demotion that fails leaves the block in RAM without a copy; a restore
// that fails removes the block and its state once its requests let go, and
// the next lookup matches the shallower prefix.
void testTransferFailures() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  {
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
    require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100 && fixture.cache.pollTransfers() &&
                fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 1,
            "no demotion");
    tier.complete(false);
    require(fixture.cache.pollTransfers(), "failed write was not consumed");
    const auto stats = fixture.cache.snapshot();
    require(stats.kvTier.demotionFailures == 1 && stats.kvTier.pendingPages == 0 &&
                stats.kvCache.blocks == 4 && stats.kvTier.diskBlocks == 0 && tier.slots == 0 &&
                stats.stateCache.diskBytes == 100,
            "failed write lost the page, kept the slot, or touched the state");
  }
  {
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
    require(fixture.cache.reclaimOne(reuse).reclaimedBytes == 100 && fixture.cache.pollTransfers() &&
                fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 1,
            "no demotion");
    tier.complete();
    require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "no disk block");
    auto lookup = fixture.lookup(129);
    fixture.cache.beginRequest(2);
    require(fixture.cache.restoreRequest(2, lookup).granted(), "restore was denied");
    tier.complete(false);
    require(fixture.cache.pollTransfers() &&
                fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::Failed &&
                fixture.cache.snapshot().kvTier.restoreFailures == 1,
            "failed read was not reported to the request");
    lookup = {};
    fixture.cache.endRequest(2);
    const auto stats = fixture.cache.snapshot();
    require(fixture.lookup(129).kvBoundary == 96 && fixture.pool.freePageCount() == 1 &&
                stats.kvCache.blocks == 3 && stats.kvTier.diskBlocks == 0 &&
                stats.stateCache.entries == 0 && tier.slots == 0,
            "poisoned block or its state survived its last user");
  }
}

// Pressure reclaim counts pages in flight toward its target instead of
// writing the whole chain at once.
void testReclaimCacheCountsPendingPages() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimCache(100, false) == 100 && tier.demotions == 0,
          "the state's RAM did not satisfy the first target");
  require(fixture.cache.pollTransfers(), "state write was not consumed");
  require(fixture.cache.reclaimCache(100, false) == 0 && tier.demotions == 1 &&
              fixture.cache.snapshot().kvCache.blocks == 4,
          "reclaim wrote more than the target while a page was on its way back");
}

void testBusyRingPreservesDiskVictim() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  for (const bool restored : {false, true}) {
    test::TestKvBacking backing{4, 100};
    KvPool pool{backing};
    test::TestKvTier tier;
    tier.capacity = 2;
    tier.stagingSlots = 1;
    engine::Cache cache{pool, cacheNamespace(), &tier};
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    control->capacity = 4;
    std::array<std::vector<uint32_t>, 4> prompts;
    for (uint32_t i = 0; i < prompts.size(); ++i) {
      prompts[i].assign(KvCache::pageTokens, 1000 + i);
      cache.beginRequest(i + 1);
      require(cache.ensureTokens(i + 1, KvCache::pageTokens).granted(), "prefix KV failed");
      auto block = cache.publishCommittedBlocks(i + 1, prompts[i], KvCache::pageTokens);
      cache.endRequest(i + 1);
      cache.publishCompositeState(block, std::make_shared<TieredState>(control));
      require(cache.reclaimOneState() && cache.pollTransfers(), "state was not demoted");
      prompts[i].push_back(9999);
    }
    require(cache.reclaimOne(reuse).madeProgress, "first demotion failed");
    tier.complete();
    require(cache.pollTransfers(), "first demotion did not finish");
    if (restored) {
      auto lookup = cache.lookup(prompts[0]);
      cache.beginRequest(9);
      require(cache.restoreRequest(9, lookup).granted(), "disk prefix did not restore");
      tier.complete();
      require(cache.pollTransfers(), "restore did not finish");
      lookup = {};
      cache.endRequest(9);
    }
    require(cache.reclaimOne(reuse).madeProgress && tier.slots == 2,
            "second demotion did not fill quota");
    require(!cache.reclaimOne(reuse).madeProgress, "busy ring did not wait");
    require(tier.slots == 2 && cache.snapshot().kvTier.diskBlocks == 2 &&
                cache.lookup(prompts[0]).kvBoundary == KvCache::pageTokens,
            "busy staging ring discarded a disk prefix without starting a write");
    tier.complete();
    require(cache.pollTransfers() && cache.reclaimOne(reuse).madeProgress &&
                tier.demotions == 3,
            "disk replacement did not resume after staging became available");
    tier.complete();
    require(cache.pollTransfers(), "resumed demotion did not finish");
  }
}

// A restore takes its pages before it pins its chain, and the chain's last
// resident block has only disk children, which makes it a leaf. The eviction
// that makes room must not take it: the restore adopts pages under it.
void testRestoreKeepsTheBlockItExtends() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(reuse).madeProgress && fixture.cache.pollTransfers(),
          "state was not demoted");
  for (int demoted = 0; demoted < 2; ++demoted) {
    require(fixture.cache.reclaimOne(reuse).madeProgress, "leaf was not demoted");
    tier.complete();
    require(fixture.cache.pollTransfers(), "demotion did not land");
  }
  require(fixture.pool.freePageCount() == 2, "the last two blocks did not go to disk");
  {
    auto lookup = fixture.lookup(129);
    fixture.cache.beginRequest(2);
    require(fixture.cache.restoreRequest(2, lookup).granted(), "first restore denied");
    tier.complete();
    require(fixture.cache.pollTransfers() &&
                fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::None,
            "first restore did not land");
    lookup = {};
    fixture.cache.endRequest(2);
  }
  // The last block gives up its page again; the one before it keeps a disk
  // copy and has only a disk child.
  require(fixture.cache.reclaimOne(reuse).madeProgress && fixture.pool.freePageCount() == 1,
          "the restored leaf did not drop its page");
  fixture.cache.beginRequest(3);
  require(fixture.cache.ensureTokens(3, 32).granted() && fixture.pool.freePageCount() == 0,
          "another request did not take the free page");
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(4);
  const TokenAdmission refused = fixture.cache.restoreRequest(4, lookup);
  require(!refused.granted() && refused.failure == KvPageAcquireFailure::LogicalCapacity &&
              fixture.cache.pageTable(4).pages.empty(),
          "a restore without a page did not report the shortfall");
  fixture.cache.endRequest(3);
  require(fixture.cache.restoreRequest(4, lookup).granted(), "the retried restore was denied");
  const auto table = fixture.cache.pageTable(4);
  require(table.pages.size() == 4 &&
              std::none_of(table.pages.begin(), table.pages.end(),
                           [](uint32_t page) { return page == KvCache::noPage; }),
          "the restored chain lost a page");
  tier.complete();
  require(fixture.cache.pollTransfers() &&
              fixture.cache.kvRestoreStatus(4) == KvRestoreStatus::None,
          "the retried restore did not land");
  lookup = {};
  fixture.cache.endRequest(4);
}

// A state published in RAM while its block's demotion is in flight keeps the
// block's page when the write lands: a state in RAM sits on resident KV.
void testDemotionKeepsThePageUnderANewState() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(reuse).madeProgress && fixture.cache.pollTransfers(),
          "state was not demoted");
  require(fixture.cache.reclaimOne(reuse).madeProgress && tier.demotions == 1,
          "the leaf's demotion did not start");
  fixture.publish(3);
  require(fixture.cache.snapshot().stateCache.bytes == 100, "the new state is not in RAM");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 0,
          "the landed demotion dropped the page under a state in RAM");
  auto lookup = fixture.lookup(129);
  require(lookup.kvBoundary == 128 && lookup.state && lookup.state->state()->residentBytes() == 100,
          "the prefix under the new state is not resident");
}

// A prefill through blocks another request is restoring keeps its own pages
// and may publish a state in RAM on one of them. When the restorer is
// cancelled before that block's read has started, the read goes on: a state
// in RAM sits on resident KV. Its later write into a full quota gives up
// other copies, never the block under it.
void testCancelledRestoreKeepsThePageUnderANewState() {
  constexpr auto reuse = CacheReclaimMode::ReuseBacking;
  test::TestKvBacking backing{8, 100};
  KvPool pool{backing};
  test::TestKvTier tier;
  engine::Cache cache{pool, cacheNamespace(), &tier};
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  std::vector<uint32_t> prompt(129);
  for (uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = 1000 + i;
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 128).granted(), "prefix KV failed");
  const uint64_t last = cache.publishCommittedBlocks(1, prompt, 128);
  const uint64_t third = cache.blockAt(1, 96);
  cache.endRequest(1);
  cache.publishCompositeState(last, std::make_shared<TieredState>(control));
  require(cache.reclaimOneState() && cache.pollTransfers(), "state was not demoted");
  for (int i = 0; i < 3; ++i) {
    require(cache.reclaimOne(reuse).madeProgress, "KV demotion did not start");
    tier.complete();
    require(cache.pollTransfers(), "KV demotion did not finish");
  }
  // The restore reads the second block; the last two wait for staging.
  tier.stagingSlots = 1;
  auto lookup = cache.lookup(prompt);
  cache.beginRequest(2);
  require(cache.restoreRequest(2, lookup).granted() && tier.restores == 1,
          "restore was denied");
  cache.beginRequest(3);
  require(cache.ensureTokens(3, 97).granted() &&
              cache.publishCommittedBlocks(3, prompt, 96) == third,
          "the prefill did not reach the block being restored");
  cache.publishCompositeState(third, std::make_shared<TieredState>(control));
  cache.endRequest(3);
  lookup = {};
  cache.endRequest(2);
  for (int i = 0; i < 2; ++i) {
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  require(tier.restores == 2 && tier.inFlight() == 0 && cache.snapshot().kvCache.blocks == 3,
          "the cancelled restore dropped the page under a state in RAM");
  control->capacity = control->slots;
  require(cache.reclaimOne(reuse).reclaimedBytes == 100 && cache.pollTransfers(),
          "the state in RAM was not written");
  lookup = cache.lookup(std::span<const uint32_t>(prompt).first(97));
  require(lookup.kvBoundary == 96 && lookup.state && lookup.state->kvBlock() == third &&
              !lookup.state->state()->residentBytes(),
          "the written state lost its block");
}

void testLargeSharedDiskRestore() {
  constexpr uint32_t pages = 4096;
  constexpr uint32_t tokens = pages * KvCache::pageTokens;
  test::TestKvBacking backing{pages, 100};
  KvPool pool{backing};
  test::TestKvTier tier;
  tier.capacity = pages;
  tier.stagingSlots = 96;
  engine::Cache cache{pool, cacheNamespace(), &tier};
  std::vector<uint32_t> prompt(tokens, 17);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, tokens).granted(), "large prefix admission failed");
  const auto boundary = cache.publishCommittedBlocks(1, prompt, tokens);
  cache.endRequest(1);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  cache.publishCompositeState(boundary, std::make_shared<TieredState>(control));
  require(cache.reclaimOneState() && cache.pollTransfers(), "large state demotion failed");
  for (uint32_t i = 0; i < pages; ++i) {
    require(cache.reclaimOne(CacheReclaimMode::ReuseBacking).madeProgress,
            "large prefix KV demotion did not progress");
    tier.complete();
    require(cache.pollTransfers(), "large prefix KV demotion did not finish");
  }
  require(pool.freePageCount() == pages && tier.demotions == pages,
          "large prefix was not fully on disk");
  prompt.push_back(18);
  auto lookup = cache.lookup(prompt);
  require(lookup.resumeBoundary() == tokens,
          "large disk prefix lookup lost its endpoint");
  for (uint64_t id = 2; id <= 5; ++id) {
    cache.beginRequest(id);
    require(cache.restoreRequest(id, lookup).granted(), "large shared restore denied");
  }
  require(tier.restores == tier.stagingSlots,
          "shared restore exceeded the transfer window");
  lookup = {};
  for (uint64_t id = 2; id < 5; ++id) cache.endRequest(id);
  for (uint32_t i = 0; i < pages && cache.kvRestoreStatus(5) == KvRestoreStatus::Pending; ++i) {
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  require(cache.kvRestoreStatus(5) == KvRestoreStatus::None &&
              tier.restores == pages && tier.inFlight() == 0,
          "large shared restore stalled or reread pages after peer cancellation");
  cache.endRequest(5);
  const auto stats = cache.snapshot();
  require(stats.activeRequests == 0 && stats.stateCache.pinned == 0 &&
              stats.kvTier.pendingPages == 0,
          "large shared restore retained a request, state pin, or transfer");
}
int main() {
  try {
    testLargeSharedDiskRestore();
    testRestoreKeepsTheBlockItExtends();
    testDemotionKeepsThePageUnderANewState();
    testCancelledRestoreKeepsThePageUnderANewState();
    testBusyRingPreservesDiskVictim();
    testCancelledRestoreStopsQueuedReads();
    testDiskCheckpointRamAccounting();
    testOrdinaryPublicationUpgradesDiskCheckpoint();
    testKvDemotionAndRestoreLifecycle();
    testTailsDropAndParentsFollowToDisk();
    testRefusedDemotionKeepsTheLeafWhileTransfersLand();
    testUnusableTierDropsTheLeafInstead();
    testSecondStateWaitsForTheWrite();
    testRefusedRingStopsTheScan();
    testRestoresInFlightMakeAShortfallPending();
    testParentOfDiskChildrenSurvivesRefusal();
    testDiskReplacementOrder();
    testPendingPagesGateAllocation();
    testTransferFailures();
    testReclaimCacheCountsPendingPages();
    testDemotionCostsNoSecondState();
    testPromotionIdentityAndDenial();
    testRepublicationKeepsTheDiskCopy();
    testLostStatesAreCounted();
    testRestoredStateKeepsItsDiskCopy();
    testDiskReplacementSpansStatesAndKv();
    testDiskPublicationLifecycle();
    testDiskPublicationFailure();
    testFailedWriteUnderALookup();
    testFailedWriteIsCountedAfterItsEntryLeft();
    testDiskPublicationMakesRoom();
    testRollingCheckpointsUseTheTier();
    testDiskQuotaReplacesByRecency();
    testInvalidationDuringOffload();
    testDemotionFreesTheBufferAtOnce();
    testTierOnlyAddsToTierOff();
    testDiskPromotionAndInvalidation();
    testTieredStateLifecycle();
    testTieredWriteReuseAndFailure();
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
