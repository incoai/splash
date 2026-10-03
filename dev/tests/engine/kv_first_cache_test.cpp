#include "Checked.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Cache.hpp"
#include "model/SlotFile.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

template <typename Error, typename Function>
void requireThrows(Function &&function, const char *message) {
  try {
    function();
  } catch (const Error &) {
    return;
  }
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

// Admits pages the way the engine does: each denial makes room in one
// reclaim step, until the pages fit, a transfer in flight holds what they
// need, or nothing more can be reclaimed.
TokenAdmission admitLikeEngine(engine::Cache &cache,
                               const std::function<TokenAdmission()> &attempt) {
  TokenAdmission admission = attempt();
  while (admission.failure == TokenAdmissionFailure::Denied) {
    const CacheReclaimResult step =
        cache.reclaimForPages(admission.additionalPages, CacheReclaimMode::KeepExtents,
                              ReclaimClass::InUse);
    if (!step.madeProgress) {
      if (step.pending)
        admission.failure = TokenAdmissionFailure::Pending;
      break;
    }
    admission = attempt();
  }
  return admission;
}

TokenAdmission admitTokens(engine::Cache &cache, uint64_t requestId, uint64_t tokens) {
  return admitLikeEngine(cache, [&] { return cache.ensureTokens(requestId, tokens); });
}

TokenAdmission admitRestore(engine::Cache &cache, uint64_t requestId,
                            const CacheLookup &lookup) {
  return admitLikeEngine(cache, [&] { return cache.restoreRequest(requestId, lookup); });
}

// One reclaim step that may take every class; returns the bytes of state
// RAM it let go of, which a step does not count as released.
uint64_t reclaimStateBytes(engine::Cache &cache, CacheReclaimMode mode) {
  const uint64_t before = cache.snapshot().stateCache.bytes;
  require(cache.reclaimOne(mode, ReclaimClass::InUse).madeProgress,
          "the reclaim step made no progress");
  return before - cache.snapshot().stateCache.bytes;
}

// Pressure-pass steps until none makes progress.
void reclaimEverything(engine::Cache &cache, bool keepResumePoint) {
  while (cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse, keepResumePoint)
             .madeProgress) {
  }
}

// Four pages, all allocated, and a budget of no more.
struct CacheFixture {
  test::TestKvStorage storage{5, 100, 1};
  KvPool pool{storage, 4};
  engine::Cache cache;
  std::vector<uint32_t> prompt;
  std::vector<uint64_t> blocks;

  // With running, the request that cached the chain stays unfinished, so
  // none of its KV is a victim, as for a request publishing its states.
  explicit CacheFixture(engine::KvTier *tier = nullptr, bool running = false)
      : cache(pool, tier, nullptr) {
    storage.budgetPages = 4;
    for (uint32_t block = 0; block < 4; ++block) {
      for (uint32_t row = 0; row < KvCache::pageTokens; ++row)
        prompt.push_back(1000 + block * 100 + row);
    }
    prompt.push_back(9999);
    cache.beginRequest(1);
    require(admitTokens(cache, 1, 128).granted(),
            "fixture KV pages were not acquired");
    cache.publishCommittedBlocks(1, prompt, 128, {});
    for (uint32_t boundary = 32; boundary <= 128; boundary += 32)
      blocks.push_back(cache.blockAt(1, boundary));
    if (!running)
      cache.endRequest(1);
  }

  void publish(uint32_t block, uint64_t bytes = 100) {
    cache.publishCompositeState(blocks.at(block),
                                std::make_shared<TestState>(bytes), false);
  }

  engine::CacheLookup lookup(uint32_t tokens) {
    return cache.lookup(std::span<const uint32_t>(prompt).first(tokens), {});
  }
};

// Four one-page prefixes, each cached by a request of its own, on four pages,
// all allocated, and a budget of no more. With continued, each prompt runs
// one token past its page, so a lookup can match the page.
struct Prefixes {
  test::TestKvStorage storage{5, 100, 1};
  KvPool pool{storage, 4};
  engine::Cache cache;
  std::array<std::vector<uint32_t>, 4> prompts;
  std::array<uint64_t, 4> blocks{};

  explicit Prefixes(test::TestKvTier *tier = nullptr, bool continued = true)
      : cache(pool, tier, nullptr) {
    storage.budgetPages = 4;
    for (uint32_t i = 0; i < prompts.size(); ++i) {
      prompts[i].assign(KvCache::pageTokens, 1000 + i);
      cache.beginRequest(i + 1);
      require(admitTokens(cache, i + 1, KvCache::pageTokens).granted(), "prefix KV failed");
      blocks[i] = test::publishBlocks(cache, i + 1, prompts[i], KvCache::pageTokens);
      cache.endRequest(i + 1);
      if (continued)
        prompts[i].push_back(9999);
    }
  }
  engine::CacheLookup lookup(uint32_t index) { return cache.lookup(prompts[index], {}); }
};

// Demotes the oldest leaves one at a time, each copy landing before the next
// starts.
void demoteLeaves(engine::Cache &cache, test::TestKvTier &tier, uint32_t leaves) {
  for (uint32_t i = 0; i < leaves; ++i) {
    require(cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress,
            "KV demotion did not start");
    tier.complete();
    require(cache.pollTransfers(), "KV demotion did not finish");
  }
}

void testSchedulingProbeDoesNotChangeCachePolicy() {
  CacheFixture fixture;
  const auto cachedTokens = [&](std::span<const uint32_t> prompt) {
    return fixture.cache.probe(prompt, {}).cachedTokens();
  };
  require(cachedTokens(fixture.prompt) == 0,
          "KV without recurrent state was counted as reusable work");
  fixture.publish(0);
  fixture.publish(3);
  const auto prefix = std::span<const uint32_t>(fixture.prompt).first(33);
  require(cachedTokens(prefix) == 32 && cachedTokens(fixture.prompt) == 128 &&
              fixture.cache.snapshot().stateCache.pinned == 0 &&
              fixture.cache.snapshot().lookup.kvHitTokens == 0,
          "scheduling probe pinned a lease or counted a cache hit");
  require(fixture.cache.reclaimOneState(false, 0, false) && cachedTokens(prefix) == 0 &&
              cachedTokens(fixture.prompt) == 128,
          "scheduling probe refreshed the oldest state's eviction order");
}

void testValidAdmissionProbePreservesLookupAndAccounting() {
  CacheFixture fixture;
  fixture.publish(0);
  fixture.publish(2);
  const CacheProbe probe = fixture.cache.probe(fixture.prompt, {});
  const auto before = fixture.cache.snapshot();
  require(probe.cachedTokens() == 96 && before.stateCache.pinned == 0 &&
              before.lookup.kvHitTokens == 0,
          "admission probe changed cache ownership or accounting");
  auto lookup = fixture.cache.lookup(fixture.prompt, {}, &probe);
  require(lookup.kvBoundary == 128 && lookup.resumeBoundary() == 96 &&
              lookup.junctionBoundary == 0,
          "valid admission probe lost the deepest state or KV tail");
  fixture.cache.recordLookup(lookup);
  const auto after = fixture.cache.snapshot();
  require(after.lookup.kvHitTokens == 128 && after.lookup.lazyJunctions == 0 &&
              after.stateCache.pinned == 1,
          "admission probe changed lookup accounting or lease ownership");
}

// A probe kept across admission passes hashes nothing while the KV graph
// stays as it was, follows the states regardless, and after a change hashes
// only from its first block that no longer matches.
void testProbeRefreshFollowsTheGraph() {
  {
    CacheFixture fixture;
    const auto hashed = [&] { return fixture.cache.snapshot().lookup.probeHashedBlocks; };
    fixture.publish(1);
    CacheProbe probe = fixture.cache.probe(fixture.prompt, {});
    require(probe.cachedTokens() == 64 && hashed() == 4, "the probe missed its state");
    fixture.publish(3);
    fixture.cache.refresh(probe, fixture.prompt, {});
    require(probe.cachedTokens() == 128 && hashed() == 4,
            "a refresh missed a new state or hashed an unchanged graph");
  }
  {
    CacheFixture fixture;
    engine::Cache &cache = fixture.cache;
    const auto hashed = [&] { return cache.snapshot().lookup.probeHashedBlocks; };
    fixture.publish(2);
    CacheProbe probe = cache.probe(fixture.prompt, {});
    require(probe.cachedTokens() == 96 &&
                cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse)
                    .madeProgress &&
                cache.snapshot().pool.pagesPrefix == 3,
            "KV eviction fixture did not evict the cached tail");
    cache.refresh(probe, fixture.prompt, {});
    auto probed = cache.lookup(fixture.prompt, {}, &probe);
    auto fresh = cache.lookup(fixture.prompt, {});
    require(probe.cachedTokens() == 96 && hashed() == 5 && probed.kvBoundary == 96 &&
                probed.kvBoundary == fresh.kvBoundary &&
                probed.resumeBoundary() == fresh.resumeBoundary(),
            "a refresh kept an evicted block or re-hashed the blocks that stayed");
    fresh = {};
    // The tail comes back, written by a request that continues the chain.
    cache.beginRequest(2);
    require(admitRestore(cache, 2, probed).granted() && admitTokens(cache, 2, 128).granted(),
            "the tail's page was not acquired");
    probed = {};
    cache.publishCommittedBlocks(2, fixture.prompt, 128, {});
    cache.endRequest(2);
    cache.refresh(probe, fixture.prompt, {});
    require(hashed() == 6 && cache.lookup(fixture.prompt, {}, &probe).kvBoundary == 128,
            "a refresh hashed more than the page that came back");
  }
  {
    // The chain's middle block fails its restore while the block below it
    // comes back: the chain ends above the poisoned block.
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    engine::Cache &cache = fixture.cache;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
    require(cache.reclaimOneState(false, 0, true) && cache.pollTransfers(),
            "state was not written");
    demoteLeaves(cache, tier, 2);
    CacheProbe probe = cache.probe(fixture.prompt, {});
    require(probe.cachedTokens() == 128, "the probe missed the state on disk");
    tier.transferLimit = 1;
    auto lookup = cache.lookup(fixture.prompt, {});
    cache.beginRequest(2);
    require(admitRestore(cache, 2, lookup).granted() && tier.restores == 1,
            "restore was denied");
    tier.complete(false);
    static_cast<void>(cache.pollTransfers());
    tier.complete();
    require(cache.pollTransfers() && tier.restores == 2 &&
                cache.kvRestoreStatus(2) == KvRestoreStatus::Failed,
            "the failed read was not reported");
    const uint64_t before = cache.snapshot().lookup.probeHashedBlocks;
    cache.refresh(probe, fixture.prompt, {});
    require(probe.cachedTokens() == 0 &&
                cache.snapshot().lookup.probeHashedBlocks == before + 1 &&
                cache.lookup(fixture.prompt, {}, &probe).kvBoundary == 64,
            "a refresh matched through a poisoned block");
    lookup = {};
    cache.endRequest(2);
  }
}

void testProbeRechecksStateChanges() {
  CacheFixture fixture;
  fixture.publish(3);
  fixture.publish(0);
  const CacheProbe probe = fixture.cache.probe(fixture.prompt, {});
  require(probe.cachedTokens() == 128 && fixture.cache.reclaimOneState(false, 0, false),
          "state eviction fixture did not remove the deepest state");
  auto lookup = fixture.cache.lookup(fixture.prompt, {}, &probe);
  require(lookup.kvBoundary == 128 && lookup.resumeBoundary() == 32,
          "admission probe reused an evicted state");
  lookup.state.reset();

  CacheFixture published;
  const CacheProbe cold = published.cache.probe(published.prompt, {});
  published.publish(2);
  auto newlyPublished = published.cache.lookup(published.prompt, {}, &cold);
  require(cold.cachedTokens() == 0 && newlyPublished.resumeBoundary() == 96,
          "admission probe missed a state published after preview");
}

void testProbeFallsBackWhenKvChanges() {
  CacheFixture fixture;
  fixture.publish(0);
  const CacheProbe probe = fixture.cache.probe(fixture.prompt, {});
  require(
      fixture.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress &&
          fixture.cache.snapshot().pool.pagesPrefix == 3,
      "KV eviction fixture did not evict the cached tail");
  auto lookup = fixture.cache.lookup(fixture.prompt, {}, &probe);
  require(lookup.kvBoundary == 96 && lookup.resumeBoundary() == 32,
          "admission probe reused an evicted KV block");

  test::TestKvStorage storage{1, 100, 1};
  KvPool pool{storage, 1};
  engine::Cache cache{pool, nullptr, nullptr};
  std::vector<uint32_t> prompt(33, 77);
  const CacheProbe cold = cache.probe(prompt, {});
  require(cold.cachedTokens() == 0, "cold admission probe found cached work");
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 32).granted(), "new KV page was not acquired");
  const uint64_t block = test::publishBlocks(cache, 1, prompt, 32);
  cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
  cache.endRequest(1);
  auto newlyCached = cache.lookup(prompt, {}, &cold);
  require(newlyCached.kvBoundary == 32 &&
              newlyCached.resumeBoundary() == 32,
          "cold probe hid a prefix published after preview");
}

void testProbeBindsImageIdentity() {
  test::TestKvStorage storage{1, 100, 1};
  KvPool pool{storage, 1};
  engine::Cache cache{pool, nullptr, nullptr};
  std::vector<uint32_t> prompt(33, 77);
  const ImageSpan image{0, 32, 1, 1, 101, 202};
  const std::span<const ImageSpan> images(&image, 1);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 32).granted(), "image KV page was not acquired");
  const uint64_t block = test::publishBlocks(cache, 1, prompt, 32, images);
  cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
  cache.endRequest(1);
  const CacheProbe probe = cache.probe(prompt, images);
  const auto valid = cache.lookup(prompt, images, &probe);
  require(valid.kvBoundary == 32 && valid.resumeBoundary() == 32,
          "matching image probe lost its cached prefix");
}

void testProbeCannotCrossCaches() {
  test::TestKvStorage firstStorage{1, 100, 1};
  test::TestKvStorage secondStorage{1, 100, 1};
  KvPool firstPool{firstStorage, 1};
  KvPool secondPool{secondStorage, 1};
  engine::Cache first{firstPool, nullptr, nullptr};
  engine::Cache second{secondPool, nullptr, nullptr};
  const std::vector<uint32_t> firstPrompt(33, 11);
  const std::vector<uint32_t> secondPrompt(33, 22);
  const auto populate = [](engine::Cache &cache,
                           const std::vector<uint32_t> &prompt) {
    cache.beginRequest(1);
    require(admitTokens(cache, 1, 32).granted(), "KV page was not acquired");
    const uint64_t block = test::publishBlocks(cache, 1, prompt, 32);
    cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
    cache.endRequest(1);
  };
  populate(first, firstPrompt);
  populate(second, secondPrompt);
  CacheProbe probe = first.probe(firstPrompt, {});
  const auto lookup = second.lookup(firstPrompt, {}, &probe);
  require(lookup.kvBoundary == 0 && lookup.resumeBoundary() == 0,
          "a probe from another cache reused a colliding block id");
  bool refused = false;
  try {
    second.refresh(probe, firstPrompt, {});
  } catch (const std::logic_error &) {
    refused = true;
  }
  require(refused, "a probe from another cache was refreshed");
}

// Every change of a request's page list moves its revision by one and says
// from which page on the list differs: an append from the length it grew
// from, a written block swapped for the cached page at its index, a restore
// from the start. A list that stays the same keeps its revision.
void testPageTableReportsWhatChanged() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 8};
  engine::Cache cache(pool, nullptr, nullptr);
  constexpr uint32_t page = KvCache::pageTokens;
  std::vector<uint32_t> prompt;
  for (uint32_t token = 0; token <= 2 * page; ++token)
    prompt.push_back(1000 + token);

  cache.beginRequest(1);
  require(admitTokens(cache, 1, page).granted() && cache.pageTable(1).revision == 1 &&
              cache.pageTable(1).firstChanged == 0,
          "a request's first page did not start its revision");
  require(admitTokens(cache, 1, 2 * page).granted() && cache.pageTable(1).revision == 2 &&
              cache.pageTable(1).firstChanged == 1 && cache.pageTable(1).pages.size() == 2,
          "an append did not report the length it grew from");
  require(admitTokens(cache, 1, page + 1).granted(), "held pages were denied");
  cache.publishCommittedBlocks(1, prompt, 2 * page, {});
  require(cache.pageTable(1).revision == 2,
          "tokens on held pages or a request's own blocks moved its revision");
  cache.publishCompositeState(cache.blockAt(1, 2 * page), std::make_shared<TestState>(100), false);

  // A second request writes the same blocks into pages of its own, and
  // publishing each swaps it for the cached page.
  cache.beginRequest(2);
  require(admitTokens(cache, 2, 2 * page).granted() && cache.pageTable(2).revision == 1,
          "a second request's pages were denied");
  cache.publishCommittedBlocks(2, prompt, page, {});
  require(cache.pageTable(2).revision == 2 && cache.pageTable(2).firstChanged == 0 &&
              cache.pageTable(2).pages[0] == cache.pageTable(1).pages[0],
          "the first swapped block did not report its index");
  cache.publishCommittedBlocks(2, prompt, 2 * page, {});
  require(cache.pageTable(2).revision == 3 && cache.pageTable(2).firstChanged == 1 &&
              cache.pageTable(2).pages[1] == cache.pageTable(1).pages[1] &&
              cache.pageTable(1).revision == 2,
          "a swapped block did not report its index or moved the writer's revision");
  cache.endRequest(2);
  cache.endRequest(1);

  cache.beginRequest(3);
  require(admitRestore(cache, 3, cache.lookup(prompt, {})).granted() &&
              cache.pageTable(3).revision == 1 && cache.pageTable(3).firstChanged == 0 &&
              cache.pageTable(3).pages.size() == 2,
          "a restore did not report its whole list");
  require(admitTokens(cache, 3, prompt.size()).granted() &&
              cache.pageTable(3).revision == 2 && cache.pageTable(3).firstChanged == 2,
          "an append after a restore did not report the restored length");
  cache.endRequest(3);
}

void testCacheLookupAndOneTokenReplay() {
  CacheFixture fixture;
  fixture.publish(0);
  fixture.publish(2);

  auto full = fixture.cache.lookup(fixture.prompt, {});
  require(full.kvBoundary == 128 && full.resumeBoundary() == 96 &&
              full.junctionBoundary == 0 && full.state &&
              full.state->kvBlock() == fixture.blocks[2],
          "KV-first lookup did not coordinate dense KV and sparse state");

  full.state.reset();
  auto exactEdge = fixture.lookup(128);
  require(exactEdge.kvBoundary == 96 && exactEdge.resumeBoundary() == 96 &&
              !exactEdge.junctionBoundary,
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

// A prompt that leaves the cached chain where it goes on to a state of its
// own branches there: its lookup asks for a junction at the branch point,
// and once one is published there the next lookup resumes from it.
void testLazyJunctionMaterialization() {
  CacheFixture fixture;
  fixture.publish(0);
  fixture.publish(3);
  std::vector<uint32_t> branch(fixture.prompt.begin(), fixture.prompt.begin() + 64);
  branch.resize(97, 7);
  {
    auto first = fixture.cache.lookup(branch, {});
    require(first.kvBoundary == 64 && first.resumeBoundary() == 32 &&
                first.junctionBoundary == 64,
            "first branch lookup did not request lazy materialization");
  }
  fixture.publish(1);
  auto second = fixture.cache.lookup(branch, {});
  require(second.resumeBoundary() == 64 && !second.junctionBoundary,
          "second request did not resume from the lazy junction");
}

// A junction is worth a state only where another branch goes on to a state
// below the matched KV: not where the cached chain ends, nor inside a tail
// past the state that holds none. Only a lookup at a branch point counts as
// a lazy junction.
void testLazyJunctionOnlyAtABranchPoint() {
  CacheFixture fixture;
  fixture.publish(0);
  std::vector<uint32_t> branch(fixture.prompt.begin(), fixture.prompt.begin() + 64);
  branch.resize(97, 7);
  const auto junction = [&](std::span<const uint32_t> prompt) {
    auto lookup = fixture.cache.lookup(prompt, {});
    require(lookup.resumeBoundary() == 32, "the lookup did not resume from the state");
    fixture.cache.recordLookup(lookup);
    return lookup.junctionBoundary;
  };
  const auto lazyJunctions = [&] { return fixture.cache.snapshot().lookup.lazyJunctions; };
  require(junction(fixture.prompt) == 0 && lazyJunctions() == 0,
          "a chain end requested a lazy junction");
  require(junction(branch) == 0 && lazyJunctions() == 0,
          "a dead tail past the state requested a lazy junction");
  fixture.publish(3);
  require(junction(branch) == 64 && lazyJunctions() == 1,
          "a branch point with a state below did not request a lazy junction");
}

void testByteLruAndPins() {
  CacheFixture fixture;
  fixture.publish(0, 150);
  fixture.publish(1, 150);
  auto pinned = fixture.lookup(33);
  require(pinned.state.has_value(), "state pin failed");
  fixture.publish(2, 150);
  // Keep the only KV leaf state-backed so this assertion isolates state LRU;
  // a state-free KV leaf would go first.
  fixture.publish(3, 150);
  require(reclaimStateBytes(fixture.cache, CacheReclaimMode::ReleaseExtents) == 150,
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
  require(reclaimStateBytes(fixture.cache, CacheReclaimMode::ReleaseExtents) == 150,
          "released state pins did not restore LRU eligibility");
}

// The field failure this guards: under host pressure a shrink that no request
// was waiting for discarded the only published state one second after it
// appeared, and the follow-up replayed its whole prompt instead of resuming.
void testSpeculativeReclaimKeepsTheResumePoint() {
  CacheFixture fixture;
  fixture.publish(0, 100);
  fixture.publish(2, 100);
  reclaimEverything(fixture.cache, true);
  require(fixture.cache.snapshot().stateCache.entries == 1,
          "an unbounded speculative shrink did not stop at the resume point");
  // The chain the kept publication needs survives with it: its own KV block
  // is not state-free, and every ancestor still has a child.
  auto resumed = fixture.cache.lookup(fixture.prompt, {});
  require(resumed.resumeBoundary() == 96,
          "the kept publication could not resume the next request");
  resumed.state.reset();
  reclaimEverything(fixture.cache, false);
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
      fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse, true);
  const auto kept = fixture.cache.snapshot().stateCache;
  require(step.madeProgress && kept.entries == 1 && kept.checkpointEntries == 0,
          "a disposable checkpoint outranked the resume point");
  require(fixture.lookup(33).resumeBoundary() == 32,
          "the warm conversation lost its resumable prefix");
}

void testKvEvictionInvalidatesStateFirst() {
  CacheFixture fixture;
  fixture.publish(3);
  require(reclaimStateBytes(fixture.cache, CacheReclaimMode::ReleaseExtents) == 100,
          "state was not reclaimed before its KV block");
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
                      .reclaimedBytes == 100 &&
              fixture.cache.lookup(fixture.prompt, {}).kvBoundary == 96,
          "KV leaf eviction did not remove the dependent prefix");

  CacheFixture pinnedFixture;
  pinnedFixture.publish(2);
  auto lease = pinnedFixture.lookup(97);
  require(lease.state.has_value(), "replacement state pin failed");
  reclaimEverything(pinnedFixture.cache, false);
  require(pinnedFixture.cache.snapshot().pool.pagesPrefix == 3 &&
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
    fixture.cache.publishCompositeState(999, std::make_shared<TestState>(100), false);
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
  require(fixture.cache.reuseCompositeState(fixture.blocks[0], false),
          "resident publication probe missed an existing state");
  require(!fixture.cache.reuseCompositeState(fixture.blocks[2], false),
          "publication probe found a state that was never published");
  const auto probed = fixture.cache.snapshot();
  require(probed.lookup.kvHitTokens == before.lookup.kvHitTokens &&
              probed.lookup.stateDiskHits == before.lookup.stateDiskHits,
          "publication probe polluted restore hit/miss accounting");
  require(reclaimStateBytes(fixture.cache, CacheReclaimMode::ReleaseExtents) == 200,
          "publication probe did not promote the existing state in LRU");
}

void testUnifiedRecencyAndReleasedByteAccounting() {
  {
    CacheFixture fixture;
    fixture.publish(0, 150);
    const CacheReclaimResult reclaimed =
        fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse);
    require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 100 &&
                fixture.cache.snapshot().pool.pagesPrefix == 3 &&
                fixture.cache.snapshot().stateCache.entries == 1,
            "global cache order did not select the older state-free KV leaf");
  }

  {
    // With the only leaf state-backed, the state is selected. Its eviction
    // releases nothing by itself: its buffers return to the model's pool.
    CacheFixture fixture;
    fixture.publish(3, 150);
    const CacheReclaimResult reclaimed =
        fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse);
    require(reclaimed.madeProgress && reclaimed.evictedState && reclaimed.reclaimedBytes == 0 &&
                fixture.cache.snapshot().stateCache.entries == 0 &&
                fixture.cache.snapshot().stateCache.bytes == 0 &&
                fixture.cache.snapshot().pool.pagesPrefix == 4,
            "the state was not evicted, or its eviction counted as released");
  }
}

// A request publishes a state on an interior block, then keeps committing
// blocks (its decode tail) before it ends. No state restores through the
// tail, so its leaves go first, as dead KV goes before any state; then the
// state, and its block after it.
void testFinishedRequestLeavesTailKvBeforeItsState() {
  test::TestKvStorage storage{4, 100, 1};
  KvPool pool{storage, 4};
  engine::Cache cache{pool, nullptr, nullptr};
  std::vector<uint32_t> prompt;
  for (uint32_t token = 0; token < 129; ++token)
    prompt.push_back(5000 + token);

  cache.beginRequest(1);
  require(admitTokens(cache, 1, 64).granted(),
          "prefix pages were not acquired");
  cache.publishCommittedBlocks(1, prompt, 64, {});
  const uint64_t stateBlock = cache.blockAt(1, 64);
  cache.publishCompositeState(stateBlock, std::make_shared<TestState>(100), false);
  require(admitTokens(cache, 1, 128).granted(), "tail pages were not acquired");
  cache.publishCommittedBlocks(1, prompt, 128, {});
  cache.endRequest(1);
  require(cache.snapshot().pool.pagesPrefix == 4 &&
              cache.snapshot().stateCache.entries == 1,
          "tail-before-state fixture geometry changed");

  for (uint32_t remaining : {3U, 2U}) {
    const CacheReclaimResult reclaimed =
        cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse);
    require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 0 &&
                cache.snapshot().pool.pagesPrefix == remaining &&
                cache.snapshot().stateCache.entries == 1,
            "finished request's state was evicted before its KV tail");
  }
  require(cache.lookup(prompt, {}).resumeBoundary() == 64,
          "state did not survive the eviction of the decode tail");
  const CacheReclaimResult state =
      cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse);
  require(state.madeProgress && state.evictedState &&
              cache.snapshot().stateCache.entries == 0 &&
              cache.snapshot().pool.pagesPrefix == 2,
          "state was not evicted once its block became the oldest leaf");
  const CacheReclaimResult leaf =
      cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse);
  require(leaf.madeProgress && cache.snapshot().pool.pagesPrefix == 1,
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
    require(admitRestore(fixture.cache, 2, lookup).granted(), "restore pages were denied");
    fixture.cache.endRequest(2);
    require(!fixture.cache.retireCheckpointState(replacement) &&
                fixture.cache.snapshot().stateCache.entries == 1,
            "retirement removed a pinned restore checkpoint");
  }
  require(fixture.cache.retireCheckpointState(replacement) &&
              fixture.cache.snapshot().stateCache.entries == 0,
          "restoration made a checkpoint ineligible for rolling retirement");

  const StateCheckpoint shared = publish();
  require(fixture.cache.reuseCompositeState(block, false),
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
    require(admitRestore(fixture.cache, 2, lookup).granted(), "restore pages were denied");
    fixture.cache.endRequest(2);
  }

  require(fixture.cache.reclaimOneState(false, 0, false) &&
              !fixture.cache.checkpointState(fixture.blocks[2]) &&
              fixture.cache.checkpointState(fixture.blocks[1]),
          "restore did not refresh recency within checkpoint LRU");
  require(fixture.cache.reclaimOneState(false, 0, false) &&
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
  require(reclaimStateBytes(fixture.cache, CacheReclaimMode::ReleaseExtents) == 200 &&
              fixture.cache.snapshot().pool.pagesPrefix == 4 &&
              fixture.lookup(33).resumeBoundary() == 32,
          "checkpoint reclaim displaced older ordinary state or KV");
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
                      .reclaimedBytes == 100 &&
              fixture.cache.snapshot().pool.pagesPrefix == 3,
          "ordinary state and KV lost their shared LRU order");
}

void testOptionalReclaimLeavesOrdinaryStateIntact() {
  CacheFixture fixture;
  fixture.publish(0, 150);
  fixture.cache.publishCompositeState(fixture.blocks[2],
                                      std::make_shared<TestState>(200), true);
  require(fixture.cache.reclaimOneState(true, 0, false) &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 0 &&
              fixture.lookup(33).resumeBoundary() == 32,
          "optional publication failed to recycle a disposable checkpoint");
  require(!fixture.cache.reclaimOneState(true, 0, false) &&
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
  require(fixture.cache.reclaimOneState(false, 0, false) &&
              fixture.lookup(33).resumeBoundary() == 0 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 2 &&
              fixture.cache.snapshot().stateCache.checkpointBytes == 500,
          "global reclaim evicted pinned checkpoint state");

  require(fixture.cache.reuseCompositeState(fixture.blocks[1], false) &&
              fixture.cache.retireCheckpointState(checkpoint),
          "pinned ordinary boundary could not upgrade its checkpoint");
  first.state.reset();
  second.state.reset();
  require(fixture.cache.reclaimOneState(false, 0, false) &&
              fixture.lookup(97).resumeBoundary() == 64 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 0,
          "pinned boundary upgrade corrupted the eviction queues");
  require(fixture.cache.reclaimOneState(false, 0, false) &&
              fixture.cache.lookup(fixture.prompt, {}).resumeBoundary() == 64,
          "upgraded checkpoint did not join ordinary LRU");

  require(fixture.cache.reuseCompositeState(fixture.blocks[1], true) &&
              !fixture.cache.checkpointState(fixture.blocks[1]),
          "optional publication downgraded an ordinary boundary");
}

void testCheckpointPressurePreservesHotPrefix() {
  SharedBudget budget;
  test::TestKvStorage storage(4, 100, 1);
  storage.growthAllowed = [&](uint32_t) { return budget.acquire(100); };
  storage.released = [&] { budget.release(100); };
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  const std::vector<uint32_t> hot(33, 11);
  const std::vector<uint32_t> cold(65, 22);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 32).granted(), "hot KV admission failed");
  const uint64_t hotBlock = test::publishBlocks(cache, 1, hot, 32);
  cache.publishCompositeState(hotBlock, std::make_shared<BudgetState>(budget), false);
  cache.endRequest(1);
  {
    auto lookup = cache.lookup(hot, {});
    require(lookup.resumeBoundary() == 32, "hot prefix did not restore");
    cache.beginRequest(2);
    require(admitRestore(cache, 2, lookup).granted(), "restore pages were denied");
    cache.endRequest(2);
  }

  cache.beginRequest(3);
  require(admitTokens(cache, 3, 32).granted(), "cold KV admission failed");
  const uint64_t coldBlock = test::publishBlocks(cache, 3, cold, 32);
  cache.publishCompositeState(coldBlock, std::make_shared<BudgetState>(budget),
                              true);
  require(budget.used == SharedBudget::capacity,
          "checkpoint did not fill the shared allocation budget");
  const TokenAdmission denied = cache.ensureTokens(3, 64);
  require(denied.failure == TokenAdmissionFailure::Denied,
          "necessary KV growth was not denied by the shared budget");
  const auto reclaimed = cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse);
  require(reclaimed.madeProgress && reclaimed.evictedState &&
              cache.ensureTokens(3, 64).granted() &&
              cache.lookup(hot, {}).resumeBoundary() == 32 &&
              cache.lookup(cold, {}).resumeBoundary() == 0 && budget.used == 500,
          "successful checkpoint allocation later displaced the hot prefix");
  cache.endRequest(3);
}

} // namespace

void recordUse(CacheFixture &fixture, uint32_t tokens) {
  auto lookup = fixture.lookup(tokens);
  fixture.cache.recordLookup(lookup);
}

void publishReusable(CacheFixture &fixture, uint64_t block,
                     std::shared_ptr<const CompositeState> state) {
  fixture.cache.publishCompositeState(block, std::move(state), false);
  const auto index = std::find(fixture.blocks.begin(), fixture.blocks.end(), block) - fixture.blocks.begin();
  recordUse(fixture, static_cast<uint32_t>((index + 1) * 32 + 1));
}

void testTieredStateLifecycle() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(reclaimStateBytes(fixture.cache, CacheReclaimMode::ReleaseExtents) == 100,
          "demotion did not free the state's RAM at once");
  auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.bytes == 0 && snapshot.diskBytes == 100 && snapshot.entries == 1 &&
              snapshot.offloads == 1 && control->slots == 1,
          "demoted entry did not become its disk copy");
  require(!fixture.cache.pollTransfers(), "unfinished write was consumed early");
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TestState>(100), true);
  require(reclaimStateBytes(fixture.cache, CacheReclaimMode::ReleaseExtents) == 100,
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
    require(!fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
                 .madeProgress,
            "pinned restore lost its KV");
  }
  require(
      fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse).madeProgress,
      "disk state permanently protected KV");
  require(fixture.cache.snapshot().stateCache.entries == 0 && control->slots == 0,
          "KV eviction leaked disk state");
}

void testTieredWriteReuseAndFailure() {
  for (bool fail : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
    require(reclaimStateBytes(fixture.cache, CacheReclaimMode::ReleaseExtents) == 100,
            "demotion did not free RAM");
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

// A full quota replaces the least recently used copy, for a checkpoint too.
void testDiskQuotaReplacesByRecency() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 1;
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control));
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
                  .madeProgress &&
              fixture.cache.pollTransfers() && control->slots == 1,
          "first write failed");
  // The second state needs the slot: the older copy, the only one of its
  // state, gives way.
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
                  .madeProgress &&
              fixture.cache.pollTransfers(),
          "replacement failed");
  require(fixture.cache.snapshot().stateCache.entries == 1 && control->slots == 1 &&
              fixture.lookup(129).resumeBoundary() == 128 && !fixture.lookup(33).state,
          "the older disk copy was not replaced");
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), true);
  require(fixture.cache.reclaimOneState(true, 0, false) && fixture.cache.pollTransfers() &&
              fixture.cache.snapshot().stateCache.offloads == 3 && control->slots == 1 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 1 &&
              fixture.lookup(129).resumeBoundary() == 64,
          "checkpoint did not replace the least recently used copy");
}

// A rolling checkpoint uses the tier like any state: straight to disk when no
// cache slot holds it, written under RAM pressure, replacing the least recently
// used copy when the quota is full, retired from both tiers with its successor;
// under KV pressure its leaf is demoted, not dropped.
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
  fixture.cache.publishCompositeState(fixture.blocks[2],
                                      std::make_shared<TieredState>(control), true);
  require(fixture.cache.reclaimOneState(true, 0, false) && fixture.cache.pollTransfers(),
          "RAM checkpoint was not reclaimed");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.offloads == 2 && stats.bytes == 0 && stats.checkpointEntries == 1 &&
              stats.checkpointBytes == 0 && control->slots == 1,
          "RAM checkpoint was dropped although the quota had room");
  control->capacity = 1;
  fixture.cache.publishCompositeState(fixture.blocks[3],
                                      std::make_shared<TieredState>(control), true);
  require(fixture.cache.reclaimOneState(true, 0, false) && fixture.cache.pollTransfers() &&
              control->slots == 1 && fixture.cache.snapshot().stateCache.offloads == 3 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 1 &&
              fixture.cache.checkpointState(fixture.blocks[3]) &&
              !fixture.cache.checkpointState(fixture.blocks[2]),
          "a full quota kept the older checkpoint");
  require(
      fixture.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress &&
          tier.demotions == 1 && fixture.cache.snapshot().pool.pagesPrefix == 4,
      "the checkpoint's leaf was dropped instead of demoted");
}

void testDiskPromotionAndInvalidation() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const auto block = fixture.blocks[3];
  publishReusable(fixture, block, std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
                  .madeProgress &&
              fixture.cache.pollTransfers(),
          "offload failed");
  auto first = fixture.lookup(129);
  auto peer = fixture.lookup(129);
  auto invalid = first.state->state();
  fixture.cache.discardState(block, invalid.get());
  require(!fixture.lookup(129).state, "invalid disk entry admitted another reader");
  require(fixture.cache.probe(fixture.prompt, {}).cachedTokens() == 0,
          "scheduling probe counted an invalid disk state as reusable work");
  first = {};
  fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
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
  require(
      fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse).madeProgress,
      "write not started");
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
  require(fixture.cache.reclaimOneState(false, 0, false).made,
          "required publication could not recycle a buffer");
  auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.entries == 2 && snapshot.bytes == 100 && snapshot.diskBytes == 100,
          "demotion took a second state or kept the victim's RAM");
  // One write in flight: an ordinary publication leaves a second victim, whose
  // write would have to wait, and one in use drops it.
  require(!fixture.cache.reclaimOneState(false, 0, false) &&
              fixture.cache.snapshot().stateCache.bytes == 100,
          "an ordinary publication dropped a victim whose write would wait");
  StateUse use = fixture.cache.useState(fixture.blocks[1]);
  require(fixture.cache.reclaimOneState(false, fixture.blocks[1], false).made,
          "second recycle failed");
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
  // Copies refused before one succeeds, as while no cache slot is free.
  uint32_t deniedCopies = 0;
  bool ready() const noexcept override { return true; }
  bool finish() override { return true; }
  void cancel() noexcept override {}
  std::shared_ptr<const CompositeState> snapshot() override {
    ++copies;
    if (deniedCopies) {
      --deniedCopies;
      return nullptr;
    }
    return available ? std::make_shared<TestState>(100) : nullptr;
  }
};

void testPromotionIdentityAndDenial() {
  for (bool available : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
    require(fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
                    .madeProgress &&
                fixture.cache.pollTransfers(),
            "promotion fixture write failed");
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

// Promotion only saves a later read: the oldest RAM copy gives it room only
// when that state keeps a disk copy, never when the RAM copy is its only one.
void testPromotionNeverDropsAUniqueState() {
  for (const bool onDisk : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
    require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
            "the state to promote was not written");
    if (onDisk) {
      // Written and published again, the older state has a RAM copy and a
      // disk copy beside it.
      fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
      require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
              "the older state was not written");
      fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
    } else {
      fixture.publish(1);
    }
    const auto lookup = fixture.lookup(129);
    PromotionTicket ticket;
    ticket.deniedCopies = 1;
    fixture.cache.promoteState(lookup, ticket);
    const auto stats = fixture.cache.snapshot().stateCache;
    if (onDisk)
      require(stats.promotions == 1 && !fixture.cache.stateResident(fixture.blocks[1]) &&
                  fixture.lookup(65).resumeBoundary() == 64,
              "promotion did not take the RAM of a state with a disk copy");
    else
      require(stats.promotions == 0 && stats.promotionsSkipped == 1 &&
                  fixture.cache.stateResident(fixture.blocks[1]),
              "promotion dropped a state's only copy");
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
  require(fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
                  .madeProgress &&
              fixture.cache.pollTransfers(),
          "state was not demoted");
  require(!fixture.cache.reuseCompositeState(fixture.blocks[0], false),
          "a disk-only copy stood in for a RAM publication");
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.diskBytes == 100 && stats.bytes == 200 && control->slots == 1 &&
              fixture.lookup(33).state->state()->residentBytes() == 100,
          "republication dropped the disk copy or did not add the RAM copy");
  require(fixture.cache.reuseCompositeState(fixture.blocks[0], false), "RAM copy was not reusable");
  // The other state is used; the republished one is the oldest RAM copy and
  // leaves RAM without a second write.
  recordUse(fixture, 129);
  require(fixture.cache.reclaimOneState(false, 0, false) && !fixture.cache.pollTransfers(),
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
  require(fixture.cache.reclaimOneState(false, 0, false).made, "state was not dropped");
  {
    auto lookup = fixture.lookup(129);
    require(lookup.kvBoundary == 128 && !lookup.state && lookup.lostState,
            "dropped state was not recognised");
    fixture.cache.recordLookup(lookup);
  }
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
          "state was not demoted");
  {
    auto hit = fixture.lookup(129);
    require(hit.state && !hit.lostState, "a disk hit was counted as a lost state");
  }
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TestState>(100), true);
  require(fixture.cache.reclaimOneState(true, 0, false).made, "checkpoint was not dropped");
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
  require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
          "state was not demoted");
  {
    auto lookup = fixture.lookup(129);
    PromotionTicket ticket;
    fixture.cache.promoteState(lookup, ticket);
    require(ticket.copies == 1, "restored state was not promoted");
  }
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.promotions == 1 && stats.bytes == 100 && stats.diskBytes == 100 && control->slots == 1,
          "promotion dropped the disk copy");
  require(fixture.cache.reclaimOneState(false, 0, false) && !fixture.cache.pollTransfers(),
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
    require(fixture.cache.reclaimOneState(true, 0, false) && fixture.cache.pollTransfers(),
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
      require(fixture.cache.reuseStoredState(fixture.blocks[3]),
              "the disk checkpoint was not reused as an ordinary state");
    } else {
      fixture.cache.publishCompositeState(fixture.blocks[3],
                                          std::make_shared<TestState>(100), false);
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
// a block already on disk is reused rather than published twice, and under
// KV pressure the stated leaf is demoted rather than dropped.
void testDiskPublicationLifecycle() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  const StateWriter write = [&](std::function<void()>) { return writeState(control); };
  require(fixture.cache.publishStateToDisk(fixture.blocks[3], write, false),
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
  require(!fixture.cache.publishStateToDisk(fixture.blocks[1], write, false) &&
              fixture.cache.snapshot().stateCache.entries == 1 && control->slots == 1,
          "a second write started beside the one in flight");
  require(fixture.cache.reuseStoredState(fixture.blocks[3]) &&
              fixture.cache.snapshot().stateCache.offloads == 1,
          "a block already on disk was written again");
  requireThrows<std::logic_error>(
      [&] { static_cast<void>(fixture.cache.publishStateToDisk(fixture.blocks[3], write, false)); },
      "a second disk publication of one block was accepted");
  control->ready = true;
  require(fixture.cache.pollTransfers() && !fixture.cache.pollTransfers(),
          "the write was not consumed exactly once");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 1 && stats.diskBytes == 100 && stats.offloadFailures == 0,
          "completion changed tier occupancy");
  require(fixture.cache.publishStateToDisk(fixture.blocks[1], write, false) && control->slots == 2,
          "the next publication did not follow the finished write");
  require(
      fixture.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress &&
          tier.demotions == 1 && fixture.cache.snapshot().pool.pagesPrefix == 4,
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
  require(fixture.cache.publishStateToDisk(fixture.blocks[3], write, false) &&
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
    require(fixture.cache.reclaimOneState(false, 0, false).made, "state was not demoted");
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
      require(fixture.cache.publishStateToDisk(block, write, false) && fixture.cache.pollTransfers(),
              "the disk publication at the emptied block failed");
    } else {
      fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
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
      require(fixture.cache.reclaimOneState(false, 0, false).made, "state was not demoted");
      {
        auto reader = fixture.lookup(129);
        fixture.cache.discardState(block, reader.state->state().get());
      }
      fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
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
  require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers() &&
              control->slots == 1,
          "the older state did not fill the quota");
  const StateWriter write = [&](std::function<void()>) { return writeState(control); };
  require(fixture.cache.publishStateToDisk(fixture.blocks[3], write, false) &&
              fixture.cache.pollTransfers(),
          "the full quota refused the lane's state");
  const auto stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 1 && stats.offloads == 2 && control->slots == 1 &&
              !fixture.lookup(65).state && fixture.lookup(129).state,
          "the older copy did not make room for the new one");
  control->capacity = 0;
  CacheFixture empty;
  require(!empty.cache.publishStateToDisk(empty.blocks[3], write, false) &&
              empty.cache.snapshot().stateCache.entries == 0 &&
              !empty.lookup(129).lostState,
          "a state was published without a disk copy");
}

// The quota is one order across both kinds of copies: a write that needs
// room drops redundant copies first, KV or state, then the oldest copy that
// is the only one.
void testDiskReplacementSpansStatesAndKv() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  tier.capacity = 1;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 1;
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
          "state was not demoted");
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == 1,
          "leaf was not written");
  tier.complete();
  require(fixture.cache.pollTransfers(), "leaf did not land");
  // Both come back: the block and its state hold RAM and disk copies.
  {
    auto lookup = fixture.lookup(129);
    fixture.cache.beginRequest(2);
    require(admitRestore(fixture.cache, 2, lookup).granted(), "restore was denied");
    tier.complete();
    require(fixture.cache.pollTransfers(), "restore did not land");
    PromotionTicket ticket;
    fixture.cache.promoteState(lookup, ticket);
    require(ticket.copies == 1, "state was not promoted");
    lookup = {};
    fixture.cache.endRequest(2);
  }
  auto stats = fixture.cache.snapshot();
  require(stats.kvTier.diskBlocks == 1 && stats.pool.pagesPrefix == 4 &&
              stats.stateCache.bytes == 100 && stats.stateCache.diskBytes == 100,
          "restore did not leave both copies of both");
  // A new state at block 2 needs disk room. The oldest RAM copy leaves RAM
  // for nothing first; then the write gives up the redundant KV copy and,
  // when that is not enough, the state copy that is the only one.
  fixture.cache.publishCompositeState(fixture.blocks[2], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, 0, false) &&
              fixture.cache.snapshot().stateCache.offloads == 1,
          "the restored state was written again");
  require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
          "new state was not written");
  stats = fixture.cache.snapshot();
  require(stats.stateCache.offloads == 2 && stats.stateCache.entries == 1 && control->slots == 1 &&
              stats.kvTier.diskBlocks == 0 && stats.pool.pagesPrefix == 4 &&
              fixture.lookup(97).state && !fixture.lookup(97).state->state()->residentBytes(),
          "disk replacement did not give up redundant copies before the only ones");
}

// When only the KV tier failed to start, the states' file still draws on the
// disk quota, and the quota and its IO are reported all the same.
void testQuotaWithoutTheKvTier() {
  constexpr uint64_t size = kHostPageBytes;
  auto budget = std::make_shared<model::DiskBudget>(4 * size);
  test::TestKvStorage storage{4, 100, 1};
  KvPool pool{storage, 4};
  engine::Cache cache(pool, nullptr, budget);
  model::SlotFile states(size, budget);
  auto slot = states.acquire();
  std::vector<std::byte> source(size, std::byte{1}), restored(size);
  require(slot && states.write(slot, {source}, {})->wait() &&
              states.read(slot, {restored}, {})->wait(),
          "state file IO failed");
  const auto tier = cache.snapshot().kvTier;
  require(tier.capacityBytes == 4 * size && tier.usedBytes == size && tier.readBytes == size &&
              tier.writtenBytes == size && tier.fileBytes == size && tier.diskBlocks == 0 &&
              tier.diskBytes == 0,
          "the disk quota went unreported without the KV tier");
}

// Independent prefixes under random publications, uses and reclaims: the
// RAM contents match a cache without the tier step for step, and every hit
// without the tier is a hit with it. The disk only adds.
void testTierOnlyAddsToTierOff() {
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
        off.cache.publishCompositeState(off.blocks[index], std::make_shared<TestState>(100), false);
        on.cache.publishCompositeState(on.blocks[index], std::make_shared<TieredState>(control), false);
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
        static_cast<void>(off.cache.reclaimOneState(false, 0, false));
        static_cast<void>(on.cache.reclaimOneState(false, 0, false));
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
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false); // B
  require(fixture.cache.reclaimOneState(false, 0, false).made,
          "no buffer was recycled for C"); // C needs a buffer
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
      tier.transferLimit = 1;
      CacheFixture fixture(&tier);
      auto control = std::make_shared<TransferControl>();
      control->ready = true;
      if (peerBlocks == 2) {
        fixture.cache.publishCompositeState(fixture.blocks[1],
                                            std::make_shared<TieredState>(control), false);
        require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
                "shared prefix state demotion failed");
      }
      fixture.cache.publishCompositeState(fixture.blocks[3],
                                          std::make_shared<TieredState>(control), false);
      require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
              "state demotion failed");
      demoteLeaves(fixture.cache, tier, 4);
      auto lookup = fixture.lookup(129);
      fixture.cache.beginRequest(2);
      require(admitRestore(fixture.cache, 2, lookup).granted(), "restore was denied");
      if (peerBlocks) {
        auto peerLookup = fixture.lookup(peerBlocks * KvCache::pageTokens + 1);
        fixture.cache.beginRequest(3);
        require(peerLookup.state && admitRestore(fixture.cache, 3, peerLookup).granted(),
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
      require(admitRestore(fixture.cache, 4, retry).granted(), "retry restore was denied");
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
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  // The state goes first and frees its RAM at once; the leaf under it is
  // then worth keeping.
  require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
              tier.demotions == 0 && fixture.cache.pollTransfers(),
          "state was not demoted ahead of its leaf");
  const auto step = fixture.cache.reclaimOne(reuse, ReclaimClass::InUse);
  require(step.madeProgress && step.reclaimedBytes == 0 && tier.demotions == 1 &&
              tier.slots == 1 && fixture.pool.freePageCount() == 0,
          "leaf under a disk state was dropped or freed before its copy landed");
  auto stats = fixture.cache.snapshot();
  require(stats.kvTier.pendingPages == 1 && stats.pool.pagesPrefix == 4 &&
              stats.kvTier.diskBlocks == 1,
          "pending demotion was not accounted");
  require(!fixture.cache.pollTransfers() &&
              !fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
          "the chain was reclaimed past its demoting leaf");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1,
          "landed copy did not free the page");
  stats = fixture.cache.snapshot();
  require(stats.kvTier.demotions == 1 && stats.kvTier.pendingPages == 0 &&
              stats.pool.pagesPrefix == 3 && stats.kvTier.diskBlocks == 1 &&
              stats.kvTier.diskBytes == 100 && stats.stateCache.diskBytes == 100,
          "tier accounting after the demotion is off");
  {
    auto lookup = fixture.lookup(129);
    require(lookup.kvBoundary == 128 && lookup.state &&
                lookup.state->kvBlock() == fixture.blocks[3] &&
                !lookup.state->state()->residentBytes(),
            "prefix on disk did not match");
    fixture.cache.beginRequest(2);
    require(admitRestore(fixture.cache, 2, lookup).granted() &&
                fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::Pending &&
                tier.restores == 1 && fixture.pool.freePageCount() == 0 &&
                fixture.cache.pageTable(2).pages.size() == 4,
            "restore did not take a page for the disk-only block");
    // A second request on the same prefix waits for the same restore.
    auto again = fixture.lookup(129);
    fixture.cache.beginRequest(3);
    require(admitRestore(fixture.cache, 3, again).granted() &&
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
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              tier.demotions == 1 && fixture.pool.freePageCount() == 1 &&
              fixture.cache.snapshot().pool.pagesPrefix == 3 &&
              fixture.cache.snapshot().stateCache.diskBytes == 100,
          "a block with a disk copy was written again or lost its state");
}

// Leaves nothing depends on are dropped, never written; a parent whose child
// went to disk is written when its turn comes, so the chain stays whole.
void testTailsDropAndParentsFollowToDisk() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              tier.demotions == 0 && fixture.pool.freePageCount() == 1,
          "a tail was written");
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[2], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
              fixture.cache.pollTransfers(),
          "state was not demoted first");
  for (uint32_t written = 1; written <= 3; ++written) {
    require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                tier.demotions == written,
            "a block the disk chain depends on was dropped");
    tier.complete();
    require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1 + written,
            "written block did not free its page");
  }
  const auto stats = fixture.cache.snapshot();
  require(stats.pool.pagesPrefix == 0 && stats.kvTier.diskBlocks == 3 && tier.slots == 3 &&
              stats.stateCache.diskBytes == 100,
          "chain did not move to disk whole");
  auto lookup = fixture.lookup(129);
  require(lookup.kvBoundary == 96 && lookup.state,
          "disk chain did not match up to its state");
}

// Only a state restores a disk-only chain. Once a failed state write or a
// full quota has taken the state a disk child was written for, the parent is
// not written for that child: both go, and the page returns at once.
void testDiskCopiesNoStateNeedsGoWithTheLeaf() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  {
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    auto control = std::make_shared<TransferControl>();
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
    require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
                fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                tier.demotions == 1,
            "the leaf under a state being written was not written");
    tier.complete();
    control->success = false;
    control->ready = true;
    require(fixture.cache.pollTransfers() && fixture.cache.snapshot().stateCache.entries == 0 &&
                fixture.cache.snapshot().kvTier.diskBlocks == 1,
            "the failed state write left its state, or the leaf did not land");
    fixture.cache.beginRequest(2);
    require(admitTokens(fixture.cache, 2, 64).granted() && tier.demotions == 1 &&
                tier.slots == 0 && fixture.cache.snapshot().kvTier.diskBlocks == 0,
            "a leaf was written for a disk child no state needs");
    fixture.cache.endRequest(2);
  }
  {
    // The quota holds one page, so the parent's demotion replaces the
    // oldest sole copy: the child, and the state on it.
    test::TestKvTier tier;
    tier.capacity = 1;
    CacheFixture fixture(&tier);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
    require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
                fixture.cache.pollTransfers() &&
                fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                tier.demotions == 1,
            "leaf was not written");
    tier.complete();
    require(fixture.cache.pollTransfers() && tier.slots == 1, "leaf did not land");
    require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                tier.demotions == 1 && tier.slots == 0 && fixture.pool.freePageCount() == 2 &&
                fixture.cache.snapshot().stateCache.entries == 0,
            "a leaf was written after making room took the state it was written for");
  }
}

// A demotion the tier cannot take right now keeps its leaf while transfers
// in flight will free room: the shortfall is Pending and the leaf is written
// once the tier has room. Room or a file that nothing will free is not worth
// waiting for: the leaf goes as without a tier, unless a disk subtree
// depends on it.
void testRefusedDemotionKeepsTheLeafWhileTransfersLand() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  tier.transferLimit = 1;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
              fixture.cache.pollTransfers(),
          "state was not demoted first");
  // One demotion is all the tier takes at a time, and it stays in flight.
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              tier.demotions == 1 && tier.inFlight() == 1,
          "the first leaf was not written");
  // Its parent is no leaf while the page being written is still resident, so
  // the pass has nothing to give and says so: wait for the transfer.
  const CacheReclaimResult busy = fixture.cache.reclaimOne(reuse, ReclaimClass::InUse);
  require(!busy.madeProgress && busy.pending && tier.demotions == 1 &&
              fixture.cache.snapshot().pool.pagesPrefix == 4,
          "a leaf was dropped or the wait was not reported while the tier was busy");
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 32).failure == TokenAdmissionFailure::Pending,
          "a request was failed while a transfer was landing");
  fixture.cache.endRequest(2);
  tier.complete();
  require(fixture.cache.pollTransfers() && tier.inFlight() == 0, "the demotion did not land");
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == 2,
          "the leaf was not written once the tier had room");
  tier.complete();
  require(fixture.cache.pollTransfers(), "second leaf did not land");
}

void testUnusableTierDropsTheLeafInstead() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  tier.transferLimit = 0;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
              fixture.cache.pollTransfers(),
          "state was not demoted first");
  // Nothing is in flight and nothing ever makes room: waiting would be
  // waiting for nothing, so the leaf and its disk copy go instead.
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              tier.demotions == 0 && fixture.cache.snapshot().pool.pagesPrefix == 3 &&
              fixture.cache.snapshot().kvTier.demotionsRefused == 1,
          "an unusable tier parked the leaf instead of dropping it");
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 32).granted(),
          "a request waited although nothing was in flight");
  fixture.cache.endRequest(2);

  // A leaf a disk subtree depends on stays while that subtree is in use: a
  // lookup holds the state below it. Once the tier takes no writes and the
  // subtree is free, the leaf goes with it, as without a tier, instead of
  // holding its RAM until the server restarts.
  test::TestKvTier second;
  CacheFixture deep(&second);
  deep.cache.publishCompositeState(deep.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(deep.cache, reuse) == 100 &&
              deep.cache.pollTransfers(),
          "deep state was not demoted");
  require(deep.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && second.demotions == 1,
          "leaf was not written");
  second.complete();
  require(deep.cache.pollTransfers(), "leaf did not land");
  second.writableFile = false;
  {
    auto held = deep.lookup(129);
    require(held.state && !deep.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                deep.cache.snapshot().pool.pagesPrefix == 3 &&
                deep.cache.snapshot().kvTier.diskBlocks == 1,
            "a leaf was dropped under a disk subtree a lookup holds");
  }
  require(deep.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              deep.cache.snapshot().pool.pagesPrefix == 2 &&
              deep.cache.snapshot().kvTier.diskBlocks == 0 &&
              deep.cache.snapshot().stateCache.entries == 0,
          "an unwritable tier kept the leaf and its disk subtree");
  deep.cache.beginRequest(2);
  require(admitTokens(deep.cache, 2, 64).granted(),
          "the dropped leaf's page did not come back");
  deep.cache.endRequest(2);
}

// The failure seen at 23G: a leaf with disk-only children whose demotion is
// refused must stay, not be erased under its children.
void testParentOfDiskChildrenSurvivesRefusal() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
              fixture.cache.pollTransfers() &&
              fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              tier.demotions == 1,
          "leaf was not written");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "leaf did not land");
  // With the tier unusable and nothing in flight the parent still stays: its
  // disk subtree depends on it. The request is told it cannot have the pages
  // rather than told to wait for something that will never happen.
  tier.transferLimit = 0;
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 64).failure == TokenAdmissionFailure::Denied &&
              fixture.cache.snapshot().pool.pagesPrefix == 3 && tier.demotions == 1,
          "the parent of a disk block was dropped, or the request was told to wait");
  tier.transferLimit = 8;
  require(admitTokens(fixture.cache, 2, 64).failure == TokenAdmissionFailure::Pending &&
              tier.demotions == 2,
          "the parent was not written once the tier had room");
  tier.complete();
  require(fixture.cache.pollTransfers() && admitTokens(fixture.cache, 2, 64).granted(),
          "pages did not return to the request");
  fixture.cache.endRequest(2);
}

// While one state write is in flight, a pressure pass keeps the next state
// in RAM rather than dropping it; the pass after the write takes it.
void testSecondStateWaitsForTheWrite() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control));
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(reclaimStateBytes(fixture.cache, reuse) == 100,
          "first state was not written");
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 100 && stats.diskBytes == 100 && stats.offloads == 1,
          "first write did not free its RAM");
  require(!fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
          "a state was dropped or a leaf under one was taken while a write was in flight");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 100 && stats.evictions == 0,
          "the waiting state did not stay");
  control->ready = true;
  require(fixture.cache.pollTransfers() &&
              reclaimStateBytes(fixture.cache, reuse) == 100,
          "the waiting state was not written after the first landed");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 0 && stats.diskBytes == 200 && stats.offloads == 2 &&
              control->slots == 2,
          "second write did not land beside the first");
}

// A checkpoint goes first, but one whose write must wait for the one in
// flight holds back nothing else: the KV leaves and ordinary states behind
// it go in recency order meanwhile, and the pass after the write takes it.
void testWaitingCheckpointHoldsBackNothingElse() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, 0, false).made, "first state was not written");
  fixture.publish(2);
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control),
                                      true);
  // The idle tail is the oldest and needs no transfer; the ordinary state
  // cannot be written and goes next, then the leaf it stood on.
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              fixture.cache.snapshot().pool.pagesPrefix == 3,
          "the waiting checkpoint held back an idle KV tail");
  require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
              fixture.cache.snapshot().stateCache.entries == 2,
          "the waiting checkpoint held back an ordinary state");
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              fixture.cache.snapshot().pool.pagesPrefix == 2,
          "the waiting checkpoint held back the leaf a state left");
  const CacheReclaimResult waiting = fixture.cache.reclaimOne(reuse, ReclaimClass::InUse);
  auto stats = fixture.cache.snapshot().stateCache;
  require(!waiting.madeProgress && waiting.pending && stats.bytes == 100 &&
              stats.checkpointEntries == 1 && stats.checkpointEvictions == 0,
          "the checkpoint did not wait for the write in flight");
  control->ready = true;
  require(fixture.cache.pollTransfers() &&
              reclaimStateBytes(fixture.cache, reuse) == 100,
          "the checkpoint was not written after the first landed");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.offloads == 2 && stats.bytes == 0 && stats.checkpointEntries == 1 &&
              control->slots == 2,
          "the checkpoint did not land beside the first write");
}

// A refusal ends the scan: the tier is full for every leaf alike, so one
// attempt costs one refusal, not one per cached block.
void testFullTierStopsTheScan() {
  test::TestKvTier tier;
  Prefixes p(&tier, false);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  for (uint64_t block : p.blocks) {
    p.cache.publishCompositeState(block, std::make_shared<TieredState>(control), false);
    require(p.cache.reclaimOneState(false, 0, false) && p.cache.pollTransfers(),
            "state was not demoted");
  }
  // One transfer at a time: the first leaf takes it and the request waits for
  // that page rather than evicting more.
  tier.transferLimit = 1;
  p.cache.beginRequest(9);
  require(admitTokens(p.cache, 9, 32).failure == TokenAdmissionFailure::Pending &&
              tier.demotions == 1 && p.cache.snapshot().pool.pagesPrefix == 4,
          "the first leaf was not written, or a leaf was dropped");
  // A larger shortfall meets a tier that the transfer in flight fills. Every
  // leaf would answer the same, so the scan asks once and waits.
  require(admitTokens(p.cache, 9, 64).failure == TokenAdmissionFailure::Pending &&
              p.cache.snapshot().kvTier.demotionsRefused == 1 &&
              p.cache.snapshot().pool.pagesPrefix == 4,
          "a full tier was asked once per leaf, or a leaf was dropped");
  tier.complete();
  require(p.cache.pollTransfers() && admitTokens(p.cache, 9, 32).granted(),
          "the page did not return");
  p.cache.endRequest(9);
}

// A shortfall while restores are in flight is pending, not exhausted: the
// restored blocks become leaves with disk copies. Pages held by an active
// request are exhausted for real.
void testRestoresInFlightMakeAShortfallPending() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
              fixture.cache.pollTransfers() &&
              fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
          "leaf was not written");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "leaf did not land");
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(2);
  require(admitRestore(fixture.cache, 2, lookup).granted() && fixture.pool.freePageCount() == 0,
          "restore did not take the free page");
  fixture.cache.beginRequest(3);
  require(admitTokens(fixture.cache, 3, 32).failure == TokenAdmissionFailure::Pending,
          "a shortfall during a restore was reported as exhausted");
  tier.complete();
  require(fixture.cache.pollTransfers() &&
              admitTokens(fixture.cache, 3, 32).failure == TokenAdmissionFailure::Denied,
          "pages held by an active request were not exhausted");
  lookup = {};
  fixture.cache.endRequest(2);
  require(admitTokens(fixture.cache, 3, 32).granted() && tier.demotions == 1,
          "the restored leaf did not give up its page for nothing");
  fixture.cache.endRequest(3);
}

// A full quota replaces the oldest redundant copy first, then the oldest
// disk-only leaf; a block whose copy was replaced stays in RAM.
void testDiskReplacementOrder() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  tier.capacity = 1;
  Prefixes p(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  for (uint64_t block : p.blocks)
    p.cache.publishCompositeState(block, std::make_shared<TieredState>(control), false);
  // The oldest state goes to disk, then the leaf under it.
  const auto demoteNext = [&] {
    require(reclaimStateBytes(p.cache, reuse) == 100 &&
                p.cache.pollTransfers(),
            "state was not demoted first");
    require(p.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
            "leaf under a disk state was not reclaimed");
  };
  // A goes to disk and comes back: RAM and disk both hold it.
  demoteNext();
  require(tier.demotions == 1, "A was not written");
  tier.complete();
  require(p.cache.pollTransfers() && p.pool.freePageCount() == 1, "A did not free its page");
  {
    auto lookup = p.cache.lookup(p.prompts[0], {});
    p.cache.beginRequest(9);
    require(admitRestore(p.cache, 9, lookup).granted(),
            "A did not restore");
    tier.complete();
    require(p.cache.pollTransfers(), "A's restore did not finish");
    lookup = {};
    p.cache.endRequest(9);
  }
  // B needs the one slot: A's redundant copy goes, A stays resident.
  demoteNext();
  require(tier.demotions == 2 && tier.slots == 1 &&
              p.cache.snapshot().pool.pagesPrefix == 4 && p.cache.snapshot().kvTier.diskBlocks == 1,
          "B did not replace A's redundant copy");
  tier.complete();
  require(p.cache.pollTransfers() && p.pool.freePageCount() == 1, "B did not free its page");
  // C needs the slot: no redundant copy is left, so the oldest disk-only
  // leaf, B, leaves with its state.
  demoteNext();
  require(tier.demotions == 3 && tier.slots == 1 &&
              p.cache.lookup(p.prompts[1], {}).kvBoundary == 0 && control->slots == 2,
          "C did not replace the oldest disk-only leaf");
}

// Pages already on their way back gate allocation: the request is told to
// wait instead of the cache demoting more than it needs.
void testPendingPagesGateAllocation() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 32).failure == TokenAdmissionFailure::Pending &&
              tier.demotions == 1 && fixture.cache.snapshot().pool.pagesPrefix == 4 &&
              fixture.cache.snapshot().kvTier.pendingPages == 1,
          "allocation evicted past the page on its way back");
  require(admitTokens(fixture.cache, 2, 32).failure == TokenAdmissionFailure::Pending &&
              tier.demotions == 1,
          "a retry before the copy landed demoted more");
  tier.complete();
  require(fixture.cache.pollTransfers() && admitTokens(fixture.cache, 2, 32).granted() &&
              fixture.cache.pageTable(2).pages.size() == 1 &&
              fixture.cache.snapshot().pool.pagesPrefix == 3,
          "pages did not return to the waiting request");
  fixture.cache.endRequest(2);
}

// A demotion that fails leaves the block in RAM without a copy; a restore
// that fails removes the block and its state once its requests let go, and
// the next lookup matches the shallower prefix.
void testTransferFailures() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  {
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
    require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
                fixture.cache.pollTransfers() &&
                fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                tier.demotions == 1,
            "no demotion");
    tier.complete(false);
    require(fixture.cache.pollTransfers(), "failed write was not consumed");
    const auto stats = fixture.cache.snapshot();
    require(stats.kvTier.demotionFailures == 1 && stats.kvTier.pendingPages == 0 &&
                stats.pool.pagesPrefix == 4 && stats.kvTier.diskBlocks == 0 && tier.slots == 0 &&
                stats.stateCache.diskBytes == 100,
            "failed write lost the page, kept the slot, or touched the state");
  }
  {
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
    require(reclaimStateBytes(fixture.cache, reuse) == 100 &&
                fixture.cache.pollTransfers() &&
                fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                tier.demotions == 1,
            "no demotion");
    tier.complete();
    require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "no disk block");
    auto lookup = fixture.lookup(129);
    fixture.cache.beginRequest(2);
    require(admitRestore(fixture.cache, 2, lookup).granted(), "restore was denied");
    tier.complete(false);
    require(fixture.cache.pollTransfers() &&
                fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::Failed &&
                fixture.cache.snapshot().kvTier.restoreFailures == 1,
            "failed read was not reported to the request");
    lookup = {};
    fixture.cache.endRequest(2);
    const auto stats = fixture.cache.snapshot();
    require(fixture.lookup(129).kvBoundary == 96 && fixture.pool.freePageCount() == 1 &&
                stats.pool.pagesPrefix == 3 && stats.kvTier.diskBlocks == 0 &&
                stats.stateCache.entries == 0 && tier.slots == 0,
            "poisoned block or its state survived its last user");
  }
}

// Nothing below a block whose read failed matches any more. A block restored
// under it and a sibling branch still on disk leave with their states once
// the request lets go, and the prefix above gives its pages up again, even
// after the fault has closed the tier.
void testFailedRestoreDropsTheBlocksBelow() {
  test::TestKvStorage storage{9, 100, 1};
  storage.budgetPages = 8;
  KvPool pool{storage, 8};
  test::TestKvTier tier;
  engine::Cache cache{pool, &tier, nullptr};
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  // Two prompts share three blocks and part at the fourth, which holds a
  // state in each.
  std::vector<uint32_t> prompt(129);
  for (uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = 1000 + i;
  std::vector<uint32_t> sibling = prompt;
  for (uint32_t i = 96; i < sibling.size(); ++i)
    sibling[i] = 2000 + i;
  for (uint64_t request : {1, 2}) {
    cache.beginRequest(request);
    require(admitTokens(cache, request, 128).granted(), "prefix KV failed");
    const uint64_t last =
        test::publishBlocks(cache, request, request == 1 ? prompt : sibling, 128);
    cache.endRequest(request);
    cache.publishCompositeState(last, std::make_shared<TieredState>(control), false);
    require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
            "state was not demoted");
  }
  // Both fourth blocks go to disk, then the third block they share.
  demoteLeaves(cache, tier, 3);
  // The shared block's read fails and the fault closes the tier; the read
  // queued behind it still lands.
  tier.transferLimit = 1;
  auto lookup = cache.lookup(prompt, {});
  cache.beginRequest(3);
  require(lookup.state && admitRestore(cache, 3, lookup).granted() && tier.restores == 1,
          "restore was denied");
  tier.complete(false);
  static_cast<void>(cache.pollTransfers());
  tier.writableFile = false;
  tier.complete();
  require(cache.pollTransfers() && tier.restores == 2 &&
              cache.kvRestoreStatus(3) == KvRestoreStatus::Failed,
          "the failed read was not reported");
  lookup = {};
  cache.endRequest(3);
  const auto stats = cache.snapshot();
  require(stats.pool.pagesPrefix == 2 && stats.kvTier.diskBlocks == 0 &&
              stats.stateCache.entries == 0 && tier.slots == 0 && control->slots == 0,
          "the blocks below a failed read or their states outlived the request");
  require(cache.lookup(prompt, {}).kvBoundary == 64, "the surviving prefix did not match");
  cache.beginRequest(4);
  require(admitTokens(cache, 4, 256).granted(),
          "the prefix above a failed read stayed pinned in RAM");
  cache.endRequest(4);
}

// A page whose demotion is in flight counts as pending bytes, which a
// pressure pass counts toward its target instead of writing the whole chain
// at once. Writing a state releases nothing by itself.
void testDemotionInFlightCountsAsPendingBytes() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  const CacheReclaimResult written =
      fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse);
  require(written.madeProgress && written.reclaimedBytes == 0 && tier.demotions == 0 &&
              fixture.cache.snapshot().stateCache.bytes == 0,
          "the state was not written first");
  require(fixture.cache.pollTransfers(), "state write was not consumed");
  const CacheReclaimResult demoted =
      fixture.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse);
  require(demoted.madeProgress && demoted.reclaimedBytes == 0 && tier.demotions == 1 &&
              fixture.cache.pendingBytes() == 100 &&
              fixture.cache.snapshot().pool.pagesPrefix == 4,
          "the leaf's page on its way back did not count as pending");
}

void testBusyTierPreservesDiskVictim() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  for (const bool restored : {false, true}) {
    test::TestKvStorage storage{4, 100, 1};
    KvPool pool{storage, 4};
    test::TestKvTier tier;
    tier.capacity = 2;
    tier.transferLimit = 1;
    engine::Cache cache{pool, &tier, nullptr};
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    control->capacity = 4;
    std::array<std::vector<uint32_t>, 4> prompts;
    for (uint32_t i = 0; i < prompts.size(); ++i) {
      prompts[i].assign(KvCache::pageTokens, 1000 + i);
      cache.beginRequest(i + 1);
      require(admitTokens(cache, i + 1, KvCache::pageTokens).granted(), "prefix KV failed");
      auto block = test::publishBlocks(cache, i + 1, prompts[i], KvCache::pageTokens);
      cache.endRequest(i + 1);
      cache.publishCompositeState(block, std::make_shared<TieredState>(control), false);
      require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
              "state was not demoted");
      prompts[i].push_back(9999);
    }
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress, "first demotion failed");
    tier.complete();
    require(cache.pollTransfers(), "first demotion did not finish");
    if (restored) {
      auto lookup = cache.lookup(prompts[0], {});
      cache.beginRequest(9);
      require(admitRestore(cache, 9, lookup).granted(), "disk prefix did not restore");
      tier.complete();
      require(cache.pollTransfers(), "restore did not finish");
      lookup = {};
      cache.endRequest(9);
    }
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.slots == 2,
            "second demotion did not fill quota");
    // While the tier is busy, a leaf that needs a write stays; the restored
    // prefix, which keeps its disk copy, still gives its page.
    if (restored)
      require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == 2 &&
                  cache.snapshot().kvTier.demotionsRefused == 1,
              "a busy tier held back the page of a leaf with a disk copy");
    require(!cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress, "busy tier did not wait");
    require(tier.slots == 2 && cache.snapshot().kvTier.diskBlocks == 2 &&
                cache.lookup(prompts[0], {}).kvBoundary == KvCache::pageTokens,
            "busy tier discarded a disk prefix without starting a write");
    tier.complete();
    require(cache.pollTransfers() && cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                tier.demotions == 3,
            "disk replacement did not resume once the tier had room");
    tier.complete();
    require(cache.pollTransfers(), "resumed demotion did not finish");
  }
}

// A denied admission makes room in one reclaim step: it evicts the leaves
// that cover the shortfall and no more, and stops once an evicted state has
// returned memory, which may let the retry grow the pool instead.
void testReclaimForPagesCoversTheShortfall() {
  for (bool withState : {false, true}) {
    test::TestKvStorage storage{16, 100, 1};
    storage.budgetPages = 8;
    KvPool pool{storage, 0};
    engine::Cache cache{pool, nullptr, nullptr};
    // Eight one-page leaves fill the budget. With states, each holds one, so
    // none is dead KV, and the first leaf's state is the oldest victim.
    for (uint32_t leaf = 0; leaf < 8; ++leaf) {
      cache.beginRequest(leaf + 1);
      require(admitTokens(cache, leaf + 1, 32).granted(), "leaf KV admission failed");
      const uint64_t block = test::publishBlocks(
          cache, leaf + 1, std::vector<uint32_t>(33, 100 + leaf), 32);
      cache.endRequest(leaf + 1);
      if (withState)
        cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
    }
    require(pool.freePageCount() == 0 && cache.snapshot().pool.pagesPrefix == 8,
            "fixture geometry changed");
    const CacheReclaimResult reclaimed =
        cache.reclaimForPages(3, CacheReclaimMode::KeepExtents, ReclaimClass::InUse);
    if (withState) {
      require(reclaimed.madeProgress && pool.freePageCount() == 0 &&
                  cache.snapshot().pool.pagesPrefix == 8 &&
                  cache.snapshot().stateCache.entries == 7,
              "the reclaim step went on evicting after a state returned memory");
    } else {
      require(reclaimed.madeProgress && pool.freePageCount() == 3 &&
                  cache.snapshot().pool.pagesPrefix == 5,
              "the reclaim step did not evict exactly the shortfall");
    }
  }
}

// A lookup touches the resident boundary of a chain with a disk suffix, not
// only its tip: the boundary is a leaf, and the reclaim that makes room for
// the restore must take an older one, not the block the restore extends.
void testLookupKeepsADiskChainsResidentBoundaryWarm() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage{4, 100, 1};
  KvPool pool{storage, 4};
  test::TestKvTier tier;
  engine::Cache cache{pool, &tier, nullptr};
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  std::vector<uint32_t> chain(65);
  for (uint32_t i = 0; i < chain.size(); ++i)
    chain[i] = 1000 + i;
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 64).granted(), "chain KV admission failed");
  const uint64_t leaf = test::publishBlocks(cache, 1, chain, 64);
  cache.endRequest(1);
  cache.publishCompositeState(leaf, std::make_shared<TieredState>(control), false);
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && cache.pollTransfers(),
          "the chain's state was not demoted");
  demoteLeaves(cache, tier, 1);
  // The chain is its resident root and its leaf on disk. A newer chain
  // follows.
  const std::vector<uint32_t> newer(33, 7);
  cache.beginRequest(2);
  require(admitTokens(cache, 2, 32).granted(), "newer KV admission failed");
  cache.publishCommittedBlocks(2, newer, 32, {});
  cache.endRequest(2);
  require(pool.freePageCount() == 2 && tier.demotions == 1,
          "fixture geometry changed");

  {
    const auto lookup = cache.lookup(chain, {});
    require(lookup.resumeBoundary() == 64, "the disk chain did not match");
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                pool.freePageCount() == 3 && tier.demotions == 1,
            "the reclaim took the chain's resident boundary");
  }
  require(cache.lookup(newer, {}).kvBoundary == 0 &&
              cache.lookup(chain, {}).resumeBoundary() == 64 &&
              cache.snapshot().kvTier.diskBlocks == 1,
          "the reclaim did not take the older leaf alone");
}

// A restore takes its pages before it pins its chain, and the chain's last
// resident block has only disk children, which makes it a leaf. The eviction
// that makes room must not take it: the restore adopts pages under it.
void testRestoreKeepsTheBlockItExtends() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              fixture.cache.pollTransfers(),
          "state was not demoted");
  demoteLeaves(fixture.cache, tier, 2);
  require(fixture.pool.freePageCount() == 2, "the last two blocks did not go to disk");
  {
    auto lookup = fixture.lookup(129);
    fixture.cache.beginRequest(2);
    require(admitRestore(fixture.cache, 2, lookup).granted(), "first restore denied");
    tier.complete();
    require(fixture.cache.pollTransfers() &&
                fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::None,
            "first restore did not land");
    lookup = {};
    fixture.cache.endRequest(2);
  }
  // The last block gives up its page again; the one before it keeps a disk
  // copy and has only a disk child.
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              fixture.pool.freePageCount() == 1,
          "the restored leaf did not drop its page");
  fixture.cache.beginRequest(3);
  require(admitTokens(fixture.cache, 3, 32).granted() && fixture.pool.freePageCount() == 0,
          "another request did not take the free page");
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(4);
  // Making room demotes a leaf: the restore waits for its copy and holds no
  // page meanwhile.
  const TokenAdmission waiting = admitRestore(fixture.cache, 4, lookup);
  require(waiting.failure == TokenAdmissionFailure::Pending &&
              fixture.cache.pageTable(4).pages.empty(),
          "a restore without a page did not wait for the room being made");
  tier.complete();
  require(fixture.cache.pollTransfers(), "the demotion making room did not land");
  fixture.cache.endRequest(3);
  require(admitRestore(fixture.cache, 4, lookup).granted(), "the retried restore was denied");
  const auto table = fixture.cache.pageTable(4);
  require(table.pages.size() == 4 &&
              std::none_of(table.pages.begin(), table.pages.end(),
                           [](uint32_t page) { return page == KvCache::noPage; }),
          "the restored chain lost a page");
  // Its copies land with the demotion queued before them.
  while (tier.inFlight()) {
    tier.complete();
    require(fixture.cache.pollTransfers(), "a transfer did not land");
  }
  require(fixture.cache.kvRestoreStatus(4) == KvRestoreStatus::None,
          "the retried restore did not land");
  lookup = {};
  fixture.cache.endRequest(4);
}

// A state published in RAM while its block's demotion is in flight keeps the
// block's page when the write lands: a state in RAM sits on resident KV.
void testDemotionKeepsThePageUnderANewState() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              fixture.cache.pollTransfers(),
          "state was not demoted");
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == 1,
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
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 8};
  test::TestKvTier tier;
  engine::Cache cache{pool, &tier, nullptr};
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  std::vector<uint32_t> prompt(129);
  for (uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = 1000 + i;
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 128).granted(), "prefix KV failed");
  const uint64_t last = test::publishBlocks(cache, 1, prompt, 128);
  const uint64_t third = cache.blockAt(1, 96);
  cache.endRequest(1);
  cache.publishCompositeState(last, std::make_shared<TieredState>(control), false);
  require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(), "state was not demoted");
  demoteLeaves(cache, tier, 3);
  // The restore reads the second block; the last two wait for their turn.
  tier.transferLimit = 1;
  auto lookup = cache.lookup(prompt, {});
  cache.beginRequest(2);
  require(admitRestore(cache, 2, lookup).granted() && tier.restores == 1,
          "restore was denied");
  cache.beginRequest(3);
  require(admitTokens(cache, 3, 97).granted() &&
              test::publishBlocks(cache, 3, prompt, 96) == third,
          "the prefill did not reach the block being restored");
  cache.publishCompositeState(third, std::make_shared<TieredState>(control), false);
  cache.endRequest(3);
  lookup = {};
  cache.endRequest(2);
  for (int i = 0; i < 2; ++i) {
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  require(tier.restores == 2 && tier.inFlight() == 0 && cache.snapshot().pool.pagesPrefix == 3,
          "the cancelled restore dropped the page under a state in RAM");
  control->capacity = control->slots;
  require(reclaimStateBytes(cache, reuse) == 100 &&
              cache.pollTransfers(),
          "the state in RAM was not written");
  lookup = cache.lookup(std::span<const uint32_t>(prompt).first(97), {});
  require(lookup.kvBoundary == 96 && lookup.state && lookup.state->kvBlock() == third &&
              !lookup.state->state()->residentBytes(),
          "the written state lost its block");
}

void testLargeSharedDiskRestore() {
  constexpr uint32_t pages = 4096;
  constexpr uint32_t tokens = pages * KvCache::pageTokens;
  test::TestKvStorage storage{pages, 100, 1};
  KvPool pool{storage, pages};
  test::TestKvTier tier;
  tier.capacity = pages;
  tier.transferLimit = 96;
  engine::Cache cache{pool, &tier, nullptr};
  std::vector<uint32_t> prompt(tokens, 17);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, tokens).granted(), "large prefix admission failed");
  const auto boundary = test::publishBlocks(cache, 1, prompt, tokens);
  cache.endRequest(1);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  cache.publishCompositeState(boundary, std::make_shared<TieredState>(control), false);
  require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
          "large state demotion failed");
  demoteLeaves(cache, tier, pages);
  require(pool.freePageCount() == pages && tier.demotions == pages,
          "large prefix was not fully on disk");
  prompt.push_back(18);
  auto lookup = cache.lookup(prompt, {});
  require(lookup.resumeBoundary() == tokens,
          "large disk prefix lookup lost its endpoint");
  for (uint64_t id = 2; id <= 5; ++id) {
    cache.beginRequest(id);
    require(admitRestore(cache, id, lookup).granted(), "large shared restore denied");
  }
  require(tier.restores == tier.transferLimit,
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

// Restores start in block order and stop at the first the tier refuses: a
// poll that finds the tier full asks it once, not once per waiting block,
// and each transfer that lands lets the next block start.
void testRefusedRestoresStopAtTheFirstRefusal() {
  constexpr uint32_t pages = 10;
  constexpr uint32_t tokens = pages * KvCache::pageTokens;
  test::TestKvStorage storage{pages, 100, 1};
  KvPool pool{storage, pages};
  test::TestKvTier tier;
  tier.capacity = pages;
  tier.transferLimit = 2;
  engine::Cache cache{pool, &tier, nullptr};
  std::vector<uint32_t> prompt(tokens, 19);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, tokens).granted(), "the disk chain's admission failed");
  const auto boundary = test::publishBlocks(cache, 1, prompt, tokens);
  cache.endRequest(1);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  cache.publishCompositeState(boundary, std::make_shared<TieredState>(control), false);
  require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
          "the chain's state demotion failed");
  demoteLeaves(cache, tier, pages);
  prompt.push_back(20);
  auto lookup = cache.lookup(prompt, {});
  cache.beginRequest(2);
  require(admitRestore(cache, 2, lookup).granted() && tier.restoreCalls == 2,
          "the restore asked the tier for more than it takes");
  for (int poll = 0; poll < 3; ++poll)
    static_cast<void>(cache.pollTransfers());
  require(tier.restoreCalls == 2, "polls asked a full tier for every waiting block");
  tier.transfers[tier.transfers.size() - 2]->ready = true;
  static_cast<void>(cache.pollTransfers());
  const std::span<const uint32_t> table = cache.pageTable(2).pages;
  const std::vector<uint32_t> chain(table.begin(), table.end());
  require(tier.restoreCalls == 3 &&
              std::ranges::equal(tier.restoredPages, std::span(chain).first(3)),
          "a landed restore did not start exactly the next block");
  for (uint32_t i = 0; i < pages && cache.kvRestoreStatus(2) == KvRestoreStatus::Pending; ++i) {
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  require(cache.kvRestoreStatus(2) == KvRestoreStatus::None &&
              std::ranges::equal(tier.restoredPages, chain),
          "the chain was not restored root first");
  lookup = {};
  cache.endRequest(2);
}

// Three prompts of three blocks on four-page extents: the first on pages 0
// to 2, the second on pages 3 to 5 and the third on pages 6 to 8, so the
// second and third lie across two extents each. Every page holds a value of
// its own. The first prompt's blocks are plain leaves; the others end in a
// state.
struct ExtentFixture {
  test::TestKvStorage storage;
  KvPool pool{storage, storage.pageCount()};
  engine::Cache cache;
  std::array<std::vector<uint32_t>, 3> prompts;
  // Each prompt's last block.
  std::array<uint64_t, 3> lastBlocks{};

  explicit ExtentFixture(uint32_t pages = 16, KvTier *tier = nullptr)
      : storage(pages, 100, 4), cache(pool, tier, nullptr) {
    for (uint32_t prompt = 0; prompt < 3; ++prompt) {
      for (uint32_t token = 0; token <= 3 * KvCache::pageTokens; ++token)
        prompts[prompt].push_back(1000 * (prompt + 1) + token);
      const uint64_t request = prompt + 1;
      cache.beginRequest(request);
      require(admitTokens(cache, request, 96).granted(),
              "extent fixture KV pages were not acquired");
      const PageTableView table = cache.pageTable(request);
      for (uint32_t block = 0; block < 3; ++block) {
        require(table.pages[block] == 3 * prompt + block,
                "extent fixture pages are not laid out in order");
        storage.content[table.pages[block]] = content(prompt, block);
      }
      lastBlocks[prompt] = test::publishBlocks(cache, request, prompts[prompt], 96);
      if (prompt)
        cache.publishCompositeState(lastBlocks[prompt], std::make_shared<TestState>(100), false);
      cache.endRequest(request);
    }
  }

  static uint64_t content(uint32_t prompt, uint32_t block) {
    return 10 * (prompt + 1) + block;
  }

  // Evicts the first prompt's three leaves, the oldest in the cache.
  void evictFirstPrompt() {
    for (uint32_t leaf = 0; leaf < 3; ++leaf) {
      require(cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress,
              "the first prompt's leaf was not evicted");
    }
    require(cache.snapshot().pool.pagesPrefix == 6,
            "eviction did not take exactly the first prompt");
  }
};

// A reclaim that releases extents returns free pages before it evicts: an
// empty extent as it is, then the pages scattered over the others, gathered
// by moving the only page of the emptiest extent. The request and the block
// on that page follow it.
void testCompactionReturnsFreePagesBeforeEvicting() {
  constexpr auto release = CacheReclaimMode::ReleaseExtents;
  ExtentFixture fixture;
  engine::Cache &cache = fixture.cache;
  auto lookup = cache.lookup(fixture.prompts[1], {});
  cache.beginRequest(4);
  require(lookup.state && admitRestore(cache, 4, lookup).granted(),
          "the second prompt was not restored");
  lookup = {};
  const uint64_t revision = cache.pageTable(4).revision;
  // Extent 0 keeps the second prompt's first page, extent 1 is full, extent
  // 2 keeps the third prompt's last page, extent 3 is empty.
  fixture.evictFirstPrompt();

  CacheReclaimResult step = cache.reclaimOne(release, ReclaimClass::InUse);
  require(step.madeProgress && step.reclaimedBytes == 400 &&
              fixture.storage.copies.empty(),
          "the empty extent was not released as it is");
  step = cache.reclaimOne(release, ReclaimClass::InUse);
  require(step.madeProgress && step.reclaimedBytes == 400 &&
              fixture.storage.copies.size() == 1 &&
              fixture.storage.copies[0].from == 3 && fixture.storage.copies[0].to == 9 &&
              cache.snapshot().pool.pagesPrefix == 6 &&
              cache.snapshot().stateCache.entries == 2 &&
              cache.snapshot().pool.pagesAllocated == 8 &&
              cache.snapshot().pool.extentCompactions == 1,
          "scattered free pages were not returned before an eviction");
  const PageTableView table = cache.pageTable(4);
  require(table.revision == revision + 1 && table.firstChanged == 0 &&
              table.pages[0] == 9 && table.pages[1] == 4 && table.pages[2] == 5 &&
              fixture.storage.content[9] == ExtentFixture::content(1, 0),
          "the request did not follow its moved page at a new table revision");
  cache.endRequest(4);
  lookup = cache.lookup(fixture.prompts[1], {});
  cache.beginRequest(5);
  require(lookup.state && admitRestore(cache, 5, lookup).granted() &&
              cache.pageTable(5).pages[0] == 9,
          "the cached block did not follow its moved page");
  lookup = {};

  // The free pages left cover no extent: the next step evicts.
  step = cache.reclaimOne(release, ReclaimClass::InUse);
  require(step.madeProgress && step.evictedState && step.reclaimedBytes == 0 &&
              cache.snapshot().stateCache.entries == 1 &&
              fixture.storage.copies.size() == 1,
          "a step with nothing to gather did not evict");
  cache.endRequest(5);
}

// The extent a step empties stays as the runway a pass keeps when it is the
// only empty one; the step made progress and returned nothing.
void testCompactionLeavesTheRunway() {
  ExtentFixture fixture(12);
  engine::Cache &cache = fixture.cache;
  fixture.evictFirstPrompt();
  const CacheReclaimResult step =
      cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse, false, true);
  const KvPoolSnapshot pool = cache.snapshot().pool;
  require(step.madeProgress && step.reclaimedBytes == 0 &&
              fixture.storage.copies.size() == 1 && pool.pagesAllocated == 12 &&
              pool.reclaimableBytes == 400,
          "the emptied extent did not stay as the runway");
  // A pass with a target keeps one empty extent and returns the others.
  ExtentFixture spare;
  spare.evictFirstPrompt();
  uint64_t released = spare.cache.releaseEmptyExtents(true);
  while (released < 400) {
    const CacheReclaimResult next =
        spare.cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse, false, true);
    require(next.madeProgress, "the pass made no progress");
    released += next.reclaimedBytes;
  }
  require(released == 400 &&
              spare.storage.copies.size() == 1 &&
              spare.cache.snapshot().pool.pagesAllocated == 12 &&
              spare.cache.snapshot().pool.reclaimableBytes == 400 &&
              spare.cache.snapshot().pool.pagesPrefix == 6,
          "a pass did not keep one runway and return the extent it emptied");
}

// Compaction re-points only the blocks on the pages it moves, and a block
// moved twice is found, and released, where it went last. The first
// prompt's state is on disk and its blocks are demoted, so they have no page
// to move.
void testCompactionFollowsOnlyTheMovedBlocks() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  constexpr auto release = CacheReclaimMode::ReleaseExtents;
  test::TestKvTier tier;
  ExtentFixture fixture(16, &tier);
  engine::Cache &cache = fixture.cache;
  KvPool &pool = fixture.pool;
  test::TestKvStorage &storage = fixture.storage;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const StateWriter write = [&](std::function<void()>) { return writeState(control); };
  require(cache.publishStateToDisk(fixture.lastBlocks[0], write, false) && cache.pollTransfers(),
          "the first prompt's state was not written");
  // Extent 0 keeps the first prompt's first block and the second prompt's
  // first block; extent 2 keeps the third prompt's last block alone.
  demoteLeaves(cache, tier, 2);
  require(cache.reclaimOne(release, ReclaimClass::InUse).reclaimedBytes == 400 &&
              cache.reclaimOne(release, ReclaimClass::InUse).madeProgress &&
              storage.copies.size() == 1 && storage.copies[0].from == 8 &&
              storage.copies[0].to == 1,
          "the third prompt's leaf did not move into extent 0");
  // The first prompt's first block goes to disk and the second prompt's
  // state and two upper blocks go, so extents 0 and 1 hold two pages each.
  demoteLeaves(cache, tier, 1);
  for (uint32_t victim = 0; victim < 3; ++victim)
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
            "the second prompt was not evicted");
  require(cache.reclaimOne(release, ReclaimClass::InUse).madeProgress &&
              storage.copies.size() == 3 && storage.copies[1].from == 1 &&
              storage.copies[1].to == 4 && storage.copies[2].from == 3 &&
              storage.copies[2].to == 5 && cache.snapshot().pool.extentCompactions == 2,
          "extent 0 was not emptied into extent 1");

  auto lookup = cache.lookup(fixture.prompts[2], {});
  cache.beginRequest(4);
  require(lookup.state && admitRestore(cache, 4, lookup).granted(),
          "the third prompt was not restored");
  const PageTableView table = cache.pageTable(4);
  require(table.pages[0] == 6 && table.pages[1] == 7 && table.pages[2] == 4,
          "the third prompt's leaf is not where it moved last");
  for (uint32_t block = 0; block < 3; ++block) {
    require(storage.content[table.pages[block]] == ExtentFixture::content(2, block),
            "a block's page does not hold its content");
  }
  lookup = {};
  cache.endRequest(4);
  // The demoted blocks still have no page: a lookup reads all three from
  // disk.
  lookup = cache.lookup(fixture.prompts[0], {});
  cache.beginRequest(5);
  require(lookup.state && admitRestore(cache, 5, lookup).granted(),
          "the first prompt was not restored");
  for (uint32_t i = 0; i < 3 && cache.kvRestoreStatus(5) == KvRestoreStatus::Pending; ++i) {
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  require(cache.kvRestoreStatus(5) == KvRestoreStatus::None && tier.restores == 3,
          "a demoted block was given a page");
  lookup = {};
  cache.endRequest(5);

  // The oldest leaves go: the second prompt's first block, the third
  // prompt's state, then its leaf, which frees the page it moved to last.
  uint32_t freePages = pool.freePageCount();
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              pool.freePageCount() == freePages + 1 &&
              cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
          "the second prompt's moved block did not free its page");
  freePages = pool.freePageCount();
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              pool.freePageCount() == freePages + 1,
          "the twice-moved leaf did not free the page it moved to last");
}

// A pass that evicts everything moves pages only once it has: the pages a
// request still holds are gathered into one extent, and nothing is copied
// and then evicted.
void testEvictAllGathersWhatRequestsHold() {
  ExtentFixture fixture;
  engine::Cache &cache = fixture.cache;
  auto lookup = cache.lookup(fixture.prompts[1], {});
  cache.beginRequest(4);
  require(lookup.state && admitRestore(cache, 4, lookup).granted(),
          "the second prompt was not restored");
  lookup = {};
  fixture.evictFirstPrompt();
  // Three extents (1200); the third prompt's pages are gone before extent
  // 0's page moves into their place.
  require(cache.evictAll().reclaimedBytes == 1200 &&
              fixture.storage.copies.size() == 1 &&
              fixture.storage.copies[0].from == 3 &&
              fixture.storage.copies[0].to / 4 == 1,
          "an evict-all pass moved a page it then evicted, or left extents");
  const CacheSnapshot stats = cache.snapshot();
  const PageTableView table = cache.pageTable(4);
  require(stats.pool.pagesAllocated == 4 && stats.pool.pagesFree == 1 &&
              stats.pool.pagesPrefix == 3 && stats.stateCache.entries == 0 &&
              table.pages[0] / 4 == 1 && table.pages[1] == 4 && table.pages[2] == 5 &&
              fixture.storage.content[table.pages[0]] == ExtentFixture::content(1, 0),
          "an evict-all pass did not gather the request's pages into one extent");
  cache.endRequest(4);
}

// A page a restore reads into stays where it is: the extent that holds it is
// passed over although it has the fewest pages, and the next one moves.
void testCompactionLeavesAPageBeingRestored() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  constexpr auto release = CacheReclaimMode::ReleaseExtents;
  test::TestKvStorage storage{16, 100, 4};
  KvPool pool{storage, 16};
  test::TestKvTier tier;
  engine::Cache cache{pool, &tier, nullptr};
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const auto prompt = [](uint32_t first, uint32_t blocks) {
    std::vector<uint32_t> tokens(blocks * KvCache::pageTokens + 1);
    for (uint32_t i = 0; i < tokens.size(); ++i)
      tokens[i] = first + i;
    return tokens;
  };
  // One block whose state and page both go to disk.
  const std::vector<uint32_t> onDisk = prompt(1000, 1);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 32).granted(), "disk prompt KV failed");
  uint64_t last = test::publishBlocks(cache, 1, onDisk, 32);
  cache.endRequest(1);
  cache.publishCompositeState(last, std::make_shared<TieredState>(control), false);
  require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(), "state was not demoted");
  demoteLeaves(cache, tier, 1);
  // Three plain blocks on pages 0 to 2.
  const std::vector<uint32_t> plain = prompt(2000, 3);
  cache.beginRequest(2);
  require(admitTokens(cache, 2, 96).granted(), "plain prompt KV failed");
  cache.publishCommittedBlocks(2, plain, 96, {});
  cache.endRequest(2);
  // The restore takes extent 0's last page and stays in flight.
  auto lookup = cache.lookup(onDisk, {});
  cache.beginRequest(3);
  require(lookup.state && admitRestore(cache, 3, lookup).granted() &&
              cache.pageTable(3).pages[0] == 3 && tier.restores == 1 &&
              cache.kvRestoreStatus(3) == KvRestoreStatus::Pending,
          "the restore did not start on extent 0's last page");
  // Three blocks on pages 4 to 6 under a state in RAM.
  const std::vector<uint32_t> kept = prompt(3000, 3);
  cache.beginRequest(4);
  require(admitTokens(cache, 4, 96).granted() && cache.pageTable(4).pages[0] == 4,
          "kept prompt KV failed");
  for (uint32_t block = 0; block < 3; ++block)
    storage.content[4 + block] = 40 + block;
  last = test::publishBlocks(cache, 4, kept, 96);
  cache.publishCompositeState(last, std::make_shared<TestState>(100), false);
  cache.endRequest(4);
  // The plain leaves go: extent 0 keeps only the page being read into.
  for (uint32_t leaf = 0; leaf < 3; ++leaf)
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
            "plain leaf was not evicted");
  require(cache.reclaimOne(release, ReclaimClass::InUse).reclaimedBytes == 400 &&
              cache.reclaimOne(release, ReclaimClass::InUse).reclaimedBytes == 400 &&
              storage.copies.empty(),
          "the empty extents were not released first");

  const CacheReclaimResult step = cache.reclaimOne(release, ReclaimClass::InUse);
  require(step.madeProgress && step.reclaimedBytes == 400 && storage.copies.size() == 3 &&
              storage.copies[0].from == 4 && storage.copies[0].to == 0 &&
              storage.copies[1].from == 5 && storage.copies[1].to == 1 &&
              storage.copies[2].from == 6 && storage.copies[2].to == 2 &&
              cache.pageTable(3).pages[0] == 3,
          "compaction moved a page a restore reads into");
  tier.complete();
  require(cache.pollTransfers() && cache.kvRestoreStatus(3) == KvRestoreStatus::None,
          "the restore did not land on its page");
  lookup = {};
  cache.endRequest(3);
  lookup = cache.lookup(kept, {});
  cache.beginRequest(5);
  require(lookup.state && admitRestore(cache, 5, lookup).granted(), "kept prompt was lost");
  const PageTableView table = cache.pageTable(5);
  require(table.pages[0] == 0 && table.pages[1] == 1 && table.pages[2] == 2 &&
              storage.content[0] == 40 && storage.content[1] == 41 && storage.content[2] == 42,
          "the kept prompt's blocks did not follow their pages");
  lookup = {};
  cache.endRequest(5);
}

// So does a page being written to disk: its extent is passed over, and the
// pages of the chain above it move next to it.
void testCompactionLeavesAPageBeingDemoted() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  constexpr auto release = CacheReclaimMode::ReleaseExtents;
  test::TestKvStorage storage{16, 100, 4};
  KvPool pool{storage, 16};
  test::TestKvTier tier;
  engine::Cache cache{pool, &tier, nullptr};
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  // One plain block on page 0, then four blocks on pages 1 to 4 whose state
  // is on disk.
  std::vector<uint32_t> plain(33);
  std::vector<uint32_t> chained(129);
  for (uint32_t i = 0; i < chained.size(); ++i) {
    if (i < plain.size())
      plain[i] = 1000 + i;
    chained[i] = 2000 + i;
  }
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 32).granted(), "plain block KV failed");
  cache.publishCommittedBlocks(1, plain, 32, {});
  cache.endRequest(1);
  cache.beginRequest(2);
  require(admitTokens(cache, 2, 128).granted() && cache.pageTable(2).pages[3] == 4,
          "chain KV failed");
  for (uint32_t block = 0; block < 4; ++block)
    storage.content[1 + block] = 20 + block;
  const uint64_t last = test::publishBlocks(cache, 2, chained, 128);
  cache.endRequest(2);
  cache.publishCompositeState(last, std::make_shared<TieredState>(control), false);
  require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(), "state was not demoted");
  // The plain block goes, then the chain's leaf starts to be written.
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == 1 &&
              cache.snapshot().pool.pagesPrefix == 4,
          "the chain's leaf did not start its demotion");
  require(cache.reclaimOne(release, ReclaimClass::InUse).reclaimedBytes == 400 &&
              cache.reclaimOne(release, ReclaimClass::InUse).reclaimedBytes == 400 &&
              storage.copies.empty(),
          "the empty extents were not released first");

  const CacheReclaimResult step = cache.reclaimOne(release, ReclaimClass::InUse);
  require(step.madeProgress && step.reclaimedBytes == 400 && storage.copies.size() == 3 &&
              storage.copies[0].from == 1 && storage.copies[0].to == 5 &&
              storage.copies[1].from == 2 && storage.copies[1].to == 6 &&
              storage.copies[2].from == 3 && storage.copies[2].to == 7,
          "compaction moved a page being written to disk");
  tier.complete();
  require(cache.pollTransfers() && cache.snapshot().pool.pagesFree == 1,
          "the demotion did not land and free its page");
  auto lookup = cache.lookup(chained, {});
  cache.beginRequest(3);
  require(lookup.state && admitRestore(cache, 3, lookup).granted(), "chain was lost");
  const PageTableView table = cache.pageTable(3);
  require(table.pages[0] == 5 && table.pages[1] == 6 && table.pages[2] == 7 &&
              table.pages[3] == 4 && storage.content[5] == 20 && storage.content[6] == 21 &&
              storage.content[7] == 22,
          "the chain's blocks did not follow their pages");
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  lookup = {};
  cache.endRequest(3);
}

// Caches a request's chain of whole pages: its prompt, one token past the
// last page, and its blocks, root first. With running the request stays
// unfinished, so its chain stays active, as a running request's does.
std::pair<std::vector<uint32_t>, std::vector<uint64_t>>
cacheChain(engine::Cache &cache, uint64_t id, uint32_t pages, bool running = false) {
  std::vector<uint32_t> prompt(pages * KvCache::pageTokens + 1);
  for (uint32_t row = 0; row < prompt.size(); ++row)
    prompt[row] = static_cast<uint32_t>(1000 * id + row);
  cache.beginRequest(id);
  require(admitTokens(cache, id, pages * KvCache::pageTokens).granted(),
          "fixture pages were not acquired");
  cache.publishCommittedBlocks(id, prompt, pages * KvCache::pageTokens, {});
  std::vector<uint64_t> blocks;
  for (uint32_t page = 1; page <= pages; ++page)
    blocks.push_back(cache.blockAt(id, page * KvCache::pageTokens));
  if (!running)
    cache.endRequest(id);
  return {std::move(prompt), std::move(blocks)};
}

// A state an unfinished request uses goes after every other state and KV
// leaf, newer or not, and still goes once nothing else is left.
void testStateInUseGoesLast() {
  CacheFixture fixture;
  fixture.publish(0);
  StateUse use = fixture.cache.useState(fixture.blocks[0]);
  fixture.publish(1);
  // The lookup makes the chain's leaf and the other state newer.
  require(fixture.lookup(129).resumeBoundary() == 64 &&
              fixture.cache.snapshot().stateCache.inUse == 1,
          "the state was not marked in use");
  while (fixture.cache.snapshot().pool.pagesPrefix > 1) {
    require(
        fixture.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress &&
            fixture.cache.stateResident(fixture.blocks[0]),
        "the state in use left before the other state and KV");
  }
  require(fixture.cache.snapshot().stateCache.entries == 1, "the other state survived");
  const CacheReclaimResult last =
      fixture.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse);
  const auto snapshot = fixture.cache.snapshot().stateCache;
  require(last.madeProgress && snapshot.entries == 0 && snapshot.inUse == 1 &&
              snapshot.inUseEvictions == 1,
          "the last state in use could not be reclaimed");
  use.reset();
  require(fixture.cache.snapshot().stateCache.inUse == 0, "the mark outlived its handle");
}

// The KV a state in use restores through goes with that state, after
// ordinary states and KV: a newer ordinary state goes before the older leaf
// of a state in use that only the disk holds.
void testKvAStateInUseNeedsGoesLast() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage{7, 100, 1};
  storage.budgetPages = 6;
  KvPool pool{storage, 6};
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const std::vector<uint64_t> used = cacheChain(cache, 1, 2).second;
  StateUse use = cache.useState(used[1]);
  cache.publishCompositeState(used[1], std::make_shared<TieredState>(control), false);
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && cache.pollTransfers() &&
              !cache.stateResident(used[1]),
          "the state in use was not written");
  const std::vector<uint64_t> ordinary = cacheChain(cache, 2, 2).second;
  cache.publishCompositeState(ordinary[1], std::make_shared<TestState>(100), false);
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              !cache.stateResident(ordinary[1]) && tier.demotions == 0,
          "the KV a state in use needs went before an ordinary state");
}

// Marks count per block: requests sharing it each hold one, a mark taken
// before the publication applies to it, and so does one on a block whose
// state left and was published again. Only a publication that is itself in
// use displaces a state in use, after every other state; ordinary and
// optional publications never do. The chain's request runs, so no KV
// competes with the states.
void testStateUsesAreCountedPerBlock() {
  CacheFixture fixture(nullptr, true);
  const auto resident = [&](uint32_t block) {
    return fixture.cache.stateResident(fixture.blocks[block]);
  };
  StateUse early = fixture.cache.useState(fixture.blocks[0]);
  fixture.publish(0);
  StateUse shared = fixture.cache.useState(fixture.blocks[0]);
  early.reset();
  require(fixture.cache.snapshot().stateCache.inUse == 1 &&
              !fixture.cache.reclaimOneState(true, fixture.blocks[1], true) &&
              !fixture.cache.reclaimOneState(false, fixture.blocks[1], true) && resident(0),
          "a shared mark was lost, or an ordinary publication took a state in use");
  fixture.publish(1);
  StateUse publishing = fixture.cache.useState(fixture.blocks[2]);
  require(fixture.cache.reclaimOneState(false, fixture.blocks[2], true) && resident(0) &&
              !resident(1),
          "recycling took the state in use before an ordinary one");
  require(!fixture.cache.reclaimOneState(true, fixture.blocks[2], true) && resident(0),
          "an optional publication at a used block took a state in use");
  require(fixture.cache.reclaimOneState(false, fixture.blocks[2], true) && !resident(0) &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 1,
          "a publication in use could not recycle the last state in use");
  fixture.publish(0);
  require(!fixture.cache.reclaimOneState(false, fixture.blocks[1], true) && resident(0),
          "a state published again at a used block was not in use");
  shared.reset();
  require(fixture.cache.snapshot().stateCache.inUse == 1 &&
              fixture.cache.reclaimOneState(false, fixture.blocks[1], true) && !resident(0),
          "a released block stayed in use");
}

// The KV above a state is in use only while that state is: once a state
// marked after its publication has left, and its mark is released, its chain
// goes by recency again, before a newer ordinary state.
void testKvInUseFollowsItsState() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage{7, 100, 1};
  storage.budgetPages = 6;
  KvPool pool{storage, 6};
  engine::Cache cache(pool, nullptr, nullptr);
  const std::vector<uint64_t> used = cacheChain(cache, 1, 2).second;
  cache.publishCompositeState(used[1], std::make_shared<TestState>(100), false);
  StateUse use = cache.useState(used[1]);
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              cache.snapshot().stateCache.entries == 0 &&
              cache.snapshot().stateCache.inUseEvictions == 1,
          "the state in use did not go once nothing else was left");
  use.reset();
  const std::vector<uint64_t> ordinary = cacheChain(cache, 2, 2).second;
  cache.publishCompositeState(ordinary[1], std::make_shared<TestState>(100), false);
  for (uint32_t leaf = 0; leaf < 2; ++leaf)
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
            "the cache could not be reclaimed");
  require(cache.stateResident(ordinary[1]) && cache.snapshot().pool.pagesPrefix == 2,
          "the KV of a state that left stayed in use");
}

// A finished request stamps only the replay point its conversation resumes
// from, when its use ends: the older points on its chain keep their age and
// go before another conversation's newer state.
void testEndRequestStampsOnlyTheReplayPoint() {
  CacheFixture fixture(nullptr, true);
  fixture.storage.budgetPages = 5;
  fixture.publish(0);
  const uint64_t other = cacheChain(fixture.cache, 2, 1).second[0];
  fixture.cache.publishCompositeState(other, std::make_shared<TestState>(100), false);
  StateUse use = fixture.cache.useState(fixture.blocks[2]);
  fixture.publish(2);
  fixture.cache.endRequest(1);
  use.reset();
  const auto resident = [&](uint64_t block) { return fixture.cache.stateResident(block); };
  // Dead KV goes between them: each step here ends with one state fewer.
  const auto evictState = [&] {
    const uint32_t entries = fixture.cache.snapshot().stateCache.entries;
    while (fixture.cache.snapshot().stateCache.entries == entries)
      require(fixture.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse)
                  .madeProgress,
              "the cache could not be reclaimed");
  };
  evictState();
  require(!resident(fixture.blocks[0]) && resident(other) && resident(fixture.blocks[2]),
          "the finished request's older point did not go first");
  evictState();
  require(!resident(other) && resident(fixture.blocks[2]),
          "the replay point went before another conversation's older state");
  evictState();
  require(fixture.cache.snapshot().stateCache.entries == 0, "the replay point did not go last");
}

// The resume-point floor keeps the newest ordinary publication: a newer
// finished point is kept over an older one in use, and states in use still
// go after the rest. A state in use is never the resume point, even when it
// is newer than every ordinary one: its class protects it, and a speculative
// shrink takes it before the ordinary point.
void testKeepResumePointKeepsTheNewestOrdinaryPublication() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  CacheFixture fixture;
  fixture.publish(0);
  StateUse running = fixture.cache.useState(fixture.blocks[1]);
  fixture.publish(1);
  fixture.publish(2);
  // Lookups refresh what they find, so the order is read without them.
  const auto held = [&](uint32_t block) {
    return fixture.cache.stateResident(fixture.blocks[block]);
  };
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse, true).madeProgress &&
              fixture.cache.snapshot().pool.pagesPrefix == 3,
          "the older KV leaf did not go first");
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse, true).madeProgress && !held(0) &&
              held(1) && held(2),
          "the older ordinary state did not go before the state in use");
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse, true).madeProgress && !held(1) &&
              held(2) && fixture.cache.snapshot().stateCache.inUseEvictions == 1,
          "the state in use outranked the newer resume point");
  require(!fixture.cache.reclaimOne(reuse, ReclaimClass::InUse, true).madeProgress && held(2),
          "the speculative shrink took the resume point");
  // A state in use newer than every ordinary one goes; the ordinary point
  // stays.
  CacheFixture newer;
  newer.publish(0);
  StateUse newest = newer.cache.useState(newer.blocks[2]);
  newer.publish(2);
  reclaimEverything(newer.cache, true);
  const auto states = newer.cache.snapshot().stateCache;
  require(states.entries == 1 && states.inUseEvictions == 1 &&
              newer.cache.stateResident(newer.blocks[0]) &&
              !newer.cache.stateResident(newer.blocks[2]),
          "the speculative shrink kept a state in use over the newest ordinary publication");
}

// A state in use is reusable: a checkpoint there becomes ordinary, and one
// published there is ordinary.
void testStatesInUseAreOrdinary() {
  CacheFixture fixture;
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TestState>(100), true);
  const StateCheckpoint checkpoint = fixture.cache.checkpointState(fixture.blocks[1]);
  StateUse use = fixture.cache.useState(fixture.blocks[1]);
  require(fixture.cache.retireCheckpointState(checkpoint) &&
              fixture.cache.snapshot().stateCache.entries == 1 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 0,
          "rolling retirement erased a state in use");
  StateUse later = fixture.cache.useState(fixture.blocks[2]);
  fixture.cache.publishCompositeState(fixture.blocks[2], std::make_shared<TestState>(100), true);
  require(!fixture.cache.checkpointState(fixture.blocks[2]) &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 0,
          "a checkpoint published at a used block stayed disposable");
}

// States in use and the KV they need wait for transfers in flight, and a
// request short of pages waits with them: what those return may be enough.
// Once they have landed, the state in use goes with its leaf.
void testInUseWaitsForTransfersInFlight() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, 0, false).made, "the first state was not written");
  // Only the transfer holds this state back: nothing needs to write it.
  StateUse use = fixture.cache.useState(fixture.blocks[1]);
  fixture.publish(1);
  CacheReclaimResult result;
  while ((result = fixture.cache.reclaimOne(reuse, ReclaimClass::InUse)).madeProgress) {
  }
  require(result.pending && fixture.cache.stateResident(fixture.blocks[1]),
          "a state in use went while a transfer was in flight");
  // Two pages are free; the third has to come from the leaf in use.
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 96).failure == TokenAdmissionFailure::Pending &&
              fixture.cache.stateResident(fixture.blocks[1]),
          "a request short of pages took the KV in use while a transfer was in flight");
  control->ready = true;
  require(fixture.cache.pollTransfers() && admitTokens(fixture.cache, 2, 96).granted() &&
              !fixture.cache.stateResident(fixture.blocks[1]) &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 1,
          "the state in use was not taken once the transfer landed");
  fixture.cache.endRequest(2);
}

// A lane that takes a cached state's buffers is running work: a state in use
// goes after every other state, newer or not, and once no transfer in flight
// can return what is needed first.
void testLaneTakesStatesInUseLast() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  StateUse use = fixture.cache.useState(fixture.blocks[0]);
  fixture.publish(0);
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  require(fixture.cache.evictableStates(ReclaimClass::InUse) == 2 &&
              fixture.cache.reclaimStateForLane(ReclaimClass::InUse).madeProgress &&
              fixture.cache.stateResident(fixture.blocks[0]) &&
              !fixture.cache.stateResident(fixture.blocks[1]),
          "a lane took the state in use before a newer ordinary state");
  // The ordinary state's write is in flight.
  const CacheReclaimResult waiting = fixture.cache.reclaimStateForLane(ReclaimClass::InUse);
  require(!waiting.madeProgress && waiting.pending &&
              fixture.cache.stateResident(fixture.blocks[0]),
          "a lane took a state in use while a transfer was in flight");
  control->ready = true;
  require(fixture.cache.pollTransfers() &&
              fixture.cache.reclaimStateForLane(ReclaimClass::InUse).madeProgress &&
              fixture.cache.evictableStates(ReclaimClass::InUse) == 0 &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 1,
          "a lane could not take the last state in use");
}

// Work that a resident lane holds back reclaims up to the ordinary class:
// the ordinary state and the stateless KV go, and then it stops short of the
// state in use and the KV that state restores through, with nothing pending.
// Of the four idle pages, only the stateless leaves' two are reusable to it.
// Running work takes that state once it is all that is left, and its KV is
// then ordinary.
void testOrdinaryClassStopsShortOfStatesInUse() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  CacheFixture fixture;
  fixture.publish(0);
  StateUse use = fixture.cache.useState(fixture.blocks[1]);
  fixture.publish(1);
  require(fixture.cache.reusablePages(ReclaimClass::Ordinary) == 2 &&
              fixture.cache.reusablePages(ReclaimClass::InUse) == 4,
          "work held back counted the KV in use as reusable");
  uint32_t steps = 0;
  CacheReclaimResult step;
  while ((step = fixture.cache.reclaimOne(reuse, ReclaimClass::Ordinary)).madeProgress)
    ++steps;
  require(steps == 3 && !step.pending && fixture.cache.snapshot().stateCache.entries == 1 &&
              fixture.cache.stateResident(fixture.blocks[1]) &&
              fixture.cache.snapshot().pool.pagesPrefix == 2,
          "work held back took what is in use, or left an ordinary victim");
  require(!fixture.cache.reclaimStateForLane(ReclaimClass::Ordinary).madeProgress &&
              fixture.cache.evictableStates(ReclaimClass::Ordinary) == 0 &&
              fixture.cache.evictableStates(ReclaimClass::InUse) == 1,
          "a lane held back could take the state in use");
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              !fixture.cache.stateResident(fixture.blocks[1]) &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 1 &&
              fixture.cache.reusablePages(ReclaimClass::Ordinary) == 4,
          "running work did not take the state in use");
}

// A publication in use never drops a state in use for a busy write slot:
// while the write in flight holds it nothing is recycled, and that
// publication goes without. Once the slot is free, a publication in use
// takes the state by writing it. The chain's request runs, so no KV is
// left to make room with either.
void testRecycleNeverDropsAStateInUseForTheWriteSlot() {
  CacheFixture fixture(nullptr, true);
  auto control = std::make_shared<TransferControl>();
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, 0, false).made, "the first state was not written");
  StateUse held = fixture.cache.useState(fixture.blocks[1]);
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  StateUse publishing = fixture.cache.useState(fixture.blocks[2]);
  require(!fixture.cache.reclaimOneState(false, fixture.blocks[2], true) &&
              fixture.cache.stateResident(fixture.blocks[1]) &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 0,
          "a state in use was dropped for a busy write slot");
  control->ready = true;
  require(fixture.cache.pollTransfers() &&
              fixture.cache.reclaimOneState(false, fixture.blocks[2], true) &&
              fixture.lookup(65).resumeBoundary() == 64 &&
              fixture.cache.snapshot().stateCache.offloads == 2,
          "the state in use was not written for the publication in use");
}

// Disk replacement follows the class of the copy it makes room for: the
// write of a state in use gives up ordinary copies first and the oldest copy
// in use last, while an ordinary state's write never replaces a copy in use;
// that state is dropped, as when the quota holds nothing to give. The
// chain's request runs, so only states compete.
void testDiskReplacementTakesStatesInUseLast() {
  CacheFixture fixture(nullptr, true);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  // Probes find the copies without refreshing them.
  const auto cached = [&](uint32_t tokens) {
    return fixture.cache.probe(std::span<const uint32_t>(fixture.prompt).first(tokens), {})
        .cachedTokens();
  };
  const auto inUseEvictions = [&] {
    return fixture.cache.snapshot().stateCache.inUseEvictions;
  };
  StateUse oldest = fixture.cache.useState(fixture.blocks[0]);
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, fixture.blocks[0], true) &&
              fixture.cache.pollTransfers(),
          "the oldest state in use was not written");
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers() &&
              control->slots == 2,
          "the ordinary state was not written");
  // The quota is full: a write in use gives up the ordinary copy.
  StateUse older = fixture.cache.useState(fixture.blocks[2]);
  fixture.cache.publishCompositeState(fixture.blocks[2], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, fixture.blocks[2], true) &&
              fixture.cache.pollTransfers() && cached(65) == 32 && cached(97) == 96 &&
              inUseEvictions() == 0,
          "a write in use replaced a copy in use before an ordinary one");
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, 0, false) && cached(129) == 96 &&
              inUseEvictions() == 0,
          "an ordinary write replaced the only copy of a state in use");
  StateUse newer = fixture.cache.useState(fixture.blocks[3]);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(
      fixture.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress &&
          fixture.cache.pollTransfers() && cached(33) == 0 && cached(97) == 96 &&
          cached(129) == 128 && inUseEvictions() == 1,
      "the write of a state in use did not replace the oldest copy in use");
}

// A request short of pages takes the KV leaf a state in use needs after a
// newer leaf nothing in use needs.
void testPageShortageTakesLeavesInUseLast() {
  test::TestKvStorage storage{7, 100, 1};
  storage.budgetPages = 6;
  KvPool pool{storage, 6};
  engine::Cache cache(pool, nullptr, nullptr);
  const auto [usedPrompt, used] = cacheChain(cache, 1, 2);
  // The older chain's leaf holds the state in use.
  StateUse use = cache.useState(used[1]);
  cache.publishCompositeState(used[1], std::make_shared<TestState>(100), false);
  static_cast<void>(cacheChain(cache, 2, 2));
  cache.beginRequest(3);
  require(admitTokens(cache, 3, 96).granted() && cache.lookup(usedPrompt, {}).resumeBoundary() == 64,
          "a request short of pages took the leaf a state in use needs before a newer one");
  cache.endRequest(3);
}

// The parent a disk-only state in use restores through goes with the states
// in use: its write may displace their copies, so a quota holding only them
// never keeps it in RAM.
void testKvAStateInUseNeedsNeverPinsMemory() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvTier tier;
  tier.capacity = 1;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  StateUse use = fixture.cache.useState(fixture.blocks[3]);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              fixture.cache.pollTransfers(),
          "the state in use was not written");
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == 1,
          "the leaf of the state in use was not written");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1,
          "the leaf did not land");
  // Block 2 is a leaf the state in use needs; the quota holds only that
  // state's KV copy.
  require(fixture.cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              fixture.cache.snapshot().pool.pagesPrefix == 2 &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 1,
          "the parent a state in use needs stayed in RAM");
}

// An ordinary leaf's write never displaces the copies a state in use needs,
// even when they are older. While a lookup holds the ordinary copy below the
// leaf, the leaf stays; then the ordinary copies give way, and the leaf,
// which nothing needs any more, goes without one.
void testOrdinaryDemotionKeepsCopiesInUse() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage{7, 100, 1};
  storage.budgetPages = 6;
  KvPool pool{storage, 6};
  test::TestKvTier tier;
  tier.capacity = 3;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const auto settle = [&](const char *message) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress, message);
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  };
  // The older chain goes to disk whole: its state in use, then its KV.
  const auto [usedPrompt, used] = cacheChain(cache, 1, 2);
  StateUse use = cache.useState(used[1]);
  cache.publishCompositeState(used[1], std::make_shared<TieredState>(control), false);
  for (uint32_t step = 0; step < 3; ++step)
    settle("the chain in use did not go to disk");
  // The newer chain's tail and its ordinary state follow; the quota is full.
  const auto [ordinaryPrompt, ordinary] = cacheChain(cache, 2, 3);
  cache.publishCompositeState(ordinary[2], std::make_shared<TieredState>(control), false);
  for (uint32_t step = 0; step < 2; ++step)
    settle("the ordinary tail did not go to disk");
  require(tier.slots == 3 && cache.snapshot().pool.pagesPrefix == 2,
          "fixture did not fill the quota");
  // The parent of that tail cannot be written without displacing a copy.
  // While a lookup holds the ordinary state, only copies in use could go.
  {
    const CacheLookup held = cache.lookup(ordinaryPrompt, {});
    require(held.state && !cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
                cache.snapshot().stateCache.inUseEvictions == 0,
            "an ordinary leaf's write displaced a copy in use");
  }
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              cache.snapshot().pool.pagesPrefix == 1 && tier.slots == 2 &&
              cache.snapshot().stateCache.entries == 1 &&
              cache.snapshot().stateCache.inUseEvictions == 0,
          "an ordinary leaf's write displaced the copies a state in use needs");
  require(cache.probe(usedPrompt, {}).cachedTokens() == 64,
          "the state in use lost the KV it restores through");
}

// Promotion is optional: without a free slot it never takes a state in use
// for one, and the restored state stays on disk.
void testPromotionSkipsForAStateInUse() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(fixture.cache.reclaimOneState(false, 0, false) && fixture.cache.pollTransfers(),
          "state was not demoted");
  StateUse use = fixture.cache.useState(fixture.blocks[1]);
  fixture.publish(1);
  auto lookup = fixture.lookup(129);
  PromotionTicket ticket;
  ticket.available = false;
  fixture.cache.promoteState(lookup, ticket);
  const auto stats = fixture.cache.snapshot().stateCache;
  require(ticket.copies == 1 && stats.promotionsSkipped == 1 && stats.inUseEvictions == 0 &&
              fixture.cache.stateResident(fixture.blocks[1]),
          "promotion took a state in use for its slot");
}

// KV no state restores through saves no prefill: a newer conversation's dead
// tail goes before an older conversation's state.
void testDeadKvGoesBeforeOlderStates() {
  test::TestKvStorage storage{6, 100, 1};
  KvPool pool{storage, 6};
  engine::Cache cache(pool, nullptr, nullptr);
  const std::vector<uint64_t> older = cacheChain(cache, 1, 2).second;
  cache.publishCompositeState(older[1], std::make_shared<TestState>(100), false);
  const std::vector<uint64_t> newer = cacheChain(cache, 2, 3).second;
  cache.publishCompositeState(newer[1], std::make_shared<TestState>(100), false);
  require(cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress &&
              cache.snapshot().pool.pagesPrefix == 4 && cache.stateResident(older[1]) &&
              cache.stateResident(newer[1]),
          "an older state went before a newer conversation's dead KV");
}

// A demotion the busy tier cannot start keeps only the leaves that need one:
// a newer leaf with a disk copy still gives its page, before a newer state.
void testPendingDemotionLeavesOtherKvOpen() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 8};
  test::TestKvTier tier;
  tier.transferLimit = 1;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 3;
  // Three one-page chains whose states only the disk holds: each leaf needs
  // a demotion to go.
  std::vector<std::vector<uint32_t>> prompts;
  for (uint64_t id = 1; id <= 3; ++id) {
    auto [prompt, blocks] = cacheChain(cache, id, 1);
    cache.publishCompositeState(blocks[0], std::make_shared<TieredState>(control), false);
    require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
            "a state was not written");
    prompts.push_back(std::move(prompt));
  }
  // The oldest leaf is written while a lookup holds it, so it keeps its page
  // beside its new disk copy; the next leaf's demotion stays in flight.
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress, "a demotion did not start");
  {
    const CacheLookup held = cache.lookup(prompts[0], {});
    tier.complete();
    require(cache.pollTransfers(), "the demotion did not land");
  }
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && !tier.canDemote(),
          "the second demotion did not fill the tier");
  const uint64_t state = cacheChain(cache, 4, 1).second[0];
  cache.publishCompositeState(state, std::make_shared<TestState>(100), false);
  const CacheSnapshot before = cache.snapshot();
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress,
          "nothing was reclaimed while a demotion had to wait");
  const CacheSnapshot after = cache.snapshot();
  require(after.pool.pagesPrefix == before.pool.pagesPrefix - 1 &&
              after.kvTier.diskBlocks == before.kvTier.diskBlocks &&
              cache.lookup(prompts[0], {}).kvBoundary == KvCache::pageTokens &&
              cache.stateResident(state) && after.kvTier.demotionsRefused == 1,
          "a demotion that had to wait closed the KV whose page frees at once");
}

// A copy that failed is not the protection giving way.
void testInUseEvictionsCountChoices() {
  CacheFixture fixture;
  StateUse use = fixture.cache.useState(fixture.blocks[1]);
  fixture.publish(1);
  const auto state = fixture.lookup(65).state->state();
  fixture.cache.discardState(fixture.blocks[1], state.get());
  require(!fixture.lookup(65).state &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 0,
          "a failed copy counted as an eviction of a state in use");
}

// A publication in use makes room as running work does: KV no state
// restores through goes first, oldest first, then the ordinary state, then
// the KV it restored through. An optional publication takes no KV, and the
// running chain the publication belongs to is never a victim.
void testPublicationInUseTakesOrdinaryKv() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 8};
  engine::Cache cache(pool, nullptr, nullptr);
  const auto older = cacheChain(cache, 1, 2).first;
  const std::vector<uint64_t> stated = cacheChain(cache, 2, 2).second;
  cache.publishCompositeState(stated[1], std::make_shared<TestState>(100), false);
  const auto newer = cacheChain(cache, 3, 2).first;
  const uint64_t point = cacheChain(cache, 4, 2, true).second[1];
  StateUse use = cache.useState(point);
  const auto blocks = [&] { return cache.snapshot().pool.pagesPrefix; };
  require(!cache.reclaimOneState(true, point, true) && blocks() == 8,
          "an optional publication took KV");
  for (uint32_t leaf = 0; leaf < 2; ++leaf)
    require(cache.reclaimOneState(false, point, true) && cache.stateResident(stated[1]),
            "the older dead KV did not go before the ordinary state");
  require(blocks() == 6 && cache.lookup(older, {}).kvBoundary == 0 &&
              cache.lookup(newer, {}).kvBoundary == 64 && pool.snapshot().pagesAllocated == 6,
          "the older dead KV did not go first, with its extents");
  for (uint32_t leaf = 0; leaf < 2; ++leaf)
    require(cache.reclaimOneState(false, point, true) && cache.stateResident(stated[1]),
            "the newer dead KV did not go before the ordinary state");
  require(cache.reclaimOneState(false, point, true) && !cache.stateResident(stated[1]) &&
              blocks() == 4,
          "the ordinary state did not go before the KV it restored through");
  for (uint32_t leaf = 0; leaf < 2; ++leaf)
    require(cache.reclaimOneState(false, point, true).made, "the KV of the old state did not go");
  require(!cache.reclaimOneState(false, point, true) && blocks() == 2 &&
              cache.snapshot().stateCache.inUseEvictions == 0,
          "a publication took its own running chain");
  cache.endRequest(4);
}

// An ordinary publication, such as a junction, makes room from cached KV as
// one in use does, an extent at a time, but takes nothing in use: once only
// a state in use and the KV it restores through are left, it gets no room.
void testOrdinaryPublicationTakesOrdinaryKvNotWhatIsInUse() {
  test::TestKvStorage storage{8, 100, 2};
  KvPool pool{storage, 8};
  engine::Cache cache(pool, nullptr, nullptr);
  static_cast<void>(cacheChain(cache, 1, 4));
  const uint64_t used = cacheChain(cache, 2, 2).second[1];
  StateUse use = cache.useState(used);
  cache.publishCompositeState(used, std::make_shared<TestState>(100), false);
  const uint64_t junction = cacheChain(cache, 3, 2, true).second[1];
  for (const uint32_t allocated : {6U, 4U}) {
    const StateRoom room = cache.reclaimOneState(false, junction, true);
    require(room && room.extentBytes == 200 && pool.snapshot().pagesAllocated == allocated,
            "an ordinary publication did not take an extent of the older KV");
  }
  require(!cache.reclaimOneState(false, junction, true) && cache.stateResident(used) &&
              cache.snapshot().stateCache.inUseEvictions == 0 &&
              cache.snapshot().pool.pagesPrefix == 4,
          "an ordinary publication took what is in use");
  cache.endRequest(3);
}

// An ordinary publication never drops a victim whose write must wait for the
// write in flight: it gets no room, and the victim is written once the
// staging buffer is free.
void testOrdinaryPublicationWaitsForTheWriteSlot() {
  CacheFixture fixture(nullptr, true);
  auto control = std::make_shared<TransferControl>();
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  require(fixture.cache.reclaimOneState(false, fixture.blocks[2], false).made,
          "the first state was not written");
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  require(!fixture.cache.reclaimOneState(false, fixture.blocks[2], false) &&
              fixture.cache.stateResident(fixture.blocks[1]),
          "an ordinary publication dropped a victim whose write had to wait");
  control->ready = true;
  require(fixture.cache.pollTransfers() &&
              fixture.cache.reclaimOneState(false, fixture.blocks[2], false) &&
              !fixture.cache.stateResident(fixture.blocks[1]) &&
              fixture.cache.snapshot().stateCache.offloads == 2 &&
              fixture.lookup(65).resumeBoundary() == 64,
          "the waiting victim was not written once the staging buffer was free");
}

// While the engine may not grow, an extent's bytes give a snapshot nothing:
// a publication in use leaves KV and extents where they are and recycles
// states like any other publication, the oldest in use after the ordinary
// ones. With growth each step says what it gave, buffers or an extent.
void testPublicationInUseWithoutGrowthTakesStatesAlone() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 8};
  engine::Cache cache(pool, nullptr, nullptr);
  const auto older = cacheChain(cache, 1, 2).first;
  const std::vector<uint64_t> stated = cacheChain(cache, 2, 2).second;
  cache.publishCompositeState(stated[1], std::make_shared<TestState>(100), false);
  const uint64_t other = cacheChain(cache, 3, 2, true).second[1];
  StateUse held = cache.useState(other);
  cache.publishCompositeState(other, std::make_shared<TestState>(100), false);
  const uint64_t point = cacheChain(cache, 4, 2, true).second[1];
  StateUse use = cache.useState(point);
  const auto blocks = [&] { return cache.snapshot().pool.pagesPrefix; };
  const auto allocated = [&] { return pool.snapshot().pagesAllocated; };
  require(blocks() == 8 && allocated() == 8, "fixture geometry changed");

  StateRoom room = cache.reclaimOneState(false, point, false);
  require(room && room.extentBytes == 0 && !cache.stateResident(stated[1]) && blocks() == 8 &&
              allocated() == 8 && cache.lookup(older, {}).kvBoundary == 64,
          "without growth the publication took KV ahead of the ordinary state");
  room = cache.reclaimOneState(false, point, false);
  require(room && room.extentBytes == 0 && !cache.stateResident(other) && blocks() == 8 &&
              allocated() == 8 && cache.snapshot().stateCache.inUseEvictions == 1,
          "without growth the publication took KV ahead of the state in use");
  require(!cache.reclaimOneState(false, point, false) && blocks() == 8 && allocated() == 8,
          "without growth the publication took KV once no state was left");
  room = cache.reclaimOneState(false, point, true);
  require(room && room.extentBytes == 100 && blocks() == 7 && allocated() == 7,
          "with growth the oldest KV did not go with its extent");
  cache.endRequest(3);
  cache.endRequest(4);
}

// A publication in use takes ordinary KV only: the KV another state in use
// restores through stays, and running work takes it, last.
void testPublicationInUseLeavesKvInUse() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage{4, 100, 1};
  KvPool pool{storage, 4};
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const std::vector<uint64_t> used = cacheChain(cache, 1, 2).second;
  StateUse held = cache.useState(used[1]);
  cache.publishCompositeState(used[1], std::make_shared<TieredState>(control), false);
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && cache.pollTransfers() &&
              !cache.stateResident(used[1]),
          "the state in use was not written");
  const uint64_t point = cacheChain(cache, 2, 2, true).second[1];
  StateUse use = cache.useState(point);
  require(!cache.reclaimOneState(false, point, true) && cache.snapshot().pool.pagesPrefix == 4 &&
              tier.demotions == 0,
          "a publication in use took the KV a state in use restores through");
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == 1,
          "running work did not take the KV in use last");
  cache.endRequest(2);
}

// A snapshot cannot wait for a demotion: a publication in use passes over
// every leaf whose page returns only once a copy is written, here each older
// leaf, which holds a state only the disk holds. It starts no demotion, so it
// displaces no disk copy, and recycles the oldest state in use instead.
void testInUsePublicationStartsNoDemotion() {
  test::TestKvStorage storage{6, 100, 1};
  KvPool pool{storage, 6};
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  for (uint64_t id : {1, 2}) {
    cache.publishCompositeState(cacheChain(cache, id, 2).second[1],
                                std::make_shared<TieredState>(control), false);
    require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
            "the older state was not written");
  }
  const uint64_t other = cacheChain(cache, 3, 1, true).second[0];
  StateUse held = cache.useState(other);
  cache.publishCompositeState(other, std::make_shared<TestState>(100), false);
  const uint64_t point = cacheChain(cache, 4, 1, true).second[0];
  StateUse use = cache.useState(point);
  const CacheSnapshot before = cache.snapshot();
  const StateRoom room = cache.reclaimOneState(false, point, true);
  const CacheSnapshot after = cache.snapshot();
  require(room && room.extentBytes == 0 && tier.demotions == 0 &&
              !cache.stateResident(other) && after.stateCache.inUseEvictions == 1 &&
              after.stateCache.diskBytes == before.stateCache.diskBytes &&
              after.kvTier.diskBlocks == before.kvTier.diskBlocks &&
              after.pool.pagesPrefix == before.pool.pagesPrefix,
          "a publication in use started a demotion or displaced a disk copy");
  cache.endRequest(3);
  cache.endRequest(4);
}

// KV gives a publication in use memory only through an extent it empties.
// Where the pages requests hold leave no extent to empty, three held in each
// of two extents, no KV goes: the stateless chain on the other two pages
// stays, nothing is copied, and the oldest state in use is recycled.
void testInUsePublicationTakesNoKvWhenNoExtentCanEmpty() {
  test::TestKvStorage storage{8, 100, 4};
  KvPool pool{storage, 8};
  engine::Cache cache(pool, nullptr, nullptr);
  const uint64_t other = cacheChain(cache, 1, 3, true).second[2];
  StateUse held = cache.useState(other);
  cache.publishCompositeState(other, std::make_shared<TestState>(100), false);
  static_cast<void>(cacheChain(cache, 2, 2));
  const uint64_t point = cacheChain(cache, 3, 3, true).second[2];
  StateUse use = cache.useState(point);
  require(std::ranges::equal(cache.pageTable(1).pages, std::vector<uint32_t>{0, 1, 2}) &&
              std::ranges::equal(cache.pageTable(3).pages, std::vector<uint32_t>{5, 6, 7}),
          "fixture geometry changed");
  const StateRoom room = cache.reclaimOneState(false, point, true);
  require(room && room.extentBytes == 0 && !cache.stateResident(other) &&
              cache.snapshot().stateCache.inUseEvictions == 1 &&
              cache.snapshot().pool.pagesPrefix == 8 && storage.copies.empty(),
          "a publication in use took KV although no extent could be emptied");
  cache.endRequest(1);
  cache.endRequest(3);
}

// While the host refuses growth only what gives pages goes: KV leaves in
// their own oldest-first order, never a checkpoint or a state older than
// every leaf, and a state only where it sits on the leaf that goes next,
// which then goes on the next step. A scan that keeps the resume point
// passes over it and its leaf.
void testPageReuseTakesKvBeforeStates() {
  constexpr auto reuse = CacheReclaimMode::ReusePages;
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 8};
  engine::Cache cache(pool, nullptr, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  // Two chains whose leaves hold states only the disk holds, so neither is
  // dead KV.
  std::vector<std::vector<uint32_t>> prompts;
  std::vector<std::vector<uint64_t>> chains;
  for (uint64_t id = 1; id <= 2; ++id) {
    auto [prompt, blocks] = cacheChain(cache, id, 2);
    cache.publishCompositeState(blocks[1], std::make_shared<TieredState>(control), false);
    require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
            "a leaf's state was not written");
    prompts.push_back(std::move(prompt));
    chains.push_back(std::move(blocks));
  }
  // An ordinary state and a checkpoint on the interior blocks, older than
  // both leaves, which the lookups refresh.
  const uint64_t ordinary = chains[0][0];
  const uint64_t checkpoint = chains[1][0];
  cache.publishCompositeState(ordinary, std::make_shared<TestState>(100), false);
  cache.publishCompositeState(checkpoint, std::make_shared<TestState>(100), true);
  for (const auto &prompt : prompts)
    static_cast<void>(cache.lookup(prompt, {}));
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              cache.snapshot().pool.pagesPrefix == 3 && cache.stateResident(ordinary) &&
              cache.stateResident(checkpoint) &&
              cache.snapshot().stateCache.checkpointEvictions == 0,
          "page reuse took a state before a KV leaf");
  // The first chain's interior block is now its oldest leaf, under the
  // ordinary state, the resume point.
  require(cache.reclaimOne(reuse, ReclaimClass::InUse, true).madeProgress &&
              cache.stateResident(ordinary) && cache.stateResident(checkpoint) &&
              cache.snapshot().pool.pagesPrefix == 2,
          "page reuse that keeps the resume point did not take the next leaf");
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              !cache.stateResident(ordinary) && cache.snapshot().pool.pagesPrefix == 2,
          "page reuse did not take the state on the leaf that goes next");
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress &&
              cache.snapshot().pool.pagesPrefix == 1 && cache.stateResident(checkpoint),
          "page reuse did not take the leaf its state left");
}

// A reclaim that keeps extents leaves the one its eviction empties allocated
// for the next request, which takes its pages without allocating; releasing
// the empty extents afterwards releases it.
void testReplacementKeepsTheExtentItEmpties() {
  test::TestKvStorage storage(8, 4096, 4);
  storage.budgetPages = 4;
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  static_cast<void>(cacheChain(cache, 1, 1));

  const auto reclaimed = cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse);
  require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 0 &&
              cache.snapshot().pool.pagesPrefix == 0 &&
              storage.allocatedPages() == 4 && storage.releasedExtents == 0,
          "replacement released the newly reusable extent");
  cache.beginRequest(2);
  require(cache.ensureTokens(2, 128).granted() &&
              pool.snapshot().extentAllocations == 1 && storage.releasedExtents == 0,
          "replacement allocated the reusable extent again");
  cache.endRequest(2);
  require(cache.releaseEmptyExtents(false) == 4 * 4096 &&
              storage.allocatedPages() == 0 && storage.releasedExtents == 1,
          "releasing the empty extents did not release the reused one");
}

// A pass that releases extents as its evictions empty them reports the
// longest release of one extent, as growth reports the longest allocation of
// one; the loop's longest tick covers the whole pass.
void testReleaseTimeCoversOneExtent() {
  constexpr uint32_t extents = 6;
  test::TestKvStorage storage(4 * extents, 4096, 4);
  storage.releaseTime = std::chrono::milliseconds(5);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  for (uint32_t chain = 0; chain < extents; ++chain)
    static_cast<void>(cacheChain(cache, chain + 1, 4));
  require(cache.snapshot().pool.reclaimableBytes == 0 &&
              cache.snapshot().pool.pagesPrefix == 4 * extents,
          "release time setup geometry changed");
  static_cast<void>(cache.evictAll());
  // The pool's timer runs around one release: no less than the storage saw
  // its longest release take, and less than all of them took together. No
  // bound in milliseconds holds on a loaded machine.
  const double longest = cache.snapshot().pool.extentReleaseMaxMilliseconds;
  require(storage.releasedExtents == extents && longest >= storage.longestRelease &&
              longest < storage.totalRelease,
          "the release time is not one extent's");
}

// KV gives a publication in use memory only through the extent it leaves
// empty. An empty extent goes first. Then leaves go, oldest first, until one
// leaves an extent empty, which is released before the call returns: the
// snapshot that follows needs the memory at once.
void testPublicationReleasesTheExtentItEmpties() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  // The publishing request runs on the first extent.
  const uint64_t point = cacheChain(cache, 1, 4, true).second[3];
  StateUse use = cache.useState(point);
  // Another conversation's chain fills the second extent and half the third.
  static_cast<void>(cacheChain(cache, 2, 6));
  // A request that cached nothing leaves the fourth extent empty.
  cache.beginRequest(3);
  require(cache.ensureTokens(3, 96).granted(), "the empty extent was not allocated");
  cache.endRequest(3);
  require(cache.snapshot().pool.reclaimableBytes == 4 * 4096 &&
              storage.allocatedPages() == 16,
          "fixture geometry changed");

  // An extent is room only for a snapshot that can allocate its bytes.
  require(!cache.reclaimOneState(false, point, false) && storage.releasedExtents == 0 &&
              cache.snapshot().pool.pagesPrefix == 10,
          "a publication that cannot allocate released an extent or took KV");
  StateRoom room = cache.reclaimOneState(false, point, true);
  require(room && room.extentBytes == 4 * 4096 && storage.releasedExtents == 1 &&
              cache.snapshot().pool.pagesPrefix == 10,
          "the publication did not release the empty extent first");
  room = cache.reclaimOneState(false, point, true);
  require(room && room.extentBytes == 4 * 4096 && storage.releasedExtents == 2 &&
              storage.allocatedPages() == 8 && cache.snapshot().pool.pagesPrefix == 8,
          "the publication took more KV than its extent or kept the extent it emptied");
  cache.endRequest(1);
}

int main() {
  try {
    testStateInUseGoesLast();
    testKvAStateInUseNeedsGoesLast();
    testStateUsesAreCountedPerBlock();
    testKvInUseFollowsItsState();
    testEndRequestStampsOnlyTheReplayPoint();
    testKeepResumePointKeepsTheNewestOrdinaryPublication();
    testStatesInUseAreOrdinary();
    testInUseWaitsForTransfersInFlight();
    testLaneTakesStatesInUseLast();
    testOrdinaryClassStopsShortOfStatesInUse();
    testRecycleNeverDropsAStateInUseForTheWriteSlot();
    testDiskReplacementTakesStatesInUseLast();
    testPageShortageTakesLeavesInUseLast();
    testKvAStateInUseNeedsNeverPinsMemory();
    testOrdinaryDemotionKeepsCopiesInUse();
    testPromotionSkipsForAStateInUse();
    testInUseEvictionsCountChoices();
    testPublicationInUseTakesOrdinaryKv();
    testDeadKvGoesBeforeOlderStates();
    testPendingDemotionLeavesOtherKvOpen();
    testPublicationInUseWithoutGrowthTakesStatesAlone();
    testOrdinaryPublicationTakesOrdinaryKvNotWhatIsInUse();
    testOrdinaryPublicationWaitsForTheWriteSlot();
    testPublicationInUseLeavesKvInUse();
    testInUsePublicationStartsNoDemotion();
    testInUsePublicationTakesNoKvWhenNoExtentCanEmpty();
    testPublicationReleasesTheExtentItEmpties();
    testLargeSharedDiskRestore();
    testRefusedRestoresStopAtTheFirstRefusal();
    testReclaimForPagesCoversTheShortfall();
    testPageReuseTakesKvBeforeStates();
    testLookupKeepsADiskChainsResidentBoundaryWarm();
    testRestoreKeepsTheBlockItExtends();
    testDemotionKeepsThePageUnderANewState();
    testCancelledRestoreKeepsThePageUnderANewState();
    testBusyTierPreservesDiskVictim();
    testCancelledRestoreStopsQueuedReads();
    testDiskCheckpointRamAccounting();
    testOrdinaryPublicationUpgradesDiskCheckpoint();
    testKvDemotionAndRestoreLifecycle();
    testTailsDropAndParentsFollowToDisk();
    testDiskCopiesNoStateNeedsGoWithTheLeaf();
    testRefusedDemotionKeepsTheLeafWhileTransfersLand();
    testUnusableTierDropsTheLeafInstead();
    testSecondStateWaitsForTheWrite();
    testWaitingCheckpointHoldsBackNothingElse();
    testFullTierStopsTheScan();
    testRestoresInFlightMakeAShortfallPending();
    testParentOfDiskChildrenSurvivesRefusal();
    testDiskReplacementOrder();
    testPendingPagesGateAllocation();
    testTransferFailures();
    testFailedRestoreDropsTheBlocksBelow();
    testDemotionInFlightCountsAsPendingBytes();
    testDemotionCostsNoSecondState();
    testPromotionIdentityAndDenial();
    testPromotionNeverDropsAUniqueState();
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
    testQuotaWithoutTheKvTier();
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
    testSchedulingProbeDoesNotChangeCachePolicy();
    testValidAdmissionProbePreservesLookupAndAccounting();
    testProbeRefreshFollowsTheGraph();
    testProbeRechecksStateChanges();
    testProbeFallsBackWhenKvChanges();
    testProbeBindsImageIdentity();
    testProbeCannotCrossCaches();
    testPageTableReportsWhatChanged();
    testCacheLookupAndOneTokenReplay();
    testPage31Page32Page33Backoff();
    testLazyJunctionMaterialization();
    testLazyJunctionOnlyAtABranchPoint();
    testByteLruAndPins();
    testSpeculativeReclaimKeepsTheResumePoint();
    testCheckpointDoesNotOutrankTheResumePoint();
    testKvEvictionInvalidatesStateFirst();
    testStatePublicationValidation();
    testDuplicateProbePromotesStateWithoutLookupAccounting();
    testUnifiedRecencyAndReleasedByteAccounting();
    testFinishedRequestLeavesTailKvBeforeItsState();
    testCompactionReturnsFreePagesBeforeEvicting();
    testCompactionFollowsOnlyTheMovedBlocks();
    testCompactionLeavesTheRunway();
    testEvictAllGathersWhatRequestsHold();
    testCompactionLeavesAPageBeingRestored();
    testCompactionLeavesAPageBeingDemoted();
    testReplacementKeepsTheExtentItEmpties();
    testReleaseTimeCoversOneExtent();
    std::cout << "KV-first cache tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "KV-first cache tests failed: " << error.what() << '\n';
    return 1;
  }
}
