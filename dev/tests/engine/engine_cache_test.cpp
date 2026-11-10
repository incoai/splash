#include "Checked.hpp"
#include "MemoryPasses.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestWriteBudget.hpp"
#include "engine/Cache.hpp"
#include "model/SlotFile.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::rejects;
using splash::test::require;

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
  // The write budget the states' file shares with the KV tier, if any.
  std::shared_ptr<const model::DiskBudget> writeBudget;
  // The states whose RAM copy went: their buffers are back in the model's
  // pool, as a model state's are once its RAM copy goes.
  uint32_t pooled = 0;
  // What the model's pool releases once `states` more buffers return to it
  // (MemoryLedger::setBufferBytes), at 100 bytes a state.
  [[nodiscard]] uint64_t poolBytes(uint32_t states) const noexcept { return uint64_t{pooled + states} * 100; }
};

// A write the tier admits while the quota has a slot; its disk copy holds
// that slot until it is dropped.
std::unique_ptr<StateOffload> writeState(const std::shared_ptr<TransferControl> &control);

class TieredState final : public CompositeState {
public:
  TieredState(std::shared_ptr<TransferControl> control, bool disk = false)
      : control_(std::move(control)), disk_(disk) { if (disk_) ++control_->slots; }
  ~TieredState() override {
    if (disk_)
      --control_->slots;
    else
      ++control_->pooled;
  }
  uint64_t bytes() const noexcept override { return 100; }
  uint64_t residentBytes() const noexcept override { return disk_ ? 0 : bytes(); }
  bool writable() const noexcept override { return !disk_; }
  bool canOffload(bool inUse) const noexcept override {
    return !disk_ && (!control_->writeBudget || control_->writeBudget->admits(inUse));
  }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override {
    return writeState(control_);
  }
  std::unique_ptr<StateOffload> persist(std::function<void()>) const override {
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

// What the ledger answers the claim a refused admission makes, the cache its
// only source (test::freeFromCache): the pages it lacks beyond the free ones,
// as the engine asks for them (Engine's inPlace).
FreeResult askLedger(engine::Cache &cache, const TokenAdmission &refused) {
  const uint32_t lacking = refused.additionalPages - std::min(refused.additionalPages, refused.availablePages);
  return test::freeFromCache(cache, test::kInServiceGrowth, {.pages = lacking, .form = refused.form});
}

// An admission made the way the engine makes it, and the ledger's answer to
// its last refusal: Met once the pages fit (or at once), else Pending while
// what lands covers what it lacks, Short when nothing does.
struct Admitted final {
  TokenAdmission admission;
  FreeOutcome ledger = FreeOutcome::Met;
  [[nodiscard]] bool granted() const noexcept { return admission.granted(); }
};

// Admits pages the way the engine does: each refusal asks the ledger for the
// pages it lacks (askLedger), until the pages fit or the ledger answers
// other than Met.
Admitted admitLikeEngine(engine::Cache &cache, const std::function<TokenAdmission()> &attempt) {
  Admitted admitted{attempt()};
  while (!admitted.granted()) {
    admitted.ledger = askLedger(cache, admitted.admission).outcome;
    if (admitted.ledger != FreeOutcome::Met)
      break;
    admitted.admission = attempt();
  }
  return admitted;
}

Admitted admitTokens(engine::Cache &cache, uint64_t requestId, uint64_t tokens) {
  return admitLikeEngine(cache, [&] { return cache.ensureTokens(requestId, tokens); });
}

Admitted admitRestore(engine::Cache &cache, uint64_t requestId,
                            const CacheLookup &lookup) {
  return admitLikeEngine(cache, [&] { return cache.restoreRequest(requestId, lookup); });
}

// A claim for one state's buffers that may take every class; returns the
// bytes of state RAM it let go of.
uint64_t reclaimStateBytes(engine::Cache &cache) {
  const uint64_t before = cache.snapshot().stateCache.bytes;
  require(test::stateStep(cache, ReclaimClass::InUse).gave(), "the claim took no state");
  return before - cache.snapshot().stateCache.bytes;
}

// Claims for an extent's bytes until none takes anything.
void reclaimEverything(engine::Cache &cache) {
  while (test::releaseStep(cache).gave()) {
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
  explicit CacheFixture(engine::KvTier *tier = nullptr, bool running = false,
                        std::shared_ptr<model::DiskBudget> budget = nullptr)
      : cache(pool, tier, std::move(budget)) {
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
// one token past its page, so a lookup can match the page. A tier's copies
// count against `budget`, if any.
struct Prefixes {
  test::TestKvStorage storage{5, 100, 1};
  KvPool pool{storage, 4};
  engine::Cache cache;
  std::array<std::vector<uint32_t>, 4> prompts;
  std::array<uint64_t, 4> blocks{};

  explicit Prefixes(test::TestKvTier *tier = nullptr, bool continued = true,
                    std::shared_ptr<model::DiskBudget> budget = nullptr)
      : cache(pool, tier, std::move(budget)) {
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
    require(test::evictStep(cache).gave(),
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
  require(test::stateStep(fixture.cache).gave() && cachedTokens(prefix) == 0 &&
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

// A kept probe hashes nothing while the KV graph is unchanged, and after a change only from the
// first block that no longer matches.
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
                test::evictStep(cache).gave() &&
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
    require(test::stateStep(cache).gave() && cache.pollTransfers(),
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
  require(probe.cachedTokens() == 128 && test::stateStep(fixture.cache).gave(),
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
      test::evictStep(fixture.cache).gave() &&
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
  const uint64_t block = test::publishBlocks(cache, 1, prompt, 32, {images, {}});
  cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
  cache.endRequest(1);
  const CacheProbe probe = cache.probe(prompt, {images, {}});
  const auto valid = cache.lookup(prompt, {images, {}}, &probe);
  require(valid.kvBoundary == 32 && valid.resumeBoundary() == 32,
          "matching image probe lost its cached prefix");
}

// A cache salt keys the first block: salted and unsalted prompts never match each other.
void testCacheSaltSeparatesPrompts() {
  const CacheSalt red{0x1111, 0x2222}, blue{0x3333, 0x4444};
  require(blockIdentity(0, 32, {}, {}) == BlockIdentity{} &&
              blockIdentity(0, 32, {}, red) != BlockIdentity{} &&
              blockIdentity(0, 32, {}, red) != blockIdentity(0, 32, {}, blue) &&
              blockIdentity(32, 32, {}, red) == BlockIdentity{},
          "a salt did not key the first block alone");
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 8};
  engine::Cache cache{pool, nullptr, nullptr};
  std::vector<uint32_t> prompt(65, 77);
  const auto publish = [&](uint64_t id, CacheSalt salt) {
    cache.beginRequest(id);
    require(admitTokens(cache, id, 64).granted(), "KV pages were not acquired");
    const uint64_t block = test::publishBlocks(cache, id, prompt, 64, {{}, salt});
    cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
    cache.endRequest(id);
  };
  const auto cached = [&](CacheSalt salt) {
    return cache.lookup(prompt, {{}, salt}).resumeBoundary();
  };
  // Probes taken before the prompt is cached match it, once refreshed, only
  // under its salt, and a lookup takes the refreshed chain.
  CacheProbe probe = cache.probe(prompt, {{}, red});
  CacheProbe other = cache.probe(prompt, {{}, blue});
  publish(1, red);
  require(cached(red) == 64 && cached(blue) == 0 && cached({}) == 0,
          "a prompt cached under a salt matched another salt or none");
  cache.refresh(probe, prompt, {{}, red});
  cache.refresh(other, prompt, {{}, blue});
  require(probe.cachedTokens() == 64 && other.cachedTokens() == 0 &&
              cache.lookup(prompt, {{}, red}, &probe).resumeBoundary() == 64,
          "a refreshed probe matched across salts or lost its own");
  publish(3, blue);
  publish(2, {});
  require(cached({}) == 64 && cached(red) == 64 && cached(blue) == 64,
          "an unsalted prompt did not part from the salted ones");
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
  rejects([&] { second.refresh(probe, firstPrompt, {}); },
          "a scheduling probe belongs to another cache",
          "a probe from another cache was refreshed");
}

// A cut page table says from which page on it changed since the last cut (append, swap, restore);
// an unchanged list keeps its revision.
void testPageTableReportsWhatChanged() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 8};
  engine::Cache cache(pool, nullptr, nullptr);
  constexpr uint32_t page = KvCache::pageTokens;
  std::vector<uint32_t> prompt;
  for (uint32_t token = 0; token <= 2 * page; ++token)
    prompt.push_back(1000 + token);
  const auto cut = [&](uint64_t request) { return cache.cutPageTable(request); };

  cache.beginRequest(1);
  require(admitTokens(cache, 1, page).granted() && cache.pageTable(1).revision == 1 &&
              cut(1).revision == 1 && cache.pageTable(1).firstChanged == 0,
          "a request's first page did not start its revision");
  require(admitTokens(cache, 1, 2 * page).granted() && cut(1).revision == 2 &&
              cache.pageTable(1).firstChanged == 1 && cache.pageTable(1).pages.size() == 2,
          "an append did not report the length it grew from");
  require(admitTokens(cache, 1, page + 1).granted(), "held pages were denied");
  require(cache.publishCommittedBlocks(1, prompt, 2 * page, {}) == 0 && cut(1).revision == 2,
          "tokens on held pages or a request's own blocks moved its revision");
  cache.publishCompositeState(cache.blockAt(1, 2 * page), std::make_shared<TestState>(100), false);

  // A second request writes the same blocks into pages of its own, and
  // publishing each swaps it for the cached page and reports its tokens.
  cache.beginRequest(2);
  require(admitTokens(cache, 2, 2 * page).granted() && cut(2).revision == 1,
          "a second request's pages were denied");
  require(cache.publishCommittedBlocks(2, prompt, page, {}) == page && cut(2).revision == 2 &&
              cache.pageTable(2).firstChanged == 0 &&
              cache.pageTable(2).pages[0] == cache.pageTable(1).pages[0],
          "the first swapped block did not report its index or tokens");
  require(cache.publishCommittedBlocks(2, prompt, 2 * page, {}) == page && cut(2).revision == 3 &&
              cache.pageTable(2).firstChanged == 1 &&
              cache.pageTable(2).pages[1] == cache.pageTable(1).pages[1] && cut(1).revision == 2,
          "a swapped block did not report its index or tokens, or moved the writer's revision");
  cache.endRequest(2);
  cache.endRequest(1);

  cache.beginRequest(3);
  require(admitRestore(cache, 3, cache.lookup(prompt, {})).granted() && cut(3).revision == 1 &&
              cache.pageTable(3).firstChanged == 0 && cache.pageTable(3).pages.size() == 2,
          "a restore did not report its whole list");
  require(admitTokens(cache, 3, prompt.size()).granted() && cut(3).revision == 2 &&
              cache.pageTable(3).firstChanged == 2,
          "an append after a restore did not report the restored length");
  cache.endRequest(3);

  // Two changes before a cut: one revision, from the first page either
  // changed, which the view shows before the cut makes it the command's.
  cache.beginRequest(4);
  require(admitTokens(cache, 4, page).granted() && cut(4).revision == 1,
          "a fourth request's page was denied");
  require(admitTokens(cache, 4, 2 * page + 1).granted() &&
              cache.publishCommittedBlocks(4, prompt, page, {}) == page &&
              cache.pageTable(4).revision == 2 && cache.pageTable(4).firstChanged == 0,
          "two changes did not show as the next revision from the first change");
  const PageTableView coalesced = cut(4);
  require(coalesced.revision == 2 && coalesced.firstChanged == 0 && coalesced.pages.size() == 3 &&
              cut(4).revision == 2 && cut(4).firstChanged == 0,
          "two changes before a cut made more than one revision, or a cut without changes made one");
  cache.endRequest(4);
}

// A new page with rows the work does not write comes cleared; pages the work writes in full are not
// cleared.
void testNewPagesTheWorkDoesNotFillComeCleared() {
  test::TestKvStorage storage{16, 4096, 4};
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  std::fill(storage.content.begin(), storage.content.end(), 7);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 64).granted() && storage.cleared.empty(),
          "a target at a page boundary cleared a page");
  require(cache.ensureTokens(1, 70).granted(), "the tail page was not admitted");
  const PageTableState table = cache.pageTable(1);
  require(table.pages.size() == 3 &&
              storage.cleared == std::vector<uint32_t>{table.pages[2]} &&
              storage.content[table.pages[2]] == 0 &&
              storage.content[table.pages[0]] == 7 && storage.content[table.pages[1]] == 7,
          "the new tail page was not cleared, or a page written in full was");
  require(cache.ensureTokens(1, 90).granted() && storage.cleared.size() == 1,
          "a target inside the tail page cleared it again");
  // A resumed lane takes back pages past where its replay ends.
  cache.beginRequest(2);
  require(cache.ensureTokens(2, 160, 70).granted(), "the resumed pages were not admitted");
  const PageTableState resumed = cache.pageTable(2);
  require(resumed.pages.size() == 5 &&
              storage.cleared == std::vector<uint32_t>{table.pages[2], resumed.pages[2],
                                                       resumed.pages[3], resumed.pages[4]} &&
              storage.content[resumed.pages[1]] == 7,
          "a page the replay does not write in full was not cleared, or one it does was");
  cache.endRequest(2);
  cache.endRequest(1);
}

void testCacheLookupAndOneTokenReplay() {
  CacheFixture fixture;
  fixture.publish(0);
  fixture.publish(2);

  auto full = fixture.cache.lookup(fixture.prompt, {});
  require(full.kvBoundary == 128 && full.resumeBoundary() == 96 &&
              full.junctionBoundary == 0 && full.state &&
              full.state->kvBlock() == fixture.blocks[2],
          "a lookup did not pair the cached KV prefix with the latest state within it");

  full.state.reset();
  auto exactEdge = fixture.lookup(128);
  require(exactEdge.kvBoundary == 96 && exactEdge.resumeBoundary() == 96 &&
              !exactEdge.junctionBoundary,
          "exact block-edge prompt did not replay one input token");

  auto shortPrompt = fixture.lookup(32);
  require(shortPrompt.kvBoundary == 0 && shortPrompt.resumeBoundary() == 0,
          "single-block prompt illegally became an exact hit");
}

// A prompt that branches off a cached chain asks for a junction at the branch point, and the next
// lookup resumes from it.
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

// A junction is asked for only at a real branch point, not where the chain ends or inside a
// stateless tail.
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
  require(reclaimStateBytes(fixture.cache) == 150,
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
  require(reclaimStateBytes(fixture.cache) == 150,
          "released state pins did not restore LRU eligibility");
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
  // Growth, which keeps the resume point.
  const FreeResult step = test::freeFromCache(
      fixture.cache, {.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate}, {.bytes = 1});
  const auto kept = fixture.cache.snapshot().stateCache;
  require(step.gave() && kept.entries == 1 && kept.checkpointEntries == 0,
          "a disposable checkpoint outranked the resume point");
  require(fixture.lookup(33).resumeBoundary() == 32,
          "the warm conversation lost its resumable prefix");
}

void testKvEvictionInvalidatesStateFirst() {
  CacheFixture fixture;
  fixture.publish(3);
  require(reclaimStateBytes(fixture.cache) == 100,
          "state was not reclaimed before its KV block");
  require(test::releaseStep(fixture.cache).nowBytes == 100 &&
              fixture.cache.lookup(fixture.prompt, {}).kvBoundary == 96,
          "KV leaf eviction did not remove the dependent prefix");

  CacheFixture pinnedFixture;
  pinnedFixture.publish(2);
  auto lease = pinnedFixture.lookup(97);
  require(lease.state.has_value(), "replacement state pin failed");
  reclaimEverything(pinnedFixture.cache);
  require(pinnedFixture.cache.snapshot().pool.pagesPrefix == 3 &&
              pinnedFixture.cache.snapshot().stateCache.entries == 1,
          "pinned composite state did not protect its KV dependency");
}

void testStatePublicationValidation() {
  CacheFixture fixture;
  fixture.publish(0);
  rejects([&] { fixture.publish(0); }, "duplicate composite state key",
          "a state was published twice at one block");
  rejects(
      [&] {
        fixture.cache.publishCompositeState(999, std::make_shared<TestState>(100), false);
      },
      "composite state KV block is unknown", "a state was published without its KV block");
  require(fixture.cache.snapshot().stateCache.entries == 1,
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
  require(reclaimStateBytes(fixture.cache) == 200,
          "publication probe did not promote the existing state in LRU");
}

void testUnifiedRecencyAndReleasedByteAccounting() {
  {
    CacheFixture fixture;
    fixture.publish(0, 150);
    const FreeResult reclaimed = test::releaseStep(fixture.cache);
    require(reclaimed.gave() && reclaimed.nowBytes == 100 &&
                fixture.cache.snapshot().pool.pagesPrefix == 3 &&
                fixture.cache.snapshot().stateCache.entries == 1,
            "global cache order did not select the older state-free KV leaf");
  }

  {
    // With the only leaf state-backed, the state is selected. Its eviction
    // releases nothing by itself: its buffers return to the model's pool.
    CacheFixture fixture;
    fixture.publish(3, 150);
    const FreeResult reclaimed = test::releaseStep(fixture.cache);
    require(reclaimed.gave() && reclaimed.stateBuffers && reclaimed.nowBytes == 0 &&
                fixture.cache.snapshot().stateCache.entries == 0 &&
                fixture.cache.snapshot().stateCache.bytes == 0 &&
                fixture.cache.snapshot().pool.pagesPrefix == 4,
            "the state was not evicted, or its eviction counted as released");
  }
}

// A finished request's decode tail past its state is dead KV and goes before the state.
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
    const FreeResult reclaimed = test::evictStep(cache);
    require(reclaimed.gave() && reclaimed.nowBytes == 0 &&
                cache.snapshot().pool.pagesPrefix == remaining &&
                cache.snapshot().stateCache.entries == 1,
            "finished request's state was evicted before its KV tail");
  }
  require(cache.lookup(prompt, {}).resumeBoundary() == 64,
          "state did not survive the eviction of the decode tail");
  const FreeResult state =
      test::evictStep(cache);
  require(state.gave() && state.stateBuffers &&
              cache.snapshot().stateCache.entries == 0 &&
              cache.snapshot().pool.pagesPrefix == 2,
          "state was not evicted once its block became the oldest leaf");
  const FreeResult leaf =
      test::evictStep(cache);
  require(leaf.gave() && cache.snapshot().pool.pagesPrefix == 1,
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

  require(test::stateStep(fixture.cache).gave() &&
              !fixture.cache.checkpointState(fixture.blocks[2]) &&
              fixture.cache.checkpointState(fixture.blocks[1]),
          "restore did not refresh recency within checkpoint LRU");
  require(test::stateStep(fixture.cache).gave() &&
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
  require(reclaimStateBytes(fixture.cache) == 200 &&
              fixture.cache.snapshot().pool.pagesPrefix == 4 &&
              fixture.lookup(33).resumeBoundary() == 32,
          "checkpoint reclaim displaced older ordinary state or KV");
  require(test::releaseStep(fixture.cache).nowBytes == 100 &&
              fixture.cache.snapshot().pool.pagesPrefix == 3,
          "ordinary state and KV lost their shared LRU order");
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
  require(test::stateStep(fixture.cache).gave() &&
              fixture.lookup(33).resumeBoundary() == 0 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 2 &&
              fixture.cache.snapshot().stateCache.checkpointBytes == 500,
          "global reclaim evicted pinned checkpoint state");

  require(fixture.cache.reuseCompositeState(fixture.blocks[1], false) &&
              fixture.cache.retireCheckpointState(checkpoint),
          "pinned ordinary boundary could not upgrade its checkpoint");
  first.state.reset();
  second.state.reset();
  require(test::stateStep(fixture.cache).gave() &&
              fixture.lookup(97).resumeBoundary() == 64 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 0,
          "pinned boundary upgrade corrupted the eviction queues");
  require(test::stateStep(fixture.cache).gave() &&
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
  const auto reclaimed = test::evictStep(cache);
  require(reclaimed.gave() && reclaimed.stateBuffers &&
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
  require(reclaimStateBytes(fixture.cache) == 100,
          "demotion did not free the state's RAM at once");
  auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.bytes == 0 && snapshot.diskBytes == 100 && snapshot.entries == 1 &&
              snapshot.offloads == 1 && control->slots == 1,
          "demoted entry did not become its disk copy");
  require(!fixture.cache.pollTransfers(), "unfinished write was consumed early");
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TestState>(100), true);
  require(reclaimStateBytes(fixture.cache) == 100,
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
    require(!test::releaseStep(fixture.cache).gave(),
            "pinned restore lost its KV");
  }
  require(
      test::releaseStep(fixture.cache).gave(),
      "disk state permanently protected KV");
  require(fixture.cache.snapshot().stateCache.entries == 0 && control->slots == 0,
          "KV eviction leaked disk state");
}

void testTieredWriteReuseAndFailure() {
  for (bool fail : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
    require(reclaimStateBytes(fixture.cache) == 100,
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

// A full quota never gives up the only disk copy of a chain a queued request will restore.
void testDiskQuotaKeepsTheOnlyCopyAQueuedRequestRestores() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 1;
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control));
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(test::releaseStep(fixture.cache).gave() &&
              fixture.cache.pollTransfers() && control->slots == 1,
          "first write failed");
  const CacheProbe queued = fixture.cache.probe(std::span<const uint32_t>(fixture.prompt).first(33), {});
  fixture.cache.setWantedChains({queued.restoredChain()});
  static_cast<void>(test::releaseStep(fixture.cache));
  static_cast<void>(fixture.cache.pollTransfers());
  require(fixture.lookup(33).resumeBoundary() == 32,
          "the quota gave up the only copy of the chain a queued request restores");
  fixture.cache.setWantedChains({});
}

// A full temporary quota still demotes a needed leaf by replacing its oldest copy, the slot the
// take finds as freeDiskSpace does.
void testAFullTemporaryQuotaStillDemotesByReplacingItsOldestCopy() {
  test::TestKvTier tier;
  auto budget = std::make_shared<model::DiskBudget>(2 * tier.slotBytes());
  tier.budget = budget;
  Prefixes prefixes(&tier, true, budget);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 8;
  for (const uint64_t block : prefixes.blocks)
    prefixes.cache.publishCompositeState(block, std::make_shared<TieredState>(control), false);
  const auto step = [&] {
    const bool progress =
        test::evictStep(prefixes.cache).gave();
    tier.complete();
    static_cast<void>(prefixes.cache.pollTransfers());
    return progress;
  };
  // Two prefixes go to disk, state then leaf; the third's state follows.
  for (int index = 0; index < 5; ++index)
    require(step(), "the cache did not leave RAM");
  const uint64_t demotions = tier.demotions;
  const uint64_t diskBlocks = prefixes.cache.snapshot().kvTier.diskBlocks;
  require(budget->usedBytes() + tier.slotBytes() > budget->limitBytes() && diskBlocks == 2 &&
              !prefixes.cache.stateResident(prefixes.blocks[2]),
          "the quota was not full of two disk-only leaves beside a state on disk");
  const uint32_t freePages = prefixes.pool.freePageCount();
  const FreeResult freed = test::freeFromCache(prefixes.cache, test::kInServiceGrowth, {.pages = 1});
  require(freed.outcome == FreeOutcome::Pending && tier.demotions == demotions + 1 &&
              prefixes.cache.snapshot().kvTier.diskBlocks == diskBlocks,
          "a full quota did not demote the needed leaf by replacing the oldest copy: outcome " +
              std::to_string(static_cast<int>(freed.outcome)) + ", " +
              std::to_string(tier.demotions - demotions) + " demotions");
  tier.complete();
  static_cast<void>(prefixes.cache.pollTransfers());
  require(prefixes.pool.freePageCount() == freePages + 1, "the leaf's page did not free once its copy landed");
}

// A take never stops at a leaf it can neither write nor drop: with the quota
// full of copies it may not give up and no transfer in flight, a leaf whose
// disk subtree needs it goes with that subtree, as with no tier, for a claim
// that can wait. An Immediate claim passes over a leaf that needs a write and
// counts it so: short of its need, it takes nothing.
void testALeafWhoseWriteFindsNoSlotGoesWithItsSubtree() {
  for (const ClaimSpec claim : {test::kInServiceGrowth,
                                ClaimSpec{.cls = ClaimClass::InService, .timing = NeedTiming::Immediate}}) {
    test::TestKvTier tier;
    tier.capacity = 1;
    test::TestKvStorage storage{4, 100, 1};
    KvPool pool{storage, 0};
    engine::Cache cache(pool, &tier, nullptr);
    std::vector<uint32_t> chain(2 * KvCache::pageTokens + 1);
    std::iota(chain.begin(), chain.end(), 1000);
    cache.beginRequest(1);
    require(admitTokens(cache, 1, 2 * KvCache::pageTokens).granted(), "the chain's pages were denied");
    const uint64_t tail = test::publishBlocks(cache, 1, chain, 2 * KvCache::pageTokens);
    cache.endRequest(1);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    cache.publishCompositeState(tail, std::make_shared<TieredState>(control), false);
    require(test::stateStep(cache).gave() && cache.pollTransfers() && !cache.stateResident(tail),
            "the chain's state was not written");
    demoteLeaves(cache, tier, 1);
    // The tail's copies are the only ones of the chain a queued request
    // restores: the full quota may give up neither.
    cache.setWantedChains({cache.probe(chain, {}).restoredChain()});
    const std::vector<uint32_t> other(KvCache::pageTokens + 1, 7);
    cache.beginRequest(2);
    require(admitTokens(cache, 2, KvCache::pageTokens).granted(), "the other prefix's page was denied");
    static_cast<void>(test::publishBlocks(cache, 2, other, KvCache::pageTokens));
    cache.endRequest(2);
    cache.beginRequest(3);
    require(admitTokens(cache, 3, uint64_t{pool.freePageCount()} * KvCache::pageTokens).granted() &&
                pool.freePageCount() == 0 && tier.slots == tier.capacity,
            "the fixture did not fill the pool and the quota");
    const uint32_t demotions = tier.demotions;
    const uint32_t cachedPages = cache.snapshot().pool.pagesPrefix;
    const FreeResult freed = test::freeFromCache(cache, claim, {.pages = 2});
    const uint32_t chainTokens = cache.probe(chain, {}).cachedTokens();
    const uint32_t leftPages = cache.snapshot().pool.pagesPrefix;
    const bool canWait = claim.timing == NeedTiming::CanWait;
    require(canWait ? freed.outcome == FreeOutcome::Met && freed.pages == 2 && !chainTokens &&
                          leftPages + 2 == cachedPages
                    : freed.outcome == FreeOutcome::Short && !freed.pages && chainTokens && leftPages == cachedPages,
            "a take stopped at a leaf no slot can take: timing " + std::to_string(static_cast<int>(claim.timing)) +
                ", outcome " + std::to_string(static_cast<int>(freed.outcome)) + ", " +
                std::to_string(freed.pages) + " pages, chain cached " + std::to_string(chainTokens) +
                " tokens, cached pages " + std::to_string(cachedPages) + " -> " + std::to_string(leftPages));
    require(tier.demotions == demotions, "a take wrote a leaf no slot can take");
    cache.setWantedChains({});
    cache.endRequest(3);
    cache.checkInvariants();
  }
}

// A waiting claim starts the demotions of the whole run up an idle chain,
// up to what completes an extent, so it frees once those writes land, not
// one page per landing. The chain fills three extents of four pages and ends in a
// state on disk, so each block needs a copy before its page frees. A parent's
// write that finishes before its child's still lands after it, freeing its
// page.
void testAWaitingClaimDemotesAWholeChainRunAtOnce() {
  test::TestKvTier tier;
  tier.transferLimit = 64;
  auto budget = std::make_shared<model::DiskBudget>(16 * tier.slotBytes());
  tier.budget = budget;
  test::TestKvStorage storage{12, 100, 4};
  KvPool pool{storage, 4};
  engine::Cache cache(pool, &tier, budget);
  std::vector<uint32_t> prompt(12 * KvCache::pageTokens + 1);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 12 * KvCache::pageTokens).granted(), "the chain's pages were denied");
  cache.publishCommittedBlocks(1, prompt, 12 * KvCache::pageTokens, {});
  const uint64_t tail = cache.blockAt(1, 12 * KvCache::pageTokens);
  cache.endRequest(1);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  cache.publishCompositeState(tail, std::make_shared<TieredState>(control), false);
  require(test::stateStep(cache).gave() && cache.pollTransfers() &&
              !cache.stateResident(tail),
          "the chain's state was not written");
  const uint64_t extent = storage.extentPages() * 100;
  require(pool.freePageCount() == 0 && pool.allocatedBytes() == 3 * extent, "fixture geometry changed");
  const ClaimSpec start{.cls = ClaimClass::Start};
  const uint32_t demotions = tier.demotions;
  const FreeResult waits = test::freeFromCache(cache, start, {.bytes = extent});
  const uint32_t started = tier.demotions - demotions;
  // The parents' writes finish before the tail's, which started first.
  for (size_t index = tier.transfers.size() - started + 1; index < tier.transfers.size(); ++index)
    tier.transfers[index]->ready = true;
  static_cast<void>(cache.pollTransfers());
  const Returning ahead = cache.returning();
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  const uint64_t allocated = pool.allocatedBytes();
  const FreeResult freed = test::freeFromCache(cache, start, {.bytes = extent});
  require(waits.outcome == FreeOutcome::Pending && started >= storage.extentPages() && waits.laterBytes >= extent &&
              ahead.landingBytes >= extent && freed.outcome == FreeOutcome::Met &&
              allocated - pool.allocatedBytes() >= extent,
          "a waiting claim did not free an extent once its writes landed: " + std::to_string(started) +
              " demotions started for " + std::to_string(waits.laterBytes) + " bytes, " +
              std::to_string(ahead.landingBytes) + " landing past the tail, then outcome " +
              std::to_string(static_cast<int>(freed.outcome)) + " with " +
              std::to_string(allocated - pool.allocatedBytes()) + " bytes released");
  cache.checkInvariants();
}

// Certain reuse per claim class: Start and Optional take the resume point but never a
// queued chain; Growth takes neither; InService takes them oldest first.
void testCertainReuseByClass() {
  struct Case final {
    ClaimSpec spec;
    bool met;
    bool keepsQueued;
    bool keepsResumePoint;
  };
  for (const Case each : {Case{{.cls = ClaimClass::Start}, true, true, false},
                          Case{{.cls = ClaimClass::Optional, .requestId = 1, .timing = NeedTiming::Immediate}, true, true, false},
                          Case{{.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate}, false, true, true},
                          Case{{.cls = ClaimClass::InService}, true, false, true}}) {
    Prefixes prefixes;
    for (const uint64_t block : prefixes.blocks)
      prefixes.cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
    const CacheProbe queued = prefixes.cache.probe(prefixes.prompts[0], {});
    prefixes.cache.setWantedChains({queued.restoredChain()});
    const FreeResult freed = test::freeFromCache(prefixes.cache, each.spec, {.pages = 3});
    const bool keptQueued = prefixes.cache.stateResident(prefixes.blocks[0]);
    const bool keptResumePoint = prefixes.cache.stateResident(prefixes.blocks[3]);
    prefixes.cache.setWantedChains({});
    require((freed.outcome == FreeOutcome::Met) == each.met && keptQueued == each.keepsQueued &&
                keptResumePoint == each.keepsResumePoint,
            "a " + std::to_string(static_cast<int>(each.spec.cls)) +
                " claim took the wrong certain reuse: met " + std::to_string(freed.outcome == FreeOutcome::Met) +
                ", queued chain kept " + std::to_string(keptQueued) + ", resume point kept " +
                std::to_string(keptResumePoint));
  }
}

// A rolling checkpoint uses the SSD tier like any state: written, replaced by recency, retired with
// its successor; its leaf is demoted, not dropped.
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
  require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
          "RAM checkpoint was not reclaimed");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.offloads == 2 && stats.bytes == 0 && stats.checkpointEntries == 1 &&
              stats.checkpointBytes == 0 && control->slots == 1,
          "RAM checkpoint was dropped although the quota had room");
  control->capacity = 1;
  fixture.cache.publishCompositeState(fixture.blocks[3],
                                      std::make_shared<TieredState>(control), true);
  require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers() &&
              control->slots == 1 && fixture.cache.snapshot().stateCache.offloads == 3 &&
              fixture.cache.snapshot().stateCache.checkpointEntries == 1 &&
              fixture.cache.checkpointState(fixture.blocks[3]) &&
              !fixture.cache.checkpointState(fixture.blocks[2]),
          "a full quota kept the older checkpoint");
  require(
      test::evictStep(fixture.cache).gave() &&
          tier.demotions == 1 && fixture.cache.snapshot().pool.pagesPrefix == 4,
      "the checkpoint's leaf was dropped instead of demoted");
}

void testDiskPromotionAndInvalidation() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const auto block = fixture.blocks[3];
  publishReusable(fixture, block, std::make_shared<TieredState>(control));
  require(test::releaseStep(fixture.cache).gave() &&
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
      test::releaseStep(fixture.cache).gave(),
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
  require(test::stateStep(fixture.cache).gave(),
          "required publication could not recycle a buffer");
  auto snapshot = fixture.cache.snapshot().stateCache;
  require(snapshot.entries == 2 && snapshot.bytes == 100 && snapshot.diskBytes == 100,
          "demotion took a second state or kept the victim's RAM");
  // One write in flight: a second victim, whose write would have to wait,
  // stays.
  require(!test::stateStep(fixture.cache, ReclaimClass::InUse).gave() &&
              fixture.cache.snapshot().stateCache.bytes == 100,
          "a claim dropped a victim whose write would wait");
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
    require(test::releaseStep(fixture.cache).gave() &&
                fixture.cache.pollTransfers(),
            "promotion fixture write failed");
    auto first = fixture.lookup(129);
    auto peer = fixture.lookup(129);
    PromotionTicket ticket;
    ticket.available = available;
    fixture.cache.promoteState(first, [&] { return ticket.snapshot(); });
    if (available) fixture.cache.promoteState(peer, [&] { return ticket.snapshot(); });
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

// Promotion takes no cache room: a refused copy leaves the restored state on disk and every RAM
// state where it was.
void testPromotionNeverDropsAUniqueState() {
  for (const bool onDisk : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
    require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
            "the state to promote was not written");
    if (onDisk) {
      // Written and published again, the older state has a RAM copy and a
      // disk copy beside it.
      fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
      require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
              "the older state was not written");
      fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
    } else {
      fixture.publish(1);
    }
    const auto lookup = fixture.lookup(129);
    PromotionTicket ticket;
    ticket.deniedCopies = 1;
    const uint64_t evictions = fixture.cache.snapshot().stateCache.evictions;
    fixture.cache.promoteState(lookup, [&] { return ticket.snapshot(); });
    const auto stats = fixture.cache.snapshot().stateCache;
    require(ticket.copies == 1 && stats.promotions == 0 && stats.promotionsSkipped == 1 &&
                stats.evictions == evictions && fixture.cache.stateResident(fixture.blocks[1]),
            onDisk ? "a refused promotion took the RAM of a state with a disk copy"
                   : "a refused promotion dropped a state's only copy");
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
  require(test::releaseStep(fixture.cache).gave() &&
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
  require(test::stateStep(fixture.cache).gave() && !fixture.cache.pollTransfers(),
          "re-eviction of a state with a disk copy wrote it again");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.bytes == 100 && stats.diskBytes == 100 && stats.offloads == 1 &&
              fixture.lookup(33).state && !fixture.lookup(33).state->state()->residentBytes(),
          "the disk copy did not serve after the RAM copy left");
}

// A lookup that finds a block's KV but no state in either tier reports a lost state; disk copies
// and dropped checkpoints are not lost.
void testLostStatesAreCounted() {
  CacheFixture fixture;
  fixture.publish(3);
  require(test::stateStep(fixture.cache).gave(), "state was not dropped");
  {
    auto lookup = fixture.lookup(129);
    require(lookup.kvBoundary == 128 && !lookup.state && lookup.lostState,
            "dropped state was not recognised");
    fixture.cache.recordLookup(lookup);
  }
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
          "state was not demoted");
  {
    auto hit = fixture.lookup(129);
    require(hit.state && !hit.lostState, "a disk hit was counted as a lost state");
  }
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TestState>(100), true);
  require(test::stateStep(fixture.cache).gave(), "checkpoint was not dropped");
  {
    auto shallow = fixture.lookup(65);
    require(!shallow.state && !shallow.lostState, "a dropped checkpoint counted as a lost state");
  }
  require(fixture.cache.snapshot().lookup.lostStateMisses == 1, "lost states were not counted");
}

void testDiskCheckpointRamAccounting() {
  for (bool republish : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    fixture.cache.publishCompositeState(fixture.blocks[3],
                                        std::make_shared<TieredState>(control), true);
    const auto point = fixture.cache.checkpointState(fixture.blocks[3]);
    require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
            "checkpoint demotion failed");
    if (republish) {
      fixture.cache.publishCompositeState(fixture.blocks[3],
                                          std::make_shared<TestState>(100), true);
    } else {
      auto lookup = fixture.lookup(129);
      PromotionTicket ticket;
      fixture.cache.promoteState(lookup, [&] { return ticket.snapshot(); });
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

// A state written straight to disk from its lane: hits during the write are disk hits, one write
// holds the staging buffer, and its leaf is demoted, not dropped.
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
  rejects([&] { static_cast<void>(fixture.cache.publishStateToDisk(fixture.blocks[3], write, false)); },
          "a block's state is published once; reuse it through reuseStoredState",
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
      test::evictStep(fixture.cache).gave() &&
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

// A write that fails under a reader leaves the block nothing; a publication meanwhile takes the
// entry over and outlives the reader.
void testFailedWriteUnderALookup() {
  for (bool onDisk : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    const auto block = fixture.blocks[3];
    publishReusable(fixture, block, std::make_shared<TieredState>(control));
    require(test::stateStep(fixture.cache).gave(), "state was not demoted");
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

// An invalidated state keeps its copies until its readers let go; a publication meanwhile takes the
// entry over.
void testInvalidatedStateUnderALookup() {
  for (bool onDisk : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    const auto block = fixture.blocks[3];
    publishReusable(fixture, block, std::make_shared<TieredState>(control));
    require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
            "state was not written");
    auto first = fixture.lookup(129);
    auto reader = fixture.lookup(129);
    PromotionTicket ticket;
    fixture.cache.promoteState(first, [&] { return ticket.snapshot(); });
    fixture.cache.invalidateState(block);
    first = {};
    auto stats = fixture.cache.snapshot().stateCache;
    require(stats.entries == 1 && stats.pinned == 1 && stats.bytes == 100 && stats.diskBytes == 100 &&
                !fixture.lookup(129).state,
            "the invalidated state did not stay under its reader, or matched a lookup");
    if (onDisk) {
      const StateWriter write = [&](std::function<void()>) { return writeState(control); };
      require(fixture.cache.publishStateToDisk(block, write, false) && fixture.cache.pollTransfers(),
              "the disk publication at the invalidated state failed");
    } else {
      fixture.cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
    }
    reader = {};
    stats = fixture.cache.snapshot().stateCache;
    require(stats.entries == 1 && stats.pinned == 0 && stats.bytes == (onDisk ? 0 : 100) &&
                stats.diskBytes == (onDisk ? 100 : 0) && control->slots == (onDisk ? 1 : 0) &&
                fixture.lookup(129).state,
            "the publication did not take over the invalidated state");
  }
}

// A failed write is counted even when its entry left before it landed: a retired checkpoint,
// or a condemned state a new publication replaced.
void testFailedWriteIsCountedAfterItsEntryLeft() {
  for (bool replaced : {false, true}) {
    CacheFixture fixture;
    auto control = std::make_shared<TransferControl>();
    const auto block = fixture.blocks[3];
    if (replaced) {
      publishReusable(fixture, block, std::make_shared<TieredState>(control));
      require(test::stateStep(fixture.cache).gave(), "state was not demoted");
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
  require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers() &&
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

// Quota replacement is one order over KV and state copies: redundant copies first, then the oldest
// unique one.
void testDiskReplacementSpansStatesAndKv() {
  test::TestKvTier tier;
  tier.capacity = 1;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 1;
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
          "state was not demoted");
  require(test::evictStep(fixture.cache).gave() && tier.demotions == 1,
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
    fixture.cache.promoteState(lookup, [&] { return ticket.snapshot(); });
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
  require(test::stateStep(fixture.cache).gave() &&
              fixture.cache.snapshot().stateCache.offloads == 1,
          "the restored state was written again");
  require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
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

// Under random traffic the SSD tier only adds hits: RAM contents match a cache without the tier
// step for step.
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
        static_cast<void>(test::stateStep(off.cache));
        static_cast<void>(test::stateStep(on.cache));
        static_cast<void>(on.cache.pollTransfers());
      }
      require(on.cache.snapshot().stateCache.bytes == off.cache.snapshot().stateCache.bytes,
              "RAM occupancy diverged from the cache without the tier");
    }
  }
}

// A demotion must not cost a second state: with room for two states the tier keeps the hit a cache
// without it gets.
// A claim the engine's limit refused for one KV page takes a cached page in
// place, as the count said, though no extent can empty: a lane's request
// holds the rest of the pool. One page is written for it, not the chain.
void testAMemoryClaimForAPageTakesACachedPageInPlace() {
  test::TestKvTier tier;
  tier.transferLimit = 64;
  auto budget = std::make_shared<model::DiskBudget>(16 * tier.slotBytes());
  tier.budget = budget;
  test::TestKvStorage storage{12, 100, 4};
  KvPool pool{storage, 4};
  engine::Cache cache(pool, &tier, budget);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, 10 * KvCache::pageTokens).granted(), "the lane's pages were denied");
  std::vector<uint32_t> prompt(2 * KvCache::pageTokens + 1);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(2);
  require(admitTokens(cache, 2, 2 * KvCache::pageTokens).granted(), "the chain's pages were denied");
  cache.publishCommittedBlocks(2, prompt, 2 * KvCache::pageTokens, {});
  const uint64_t tail = cache.blockAt(2, 2 * KvCache::pageTokens);
  cache.endRequest(2);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  cache.publishCompositeState(tail, std::make_shared<TieredState>(control), false);
  require(test::stateStep(cache).gave() && cache.pollTransfers() && !cache.stateResident(tail),
          "the chain's state was not written");
  require(pool.freePageCount() == 0 && pool.reusablePages() == 2, "fixture geometry changed");
  const uint32_t demotions = tier.demotions;
  const FreeResult freed = test::freeFromCache(cache, test::kInServiceGrowth, {.bytes = 100, .pages = 1});
  require(freed.outcome != FreeOutcome::Short && freed.pages == 1 && tier.demotions - demotions == 1,
          "a memory claim for a page took " + std::to_string(freed.pages) + " pages with " +
              std::to_string(tier.demotions - demotions) + " writes, outcome " +
              std::to_string(static_cast<int>(freed.outcome)));
  cache.checkInvariants();
}

void testDemotionCostsNoSecondState() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control)); // A, A
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false); // B
  require(test::stateStep(fixture.cache).gave(),
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
        require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
                "shared prefix state demotion failed");
      }
      fixture.cache.publishCompositeState(fixture.blocks[3],
                                          std::make_shared<TieredState>(control), false);
      require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
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

// A leaf whose state went to disk is written, not dropped; its prefix then matches through disk and
// requests wait for the restore.
void testKvDemotionAndRestoreLifecycle() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  // The state goes first and frees its RAM at once; the leaf under it is
  // then worth keeping.
  require(reclaimStateBytes(fixture.cache) == 100 &&
              tier.demotions == 0 && fixture.cache.pollTransfers(),
          "state was not demoted ahead of its leaf");
  const auto step = test::evictStep(fixture.cache);
  require(step.gave() && step.nowBytes == 0 && tier.demotions == 1 &&
              tier.slots == 1 && fixture.pool.freePageCount() == 0,
          "leaf under a disk state was dropped or freed before its copy landed");
  auto stats = fixture.cache.snapshot();
  require(stats.kvTier.pendingPages == 1 && stats.pool.pagesPrefix == 4 &&
              stats.kvTier.diskBlocks == 1,
          "pending demotion was not accounted");
  require(!fixture.cache.pollTransfers() &&
              !test::evictStep(fixture.cache).gave(),
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
  require(test::evictStep(fixture.cache).gave() &&
              tier.demotions == 1 && fixture.pool.freePageCount() == 1 &&
              fixture.cache.snapshot().pool.pagesPrefix == 3 &&
              fixture.cache.snapshot().stateCache.diskBytes == 100,
          "a block with a disk copy was written again or lost its state");
}

// Pressure writes every state it evicts to the SSD cache, which keeps them by its own LRU: a
// prompt's last checkpoint, which another conversation with the same long system prompt resumes
// from, and a point its conversation went on from. Both stay restorable from their disk copies.
void testEvictedCheckpointsAndSupersededPointsStayOnTheSsd() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  // The system prompt ends at the first block, with its last checkpoint; the
  // conversation's point at the second block is superseded by the one at its
  // last.
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), true);
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(!fixture.cache.newestPoint(fixture.blocks[1]) && fixture.cache.newestPoint(fixture.blocks[3]),
          "the conversation's points are not as the test needs");
  // In-service work takes everything, each write landing before its next
  // step.
  for (uint32_t step = 0; step < 64 && test::releaseStep(fixture.cache).gave(); ++step) {
    tier.complete();
    static_cast<void>(fixture.cache.pollTransfers());
  }
  require(fixture.cache.snapshot().stateCache.offloads == 3 && !fixture.cache.stateResident(fixture.blocks[0]) &&
              !fixture.cache.stateResident(fixture.blocks[1]) && !fixture.cache.stateResident(fixture.blocks[3]) &&
              tier.demotions == 4 && !fixture.pool.allocatedBytes(),
          "pressure did not write every state it evicted and the KV they restore through");

  // Another conversation with the same system prompt resumes from the
  // checkpoint's disk copy.
  std::vector<uint32_t> other(fixture.prompt.begin(), fixture.prompt.begin() + KvCache::pageTokens);
  other.push_back(4242);
  {
    CacheLookup lookup = fixture.cache.lookup(other, {});
    require(lookup.state && lookup.state->kvBlock() == fixture.blocks[0] && !lookup.state->state()->residentBytes() &&
                lookup.resumeBoundary() == KvCache::pageTokens,
            "a conversation with the same system prompt did not find the checkpoint on the SSD");
    fixture.cache.beginRequest(2);
    require(admitRestore(fixture.cache, 2, lookup).granted() &&
                fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::Pending,
            "the system prompt's KV was not restored");
    tier.complete();
    require(fixture.cache.pollTransfers() && fixture.cache.kvRestoreStatus(2) == KvRestoreStatus::None,
            "the system prompt's restore did not complete");
  }
  fixture.cache.endRequest(2);
  // The conversation's earlier turn resumes from the superseded point's.
  {
    CacheLookup lookup = fixture.lookup(2 * KvCache::pageTokens + 1);
    require(lookup.state && lookup.state->kvBlock() == fixture.blocks[1] && !lookup.state->state()->residentBytes() &&
                lookup.resumeBoundary() == 2 * KvCache::pageTokens,
            "the conversation's earlier turn did not find the superseded point on the SSD");
  }
  fixture.cache.checkInvariants();
}

// Leaves nothing depends on are dropped, never written; a parent whose child
// went to disk is written when its turn comes, so the chain stays whole.
void testTailsDropAndParentsFollowToDisk() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  require(test::evictStep(fixture.cache).gave() &&
              tier.demotions == 0 && fixture.pool.freePageCount() == 1,
          "a tail was written");
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[2], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache) == 100 &&
              fixture.cache.pollTransfers(),
          "state was not demoted first");
  for (uint32_t written = 1; written <= 3; ++written) {
    require(test::evictStep(fixture.cache).gave() &&
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

// Once the state a disk child was written for is gone, the parent is not written for that child:
// both go and the page returns.
void testDiskCopiesNoStateNeedsGoWithTheLeaf() {
  {
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    auto control = std::make_shared<TransferControl>();
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
    require(reclaimStateBytes(fixture.cache) == 100 &&
                test::evictStep(fixture.cache).gave() &&
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
    require(reclaimStateBytes(fixture.cache) == 100 &&
                fixture.cache.pollTransfers() &&
                test::evictStep(fixture.cache).gave() &&
                tier.demotions == 1,
            "leaf was not written");
    tier.complete();
    require(fixture.cache.pollTransfers() && tier.slots == 1, "leaf did not land");
    require(test::evictStep(fixture.cache).gave() &&
                tier.demotions == 1 && tier.slots == 0 && fixture.pool.freePageCount() == 2 &&
                fixture.cache.snapshot().stateCache.entries == 0,
            "a leaf was written after making room took the state it was written for");
  }
}

// A demotion the busy tier refuses keeps its leaf (Pending) while transfers will free room; with
// nothing in flight the leaf goes.
void testRefusedDemotionKeepsTheLeafWhileTransfersLand() {
  test::TestKvTier tier;
  tier.transferLimit = 1;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache) == 100 &&
              fixture.cache.pollTransfers(),
          "state was not demoted first");
  // One demotion is all the tier takes at a time, and it stays in flight.
  require(test::evictStep(fixture.cache).gave() &&
              tier.demotions == 1 && tier.inFlight() == 1,
          "the first leaf was not written");
  // Its parent is no leaf while the page being written is still resident, so
  // the pass has nothing to give and says so: wait for the transfer.
  const FreeResult busy = test::evictStep(fixture.cache);
  require(!busy.gave() && busy.outcome == FreeOutcome::Pending && tier.demotions == 1 &&
              fixture.cache.snapshot().pool.pagesPrefix == 4,
          "a leaf was dropped or the wait was not reported while the tier was busy");
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 32).ledger == FreeOutcome::Pending,
          "a request was failed while a transfer was landing");
  fixture.cache.endRequest(2);
  tier.complete();
  require(fixture.cache.pollTransfers() && tier.inFlight() == 0, "the demotion did not land");
  require(test::evictStep(fixture.cache).gave() && tier.demotions == 2,
          "the leaf was not written once the tier had room");
  tier.complete();
  require(fixture.cache.pollTransfers(), "second leaf did not land");
}

void testUnusableTierDropsTheLeafInstead() {
  test::TestKvTier tier;
  tier.transferLimit = 0;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache) == 100 &&
              fixture.cache.pollTransfers(),
          "state was not demoted first");
  // Nothing is in flight and nothing ever makes room: waiting would be
  // waiting for nothing, so the leaf and its disk copy go instead.
  require(test::evictStep(fixture.cache).gave() &&
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
  require(reclaimStateBytes(deep.cache) == 100 &&
              deep.cache.pollTransfers(),
          "deep state was not demoted");
  require(test::evictStep(deep.cache).gave() && second.demotions == 1,
          "leaf was not written");
  second.complete();
  require(deep.cache.pollTransfers(), "leaf did not land");
  second.writableFile = false;
  {
    auto held = deep.lookup(129);
    require(held.state && !test::evictStep(deep.cache).gave() &&
                deep.cache.snapshot().pool.pagesPrefix == 3 &&
                deep.cache.snapshot().kvTier.diskBlocks == 1,
            "a leaf was dropped under a disk subtree a lookup holds");
  }
  require(test::evictStep(deep.cache).gave() &&
              deep.cache.snapshot().pool.pagesPrefix == 2 &&
              deep.cache.snapshot().kvTier.diskBlocks == 0 &&
              deep.cache.snapshot().stateCache.entries == 0,
          "an unwritable tier kept the leaf and its disk subtree");
  deep.cache.beginRequest(2);
  require(admitTokens(deep.cache, 2, 64).granted(),
          "the dropped leaf's page did not come back");
  deep.cache.endRequest(2);
}

// While the write budget is spent a state not in use goes without a copy; one in use is still
// written.
void testSpentWriteBudgetHoldsBackStatesNotInUse() {
  test::WriteBudget writes;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->writeBudget = writes.budget;
  writes.spend();
  {
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control),
                                        false);
    require(reclaimStateBytes(fixture.cache) == 100 && control->slots == 0 &&
                !fixture.cache.snapshot().stateCache.entries,
            "a state not in use was written, or kept, while the write budget was spent");
  }
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  const StateUse use = fixture.cache.useState(fixture.blocks[3]);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control),
                                      false);
  require(reclaimStateBytes(fixture.cache) == 100 && fixture.cache.pollTransfers() &&
              control->slots == 1,
          "a state in use was not written while the write budget was spent");
}

// A state written while the budget had room keeps its whole chain after the budget runs out.
void testSpentWriteBudgetKeepsTheChainsOfWrittenStates() {
  test::WriteBudget writes;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->writeBudget = writes.budget;
  test::TestKvTier tier;
  tier.capacity = 8;
  CacheFixture fixture(&tier);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control),
                                      false);
  require(reclaimStateBytes(fixture.cache) == 100 && fixture.cache.pollTransfers() &&
              control->slots == 1,
          "the state was not written while the budget had room");
  writes.spend();
  for (uint32_t leaf = 0; leaf < 4; ++leaf) {
    require(test::evictStep(fixture.cache).gave() &&
                tier.demotions == leaf + 1,
            "a page the written state needs was not written while the budget was spent");
    tier.complete();
    require(fixture.cache.pollTransfers(), "the page did not land");
  }
  const CacheSnapshot held = fixture.cache.snapshot();
  require(held.kvTier.diskBlocks == 4 && held.stateCache.entries == 1 && control->slots == 1 &&
              !held.kvTier.demotionsRefused && fixture.lookup(129).state.has_value(),
          "the written state lost its chain when the budget ran out");
}

// A limit that cannot hold one slot makes no room: a leaf a written state
// needs is not written, nothing the tier holds is given up for it, and it
// goes as with no tier, with the copies below it that need it.
void testLimitBelowASlotMakesNoRoom() {
  auto budget = std::make_shared<model::DiskBudget>(uint64_t{1} << 30);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  test::TestKvTier tier;
  tier.capacity = 8;
  CacheFixture fixture(&tier, false, budget);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control),
                                      false);
  require(reclaimStateBytes(fixture.cache) == 100 && fixture.cache.pollTransfers() &&
              control->slots == 1,
          "the state was not written");
  require(test::evictStep(fixture.cache).gave() &&
              tier.demotions == 1,
          "the deepest page the state needs was not written");
  tier.complete();
  require(fixture.cache.pollTransfers(), "the page did not land");
  budget->setLimitBytes(tier.slotBytes() - 1);
  static_cast<void>(test::evictStep(fixture.cache));
  const CacheSnapshot held = fixture.cache.snapshot();
  require(tier.demotions == 1 && !held.kvTier.diskBlocks && !fixture.lookup(129).state,
          "a limit below a slot wrote a page, or kept a leaf no write can take");
  fixture.cache.checkInvariants();
}

// A leaf with disk-only children whose demotion cannot start, with nothing
// in flight to wait for, goes with those children, as with no tier: it is
// never erased under them, and the request has its pages rather than wait
// for something that will never happen.
void testParentGoesWithItsDiskChildrenWhenNoWriteCanStart() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier, false, test::ampleDiskBudget());
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache) == 100 &&
              fixture.cache.pollTransfers() &&
              test::evictStep(fixture.cache).gave() &&
              tier.demotions == 1,
          "leaf was not written");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "leaf did not land");
  tier.transferLimit = 0;
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 64).granted() && tier.demotions == 1 &&
              !fixture.cache.snapshot().kvTier.diskBlocks && !fixture.lookup(129).state,
          "the parent of a disk block stayed with no write to wait for, or went without its subtree");
  fixture.cache.endRequest(2);
  fixture.cache.checkInvariants();
}

// While one state write is in flight, a pressure pass keeps the next state
// in RAM rather than dropping it; the pass after the write takes it.
void testSecondStateWaitsForTheWrite() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  publishReusable(fixture, fixture.blocks[0], std::make_shared<TieredState>(control));
  publishReusable(fixture, fixture.blocks[3], std::make_shared<TieredState>(control));
  require(reclaimStateBytes(fixture.cache) == 100,
          "first state was not written");
  auto stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 100 && stats.diskBytes == 100 && stats.offloads == 1,
          "first write did not free its RAM");
  require(!test::evictStep(fixture.cache).gave(),
          "a state was dropped or a leaf under one was taken while a write was in flight");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 100 && stats.evictions == 0,
          "the waiting state did not stay");
  control->ready = true;
  require(fixture.cache.pollTransfers() &&
              reclaimStateBytes(fixture.cache) == 100,
          "the waiting state was not written after the first landed");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.entries == 2 && stats.bytes == 0 && stats.diskBytes == 200 && stats.offloads == 2 &&
              control->slots == 2,
          "second write did not land beside the first");
}

// A checkpoint whose write waits for the slot holds nothing else back: KV and ordinary states
// behind it go meanwhile.
void testWaitingCheckpointHoldsBackNothingElse() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  require(test::stateStep(fixture.cache).gave(), "first state was not written");
  fixture.publish(2);
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control),
                                      true);
  // The idle tail is the oldest and needs no transfer; the ordinary state
  // cannot be written and goes next, then the leaf it stood on.
  require(test::evictStep(fixture.cache).gave() &&
              fixture.cache.snapshot().pool.pagesPrefix == 3,
          "the waiting checkpoint held back an idle KV tail");
  require(reclaimStateBytes(fixture.cache) == 100 &&
              fixture.cache.snapshot().stateCache.entries == 2,
          "the waiting checkpoint held back an ordinary state");
  require(test::evictStep(fixture.cache).gave() &&
              fixture.cache.snapshot().pool.pagesPrefix == 2,
          "the waiting checkpoint held back the leaf a state left");
  const FreeResult waiting = test::evictStep(fixture.cache);
  auto stats = fixture.cache.snapshot().stateCache;
  require(!waiting.gave() && (waiting.outcome == FreeOutcome::Pending) && stats.bytes == 100 &&
              stats.checkpointEntries == 1 && stats.checkpointEvictions == 0,
          "the checkpoint did not wait for the write in flight");
  control->ready = true;
  require(fixture.cache.pollTransfers() &&
              reclaimStateBytes(fixture.cache) == 100,
          "the checkpoint was not written after the first landed");
  stats = fixture.cache.snapshot().stateCache;
  require(stats.offloads == 2 && stats.bytes == 0 && stats.checkpointEntries == 1 &&
              control->slots == 2,
          "the checkpoint did not land beside the first write");
}

// A refusal ends the scan: the tier is full for every leaf alike, so one attempt costs one
// refusal, not one per cached block.
void testFullTierStopsTheScan() {
  test::TestKvTier tier;
  Prefixes p(&tier, false, test::ampleDiskBudget());
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  for (uint64_t block : p.blocks) {
    p.cache.publishCompositeState(block, std::make_shared<TieredState>(control), false);
    require(test::stateStep(p.cache).gave() && p.cache.pollTransfers(),
            "state was not demoted");
  }
  // One transfer at a time: the first leaf takes it and the request waits for
  // that page rather than evicting more.
  tier.transferLimit = 1;
  p.cache.beginRequest(9);
  require(admitTokens(p.cache, 9, 32).ledger == FreeOutcome::Pending &&
              tier.demotions == 1 && p.cache.snapshot().pool.pagesPrefix == 4,
          "the first leaf was not written, or a leaf was dropped");
  // A larger shortfall meets a tier that the transfer in flight fills. Every
  // leaf would answer the same, so the scan asks once and waits.
  require(admitTokens(p.cache, 9, 64).ledger == FreeOutcome::Pending &&
              p.cache.snapshot().kvTier.demotionsRefused == 1 &&
              p.cache.snapshot().pool.pagesPrefix == 4,
          "a full tier was asked once per leaf, or a leaf was dropped");
  tier.complete();
  require(p.cache.pollTransfers() && admitTokens(p.cache, 9, 32).granted(),
          "the page did not return");
  p.cache.endRequest(9);
}

// Restores in flight return no page: a shortfall they hold every page of is denied, and the
// restored leaf frees its page once its request ends.
void testRestoresInFlightReturnNothingToAShortfall() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache) == 100 &&
              fixture.cache.pollTransfers() &&
              test::evictStep(fixture.cache).gave(),
          "leaf was not written");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "leaf did not land");
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(2);
  require(admitRestore(fixture.cache, 2, lookup).granted() && fixture.pool.freePageCount() == 0,
          "restore did not take the free page");
  fixture.cache.beginRequest(3);
  require(admitTokens(fixture.cache, 3, 32).ledger == FreeOutcome::Short,
          "a shortfall during a restore waited for what the restore returns");
  tier.complete();
  require(fixture.cache.pollTransfers() &&
              admitTokens(fixture.cache, 3, 32).ledger == FreeOutcome::Short,
          "pages held by an active request were not exhausted");
  lookup = {};
  fixture.cache.endRequest(2);
  require(admitTokens(fixture.cache, 3, 32).granted() && tier.demotions == 1,
          "the restored leaf did not give up its page for nothing");
  fixture.cache.endRequest(3);
}

// Pages already on their way back make a request wait instead of the cache demoting more than it
// needs.
void testPendingPagesGateAllocation() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier, false, test::ampleDiskBudget());
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 32).ledger == FreeOutcome::Pending &&
              tier.demotions == 1 && fixture.cache.snapshot().pool.pagesPrefix == 4 &&
              fixture.cache.snapshot().kvTier.pendingPages == 1,
          "allocation evicted past the page on its way back");
  require(admitTokens(fixture.cache, 2, 32).ledger == FreeOutcome::Pending &&
              tier.demotions == 1,
          "a retry before the copy landed demoted more");
  tier.complete();
  require(fixture.cache.pollTransfers() && admitTokens(fixture.cache, 2, 32).granted() &&
              fixture.cache.pageTable(2).pages.size() == 1 &&
              fixture.cache.snapshot().pool.pagesPrefix == 3,
          "pages did not return to the waiting request");
  fixture.cache.endRequest(2);
}

// A failed demotion leaves the block in RAM without a copy; a failed restore removes the block and
// its state once its requests let go.
void testTransferFailures() {
  {
    test::TestKvTier tier;
    CacheFixture fixture(&tier);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
    require(reclaimStateBytes(fixture.cache) == 100 &&
                fixture.cache.pollTransfers() &&
                test::evictStep(fixture.cache).gave() &&
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
    require(reclaimStateBytes(fixture.cache) == 100 &&
                fixture.cache.pollTransfers() &&
                test::evictStep(fixture.cache).gave() &&
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

// Nothing below a block whose read failed matches any more; restored and disk blocks below it leave
// once the request lets go.
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
    require(test::stateStep(cache).gave() && cache.pollTransfers(),
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

// A busy tier gives up no disk copy for a demotion that cannot start yet.
void testBusyTierPreservesDiskVictim() {
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
      require(test::stateStep(cache).gave() && cache.pollTransfers(),
              "state was not demoted");
      prompts[i].push_back(9999);
    }
    require(test::evictStep(cache).gave(), "first demotion failed");
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
      // The restored prefix keeps its disk copy: its page goes first, at
      // once, writing nothing.
      require(test::evictStep(cache).gave() && tier.demotions == 1,
              "the restored prefix did not give its page before a write");
    }
    require(test::evictStep(cache).gave() && tier.slots == 2,
            "second demotion did not fill quota");
    // While the tier is busy, a leaf that needs a write stays.
    require(!test::evictStep(cache).gave(), "busy tier did not wait");
    require(tier.slots == 2 && cache.snapshot().kvTier.diskBlocks == 2 &&
                cache.lookup(prompts[0], {}).kvBoundary == KvCache::pageTokens,
            "busy tier discarded a disk prefix without starting a write");
    tier.complete();
    require(cache.pollTransfers() && test::evictStep(cache).gave() &&
                tier.demotions == 3,
            "disk replacement did not resume once the tier had room");
    tier.complete();
    require(cache.pollTransfers(), "resumed demotion did not finish");
  }
}

// A lookup touches the resident boundary of a chain with a disk suffix, so the reclaim for its
// restore takes an older leaf.
void testLookupKeepsADiskChainsResidentBoundaryWarm() {
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
  require(test::evictStep(cache).gave() && cache.pollTransfers(),
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
    require(test::evictStep(cache).gave() &&
                pool.freePageCount() == 3 && tier.demotions == 1,
            "the reclaim took the chain's resident boundary");
  }
  require(cache.lookup(newer, {}).kvBoundary == 0 &&
              cache.lookup(chain, {}).resumeBoundary() == 64 &&
              cache.snapshot().kvTier.diskBlocks == 1,
          "the reclaim did not take the older leaf alone");
}

// The reclaim that makes room for a restore never takes the resident block the restore extends.
void testRestoreKeepsTheBlockItExtends() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier, false, test::ampleDiskBudget());
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(test::evictStep(fixture.cache).gave() &&
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
  require(test::evictStep(fixture.cache).gave() &&
              fixture.pool.freePageCount() == 1,
          "the restored leaf did not drop its page");
  fixture.cache.beginRequest(3);
  require(admitTokens(fixture.cache, 3, 32).granted() && fixture.pool.freePageCount() == 0,
          "another request did not take the free page");
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(4);
  // Making room demotes a leaf: the restore waits for its copy and holds no
  // page meanwhile.
  const Admitted waiting = admitRestore(fixture.cache, 4, lookup);
  require(waiting.ledger == FreeOutcome::Pending &&
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
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(test::evictStep(fixture.cache).gave() &&
              fixture.cache.pollTransfers(),
          "state was not demoted");
  require(test::evictStep(fixture.cache).gave() && tier.demotions == 1,
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

// A state in RAM sits on resident KV, which the counts of what a reclaim frees take for granted
// (KvCache::pageReach): the cache's invariants reject a RAM copy on a block whose page has gone
// to the SSD.
void testAStateInRamSitsOnAPage() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(test::evictStep(fixture.cache).gave() && fixture.cache.pollTransfers(), "state was not demoted");
  require(test::evictStep(fixture.cache).gave() && tier.demotions == 1, "the leaf's demotion did not start");
  tier.complete();
  require(fixture.cache.pollTransfers() && !fixture.cache.residentChainPages({&fixture.blocks[3], 1}),
          "the leaf did not leave for the SSD");
  fixture.cache.checkInvariants();
  fixture.publish(3);
  rejects([&] { fixture.cache.checkInvariants(); }, "a state in RAM sits on a block with a page",
          "a state in RAM on a block without a page passed the cache's invariants");
}

// A state published in RAM on a block another request was restoring keeps its page when that
// restorer is cancelled.
void testCancelledRestoreKeepsThePageUnderANewState() {
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
  require(test::stateStep(cache).gave() && cache.pollTransfers(),
          "state was not demoted");
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
  require(reclaimStateBytes(cache) == 100 &&
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
  require(test::stateStep(cache).gave() && cache.pollTransfers(),
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

// Restores start root first and stop at the first refusal; each landed transfer starts the next
// block.
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
  require(test::stateStep(cache).gave() && cache.pollTransfers(),
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
      const PageTableState table = cache.pageTable(request);
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
      require(test::evictStep(cache).gave(),
              "the first prompt's leaf was not evicted");
    }
    require(cache.snapshot().pool.pagesPrefix == 6,
            "eviction did not take exactly the first prompt");
  }
};

// A reclaim that releases extents releases an empty extent, then compacts scattered free pages,
// before it evicts anything.
void testCompactionReturnsFreePagesBeforeEvicting() {
  ExtentFixture fixture;
  engine::Cache &cache = fixture.cache;
  auto lookup = cache.lookup(fixture.prompts[1], {});
  cache.beginRequest(4);
  require(lookup.state && admitRestore(cache, 4, lookup).granted(),
          "the second prompt was not restored");
  lookup = {};
  // A command read the list as it was restored.
  const uint64_t revision = cache.cutPageTable(4).revision;
  // Extent 0 keeps the second prompt's first page, extent 1 is full, extent
  // 2 keeps the third prompt's last page, extent 3 is empty.
  fixture.evictFirstPrompt();

  FreeResult step = test::releaseStep(cache);
  require(step.gave() && step.nowBytes == 400 &&
              fixture.storage.copies.empty(),
          "the empty extent was not released as it is");
  step = test::releaseStep(cache);
  require(step.gave() && step.nowBytes == 400 &&
              fixture.storage.copies.size() == 1 &&
              fixture.storage.copies[0].from == 3 && fixture.storage.copies[0].to == 9 &&
              cache.snapshot().pool.pagesPrefix == 6 &&
              cache.snapshot().stateCache.entries == 2 &&
              cache.snapshot().pool.pagesAllocated == 8 &&
              cache.snapshot().pool.extentCompactions == 1,
          "scattered free pages were not returned before an eviction");
  const PageTableState table = cache.pageTable(4);
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
  step = test::releaseStep(cache);
  require(step.gave() && step.stateBuffers && step.nowBytes == 0 &&
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
  // A step of the host's pass, which keeps the runway.
  const FreeResult step = test::takeStep(cache, {.cls = ClaimClass::Host}, {.bytes = 1});
  const KvPoolSnapshot pool = cache.snapshot().pool;
  require(step.gave() && step.nowBytes == 0 &&
              fixture.storage.copies.size() == 1 && pool.pagesAllocated == 12 &&
              pool.reclaimableBytes == 400,
          "the emptied extent did not stay as the runway");
  // A pass with a target keeps one empty extent and returns the others.
  ExtentFixture spare;
  spare.evictFirstPrompt();
  uint64_t released = spare.cache.releaseEmptyExtents(true);
  while (released < 400) {
    const FreeResult next = test::takeStep(spare.cache, {.cls = ClaimClass::Host}, {.bytes = 1});
    require(next.gave(), "the pass made no progress");
    released += next.nowBytes;
  }
  require(released == 400 &&
              spare.storage.copies.size() == 1 &&
              spare.cache.snapshot().pool.pagesAllocated == 12 &&
              spare.cache.snapshot().pool.reclaimableBytes == 400 &&
              spare.cache.snapshot().pool.pagesPrefix == 6,
          "a pass did not keep one runway and return the extent it emptied");
}

// Compaction re-points only the blocks on moved pages, and a block moved twice is found where it
// went last.
void testCompactionFollowsOnlyTheMovedBlocks() {
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
  {
    // Unfinished requests use the other prompts' states, which go only after
    // the first prompt's leaves.
    const StateUse second = cache.useState(fixture.lastBlocks[1]);
    const StateUse third = cache.useState(fixture.lastBlocks[2]);
    // Extent 0 keeps the first prompt's first block and the second prompt's
    // first block; extent 2 keeps the third prompt's last block alone.
    demoteLeaves(cache, tier, 2);
    require(test::releaseStep(cache).nowBytes == 400 &&
                test::releaseStep(cache).gave() &&
                storage.copies.size() == 1 && storage.copies[0].from == 8 &&
                storage.copies[0].to == 1,
            "the third prompt's leaf did not move into extent 0");
    // The first prompt's first block goes to disk and the second prompt's
    // state and two upper blocks go, so extents 0 and 1 hold two pages each.
    demoteLeaves(cache, tier, 1);
    for (uint32_t victim = 0; victim < 3; ++victim)
      require(test::evictStep(cache).gave(),
              "the second prompt was not evicted");
    require(test::releaseStep(cache).gave() &&
                storage.copies.size() == 3 && storage.copies[1].from == 1 &&
                storage.copies[1].to == 4 && storage.copies[2].from == 3 &&
                storage.copies[2].to == 5 && cache.snapshot().pool.extentCompactions == 2,
            "extent 0 was not emptied into extent 1");
  }

  auto lookup = cache.lookup(fixture.prompts[2], {});
  cache.beginRequest(4);
  require(lookup.state && admitRestore(cache, 4, lookup).granted(),
          "the third prompt was not restored");
  const PageTableState table = cache.pageTable(4);
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
  require(test::evictStep(cache).gave() &&
              pool.freePageCount() == freePages + 1 &&
              test::evictStep(cache).gave(),
          "the second prompt's moved block did not free its page");
  freePages = pool.freePageCount();
  require(test::evictStep(cache).gave() &&
              pool.freePageCount() == freePages + 1,
          "the twice-moved leaf did not free the page it moved to last");
}

// A page a restore reads into stays where it is: the extent that holds it is
// passed over although it has the fewest pages, and the next one moves.
void testCompactionLeavesAPageBeingRestored() {
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
  require(test::stateStep(cache).gave() && cache.pollTransfers(),
          "state was not demoted");
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
    require(test::evictStep(cache).gave(),
            "plain leaf was not evicted");
  require(test::releaseStep(cache).nowBytes == 400 &&
              test::releaseStep(cache).nowBytes == 400 &&
              storage.copies.empty(),
          "the empty extents were not released first");

  const FreeResult step = test::releaseStep(cache);
  require(step.gave() && step.nowBytes == 400 && storage.copies.size() == 3 &&
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
  const PageTableState table = cache.pageTable(5);
  require(table.pages[0] == 0 && table.pages[1] == 1 && table.pages[2] == 2 &&
              storage.content[0] == 40 && storage.content[1] == 41 && storage.content[2] == 42,
          "the kept prompt's blocks did not follow their pages");
  lookup = {};
  cache.endRequest(5);
}

// So does a page being written to disk: its extent is passed over, and the
// pages of the chain above it move next to it.
void testCompactionLeavesAPageBeingDemoted() {
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
  require(test::stateStep(cache).gave() && cache.pollTransfers(),
          "state was not demoted");
  // The plain block goes, then the chain's leaf starts to be written.
  require(test::evictStep(cache).gave() &&
              test::evictStep(cache).gave() && tier.demotions == 1 &&
              cache.snapshot().pool.pagesPrefix == 4,
          "the chain's leaf did not start its demotion");
  require(test::releaseStep(cache).nowBytes == 400 &&
              test::releaseStep(cache).nowBytes == 400 &&
              storage.copies.empty(),
          "the empty extents were not released first");

  const FreeResult step = test::releaseStep(cache);
  require(step.gave() && step.nowBytes == 400 && storage.copies.size() == 3 &&
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
  const PageTableState table = cache.pageTable(3);
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
        test::evictStep(fixture.cache).gave() &&
            fixture.cache.stateResident(fixture.blocks[0]),
        "the state in use left before the other state and KV");
  }
  require(fixture.cache.snapshot().stateCache.entries == 1, "the other state survived");
  const FreeResult last =
      test::evictStep(fixture.cache);
  const auto snapshot = fixture.cache.snapshot().stateCache;
  require(last.gave() && snapshot.entries == 0 && snapshot.inUse == 1 &&
              snapshot.inUseEvictions == 1,
          "the last state in use could not be reclaimed");
  use.reset();
  require(fixture.cache.snapshot().stateCache.inUse == 0, "the mark outlived its handle");
}

// The KV a state in use restores through goes with that state, after ordinary states and KV.
void testKvAStateInUseNeedsGoesLast() {
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
  require(test::stateStep(cache, ReclaimClass::InUse).gave() && cache.pollTransfers() &&
              !cache.stateResident(used[1]),
          "the state in use was not written");
  const std::vector<uint64_t> ordinary = cacheChain(cache, 2, 2).second;
  cache.publishCompositeState(ordinary[1], std::make_shared<TestState>(100), false);
  require(test::evictStep(cache).gave() &&
              !cache.stateResident(ordinary[1]) && tier.demotions == 0,
          "the KV a state in use needs went before an ordinary state");
}

// State-use marks count per block: a mark taken before the publication applies to it, shared
// marks each hold one, and a state published again at a used block is in use again.
void testStateUsesAreCountedPerBlock() {
  CacheFixture fixture(nullptr, true);
  const auto inUse = [&] { return fixture.cache.snapshot().stateCache.inUse; };
  // Whether a state claim of the given reach takes the cache's one state.
  const auto takes = [&](ReclaimClass reach) {
    const ClaimSpec claim{.cls = ClaimClass::InService, .reach = reach};
    return test::takeFirst(fixture.cache, claim, {.stateBuffers = 1}).outcome == FreeOutcome::Met;
  };
  StateUse early = fixture.cache.useState(fixture.blocks[0]);
  fixture.publish(0);
  StateUse shared = fixture.cache.useState(fixture.blocks[0]);
  early.reset();
  require(inUse() == 1 && !takes(ReclaimClass::Ordinary) && fixture.cache.stateResident(fixture.blocks[0]),
          "a mark taken before the publication, or a shared one, was lost");
  require(takes(ReclaimClass::InUse) && !fixture.cache.stateResident(fixture.blocks[0]),
          "a claim that reaches what is in use did not take the state in use");
  fixture.publish(0);
  require(inUse() == 1 && !takes(ReclaimClass::Ordinary), "a state published again at a used block was not in use");
  shared.reset();
  require(inUse() == 0 && takes(ReclaimClass::Ordinary), "a released block stayed in use");
}

// KV above a state is in use only while that state is; afterwards its chain goes by recency again.
void testKvInUseFollowsItsState() {
  test::TestKvStorage storage{7, 100, 1};
  storage.budgetPages = 6;
  KvPool pool{storage, 6};
  engine::Cache cache(pool, nullptr, nullptr);
  const std::vector<uint64_t> used = cacheChain(cache, 1, 2).second;
  cache.publishCompositeState(used[1], std::make_shared<TestState>(100), false);
  StateUse use = cache.useState(used[1]);
  require(test::stateStep(cache, ReclaimClass::InUse).gave() &&
              cache.snapshot().stateCache.entries == 0 &&
              cache.snapshot().stateCache.inUseEvictions == 1,
          "the state in use did not go once nothing else was left");
  use.reset();
  const std::vector<uint64_t> ordinary = cacheChain(cache, 2, 2).second;
  cache.publishCompositeState(ordinary[1], std::make_shared<TestState>(100), false);
  for (uint32_t leaf = 0; leaf < 2; ++leaf)
    require(test::evictStep(cache).gave(),
            "the cache could not be reclaimed");
  require(cache.stateResident(ordinary[1]) && cache.snapshot().pool.pagesPrefix == 2,
          "the KV of a state that left stayed in use");
}

// A finished request refreshes only its replay point; older points on its chain keep their age.
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
  // Each claim for a state's buffers takes the oldest state.
  const auto evictState = [&] {
    require(test::stateStep(fixture.cache).gave(), "the cache could not be reclaimed");
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

// The resume point is the newest ordinary publication, never a state in use: growth, which keeps
// it, takes the older ordinary state and its KV and stops there.
void testKeepResumePointKeepsTheNewestOrdinaryPublication() {
  const ClaimSpec growth{.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate};
  const auto shrink = [&](engine::Cache &cache) {
    while (test::freeFromCache(cache, growth, {.pages = 1}).gave()) {
    }
    while (test::freeFromCache(cache, growth, {.stateBuffers = 1}).gave()) {
    }
  };
  CacheFixture fixture;
  fixture.publish(0);
  StateUse running = fixture.cache.useState(fixture.blocks[1]);
  fixture.publish(1);
  fixture.publish(2);
  const auto held = [&](const CacheFixture &each, uint32_t block) {
    return each.cache.stateResident(each.blocks[block]);
  };
  shrink(fixture.cache);
  require(!held(fixture, 0) && held(fixture, 1) && held(fixture, 2),
          "growth took the resume point or the state in use, or left the older ordinary state");
  // A state in use newer than every ordinary one is not the resume point.
  CacheFixture newer;
  newer.publish(0);
  StateUse newest = newer.cache.useState(newer.blocks[2]);
  newer.publish(2);
  shrink(newer.cache);
  require(held(newer, 0) && held(newer, 2), "the resume point was the state in use, not the newest ordinary one");
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

// A can-wait page claim waits only for landings that cover it; a state write returns no page, so it
// takes what is in use at once.
void testInUseGoesWhenWhatLandsReturnsNoPage() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  require(test::stateStep(fixture.cache).gave(),
          "the first state was not written");
  // Only the transfer holds this state back: nothing needs to write it.
  StateUse use = fixture.cache.useState(fixture.blocks[1]);
  fixture.publish(1);
  while (test::evictStep(fixture.cache, ReclaimClass::Ordinary).gave()) {
  }
  require(fixture.cache.stateResident(fixture.blocks[1]) && fixture.pool.freePageCount() == 2,
          "the ordinary class did not stop short of the state in use");
  // Two pages are free; the third has to come from the leaf in use.
  fixture.cache.beginRequest(2);
  require(admitTokens(fixture.cache, 2, 96).granted() && !fixture.cache.stateResident(fixture.blocks[1]) &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 1,
          "a request short of pages waited for a transfer that returns no page");
  control->ready = true;
  require(fixture.cache.pollTransfers(), "the state write did not land");
  fixture.cache.endRequest(2);
}

// Quota replacement follows the writer's class: an in-use write gives up in-use copies last; an
// ordinary write never does.
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
  require(test::stateStep(fixture.cache, ReclaimClass::InUse).gave() && fixture.cache.pollTransfers(),
          "the oldest state in use was not written");
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers() &&
              control->slots == 2,
          "the ordinary state was not written");
  // The quota is full: a write in use gives up the ordinary copy.
  StateUse older = fixture.cache.useState(fixture.blocks[2]);
  fixture.cache.publishCompositeState(fixture.blocks[2], std::make_shared<TieredState>(control), false);
  require(test::stateStep(fixture.cache, ReclaimClass::InUse).gave() && fixture.cache.pollTransfers() &&
              cached(65) == 32 && cached(97) == 96 && inUseEvictions() == 0,
          "a write in use replaced a copy in use before an ordinary one");
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(test::stateStep(fixture.cache).gave() && cached(129) == 96 &&
              inUseEvictions() == 0,
          "an ordinary write replaced the only copy of a state in use");
  StateUse newer = fixture.cache.useState(fixture.blocks[3]);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(
      test::stateStep(fixture.cache, ReclaimClass::InUse).gave() &&
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

// A quota holding only in-use copies never keeps in RAM the parent a disk-only state in use
// restores through.
void testKvAStateInUseNeedsNeverPinsMemory() {
  test::TestKvTier tier;
  tier.capacity = 1;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  StateUse use = fixture.cache.useState(fixture.blocks[3]);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(test::evictStep(fixture.cache).gave() &&
              fixture.cache.pollTransfers(),
          "the state in use was not written");
  require(test::evictStep(fixture.cache).gave() && tier.demotions == 1,
          "the leaf of the state in use was not written");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1,
          "the leaf did not land");
  // Block 2 is a leaf the state in use needs; the quota holds only that
  // state's KV copy.
  require(test::evictStep(fixture.cache).gave() &&
              fixture.cache.snapshot().pool.pagesPrefix == 2 &&
              fixture.cache.snapshot().stateCache.inUseEvictions == 1,
          "the parent a state in use needs stayed in RAM");
}

// An ordinary leaf's write never displaces the copies a state in use needs, even older ones.
void testOrdinaryDemotionKeepsCopiesInUse() {
  test::TestKvStorage storage{7, 100, 1};
  storage.budgetPages = 6;
  KvPool pool{storage, 6};
  test::TestKvTier tier;
  tier.capacity = 3;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const auto settle = [&](const char *message) {
    require(test::evictStep(cache).gave(), message);
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
    require(held.state && !test::evictStep(cache).gave() &&
                cache.snapshot().stateCache.inUseEvictions == 0,
            "an ordinary leaf's write displaced a copy in use");
  }
  require(test::evictStep(cache).gave() &&
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
  require(test::stateStep(fixture.cache).gave() && fixture.cache.pollTransfers(),
          "state was not demoted");
  StateUse use = fixture.cache.useState(fixture.blocks[1]);
  fixture.publish(1);
  auto lookup = fixture.lookup(129);
  PromotionTicket ticket;
  ticket.available = false;
  fixture.cache.promoteState(lookup, [&] { return ticket.snapshot(); });
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
  require(test::evictStep(cache).gave() &&
              cache.snapshot().pool.pagesPrefix == 4 && cache.stateResident(older[1]) &&
              cache.stateResident(newer[1]),
          "an older state went before a newer conversation's dead KV");
}

// A demotion the busy tier cannot start keeps only the leaves that need one:
// a leaf with a disk copy still gives its page, before a newer state,
// without asking the busy tier.
void testPendingDemotionLeavesOtherKvOpen() {
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
    require(test::stateStep(cache).gave() && cache.pollTransfers(),
            "a state was not written");
    prompts.push_back(std::move(prompt));
  }
  // The oldest leaf is written while a lookup holds it, so it keeps its page
  // beside its new disk copy; the next leaf's demotion stays in flight.
  require(test::evictStep(cache).gave(), "a demotion did not start");
  {
    const CacheLookup held = cache.lookup(prompts[0], {});
    tier.complete();
    require(cache.pollTransfers(), "the demotion did not land");
    require(test::evictStep(cache).gave() && !tier.canDemote(WriteClass::Reclaim),
            "the second demotion did not fill the tier");
  }
  const uint64_t state = cacheChain(cache, 4, 1).second[0];
  cache.publishCompositeState(state, std::make_shared<TestState>(100), false);
  const CacheSnapshot before = cache.snapshot();
  require(test::evictStep(cache).gave(),
          "nothing was reclaimed while a demotion had to wait");
  const CacheSnapshot after = cache.snapshot();
  require(after.pool.pagesPrefix == before.pool.pagesPrefix - 1 &&
              after.kvTier.diskBlocks == before.kvTier.diskBlocks &&
              cache.lookup(prompts[0], {}).kvBoundary == KvCache::pageTokens &&
              cache.stateResident(state) && after.kvTier.demotionsRefused == before.kvTier.demotionsRefused,
          "a demotion that had to wait closed the KV whose page frees at once");
}

// A failed copy of a state in use is not counted as an eviction of a state in use.
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

// While the host refuses growth only what frees pages goes: KV leaves oldest first, a state only
// with the leaf under it.
void testPageReuseTakesKvBeforeStates() {
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
    require(test::stateStep(cache).gave() && cache.pollTransfers(),
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
  // Steps of page scans, a refusal the host refused (pages in place).
  const auto scan = [&](const ClaimSpec &claim) {
    return test::takeStep(cache, claim, {.hostRefused = true, .pages = 1}).gave();
  };
  require(scan(test::kInServiceGrowth) &&
              cache.snapshot().pool.pagesPrefix == 3 && cache.stateResident(ordinary) &&
              cache.stateResident(checkpoint) &&
              cache.snapshot().stateCache.checkpointEvictions == 0,
          "page reuse took a state before a KV leaf");
  // The first chain's interior block is now its oldest leaf, under the
  // ordinary state, the resume point.
  require(scan({.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate}) &&
              cache.stateResident(ordinary) && cache.stateResident(checkpoint) &&
              cache.snapshot().pool.pagesPrefix == 2,
          "page reuse that keeps the resume point did not take the next leaf");
  require(scan(test::kInServiceGrowth) &&
              !cache.stateResident(ordinary) && cache.snapshot().pool.pagesPrefix == 2,
          "page reuse did not take the state on the leaf that goes next");
  require(scan(test::kInServiceGrowth) &&
              cache.snapshot().pool.pagesPrefix == 1 && cache.stateResident(checkpoint),
          "page reuse did not take the leaf its state left");
}

// A page need's eviction leaves the extent it empties allocated for the next request.
void testReplacementKeepsTheExtentItEmpties() {
  test::TestKvStorage storage(8, 4096, 4);
  storage.budgetPages = 4;
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  static_cast<void>(cacheChain(cache, 1, 1));

  const auto reclaimed = test::evictStep(cache);
  require(reclaimed.gave() && reclaimed.nowBytes == 0 &&
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

// A cache whose requests may spill, under --kv-spill always: their pages
// move to the index form as their copies land (Cache::spillWritten). Its
// pool's extents hold four pages, 100 bytes a Full page and 10 an Index
// page, one extent of each form allocated from the start (the Index one the
// runway spilling draws on, as the cache's first spill pass allocates it),
// and its tier's slots of 100 bytes draw on a quota of `slots` of them. The
// dense history and the tail window are a page each unless a test says
// otherwise.
struct Spilling {
  explicit Spilling(CacheSpillPolicy policy = {1, 1, true}, uint32_t slots = 64)
      : budget(std::make_shared<model::DiskBudget>(uint64_t{slots} * 100)),
        cache(pool, &tier, budget, policy) {
    tier.budget = budget;
    tier.transferLimit = 1024;
    require(cache.spillWritten(SpilledExtents::KeepRunway).granted() &&
                pool.freePageCount(kv::PageForm::Index) == 4,
            "the cache did not allocate the Index runway");
  }

  // `pages` whole pages of tokens from `first` on, and one token past them.
  static std::vector<uint32_t> prompt(uint32_t first, uint32_t pages) {
    std::vector<uint32_t> tokens(pages * KvCache::pageTokens + 1);
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
  }

  // A request in the mode, which commits quota for its pages as a Spill
  // admission does, holds pages for `pages` pages of the prompt and one more
  // token, and publishes them.
  void start(uint64_t id, std::span<const uint32_t> tokens, uint32_t pages,
             KvMode mode = KvMode::Spill) {
    cache.beginRequest(id, mode);
    if (mode == KvMode::Spill)
      require(cache.commitSpill(id, pages), "a Spill request's commitment was refused");
    grow(id, tokens, pages);
  }
  void grow(uint64_t id, std::span<const uint32_t> tokens, uint32_t pages) {
    require(admitTokens(cache, id, pages * KvCache::pageTokens + 1).granted(),
            "a request's pages were denied");
    cache.publishCommittedBlocks(id, tokens, pages * KvCache::pageTokens, {});
  }
  // The block at the request's page.
  [[nodiscard]] uint64_t block(uint64_t id, uint32_t page) {
    return cache.blockAt(id, (page + 1) * KvCache::pageTokens);
  }
  // A state at the block, which reclaim writes to its disk copy.
  void publishState(uint64_t block) {
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    cache.publishCompositeState(block, std::make_shared<TieredState>(control), false);
  }
  // Everything cached leaves RAM: states for their disk copies, written as
  // they go, and KV for its copies.
  void toDisk() {
    while (cache.snapshot().pool.pagesPrefix) {
      const bool took = test::evictStep(cache).gave();
      const bool landed = cache.pollTransfers();
      require(took || landed, "the cache did not leave RAM");
    }
  }
  // Which of the request's pages read their copies, by page: true for an
  // Index page.
  [[nodiscard]] std::vector<bool> spilled(uint64_t id) {
    std::vector<bool> result;
    for (uint64_t block : test::indexBlocks(cache, id))
      result.push_back(block != 0);
    return result;
  }

  test::TestKvStorage storage{1024, 100, 4, 10};
  KvPool pool{storage, 4};
  test::TestKvTier tier;
  std::shared_ptr<model::DiskBudget> budget;
  engine::Cache cache;
};

// Spill pages past the dense history and window move to the Index form once their copies land; the
// next cut names the copies.
void testSpillMovesLandedPagesToTheIndexForm() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  require(spill.budget->committedBytes() == 600 && spill.cache.spillRoom() == 5800,
          "the commitment did not lock the request's quota");
  require(spill.cache.writeThrough() == 6 && spill.tier.writes.size() == 6 &&
              std::ranges::all_of(spill.tier.writes,
                                  [](const auto &write) { return write.write == WriteClass::Spill; }) &&
              std::ranges::all_of(spill.tier.draws,
                                  [](auto draw) { return draw == model::DiskBudget::Draw::Commitment; }) &&
              spill.budget->committedBytes() == 0 && spill.budget->usedBytes() == 600,
          "the published pages were not written through on the request's commitment");
  require(spill.cache.spillWritten(SpilledExtents::KeepRunway).granted() &&
              spill.spilled(1) == std::vector<bool>(7, false),
          "a page spilled before its copy landed");
  spill.tier.complete();
  require(spill.cache.pollTransfers(), "the writes did not land");
  const uint32_t free = spill.pool.freePageCount();
  const uint32_t allocations = spill.storage.allocationAttempts;
  require(spill.cache.spillWritten(SpilledExtents::KeepRunway).granted() &&
              spill.spilled(1) == std::vector<bool>{false, true, true, true, true, false, false} &&
              spill.pool.freePageCount() == free + 4 &&
              spill.pool.freePageCount(kv::PageForm::Index) == 4 &&
              spill.storage.allocationAttempts == allocations + 1,
          "the landed pages past the dense history and the window did not spill into the runway, or it "
          "was not refilled");
  const KvSpillSnapshot spilled = spill.cache.spillSnapshot();
  require(spilled.spilledPages == 4 && spilled.spilledBytes == 400 && spilled.spillMoves == 4 &&
              spilled.writeThroughPages == 6 && spilled.heldBytes == 600 && spilled.committedBytes == 0 &&
              spill.cache.holdsIndexPages(1),
          "spilling did not count what it holds and did");
  std::shared_ptr<const model::KvPageSources> sources = spill.cache.cutPageTable(1).sources;
  require(sources && sources->indexPages() == 4 && (*sources)[1].block == spill.block(1, 1) &&
              (*sources)[1].copy && !(*sources)[5].indexed() && !(*sources)[0].indexed(),
          "the cut did not name the spilled blocks and their copies");
  spill.cache.checkInvariants();
  // The command that read them completed.
  sources.reset();
  spill.cache.endRequest(1);
  require(spill.cache.spillSnapshot().heldBytes == 0 && spill.cache.spillSnapshot().spilledPages == 4,
          "the finished request's spilled chain did not stay cached");
  spill.cache.checkInvariants();
}

// The Full extents a spill empties are released (all, or all but the runway); a spill that moves
// nothing releases nothing.
void testSpilledFullExtentsLeaveMemory() {
  for (const SpilledExtents extents : {SpilledExtents::KeepRunway, SpilledExtents::Released}) {
    Spilling spill;
    const auto emptyFullExtents = [&] {
      const KvPoolSnapshot pool = spill.pool.snapshot();
      return static_cast<uint32_t>((pool.reclaimableBytes - pool.index.reclaimableBytes) /
                                   (spill.pool.extentPages() * spill.pool.bytesPerPage()));
    };
    const auto fullExtents = [&] {
      const KvPoolSnapshot pool = spill.pool.snapshot();
      return (pool.pagesAllocated - pool.index.pagesAllocated) / spill.pool.extentPages();
    };
    // Fourteen pages and the open one fill four Full extents, the runway's
    // first. Pages 1-12 spill: the dense history stays in the first and the
    // tail window in the last, the two between empty; released, the dense
    // history's page moves beside the window's.
    const std::vector<uint32_t> prompt = Spilling::prompt(1000, 14);
    spill.start(1, prompt, 14);
    static_cast<void>(spill.cache.writeThrough());
    spill.tier.complete();
    require(spill.cache.pollTransfers() && !emptyFullExtents(), "the writes did not land, or an extent was empty");
    const bool runway = extents == SpilledExtents::KeepRunway;
    require(spill.cache.spillWritten(extents).granted() && test::indexBlocks(spill.cache, 1)[12] &&
                emptyFullExtents() == runway && fullExtents() == (runway ? 3u : 1u),
            "the Full extents the moves emptied did not go as the caller said");
    // A Resident request's pages, which an idle reclaim drops once it ended,
    // empty extents that a spill moving nothing leaves.
    spill.start(2, Spilling::prompt(5000, 20), 20, KvMode::Resident);
    spill.cache.endRequest(2);
    while (test::evictStep(spill.cache, ReclaimClass::Ordinary).gave()) {
    }
    const uint32_t before = emptyFullExtents();
    require(before > 1 && spill.cache.spillWritten(extents).granted() &&
                emptyFullExtents() == before,
            "a spill that moved nothing released an extent");
    spill.cache.checkInvariants();
    spill.cache.endRequest(1);
  }
}

// At the memory wall spills draw on the Index runway alone; once growth returns, held pages spill
// and the runway refills.
void testSpillAtTheMemoryWall() {
  for (const auto failure :
       {metal::AllocationFailure::EngineBudget, metal::AllocationFailure::HostPressure}) {
    Spilling spill;
    spill.storage.growthBlocked = true;
    spill.storage.allocationFailure = failure;
    const std::vector<uint32_t> prompt = Spilling::prompt(1000, 12);
    const uint32_t allocated = spill.storage.allocatedPages();
    // The Full runway holds three pages and the open one. Each page that
    // spills frees the one the next grows into.
    spill.start(1, prompt, 3);
    require(test::spillRound(spill.cache, spill.tier) == 3, "the pages were not written through");
    for (uint32_t pages = 4; pages <= 6; ++pages) {
      spill.grow(1, prompt, pages);
      require(test::spillRound(spill.cache, spill.tier) == 1, "a page was not written through");
    }
    require(spill.spilled(1) == std::vector<bool>{false, true, true, true, true, false, false} &&
                spill.storage.allocatedPages() == allocated,
            "spilling at the wall allocated, or did not free the pages the request grew into");
    spill.grow(1, prompt, 7);
    static_cast<void>(spill.cache.writeThrough());
    spill.tier.complete();
    static_cast<void>(spill.cache.pollTransfers());
    const TokenAdmission refused = spill.cache.spillWritten(SpilledExtents::KeepRunway);
    require(refused.failure == TokenAdmissionFailure::Denied && refused.form == kv::PageForm::Index &&
                refused.additionalPages == 1 && refused.allocationFailure == failure &&
                !spill.spilled(1)[5],
            "a spill past the runway did not ask for an Index extent, or moved its page anyway");
    spill.storage.growthBlocked = false;
    require(spill.cache.spillWritten(SpilledExtents::KeepRunway).granted() && spill.spilled(1)[5] &&
                spill.pool.freePageCount(kv::PageForm::Index) >= 4,
            "once growth was admitted the page did not spill, or the runway was not refilled");
    spill.storage.growthBlocked = true;
    const uint32_t refilled = spill.storage.allocatedPages();
    for (uint32_t pages = 8; pages <= 11; ++pages) {
      spill.grow(1, prompt, pages);
      static_cast<void>(test::spillRound(spill.cache, spill.tier));
    }
    std::vector<bool> spilled(12, true);
    spilled[0] = spilled[10] = spilled[11] = false;
    require(spill.spilled(1) == spilled && spill.storage.allocatedPages() == refilled,
            "at the wall once more the request did not spill through the refilled runway");
    spill.cache.checkInvariants();
    spill.cache.endRequest(1);
  }
}

// A Full shortfall reaches an idle spilled chain's Full pages; its Index leaves go first, oldest
// first.
void testFullReclaimReachesAnIdleSpilledChain() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  spill.cache.endRequest(1);
  require(spill.cache.spillSnapshot().spilledPages == 4 && spill.pool.freePageCount() == 6,
          "the idle spilled chain was not set up");
  // Nothing grows: the eight Full pages are all there is.
  spill.storage.growthBlocked = true;
  spill.cache.beginRequest(2);
  require(admitTokens(spill.cache, 2, 8 * KvCache::pageTokens).granted(),
          "the Full shortfall did not reach the idle chain's dense history");
  const CacheSnapshot after = spill.cache.snapshot();
  require(after.kvTier.diskBlocks == 6 && spill.cache.spillSnapshot().spilledPages == 0 &&
              spill.pool.freePageCount(kv::PageForm::Index) == 8 && spill.storage.releasedExtents == 0,
          "the idle chain did not leave RAM for its copies, or a page need released an extent");
  const FreeResult released = test::freeFromCache(
      spill.cache, {.cls = ClaimClass::InService}, {.bytes = 4 * spill.storage.bytesPerPage(kv::PageForm::Index)});
  require(released.outcome == FreeOutcome::Met && spill.storage.releasedExtents == 1 &&
              spill.pool.freePageCount(kv::PageForm::Index) == 4,
          "a need of bytes did not take the emptied Index extent first, or its runway went");
  spill.cache.checkInvariants();
  spill.cache.endRequest(2);
}

// A Full shortfall first releases empty Index extents past the runway.
void testFullShortfallReleasesEmptyIndexExtents() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 10);
  spill.start(1, prompt, 10);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  require(spill.cache.spillSnapshot().spilledPages == 8, "two Index extents were not filled");
  spill.cache.endRequest(1);
  // Steps for Index pages, which evict the spilled leaves.
  while (spill.cache.spillSnapshot().spilledPages)
    require(test::takeStep(spill.cache, test::kInServiceGrowth, {.pages = 1, .form = kv::PageForm::Index}).gave(),
            "an idle spilled leaf did not go");
  const uint32_t releasedBefore = spill.storage.releasedExtents;
  const FreeResult released = test::takeFirst(spill.cache, test::kInServiceGrowth, {.pages = 1});
  require(released.outcome == FreeOutcome::Met && released.nowBytes == 80 &&
              spill.storage.releasedExtents - releasedBefore == 2 &&
              spill.pool.freePageCount(kv::PageForm::Index) == 4,
          "a Full shortfall did not release the empty Index extents past the runway first");
}

// Under critical pressure the host's pass takes the empty Full extent the next request would start from, never the
// Index runway: a spill's moves land there without an allocation.
void testCriticalKeepsTheIndexRunway() {
  Spilling spill;
  engine::MemoryLedger ledger;
  ledger.addSource(spill.cache);
  // As the host's pass asks (MemoryGovernor::hostPass): what it finds of
  // all the cache holds, a Full and an Index extent, both empty.
  const ClaimSpec host{.cls = ClaimClass::Host};
  const uint64_t found = ledger.available(host, {.bytes = spill.storage.allocatedBytes()}, 0, MemoryPressure::Critical);
  const engine::ClaimId claim = ledger.open(host);
  const FreeResult freed = ledger.free(claim, {.bytes = found}, 0, MemoryPressure::Critical);
  ledger.close(claim);
  require(freed.nowBytes == 4 * spill.storage.bytesPerPage(kv::PageForm::Full) &&
              !spill.pool.allocatedExtents(kv::PageForm::Full) && spill.pool.freePageCount(kv::PageForm::Index) == 4,
          "under critical pressure the host's pass took the Index runway, or kept the empty Full extent");
}

// An Index shortfall empties only Full extents; moving Index pages frees nothing it can grow by.
void testIndexShortfallCompactsOnlyFullExtents() {
  Spilling spill;
  KvPageAcquisition index = spill.pool.acquirePages(5, kv::PageForm::Index);
  require(index.granted() && spill.pool.freePageCount(kv::PageForm::Index) == 3,
          "the Index pages did not take a second extent");
  for (uint32_t page : std::span(index.pages).first(3))
    spill.pool.releasePage(page, false);
  spill.storage.growthBlocked = true;
  const FreeResult reclaimed =
      test::takeFirst(spill.cache, test::kInServiceGrowth, {.pages = 8, .form = kv::PageForm::Index});
  require(reclaimed.outcome != FreeOutcome::Met && spill.storage.copies.empty() &&
              spill.pool.snapshot().extentCompactions == 0,
          "an Index shortfall compacted an Index extent");
  for (uint32_t page : std::span(index.pages).last(2))
    spill.pool.releasePage(page, false);
}

// A need of bytes and pages takes its pages in place: each take ends once it has a page more there
// (Cache::takeReclaim), which the next keeps with the other free ones, so neither its takes nor its count empty an
// extent of its form. A prefix's eight pages in two extents, none free, give a KV need of an extent's bytes and twelve
// pages eight pages in place and no bytes: short, it takes nothing and the prefix stays. Counted as the two extents
// those pages fill, the bytes were covered, the takes freed the eight pages one at a time, no extent, and the run was
// overcounted.
void testANeedInPlaceEmptiesNoExtentOfItsForm() {
  test::TestKvStorage storage{8, 100, 4};
  KvPool pool{storage, 0};
  engine::Cache cache(pool, nullptr, nullptr);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 8 * KvCache::pageTokens).granted(), "the prefix's pages were denied");
  cache.publishCommittedBlocks(1, Spilling::prompt(1000, 8), 8 * KvCache::pageTokens, {});
  cache.endRequest(1);
  require(pool.allocatedExtents(kv::PageForm::Full) == 2 && !pool.freePageCount(kv::PageForm::Full),
          "the prefix did not fill two extents");
  MemoryLedger ledger;
  ledger.addSource(cache);
  const ClaimId claim = ledger.open(test::kInServiceGrowth);
  const FreeResult freed = ledger.free(claim, {.bytes = 400, .pages = 12, .kv = true}, 0, MemoryPressure::Normal);
  ledger.close(claim);
  MemoryGovernorSnapshot snapshot;
  ledger.fill(snapshot, 0);
  require(freed.outcome == FreeOutcome::Short && !freed.pages && !pool.freePageCount(kv::PageForm::Full) &&
              !snapshot.countChecks.overcountedRuns,
          "a need of an extent's bytes and twelve pages, of eight pages in place, freed " +
              std::to_string(freed.pages) + " pages, left " +
              std::to_string(pool.freePageCount(kv::PageForm::Full)) + " free, and was overcounted " +
              std::to_string(snapshot.countChecks.overcountedRuns) + " times");
}

// A page need of a form the cache holds none of gets no eviction offered.
void testAPageNeedOfAFormTheCacheHoldsNoneOfCountsNothing() {
  Spilling spill;
  for (uint64_t id = 1; id <= 3; ++id) {
    spill.start(id, Spilling::prompt(1000 * static_cast<uint32_t>(id), 4), 4, KvMode::Resident);
    spill.cache.endRequest(id);
  }
  require(spill.pool.reclaimEmptyExtents(false, 1, kv::PageForm::Index) && !spill.pool.allocatedExtents(kv::PageForm::Index),
          "the Index runway stayed");
  const ClaimSpec refill{.cls = ClaimClass::Optional, .timing = NeedTiming::Immediate};
  const VictimQuery query =
      test::query(refill, {.bytes = 40, .hostRefused = true, .pages = 4, .form = kv::PageForm::Index});
  uint32_t evictions = 0;
  for (std::optional<engine::Victim> victim = spill.cache.peek(query, nullptr); victim;
       victim = spill.cache.peek(query, &*victim))
    evictions += !victim->pages && !victim->bytes;
  require(!evictions, "a page need of a form the cache holds no page of was offered " + std::to_string(evictions) +
                          " units that free none");
  // A Spill request's pages spill into Index extents, and stay cached.
  spill.start(4, Spilling::prompt(5000, 10), 10);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  spill.cache.endRequest(4);
  bool freesIndexPages = false;
  for (std::optional<engine::Victim> victim = spill.cache.peek(query, nullptr); victim && !freesIndexPages;
       victim = spill.cache.peek(query, &*victim))
    freesIndexPages = victim->pages != 0 && victim->form == kv::PageForm::Index;
  require(freesIndexPages, "cached Index pages were not offered to a page need of their form");
}

// Idle pages a state in use restores through count against the reusable pages of their own form
// only.
void testReusablePagesCountTheirForm() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 12);
  spill.start(1, prompt, 12);
  while (test::spillRound(spill.cache, spill.tier)) {
  }
  // Memory reclaims until none takes anything: the empty extents go, those
  // compaction empties too.
  while (test::releaseStep(spill.cache).gave()) {
  }
  const uint64_t tip = spill.block(1, 11);
  spill.cache.publishCompositeState(tip, std::make_shared<TestState>(100), false);
  StateUse use = spill.cache.useState(tip);
  spill.cache.endRequest(1);
  const uint32_t full = spill.pool.reusablePages(kv::PageForm::Full);
  const uint32_t index = spill.pool.reusablePages(kv::PageForm::Index);
  require(spill.cache.spillSnapshot().spilledPages == 10 && full < 10 &&
              spill.cache.reusablePages(ReclaimClass::Ordinary) == full - 2 &&
              spill.cache.reusablePages(ReclaimClass::Ordinary, kv::PageForm::Index) == index - 10 &&
              spill.cache.reusablePages(ReclaimClass::InUse, kv::PageForm::Index) == index,
          "the pages a state in use restores through were counted against another form");
}

// Pages written through that will spill count as pending for a Full shortfall, so a covered
// admission waits instead of reclaiming.
void testPendingPagesCountWritesThatSpill() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  require(spill.cache.writeThrough() == 6 && spill.cache.pendingPages(kv::PageForm::Full) == 4 &&
              spill.cache.transfersInFlight(),
          "the writes of pages that will spill were not pending");
  // A Resident request on the first three pages keeps two of them Full.
  spill.start(2, prompt, 3, KvMode::Resident);
  require(spill.cache.pendingPages(kv::PageForm::Full) == 2,
          "writes of pages a Resident request holds were pending");
  KvPageAcquisition index = spill.pool.acquirePages(3, kv::PageForm::Index);
  require(index.granted() && spill.cache.pendingPages(kv::PageForm::Full) == 1,
          "more writes were pending than Index pages could take");
  // The pool cannot grow: a page more than the free ones waits for the
  // write that frees one.
  spill.storage.growthBlocked = true;
  spill.cache.beginRequest(3);
  const uint32_t free = spill.pool.freePageCount();
  const TokenAdmission refused = spill.cache.ensureTokens(3, (free + 1) * KvCache::pageTokens);
  require(!refused.granted() && askLedger(spill.cache, refused).outcome == FreeOutcome::Pending,
          "an admission the spilling writes cover did not wait for them");
  for (uint32_t page : index.pages)
    spill.pool.releasePage(page, false);
  spill.cache.endRequest(3);
  spill.cache.endRequest(2);
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.endRequest(1);
}

// Writes of pages that will not spill free nothing, so a reclaim does not wait for them.
void testWritesThatFreeNothingHoldNothingBack() {
  Spilling spill;
  spill.start(1, Spilling::prompt(1000, 2), 2);
  require(spill.cache.writeThrough() == 2 && spill.cache.pendingPages(kv::PageForm::Full) == 0 &&
              !spill.cache.transfersInFlight(),
          "writes that free nothing counted as transfers a reclaim waits for");
  spill.start(2, Spilling::prompt(5000, 1), 1, KvMode::Resident);
  const uint64_t block = spill.block(2, 0);
  StateUse use = spill.cache.useState(block);
  spill.cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
  spill.cache.endRequest(2);
  while (spill.cache.stateResident(block) &&
         test::evictStep(spill.cache).gave()) {
  }
  require(!spill.cache.stateResident(block), "a state in use waited for writes that free nothing");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.endRequest(1);
}

// A refused allocation waits for spilling writes only for what they and free pages cover; for the
// rest it reclaims.
void testWritesShortOfARefusalHoldNothingBack() {
  Spilling spill;
  spill.start(1, Spilling::prompt(1000, 6), 6);
  spill.start(2, Spilling::prompt(5000, 2), 2, KvMode::Resident);
  const uint64_t block = spill.block(2, 1);
  StateUse use = spill.cache.useState(block);
  spill.cache.publishCompositeState(block, std::make_shared<TestState>(100), false);
  spill.cache.endRequest(2);
  require(spill.cache.writeThrough() == 6 && spill.cache.pendingPages(kv::PageForm::Full) == 4,
          "the writes of pages that will spill were not pending");
  // The pool cannot grow: a page more than the free ones and those the
  // writes free.
  spill.storage.growthBlocked = true;
  spill.cache.beginRequest(3);
  const uint64_t tokens =
      uint64_t{spill.pool.freePageCount() + spill.cache.pendingPages(kv::PageForm::Full) + 1} *
      KvCache::pageTokens;
  // The ledger takes what is in use for the page the writes do not cover,
  // and waits for them for the rest.
  require(admitTokens(spill.cache, 3, tokens).ledger == FreeOutcome::Pending && !spill.cache.stateResident(block),
          "the claim did not take what is in use for the page the writes through do not cover, or did not wait "
          "for them for the rest");
  spill.cache.endRequest(3);
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.endRequest(1);
}

// A memory reclaim still empties and releases an idle chain's extent while a Spill request's writes
// are in flight.
void testMemoryReclaimEmptiesAnExtentBesideWritesThatSpill() {
  Spilling spill;
  // Two Index pages hold other blocks' index: the Index extent is not empty,
  // and of the four writes that will spill, the two the free Index pages
  // can take are pending.
  KvPageAcquisition index = spill.pool.acquirePages(2, kv::PageForm::Index);
  spill.start(1, Spilling::prompt(1000, 6), 6);
  // An idle chain on the second extent's last page and the third's first
  // three, whose fourth is free.
  spill.start(2, Spilling::prompt(5000, 4), 4, KvMode::Resident);
  spill.cache.endRequest(2);
  require(index.granted() && spill.cache.writeThrough() == 6 &&
              spill.cache.pendingPages(kv::PageForm::Full) == 2 && spill.pool.freePageCount() == 1 &&
              spill.cache.reusablePages(ReclaimClass::Ordinary) == 5,
          "fixture geometry changed");
  // A device need of in-service work within Ordinary, its state buffers
  // counted too (Cache::walkFor).
  const FreeResult reclaimed =
      test::takeFirst(spill.cache, {.cls = ClaimClass::InService, .reach = ReclaimClass::Ordinary},
                      {.bytes = 400, .stateBuffers = 1});
  require(reclaimed.nowBytes == 400 && spill.storage.releasedExtents == 1 &&
              spill.cache.pendingPages(kv::PageForm::Full) == 2,
          "the reclaim for memory stopped short of the extent an idle chain empties beside writes "
          "that spill");
  spill.cache.checkInvariants();
  for (uint32_t page : index.pages)
    spill.pool.releasePage(page, false);
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.endRequest(1);
}

// A Resident request denied growth converts to Spill and waits for its pages to spill; tier, budget
// or quota refusal changes nothing.
void testConversionWaitsForItsPagesToSpill() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 9);
  spill.start(1, prompt, 6, KvMode::Resident);
  spill.storage.growthBlocked = true;
  const uint64_t tokens = 9 * KvCache::pageTokens + 1;
  require(spill.cache.ensureTokens(1, tokens).failure == TokenAdmissionFailure::Denied,
          "the fixture's growth was not denied");
  spill.tier.writableFile = false;
  require(!spill.cache.convertToSpill(1, 4) && spill.budget->committedBytes() == 0,
          "a request converted with a closed tier");
  spill.tier.writableFile = true;
  require(!spill.cache.convertToSpill(1, 65) && spill.budget->committedBytes() == 0,
          "a request converted past its quota");
  const uint32_t allocations = spill.storage.allocationAttempts;
  require(spill.cache.convertToSpill(1, 4) && spill.budget->committedBytes() == 0 &&
              spill.tier.writes.size() == 6 && spill.storage.allocationAttempts == allocations,
          "the conversion did not commit and start its writes, or allocated");
  const TokenAdmission converted = spill.cache.ensureTokens(1, tokens);
  require(!converted.granted() && askLedger(spill.cache, converted).outcome == FreeOutcome::Pending,
          "the converted request did not wait for its pages to spill");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  const TokenAdmission runway = spill.cache.spillWritten(SpilledExtents::KeepRunway);
  require(runway.failure == TokenAdmissionFailure::Denied && runway.form == kv::PageForm::Index &&
              spill.cache.spillSnapshot().spilledPages == 4 && spill.cache.ensureTokens(1, tokens).granted(),
          "the converted request did not grow into the pages its spill freed");
  spill.cache.checkInvariants();
  spill.cache.endRequest(1);
  require(spill.budget->committedBytes() == 0, "the converted request kept its commitment");
}

// The cache refills the runway, so a conversion at the memory wall waits for its writes instead of
// being refused.
void testConversionAtTheWallFindsTheRunway() {
  Spilling spill;
  spill.start(1, Spilling::prompt(1000, 6), 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  require(spill.cache.spillSnapshot().spilledPages == 4 && spill.pool.freePageCount(kv::PageForm::Index) == 4,
          "the runway the first spill took was not refilled");
  spill.start(2, Spilling::prompt(5000, 5), 4, KvMode::Resident);
  spill.storage.growthBlocked = true;
  const uint64_t tokens = 5 * KvCache::pageTokens + 1;
  require(spill.cache.ensureTokens(2, tokens).failure == TokenAdmissionFailure::Denied,
          "the fixture's growth was not denied");
  require(spill.cache.convertToSpill(2, 5) && spill.cache.pendingPages(kv::PageForm::Full) == 2,
          "the request that converted at the wall did not start the writes of its pages");
  const TokenAdmission converted = spill.cache.ensureTokens(2, tokens);
  require(!converted.granted() && askLedger(spill.cache, converted).outcome == FreeOutcome::Pending,
          "the request that converted at the wall did not wait for its pages to spill");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  static_cast<void>(spill.cache.spillWritten(SpilledExtents::KeepRunway));
  require(spill.cache.spillSnapshot().spilledPages == 6 && spill.cache.ensureTokens(2, tokens).granted(),
          "the converted request's pages did not spill through the runway");
  spill.cache.checkInvariants();
  spill.cache.endRequest(2);
  spill.cache.endRequest(1);
}

// A Resident request unspills a block only as its only holder with no transfer moving it; a Spill
// holder reads the copies.
void testUnspillOnlyAsTheOnlyHolder() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  spill.publishState(spill.block(1, 5));
  const std::vector<bool> spilledChain{false, true, true, true, true, false};

  CacheLookup lookup = spill.cache.lookup(prompt, {});
  spill.cache.beginRequest(2);
  require(admitRestore(spill.cache, 2, lookup).granted() && spill.tier.restores == 0 &&
              spill.spilled(2) == spilledChain && spill.cache.holdsIndexPages(2),
          "a Resident request moved blocks another request holds spilled");
  lookup = {};
  spill.cache.endRequest(2);
  spill.cache.endRequest(1);

  lookup = spill.cache.lookup(prompt, {});
  spill.cache.beginRequest(3);
  require(admitRestore(spill.cache, 3, lookup).granted() && spill.tier.restores == 4 &&
              spill.cache.kvRestoreStatus(3) == KvRestoreStatus::Pending &&
              spill.spilled(3) == std::vector<bool>(6, false) && spill.cache.spillSnapshot().unspills == 4,
          "the only holder did not move the spilled blocks back to Full pages and restore them");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  require(spill.cache.kvRestoreStatus(3) == KvRestoreStatus::None &&
              spill.cache.spillSnapshot().wholeRestores == 4,
          "the whole restores did not land");
  lookup = {};
  spill.cache.endRequest(3);

  spill.toDisk();
  const uint32_t restores = spill.tier.restores;
  lookup = spill.cache.lookup(prompt, {});
  spill.cache.beginRequest(4, KvMode::Spill);
  require(spill.cache.commitSpill(4, 0) && admitRestore(spill.cache, 4, lookup).granted() &&
              spill.tier.restores == restores + 6 &&
              spill.spilled(4) == std::vector<bool>{false, true, true, true, true, true},
          "a Spill request did not restore the chain's index past its dense history");
  CacheLookup matched = spill.cache.lookup(prompt, {});
  spill.cache.beginRequest(5);
  require(admitRestore(spill.cache, 5, matched).granted() && spill.tier.restores == restores + 6 &&
              spill.cache.kvRestoreStatus(5) == KvRestoreStatus::Pending &&
              spill.spilled(5) == spill.spilled(4),
          "a Resident request moved blocks a restore was filling, or did not wait for it");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  require(spill.cache.kvRestoreStatus(5) == KvRestoreStatus::None &&
              spill.cache.spillSnapshot().indexRestores == 5,
          "the index restores did not land");
  spill.cache.checkInvariants();
  lookup = {};
  matched = {};
  spill.cache.endRequest(5);
  spill.cache.endRequest(4);
}

// An unspill takes free pages only beyond those higher-ranked waiting claims hold.
void testAnUnspillLeavesTheHeldFreePages() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  spill.publishState(spill.block(1, 5));
  spill.cache.endRequest(1);
  CacheLookup lookup = spill.cache.lookup(prompt, {});
  spill.cache.beginRequest(2);
  const uint32_t free = spill.pool.freePageCount(kv::PageForm::Full);
  spill.storage.growthBlocked = true;
  require(free >= 4 && spill.cache.restoreRequest(2, lookup, {free, 0, 0}).failure == TokenAdmissionFailure::Denied &&
              spill.pool.freePageCount(kv::PageForm::Full) == free && spill.cache.spillSnapshot().unspills == 0,
          "a Resident restore moved spilled blocks onto free pages waiting claims hold");
  spill.storage.growthBlocked = false;
  require(spill.cache.restoreRequest(2, lookup, {}).granted() && spill.cache.spillSnapshot().unspills == 4,
          "the Resident restore did not move the spilled blocks with no page held");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.checkInvariants();
  lookup = {};
  spill.cache.endRequest(2);
}

// A writer moves onto a block's page while a write reads it, but keeps its own page while a restore
// fills it.
void testWritersMoveOntoPagesBeingWritten() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 3);
  spill.start(1, prompt, 3);
  require(spill.cache.writeThrough() == 3, "the pages were not written through");
  spill.cache.beginRequest(2);
  require(admitTokens(spill.cache, 2, 3 * KvCache::pageTokens + 1).granted(),
          "the writer's pages were denied");
  require(spill.cache.publishCommittedBlocks(2, prompt, 3 * KvCache::pageTokens, {}) ==
                  3 * KvCache::pageTokens &&
              std::ranges::equal(spill.cache.pageTable(2).pages.first(3),
                                 spill.cache.pageTable(1).pages.first(3)),
          "a writer kept its own pages beside blocks being written");
  spill.publishState(spill.block(2, 2));
  spill.cache.endRequest(2);
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.endRequest(1);
  spill.toDisk();
  CacheLookup lookup = spill.cache.lookup(prompt, {});
  spill.cache.beginRequest(3);
  require(admitRestore(spill.cache, 3, lookup).granted() &&
              spill.cache.kvRestoreStatus(3) == KvRestoreStatus::Pending,
          "the chain's restore did not start");
  spill.cache.beginRequest(4);
  require(admitTokens(spill.cache, 4, 3 * KvCache::pageTokens + 1).granted() &&
              spill.cache.publishCommittedBlocks(4, prompt, 3 * KvCache::pageTokens, {}) == 0,
          "a writer moved onto pages a restore was filling");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.checkInvariants();
  lookup = {};
  spill.cache.endRequest(4);
  spill.cache.endRequest(3);
}

// A block whose restore failed keeps a page of whatever the pool held, and with no request holding
// that page it gives up its copy too. A writer that kept pages of its own beside the blocks a
// restore was filling goes on below them, and nothing writes that page as the block's: a persistent
// tier's next process would take it for the K/V.
void testAFailedRestoresPageIsNeverWritten() {
  for (const bool spills : {false, true}) {
    Spilling spill;
    const std::vector<uint32_t> prompt = Spilling::prompt(1000, 4);
    spill.start(1, prompt, 3);
    require(spill.cache.writeThrough() == 3, "the pages were not written through");
    spill.publishState(spill.block(1, 2));
    spill.tier.complete();
    static_cast<void>(spill.cache.pollTransfers());
    spill.cache.endRequest(1);
    spill.toDisk();
    CacheLookup lookup = spill.cache.lookup(std::span<const uint32_t>(prompt).first(3 * KvCache::pageTokens + 1), {});
    spill.cache.beginRequest(2);
    require(admitRestore(spill.cache, 2, lookup).granted() &&
                spill.cache.kvRestoreStatus(2) == KvRestoreStatus::Pending,
            "the chain's restore did not start");
    spill.start(3, prompt, 3, KvMode::Resident);
    lookup = {};
    spill.cache.endRequest(2);
    spill.tier.complete(false);
    static_cast<void>(spill.cache.pollTransfers());
    require(spill.cache.snapshot().kvTier.restoreFailures == 3, "the reads did not fail");
    spill.grow(3, prompt, 4);
    const uint64_t point = spill.block(3, 3);
    spill.publishState(point);
    const size_t before = spill.tier.writes.size();
    if (spills) {
      require(spill.cache.convertToSpill(3, 4), "the writer did not convert to Spill");
      static_cast<void>(spill.cache.writeThrough());
      require(spill.tier.writes.size() == before + 1, "a Spill request wrote a page whose restore failed");
    } else {
      require(spill.cache.persist(point) == PersistStatus::Unneeded && spill.tier.writes.size() == before,
              "a point's copies wrote a page whose restore failed");
    }
    spill.cache.checkInvariants();
    spill.tier.complete();
    static_cast<void>(spill.cache.pollTransfers());
    spill.cache.endRequest(3);
  }
}

// Quota never gives up a copy a running request reads; its slots return once the command's cut
// goes.
void testQuotaGivesUpOnlyCopiesNothingReads() {
  Spilling spill({1, 1, true}, 64);
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 5);
  spill.start(1, prompt, 5);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  std::optional<PageTableView> command = spill.cache.cutPageTable(1);
  spill.budget->setLimitBytes(0);
  spill.cache.keepDiskLimit();
  require(spill.tier.slots == 5 && spill.cache.spillSnapshot().heldBytes == 500,
          "quota gave up the copies a running request reads");
  spill.cache.checkInvariants();
  spill.cache.endRequest(1);
  do {
    spill.cache.keepDiskLimit();
  } while (test::evictStep(spill.cache, ReclaimClass::Ordinary).gave());
  require(spill.cache.snapshot().kvTier.diskBlocks == 0 && spill.cache.snapshot().pool.pagesPrefix == 0 &&
              spill.tier.slots == 3,
          "quota and reclaim did not give the ended request's blocks up, or the cut lost its copies");
  command.reset();
  require(spill.tier.slots == 0 && spill.budget->usedBytes() == 0,
          "the slots of the copies the cut named did not return once it went");
  spill.cache.checkInvariants();
}

// Quota never gives up the spilled leaf a state in RAM sits on; reclaim drops the state first.
void testQuotaKeepsTheSpilledLeafOfAStateInRam() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  const uint64_t leaf = spill.block(1, 4);
  auto noRoom = std::make_shared<TransferControl>();
  noRoom->capacity = 0;
  spill.cache.publishCompositeState(leaf, std::make_shared<TieredState>(noRoom), false);
  spill.cache.endRequest(1);
  require(test::evictStep(spill.cache, ReclaimClass::Ordinary).gave() &&
              spill.cache.snapshot().pool.pagesPrefix == 5,
          "the tail below the state did not leave RAM");
  const auto resume = std::span<const uint32_t>(prompt).first(5 * KvCache::pageTokens + 1);
  spill.budget->setLimitBytes(0);
  spill.cache.keepDiskLimit();
  require(spill.tier.slots == 4 && spill.cache.stateResident(leaf) &&
              spill.cache.lookup(resume, {}).resumeBoundary() == 5 * KvCache::pageTokens,
          "quota gave up the spilled leaf of a state in RAM");
  const FreeResult reclaimed = test::stateStep(spill.cache, ReclaimClass::Ordinary);
  require(reclaimed.gave() && !spill.cache.stateResident(leaf) && spill.tier.slots == 4 &&
              spill.cache.lookup(resume, {}).kvBoundary == 5 * KvCache::pageTokens,
          "making room for the state's write gave up its own leaf");
  spill.cache.keepDiskLimit();
  require(spill.tier.slots == 0 && spill.cache.snapshot().pool.pagesPrefix == 1,
          "quota kept the spilled chain once its state was gone");
  spill.cache.checkInvariants();
}

// A committed page's slot comes before any cached copy, the spilled chain of a state in RAM
// included.
void testCommittedPagesTakeTheSpilledChainOfAStateInRam() {
  Spilling spill({1, 1, true}, 6);
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  const uint64_t leaf = spill.block(1, 4);
  spill.publishState(leaf);
  spill.cache.endRequest(1);
  require(test::evictStep(spill.cache, ReclaimClass::Ordinary).gave() &&
              spill.cache.snapshot().pool.pagesPrefix == 5 && spill.tier.slots == 6,
          "the tail below the state did not leave RAM");
  const auto resume = std::span<const uint32_t>(prompt).first(5 * KvCache::pageTokens + 1);
  spill.start(2, Spilling::prompt(5000, 2), 2);
  require(spill.cache.writeThrough() == 2 && spill.cache.stateResident(leaf) &&
              spill.cache.lookup(resume, {}).resumeBoundary() == 5 * KvCache::pageTokens,
          "the first pages did not take the copies no state in RAM holds first");
  spill.grow(2, Spilling::prompt(5000, 6), 6);
  require(spill.cache.writeThrough() == 4 && !spill.cache.stateResident(leaf) &&
              spill.tier.slots == 6 && spill.cache.spillSnapshot().heldBytes == 600 &&
              spill.cache.lookup(resume, {}).kvBoundary == KvCache::pageTokens,
          "a committed page did not take the spilled chain of a state in RAM");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.checkInvariants();
}

// An unreadable copy poisons its block, which leaves at once with those below; the slots return
// with the cut.
void testUnreadableCopyLeavesBeforeTheCommandThatReadIt() {
  for (const bool poisonFirst : {false, true}) {
    for (const uint32_t page : {1u, 4u}) {
      Spilling spill;
      spill.start(1, Spilling::prompt(1000, 6), 6);
      static_cast<void>(test::spillRound(spill.cache, spill.tier));
      std::optional<PageTableView> command = spill.cache.cutPageTable(1);
      const uint64_t unreadable = spill.block(1, page);
      if (poisonFirst)
        spill.cache.poisonCopy(unreadable);
      spill.cache.endRequest(1);
      if (!poisonFirst)
        spill.cache.poisonCopy(unreadable);
      // The spilled pages past it keep their slots until the cut goes.
      require(spill.cache.snapshot().kvTier.diskBlocks == page && spill.tier.slots == 5,
              "the unreadable block and those below it did not leave, or the cut lost its copies");
      command.reset();
      require(spill.tier.slots == page, "the cut's slots did not return once it went");
      spill.cache.checkInvariants();
    }
  }
}

// A commitment locks quota until its pages are written; a lower active limit drops the newest
// commitment first.
void testCommitmentsLockQuotaUntilWritten() {
  Spilling spill({1, 1, true}, 10);
  const std::vector<uint32_t> cached = Spilling::prompt(1000, 4);
  spill.start(1, cached, 4);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  spill.cache.endRequest(1);
  require(spill.tier.slots == 4 && spill.budget->committedBytes() == 0, "the cached copies were not set up");

  // A commitment within the room takes no copy; one past it is refused.
  spill.cache.beginRequest(2, KvMode::Spill);
  require(spill.cache.commitSpill(2, 5) && spill.tier.slots == 4 && spill.cache.spillRoom() == 500,
          "a commitment within the room gave a copy up");
  spill.cache.beginRequest(3, KvMode::Spill);
  require(!spill.cache.commitSpill(3, 6) && spill.budget->committedBytes() == 500,
          "a commitment past the room was granted");
  spill.cache.endRequest(3);

  // Pages with a copy return their slots; new ones draw them.
  std::vector<uint32_t> longer = Spilling::prompt(1000, 6);
  spill.grow(2, longer, 6);
  require(spill.budget->committedBytes() == 100 && spill.cache.writeThrough() == 2 &&
              spill.budget->committedBytes() == 0 && spill.budget->usedBytes() == 600,
          "published pages did not return or draw the commitment");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.endRequest(2);

  // Two requests on one new prefix: the second's pages have copies on
  // their way, and its commitment goes back.
  const std::vector<uint32_t> shared = Spilling::prompt(5000, 2);
  spill.start(4, shared, 2);
  require(spill.cache.writeThrough() == 2, "the first request's pages were not written");
  spill.start(5, shared, 2);
  require(spill.budget->committedBytes() == 0, "two requests on one prefix committed it twice");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());

  // A falling active limit revokes the newest commitment first.
  spill.cache.endRequest(5);
  spill.cache.endRequest(4);
  spill.cache.beginRequest(6, KvMode::Spill);
  spill.cache.beginRequest(7, KvMode::Spill);
  require(spill.cache.commitSpill(6, 1) && spill.cache.commitSpill(7, 1), "the commitments were refused");
  spill.budget->setActiveLimitBytes(100);
  spill.cache.keepDiskLimit();
  require(spill.budget->committedBytes() == 100 && spill.cache.spillSnapshot().revokedCommitments == 1,
          "the newest commitment was not revoked first");
  spill.cache.endRequest(6);
  require(spill.budget->committedBytes() == 0, "an ending request kept its commitment");

  // A write that found the volume full pauses write-through.
  spill.budget->setActiveLimitBytes(1000);
  spill.grow(7, Spilling::prompt(9000, 1), 1);
  require(spill.cache.writeThrough() == 1, "the page was not written");
  spill.tier.complete(false);
  static_cast<void>(spill.cache.pollTransfers());
  spill.budget->setActiveLimitBytes(spill.cache.spillSnapshot().heldBytes);
  require(spill.cache.spillSnapshot().writeThroughPaused && spill.cache.writeThrough() == 0,
          "write-through went on after the volume filled");
  spill.budget->setActiveLimitBytes(1000);
  require(spill.cache.writeThrough() == 1 && !spill.cache.spillSnapshot().writeThroughPaused &&
              spill.cache.spillSnapshot().writeThroughFailures == 1,
          "write-through did not resume once the limit had room");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.checkInvariants();
  spill.cache.endRequest(7);
}

// Write-through resumes after a full volume where the blocks requests hold leave a slot, giving a
// cached copy up.
void testWriteThroughResumesBesideCopiesThatFillTheRoom() {
  // Every page stays Full, so that each cached copy is a redundant one.
  Spilling spill({8, 1, true}, 10);
  spill.start(1, Spilling::prompt(5000, 4), 4);
  require(spill.cache.writeThrough() == 4, "the cached copies were not written");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.endRequest(1);

  // A request whose three pages have copies commits two more, and the first
  // page past them finds the volume full.
  spill.start(2, Spilling::prompt(1000, 3), 3);
  require(spill.cache.writeThrough() == 3, "the request's pages were not written");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  require(spill.cache.commitSpill(2, 2), "the request's commitment was refused");
  spill.grow(2, Spilling::prompt(1000, 4), 4);
  require(spill.cache.writeThrough() == 1, "the page was not written");
  spill.tier.complete(false);
  static_cast<void>(spill.cache.pollTransfers());
  require(spill.cache.spillSnapshot().writeThroughPaused, "a full volume did not pause write-through");

  // The volume measured again leaves the size's limits 450 bytes: the
  // commitment goes, and the copies keep within a slot of the limit.
  spill.budget->setLimitBytes(450);
  spill.budget->setActiveLimitBytes(450);
  spill.cache.keepDiskLimit();
  const KvSpillSnapshot measured = spill.cache.spillSnapshot();
  require(measured.committedBytes == 0 && measured.heldBytes == 300 && spill.budget->usedBytes() == 400,
          "the measure did not take the commitment back and cut the copies to the limit");
  require(spill.cache.writeThrough() == 1 && !spill.cache.spillSnapshot().writeThroughPaused &&
              spill.budget->usedBytes() == 400,
          "write-through stayed paused while the blocks requests hold left a slot of room");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.checkInvariants();
  spill.cache.endRequest(2);
}

// A Spill start commits quota for the pages of the chain it restores that have no copy. A reclaim
// that writes one of them a copy before the restore holds it, as allocate() runs one around the
// restore, leaves that page written: the restore returns its page of the commitment, and the copies
// it holds and its commitment stay within active KV's limit.
void testACopyWrittenBeforeTheRestoreReturnsItsCommitment() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 4);
  spill.start(1, prompt, 4, KvMode::Resident);
  spill.publishState(spill.block(1, 3));
  spill.cache.endRequest(1);
  // The chain's last two blocks leave RAM for copies; the first two stay,
  // without copies.
  for (int step = 0; step < 50 && spill.cache.snapshot().pool.pagesPrefix > 2; ++step) {
    static_cast<void>(test::evictStep(spill.cache));
    spill.tier.complete();
    static_cast<void>(spill.cache.pollTransfers());
  }
  CacheLookup lookup = spill.cache.lookup(prompt, {});
  require(static_cast<bool>(lookup.state), "the chain's state was not found");
  const uint64_t chainEnd = lookup.state->kvBlock();
  const Cache::ChainCopies copies = spill.cache.chainCopies(chainEnd);
  require(copies.copied == 2 && copies.unheld == 2, "the chain did not keep two blocks without copies");
  // The limit holds exactly the start's commitment beside the copies it
  // restores.
  const uint32_t pages = 5 - copies.copied;
  spill.budget->setActiveLimitBytes(uint64_t{pages + copies.unheld} * 100);
  spill.cache.beginRequest(5, KvMode::Spill);
  require(spill.cache.commitSpill(5, pages, copies), "the commitment was refused");
  static_cast<void>(test::evictStep(spill.cache));
  require(spill.cache.chainCopies(chainEnd).copied == 3, "the reclaim did not write a block of the chain");
  require(admitRestore(spill.cache, 5, lookup).granted(), "the restore was refused");
  const auto withinLimit = [&] {
    const KvSpillSnapshot snapshot = spill.cache.spillSnapshot();
    return snapshot.heldBytes == 300 && snapshot.committedBytes == 200 &&
           snapshot.heldBytes + snapshot.committedBytes <= snapshot.activeLimitBytes;
  };
  require(withinLimit(), "the commitment still counted a page whose copy the restore holds");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  require(withinLimit(), "active KV passed its limit once the restore landed");
  spill.cache.checkInvariants();
  lookup = {};
  spill.cache.endRequest(5);
  require(spill.budget->committedBytes() == 0, "the commitment was not returned");
}

// A writer that computes blocks another request is restoring keeps its own pages beside them
// (KvCache::insert): the copies those blocks are read from are not its to hold, before the restore
// lands, after it, and after the restorer ends.
void testAWriterBesideARestoreHoldsNoCopyOfIt() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 4);
  spill.start(1, prompt, 4);
  require(spill.cache.writeThrough() == 4, "the pages were not written through");
  spill.tier.complete();
  require(spill.cache.pollTransfers(), "the writes did not land");
  spill.publishState(spill.block(1, 3));
  spill.cache.endRequest(1);
  spill.toDisk();
  // The restorer's restore stays in flight.
  spill.cache.beginRequest(2, KvMode::Resident);
  CacheLookup lookup = spill.cache.lookup(prompt, {});
  require(static_cast<bool>(lookup.state) && admitRestore(spill.cache, 2, lookup).granted() &&
              spill.cache.holdsActiveKvQuota(2),
          "the restorer did not hold the chain's copies");
  // The writer computes the chain again beside it.
  spill.cache.beginRequest(3, KvMode::Resident);
  spill.grow(3, prompt, 4);
  require(spill.cache.blockAt(3, 4 * KvCache::pageTokens) == spill.cache.blockAt(2, 4 * KvCache::pageTokens) &&
              !spill.cache.holdsActiveKvQuota(3),
          "a writer beside a restore held the copies it is restored from");
  spill.cache.checkInvariants();
  spill.tier.complete();
  require(spill.cache.pollTransfers() && !spill.cache.holdsActiveKvQuota(3),
          "a writer beside a restore held its copies once it landed");
  spill.cache.checkInvariants();
  lookup = {};
  spill.cache.endRequest(2);
  require(!spill.cache.holdsActiveKvQuota(3), "a writer held the copies once their restorer ended");
  spill.cache.checkInvariants();
  spill.cache.endRequest(3);
}

// A Spill start's commitment leaves room for the copies no request holds on the chain it restores,
// which it holds once it does: a commitment past what the limit leaves beside them is refused, a
// commitment of no page too when the limit cannot hold the copies alone.
void testACommitmentLeavesRoomForTheCopiesItsRestoreHolds() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 4);
  spill.start(1, prompt, 4);
  require(spill.cache.writeThrough() == 4, "the pages were not written through");
  spill.tier.complete();
  require(spill.cache.pollTransfers(), "the writes did not land");
  spill.publishState(spill.block(1, 3));
  spill.cache.endRequest(1);
  const CacheLookup lookup = spill.cache.lookup(prompt, {});
  require(static_cast<bool>(lookup.state), "the chain's state was not found");
  const Cache::ChainCopies copies = spill.cache.chainCopies(lookup.state->kvBlock());
  require(copies.unheld >= 2 && spill.cache.spillSnapshot().heldBytes == 0, "the chain kept no copies unheld");
  spill.budget->setActiveLimitBytes(uint64_t{copies.unheld + 2} * 100);
  spill.cache.beginRequest(5, KvMode::Spill);
  require(!spill.cache.commitSpill(5, 3, copies) && spill.budget->committedBytes() == 0,
          "a commitment took the room of the copies its restore holds");
  require(spill.cache.commitSpill(5, 2, copies) && spill.budget->committedBytes() == 200,
          "a commitment that fits beside the copies was refused");
  spill.cache.endRequest(5);
  spill.budget->setActiveLimitBytes(uint64_t{copies.unheld - 1} * 100);
  spill.cache.beginRequest(6, KvMode::Spill);
  require(!spill.cache.commitSpill(6, 0, copies) && spill.cache.commitSpill(6, 0),
          "a commitment of no page held copies the limit cannot hold");
  spill.cache.endRequest(6);
}

// A start's probe may hold a chain whose first block's read failed since: nothing past that block
// matches any more, and the copies past it take no quota for the start, as a refreshed probe finds.
void testARestoreReadsAChainOnlyAsFarAsItMatches() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 4);
  spill.start(1, prompt, 4);
  require(spill.cache.writeThrough() == 4, "the pages were not written through");
  spill.tier.complete();
  require(spill.cache.pollTransfers(), "the writes did not land");
  spill.publishState(spill.block(1, 3));
  spill.cache.endRequest(1);
  const CacheProbe probe = spill.cache.probe(prompt, {});
  const std::vector<uint64_t> chain(probe.restoredChain().begin(), probe.restoredChain().end());
  require(chain.size() == 4 && spill.cache.restoreTakesQuota(chain), "the chain's copies took no quota");
  // A lookup keeps the chain in the graph while its first block's read
  // fails.
  CacheLookup lookup = spill.cache.lookup(prompt, {});
  spill.cache.poisonCopy(chain.front());
  require(spill.cache.chainCopies(chain.back()).unheld == 3 &&
              !spill.cache.restoreTakesQuota(probe.restoredChain()),
          "a start's chain took quota for copies past a block that no longer matches");
  lookup = {};
}

// A failed write-through recommits its slot and owes the copy again; pages past it spill once it
// lands.
void testFailedWritesKeepTheirCommitment() {
  for (const bool everyWrite : {true, false}) {
    Spilling spill;
    // One write through at a time, for one of them to fail.
    if (!everyWrite)
      spill.tier.writeLimits = {~0u, 16, 1};
    spill.start(1, Spilling::prompt(1000, 6), 6);
    spill.budget->setLimitBytes(0);
    for (int round = 0; spill.cache.writeThrough(); ++round) {
      spill.tier.complete(!(everyWrite ? round == 0 : round == 1));
      static_cast<void>(spill.cache.pollTransfers());
      if (everyWrite && !round)
        require(spill.budget->committedBytes() == 600 && spill.budget->usedBytes() == 0,
                "the failed writes did not give their slots back to the commitment");
    }
    static_cast<void>(spill.cache.spillWritten(SpilledExtents::KeepRunway));
    const KvSpillSnapshot spilled = spill.cache.spillSnapshot();
    require(spilled.writeThroughPages == 6 && spilled.writeThroughFailures == (everyWrite ? 6u : 1u) &&
                spill.budget->committedBytes() == 0 && spilled.spilledPages == 4 &&
                std::ranges::all_of(spill.tier.draws,
                                    [](auto draw) { return draw == model::DiskBudget::Draw::Commitment; }),
            "a failed write's copy was not written again on its commitment, or held the pages past it back");
    spill.cache.checkInvariants();
    spill.cache.endRequest(1);
  }
}

// Spill pages past the commitment commit slots as they are written, within the active limit.
void testPagesPastTheCommitmentCommitTheirSlots() {
  Spilling spill;
  spill.budget->setLimitBytes(0);
  // The first request committed two of the four pages it published, the
  // second all four of its own.
  const std::vector<uint32_t> first = Spilling::prompt(1000, 4);
  spill.cache.beginRequest(1, KvMode::Spill);
  require(spill.cache.commitSpill(1, 2), "the first request's commitment was refused");
  spill.grow(1, first, 4);
  spill.start(2, Spilling::prompt(5000, 4), 4);
  // Active KV may hold the six committed slots and no more.
  spill.budget->setActiveLimitBytes(600);
  require(spill.cache.writeThrough() == 6 && spill.tier.writes.size() == 6 &&
              spill.budget->committedBytes() == 0 && spill.budget->usedBytes() == 600,
          "the first request's pages past its commitment held the second's committed pages back");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  static_cast<void>(spill.cache.spillWritten(SpilledExtents::KeepRunway));
  require(spill.spilled(1) == std::vector<bool>{false, true, false, false, false} &&
              spill.spilled(2) == std::vector<bool>{false, true, true, false, false},
          "a page without a copy spilled, or one with a copy did not");
  spill.budget->setActiveLimitBytes(6400);
  require(test::spillRound(spill.cache, spill.tier) == 2 &&
              spill.spilled(1) == std::vector<bool>{false, true, true, false, false} &&
              spill.budget->committedBytes() == 0 &&
              std::ranges::all_of(spill.tier.draws,
                                  [](auto draw) { return draw == model::DiskBudget::Draw::Commitment; }),
          "the pages past the commitment were not written on slots committed for them once the active "
          "limit had room");
  spill.cache.checkInvariants();
  spill.cache.endRequest(2);
  spill.cache.endRequest(1);
}

// A conversion whose history already has copies writes none of them again and moves nothing: its claim releases
// those pages at once, into free Index pages without an allocation (Cache::releasePages).
void testAConversionsHistoryWithCopiesIsReleasedAtOnce() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  spill.publishState(spill.block(1, 5));
  spill.cache.endRequest(1);
  // The only holder of the spilled chain moves it back to Full pages and
  // restores them whole, their copies kept.
  CacheLookup lookup = spill.cache.lookup(prompt, {});
  spill.cache.beginRequest(2);
  require(admitRestore(spill.cache, 2, lookup).granted() && spill.cache.spillSnapshot().unspills == 4,
          "the Resident request did not restore the spilled chain whole");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.storage.growthBlocked = true;
  const uint32_t held = static_cast<uint32_t>(spill.cache.pageTable(2).pages.size());
  const uint64_t tokens = uint64_t{held + spill.pool.freePageCount() + 1} * KvCache::pageTokens;
  require(spill.cache.ensureTokens(2, tokens).failure == TokenAdmissionFailure::Denied,
          "the fixture's growth was not denied");
  const uint32_t writes = static_cast<uint32_t>(spill.tier.writes.size());
  const uint32_t allocations = spill.storage.allocationAttempts;
  const ReleaseRule every{.takesFrom = [](uint64_t) { return true; }};
  require(spill.cache.convertToSpill(2, 0) && spill.tier.writes.size() == writes &&
              spill.spilled(2) == std::vector<bool>(6, false) && spill.cache.releasablePages(2, every).clean == 4,
          "the conversion wrote its history's pages again, moved them, or left them no release's");
  require(spill.cache.releasePages(2, 4, true, every) == 4 &&
              spill.spilled(2) == std::vector<bool>{false, true, true, true, true, false} &&
              spill.storage.allocationAttempts == allocations && spill.cache.spillSnapshot().releasedPages == 4,
          "a release of the pages whose copies had landed did not move them at once, or allocated");
  require(spill.cache.ensureTokens(2, tokens).granted(),
          "the converted request did not grow into the pages its release freed");
  spill.cache.checkInvariants();
  lookup = {};
  spill.cache.endRequest(2);
}

// A page spills only once every holder spills and holds it past its dense history and window.
void testEveryHolderDecidesWhetherAPageSpills() {
  Spilling spill({2, 2, true});
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 8);
  spill.start(1, prompt, 8);
  spill.start(2, prompt, 5);
  spill.start(3, prompt, 3, KvMode::Resident);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  require(spill.spilled(1) == std::vector<bool>{false, false, false, false, false, true, false, false, false},
          "a page spilled inside the dense history, a window, or beside a Resident holder");
  spill.cache.endRequest(3);
  spill.grow(2, Spilling::prompt(1000, 7), 7);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  require(spill.spilled(1) == std::vector<bool>{false, false, true, true, true, true, false, false, false},
          "the pages did not spill once their Resident holder ended and the other's window moved on");
  spill.cache.checkInvariants();
  spill.cache.endRequest(2);
  spill.cache.endRequest(1);
}

// Without quota pressure a block is written once; one whose redundant copy
// quota gave up is written again when a Spill request holds it, and counted.
void testBlocksAreWrittenOnce() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  for (int round = 0; round < 3; ++round)
    static_cast<void>(test::spillRound(spill.cache, spill.tier));
  require(spill.tier.writes.size() == 6 && spill.cache.spillSnapshot().rewrites == 0,
          "a block was written more than once");
  spill.cache.endRequest(1);
  spill.budget->setLimitBytes(400);
  spill.cache.keepDiskLimit();
  spill.budget->setLimitBytes(6400);
  spill.start(2, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  require(spill.tier.writes.size() == 8 && spill.cache.spillSnapshot().rewrites == 2,
          "the blocks whose copies went were not written again, or not counted");
  spill.cache.endRequest(2);
}

// A Spill request's owed copies count as returning Full pages while any of its copies is written.
void testOwedCopiesReturnWhileAnyIsWritten() {
  // A dense history of four pages; one write at a time.
  Spilling spill(CacheSpillPolicy{4, 1, true});
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 1;
  const std::vector<uint32_t> tokens = Spilling::prompt(1, 12);
  spill.start(1, tokens, 12);
  // Whether a claim of in-service work that lacks `pages` of the form beyond
  // the free ones waits for what lands and the copies owed (MemoryLedger::free).
  const auto waits = [&](uint32_t pages, kv::PageForm form = kv::PageForm::Full) {
    return test::freeFromCache(spill.cache, test::kInServiceGrowth, {.pages = pages, .form = form}).outcome ==
           FreeOutcome::Pending;
  };
  require(!waits(1), "a claim waited for copies no write had started");
  require(spill.cache.writeThrough() == 1, "the first copy was not written");
  // Seven of its pages spill once their copies land: past the dense history
  // and before the window. Three Full pages are free, four Index pages.
  require(spill.pool.freePageCount(kv::PageForm::Full) == 3 &&
              spill.pool.freePageCount(kv::PageForm::Index) == 4,
          "the pool did not hold the expected free pages");
  require(waits(7) && !waits(8), "a claim waited other than for the pages the copies owed behind the dense history's give back");
  require(!waits(1, kv::PageForm::Index), "a claim of Index pages waited for owed copies");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  spill.cache.endRequest(1);
  spill.cache.checkInvariants();
}

// A recomputed chain keeps its old copies; a Spill holder later owes new copies and spills only
// once they land.
void testRecomputedBlocksSpillToCopiesOfTheirPages() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  spill.publishState(spill.block(1, 5));
  spill.cache.endRequest(1);
  spill.toDisk();
  const size_t written = spill.tier.writes.size();
  spill.start(2, prompt, 6, KvMode::Resident);
  const uint64_t chainEnd = spill.block(2, 5);
  require(spill.tier.writes.size() == written && spill.cache.snapshot().kvTier.diskBlocks == 6 &&
              !spill.cache.chainCopies(chainEnd).copied,
          "a Resident request that computed a chain again wrote it, gave its copies up, or took them "
          "for copies of its pages");
  spill.cache.endRequest(2);
  spill.start(3, prompt, 6);
  require(spill.cache.writeThrough() == 6 && spill.cache.spillWritten(SpilledExtents::KeepRunway).granted() &&
              spill.spilled(3) == std::vector<bool>(7, false),
          "a Spill request did not write copies of its pages, or a page spilled to another's copy");
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  require(spill.cache.spillWritten(SpilledExtents::KeepRunway).granted() &&
              spill.spilled(3) == std::vector<bool>{false, true, true, true, true, false, false} &&
              spill.cache.chainCopies(chainEnd).copied == 6 && spill.cache.spillSnapshot().rewrites == 6 &&
              spill.cache.snapshot().kvTier.diskBlocks == 6,
          "the pages did not spill to the copies that replaced the old ones");
  spill.cache.checkInvariants();
  spill.cache.endRequest(3);
}

// An unreadable copy matches nothing, and leaves with its block once the requests holding it end.
void testUnreadableCopyLeavesWithItsHolders() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  const uint64_t unreadable = spill.block(1, 2);
  spill.cache.poisonCopy(unreadable);
  spill.cache.poisonCopy(unreadable);
  require(spill.cache.lookup(prompt, {}).kvBoundary == 2 * KvCache::pageTokens &&
              spill.tier.slots == 6 && (*spill.cache.cutPageTable(1).sources)[2].block == unreadable,
          "an unreadable copy still matched, or left while its holder reads it");
  spill.cache.checkInvariants();
  spill.cache.endRequest(1);
  require(spill.tier.slots == 2 && spill.cache.snapshot().kvTier.diskBlocks == 2,
          "the unreadable block and those below it did not leave with their copies");
}

// A poisoned block leaves once nothing refers to it, its own write in flight included.
void testPoisonedBlockLeavesOnceItsWriteLands() {
  test::TestKvTier tier;
  tier.transferLimit = 8;
  CacheFixture fixture(&tier, true);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(fixture.cache.persist(fixture.blocks[3]) == PersistStatus::Started && tier.writes.size() == 4,
          "the point's copies did not start");
  fixture.cache.poisonCopy(fixture.blocks[1]);
  fixture.cache.endRequest(1);
  tier.transfers[2]->ready = tier.transfers[3]->ready = true;
  static_cast<void>(fixture.cache.pollTransfers());
  require(fixture.cache.snapshot().pool.pagesPrefix == 2 && !fixture.cache.snapshot().stateCache.entries,
          "the subtree below the poisoned block did not leave once its writes landed");
  tier.complete();
  static_cast<void>(fixture.cache.pollTransfers());
  const CacheSnapshot landed = fixture.cache.snapshot();
  require(landed.pool.pagesPrefix == 1 && landed.kvTier.diskBlocks == 1,
          "the poisoned block did not leave with its page once its write landed");
  fixture.cache.checkInvariants();
}

// A windowed layout's cache, of a window of two pages, over pages of 100
// bytes, Slim ones of 30, in extents of four: requests run the prompt's
// pages as commands do, publishing the blocks they commit, and states land
// at the blocks they name.
struct Windowed final {
  static constexpr uint32_t kWindowPages = 2;
  test::TestKvStorage storage{64, 100, 4, 0, 30};
  KvPool pool{storage, 4};
  engine::Cache cache;

  explicit Windowed(engine::KvTier *tier = nullptr, bool trim = true)
      : cache(pool, tier, nullptr, std::nullopt, CacheWindowPolicy{kWindowPages, trim}) {}

  // `pages` pages of distinct tokens from `first`, and one token past them.
  static std::vector<uint32_t> prompt(uint32_t first, uint32_t pages) {
    std::vector<uint32_t> tokens(pages * KvCache::pageTokens + 1);
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
  }
  // The request's command commits its pages up to `pages`.
  void run(uint64_t id, const std::vector<uint32_t> &tokens, uint32_t pages) {
    require(admitTokens(cache, id, pages * KvCache::pageTokens + 1).granted(), "windowed KV was refused");
    cache.publishCommittedBlocks(id, tokens, pages * KvCache::pageTokens, {});
  }
  void publish(uint64_t id, uint32_t pages, bool checkpoint = false) {
    cache.publishCompositeState(cache.blockAt(id, pages * KvCache::pageTokens), std::make_shared<TestState>(100),
                                checkpoint);
  }
  // The forms of the request's pages, F for Full and S for Slim.
  std::string forms(uint64_t id) const {
    std::string result;
    for (const uint32_t page : cache.pageTable(id).pages)
      result += pool.form(page) == kv::PageForm::Slim ? 'S' : 'F';
    return result;
  }
  [[nodiscard]] uint32_t slimPages() const {
    const KvPoolSnapshot snapshot = pool.snapshot();
    return snapshot.slim.pagesAllocated - snapshot.slim.pagesFree;
  }
};

// A page drops its windowed layers' K/V once every holder is two blocks past it; trimming off keeps
// every page Full.
void testWindowsTrimWhatEveryHolderPassed() {
  Windowed windowed;
  engine::Cache &cache = windowed.cache;
  const std::vector<uint32_t> prompt = Windowed::prompt(5000, 8);
  cache.beginRequest(1);
  windowed.run(1, prompt, 3);
  require(cache.trimWindows() == 1 && windowed.forms(1) == "SFFF", "three published pages did not trim the first");
  windowed.run(1, prompt, 8);
  require(cache.trimWindows() == 5 && windowed.forms(1) == "SSSSSSFFF" &&
              cache.snapshot().window->trimmedPages == 6 && cache.trimWindows() == 0,
          "the pages the window passed were not trimmed, or trimmed twice");
  static_cast<void>(cache.cutPageTable(1));

  // A second request computes the first three blocks again: it takes them
  // back Full, while the first keeps the Slim pages, and only what it has
  // passed itself trims.
  cache.beginRequest(2);
  windowed.run(2, prompt, 3);
  const PageTableState first = cache.pageTable(1), second = cache.pageTable(2);
  require(cache.snapshot().window->upgrades == 3 && windowed.forms(2) == "FFFF" &&
              windowed.forms(1) == "SSSSSSFFF" && cache.blockAt(2, 32) == cache.blockAt(1, 32) &&
              first.pages[0] != second.pages[0],
          "a writer did not give the Slim blocks it computed again its own Full pages");
  require(cache.trimWindows() == 1 && windowed.forms(2) == "SFFF", "the writer's pages did not trim on its window");
  cache.checkInvariants();
  cache.endRequest(2);
  cache.endRequest(1);
  require(cache.trimWindows() == 4 && windowed.slimPages() == 8 && cache.snapshot().pool.pagesActive == 0,
          "the pages no request holds kept their windowed layers");
  cache.checkInvariants();

  Windowed full(nullptr, false);
  full.cache.beginRequest(1);
  full.run(1, prompt, 8);
  require(full.cache.trimWindows() == 0 && full.forms(1) == "FFFFFFFFF" && !full.cache.snapshot().window->trim,
          "a cache with trimming off trimmed");
  full.cache.endRequest(1);
}

// A writer recomputing trimmed or untrimmed blocks keeps its own pages; reads are the same with
// trimming on or off.
void testWritersKeepTheirPagesTrimmedOrNot() {
  for (const bool trim : {true, false}) {
    Windowed windowed(nullptr, trim);
    engine::Cache &cache = windowed.cache;
    const std::vector<uint32_t> prompt = Windowed::prompt(9000, 12);
    cache.beginRequest(1);
    windowed.run(1, prompt, 8);
    require(cache.trimWindows() == (trim ? 6u : 0u), "the first request's passed pages did not trim");
    const std::vector<uint32_t> first(cache.pageTable(1).pages.begin(), cache.pageTable(1).pages.end());

    // The second request computes the eight blocks again beside the first,
    // whose window holds the last two.
    cache.beginRequest(2);
    require(admitTokens(cache, 2, 8 * KvCache::pageTokens + 1).granted(), "the second request was refused");
    const std::vector<uint32_t> own(cache.pageTable(2).pages.begin(), cache.pageTable(2).pages.end());
    require(cache.publishCommittedBlocks(2, prompt, 8 * KvCache::pageTokens, {}) == 0 &&
                std::ranges::equal(cache.pageTable(2).pages, own) &&
                std::ranges::equal(cache.pageTable(1).pages, first) &&
                cache.blockAt(2, 8 * KvCache::pageTokens) == cache.blockAt(1, 8 * KvCache::pageTokens) &&
                cache.snapshot().window->upgrades == 8,
            "a writer moved onto the pages of the blocks it computed again, or they kept theirs");
    cache.checkInvariants();

    // Once the first has passed the two pages it kept, they trim, beside
    // the second's six and the two the first passed of its own.
    windowed.run(1, prompt, 12);
    require(cache.trimWindows() == (trim ? 10u : 0u) &&
                windowed.forms(1) == (trim ? "SSSSSSSSSSFFF" : "FFFFFFFFFFFFF") &&
                windowed.forms(2) == (trim ? "SSSSSSFFF" : "FFFFFFFFF"),
            "the pages a request kept beside its blocks did not trim once it passed them");
    cache.checkInvariants();
    cache.endRequest(1);
    cache.endRequest(2);
    cache.checkInvariants();
  }
}

// A state pins its two-block tail Full; an evicted state unpins it for the next trim.
void testStateTailsPinTheirWindows() {
  Windowed windowed;
  engine::Cache &cache = windowed.cache;
  const std::vector<uint32_t> prompt = Windowed::prompt(6000, 8);
  cache.beginRequest(1);
  windowed.run(1, prompt, 8);
  windowed.publish(1, 4);
  windowed.publish(1, 5);
  require(cache.snapshot().window->pinnedBlocks == 3, "adjacent states did not share their tails");
  cache.endRequest(1);
  require(cache.trimWindows() == 5, "the pages no tail holds were not trimmed");
  cache.checkInvariants();

  CacheLookup lookup = cache.lookup(std::span<const uint32_t>(prompt).first(4 * KvCache::pageTokens + 1), {});
  require(lookup.state && lookup.resumeBoundary() == 4 * KvCache::pageTokens, "the earlier state was not found");
  const uint64_t earlier = lookup.state->kvBlock();
  lookup = {};
  cache.invalidateState(earlier);
  require(cache.snapshot().window->pinnedBlocks == 2 && cache.trimWindows() == 1,
          "an evicted state's tail was not trimmed once unpinned");

  cache.beginRequest(2);
  CacheLookup resume = cache.lookup(prompt, {});
  require(resume.state && resume.resumeBoundary() == 5 * KvCache::pageTokens &&
              admitRestore(cache, 2, resume).granted() && windowed.forms(2) == "SSSFF",
          "a restore did not find its tail Full and its history Slim");
  static_cast<void>(cache.cutPageTable(2));
  cache.checkInvariants();
  cache.endRequest(2);
}

// A state whose tail is not Full is refused (a logic error), and the refusal is counted.
void testStatesNeedTheirTailFull() {
  Windowed windowed;
  engine::Cache &cache = windowed.cache;
  const std::vector<uint32_t> prompt = Windowed::prompt(7000, 6);
  cache.beginRequest(1);
  windowed.run(1, prompt, 6);
  require(cache.trimWindows() == 4, "the window's past was not trimmed");
  const uint64_t passed = cache.blockAt(1, 3 * KvCache::pageTokens);
  require(!cache.stateTailFull(passed) && cache.stateTailFull(cache.blockAt(1, 6 * KvCache::pageTokens)) &&
              cache.snapshot().window->refusedStates == 1,
          "a tail past the window was taken for Full, or the window's for less");
  rejects([&] { cache.publishCompositeState(passed, std::make_shared<TestState>(100), false); },
          "tail holds only the layers no window bounds", "a state was published on a Slim tail");
  windowed.publish(1, 6);
  cache.checkInvariants();
  cache.endRequest(1);
}

// A Full shortfall in a windowed cache trims first, then takes states (checkpoint first); all KV
// stays cached Slim.
void testFullShortfallsTrimBeforeAnyLeafGoes() {
  Windowed windowed;
  windowed.storage.budgetPages = 16;
  engine::Cache &cache = windowed.cache;
  const std::vector<uint32_t> cached = Windowed::prompt(8000, 8);
  cache.beginRequest(1);
  windowed.run(1, cached, 8);
  windowed.publish(1, 4, true);
  windowed.publish(1, 8);
  cache.endRequest(1);
  require(cache.trimWindows() == 4 && windowed.slimPages() == 4, "the unpinned pages were not trimmed");

  // Twelve Full pages are three Full extents, which leave the Slim ones no
  // room in the budget: the eight free pages need four more.
  cache.beginRequest(2);
  const TokenAdmission refused = cache.ensureTokens(2, 12 * KvCache::pageTokens);
  require(!refused.granted() && refused.form == kv::PageForm::Full, "the shortfall setup was granted");
  // The refusal's units until a state went.
  while (cache.snapshot().stateCache.entries == 2)
    require(test::takeFirst(cache, test::kInServiceGrowth, {.pages = refused.additionalPages}).outcome ==
                FreeOutcome::Met,
            "the refusal took no state");
  require(windowed.slimPages() == 6 &&
              cache.snapshot().stateCache.entries == 1 && cache.snapshot().stateCache.checkpointEntries == 0,
          "the checkpoint did not go first, or its tail was not trimmed");
  require(admitTokens(cache, 2, 12 * KvCache::pageTokens).granted() && windowed.slimPages() == 8 &&
              cache.snapshot().stateCache.entries == 0,
          "the shortfall was not met by the states' tails");
  const CacheLookup kept = cache.lookup(cached, {});
  require(kept.kvBoundary == 8 * KvCache::pageTokens && !kept.state, "a KV leaf went for Full pages");
  cache.checkInvariants();
  cache.endRequest(2);
}

// A state the SSD keeps still pins its tail Full; the shortfall moves on to the ordinary state.
void testFullShortfallsKeepTheTailsOfStatesTheSsdKeeps() {
  Windowed windowed;
  windowed.storage.budgetPages = 16;
  engine::Cache &cache = windowed.cache;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const std::vector<uint32_t> cached = Windowed::prompt(8000, 8);
  cache.beginRequest(1);
  windowed.run(1, cached, 8);
  const uint64_t checkpoint = cache.blockAt(1, 4 * KvCache::pageTokens);
  cache.publishCompositeState(checkpoint, std::make_shared<TieredState>(control), true);
  windowed.publish(1, 8);
  cache.endRequest(1);
  require(cache.trimWindows() == 4 && windowed.slimPages() == 4, "the unpinned pages were not trimmed");

  // A claim for a state's buffers writes the checkpoint first.
  require(test::freeFromCache(cache, test::kInServiceGrowth, {.stateBuffers = 1}).outcome == FreeOutcome::Met &&
              control->slots == 1 && cache.pollTransfers(),
          "the checkpoint was not written to the SSD");
  cache.beginRequest(2);
  const TokenAdmission refused = cache.ensureTokens(2, 12 * KvCache::pageTokens);
  require(!refused.granted() && refused.form == kv::PageForm::Full, "the shortfall setup was granted");
  static_cast<void>(test::takeFirst(cache, test::kInServiceGrowth, {.pages = refused.additionalPages}));
  const auto states = cache.snapshot().stateCache;
  require(states.entries == 1 && states.checkpointEntries == 1 && states.diskBytes == 100 && control->slots == 1 &&
              windowed.slimPages() == 6 && cache.stateTailFull(checkpoint),
          "the checkpoint the SSD keeps lost its entry, or its tail was trimmed");
  cache.checkInvariants();
  cache.endRequest(2);
}

// At the memory wall a Full shortfall takes dead leaves first, keeping the states a resume needs.
void testFullShortfallsAtTheWallTakeDeadLeavesFirst() {
  for (const bool trim : {true, false}) {
    Windowed windowed(nullptr, trim);
    windowed.storage.budgetPages = 12;
    engine::Cache &cache = windowed.cache;
    const std::vector<uint32_t> first = Windowed::prompt(1000, 2), second = Windowed::prompt(2000, 2),
                                dead = Windowed::prompt(3000, 6);
    cache.beginRequest(1);
    windowed.run(1, first, 2);
    windowed.publish(1, 2);
    cache.endRequest(1);
    cache.beginRequest(2);
    windowed.run(2, second, 2);
    windowed.publish(2, 2);
    cache.endRequest(2);
    cache.beginRequest(3);
    windowed.run(3, dead, 6);
    cache.endRequest(3);
    require(cache.trimWindows() == 0 && windowed.slimPages() == 0 && cache.snapshot().pool.pagesFree == 2,
            "the Full extents left room in the budget");

    // Four pages, two of them free: two of the dead chain's leaves go.
    cache.beginRequest(4);
    require(admitTokens(cache, 4, 4 * KvCache::pageTokens).granted() && cache.snapshot().stateCache.entries == 2 &&
                cache.lookup(first, {}).state && cache.lookup(second, {}).state &&
                cache.lookup(dead, {}).kvBoundary == 4 * KvCache::pageTokens,
            "a Full shortfall at the wall took a state before the dead leaves");
    cache.checkInvariants();
    cache.endRequest(4);
  }
}

// A demotion returns its page as it lands; a restore returns nothing.
void testRestoresReturnNothing() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache) == 100 && fixture.cache.pollTransfers() &&
              test::evictStep(fixture.cache).gave(),
          "the leaf was not demoted");
  const Returning demoting = fixture.cache.returning();
  require(demoting.transfers == 1 && demoting.landingPages[static_cast<size_t>(kv::PageForm::Full)] == 1 &&
              !demoting.owedPages[static_cast<size_t>(kv::PageForm::Full)],
          "a demotion in flight did not return its page");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 1, "the leaf did not land");
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(2);
  require(admitRestore(fixture.cache, 2, lookup).granted() && fixture.cache.transfersInFlight(),
          "the restore did not start");
  const Returning restoring = fixture.cache.returning();
  require(!restoring.transfers && restoring.landingPages == std::array<uint32_t, 3>{} && !restoring.landingBytes &&
              fixture.cache.returningFor(std::vector<uint64_t>{2}).landingPages == std::array<uint32_t, 3>{},
          "a restore counted as memory on its way back");
  tier.complete();
  static_cast<void>(fixture.cache.pollTransfers());
  lookup = {};
  fixture.cache.endRequest(2);
}

// A demotion returns a page only from a leaf; once a block was written below it, a can-wait claim
// finds nothing to wait for.
void testHeldDemotionReturnsNoPage() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache) == 100 && fixture.cache.pollTransfers() &&
              test::evictStep(fixture.cache).gave() &&
              tier.demotions == 1,
          "the leaf's demotion did not start");
  constexpr auto full = static_cast<size_t>(kv::PageForm::Full);
  constexpr uint32_t extended = 5 * KvCache::pageTokens;
  const Need page{.pages = 1};
  std::vector<uint32_t> longer = fixture.prompt;
  longer.pop_back();
  for (uint32_t row = 0; row <= KvCache::pageTokens; ++row)
    longer.push_back(5000 + row);
  fixture.storage.budgetPages = 5;
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(2);
  require(admitRestore(fixture.cache, 2, lookup).granted() && admitTokens(fixture.cache, 2, extended).granted(),
          "the request did not extend the chain being demoted");
  lookup = {};
  fixture.cache.publishCommittedBlocks(2, longer, extended, {});
  const Returning held = fixture.cache.returning();
  require(fixture.cache.blockAt(2, extended) && held.transfers == 1 && !held.landingPages[full] &&
              !held.landingBytes,
          "a demotion whose block keeps a resident child returned its page");
  require(test::freeFromCache(fixture.cache, {}, page).outcome == FreeOutcome::Short,
          "a claim that can wait waited for the page of a demotion a resident child holds");
  fixture.cache.endRequest(2);
  require(test::evictStep(fixture.cache).gave() &&
              fixture.pool.freePageCount() == 1,
          "the block below the leaf did not go");
  require(fixture.cache.returning().landingPages[full] == 1 &&
              test::freeFromCache(fixture.cache, {}, page).outcome == FreeOutcome::Pending,
          "a demotion whose block is a leaf again did not return its page");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 2, "the demotion did not free its page");
}

// A claim that cannot wait counts no landing page: the dead leaf it evicts under a block being
// demoted gives it a free page now, though the demotion then frees its page as it lands.
void testImmediateEvictionCountsNoLandingPage() {
  test::TestKvTier tier;
  CacheFixture fixture(&tier);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  require(reclaimStateBytes(fixture.cache) == 100 && fixture.cache.pollTransfers() &&
              test::evictStep(fixture.cache).gave() && tier.demotions == 1,
          "the leaf's demotion did not start");
  constexpr auto full = static_cast<size_t>(kv::PageForm::Full);
  constexpr uint32_t extended = 5 * KvCache::pageTokens;
  std::vector<uint32_t> longer = fixture.prompt;
  longer.pop_back();
  for (uint32_t row = 0; row <= KvCache::pageTokens; ++row)
    longer.push_back(5000 + row);
  fixture.storage.budgetPages = 5;
  auto lookup = fixture.lookup(129);
  fixture.cache.beginRequest(2);
  require(admitRestore(fixture.cache, 2, lookup).granted() && admitTokens(fixture.cache, 2, extended).granted(),
          "the request did not extend the chain being demoted");
  lookup = {};
  fixture.cache.publishCommittedBlocks(2, longer, extended, {});
  fixture.cache.endRequest(2);
  require(!fixture.cache.returning().landingPages[full], "the demotion under the dead leaf returned its page");
  const ClaimSpec cannotWait{.cls = ClaimClass::InService, .timing = NeedTiming::Immediate};
  const FreeResult freed = test::freeFromCache(fixture.cache, cannotWait, {.pages = 1});
  require(freed.outcome == FreeOutcome::Met && freed.pages == 1 && fixture.pool.freePageCount() == 1 &&
              fixture.cache.returning().landingPages[full] == 1,
          "a claim that cannot wait was left waiting for the page of a demotion its eviction made a leaf");
  tier.complete();
  require(fixture.cache.pollTransfers() && fixture.pool.freePageCount() == 2, "the demotion did not free its page");
}

// What spilling returns splits into landing (in-flight writes that spill) and owed (copies not
// started); per request.
void testReturningSplitsLandingAndOwed() {
  Spilling spill(CacheSpillPolicy{1, 1, true});
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 2;
  spill.start(1, Spilling::prompt(1000, 8), 8);
  spill.start(2, Spilling::prompt(5000, 4), 4);
  require(spill.cache.writeThrough() == 2, "the first copies were not written");
  // Request 1's dense page and its second, past the dense history: the
  // second spills once it lands. Pages 2 to 6 spill once copied, the last
  // stays in its window; request 2's pages 1 and 2 spill once copied.
  constexpr auto full = static_cast<size_t>(kv::PageForm::Full);
  const Returning all = spill.cache.returning();
  const Returning first = spill.cache.returningFor(std::vector<uint64_t>{1});
  const Returning second = spill.cache.returningFor(std::vector<uint64_t>{2});
  require(all.transfers == 1 && all.landingPages[full] == 1 && all.owedPages[full] == 7,
          "the spilling writes and the owed copies were not split into landing and owed");
  require(first.transfers == 1 && first.landingPages[full] == 1 && first.owedPages[full] == 5 &&
              !second.transfers && !second.landingPages[full] && second.owedPages[full] == 2,
          "a request's own count took in another's pages");
  // Two requests' owed copies counted for one claim: the extents they complete
  // beside every landing, the free Full pages counted once.
  const Returning both = spill.cache.returningFor(std::vector<uint64_t>{1, 2});
  const uint32_t free = spill.pool.freePageCount(kv::PageForm::Full);
  const auto extents = [](uint64_t pages) { return pages / 4; };
  const uint64_t completed = (extents(free + all.landingPages[full] + 7) - extents(free)) * 4 * 100;
  require(both.owedPages[full] == 7 && all.landingBytes + both.owedBytes == completed,
          "the requests' owed copies counted the free Full pages once a request: " +
              std::to_string(all.landingBytes + both.owedBytes) + " bytes come back, not " +
              std::to_string(completed));
  // With no free Index page a landing spills nothing; the owed copies count
  // all the same.
  KvPageAcquisition index = spill.pool.acquirePages(spill.pool.freePageCount(kv::PageForm::Index), kv::PageForm::Index);
  const Returning capped = spill.cache.returning();
  require(index.granted() && !capped.landingPages[full] && capped.owedPages[full] == 7,
          "the landing pages were not capped by the free Index pages, or the owed copies were");
  for (const uint32_t page : index.pages)
    spill.pool.releasePage(page, false);
  spill.tier.complete();
  static_cast<void>(spill.cache.pollTransfers());
  require(!spill.cache.returning().owedPages[full], "copies were owed while none was written");
  spill.cache.endRequest(1);
  spill.cache.endRequest(2);
}

// Walking every unit of every query leaves the cache as it was.
void testPeekHasNoSideEffects() {
  test::TestKvTier tier;
  tier.transferLimit = 8;
  CacheFixture fixture(&tier, true);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), true);
  fixture.cache.publishCompositeState(fixture.blocks[3], std::make_shared<TieredState>(control), false);
  fixture.cache.endRequest(1);
  const auto observe = [&] {
    const CacheSnapshot snapshot = fixture.cache.snapshot();
    return std::array<uint64_t, 9>{snapshot.pool.pagesPrefix, snapshot.pool.pagesFree, snapshot.pool.allocatedBytes,
                                   snapshot.stateCache.entries, snapshot.stateCache.bytes, snapshot.stateCache.diskBytes,
                                   snapshot.kvTier.demotions, snapshot.kvTier.demotionsRefused, tier.writes.size()};
  };
  const auto before = observe();
  uint32_t units = 0;
  for (const ClaimClass cls : {ClaimClass::InService, ClaimClass::Growth, ClaimClass::Host})
    for (const NeedTiming timing : {NeedTiming::CanWait, NeedTiming::Immediate})
      for (const ReclaimClass reach : {ReclaimClass::Ordinary, ReclaimClass::InUse})
        for (const Need need : {Need{.bytes = 1}, Need{.pages = 2}, Need{.hostRefused = true, .pages = 2},
                                Need{.stateBuffers = 1}}) {
          const ClaimSpec claim{.cls = cls, .reach = reach, .timing = timing};
          const VictimQuery query = test::query(claim, need);
          for (std::optional<engine::Victim> victim = fixture.cache.peek(query, nullptr); victim;
               victim = fixture.cache.peek(query, &*victim))
            ++units;
        }
  fixture.cache.checkInvariants();
  require(units && observe() == before, "the walk changed the cache");
}

// The resume point and wanted chains come in their place to in-service work, the resume point alone
// to a start; Growth and a host below Critical are never offered them.
void testResumePointAndWantedChainsAreCertainReuse() {
  test::TestKvStorage storage{16, 100, 1};
  KvPool pool{storage, 16};
  engine::Cache cache(pool, nullptr, nullptr);
  const auto cachePrompt = [&](uint64_t id, uint32_t first) {
    std::vector<uint32_t> prompt(2 * KvCache::pageTokens + 1);
    std::iota(prompt.begin(), prompt.end(), first);
    cache.beginRequest(id);
    require(admitTokens(cache, id, prompt.size()).granted(), "a prompt's pages were denied");
    cache.publishCommittedBlocks(id, prompt, 2 * KvCache::pageTokens, {});
    cache.publishCompositeState(cache.blockAt(id, 2 * KvCache::pageTokens), std::make_shared<TestState>(100), false);
    cache.endRequest(id);
    return prompt;
  };
  const std::vector<uint32_t> wanted = cachePrompt(1, 1000);
  const std::vector<uint32_t> other = cachePrompt(2, 5000);
  const std::vector<uint32_t> newest = cachePrompt(3, 9000);
  CacheProbe probe = cache.probe(wanted, {});
  cache.setWantedChains({probe.restoredChain()});
  // The units and pages a claim of the class is offered.
  const auto offered = [&](ClaimClass cls, const Need &need) {
    const VictimQuery query = test::query({.cls = cls}, need);
    std::pair<uint32_t, uint32_t> units;
    for (std::optional<engine::Victim> victim = cache.peek(query, nullptr); victim;
         victim = cache.peek(query, &*victim)) {
      ++units.first;
      units.second += victim->pages;
    }
    return units;
  };
  // Prompt 1's state (wanted), prompt 2's and prompt 3's (the resume point).
  require(offered(ClaimClass::InService, {.stateBuffers = 1}).first == 3,
          "in-service work was not offered the resume point and the wanted chain's state");
  require(offered(ClaimClass::Start, {.stateBuffers = 1}).first == 2,
          "a start was not offered the resume point alone of certain reuse");
  // Growth and a host pass short of critical pressure leave them: a page
  // need's units free prompt 2's pages alone, its state and then its
  // leaves, and a device need's units every extent but theirs.
  for (const ClaimClass cls : {ClaimClass::Host, ClaimClass::Growth})
    require(offered(cls, {.stateBuffers = 1}).first == 1 && offered(cls, {.pages = 1}).second == 2,
            "a claim that may take nothing of certain reuse was offered some, or not the rest");
  const ClaimSpec growth{.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate};
  require(test::freeFromCache(cache, growth, {.pages = 2}).outcome == FreeOutcome::Met,
          "growth did not take what is not of certain reuse");
  cache.refresh(probe, wanted, {});
  require(probe.cachedTokens() == 2 * KvCache::pageTokens &&
              cache.stateResident(probe.restoredChain().back()) &&
              cache.probe(newest, {}).cachedTokens() == 2 * KvCache::pageTokens &&
              !cache.probe(other, {}).cachedTokens(),
          "growth's memory reclaims took the wanted chain or the resume point");
  cache.setWantedChains({});
  require(offered(ClaimClass::Growth, {.stateBuffers = 1}).first == 1,
          "a chain no request waits for any more stayed of certain reuse");
}

// Units in use come only to a claim with reach InUse, after everything else.
void testInUseUnitsComeOnlyWithReachInUse() {
  CacheFixture fixture;
  fixture.publish(1);
  fixture.publish(3);
  StateUse use = fixture.cache.useState(fixture.blocks[3]);
  const auto units = [&](ReclaimClass reach) {
    const ClaimSpec claim{.reach = reach};
    const VictimQuery query = test::query(claim, {.stateBuffers = 1});
    std::vector<engine::Victim> result;
    for (std::optional<engine::Victim> victim = fixture.cache.peek(query, nullptr); victim;
         victim = fixture.cache.peek(query, &*victim))
      result.push_back(*victim);
    return result;
  };
  const std::vector<engine::Victim> ordinary = units(ReclaimClass::Ordinary);
  const std::vector<engine::Victim> inUse = units(ReclaimClass::InUse);
  require(ordinary.size() == 1 && ordinary[0].cls == ReclaimClass::Ordinary && inUse.size() == 2 &&
              inUse[1].cls == ReclaimClass::InUse,
          "what is in use came to a claim that does not reach it, or before the rest");
}

// Each take of a device need releases one whole extent, taking only the leaves that empty it.
void testBurstReclaimTakesWholeExtentsOncePerNeed() {
  test::TestKvStorage storage{64, 100, 4};
  KvPool pool{storage, 4};
  engine::Cache cache(pool, nullptr, nullptr);
  for (uint64_t id = 1; id <= 4; ++id) {
    std::vector<uint32_t> prompt(8 * KvCache::pageTokens + 1);
    std::iota(prompt.begin(), prompt.end(), 1000 * id);
    cache.beginRequest(id);
    require(admitTokens(cache, id, prompt.size()).granted(), "a conversation's pages were denied");
    cache.publishCommittedBlocks(id, prompt, 8 * KvCache::pageTokens, {});
    cache.endRequest(id);
  }
  // The empty extent the conversations' open pages left goes first.
  static_cast<void>(cache.releaseEmptyExtents(false));
  const uint64_t extent = 4 * 100;
  const ClaimSpec claim{.cls = ClaimClass::InService};
  const VictimQuery query = test::query(claim, {.bytes = 3 * extent});
  for (uint32_t take = 0; take < 3; ++take) {
    const std::optional<engine::Victim> unit = cache.peek(query, nullptr);
    const uint64_t allocated = pool.allocatedBytes(), prefix = cache.snapshot().pool.pagesPrefix;
    require(unit && cache.take(query, *unit).outcome == FreeOutcome::Met &&
                allocated - pool.allocatedBytes() == extent && prefix - cache.snapshot().pool.pagesPrefix == 4,
            "a memory reclaim did not take the leaves that empty one extent and release it");
  }
  cache.checkInvariants();
}

// A lane's growth (Cache::growthNeed), as placement weighs it, takes the
// free pages first and the rest in whole extents, and the walk for it keeps
// those pages, as the claims that grow the lane keep them: beside a
// conversation cached on one and a half extents, the walk finds only the
// extent the conversation's leaves empty. Counted as free and again as
// room, the free pages let the walk release the half-used extent too, and a
// lane placed on that failed at its last extent.
void testGrowthNeedKeepsTheFreePagesItTakes() {
  test::TestKvStorage storage{64, 100, 4};
  KvPool pool{storage, 0};
  engine::Cache cache(pool, nullptr, nullptr);
  std::vector<uint32_t> prompt(6 * KvCache::pageTokens + 1);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, prompt.size()).granted(), "a conversation's pages were denied");
  cache.publishCommittedBlocks(1, prompt, 6 * KvCache::pageTokens, {});
  cache.endRequest(1);
  const uint64_t extent = 4 * 100;
  require(pool.freePageCount(kv::PageForm::Full) == 2 && pool.allocatedBytes() == 2 * extent,
          "the conversation did not leave two free pages in its second extent");
  const Need need = cache.growthNeed(10);
  const ClaimSpec start{.cls = ClaimClass::Start};
  const VictimQuery query = test::query(start, need);
  uint64_t found = 0;
  for (std::optional<engine::Victim> victim = cache.peek(query, nullptr); victim; victim = cache.peek(query, &*victim))
    found += victim->bytes;
  require(need.bytes == 2 * extent && found == extent,
          "a lane's growth counted the free pages it takes as room: needs " + std::to_string(need.bytes) +
              " B, found " + std::to_string(found) + " B");
}

// A device need's walk reaches the leaves past the states it gives up: every extent goes, the
// checkpoint written.
void testDeviceNeedReachesTheLeavesPastTheStates() {
  test::TestKvStorage storage{48, 100, 4};
  KvPool pool{storage, 4};
  engine::Cache cache(pool, nullptr, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const auto cachePrompt = [&](uint64_t id, uint32_t pages) {
    std::vector<uint32_t> prompt(pages * KvCache::pageTokens + 1);
    std::iota(prompt.begin(), prompt.end(), 1000 * id);
    cache.beginRequest(id);
    require(admitTokens(cache, id, prompt.size()).granted(), "a conversation's pages were denied");
    cache.publishCommittedBlocks(id, prompt, pages * KvCache::pageTokens, {});
    const std::array blocks{cache.blockAt(id, pages / 2 * KvCache::pageTokens),
                            cache.blockAt(id, pages * KvCache::pageTokens)};
    cache.endRequest(id);
    return blocks;
  };
  const std::array<uint64_t, 2> oldest = cachePrompt(1, 8);
  cache.publishCompositeState(oldest[0], std::make_shared<TieredState>(control), true);
  cache.publishCompositeState(oldest[1], std::make_shared<TestState>(100), false);
  for (uint64_t id = 2; id <= 5; ++id)
    static_cast<void>(cachePrompt(id, 4));
  static_cast<void>(cache.releaseEmptyExtents(false));
  const ClaimSpec claim{.cls = ClaimClass::InService};
  require(test::freeFromCache(cache, claim, {.bytes = pool.allocatedBytes()}).outcome == FreeOutcome::Met,
          "the cache did not give its extents");
  require(!pool.allocatedBytes() && cache.snapshot().stateCache.offloads == 1,
          "the units left extents, or the checkpoint unwritten");
  cache.checkInvariants();
}

// A lone request's KV growth that needs the extents of every idle chain but the resume point's, while
// each chain's states need writes and one write holds the write slot: the claim's own first write,
// or with `inFlight` the write of a claim before it. The walk takes each
// state behind that write as a unit that lands later, the ordinary ones at the chains' leaves (with
// `inUse` held in use) and with `checkpoints` the checkpoints on the chains' second blocks, and the
// newer states of its phase follow it, so the walk reaches every chain's pages and the claim waits
// instead of being short. The writes land one at a time, each releasing its chain's extent to the
// claims that follow.
void growPastAStateWrite(bool checkpoints, bool inUse, bool inFlight) {
  const std::string label = std::string(checkpoints ? "checkpoints" : "leaf states") + (inUse ? " in use" : "") +
                            (inFlight ? ", write in flight: " : ": ");
  test::TestKvStorage storage{48, 100, 4};
  KvPool pool{storage, 4};
  engine::Cache cache(pool, nullptr, nullptr);
  auto control = std::make_shared<TransferControl>();
  std::vector<StateUse> uses;
  for (uint64_t id = 1; id <= 5; ++id) {
    std::vector<uint32_t> prompt(4 * KvCache::pageTokens + 1);
    std::iota(prompt.begin(), prompt.end(), 1000 * id);
    cache.beginRequest(id);
    require(admitTokens(cache, id, prompt.size()).granted(), label + "a conversation's pages were denied");
    cache.publishCommittedBlocks(id, prompt, 4 * KvCache::pageTokens, {});
    const uint64_t inner = cache.blockAt(id, 2 * KvCache::pageTokens);
    const uint64_t leaf = cache.blockAt(id, 4 * KvCache::pageTokens);
    cache.endRequest(id);
    if (checkpoints)
      cache.publishCompositeState(inner, std::make_shared<TieredState>(control), true);
    cache.publishCompositeState(leaf, std::make_shared<TieredState>(control), !checkpoints && id == 1);
    if (inUse)
      uses.push_back(cache.useState(leaf));
  }
  static_cast<void>(cache.releaseEmptyExtents(false));
  const uint64_t extent = storage.extentPages() * 100, need = 4 * extent;
  const uint64_t allocated = pool.allocatedBytes();
  const auto released = [&] { return allocated - pool.allocatedBytes(); };
  const auto claim = [&](const char *when) {
    const FreeResult freed = test::freeFromCache(cache, test::kInServiceGrowth, {.bytes = need - released()});
    require(freed.outcome != FreeOutcome::Short,
            label + "growth was short with idle chains behind the state write " + when);
  };
  if (inFlight)
    require(test::freeFromCache(cache, test::kInServiceGrowth, {.bytes = extent}).outcome != FreeOutcome::Short,
            label + "the first claim was short");
  if (released() < need)
    claim("before the writes landed");
  control->ready = true;
  for (int round = 0; round < 16 && released() < need; ++round) {
    static_cast<void>(cache.pollTransfers());
    if (released() < need)
      claim("as the writes landed");
  }
  require(released() >= need, label + "the idle chains did not free their extents once the writes landed: " +
                                  std::to_string(released()) + " bytes released");
  cache.checkInvariants();
}

void testGrowthPastAStateWriteReachesTheIdleChains() {
  for (const bool checkpoints : {false, true})
    for (const bool inUse : {false, true})
      for (const bool inFlight : {false, true})
        growPastAStateWrite(checkpoints, inUse, inFlight);
}

// A device need takes KV only while it can empty an extent past the kept runways; otherwise it
// takes a state.
void testDeviceNeedKeepsKvBesideTheIndexRunway() {
  Spilling spill;
  const uint64_t needed = cacheChain(spill.cache, 1, 1).second[0];
  spill.publishState(needed);
  require(test::stateStep(spill.cache).gave() && spill.cache.pollTransfers(),
          "the leaf's state was not written");
  static_cast<void>(cacheChain(spill.cache, 2, 3, true));
  static_cast<void>(cacheChain(spill.cache, 3, 1));
  const uint64_t held = cacheChain(spill.cache, 4, 3, true).second[2];
  spill.cache.publishCompositeState(held, std::make_shared<TestState>(100), false);
  require(spill.cache.reusablePages(ReclaimClass::Ordinary) == 2 &&
              spill.cache.reusablePages(ReclaimClass::Ordinary, kv::PageForm::Index) == 4 &&
              spill.pool.freePageCount(kv::PageForm::Index) == 4 && spill.tier.writes.empty(),
          "fixture geometry changed");
  const uint64_t cached = spill.cache.snapshot().pool.pagesPrefix;
  const ClaimSpec claim{.cls = ClaimClass::InService};
  const VictimQuery query = test::query(claim, {.bytes = 400});
  const std::optional<engine::Victim> unit = spill.cache.peek(query, nullptr);
  require(unit && unit->stateBuffers == 1 && !unit->bytes && !unit->later,
          "the walk offered KV that empties no extent but the Index runway it keeps");
  require(spill.cache.take(query, *unit).outcome == FreeOutcome::Met && !spill.cache.stateResident(held) &&
              spill.cache.snapshot().pool.pagesPrefix == cached && spill.tier.writes.empty(),
          "a device need took KV that empties no extent but the Index runway it keeps");
  spill.cache.checkInvariants();
  spill.cache.endRequest(2);
  spill.cache.endRequest(4);
}

// A device need's walk passes over leaves whose extents its KV cannot empty.
void testDeviceNeedPassesOverLeavesOfFormsOutOfReach() {
  Spilling spill;
  const std::vector<uint32_t> prompt = Spilling::prompt(1000, 6);
  spill.start(1, prompt, 6);
  static_cast<void>(test::spillRound(spill.cache, spill.tier));
  spill.cache.endRequest(1);
  // The chain's Full tail goes for its copy: its last spilled block is a
  // leaf, which the lookup leaves newer than the dead Full leaf.
  require(spill.cache.spillSnapshot().spilledPages == 4 &&
              test::evictStep(spill.cache).gave(),
          "the idle spilled chain was not set up");
  static_cast<void>(cacheChain(spill.cache, 2, 1));
  static_cast<void>(spill.cache.lookup(prompt, {}));
  static_cast<void>(cacheChain(spill.cache, 3, spill.pool.freePageCount(), true));
  require(spill.pool.freePageCount() == 0 && spill.cache.reusablePages(ReclaimClass::Ordinary) == 2 &&
              spill.cache.reusablePages(ReclaimClass::Ordinary, kv::PageForm::Index) == 8 &&
              spill.pool.freePageCount(kv::PageForm::Index) == 4,
          "fixture geometry changed");
  const uint64_t extent = 4 * spill.storage.bytesPerPage(kv::PageForm::Index);
  const ClaimSpec claim{.cls = ClaimClass::InService};
  const VictimQuery query = test::query(claim, {.bytes = extent});
  const std::optional<engine::Victim> unit = spill.cache.peek(query, nullptr);
  const uint64_t allocated = spill.pool.allocatedBytes();
  require(unit && unit->bytes == extent && spill.cache.take(query, *unit).outcome == FreeOutcome::Met &&
              allocated - spill.pool.allocatedBytes() == extent && spill.pool.freePageCount() == 0,
          "a device need took a leaf of a form no extent of which its KV could empty");
  spill.cache.checkInvariants();
  spill.cache.endRequest(3);
}

// A request short of a page waits for the one on its way back rather than demote more, and so does
// one ranked below a claim that takes that page first: it waits as that hold ends
// (MemoryLedger::free), taking nothing.
void testARefusedRequestWaitsForThePageOnItsWayBack() {
  test::TestKvStorage storage{24, 100, 1};
  KvPool pool{storage, 16};
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  std::vector<uint32_t> prompt(KvCache::pageTokens + 1);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, prompt.size()).granted(), "the prompt's pages were denied");
  cache.publishCommittedBlocks(1, prompt, KvCache::pageTokens, {});
  cache.publishCompositeState(cache.blockAt(1, KvCache::pageTokens), std::make_shared<TieredState>(control), false);
  cache.endRequest(1);
  require(reclaimStateBytes(cache) == 100 && cache.pollTransfers() &&
              test::evictStep(cache).gave() &&
              cache.transfersInFlight() && tier.demotions == 1,
          "the leaf's demotion did not start");
  storage.growthBlocked = true;
  const uint32_t free = pool.freePageCount();
  cache.beginRequest(2);
  require(!cache.ensureTokens(2, (free + 1) * KvCache::pageTokens).granted(), "the fixture's growth was not refused");
  for (const bool heldAbove : {false, true}) {
    engine::MemoryLedger ledger;
    ledger.addSource(cache);
    ledger.setReturning([&cache] { return cache.returning(); },
                        [&cache](std::span<const uint64_t> requestIds) { return cache.returningFor(requestIds); });
    if (heldAbove) {
      const engine::ClaimId above = ledger.open({.cls = ClaimClass::InService, .requestId = 3, .rank = {.priority = 2}});
      require(ledger.free(above, {.pages = 1}, 0, MemoryPressure::Normal).outcome == FreeOutcome::Pending,
              "the claim above did not wait for the page on its way back");
    }
    const engine::ClaimId below = ledger.open({.cls = ClaimClass::InService, .requestId = 2, .rank = {.priority = 1}});
    const FreeOutcome outcome = ledger.free(below, {.pages = 1}, 0, MemoryPressure::Normal).outcome;
    require(outcome == FreeOutcome::Pending && tier.demotions == 1,
            heldAbove ? "a request below a claim that takes the page first did not wait as its hold ends"
                      : "a request did not wait for the page on its way back");
  }
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  cache.endRequest(2);
}

// A state that needs a write frees now while no write holds the staging buffer, which its write
// copies it into: claims that cannot wait take it as claims that can do, starting the write. While
// that write is in flight the next such state is offered in its place marked later: a claim that
// cannot wait passes over it to the newer states, and one that can waits for it, holding them back.
// A leaf that needs a demotion is offered in its place marked later too.
// The staging buffer writes one state at a time, so a state need counts one state that needs a write as freeing
// now, not every one: a claim of two buffers that two such states would give is short before it starts a write
// it cannot use.
void testAStateNeedCountsTheStagingBufferOnce() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  const ClaimSpec claim{.cls = ClaimClass::InService, .timing = NeedTiming::Immediate};
  const VictimQuery query = test::query(claim, {.stateBuffers = 2});
  uint32_t now = 0;
  for (std::optional<engine::Victim> victim = fixture.cache.peek(query, nullptr); victim;
       victim = fixture.cache.peek(query, &*victim))
    now += victim->later ? 0 : victim->stateBuffers;
  require(now == 1, "the peeks counted " + std::to_string(now) + " states freeing now through one staging buffer");
  require(test::freeFromCache(fixture.cache, claim, {.stateBuffers = 2}).outcome == FreeOutcome::Short &&
              control->slots == 0 && fixture.cache.stateResident(fixture.blocks[0]) &&
              fixture.cache.stateResident(fixture.blocks[1]),
          "a claim the staging buffer cannot meet started a write it could not use");
}

void testImmediateWalksOfferWhatLandsLaterInItsPlace() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  // A state the tier takes no write of, which goes at once, and the resume
  // point.
  fixture.publish(2);
  fixture.publish(3);
  const auto unitsOf = [&](ClaimClass cls, NeedTiming timing) {
    const ClaimSpec claim{.cls = cls, .timing = timing};
    const VictimQuery query = test::query(claim, {.stateBuffers = 1});
    std::vector<engine::Victim> units;
    for (std::optional<engine::Victim> victim = fixture.cache.peek(query, nullptr); victim;
         victim = fixture.cache.peek(query, &*victim))
      units.push_back(*victim);
    return units;
  };
  const auto blockOf = [](const engine::Victim &victim) { return victim.key & ((uint64_t{1} << 56) - 1); };
  const auto same = [](const engine::Victim &left, const engine::Victim &right) {
    return left.key == right.key && left.write == right.write && left.later == right.later &&
           left.stateBuffers == right.stateBuffers && left.pages == right.pages;
  };
  // The staging buffer writes one state at a time: the first that needs a
  // write frees now through it, the next waits for it, which a claim that
  // cannot wait passes over and one that can is held back behind.
  for (const ClaimClass cls : {ClaimClass::InService, ClaimClass::Growth}) {
    const std::vector<engine::Victim> atOnce = unitsOf(cls, NeedTiming::Immediate);
    require(atOnce.size() >= 3 && blockOf(atOnce[0]) == fixture.blocks[0] && atOnce[0].write &&
                !atOnce[0].later && atOnce[0].stateBuffers == 1 && blockOf(atOnce[1]) == fixture.blocks[1] &&
                atOnce[1].write && atOnce[1].later && !atOnce[1].stateBuffers &&
                blockOf(atOnce[2]) == fixture.blocks[2] && !atOnce[2].write && !atOnce[2].later,
            "the first state that needs a write was not offered as freeing now through the staging buffer, or "
            "the next as waiting for it");
    const std::vector<engine::Victim> canWait = unitsOf(cls, NeedTiming::CanWait);
    require(canWait.size() == 2 && same(canWait[0], atOnce[0]) && same(canWait[1], atOnce[1]),
            "a claim that can wait was not held back behind the state that waits for the staging buffer");
  }
  const ClaimSpec cannotWait{.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate};
  require(test::freeFromCache(fixture.cache, cannotWait, {.stateBuffers = 1}).outcome == FreeOutcome::Met &&
              control->slots == 1 && !fixture.cache.stateResident(fixture.blocks[0]) &&
              fixture.cache.stateResident(fixture.blocks[1]) && fixture.cache.stateResident(fixture.blocks[2]),
          "a claim that cannot wait did not write the oldest state through the idle staging buffer");

  // That write holds the staging buffer: the next state waits for it.
  const std::vector<engine::Victim> busy = unitsOf(ClaimClass::Growth, NeedTiming::Immediate);
  require(busy.size() == 2 && blockOf(busy[0]) == fixture.blocks[1] && busy[0].write && busy[0].later &&
              !busy[0].stateBuffers && blockOf(busy[1]) == fixture.blocks[2] && !busy[1].later,
          "a claim that cannot wait was not offered the state that waits for the staging buffer marked later, "
          "then the newer state");
  const std::vector<engine::Victim> waits = unitsOf(ClaimClass::Growth, NeedTiming::CanWait);
  require(waits.size() == 1 && blockOf(waits[0]) == fixture.blocks[1] && waits[0].later,
          "a claim that can wait was offered the newer states past the state that waits for the staging buffer");
  require(fixture.cache.take(test::query(cannotWait, {.stateBuffers = 1}), busy[0]).outcome == FreeOutcome::Short &&
              fixture.cache.take(test::query({.cls = ClaimClass::Growth}, {.stateBuffers = 1}), waits[0]).outcome ==
                  FreeOutcome::Pending &&
              fixture.cache.stateResident(fixture.blocks[1]) && control->slots == 1,
          "a claim that cannot wait waited for the staging buffer, or one that can did not");
  require(test::freeFromCache(fixture.cache, cannotWait, {.stateBuffers = 1}).outcome == FreeOutcome::Met &&
              fixture.cache.stateResident(fixture.blocks[1]) && !fixture.cache.stateResident(fixture.blocks[2]) &&
              control->slots == 1,
          "a claim that cannot wait did not pass over the state that waits for the staging buffer");

  // The write lands, and the staging buffer takes the next.
  require(fixture.cache.pollTransfers() &&
              test::freeFromCache(fixture.cache, cannotWait, {.stateBuffers = 1}).outcome == FreeOutcome::Met &&
              !fixture.cache.stateResident(fixture.blocks[1]) && control->slots == 2 &&
              fixture.cache.stateResident(fixture.blocks[3]),
          "the staging buffer did not take the next state once the write landed, or growth took the resume point");
  fixture.cache.checkInvariants();

  // The leaf a state on disk restores through, without a copy while the
  // tier takes writes: the unit of its demotion, which the tier and the
  // quota have room for, frees its page as the copy lands.
  test::TestKvTier tier;
  CacheFixture kv(&tier, false, std::make_shared<model::DiskBudget>(1000));
  auto written = std::make_shared<TransferControl>();
  written->ready = true;
  kv.cache.publishCompositeState(kv.blocks[3], std::make_shared<TieredState>(written), false);
  require(test::stateStep(kv.cache).gave() && !kv.cache.stateResident(kv.blocks[3]),
          "the state was not written");
  static_cast<void>(kv.cache.pollTransfers());
  const auto leavesOf = [&](NeedTiming timing) {
    const ClaimSpec claim{.cls = ClaimClass::InService, .timing = timing};
    const VictimQuery pages = test::query(claim, {.pages = 1});
    std::vector<engine::Victim> units;
    for (std::optional<engine::Victim> victim = kv.cache.peek(pages, nullptr); victim;
         victim = kv.cache.peek(pages, &*victim))
      units.push_back(*victim);
    return units;
  };
  const std::vector<engine::Victim> demoted = leavesOf(NeedTiming::Immediate);
  require(demoted.size() == 1 && demoted[0].later && demoted[0].pages,
          "a claim that cannot wait was not offered the leaf that needs a demotion in its place, marked later");
  require(test::freeFromCache(kv.cache, test::kInServiceGrowth, {.pages = 1}).outcome == FreeOutcome::Pending &&
              tier.demotions == 1,
          "a claim that can wait did not demote the leaf");
}

// An Immediate device need takes what frees with no write first: it passes over the checkpoint that
// needs a write and takes the leaves after it, writing nothing while they meet it.
void testImmediateDeviceWalkPassesOverWhatAWriteFrees() {
  test::TestKvStorage storage{48, 100, 4};
  KvPool pool{storage, 4};
  engine::Cache cache(pool, nullptr, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const auto cachePrompt = [&](uint64_t id, uint32_t pages) {
    std::vector<uint32_t> prompt(pages * KvCache::pageTokens + 1);
    std::iota(prompt.begin(), prompt.end(), 1000 * id);
    cache.beginRequest(id);
    require(admitTokens(cache, id, prompt.size()).granted(), "a conversation's pages were denied");
    cache.publishCommittedBlocks(id, prompt, pages * KvCache::pageTokens, {});
    const std::array blocks{cache.blockAt(id, pages / 2 * KvCache::pageTokens),
                            cache.blockAt(id, pages * KvCache::pageTokens)};
    cache.endRequest(id);
    return blocks;
  };
  const std::array<uint64_t, 2> oldest = cachePrompt(1, 8);
  cache.publishCompositeState(oldest[0], std::make_shared<TieredState>(control), true);
  cache.publishCompositeState(oldest[1], std::make_shared<TestState>(100), false);
  for (uint64_t id = 2; id <= 5; ++id)
    static_cast<void>(cachePrompt(id, 4));
  static_cast<void>(cache.releaseEmptyExtents(false));
  const ClaimSpec claim{.cls = ClaimClass::Optional, .requestId = 1, .timing = NeedTiming::Immediate};
  const uint64_t allocated = pool.allocatedBytes();
  require(test::freeFromCache(cache, claim, {.bytes = 2 * 4 * 100}).outcome == FreeOutcome::Met,
          "an Immediate claim did not take the leaves after the checkpoint");
  const uint64_t released = allocated - pool.allocatedBytes();
  require(released && !control->slots && !cache.snapshot().stateCache.offloads && cache.stateResident(oldest[0]),
          "the units that go at once wrote the checkpoint");
  cache.checkInvariants();
}

// A device need's walk offers the state the idle staging buffer writes as a unit of its own, after
// what frees with no write: it frees now, and a claim that cannot wait takes it, starting the
// write. While that write is in flight the walk offers no such unit, and the next state that needs
// a write is among what frees as writes land.
void testDeviceWalkOffersTheStagedWriteAsAUnit() {
  CacheFixture fixture;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  fixture.cache.publishCompositeState(fixture.blocks[0], std::make_shared<TieredState>(control), false);
  fixture.cache.publishCompositeState(fixture.blocks[1], std::make_shared<TieredState>(control), false);
  // The resume point, which growth keeps, and with it the chain.
  fixture.publish(3);
  const ClaimSpec cannotWait{.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate};
  const VictimQuery query = test::query(cannotWait, {.bytes = 1});
  const auto units = [&] {
    std::vector<engine::Victim> offered;
    for (std::optional<engine::Victim> unit = fixture.cache.peek(query, nullptr); unit;
         unit = fixture.cache.peek(query, &*unit))
      offered.push_back(*unit);
    return offered;
  };
  const std::vector<engine::Victim> idle = units();
  require(idle.size() == 2 && idle[0].write && !idle[0].later && idle[0].stateBuffers == 1 && !idle[0].bytes &&
              idle[1].write && idle[1].later && idle[1].stateBuffers == 1,
          "the walk did not offer one state the idle staging buffer writes, then the one that waits for it");
  const FreeResult taken = fixture.cache.take(query, idle[0]);
  require(taken.outcome == FreeOutcome::Met && taken.stateBuffers == 1 && control->slots == 1 &&
              !fixture.cache.stateResident(fixture.blocks[0]) && fixture.cache.stateResident(fixture.blocks[1]),
          "a claim that cannot wait did not write the oldest state through the idle staging buffer");
  const std::vector<engine::Victim> busy = units();
  require(busy.size() == 1 && busy[0].later && busy[0].stateBuffers == 1,
          "the walk offered a state the staging buffer writes while a write held it");
  require(fixture.cache.pollTransfers(), "the state write did not land");
  const std::vector<engine::Victim> landed = units();
  require(landed.size() == 1 && landed[0].write && !landed[0].later && landed[0].stateBuffers == 1,
          "the walk did not offer the next state once the write landed");
  fixture.cache.checkInvariants();
}

// Caches a conversation of `pages` pages through a finished request, with
// an ordinary state on its last block when one is given; returns that
// block.
uint64_t cacheConversation(engine::Cache &cache, uint64_t id, uint32_t pages,
                           std::shared_ptr<const CompositeState> state = nullptr) {
  std::vector<uint32_t> prompt(pages * KvCache::pageTokens + 1);
  std::iota(prompt.begin(), prompt.end(), 1000 * id);
  cache.beginRequest(id);
  require(admitTokens(cache, id, prompt.size()).granted(), "a conversation's pages were denied");
  cache.publishCommittedBlocks(id, prompt, pages * KvCache::pageTokens, {});
  const uint64_t last = cache.blockAt(id, pages * KvCache::pageTokens);
  if (state)
    cache.publishCompositeState(last, std::move(state), false);
  cache.endRequest(id);
  return last;
}

// Counting what a reclaim frees costs as much in a cache of many conversations as in one of few: every phase of a
// page need and a device need, of what is in use too, beside states that need a write, a state in use and a waiting
// request's chain, looks at the same blocks, states and request pages (Cache::reachVisits) and gives the same units.
// So does it in a windowed cache that trims, beside a running request's window (the Trim unit and trimmableCount),
// and in a spilling one (the Extent unit), where a page need counts the other forms' empty extents first.
void testReclaimCountsDoNotGrowWithTheCache() {
  enum class Kind { Plain, Windowed, Spilling };
  const auto count = [](uint32_t conversations, Kind kind) {
    const uint32_t pages = conversations * 4 + 16;
    test::TestKvStorage storage{pages, 100, 4, kind == Kind::Spilling ? 30u : 0u, kind == Kind::Windowed ? 30u : 0u};
    storage.budgetPages = pages;
    KvPool pool{storage, 4};
    test::TestKvTier tier;
    auto budget = std::make_shared<model::DiskBudget>(uint64_t{pages} * 100);
    tier.budget = budget;
    engine::Cache cache(pool, &tier, budget,
                        kind == Kind::Spilling ? std::optional(CacheSpillPolicy{1, 1}) : std::nullopt,
                        kind == Kind::Windowed ? std::optional(CacheWindowPolicy{1}) : std::nullopt);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    control->capacity = conversations;
    std::vector<uint64_t> last;
    for (uint64_t id = 1; id <= conversations; ++id)
      last.push_back(cacheConversation(cache, id, 4, std::make_shared<TieredState>(control)));
    const StateUse use = cache.useState(last[1]);
    std::vector<uint32_t> prompt(4 * KvCache::pageTokens + 1);
    std::iota(prompt.begin(), prompt.end(), 1000 * conversations);
    const CacheProbe waiting = cache.probe(prompt, {});
    cache.setWantedChains({waiting.restoredChain()});
    // A running request whose window the counts of a windowed cache read.
    const uint64_t running = conversations + 1;
    cache.beginRequest(running);
    require(admitTokens(cache, running, 3 * KvCache::pageTokens).granted(), "the running request's pages were denied");
    const uint64_t before = cache.reachVisits();
    std::vector<engine::Victim> units;
    for (const VictimQuery &query : {test::query(test::kInServiceGrowth, {.pages = 1}),
                                     test::query(test::kInServiceGrowth, {.bytes = 1, .stateBuffers = 1})})
      for (std::optional<engine::Victim> unit = cache.peek(query, nullptr); unit; unit = cache.peek(query, &*unit))
        units.push_back(*unit);
    const uint64_t visits = cache.reachVisits() - before;
    cache.endRequest(running);
    cache.checkInvariants();
    return std::pair{visits, units};
  };
  for (const Kind kind : {Kind::Plain, Kind::Windowed, Kind::Spilling}) {
    const std::string name = kind == Kind::Plain ? "plain" : kind == Kind::Windowed ? "windowed" : "spilling";
    const auto [fewVisits, few] = count(8, kind);
    const auto [manyVisits, many] = count(32, kind);
    require(fewVisits == manyVisits && few.size() == many.size() && few.size() >= 4,
            "counting a " + name + " cache of 32 conversations looked at " + std::to_string(manyVisits) +
                " blocks, states and request pages, one of 8 " + std::to_string(fewVisits) + " (" +
                std::to_string(many.size()) + " and " + std::to_string(few.size()) + " units)");
    for (size_t index = 0; index < few.size(); ++index)
      require(few[index].key == many[index].key, "the larger " + name + " cache gave other units");
  }
}

// A take costs the leaves it tries, not the cache. In a cache of many conversations as in one of few, with a state
// in use and a waiting request's chain, claims through the ledger are each given as many leaves (Cache::leafVisits)
// and free the same: of a page and of an extent's bytes beside states that need a write, which go before their
// leaves; once the states are on the SSD, of a page, in a page scan and not, that cannot wait for a write, beside a
// newer copied chain and older chains without a copy; then beside a newer chain no state needs; and as the leaves
// are demoted and the transfers land.
void testTakesDoNotGrowWithTheCache() {
  struct Take final {
    uint64_t leaves = 0;
    FreeResult freed;
  };
  const auto run = [](uint32_t conversations) {
    test::TestKvStorage storage{conversations * 4 + 8, 100, 4};
    storage.budgetPages = conversations * 4 + 8;
    KvPool pool{storage, 4};
    test::TestKvTier tier;
    tier.capacity = conversations * 4;
    engine::Cache cache(pool, &tier, nullptr);
    auto control = std::make_shared<TransferControl>();
    control->ready = true;
    control->capacity = conversations;
    std::vector<uint64_t> last;
    for (uint64_t id = 1; id <= conversations; ++id)
      last.push_back(cacheConversation(cache, id, 4, std::make_shared<TieredState>(control)));
    const StateUse use = cache.useState(last[1]);
    std::vector<uint32_t> prompt(4 * KvCache::pageTokens + 1);
    std::iota(prompt.begin(), prompt.end(), 1000 * conversations);
    const CacheProbe waiting = cache.probe(prompt, {});
    cache.setWantedChains({waiting.restoredChain()});
    const ClaimSpec cannotWait{.cls = ClaimClass::InService, .reach = ReclaimClass::InUse,
                               .timing = NeedTiming::Immediate};
    std::vector<Take> takes;
    const auto claim = [&](const ClaimSpec &spec, const Need &need) {
      const uint64_t before = cache.leafVisits();
      const FreeResult freed = test::freeFromCache(cache, spec, need);
      takes.push_back({cache.leafVisits() - before, freed});
      tier.complete();
      static_cast<void>(cache.pollTransfers());
      cache.checkInvariants();
    };
    claim(test::kInServiceGrowth, {.pages = 1});
    claim(test::kInServiceGrowth, {.bytes = 1});
    while (test::stateStep(cache, ReclaimClass::InUse).gave())
      static_cast<void>(cache.pollTransfers());
    for (PersistStatus status; (status = cache.persist(last[conversations - 2])) != PersistStatus::Durable;) {
      require(status == PersistStatus::Started, "a newer chain was not copied ahead");
      tier.complete();
      static_cast<void>(cache.pollTransfers());
    }
    claim(cannotWait, {.pages = 1});
    claim(test::kInServiceGrowth, {.hostRefused = true, .pages = 1});
    cacheConversation(cache, conversations + 1, 4);
    claim(cannotWait, {.pages = 1});
    for (uint32_t round = 0; round < 4; ++round) {
      claim(test::kInServiceGrowth, {.pages = 1});
      claim(test::kInServiceGrowth, {.bytes = 1});
    }
    return takes;
  };
  const std::vector<Take> few = run(8);
  const std::vector<Take> many = run(32);
  uint64_t leaves = 0;
  for (size_t index = 0; index < few.size(); ++index) {
    require(few[index].leaves == many[index].leaves,
            "take " + std::to_string(index) + " in a cache of 32 conversations was given " +
                std::to_string(many[index].leaves) + " leaves, in one of 8 " + std::to_string(few[index].leaves));
    require(few[index].freed.outcome == many[index].freed.outcome && few[index].freed.pages == many[index].freed.pages &&
                few[index].freed.stateBuffers == many[index].freed.stateBuffers,
            "the larger cache's take " + std::to_string(index) + " freed something else");
    leaves += few[index].leaves;
  }
  require(leaves > 0, "no take was given a leaf");
}

// A resident leaf waits among the leaves of what is in use while a state below it, on the SSD at a block that left
// RAM, is in use, and among the others again once the use ends (checkInvariants checks its kind).
void testLeafFollowsTheUseOfAStateBelowIt() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const uint64_t last = cacheConversation(cache, 1, 2, std::make_shared<TieredState>(control));
  require(test::stateStep(cache).gave() && cache.pollTransfers(), "the state was not written to the SSD");
  require(test::evictStep(cache).gave(), "the leaf under the state was not demoted");
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  require(!cache.residentChainPages({&last, 1}) && cache.snapshot().stateCache.entries == 1 &&
              cache.snapshot().stateCache.diskBytes,
          "the state's block did not leave RAM with its state on the SSD");
  cache.checkInvariants();
  {
    const StateUse use = cache.useState(last);
    cache.checkInvariants();
  }
  cache.checkInvariants();
}

// The pages a need's units count at most (peek), and those its takes free,
// of the need's form, now or as the demotions they start land, taken one at
// a time until one takes nothing.
std::pair<uint32_t, uint32_t> countAndTakePages(engine::Cache &cache, KvPool &pool, const VictimQuery &query) {
  uint32_t counted = 0;
  for (std::optional<engine::Victim> victim = cache.peek(query, nullptr); victim;
       victim = cache.peek(query, &*victim))
    counted += victim->form == query.need.form ? victim->pages : 0;
  const auto pages = [&] {
    return pool.freePageCount(query.need.form) +
           cache.returning().landingPages[static_cast<size_t>(query.need.form)];
  };
  const uint32_t before = pages();
  for (std::optional<engine::Victim> unit = cache.peek(query, nullptr);
       unit && cache.take(query, *unit).outcome != FreeOutcome::Short; unit = cache.peek(query, nullptr)) {
  }
  cache.checkInvariants();
  return {counted, pages() - before};
}

// A page need's walk offers each dead leaf once and never overstates what it frees.
void testPageWalkOffersEachDeadLeafOnce() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 1};
  engine::Cache cache(pool, nullptr, nullptr);
  std::vector<uint32_t> prompt(2 * KvCache::pageTokens + 1);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, prompt.size()).granted(), "the prompt's pages were denied");
  cache.publishCommittedBlocks(1, prompt, 2 * KvCache::pageTokens, {});
  const uint64_t first = cache.blockAt(1, KvCache::pageTokens);
  cache.publishCompositeState(first, std::make_shared<TestState>(100), false);
  cache.endRequest(1);
  const ClaimSpec growth{.cls = ClaimClass::Growth};
  const auto [counted, freed] = countAndTakePages(cache, pool, test::query(growth, {.pages = 2}));
  require(counted == 1 && freed == 1 && cache.stateResident(first),
          "the walk counted " + std::to_string(counted) + " pages, taking its units freed " + std::to_string(freed));
}

// A need the engine's limit refused is a device need: each take releases whole extents, or gives
// up a state.
void testEngineRefusedPageNeedIsADeviceNeed() {
  test::TestKvStorage storage{16, 100, 1};
  KvPool pool{storage, 16};
  engine::Cache cache(pool, nullptr, nullptr);
  for (uint64_t id = 1; id <= 3; ++id)
    static_cast<void>(cacheConversation(cache, id, 2, std::make_shared<TestState>(100)));
  static_cast<void>(cache.releaseEmptyExtents(false));
  const ClaimSpec claim{.cls = ClaimClass::InService};
  uint32_t takes = 0;
  for (FreeResult step; (step = test::takeFirst(cache, claim, {.bytes = 100, .pages = 2})).outcome == FreeOutcome::Met;
       ++takes)
    require(step.nowBytes % 100 == 0 && (step.nowBytes || step.stateBuffers == 1),
            "a take released no whole extent and gave up no state");
  require(takes == 9 && !pool.allocatedBytes() && !cache.snapshot().stateCache.entries,
          "the takes left extents or states");
}

// A page need's walk reaches parents that become leaves and leaves under states; its takes free
// what its units counted.
void testPageNeedReachesParentsAndLeavesUnderStates() {
  const ClaimSpec claim{.cls = ClaimClass::InService};
  for (const bool hostRefused : {false, true}) {
    test::TestKvStorage storage{16, 100, 1};
    KvPool pool{storage, 1};
    engine::Cache cache(pool, nullptr, nullptr);
    static_cast<void>(cacheConversation(cache, 1, 8));
    const auto [counted, freed] =
        countAndTakePages(cache, pool, test::query(claim, {.hostRefused = hostRefused, .pages = 4}));
    require(counted == 8 && freed == 8,
            std::string(hostRefused ? "a page scan" : "a page need") + " of an idle conversation counted " +
                std::to_string(counted) + " pages, freed " + std::to_string(freed));
  }
  for (const bool hostRefused : {false, true}) {
    test::TestKvStorage storage{16, 100, 1};
    KvPool pool{storage, 1};
    engine::Cache cache(pool, nullptr, nullptr);
    for (uint64_t id = 1; id <= 3; ++id)
      static_cast<void>(cacheConversation(cache, id, 2, std::make_shared<TestState>(100)));
    const VictimQuery query = test::query(claim, {.hostRefused = hostRefused, .pages = 6});
    const auto [counted, freed] = countAndTakePages(cache, pool, query);
    require(counted == 6 && freed == 6 && !cache.snapshot().stateCache.entries,
            std::string(hostRefused ? "a page scan" : "a page need") + " under states counted " +
                std::to_string(counted) + " pages, freed " + std::to_string(freed));
  }
}

// A device unit freed only by the demotions it starts is marked later, and take() reports the
// extent its page completes as it lands as later bytes.
void testDeviceUnitReportsWhatItsDemotionsReturn() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  auto budget = std::make_shared<model::DiskBudget>(1000);
  tier.budget = budget;
  engine::Cache cache(pool, &tier, budget);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const uint64_t last = cacheConversation(cache, 1, 4, std::make_shared<TieredState>(control));
  require(test::stateStep(cache).gave() && !cache.stateResident(last),
          "the state was not written");
  static_cast<void>(cache.pollTransfers());
  require(pool.freePageCount() == 1, "the open page's extent is not the one empty extent");
  const ClaimSpec host{.cls = ClaimClass::Host};
  const VictimQuery query = test::query(host, {.bytes = 100});
  const std::optional<engine::Victim> unit = cache.peek(query, nullptr);
  require(unit && unit->later, "the unit that demotes the leaf was not marked later");
  const uint64_t allocated = pool.allocatedBytes();
  const FreeResult taken = cache.take(query, *unit);
  require(taken.outcome == FreeOutcome::Pending && !taken.nowBytes && taken.laterBytes == 100 &&
              pool.allocatedBytes() == allocated && tier.demotions == 1,
          "taking the unit did not start the demotion, or did not say what lands");
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  require(cache.releaseEmptyExtents(true) == 100, "the extent the landed page completed was not released");
}

// One unit of `unitBytes` above an empty floor: a pool that costs only its rebuild (an idle vision
// arena) or one the lanes read through (the expert store).
class UnitPool final : public model::ElasticPool {
public:
  explicit UnitPool(bool rebuildOnly, uint64_t unitBytes = 100) : rebuildOnly_(rebuildOnly), unitBytes_(unitBytes) {}
  [[nodiscard]] model::PoolState state() const override {
    return {.name = "unit", .currentBytes = units * unitBytes_, .capBytes = unitBytes_,
            .releaseUnitBytes = units ? unitBytes_ : 0u, .rebuildOnly = rebuildOnly_};
  }
  [[nodiscard]] uint64_t release(uint64_t) override { return std::exchange(units, 0) * unitBytes_; }
  [[nodiscard]] metal::AllocationResult grow(const metal::AllocationAdmission &) override { return {}; }
  uint64_t units = 1;

private:
  bool rebuildOnly_;
  uint64_t unitBytes_;
};

// A claim of `bytes` that can wait, for in-service work that reaches what is in use.
FreeResult claimBytes(engine::MemoryLedger &ledger, uint64_t bytes) {
  const engine::ClaimId id = ledger.open(test::kInServiceGrowth);
  const FreeResult freed = ledger.free(id, {.bytes = bytes}, 0, MemoryPressure::Normal);
  ledger.close(id);
  return freed;
}

// A claim that can wait takes the units in the one order: the cache that frees now, a pool that
// costs only its rebuild, then the cache that needs a write, and the store after them. A newer
// copied leaf meets an extent's need while an older leaf a state on the SSD needs, without a copy,
// stays; the next need takes the arena, still without a write, and the one after demotes the older
// leaf and leaves the store.
void testClaimTakesCopiedCacheThenRebuiltPoolsThenWrites() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  auto budget = std::make_shared<model::DiskBudget>(1000);
  tier.budget = budget;
  engine::Cache cache(pool, &tier, budget);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const uint64_t older = cacheConversation(cache, 1, 1, std::make_shared<TieredState>(control));
  const uint64_t newer = cacheConversation(cache, 2, 1, std::make_shared<TieredState>(control));
  for (int state = 0; state < 2; ++state)
    require(test::stateStep(cache).gave() && cache.pollTransfers(), "a state was not written to the SSD");
  require(cache.persist(newer) == PersistStatus::Started, "the newer leaf was not copied ahead");
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  static_cast<void>(cache.releaseEmptyExtents(false));
  UnitPool arena(true), store(false);
  engine::MemoryLedger ledger;
  ledger.addSource(cache);
  static_cast<void>(ledger.addPool(arena));
  static_cast<void>(ledger.addPool(store));
  const auto inMemory = [&](uint64_t block) { return cache.residentChainPages({&block, 1}) != 0; };
  const uint64_t demotions = tier.demotions;
  require(claimBytes(ledger, 100).outcome == FreeOutcome::Met && tier.demotions == demotions && !inMemory(newer) &&
              inMemory(older) && arena.units && store.units,
          "a claim that can wait took more than the copied leaf that met its need");
  require(claimBytes(ledger, 100).outcome == FreeOutcome::Met && tier.demotions == demotions && inMemory(older) &&
              !arena.units && store.units,
          "a claim demoted a leaf before it took the pool that costs only its rebuild");
  const FreeResult freed = claimBytes(ledger, 100);
  require(freed.outcome == FreeOutcome::Pending && tier.demotions == demotions + 1 && store.units &&
              freed.laterBytes == 100,
          "a claim took the store before it demoted the older leaf");
  cache.checkInvariants();
}

// A claim that can wait stops at the cache's writes while one waits for the tier, though the
// writes in flight are its own: a leaf whose demotions fill the tier leaves the store in place,
// and the claim waits for them to land.
void testOwnDemotionsHoldBackTheStore() {
  test::TestKvStorage storage{24, 100, 8};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  auto budget = std::make_shared<model::DiskBudget>(10000);
  tier.budget = budget;
  engine::Cache cache(pool, &tier, budget);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const uint64_t leaf = cacheConversation(cache, 1, 16, std::make_shared<TieredState>(control));
  require(test::stateStep(cache).gave() && cache.pollTransfers(), "the state was not written to the SSD");
  static_cast<void>(cache.releaseEmptyExtents(false));
  UnitPool store(false, 800);
  engine::MemoryLedger ledger;
  ledger.addSource(cache);
  ledger.setReturning([&cache] { return cache.returning(); },
                      [&cache](std::span<const uint64_t> requestIds) { return cache.returningFor(requestIds); });
  static_cast<void>(ledger.addPool(store));
  const FreeResult freed = claimBytes(ledger, 800);
  require(freed.outcome == FreeOutcome::Pending && tier.demotions == tier.transferLimit && store.units &&
              cache.residentChainPages({&leaf, 1}),
          "a claim whose own demotions filled the tier took the store");
  cache.checkInvariants();
}

// A memory reclaim demotes only what completes an extent beside the free pages: with half an
// extent free, half an extent of demotions meets an extent's need.
void testDemotionsStopWhereTheyCompleteAnExtent() {
  test::TestKvStorage storage{24, 100, 8};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  tier.transferLimit = 64;
  auto budget = std::make_shared<model::DiskBudget>(10000);
  tier.budget = budget;
  engine::Cache cache(pool, &tier, budget);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  static_cast<void>(cacheConversation(cache, 1, 8, std::make_shared<TieredState>(control)));
  require(test::stateStep(cache).gave() && cache.pollTransfers(), "the state was not written to the SSD");
  // A request holds half of the second extent.
  cache.beginRequest(2);
  require(admitTokens(cache, 2, 4 * KvCache::pageTokens).granted(), "the request's pages were denied");
  static_cast<void>(cache.releaseEmptyExtents(false));
  engine::MemoryLedger ledger;
  ledger.addSource(cache);
  const FreeResult freed = claimBytes(ledger, 800);
  require(freed.outcome == FreeOutcome::Pending && freed.laterBytes == 800 && tier.demotions == 4,
          "a claim demoted " + std::to_string(tier.demotions) + " pages for an extent half free");
  cache.endRequest(2);
  cache.checkInvariants();
}

// A claim that can wait counts the extent a landing page completes beside the free pages as landing;
// a release of those free pages now is not counted again, since the landing then completes none. A
// leaf's demotion lands beside three free pages; a request then holds them, and a dead page beside
// three new free pages completes an extent now: a claim of two extents takes that extent and the
// store, not that extent alone.
void testAReleaseOfWhatALandingCompletesCountsOnce() {
  test::TestKvStorage storage{16, 100, 4};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  auto budget = std::make_shared<model::DiskBudget>(10000);
  tier.budget = budget;
  engine::Cache cache(pool, &tier, budget);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  static_cast<void>(cacheConversation(cache, 1, 1, std::make_shared<TieredState>(control)));
  require(test::stateStep(cache).gave() && cache.pollTransfers(), "the state was not written to the SSD");
  static_cast<void>(cache.releaseEmptyExtents(false));
  UnitPool store(false, 400);
  engine::MemoryLedger ledger;
  ledger.addSource(cache);
  ledger.setReturning([&cache] { return cache.returning(); },
                      [&cache](std::span<const uint64_t> requestIds) { return cache.returningFor(requestIds); });
  static_cast<void>(ledger.addPool(store));
  require(claimBytes(ledger, 100).laterBytes == 400 && store.units, "the leaf's demotion did not complete an extent");
  cache.beginRequest(2);
  require(admitTokens(cache, 2, 3 * KvCache::pageTokens).granted(), "the request's pages were denied");
  static_cast<void>(cacheConversation(cache, 3, 1));
  require(pool.freePageCount() == 3 && cache.returning().landingBytes == 400, "fixture geometry changed");
  static_cast<void>(claimBytes(ledger, 800));
  require(pool.freePageCount() == 0 && !store.units,
          "a claim counted the free pages a landing completes an extent with twice");
  cache.endRequest(2);
  cache.checkInvariants();
}

// A claim that can wait counts the landing a take of its first unit lets free. A leaf's demotion
// lands beside three free pages; a state then published in RAM on the leaf keeps its page as it
// lands, so the landing completes nothing. Taking that state, with no write, lets it complete the
// extent, which the take credits: the unit offered for the state counts that extent.
void testATakeCountsTheLandingItLetsFree() {
  test::TestKvStorage storage{16, 100, 4};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  auto budget = std::make_shared<model::DiskBudget>(10000);
  tier.budget = budget;
  engine::Cache cache(pool, &tier, budget);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const uint64_t leaf = cacheConversation(cache, 1, 1, std::make_shared<TieredState>(control));
  require(test::stateStep(cache).gave() && cache.pollTransfers(), "the state was not written to the SSD");
  static_cast<void>(cache.releaseEmptyExtents(false));
  require(test::freeFromCache(cache, test::kInServiceGrowth, {.bytes = 100}).laterBytes == 400,
          "the leaf's demotion did not complete an extent");
  cache.publishCompositeState(leaf, std::make_shared<TestState>(100), false);
  require(pool.freePageCount() == 3 && tier.demotions == 1 && cache.returning().landingBytes == 0,
          "fixture geometry changed");
  const VictimQuery query = test::query(test::kInServiceGrowth, {.bytes = 400, .stateBuffers = 1});
  const std::optional<engine::Victim> unit = cache.peek(query, nullptr);
  require(unit && !unit->write && !unit->later && unit->stateBuffers == 1, "the state was not offered first");
  const FreeResult taken = cache.take(query, *unit);
  require(!cache.stateResident(leaf) && taken.laterBytes == 400 && cache.returning().landingBytes == 400,
          "taking the state did not let the landing complete the extent");
  require(unit->bytes >= taken.nowBytes + taken.laterBytes,
          "the unit counted " + std::to_string(unit->bytes) + " bytes; its take freed " +
              std::to_string(taken.nowBytes + taken.laterBytes));
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  cache.checkInvariants();
}

// Growth counts no page a copy in flight holds: the copy frees nothing until it lands, so the dead
// leaf beside it, which alone cannot fund the unit, stays.
void testGrowthCountsNoPageInTransfer() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  tier.transferLimit = 64;
  auto budget = std::make_shared<model::DiskBudget>(10000);
  tier.budget = budget;
  engine::Cache cache(pool, &tier, budget);
  tier.capacity = 16;
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  const uint64_t dead = cacheConversation(cache, 1, 1);
  const uint64_t copied = cacheConversation(cache, 2, 3, std::make_shared<TieredState>(control));
  require(test::stateStep(cache).gave() && cache.pollTransfers(), "the state was not written to the SSD");
  require(cache.persist(copied) == PersistStatus::Started, "the chain was not copied ahead");
  static_cast<void>(cache.releaseEmptyExtents(false));
  engine::MemoryLedger ledger;
  ledger.addSource(cache);
  const engine::ClaimId claim = ledger.open({.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate});
  const FreeResult freed = ledger.free(claim, {.bytes = 300}, 0, MemoryPressure::Normal);
  ledger.close(claim);
  require(freed.outcome == FreeOutcome::Short && !freed.nowBytes && cache.residentChainPages({&dead, 1}),
          "growth took the dead leaf for a unit only copies in flight could fund");
}

// Growth takes nothing when what the cache frees now cannot fund its unit: a state whose write
// waits for the one in flight counts for nothing, so the dead chain beside it stays (one of its two
// extents would go, the other is the runway growth keeps).
void testGrowthCountsNoStateThatWaitsForAWrite() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  const uint64_t dead = cacheConversation(cache, 1, 2);
  static_cast<void>(cacheConversation(cache, 2, 1, std::make_shared<TieredState>(control)));
  // The resume point, which growth keeps, copied ahead: its write holds the
  // staging buffer.
  const uint64_t point = cacheConversation(cache, 3, 1, std::make_shared<TieredState>(control));
  require(cache.persist(point) == PersistStatus::Started && control->slots == 1, "the resume point was not copied");
  static_cast<void>(cache.releaseEmptyExtents(false));
  engine::MemoryLedger ledger;
  ledger.addSource(cache);
  ledger.setBufferBytes([&control](uint32_t states, bool, bool) { return control->poolBytes(states); });
  const engine::ClaimId claim = ledger.open({.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate});
  const FreeResult freed = ledger.free(claim, {.bytes = 200}, 0, MemoryPressure::Normal);
  ledger.close(claim);
  require(freed.outcome == FreeOutcome::Short && !freed.nowBytes && cache.residentChainPages({&dead, 1}),
          "growth took the dead leaf for a unit only a state's write could fund");
}

// The state the idle staging buffer writes lets go of the copied blocks of its chain as it leaves
// RAM: the walk counts them with it, and growth that cannot wait takes the state, starting its
// write, then those blocks, with no demotion.
void testStagedStateFreesItsCopiedChain() {
  test::TestKvStorage storage{8, 100, 1};
  KvPool pool{storage, 1};
  test::TestKvTier tier;
  tier.transferLimit = 64;
  tier.capacity = 16;
  engine::Cache cache(pool, &tier, nullptr);
  auto control = std::make_shared<TransferControl>();
  control->ready = true;
  control->capacity = 4;
  // A conversation of three blocks, its point at the second and the one it
  // went on to at the third, which write-behind copied with its chain.
  std::vector<uint32_t> prompt(3 * KvCache::pageTokens + 1);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, prompt.size()).granted(), "the conversation's pages were denied");
  cache.publishCommittedBlocks(1, prompt, 3 * KvCache::pageTokens, {});
  const uint64_t earlier = cache.blockAt(1, 2 * KvCache::pageTokens);
  const uint64_t newest = cache.blockAt(1, 3 * KvCache::pageTokens);
  cache.endRequest(1);
  cache.publishCompositeState(earlier, std::make_shared<TieredState>(control), false);
  cache.publishCompositeState(newest, std::make_shared<TieredState>(control), false);
  PersistStatus status = PersistStatus::Started;
  for (int step = 0; step < 16 && status != PersistStatus::Durable; ++step) {
    status = cache.persist(newest);
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  require(status == PersistStatus::Durable && tier.demotions == 3, "the newest point was not copied with its chain");
  // The resume point, which growth keeps, on a block of its own.
  static_cast<void>(cacheConversation(cache, 2, 1, std::make_shared<TestState>(100)));
  static_cast<void>(cache.releaseEmptyExtents(false));
  engine::MemoryLedger ledger;
  ledger.addSource(cache);
  ledger.setBufferBytes([&control](uint32_t states, bool, bool) { return control->poolBytes(states); });
  const engine::ClaimId claim = ledger.open({.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate});
  // Both states' buffers and two of the chain's three extents, the third
  // the runway growth keeps.
  const FreeResult freed = ledger.free(claim, {.bytes = 400}, 0, MemoryPressure::Normal);
  ledger.close(claim);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 200 && !cache.stateResident(earlier) &&
              cache.snapshot().stateCache.offloads == 2 && tier.demotions == 3,
          "growth did not take the state the staging buffer writes and the copied chain it frees");
  cache.checkInvariants();
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
    testInUseGoesWhenWhatLandsReturnsNoPage();
    testDiskReplacementTakesStatesInUseLast();
    testPageShortageTakesLeavesInUseLast();
    testKvAStateInUseNeedsNeverPinsMemory();
    testOrdinaryDemotionKeepsCopiesInUse();
    testPromotionSkipsForAStateInUse();
    testInUseEvictionsCountChoices();
    testDeadKvGoesBeforeOlderStates();
    testPendingDemotionLeavesOtherKvOpen();
    testLargeSharedDiskRestore();
    testRefusedRestoresStopAtTheFirstRefusal();
    testPageReuseTakesKvBeforeStates();
    testLookupKeepsADiskChainsResidentBoundaryWarm();
    testRestoreKeepsTheBlockItExtends();
    testDemotionKeepsThePageUnderANewState();
    testAStateInRamSitsOnAPage();
    testCancelledRestoreKeepsThePageUnderANewState();
    testBusyTierPreservesDiskVictim();
    testCancelledRestoreStopsQueuedReads();
    testDiskCheckpointRamAccounting();
    testOrdinaryPublicationUpgradesDiskCheckpoint();
    testKvDemotionAndRestoreLifecycle();
    testEvictedCheckpointsAndSupersededPointsStayOnTheSsd();
    testTailsDropAndParentsFollowToDisk();
    testDiskCopiesNoStateNeedsGoWithTheLeaf();
    testRefusedDemotionKeepsTheLeafWhileTransfersLand();
    testUnusableTierDropsTheLeafInstead();
    testSpentWriteBudgetHoldsBackStatesNotInUse();
    testSpentWriteBudgetKeepsTheChainsOfWrittenStates();
    testLimitBelowASlotMakesNoRoom();
    testSecondStateWaitsForTheWrite();
    testWaitingCheckpointHoldsBackNothingElse();
    testFullTierStopsTheScan();
    testRestoresInFlightReturnNothingToAShortfall();
    testParentGoesWithItsDiskChildrenWhenNoWriteCanStart();
    testPendingPagesGateAllocation();
    testTransferFailures();
    testFailedRestoreDropsTheBlocksBelow();
    testDemotionCostsNoSecondState();
    testPromotionIdentityAndDenial();
    testPromotionNeverDropsAUniqueState();
    testRepublicationKeepsTheDiskCopy();
    testLostStatesAreCounted();
    testDiskReplacementSpansStatesAndKv();
    testDiskPublicationLifecycle();
    testDiskPublicationFailure();
    testFailedWriteUnderALookup();
    testInvalidatedStateUnderALookup();
    testFailedWriteIsCountedAfterItsEntryLeft();
    testDiskPublicationMakesRoom();
    testRollingCheckpointsUseTheTier();
    testDiskQuotaKeepsTheOnlyCopyAQueuedRequestRestores();
    testCertainReuseByClass();
    testAFullTemporaryQuotaStillDemotesByReplacingItsOldestCopy();
    testALeafWhoseWriteFindsNoSlotGoesWithItsSubtree();
    testAWaitingClaimDemotesAWholeChainRunAtOnce();
    testAMemoryClaimForAPageTakesACachedPageInPlace();
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
    testCheckpointPinsAndBoundaryUpgrade();
    testCheckpointPressurePreservesHotPrefix();
    testSchedulingProbeDoesNotChangeCachePolicy();
    testValidAdmissionProbePreservesLookupAndAccounting();
    testProbeRefreshFollowsTheGraph();
    testProbeRechecksStateChanges();
    testProbeFallsBackWhenKvChanges();
    testProbeBindsImageIdentity();
    testCacheSaltSeparatesPrompts();
    testProbeCannotCrossCaches();
    testPageTableReportsWhatChanged();
    testNewPagesTheWorkDoesNotFillComeCleared();
    testCacheLookupAndOneTokenReplay();
    testLazyJunctionMaterialization();
    testLazyJunctionOnlyAtABranchPoint();
    testByteLruAndPins();
    testCheckpointDoesNotOutrankTheResumePoint();
    testKvEvictionInvalidatesStateFirst();
    testStatePublicationValidation();
    testDuplicateProbePromotesStateWithoutLookupAccounting();
    testUnifiedRecencyAndReleasedByteAccounting();
    testFinishedRequestLeavesTailKvBeforeItsState();
    testReclaimCountsDoNotGrowWithTheCache();
    testTakesDoNotGrowWithTheCache();
    testLeafFollowsTheUseOfAStateBelowIt();
    testCompactionReturnsFreePagesBeforeEvicting();
    testCompactionFollowsOnlyTheMovedBlocks();
    testCompactionLeavesTheRunway();
    testCompactionLeavesAPageBeingRestored();
    testCompactionLeavesAPageBeingDemoted();
    testReplacementKeepsTheExtentItEmpties();
    testSpillMovesLandedPagesToTheIndexForm();
    testSpilledFullExtentsLeaveMemory();
    testSpillAtTheMemoryWall();
    testFullReclaimReachesAnIdleSpilledChain();
    testFullShortfallReleasesEmptyIndexExtents();
    testCriticalKeepsTheIndexRunway();
    testIndexShortfallCompactsOnlyFullExtents();
    testANeedInPlaceEmptiesNoExtentOfItsForm();
    testAPageNeedOfAFormTheCacheHoldsNoneOfCountsNothing();
    testReusablePagesCountTheirForm();
    testPendingPagesCountWritesThatSpill();
    testWritesThatFreeNothingHoldNothingBack();
    testWritesShortOfARefusalHoldNothingBack();
    testMemoryReclaimEmptiesAnExtentBesideWritesThatSpill();
    testConversionWaitsForItsPagesToSpill();
    testConversionAtTheWallFindsTheRunway();
    testUnspillOnlyAsTheOnlyHolder();
    testAnUnspillLeavesTheHeldFreePages();
    testWritersMoveOntoPagesBeingWritten();
    testAFailedRestoresPageIsNeverWritten();
    testQuotaGivesUpOnlyCopiesNothingReads();
    testQuotaKeepsTheSpilledLeafOfAStateInRam();
    testCommittedPagesTakeTheSpilledChainOfAStateInRam();
    testUnreadableCopyLeavesBeforeTheCommandThatReadIt();
    testCommitmentsLockQuotaUntilWritten();
    testACopyWrittenBeforeTheRestoreReturnsItsCommitment();
    testAWriterBesideARestoreHoldsNoCopyOfIt();
    testACommitmentLeavesRoomForTheCopiesItsRestoreHolds();
    testARestoreReadsAChainOnlyAsFarAsItMatches();
    testWriteThroughResumesBesideCopiesThatFillTheRoom();
    testFailedWritesKeepTheirCommitment();
    testPagesPastTheCommitmentCommitTheirSlots();
    testAConversionsHistoryWithCopiesIsReleasedAtOnce();
    testEveryHolderDecidesWhetherAPageSpills();
    testBlocksAreWrittenOnce();
    testOwedCopiesReturnWhileAnyIsWritten();
    testRecomputedBlocksSpillToCopiesOfTheirPages();
    testUnreadableCopyLeavesWithItsHolders();
    testPoisonedBlockLeavesOnceItsWriteLands();
    testWindowsTrimWhatEveryHolderPassed();
    testWritersKeepTheirPagesTrimmedOrNot();
    testStateTailsPinTheirWindows();
    testStatesNeedTheirTailFull();
    testFullShortfallsTrimBeforeAnyLeafGoes();
    testFullShortfallsKeepTheTailsOfStatesTheSsdKeeps();
    testFullShortfallsAtTheWallTakeDeadLeavesFirst();
    testRestoresReturnNothing();
    testHeldDemotionReturnsNoPage();
    testImmediateEvictionCountsNoLandingPage();
    testReturningSplitsLandingAndOwed();
    testPeekHasNoSideEffects();
    testResumePointAndWantedChainsAreCertainReuse();
    testInUseUnitsComeOnlyWithReachInUse();
    testBurstReclaimTakesWholeExtentsOncePerNeed();
    testGrowthNeedKeepsTheFreePagesItTakes();
    testDeviceNeedReachesTheLeavesPastTheStates();
    testGrowthPastAStateWriteReachesTheIdleChains();
    testDeviceNeedKeepsKvBesideTheIndexRunway();
    testDeviceNeedPassesOverLeavesOfFormsOutOfReach();
    testARefusedRequestWaitsForThePageOnItsWayBack();
    testAStateNeedCountsTheStagingBufferOnce();
    testImmediateWalksOfferWhatLandsLaterInItsPlace();
    testImmediateDeviceWalkPassesOverWhatAWriteFrees();
    testDeviceWalkOffersTheStagedWriteAsAUnit();
    testPageWalkOffersEachDeadLeafOnce();
    testEngineRefusedPageNeedIsADeviceNeed();
    testPageNeedReachesParentsAndLeavesUnderStates();
    testDeviceUnitReportsWhatItsDemotionsReturn();
    testClaimTakesCopiedCacheThenRebuiltPoolsThenWrites();
    testOwnDemotionsHoldBackTheStore();
    testDemotionsStopWhereTheyCompleteAnExtent();
    testAReleaseOfWhatALandingCompletesCountsOnce();
    testATakeCountsTheLandingItLetsFree();
    testGrowthCountsNoPageInTransfer();
    testGrowthCountsNoStateThatWaitsForAWrite();
    testStagedStateFreesItsCopiedChain();
    std::cout << "engine cache tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "engine cache tests failed: " << error.what() << '\n';
    return 1;
  }
}
