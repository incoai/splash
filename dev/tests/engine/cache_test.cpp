#include "engine/Cache.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace splash;
using namespace splash::engine;

namespace {

class Backing final : public KvBacking {
public:
  explicit Backing(uint32_t pages, uint32_t maximumResidentPages =
                                       std::numeric_limits<uint32_t>::max())
      : resident_(pages), maximumResidentPages_(maximumResidentPages) {}
  uint32_t pageCount() const noexcept override { return resident_.size(); }
  uint64_t bytesPerPage() const noexcept override { return 4096; }
  bool isResident(uint32_t page) const override { return resident_.at(page); }
  splash::metal::AllocationResult ensureResident(uint32_t page) override {
    const uint32_t first = extentFirstPage(page);
    const uint32_t count = extentPageCount(page);
    uint32_t additional = 0;
    for (uint32_t index = first; index < first + count; ++index)
      additional += !resident_.at(index);
    if (uint64_t{residentPages()} + additional > maximumResidentPages_) {
      return false;
    }
    for (uint32_t index = first; index < first + count; ++index) {
      resident_.at(index) = true;
    }
    if (additional)
      ++mappedExtents;
    return true;
  }
  bool releaseBackingForPage(uint32_t page) override {
    const uint32_t first = extentFirstPage(page);
    const uint32_t count = extentPageCount(page);
    for (uint32_t index = first; index < first + count; ++index) {
      resident_.at(index) = false;
    }
    ++unmappedExtents;
    if (pacedReleases)
      ready = false;
    return true;
  }
  bool releaseReady() const noexcept override { return ready; }
  void awaitRelease() override { ready = true; }
  bool ready = true;
  bool pacedReleases = false;
  uint32_t extentFirstPage(uint32_t page) const override {
    return page - page % 4;
  }
  uint32_t extentPageCount(uint32_t page) const override {
    return std::min<uint32_t>(4, resident_.size() - extentFirstPage(page));
  }
  uint32_t residentPages() const noexcept {
    uint32_t count = 0;
    for (bool value : resident_)
      count += value;
    return count;
  }
  uint32_t mappedExtents = 0;
  uint32_t unmappedExtents = 0;
private:
  std::vector<bool> resident_;
  uint32_t maximumResidentPages_;
};

class State final : public CompositeState {
public:
  explicit State(uint64_t bytes) : bytes_(bytes) {}
  uint64_t bytes() const noexcept override { return bytes_; }

private:
  uint64_t bytes_;
};

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

CacheNamespace cacheNamespace() {
  CacheNamespace result;
  result.digest.fill(0x5a);
  return result;
}

void publish(engine::Cache &resources, uint64_t block,
             uint64_t bytes) {
  resources.publishCompositeState(block, std::make_shared<State>(bytes));
}

std::vector<uint32_t> tokens(uint32_t count, uint32_t salt = 0) {
  std::vector<uint32_t> result(count);
  for (uint32_t i = 0; i < count; ++i)
    result[i] = salt + i + 1;
  return result;
}

void testCanonicalPagesAndSparseState() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(65);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, prompt.size()).granted(),
          "request pages were not admitted");
  uint64_t deepest = resources.publishCommittedBlocks(1, prompt, 64);
  require(deepest && resources.blockAt(1, 64) == deepest,
          "complete Page32 chain was not published");
  publish(resources, deepest, 100);
  resources.endRequest(1);

  auto lookup = resources.lookup(prompt);
  require(lookup.kvBoundary == 64 && lookup.resumeBoundary() == 64 &&
              !lookup.junctionBoundary(),
          "KV-first lookup did not coordinate the sparse state");
  resources.beginRequest(2);
  require(resources.restoreRequest(2, lookup).granted(), "restore pages were denied");
  require(resources.pageTable(2).pages.size() == 2,
          "restored request did not retain the state KV chain");
  resources.endRequest(2);
}

void testKvDeeperThanStateAndDependencyEviction() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(97);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 96).granted(), "KV allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 96));
  const uint64_t middle = resources.blockAt(1, 64);
  publish(resources, middle, 100);
  resources.endRequest(1);
  auto lookup = resources.lookup(prompt);
  require(lookup.kvBoundary == 96 && lookup.resumeBoundary() == 64 &&
              lookup.junctionBoundary() == 96,
          "dense KV did not expose the lazy state junction");
  lookup.state.reset();
  require(resources.reclaimCache(1, false) >= 100,
          "unreferenced composite state was not reclaimed first");
  require(resources.snapshot().kvCache.blocks == 2,
          "LRU reclaim did not remove the older fragmented KV leaf first");
  require(resources.reclaimCache(1, false) != 0,
          "KV backing was not reclaimed after cached state");
  require(resources.snapshot().stateCache.entries == 0,
          "composite state outlived its KV dependency");
}

void testActiveTipProtectsTheContentChain() {
  Backing backing(4);
  KvPool pool(backing);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(97, 1000);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 64).granted(),
          "active request KV allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 64));
  resources.beginRequest(2);

  engine::TokenAdmission blocked = resources.ensureTokens(2, 96);
  require(!blocked.granted() && resources.snapshot().kvCache.blocks == 2,
          "memory pressure evicted an active request KV tip");

  resources.endRequest(1);
  require(resources.ensureTokens(2, 96).granted(),
          "released KV tip did not become reclaimable");
  resources.endRequest(2);
}

void testPhysicalGrowthReclaimsOneWholeCachedExtent() {
  Backing backing(8, 4);
  KvPool pool(backing);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(129, 2000);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 128).granted(),
          "initial KV extent allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 128));
  resources.endRequest(1);
  require(resources.snapshot().kvCache.blocks == 4 &&
              resources.snapshot().pool.pagesResident == 4,
          "cached extent setup is wrong");

  resources.beginRequest(2);
  require(!resources.ensureTokens(2, 1).granted(),
          "physical growth bypassed engine-coordinated reclaim");
  require(resources.reclaimCache(1, false) != 0 &&
              resources.ensureTokens(2, 1).granted(),
          "explicit backend reclaim did not release cached KV backing");
  const auto snapshot = resources.snapshot();
  require(snapshot.kvCache.blocks == 0 && snapshot.pool.pagesResident == 4 &&
              snapshot.pool.pagesActive == 1,
          "growth reclaim did not atomically replace the cached extent");
  resources.endRequest(2);
}

void testFragmentedColdKvPrecedesNewerState() {
  Backing backing(8, 4);
  KvPool pool(backing);
  engine::Cache resources(pool, cacheNamespace());
  const auto prompt = tokens(129);
  resources.beginRequest(1);
  require(resources.ensureTokens(1, 128).granted(), "fixture allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 128));
  const uint64_t stateBlock = resources.blockAt(1, 32);
  resources.endRequest(1);
  publish(resources, stateBlock, 100);

  const auto reclaimed = resources.reclaimOne();
  require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 0 &&
              resources.snapshot().kvCache.blocks == 3 &&
              resources.snapshot().stateCache.entries == 1 &&
              backing.unmappedExtents == 0,
          "physical-byte preference evicted newer state before cold KV");
}

// A request short of logical pages evicts the least recently used KV leaf
// together with its composite state. A leaf whose state a lookup holds is
// not a candidate, so the next leaf goes instead.
void testLogicalPressureEvictsALeafWithItsState() {
  for (bool leased : {false, true}) {
    Backing backing(4);
    KvPool pool(backing);
    engine::Cache resources(pool, cacheNamespace());
    const auto older = tokens(33, 100);
    const auto newer = tokens(33, 200);
    uint64_t id = 1;
    for (const auto &prompt : {older, newer}) {
      resources.beginRequest(id);
      require(resources.ensureTokens(id, 32).granted(),
              "fixture allocation failed");
      publish(resources, resources.publishCommittedBlocks(id, prompt, 32), 100);
      resources.endRequest(id++);
    }
    std::optional<CacheLookup> lease;
    if (leased)
      lease = resources.lookup(older);
    resources.beginRequest(id);
    require(resources.ensureTokens(id, 96).granted(),
            "logical pressure did not take a cached page");
    const auto snapshot = resources.snapshot();
    require(snapshot.kvCache.blocks == 1 &&
                snapshot.stateCache.entries == 1 &&
                snapshot.stateCache.evictions == 1,
            "logical eviction did not take exactly one leaf with its state");
    require(resources.probe(leased ? older : newer).cachedTokens() == 32 &&
                resources.probe(leased ? newer : older).cachedTokens() == 0,
            "logical eviction took the leased or the newer leaf");
    resources.endRequest(id);
  }
}

void testReplacementPreservesBackingEvenWhenExtentBecomesEmpty() {
  Backing backing(8, 4);
  KvPool pool(backing);
  engine::Cache resources(pool, cacheNamespace());
  resources.beginRequest(1);
  const auto prompt = tokens(33);
  require(resources.ensureTokens(1, 32).granted(), "fixture allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(1, prompt, 32));
  resources.endRequest(1);

  const auto reclaimed = resources.reclaimOne(CacheReclaimMode::ReuseBacking);
  require(reclaimed.madeProgress && reclaimed.reclaimedBytes == 0 &&
              resources.snapshot().kvCache.blocks == 0 &&
              backing.residentPages() == 4 && backing.unmappedExtents == 0,
          "replacement unmapped the newly reusable extent");
  resources.beginRequest(2);
  require(resources.ensureTokens(2, 128).granted() &&
              backing.mappedExtents == 1 && backing.unmappedExtents == 0,
          "replacement unnecessarily remapped reusable backing");
  resources.endRequest(2);
  require(resources.reclaimCache(0, false) == 4 * 4096 &&
              backing.residentPages() == 0 && backing.unmappedExtents == 1,
          "zero-target physical shrink did not release the empty extent");
}

} // namespace

// Physical release is paced by the backing. While an earlier release is
// still in flight and more empty extents wait, a reclaim pass neither queues
// another unmap nor evicts reusable cache; it resumes once the backing is
// ready again.
void testReclaimDefersBehindInFlightRelease() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, cacheNamespace());
  auto prompt = tokens(33);
  resources.beginRequest(3);
  require(resources.ensureTokens(3, 32).granted(), "cached page allocation failed");
  static_cast<void>(resources.publishCommittedBlocks(3, prompt, 32));
  resources.endRequest(3);
  // Request 4 fills the runway extent and maps a second one; request 5 needs
  // more pages than remain resident and maps a third. Both end empty.
  for (const auto [request, tokenCount] : {std::pair{4, 128}, std::pair{5, 256}}) {
    resources.beginRequest(request);
    require(resources.ensureTokens(request, tokenCount).granted(),
            "empty extent allocation failed");
    resources.endRequest(request);
  }
  auto snapshot = resources.snapshot();
  require(snapshot.kvCache.blocks == 1 && snapshot.pool.reclaimableExtents == 2 &&
              backing.residentPages() == 12 && !resources.releaseDeferred(),
          "paced-release setup geometry changed");

  backing.pacedReleases = true;
  require(resources.reclaimCache(1ULL << 30, false) == 4 * 4096 &&
              backing.unmappedExtents == 1 && resources.releaseDeferred() &&
              resources.snapshot().kvCache.blocks == 1 &&
              resources.snapshot().pool.reclaimableExtents == 1,
          "reclaim queued a second unmap or evicted cache behind an in-flight release");
  require(resources.reclaimCache(1ULL << 30, false) == 0 &&
              !resources.reclaimOne().madeProgress &&
              backing.unmappedExtents == 1 &&
              resources.snapshot().kvCache.blocks == 1,
          "deferred reclaim made progress while the release was in flight");
  backing.awaitRelease();
  require(!resources.releaseDeferred(), "completed release still reported deferred");
  // The next pass releases the last empty extent, then evicts the cached
  // block whose extent cannot be released until that unmap completes.
  require(resources.reclaimCache(1ULL << 30, false) == 4 * 4096 &&
              backing.unmappedExtents == 2 &&
              resources.snapshot().kvCache.blocks == 0 &&
              resources.releaseDeferred(),
          "reclaim did not resume after the release completed");
  backing.awaitRelease();
  require(resources.reclaimCache(0, false) == 4 * 4096 &&
              backing.unmappedExtents == 3 && backing.residentPages() == 0 &&
              !resources.releaseDeferred(),
          "final paced pass did not return the last empty extent");
}

int main() {
  try {
    testReclaimDefersBehindInFlightRelease();
    testCanonicalPagesAndSparseState();
    testKvDeeperThanStateAndDependencyEviction();
    testActiveTipProtectsTheContentChain();
    testPhysicalGrowthReclaimsOneWholeCachedExtent();
    testFragmentedColdKvPrecedesNewerState();
    testLogicalPressureEvictsALeafWithItsState();
    testReplacementPreservesBackingEvenWhenExtentBecomesEmpty();
    std::cout << "engine cache tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "engine cache tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
