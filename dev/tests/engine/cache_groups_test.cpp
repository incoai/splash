#include "TestKvPool.hpp"
#include "engine/Cache.hpp"

#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Payload final : StatePayload {
  explicit Payload(uint64_t size = 100) : size(size) {}
  uint64_t size;
  uint64_t bytes() const noexcept override { return size; }
};
struct Fixture {
  test::TestKvBacking backing{64, 100};
  KvPool pool{backing};
  Cache cache{pool, {}};
  std::vector<uint32_t> tokens;
  std::vector<uint64_t> blocks;
  Fixture(std::vector<CacheGroupSpec> groups) : tokens(257) {
    cache.configureGroups(std::move(groups));
    std::iota(tokens.begin(), tokens.end(), 1000);
    cache.beginRequest(1);
    require(cache.ensureTokens(1, 256).granted(), "fixture allocation failed");
    static_cast<void>(cache.publishCommittedBlocks(1, tokens, 256));
    for (uint32_t end = 32; end <= 256; end += 32)
      blocks.push_back(cache.blockAt(1, end));
    cache.endRequest(1);
  }
  void publish(uint32_t boundary, std::vector<CachedStateBlock> parts,
               bool checkpoint = false) {
    cache.publishCompositeState(blocks.at(boundary / 32 - 1),
                                std::make_shared<RestoreState>(
                                    RestoreState{boundary, std::move(parts)}),
                                checkpoint);
  }
  CacheLookup lookup(uint32_t length) {
    return cache.lookup(std::span(tokens).first(length));
  }
};
CachedStateBlock part(uint32_t group, uint32_t begin, uint32_t end) {
  return {group, begin, end, std::make_shared<Payload>()};
}
void noAuxiliaryState() {
  Fixture f({});
  auto hit = f.lookup(257);
  require(hit.resumeBoundary() == 256 && hit.state->state()->blocks.empty(),
          "full-attention target incorrectly requires recurrent state");
  require(f.cache.snapshot().stateCache.bytes == 0,
          "empty groups allocated state");
  f.cache.beginRequest(2);
  require(f.cache.restoreRequest(2, hit).granted(), "KV-only restore failed");
  require(f.cache.pageTable(2).pages.size() == 8, "KV-only restore lost pages");
  f.cache.endRequest(2);
}
void independentGroups() {
  Fixture f({{7, CacheGroupKind::Checkpoint, 0},
             {19, CacheGroupKind::SlidingWindow, 64}});
  auto a = part(19, 0, 64), b = part(19, 64, 96);
  auto old = part(7, 64, 64), latest = part(7, 96, 96);
  f.publish(64, {old, a});
  f.publish(96, {latest, a, b});
  {
    auto hit = f.lookup(97);
    require(hit.resumeBoundary() == 96 &&
                hit.state->state()->blocks.size() == 3,
            "groups did not coordinate a complete restore");
    require(f.cache.snapshot().stateCache.bytes == 400,
            "shared group block counted twice");
    auto missing = RestoreState{96, {b}};
    f.cache.discardState(f.blocks[2], &missing);
    require(hit.state->state()->blocks.size() == 3,
            "invalidation destroyed a live restore lease");
  }
  auto fallback = f.lookup(97);
  require(fallback.resumeBoundary() == 64,
          "missing draft fragment was reported as a complete hit");
  require(f.cache.stateResident(f.blocks[2]),
          "draft invalidation also discarded checkpoint");
}
void retireCheckpointKeepsSharedWindow() {
  Fixture f({{7}, {19, CacheGroupKind::SlidingWindow, 64}});
  auto shared = part(19, 0, 64);
  f.publish(64, {part(7, 64, 64), shared}, true);
  auto old = f.cache.checkpointState(f.blocks[1]);
  f.publish(96, {part(7, 96, 96), shared, part(19, 64, 96)}, true);
  require(f.cache.retireCheckpointState(old), "checkpoint retirement failed");
  require(f.lookup(97).resumeBoundary() == 96,
          "retiring a checkpoint removed a newer checkpoint's window");
}
void retirementReleasesUnsharedWindow() {
  Fixture f({{7}, {19, CacheGroupKind::SlidingWindow, 64}});
  f.publish(64, {part(7, 64, 64), part(19, 0, 64)}, true);
  auto old = f.cache.checkpointState(f.blocks[1]);
  f.publish(192, {part(7, 192, 192), part(19, 128, 192)}, true);
  require(f.cache.retireCheckpointState(old), "checkpoint retirement failed");
  require(f.cache.snapshot().stateCache.entries == 2 &&
              f.lookup(193).resumeBoundary() == 192,
          "retired checkpoint left an unshared window");
}
void leaseReleasePreservesDependencyOrder() {
  Fixture f({{7}, {19, CacheGroupKind::SlidingWindow, 64}});
  f.publish(64, {{7, 64, 64, std::make_shared<Payload>(100)},
                 {19, 0, 64, std::make_shared<Payload>(3)}});
  {
    auto hit = f.lookup(65);
    require(bool(hit.state), "lease fixture missed");
  }
  require(
      f.cache.reclaimOneState().madeProgress &&
          f.cache.snapshot().stateCache.entries == 0,
      "lease release left an unusable checkpoint after evicting its window");
}

void invalidatedLeaseReleasesUnneededWindow() {
  Fixture f({{7}, {19, CacheGroupKind::SlidingWindow, 64}});
  auto checkpoint = part(7, 64, 64);
  f.publish(64, {checkpoint, part(19, 0, 64)});
  {
    auto hit = f.lookup(65);
    auto failed = RestoreState{64, {checkpoint}};
    f.cache.discardState(f.blocks[1], &failed);
    require(hit.state->state()->blocks.size() == 2,
            "invalidation destroyed the active restore payload");
  }
  require(f.cache.snapshot().stateCache.entries == 0,
          "invalidated restore lease left an unneeded window");
}

void incompleteCheckpointGroupsDoNotRetainWindow() {
  Fixture f({{7}, {8}, {19, CacheGroupKind::SlidingWindow, 64}});
  auto checkpoint = part(7, 64, 64);
  f.publish(64, {checkpoint, part(8, 64, 64), part(19, 0, 64)});
  auto failed = RestoreState{64, {checkpoint}};
  f.cache.discardState(f.blocks[1], &failed);
  require(f.cache.snapshot().stateCache.entries == 1 &&
              f.lookup(65).resumeBoundary() == 0,
          "incomplete checkpoint groups retained an unusable window");
}

void diskQuotaReclaimCanRetireTheWriteSource() {
  struct TieredPayload final : StatePayload {
    explicit TieredPayload(bool disk) : disk(disk) {}
    bool disk;
    mutable uint32_t writes = 0;
    uint64_t bytes() const noexcept override { return 100; }
    uint64_t residentBytes() const noexcept override {
      return disk ? 0 : bytes();
    }
    bool canOffload() const noexcept override { return !disk; }
    std::unique_ptr<StateOffload>
    offload(std::function<void()>) const override {
      ++writes;
      return {}; // Full disk quota: the caller must make room.
    }
  };
  for (bool copy : {false, true}) {
    test::TestKvBacking backing{4, 100};
    KvPool pool(backing);
    CacheRecency recency;
    KvCache kv(pool, {}, recency);
    StateCache states(kv, recency);
    states.configure({{7}, {19, CacheGroupKind::SlidingWindow, 32}});
    auto allocation = pool.acquirePages(1, false);
    require(allocation.granted(), "quota reclaim fixture allocation failed");
    std::array<uint32_t, 32> tokens{};
    const auto leaf = kv.insert(0, tokens, allocation.pages[0]).id;
    pool.releasePage(allocation.pages[0], false);
    states.importDisk(
        leaf, std::make_shared<RestoreState>(RestoreState{
                  32, {{7, 32, 32, std::make_shared<TieredPayload>(true)}}}));
    auto source = std::make_shared<TieredPayload>(false);
    states.publish(leaf, std::make_shared<RestoreState>(
                             RestoreState{32, {{19, 0, 32, source}}}));
    uint32_t attempts = 0;
    const auto makeRoom = [&] {
      if (++attempts != 1)
        return false;
      states.evictDiskOnly(leaf, 7);
      return true;
    };
    if (copy)
      require(!states.copyToDisk(leaf, {}, makeRoom),
              "copied a window with no remaining checkpoint");
    else
      require(states.reclaim(leaf, {}, makeRoom, 19).evicted,
              "quota reclaim did not release the obsolete source");
    require(states.snapshot().entries == 0 && !states.writing() &&
                attempts == 1 && source->writes == 1,
            "quota reclaim retained or rewrote an orphan window");
  }
}

void longPrefillKeepsOnlyLiveWindows() {
  test::TestKvBacking backing(4096, 1);
  KvPool pool(backing);
  Cache cache(pool, {});
  cache.configureGroups({{7}, {19, CacheGroupKind::SlidingWindow, 2048}});
  std::vector<uint32_t> tokens(131073);
  std::iota(tokens.begin(), tokens.end(), 1);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 131072).granted(),
          "long prefill fixture failed");
  static_cast<void>(cache.publishCommittedBlocks(1, tokens, 131072));
  StateCheckpoint previous;
  for (uint32_t end = 4096; end <= 131072; end += 4096) {
    require(cache.retireCheckpointState(previous),
            "long prefill retirement failed");
    auto state = std::make_shared<RestoreState>();
    state->boundary = end;
    state->blocks.push_back(part(7, end, end));
    for (uint32_t page = end - 2048 + 128; page <= end; page += 128)
      state->blocks.push_back(part(19, page - 128, page));
    const auto leaf = cache.blockAt(1, end);
    cache.publishCompositeState(leaf, std::move(state), true);
    previous = cache.checkpointState(leaf);
    require(cache.snapshot().stateCache.entries == 17,
            "long prefill accumulated orphan windows");
  }
  require(cache.lookup(tokens).resumeBoundary() == 131072,
          "long prefill lost latest recovery point");
  cache.endRequest(1);
}

void retirementPreservesOtherBranch() {
  Fixture f({{7}, {19, CacheGroupKind::SlidingWindow, 128}});
  auto shared = part(19, 0, 64);
  f.publish(96, {part(7, 96, 96), shared, part(19, 64, 96)}, true);
  auto old = f.cache.checkpointState(f.blocks[2]);
  auto branch = f.tokens;
  branch[64] += 10000;
  f.cache.beginRequest(2);
  require(f.cache.ensureTokens(2, 96).granted(), "branch allocation failed");
  static_cast<void>(f.cache.publishCommittedBlocks(2, branch, 96));
  const auto leaf = f.cache.blockAt(2, 96);
  f.cache.publishCompositeState(
      leaf,
      std::make_shared<RestoreState>(
          RestoreState{96, {part(7, 96, 96), shared, part(19, 64, 96)}}),
      true);
  f.cache.endRequest(2);
  require(f.cache.retireCheckpointState(old) &&
              f.cache.lookup(std::span(branch).first(97)).resumeBoundary() ==
                  96 &&
              f.cache.snapshot().stateCache.entries == 3,
          "retirement removed a sibling branch's shared window");
}

void requestTouchesRestoreDependencies() {
  for (bool reverse : {false, true}) {
    std::vector<CacheGroupSpec> groups{{7},
                                       {19, CacheGroupKind::SlidingWindow, 64}};
    if (reverse)
      std::reverse(groups.begin(), groups.end());
    Fixture f(groups);
    CachedStateBlock shared{19, 0, 64, std::make_shared<Payload>(1)};
    CachedStateBlock tail{19, 64, 96, std::make_shared<Payload>(1)};
    f.publish(64, {part(7, 64, 64), shared});
    f.publish(96, {part(7, 96, 96), shared, tail});
    f.cache.beginRequest(2);
    {
      auto hit = f.lookup(97);
      require(f.cache.restoreRequest(2, hit).granted(), "restore failed");
    }
    f.cache.endRequest(2);
    require(f.cache.reclaimOneState().madeProgress,
            "group reclaim made no progress");
    require(f.lookup(97).resumeBoundary() == 96,
            "LRU removed a shared dependency before its older checkpoint");
    require(f.cache.reclaimOneState().madeProgress &&
                f.cache.snapshot().stateCache.entries == 0,
            "last checkpoint left orphan window dependencies");
  }
}
void windowWithoutCheckpoint() {
  Fixture f({{42, CacheGroupKind::SlidingWindow, 64}});
  f.publish(96, {part(42, 0, 64), part(42, 64, 96)});
  auto hit = f.lookup(97);
  require(hit.resumeBoundary() == 96,
          "window-only target requires a checkpoint group");
  auto shorter = f.lookup(65);
  require(shorter.resumeBoundary() == 64,
          "window block cannot independently restore its endpoint");
  require(!f.lookup(33).state, "cache fabricated an unpublished partial block");
}
void severalGroupsAndHoles() {
  Fixture f({{3, CacheGroupKind::Checkpoint, 0},
             {9, CacheGroupKind::Checkpoint, 0},
             {25, CacheGroupKind::SlidingWindow, 128},
             {80, CacheGroupKind::SlidingWindow, 64}});
  f.publish(128, {part(3, 128, 128), part(9, 128, 128), part(25, 0, 64),
                  part(25, 64, 128), part(80, 64, 128)});
  require(f.lookup(129).resumeBoundary() == 128,
          "multiple target/draft groups do not compose");
  f.publish(192, {part(3, 192, 192), part(9, 192, 192), part(25, 160, 192),
                  part(80, 128, 192)});
  require(f.lookup(193).resumeBoundary() == 128, "window hole was ignored");
}
void protectWholeWindow() {
  Fixture f({{7}, {19, CacheGroupKind::SlidingWindow, 128}});
  f.publish(128, {part(7, 128, 128), part(19, 0, 64), part(19, 64, 128)});
  f.publish(192, {part(7, 192, 192), part(19, 64, 128), part(19, 128, 192)});
  static_cast<void>(f.cache.reclaimCache(UINT64_MAX, false, true));
  require(f.cache.snapshot().stateCache.entries == 3,
          "speculative reclaim did not retain exactly one complete window");
  require(f.lookup(193).resumeBoundary() == 192,
          "speculative reclaim punched a hole in the protected window");
  static_cast<void>(f.cache.reclaimCache(UINT64_MAX, false, false));
  require(!f.cache.snapshot().stateCache.entries,
          "demanded reclaim could not release protected groups");
}
void independentEviction() {
  Fixture f({{7}, {19, CacheGroupKind::SlidingWindow, 64}});
  f.publish(64, {part(7, 64, 64), part(19, 0, 64)});
  require(f.cache.reclaimOneState().madeProgress, "group LRU made no progress");
  require(f.cache.snapshot().stateCache.entries == 0,
          "checkpoint eviction retained an unusable window");
  require(!f.lookup(65).state, "incomplete group set was treated as a hit");
}
void branchIsolation() {
  Fixture f({{42, CacheGroupKind::SlidingWindow, 64}});
  f.publish(96, {part(42, 0, 64), part(42, 64, 96)});
  auto branch = f.tokens;
  branch[64] += 10000;
  f.cache.beginRequest(2);
  require(f.cache.ensureTokens(2, 96).granted(), "branch allocation failed");
  static_cast<void>(f.cache.publishCommittedBlocks(2, branch, 96));
  const auto leaf = f.cache.blockAt(2, 96);
  f.cache.publishCompositeState(
      leaf,
      std::make_shared<RestoreState>(RestoreState{96, {part(42, 64, 96)}}));
  f.cache.endRequest(2);
  require(f.cache.lookup(std::span(branch).first(97)).resumeBoundary() == 96,
          "branch could not reuse its shared ancestor window block");
  branch[0] += 10000;
  require(!f.cache.lookup(std::span(branch).first(97)).state,
          "unrelated token chain reused another branch's group blocks");
}
void boundarySearchMatchesExhaustiveCoverage() {
  std::mt19937 random(314159);
  for (unsigned trial = 0; trial < 500; ++trial) {
    std::vector<CacheGroupSpec> groups{{1, CacheGroupKind::SlidingWindow, 128},
                                       {2, CacheGroupKind::SlidingWindow, 64}};
    if (trial % 2)
      groups.push_back({3});
    CacheGroupCoordinator coordinator(groups);
    std::map<std::pair<uint32_t, uint32_t>, CachedStateBlock> records;
    for (const auto &group : groups)
      for (uint32_t end = 32; end <= 512; end += 32)
        if (random() % 4) {
          const auto begin =
              group.kind == CacheGroupKind::Checkpoint
                  ? end
                  : end - std::min(end, 32 * (1 + uint32_t(random() % 4)));
          records.emplace(std::pair{group.id, end}, part(group.id, begin, end));
        }
    const auto find = [&](CacheGroupId group,
                          uint32_t end) -> std::optional<CachedStateBlock> {
      auto found = records.find({group, end});
      return found == records.end() ? std::nullopt
                                    : std::optional(found->second);
    };
    for (uint32_t maximum = 32; maximum <= 512; maximum += 32) {
      uint32_t expected = maximum;
      while (expected && !coordinator.complete(expected, find))
        expected -= 32;
      const auto actual = coordinator.match(maximum, 32, find);
      require((actual ? actual->boundary : 0) == expected,
              "window search skipped a recoverable boundary");
    }
  }
}
void incompletePointDoesNotRefreshDependencies() {
  CacheGroupCoordinator coordinator(
      {{19, CacheGroupKind::SlidingWindow, 64}, {7}});
  std::vector<CachedStateBlock> records{part(7, 96, 96), part(19, 64, 96)};
  const auto find = [&](CacheGroupId group,
                        uint32_t end) -> std::optional<CachedStateBlock> {
    for (const auto &record : records)
      if (record.group == group && record.end == end)
        return record;
    return {};
  };
  std::vector<CacheGroupId> visited;
  const auto visit = [&](const auto &block) { visited.push_back(block.group); };
  coordinator.visitComplete(96, find, visit);
  require(visited.empty(), "incomplete restore refreshed orphan dependencies");
  records.push_back(part(19, 0, 64));
  coordinator.visitComplete(96, find, visit);
  require(visited == std::vector<CacheGroupId>{7, 19, 19},
          "restore visit did not order checkpoint before dependencies");
}
void sharedPhysicalSlices() {
  auto payload = std::make_shared<Payload>();
  RestoreState state{224, {{1, 32, 64, payload}, {1, 192, 224, payload}}};
  require(state.bytes() == 100,
          "two logical slices double-count one physical page");
}
void declarations() {
  for (const auto &specs :
       {std::vector<CacheGroupSpec>{{1}, {1}},
        std::vector<CacheGroupSpec>{{2, CacheGroupKind::SlidingWindow, 0}},
        std::vector<CacheGroupSpec>{{2, CacheGroupKind::Checkpoint, 64}}}) {
    bool rejected = false;
    try {
      validateCacheGroups(specs);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "malformed cache group declaration accepted");
  }
}
} // namespace
int main() {
  try {
    declarations();
    noAuxiliaryState();
    independentGroups();
    retireCheckpointKeepsSharedWindow();
    retirementReleasesUnsharedWindow();
    longPrefillKeepsOnlyLiveWindows();
    retirementPreservesOtherBranch();
    leaseReleasePreservesDependencyOrder();
    invalidatedLeaseReleasesUnneededWindow();
    incompleteCheckpointGroupsDoNotRetainWindow();
    diskQuotaReclaimCanRetireTheWriteSource();
    requestTouchesRestoreDependencies();
    windowWithoutCheckpoint();
    severalGroupsAndHoles();
    sharedPhysicalSlices();
    incompletePointDoesNotRefreshDependencies();
    boundarySearchMatchesExhaustiveCoverage();
    protectWholeWindow();
    independentEviction();
    branchIsolation();
    std::cout << "cache group coordination tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
