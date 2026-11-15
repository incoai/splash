// What a claim counts of the Spill lanes' releases before it takes any (MemoryLedger's bound), held against what
// their takes free, the Index runway and the starts placed Resident on that count, over random conversation trees
// of Spill lanes at the memory wall; and the cases those runs reduce to.

#include "EngineTestSupport.hpp"
#include "MemoryPasses.hpp"
#include "TestChecks.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::Executor;
using splash::test::idle;
using splash::test::Spilling;

namespace {

using splash::test::require;

constexpr uint64_t kFullPageBytes = 423'936, kIndexPageBytes = 24'576;

// Spill lanes under the governor's 40 GiB limit at Qwen3.8-Flash-Next's page bytes and history staging, in extents
// of `extentPages` pages, its 390 MiB hot block floor scaled to them, with `pageIds` page ids: with ids for every
// extent the lanes reach, the governor limits the pool and never the id range, as production's ids hold every
// extent the limit could hold of Index pages (RuntimeResources).
struct Wall final {
  explicit Wall(uint32_t extentPages, uint32_t pageIds, const std::function<void(Executor &)> &shape = {})
      : spill{65'536, false, 0, 200'000, Spilling::Extent{extentPages}, shape, {}, true,
              Spilling::Pages{.count = pageIds, .fullBytes = kFullPageBytes, .indexBytes = kIndexPageBytes}} {
    spill.model.spilledReadFloorBytes = (390ULL << 20) / 256 * extentPages;
    spill.model.stagingPageBytes = 36'864;
  }
  // Starts `request` where the room it finds holds its first command's extents and state, not its Resident KV,
  // beside what the lanes' releases give it: Spill, but where those hold its Resident KV.
  void start(EngineRequest request) {
    const uint64_t id = request.id;
    spill.host.leaveRoom(startRoomBytes());
    spill.engine.submit(std::move(request));
    spill.runUntil([&] { return std::ranges::count(spill.events.startIds, id) != 0; }, "the start did not start");
    spill.host.leaveRoom(std::nullopt);
  }
  void startSpill(EngineRequest request) {
    const uint64_t id = request.id;
    start(std::move(request));
    require(kvMode(id) == KvMode::Spill, "the start was not placed Spill");
  }
  [[nodiscard]] uint64_t startRoomBytes() const {
    const uint64_t command = spill.model.groupRows / KvCache::pageTokens;
    return (command + spill.pool.extentPages()) * kFullPageBytes + (20ULL << 20);
  }
  // The first reads the lanes holding Index pages lack (Engine::readsShortfall): a decoding lane's hot block floor,
  // a prefilling lane's history staging through its next command.
  [[nodiscard]] uint64_t owedReads() const {
    uint64_t bytes = 0;
    std::vector<uint64_t> decoding;
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests) {
      if (!spill.model.requests.contains(active.id) || !spill.model.requests.at(active.id).resident ||
          !spill.cache.holdsIndexPages(active.id))
        continue;
      if (active.phase != Phase::Prefill) {
        decoding.push_back(active.id);
        continue;
      }
      const uint64_t end = std::min<uint64_t>(active.promptProcessed + spill.model.groupRows, active.promptTokens);
      const auto pages = static_cast<uint32_t>((end + KvCache::pageTokens - 1) / KvCache::pageTokens);
      bytes += spill.model.stagingShortfall(active.id, pages);
    }
    return bytes + spill.model.hotFloorShortfall(decoding);
  }
  // What a claim keeps of the room now, `owed` the lanes' first reads before it: the room less the first reads its
  // releases created. A start's (Need::afterStart) keeps too what its start, which allocated `allocated` before it, no
  // longer allocates (Model::startStateBytes).
  [[nodiscard]] uint64_t kept(uint64_t owed, std::optional<uint64_t> allocated = std::nullopt) const {
    const uint64_t created = owedReads() - std::min(owedReads(), owed);
    const uint64_t room = spill.host.governor.snapshot().headroomBytes + allocated.value_or(0);
    const uint64_t taken = created + (allocated ? spill.model.startStateBytes() : 0);
    return room - std::min(room, taken);
  }
  // Whether placement starts request `id`, of `tokens` projected tokens, Resident only where a Start claim's count
  // holds its Resident KV (Engine::residentRoom): that KV takes more than Spill's peak, its floor and the larger of its
  // history's staging and its hot block floor (Engine::placement).
  [[nodiscard]] bool residentOnTheCount(uint64_t id, uint64_t tokens) const {
    const SpillGeometry geometry = Spilling::geometry(0, {.fullBytes = kFullPageBytes, .indexBytes = kIndexPageBytes});
    const auto pages = static_cast<uint32_t>((tokens + KvCache::pageTokens - 1) / KvCache::pageTokens);
    const uint64_t reads =
        std::max(spill.model.stagingShortfall(id, pages), spill.model.hotFloorShortfall(std::span(&id, 1)));
    return geometry.residentBytes(tokens) > geometry.spillFloorBytes(tokens) + reads;
  }
  // The memory wall: the engine holds what it does now and `room` more.
  void wall(uint64_t room = 0) { spill.host.leaveRoom(room); }
  // Ticks until the lane has decoded `tokens` tokens, between commands.
  void decode(uint64_t id, size_t tokens) {
    const auto decoded = [&] { return spill.events.outputs[id].size() >= tokens && !spill.engine.commandInFlight(); };
    for (uint32_t step = 0; step < 4'000 && !decoded(); ++step) {
      static_cast<void>(spill.engine.tick(spill.now++));
      spill.tier.complete();
    }
    require(decoded(), "lane " + std::to_string(id) + " did not decode: " +
                           (spill.events.failureDetails.empty() ? std::string("no failure")
                                                                : spill.events.failureDetails.front().first) +
                           "; " + pool());
  }
  // Between commands with no transfer in flight: the lanes' writes paused until those in flight land.
  void quiet() {
    spill.allowSpillWrites(false);
    spill.runUntil([&] { return spill.tier.inFlight() == 0 && !spill.engine.commandInFlight(); },
                   "the transfers in flight did not land");
  }
  [[nodiscard]] KvMode kvMode(uint64_t id) const {
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
      if (active.id == id)
        return active.kvMode;
    return KvMode::Resident;
  }
  [[nodiscard]] std::optional<Phase> phase(uint64_t id) const {
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
      if (active.id == id)
        return active.phase;
    return std::nullopt;
  }
  // The extent ids no form holds.
  [[nodiscard]] uint32_t freeIds() const {
    uint32_t held = 0;
    for (const kv::PageForm form : {kv::PageForm::Full, kv::PageForm::Index, kv::PageForm::Slim})
      held += spill.pool.allocatedExtents(form);
    return spill.pool.pageCount() / spill.pool.extentPages() - held;
  }
  [[nodiscard]] std::string pool() const {
    return "free Full " + std::to_string(spill.pool.freePageCount(kv::PageForm::Full)) + ", free Index " +
           std::to_string(spill.pool.freePageCount(kv::PageForm::Index)) + ", extents Full " +
           std::to_string(spill.pool.allocatedExtents(kv::PageForm::Full)) + " Index " +
           std::to_string(spill.pool.allocatedExtents(kv::PageForm::Index)) + ", free ids " +
           std::to_string(freeIds()) + ", room " + std::to_string(spill.host.governor.snapshot().headroomBytes);
  }

  Spilling spill;
};

// Page ids for every extent of 256 pages, Flash-Next's, the lanes reach (FlashNextLanes).
constexpr uint32_t kPageIds = 65'536;

const ClaimSpec kNormalStart{.cls = ClaimClass::Start, .requestId = 99, .priority = RequestPriority::Normal};

// Two Normal Spill lanes decode at the memory wall: lane 1 a 3,000-page prompt, lane 2 lane 1's prompt and 1,000
// pages more (`shared`), or 4,000 pages of its own. Neither holds Index pages.
void twoLanes(Wall &lanes, bool shared) {
  lanes.startSpill(Spilling::prompt(1, 1, 3'000, 400));
  lanes.decode(1, 4);
  lanes.startSpill(Spilling::prompt(2, shared ? 1 : 2, 4'000, 400));
  lanes.decode(2, 4);
  require(!lanes.spill.cache.holdsIndexPages(1) && !lanes.spill.cache.holdsIndexPages(2),
          "a lane held Index pages before the wall");
}

// Invariants 1 and 2: a claim's count of the lanes' releases (MemoryGovernor::available) is what their takes in
// turn leave it, with no transfer in flight and the host at normal pressure: a Normal start's claim of all it counts
// is met, and the room it leaves holds that count beside the first reads (hot block floors) the releases gave the
// lanes, which the count is net of. Two lanes, on conversations of their own or on one, which then share its pages.
void claimTwoLanesCount(bool shared) {
  Wall lanes(256, kPageIds);
  Spilling &spill = lanes.spill;
  twoLanes(lanes, shared);
  lanes.wall();
  lanes.quiet();
  MemoryGovernor &governor = spill.host.governor;
  const uint64_t counted = governor.available(kNormalStart, {.bytes = 20ULL << 30, .afterStart = true});
  const uint64_t owed = lanes.owedReads();
  const uint64_t allocated = spill.model.startStateBytes();
  const ClaimId claim = governor.ledger().open(kNormalStart);
  const FreeResult freed = governor.free(claim, {.bytes = counted, .afterStart = true});
  governor.ledger().close(claim);
  const uint64_t kept = lanes.kept(owed, allocated);
  require(freed.outcome == FreeOutcome::Met && kept >= counted,
          std::string(shared ? "two lanes on one conversation" : "two lanes") + ": a start counted " +
              std::to_string(counted) + " B of their releases; a claim of that much was " +
              (freed.outcome == FreeOutcome::Met ? "met" : "not met") + ", freeing " + std::to_string(freed.nowBytes) +
              " B, of which the first reads of the lanes it gave Index pages take " +
              std::to_string(lanes.owedReads() - owed) + " B: it keeps " + std::to_string(kept) + " B");
  spill.allowSpillWrites(true);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
}
void testAClaimOfTwoLanesReleasesKeepsWhatItCounted() { claimTwoLanesCount(false); }
void testAClaimOfTwoLanesOnOneConversationKeepsWhatItCounted() { claimTwoLanesCount(true); }

// Invariants 3 and 4: a Normal start of 1,500 pages that cannot spill (the SSD cache takes no writes) beside two
// decoding Normal Spill lanes of their own conversations at the wall is placed Resident on their releases, prefills
// with no lane yielding, and between commands the Index runway holds an extent while the lanes hold clean pages and
// the room holds an Index extent.
void testAStartPlacedResidentBesideTwoLanesKeepsTheRunway() {
  Wall lanes(256, kPageIds);
  Spilling &spill = lanes.spill;
  twoLanes(lanes, false);
  spill.tier.writableFile = false;
  lanes.wall();
  spill.engine.submit(Spilling::prompt(3, 3, 1'500, 8));
  std::optional<KvMode> placed;
  std::string runway;
  spill.runUntil(
      [&] {
        if (!placed && std::ranges::count(spill.events.startIds, 3) != 0)
          placed = lanes.kvMode(3);
        const bool clean = std::ranges::any_of(std::array<uint64_t, 2>{1, 2}, [&](uint64_t id) {
          return spill.model.requests.contains(id) && spill.model.requests.at(id).resident &&
                 spill.cache.releasablePages(id, {}).clean != 0;
        });
        if (runway.empty() && !spill.engine.commandInFlight() && clean &&
            spill.host.governor.snapshot().headroomBytes >= spill.pool.extentPages() * kIndexPageBytes &&
            spill.pool.freePageCount(kv::PageForm::Index) < spill.pool.extentPages())
          runway = lanes.pool();
        return idle(spill.engine);
      },
      "the requests did not end");
  const uint64_t yields = spill.engine.snapshot().resourceSuspensions;
  require(placed == KvMode::Resident && yields == 0 && runway.empty() && spill.events.failures.empty(),
          std::string("the start was placed ") + (placed == KvMode::Resident ? "Resident" : "Spill or not at all") +
              ", " + std::to_string(yields) + " lanes yielded" +
              (runway.empty() ? std::string() : ", the runway fell short between commands: " + runway));
}

// Invariant 3: a start placed Resident on the lanes' releases (Engine::residentRoom) prefills with no lane yielding
// and none of its claims short, while the host and the wall stay: a Normal start that cannot spill (the SSD cache
// takes no writes) beside two decoding Normal Spill lanes at the wall, of their own conversations (4,100 pages, 1.74
// GB of Resident KV, within the 1.84 GB a start counts of their releases) or on one (2,300 pages, 975 MB).
void placeBesideTwoLanes(bool shared, uint32_t pages) {
  Wall lanes(256, kPageIds);
  Spilling &spill = lanes.spill;
  twoLanes(lanes, shared);
  spill.tier.writableFile = false;
  lanes.wall();
  const uint64_t counted = spill.host.governor.available(kNormalStart, {.bytes = 20ULL << 30, .afterStart = true});
  spill.engine.submit(Spilling::prompt(3, 3, pages, 8));
  std::optional<KvMode> placed;
  bool shortClaim = false;
  spill.host.governor.ledger().setDecisionLog([&](std::string_view line) {
    shortClaim = shortClaim || (line.find("in-service claim of request 3 ") != std::string_view::npos &&
                                line.find(": short") != std::string_view::npos);
  });
  spill.runUntil(
      [&] {
        if (!placed && std::ranges::count(spill.events.startIds, 3) != 0)
          placed = lanes.kvMode(3);
        return idle(spill.engine);
      },
      "the requests did not end");
  const uint64_t yields = spill.engine.snapshot().resourceSuspensions;
  require(placed != KvMode::Resident || (yields == 0 && !shortClaim && spill.events.failures.empty()),
          std::string(shared ? "two lanes on one conversation" : "two lanes") + ": a start of " +
              std::to_string(pages) + " pages beside them, which counted " + std::to_string(counted) +
              " B of their releases, was placed Resident, and then " + std::to_string(yields) +
              " lanes yielded and its claims " + (shortClaim ? "went short" : "were met"));
}
void testAStartPlacedResidentOnTwoLanesReleasesPrefillsWithoutAYield() { placeBesideTwoLanes(false, 4'100); }
void testAStartPlacedResidentOnALanesConversationPrefillsWithoutAYield() { placeBesideTwoLanes(true, 2'300); }

// Invariant 3 at a start: a start is placed in the memory it starts in. A Normal start of 600 pages beside a Resident
// lane in service backs off from a state the model refused (Engine::waitForResources), placed Resident while the room
// holds its KV. The room then falls to half its Resident KV, which holds its prefill as a Spill lane's: a move of the
// memory outside the engine's buffers that no change of the engine's own state names. Started on a placement from
// before the move, it went on Resident and converted itself in its prefill; placed in the memory its attempt finds, it
// starts Spill, and no lane converts or yields in its prefill.
void testAStartIsPlacedInTheMemoryItStartsIn() {
  Wall lanes(16, kPageIds);
  Spilling &spill = lanes.spill;
  spill.engine.submit(Spilling::prompt(1, 1, 8, 400));
  lanes.decode(1, 4);
  require(lanes.kvMode(1) == KvMode::Resident, "the lane did not start Resident");
  spill.model.deniedBegins = 1;
  spill.engine.submit(Spilling::prompt(2, 2, 600, 8));
  spill.runUntil([&] { return spill.model.deniedBegins == 0; }, "the start's state was not refused");
  // A tick of its backoff places it in the memory its refusal left.
  static_cast<void>(spill.engine.tick(spill.now++));
  spill.tier.complete();
  require(lanes.phase(2) == Phase::WaitingResources, "the start did not back off");
  spill.host.leaveRoom(600 * kFullPageBytes / 2);
  spill.runUntil([&] { return std::ranges::count(spill.events.startIds, 2) != 0; }, "the start did not start");
  const KvMode placed = lanes.kvMode(2);
  spill.runUntil([&] { return lanes.phase(2) != Phase::Prefill && !spill.engine.commandInFlight(); },
                 "the start did not leave its prefill");
  const EngineSnapshot prefilled = spill.engine.snapshot();
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(placed == KvMode::Spill && prefilled.spill.converted == 0 && prefilled.resourceSuspensions == 0 &&
              spill.events.failures.empty() && spill.events.completedCount == 2,
          std::string("the start was placed ") + (placed == KvMode::Spill ? "Spill" : "Resident") + " at the wall; " +
              std::to_string(prefilled.spill.converted) + " lanes converted and " +
              std::to_string(prefilled.resourceSuspensions) + " yielded in its prefill, and " +
              std::to_string(spill.events.completedCount) + " requests completed");
}

// Invariant 2 for a page in place: a lone Spill lane decodes at the wall, with no free Full page and no room, beside
// the two empty Index extents a lane's end left, the runway and one more. Its growth lacks a Full extent's bytes or a
// Full page in place, and its count covers the page: the lane's release moves its oldest page whose copy landed into
// those free Index pages, beside the runway (Cache::inPlaceMoves): fewer than an extent and a page of such pages move
// no extent more. A reclaim also counted the empty Index extent past the runway (Cache::releaseExtent): taken in turn,
// the reclaim went first, the release found only the runway and moved nothing, and the lane ended at its length after
// 24 of its 400 tokens. The claim's reclaims now keep those Index pages (EngineVictims::movesInto): the run frees what
// it counted, and the lane decodes on.
void testAReleaseInPlaceAfterAReclaimOfItsIndexPagesFreesWhatItCounted() {
  constexpr uint32_t kExtentPages = 16;
  Wall lanes(kExtentPages, kPageIds);
  Spilling &spill = lanes.spill;
  lanes.startSpill(Spilling::prompt(1, 1, 200, 400));
  lanes.decode(1, 4);
  lanes.quiet();
  // An earlier wall's release: the lane's pages whose copies landed in whole extents, which leaves it fewer than an
  // extent of them; then its hot block floor, while the room holds it.
  const ReleaseRule every{.takesFrom = [](uint64_t) { return true; }};
  {
    const test::ClaimScope scope(spill.host.governor, ClaimSpec{.cls = ClaimClass::InService, .requestId = 1});
    const uint32_t clean = spill.cache.releasablePages(1, every).clean;
    require(spill.cache.releasePages(1, clean / kExtentPages * kExtentPages, false, every) != 0,
            "the lane released none of its pages");
  }
  spill.allowSpillWrites(true);
  lanes.decode(1, 8);
  lanes.quiet();
  // The Index extents a lane's end left empty, and no free Full page.
  {
    const test::ClaimScope scope(spill.host.governor, ClaimSpec{.cls = ClaimClass::InService, .requestId = 1});
    const KvPageAcquisition index = spill.pool.acquirePages(2 * kExtentPages, kv::PageForm::Index);
    require(index.granted(), "the Index extents were not allocated");
    for (const uint32_t page : index.pages)
      spill.pool.releasePage(page, false);
  }
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  const uint32_t clean = spill.cache.releasablePages(1, every).clean;
  require(held.granted() && spill.pool.emptyExtents(kv::PageForm::Index) == 2 && clean != 0 && clean <= kExtentPages,
          "the lane held " + std::to_string(clean) + " pages whose copies landed beside " +
              std::to_string(spill.pool.emptyExtents(kv::PageForm::Index)) + " empty Index extents: " + lanes.pool());
  lanes.wall();
  spill.allowSpillWrites(true);
  const MemoryLedger::CountChecks before = spill.host.governor.snapshot().countChecks;
  const uint32_t indexPages = spill.cache.indexPages(1);
  spill.runUntil(
      [&] {
        return spill.events.usage.contains(1) || spill.engine.snapshot().resourceSuspensions != 0 ||
               spill.cache.indexPages(1) > indexPages;
      },
      "the lane neither moved a page in place nor ended");
  const MemoryLedger::CountChecks after = spill.host.governor.snapshot().countChecks;
  const bool ended = spill.events.usage.contains(1);
  require(after.coveredRuns != before.coveredRuns && after.overcountedRuns == before.overcountedRuns && !ended &&
              spill.engine.snapshot().resourceSuspensions == 0,
          "at the wall " + std::to_string(after.coveredRuns - before.coveredRuns) + " counts covered the lane's page, " +
              std::to_string(after.overcountedRuns - before.overcountedRuns) + " of them freed less, and the lane " +
              (ended ? "ended after " + std::to_string(spill.events.outputs[1].size()) + " tokens"
                     : spill.engine.snapshot().resourceSuspensions ? std::string("yielded")
                                                                   : std::string("moved its page in place")));
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
  spill.host.leaveRoom(std::nullopt);
  spill.runUntilIdle();
  require(spill.events.failures.empty() && spill.events.usage.at(1).second == 400, "the lane did not complete");
}

// A conversation's prompt: `pages` whole pages of tokens from `first` on, and a token more.
std::vector<uint32_t> conversation(uint32_t first, uint32_t pages) {
  std::vector<uint32_t> tokens(pages * KvCache::pageTokens + 1);
  std::iota(tokens.begin(), tokens.end(), first);
  return tokens;
}

// Invariants 2 and 5 beside transfers in flight: a page in transfer pins its extent (Cache::compactExtent), so a
// release of a Spill request's pages for bytes (Cache::releasePages) moves only the whole extents of them compaction
// can empty, which an empty extent is not, and each page it moves frees its Full bytes less its Index page's, as a
// claim's unit counts them (EngineVictims::releases). Two requests took their 12 pages in turn, of 100 B Full and 10 B
// Index pages in 4-page extents, and a third an extent of pages it left empty; the first's copies landed. While the
// second's are being written they pin every Full extent that holds pages: the release moves none and allocates
// nothing, where moving its 8 pages emptied none of those and took the pool from 2,840 B to 2,520 B, the empty extent
// leaving beside two refills of the Index runway. Once those copies land it moves the 8 and frees the 720 B its count
// gives them, and the empty extent's 400 B.
void testAReleaseBesideWritesInFlightFreesWhatItCounts() {
  test::TestKvStorage storage{1024, 100, 4, 10};
  KvPool pool{storage, 4};
  test::TestKvTier tier;
  const auto budget = std::make_shared<model::DiskBudget>(25'600);
  tier.budget = budget;
  tier.transferLimit = 1024;
  engine::Cache cache(pool, &tier, budget, CacheSpillPolicy{1, 1, false});
  require(cache.spillWritten(SpilledExtents::KeepRunway).granted(), "the Index runway was not allocated");
  const std::vector<uint32_t> first = conversation(1000, 12), second = conversation(5000, 12);
  for (const uint64_t id : {1, 2, 3}) {
    cache.beginRequest(id, KvMode::Spill);
    require(cache.commitSpill(id, 13), "a Spill request's commitment was refused");
  }
  for (uint32_t pages = 1; pages <= 12; ++pages)
    for (const uint64_t id : {1, 2})
      require(cache.ensureTokens(id, pages * KvCache::pageTokens).granted(), "a request's page was denied");
  require(cache.ensureTokens(3, 4 * KvCache::pageTokens).granted(), "the third request's pages were denied");
  cache.endRequest(3);
  cache.publishCommittedBlocks(1, first, 12 * KvCache::pageTokens, {});
  static_cast<void>(cache.writeThrough());
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  cache.publishCommittedBlocks(2, second, 12 * KvCache::pageTokens, {});
  require(cache.writeThrough() != 0, "the second request's copies were not written");
  const uint32_t extent = pool.extentPages();
  const uint32_t pages = cache.releasablePages(1, {}).clean / extent * extent;
  require(pages == 8 && pool.emptyExtents(kv::PageForm::Full) == 1,
          "the first request did not hold two extents of pages whose copies landed beside the third's empty extent");
  const uint64_t pinned = pool.allocatedBytes();
  const uint32_t movedPinned = cache.releasePages(1, pages, false, {});
  const uint64_t besideWrites = pool.allocatedBytes();
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  const uint64_t landed = pool.allocatedBytes();
  const uint32_t moved = cache.releasePages(1, pages, false, {});
  const uint64_t after = pool.allocatedBytes();
  const uint64_t counted = uint64_t{pages} * (100 - 10);
  cache.endRequest(1);
  cache.endRequest(2);
  cache.checkInvariants();
  require(movedPinned == 0 && besideWrites == pinned,
          "a release of " + std::to_string(pages) + " pages beside copies in flight in each used Full extent moved " +
              std::to_string(movedPinned) + " and took the pool from " + std::to_string(pinned) + " B to " +
              std::to_string(besideWrites) + " B");
  require(moved == pages && landed >= after + counted,
          "a release of " + std::to_string(pages) + " pages once the copies landed moved " + std::to_string(moved) +
              " and took the pool from " + std::to_string(landed) + " B to " + std::to_string(after) +
              " B, where its count gives " + std::to_string(counted) + " B");
}

// The property runs. A seed draws two to five requests as a conversation tree: each on a prefix of an earlier one
// (a fork at a random page; a third of the forks a short turn of 8 to 40 pages) or a conversation of its own, of a
// random priority, prompt and output. The first two or more start one by one, where the room holds their first
// command: Spill, but where the lanes' releases hold their Resident KV. Then the room falls to a random wall (none,
// less than an Index extent, a Full extent, two to six), and the others arrive as the lanes run: on a lane's prefix,
// where placement may start them Resident, or, the SSD cache taking no writes until they start, Resident where memory
// holds them. Commands run in one pass or, for half the seeds, in passes, and between commands the host turns
// critical, refuses what lanes do not need, or comes back, the wall moves, and the test claims at random as the
// engine's claims do: a lane's growth of bytes or of pages, a start, the host's pass at critical pressure, a need the
// host refused.
struct Violations final {
  uint64_t count = 0;
  std::vector<std::string> first;
  void add(std::string what) {
    if (first.size() < 6)
      first.push_back(std::move(what));
    ++count;
  }
};

struct PropertyTotals final {
  uint64_t seeds = 0, claims = 0, quietClaims = 0, countedClaims = 0, coveredRuns = 0, releases = 0,
           engineOvercounts = 0, residentStarts = 0, completed = 0, failed = 0;
  // Invariant 2 as the ledger sees it (a count that covered its need, takes that did not) and net of the reads the
  // releases created (notKept), each with no transfer in flight and apart with transfers; 5 (units); 4 (runway); 3
  // (placement).
  Violations overcounted, notKept, shortUnits, blockedByTransfers, notKeptInFlight, runway, placement,
      engineOvercounted, failures;
};

class TreeRun final {
public:
  TreeRun(uint32_t seed, PropertyTotals &totals, bool verbose = false)
      : random_(seed), seed_(seed), totals_(totals), verbose_(verbose),
        lanes_(Lanes::kExtentPages, Lanes::kPageIds,
               chance(50) ? test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 4, true)
                          : std::function<void(Executor &)>{}) {
    Spilling &spill = lanes_.spill;
    spill.model.decodeTokens = model::ExecutionLimits::targetVerifyRows;
    spill.host.governor.ledger().setDecisionLog([this](std::string_view line) {
      log_.emplace_back(line);
      if (verbose_)
        std::cout << where() << ": " << line << "\n";
      if (log_.size() > 40)
        log_.erase(log_.begin());
      noteShort(line);
    });
  }

  void run() {
    Spilling &spill = lanes_.spill;
    draw();
    // The first starts, one by one, then the wall.
    const uint32_t first = pick(2, static_cast<uint32_t>(requests_.size()));
    for (uint32_t index = 0; index < first; ++index) {
      lanes_.start(engineRequest(requests_[index]));
      lanes_.decode(requests_[index].id, 1);
      started_.insert(requests_[index].id);
      admitted_.insert(requests_[index].id);
    }
    room_ = rooms()[pick(0, 3)];
    lanes_.wall(room_);
    size_t next = first;
    for (uint32_t step = 0; step < kSteps && !(next == requests_.size() && idle(spill.engine)); ++step) {
      tick();
      if (spill.engine.commandInFlight())
        continue;
      checkBetweenCommands(step);
      if (next < requests_.size() && chance(4))
        submit(requests_[next++]);
      if (chance(30))
        claim(step);
      if (chance(3))
        moveHost();
      if (chance(2)) {
        room_ = rooms()[pick(0, 3)];
        lanes_.wall(room_);
        memoryMoved();
        if (verbose_)
          std::cout << where() << ": the wall leaves " << room_ << " B\n";
      }
    }
    // Everything back: ample room, normal pressure, writes taken.
    while (next < requests_.size())
      submit(requests_[next++]);
    settleHost();
    spill.tier.writableFile = true;
    spill.host.leaveRoom(std::nullopt);
    for (uint32_t step = 0; step < kSteps && !idle(spill.engine); ++step)
      tick();
    require(idle(spill.engine), "seed " + std::to_string(seed_) + ": the requests did not end");
    spill.cache.checkInvariants();
    totals_.completed += spill.events.completedCount;
    totals_.failed += spill.events.failedCount;
    for (const auto &[message, _] : spill.events.failureDetails)
      totals_.failures.add("seed " + std::to_string(seed_) + ": " + message);
    totals_.releases += spill.cache.spillSnapshot().releases;
    ++totals_.seeds;
  }

private:
  // Flash-Next's extents scaled to 16 pages: a lane of a few hundred pages crosses as many extents as a Flash-Next
  // lane of a few thousand, so a seed is cheap.
  struct Lanes final {
    static constexpr uint32_t kExtentPages = 16, kPageIds = 32'768;
  };
  static constexpr uint32_t kSteps = 3'000;
  struct Planned final {
    uint64_t id = 0;
    std::vector<uint32_t> tokens;
    uint32_t output = 0;
    RequestPriority priority = RequestPriority::Normal;
  };

  uint32_t pick(uint32_t first, uint32_t last) { return std::uniform_int_distribution<uint32_t>(first, last)(random_); }
  bool chance(uint32_t percent) { return pick(1, 100) <= percent; }
  [[nodiscard]] uint64_t fullExtentBytes() const { return uint64_t{Lanes::kExtentPages} * kFullPageBytes; }
  [[nodiscard]] std::array<uint64_t, 4> rooms() const {
    const uint64_t extent = fullExtentBytes();
    return {0, Lanes::kExtentPages * kIndexPageBytes / 2, extent, extent * (2 + (seed_ % 5))};
  }

  void draw() {
    const uint32_t count = pick(2, 5);
    for (uint32_t index = 0; index < count; ++index) {
      Planned planned{.id = index + 1u,
                      .output = 8 * pick(16, 64),
                      .priority = static_cast<RequestPriority>(pick(0, 2))};
      // A fork carries on its parent's conversation; in a third of them a short turn, which a start may place
      // Resident on the parent's pages in memory.
      bool turn = false;
      if (index && chance(60)) {
        const Planned &parent = requests_[pick(0, index - 1)];
        const uint32_t parentPages = static_cast<uint32_t>(parent.tokens.size() / KvCache::pageTokens);
        planned.tokens.assign(parent.tokens.begin(),
                              parent.tokens.begin() + pick(1, parentPages) * KvCache::pageTokens);
        turn = chance(33);
      }
      const uint32_t own = (turn ? pick(8, 40) : pick(120, 480)) * KvCache::pageTokens + 1;
      for (uint32_t token = 0; token < own; ++token)
        planned.tokens.push_back(100'000 * (index + 1) + token % 50'000);
      requests_.push_back(std::move(planned));
    }
  }
  [[nodiscard]] static EngineRequest engineRequest(const Planned &planned) {
    EngineRequest result = test::request(planned.id, planned.tokens);
    result.maxNewTokens = planned.output;
    result.priority = planned.priority;
    result.deadlineMilliseconds = 1e9;
    return result;
  }

  // A later request: on a lane's prefix, where placement decides; or, in a third of them, while the SSD cache takes
  // no writes, until it starts, so that it starts Resident only where memory holds it.
  void submit(const Planned &planned) {
    Spilling &spill = lanes_.spill;
    if (chance(33)) {
      spill.tier.writableFile = false;
      closedFor_ = planned.id;
    }
    spill.engine.submit(engineRequest(planned));
  }

  void tick() {
    Spilling &spill = lanes_.spill;
    const bool quiet = spill.tier.inFlight() == 0;
    const uint32_t transfers = spill.tier.demotions + spill.tier.restores;
    const size_t suspended = spill.model.suspendedIds.size();
    const uint64_t preemptions = spill.engine.snapshot().prioritySuspensions;
    const MemoryLedger::CountChecks before = spill.host.governor.snapshot().countChecks;
    const uint32_t resumptions = spill.model.resumptions;
    static_cast<void>(spill.engine.tick(spill.now++));
    // A suspended lane resumes, below a running priority too (Engine::admitQueued): its replay takes memory no
    // placement counted, which counts the lanes in service alone (Engine::laneDemand), as a move of the wall does.
    if (spill.model.resumptions != resumptions) {
      memoryMoved();
      if (verbose_)
        std::cout << where() << ": a suspended lane resumed\n";
    }
    if (spill.engine.snapshot().prioritySuspensions != preemptions)
      preempted_.insert(spill.model.suspendedIds.begin() + static_cast<std::ptrdiff_t>(suspended),
                        spill.model.suspendedIds.end());
    const MemoryLedger::CountChecks after = spill.host.governor.snapshot().countChecks;
    const uint64_t overcounted = after.overcountedRuns - before.overcountedRuns;
    totals_.engineOvercounts += overcounted;
    if (overcounted && quiet && transfers == spill.tier.demotions + spill.tier.restores)
      totals_.engineOvercounted.add(where() + ": " + lanes_.pool() + "; " + lastLog());
    spill.tier.complete();
    noteStarts();
  }

  // The host or the wall moved, or a suspended lane resumed: memory no count held.
  void memoryMoved() {
    ++epoch_;
    placed_.clear();
  }

  // The starts placed Resident beside a lane at normal pressure only on a Start claim's count, the lanes' releases
  // among it (Wall::residentOnTheCount), judged while their counts hold. A start is placed as admission gives it its
  // lane and model state (Engine::admit), which for one that restores its chain from the SSD comes commands of the
  // other lanes before its start (Engine::completeAdmission).
  void noteStarts() {
    Spilling &spill = lanes_.spill;
    for (const uint64_t id : spill.events.startIds) {
      if (!started_.insert(id).second)
        continue;
      if (closedFor_ == id) {
        spill.tier.writableFile = true;
        closedFor_ = 0;
      }
      if (verbose_)
        std::cout << where() << ": request " << id << " started "
                  << (lanes_.kvMode(id) == KvMode::Resident ? "Resident" : "Spill") << "\n";
    }
    for (const Planned &planned : requests_) {
      const uint64_t id = planned.id;
      // An attempt that left the model before its start places the request again at the next.
      if (!spill.model.requests.contains(id)) {
        if (!started_.contains(id))
          admitted_.erase(id);
        continue;
      }
      if (!admitted_.insert(id).second)
        continue;
      placed_.erase(id);
      const bool beside = std::ranges::any_of(spill.model.requests, [&](const auto &entry) {
        return entry.first != id && entry.second.resident;
      });
      const uint64_t tokens = planned.tokens.size() + std::min<uint64_t>(planned.output, kProjectedOutputTokens);
      // A start of a higher priority than the starts placed Resident before it takes memory before theirs, which
      // their counts did not hold: its claims rank above theirs, and they are the lanes that yield first
      // (ClaimRank, Engine::laneToYield), or convert first where they can.
      std::erase_if(placed_, [&](uint64_t placed) { return requests_.at(placed - 1).priority > planned.priority; });
      if (lanes_.kvMode(id) == KvMode::Resident && beside && host_ == Host::Normal &&
          lanes_.residentOnTheCount(id, tokens)) {
        placed_.insert(id);
        ++totals_.residentStarts;
      }
    }
    // A start placed Resident yields, goes short or converts itself to Spill in its own prefill only where the host or
    // the wall moved since, a suspended lane resumed or a start of a higher priority began. Only its own growth
    // converts it, never the test's claims (Engine::EngineVictims::conversions).
    for (auto entry = placed_.begin(); entry != placed_.end();) {
      const uint64_t id = *entry;
      const std::optional<Phase> phase = lanes_.phase(id);
      const bool suspended = std::ranges::count(spill.model.suspendedIds, id) != 0 && !preempted_.contains(id);
      const bool converted = lanes_.kvMode(id) == KvMode::Spill;
      const bool refused = suspended || converted || shortClaims_.contains(id);
      if (refused)
        totals_.placement.add(where() + ": request " + std::to_string(id) + " placed Resident " +
                              (suspended ? "yielded" : converted ? "converted" : "went short") + " in its prefill; " +
                              lanes_.pool() + "; " + lastLog());
      // Judged through its restore (queued until it lands) and its prefill, while it holds its lane.
      if (refused || !spill.model.requests.contains(id) || !phase ||
          (*phase != Phase::Queued && *phase != Phase::Prefill))
        entry = placed_.erase(entry);
      else
        ++entry;
    }
    shortClaims_.clear();
  }
  // A lane's own claim in service that went short; the test's claims are not its.
  void noteShort(std::string_view line) {
    constexpr std::string_view of = "in-service claim of request ";
    const size_t at = line.find(of);
    if (claiming_ || at == std::string_view::npos || line.find(": short") == std::string_view::npos)
      return;
    shortClaims_.insert(std::stoull(std::string(line.substr(at + of.size()))));
  }

  // Invariant 4: between commands, while a Spill lane holds pages a release may move and the room holds an Index
  // extent, no waiting claim holding it, at normal pressure, the free Index pages hold an extent once a control pass
  // could refill them: short at two checks in a row, the wall and the host unmoved between them.
  void checkBetweenCommands(uint32_t step) {
    Spilling &spill = lanes_.spill;
    if (step % 16 == 0)
      spill.cache.checkInvariants();
    const auto releasable = [&](const ActiveRequestSnapshot &active) {
      return active.kvMode == KvMode::Spill && spill.model.requests.contains(active.id) &&
             spill.model.requests.at(active.id).resident && spill.cache.releasablePages(active.id, {}).clean != 0;
    };
    const bool clean = std::ranges::any_of(spill.engine.snapshot().activeRequests, releasable);
    const bool shortRunway = host_ == Host::Normal && clean && !spill.host.governor.ledger().claimWaiting() &&
                             spill.host.governor.snapshot().headroomBytes >= Lanes::kExtentPages * kIndexPageBytes &&
                             spill.pool.freePageCount(kv::PageForm::Index) < spill.pool.extentPages();
    if (shortRunway && runwayShortAt_ == epoch_)
      totals_.runway.add(where() + ": " + lanes_.pool() + "; " + lastLog());
    runwayShortAt_ = shortRunway ? std::optional(epoch_) : std::nullopt;
  }

  // The host turns critical (and its pass takes what it may), refuses what lanes do not need, or comes back.
  enum class Host : uint8_t { Normal, Critical, Refusing };
  void moveHost() {
    Spilling &spill = lanes_.spill;
    settleHost();
    switch (pick(0, 2)) {
    case 0:
      break;
    case 1:
      host_ = Host::Critical;
      spill.host.setPressure(MemoryPressure::Critical);
      static_cast<void>(test::criticalPass(spill.host.governor));
      lanes_.wall(room_);
      break;
    case 2:
      host_ = Host::Refusing;
      spill.host.pause(true);
      break;
    }
    memoryMoved();
    if (verbose_)
      std::cout << where() << ": the host is "
                << (host_ == Host::Normal ? "normal" : host_ == Host::Critical ? "critical" : "refusing") << "\n";
  }
  void settleHost() {
    Spilling &spill = lanes_.spill;
    if (host_ == Host::Critical)
      spill.host.setPressure(MemoryPressure::Normal);
    if (host_ == Host::Refusing)
      spill.host.pause(false);
    host_ = Host::Normal;
  }

  // A claim the test makes between commands as one of the engine's: a lane's growth of bytes or of pages in place, a
  // start, the host's pass at critical pressure, or a need the host refused, its lane's pages in place. What it frees
  // stays as room, which no claim takes, as the claim would take its need.
  void claim(uint32_t step) {
    Spilling &spill = lanes_.spill;
    MemoryGovernor &governor = spill.host.governor;
    MemoryLedger &ledger = governor.ledger();
    std::vector<const ActiveRequestSnapshot *> lanes;
    const EngineSnapshot snapshot = spill.engine.snapshot();
    for (const ActiveRequestSnapshot &active : snapshot.activeRequests)
      if (spill.model.requests.contains(active.id) && spill.model.requests.at(active.id).resident)
        lanes.push_back(&active);
    const uint32_t kind = pick(0, 4);
    if (lanes.empty() && kind != 2 && kind != 3)
      return;
    const ActiveRequestSnapshot *lane =
        lanes.empty() ? nullptr : lanes[pick(0, static_cast<uint32_t>(lanes.size()) - 1)];
    const auto rank = [](RequestPriority priority) {
      return ClaimRank{.priority = static_cast<uint8_t>(2 - static_cast<uint8_t>(priority))};
    };
    const RequestPriority priority = lane ? lane->priority : static_cast<RequestPriority>(pick(0, 2));
    ClaimSpec spec{.cls = ClaimClass::InService, .requestId = lane ? lane->id : 0, .rank = rank(priority),
                   .priority = priority};
    Need need;
    const char *what = "";
    switch (kind) {
    case 0:
      need = {.bytes = pick(1, 12) * fullExtentBytes(), .kv = true};
      what = "a lane's growth of bytes";
      break;
    case 1:
      need = {.pages = pick(1, 2 * Lanes::kExtentPages), .form = kv::PageForm::Full, .kv = true};
      what = "a lane's growth of pages";
      break;
    case 2:
      spec = {.cls = ClaimClass::Start, .requestId = 1'000 + step, .rank = rank(priority), .priority = priority};
      need = {.bytes = pick(1, 16) * fullExtentBytes(), .afterStart = true};
      what = "a start";
      break;
    case 4:
      need = {.hostRefused = true, .pages = pick(1, Lanes::kExtentPages), .form = kv::PageForm::Full, .kv = true};
      what = "a need the host refused";
      break;
    default:
      break;
    }
    const bool quiet = spill.tier.inFlight() == 0;
    const MemoryLedger::CountChecks before = governor.snapshot().countChecks;
    const std::string pool = lanes_.pool();
    // A need of bytes its count covers, with no claim waiting to hold room above it: what the claim keeps of the
    // room once the lanes' first reads its releases created are set aside, a start's with what its start no longer
    // allocates, holds the need.
    const bool counts = (kind == 0 || kind == 2) && host_ == Host::Normal && !ledger.claimWaiting() &&
                        governor.available(spec, need) >= need.bytes;
    const uint64_t owed = lanes_.owedReads();
    const std::optional<uint64_t> allocated =
        need.afterStart ? std::optional(spill.model.startStateBytes()) : std::nullopt;
    FreeResult freed;
    claiming_ = true;
    if (kind == 3) {
      what = "the host's pass at critical pressure";
      const bool critical = host_ == Host::Critical;
      if (!critical)
        spill.host.setPressure(MemoryPressure::Critical);
      freed = test::hostPass(governor, pick(1, 6) * fullExtentBytes(), true);
      if (!critical)
        spill.host.setPressure(MemoryPressure::Normal);
    } else {
      const ClaimId id = ledger.open(spec);
      freed = governor.free(id, need);
      ledger.close(id);
    }
    claiming_ = false;
    const MemoryLedger::CountChecks after = governor.snapshot().countChecks;
    ++totals_.claims;
    totals_.quietClaims += quiet;
    totals_.coveredRuns += after.coveredRuns - before.coveredRuns;
    const std::string detail = where() + ": " + what + (quiet ? "" : " (transfers in flight)") + ", outcome " +
                               std::to_string(static_cast<int>(freed.outcome)) + ", freed " +
                               std::to_string(freed.nowBytes) + " B " + std::to_string(freed.pages) +
                               " pages; before: " + pool + "; " + lastLog();
    if (verbose_)
      std::cout << "claim: " << detail << "\n";
    if (after.overcountedRuns != before.overcountedRuns)
      (quiet ? totals_.overcounted : totals_.blockedByTransfers).add(detail);
    if (after.shortReleases != before.shortReleases && quiet)
      totals_.shortUnits.add(detail);
    // Met, the claim takes its need from the room; one that waits or falls short asks again.
    if (counts && freed.outcome == FreeOutcome::Met) {
      const uint64_t kept = lanes_.kept(owed, allocated);
      ++totals_.countedClaims;
      if (kept < need.bytes)
        (quiet ? totals_.notKept : totals_.notKeptInFlight)
            .add(detail + "; kept " + std::to_string(kept) + " B of a need of " + std::to_string(need.bytes) +
                 " B, the reads it created taking " +
                 std::to_string(lanes_.owedReads() - std::min(lanes_.owedReads(), owed)));
    }
  }

  [[nodiscard]] std::string where() const {
    return "seed " + std::to_string(seed_) + " tick " + std::to_string(static_cast<uint64_t>(lanes_.spill.now));
  }
  [[nodiscard]] std::string lastLog() const {
    std::string lines;
    for (size_t index = log_.size() > 3 ? log_.size() - 3 : 0; index < log_.size(); ++index)
      lines += (lines.empty() ? "" : " | ") + log_[index];
    return lines;
  }

  std::mt19937 random_;
  uint32_t seed_;
  PropertyTotals &totals_;
  // Every decision, start and claim printed as it comes (SPLASH_RELEASE_SEED).
  bool verbose_;
  Wall lanes_;
  std::vector<Planned> requests_;
  std::set<uint64_t> started_;
  // The requests admission placed and gave a lane (noteStarts), the first starts among them.
  std::set<uint64_t> admitted_;
  // The starts placed Resident beside a lane on the count whose counts hold: no memory they did not count moved since
  // their placement (noteStarts).
  std::set<uint64_t> placed_;
  // The moves of the host and the wall, a suspended lane's resume among them.
  uint64_t epoch_ = 0;
  // The wall's and the host's moves at the last check that found the runway short.
  std::optional<uint64_t> runwayShortAt_;
  std::set<uint64_t> shortClaims_;
  // The requests a start of a higher priority preempted.
  std::set<uint64_t> preempted_;
  bool claiming_ = false;
  uint64_t room_ = 0;
  uint64_t closedFor_ = 0;
  Host host_ = Host::Normal;
  std::vector<std::string> log_;
};

void report(std::string_view name, const Violations &violations, std::string &failures) {
  if (!violations.count)
    return;
  failures += "\n  " + std::string(name) + ": " + std::to_string(violations.count);
  for (const std::string &first : violations.first)
    failures += "\n    " + first;
}

// Invariants 2 to 5 over SPLASH_RELEASE_SEEDS seeds (20 by default; SPLASH_RELEASE_SEED runs one, printing its
// decisions): with no transfer in flight, a claim's count that covered its need is met by its takes, the ledger's
// own runs in the engine's ticks too, and what a met claim keeps of the room beside the first reads its releases
// created holds its need; a release taken first frees what it counted; between commands at normal pressure the Index
// runway holds an extent where the room does; a start placed Resident beside the lanes on their count prefills with
// no yield, no short claim and no conversion while the host and the wall stay, no suspended lane resumes and no start
// of a higher priority begins. With transfers in flight they are counted apart.
void testRandomConversationTreesKeepTheReleaseCounts() {
  const char *seeds = std::getenv("SPLASH_RELEASE_SEEDS");
  const char *one = std::getenv("SPLASH_RELEASE_SEED");
  const uint32_t first = one ? static_cast<uint32_t>(std::stoul(one)) : 1;
  const uint32_t last = one ? first : seeds ? static_cast<uint32_t>(std::stoul(seeds)) : 20;
  PropertyTotals totals;
  for (uint32_t seed = first; seed <= last; ++seed) {
    TreeRun run(seed, totals, one != nullptr);
    run.run();
  }
  std::cout << "release trees: " << totals.seeds << " seeds, " << totals.claims << " claims (" << totals.quietClaims
            << " with no transfer in flight, " << totals.countedClaims << " of bytes their count covered), "
            << totals.coveredRuns << " taken on their count; " << totals.releases << " releases; "
            << totals.residentStarts << " starts placed Resident beside lanes on the count; " << totals.engineOvercounts
            << " of the engine's runs overcounted; " << totals.completed << " completed, " << totals.failed
            << " failed\n";
  std::string failures;
  report("invariant 2, a count that covered its need, takes that did not (no transfer in flight)", totals.overcounted,
         failures);
  report("invariant 2, a count that covered its need, less kept beside the reads its releases created (no transfer "
         "in flight)",
         totals.notKept, failures);
  report("invariant 2, the engine's own runs in ticks with no transfer", totals.engineOvercounted, failures);
  report("invariant 5, a release taken first freed less than it counted", totals.shortUnits, failures);
  report("invariant 4, the Index runway short of an extent between commands", totals.runway, failures);
  report("invariant 3, a start placed Resident yielded, went short or converted", totals.placement, failures);
  std::string apart;
  report("a count not met, transfers in flight", totals.blockedByTransfers, apart);
  report("a count not kept, transfers in flight", totals.notKeptInFlight, apart);
  report("requests that failed", totals.failures, apart);
  if (!apart.empty())
    std::cout << "release trees, apart:" << apart << "\n";
  require(failures.empty(), "release trees broke an invariant:" + failures);
  require(one || (totals.releases != 0 && totals.countedClaims != 0),
          "no tree released a lane's pages on a counted claim");
}

} // namespace

std::span<const test::TestCase> releaseTests() {
  static constexpr test::TestCase kTests[] = {
      {"testAClaimOfTwoLanesReleasesKeepsWhatItCounted", testAClaimOfTwoLanesReleasesKeepsWhatItCounted},
      {"testAClaimOfTwoLanesOnOneConversationKeepsWhatItCounted",
       testAClaimOfTwoLanesOnOneConversationKeepsWhatItCounted},
      {"testAStartPlacedResidentBesideTwoLanesKeepsTheRunway", testAStartPlacedResidentBesideTwoLanesKeepsTheRunway},
      {"testAStartPlacedResidentOnTwoLanesReleasesPrefillsWithoutAYield",
       testAStartPlacedResidentOnTwoLanesReleasesPrefillsWithoutAYield},
      {"testAStartPlacedResidentOnALanesConversationPrefillsWithoutAYield",
       testAStartPlacedResidentOnALanesConversationPrefillsWithoutAYield},
      {"testAStartIsPlacedInTheMemoryItStartsIn", testAStartIsPlacedInTheMemoryItStartsIn},
      {"testAReleaseInPlaceAfterAReclaimOfItsIndexPagesFreesWhatItCounted",
       testAReleaseInPlaceAfterAReclaimOfItsIndexPagesFreesWhatItCounted},
      {"testAReleaseBesideWritesInFlightFreesWhatItCounts", testAReleaseBesideWritesInFlightFreesWhatItCounts},
      {"testRandomConversationTreesKeepTheReleaseCounts", testRandomConversationTreesKeepTheReleaseCounts},
  };
  return kTests;
}
