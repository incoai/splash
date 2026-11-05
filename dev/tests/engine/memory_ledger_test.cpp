// The memory ledger and the governor's rule over it: accounts, claims and the room they hold, the host rule,
// free(), available(), growth and the control pass, on fakes.

#include "FakeElasticPool.hpp"
#include "FakeMemory.hpp"
#include "TestChecks.hpp"
#include "TestMetalMemory.hpp"
#include "TestRegistry.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryLedger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using test::FakeElasticPool;
using test::FakeMemory;
using test::HostEvent;
using test::require;

constexpr uint64_t kGiB = 1ULL << 30;
constexpr uint64_t kMiB = 1ULL << 20;

// An allocation the test expects granted.
void require(const metal::AllocationResult &granted, std::string_view message) {
  test::require(static_cast<bool>(granted), message);
}

// Units in the order the test lists them. A unit that frees memory now gives
// its bytes back to FakeMemory's holder when taken; one that lands later
// does when the test lands it.
class TestSource final : public VictimSource {
public:
  explicit TestSource(std::string name, FakeMemory *memory = nullptr) : name_(std::move(name)), memory_(memory) {}

  [[nodiscard]] std::string_view name() const noexcept override { return name_; }
  // Only the units the query's rights take (VictimSource): units in use,
  // the pass command, the buffers a lane starts from and a conversion of the
  // claimant's own lane (its key) only for a claim whose rights take them.
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &query, const Victim *after) const override {
    ++peeks;
    peeked.push_back(after ? after->key : 0);
    const auto takes = [&](const Victim &unit) {
      switch (unit.kind) {
      case VictimKind::Memory: return unit.cls != ReclaimClass::InUse || query.rights.inUse;
      case VictimKind::Conversion: return unit.key == query.rights.convertLane;
      case VictimKind::PassCommand: return query.rights.passCommand;
      case VictimKind::LaneStart: return query.rights.laneStart == TakeRights::LaneStart::Last;
      }
      return false;
    };
    auto unit = units.begin();
    if (after)
      unit = std::next(std::ranges::find(units, after->key, &Victim::key));
    while (unit != units.end() && !takes(*unit))
      ++unit;
    if (unit == units.end())
      return std::nullopt;
    return *unit;
  }
  // A unit goes whole: its pages of the need's form and its state buffers.
  FreeResult take(const VictimQuery &query, const Victim &victim) override {
    const auto unit = std::ranges::find(units, victim.key, &Victim::key);
    if (unit == units.end())
      throw std::logic_error("took a unit the source does not hold");
    taken.push_back(unit->key);
    const Victim gone = *unit;
    units.erase(unit);
    FreeResult freed{.outcome = FreeOutcome::Met,
                     .pages = gone.form == query.need.form ? gone.pages : 0,
                     .stateBuffers = gone.stateBuffers};
    if (gone.later) {
      landing.push_back(gone);
      freed.laterBytes = gone.bytes;
      return freed;
    }
    freed.nowBytes = gone.sized ? gone.bytes : unsizedBytes;
    if (memory_ && freed.nowBytes)
      memory_->release(name_, freed.nowBytes);
    return freed;
  }
  // The units taken to land later free their memory. With
  // landsIntoExtents, as the cache's demotions do, a landing only empties
  // the page's extent: a unit of its own that costs nothing, which a later
  // take releases.
  void land() {
    for (const Victim &unit : landing) {
      if (landsIntoExtents)
        units.push_back(Victim{.key = 1000 + unit.key, .bytes = unit.bytes});
      else if (memory_ && unit.bytes)
        memory_->release(name_, unit.bytes);
    }
    landing.clear();
  }

  std::vector<Victim> units;
  std::vector<uint64_t> taken;
  std::vector<Victim> landing;
  uint64_t unsizedBytes = 0;
  bool landsIntoExtents = false;
  mutable uint32_t peeks = 0;
  // Each peek's key it asked after, 0 for the head.
  mutable std::vector<uint64_t> peeked;

private:
  std::string name_;
  FakeMemory *memory_ = nullptr;
};

// A pool whose units above its floor are listed, the last given back first,
// the fewest whose bytes reach what release() asks, as the expert store
// gives its last live chunk back, whatever its size (ExpertStorePool).
class ListedPool final : public model::ElasticPool {
public:
  ListedPool(uint64_t floorBytes, std::vector<uint64_t> units)
      : floorBytes_(floorBytes), units_(std::move(units)) {}

  [[nodiscard]] model::PoolState state() const override {
    const uint64_t current = currentBytes();
    return {.name = "experts",
            .floorBytes = floorBytes_,
            .currentBytes = current,
            .capBytes = current,
            .releaseUnitBytes = units_.empty() ? 0 : units_.back(),
            .releaseUnits = {units_.rbegin(), units_.rend()}};
  }
  [[nodiscard]] uint64_t release(uint64_t bytes) override {
    uint64_t released = 0;
    while (released < bytes && !units_.empty()) {
      released += units_.back();
      units_.pop_back();
    }
    return released;
  }
  [[nodiscard]] metal::AllocationResult grow(const metal::AllocationAdmission &) override { return {}; }

  [[nodiscard]] uint64_t currentBytes() const noexcept {
    uint64_t bytes = floorBytes_;
    for (const uint64_t unit : units_)
      bytes += unit;
    return bytes;
  }

private:
  uint64_t floorBytes_;
  std::vector<uint64_t> units_;
};

// A cached unit; `later`: one that frees only once its write lands.
Victim unit(uint64_t key, uint64_t bytes, bool later = false) {
  return {.key = key, .bytes = bytes, .write = later, .later = later};
}

// A host of 64 GiB whose headroom, its spare memory above the governor's
// 2 GiB spare reserve, is `headroom` while the engine holds nothing.
FakeMemory::Config host(uint64_t headroom, MemoryPressure pressure = MemoryPressure::Normal,
                        std::vector<HostEvent> later = {}) {
  std::vector<HostEvent> script{{0.0, 62 * kGiB - headroom, pressure}};
  script.insert(script.end(), later.begin(), later.end());
  return {.hostBytes = 64 * kGiB, .otherBytes = 62 * kGiB - headroom, .script = std::move(script)};
}
HostEvent at(double milliseconds, uint64_t headroom, MemoryPressure pressure = MemoryPressure::Normal) {
  return {milliseconds, 62 * kGiB - headroom, pressure};
}

// The governor over FakeMemory's host, told macOS's pressure as the control
// pass tells it.
struct Fixture final {
  explicit Fixture(FakeMemory::Config config, uint64_t limit = 40 * kGiB)
      : memory(std::move(config)), governor(test::governorOver(backend, memory, {.limitBytes = limit})) {
    governor->setPressure(memory.pressure());
  }
  MemoryLedger &ledger() { return governor->ledger(); }
  // An allocation of `holder` for `claim`, in its scope.
  metal::AllocationResult allocate(ClaimId claim, uint64_t bytes, std::string_view holder = "kv") {
    governor->setPressure(memory.pressure());
    const MemoryGovernor::Scope scope(*governor, claim);
    return memory.allocate(governor->allocationAdmission(), holder, bytes);
  }
  // Runs the control pass at `seconds`.
  bool pass(double seconds) {
    memory.advanceTo(seconds * 1000.0);
    governor->setPressure(memory.pressure());
    return governor->controlPass(seconds * 1000.0);
  }
  // The allocation as the engine makes it: refused, free() what it lacks
  // and try again, while free() meets it.
  metal::AllocationResult allocateFreeing(ClaimId claim, uint64_t bytes) {
    metal::AllocationResult result = allocate(claim, bytes);
    for (int attempt = 0; !result && attempt < 3; ++attempt) {
      if (governor->free(claim, governor->need(claim)).outcome != FreeOutcome::Met)
        break;
      result = allocate(claim, bytes);
    }
    return result;
  }

  FakeMemory memory;
  metal::MetalBackend backend{""};
  std::optional<MemoryGovernor> governor;
};

ClaimSpec request(uint64_t id, RequestPriority priority, ClaimClass cls = ClaimClass::InService) {
  return {.cls = cls,
          .requestId = id,
          .rank = {.priority = static_cast<uint8_t>(2 - static_cast<uint8_t>(priority)),
                   .earliness = std::numeric_limits<uint64_t>::max() - id},
          .priority = priority};
}

// A lane's claim for its own KV growth, the one claim that may convert it
// (ClaimSpec::kvGrowth).
ClaimSpec growth(uint64_t id, RequestPriority priority) {
  ClaimSpec spec = request(id, priority);
  spec.kvGrowth = true;
  return spec;
}

// Leaves a request's claim waiting for `need`, which its own lane's owed
// copies meet and no other claim counts: the free() it waits on holds it.
void waitForItsCopies(MemoryLedger &ledger, ClaimId claim, const Need &need) {
  const uint64_t id = ledger.claim(claim).spec.requestId;
  Returning own{.owedBytes = need.bytes};
  own.owedPages[static_cast<size_t>(need.form)] = need.pages;
  ledger.setReturning([] { return Returning{}; }, [=](std::span<const uint64_t> lanes) {
    return std::ranges::find(lanes, id) != lanes.end() ? own : Returning{};
  });
  require(ledger.free(claim, need, 0, false).outcome == FreeOutcome::Pending && ledger.claim(claim).waiting,
          "a claim its lane's owed copies cover did not wait");
}

uint64_t accountCurrent(const MemoryGovernorSnapshot &snapshot, std::string_view name) {
  for (const MemoryAccountSnapshot &account : snapshot.accounts)
    if (account.name == name)
      return account.currentBytes;
  throw std::runtime_error("no account " + std::string(name));
}

// Over seeded sequences of allocations and releases, the accounts sum to the charged bytes, the fixed account
// holding the rest.
void testAccountsSumToCharged() {
  for (uint64_t seed = 1; seed <= 200; ++seed) {
    std::mt19937_64 random(seed);
    Fixture fixture(host(32 * kGiB));
    FakeElasticPool experts("experts", fixture.memory, 256 * kMiB, 2, 8, 16, 4);
    fixture.ledger().addAccount("kv", [&] { return fixture.memory.held("kv"); });
    fixture.ledger().addAccount("states", [&] { return fixture.memory.held("states"); });
    fixture.ledger().addPool(experts);
    for (int step = 0; step < 60; ++step) {
      const char *holders[] = {"kv", "states", "weights"};
      const char *holder = holders[random() % 3];
      const uint64_t bytes = (1 + random() % 64) * kMiB;
      switch (random() % 4) {
      case 0:
        fixture.memory.hold(holder, bytes);
        break;
      case 1:
        if (fixture.memory.held(holder) >= bytes)
          fixture.memory.release(holder, bytes);
        break;
      case 2:
        static_cast<void>(experts.release(1));
        break;
      default: {
        // The pool grows inside a growth claim's scope, as the ledger grows it.
        const ClaimId growth = fixture.ledger().open({.cls = ClaimClass::Growth});
        {
          const MemoryGovernor::Scope scope(*fixture.governor, growth);
          static_cast<void>(experts.grow(fixture.governor->allocationAdmission()));
        }
        fixture.ledger().close(growth);
      }
      }
      const MemoryGovernorSnapshot snapshot = fixture.governor->snapshot();
      uint64_t sum = 0;
      for (const MemoryAccountSnapshot &account : snapshot.accounts)
        sum += account.currentBytes;
      require(sum == snapshot.chargedBytes, "seed " + std::to_string(seed) + ": the accounts do not sum to charged");
      require(accountCurrent(snapshot, "fixed") == fixture.memory.held("weights") &&
                  accountCurrent(snapshot, "experts") == fixture.memory.held("experts") &&
                  accountCurrent(snapshot, "kv") == fixture.memory.held("kv"),
              "an account does not pull its owner's bytes");
    }
  }
}

// Each account renders its floor, current, cap, unit and moves; owed bytes come from what is returning, claims from
// the waiting ones.
void testSnapshotRendersEveryAccount() {
  Fixture fixture(host(32 * kGiB));
  fixture.memory.hold("kv", 3 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 4, 3);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addAccount("kv", [&] { return fixture.memory.held("kv"); });
  ledger.addPool(experts);
  ledger.setReturning([] { return Returning{.landingBytes = 3 * kMiB, .owedBytes = 5 * kMiB, .transfers = 2,
                                            .oldestMilliseconds = 40.0}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  ledger.setNow(10.0);
  const ClaimId waiting = ledger.open(request(7, RequestPriority::Normal));
  require(ledger.free(waiting, {.bytes = 2 * kMiB}, 0, false).outcome == FreeOutcome::Pending,
          "a claim the landings cover did not wait");
  ledger.setNow(14.0);
  const MemoryGovernorSnapshot snapshot = fixture.governor->snapshot();
  require(snapshot.accounts.size() == 3 && snapshot.accounts[0].name == "fixed" && snapshot.accounts[1].name == "kv" &&
              snapshot.accounts[2].name == "experts",
          "the accounts are not fixed, kv and the pool in order");
  const MemoryAccountSnapshot &pool = snapshot.accounts[2];
  require(pool.floorBytes == 2 * kGiB && pool.currentBytes == 3 * kGiB && pool.capBytes == 4 * kGiB &&
              pool.unitBytes == kGiB,
          "the pool's row does not render its state");
  require(snapshot.owed.transfers == 2 && snapshot.owed.bytes == 8 * kMiB && snapshot.owed.toClaimsBytes == 2 * kMiB &&
              snapshot.owed.oldestMilliseconds == 40.0,
          "owed bytes do not render what is returning");
  require(snapshot.claims.waiting == 1 && snapshot.claims.bytes == 2 * kMiB &&
              std::abs(snapshot.claims.oldestMilliseconds - 4000.0) < 1e-6,
          "the waiting claim does not render");
}

// A waiting claim holds what it lacks against every claim ranked below it (lower priority, or a later submission at
// its priority).
void testWaitingClaimHoldsRoomForItsRank() {
  Fixture fixture(host(32 * kGiB), 10 * kGiB);
  fixture.memory.hold("weights", 9 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  ledger.setReturning([] { return Returning{.landingBytes = 2 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  const auto waits = [&](ClaimId claim) {
    return fixture.governor->free(claim, fixture.governor->need(claim)).outcome == FreeOutcome::Pending;
  };
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  const ClaimId low = ledger.open(request(2, RequestPriority::Background));
  require(!fixture.allocate(high, 2 * kGiB), "2 GiB fit in 1 GiB of room");
  require(waits(high), "a claim the landings cover did not wait");
  require(fixture.allocate(low, 512 * kMiB).failure == metal::AllocationFailure::EngineBudget,
          "a lower-ranked claim took room a waiting claim lacks");
  require(fixture.allocate(high, 512 * kMiB), "the waiting claim was refused its own room");
  const ClaimId earlier = ledger.open(request(3, RequestPriority::Normal));
  const ClaimId later = ledger.open(request(4, RequestPriority::Normal));
  ledger.setWaiting(high, false);
  require(!fixture.allocate(earlier, 2 * kGiB), "2 GiB fit in the room left");
  require(waits(earlier), "a claim the landings cover did not wait");
  require(!fixture.allocate(later, 256 * kMiB), "a later submission took room an earlier one waits for");
  require(fixture.allocate(earlier, 256 * kMiB), "the earlier submission was refused room");
}

// A claim that does not wait holds nothing.
void testShortClaimHoldsNothing() {
  Fixture fixture(host(32 * kGiB), 10 * kGiB);
  fixture.memory.hold("weights", 9 * kGiB);
  const ClaimId high = fixture.ledger().open(request(1, RequestPriority::Foreground));
  const ClaimId low = fixture.ledger().open(request(2, RequestPriority::Background));
  require(!fixture.allocate(high, 2 * kGiB), "2 GiB fit in 1 GiB of room");
  require(fixture.allocate(low, 512 * kMiB), "a claim that does not wait held room");
}

// A scope restores the one it replaced: refusals go to the innermost claim, and with none open admission is as
// before.
void testNestedScopesRestore() {
  Fixture fixture(host(32 * kGiB), 10 * kGiB);
  fixture.memory.hold("weights", 9 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  const ClaimId outer = ledger.open(request(1, RequestPriority::Normal));
  const ClaimId inner = ledger.open(request(2, RequestPriority::Normal));
  {
    const MemoryGovernor::Scope a(*fixture.governor, outer);
    {
      const MemoryGovernor::Scope b(*fixture.governor, inner);
      require(!fixture.memory.allocate(fixture.governor->allocationAdmission(), "kv", 2 * kGiB), "2 GiB fit");
    }
    require(!fixture.memory.allocate(fixture.governor->allocationAdmission(), "kv", 3 * kGiB), "3 GiB fit");
  }
  require(ledger.claim(inner).refusal.bytes == 2 * kGiB && ledger.claim(outer).refusal.bytes == 3 * kGiB,
          "a refusal was recorded on the wrong claim");
  // Once they end, an admission is for the claim whose scope is open then: a
  // claim ranked above the one that waits, for what lands, is not held back
  // by it.
  ledger.setReturning([] { return Returning{.landingBytes = 3 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  require(fixture.governor->free(outer, fixture.governor->need(outer)).outcome == FreeOutcome::Pending,
          "a claim the landings cover did not wait");
  const ClaimId above = ledger.open(request(3, RequestPriority::Foreground));
  {
    const MemoryGovernor::Scope scope(*fixture.governor, above);
    require(fixture.memory.allocate(fixture.governor->allocationAdmission(), "kv", 512 * kMiB),
            "an admission of a claim ranked above the waiting one was held back by it");
  }
  ledger.close(above);
}

// A claim's id names it alone for the ledger's life: ids only grow, and are 64 bits wide, so a
// long-running server never wraps onto a claim still open.
void testClaimIdsNeverRepeat() {
  require(std::numeric_limits<decltype(ClaimId::value)>::digits == 64, "claim ids are narrower than 64 bits");
  MemoryLedger ledger;
  const ClaimId kept = ledger.open(request(1, RequestPriority::Normal));
  ClaimId last = kept;
  for (int round = 0; round < 1000; ++round) {
    const ClaimId next = ledger.open(request(2, RequestPriority::Normal));
    require(next && next.value > last.value && !(next == kept), "a claim took an id given before");
    ledger.close(next);
    last = next;
  }
  require(ledger.claims().size() == 1 && ledger.claims().contains(kept.value), "the open claim lost its id");
}

// KV pages a waiting claim lacks are held, by form, against the claims ranked below it.
void testHeldPagesFollowRank() {
  MemoryLedger ledger;
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  const ClaimId mid = ledger.open(request(2, RequestPriority::Normal));
  const ClaimId low = ledger.open(request(3, RequestPriority::Background));
  TestSource none("cache");
  ledger.addSource(none);
  waitForItsCopies(ledger, high, {.pages = 6, .form = kv::PageForm::Index});
  waitForItsCopies(ledger, mid, {.pages = 4});
  const std::array<uint32_t, 3> forLow = ledger.heldPages(low, 0);
  const std::array<uint32_t, 3> forMid = ledger.heldPages(mid, 0);
  require(forLow[static_cast<size_t>(kv::PageForm::Index)] == 6 && forLow[static_cast<size_t>(kv::PageForm::Full)] == 4,
          "a lower-ranked claim is not held the pages of both claims above it");
  require(forMid[static_cast<size_t>(kv::PageForm::Index)] == 6 && forMid[static_cast<size_t>(kv::PageForm::Full)] == 0,
          "a claim is held its own pages or a lower claim's");
}

// A waiting claim's bytes are held as each form's pages against lower-ranked claims (10,000 bytes: 3 Full pages of
// 4096, 4 Slim of 3000); a claim free() found short is held against no one, and nothing is held at or above its rank.
void testWaitingBytesHoldPagesOfEachForm() {
  MemoryLedger ledger;
  ledger.setPageBytes({4096, 0, 3000});
  const ClaimId waiting = ledger.open(request(5, RequestPriority::Normal));
  const ClaimId shortClaim = ledger.open(request(4, RequestPriority::Normal));
  const ClaimId same = ledger.open(request(5, RequestPriority::Normal));
  const ClaimId above = ledger.open(request(6, RequestPriority::Foreground));
  const ClaimId below = ledger.open(request(7, RequestPriority::Background));
  require(ledger.free(shortClaim, {.bytes = 20'000}, 0, false).outcome == FreeOutcome::Short,
          "a claim nothing covers did not find free() short");
  waitForItsCopies(ledger, waiting, {.bytes = 10'000});
  require(ledger.heldPages(below, 0) == std::array<uint32_t, 3>{3, 0, 4},
          "a lower-ranked claim was not held the pages of the bytes a claim above it waits for");
  require(ledger.heldPages(same, 0) == std::array<uint32_t, 3>{} && ledger.heldPages(above, 0) == std::array<uint32_t, 3>{},
          "a claim of the waiting claim's rank or above it was held its pages");
}

// A request's wait spans its allocations: a met free() for another of them keeps the claim waiting, a short one
// ends the wait.
void testAMetNeedKeepsTheWaitAShortOneEndsIt() {
  MemoryLedger ledger;
  TestSource cache("cache");
  Victim leaf = unit(1, 0);
  leaf.pages = 4;
  cache.units = {leaf};
  ledger.addSource(cache);
  const ClaimId waiting = ledger.open(request(1, RequestPriority::Foreground));
  const ClaimId below = ledger.open(request(2, RequestPriority::Background));
  waitForItsCopies(ledger, waiting, {.bytes = kGiB});
  require(ledger.free(waiting, {.pages = 4}, 0, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1},
          "the leaf did not meet the claim's need of pages");
  require(ledger.claim(waiting).waiting && ledger.claim(waiting).need.bytes == kGiB &&
              ledger.heldBytes(below) == kGiB,
          "a claim whose free() for another need was met stopped waiting for its copies");
  require(ledger.free(waiting, {.bytes = 2 * kGiB}, 0, false).outcome == FreeOutcome::Short &&
              !ledger.claim(waiting).waiting && ledger.heldBytes(below) == 0,
          "a claim free() found short still waits");
}

// One host rule for every class: the floor passes, in-service work needs the margin (or an exemption), elective
// work the band too, and macOS's pressure levels refuse in order.
void testOneHostRule() {
  const auto admits = [](uint64_t headroom, MemoryPressure pressure, ClaimClass cls, uint64_t bytes,
                         bool exempt = false) {
    Fixture fixture(host(headroom, pressure));
    const ClaimId claim = fixture.ledger().open({.cls = cls});
    if (exempt)
      fixture.ledger().setHostExempt(claim, true);
    return fixture.allocate(claim, bytes);
  };
  const MemoryPressure normal = MemoryPressure::Normal, warning = MemoryPressure::Warning,
                       critical = MemoryPressure::Critical;
  require(admits(512 * kMiB, warning, ClaimClass::Floor, kGiB), "the floor was refused under warning");
  require(!admits(8 * kGiB, critical, ClaimClass::Floor, kGiB), "critical pressure admitted the floor");
  require(admits(2560 * kMiB, normal, ClaimClass::InService, 1536 * kMiB), "in-service work keeping M was refused");
  require(admits(2560 * kMiB, normal, ClaimClass::InService, 1792 * kMiB).failure ==
              metal::AllocationFailure::HostPressure,
          "in-service work inside the margin was admitted unexempt");
  require(admits(512 * kMiB, warning, ClaimClass::InService, kGiB, true), "an exempt in-service claim was refused");
  require(!admits(8 * kGiB, critical, ClaimClass::InService, kGiB, true), "critical pressure admitted exempt work");
  for (const ClaimClass cls : {ClaimClass::Start, ClaimClass::Optional, ClaimClass::Growth}) {
    require(admits(2560 * kMiB, normal, cls, 1536 * kMiB), "elective work keeping the margin was refused");
    require(!admits(2560 * kMiB, normal, cls, 1792 * kMiB), "elective work inside the margin was admitted");
    require(!admits(8 * kGiB, warning, cls, kGiB), "macOS's warning admitted elective work");
  }
  // The query the engine asks of growth, with no bytes and no credit: the
  // margin, the band and macOS's warning hold it as they hold growth.
  const auto growthAllowed = [](uint64_t headroom, MemoryPressure pressure) {
    Fixture fixture(host(headroom, pressure));
    static_cast<void>(fixture.governor->evaluate());
    return fixture.governor->admits(ClaimClass::Growth);
  };
  require(growthAllowed(8 * kGiB, normal), "growth was not allowed outside the band");
  require(!growthAllowed(900 * kMiB, normal), "growth was allowed in the band");
  require(!growthAllowed(8 * kGiB, warning), "growth was allowed under macOS's warning");
  require(!growthAllowed(8 * kGiB, critical), "growth was allowed under critical pressure");
  // The band: in below M, out at M + H, H the largest growth unit. The
  // pool's two units are held on the host too.
  for (const uint64_t unit : {kGiB, 512 * kMiB}) {
    const uint64_t pool = 2 * unit;
    Fixture fixture(host(900 * kMiB + pool, normal,
                         {at(100.0, kGiB + unit - 100 * kMiB + pool), at(200.0, kGiB + unit + pool)}));
    FakeElasticPool experts("experts", fixture.memory, unit, 1, 4, 5, 2);
    fixture.ledger().addPool(experts);
    static_cast<void>(fixture.governor->evaluate());
    const ClaimId start = fixture.ledger().open({.cls = ClaimClass::Start});
    require(!fixture.allocate(start, 64 * kMiB), "a start was admitted below the margin");
    fixture.memory.advanceTo(100.0);
    require(fixture.allocate(start, 64 * kMiB).failure == metal::AllocationFailure::HostPressure,
            "the band ended short of M + H");
    fixture.memory.advanceTo(200.0);
    require(fixture.allocate(start, 64 * kMiB), "the band outlasted M + the largest unit");
  }
}

// Memory a lane in service takes anyway (ClaimSpec::laneNeeds) is judged by the host as that lane's growth: by the
// margin, not the band.
void testTheHostJudgesWhatALaneNeedsAsItsGrowth() {
  const ClaimSpec needed{.cls = ClaimClass::Optional, .requestId = 1, .laneNeeds = true};
  const ClaimSpec optional{.cls = ClaimClass::Optional, .requestId = 1};
  // In the band, past the margin by less than the largest growth unit.
  {
    const uint64_t pool = 2 * kGiB;
    Fixture fixture(host(900 * kMiB + pool, MemoryPressure::Normal, {at(100.0, 2 * kGiB - 100 * kMiB + pool)}));
    FakeElasticPool experts("experts", fixture.memory, kGiB, 1, 4, 5, 2);
    fixture.ledger().addPool(experts);
    static_cast<void>(fixture.governor->evaluate());
    fixture.memory.advanceTo(100.0);
    require(fixture.allocate(fixture.ledger().open(optional), 64 * kMiB).failure ==
                metal::AllocationFailure::HostPressure,
            "the band admitted optional work");
    require(fixture.allocate(fixture.ledger().open(needed), 64 * kMiB),
            "the band refused what a lane needs, which keeps the margin");
  }
  // Inside the margin.
  for (const MemoryPressure pressure : {MemoryPressure::Normal, MemoryPressure::Warning, MemoryPressure::Critical}) {
    Fixture fixture(host(512 * kMiB, pressure));
    const ClaimId lane = fixture.ledger().open(needed);
    require(!fixture.allocate(lane, 256 * kMiB), "what a lane needs was admitted inside the margin unexempt");
    require(static_cast<bool>(fixture.allocateFreeing(lane, 256 * kMiB)) == (pressure != MemoryPressure::Critical),
            "what a lane needs, with nothing in place to meet it, was " +
                std::string(pressure == MemoryPressure::Critical ? "admitted under critical pressure"
                                                                 : "refused past the margin"));
    require(!fixture.allocateFreeing(fixture.ledger().open(optional), 256 * kMiB),
            "optional work passed the margin");
  }
}

// The band ends once a host pass finds nothing to take: only other applications could return the rest.
void testBandWaivedWhenTheHostClaimFindsNothing() {
  // A pool of one unit at its floor that could grow one more: H = 1 GiB.
  Fixture fixture(host(800 * kMiB + kGiB, MemoryPressure::Normal, {at(100.0, 1536 * kMiB + kGiB)}));
  FakeElasticPool pool("experts", fixture.memory, kGiB, 1, 2, 3, 1);
  fixture.ledger().addPool(pool);
  static_cast<void>(fixture.governor->evaluate());
  const ClaimId start = fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(start, kGiB), "a start was admitted below the margin");
  fixture.ledger().setWaiting(start, true);
  fixture.memory.advanceTo(100.0);
  const ClaimId probe = fixture.ledger().open(request(2, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(probe, 128 * kMiB), "the band ended short of M + H");
  static_cast<void>(fixture.pass(0.1));
  require(fixture.allocate(probe, 128 * kMiB), "the band held after the host pass found nothing");
}

// A critical pass that finds nothing to take is Short while memory lands: the host's claim at critical pressure
// waits for no write, so it never stays pending.
void testACriticalPassThatFindsNothingIsShortWhileMemoryLands() {
  Fixture fixture(host(8 * kGiB, MemoryPressure::Critical));
  fixture.ledger().setReturning([] { return Returning{.landingBytes = kGiB, .transfers = 1}; },
                                [](std::span<const uint64_t>) { return Returning{}; });
  require(!fixture.pass(0.5), "a critical pass that found nothing waited for what lands");
}

// Every host pass gives back the idle state buffers, which cost nothing, whatever else it finds.
void testEveryHostPassGivesBackIdleBuffers() {
  Fixture fixture(host(512 * kMiB));
  TestSource engine("engine", &fixture.memory);
  Victim buffers = unit(1, 0);
  buffers.sized = false;
  engine.units = {buffers};
  engine.unsizedBytes = 226 * kMiB;
  fixture.memory.hold("engine", 226 * kMiB);
  fixture.ledger().addSource(engine);
  static_cast<void>(fixture.pass(1.0));
  require(engine.taken == std::vector<uint64_t>{1} && fixture.memory.total() == 0,
          "a host pass that found no sized unit kept the idle buffers");
}

// The host deficit is the margin plus what host-refused waiting claims lack: with nothing waiting a pass frees only
// what restores the margin.
void testDeficitCountsHostRefusedClaimsNotAnIdleMargin() {
  // 800 MiB of headroom beside the cache's 2 GiB: the deficit is 224 MiB of
  // M, and 512 MiB more for a start the host refused and that waits.
  for (const bool waits : {false, true}) {
    Fixture fixture(host(800 * kMiB + 2 * kGiB));
    TestSource cache("cache", &fixture.memory);
    for (uint64_t key = 1; key <= 16; ++key)
      cache.units.push_back(unit(key, 128 * kMiB));
    fixture.memory.hold("cache", 2 * kGiB);
    fixture.ledger().addSource(cache);
    if (waits) {
      const ClaimId start = fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::Start));
      require(fixture.allocate(start, 512 * kMiB).failure == metal::AllocationFailure::HostPressure,
              "the start was not refused by the host");
      fixture.ledger().setWaiting(start, true);
    }
    static_cast<void>(fixture.pass(1.0));
    require(cache.taken.size() == (waits ? 6u : 2u),
            waits ? "a waiting host-refused claim's need was not freed" : "an idle band pass freed past the margin");
  }
}

// A start the host refused is admitted within two control passes, in the band or out of it, rather than held to its
// wait limit.
void testAHostRefusedStartIsAdmittedWithinTwoPasses() {
  struct Case final {
    bool band;
    uint64_t headroom, need;
    bool later;
    const char *what;
  };
  for (const Case &scenario :
       {Case{true, 1100 * kMiB, 300 * kMiB, false, "a start the host refused waited in the band after a pass met its deficit"},
        Case{true, 1100 * kMiB, 300 * kMiB, true,
             "a start the host refused waited in the band after its deficit landed"},
        Case{true, 1200 * kMiB, 108'527'616, false,
             "a start the host refused waited in the band although the headroom past the margin held it"},
        Case{false, 1536 * kMiB, 600 * kMiB, false, "no pass freed for a start the host refused outside the band"}}) {
    const bool band = scenario.band;
    const uint64_t headroom = scenario.headroom;
    const uint64_t need = scenario.need;
    Fixture fixture(host(headroom + kGiB + 2 * kGiB));
    FakeElasticPool experts("experts", fixture.memory, kGiB, 1, band ? 2 : 1, 3, 1);
    fixture.ledger().addPool(experts);
    TestSource cache("cache", &fixture.memory);
    for (uint64_t key = 1; key <= 16; ++key)
      cache.units.push_back(unit(key, 128 * kMiB, scenario.later));
    fixture.memory.hold("cache", 2 * kGiB);
    fixture.ledger().addSource(cache);
    static_cast<void>(fixture.governor->evaluate());
    const ClaimId start = fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::Start));
    bool admitted = false;
    for (uint32_t pass = 0; pass < 2 && !admitted; ++pass) {
      // As the engine has it: the request waits to retry what the host
      // refused (Engine::syncClaims), its allocation is refused and the
      // ledger asked, then the control pass runs.
      fixture.ledger().setHostRefusedWaiting(start, true);
      const metal::AllocationResult result = fixture.allocate(start, need);
      admitted = static_cast<bool>(result);
      if (admitted)
        break;
      require(result.failure == metal::AllocationFailure::HostPressure, "the start was not refused by the host");
      static_cast<void>(fixture.governor->free(start, fixture.governor->need(start)));
      static_cast<void>(fixture.pass(1.0 + 1.5 * pass));
      cache.land();
    }
    admitted = admitted || static_cast<bool>(fixture.allocate(start, need));
    require(admitted, scenario.what);
  }
}

// A pool's growth leaves the host what the starts it refused wait for, so the room freed for a start goes to it;
// the store regrows once the start is done.
void testGrowthLeavesTheHostWhatAStartItRefusedWaitsFor() {
  constexpr uint64_t kChunk = 117 * kMiB;
  constexpr uint64_t kNeed = 300 * kMiB;
  struct Case final {
    uint64_t headroom;
    bool later;
    const char *what;
  };
  for (const Case &scenario :
       {Case{1124 * kMiB, true, "the store grew in the band into what landed for a start the host refused"},
        Case{1224 * kMiB, false, "the store grew into what a pass freed for a start the host refused"}}) {
    Fixture fixture(host(scenario.headroom + kChunk + 2 * kGiB));
    FakeElasticPool experts("experts", fixture.memory, kChunk, 1, 2, 3, 1);
    fixture.ledger().addPool(experts);
    TestSource cache("cache", &fixture.memory);
    for (uint64_t key = 1; key <= 16; ++key)
      cache.units.push_back(unit(key, 128 * kMiB, scenario.later));
    fixture.memory.hold("cache", 2 * kGiB);
    fixture.ledger().addSource(cache);
    static_cast<void>(fixture.governor->evaluate());
    const ClaimId start = fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::Start));
    fixture.ledger().setHostRefusedWaiting(start, true);
    require(fixture.allocate(start, kNeed).failure == metal::AllocationFailure::HostPressure,
            "the start was not refused by the host");
    static_cast<void>(fixture.governor->free(start, fixture.governor->need(start)));
    // Past the quiet time, where only the waiting start holds the store.
    static_cast<void>(fixture.pass(kRegrowQuietSeconds + 1.0));
    if (scenario.later) {
      cache.land();
      static_cast<void>(fixture.pass(kRegrowQuietSeconds + 2.5));
    }
    const bool admitted = static_cast<bool>(fixture.allocate(start, kNeed));
    require(admitted && experts.units() == 1, scenario.what);
    fixture.ledger().close(start);
    fixture.memory.release("kv", kNeed);
    static_cast<void>(fixture.pass(2 * kRegrowQuietSeconds + 3.0));
    require(experts.units() == 2, "the store did not grow into what a start the host refused gave back");
  }
}

// Inside a source its own order holds: a unit behind another waits for it.
void testSourceOrderIsKeptWithinASource() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB), unit(2, kGiB)};
  ledger.addSource(cache);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1},
          "free() skipped ahead inside a source's order");
}

// A source's units keyed by their place in its order: taking one moves
// every unit after it up a place and leaves those before it in place.
class PlacedSource final : public VictimSource {
public:
  [[nodiscard]] std::string_view name() const noexcept override { return "placed"; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &, const Victim *after) const override {
    const size_t place = after ? after->key + 1 : 0;
    if (place >= units.size())
      return std::nullopt;
    Victim victim = units[place].second;
    victim.key = place;
    return victim;
  }
  FreeResult take(const VictimQuery &, const Victim &victim) override {
    taken.push_back(units.at(victim.key).first);
    const Victim gone = units[victim.key].second;
    units.erase(units.begin() + static_cast<std::ptrdiff_t>(victim.key));
    return {.outcome = FreeOutcome::Met, .nowBytes = gone.bytes, .stateBuffers = gone.stateBuffers};
  }
  std::vector<std::pair<uint64_t, Victim>> units;  // the test's name for each unit, and the unit
  std::vector<uint64_t> taken;
};

// free() takes the units of a source keyed by place, each at its place after the takes before it; a unit the
// claim passes over (one that lands later, for a claim that does not wait for writes) stays.
void testUnitsKeyedByTheirPlaceAreTakenAsChosen() {
  MemoryLedger ledger;
  PlacedSource placed;
  TestSource other("other");
  placed.units = {{1, unit(0, kGiB)}, {2, unit(0, kGiB, true)}, {3, unit(0, kGiB)}, {4, unit(0, kGiB)}};
  other.units = {unit(5, kGiB)};
  ledger.addSource(placed);
  ledger.addSource(other);
  ClaimSpec spec = request(1, RequestPriority::Normal);
  spec.timing = NeedTiming::Immediate;
  const ClaimId claim = ledger.open(spec);
  require(ledger.free(claim, {.bytes = 4 * kGiB}, 0, false).outcome == FreeOutcome::Met,
          "the claim was not met");
  require(placed.taken == std::vector<uint64_t>{1, 3, 4} && other.taken == std::vector<uint64_t>{5} &&
              placed.units.size() == 1 && placed.units.front().first == 2,
          "free() did not take the units of a source keyed by place at their places");
}

// The take table, row by row (one claim per row of the rules' table): what
// each claimant may take.
void testTakeRightsTable() {
  using Reuse = TakeRights::CertainReuse;
  using Reserves = TakeRights::Reserves;
  using LaneStart = TakeRights::LaneStart;
  const auto rights = [](const ClaimSpec &spec, const Need &need = {.bytes = kMiB}, bool critical = false,
                         std::optional<RequestPriority> highest = std::nullopt) {
    return takeRights(spec, need, critical, highest);
  };
  const ClaimSpec everything{.cls = ClaimClass::InService, .requestId = 7, .reach = ReclaimClass::InUse,
                             .supersedes = true, .kvGrowth = true};
  ClaimSpec immediate = everything;
  immediate.timing = NeedTiming::Immediate;
  ClaimSpec start = everything;
  start.cls = ClaimClass::Start;
  ClaimSpec optional = immediate;
  optional.cls = ClaimClass::Optional;
  ClaimSpec below = everything;
  below.priority = RequestPriority::Background;
  struct Row final {
    std::string name;
    TakeRights actual;
    TakeRights expected;
  };
  const std::vector<Row> rows{
      {"in service, can wait", rights(everything),
       {.units = true, .waitsForWrites = true, .waitsForLandings = true, .inUse = true, .certainReuse = Reuse::All,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = true, .passCommand = true,
        .convertLane = 7}},
      {"in service, can wait, for KV", rights(everything, {.bytes = kMiB, .kv = true}),
       {.units = true, .waitsForWrites = true, .waitsForLandings = true, .inUse = true, .certainReuse = Reuse::All,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = true, .passCommand = true,
        .convertLane = 7, .laneStart = LaneStart::Last}},
      {"in service, can wait, for a state's buffers", rights(everything, {.bytes = kMiB, .stateBuffers = 1}),
       {.units = true, .waitsForWrites = true, .waitsForLandings = true, .inUse = true, .certainReuse = Reuse::All,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = true, .passCommand = true,
        .convertLane = 7}},
      {"in service, can wait, ordinary reach", rights({.cls = ClaimClass::InService}),
       {.units = true, .waitsForWrites = true, .waitsForLandings = true, .inUse = false, .certainReuse = Reuse::All,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = true, .passCommand = false,
        .convertLane = 0}},
      {"in service, can wait, ordinary reach, for KV", rights({.cls = ClaimClass::InService}, {.pages = 1, .kv = true}),
       {.units = true, .waitsForWrites = true, .waitsForLandings = true, .inUse = false, .certainReuse = Reuse::All,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = true, .passCommand = false,
        .convertLane = 0, .laneStart = LaneStart::Last}},
      {"in service, Immediate", rights(immediate, {.bytes = kMiB, .kv = true}),
       {.units = true, .waitsForWrites = false, .waitsForLandings = false, .inUse = false, .certainReuse = Reuse::All,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = false, .passCommand = false,
        .convertLane = 0}},
      {"start", rights(start, {.bytes = kMiB, .kv = true}),
       {.units = true, .waitsForWrites = true, .waitsForLandings = true, .inUse = false, .certainReuse = Reuse::ResumePoint,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = false, .passCommand = false,
        .convertLane = 0}},
      {"optional", rights(optional, {.bytes = kMiB, .kv = true}),
       {.units = true, .waitsForWrites = false, .waitsForLandings = false, .inUse = false, .certainReuse = Reuse::ResumePoint,
        .reserves = Reserves::None, .rebuiltPools = true, .storeAboveFloor = false, .passCommand = false,
        .convertLane = 0}},
      {"growth", rights({.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate}), {.units = true}},
      {"the engine's own optional work", rights({.cls = ClaimClass::Optional, .timing = NeedTiming::Immediate}), {}},
      {"the host with a deficit", rights({.cls = ClaimClass::Host}),
       {.units = true, .waitsForWrites = true, .waitsForLandings = false, .inUse = false, .certainReuse = Reuse::None,
        .reserves = Reserves::None, .rebuiltPools = true, .storeAboveFloor = false, .passCommand = false,
        .convertLane = 0}},
      {"the host at critical pressure", rights({.cls = ClaimClass::Host}, {.bytes = kMiB}, true),
       {.units = true, .waitsForWrites = false, .waitsForLandings = false, .inUse = true, .certainReuse = Reuse::All,
        .reserves = Reserves::All, .rebuiltPools = true, .storeAboveFloor = true, .passCommand = false,
        .convertLane = 0, .laneStart = LaneStart::First}},
      {"a need the host refused in service", rights(everything, {.bytes = kMiB, .hostRefused = true, .kv = true}),
       {.units = true, .waitsForWrites = false, .waitsForLandings = false, .inUse = false, .certainReuse = Reuse::All,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = false, .passCommand = false,
        .convertLane = 0}},
      {"a need the host refused a start", rights(start, {.bytes = kMiB, .hostRefused = true}),
       {.units = true, .waitsForWrites = false, .waitsForLandings = false, .inUse = false, .certainReuse = Reuse::ResumePoint,
        .reserves = Reserves::NextExtent, .rebuiltPools = true, .storeAboveFloor = false, .passCommand = false,
        .convertLane = 0}},
      {"in service below a lane", rights(below, {.bytes = kMiB, .kv = true}, false, RequestPriority::Foreground),
       {.units = true, .waitsForWrites = true, .waitsForLandings = true, .inUse = true, .certainReuse = Reuse::All,
        .reserves = Reserves::NextExtent, .rebuiltPools = false, .storeAboveFloor = false, .passCommand = true,
        .convertLane = 7, .laneStart = LaneStart::Last}},
      {"in service at critical pressure", rights(everything, {.bytes = kMiB}, true),
       {.units = true, .waitsForWrites = true, .waitsForLandings = true, .inUse = true, .certainReuse = Reuse::All,
        .reserves = Reserves::All, .rebuiltPools = true, .storeAboveFloor = true, .passCommand = true,
        .convertLane = 7, .laneStart = LaneStart::First}},
  };
  for (const Row &row : rows)
    require(row.actual == row.expected, row.name + ": the take rights differ from the table");
}

// A Floor claim is a host rule only: asking the ledger to take or count
// units for one is a caller's error, not an empty take.
void testAFloorClaimTakesNothing() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kMiB)};
  ledger.addSource(cache);
  const ClaimSpec floor{.cls = ClaimClass::Floor};
  test::rejects([&] { static_cast<void>(ledger.available(floor, {.bytes = kMiB}, 0, false)); },
                "a Floor claim takes nothing", "available() counted units for a Floor claim");
  const ClaimId claim = ledger.open(floor);
  test::rejects([&] { static_cast<void>(ledger.free(claim, {.bytes = kMiB}, 0, false)); },
                "a Floor claim takes nothing", "free() took units for a Floor claim");
  require(cache.taken.empty(), "a Floor claim took a unit");
}

// What the ledger itself takes by the rights, from units of distinct sizes:
// the pools, and cache that frees once its write lands. What else a claim
// may take its sources leave out (VictimSource).
void testThePoolsAndWritesEachClaimTakes() {
  FakeMemory memory({});
  FakeElasticPool pool("experts", memory, 32 * kMiB, 0, 1, 2, 1);
  FakeElasticPool rebuilt("vision_arena", memory, 64 * kMiB, 0, 1, 1, 1);
  rebuilt.rebuildOnly = true;
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kMiB), unit(2, 2 * kMiB, true)};
  ledger.addSource(cache);
  ledger.addPool(pool);
  ledger.addPool(rebuilt);
  const auto counted = [&](ClaimSpec spec, bool critical = false, bool hostRefused = false) {
    return ledger.available(spec, {.bytes = kGiB, .hostRefused = hostRefused}, 0, critical) / kMiB;
  };
  // 1 cache that frees now, 2 cache that needs a write, 32 a store unit, 64
  // a unit that costs only its rebuild.
  require(counted({.cls = ClaimClass::InService}) == 1 + 2 + 32 + 64, "in-service work took the wrong units");
  require(counted({.cls = ClaimClass::InService, .timing = NeedTiming::Immediate}) == 1 + 64,
          "an Immediate in-service claim took a write or the store");
  require(counted({.cls = ClaimClass::InService}, false, true) == 1 + 64,
          "a need the host refused took a write or the store");
  require(counted({.cls = ClaimClass::Start}) == 1 + 2 + 64, "a start took the store or no write");
  require(counted({.cls = ClaimClass::Optional, .requestId = 1, .timing = NeedTiming::Immediate}) == 1 + 64,
          "optional work took a write or the store");
  require(counted({.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate}) == 1,
          "growth took more than cache that frees now");
  require(counted({.cls = ClaimClass::Host}) == 1 + 2 + 64,
          "the host's deficit took the store short of critical pressure, or not the units that cost only their "
          "rebuild");
  require(counted({.cls = ClaimClass::Host}, true) == 1 + 32 + 64,
          "critical pressure did not give the host the store, or gave it a write");
  ledger.setLanes(0, RequestPriority::Foreground);
  require(counted(request(1, RequestPriority::Background)) == 1 + 2,
          "a claim below a lane in service took a pool unit");
  require(counted(request(1, RequestPriority::Background, ClaimClass::Start)) == 1 + 2,
          "a start below a lane in service took a pool unit");
  require(counted(request(1, RequestPriority::Foreground)) == 1 + 2 + 32 + 64,
          "a claim at the lanes' priority was refused a pool unit");
}

// A need the host refused takes only what frees now, whatever its class: no conversion, pass command or unit in
// use; short when that does not cover it.
void testAHostRefusedNeedTakesOnlyWhatFreesNow() {
  for (const uint32_t pages : {1u, 2u}) {
    MemoryLedger ledger;
    TestSource cache("cache"), engine("engine");
    Victim inUse{.key = 2, .pages = 2, .cls = ReclaimClass::InUse};
    cache.units = {Victim{.key = 1, .pages = 1}, inUse};
    Victim conversion{.key = 3, .pages = 4, .kind = VictimKind::Conversion};
    Victim passes{.key = 4, .pages = 8, .kind = VictimKind::PassCommand};
    engine.units = {conversion, passes};
    ledger.addSource(cache);
    ledger.addSource(engine);
    ClaimSpec spec = request(1, RequestPriority::Normal);
    spec.reach = ReclaimClass::InUse;
    spec.supersedes = true;
    const ClaimId claim = ledger.open(spec);
    const FreeOutcome outcome = ledger.free(claim, {.hostRefused = true, .pages = pages}, 0, false).outcome;
    if (pages == 1)
      require(outcome == FreeOutcome::Met && cache.taken == std::vector<uint64_t>{1} && engine.taken.empty(),
              "a host-refused need took more than what frees now");
    else
      require(outcome == FreeOutcome::Short && cache.taken.empty() && engine.taken.empty(),
              "a host-refused need took a conversion, the pass command or a unit in use");
  }
}

// A claim whose need fits the free room but for the room waiting claims ranked above it hold waits for them,
// taking nothing.
void testRoomHeldForHigherRanksMakesAWait() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB)};
  ledger.addSource(cache);
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  waitForItsCopies(ledger, high, {.bytes = kGiB});
  const ClaimId low = ledger.open(request(2, RequestPriority::Background));
  require(ledger.free(low, {.bytes = 512 * kMiB}, kGiB, false).outcome == FreeOutcome::Pending &&
              cache.taken.empty(),
          "a claim the room held above it fits took a victim instead of waiting");
}

// A claim the units would meet but for the room held for higher ranks waits, taking nothing; a need they would not
// meet anyway, or an Immediate one, is short.
void testUnitsHeldForHigherRanksMakeAWait() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB)};
  ledger.addSource(cache);
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  waitForItsCopies(ledger, high, {.bytes = kGiB});
  const ClaimId low = ledger.open(request(2, RequestPriority::Normal));
  const FreeOutcome outcome = ledger.free(low, {.bytes = kGiB}, 0, false).outcome;
  require(outcome == FreeOutcome::Pending && cache.taken.empty() && ledger.claim(low).waiting,
          std::string("a claim the cache meets but for the room a waiting claim ranked above it holds found free() ") +
              (outcome == FreeOutcome::Short ? "short" : outcome == FreeOutcome::Met ? "met" : "pending") +
              ", " + std::to_string(cache.taken.size()) + " units taken");
  const ClaimId larger = ledger.open(request(4, RequestPriority::Background));
  require(ledger.free(larger, {.bytes = 2 * kGiB}, 0, false).outcome == FreeOutcome::Short &&
              cache.taken.empty() && !ledger.claim(larger).waiting,
          "a claim the cache does not meet without the room held above it either waited for that room");
  ClaimSpec spec = request(3, RequestPriority::Normal);
  spec.timing = NeedTiming::Immediate;
  const ClaimId now = ledger.open(spec);
  require(ledger.free(now, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Short && cache.taken.empty(),
          "an Immediate claim waited behind the room held above it");
}

// Landings count first, then victims in the fixed order, then owed copies: a claim takes a victim before it waits
// for another lane's write-through.
void testLandingThenVictimsThenOwed() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB)};
  ledger.addSource(cache);
  ledger.setReturning([] { return Returning{.landingBytes = 256 * kMiB, .owedBytes = 4 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1},
          "the claim waited on owed copies before it took a victim");
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Pending &&
              ledger.claim(claim).waiting,
          "with no victim left the claim did not wait for the owed copies");
  ledger.setReturning([] { return Returning{.landingBytes = 2 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  cache.units = {unit(2, kGiB)};
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Pending && cache.taken.size() == 1,
          "landings that meet the need did not come first");
}

// An Immediate claim does not wait for landings and takes nothing that lands later.
void testImmediateSkipsLandings() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, true), unit(2, kGiB)};
  ledger.addSource(cache);
  ledger.setReturning([] { return Returning{.landingBytes = 4 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .requestId = 1, .timing = NeedTiming::Immediate});
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{2},
          "an Immediate claim waited or took a unit that lands later");
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Short,
          "an Immediate claim waited for landings");
}

// A cached unit that needs a write but frees now, as a state the idle staging buffer writes does,
// goes to a claim that cannot wait in the stage of the cache that needs a write: after the pools
// that cost only their rebuild. One that frees only once its write lands it passes over.
void testImmediateTakesAWriteThatFreesNowAfterRebuiltPools() {
  FakeMemory memory({});
  FakeElasticPool vision("vision", memory, 256 * kMiB, 0, 1, 1, 1);
  vision.rebuildOnly = true;
  MemoryLedger ledger;
  TestSource cache("cache");
  Victim staged = unit(3, 256 * kMiB);
  staged.write = true;
  cache.units = {unit(1, 256 * kMiB), unit(2, 256 * kMiB, true), staged};
  ledger.addSource(cache);
  ledger.addPool(vision);
  const auto lacks = [&](uint64_t bytes) {
    const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .requestId = 1, .timing = NeedTiming::Immediate});
    const FreeOutcome outcome = ledger.free(claim, {.bytes = bytes}, 0, false).outcome;
    ledger.close(claim);
    return outcome;
  };
  require(lacks(512 * kMiB) == FreeOutcome::Met && cache.taken == std::vector<uint64_t>{1} && vision.units() == 0,
          "the pool that costs only its rebuild did not go before the unit that needs a write");
  require(lacks(256 * kMiB) == FreeOutcome::Met && cache.taken == std::vector<uint64_t>{1, 3},
          "a claim that cannot wait did not take the unit whose write frees it now");
  require(lacks(256 * kMiB) == FreeOutcome::Short && cache.taken.size() == 2,
          "a claim that cannot wait took the unit that frees only once its write lands");
}

// What lands and what the claim's own lanes owe count before victims in part too: a waiting claim takes victims
// only for the rest and waits for the landing.
void testLandingCountsBeforeVictimsInPart() {
  struct Case final {
    Returning all;
    Returning own;
    NeedTiming timing = NeedTiming::CanWait;
    FreeOutcome outcome = FreeOutcome::Pending;
    std::vector<uint64_t> taken;
  };
  const Case cases[] = {
      {.all = {.landingBytes = 512 * kMiB}, .taken = {1}},
      {.all = {.owedBytes = 512 * kMiB}, .own = {.owedBytes = 512 * kMiB}, .taken = {1}},
      {.all = {.owedBytes = 512 * kMiB}, .outcome = FreeOutcome::Met, .taken = {1, 2}},
      {.all = {.landingBytes = 512 * kMiB}, .timing = NeedTiming::Immediate, .outcome = FreeOutcome::Met,
       .taken = {1, 2}},
  };
  for (const Case &each : cases) {
    MemoryLedger ledger;
    TestSource cache("cache");
    cache.units = {unit(1, 512 * kMiB), unit(2, 512 * kMiB)};
    ledger.addSource(cache);
    ledger.setReturning([&] { return each.all; }, [&](std::span<const uint64_t> lanes) {
      return std::ranges::find(lanes, 1) != lanes.end() ? each.own : Returning{};
    });
    ClaimSpec spec = request(1, RequestPriority::Normal);
    spec.timing = each.timing;
    const ClaimId claim = ledger.open(spec);
    require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == each.outcome && cache.taken == each.taken &&
                ledger.claim(claim).waiting == (each.outcome == FreeOutcome::Pending),
            "the claim took victims for what lands, or did not wait for it after taking the rest");
  }
  // In pages: two land, a waiting claim ranked above them takes them first.
  for (const bool heldAbove : {false, true}) {
    MemoryLedger ledger;
    TestSource cache("cache");
    ledger.addSource(cache);
    ledger.setReturning([] { return Returning{.landingPages = {2, 0, 0}}; },
                        [](std::span<const uint64_t>) { return Returning{}; });
    if (heldAbove) {
      const ClaimId high = ledger.open(request(2, RequestPriority::Foreground));
      require(ledger.free(high, {.pages = 2}, 0, false).outcome == FreeOutcome::Pending,
              "the claim above did not wait for the pages that land");
    }
    for (uint64_t key : {1, 2})
      cache.units.push_back({.key = key, .pages = 2});
    const ClaimId claim = ledger.open(request(3, RequestPriority::Background));
    const FreeOutcome outcome = ledger.free(claim, {.pages = 4}, 0, false).outcome;
    require(heldAbove ? outcome == FreeOutcome::Met && cache.taken == std::vector<uint64_t>{1, 2}
                      : outcome == FreeOutcome::Pending && cache.taken == std::vector<uint64_t>{1},
            "the pages that land were not counted before victims, or counted although held above");
  }
  // Bytes a claim above waits for that the room under the limit covers are
  // held there, not again as landing pages: the claim below counts the two
  // that land, as with nothing above it.
  {
    MemoryLedger ledger;
    ledger.setPageBytes({4096, 0, 0});
    TestSource cache("cache");
    ledger.addSource(cache);
    ledger.setReturning([] { return Returning{.landingPages = {2, 0, 0}}; }, [](std::span<const uint64_t> lanes) {
      return std::ranges::find(lanes, 2) != lanes.end() ? Returning{.owedBytes = 8192} : Returning{};
    });
    const ClaimId high = ledger.open(request(2, RequestPriority::Foreground));
    require(ledger.free(high, {.bytes = 8192}, 0, false).outcome == FreeOutcome::Pending,
            "the claim above did not wait for its copies");
    for (uint64_t key : {1, 2})
      cache.units.push_back({.key = key, .pages = 2});
    const ClaimId claim = ledger.open(request(3, RequestPriority::Background));
    require(ledger.free(claim, {.pages = 4}, 8192, false).outcome == FreeOutcome::Pending &&
                cache.taken == std::vector<uint64_t>{1},
            "the bytes a claim above waits for, which the room covers, were held as landing pages too");
  }
}

// A model's pool of state buffers: those idle beyond the `kept` ones a lane
// starts from go one at a time, at no cost, as a unit sized only when taken
// (Model::releaseIdleBuffer); the kept ones go all at once, as the last unit
// of the claims whose rights take them (TakeRights::LaneStart::Last).
class BufferPool final : public VictimSource {
public:
  explicit BufferPool(uint64_t bufferBytes) : bufferBytes_(bufferBytes) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "buffers"; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &query, const Victim *after) const override {
    if (!after)
      return Victim{.sized = false};
    if (after->kind == VictimKind::Memory && query.rights.laneStart == TakeRights::LaneStart::Last)
      return Victim{.kind = VictimKind::LaneStart};
    return std::nullopt;
  }
  FreeResult take(const VictimQuery &, const Victim &victim) override {
    if (victim.kind == VictimKind::LaneStart) {
      const uint64_t bytes = uint64_t{std::exchange(idle, 0)} * bufferBytes_;
      return bytes ? FreeResult{FreeOutcome::Met, bytes} : FreeResult{};
    }
    if (idle <= kept)
      return {};
    --idle;
    return {FreeOutcome::Met, bufferBytes_};
  }
  // What it releases once `more` buffers return to it, beyond the kept ones
  // with keepLane (Model::idleBufferBytes).
  [[nodiscard]] uint64_t releases(uint32_t more, bool keepLane = true) const noexcept {
    const uint32_t held = keepLane ? kept : 0;
    return uint64_t{idle + more - std::min(idle + more, held)} * bufferBytes_;
  }
  uint32_t idle = 0;
  uint32_t kept = 0;

private:
  uint64_t bufferBytes_;
};

// Cached states in the test's order, whose buffers return to the pool as
// they go.
class CachedStates final : public VictimSource {
public:
  explicit CachedStates(BufferPool &pool) : pool_(pool) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "cache"; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &, const Victim *after) const override {
    auto unit = units.begin();
    if (after)
      unit = std::next(std::ranges::find(units, after->key, &Victim::key));
    return unit == units.end() ? std::nullopt : std::optional<Victim>(*unit);
  }
  FreeResult take(const VictimQuery &, const Victim &victim) override {
    const auto unit = std::ranges::find(units, victim.key, &Victim::key);
    taken.push_back(unit->key);
    const Victim gone = *unit;
    pool_.idle += gone.stateBuffers;
    units.erase(unit);
    return {.outcome = FreeOutcome::Met, .nowBytes = gone.bytes, .stateBuffers = gone.stateBuffers};
  }
  std::vector<Victim> units;
  std::vector<uint64_t> taken;

private:
  BufferPool &pool_;
};

// A need of bytes counts the state buffers its units free as what the model's pool releases of them, once: a
// state taken returns its buffers to the pool at once, and the pool keeps a lane's. Of a pool below them the first
// state's buffer gives nothing and the second's its bytes, so the two states and one unit meet a need of two units,
// and the pool releases what was counted.
void testStateBuffersCountBeyondThoseThePoolKeeps() {
  MemoryLedger ledger;
  BufferPool pool(kGiB);
  pool.kept = 1;
  CachedStates cache(pool);
  cache.units = {{.key = 1, .stateBuffers = 1}, {.key = 2, .stateBuffers = 1}, unit(3, kGiB), unit(4, kGiB)};
  ledger.addSource(cache);
  ledger.addSource(pool);
  ledger.setBufferBytes([&pool](uint32_t states, bool, bool) { return pool.releases(states); });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  const FreeResult freed = ledger.free(claim, {.bytes = 2 * kGiB}, 0, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 2 * kGiB &&
              cache.taken == std::vector<uint64_t>{1, 2, 3} && pool.idle == 1,
          "free() did not count the bytes of the state buffers the model's pool releases, once: " +
              std::to_string(cache.taken.size()) + " units taken, " + std::to_string(freed.nowBytes / kMiB) +
              " MiB freed");
}

// The buffers a lane starts from go to in-service KV that can wait, after every other unit, counted with the state
// buffers that refill them first: from an empty pool a state's buffer refills the lane's, and the state, a unit and
// the lane's buffer meet a KV need of two units. A KV need the units before them meet keeps them; so do a need of a
// state's buffers, the model's other memory, a start's KV and an Immediate claim's, which are short.
void testTheBuffersALaneStartsFromGoLast() {
  struct Case final {
    std::string name;
    ClaimSpec spec;
    Need need;
    FreeOutcome outcome;
    std::vector<uint64_t> taken;
    uint32_t idle;
  };
  ClaimSpec immediate = request(1, RequestPriority::Normal);
  immediate.timing = NeedTiming::Immediate;
  const ClaimSpec inService = request(1, RequestPriority::Normal);
  const ClaimSpec start = request(1, RequestPriority::Normal, ClaimClass::Start);
  for (const Case &each : {
           Case{"a lane's KV", inService, {.bytes = 2 * kGiB, .kv = true}, FreeOutcome::Met, {1, 2}, 0},
           Case{"KV the cache meets", inService, {.bytes = kGiB, .kv = true}, FreeOutcome::Met, {1, 2}, 1},
           Case{"a state's buffers", inService, {.bytes = 2 * kGiB, .unheldStateBuffer = true}, FreeOutcome::Short,
                {}, 0},
           Case{"the model's other memory", inService, {.bytes = 2 * kGiB}, FreeOutcome::Short, {}, 0},
           Case{"a start's KV", start, {.bytes = 2 * kGiB, .kv = true}, FreeOutcome::Short, {}, 0},
           Case{"Immediate KV", immediate, {.bytes = 2 * kGiB, .kv = true}, FreeOutcome::Short, {}, 0}}) {
    MemoryLedger ledger;
    BufferPool pool(kGiB);
    pool.kept = 1;
    CachedStates cache(pool);
    cache.units = {{.key = 1, .stateBuffers = 1}, unit(2, kGiB)};
    ledger.addSource(cache);
    ledger.addSource(pool);
    ledger.setBufferBytes([&pool](uint32_t states, bool, bool keepLane) { return pool.releases(states, keepLane); });
    const FreeResult freed = ledger.free(ledger.open(each.spec), each.need, 0, false);
    require(freed.outcome == each.outcome && cache.taken == each.taken && pool.idle == each.idle &&
                freed.nowBytes == (each.outcome == FreeOutcome::Met ? each.need.bytes : 0),
            each.name + ": the buffers a lane starts from were taken out of order or miscounted: " +
                std::to_string(cache.taken.size()) + " units taken, " + std::to_string(pool.idle) +
                " buffers pooled, " + std::to_string(freed.nowBytes / kMiB) + " MiB freed");
  }
}

// The buffers a state frees meet a need of bytes, counted at what the pool releases of them; a need of buffers in
// place keeps them pooled.
void testStateBuffersMeetANeedOfBytes() {
  MemoryLedger ledger;
  BufferPool pool(64 * kMiB);
  CachedStates cache(pool);
  for (uint64_t key = 1; key <= 3; ++key)
    cache.units.push_back({.key = key, .stateBuffers = 1});
  ledger.addSource(cache);
  ledger.addSource(pool);
  // The first state's buffers release 64 MiB, each after it 32 MiB more:
  // two cover no 100 MiB, three do.
  ledger.setBufferBytes([&pool](uint32_t states, bool, bool) {
    const uint32_t pooled = pool.idle + states;
    return pooled ? uint64_t{pooled + 1} * 32 * kMiB : 0;
  });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  const FreeResult freed = ledger.free(claim, {.bytes = 100 * kMiB}, 0, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 192 * kMiB &&
              cache.taken == std::vector<uint64_t>{1, 2, 3} && !pool.idle,
          "a need of bytes did not take the states whose buffers the pool releases, or left them in the pool");
  cache.units.push_back({.key = 4, .stateBuffers = 1});
  const ClaimId lane = ledger.open(request(2, RequestPriority::Normal));
  const FreeResult inPlace = ledger.free(lane, {.stateBuffers = 1}, 0, false);
  require(inPlace.outcome == FreeOutcome::Met && !inPlace.nowBytes && cache.taken.back() == 4 && pool.idle == 1,
          "a need of buffers in place did not keep the buffers its state freed");
}

// Idle buffers go first and the fit is tried again: a refusal they cover takes nothing in place; a need that
// refused no bytes is met in place.
void testIdleBuffersThatCoverARefusalTakeNothingInPlace() {
  for (const bool pages : {false, true}) {
    MemoryLedger ledger;
    BufferPool pool(64 * kMiB);
    CachedStates cache(pool);
    cache.units.push_back({.key = 1, .pages = 4, .stateBuffers = 1});
    ledger.addSource(cache);
    ledger.addSource(pool);
    pool.idle = 1;
    const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
    const Need need = pages ? Need{.bytes = 48 * kMiB, .pages = 4} : Need{.bytes = 48 * kMiB, .stateBuffers = 1};
    const FreeResult freed = ledger.free(claim, need, 0, false);
    require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 64 * kMiB && cache.taken.empty() && !pool.idle,
            std::string("a refusal of bytes the idle buffers cover took ") + (pages ? "a unit's pages" : "a state") +
                " in place: " + std::to_string(cache.taken.size()) + " taken, " +
                std::to_string(freed.nowBytes / kMiB) + " MiB freed");
  }
  MemoryLedger ledger;
  BufferPool pool(64 * kMiB);
  CachedStates cache(pool);
  cache.units.push_back({.key = 1, .stateBuffers = 1});
  ledger.addSource(cache);
  ledger.addSource(pool);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.stateBuffers = 1}, 0, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1} && pool.idle == 1,
          "a need of buffers alone was not met in place");
}

// A need the host refused is met in place first; failing that, in-service work is exempted from the host for its
// retries and a start is short.
void testHostRefusedNeedIsMetInPlaceFirst() {
  Fixture fixture(host(1200 * kMiB));
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", kGiB);
  Victim leaf = unit(1, 0);
  leaf.pages = 8;
  cache.units = {unit(2, kGiB), leaf};
  fixture.ledger().addSource(cache);
  const ClaimId lane = fixture.ledger().open(request(1, RequestPriority::Normal));
  require(fixture.allocate(lane, 512 * kMiB).failure == metal::AllocationFailure::HostPressure,
          "the host did not refuse the lane");
  Need need = fixture.governor->need(lane, Need{.pages = 8});
  require(need.hostRefused && need.bytes == 512 * kMiB && need.pages == 8, "the need is not the refusal and pages");
  require(fixture.governor->free(lane, need).outcome == FreeOutcome::Met && cache.taken == std::vector<uint64_t>{1},
          "the host-refused need was not met in place by the leaf");
  require(!fixture.ledger().claim(lane).hostExempt, "a need met in place exempted the claim");
  need.pages = 0;
  require(fixture.governor->free(lane, need).outcome == FreeOutcome::Met && fixture.ledger().claim(lane).hostExempt,
          "in-service work nothing in place meets was not exempted");
  require(fixture.allocate(lane, 512 * kMiB) && fixture.ledger().claim(lane).hostExempt,
          "the exempt retry was refused, or its grant ended the exemption before the allocation did");
  fixture.ledger().setHostExempt(lane, false);
  const ClaimId start = fixture.ledger().open(request(2, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(start, 512 * kMiB), "the host admitted the start");
  require(fixture.governor->free(start, fixture.governor->need(start)).outcome == FreeOutcome::Short &&
              !fixture.ledger().claim(start).hostExempt,
          "a start was exempted from the host");
}

// What a lane needs (ClaimSpec::laneNeeds), refused by the host, is met in place by any unit its class may take
// before it passes the host.
void testWhatALaneNeedsIsMetInPlaceBeforeTheHost() {
  Fixture fixture(host(1200 * kMiB));
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", kGiB);
  Victim leaf = unit(1, 0);
  leaf.pages = 8;
  cache.units = {leaf};
  fixture.ledger().addSource(cache);
  const ClaimId claim = fixture.ledger().open(
      {.cls = ClaimClass::Optional, .requestId = 1, .timing = NeedTiming::Immediate, .laneNeeds = true});
  require(fixture.allocate(claim, 512 * kMiB).failure == metal::AllocationFailure::HostPressure,
          "the host did not refuse the claim");
  const FreeResult freed = fixture.governor->free(claim, fixture.governor->need(claim, Need{.pages = 8}));
  require(freed.outcome == FreeOutcome::Met && cache.taken == std::vector<uint64_t>{1} &&
              !fixture.ledger().claim(claim).hostExempt,
          "what a lane needs passed the host beside pages in place");
}

// A host exemption covers the whole retried allocation: each of its reservations passes the host, not only the
// first.
void testAnExemptRetryPassesTheHostWithEachReservation() {
  constexpr uint64_t extent = 108'527'616;
  Fixture fixture(host(kHostMarginBytes + extent / 2));
  const ClaimId lane = fixture.ledger().open(request(1, RequestPriority::Normal));
  const auto attempt = [&] {
    std::vector<uint64_t> granted;
    for (int reservation = 0; reservation < 4; ++reservation) {
      if (!fixture.allocate(lane, extent)) {
        for (const uint64_t bytes : granted)
          fixture.memory.release("kv", bytes);
        return false;
      }
      granted.push_back(extent);
    }
    return true;
  };
  require(!attempt(), "the host admitted the floor beside its margin");
  require(fixture.governor->free(lane, fixture.governor->need(lane)).outcome == FreeOutcome::Met &&
              fixture.ledger().claim(lane).hostExempt,
          "in-service work nothing in place meets was not exempted");
  require(attempt(), "the exempt retry was refused the host after its first reservation");
}

// Units that cost nothing and are sized only when taken go first.
void testZeroCostUnsizedUnitsGoFirst() {
  MemoryLedger ledger;
  TestSource engine("engine"), cache("cache");
  Victim buffers = unit(1, 0);
  buffers.sized = false;
  engine.units = {buffers};
  engine.unsizedBytes = 600 * kMiB;
  cache.units = {unit(2, kGiB)};
  ledger.addSource(cache);
  ledger.addSource(engine);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  const FreeResult freed = ledger.free(claim, {.bytes = kGiB}, 512 * kMiB, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 600 * kMiB && engine.taken.size() == 1 &&
              cache.taken.empty(),
          "the unsized buffers did not go first and end the need");
}

// The host's claim takes its target: the idle buffers that go first count in it. Taking the whole target beyond
// them gave the host 256 MiB more than it asked for here.
void testTheHostClaimCountsTheIdleBuffersInItsTarget() {
  MemoryLedger ledger;
  TestSource engine("engine"), cache("cache");
  Victim buffers = unit(1, 0);
  buffers.sized = false;
  engine.units = {buffers};
  engine.unsizedBytes = 256 * kMiB;
  for (uint64_t key = 2; key <= 5; ++key)
    cache.units.push_back(unit(key, 128 * kMiB));
  ledger.addSource(cache);
  ledger.addSource(engine);
  const ClaimId claim = ledger.open({.cls = ClaimClass::Host});
  const FreeResult freed = ledger.free(claim, {.bytes = 512 * kMiB}, 0, false);
  ledger.close(claim);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 512 * kMiB && cache.taken.size() == 2,
          "the host's claim took " + std::to_string(freed.nowBytes / kMiB) + " MiB for a target of 512 MiB");
}

// The engine's own optional work, an Optional claim of no request, takes no unit (TakeRights::units):
// only what costs nothing and what lands. A request's optional work takes them.
void testAClaimThatTakesNoUnitsTakesNone() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, 512 * kMiB), unit(2, 512 * kMiB)};
  ledger.addSource(cache);
  for (const uint64_t requestId : {uint64_t{0}, uint64_t{7}}) {
    const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .requestId = requestId});
    const FreeOutcome outcome = ledger.free(claim, {.bytes = kGiB}, 0, false).outcome;
    ledger.close(claim);
    require(requestId ? outcome == FreeOutcome::Met && cache.taken.size() == 2
                      : outcome == FreeOutcome::Short && cache.taken.empty(),
            "whether the Optional claim was of a request did not decide whether it took the source's units");
  }
}

// A claim that takes no units counts none: it peeks a source only for the units that cost nothing, and asks no
// copies ahead of cache it would not take copied either.
void testAClaimThatTakesNoUnitsCountsNone() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, true), unit(2, kGiB)};
  ledger.addSource(cache);
  const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .timing = NeedTiming::Immediate});
  const FreeOutcome outcome = ledger.free(claim, {.bytes = kGiB}, 0, false).outcome;
  ledger.close(claim);
  require(outcome == FreeOutcome::Short && cache.taken.empty(), "a claim that takes no units took one");
  require(cache.peeks == 1, "a claim that takes no units peeked " + std::to_string(cache.peeks) + " units");
  require(!ledger.copyDemand(), "a claim that takes no units asked copies ahead");
}

// available() counts the room, the cache and pool units above the floor up to the need, and takes nothing.
void testAvailableTakesNothing() {
  FakeMemory memory({});
  FakeElasticPool pool("experts", memory, kGiB, 1, 3, 4, 3);
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB), unit(2, kGiB, true)};
  ledger.addSource(cache);
  ledger.addPool(pool);
  require(ledger.available(request(1, RequestPriority::Normal), {.bytes = 10 * kGiB}, 256 * kMiB, false) ==
              256 * kMiB + 4 * kGiB,
          "available() did not count the room, the cache and the pool's units above its floor");
  require(ledger.available(request(1, RequestPriority::Normal), {.bytes = kGiB}, 256 * kMiB, false) == kGiB,
          "available() did not stop at enough");
  require(cache.taken.empty() && pool.units() == 3 && pool.shrinks == 0, "available() took something");
}

// free() and available() peek no unit past the one that covers the lack: a count costs what the claim takes, not
// the cache behind it.
void testCountsStopAtTheLack() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, 2 * kGiB), unit(2, kGiB)};
  ledger.addSource(cache);
  const auto stopped = [&] { return std::ranges::find(cache.peeked, uint64_t{1}) == cache.peeked.end(); };
  const ClaimSpec spec = request(1, RequestPriority::Normal);
  require(ledger.available(spec, {.bytes = kGiB}, 256 * kMiB, false) == kGiB && stopped(),
          "available() counted past what the claim lacks");
  cache.peeked.clear();
  const ClaimId claim = ledger.open(spec);
  require(ledger.free(claim, {.bytes = kGiB}, 256 * kMiB, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1} && stopped(),
          "free() counted past what the claim lacks");
  ledger.close(claim);
}

// available() counts what free() would take, taking nothing: what lands and what the claim's request owes count for
// a claim that can wait, not for an Immediate one.
void testAvailableCountsWhatLandsForAClaimThatCanWait() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, 256 * kMiB)};
  ledger.addSource(cache);
  ledger.setReturning([] { return Returning{.landingBytes = 512 * kMiB}; },
                      [](std::span<const uint64_t> lanes) {
                        return std::ranges::find(lanes, 1u) != lanes.end() ? Returning{.owedBytes = 128 * kMiB}
                                                                           : Returning{};
                      });
  const ClaimSpec start = request(1, RequestPriority::Normal, ClaimClass::Start);
  require(ledger.available(start, {.bytes = 2 * kGiB}, 128 * kMiB, false) == 1024 * kMiB,
          "available() did not count what lands and the copies the request owes before its victims");
  ClaimSpec immediate = start;
  immediate.timing = NeedTiming::Immediate;
  require(ledger.available(immediate, {.bytes = 2 * kGiB}, 128 * kMiB, false) == 384 * kMiB,
          "available() counted what lands for a claim that cannot wait");
  require(cache.taken.empty(), "available() took something");
}

// Free room beyond the lanes' projected growth regrows the pools one unit per quiet period, never into what the
// lanes project.
void testGrowthRetakesFreeRoomBeyondLaneDemand() {
  Fixture fixture(host(32 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 8, 9, 2);
  fixture.memory.hold("weights", 3 * kGiB);
  fixture.ledger().addPool(experts);
  fixture.ledger().setLanes(kGiB, std::nullopt);
  const auto unitsAt = [&](double seconds) {
    static_cast<void>(fixture.pass(seconds));
    return experts.units();
  };
  require(unitsAt(kRegrowQuietSeconds - 0.5) == 2, "the pool grew before the quiet time passed");
  require(unitsAt(kRegrowQuietSeconds) == 3, "the pool did not grow a unit once the quiet time passed");
  require(unitsAt(kRegrowQuietSeconds + 0.5) == 3, "the pool grew again within the quiet time after its growth");
  require(unitsAt(2 * kRegrowQuietSeconds) == 4, "the pool did not grow its next unit after another quiet time");
  require(unitsAt(3 * kRegrowQuietSeconds) == 4, "the pool grew into the room the lanes are projected to grow into");
  fixture.ledger().setLanes(0, std::nullopt);
  require(unitsAt(4 * kRegrowQuietSeconds - 0.5) == 4, "the pool tried again within the quiet time after a try");
  require(unitsAt(4 * kRegrowQuietSeconds) == 5, "the pool did not take the room the lanes no longer project");
  // A request that lacks room takes a unit back and restarts the wait.
  const ClaimId claim = fixture.ledger().open(request(1, RequestPriority::Normal));
  require(fixture.ledger().free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Met &&
              experts.units() == 4,
          "the request did not take a unit");
  fixture.ledger().close(claim);
  require(unitsAt(5 * kRegrowQuietSeconds - 0.5) == 4, "the pool grew within the quiet time after a request lacked room");
  require(unitsAt(5 * kRegrowQuietSeconds) == 5, "the pool did not grow back after the quiet time");
}

// After a burst shrank the store, each quiet period regrows a unit from cache that frees now, never what is in use,
// reused or unwritten; unwritten cache gets copies ahead.
void testGrowthTakesIdleCacheAfterABurst() {
  Fixture fixture(host(32 * kGiB), 10 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 1, 4, 4, 4);
  fixture.memory.hold("weights", 3 * kGiB);
  fixture.ledger().addPool(experts);
  const ClaimId burst = fixture.ledger().open(request(1, RequestPriority::Normal));
  require(fixture.allocateFreeing(burst, 6 * kGiB) && experts.units() == 1,
          "the burst did not shrink the store to its floor");
  fixture.ledger().close(burst);
  fixture.memory.release("kv", 6 * kGiB);
  fixture.memory.hold("cache", 6 * kGiB);
  TestSource cache("cache", &fixture.memory);
  cache.units = {Victim{.key = 1, .bytes = kGiB, .cls = ReclaimClass::InUse},
                 unit(3, kGiB, true),
                 unit(4, kGiB, true),
                 unit(5, kGiB, true),
                 unit(6, kGiB, true)};
  fixture.ledger().addSource(cache);
  static_cast<void>(fixture.pass(kRegrowQuietSeconds - 1.0));
  require(experts.units() == 1 && !fixture.ledger().copyDemand(), "the store reached for cache before the quiet time");
  static_cast<void>(fixture.pass(kRegrowQuietSeconds + 1.0));
  require(experts.units() == 1 && cache.taken.empty() && fixture.ledger().copyDemand() &&
              fixture.ledger().growthLackBytes() != 0,
          "growth that found only cache awaiting its copy took it or asked for no copies ahead");
  for (Victim &each : cache.units)
    each.write = each.later = each.key == 3;
  for (int period = 2; period <= 5; ++period)
    static_cast<void>(fixture.pass(period * kRegrowQuietSeconds + 1.0));
  require(experts.units() == 4, "the store did not return to its cap: units " + std::to_string(experts.units()));
  require(cache.taken == std::vector<uint64_t>{4, 5, 6},
          "growth took a unit in use or that frees only once written");
  fixture.ledger().removeSource(cache);
}

// Under critical pressure the host claim takes the margin once per settle time in the one order (cache, store above
// floor, in use), never below the floor or unwritten cache.
void testCriticalTakesTheMarginASettledSecondInTheOrder() {
  Fixture fixture(host(8 * kGiB, MemoryPressure::Critical));
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 4);
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", 2 * kGiB);
  Victim inUse = unit(2, kGiB);
  inUse.cls = ReclaimClass::InUse;
  cache.units = {unit(3, 512 * kMiB, true), unit(1, 512 * kMiB), inUse};
  fixture.ledger().addSource(cache);
  fixture.ledger().addPool(experts);
  static_cast<void>(fixture.pass(0.5));
  require(cache.taken == std::vector<uint64_t>{1} && experts.units() == 3,
          "a critical pass did not take the margin from cache that frees now, then the store");
  static_cast<void>(fixture.pass(1.0));
  require(experts.units() == 3 && cache.units.size() == 2, "a critical pass ran before the host settled");
  static_cast<void>(fixture.pass(1.5));
  require(experts.units() == 2 && cache.units.size() == 2, "the second critical pass did not take the store first");
  static_cast<void>(fixture.pass(2.5));
  require(experts.units() == 2 && cache.taken == std::vector<uint64_t>{1, 2},
          "the third critical pass took the store below its floor, uncopied cache, or left what is in use");
}

// Under critical pressure the host claim takes what it finds up to its target at once, whatever is on its way back.
void testCriticalTakesWhatItFindsWhateverLands() {
  Fixture fixture(host(8 * kGiB, MemoryPressure::Critical));
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", 400 * kMiB);
  Victim inUse = unit(1, 200 * kMiB);
  inUse.cls = ReclaimClass::InUse;
  cache.units = {inUse, unit(2, 200 * kMiB)};
  fixture.ledger().setReturning([] { return Returning{.landingBytes = kGiB, .transfers = 4}; },
                                [](std::span<const uint64_t>) { return Returning{}; });
  fixture.ledger().addSource(cache);
  static_cast<void>(fixture.pass(0.5));
  require(cache.units.empty() && fixture.memory.held("cache") == 0,
          "a critical pass left the cache it found while more than it was on its way back");
}

// macOS's warning alone, with no deficit, takes nothing.
void testWarningWithoutADeficitTakesNothing() {
  Fixture fixture(host(1536 * kMiB + 3 * kGiB, MemoryPressure::Warning));
  TestSource cache("cache", &fixture.memory);
  for (uint64_t key = 1; key <= 6; ++key)
    cache.units.push_back(unit(key, 512 * kMiB));
  fixture.memory.hold("cache", 3 * kGiB);
  fixture.ledger().addSource(cache);
  for (double seconds = 0.5; seconds <= 60.0; seconds += 0.5)
    static_cast<void>(fixture.pass(seconds));
  require(cache.taken.empty() && fixture.memory.held("cache") == 3 * kGiB,
          "macOS's warning without a deficit took cache");
}

// The deficit counts what is landing: a pass frees only the rest.
void testDeficitNetOfReturning() {
  // 512 MiB of headroom beside the cache's 800 MiB: a deficit of 512 MiB,
  // all of it landing (no pass), or 312 MiB of it beyond 200 MiB landing.
  for (const uint64_t landing : {512 * kMiB, 200 * kMiB}) {
    Fixture fixture(host(512 * kMiB + 800 * kMiB));
    TestSource cache("cache", &fixture.memory);
    for (uint64_t key = 1; key <= 8; ++key)
      cache.units.push_back(unit(key, 100 * kMiB));
    fixture.memory.hold("cache", 800 * kMiB);
    fixture.ledger().addSource(cache);
    fixture.ledger().setReturning([&] { return Returning{.landingBytes = landing}; },
                                  [](std::span<const uint64_t>) { return Returning{}; });
    static_cast<void>(fixture.pass(1.0));
    require(cache.taken.size() == (landing == 512 * kMiB ? 0u : 4u),
            "the band pass did not free the deficit net of what is landing");
  }
}

// Missing telemetry holds elective work back but gives no cache back: no deficit is measured.
void testMissingTelemetryGivesNoCacheBack() {
  bool measured = true;
  metal::MetalBackend backend("");
  MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB, [&]() -> std::optional<HostMemory> {
    return measured ? std::optional<HostMemory>(HostMemory{16 * kGiB, 16 * kGiB}) : std::nullopt;
  }, 0);
  TestSource cache("cache");
  cache.units = {unit(1, kGiB)};
  governor.ledger().addSource(cache);
  measured = false;
  static_cast<void>(governor.controlPass(1000.0));
  require(cache.taken.empty() && !governor.admits(ClaimClass::Start),
          "a pass without telemetry gave cache back or admitted a start");
}

// A host pass that waits on landings asks for another pass at the next command-free point.
void testRerunWhileTheDeficitWaitsOnLandings() {
  // 512 MiB of headroom beside the cache's 1 GiB: a deficit of 512 MiB.
  Fixture fixture(host(512 * kMiB + kGiB));
  TestSource cache("cache", &fixture.memory);
  cache.units = {unit(1, kGiB, true)};
  fixture.memory.hold("cache", kGiB);
  fixture.ledger().addSource(cache);
  require(fixture.pass(1.0) && cache.taken.size() == 1, "a host pass that took a demotion did not ask to run again");
  require(!fixture.pass(3.0), "a pass with nothing returning asked to run again");
}

// What lands counts once: the deficit is net of it, and the host's claim takes the rest at once. Counted again
// by the claim, 640 MiB landing met its 384 MiB claim, and the host stayed short for the whole landing.
void testTheHostClaimTakesTheDeficitBeyondWhatLands() {
  // No headroom beside the cache's 1 GiB: 1 GiB short of the margin, 640 MiB
  // of it landing.
  Fixture fixture(host(kGiB));
  TestSource cache("cache", &fixture.memory);
  for (uint64_t key = 1; key <= 8; ++key)
    cache.units.push_back(unit(key, 128 * kMiB));
  fixture.memory.hold("cache", kGiB);
  fixture.ledger().addSource(cache);
  fixture.ledger().setReturning([] { return Returning{.landingBytes = 640 * kMiB}; },
                                [](std::span<const uint64_t>) { return Returning{}; });
  static_cast<void>(fixture.pass(1.0));
  require(cache.taken.size() == 3, "the host pass took " + std::to_string(cache.taken.size()) +
                                       " units of 128 MiB for a deficit of 384 MiB beyond what lands");
}

// A host pass that finds nothing while memory lands waits for it, judged again once the host has settled: run at
// every command-free point, the whole control pass reran each loop iteration while the landing lasted.
void testAHostPassWaitingOnLandingsRunsOncePerSettle() {
  Fixture fixture(host(kGiB));
  fixture.memory.hold("cache", kGiB);
  fixture.ledger().setReturning([] { return Returning{.landingBytes = 640 * kMiB}; },
                                [](std::span<const uint64_t>) { return Returning{}; });
  require(fixture.pass(1.0), "a host pass that found nothing while memory lands did not wait for it");
  require(!fixture.pass(1.001), "the host pass ran again before the host settled");
  require(fixture.pass(2.0), "the host pass was not judged again once the host settled");
}

// A deficit under macOS's warning takes about the deficit of cache, never a pool unit or a conversion.
void testAWarningDeficitTakesNoPoolUnit() {
  Fixture fixture(host(800 * kMiB + 3 * kGiB, MemoryPressure::Warning));
  FakeElasticPool experts("experts", fixture.memory, kGiB, 1, 3, 4, 2);
  TestSource cache("cache", &fixture.memory), engine("engine");
  fixture.memory.hold("cache", kGiB);
  cache.units = {unit(1, 512 * kMiB), unit(2, 512 * kMiB)};
  Victim conversion = unit(3, kGiB, true);
  conversion.kind = VictimKind::Conversion;
  engine.units = {conversion};
  fixture.ledger().addSource(cache);
  fixture.ledger().addSource(engine);
  fixture.ledger().addPool(experts);
  static_cast<void>(fixture.pass(1.0));
  require(cache.taken == std::vector<uint64_t>{1} && experts.units() == 2 && engine.taken.empty(),
          "a deficit under macOS's warning took other than about the deficit of cache, a pool unit or a "
          "conversion");
}

// Short of critical pressure the host claim takes no store unit and still refuses a start; under critical pressure
// the store goes to its floor.
void testTheHostTakesNoStoreUnitShortOfCritical() {
  constexpr uint64_t kLarge = 1023 * kMiB, kSmall = 188'000'000, kFloor = 7210 * kMiB;
  constexpr uint64_t kShort = 343'617'126;  // 327.7 MiB
  constexpr uint64_t kAll = kFloor + kLarge + kSmall;
  for (const MemoryPressure pressure : {MemoryPressure::Normal, MemoryPressure::Warning, MemoryPressure::Critical}) {
    Fixture fixture(host(kAll + kHostMarginBytes - kShort, pressure));
    ListedPool store(kFloor, {kLarge, kSmall});
    fixture.memory.hold("experts", kAll);
    TestSource cache("cache", &fixture.memory);
    MemoryLedger &ledger = fixture.ledger();
    std::string lines;
    ledger.setDecisionLog([&](std::string_view line) { lines += " " + std::string(line) + ";"; });
    ledger.addSource(cache);
    ledger.addPool(store);
    static_cast<void>(fixture.pass(1.0));
    if (pressure == MemoryPressure::Critical) {
      require(store.currentBytes() == kFloor, "critical pressure left the store above its floor:" + lines);
      continue;
    }
    const std::string when = pressure == MemoryPressure::Normal ? "at normal pressure" : "under macOS's warning";
    require(store.currentBytes() == kAll,
            when + " the host's pass took " + std::to_string((kAll - store.currentBytes()) / kMiB) +
                " MiB of the store's chunks above its floor:" + lines);
    const ClaimId start = ledger.open(request(1, RequestPriority::Normal, ClaimClass::Start));
    require(fixture.allocate(start, 128 * kMiB).failure == metal::AllocationFailure::HostPressure,
            when + " the host short of its margin admitted a start");
  }
}

// While a request's start waits, the store does not grow into the room it waits for: the start and its state
// buffer are met from that room with no unit taken back, and the unit grows back once nothing waits.
void testAPoolDoesNotGrowWhileAClaimWaits() {
  Fixture fixture(host(32 * kGiB), 16 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 104, 3);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addPool(experts);
  fixture.memory.hold("weights", 11 * kGiB);
  fixture.memory.hold("kv", 2 * kGiB);
  const ClaimId start = ledger.open(request(10, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(start, 43 * kMiB), "the start was admitted at the engine's limit");
  ledger.setWaiting(start, true);
  fixture.memory.release("kv", kGiB + 50 * kMiB);
  std::string history;
  uint32_t before = experts.units();
  std::optional<double> grewAt;
  uint32_t retaken = 0;
  const auto track = [&](double seconds) {
    const uint32_t after = experts.units();
    if (after > before) {
      grewAt = seconds;
      history += " grew at " + std::to_string(seconds) + " s;";
    } else if (after < before) {
      retaken += grewAt && seconds - *grewAt <= kRegrowQuietSeconds;
      history += " gave back at " + std::to_string(seconds) + " s;";
    }
    before = after;
  };
  constexpr double kQuiet = kRegrowQuietSeconds;
  static_cast<void>(fixture.pass(kQuiet + 1.0));
  track(kQuiet + 1.0);
  require(fixture.allocateFreeing(start, 43 * kMiB), "the waiting start was not met once memory was freed");
  ledger.update(start, request(10, RequestPriority::Normal, ClaimClass::InService));
  ledger.setWaiting(start, false);
  require(fixture.allocateFreeing(start, 15 * kMiB), "the started request's state buffer was refused");
  track(kQuiet + 1.01);
  static_cast<void>(fixture.pass(kQuiet + 1.5));
  track(kQuiet + 1.5);
  require(retaken == 0 && experts.units() == 3,
          "a store unit grown while a claim waited was taken back within the horizon:" + history);
  ledger.close(start);
  fixture.memory.release("kv", 58 * kMiB);
  static_cast<void>(fixture.pass(2 * kQuiet + 2.0));
  track(2 * kQuiet + 2.0);
  require(experts.units() == 4, "the store did not grow back once no claim waited:" + history);
}

// Only a request's waiting claim holds the pools' growth into free room; claims of no request never hold the store
// down.
void testOnlyARequestsWaitingClaimHoldsFreeRoom() {
  Fixture fixture(host(32 * kGiB), 16 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 104, 3);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addPool(experts);
  fixture.memory.hold("weights", 11 * kGiB);
  fixture.memory.hold("kv", 2 * kGiB);
  const ClaimId own = ledger.open({});
  require(!fixture.allocate(own, 43 * kMiB), "the engine's own claim was admitted at the engine's limit");
  ledger.setWaiting(own, true);
  const ClaimId start = ledger.open(request(10, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(start, 43 * kMiB), "the start was admitted at the engine's limit");
  ledger.setWaiting(start, true);
  fixture.memory.release("kv", kGiB + 100 * kMiB);
  // Past the quiet time, where only the waiting start holds the store.
  static_cast<void>(fixture.pass(kRegrowQuietSeconds + 1.0));
  require(experts.units() == 3, "the store grew into free room while a request's start waited");
  require(fixture.allocateFreeing(start, 43 * kMiB), "the waiting start was not met once memory was freed");
  ledger.update(start, request(10, RequestPriority::Normal, ClaimClass::InService));
  ledger.setWaiting(start, false);
  std::optional<double> grewAt;
  for (double seconds = kRegrowQuietSeconds + 1.5; seconds <= 2 * kRegrowQuietSeconds + 1.5 && !grewAt;
       seconds += 0.5) {
    static_cast<void>(fixture.pass(seconds));
    if (experts.units() == 4)
      grewAt = seconds;
  }
  require(grewAt.has_value(), "the store did not grow back after the start's wait ended, while only the engine's "
                              "own claim waited: " + std::to_string(experts.units()) + " units");
}

// A claim lacks its need plus the room held above it, beyond the free room: it takes units only once they cover
// both, and waits meanwhile.
void testAClaimLacksTheRoomHeldAboveIt() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB)};
  ledger.addSource(cache);
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  waitForItsCopies(ledger, high, {.bytes = kGiB});
  const ClaimSpec spec = request(2, RequestPriority::Background);
  const ClaimId low = ledger.open(spec);
  require(ledger.available(spec, {.bytes = kGiB}, 512 * kMiB, false) == 512 * kMiB,
          "available() counted the room the claim above holds");
  require(ledger.free(low, {.bytes = kGiB}, 512 * kMiB, false).outcome == FreeOutcome::Pending &&
              cache.taken.empty(),
          "the claim took cache for an allocation the room held above it refuses, or did not wait for it");
  cache.units.push_back(unit(2, kGiB));
  require(ledger.free(low, {.bytes = kGiB}, 512 * kMiB, false).outcome == FreeOutcome::Met &&
              cache.taken.size() == 2,
          "the claim did not take what it and the room held above it lack");
}

// A need met in place asks admission for no bytes, so the room held above it is not part of what it lacks.
void testANeedMetInPlaceLacksNoHeldBytes() {
  MemoryLedger ledger;
  TestSource cache("cache");
  Victim leaf = unit(1, 0);
  leaf.pages = 8;
  cache.units = {leaf};
  ledger.addSource(cache);
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  waitForItsCopies(ledger, high, {.bytes = kGiB});
  const ClaimId low = ledger.open(request(2, RequestPriority::Background));
  require(ledger.free(low, {.pages = 8}, 0, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1} && !ledger.claim(low).waiting,
          "a pages-only need waited on the room held above it instead of taking the leaf");
  leaf.key = 3;
  cache.units = {unit(2, kGiB), leaf};
  ClaimSpec spec = request(3, RequestPriority::Background);
  spec.timing = NeedTiming::Immediate;
  const ClaimId now = ledger.open(spec);
  require(ledger.free(now, {.pages = 8}, 0, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1, 3},
          "an Immediate pages-only need took cache bytes that leave no page");
}

// A store that started short of host memory regrows to its cap within two passes once the host frees room, with no
// quiet period.
void testShortStartRegrowsWhenTheHostFrees() {
  // Headroom 512 MiB beside the store's 2 GiB until 60 s, then M + 1.5 units.
  const uint64_t store = 2 * kGiB;
  Fixture fixture({.hostBytes = 64 * kGiB,
                   .otherBytes = 62 * kGiB - 512 * kMiB - store,
                   .script = test::lowHostAtStart(62 * kGiB - 512 * kMiB - store, 62 * kGiB - 2560 * kMiB - store,
                                                  60000.0)},
                  16 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 3, 4, 2);
  fixture.ledger().addPool(experts);
  for (int step = 0; step < 120; ++step)
    static_cast<void>(fixture.pass(step * 0.5));
  require(experts.units() == 2, "the store grew while the host was short");
  static_cast<void>(fixture.pass(60.0));
  static_cast<void>(fixture.pass(60.5));
  require(experts.units() == 3, "the store did not reach its cap within two passes of the host freeing");
}

// A converted lane waits for its own copies before it takes a store unit.
void testALoneConvertedLaneWaitsForItsOwnCopies() {
  FakeMemory memory({});
  FakeElasticPool experts("experts", memory, kGiB, 1, 3, 4, 3);
  MemoryLedger ledger;
  ledger.addPool(experts);
  ledger.setReturning([] { return Returning{.owedBytes = 2 * kGiB}; },
                      [](std::span<const uint64_t> lanes) {
                        return std::ranges::find(lanes, 5) != lanes.end() ? Returning{.owedBytes = 2 * kGiB}
                                                                          : Returning{};
                      });
  const ClaimId lane = ledger.open(request(5, RequestPriority::Normal));
  require(ledger.free(lane, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Pending && experts.units() == 3,
          "a lone Spill lane took a store unit instead of waiting for its own copies");
}

// A claim that converted its lane waits for that lane's copies: its retries take nothing else.
void testARetryAfterItsConversionTakesNothingElse() {
  MemoryLedger ledger;
  TestSource engine("engine");
  Victim conversion = unit(1, kGiB, true);
  conversion.kind = VictimKind::Conversion;
  engine.units = {conversion, unit(2, kGiB, true)};
  ledger.addSource(engine);
  ledger.setReturning([&] { return Returning{}; },
                      [&](std::span<const uint64_t> lanes) {
                        const auto converted = [&](uint64_t id) { return std::ranges::count(engine.taken, id) != 0; };
                        return std::ranges::any_of(lanes, converted) ? Returning{.owedBytes = kGiB} : Returning{};
                      });
  const ClaimId claim = ledger.open(growth(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Pending &&
              engine.taken == std::vector<uint64_t>{1},
          "the claim did not convert its lane");
  for (int retry = 0; retry < 5; ++retry)
    require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Pending &&
                engine.taken.size() == 1,
            "a retry took another unit while its lane's copies were owed");
}

// Another lane's owed copies do not hold a claim back from a store unit.
void testOneConversionDoesNotStallTheStore() {
  FakeMemory memory({});
  FakeElasticPool experts("experts", memory, kGiB, 1, 3, 4, 3);
  MemoryLedger ledger;
  ledger.addPool(experts);
  ledger.setReturning([] { return Returning{.owedBytes = 4 * kGiB}; },
                      [](std::span<const uint64_t> lanes) {
                        return std::ranges::find(lanes, 9) != lanes.end() ? Returning{.owedBytes = 4 * kGiB}
                                                                          : Returning{};
                      });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Met && experts.units() == 2,
          "another lane's write-through held the claim back from the store");
}

// Copies ahead are asked for only after a free() reached uncopied cache, and only for a while after it.
void testCopiesAheadOnlyAfterDemandReachedUncopiedCache() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, true), unit(2, kGiB)};
  ledger.addSource(cache);
  ledger.setNow(100.0);
  require(!ledger.copyDemand(), "copies ahead were wanted before any free()");
  const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .requestId = 1, .timing = NeedTiming::Immediate});
  require(ledger.free(claim, {.bytes = kGiB}, 0, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{2},
          "the Immediate claim did not skip the uncopied unit");
  require(ledger.copyDemand(), "an Immediate claim's skip of an uncopied unit asked no copies");
  ledger.setNow(100.0 + kCopyDemandSeconds);
  require(ledger.copyDemand(), "the demand ended within its time");
  ledger.setNow(100.5 + kCopyDemandSeconds);
  require(!ledger.copyDemand(), "copies ahead outlasted their time");
}

// A shortfall of X frees at most X and one pool unit, also when the smallest unit goes first.
void testAShortfallTakesAtMostOneUnitBeyondIt() {
  for (const uint64_t unitBytes : {kGiB, 512 * kMiB}) {
    FakeMemory memory({});
    FakeElasticPool experts("experts", memory, unitBytes, 0, 8, 9, 8);
    MemoryLedger ledger;
    ledger.addPool(experts);
    const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
    const uint64_t shortfall = 1331 * kMiB;
    const FreeResult freed = ledger.free(claim, {.bytes = shortfall}, 0, false);
    require(freed.outcome == FreeOutcome::Met && freed.nowBytes >= shortfall &&
                freed.nowBytes < shortfall + unitBytes,
            "a shortfall took more than itself and one unit");
  }
  ListedPool experts(kGiB, {1023 * kMiB, 1023 * kMiB, 117 * kMiB});
  MemoryLedger ledger;
  ledger.addPool(experts);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  const FreeResult freed = ledger.free(claim, {.bytes = 300 * kMiB}, 0, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 1140 * kMiB &&
              experts.currentBytes() == kGiB + 1023 * kMiB,
          "a shortfall of 300 MiB took " + std::to_string(freed.nowBytes / kMiB) +
              " MiB of a store whose small chunk goes first, not it and one large chunk");
}

// A host reading that lags its release takes nothing more: the next pass waits for the host to settle.
void testALaggingHostReadingTakesNoSecondRelease() {
  // 512 MiB of headroom beside the cache's 2 GiB; after the pass other
  // applications' bytes hide the 512 MiB it returns for 800 ms.
  Fixture fixture(host(2560 * kMiB, MemoryPressure::Normal, {at(1001.0, 2 * kGiB), at(1800.0, 2560 * kMiB)}));
  TestSource cache("cache", &fixture.memory);
  for (uint64_t key = 1; key <= 8; ++key)
    cache.units.push_back(unit(key, 256 * kMiB));
  fixture.memory.hold("cache", 2 * kGiB);
  fixture.ledger().addSource(cache);
  static_cast<void>(fixture.pass(1.0));
  const size_t first = cache.taken.size();
  require(first == 2, "the band pass did not free its deficit");
  static_cast<void>(fixture.pass(1.5));
  require(cache.taken.size() == first, "a lagging host reading took a second release");
  static_cast<void>(fixture.pass(2.0));
  require(cache.taken.size() == first, "the settled reading took more than the deficit");
}

// A pool's floor bounds every take: a pool whose floor is its cap gives nothing, one below it gives down to the floor.
void testPoolFloorBoundsEveryTake() {
  for (const uint32_t floor : {20u, 17u}) {
    FakeMemory memory({});
    FakeElasticPool experts("experts", memory, 256 * kMiB, floor, 20, 21, 20);
    MemoryLedger ledger;
    ledger.addPool(experts);
    const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
    static_cast<void>(ledger.free(claim, {.bytes = 2 * kGiB}, 0, false));
    static_cast<void>(ledger.free(claim, {.bytes = 512 * kMiB}, 0, false));
    require(experts.units() == (floor == 20 ? 20 : 18),
            "a claim took a pool below its floor or less than the need");
    static_cast<void>(ledger.free(claim, {.bytes = 256 * kMiB}, 0, false));
    require(experts.units() == floor, "the pool did not give units down to its floor");
  }
}

// Each decision logs one line; a retry that decides the same again logs nothing.
void testEachDecisionLogsOneLine() {
  MemoryLedger ledger;
  std::vector<std::string> lines;
  ledger.setDecisionLog([&](std::string_view line) { lines.emplace_back(line); });
  TestSource cache("cache");
  cache.units = {unit(1, kGiB)};
  ledger.addSource(cache);
  const ClaimId claim = ledger.open(request(4, RequestPriority::Normal));
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false));
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false));
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false));
  require(lines.size() == 2 && lines[0].find("request 4") != std::string::npos &&
              lines[0].find(": met, took cache") != std::string::npos &&
              lines[1].find(": short") != std::string::npos,
          "the decisions were not logged once each: " + (lines.empty() ? std::string() : lines[0]));
}

// A claim that flips between waiting for what lands and short of it as
// copies land logs each outcome once for each kind of need until an
// allocation of it is granted, however much the need still lacks: six
// retries of a need that shrinks as they go log two lines, a need of pages
// its own short, and past the grant the first need logs again.
void testARetryBackToALoggedOutcomeLogsNothing() {
  MemoryLedger ledger;
  std::vector<std::string> lines;
  ledger.setDecisionLog([&](std::string_view line) { lines.emplace_back(line); });
  uint64_t landing = 0;
  ledger.setReturning([&] { return Returning{.landingBytes = landing}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  const ClaimId claim = ledger.open(request(4, RequestPriority::Normal));
  for (uint32_t retry = 0; retry < 6; ++retry) {
    landing = retry % 2 ? kGiB : 0;
    static_cast<void>(ledger.free(claim, {.bytes = kGiB - retry * kMiB, .pages = 6 - retry}, 0, false));
  }
  static_cast<void>(ledger.free(claim, {.pages = 3}, 0, false));
  ledger.granted(claim);
  landing = 0;
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false));
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, kGiB, false));
  const std::array<std::string_view, 5> outcomes{": short", ": pending", ": short", ": short", ": met"};
  bool each = lines.size() == outcomes.size();
  for (size_t line = 0; each && line < lines.size(); ++line)
    each = lines[line].find(outcomes[line]) != std::string::npos;
  require(each, "a claim's retries logged " + std::to_string(lines.size()) + " lines, not one per need, outcome "
                "and grant");
}

// A need the host refused that nothing in place meets is met past the host
// (the host exemption) and its line says so, once, after a need of the
// claim met before: logged short, or not at all, a prefill the host refused
// every extent of read as short for 67 s while it ran.
void testAHostExemptionLogsMetPastTheHost() {
  Fixture fixture(host(32 * kGiB), 16 * kGiB);
  std::vector<std::string> lines;
  fixture.ledger().setDecisionLog([&](std::string_view line) { lines.emplace_back(line); });
  const ClaimId claim = fixture.ledger().open(request(4, RequestPriority::Normal));
  require(fixture.governor->free(claim, {.bytes = 128 * kMiB}).outcome == FreeOutcome::Met,
          "a need the room holds was not met");
  for (int retry = 0; retry < 3; ++retry)
    require(fixture.governor->free(claim, {.bytes = 128 * kMiB, .hostRefused = true}).outcome == FreeOutcome::Met,
            "a refused need of work in service was not met past the host");
  require(lines.size() == 2 && lines[0].find(": met") != std::string::npos &&
              lines[1].find(": met past the host") != std::string::npos,
          "the host exemption did not log one met line: " + (lines.empty() ? std::string() : lines.back()));
}

// The fixed take order: cache, rebuild-only pools, the lane's conversion, then the store, each only for what the
// ones before leave.
void testTheOrderTakesCacheThenRebuiltPoolsThenConversionsThenTheStore() {
  FakeMemory memory({});
  FakeElasticPool experts("experts", memory, kGiB, 2, 6, 7, 6);
  FakeElasticPool vision("vision", memory, 256 * kMiB, 0, 1, 1, 1);
  vision.rebuildOnly = true;
  MemoryLedger ledger;
  TestSource engine("engine"), cache("cache");
  // The lane of request 3, which its own growth converts.
  Victim lane = unit(3, kGiB, true);
  lane.kind = VictimKind::Conversion;
  engine.units = {lane};
  cache.units = {unit(1, 256 * kMiB)};
  ledger.addPool(experts);
  ledger.addSource(engine);
  ledger.addSource(cache);
  ledger.addPool(vision);
  const auto lacks = [&](uint64_t id, uint64_t bytes) {
    const ClaimId claim = ledger.open(growth(id, RequestPriority::Normal));
    static_cast<void>(ledger.free(claim, {.bytes = bytes}, 0, false));
    ledger.close(claim);
  };
  lacks(1, 256 * kMiB);
  require(cache.taken == std::vector<uint64_t>{1} && vision.units() == 1 && engine.taken.empty() &&
              experts.units() == 6,
          "the cache did not go first");
  lacks(2, 256 * kMiB);
  require(vision.units() == 0 && engine.taken.empty() && experts.units() == 6,
          "the pool that costs only its rebuild did not go before conversions and the store");
  lacks(3, kGiB);
  require(engine.taken == std::vector<uint64_t>{3} && experts.units() == 6,
          "the lane's conversion did not go before the store");
  lacks(4, kGiB);
  require(experts.units() == 5, "the store did not give a unit once nothing else was left");
}

// Growth counts what the host's reading does not show yet, so a lagging reading grows no unit that later goes back.
void testGrowthCountsWhatTheHostDoesNotShowYet() {
  // M + 1.5 GiB of headroom beside the store's 2 GiB.
  FakeMemory::Config config = host(2 * kGiB + kGiB + 1536 * kMiB);
  config.readingLagMilliseconds = 1000.0;
  Fixture fixture(std::move(config));
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 6, 7, 2);
  fixture.ledger().addPool(experts);
  fixture.memory.advanceTo(1000.0);
  fixture.governor->growToTargets(1000.0, MemoryPressure::Normal);
  for (double t = 1.5; t <= 10.0; t += 0.5)
    static_cast<void>(fixture.pass(t));
  require(experts.units() == 3 && experts.shrinks == 0,
          "growth admitted against a reading that did not show it: " + std::to_string(experts.grows) + " grown, " +
              std::to_string(experts.shrinks) + " given back");
}

// Growth leaves the host what the lanes in service will grow into, or the host would take the unit back.
void testGrowthLeavesTheHostWhatTheLanesGrowInto() {
  Fixture fixture(host(2 * kGiB + kGiB + 1536 * kMiB));
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 6, 7, 2);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addPool(experts);
  ledger.setLanes(kGiB, RequestPriority::Normal);
  // Past the quiet time, where only the lanes' projection holds the store.
  static_cast<void>(fixture.pass(kRegrowQuietSeconds + 1.0));
  require(experts.units() == 2, "the store grew into the host's margin the lanes are projected to grow into");
  ledger.setLanes(0, std::nullopt);
  static_cast<void>(fixture.pass(kRegrowQuietSeconds + 1.5));
  require(experts.units() == 3, "the store did not grow once the lanes projected nothing");
}

// Only a request claim that is not met, or that takes costly units, restarts the regrowth quiet period; claims met
// from cache do not.
void testRequestsMetFromCacheLeaveRegrowthQuiet() {
  Fixture fixture(host(32 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 2);
  fixture.memory.hold("weights", 3 * kGiB);
  fixture.ledger().addPool(experts);
  TestSource cache("cache");
  for (uint64_t key = 1; key <= 8; ++key)
    cache.units.push_back(unit(key, 256 * kMiB));
  fixture.ledger().addSource(cache);
  const auto request = [&](uint64_t id, uint64_t bytes, ClaimClass cls = ClaimClass::InService) {
    const ClaimId claim = fixture.ledger().open(::request(id, RequestPriority::Normal, cls));
    const FreeOutcome outcome = fixture.ledger().free(claim, {.bytes = bytes}, 0, false).outcome;
    fixture.ledger().close(claim);
    return outcome;
  };
  for (uint64_t id = 1; id <= 5; ++id) {
    static_cast<void>(fixture.pass(5.0 * double(id)));
    require(request(id, 256 * kMiB) == FreeOutcome::Met, "a request was not met from cache");
  }
  static_cast<void>(fixture.pass(kRegrowQuietSeconds));
  require(experts.units() == 3, "requests met from cache held the store from regrowing");
  // Short at 45 s, past the try at 30 s.
  static_cast<void>(fixture.pass(45.0));
  require(request(6, 64 * kGiB, ClaimClass::Start) == FreeOutcome::Short, "the oversized start was not short");
  static_cast<void>(fixture.pass(45.0 + kRegrowQuietSeconds - 0.5));
  require(experts.units() == 3, "the store regrew within the quiet time after a short claim");
  static_cast<void>(fixture.pass(45.0 + kRegrowQuietSeconds + 0.5));
  require(experts.units() == 4, "the store did not regrow a quiet time after the short claim");
  // Met at 80 s by the rest of the cache and a store chunk above the floor.
  static_cast<void>(fixture.pass(80.0));
  require(request(7, 768 * kMiB + kGiB) == FreeOutcome::Met && experts.units() == 3,
          "the claim was not met by the cache and a store chunk");
  static_cast<void>(fixture.pass(80.0 + kRegrowQuietSeconds - 0.5));
  require(experts.units() == 3, "the store regrew within the quiet time after a claim took a store chunk");
  static_cast<void>(fixture.pass(80.0 + kRegrowQuietSeconds + 0.5));
  require(experts.units() == 4, "the store did not regrow a quiet time after the claim that took a chunk");
}

// Under macOS's warning only a unit the cache funds whole regrows, never one that needs free room or any while the
// host has a deficit.
void testWarningRegrowsOnlyWhatTheCacheFunds() {
  enum class Case { Funded, Deficit, FreeRoom };
  for (const Case each : {Case::Funded, Case::Deficit, Case::FreeRoom}) {
    const uint64_t cacheBytes = each == Case::FreeRoom ? 0 : each == Case::Deficit ? 3 * 512 * kMiB : kGiB;
    const uint64_t limit = each == Case::FreeRoom ? 8 * kGiB : 5 * kGiB + cacheBytes;
    const uint64_t headroom = each == Case::Deficit ? 800 * kMiB : 8 * kGiB;
    Fixture fixture(host(headroom + 5 * kGiB + cacheBytes, MemoryPressure::Warning), limit);
    FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 3, 4, 2);
    fixture.memory.hold("weights", 3 * kGiB);
    TestSource cache("cache", &fixture.memory);
    fixture.ledger().addPool(experts);
    if (cacheBytes) {
      fixture.memory.hold("cache", cacheBytes);
      for (uint64_t key = 1; key * 512 * kMiB <= cacheBytes; ++key)
        cache.units.push_back(unit(key, 512 * kMiB));
      fixture.ledger().addSource(cache);
    }
    if (each == Case::Deficit) {
      static_cast<void>(fixture.pass(kRegrowQuietSeconds - 0.5));
      require(cache.taken.size() == 1, "the deficit pass did not take one unit");
      fixture.memory.hold("lanes", 512 * kMiB);
      static_cast<void>(fixture.pass(kRegrowQuietSeconds));
      require(experts.units() == 2 && cache.taken.size() == 1, "a unit regrew while the host had a deficit");
      continue;
    }
    static_cast<void>(fixture.pass(kRegrowQuietSeconds + 1.0));
    const bool grew = experts.units() == 3;
    if (each == Case::Funded)
      require(grew && cache.taken.size() == 2, "a unit the cache funds whole did not regrow under a warning");
    else
      require(!grew, "a unit into free room regrew under a warning");
  }
}

// Under macOS's warning a unit the cache funds whole regrows though the room lacks less than the unit: the host
// judges what the cache frees of the whole unit, not only of what the room lacks.
void testWarningRegrowsAUnitTheCacheFundsBeyondTheLack() {
  Fixture fixture(host(8 * kGiB + 7 * kGiB, MemoryPressure::Warning), 7 * kGiB + 512 * kMiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 3, 4, 2);
  fixture.memory.hold("weights", 3 * kGiB);
  fixture.memory.hold("cache", 2 * kGiB);
  TestSource cache("cache", &fixture.memory);
  cache.units = {unit(1, 2 * kGiB)};
  fixture.ledger().addPool(experts);
  fixture.ledger().addSource(cache);
  static_cast<void>(fixture.pass(kRegrowQuietSeconds + 1.0));
  require(experts.units() == 3 && cache.taken == std::vector<uint64_t>{1},
          "a unit the cache funds whole did not regrow under a warning when the room lacked less than the unit");
}

// A bytes need beside buffers no state can give is met by the bytes of the states it evicts, which leave the
// model's pool; its decision says it lacks such a buffer, not a count of them.
void testABytesNeedBesideBuffersNoStateGivesFreesTheirBytes() {
  MemoryLedger ledger;
  std::vector<std::string> lines;
  ledger.setDecisionLog([&](std::string_view line) { lines.emplace_back(line); });
  BufferPool pool(64 * kMiB);
  CachedStates cache(pool);
  cache.units.push_back({.key = 1, .stateBuffers = 1});
  ledger.addSource(cache);
  ledger.addSource(pool);
  ledger.setBufferBytes([&pool](uint32_t states, bool, bool) { return pool.releases(states); });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal, ClaimClass::Start));
  const FreeResult freed = ledger.free(claim, {.bytes = 48 * kMiB, .unheldStateBuffer = true}, 0, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 64 * kMiB &&
              cache.taken == std::vector<uint64_t>{1} && !pool.idle,
          "the evicted state's buffers stayed in the pool for a need no state meets in place");
  require(lines.size() == 1 && lines[0].find(", a state buffer no cached state holds") != std::string::npos &&
              lines[0].find("4294967295") == std::string::npos,
          "the decision did not say the need lacks a buffer no cached state holds: " +
              (lines.empty() ? std::string() : lines[0]));
}

// A need of only a buffer no cached state holds reaches no cached state, so it asks no copies ahead.
void testABufferNoStateHoldsReachesNoCachedState() {
  MemoryLedger ledger;
  BufferPool pool(64 * kMiB);
  CachedStates cache(pool);
  cache.units.push_back({.key = 1, .stateBuffers = 1, .write = true, .later = true});
  ledger.addSource(cache);
  ledger.addSource(pool);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal, ClaimClass::Start));
  const FreeResult freed = ledger.free(claim, {.hostRefused = true, .unheldStateBuffer = true}, 0, false);
  require(freed.outcome == FreeOutcome::Short && cache.taken.empty() && !ledger.copyDemand(),
          "a need no cached state meets reached an uncopied state and asked copies ahead");
}

// The pools grow before Ready by the host rule at the current pressure: a warning holds them at their floors.
void testStartGrowthFollowsThePressureNow() {
  for (const MemoryPressure pressure : {MemoryPressure::Warning, MemoryPressure::Normal}) {
    Fixture fixture(host(32 * kGiB, MemoryPressure::Normal, {at(100.0, 32 * kGiB, pressure)}));
    FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 2);
    fixture.ledger().addPool(experts);
    fixture.memory.advanceTo(100.0);
    fixture.governor->growToTargets(100.0, fixture.memory.pressure());
    require(experts.units() == (pressure == MemoryPressure::Warning ? 2u : 4u),
            pressure == MemoryPressure::Warning ? "the start's growth went on under a warning that began after the "
                                                  "governor was built"
                                                : "the start's growth did not reach the cap at normal pressure");
  }
}

} // namespace

std::span<const test::TestCase> ledgerTests() {
  static constexpr test::TestCase kTests[] = {
      {"testAccountsSumToCharged", testAccountsSumToCharged},
      {"testSnapshotRendersEveryAccount", testSnapshotRendersEveryAccount},
      {"testWaitingClaimHoldsRoomForItsRank", testWaitingClaimHoldsRoomForItsRank},
      {"testShortClaimHoldsNothing", testShortClaimHoldsNothing},
      {"testNestedScopesRestore", testNestedScopesRestore},
      {"testClaimIdsNeverRepeat", testClaimIdsNeverRepeat},
      {"testHeldPagesFollowRank", testHeldPagesFollowRank},
      {"testWaitingBytesHoldPagesOfEachForm", testWaitingBytesHoldPagesOfEachForm},
      {"testAMetNeedKeepsTheWaitAShortOneEndsIt", testAMetNeedKeepsTheWaitAShortOneEndsIt},
      {"testOneHostRule", testOneHostRule},
      {"testTheHostJudgesWhatALaneNeedsAsItsGrowth", testTheHostJudgesWhatALaneNeedsAsItsGrowth},
      {"testWhatALaneNeedsIsMetInPlaceBeforeTheHost", testWhatALaneNeedsIsMetInPlaceBeforeTheHost},
      {"testBandWaivedWhenTheHostClaimFindsNothing", testBandWaivedWhenTheHostClaimFindsNothing},
      {"testEveryHostPassGivesBackIdleBuffers", testEveryHostPassGivesBackIdleBuffers},
      {"testACriticalPassThatFindsNothingIsShortWhileMemoryLands",
       testACriticalPassThatFindsNothingIsShortWhileMemoryLands},
      {"testDeficitCountsHostRefusedClaimsNotAnIdleMargin", testDeficitCountsHostRefusedClaimsNotAnIdleMargin},
      {"testAHostRefusedStartIsAdmittedWithinTwoPasses", testAHostRefusedStartIsAdmittedWithinTwoPasses},
      {"testGrowthLeavesTheHostWhatAStartItRefusedWaitsFor", testGrowthLeavesTheHostWhatAStartItRefusedWaitsFor},
      {"testSourceOrderIsKeptWithinASource", testSourceOrderIsKeptWithinASource},
      {"testUnitsKeyedByTheirPlaceAreTakenAsChosen", testUnitsKeyedByTheirPlaceAreTakenAsChosen},
      {"testThePoolsAndWritesEachClaimTakes", testThePoolsAndWritesEachClaimTakes},
      {"testAFloorClaimTakesNothing", testAFloorClaimTakesNothing},
      {"testTakeRightsTable", testTakeRightsTable},
      {"testAHostRefusedNeedTakesOnlyWhatFreesNow", testAHostRefusedNeedTakesOnlyWhatFreesNow},
      {"testRoomHeldForHigherRanksMakesAWait", testRoomHeldForHigherRanksMakesAWait},
      {"testUnitsHeldForHigherRanksMakeAWait", testUnitsHeldForHigherRanksMakeAWait},
      {"testLandingThenVictimsThenOwed", testLandingThenVictimsThenOwed},
      {"testImmediateSkipsLandings", testImmediateSkipsLandings},
      {"testImmediateTakesAWriteThatFreesNowAfterRebuiltPools", testImmediateTakesAWriteThatFreesNowAfterRebuiltPools},
      {"testLandingCountsBeforeVictimsInPart", testLandingCountsBeforeVictimsInPart},
      {"testStateBuffersCountBeyondThoseThePoolKeeps", testStateBuffersCountBeyondThoseThePoolKeeps},
      {"testStateBuffersMeetANeedOfBytes", testStateBuffersMeetANeedOfBytes},
      {"testTheBuffersALaneStartsFromGoLast", testTheBuffersALaneStartsFromGoLast},
      {"testIdleBuffersThatCoverARefusalTakeNothingInPlace", testIdleBuffersThatCoverARefusalTakeNothingInPlace},
      {"testHostRefusedNeedIsMetInPlaceFirst", testHostRefusedNeedIsMetInPlaceFirst},
      {"testAnExemptRetryPassesTheHostWithEachReservation", testAnExemptRetryPassesTheHostWithEachReservation},
      {"testZeroCostUnsizedUnitsGoFirst", testZeroCostUnsizedUnitsGoFirst},
      {"testTheHostClaimCountsTheIdleBuffersInItsTarget", testTheHostClaimCountsTheIdleBuffersInItsTarget},
      {"testAClaimThatTakesNoUnitsTakesNone", testAClaimThatTakesNoUnitsTakesNone},
      {"testAClaimThatTakesNoUnitsCountsNone", testAClaimThatTakesNoUnitsCountsNone},
      {"testAvailableTakesNothing", testAvailableTakesNothing},
      {"testCountsStopAtTheLack", testCountsStopAtTheLack},
      {"testAvailableCountsWhatLandsForAClaimThatCanWait", testAvailableCountsWhatLandsForAClaimThatCanWait},
      {"testGrowthRetakesFreeRoomBeyondLaneDemand", testGrowthRetakesFreeRoomBeyondLaneDemand},
      {"testGrowthTakesIdleCacheAfterABurst", testGrowthTakesIdleCacheAfterABurst},
      {"testCriticalTakesTheMarginASettledSecondInTheOrder", testCriticalTakesTheMarginASettledSecondInTheOrder},
      {"testCriticalTakesWhatItFindsWhateverLands", testCriticalTakesWhatItFindsWhateverLands},
      {"testWarningWithoutADeficitTakesNothing", testWarningWithoutADeficitTakesNothing},
      {"testDeficitNetOfReturning", testDeficitNetOfReturning},
      {"testMissingTelemetryGivesNoCacheBack", testMissingTelemetryGivesNoCacheBack},
      {"testRerunWhileTheDeficitWaitsOnLandings", testRerunWhileTheDeficitWaitsOnLandings},
      {"testTheHostClaimTakesTheDeficitBeyondWhatLands", testTheHostClaimTakesTheDeficitBeyondWhatLands},
      {"testAHostPassWaitingOnLandingsRunsOncePerSettle", testAHostPassWaitingOnLandingsRunsOncePerSettle},
      {"testAWarningDeficitTakesNoPoolUnit", testAWarningDeficitTakesNoPoolUnit},
      {"testTheHostTakesNoStoreUnitShortOfCritical", testTheHostTakesNoStoreUnitShortOfCritical},
      {"testAPoolDoesNotGrowWhileAClaimWaits", testAPoolDoesNotGrowWhileAClaimWaits},
      {"testOnlyARequestsWaitingClaimHoldsFreeRoom", testOnlyARequestsWaitingClaimHoldsFreeRoom},
      {"testAClaimLacksTheRoomHeldAboveIt", testAClaimLacksTheRoomHeldAboveIt},
      {"testANeedMetInPlaceLacksNoHeldBytes", testANeedMetInPlaceLacksNoHeldBytes},
      {"testShortStartRegrowsWhenTheHostFrees", testShortStartRegrowsWhenTheHostFrees},
      {"testALoneConvertedLaneWaitsForItsOwnCopies", testALoneConvertedLaneWaitsForItsOwnCopies},
      {"testARetryAfterItsConversionTakesNothingElse", testARetryAfterItsConversionTakesNothingElse},
      {"testOneConversionDoesNotStallTheStore", testOneConversionDoesNotStallTheStore},
      {"testCopiesAheadOnlyAfterDemandReachedUncopiedCache", testCopiesAheadOnlyAfterDemandReachedUncopiedCache},
      {"testAShortfallTakesAtMostOneUnitBeyondIt", testAShortfallTakesAtMostOneUnitBeyondIt},
      {"testALaggingHostReadingTakesNoSecondRelease", testALaggingHostReadingTakesNoSecondRelease},
      {"testPoolFloorBoundsEveryTake", testPoolFloorBoundsEveryTake},
      {"testEachDecisionLogsOneLine", testEachDecisionLogsOneLine},
      {"testARetryBackToALoggedOutcomeLogsNothing", testARetryBackToALoggedOutcomeLogsNothing},
      {"testAHostExemptionLogsMetPastTheHost", testAHostExemptionLogsMetPastTheHost},
      {"testTheOrderTakesCacheThenRebuiltPoolsThenConversionsThenTheStore",
       testTheOrderTakesCacheThenRebuiltPoolsThenConversionsThenTheStore},
      {"testGrowthCountsWhatTheHostDoesNotShowYet", testGrowthCountsWhatTheHostDoesNotShowYet},
      {"testGrowthLeavesTheHostWhatTheLanesGrowInto", testGrowthLeavesTheHostWhatTheLanesGrowInto},
      {"testRequestsMetFromCacheLeaveRegrowthQuiet", testRequestsMetFromCacheLeaveRegrowthQuiet},
      {"testWarningRegrowsOnlyWhatTheCacheFunds", testWarningRegrowsOnlyWhatTheCacheFunds},
      {"testWarningRegrowsAUnitTheCacheFundsBeyondTheLack", testWarningRegrowsAUnitTheCacheFundsBeyondTheLack},
      {"testABytesNeedBesideBuffersNoStateGivesFreesTheirBytes",
       testABytesNeedBesideBuffersNoStateGivesFreesTheirBytes},
      {"testABufferNoStateHoldsReachesNoCachedState", testABufferNoStateHoldsReachesNoCachedState},
      {"testStartGrowthFollowsThePressureNow", testStartGrowthFollowsThePressureNow},
  };
  return kTests;
}

// memory-ledger [--list] [--filter ...]
int main(int argc, char **argv) {
  const test::TestGroup groups[] = {{"ledger", ledgerTests()}};
  return test::runTests("memory-ledger", groups, argc, argv, [] { test::metalStatistics() = {}; });
}
