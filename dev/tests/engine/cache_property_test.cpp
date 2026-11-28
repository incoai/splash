#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Cache.hpp"
#include "model/SlotFile.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

// A state the tier takes whenever reclaim or persist writes it, as long as
// the quota has room for its copy and the file takes writes: the restore
// points requests resume from. Its copies draw on the quota the KV tier's
// copies draw on, as the state file's slots do (openDiskTier).
struct Shelf final {
  std::shared_ptr<model::DiskBudget> budget;
  bool takesWrites = true;
};

class ShelvedState final : public CompositeState {
public:
  ShelvedState(std::shared_ptr<Shelf> shelf, bool disk) : shelf_(std::move(shelf)), disk_(disk) {}
  ~ShelvedState() override {
    if (disk_) shelf_->budget->release(bytes());
  }
  uint64_t bytes() const noexcept override { return 100; }
  uint64_t residentBytes() const noexcept override { return disk_ ? 0 : bytes(); }
  bool writable() const noexcept override { return !disk_; }
  bool canOffload() const noexcept override { return !disk_ && shelf_->takesWrites; }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override { return shelve(); }
  std::unique_ptr<StateOffload> persist(std::function<void()>) const override { return shelve(); }

private:
  std::unique_ptr<StateOffload> shelve() const {
    if (!shelf_->budget->reserve(bytes())) return {};
    struct Written final : StateOffload {
      std::shared_ptr<const CompositeState> disk;
      bool ready() const noexcept override { return true; }
      bool finish() override { return true; }
      const std::shared_ptr<const CompositeState> &state() const noexcept override { return disk; }
    };
    auto written = std::make_unique<Written>();
    written->disk = std::make_shared<ShelvedState>(shelf_, true);
    return written;
  }

  std::shared_ptr<Shelf> shelf_;
  bool disk_;
};

// One randomized run of the tiered block store: requests publish, spill, restore, convert, release and end
// beside reclaims, quota and failing transfers; every step checks Cache::checkInvariants. A run that spills does
// so under --kv-spill always in half the runs (CacheSpillPolicy::always).
class Run final {
public:
  // With peeks, random claims count the walk's units and take them too
  // (checkTake).
  // With usedStates, a request uses the state it resumes from, as the
  // engine's requests do (Cache::useState).
  // Tight, a windowed run's conversations outgrow its budget, as long ones
  // do at the memory wall: up to 36 blocks on 16 to 48 blocks' pages, and a
  // state on 60% of growths, so that the budget refuses Cache::ensureTokens.
  // A windowed run's blocks hold window units of 0 to 3 pages, and copies of
  // them a window part of 100 bytes a page beside their base part's 100.
  // Loose, 3 conversations of 12 pages never reach the 96 of the budget.
  // Forked, a tight windowed run's conversations part from one another: the
  // second base prompt shares the first's three pages, the third its first
  // two, so that the states of conversations on one head pin its blocks
  // together, as a regenerated reply's or a second conversation's do.
  // Promoting, a start that read its state back from the SSD promotes it to
  // RAM half the time, its copy a claim that reclaims (start).
  // Sparse, sparse history spaces its writes every 4 to 8 pages by the seed
  // (Cache::sparseHistory), so that its chains of 12 or 36 pages reach the
  // spacing; else at the cache's default, past every chain.
  Run(uint32_t seed, bool windowed, bool peeks = false, bool usedStates = false, bool tight = false,
      bool forked = false, bool promotes = false, bool sparse = true)
      : random_(seed), windowed_(windowed), peeks_(peeks), usedStates_(usedStates), promotes_(promotes),
        tight_(windowed && tight),
        unitPages_(windowed ? pick(0, 3) : 0), tier_(chance(50), 100 * unitPages_),
        budget_(std::make_shared<model::DiskBudget>(kSlots * tier_.slotBytes())),
        spill_(windowed ? std::nullopt : std::optional(CacheSpillPolicy{pick(0, 2), pick(1, 2), chance(50)})),
        cache_(pool_, &tier_, budget_, spill_,
               windowed ? std::optional(CacheWindowPolicy{pick(1, 3), unitPages_}) : std::nullopt),
        maximumPages_(tight_ ? 36 : 12) {
    tier_.budget = budget_;
    tier_.transferLimit = 64;
    tier_.writeLimits = {8, 4, 8};
    // Each block takes its base page and its window unit.
    storage_.budgetPages = (tight_ ? pick(16, 48) : 96) * (1 + unitPages_);
    // Every count a peek or a take makes of what a reclaim reaches follows
    // its rule over the whole cache, as kv-page-cache's
    // testPageReachFollowsItsRule states it.
    cache_.checkReaches(true);
    // Every release of a state frees exactly the window units its count
    // names (Cache::releaseState).
    cache_.checkReleases(true);
    // And each step of a take's walk checks the answers the cache keeps
    // before it reads one: a take of many releases reads them between its
    // own changes.
    cache_.checkKeptAnswers(true);
    cache_.checkResumes(true);
    cache_.checkTails(true);
    if (sparse)
      cache_.sparseHistory((4 + seed % 5) * KvCache::pageTokens);
    for (uint32_t base = 0; base < bases_.size(); ++base) {
      bases_[base].resize(maximumPages_ * KvCache::pageTokens + 1);
      for (uint32_t row = 0; row < bases_[base].size(); ++row)
        bases_[base][row] = 1000 * (base + 1) + row;
    }
    if (forked)
      for (uint32_t base = 1; base < bases_.size(); ++base)
        std::copy_n(bases_[0].begin(), (4 - base) * KvCache::pageTokens, bases_[base].begin());
  }

  void step() {
    switch (pick(0, peeks_ ? 17 : 16)) {
    case 0: start(); break;
    case 1: case 2: grow(); break;
    case 3: static_cast<void>(cache_.writeThrough()); break;
    case 4: land(); break;
    case 5:
      if (!windowed_)
        static_cast<void>(
            cache_.spillWritten(chance(50) ? SpilledExtents::KeepRunway : SpilledExtents::Released).additionalPages);
      break;
    case 6: reclaim(); break;
    case 7: keepQuota(); break;
    case 8: convert(); break;
    case 9: failRead(); break;
    case 10: finish(); break;
    case 11: command(); break;
    case 12: persist(); break;
    case 13: failWrites(); break;
    case 14: release(); break;
    case 15: countReleases(); break;
    case 16: releaseExtents(); break;
    case 17: checkTake(); break;
    }
    cache_.checkInvariants();
  }

  uint64_t steps = 0;
  // Growths the budget refused in the end, and admissions refused again for a
  // need the ledger had met.
  uint64_t refusedGrowths = 0, refusedAfterMet = 0;
  // Units taken, by whether their takes freed, in the need's measure, all
  // peek counted when it first offered them (exact) or less (short), and of
  // those offered as needing no write the same: a short one of those is one
  // of the classes checkTake names.
  uint64_t exactUnits = 0, shortUnits = 0, exactNoWrite = 0, shortNoWrite = 0;
  // Takes that wrote and whose room on the SSD gave up a state holding
  // window units (Quota, checkTake).
  uint64_t quotaTakes = 0;
  // States read back from the SSD and promoted, their copies' claims reclaiming.
  uint64_t promotions = 0;
  // Claims whose count stopped short of the full count once it covered
  // their need (checkTake).
  uint64_t boundedCounts = 0;
  // Device needs in place among the claims (checkTake), drawn only where
  // inPlaceNeeds_ is set: the pinned seeds keep the claims they were found
  // with.
  bool inPlaceNeeds_ = false;
  uint64_t inPlaceClaims = 0;
  // Releases of whole extents for bytes (releaseExtents): those taken, those whose last extent the free Full pages
  // beside completed, those beside transfers in flight, and those that freed less than their count where the runway
  // held less than an extent before, or where the runway's refill was refused and moved fewer pages.
  uint64_t extentReleases = 0, besideReleases = 0, inFlightReleases = 0, shortRunwayReleases = 0, refusedReleases = 0;

  // Every request ends, every transfer lands, and nothing leaks: the pool's
  // pages and the tier's slots belong to the cache's blocks alone.
  void drain() {
    endRequests();
    tier_.complete();
    static_cast<void>(cache_.pollTransfers());
    cache_.checkInvariants();
    const KvPoolSnapshot pool = pool_.snapshot();
    require(pool.pagesActive == 0 && budget_->committedBytes() == 0 && cache_.spillSnapshot().heldBytes == 0,
            "a finished run left pages held, quota committed or copies held");
    // The quota counts what the cache's copies hold and no more: each KV
    // copy's slot and each state.
    const CacheSnapshot held = cache_.snapshot();
    require(budget_->usedBytes() == held.kvTier.diskBytes + held.stateCache.diskBytes,
            "the quota counts other than what the cache's copies hold");
  }

  // Every request ends; the transfers in flight stay so.
  void endRequests() {
    while (!requests_.empty()) {
      sources_.erase(requests_.begin()->first);
      cache_.endRequest(requests_.begin()->first);
      requests_.erase(requests_.begin());
    }
  }

  // The quota is whole again, and in half the runs each the KV tier and the
  // state file take no more writes; what was in flight has landed.
  void settle() {
    budget_->setLimitBytes(kSlots * 100);
    budget_->setActiveLimitBytes(kSlots * 100);
    cache_.keepDiskLimit();
    tier_.writableFile = chance(50);
    shelf_->takesWrites = chance(50);
    tier_.complete();
    static_cast<void>(cache_.pollTransfers());
    cache_.checkInvariants();
  }

  // A claim's takes never free more than peek counted for the units from
  // the head of the order through each (VictimSource's contract), and
  // never take in-use or certain-reuse units its rights do not take. A unit
  // peek offers as needing no write starts none when taken; one that needs
  // a write but frees now (the state the idle staging buffer writes) starts
  // at most that state's write and is one state; a state that frees only
  // once a write lands gives nothing when taken.
  void checkTake() {
    const ClaimClass cls = std::array{ClaimClass::InService, ClaimClass::Start, ClaimClass::Optional,
                                      ClaimClass::Growth, ClaimClass::Host}[pick(0, 4)];
    // A request's claim, but the host's.
    const ClaimSpec claim{.cls = cls,
                          .requestId = cls == ClaimClass::Host ? 0u : 1u,
                          .reach = chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary,
                          .timing = chance(50) ? NeedTiming::Immediate : NeedTiming::CanWait};
    const kv::PageForm form = !chance(50) || windowed_ ? kv::PageForm::Full : kv::PageForm::Index;
    const uint32_t stateBuffers = chance(30) ? 1u : 0u;
    // A device need, a page need, or a need of state buffers alone.
    const uint32_t kind = pick(0, 2);
    const bool device = kind == 0;
    const bool states = kind == 2;
    // A device need in place, as the engine makes it when refused memory
    // for a lane's pages: bytes and pages of its form, met by either
    // (MemoryLedger's covered), its units counted by those pages (reach).
    const bool inPlace = device && inPlaceNeeds_ && chance(50);
    const uint32_t placePages = inPlace ? pick(1, 8) : 0;
    inPlaceClaims += inPlace ? 1 : 0;
    const VictimQuery query =
        test::query(claim,
                    inPlace  ? Need{.bytes = pick(1, 8) * 400ull, .pages = placePages, .form = form,
                                    .stateBuffers = stateBuffers}
                    : device ? Need{.bytes = pick(1, 8) * 400ull, .stateBuffers = stateBuffers}
                    : states ? Need{.hostRefused = chance(50), .stateBuffers = pick(1, 2)}
                             : Need{.hostRefused = chance(50), .pages = pick(1, 8), .form = form,
                                    .stateBuffers = stateBuffers},
                    chance(10));
    const bool immediate = !query.rights.waitsForWrites;
    std::vector<uint64_t> inUse;
    for (const auto &[_, request] : requests_)
      if (request.state && cache_.stateResident(request.state))
        inUse.push_back(request.state);
    // With states in use, a request waits to start on a prefix of a base
    // prompt: the state its chain restores is of certain reuse, which only
    // work in service and the host under critical pressure take.
    const bool takesCertain = query.rights.certainReuse == TakeRights::CertainReuse::All;
    std::vector<uint64_t> wanted;
    if (usedStates_) {
      const std::span<const uint32_t> base = bases_[pick(0, static_cast<uint32_t>(bases_.size()) - 1)];
      const CacheProbe probe = cache_.probe(base.first(pick(1, maximumPages_ - 2) * KvCache::pageTokens + 1), {});
      wanted.assign(probe.restoredChain().begin(), probe.restoredChain().end());
      cache_.setWantedChains({wanted});
    }
    const bool wantedResident = !wanted.empty() && cache_.stateResident(wanted.back());
    // A unit's measure: bytes for a device need, pages of its form for a
    // page need, buffers for a state need.
    // In place, the pages of its form it keeps free, and beside them the
    // bytes of the other forms' extents (bytesThrough), either of which the
    // ledger's covered() takes.
    const auto measure = [&](const engine::Victim &unit) -> uint64_t {
      return device && !inPlace ? unit.bytes : states ? unit.stateBuffers : unit.form == form ? unit.pages : 0;
    };
    const auto gaveOf = [&](const FreeResult &taken) -> uint64_t {
      return device && !inPlace ? taken.nowBytes + taken.laterBytes : states ? taken.stateBuffers : taken.pages;
    };
    const auto bytesThrough = [&](const engine::Victim &unit) {
      uint64_t sum = 0;
      for (std::optional<engine::Victim> each = cache_.peek(query, nullptr); each; each = cache_.peek(query, &*each)) {
        sum += each->later && immediate ? 0 : each->bytes;
        if (each->key == unit.key)
          break;
      }
      return sum;
    };
    // What the units from the head of the cache's order through `unit`
    // count, as the ledger counts them (MemoryLedger::bound): a claim that
    // does not wait for writes passes over what lands later.
    const auto throughUnit = [&](const engine::Victim &unit) {
      uint64_t sum = 0;
      for (std::optional<engine::Victim> each = cache_.peek(query, nullptr); each; each = cache_.peek(query, &*each)) {
        require(!each->later || each->write, "a unit that frees once a write lands was not marked as needing one");
        require(!each->write || each->later || each->stateBuffers <= 1,
                "a unit that frees now through a write counted more than the one state the staging buffer takes");
        if (!(each->later && immediate))
          sum += measure(*each);
        if (each->key == unit.key)
          break;
      }
      return sum;
    };
    // A count stops once it covers the need and the most one take frees
    // past it (Cache::reach): the units it offers count no more, through
    // each, than the count of all the walk reaches, and cover the need
    // exactly when that count does; the takes below check that each frees
    // no more than the bounded count offered through its unit.
    if (!states) {
      const auto counts = [&](bool bounded) {
        cache_.boundCounts(bounded);
        std::map<uint64_t, uint64_t> through;
        uint64_t sum = 0;
        for (std::optional<engine::Victim> each = cache_.peek(query, nullptr); each; each = cache_.peek(query, &*each)) {
          sum += each->later && immediate ? 0 : measure(*each);
          through[each->key] = sum;
        }
        cache_.boundCounts(true);
        return std::pair{through, sum};
      };
      const auto [bounded, boundedSum] = counts(true);
      const auto [full, fullSum] = counts(false);
      const uint64_t need = device && !inPlace ? query.need.bytes : query.need.pages;
      require((boundedSum >= need) == (fullSum >= need) && boundedSum <= fullSum,
              "the bounded count gave " + std::to_string(boundedSum) + " where the full count gave " +
                  std::to_string(fullSum) + " of a need of " + std::to_string(need));
      for (const auto &[key, sum] : bounded)
        if (const auto other = full.find(key); other != full.end())
          require(sum <= other->second, "the bounded count through a unit gave " + std::to_string(sum) +
                                            ", the full count " + std::to_string(other->second));
      boundedCounts += boundedSum < fullSum ? 1 : 0;
    }
    // What a take that gives something changes: the memory the pool holds,
    // its free, landing and cached pages, the states in RAM, the writes
    // started.
    const auto fingerprint = [&] {
      const CacheSnapshot snapshot = cache_.snapshot();
      return std::tuple{pool_.allocatedBytes(),          pool_.freePageCount(kv::PageForm::Full),
                        pool_.freePageCount(kv::PageForm::Index), cache_.demotedPages(),
                        snapshot.pool.pagesPrefix,
                        snapshot.stateCache.entries,     snapshot.stateCache.bytes,
                        snapshot.stateCache.offloads,    tier_.demotions};
    };
    // Each unit's count when it was first offered, what its takes freed in
    // the need's measure, and what they reported, by the classes of unit
    // that may free less than counted (below).
    struct Taken final {
      uint64_t counted = 0;
      uint64_t freed = 0;
      bool write = false;
      bool pending = false;
      // The landing its takes netted (FreeResult::nettedBytes).
      uint64_t netted = 0;
      std::string takes;
    };
    std::map<uint64_t, Taken> taken;
    // The takes as the ledger takes (MemoryLedger::take): the first unit from
    // the head of the order, again from the head after a take that gave
    // something, the next one after a take that gave nothing; a claim that
    // does not wait for writes passes over what lands later.
    // The landing the takes credited (laterBytes) and no release netted
    // since: what the next take may net (VictimQuery::landingCredited).
    uint64_t credited = 0, gaveAll = 0, gaveBytes = 0;
    // The claim as a whole, against what the pool really released (below):
    // what its takes reported now and later, and netted.
    uint64_t claimReported = 0, claimNetted = 0, claimQuota = 0;
    const uint64_t claimAllocated = pool_.allocatedBytes(), claimLanding = cache_.returning().landingBytes;
    for (std::optional<engine::Victim> unit = cache_.peek(query, nullptr); unit;) {
      if (unit->later && immediate) {
        unit = cache_.peek(query, &*unit);
        continue;
      }
      const uint64_t through = throughUnit(*unit);
      const bool firstTake = !taken.contains(unit->key);
      Taken &record = taken[unit->key];
      if (firstTake)
        record = {.counted = measure(*unit), .write = unit->write};
      const auto unchanged = fingerprint();
      const uint64_t before = pool_.allocatedBytes();
      const uint64_t offloads = cache_.snapshot().stateCache.offloads;
      const uint32_t demotions = tier_.demotions;
      const uint64_t landingBefore = cache_.returning().landingBytes;
      const uint64_t elsewhere = cache_.unitsFreedElsewhere();
      // The landing the takes before it credited, and what the claim lacks
      // past what they freed, as the ledger passes them: a take of the
      // cache's reclaims takes steps up to that lack.
      VictimQuery taking = query;
      taking.landingCredited = credited;
      taking.lackingBytes = device ? query.need.bytes - std::min<uint64_t>(query.need.bytes, gaveBytes) : 0;
      taking.lackingPages = (device && !inPlace) || states ? 0 : query.need.pages - std::min<uint64_t>(query.need.pages, gaveAll);
      const uint64_t throughBytes = inPlace ? bytesThrough(*unit) : 0;
      const FreeResult result = cache_.take(taking, *unit);
      // A take nets no more of the landing than the claim was credited and
      // still holds: by the takes before it and its own steps, and, waiting
      // for landings, the drop of the ledger's count of what lands
      // (reclaimStep).
      const uint64_t landingDrop = landingBefore - std::min(landingBefore, cache_.returning().landingBytes);
      require(result.nettedCredited <= result.nettedBytes && result.nettedCredited <= credited + result.laterBytes &&
                  result.nettedBytes - result.nettedCredited <= (query.rights.waitsForLandings ? landingDrop : 0),
              "a take netted " + std::to_string(result.nettedBytes) + " (" + std::to_string(result.nettedCredited) +
                  " of the claim's own credit of " + std::to_string(credited) + ")");
      credited = credited - std::min(credited, result.nettedCredited) + result.laterBytes;
      claimReported += result.nowBytes + result.laterBytes;
      claimNetted += result.nettedBytes;
      const uint64_t released = before - std::min(before, pool_.allocatedBytes());
      // Quota: a take that writes makes room on the SSD as the quota gives
      // copies up, and a state given up there frees the window units its
      // tail held, which no count can know (Cache::unitsFreedElsewhere), as
      // are those of a point the give-up left history that the same take
      // releases with no write (take-used forked seed 21777). Its
      // pages are subtracted from what the take freed, in the need's
      // measure: for bytes, the whole extents they fill at most.
      const uint64_t quotaPages = (cache_.unitsFreedElsewhere() - elsewhere) * unitPages_;
      require(unit->write || !quotaPages, "a take of a unit offered as needing no write gave copies up for the quota");
      const uint64_t extentBytes = uint64_t{pool_.extentPages()} * storage_.bytesPerPage(kv::PageForm::Full);
      const uint64_t quotaBytes = (quotaPages + pool_.extentPages() - 1) / pool_.extentPages() * extentBytes;
      const uint64_t quota = device && !inPlace ? quotaBytes
                             : states           ? 0
                                                : quotaPages;
      quotaTakes += quotaPages ? 1 : 0;
      claimQuota += quotaBytes;
      require(result.nowBytes <= released, "a take released " + std::to_string(released) + " bytes, reported " +
                                               std::to_string(result.nowBytes));
      const uint64_t written = cache_.snapshot().stateCache.offloads - offloads;
      require(unit->write || (!written && tier_.demotions == demotions),
              "a take of a unit offered as needing no write started one");
      require(!unit->write || unit->later || (written <= 1 && tier_.demotions == demotions),
              "a take of a unit offered as freeing now through the staging buffer started more than its state's "
              "write");
      require(!states || !unit->later || !result.gave(),
              "a take of a state offered as freeing once a write lands gave something now");
      // Peek never counts less than its take frees: a take frees no more
      // than the units from the head through it count, the units before it
      // being what the take may free first (free pages, the extents they
      // fill). Judged at each take against the units as then offered, a unit
      // offered again counting afresh: for a device need by bytes its bytes
      // now and later, in place its pages and, below, its bytes.
      require(gaveOf(result) - std::min(gaveOf(result), quota) <= through,
              "a take freed " + std::to_string(gaveOf(result)) + " (" + std::to_string(quota) +
                  " of it through the quota) where the units through it counted " + std::to_string(through));
      const uint64_t resultBytes = result.nowBytes + result.laterBytes;
      require(!inPlace || resultBytes - std::min(resultBytes, quotaBytes) <= throughBytes,
              "in place, a take freed " + std::to_string(resultBytes) + " bytes where the units through it counted " +
                  std::to_string(throughBytes));
      record.freed += gaveOf(result);
      gaveAll += gaveOf(result);
      gaveBytes += resultBytes;
      record.pending = record.pending || result.outcome == FreeOutcome::Pending;
      record.netted += result.nettedBytes;
      record.takes += " [" + std::to_string(measure(*unit)) + " counted, " + std::to_string(gaveOf(result)) +
                      " freed, " + std::to_string(released) + " released, " + std::to_string(result.nettedBytes) +
                      " netted, " + std::to_string(result.stateBuffers) +
                      " states, outcome " + std::to_string(static_cast<int>(result.outcome)) + "]";
      // Progress: a take that says it gave something changed the cache, so
      // the ledger's peek again from the head never finds the same unit for
      // ever; and a unit peek offers is one its first take gives something
      // of, or that waits for a transfer in flight, never a count of what
      // the walk cannot reach.
      require(!result.gave() || fingerprint() != unchanged, "a take reported giving something and changed nothing");
      require(!firstTake || result.gave() || result.outcome == FreeOutcome::Pending,
              "a unit peek offered gave nothing and waited for nothing at its first take");
      if (result.gave()) {
        ++steps;
        unit = cache_.peek(query, nullptr);
      } else {
        unit = cache_.peek(query, &*unit);
      }
    }
    // What the claim's takes reported and netted covers what the pool
    // released, read from the pool rather than from the takes' own account.
    // Not yet held (kLandingTold, the oracle of its own task): that a claim
    // is told no more, now and later, than the pool released and what lands
    // at its end, past the landing it was credited at its start when it
    // waits for landings (MemoryLedger::free). It fails from take plain seed
    // 10 on: a waiting claim credited 400 of landing by the ledger's count
    // (returning, no runway, spills beside) whose release took the free pages
    // that landing needed, which the take's count (landingFill, the runway
    // kept, demotions alone) never saw, and so reported as now.
    constexpr bool kLandingTold = false;
    if (device && !inPlace) {
      const uint64_t released = claimAllocated - std::min(claimAllocated, pool_.allocatedBytes());
      const uint64_t told = claimReported + (query.rights.waitsForLandings ? claimLanding : 0);
      require(!kLandingTold || told <= released + cache_.returning().landingBytes + claimQuota,
              "a claim's takes reported " + std::to_string(claimReported) + " where the pool released " +
                  std::to_string(released) + " and " + std::to_string(cache_.returning().landingBytes) + " lands");
      require(claimReported + claimNetted >= released,
              "a claim's takes reported " + std::to_string(claimReported) + " and netted " +
                  std::to_string(claimNetted) + " where the pool released " + std::to_string(released));
    }
    // Tightness: a unit offered as needing no write frees all it counted
    // when first offered, but for two classes, each named by what its takes
    // report:
    // - Pending: a take of it ended Pending, a transfer in flight holding
    //   the rest, which comes back as it lands.
    // - Netted: a device need by bytes whose take released an extent the
    //   landing pages would have completed, a landing the claim was credited
    //   and still held (checked at each take): by the ledger's count of what
    //   lands, or by its own takes before (take plain seed 1226654: a Later
    //   unit's evictions let a demotion complete an extent, credited later;
    //   the Now unit's release then took the free pages it needed). The take
    //   credits the release less the landing it cancels and reports that
    //   (FreeResult::nettedBytes), while the count keeps the release as its
    //   least (Cache::reach), so that a claim takes that memory now rather
    //   than waiting for the landing: what it freed and netted is exactly
    //   what it counted (testAReleaseOfWhatALandingCompletesCountsOnce,
    //   testAReleaseNetsTheLandingAnEarlierTakeCredited).
    for (const auto &[key, record] : taken) {
      const bool exact = record.freed >= record.counted;
      ++(exact ? exactUnits : shortUnits);
      if (!record.write)
        ++(exact ? exactNoWrite : shortNoWrite);
      require(exact || record.write || record.pending ||
                  (device && !inPlace && record.netted && record.freed + record.netted >= record.counted),
              "a unit offered as needing no write freed " + std::to_string(record.freed) + " of the " +
                  std::to_string(record.counted) + " it counted (unit " + std::to_string(key >> 56) + ":" +
                  std::to_string(key & 0xff) + (device ? ", device" : states ? ", states" : ", pages") +
                  (query.need.hostRefused ? ", scan" : "") + (immediate ? ", immediate" : "") +
                  "); its takes:" + record.takes);
    }
    for (const uint64_t block : inUse)
      require(query.rights.inUse || cache_.stateResident(block),
              "a claim that does not reach what is in use took a state in use");
    require(takesCertain || !wantedResident || cache_.stateResident(wanted.back()),
            "a claim that does not take what is of certain reuse took the state a waiting request restores");
    if (!wanted.empty())
      cache_.setWantedChains({});
  }

private:
  static constexpr uint64_t kSlots = 160;

  struct Request final {
    uint32_t base = 0;
    uint32_t pages = 0;
    KvMode mode = KvMode::Resident;
    uint64_t state = 0;  // the block of the state it uses
    StateUse use;
  };

  uint32_t pick(uint32_t first, uint32_t last) {
    return std::uniform_int_distribution<uint32_t>(first, last)(random_);
  }
  bool chance(uint32_t percent) { return pick(1, 100) <= percent; }
  std::span<const uint32_t> prompt(const Request &request) const {
    return std::span<const uint32_t>(bases_[request.base]).first(request.pages * KvCache::pageTokens + 1);
  }
  Request *anyRequest() {
    if (requests_.empty()) return nullptr;
    auto entry = requests_.begin();
    std::advance(entry, pick(0, static_cast<uint32_t>(requests_.size()) - 1));
    return &entry->second;
  }
  uint64_t idOf(const Request &request) const {
    for (const auto &[id, each] : requests_)
      if (&each == &request) return id;
    throw std::logic_error("unknown request");
  }
  bool waiting(uint64_t id) const { return cache_.kvRestoreStatus(id) == KvRestoreStatus::Pending; }

  // Admits as the engine does (Engine::allocate): a denial makes room in
  // reclaim steps, of the form denied, until the pages fit, nothing more
  // goes, or the need is one already freed for once: at the wall a need met
  // for Index pages can free a Full extent the retry refills first, and is
  // refused again for ever.
  template <class Attempt> TokenAdmission admit(Attempt attempt) {
    TokenAdmission admission = attempt();
    std::vector<Need> freedFor;
    while (admission.failure == TokenAdmissionFailure::Denied) {
      const Need need{.pages = admission.additionalPages, .form = admission.form};
      if (std::ranges::find(freedFor, need) != freedFor.end()) {
        ++refusedAfterMet;
        break;
      }
      freedFor.push_back(need);
      if (test::takeFirst(cache_, test::kInServiceGrowth, need).outcome != FreeOutcome::Met)
        break;
      admission = attempt();
    }
    return admission;
  }

  // A new request on a prefix of a base prompt, restored from what the
  // cache holds of it, a Spill request committing quota for its pages
  // whether the tier takes writes or not, as a resume does.
  void start() {
    if (requests_.size() >= 6) return;
    const uint64_t id = ++nextId_;
    Request request{.base = pick(0, static_cast<uint32_t>(bases_.size()) - 1),
                    .pages = pick(1, maximumPages_ - 2),
                    .mode = !windowed_ && chance(60) ? KvMode::Spill : KvMode::Resident};
    CacheLookup lookup = cache_.lookup(prompt(request), {});
    cache_.beginRequest(id, request.mode);
    const Cache::ChainCopies copies = lookup.state ? cache_.chainCopies(lookup.state->kvBlock()) : Cache::ChainCopies{};
    if (request.mode == KvMode::Spill && !cache_.commitSpill(id, maximumPages_ - copies.copied, copies)) {
      cache_.endRequest(id);
      return;
    }
    if (!admit([&] { return cache_.restoreRequest(id, lookup); }).granted()) {
      cache_.endRequest(id);
      return;
    }
    request.pages = lookup.resumeBoundary() / KvCache::pageTokens;
    // As the engine does once a state's read lands (Engine::pollRestores), the
    // state is promoted to RAM, its copy a claim that may reclaim
    // (Engine::promotedCopy), before the lane goes on.
    if (promotes_ && lookup.state && !lookup.state->state()->residentBytes() && chance(50)) {
      tier_.complete();
      static_cast<void>(cache_.pollTransfers());
      if (cache_.kvRestoreStatus(id) == KvRestoreStatus::None) {
        const uint64_t before = cache_.snapshot().stateCache.promotions;
        cache_.promoteState(lookup, [&] {
          reclaim();
          return std::make_shared<ShelvedState>(shelf_, false);
        }, id);
        promotions += cache_.snapshot().stateCache.promotions - before;
      }
    }
    if (usedStates_ && lookup.state) {
      request.state = lookup.state->kvBlock();
      request.use = cache_.useState(request.state);
    }
    requests_.emplace(id, std::move(request));
  }

  // A request's command commits more pages, which it publishes; now and then
  // a state lands at its newest block.
  void grow() {
    Request *request = anyRequest();
    if (!request || request->pages >= maximumPages_) return;
    const uint64_t id = idOf(*request);
    if (waiting(id)) return;
    const uint32_t pages = std::min(maximumPages_, request->pages + pick(1, 3));
    const TokenAdmission grown = admit([&] { return cache_.ensureTokens(id, pages * KvCache::pageTokens + 1); });
    refusedGrowths += !grown.granted();
    if (!grown.granted()) return;
    request->pages = pages;
    cache_.publishCommittedBlocks(id, prompt(*request), pages * KvCache::pageTokens, {});
    // As the engine does, a state goes only where its tail holds its window
    // units, and one at a block another request published one at is reused.
    if (const uint64_t block = cache_.blockAt(id, pages * KvCache::pageTokens);
        chance(tight_ ? 60 : 30) && cache_.stateTailPresent(block, id) && !cache_.reuseStoredState(block)) {
      const bool checkpoint = peeks_ && chance(30);
      cache_.publishCompositeState(block, std::make_shared<ShelvedState>(shelf_, false), checkpoint, id);
      // Half the checkpoints are replies' states, which a reply's decode
      // captures, half of those kept, as its request's end keeps one.
      if (checkpoint && chance(50)) {
        cache_.markReply(cache_.checkpointState(block));
        if (chance(50))
          cache_.keepReply(cache_.checkpointState(block));
      }
    }
  }

  // The tier's transfers land, a few of them failing.
  void land() {
    tier_.complete(!chance(5));
    static_cast<void>(cache_.pollTransfers());
    for (auto entry = requests_.begin(); entry != requests_.end();) {
      if (cache_.kvRestoreStatus(entry->first) != KvRestoreStatus::Failed) {
        ++entry;
        continue;
      }
      sources_.erase(entry->first);
      cache_.endRequest(entry->first);
      entry = requests_.erase(entry);
    }
  }

  // Write-behind keeps a restore point at a request's newest block.
  void persist() {
    Request *request = anyRequest();
    if (!cache_.persistent() || !request || !request->pages) return;
    static_cast<void>(cache_.persist(cache_.blockAt(idOf(*request), request->pages * KvCache::pageTokens)));
  }

  // The writes of one class in flight fail, whichever requests hold their
  // blocks.
  void failWrites() {
    tier_.fail(std::array{WriteClass::Reclaim, WriteClass::Persist, WriteClass::Spill}[pick(0, 2)]);
    static_cast<void>(cache_.pollTransfers());
  }

  void reclaim() {
    // A step of in-service work's take: an eviction, a memory reclaim or a
    // page scan's eviction.
    const Need need = std::array{Need{.pages = 1}, Need{.bytes = 1}, Need{.hostRefused = true, .pages = 1}}[pick(0, 2)];
    const ClaimSpec claim{.cls = ClaimClass::InService,
                          .reach = chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary};
    static_cast<void>(test::takeStep(cache_, claim, need));
  }

  // The volume's room changes, and the quota follows it, which the state
  // file shares: its writes then make room or are refused.
  void keepQuota() {
    budget_->setLimitBytes(pick(0, kSlots) * 100);
    budget_->setActiveLimitBytes(pick(0, kSlots) * 100);
    cache_.keepDiskLimit();
    const KvSpillSnapshot spilled = cache_.spillSnapshot();
    require(!spilled.committedBytes || spilled.heldBytes + spilled.committedBytes <= spilled.activeLimitBytes,
            "active KV stayed past its limit");
  }

  void convert() {
    Request *request = anyRequest();
    if (windowed_ || !request || request->mode == KvMode::Spill) return;
    if (cache_.convertToSpill(idOf(*request), maximumPages_)) request->mode = KvMode::Spill;
  }

  // Guard: a claim releases a Spill request's oldest pages whose copies
  // landed, for bytes or in place (Cache::releasePages), by a random rule: the
  // requests it takes pages from, the request among them, and in half the
  // steps the chain a Resident start on a base prompt's prefix restores, which
  // it keeps. It moves no more than were clean, for bytes only where free
  // Index and Full pages can empty an extent, and every page it moves is one
  // each holder may read through its index and the rule lets the claim take:
  // past every holder's dense history and short of its window, no Resident
  // request's, none of a request the rule leaves out and none on the kept
  // chain; and it gives Index pages only to the holders of the oldest page it
  // counted. The KV cache's invariants check each had its copy.
  void release() {
    Request *request = anyRequest();
    if (!spill_ || !request || request->mode != KvMode::Spill) return;
    const uint64_t id = idOf(*request);
    if (waiting(id)) return;
    std::set<uint64_t> released{id};
    for (const auto &[each, _] : requests_)
      if (chance(50)) released.insert(each);
    std::optional<CacheProbe> start;
    if (chance(50)) {
      const std::span<const uint32_t> base = bases_[pick(0, static_cast<uint32_t>(bases_.size()) - 1)];
      start = cache_.probe(base.first(pick(1, maximumPages_ - 2) * KvCache::pageTokens + 1), {});
    }
    const std::span<const uint64_t> kept = start ? start->restoredChain() : std::span<const uint64_t>{};
    const ReleaseRule rule{.takesFrom = [&](uint64_t holder) { return released.contains(holder); }, .kept = kept};
    const auto indexPages = [&] {
      std::set<std::pair<uint64_t, uint32_t>> pages;
      for (const auto &[each, _] : requests_) {
        const std::span<const uint32_t> table = cache_.pageTable(each).pages;
        for (uint32_t index = 0; index < table.size(); ++index)
          if (pool_.form(table[index]) == kv::PageForm::Index) pages.insert({each, index});
      }
      return pages;
    };
    const std::set<std::pair<uint64_t, uint32_t>> before = indexPages();
    const engine::Cache::ReleasablePages releasable = cache_.releasablePages(id, rule);
    const uint32_t freeBefore =
        pool_.freePageCount(kv::PageForm::Index) + pool_.freePageCount(kv::PageForm::Full);
    const bool inPlace = chance(50);
    const uint32_t moved = cache_.releasePages(id, pick(1, 4), inPlace, rule);
    require(moved <= releasable.clean, "a release moved " + std::to_string(moved) + " pages of " +
                                           std::to_string(releasable.clean) + " whose copies landed");
    require(inPlace || !moved || freeBefore >= pool_.extentPages(),
            "a release for bytes moved pages where it could empty no Full extent");
    for (const auto &[holder, index] : indexPages()) {
      if (before.contains({holder, index})) continue;
      const Request &held = requests_.at(holder);
      require(held.mode == KvMode::Spill, "a release moved a page a Resident request holds");
      require(index >= spill_->densePages && index + spill_->windowPages < held.pages,
              "a release moved a page of a holder's dense history or window");
      require(released.contains(holder), "a release moved a page a request it does not take pages from holds");
      require(std::ranges::count(kept, cache_.blockAt(holder, (index + 1) * KvCache::pageTokens)) == 0,
              "a release moved a page of the chain a start keeps");
      require(std::ranges::count(releasable.holders, holder) != 0,
              "a release gave Index pages to a request that does not hold its oldest page");
    }
  }

  // Guard: a count of releases of Spill requests in turn, each of the whole
  // extents of its pages whose copies landed, as a claim's units for bytes
  // count them (ReleaseCount), never counts more than their takes in
  // turn free: each request's count stays within the clean pages it holds
  // when its take comes, a page several hold counting for the first, and a
  // take gives a request its first Index pages only where a count named it.
  // Counted for each request alone, a prefix two hold counts twice.
  void countReleases() {
    std::vector<uint64_t> order;
    for (const auto &[id, request] : requests_)
      if (request.mode == KvMode::Spill && !waiting(id) && chance(70)) order.push_back(id);
    std::shuffle(order.begin(), order.end(), random_);
    const ReleaseRule rule{.takesFrom = [&](uint64_t holder) { return std::ranges::count(order, holder) != 0; }};
    const uint32_t extent = pool_.extentPages();
    engine::ReleaseCount count;
    std::vector<engine::Cache::ReleasablePages> counted;
    for (const uint64_t id : order) {
      counted.push_back(cache_.releasablePages(id, rule, count));
      const uint32_t whole = counted.back().clean / extent;
      cache_.countRelease(id, {.pages = whole * extent, .extents = whole}, false, rule, count);
    }
    std::set<uint64_t> named;
    for (size_t each = 0; each < order.size(); ++each) {
      const uint32_t pages = counted[each].clean / extent * extent;
      const uint32_t clean = cache_.releasablePages(order[each], rule).clean;
      require(clean >= pages, "a count of releases counted " + std::to_string(pages) + " pages of request " +
                                  std::to_string(order[each]) + ", which held " + std::to_string(clean) +
                                  " clean when its take came");
      if (pages) named.insert(counted[each].holders.begin(), counted[each].holders.end());
      std::set<uint64_t> reading;
      for (const auto &[id, _] : requests_)
        if (cache_.holdsIndexPages(id)) reading.insert(id);
      static_cast<void>(cache_.releasePages(order[each], pages, false, rule));
      for (const auto &[id, _] : requests_)
        require(reading.contains(id) || !cache_.holdsIndexPages(id) || named.contains(id),
                "a take gave request " + std::to_string(id) + " its first Index pages, whose reads no count named");
    }
  }

  // A release of a Spill request's pages whose copies landed, in whole
  // extents and the one the free Full pages beside complete
  // (Cache::releaseBeside), as a claim's unit for bytes takes it
  // (EngineVictims::releases), frees the bytes that unit counts of the
  // extents it empties (Cache::releaseMoves), each page of them its Full
  // bytes less its Index page's, wherever the Index runway held an extent
  // before it, transfers in flight or not: its moves empty that many Full
  // extents, none a page in transfer pins, which leave, and refill the runway
  // an extent at a time. Short of the runway, its refill takes an Index
  // extent more; a refused refill moves fewer pages: those are counted apart.
  void releaseExtents() {
    Request *request = anyRequest();
    if (!spill_ || !request || request->mode != KvMode::Spill) return;
    const uint64_t id = idOf(*request);
    if (waiting(id)) return;
    const uint32_t extent = pool_.extentPages();
    const uint32_t beside = cache_.releaseBeside({});
    const engine::Cache::ReleaseMoves moves =
        cache_.releaseMoves(cache_.releasablePages(id, {}).clean, false, {.beside = beside});
    if (!moves.pages) return;
    const bool runway = pool_.freePageCount(kv::PageForm::Index) >= extent;
    const bool inFlight = tier_.inFlight() != 0;
    const uint64_t before = pool_.allocatedBytes();
    const uint32_t moved = cache_.releasePages(id, moves.pages, false, {}, {}, beside);
    const uint64_t freed = before - std::min(before, pool_.allocatedBytes());
    const uint64_t counted = uint64_t{moves.extents} * extent *
                             (storage_.bytesPerPage(kv::PageForm::Full) - storage_.bytesPerPage(kv::PageForm::Index));
    ++extentReleases;
    besideReleases += moves.pages < moves.extents * extent;
    inFlightReleases += inFlight;
    if (freed >= counted) return;
    ++(moved < moves.pages ? refusedReleases : shortRunwayReleases);
    require(moved < moves.pages || !runway,
            "a release of " + std::to_string(moves.extents) + " extents of pages, moving " + std::to_string(moved) +
                " beside " + std::to_string(beside) + " free Full pages" +
                (inFlight ? " and transfers in flight" : "") + " with the runway whole, freed " +
                std::to_string(freed) + " B of the " + std::to_string(counted) + " its count gives them");
  }

  // A read of a spilled block fails. The engine applies the results of the
  // command that read it while the command, and what it read, lives on: the
  // lane ends and the block's copy is given up.
  void failRead() {
    Request *request = anyRequest();
    if (!request) return;
    const uint64_t id = idOf(*request);
    if (waiting(id)) return;
    const PageTableView cut = cache_.cutPageTable(id);
    const std::shared_ptr<const model::KvPageSources> command = cut.sources;
    std::vector<uint64_t> blocks;
    for (uint32_t page = 0; command && page < cut.pages.size(); ++page)
      if ((*command)[page].indexed()) blocks.push_back((*command)[page].block);
    if (blocks.empty()) return;
    sources_.erase(id);
    cache_.endRequest(id);
    requests_.erase(id);
    cache_.poisonCopy(blocks[pick(0, static_cast<uint32_t>(blocks.size()) - 1)]);
    whileApplying();
  }

  // A request ends as the engine applies its last command, which holds what
  // it read until then.
  void finish() {
    Request *request = anyRequest();
    if (!request) return;
    const uint64_t id = idOf(*request);
    const auto command = sources_.extract(id);
    cache_.endRequest(id);
    requests_.erase(id);
    whileApplying();
  }

  // What else applying a command may do before the command goes: give up for
  // quota or memory what nothing holds any more.
  void whileApplying() {
    if (chance(50)) keepQuota();
    if (chance(50)) reclaim();
  }

  // A command cuts a request's page table and reads its sources until the
  // next one.
  void command() {
    Request *request = anyRequest();
    if (!request) return;
    const uint64_t id = idOf(*request);
    if (!waiting(id)) sources_[id] = cache_.cutPageTable(id).sources;
  }

  std::mt19937 random_;
  bool windowed_;
  bool peeks_;
  bool usedStates_;
  bool promotes_;
  bool tight_;
  // A windowed run's window unit, beside each block's base page.
  uint32_t unitPages_;
  // Index pages of 10 bytes for a run that spills.
  test::TestKvStorage storage_{1024, 100, 4, 10};
  KvPool pool_{storage_, 4};
  test::TestKvTier tier_;
  std::shared_ptr<model::DiskBudget> budget_;
  std::optional<CacheSpillPolicy> spill_;
  engine::Cache cache_;
  std::shared_ptr<Shelf> shelf_ = std::make_shared<Shelf>(Shelf{.budget = budget_});
  uint32_t maximumPages_;
  std::array<std::vector<uint32_t>, 3> bases_;
  std::map<uint64_t, Request> requests_;
  // What the commands in flight read.
  std::map<uint64_t, std::shared_ptr<const model::KvPageSources>> sources_;
  uint64_t nextId_ = 0;
};

// The runs' kinds: plain, windowed, windowed on a tight budget, and the same
// with conversations that part from one another.
struct Kind final {
  bool windowed = false, tight = false, forked = false;
  [[nodiscard]] std::string name() const {
    return forked ? "forked tight windowed " : tight ? "tight windowed " : windowed ? "windowed " : "";
  }
};
constexpr std::array kKinds{Kind{}, Kind{.windowed = true}, Kind{.windowed = true, .tight = true},
                            Kind{.windowed = true, .tight = true, .forked = true}};

// The seeds a test runs: of every kind 1..plain or 1..windowed and its pinned seeds, or of one kind (`only`)
// first..last alone, with requests using their states or not (`usedStates`, the take test's).
struct Seeds final {
  uint32_t plain = 8000, windowed = 400;
  std::optional<Kind> only;
  uint32_t first = 1, last = 0;
  std::optional<bool> usedStates;
  [[nodiscard]] bool runs(Kind kind) const {
    return !only || (only->windowed == kind.windowed && only->tight == kind.tight && only->forked == kind.forked);
  }
  [[nodiscard]] uint32_t from() const { return only ? first : 1; }
  [[nodiscard]] uint32_t to(Kind kind) const { return only ? last : kind.windowed ? windowed : plain; }
};

// Seeds 1..plainSeeds plain and 1..windowedSeeds windowed, tight and forked, plus windowed seed 9454 (a
// poisoned block's write lands after its subtree left) and tight windowed seed 173965 (a part acquired into a
// block's copy for a write that does not start goes back).
void testRandomRuns(const Seeds &seeds) {
  uint64_t releases = 0, beside = 0, inFlight = 0, shortRunway = 0, refused = 0;
  uint64_t refusedGrowths = 0, refusedAfterMet = 0, promotions = 0;
  // The pinned seeds run as they were found, without promotions, sparse
  // history at the cache's spacing.
  const auto check = [&](uint32_t seed, Kind kind, bool promotes = true) {
    Run run(seed, kind.windowed, false, false, kind.tight, kind.forked, promotes, promotes);
    try {
      for (int step = 0; step < 600; ++step)
        run.step();
      run.endRequests();
      run.drain();
      run.settle();
    } catch (const std::exception &error) {
      throw std::runtime_error(kind.name() + "seed " + std::to_string(seed) + ": " + error.what());
    }
    refusedGrowths += kind.tight ? run.refusedGrowths : 0;
    refusedAfterMet += kind.tight ? run.refusedAfterMet : 0;
    promotions += run.promotions;
    releases += run.extentReleases;
    beside += run.besideReleases;
    inFlight += run.inFlightReleases;
    shortRunway += run.shortRunwayReleases;
    refused += run.refusedReleases;
  };
  for (const Kind kind : kKinds)
    for (uint32_t seed = seeds.from(); seeds.runs(kind) && seed <= seeds.to(kind); ++seed)
      check(seed, kind);
  if (seeds.only) {
    std::cout << "cache property runs ok: " << seeds.only->name() << "seeds " << seeds.first << " to " << seeds.last
              << ", " << promotions << " promotions whose copies reclaimed\n";
    return;
  }
  check(9454, {.windowed = true}, false);
  check(173965, {.windowed = true, .tight = true}, false);
  require(refusedGrowths > 0, "no tight run's budget refused a growth");
  std::cout << "cache property runs ok: " << seeds.plain << " plain seeds, " << seeds.windowed
            << " windowed, as many tight and forked, sparse history every 4 to 8 pages, windowed seed 9454 and tight "
               "seed 173965; the tight ones' budgets "
               "refused " << refusedGrowths
            << " growths in the end, " << refusedAfterMet << " admissions refused again for a need met; "
            << releases << " releases of whole extents, " << beside
            << " of them completed by free pages beside, " << inFlight << " beside transfers in flight; of all, "
            << shortRunway << " freed less than counted with the runway short of an extent and " << refused
            << " with its refill refused; " << promotions
            << " promotions whose copies reclaimed; the quota counted what the copies held at every run's end\n";
}

// The same runs with random claims counted and taken (checkTake), again with requests using the states they resume
// from, plus windowed seed 3965 (a page need counts the leaf under a superseded state in RAM, which only the state's
// going lets the walk take: it goes because its tail holds the leaf), plain seed 380 with used states (a page need
// counts the leaf under a history state that holds no page) and plain seed 16561 (a leaf's state leaves as room is
// made for its copy, which finds no slot) and plain seed 11843 with used states (a release beside a landing the claim
// was never credited gives its bytes now).
void testTakeNeverExceedsPeek(const Seeds &seeds) {
  uint64_t steps = 0, exact = 0, shortOf = 0, exactNoWrite = 0, shortNoWrite = 0, quotaTakes = 0, bounded = 0,
           inPlace = 0;
  bool inPlaceNeeds = true;
  // The pinned seeds run as they were found, sparse history at the cache's
  // spacing.
  bool sparse = true;
  const auto check = [&](uint32_t seed, Kind kind, bool usedStates) {
    Run run(seed, kind.windowed, true, usedStates, kind.tight, kind.forked, false, sparse);
    run.inPlaceNeeds_ = inPlaceNeeds;
    try {
      for (int step = 0; step < 600; ++step)
        run.step();
      run.endRequests();
      run.drain();
    } catch (const std::exception &error) {
      throw std::runtime_error(std::string(usedStates ? "used states " : "") + "take " + kind.name() + "seed " +
                               std::to_string(seed) + ": " + error.what());
    }
    steps += run.steps;
    exact += run.exactUnits;
    shortOf += run.shortUnits;
    exactNoWrite += run.exactNoWrite;
    shortNoWrite += run.shortNoWrite;
    quotaTakes += run.quotaTakes;
    bounded += run.boundedCounts;
    inPlace += run.inPlaceClaims;
  };
  for (const bool usedStates : {false, true})
    for (const Kind kind : kKinds)
      for (uint32_t seed = seeds.from(); seeds.usedStates.value_or(usedStates) == usedStates && seeds.runs(kind) &&
                                         seed <= seeds.to(kind);
           ++seed)
        check(seed, kind, usedStates);
  if (seeds.only) {
    std::cout << "cache take runs ok: " << (*seeds.usedStates ? "used states " : "") << seeds.only->name()
              << "seeds " << seeds.first << " to " << seeds.last << ", " << bounded
              << " counts bounded short of the full count, " << inPlace << " device claims in place\n";
    return;
  }
  inPlaceNeeds = false;
  sparse = false;
  check(3965, {.windowed = true}, false);
  check(380, {}, true);
  check(16561, {}, false);
  check(11843, {}, true);
  // Plain seed 138904 with used states: a take whose leaf waited for a transfer in flight says Pending.
  check(138904, {}, true);
  // Windowed layouts, each a rule of the count of points on the SSD: forked seed 32532 (stage 3's tail releases,
  // count and take apart), windowed seed 5171 with used states (a unit a write in flight frees as a state's release
  // lands), 21947 with used states (a unit's write into its block's copy, which then shows the window part), forked
  // 35860 with used states (tail releases counted one at a time, history and modes judged on those before).
  check(32532, {.windowed = true, .tight = true, .forked = true}, false);
  check(5171, {.windowed = true}, true);
  check(21947, {.windowed = true}, true);
  check(35860, {.windowed = true, .tight = true, .forked = true}, true);
  // Forked seeds: 301249, a state a count released earlier supersedes nothing, nor holds a state below a block;
  // 39667 with used states, a point whose RAM copy a count released is from then on a point only on the SSD;
  // 412049, a release with no write leaves its units to the writes of their window parts.
  check(301249, {.windowed = true, .tight = true, .forked = true}, false);
  check(39667, {.windowed = true, .tight = true, .forked = true}, true);
  check(412049, {.windowed = true, .tight = true, .forked = true}, false);
  // The take scan of e34562a0: tight 60576 with used states (a disk subtree kept for a state write elsewhere),
  // forked 66773 with used states (a demotion's landing counted before its child's), forked 143938 and forked 21777
  // with used states (a point the quota's give-up left history, released in the same take).
  check(60576, {.windowed = true, .tight = true}, true);
  check(66773, {.windowed = true, .tight = true, .forked = true}, true);
  check(143938, {.windowed = true, .tight = true, .forked = true}, false);
  check(21777, {.windowed = true, .tight = true, .forked = true}, true);
  // A persistent quota's demotion gives up a copy under a landed point's state in a take, and the point above it,
  // superseded only while that chain had landed, is history and released in the same take (tight 6613, used states).
  check(6613, {.windowed = true, .tight = true}, true);
  // With device needs in place drawn (the scan's stream): forked 12632 and 22423, a take's step took pins off a
  // tail whose release its next step freed, past what a count that ran to its end had counted.
  inPlaceNeeds = true;
  check(12632, {.windowed = true, .tight = true, .forked = true}, false);
  check(22423, {.windowed = true, .tight = true, .forked = true}, false);
  // Plain 644947: a device need in place whose second step's landing completed an extent of another form, bytes no
  // count bounded past the first step.
  check(644947, {}, false);
  // Plain 1226654: a release that cancelled the landing an earlier take of the claim credited, which the take nets
  // (Netted) though the landing count stayed.
  check(1226654, {}, false);
  // Plain 101189: a take's first step released the extent free pages it had just freed completed alone, the
  // ledger's count of what lands dropping, and its second freed a page that let the landing complete it again: the
  // take nets only what it cancelled over all its steps (Cache::take).
  check(101189, {}, false);
  // The release mode's scan seeds, run as the scan ran them, sparse history every 4 + seed % 5 pages: forked 24004
  // (8 pages), a point sparse history kept and queued stopped being kept as a write the take's walk started on its
  // chain made the point below it no history (f2250deb); windowed 144893 (7 pages), a release with no write after
  // the quota gave up a copy under a state in the same take (quotaGaveUpState_); tight 105583 (7 pages), and with
  // used states, a point the count demoted judged as on the SSD alone (copyStays). Tight 15667 at the cache's
  // spacing, where the scan found it: a point whose release wrote its copy counted as one to space from, and once
  // on the SSD alone one the tails let go (copyStays).
  sparse = true;
  check(24004, {.windowed = true, .tight = true, .forked = true}, false);
  check(144893, {.windowed = true}, false);
  check(105583, {.windowed = true, .tight = true}, false);
  check(105583, {.windowed = true, .tight = true}, true);
  sparse = false;
  check(15667, {.windowed = true, .tight = true}, false);
  inPlaceNeeds = false;
  std::cout << "cache take runs ok: " << seeds.plain << " plain seeds, " << seeds.windowed
            << " windowed, tight and forked, sparse history every 4 to 8 pages, the pinned seeds; "
            << steps
            << " reclaim steps taken as the ledger takes, each that gave something changing the cache and freeing no "
            << "more than the units through its own counted, every offered unit's first take giving something or "
            << "waiting; " << exact << " units freed all peek counted, " << shortOf << " less (" << exactNoWrite
            << " and " << shortNoWrite << " of those needing no write, each Pending or Netted); " << quotaTakes
            << " takes that wrote freed window units through the quota's copies given up (Quota); " << bounded
            << " counts stopped short of the full count once they covered the need; " << inPlace
            << " device claims in place\n";
}

} // namespace

// cache-property [plainSeeds windowedSeeds]: 8000 and 400 by default (a plain run costs a fifth of a windowed one), every
// kind and test with the pinned seeds.
// cache-property TEST KIND FIRST [LAST]: one test (runs, take, or take-used: the take test with requests using
// their states) of one kind (plain, windowed, tight, forked), seeds FIRST to LAST (FIRST alone by default), so that
// a scan splits across cores and a failing seed runs again alone.
int main(int argc, char **argv) {
  const std::array<std::pair<std::string_view, Kind>, 4> kinds{
      std::pair{std::string_view{"plain"}, kKinds[0]}, std::pair{std::string_view{"windowed"}, kKinds[1]},
      std::pair{std::string_view{"tight"}, kKinds[2]}, std::pair{std::string_view{"forked"}, kKinds[3]}};
  const std::array<std::string_view, 3> tests{"runs", "take", "take-used"};
  const auto kind = argc >= 4 ? std::ranges::find(kinds, std::string_view{argv[2]}, &decltype(kinds)::value_type::first)
                              : kinds.end();
  const auto test = argc >= 4 ? std::ranges::find(tests, std::string_view{argv[1]}) : tests.end();
  if ((argc != 1 && argc != 3 && argc != 4 && argc != 5) || (argc >= 4 && (kind == kinds.end() || test == tests.end()))) {
    std::cerr << "usage: cache-property [plainSeeds windowedSeeds] | cache-property runs|take|take-used "
                 "plain|windowed|tight|forked FIRST [LAST]\n";
    return 2;
  }
  try {
    Seeds seeds;
    if (argc == 3) {
      seeds.plain = static_cast<uint32_t>(std::stoul(argv[1]));
      seeds.windowed = static_cast<uint32_t>(std::stoul(argv[2]));
    } else if (argc >= 4) {
      seeds.only = kind->second;
      seeds.first = static_cast<uint32_t>(std::stoul(argv[3]));
      seeds.last = argc == 5 ? static_cast<uint32_t>(std::stoul(argv[4])) : seeds.first;
      seeds.usedStates = *test == "take-used";
      if (*test == "runs")
        testRandomRuns(seeds);
      else
        testTakeNeverExceedsPeek(seeds);
      return 0;
    }
    testRandomRuns(seeds);
    testTakeNeverExceedsPeek(seeds);
    std::cout << "cache property tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache property tests failed: " << error.what() << '\n';
    return 1;
  }
}
