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
#include <tuple>
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
  bool canOffload(bool) const noexcept override { return !disk_ && shelf_->takesWrites; }
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
  Run(uint32_t seed, bool windowed, bool peeks = false, bool usedStates = false)
      : random_(seed), windowed_(windowed), peeks_(peeks), usedStates_(usedStates),
        tier_(chance(50)),
        budget_(std::make_shared<model::DiskBudget>(kSlots * 100)),
        spill_(windowed ? std::nullopt : std::optional(CacheSpillPolicy{pick(0, 2), pick(1, 2), chance(50)})),
        cache_(pool_, &tier_, budget_, spill_,
               windowed ? std::optional(CacheWindowPolicy{pick(1, 3)}) : std::nullopt) {
    tier_.budget = budget_;
    tier_.transferLimit = 64;
    tier_.writeLimits = {8, 4, 8};
    storage_.budgetPages = 96;
    // Every count a peek or a take makes of what a reclaim reaches follows
    // its rule over the whole cache, as kv-page-cache's
    // testPageReachFollowsItsRule states it.
    cache_.checkReaches(true);
    for (uint32_t base = 0; base < bases_.size(); ++base) {
      bases_[base].resize(kMaximumPages * KvCache::pageTokens + 1);
      for (uint32_t row = 0; row < bases_[base].size(); ++row)
        bases_[base][row] = 1000 * (base + 1) + row;
    }
  }

  void step() {
    switch (pick(0, peeks_ ? 17 : 16)) {
    case 0: start(); break;
    case 1: case 2: grow(); break;
    case 3: static_cast<void>(cache_.writeThrough()); break;
    case 4: land(); break;
    case 5:
      static_cast<void>(windowed_ ? cache_.trimWindows(chance(80))
                                  : cache_.spillWritten(chance(50) ? SpilledExtents::KeepRunway
                                                                   : SpilledExtents::Released)
                                        .additionalPages);
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
  // Units taken, by whether their takes freed, in the need's measure, all
  // peek counted when it first offered them (exact) or less (short), and of
  // those offered as needing no write the same: a short one of those is one
  // of the classes checkTake names.
  uint64_t exactUnits = 0, shortUnits = 0, exactNoWrite = 0, shortNoWrite = 0;
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
    const kv::PageForm form = !chance(50) ? kv::PageForm::Full : windowed_ ? kv::PageForm::Slim : kv::PageForm::Index;
    const uint32_t stateBuffers = chance(30) ? 1u : 0u;
    // A device need, a page need, or a need of state buffers alone.
    const uint32_t kind = pick(0, 2);
    const bool device = kind == 0;
    const bool states = kind == 2;
    const VictimQuery query =
        test::query(claim,
                    device   ? Need{.bytes = pick(1, 8) * 400ull, .stateBuffers = stateBuffers}
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
      const CacheProbe probe = cache_.probe(base.first(pick(1, kMaximumPages - 2) * KvCache::pageTokens + 1), {});
      wanted.assign(probe.restoredChain().begin(), probe.restoredChain().end());
      cache_.setWantedChains({wanted});
    }
    const bool wantedResident = !wanted.empty() && cache_.stateResident(wanted.back());
    // A unit's measure: bytes for a device need, pages of its form for a
    // page need, buffers for a state need.
    const auto measure = [&](const engine::Victim &unit) -> uint64_t {
      return device ? unit.bytes : states ? unit.stateBuffers : unit.form == form ? unit.pages : 0;
    };
    const auto gaveOf = [&](const FreeResult &taken) -> uint64_t {
      return device ? taken.nowBytes + taken.laterBytes : states ? taken.stateBuffers : taken.pages;
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
    // What a take that gives something changes: the memory the pool holds,
    // its free, landing and cached pages, the states in RAM, the writes
    // started.
    const auto fingerprint = [&] {
      const CacheSnapshot snapshot = cache_.snapshot();
      return std::tuple{pool_.allocatedBytes(),          pool_.freePageCount(kv::PageForm::Full),
                        pool_.freePageCount(kv::PageForm::Slim), pool_.freePageCount(kv::PageForm::Index),
                        cache_.demotedPages(),           snapshot.pool.pagesPrefix,
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
      uint64_t slimRoom = 0;
      bool netted = false;
      std::string takes;
    };
    std::map<uint64_t, Taken> taken;
    // The takes as the ledger takes (MemoryLedger::take): the first unit from
    // the head of the order, again from the head after a take that gave
    // something, the next one after a take that gave nothing; a claim that
    // does not wait for writes passes over what lands later.
    for (std::optional<engine::Victim> unit = cache_.peek(query, nullptr); unit;) {
      if (unit->later && immediate) {
        unit = cache_.peek(query, &*unit);
        continue;
      }
      const uint64_t through = throughUnit(*unit);
      const bool firstTake = !taken.contains(unit->key);
      Taken &record = taken[unit->key];
      if (firstTake)
        record = {.counted = measure(*unit),
                  .write = unit->write,
                  .slimRoom = uint64_t{pool_.allocatedExtents(kv::PageForm::Slim)} * pool_.extentPages()};
      const auto unchanged = fingerprint();
      const uint64_t before = pool_.allocatedBytes();
      const uint64_t offloads = cache_.snapshot().stateCache.offloads;
      const uint32_t demotions = tier_.demotions;
      const uint64_t landingBefore = cache_.returning().landingBytes;
      const uint32_t slimBefore = pool_.freePageCount(kv::PageForm::Slim);
      const FreeResult result = cache_.take(query, *unit);
      const uint64_t released = before - std::min(before, pool_.allocatedBytes());
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
      // fill).
      require(gaveOf(result) <= through, "a take freed " + std::to_string(gaveOf(result)) +
                                             " where the units through it counted " + std::to_string(through));
      record.freed += gaveOf(result);
      record.pending = record.pending || result.outcome == FreeOutcome::Pending;
      const uint32_t slimAfter = pool_.freePageCount(kv::PageForm::Slim);
      record.slimRoom += slimAfter - std::min(slimAfter, slimBefore);
      record.netted = record.netted || (device && released > gaveOf(result) &&
                                        cache_.returning().landingBytes < landingBefore);
      record.takes += " [" + std::to_string(measure(*unit)) + " counted, " + std::to_string(gaveOf(result)) +
                      " freed, " + std::to_string(released) + " released, " + std::to_string(result.stateBuffers) +
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
    // Tightness: a unit offered as needing no write frees all it counted
    // when first offered, but for three classes, each named by what its
    // takes report:
    // - Pending: a take of it ended Pending, a transfer in flight holding
    //   the rest, which comes back as it lands.
    // - Trims: a page need of Full pages in a windowed cache, short by at
    //   most the Slim pages its trims could go into (those allocated when it
    //   was first offered, and each its takes freed). Its count trims the
    //   pages the windows passed and the tails of the states the phase lets
    //   go into the Slim pages free or freed in the phase; its takes trim
    //   only once a state goes, into the Slim pages free then, and only what
    //   no other tail or window still holds, and the Trim unit before it
    //   trims into the Slim pages its takes free: what it trims itself
    //   depends on the order of the phase's takes.
    // - Netted: a device need whose take released an extent the landing
    //   pages would have completed, as the landing count dropped. The take
    //   credits the release less the landing extent it cancels (net
    //   nothing), while the count keeps the release as its least (Cache::
    //   reach), so that a claim takes that memory now rather than waiting for
    //   the landing (testAReleaseOfWhatALandingCompletesCountsOnce).
    for (const auto &[key, record] : taken) {
      const bool exact = record.freed >= record.counted;
      ++(exact ? exactUnits : shortUnits);
      if (!record.write)
        ++(exact ? exactNoWrite : shortNoWrite);
      const bool trims = windowed_ && !device && !states && form == kv::PageForm::Full &&
                         record.counted - std::min(record.counted, record.freed) <= record.slimRoom;
      require(exact || record.write || record.pending || trims || record.netted,
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
  static constexpr uint32_t kMaximumPages = 12;

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

  // Admits as the engine does: a denial makes room in reclaim steps, of
  // the form denied, until the pages fit or nothing more goes.
  template <class Attempt> TokenAdmission admit(Attempt attempt) {
    TokenAdmission admission = attempt();
    while (admission.failure == TokenAdmissionFailure::Denied &&
           test::takeFirst(cache_, test::kInServiceGrowth,
                           {.pages = admission.additionalPages, .form = admission.form})
                   .outcome == FreeOutcome::Met)
      admission = attempt();
    return admission;
  }

  // A new request on a prefix of a base prompt, restored from what the
  // cache holds of it, a Spill request committing quota for its pages
  // whether the tier takes writes or not, as a resume does.
  void start() {
    if (requests_.size() >= 6) return;
    const uint64_t id = ++nextId_;
    Request request{.base = pick(0, static_cast<uint32_t>(bases_.size()) - 1),
                    .pages = pick(1, kMaximumPages - 2),
                    .mode = !windowed_ && chance(60) ? KvMode::Spill : KvMode::Resident};
    CacheLookup lookup = cache_.lookup(prompt(request), {});
    cache_.beginRequest(id, request.mode);
    const Cache::ChainCopies copies = lookup.state ? cache_.chainCopies(lookup.state->kvBlock()) : Cache::ChainCopies{};
    if (request.mode == KvMode::Spill && !cache_.commitSpill(id, kMaximumPages - copies.copied, copies)) {
      cache_.endRequest(id);
      return;
    }
    if (!admit([&] { return cache_.restoreRequest(id, lookup); }).granted()) {
      cache_.endRequest(id);
      return;
    }
    request.pages = lookup.resumeBoundary() / KvCache::pageTokens;
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
    if (!request || request->pages >= kMaximumPages) return;
    const uint64_t id = idOf(*request);
    if (waiting(id)) return;
    const uint32_t pages = std::min(kMaximumPages, request->pages + pick(1, 3));
    if (!admit([&] { return cache_.ensureTokens(id, pages * KvCache::pageTokens + 1); }).granted()) return;
    request->pages = pages;
    cache_.publishCommittedBlocks(id, prompt(*request), pages * KvCache::pageTokens, {});
    // As the engine does, a state at a block another request published one
    // at is reused, and one is published only with a tail it restores from.
    if (const uint64_t block = cache_.blockAt(id, pages * KvCache::pageTokens);
        chance(30) && !cache_.reuseStoredState(block) && cache_.stateTailFull(block))
      cache_.publishCompositeState(block, std::make_shared<ShelvedState>(shelf_, false), peeks_ && chance(30));
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
    if (cache_.convertToSpill(idOf(*request), kMaximumPages)) request->mode = KvMode::Spill;
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
      start = cache_.probe(base.first(pick(1, kMaximumPages - 2) * KvCache::pageTokens + 1), {});
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
  // Index pages of 10 bytes for a run that spills, Slim ones of 30 for a
  // windowed one.
  test::TestKvStorage storage_{1024, 100, 4, 10, 30};
  KvPool pool_{storage_, 4};
  test::TestKvTier tier_;
  std::shared_ptr<model::DiskBudget> budget_;
  std::optional<CacheSpillPolicy> spill_;
  engine::Cache cache_;
  std::shared_ptr<Shelf> shelf_ = std::make_shared<Shelf>(Shelf{.budget = budget_});
  std::array<std::vector<uint32_t>, 3> bases_;
  std::map<uint64_t, Request> requests_;
  // What the commands in flight read.
  std::map<uint64_t, std::shared_ptr<const model::KvPageSources>> sources_;
  uint64_t nextId_ = 0;
};

// Seeds 1..plainSeeds plain and 1..windowedSeeds windowed, plus windowed seed 9454 (a poisoned
// block's write lands after its subtree left).
void testRandomRuns(uint32_t plainSeeds, uint32_t windowedSeeds) {
  uint64_t releases = 0, beside = 0, inFlight = 0, shortRunway = 0, refused = 0;
  const auto check = [&](uint32_t seed, bool windowed) {
    Run run(seed, windowed);
    try {
      for (int step = 0; step < 600; ++step)
        run.step();
      run.endRequests();
      run.drain();
      run.settle();
    } catch (const std::exception &error) {
      throw std::runtime_error(std::string(windowed ? "windowed " : "") + "seed " + std::to_string(seed) + ": " +
                               error.what());
    }
    releases += run.extentReleases;
    beside += run.besideReleases;
    inFlight += run.inFlightReleases;
    shortRunway += run.shortRunwayReleases;
    refused += run.refusedReleases;
  };
  for (const bool windowed : {false, true})
    for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed)
      check(seed, windowed);
  check(9454, true);
  std::cout << "cache property runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds
            << " windowed and windowed seed 9454; " << releases << " releases of whole extents, " << beside
            << " of them completed by free pages beside, " << inFlight << " beside transfers in flight; of all, "
            << shortRunway << " freed less than counted with the runway short of an extent and " << refused
            << " with its refill refused\n";
}

// The same runs with random claims counted and taken (checkTake), again with requests using the states they resume
// from.
void testTakeNeverExceedsPeek(uint32_t plainSeeds, uint32_t windowedSeeds) {
  uint64_t steps = 0, exact = 0, shortOf = 0, exactNoWrite = 0, shortNoWrite = 0;
  const auto check = [&](uint32_t seed, bool windowed, bool usedStates) {
    Run run(seed, windowed, true, usedStates);
    try {
      for (int step = 0; step < 600; ++step)
        run.step();
      run.endRequests();
      run.drain();
    } catch (const std::exception &error) {
      throw std::runtime_error(std::string(usedStates ? "used states " : "") +
                               (windowed ? "take windowed " : "take ") + "seed " + std::to_string(seed) + ": " +
                               error.what());
    }
    steps += run.steps;
    exact += run.exactUnits;
    shortOf += run.shortUnits;
    exactNoWrite += run.exactNoWrite;
    shortNoWrite += run.shortNoWrite;
  };
  for (const bool usedStates : {false, true})
    for (const bool windowed : {false, true})
      for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed)
        check(seed, windowed, usedStates);
  std::cout << "cache take runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds << " windowed; " << steps
            << " reclaim steps taken as the ledger takes, each that gave something changing the cache and freeing no "
            << "more than the units through its own counted, every offered unit's first take giving something or "
            << "waiting; " << exact << " units freed all peek counted, " << shortOf << " less (" << exactNoWrite
            << " and " << shortNoWrite << " of those needing no write, each Pending, Trims or Netted)\n";
}

} // namespace

// cache-property [plainSeeds windowedSeeds]: 40 and 400 by default.
int main(int argc, char **argv) {
  if (argc != 1 && argc != 3) {
    std::cerr << "usage: cache-property [plainSeeds windowedSeeds]\n";
    return 2;
  }
  try {
    const uint32_t plainSeeds = argc == 3 ? static_cast<uint32_t>(std::stoul(argv[1])) : 40u;
    const uint32_t windowedSeeds = argc == 3 ? static_cast<uint32_t>(std::stoul(argv[2])) : 400u;
    testRandomRuns(plainSeeds, windowedSeeds);
    testTakeNeverExceedsPeek(plainSeeds, windowedSeeds);
    std::cout << "cache property tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache property tests failed: " << error.what() << '\n';
    return 1;
  }
}
