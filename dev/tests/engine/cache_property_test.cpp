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
#include <random>
#include <span>
#include <stdexcept>
#include <string>
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

// One randomized run of the tiered block store: requests publish, spill, restore, convert and end
// beside reclaims, quota and failing transfers; every step checks Cache::checkInvariants.
class Run final {
public:
  // With peeks, random claims count the walk's units and take them too
  // (checkTake).
  // With usedStates, a request uses the state it resumes from, as the
  // engine's requests do (Cache::useState).
  Run(uint32_t seed, bool windowed, bool peeks = false, bool usedStates = false)
      : random_(seed), enoughRandom_(seed), windowed_(windowed), peeks_(peeks), usedStates_(usedStates),
        tier_(chance(50)),
        budget_(std::make_shared<model::DiskBudget>(kSlots * 100)),
        cache_(pool_, &tier_, budget_,
               windowed ? std::nullopt : std::optional(CacheSpillPolicy{pick(0, 2), pick(1, 2)}),
               windowed ? std::optional(CacheWindowPolicy{pick(1, 3)}) : std::nullopt) {
    tier_.budget = budget_;
    tier_.transferLimit = 64;
    tier_.writeLimits = {8, 4, 8};
    storage_.budgetPages = 96;
    for (uint32_t base = 0; base < bases_.size(); ++base) {
      bases_[base].resize(kMaximumPages * KvCache::pageTokens + 1);
      for (uint32_t row = 0; row < bases_[base].size(); ++row)
        bases_[base][row] = 1000 * (base + 1) + row;
    }
  }

  void step() {
    switch (pick(0, peeks_ ? 14 : 13)) {
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
    case 14: checkTake(); break;
    }
    cache_.checkInvariants();
  }

  uint64_t steps = 0;

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

  // A claim's takes never free more than peek counted for its units, and
  // never take in-use or certain-reuse units its rights do not take. A unit
  // peek offers as needing no write starts none when taken; one that needs
  // a write but frees now (the state the idle staging buffer writes) starts
  // at most that state's write and is one state; a state that frees only
  // once a write lands gives nothing when taken.
  void checkTake() {
    const ClaimSpec claim{.cls = std::array{ClaimClass::InService, ClaimClass::Start, ClaimClass::Optional,
                                            ClaimClass::Growth, ClaimClass::Host}[pick(0, 4)],
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
    checkCountUpTo(query);
    // Each unit, with the one before it in the cache's order.
    std::vector<std::pair<std::optional<engine::Victim>, engine::Victim>> units;
    uint64_t counted = 0;
    uint32_t countedPages = 0;
    uint32_t countedBuffers = 0;
    std::optional<engine::Victim> previous;
    for (std::optional<engine::Victim> unit = cache_.peek(query, nullptr); unit;
         previous = unit, unit = cache_.peek(query, &*previous)) {
      require(!unit->later || unit->write, "a unit that frees once a write lands was not marked as needing one");
      require(!unit->write || unit->later || unit->stateBuffers <= 1,
              "a unit that frees now through a write counted more than the one state the staging buffer takes");
      if (unit->later && immediate)
        continue;
      counted += unit->bytes;
      countedPages += unit->form == form ? unit->pages : 0;
      countedBuffers += unit->stateBuffers;
      units.emplace_back(previous, *unit);
    }
    // The free pages of the form, and for a claim that can wait the landing ones.
    const auto pagesOf = [&] {
      return pool_.freePageCount(form) + (immediate ? 0 : cache_.demotedPages()[static_cast<size_t>(form)]);
    };
    const uint32_t pagesBefore = pagesOf();
    // A claim that can wait counts the extents landing pages complete beside
    // the units, and a take may release one of them before they land.
    const uint64_t landing = immediate ? 0 : cache_.returning().landingBytes;
    // What the takes report, as the ledger sums it, and what left memory now.
    uint64_t freed = 0;
    uint64_t releasedNow = 0;
    uint32_t reportedPages = 0;
    uint32_t reportedBuffers = 0;
    for (const auto &[after, first] : units) {
      // A unit's takes while the cache offers it again in its place, until
      // one takes nothing: a reclaim phase goes a step at a time, each a
      // reclaim of a finite cache.
      for (std::optional<engine::Victim> unit = first; unit && unit->key == first.key;
           unit = cache_.peek(query, after ? &*after : nullptr)) {
        const uint64_t before = pool_.allocatedBytes();
        const uint64_t offloads = cache_.snapshot().stateCache.offloads;
        const uint32_t demotions = tier_.demotions;
        const FreeResult taken = cache_.take(query, *unit);
        const uint64_t released = before - std::min(before, pool_.allocatedBytes());
        require(taken.nowBytes <= released, "a take released " + std::to_string(released) + " bytes, reported " +
                                                std::to_string(taken.nowBytes));
        const uint64_t written = cache_.snapshot().stateCache.offloads - offloads;
        require(unit->write || (!written && tier_.demotions == demotions),
                "a take of a unit offered as needing no write started one");
        require(!unit->write || unit->later || (written <= 1 && tier_.demotions == demotions),
                "a take of a unit offered as freeing now through the staging buffer started more than its state's "
                "write");
        require(!states || !unit->later || !taken.gave(),
                "a take of a state offered as freeing once a write lands gave something now");
        freed += taken.nowBytes + taken.laterBytes;
        releasedNow += released;
        reportedPages += taken.pages;
        reportedBuffers += taken.stateBuffers;
        if (!taken.gave())
          break;
        ++steps;
      }
    }
    const uint32_t pagesAfter = pagesOf();
    const uint32_t freedPages = pagesAfter - std::min(pagesAfter, pagesBefore);
    for (const uint64_t block : inUse)
      require(query.rights.inUse || cache_.stateResident(block),
              "a claim that does not reach what is in use took a state in use");
    require(takesCertain || !wantedResident || cache_.stateResident(wanted.back()),
            "a claim that does not take what is of certain reuse took the state a waiting request restores");
    if (!wanted.empty())
      cache_.setWantedChains({});
    // A device need counts bytes, a page need pages of its form, a state
    // need buffers.
    require(device   ? freed <= counted && releasedNow <= counted + landing
            : states ? reportedBuffers <= countedBuffers
                     : freedPages <= countedPages && reportedPages <= countedPages,
            "the claim's takes freed more than peek counted: counted " + std::to_string(counted) + " bytes and " +
                std::to_string(countedPages) + " pages, freed " + std::to_string(freed) + " bytes (" +
                std::to_string(releasedNow) + " now) and " +
                std::to_string(freedPages) + " pages (" + std::to_string(reportedPages) + " reported), " +
                std::to_string(reportedBuffers) + " of " + std::to_string(countedBuffers) + " state buffers");
  }

  // Counted up to a random enough (VictimQuery::enough), the walk gives the
  // same units as counted whole, none of them more: below enough in every
  // measure the whole unit, else at least enough in a measure enough has
  // and the whole unit reaches.
  void checkCountUpTo(const VictimQuery &whole) {
    const auto pick = [&](uint32_t first, uint32_t last) {
      return std::uniform_int_distribution<uint32_t>(first, last)(enoughRandom_);
    };
    const auto chance = [&](uint32_t percent) { return pick(1, 100) <= percent; };
    VictimQuery query = whole;
    query.enough = {.bytes = chance(50) ? pick(1, 8) * 400ull : 0,
                    .pages = chance(50) ? pick(1, 8) : 0,
                    .form = whole.need.form,
                    .stateBuffers = chance(30) ? 1u : 0u};
    const auto covers = [&](const engine::Victim &unit) {
      return query.enough.reached(unit.bytes, unit.pages, unit.form, unit.stateBuffers);
    };
    std::optional<engine::Victim> all = cache_.peek(whole, nullptr);
    std::optional<engine::Victim> counted = cache_.peek(query, nullptr);
    for (; all || counted; all = cache_.peek(whole, &*all), counted = cache_.peek(query, &*counted)) {
      require(all && counted && all->key == counted->key && all->later == counted->later,
              "a walk counted up to enough gave other units than counted whole");
      const bool same = all->bytes == counted->bytes && all->pages == counted->pages &&
                        all->stateBuffers == counted->stateBuffers;
      require(counted->bytes <= all->bytes && counted->pages <= all->pages &&
                  counted->stateBuffers <= all->stateBuffers && (covers(*all) ? covers(*counted) : same),
              "a unit counted up to enough gave " + std::to_string(counted->bytes) + " bytes, " +
                  std::to_string(counted->pages) + " pages and " + std::to_string(counted->stateBuffers) +
                  " states, counted whole " + std::to_string(all->bytes) + ", " + std::to_string(all->pages) +
                  " and " + std::to_string(all->stateBuffers));
    }
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
    const uint32_t copied = lookup.state ? cache_.copiedBlocks(lookup.state->kvBlock()) : 0;
    if (request.mode == KvMode::Spill && !cache_.commitSpill(id, kMaximumPages - copied)) {
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
  // checkCountUpTo's own, so that a run's claims and takes follow the same
  // paths with the check as without it.
  std::mt19937 enoughRandom_;
  bool windowed_;
  bool peeks_;
  bool usedStates_;
  // Index pages of 10 bytes for a run that spills, Slim ones of 30 for a
  // windowed one.
  test::TestKvStorage storage_{1024, 100, 4, 10, 30};
  KvPool pool_{storage_, 4};
  test::TestKvTier tier_;
  std::shared_ptr<model::DiskBudget> budget_;
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
  const auto check = [](uint32_t seed, bool windowed) {
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
  };
  for (const bool windowed : {false, true})
    for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed)
      check(seed, windowed);
  check(9454, true);
  std::cout << "cache property runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds
            << " windowed and windowed seed 9454\n";
}

// The same runs with random claims counted, whole and up to a random enough (checkCountUpTo), and
// taken (checkTake), again with requests using the states they resume from.
void testTakeNeverExceedsPeek(uint32_t plainSeeds, uint32_t windowedSeeds) {
  uint64_t steps = 0;
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
  };
  for (const bool usedStates : {false, true})
    for (const bool windowed : {false, true})
      for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed)
        check(seed, windowed, usedStates);
  std::cout << "cache take runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds << " windowed; " << steps
            << " reclaim steps, none freeing more than peek counted\n";
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
