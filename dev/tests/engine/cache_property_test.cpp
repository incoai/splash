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
  // With peeks, the walk's units for a device or a page need are weighed
  // and taken too (checkPeek), and in half the runs the tier says how many
  // demotions it admits (test::RoomTier), which the walk then counts on.
  // With usedStates, a request uses the state it resumes from, as the
  // engine's requests do (Cache::useState).
  Run(uint32_t seed, bool windowed, bool peeks = false, bool usedStates = false)
      : random_(seed), windowed_(windowed), peeks_(peeks), usedStates_(usedStates), tier_(chance(50)),
        saysRoom_(peeks && chance(50)),
        budget_(std::make_shared<model::DiskBudget>(kSlots * 100)),
        cache_(pool_, saysRoom_ ? static_cast<engine::KvTier *>(&room_) : &tier_, budget_,
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
    case 14: checkPeek(); break;
    }
    cache_.checkInvariants();
  }

  uint64_t exactUnits = 0, unsureUnits = 0, landingUnits = 0;

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

  // The walk never overstates: taking a random claim's first units frees at least what peek
  // weighed, and never takes in-use or certain-reuse units its class may not take.
  void checkPeek() {
    const ClaimSpec claim{.cls = std::array{ClaimClass::InService, ClaimClass::Start, ClaimClass::Optional,
                                            ClaimClass::Floor, ClaimClass::Growth, ClaimClass::Host}[pick(0, 5)],
                          .reach = chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary,
                          .timing = chance(50) ? NeedTiming::Immediate : NeedTiming::CanWait};
    const kv::PageForm form = !chance(50) ? kv::PageForm::Full : windowed_ ? kv::PageForm::Slim : kv::PageForm::Index;
    const uint32_t stateBuffers = chance(30) ? 1u : 0u;
    const bool device = chance(50);
    const VictimQuery query{.claim = &claim,
                            .need = device ? Need{.bytes = pick(1, 8) * 400ull, .stateBuffers = stateBuffers}
                                           : Need{.hostRefused = chance(50), .pages = pick(1, 8), .form = form,
                                                  .stateBuffers = stateBuffers},
                            .critical = chance(10)};
    // A need the host refused runs as an Immediate claim of ordinary reach.
    const bool immediate = claim.timing == NeedTiming::Immediate || query.need.hostRefused;
    const bool reachesInUse =
        (claim.cls == ClaimClass::InService && !immediate && claim.reach == ReclaimClass::InUse) ||
        (claim.cls == ClaimClass::Host && query.critical);
    std::vector<uint64_t> inUse;
    for (const auto &[_, request] : requests_)
      if (request.state && cache_.stateResident(request.state))
        inUse.push_back(request.state);
    // With states in use, a request waits to start on a prefix of a base
    // prompt: the state its chain restores is of certain reuse, which only
    // work in service and the host under critical pressure take.
    const bool takesCertain = claim.cls == ClaimClass::InService || (claim.cls == ClaimClass::Host && query.critical);
    std::vector<uint64_t> wanted;
    if (usedStates_) {
      const std::span<const uint32_t> base = bases_[pick(0, static_cast<uint32_t>(bases_.size()) - 1)];
      const CacheProbe probe = cache_.probe(base.first(pick(1, kMaximumPages - 2) * KvCache::pageTokens + 1), {});
      wanted.assign(probe.restoredChain().begin(), probe.restoredChain().end());
      cache_.setWantedChains({wanted});
    }
    const bool wantedResident = !wanted.empty() && cache_.stateResident(wanted.back());
    std::vector<engine::Victim> victims;
    uint64_t predicted = 0;
    uint32_t predictedPages = 0;
    for (std::optional<engine::Victim> victim = cache_.peek(query, nullptr); victim && victims.size() < 4;
         victim = cache_.peek(query, &*victim)) {
      if (victim->later && immediate)
        continue;
      predicted += victim->bytes;
      predictedPages += victim->form == form ? victim->pages : 0;
      victims.push_back(*victim);
    }
    const auto pagesOf = [&] {
      return pool_.freePageCount(form) + cache_.demotedPages()[static_cast<size_t>(form)];
    };
    const size_t writes = tier_.writes.size();
    const uint64_t offloads = cache_.snapshot().stateCache.offloads;
    bool sure = true;
    uint64_t freed = 0;
    uint32_t freedPages = 0;
    for (const engine::Victim &victim : victims) {
      sure = sure && !victim.later;
      // A windowed cache's trims may take Slim extents for the Full pages
      // they free.
      const uint64_t before = pool_.allocatedBytes();
      const uint32_t pagesBefore = pagesOf();
      const FreeResult taken = cache_.take(query, victim);
      const uint64_t released = before - std::min(before, pool_.allocatedBytes());
      require(taken.nowBytes <= released, "a take released " + std::to_string(released) + " bytes, reported " +
                                              std::to_string(taken.nowBytes));
      freed += released + taken.laterBytes;
      const uint32_t pagesAfter = pagesOf();
      freedPages += pagesAfter - std::min(pagesAfter, pagesBefore);
      sure = sure && tier_.writes.size() == writes && cache_.snapshot().stateCache.offloads == offloads;
      if (!device)
        continue;
      landingUnits += taken.laterBytes != 0;
      ++(sure ? exactUnits : unsureUnits);
      require(!sure || released == victim.bytes,
              "a unit released " + std::to_string(released) + " bytes, weighed at " + std::to_string(victim.bytes));
    }
    for (const uint64_t block : inUse)
      require(reachesInUse || cache_.stateResident(block),
              "a claim that does not reach what is in use took a state in use");
    require(takesCertain || !wantedResident || cache_.stateResident(wanted.back()),
            "a claim that does not take what is of certain reuse took the state a waiting request restores");
    if (!wanted.empty())
      cache_.setWantedChains({});
    // A unit's later steps free once the transfers before them land, which
    // the claim's retries take: what was started lands, the pages that frees
    // past the landing pages counted when it started count, and the query's
    // units are taken again until they free what was weighed or none goes.
    const auto landingOf = [&] { return cache_.demotedPages()[static_cast<size_t>(form)]; };
    for (uint32_t retry = 0; retry < 64 && (freed < predicted || freedPages < predictedPages); ++retry) {
      const uint64_t before = pool_.allocatedBytes();
      const uint32_t freeBefore = pool_.freePageCount(form), landingBefore = landingOf();
      tier_.complete();
      static_cast<void>(cache_.pollTransfers());
      const uint32_t freeAfter = pool_.freePageCount(form), landingAfter = landingOf();
      const uint32_t landed = landingBefore - std::min(landingBefore, landingAfter);
      freedPages += freeAfter - std::min(freeAfter, freeBefore + landed);
      std::optional<engine::Victim> unit = cache_.peek(query, nullptr);
      while (unit && unit->later && immediate)
        unit = cache_.peek(query, &*unit);
      const uint32_t pagesBefore = pagesOf();
      const FreeResult taken = unit ? cache_.take(query, *unit) : FreeResult{};
      freed += before - std::min(before, pool_.allocatedBytes()) + taken.laterBytes;
      const uint32_t pagesAfter = pagesOf();
      freedPages += pagesAfter - std::min(pagesAfter, pagesBefore);
      if (taken.outcome != FreeOutcome::Met)
        break;
    }
    require(freed >= predicted && freedPages >= predictedPages,
            "the walk weighed its victims to free more than taking them freed: weighed " + std::to_string(predicted) +
                " bytes and " + std::to_string(predictedPages) + " pages, freed " + std::to_string(freed) +
                " bytes and " + std::to_string(freedPages) + " pages");
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
    const CacheReclaimMode mode = std::array{CacheReclaimMode::KeepExtents, CacheReclaimMode::ReleaseExtents,
                                             CacheReclaimMode::ReusePages}[pick(0, 2)];
    static_cast<void>(cache_.reclaimOne(mode, chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary));
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
  bool windowed_;
  bool peeks_;
  bool usedStates_;
  // Index pages of 10 bytes for a run that spills, Slim ones of 30 for a
  // windowed one.
  test::TestKvStorage storage_{1024, 100, 4, 10, 30};
  KvPool pool_{storage_, 4};
  test::TestKvTier tier_;
  test::RoomTier room_{tier_};
  bool saysRoom_;
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

// The same runs with peeks, again with requests using the states they resume from; the fixed
// seeds reach walk paths the default seeds miss.
void testPeekNeverOverstates(uint32_t plainSeeds, uint32_t windowedSeeds) {
  uint64_t exact = 0, unsure = 0, landing = 0;
  const auto check = [&](uint32_t seed, bool windowed, bool usedStates = false) {
    Run run(seed, windowed, true, usedStates);
    try {
      for (int step = 0; step < 600; ++step)
        run.step();
      run.endRequests();
      run.drain();
    } catch (const std::exception &error) {
      throw std::runtime_error(std::string(usedStates ? "used states " : "") +
                               (windowed ? "peek windowed " : "peek ") + "seed " + std::to_string(seed) + ": " +
                               error.what());
    }
    exact += run.exactUnits;
    unsure += run.unsureUnits;
    landing += run.landingUnits;
  };
  for (const bool usedStates : {false, true})
    for (const bool windowed : {false, true})
      for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed)
        check(seed, windowed, usedStates);
  // Units around a block whose demotion is in flight: its last child, restores, a chain ending on it.
  check(2484, true);
  check(9349, true);
  check(25949, true);
  check(28462, true);
  // A weighed page frees at a landing that lets a poisoned subtree go.
  for (const uint32_t seed : {11208u, 99256u, 229505u})
    check(seed, false);
  // A leaf passes its recency to a parent with a disk-only subtree or a demotion landing.
  check(1836083, true);
  check(396221, false);
  // The resume point lies below a poisoned block whose subtree goes as transfers land.
  for (const uint32_t seed : {10248u, 15837u, 24025u, 105144u, 386046u, 470525u, 482345u, 247559u, 364835u})
    check(seed, false);
  // Retries release an extent placed elsewhere; Slim and Index units reach only their own form.
  for (const uint32_t seed : {55544u, 63544u, 546782u, 1061965u, 1748006u, 2429048u})
    check(seed, true);
  for (const uint32_t seed : {415497u, 478646u, 483524u})
    check(seed, false);
  std::cout << "cache peek runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds
            << " windowed and the fixed seeds; " << exact
            << " units released what they were weighed at, "
            << unsure << " after a copy or a write, " << landing
            << " freeing extents as their copies land, never more than taken\n";
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
    testPeekNeverOverstates(plainSeeds, windowedSeeds);
    std::cout << "cache property tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache property tests failed: " << error.what() << '\n';
    return 1;
  }
}
