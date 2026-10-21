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
// `slots` leave room: the restore points requests resume from.
struct Shelf final {
  uint32_t used = 0;
  uint32_t slots = 64;
};

class ShelvedState final : public CompositeState {
public:
  ShelvedState(std::shared_ptr<Shelf> shelf, bool disk) : shelf_(std::move(shelf)), disk_(disk) {
    if (disk_) ++shelf_->used;
  }
  ~ShelvedState() override {
    if (disk_) --shelf_->used;
  }
  uint64_t bytes() const noexcept override { return 100; }
  uint64_t residentBytes() const noexcept override { return disk_ ? 0 : bytes(); }
  bool canOffload(bool) const noexcept override { return !disk_; }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override { return shelve(); }
  std::unique_ptr<StateOffload> persist(std::function<void()>) const override { return shelve(); }

private:
  std::unique_ptr<StateOffload> shelve() const {
    if (shelf_->used >= shelf_->slots) return {};
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

// One randomized run of the tiered block store. Requests of both modes on
// prompts that share prefixes publish, spill, restore, convert, read
// unreadable copies and end; RAM reclaim and the disk quota take what they
// may, a persistent tier keeps restore points, the pool's growth and the
// tier's writes of every class fail now and then, and the tier's transfers
// land at random. The cache checks every invariant spilling relies on after
// each step (Cache::checkInvariants), and active KV never stays past its
// limit after the quota was kept.
class Run final {
public:
  explicit Run(uint32_t seed)
      : random_(seed), tier_(chance(50)), budget_(std::make_shared<model::DiskBudget>(kSlots * 100)),
        cache_(pool_, &tier_, budget_, CacheSpillPolicy{pick(0, 2), pick(1, 2)}) {
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
    switch (pick(0, 13)) {
    case 0: start(); break;
    case 1: case 2: grow(); break;
    case 3: static_cast<void>(cache_.writeThrough()); break;
    case 4: land(); break;
    case 5: static_cast<void>(cache_.spillWritten()); break;
    case 6: reclaim(); break;
    case 7: keepQuota(); break;
    case 8: convert(); break;
    case 9: failRead(); break;
    case 10: finish(); break;
    case 11: command(); break;
    case 12: persist(); break;
    case 13: failWrites(); break;
    }
    cache_.checkInvariants();
  }

  // Every request ends, every transfer lands, and nothing leaks: the pool's
  // pages and the tier's slots belong to the cache's blocks alone.
  void drain() {
    while (!requests_.empty()) {
      sources_.erase(requests_.begin()->first);
      cache_.endRequest(requests_.begin()->first);
      requests_.erase(requests_.begin());
    }
    tier_.complete();
    static_cast<void>(cache_.pollTransfers());
    cache_.checkInvariants();
    const KvPoolSnapshot pool = pool_.snapshot();
    require(pool.pagesActive == 0 && budget_->committedBytes() == 0 && cache_.spillSnapshot().heldBytes == 0,
            "a finished run left pages held, quota committed or copies held");
  }

private:
  static constexpr uint64_t kSlots = 160;
  static constexpr uint32_t kMaximumPages = 12;

  struct Request final {
    uint32_t base = 0;
    uint32_t pages = 0;
    KvMode mode = KvMode::Resident;
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
           cache_.reclaimForPages(admission.additionalPages, CacheReclaimMode::KeepExtents,
                                  ReclaimClass::InUse, admission.form)
               .madeProgress)
      admission = attempt();
    return admission;
  }

  // A new request on a prefix of a base prompt, restored from what the
  // cache holds of it, a Spill request committing quota for its pages.
  void start() {
    if (requests_.size() >= 6) return;
    const uint64_t id = ++nextId_;
    Request request{pick(0, static_cast<uint32_t>(bases_.size()) - 1), pick(1, kMaximumPages - 2),
                    chance(60) ? KvMode::Spill : KvMode::Resident};
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
    requests_.emplace(id, request);
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
    // at is reused.
    if (const uint64_t block = cache_.blockAt(id, pages * KvCache::pageTokens);
        chance(30) && !cache_.reuseStoredState(block))
      cache_.publishCompositeState(block, std::make_shared<ShelvedState>(shelf_, false), false);
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

  // The volume's room changes, and the quota follows it; so does the state
  // file's, whose writes then make room or are refused.
  void keepQuota() {
    budget_->setLimitBytes(pick(0, kSlots) * 100);
    budget_->setActiveLimitBytes(pick(0, kSlots) * 100);
    shelf_->slots = pick(0, 64);
    cache_.keepDiskLimit();
    const KvSpillSnapshot spilled = cache_.spillSnapshot();
    require(!spilled.committedBytes || spilled.heldBytes + spilled.committedBytes <= spilled.activeLimitBytes,
            "active KV stayed past its limit");
  }

  void convert() {
    Request *request = anyRequest();
    if (!request || request->mode == KvMode::Spill) return;
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
  test::TestKvStorage storage_{1024, 100, 4, 10};
  KvPool pool_{storage_, 4};
  test::TestKvTier tier_;
  std::shared_ptr<model::DiskBudget> budget_;
  engine::Cache cache_;
  std::shared_ptr<Shelf> shelf_ = std::make_shared<Shelf>();
  std::array<std::vector<uint32_t>, 3> bases_;
  std::map<uint64_t, Request> requests_;
  // What the commands in flight read.
  std::map<uint64_t, std::shared_ptr<const model::KvPageSources>> sources_;
  uint64_t nextId_ = 0;
};

void testRandomRuns() {
  for (uint32_t seed = 1; seed <= 40; ++seed) {
    Run run(seed);
    try {
      for (int step = 0; step < 600; ++step)
        run.step();
      run.drain();
    } catch (const std::exception &error) {
      throw std::runtime_error("seed " + std::to_string(seed) + ": " + error.what());
    }
  }
  std::cout << "cache property runs ok\n";
}

} // namespace

int main() {
  try {
    testRandomRuns();
    std::cout << "cache property tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache property tests failed: " << error.what() << '\n';
    return 1;
  }
}
