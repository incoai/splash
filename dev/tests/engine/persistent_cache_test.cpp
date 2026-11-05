#include "Checked.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestWriteBudget.hpp"
#include "engine/Cache.hpp"
#include "engine/DiskLabels.hpp"
#include "engine/WriteBehind.hpp"
#include "model/SlotFile.hpp"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

// The state copies' disk in these tests: each copy keeps its write's label, as a slot file's record
// does, and writes land when the shelf is ready.
class ShelvedState;
struct Shelf final {
  std::set<const ShelvedState *> disk;
  uint32_t capacity = 8;
  bool ready = true;
  // The state file takes writes; the write budget it shares with the KV
  // tier, if any.
  bool writable = true;
  std::shared_ptr<const model::DiskBudget> writeBudget;
};

class ShelvedState final : public CompositeState {
public:
  ShelvedState(std::shared_ptr<Shelf> shelf, bool disk) : shelf_(std::move(shelf)), disk_(disk) {
    if (disk_)
      shelf_->disk.insert(this);
  }
  ~ShelvedState() override {
    if (disk_)
      shelf_->disk.erase(this);
  }
  uint64_t bytes() const noexcept override { return 100; }
  uint64_t residentBytes() const noexcept override { return disk_ ? 0 : bytes(); }
  bool writable() const noexcept override { return !disk_; }
  bool canOffload(bool inUse) const noexcept override {
    return !disk_ && shelf_->writable &&
           (!shelf_->writeBudget || shelf_->writeBudget->admits(inUse));
  }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override {
    if (shelf_->disk.size() >= shelf_->capacity)
      return {};
    return std::make_unique<Written>(shelf_, std::make_shared<ShelvedState>(shelf_, true));
  }
  std::unique_ptr<StateOffload> persist(std::function<void()> done) const override {
    return offload(std::move(done));
  }
  void label(std::vector<std::byte> label) const override { label_ = std::move(label); }
  [[nodiscard]] const std::vector<std::byte> &labelled() const noexcept { return label_; }

private:
  struct Written final : StateOffload {
    Written(std::shared_ptr<Shelf> owner, std::shared_ptr<const CompositeState> copy)
        : shelf(std::move(owner)), disk(std::move(copy)) {}
    bool ready() const noexcept override { return shelf->ready; }
    bool finish() override { return true; }
    const std::shared_ptr<const CompositeState> &state() const noexcept override { return disk; }
    std::shared_ptr<Shelf> shelf;
    std::shared_ptr<const CompositeState> disk;
  };

  std::shared_ptr<Shelf> shelf_;
  bool disk_;
  mutable std::vector<std::byte> label_;
};

std::vector<std::vector<std::byte>> labels(const Shelf &shelf) {
  std::vector<std::vector<std::byte>> result;
  for (const ShelvedState *state : shelf.disk)
    if (!state->labelled().empty())
      result.push_back(state->labelled());
  return result;
}

TokenAdmission admitTokens(engine::Cache &cache, uint64_t requestId, uint64_t tokens) {
  TokenAdmission admission = cache.ensureTokens(requestId, tokens);
  while (!admission.granted() &&
         test::takeFirst(cache, test::kInServiceGrowth, {.pages = admission.additionalPages, .form = admission.form})
                 .outcome == FreeOutcome::Met)
    admission = cache.ensureTokens(requestId, tokens);
  return admission;
}

// One process's cache over eight pages, a KV tier and a shelf, persistent or not, with an optional
// spill policy (Index pages beside the eight) or window policy (Slim pages of 30 bytes).
struct Process final {
  test::TestKvStorage storage{16, 100, 1, 10, 30};
  KvPool pool{storage, 8};
  test::TestKvTier tier;
  std::shared_ptr<Shelf> shelf = std::make_shared<Shelf>();
  engine::Cache cache;

  explicit Process(bool persistent, std::shared_ptr<model::DiskBudget> budget = nullptr,
                   std::optional<CacheSpillPolicy> spill = std::nullopt,
                   std::optional<CacheWindowPolicy> window = std::nullopt)
      : tier(persistent), cache(pool, &tier, std::move(budget), spill, window) {
    storage.budgetPages = 8;
    tier.capacity = 16;
    tier.transferLimit = 8;
  }

  // Caches the prompt through a finished request and returns
  // its blocks, root first.
  std::vector<uint64_t> cachePrompt(uint64_t requestId, const std::vector<uint32_t> &prompt,
                                    CacheSalt salt = {}) {
    cache.beginRequest(requestId);
    require(admitTokens(cache, requestId, prompt.size()).granted(), "prompt KV was not admitted");
    cache.publishCommittedBlocks(requestId, prompt, static_cast<uint32_t>(prompt.size()),
                                 {{}, salt});
    std::vector<uint64_t> blocks;
    for (uint32_t end = KvCache::pageTokens; end <= prompt.size(); end += KvCache::pageTokens)
      blocks.push_back(cache.blockAt(requestId, end));
    cache.endRequest(requestId);
    return blocks;
  }

  void publish(uint64_t block, bool checkpoint = false) {
    cache.publishCompositeState(block, std::make_shared<ShelvedState>(shelf, false), checkpoint);
  }

  // Copies the restore point at this block ahead of need, as write-behind
  // does, until it is durable.
  PersistStatus persist(uint64_t block) {
    PersistStatus status = PersistStatus::Started;
    for (int step = 0; step < 16 && status != PersistStatus::Durable; ++step) {
      status = cache.persist(block);
      tier.complete();
      static_cast<void>(cache.pollTransfers());
    }
    return status;
  }

  // Everything leaves RAM: each state is written as it goes, and each KV
  // leaf a state needs is demoted, landing before the next step. Memory
  // reclaims: a windowed cache's page need would trim first.
  void writeEverything() {
    for (;;) {
      const FreeResult step = test::releaseStep(cache);
      tier.complete();
      const bool landed = cache.pollTransfers();
      if (!step.gave() && !landed)
        return;
    }
  }

  // What the next process finds of this one's tier.
  [[nodiscard]] std::vector<std::vector<std::byte>> kvLabels() const { return tier.labels(); }
  [[nodiscard]] std::vector<std::vector<std::byte>> stateLabels() const { return labels(*shelf); }

  // Takes back what an earlier process's tier recorded.
  CacheAdoption adopt(const std::vector<std::vector<std::byte>> &kv,
                      const std::vector<std::vector<std::byte>> &states) {
    std::vector<PersistedKv> blocks;
    for (const auto &label : kv)
      blocks.push_back({label, [this] { return tier.adopt(); }});
    std::vector<PersistedState> recorded;
    for (const auto &label : states)
      recorded.push_back({label, [this](uint32_t) -> std::shared_ptr<const CompositeState> {
                            return std::make_shared<ShelvedState>(shelf, true);
                          }});
    return cache.adopt(std::move(blocks), std::move(recorded));
  }
};

std::vector<uint32_t> promptOf(uint32_t seed, uint32_t pages) {
  std::vector<uint32_t> prompt;
  for (uint32_t page = 0; page < pages; ++page)
    for (uint32_t row = 0; row < KvCache::pageTokens; ++row)
      prompt.push_back(seed + page * 100 + row);
  return prompt;
}

// Memory runs short once, before anything else is cached: a conversation's state and the leaf it
// needs are written out, and nothing of it stays in RAM.
void runShortOfMemory(Process &process) {
  process.publish(process.cachePrompt(900, promptOf(9000, 1))[0]);
  require(test::stateStep(process.cache).gave() &&
              test::evictStep(process.cache, ReclaimClass::Ordinary).gave(),
          "memory did not run short");
  process.tier.complete();
  require(process.cache.pollTransfers() && process.tier.demotions == 1 &&
              process.shelf->disk.size() == 1 && !process.cache.snapshot().pool.pagesPrefix,
          "the reclaim's writes did not land");
}

// A persistent tier labels every KV and state copy with what the next process needs; a temporary
// tier labels none.
void testCopiesAreLabelled() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  for (const bool persistent : {true, false}) {
    Process process(persistent);
    const std::vector<uint64_t> blocks = process.cachePrompt(1, prompt);
    process.publish(blocks[3]);
    process.writeEverything();
    const auto kv = process.kvLabels();
    const auto states = process.stateLabels();
    require(process.tier.slots == 4 && process.shelf->disk.size() == 1,
            "the conversation did not leave RAM for the tier");
    if (!persistent) {
      require(kv.empty() && states.empty(), "a temporary tier labelled its copies");
      continue;
    }
    require(kv.size() == 4 && states.size() == 1, "a persistent tier left a copy unlabelled");
    for (const auto &bytes : kv) {
      const auto label = decodeLabel<KvBlockLabel>(bytes);
      require(label.has_value(), "a KV label did not decode");
      const auto index = std::find(blocks.begin(), blocks.end(), label->id) - blocks.begin();
      require(index < 4 && label->parent == (index ? blocks[index - 1] : 0) && label->lastUsed &&
                  std::equal(label->tokens.begin(), label->tokens.end(),
                             prompt.begin() + index * KvCache::pageTokens),
              "a KV label did not name its block, parent, tokens and recency");
    }
    const auto state = decodeLabel<StateLabel>(states.front());
    require(state && state->block == blocks[3] && state->tokens == 128 && !state->checkpoint &&
                state->lastUsed,
            "the state label did not name its block, boundary and class");
  }
}

// A windowed layout's KV labels name their page form; the next process drops a copy of an unknown
// form with the restore point it breaks.
void testWindowedCopiesNameTheirForm() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  constexpr CacheWindowPolicy window{2};
  Process earlier(true, nullptr, std::nullopt, window);
  const std::vector<uint64_t> blocks = earlier.cachePrompt(1, prompt);
  earlier.publish(blocks[1]);
  earlier.publish(blocks[3]);
  earlier.writeEverything();
  std::vector<std::vector<std::byte>> copies = earlier.kvLabels();
  require(copies.size() == 4, "a windowed layout's persistent tier left a copy unlabelled");
  for (const auto &bytes : copies) {
    const auto label = decodeKvSlotLabel(bytes, true);
    require(bytes.size() == kvSlotLabelBytes(true) && kvSlotLabelBytes(true) == sizeof(KvBlockLabel) + 1 &&
                label && label->form == kv::PageForm::Full &&
                std::ranges::find(blocks, label->block.id) != blocks.end(),
            "a windowed layout's KV label does not name its block and its page's form");
  }
  {
    Process next(true, nullptr, std::nullopt, window);
    const CacheAdoption adoption = next.adopt(copies, earlier.stateLabels());
    require(adoption.blocks == 4 && adoption.states == 2 && adoption.dropped == 0,
            "the next process did not take back a windowed layout's restore points");
  }
  for (auto &bytes : copies)
    if (decodeKvSlotLabel(bytes, true)->block.id == blocks[2])
      bytes.back() = std::byte{0xff};
  Process next(true, nullptr, std::nullopt, window);
  const CacheAdoption adoption = next.adopt(copies, earlier.stateLabels());
  require(adoption.blocks == 2 && adoption.states == 1 && adoption.dropped == 3,
          "a copy of a form the process does not know was taken back");
}

// A windowed restore point copies trimmed pages Slim and its tail Full and restores them so; a
// state whose tail is Slim is not taken back.
void testWindowedPointsKeepTheirTailsFull() {
  const std::vector<uint32_t> prompt = promptOf(3000, 6);
  constexpr CacheWindowPolicy window{2};
  // Room beside the eight Full pages for the Slim ones.
  Process earlier(true, nullptr, std::nullopt, window);
  earlier.storage.budgetPages = 16;
  const std::vector<uint64_t> blocks = earlier.cachePrompt(1, prompt);
  earlier.publish(blocks[5]);
  require(earlier.cache.trimWindows() == 4, "the blocks past the tail were not trimmed");
  const PersistStatus status = earlier.persist(blocks[5]);
  std::vector<std::vector<std::byte>> copies = earlier.kvLabels();
  require(status == PersistStatus::Durable && copies.size() == 6, "the restore point did not become durable");
  for (const auto &bytes : copies) {
    const auto label = decodeKvSlotLabel(bytes, true);
    const auto index = std::ranges::find(blocks, label->block.id) - blocks.begin();
    require(label->form == (index < 4 ? kv::PageForm::Slim : kv::PageForm::Full),
            "a copy did not name the form of the page it was written from");
  }
  {
    Process next(true, nullptr, std::nullopt, window);
    next.storage.budgetPages = 16;
    const CacheAdoption adoption = next.adopt(copies, earlier.stateLabels());
    require(adoption.blocks == 6 && adoption.states == 1 && adoption.dropped == 0,
            "the next process did not take back the restore point");
    std::vector<uint32_t> resumed = prompt;
    resumed.push_back(1);
    next.cache.beginRequest(1);
    const CacheLookup lookup = next.cache.lookup(resumed, {});
    require(lookup.resumeBoundary() == 6 * KvCache::pageTokens &&
                next.cache.restoreRequest(1, lookup).granted(),
            "the restore point did not restore");
    std::string forms;
    for (const uint32_t page : next.cache.pageTable(1).pages)
      forms += next.pool.form(page) == kv::PageForm::Slim ? 'S' : 'F';
    require(forms == "SSSSFF" && next.tier.restores == 6, "the chain did not come back in its copies' forms");
    next.tier.complete();
    static_cast<void>(next.cache.pollTransfers());
    static_cast<void>(next.cache.cutPageTable(1));
    next.cache.checkInvariants();
    next.cache.endRequest(1);
  }
  for (auto &bytes : copies)
    if (decodeKvSlotLabel(bytes, true)->block.id == blocks[4])
      bytes.back() = static_cast<std::byte>(kv::PageForm::Slim);
  Process next(true, nullptr, std::nullopt, window);
  const CacheAdoption adoption = next.adopt(copies, earlier.stateLabels());
  require(adoption.blocks == 0 && adoption.states == 0 && adoption.dropped == 7,
          "a state whose tail a Slim copy holds was taken back");
}

// Blocks recomputed over Slim-only copies get Full copies, so a state published there restores
// whole, in this process and the next.
void testRecomputedTailsKeepFullCopies() {
  for (const bool persistent : {false, true}) {
    const std::vector<uint32_t> earlier = promptOf(5000, 6);
    constexpr CacheWindowPolicy window{2};
    Process process(persistent, nullptr, std::nullopt, window);
    process.storage.budgetPages = 16;
    const std::vector<uint64_t> blocks = process.cachePrompt(1, earlier);
    process.publish(blocks[5]);
    require(process.cache.trimWindows() == 4, "the blocks past the tail were not trimmed");
    process.writeEverything();

    // A later prompt shares the first four blocks, which copies of Slim
    // pages hold, and no state there: its request computes them again.
    std::vector<uint32_t> later(earlier.begin(), earlier.begin() + 4 * KvCache::pageTokens);
    for (uint32_t row = 0; row < 2 * KvCache::pageTokens; ++row)
      later.push_back(90000 + row);
    const CacheLookup none = process.cache.lookup(later, {});
    require(!none.state && none.kvBoundary == 4 * KvCache::pageTokens, "the shared blocks were not cached alone");
    process.cache.beginRequest(2);
    require(admitTokens(process.cache, 2, later.size()).granted(), "the later prompt's KV was not admitted");
    process.cache.publishCommittedBlocks(2, later, static_cast<uint32_t>(later.size()), {});
    require(process.cache.blockAt(2, 4 * KvCache::pageTokens) == blocks[3] && process.cache.stateTailFull(blocks[3]),
            "the writer's blocks were not the cached ones, or their tail not Full");
    process.publish(blocks[3]);
    process.cache.endRequest(2);
    process.writeEverything();
    process.cache.checkInvariants();

    std::vector<uint32_t> resumed(later.begin(), later.begin() + 4 * KvCache::pageTokens + 1);
    process.cache.beginRequest(3);
    const CacheLookup lookup = process.cache.lookup(resumed, {});
    require(lookup.state && lookup.resumeBoundary() == 4 * KvCache::pageTokens &&
                process.cache.restoreRequest(3, lookup).granted(),
            "the state on the blocks computed again did not restore");
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
    const PageTableState table = process.cache.pageTable(3);
    require(table.pages.size() == 4 && process.pool.form(table.pages[2]) == kv::PageForm::Full &&
                process.pool.form(table.pages[3]) == kv::PageForm::Full,
            "a restore found its tail Slim");
    static_cast<void>(process.cache.cutPageTable(3));
    process.cache.checkInvariants();
    process.cache.endRequest(3);
    if (!persistent)
      continue;
    Process next(true, nullptr, std::nullopt, window);
    next.storage.budgetPages = 16;
    const CacheAdoption adoption = next.adopt(process.kvLabels(), process.stateLabels());
    require(adoption.states == 2 && adoption.dropped == 0,
            "the next process did not take back both restore points");
  }
}

// The next process takes each restore point back whole under its ids and recency, and new blocks
// take new ids.
void testRestorePointsComeBack() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  Process earlier(true);
  const std::vector<uint64_t> blocks = earlier.cachePrompt(1, prompt);
  earlier.publish(blocks[1]);
  earlier.publish(blocks[3]);
  earlier.writeEverything();
  Process next(true);
  const CacheAdoption adoption = next.adopt(earlier.kvLabels(), earlier.stateLabels());
  require(adoption.blocks == 4 && adoption.states == 2 && adoption.dropped == 0 &&
              adoption.bytes == 600 && next.tier.slots == 4 && next.shelf->disk.size() == 2,
          "the next process did not take back both restore points whole");
  std::vector<uint32_t> continued = prompt;
  continued.push_back(7);
  CacheLookup lookup = next.cache.lookup(continued, {});
  require(lookup.state && lookup.state->kvBlock() == blocks[3] && lookup.resumeBoundary() == 128 &&
              !lookup.state->state()->residentBytes(),
          "a lookup did not find the deepest restore point on disk");
  next.cache.beginRequest(2);
  require(next.cache.restoreRequest(2, lookup).granted() && next.tier.restores == 4,
          "the restore point's chain was not restored from disk");
  lookup = {};
  next.cache.endRequest(2);
  const std::vector<uint64_t> later = next.cachePrompt(3, promptOf(5000, 1));
  require(later.front() > blocks.back(), "new blocks reused an id taken back");
}

// A salted restore point comes back under its salt alone.
void testSaltedPointsComeBackUnderTheirSalt() {
  const std::vector<uint32_t> prompt = promptOf(1000, 2);
  const CacheSalt salt{0x5151, 0x7373};
  Process earlier(true);
  const std::vector<uint64_t> blocks = earlier.cachePrompt(1, prompt, salt);
  earlier.publish(blocks[1]);
  earlier.writeEverything();
  Process next(true);
  const CacheAdoption adoption = next.adopt(earlier.kvLabels(), earlier.stateLabels());
  require(adoption.blocks == 2 && adoption.states == 1 && adoption.dropped == 0,
          "the next process did not take back the salted restore point");
  std::vector<uint32_t> continued = prompt;
  continued.push_back(7);
  require(next.cache.lookup(continued, {{}, salt}).resumeBoundary() == 64 &&
              next.cache.lookup(continued, {}).resumeBoundary() == 0,
          "a salted restore point did not come back under its salt alone");
}

// Broken chains, KV no state needs and damaged or stale records are not taken back, and new blocks
// skip their ids.
void testBrokenChainsStayBehind() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  Process earlier(true);
  const std::vector<uint64_t> blocks = earlier.cachePrompt(1, prompt);
  earlier.publish(blocks[1]);
  earlier.publish(blocks[3]);
  earlier.writeEverything();
  std::vector<std::vector<std::byte>> kv;
  for (const auto &label : earlier.kvLabels())
    if (decodeLabel<KvBlockLabel>(label)->id != blocks[2])
      kv.push_back(label);
  // A record repeated under its id is a damaged one, and a second state at
  // one block a stale one.
  kv.push_back(kv.front());
  std::vector<std::vector<std::byte>> states = earlier.stateLabels();
  states.push_back(states.front());
  Process next(true);
  const CacheAdoption adoption = next.adopt(kv, states);
  require(adoption.blocks == 2 && adoption.states == 1 && adoption.dropped == 4 &&
              next.tier.slots == 2 && next.shelf->disk.size() == 1,
          "a broken chain, or KV no state needs, was taken back");
  CacheLookup lookup = next.cache.lookup(prompt, {});
  require(lookup.state && lookup.state->kvBlock() == blocks[1],
          "the intact restore point did not come back");
  lookup = {};
  // A block left behind keeps its id: a record of it may outlive a crash.
  require(next.cachePrompt(2, promptOf(5000, 1)).front() > blocks[3],
          "a new block took an id a record names");
}

// A full persistent tier gives up the oldest point, its state and then its own copies, keeping a
// prefix a newer point shares.
void testOldestPointGoes() {
  const std::vector<uint32_t> shared = promptOf(1000, 2);
  const auto conversation = [&](uint32_t seed) {
    std::vector<uint32_t> prompt = shared;
    const std::vector<uint32_t> own = promptOf(seed, 2);
    prompt.insert(prompt.end(), own.begin(), own.end());
    return prompt;
  };
  const std::vector<uint32_t> older = conversation(2000), newer = conversation(3000);
  Process process(true);
  process.tier.capacity = 7;
  process.shelf->capacity = 2;
  const std::vector<uint64_t> olderBlocks = process.cachePrompt(1, older);
  process.publish(olderBlocks[3]);
  const std::vector<uint64_t> newerBlocks = process.cachePrompt(2, newer);
  process.publish(newerBlocks[3]);
  process.writeEverything();
  require(process.tier.slots == 6 && process.shelf->disk.size() == 2,
          "both points did not reach the tier");
  const auto stateAt = [&](const std::vector<uint32_t> &prompt) {
    std::vector<uint32_t> probe = prompt;
    probe.push_back(7);
    const CacheLookup lookup = process.cache.lookup(probe, {});
    return lookup.state ? lookup.state->kvBlock() : 0;
  };
  const std::vector<uint32_t> thirdPrompt = promptOf(4000, 2);
  const std::vector<uint64_t> third = process.cachePrompt(3, thirdPrompt);
  process.publish(third[1]);
  process.writeEverything();
  // The oldest point's state went, and of its own copies the leaf the new
  // point needed room for; the other stays until room is needed again.
  require(process.tier.slots == 7 && process.shelf->disk.size() == 2 && !stateAt(older) &&
              stateAt(newer) == newerBlocks[3] && stateAt(thirdPrompt) == third[1],
          "the new point did not take the oldest point's room");
  process.shelf->capacity = 3;
  const std::vector<uint32_t> fourthPrompt = promptOf(5000, 1);
  const uint64_t fourth = process.cachePrompt(4, fourthPrompt)[0];
  process.publish(fourth);
  process.writeEverything();
  require(process.tier.slots == 7 && process.shelf->disk.size() == 3 &&
              stateAt(newer) == newerBlocks[3] && stateAt(thirdPrompt) == third[1] &&
              stateAt(fourthPrompt) == fourth,
          "a copy no state needs stayed while a point was given up");
}

// A full persistent tier keeps the only copy of a point a queued request restores and gives up the
// next oldest instead.
void testAPointAQueuedRequestRestoresStays() {
  const std::vector<uint32_t> shared = promptOf(1000, 2);
  const auto conversation = [&](uint32_t seed) {
    std::vector<uint32_t> prompt = shared;
    const std::vector<uint32_t> own = promptOf(seed, 2);
    prompt.insert(prompt.end(), own.begin(), own.end());
    return prompt;
  };
  const std::vector<uint32_t> older = conversation(2000), newer = conversation(3000);
  Process process(true);
  process.tier.capacity = 7;
  process.shelf->capacity = 2;
  const std::vector<uint64_t> olderBlocks = process.cachePrompt(1, older);
  process.publish(olderBlocks[3]);
  const std::vector<uint64_t> newerBlocks = process.cachePrompt(2, newer);
  process.publish(newerBlocks[3]);
  process.writeEverything();
  require(process.tier.slots == 6 && process.shelf->disk.size() == 2, "both points did not reach the tier");
  const auto probeOf = [](const std::vector<uint32_t> &prompt) {
    std::vector<uint32_t> probe = prompt;
    probe.push_back(7);
    return probe;
  };
  const auto stateAt = [&](const std::vector<uint32_t> &prompt) {
    const CacheLookup lookup = process.cache.lookup(probeOf(prompt), {});
    return lookup.state ? lookup.state->kvBlock() : 0;
  };
  const std::vector<uint32_t> queuedPrompt = probeOf(older);
  const CacheProbe queued = process.cache.probe(queuedPrompt, {});
  process.cache.setWantedChains({queued.restoredChain()});
  const std::vector<uint32_t> thirdPrompt = promptOf(4000, 2);
  const std::vector<uint64_t> third = process.cachePrompt(3, thirdPrompt);
  process.publish(third[1]);
  process.writeEverything();
  process.cache.setWantedChains({});
  require(stateAt(older) == olderBlocks[3] && !stateAt(newer) && stateAt(thirdPrompt) == third[1],
          "the new point did not take the room of the point no queued request restores");
}

// The oldest point whose state stays in RAM gives up its copies first; the newer points stay
// durable and come back.
void testOldestPointInRamGoesWithItsChain() {
  Process process(true);
  process.tier.capacity = 6;
  process.shelf->capacity = 2;
  const auto keep = [&](uint64_t point) {
    const PersistStatus status = process.cache.persist(point);
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
    return status;
  };
  const std::vector<uint64_t> oldest = process.cachePrompt(1, promptOf(1000, 3));
  process.publish(oldest[2]);
  const uint64_t older = process.cachePrompt(2, promptOf(2000, 3))[2];
  process.publish(older);
  require(keep(oldest[2]) == PersistStatus::Started && keep(older) == PersistStatus::Started &&
              process.tier.slots == 6 && process.shelf->disk.size() == 2,
          "the two points did not fill the tier");
  const uint64_t newest = process.cachePrompt(3, promptOf(3000, 2))[1];
  process.publish(newest);
  const auto copiesOf = [&](const std::vector<uint64_t> &chain) {
    return std::ranges::count_if(process.kvLabels(), [&](const auto &bytes) {
      return std::ranges::find(chain, decodeLabel<KvBlockLabel>(bytes)->id) != chain.end();
    });
  };
  require(keep(newest) == PersistStatus::Started && process.cache.persist(older) == PersistStatus::Durable &&
              process.cache.persist(newest) == PersistStatus::Durable && copiesOf(oldest) == 1 &&
              process.cache.stateResident(oldest[2]),
          "the newest point did not take the room of the oldest point's chain");
  Process next(true);
  const CacheAdoption adoption = next.adopt(process.kvLabels(), process.stateLabels());
  require(adoption.states == 2 && adoption.blocks == 5 && adoption.dropped == 1,
          "the next process did not take the two newer points back whole");
  next.cache.checkInvariants();
}

// Room for a point's state never takes its own chain: the deeper point's copies go, then the
// oldest other point, and only the state is written.
void testPointKeepsItsChainWhileItsStateIsWritten() {
  Process process(true);
  process.shelf->capacity = 0;
  const auto keep = [&](uint64_t point) {
    const PersistStatus status = process.cache.persist(point);
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
    return status;
  };
  const auto copiesOf = [&](std::initializer_list<uint64_t> blocks) {
    return std::ranges::count_if(process.kvLabels(), [&](const auto &bytes) {
      return std::ranges::find(blocks, decodeLabel<KvBlockLabel>(bytes)->id) != blocks.end();
    });
  };
  const std::vector<uint64_t> chain = process.cachePrompt(1, promptOf(1000, 4));
  process.publish(chain[3]);
  require(keep(chain[3]) == PersistStatus::Refused, "a point the state file has no room for was kept");
  process.cache.checkInvariants();
  process.shelf->capacity = 1;
  require(keep(chain[3]) == PersistStatus::Started && process.tier.slots == 4 &&
              test::stateStep(process.cache).gave() &&
              !process.cache.stateResident(chain[3]),
          "the deeper point did not reach the tier and leave RAM");
  // The deeper point, on disk alone, gives way to the next one.
  const uint64_t other = process.cachePrompt(2, promptOf(2000, 1))[0];
  process.publish(other);
  require(keep(other) == PersistStatus::Started && process.shelf->disk.size() == 1 &&
              process.tier.slots == 5,
          "the deeper point did not give way to the next one");
  process.publish(chain[1]);
  const uint32_t writes = process.tier.demotions;
  const uint64_t stateWrites = process.cache.snapshot().stateCache.offloads;
  require(keep(chain[1]) == PersistStatus::Started &&
              process.cache.persist(chain[1]) == PersistStatus::Durable &&
              process.tier.demotions == writes &&
              process.cache.snapshot().stateCache.offloads == stateWrites + 1,
          "making room for the point's state took its own chain, which it wrote again");
  require(copiesOf({chain[0], chain[1]}) == 2 && !copiesOf({chain[2], chain[3]}) &&
              process.cache.stateResident(other) && process.shelf->disk.size() == 1,
          "room for the point's state was not the deeper point's copies, then the oldest point");
  process.cache.checkInvariants();
  Process next(true);
  const CacheAdoption adoption = next.adopt(process.kvLabels(), process.stateLabels());
  require(adoption.states == 1 && adoption.blocks == 2 && adoption.dropped == 1,
          "the next process did not take the point back whole");
  next.cache.checkInvariants();
}

// A point whose room would take its own state is refused, and its chain is not touched.
void testPointGivenUpForItselfIsRefused() {
  Process process(true);
  process.tier.capacity = 2;
  const std::vector<uint64_t> blocks = process.cachePrompt(1, promptOf(1000, 4));
  process.publish(blocks[3]);
  // The state is written, and the two leaves under it are demoted: the
  // chain is two resident pages without copies over two disk-only blocks.
  while (process.tier.slots < 2) {
    require(test::evictStep(process.cache).gave(),
            "the point did not leave RAM");
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
  }
  require(!process.cache.stateResident(blocks[3]) && process.shelf->disk.size() == 1,
          "the point's state was not written");
  require(process.cache.persist(blocks[3]) == PersistStatus::Refused &&
              !process.shelf->disk.size(),
          "a point given up for itself was not refused");
}

// A point larger than the whole quota is refused before anything gives way to it.
void testPointLargerThanTheQuotaIsRefused() {
  // Two pages and a state fit; four pages and a state do not.
  Process process(true, std::make_shared<model::DiskBudget>(450));
  const uint64_t small = process.cachePrompt(1, promptOf(1000, 2))[1];
  process.publish(small);
  require(process.cache.persist(small) == PersistStatus::Started, "the small point was not kept");
  process.tier.complete();
  static_cast<void>(process.cache.pollTransfers());
  require(process.cache.persist(small) == PersistStatus::Durable, "the small point is not durable");
  const uint64_t large = process.cachePrompt(2, promptOf(2000, 4))[3];
  process.publish(large);
  require(process.cache.persist(large) == PersistStatus::Refused && process.tier.slots == 2 &&
              process.shelf->disk.size() == 1 &&
              process.cache.persist(small) == PersistStatus::Durable,
          "a point larger than the quota displaced another");
}

// No reclaim frees a state's RAM copy while its write still reads it.
void testPersistingStateStaysInRam() {
  Process process(true);
  const uint64_t block = process.cachePrompt(1, promptOf(1000, 2))[1];
  process.publish(block);
  process.shelf->ready = false;
  require(process.cache.persist(block) == PersistStatus::Started, "the point was not written");
  process.tier.complete();
  static_cast<void>(process.cache.pollTransfers());
  const auto scan = [&] {
    return test::takeStep(process.cache, test::kInServiceGrowth, {.hostRefused = true, .pages = 1}).gave();
  };
  static_cast<void>(scan());
  require(process.cache.stateResident(block), "a page scan freed a state its write still read");
  process.shelf->ready = true;
  static_cast<void>(process.cache.pollTransfers());
  require(process.cache.persist(block) == PersistStatus::Durable &&
              scan() &&
              !process.cache.stateResident(block),
          "the state's RAM copy did not go once its write landed");
}

// A clean stop relabels each state copy with its current recency; until then labels keep their
// write's.
void testCleanStopRecordsRecency() {
  Process process(true);
  const std::vector<uint32_t> firstPrompt = promptOf(1000, 2);
  const uint64_t first = process.cachePrompt(1, firstPrompt)[1];
  process.publish(first);
  const uint64_t second = process.cachePrompt(2, promptOf(2000, 2))[1];
  process.publish(second);
  process.writeEverything();
  const auto labelled = [&] {
    std::map<uint64_t, uint64_t> recency;
    for (const auto &bytes : process.stateLabels()) {
      const auto label = decodeLabel<StateLabel>(bytes);
      recency[label->block] = label->lastUsed;
    }
    return recency;
  };
  require(labelled()[first] < labelled()[second], "the labels did not follow the writes");
  std::vector<uint32_t> again = firstPrompt;
  again.push_back(7);
  require(process.cache.lookup(again, {}).state.has_value(), "the first point was not found");
  require(labelled()[first] < labelled()[second], "a use relabelled a copy before the stop");
  process.cache.relabelStates();
  require(labelled()[first] > labelled()[second], "the clean stop did not record the newest use");
}

// A disk checkpoint reused as an ordinary state is relabelled at once, so it comes back ordinary
// after a crash.
void testReusedCheckpointsComeBackOrdinary() {
  Process process(true);
  const uint64_t reused = process.cachePrompt(1, promptOf(1000, 2))[1];
  process.publish(reused, true);
  const uint64_t resumed = process.cachePrompt(2, promptOf(2000, 2))[1];
  process.publish(resumed, true);
  process.writeEverything();
  const auto checkpoints = [&] {
    std::set<uint64_t> labelled;
    for (const auto &bytes : process.stateLabels())
      if (const auto label = decodeLabel<StateLabel>(bytes); label->checkpoint)
        labelled.insert(label->block);
    return labelled;
  };
  require(checkpoints() == std::set<uint64_t>{reused, resumed}, "the checkpoints were not written as such");
  require(process.cache.reuseStoredState(reused), "the checkpoint on disk was not reused");
  const StateUse use = process.cache.useState(resumed);
  require(checkpoints().empty(), "a checkpoint that became ordinary kept its label until a clean stop");
  Process next(true);
  require(next.adopt(process.kvLabels(), process.stateLabels()).states == 2 &&
              !next.cache.checkpointState(reused) && !next.cache.checkpointState(resumed),
          "the next process took a reused checkpoint back as a checkpoint");
}

// States of a slot file kept for the next process: each holds a slot of it.
class SlotState final : public CompositeState {
public:
  explicit SlotState(std::shared_ptr<model::SlotFile::Slot> slot) : slot_(std::move(slot)) {}
  uint64_t bytes() const noexcept override { return kHostPageBytes; }
  uint64_t residentBytes() const noexcept override { return 0; }

private:
  std::shared_ptr<model::SlotFile::Slot> slot_;
};

// Writes an earlier process's three one-page states at blocks 1 to 3, newest last, to a persistent
// state file; returns their KV blocks' labels.
std::vector<std::vector<std::byte>> writeThreeStates(const model::SlotFile::Persistence &persistence) {
  std::vector<std::vector<std::byte>> kv;
  model::SlotFile file(kHostPageBytes, std::make_shared<model::DiskBudget>(3 * kHostPageBytes),
                       persistence);
  file.finishAdoption();
  std::vector<std::shared_ptr<model::SlotFile::Slot>> kept;
  std::vector<std::byte> payload(kHostPageBytes, std::byte{1});
  for (uint64_t block = 1; block <= 3; ++block) {
    kept.push_back(file.acquire());
    require(file.write(kept.back(), {payload}, {})->wait(), "state slot write failed");
    file.label(kept.back(), encodeLabel(StateLabel{block, 10 * block, KvCache::pageTokens, 0}));
    KvBlockLabel label{block, 0, 10 * block, 0, 0, {}};
    label.tokens.fill(static_cast<uint32_t>(block));
    kv.push_back(encodeLabel(label));
  }
  require(file.synchronize({})->wait(), "state file did not synchronize");
  file.seal();
  return kv;
}

// The next process takes the states back into `next`'s cache.
void adoptStates(Process &next, model::SlotFile &file, const std::vector<std::vector<std::byte>> &kv) {
  std::vector<PersistedKv> blocks;
  for (const auto &label : kv)
    blocks.push_back({label, [&] { return next.tier.adopt(); }});
  std::vector<PersistedState> states;
  for (const model::SlotRecord &record : file.records())
    states.push_back({record.label, [&file, record](uint32_t) -> std::shared_ptr<const CompositeState> {
                        return std::make_shared<SlotState>(file.adopt(record));
                      }});
  static_cast<void>(next.cache.adopt(std::move(blocks), std::move(states)));
  file.finishAdoption();
}

std::vector<uint32_t> promptOfState(uint32_t block) {
  return std::vector<uint32_t>(KvCache::pageTokens + 1, block);
}

// Adoption under a lower limit gives up the oldest copies until the rest fits.
void testAdoptionTrimsToTheLimit() {
  const test::TemporaryDirectory directory("splash-adoption");
  const model::SlotFile::Persistence persistence{
      directory.path() / "state.slots", directory.path() / "state.records", {}, sizeof(StateLabel)};
  const std::vector<std::vector<std::byte>> kv = writeThreeStates(persistence);
  // The volume leaves less than the quota now.
  auto budget = std::make_shared<model::DiskBudget>(3 * kHostPageBytes);
  budget->setLimitBytes(2 * kHostPageBytes);
  model::SlotFile file(kHostPageBytes, budget, persistence);
  Process next(true, budget);
  adoptStates(next, file, kv);
  require(next.cache.snapshot().stateCache.entries == 2 &&
              budget->usedBytes() == 2 * kHostPageBytes,
          "adoption past the limit was not trimmed to it");
  for (uint32_t block = 2; block <= 3; ++block)
    require(next.cache.lookup(promptOfState(block), {}).state.has_value(),
            "a newer state was given up first");
  require(!next.cache.lookup(promptOfState(1), {}).state,
          "the oldest state was kept past the limit");
}

// A limit the volume lowers gives up the oldest copies not in use at the next measure; a copy in
// use stays.
void testLowerLimitKeepsCopiesInUse() {
  const test::TemporaryDirectory directory("splash-lower-limit");
  const model::SlotFile::Persistence persistence{
      directory.path() / "state.slots", directory.path() / "state.records", {}, sizeof(StateLabel)};
  const std::vector<std::vector<std::byte>> kv = writeThreeStates(persistence);
  auto budget = std::make_shared<model::DiskBudget>(3 * kHostPageBytes);
  model::SlotFile file(kHostPageBytes, budget, persistence);
  Process next(true, budget);
  adoptStates(next, file, kv);
  uint32_t measures = 0;
  uint64_t volumeLeaves = 2 * kHostPageBytes;
  next.cache.setDiskMeasure([&] {
    ++measures;
    static_cast<void>(budget->takeFullReport());
    budget->setLimitBytes(volumeLeaves);
  });
  uint64_t oldest = 0;
  {
    const CacheLookup found = next.cache.lookup(promptOfState(1), {});
    require(found.state.has_value() && budget->usedBytes() == 3 * kHostPageBytes,
            "the states were not taken back");
    oldest = found.state->kvBlock();
  }
  const StateUse use = next.cache.useState(oldest);
  next.cache.keepDiskLimit();
  require(measures == 1 && budget->usedBytes() == 2 * kHostPageBytes &&
              next.cache.lookup(promptOfState(1), {}).state &&
              !next.cache.lookup(promptOfState(2), {}).state &&
              next.cache.lookup(promptOfState(3), {}).state,
          "a lower limit did not give up the oldest copy not in use");
  // Transfers polled without a report measure nothing; a write that found
  // the volume full has it measured at once.
  volumeLeaves = 0;
  static_cast<void>(next.cache.pollTransfers());
  require(measures == 1 && budget->usedBytes() == 2 * kHostPageBytes,
          "a poll without a full volume measured it");
  budget->reportFull();
  static_cast<void>(next.cache.pollTransfers());
  require(measures == 2 && !budget->fullReported() && budget->usedBytes() == kHostPageBytes &&
              next.cache.lookup(promptOfState(1), {}).state &&
              !next.cache.lookup(promptOfState(3), {}).state,
          "a full volume was not measured at once, or a copy in use went for it");
}

// A point made durable stays in RAM, and is durable once every write of it landed.
void testDurablePointsStayInRam() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  Process process(true);
  const std::vector<uint64_t> blocks = process.cachePrompt(1, prompt);
  process.publish(blocks[3]);
  process.shelf->ready = false;
  require(process.cache.persist(blocks[3]) == PersistStatus::Started &&
              process.tier.demotions == 4 && process.shelf->disk.size() == 1,
          "the point's chain and state were not written");
  process.tier.complete();
  require(process.cache.pollTransfers() &&
              process.cache.persist(blocks[3]) == PersistStatus::Started,
          "the point was durable before its state landed");
  process.shelf->ready = true;
  require(process.cache.pollTransfers() &&
              process.cache.persist(blocks[3]) == PersistStatus::Durable &&
              process.tier.demotions == 4,
          "the point did not become durable once its writes landed");
  const CacheSnapshot snapshot = process.cache.snapshot();
  require(snapshot.pool.pagesPrefix == 4 && snapshot.kvTier.copies == 4 &&
              process.cache.stateResident(blocks[3]),
          "the durable point left RAM");
  Process next(true);
  const CacheAdoption adoption = next.adopt(process.kvLabels(), process.stateLabels());
  require(adoption.blocks == 4 && adoption.states == 1,
          "the next process did not take the durable point back whole");
}

// Only the newest points are copied: none a conversation went on from, a gone one or a checkpoint;
// one two conversations share, or only a checkpoint follows, is.
void testOnlyTheNewestPointsAreKept() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  for (const bool persistent : {true, false}) {
    Process process(persistent);
    const std::vector<uint64_t> blocks = process.cachePrompt(1, prompt);
    process.publish(blocks[1]);
    process.publish(blocks[2], true);
    process.publish(blocks[3]);
    require(process.cache.persist(blocks[1]) == PersistStatus::Unneeded &&
                process.cache.persist(blocks[2]) == PersistStatus::Unneeded &&
                process.cache.persist(blocks[3] + 100) == PersistStatus::Unneeded &&
                !process.tier.demotions,
            "a superseded point, a checkpoint or a gone point was written");
    std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + 2 * KvCache::pageTokens);
    const std::vector<uint32_t> own = promptOf(5000, 2);
    branch.insert(branch.end(), own.begin(), own.end());
    process.publish(process.cachePrompt(2, branch)[3]);
    require(process.cache.persist(blocks[1]) == PersistStatus::Started,
            "a point two conversations went on from was not kept");

    Process followed(persistent);
    const std::vector<uint64_t> chain = followed.cachePrompt(1, prompt);
    followed.publish(chain[1]);
    followed.publish(chain[3], true);
    require(followed.cache.persist(chain[1]) == PersistStatus::Started,
            "a point only a checkpoint follows was not kept");
  }
}

// Nothing is copied while a restore waits on the tier.
void testRestoresGoFirst() {
  const std::vector<uint32_t> earlier = promptOf(1000, 2);
  for (const bool persistent : {true, false}) {
    Process process(persistent);
    process.publish(process.cachePrompt(1, earlier)[1]);
    process.writeEverything();
    std::vector<uint32_t> continued = earlier;
    continued.push_back(7);
    CacheLookup lookup = process.cache.lookup(continued, {});
    process.cache.beginRequest(2);
    require(lookup.state && process.cache.restoreRequest(2, lookup).granted() &&
                process.tier.restores == 2,
            "the earlier point was not restored");
    const std::vector<uint64_t> blocks = process.cachePrompt(3, promptOf(5000, 2));
    process.publish(blocks[1]);
    const uint32_t demotions = process.tier.demotions;
    require(process.cache.persist(blocks[1]) == PersistStatus::Busy &&
                process.tier.demotions == demotions,
            "a point was written while a restore waited");
    process.tier.complete();
    require(process.cache.pollTransfers() &&
                process.cache.persist(blocks[1]) == PersistStatus::Started,
            "the point was not written once the restore landed");
    lookup = {};
    process.cache.endRequest(2);
  }
}

// A point's copies wait for writes needed now, a reclaim's demotion or a Spill write-through, and
// start once they land.
void testCopiesWaitForWritesNeededNow() {
  for (const bool persistent : {true, false}) {
    // The tier's slots draw on the quota, as a slot file's do.
    const auto budget = std::make_shared<model::DiskBudget>(1600);
    Process process(persistent, budget, CacheSpillPolicy{1, 1});
    process.tier.budget = budget;
    const uint64_t demoted = process.cachePrompt(1, promptOf(1000, 1))[0];
    process.publish(demoted);
    const uint64_t point = process.cachePrompt(2, promptOf(2000, 2))[1];
    process.publish(point);
    // The older state is written as it leaves RAM, and its leaf, which that
    // state needs, is demoted.
    require(test::stateStep(process.cache).gave() &&
                process.cache.pollTransfers() && !process.cache.stateResident(demoted),
            "the older state did not leave for the SSD");
    require(test::evictStep(process.cache, ReclaimClass::Ordinary).gave() &&
                process.tier.demotions == 1 && process.cache.pendingPages(kv::PageForm::Full) == 1,
            "the older state's leaf was not demoted");
    require(process.cache.persist(point) == PersistStatus::Busy && process.tier.demotions == 1 &&
                process.shelf->disk.size() == 1,
            "a point was copied beside a reclaim's demotion");
    process.tier.complete();
    require(process.cache.pollTransfers() && process.cache.persist(point) == PersistStatus::Started &&
                process.tier.demotions == 3 && process.shelf->disk.size() == 2,
            "the point was not copied once the demotion landed");
    process.tier.complete();
    require(process.cache.pollTransfers() && process.cache.persist(point) == PersistStatus::Durable,
            "the point did not get its copies");

    const uint64_t later = process.cachePrompt(3, promptOf(3000, 1))[0];
    process.publish(later);
    const std::vector<uint32_t> spilling = promptOf(4000, 2);
    process.cache.beginRequest(4, KvMode::Spill);
    require(process.cache.commitSpill(4, 2) &&
                admitTokens(process.cache, 4, spilling.size()).granted(),
            "the Spill request was not admitted");
    process.cache.publishCommittedBlocks(4, spilling, static_cast<uint32_t>(spilling.size()), {});
    require(process.cache.writeThrough() == 2, "the Spill request's pages were not written through");
    require(process.cache.persist(later) == PersistStatus::Busy && process.tier.demotions == 5 &&
                process.shelf->disk.size() == 2,
            "a point was copied beside a Spill request's writes through");
    process.tier.complete();
    require(process.cache.pollTransfers() && process.cache.persist(later) == PersistStatus::Started &&
                process.tier.demotions == 6 && process.shelf->disk.size() == 3,
            "the point was not copied once the writes through landed");
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
    process.cache.endRequest(4);
    process.cache.checkInvariants();
  }
}

// The memory's demand for copies ahead a step passes where a test wants them
// made: more than any cache's victims hold (WriteBehind::run).
constexpr uint64_t kCopiesDemanded = std::numeric_limits<uint64_t>::max();

// A temporary tier's copies take only free quota, never room Spill requests committed, and give no
// copy up for a point.
void testTemporaryCopiesTakeOnlyFreeRoom() {
  const std::vector<uint32_t> olderPrompt = promptOf(1000, 2);
  const auto stateAt = [](Process &process, const std::vector<uint32_t> &prompt) {
    std::vector<uint32_t> probe = prompt;
    probe.push_back(7);
    const CacheLookup lookup = process.cache.lookup(probe, {});
    return lookup.state ? lookup.state->kvBlock() : 0;
  };
  {
    // The older point's two copies leave room for one page more: the new
    // point fits the quota, not the room left. The tier's slots draw on the
    // quota, as a slot file's do.
    const auto budget = std::make_shared<model::DiskBudget>(350);
    Process process(false, budget);
    process.tier.budget = budget;
    const uint64_t older = process.cachePrompt(1, olderPrompt)[1];
    process.publish(older);
    process.writeEverything();
    require(process.tier.slots == 2 && process.shelf->disk.size() == 1,
            "the older point did not reach the tier");
    const uint64_t point = process.cachePrompt(2, promptOf(2000, 2))[1];
    process.publish(point);
    require(process.cache.persist(point) == PersistStatus::Refused && process.tier.demotions == 2 &&
                process.tier.slots == 2 && process.shelf->disk.size() == 1 &&
                stateAt(process, olderPrompt) == older,
            "a temporary tier gave a copy up for a point, or wrote what the room cannot hold");
  }
  // Room Spill requests committed is no room for copies.
  const auto budget = std::make_shared<model::DiskBudget>(600);
  Process process(false, budget, CacheSpillPolicy{1, 1});
  process.tier.budget = budget;
  const uint64_t point = process.cachePrompt(1, promptOf(1000, 2))[1];
  process.publish(point);
  process.cache.beginRequest(2, KvMode::Spill);
  require(process.cache.commitSpill(2, 4), "the Spill request did not commit its pages");
  require(process.cache.persist(point) == PersistStatus::Refused && !process.tier.demotions &&
              process.shelf->disk.empty(),
          "a point took room a Spill request committed");
  process.cache.endRequest(2);
  require(process.cache.persist(point) == PersistStatus::Started && process.tier.demotions == 2,
          "the point was not copied once the commitment returned");
}

// A point is written once it waited its delay, unless its conversation went on; a short one never;
// a clean stop writes every waiting point, the newest first.
void testPointsWaitTheirTurn() {
  constexpr double kDelay = WriteBehind::kDelayMilliseconds;
  Process process(true);
  WriteBehind writes(process.cache);
  const std::vector<uint64_t> first = process.cachePrompt(1, promptOf(1000, 2));
  process.publish(first[1]);
  writes.published(first[1], WriteBehind::kMinimumTokens, 0.0);
  const std::vector<uint64_t> second = process.cachePrompt(2, promptOf(2000, 1));
  process.publish(second[0]);
  writes.published(second[0], WriteBehind::kMinimumTokens - 1, 0.0);
  require(writes.snapshot().waiting == 1 && !writes.run(kDelay - 1, kCopiesDemanded, false) && !process.tier.demotions &&
              writes.nextWakeup() == kDelay,
          "a point was written before its delay, or a short one was queued");
  require(!writes.run(kDelay, kCopiesDemanded, false) && process.tier.demotions == 2 && !writes.nextWakeup(),
          "a due point was not written, or its writes did not hold the queue");
  process.tier.complete();
  require(process.cache.pollTransfers() && writes.run(kDelay + 1, kCopiesDemanded, false) &&
              writes.snapshot().durable == 1 && !writes.snapshot().waiting,
          "the point did not become durable once its writes landed");

  // The conversation goes on before its point falls due.
  std::vector<uint32_t> longer = promptOf(3000, 2);
  const std::vector<uint64_t> turn = process.cachePrompt(3, longer);
  process.publish(turn[1]);
  writes.published(turn[1], WriteBehind::kMinimumTokens, 2 * kDelay);
  const std::vector<uint32_t> more = promptOf(4000, 1);
  longer.insert(longer.end(), more.begin(), more.end());
  const uint64_t next = process.cachePrompt(4, longer)[2];
  process.publish(next);
  writes.published(next, WriteBehind::kMinimumTokens, 2 * kDelay + 1);
  require(writes.run(3 * kDelay, kCopiesDemanded, false) && writes.snapshot().unneeded == 1 &&
              writes.snapshot().waiting == 1 && process.tier.demotions == 2,
          "a point its conversation went on from was written");

  // A clean stop takes the newest point first, before it is due.
  const std::vector<uint64_t> newest = process.cachePrompt(5, promptOf(6000, 1));
  process.publish(newest[0]);
  writes.published(newest[0], WriteBehind::kMinimumTokens, 3 * kDelay);
  process.shelf->ready = false;
  require(!writes.flush() && process.tier.demotions == 3,
          "the clean stop did not start with the newest point");
  process.tier.complete();
  process.shelf->ready = true;
  require(process.cache.pollTransfers() && !writes.flush(),
          "the clean stop did not go on to the next point");
  process.tier.complete();
  require(process.cache.pollTransfers() && writes.flush() && writes.snapshot().durable == 3 &&
              !writes.snapshot().waiting,
          "the clean stop did not leave every point durable");
}

// A due point waits while the write budget is spent and wakes when it refills; a clean stop writes
// whatever the budget, but nothing past the volume's limit.
void testWritesKeepToTheWriteBudget() {
  constexpr double kDelay = WriteBehind::kDelayMilliseconds;
  test::WriteBudget budget;
  Process process(true, budget.budget);
  process.shelf->writeBudget = budget.budget;
  WriteBehind writes(process.cache);
  const std::vector<uint64_t> blocks = process.cachePrompt(1, promptOf(1000, 2));
  process.publish(blocks[1]);
  writes.published(blocks[1], WriteBehind::kMinimumTokens, budget.now);
  budget.spend();
  budget.now = kDelay;
  require(!writes.run(budget.now, kCopiesDemanded, false) && !process.tier.demotions && process.shelf->disk.empty(),
          "a due point was written while the write budget was spent");
  const std::optional<double> wakeup = writes.nextWakeup();
  require(wakeup && std::abs(*wakeup - test::WriteBudget::kRefillMilliseconds) < 1.0,
          "the point was not woken when the write budget has room again");
  budget.refill();
  require(!writes.run(budget.now, kCopiesDemanded, false) && process.shelf->disk.size() == 1 &&
              process.tier.demotions == 2,
          "a due point waited once the write budget had room");
  process.tier.complete();
  require(process.cache.pollTransfers() && writes.run(budget.now, kCopiesDemanded, false) &&
              writes.snapshot().durable == 1,
          "the point did not become durable once its writes landed");

  const std::vector<uint64_t> next = process.cachePrompt(2, promptOf(2000, 1));
  process.publish(next[0]);
  writes.published(next[0], WriteBehind::kMinimumTokens, budget.now);
  budget.spend();
  require(!writes.flush() && process.tier.demotions == 3,
          "the clean stop waited for the write budget");
  process.tier.complete();
  require(process.cache.pollTransfers() && writes.flush() && writes.snapshot().durable == 2,
          "the clean stop did not leave the point durable");

  // Nothing at a stop raises a limit the volume holds below a point.
  const std::vector<uint64_t> last = process.cachePrompt(3, promptOf(3000, 2));
  process.publish(last[1]);
  writes.published(last[1], WriteBehind::kMinimumTokens, budget.now);
  budget.budget->setLimitBytes(250);
  require(writes.flush() && writes.snapshot().refused == 1 && process.tier.demotions == 3,
          "the clean stop waited for a limit below the point");
}

// A point waits while the write budget holds its state back, and is refused past the limit or when
// a file takes no writes.
void testPointsWaitForTheTier() {
  for (const bool persistent : {true, false}) {
    test::WriteBudget budget;
    Process process(persistent, budget.budget);
    process.shelf->writeBudget = budget.budget;
    const uint64_t point = process.cachePrompt(1, promptOf(1000, 2))[1];
    process.publish(point);
    budget.spend();
    require(process.cache.persist(point) == PersistStatus::Busy && !process.tier.demotions &&
                process.shelf->disk.empty(),
            "a point was written or refused while the write budget was spent");
    budget.refill();
    // Two pages and a state are more than the limit, not than the quota.
    budget.budget->setLimitBytes(250);
    require(process.cache.persist(point) == PersistStatus::Refused && !process.tier.demotions &&
                process.shelf->disk.empty(),
            "a point larger than the limit was written or kept waiting");
    budget.budget->setLimitBytes(budget.budget->capacityBytes());
    require(process.cache.persist(point) == PersistStatus::Started &&
                process.shelf->disk.size() == 1 && process.tier.demotions == 2,
            "the point was not written once the tier had room");
    process.tier.complete();
    require(process.cache.pollTransfers() &&
                process.cache.persist(point) == PersistStatus::Durable,
            "the point is not durable");

    const uint64_t second = process.cachePrompt(2, promptOf(2000, 1))[0];
    process.publish(second);
    process.shelf->writable = false;
    require(process.cache.persist(second) == PersistStatus::Refused,
            "a point was kept waiting for a state file that takes no writes");
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
    const uint64_t third = process.cachePrompt(3, promptOf(3000, 1))[0];
    process.publish(third);
    process.tier.writableFile = false;
    require(process.cache.persist(third) == PersistStatus::Refused,
            "a point was kept waiting for a KV file that takes no writes");
  }
}

// Copies ahead of a Spill request's pages count as pages on their way back: a refused page waits
// for them and is free once they land.
void testCopiesAheadOfSpilledPagesArePending() {
  const auto budget = std::make_shared<model::DiskBudget>(1600);
  Process process(true, budget, CacheSpillPolicy{1, 1});
  process.tier.budget = budget;
  // Room for an Index extent beside the eight Full pages.
  process.storage.budgetPages = 9;
  require(process.cache.spillWritten(SpilledExtents::KeepRunway).granted(), "the cache did not allocate the Index runway");
  std::vector<uint32_t> prompt = promptOf(1000, 4);
  const std::vector<uint64_t> chain = process.cachePrompt(1, prompt);
  process.publish(chain[3]);
  require(process.cache.persist(chain[3]) == PersistStatus::Started && process.cache.pollTransfers() &&
              process.tier.inFlight() == 4,
          "the point's copies did not start");
  prompt.push_back(1);
  const CacheLookup lookup = process.cache.lookup(prompt, {});
  process.cache.beginRequest(2, KvMode::Spill);
  require(process.cache.commitSpill(2, 0) && process.cache.restoreRequest(2, lookup).granted() &&
              process.cache.pendingPages(kv::PageForm::Full) == 1,
          "the copies ahead of the Spill request's pages did not count as pages on their way back");
  process.storage.growthBlocked = true;
  const KvPageAcquisition held = process.pool.acquirePages(process.pool.freePageCount());
  // Its open page, past the restored chain.
  const uint64_t grown = prompt.size();
  // What the ledger answers a claim of in-service work that lacks `pages`
  // Full pages (MemoryLedger::free): it waits for the page the copies ahead
  // free as they spill into the free Index page (Cache::returning), and for
  // no more.
  const auto answer = [&](uint32_t pages) {
    return test::freeFromCache(process.cache, test::kInServiceGrowth, {.pages = pages}).outcome;
  };
  require(!process.cache.ensureTokens(2, grown).granted() && answer(1) == FreeOutcome::Pending,
          "a page the copies ahead return was not waited for");
  require(answer(2) != FreeOutcome::Pending,
          "a claim waited for more pages than the copies ahead free into the free Index pages");
  process.tier.complete();
  require(process.cache.pollTransfers(), "the copies did not land");
  static_cast<void>(process.cache.spillWritten(SpilledExtents::KeepRunway));
  require(process.cache.ensureTokens(2, grown).granted(), "the landed copies did not free a page as it spilled");
  for (const uint32_t page : held.pages)
    process.pool.releasePage(page, false);
  process.cache.endRequest(2);
}

// A leaf the full quota cannot demote goes with its state at once, writing nothing, when only copies ahead that
// return nothing to the reclaim are in flight, which free no slot either: copies of a point the reclaim cannot take,
// or of the blocks above it alone, its request running, its state in use for a reclaim of the ordinary class, or
// leased for any (KvCache::keeps). A start's claim, which keeps the chains of states in use, counts none of their
// pages.
void testRefusedDemotionTakesTheLeafBesideCopiesAhead() {
  enum class Holder : uint8_t { Running, Used, Leased };
  for (const Holder holder : {Holder::Running, Holder::Used, Holder::Leased})
    for (const uint32_t copies : {3u, 2u}) {
      Process process(true);
      const uint64_t leaf = process.cachePrompt(1, promptOf(1000, 1))[0];
      process.publish(leaf);
      require(test::stateStep(process.cache).gave() && process.cache.pollTransfers() &&
                  !process.cache.stateResident(leaf),
              "the leaf's state did not move to the SSD");
      std::vector<uint32_t> prompt = promptOf(2000, 3);
      process.cache.beginRequest(2);
      require(admitTokens(process.cache, 2, prompt.size()).granted(), "the request's KV was not admitted");
      process.cache.publishCommittedBlocks(2, prompt, static_cast<uint32_t>(prompt.size()), {});
      const uint64_t point = process.cache.blockAt(2, static_cast<uint32_t>(prompt.size()));
      process.publish(point);
      std::optional<StateUse> use;
      if (holder != Holder::Leased)
        use.emplace(process.cache.useState(point));
      if (holder != Holder::Running)
        process.cache.endRequest(2);
      prompt.push_back(1);
      const CacheLookup lookup = holder == Holder::Leased ? process.cache.lookup(prompt, {}) : CacheLookup{};
      require(holder != Holder::Leased || (lookup.state && lookup.state->kvBlock() == point),
              "the point's state was not leased");
      // The copies of the chain's first blocks, root first, stay in flight.
      process.tier.writeLimits[static_cast<size_t>(WriteClass::Persist)] = copies;
      require(process.cache.persist(point) == PersistStatus::Started && process.cache.pollTransfers() &&
                  process.tier.inFlight() == copies,
              "the point's chain was not copied ahead");
      process.tier.capacity = process.tier.slots;
      const VictimQuery start = test::query({.cls = ClaimClass::Start}, {.pages = 8});
      uint32_t counted = 0;
      for (std::optional<Victim> unit = process.cache.peek(start, nullptr); unit;
           unit = process.cache.peek(start, &*unit))
        counted += unit->pages;
      require(counted == 1, "a start's claim counted " + std::to_string(counted) +
                                " pages, the leaf's and those of a held chain that copies ahead hold");
      const uint32_t demotions = process.tier.demotions, prefix = process.cache.snapshot().pool.pagesPrefix;
      const FreeResult step =
          test::evictStep(process.cache, holder == Holder::Leased ? ReclaimClass::InUse : ReclaimClass::Ordinary);
      require(step.gave() && process.tier.demotions == demotions &&
                  process.cache.snapshot().pool.pagesPrefix == prefix - 1,
              "a leaf the quota could not demote stayed beside copies ahead that return nothing to the reclaim");
      process.tier.complete();
      static_cast<void>(process.cache.pollTransfers());
      if (holder == Holder::Running)
        process.cache.endRequest(2);
    }
}

// A temporary tier copies a point once its request ended, the points after it first, and a reclaim
// then frees it at once; a persistent tier copies points in use.
void testTemporaryPointsWaitForTheirRequests() {
  constexpr double kDelay = WriteBehind::kDelayMilliseconds;
  for (const bool persistent : {false, true}) {
    Process process(persistent);
    runShortOfMemory(process);
    WriteBehind writes(process.cache);
    // The engine's step before the points: memory demands copies.
    static_cast<void>(writes.run(0.0, kCopiesDemanded, false));
    const std::vector<uint64_t> used = process.cachePrompt(1, promptOf(1000, 2));
    process.publish(used[1]);
    StateUse use = process.cache.useState(used[1]);
    writes.published(used[1], WriteBehind::kMinimumTokens, 0.0);
    const uint64_t unused = process.cachePrompt(2, promptOf(2000, 1))[0];
    process.publish(unused);
    writes.published(unused, WriteBehind::kMinimumTokens, 1.0);
    if (persistent) {
      require(writes.nextWakeup() == kDelay && !writes.run(kDelay, kCopiesDemanded, false) && process.tier.demotions == 3,
              "a persistent tier did not copy a point in use");
      continue;
    }
    require(writes.nextWakeup() == kDelay + 1 && !writes.run(kDelay + 1, kCopiesDemanded, false) &&
                process.tier.demotions == 2 && writes.snapshot().waiting == 2,
            "the point in use was copied, or held back the one after it");
    process.tier.complete();
    require(process.cache.pollTransfers() && writes.run(kDelay + 2, kCopiesDemanded, false) &&
                writes.snapshot().durable == 1 && writes.snapshot().waiting == 1 &&
                !writes.nextWakeup(),
            "the point after the one in use was not copied, or the one in use woke the queue");
    use.reset();
    require(writes.nextWakeup() == kDelay && !writes.run(kDelay + 3, kCopiesDemanded, false) && process.tier.demotions == 4,
            "the point was not copied once its request ended");
    process.tier.complete();
    require(process.cache.pollTransfers() && writes.run(kDelay + 4, kCopiesDemanded, false) &&
                writes.snapshot().durable == 2 && !writes.snapshot().waiting,
            "the point whose request ended did not get its copies");
    bool reclaimed = false;
    while (test::releaseStep(process.cache, ReclaimClass::Ordinary).gave())
      reclaimed = true;
    require(reclaimed && !process.cache.transfersInFlight() && process.tier.demotions == 4 &&
                !process.cache.snapshot().pool.pagesPrefix &&
                !process.cache.stateResident(used[1]) && !process.cache.stateResident(unused),
            "a reclaim wrote the copied points again or waited for a write");

    const uint64_t last = process.cachePrompt(3, promptOf(3000, 1))[0];
    process.publish(last);
    writes.published(last, WriteBehind::kMinimumTokens, kDelay);
    require(writes.flush() && process.tier.demotions == 4 && process.shelf->disk.size() == 3,
            "a temporary tier's clean stop wrote a point");
    process.tier.writableFile = false;
    const uint64_t closed = process.cachePrompt(4, promptOf(4000, 1))[0];
    process.publish(closed);
    writes.published(closed, WriteBehind::kMinimumTokens, kDelay);
    require(writes.snapshot().waiting == 1, "a tier that takes no writes queued a point");
  }
}

// A temporary tier copies ahead only for demand beyond what goes at once or is copying already, and
// not while a claim waits; such a step asks for no wakeup.
void testCopiesFollowDemandNotAWindow() {
  constexpr double kDelay = WriteBehind::kDelayMilliseconds;
  {
    Process process(false);
    runShortOfMemory(process);
    WriteBehind writes(process.cache);
    const uint64_t point = process.cachePrompt(1, promptOf(1000, 2))[1];
    process.publish(point);
    writes.published(point, WriteBehind::kMinimumTokens, 0.0);
    const uint32_t demotions = process.tier.demotions;
    require(!writes.run(kDelay, uint64_t{0}, false) && process.tier.demotions == demotions &&
                writes.snapshot().waiting == 1 && !writes.nextWakeup(),
            "a point was copied ahead after a reclaim, with no demand for copies");
  }
  Process process(false);
  WriteBehind writes(process.cache);
  const uint64_t point = process.cachePrompt(1, promptOf(1000, 2))[1];
  process.publish(point);
  writes.published(point, WriteBehind::kMinimumTokens, 0.0);
  // Another conversation's point, which the test copies ahead itself.
  const uint64_t other = process.cachePrompt(2, promptOf(2000, 1))[0];
  process.publish(other);
  // An extent is a page here: the empty ones but the runway go at once.
  const uint64_t atOnce = uint64_t{process.pool.freePageCount() - 1} * process.pool.bytesPerPage();
  require(!writes.run(kDelay, uint64_t{4 * atOnce}, true) && !process.tier.demotions && !writes.nextWakeup(),
          "a point was copied ahead while a claim waited for memory on its way back, or its step asked to wake");
  require(!writes.run(kDelay, atOnce, false) && !process.tier.demotions && writes.snapshot().waiting == 1 &&
              !writes.nextWakeup(),
          "a point was copied ahead though what goes at once held the demand, or its step asked to wake");
  require(process.cache.persist(other) == PersistStatus::Started && process.tier.demotions == 1,
          "the other point's copies did not start");
  require(!writes.run(kDelay, atOnce + 1, false) && process.tier.demotions == 1 && writes.snapshot().waiting == 1 &&
              !writes.nextWakeup(),
          "a point was copied ahead though the copies in flight held the demand");
  process.tier.complete();
  require(process.cache.pollTransfers() && !writes.run(kDelay, atOnce + 1, false) && process.tier.demotions == 3 &&
              process.shelf->disk.size() == 2,
          "a point was not copied ahead on the memory's demand");
}

// Copy demand counts only cache that goes at once without a write: one byte beyond it copies a
// point.
void testCopyDemandCountsOnlyWhatGoesAtOnce() {
  constexpr double kDelay = WriteBehind::kDelayMilliseconds;
  Process process(false);
  WriteBehind writes(process.cache);
  process.publish(process.cachePrompt(1, promptOf(9000, 2))[0], true);
  static_cast<void>(process.cachePrompt(2, promptOf(5000, 2)));
  const uint64_t point = process.cachePrompt(3, promptOf(1000, 1))[0];
  process.publish(point);
  writes.published(point, WriteBehind::kMinimumTokens, 0.0);
  const ClaimSpec growth{.cls = ClaimClass::Growth, .timing = NeedTiming::Immediate};
  const VictimQuery query = test::query(growth, {.bytes = 1});
  uint64_t atOnce = 0;
  bool later = false;
  for (std::optional<Victim> unit = process.cache.peek(query, nullptr); unit;
       unit = process.cache.peek(query, &*unit)) {
    if (unit->later)
      later = true;
    else
      atOnce += unit->bytes;
  }
  require(atOnce && later, "the walk did not offer what writing the checkpoint frees beside the KV that goes at once");
  require(!writes.run(kDelay, atOnce, false) && !process.tier.demotions && writes.snapshot().waiting == 1,
          "a point was copied ahead though what goes at once held the demand");
  require(!writes.run(kDelay, atOnce + 1, false) && process.tier.demotions == 1,
          "a demand beyond what goes at once copied no point");
}

// A Spill request's written-through copies are labelled and serve its restore point: persisting it
// writes only the state.
void testWriteThroughCopiesAreRestorePoints() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  Process earlier(true, std::make_shared<model::DiskBudget>(1600), CacheSpillPolicy{1, 1});
  // Room for an Index extent beside the eight Full pages.
  earlier.storage.budgetPages = 9;
  earlier.cache.beginRequest(1, KvMode::Spill);
  require(earlier.cache.commitSpill(1, 4) && admitTokens(earlier.cache, 1, prompt.size()).granted(),
          "the Spill request was not admitted");
  earlier.cache.publishCommittedBlocks(1, prompt, static_cast<uint32_t>(prompt.size()), {});
  const uint64_t point = earlier.cache.blockAt(1, static_cast<uint32_t>(prompt.size()));
  static_cast<void>(test::spillRound(earlier.cache, earlier.tier));
  require(earlier.tier.demotions == 4 && earlier.kvLabels().size() == 4 &&
              earlier.cache.spillSnapshot().spilledPages == 2,
          "the written-through copies were not labelled, or the pages past the window did not spill");
  earlier.publish(point);
  earlier.cache.endRequest(1);
  require(earlier.cache.persist(point) == PersistStatus::Started && earlier.cache.pollTransfers() &&
              earlier.cache.persist(point) == PersistStatus::Durable && earlier.tier.demotions == 4,
          "the point did not become durable, or wrote its chain again");
  Process next(true);
  const CacheAdoption adoption = next.adopt(earlier.kvLabels(), earlier.stateLabels());
  require(adoption.blocks == 4 && adoption.states == 1 && adoption.dropped == 0,
          "the next process did not take the written-through point back whole");
}

// Recomputing a point's chain leaves its copies to the point, which stays whole, and they count as
// no KV of the request.
void testRecomputedChainsStayWhole() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  for (const KvMode mode : {KvMode::Resident, KvMode::Spill}) {
    const auto budget = std::make_shared<model::DiskBudget>(1600);
    Process process(true, budget, CacheSpillPolicy{1, 1});
    process.tier.budget = budget;
    // Room for an Index extent beside the eight Full pages.
    process.storage.budgetPages = 9;
    const std::vector<uint64_t> chain = process.cachePrompt(1, prompt);
    process.publish(chain[3]);
    process.writeEverything();
    const auto whole = [&] {
      Process next(true);
      const CacheAdoption adoption = next.adopt(process.kvLabels(), process.stateLabels());
      return adoption.states == 1 && adoption.blocks == 4;
    };
    require(whole() && !process.cache.stateResident(chain[3]) && !process.cache.snapshot().pool.pagesPrefix,
            "the point did not leave RAM for the tier whole");
    process.cache.beginRequest(2, mode);
    require((mode == KvMode::Resident || process.cache.commitSpill(2, 4)) &&
                admitTokens(process.cache, 2, prompt.size()).granted(),
            "the request computing the chain again was not admitted");
    process.cache.publishCommittedBlocks(2, prompt, static_cast<uint32_t>(prompt.size()), {});
    require(process.cache.blockAt(2, static_cast<uint32_t>(prompt.size())) == chain[3] && whole(),
            "computing the chain again gave its copies up");
    const bool spills = mode == KvMode::Spill;
    const auto activeKv = [&](uint64_t held, uint64_t committed) {
      const KvSpillSnapshot spilled = process.cache.spillSnapshot();
      return spilled.heldBytes == held && spilled.committedBytes == committed;
    };
    require(activeKv(0, spills ? 400 : 0),
            "the copies of another command's K/V counted as the request's KV, or its commitment went");
    const uint32_t writes = process.tier.demotions;
    static_cast<void>(test::spillRound(process.cache, process.tier));
    require(whole() && process.tier.demotions == writes + (spills ? 4 : 0) &&
                process.cache.spillSnapshot().spilledPages == (spills ? 2u : 0u) &&
                activeKv(spills ? 400 : 0, 0),
            "a Spill request did not write its own copies before its pages spilled, or another wrote");
    process.cache.endRequest(2);
    process.cache.checkInvariants();
  }
}

// A full quota keeps a RAM state's spilled chain over every other copy, until a committed page
// takes it, the state first.
void testCommittedPagesTakeTheSpilledChainOfAStateInRam() {
  const auto budget = std::make_shared<model::DiskBudget>(400);
  Process process(true, budget, CacheSpillPolicy{1, 1});
  process.tier.budget = budget;
  // Room for an Index extent beside the eight Full pages.
  process.storage.budgetPages = 9;
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  process.cache.beginRequest(1, KvMode::Spill);
  require(process.cache.commitSpill(1, 4) && admitTokens(process.cache, 1, prompt.size()).granted(),
          "the Spill request was not admitted");
  process.cache.publishCommittedBlocks(1, prompt, static_cast<uint32_t>(prompt.size()), {});
  static_cast<void>(test::spillRound(process.cache, process.tier));
  const uint64_t dense = process.cache.blockAt(1, KvCache::pageTokens);
  const uint64_t point = process.cache.blockAt(1, 3 * KvCache::pageTokens);
  process.publish(point);
  process.cache.endRequest(1);
  require(test::evictStep(process.cache, ReclaimClass::Ordinary).gave() &&
              process.tier.slots == 4 && process.cache.stateResident(point),
          "the tail below the replay point did not leave RAM");
  const std::vector<uint32_t> next = promptOf(5000, 3);
  process.cache.beginRequest(2, KvMode::Spill);
  require(process.cache.commitSpill(2, 3) && admitTokens(process.cache, 2, next.size()).granted(),
          "the second Spill request was not admitted");
  process.cache.publishCommittedBlocks(2, next, static_cast<uint32_t>(next.size()), {});
  require(process.cache.writeThrough() == 3 && !process.cache.stateResident(point) &&
              !process.cache.chainCopies(dense).copied && process.tier.slots == 4,
          "a committed page did not take the spilled chain of a state in RAM, or took it first");
  process.tier.complete();
  static_cast<void>(process.cache.pollTransfers());
  process.cache.checkInvariants();
}

// A prefix that keeps another conversation's copies holds back none of the later blocks from
// spilling.
void testAPrefixWithoutOwnCopiesHoldsBackNoSpill() {
  const auto budget = std::make_shared<model::DiskBudget>(100'000);
  test::TestKvStorage storage{256, 100, 4, 10};
  KvPool pool{storage, 4};
  test::TestKvTier tier(true);
  tier.budget = budget;
  tier.transferLimit = 1024;
  const auto shelf = std::make_shared<Shelf>();
  shelf->capacity = 100;
  engine::Cache cache(pool, &tier, budget, CacheSpillPolicy{1, 1});
  require(cache.spillWritten(SpilledExtents::KeepRunway).granted(), "the Index runway was not allocated");
  const auto publish = [&](uint64_t block) {
    cache.publishCompositeState(block, std::make_shared<ShelvedState>(shelf, false), false);
  };
  // The first conversation's four pages leave RAM for their copies.
  const std::vector<uint32_t> prefix = promptOf(1000, 4);
  cache.beginRequest(1);
  require(admitTokens(cache, 1, prefix.size()).granted(), "the first conversation was not admitted");
  cache.publishCommittedBlocks(1, prefix, static_cast<uint32_t>(prefix.size()), {});
  const uint64_t prefixEnd = cache.blockAt(1, static_cast<uint32_t>(prefix.size()));
  cache.endRequest(1);
  publish(prefixEnd);
  for (;;) {
    const FreeResult step = test::evictStep(cache);
    tier.complete();
    const bool landed = cache.pollTransfers();
    if (!step.gave() && !landed)
      break;
  }
  require(!cache.snapshot().pool.pagesPrefix, "the first conversation's prefix stayed in RAM");
  // The second computes the prefix again and eight pages of its own, and its
  // restore point persists.
  std::vector<uint32_t> second = prefix;
  const std::vector<uint32_t> own = promptOf(5000, 8);
  second.insert(second.end(), own.begin(), own.end());
  cache.beginRequest(2);
  require(admitTokens(cache, 2, second.size()).granted(), "the second conversation was not admitted");
  cache.publishCommittedBlocks(2, second, static_cast<uint32_t>(second.size()), {});
  const uint64_t point = cache.blockAt(2, static_cast<uint32_t>(second.size()));
  cache.endRequest(2);
  publish(point);
  PersistStatus status = PersistStatus::Busy;
  for (uint32_t attempt = 0; attempt < 100 && status != PersistStatus::Durable; ++attempt) {
    status = cache.persist(point);
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  require(status == PersistStatus::Durable && cache.chainCopies(point).copied == 8,
          "the restore point did not persist its own blocks alone");
  // Its next turn spills the chain it restores: the pages past the dense
  // history and before the window whose copies are its own, seven, spill at
  // the first scan.
  std::vector<uint32_t> next = second;
  next.push_back(7);
  cache.beginRequest(3, KvMode::Spill);
  const CacheLookup found = cache.lookup(next, {});
  require(found.state && cache.commitSpill(3, 13, cache.chainCopies(found.state->kvBlock())),
          "the next turn's commitment was refused");
  require(cache.restoreRequest(3, found).granted(), "the next turn did not restore the chain");
  require(cache.spillWritten(SpilledExtents::KeepRunway).granted(), "the spill was refused");
  require(std::ranges::count_if(test::indexBlocks(cache, 3), [](uint64_t block) { return block != 0; }) == 7,
          "the prefix waiting for its own copies held back the pages that had theirs");
  cache.checkInvariants();
  cache.endRequest(3);
}

} // namespace

int main() {
  try {
    testCopiesAreLabelled();
    testWindowedCopiesNameTheirForm();
    testWindowedPointsKeepTheirTailsFull();
    testRecomputedTailsKeepFullCopies();
    testRestorePointsComeBack();
    testSaltedPointsComeBackUnderTheirSalt();
    testBrokenChainsStayBehind();
    testOldestPointGoes();
    testAPointAQueuedRequestRestoresStays();
    testOldestPointInRamGoesWithItsChain();
    testPointKeepsItsChainWhileItsStateIsWritten();
    testAdoptionTrimsToTheLimit();
    testLowerLimitKeepsCopiesInUse();
    testCleanStopRecordsRecency();
    testReusedCheckpointsComeBackOrdinary();
    testPointGivenUpForItselfIsRefused();
    testPointLargerThanTheQuotaIsRefused();
    testPersistingStateStaysInRam();
    testDurablePointsStayInRam();
    testOnlyTheNewestPointsAreKept();
    testRestoresGoFirst();
    testCopiesWaitForWritesNeededNow();
    testTemporaryCopiesTakeOnlyFreeRoom();
    testPointsWaitTheirTurn();
    testWritesKeepToTheWriteBudget();
    testPointsWaitForTheTier();
    testRefusedDemotionTakesTheLeafBesideCopiesAhead();
    testCopiesAheadOfSpilledPagesArePending();
    testTemporaryPointsWaitForTheirRequests();
    testCopiesFollowDemandNotAWindow();
    testCopyDemandCountsOnlyWhatGoesAtOnce();
    testWriteThroughCopiesAreRestorePoints();
    testRecomputedChainsStayWhole();
    testCommittedPagesTakeTheSpilledChainOfAStateInRam();
    testAPrefixWithoutOwnCopiesHoldsBackNoSpill();
    std::cout << "Persistent cache tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
