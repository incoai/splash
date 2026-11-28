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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
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
  // The quota its copies share with the KV tier's, if any, as the state
  // file's slots do.
  std::shared_ptr<model::DiskBudget> quota;
};

class ShelvedState final : public CompositeState {
public:
  ShelvedState(std::shared_ptr<Shelf> shelf, bool disk) : shelf_(std::move(shelf)), disk_(disk) {
    if (disk_)
      shelf_->disk.insert(this);
  }
  ~ShelvedState() override {
    if (!disk_)
      return;
    shelf_->disk.erase(this);
    if (shelf_->quota)
      shelf_->quota->release(bytes());
  }
  uint64_t bytes() const noexcept override { return 100; }
  uint64_t residentBytes() const noexcept override { return disk_ ? 0 : bytes(); }
  bool writable() const noexcept override { return !disk_; }
  bool canOffload() const noexcept override {
    return !disk_ && shelf_->writable && (!shelf_->writeBudget || shelf_->writeBudget->admits());
  }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override {
    if (shelf_->disk.size() >= shelf_->capacity || (shelf_->quota && !shelf_->quota->reserve(bytes())))
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
// spill policy (Index pages beside the eight; the tests spill under --kv-spill always, pages moving as
// their copies land) or window policy, whose window unit pages give each copy a window part of 100
// bytes a page beside its base part's 100.
struct Process final {
  test::TestKvStorage storage{16, 100, 1, 10};
  KvPool pool{storage, 8};
  test::TestKvTier tier;
  std::shared_ptr<Shelf> shelf = std::make_shared<Shelf>();
  bool windowed;
  engine::Cache cache;

  explicit Process(bool persistent, std::shared_ptr<model::DiskBudget> budget = nullptr,
                   std::optional<CacheSpillPolicy> spill = std::nullopt,
                   std::optional<CacheWindowPolicy> window = std::nullopt)
      : tier(persistent, window ? 100 * window->unitPages : 0), windowed(window.has_value()),
        cache(pool, &tier, std::move(budget), spill, window) {
    storage.budgetPages = 8;
    tier.capacity = 16;
    tier.transferLimit = 8;
  }
  // Room for a conversation a few strides of write-behind deep
  // (WriteBehind::kMinimumTokens), in memory and on the tier.
  Process(bool persistent, uint32_t pages, std::shared_ptr<model::DiskBudget> budget = nullptr,
          std::optional<CacheWindowPolicy> window = std::nullopt)
      : storage(pages, 100, 1, 10), tier(persistent, window ? 100 * window->unitPages : 0),
        windowed(window.has_value()), cache(pool, &tier, std::move(budget), std::nullopt, window) {
    storage.budgetPages = pages;
    tier.capacity = pages;
    tier.transferLimit = 64;
    shelf->capacity = 64;
  }
  // A windowed one with room for a conversation a few strides of
  // write-behind deep (WriteBehind::kMinimumTokens), its memory `budget`
  // pages.
  Process(uint32_t pages, uint32_t budget, CacheWindowPolicy window)
      : storage(pages, 100, 1, 10), tier(true, 100 * window.unitPages), windowed(true),
        cache(pool, &tier, nullptr, std::nullopt, window) {
    storage.budgetPages = budget;
    tier.capacity = 4 * pages;
    tier.transferLimit = 64;
    shelf->capacity = 64;
  }

  // Caches the prompt through a finished request and returns
  // its blocks, root first.
  // The prompt's blocks, cached by a request that publishes a state at each
  // of `states` (by block index) while it holds them, as a lane holds the
  // window units its states' tails take.
  std::vector<uint64_t> cachePrompt(uint64_t requestId, const std::vector<uint32_t> &prompt,
                                    CacheSalt salt = {}, std::initializer_list<uint32_t> states = {}) {
    cache.beginRequest(requestId);
    require(admitTokens(cache, requestId, prompt.size()).granted(), "prompt KV was not admitted");
    cache.publishCommittedBlocks(requestId, prompt, static_cast<uint32_t>(prompt.size()),
                                 {{}, salt});
    std::vector<uint64_t> blocks;
    for (uint32_t end = KvCache::pageTokens; end <= prompt.size(); end += KvCache::pageTokens)
      blocks.push_back(cache.blockAt(requestId, end));
    for (const uint32_t state : states)
      publish(blocks.at(state), false, requestId);
    cache.endRequest(requestId);
    return blocks;
  }

  void publish(uint64_t block, bool checkpoint = false, std::optional<uint64_t> lane = std::nullopt) {
    cache.publishCompositeState(block, std::make_shared<ShelvedState>(shelf, false), checkpoint, lane);
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
  // reclaims, as a device need takes them.
  void writeEverything() {
    for (;;) {
      const FreeResult step = test::releaseStep(cache);
      tier.complete();
      const bool landed = cache.pollTransfers();
      if (!step.gave() && !landed)
        return;
    }
  }

  // What the next process finds of this one's tier: the labels of its copies' base parts, and of
  // their window parts.
  [[nodiscard]] std::vector<std::vector<std::byte>> kvLabels() const { return tier.labels(); }
  [[nodiscard]] std::vector<std::vector<std::byte>> windowLabels() const {
    return tier.labels(kv::Component::Window);
  }
  [[nodiscard]] std::vector<std::vector<std::byte>> stateLabels() const { return labels(*shelf); }

  // Takes back what an earlier process's tier recorded, its copies' base parts and window parts.
  CacheAdoption adopt(const std::vector<std::vector<std::byte>> &kv,
                      const std::vector<std::vector<std::byte>> &states,
                      const std::vector<std::vector<std::byte>> &windows = {}) {
    std::vector<PersistedKv> blocks;
    for (const kv::Component part : kv::kAllComponents)
      for (const auto &label : part == kv::Component::Base ? kv : windows)
        blocks.push_back({label,
                          [this, part](std::shared_ptr<KvDiskSlot> into) { return tier.adopt(part, std::move(into)); },
                          part});
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

// The labels of a windowed layout's copies, by block: its base part's and its window part's.
std::map<uint64_t, std::pair<std::vector<std::byte>, std::vector<std::byte>>>
partLabels(const std::vector<std::vector<std::byte>> &bases, const std::vector<std::vector<std::byte>> &windows) {
  std::map<uint64_t, std::pair<std::vector<std::byte>, std::vector<std::byte>>> result;
  for (const auto &bytes : bases)
    result[decodeKvSlotLabel(bytes, true)->block.id].first = bytes;
  for (const auto &bytes : windows)
    result[decodeKvSlotLabel(bytes, true)->block.id].second = bytes;
  return result;
}

// A windowed layout's copy is a base part, and a window part where its block held its unit, each labelled with
// its block and the copy's serial, which no other copy has. Two conversations part after their second block:
// the state there and each one's newest are what a reclaim writes, and every block is in a tail, so every copy
// has both parts. The next process pairs a block's parts by its id, identity and serial whatever else their
// labels hold, so a crash between the updates of two labels loses nothing. A crash between the writes of the
// two parts, at either step, leaves block 2 a base part alone, which a tail's resume cannot read, or a window
// part alone, no copy: the next process takes back neither the state whose tail holds it nor what lies below
// it; neither does it pair parts of two copies.
void testWindowedCopiesAreTwoParts() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  std::vector<uint32_t> other = prompt;
  std::iota(other.begin() + 2 * KvCache::pageTokens, other.end(), 5000);
  constexpr CacheWindowPolicy window{2, 1};
  Process earlier(true, nullptr, std::nullopt, window);
  earlier.storage.budgetPages = 16;
  const std::vector<uint64_t> blocks = earlier.cachePrompt(1, prompt, {}, {1, 3});
  const std::vector<uint64_t> branch = earlier.cachePrompt(2, other, {}, {3});
  earlier.writeEverything();
  const auto labels = partLabels(earlier.kvLabels(), earlier.windowLabels());
  require(branch[1] == blocks[1] && labels.size() == 6 && earlier.cache.snapshot().kvTier.diskBytes == 6 * 200,
          "a windowed layout's copies are not six blocks of two parts");
  std::set<uint64_t> serials;
  for (const auto &[id, parts] : labels) {
    const auto base = decodeKvSlotLabel(parts.first, true), part = decodeKvSlotLabel(parts.second, true);
    require(parts.first.size() == kvSlotLabelBytes(true) && kvSlotLabelBytes(true) == sizeof(KvBlockLabel) + 8 &&
                base && part && base->block.id == id && part->block.id == id && base->copy == part->copy &&
                serials.insert(base->copy).second,
            "a windowed layout's parts are not labelled with their block and their copy's own serial");
  }
  const auto adopt = [&](const std::function<void(std::vector<std::vector<std::byte>> &,
                                                  std::vector<std::vector<std::byte>> &)> &crash) {
    std::vector<std::vector<std::byte>> bases, windows;
    for (const auto &[id, parts] : labels) {
      bases.push_back(parts.first);
      windows.push_back(parts.second);
    }
    crash(bases, windows);
    Process next(true, nullptr, std::nullopt, window);
    next.storage.budgetPages = 16;
    return next.adopt(bases, earlier.stateLabels(), windows);
  };
  // The label of block 2's part, in either file.
  const auto at = [&](std::vector<std::vector<std::byte>> &labelled) {
    return std::ranges::find_if(labelled, [&](const auto &bytes) {
      return decodeKvSlotLabel(bytes, true)->block.id == blocks[2];
    });
  };
  CacheAdoption whole = adopt([](auto &, auto &) {});
  require(whole.blocks == 6 && whole.states == 3 && whole.dropped == 0 && whole.bytes == 6 * 200 + 3 * 100,
          "the next process did not take back a windowed layout's restore points with both parts");
  // Two label updates, the second lost: the parts' recencies differ.
  CacheAdoption relabelled = adopt([&](auto &, auto &windows) {
    auto label = decodeKvSlotLabel(*at(windows), true);
    label->block.lastUsed += 1000;
    *at(windows) = encodeKvSlotLabel(*label, true);
  });
  require(relabelled.blocks == 6 && relabelled.states == 3 && relabelled.dropped == 0,
          "a crash between the updates of two labels of a copy lost it");
  // The crash steps of block 2's copy: nothing landed, the base part alone, the window part alone; and two
  // copies' parts, which share no serial.
  const CacheAdoption none = adopt([&](auto &bases, auto &windows) {
    bases.erase(at(bases));
    windows.erase(at(windows));
  });
  const CacheAdoption baseAlone = adopt([&](auto &, auto &windows) { windows.erase(at(windows)); });
  const CacheAdoption windowAlone = adopt([&](auto &bases, auto &) { bases.erase(at(bases)); });
  const CacheAdoption crossed = adopt([&](auto &, auto &windows) {
    auto label = decodeKvSlotLabel(*at(windows), true);
    label->copy += 1000;
    *at(windows) = encodeKvSlotLabel(*label, true);
  });
  for (const auto &[name, adoption, dropped] : {std::tuple{"no part landed", none, 2u},
                                               std::tuple{"the base part alone landed", baseAlone, 3u},
                                               std::tuple{"the window part alone landed", windowAlone, 3u},
                                               std::tuple{"two copies' parts", crossed, 4u}})
    require(adoption.blocks == 4 && adoption.states == 2 && adoption.dropped == dropped &&
                adoption.bytes == 4 * 200 + 2 * 100,
            std::string("a crash with ") + name + " took back a copy of the block, or what lies below it");

  // New copies continue after every serial the earlier process recorded.
  Process next(true, nullptr, std::nullopt, window);
  next.storage.budgetPages = 16;
  static_cast<void>(next.adopt(earlier.kvLabels(), earlier.stateLabels(), earlier.windowLabels()));
  const std::vector<uint64_t> later = next.cachePrompt(3, promptOf(7000, 1), {}, {0});
  next.writeEverything();
  const auto copies = partLabels(next.kvLabels(), next.windowLabels());
  require(copies.contains(later[0]) && decodeKvSlotLabel(copies.at(later[0]).first, true)->copy > *serials.rbegin(),
          "a later process's copy took a serial an earlier one recorded");
}

// A copy holds what its block holds, and gains a part as its block gains a unit: a conversation of four blocks
// makes its state at the end durable, base parts for every block and window parts for its two-block tail; a
// state at its second block then gives the first two blocks their units, and making it durable adds their window
// parts to the copies whose base parts landed, under each copy's serial. The next process pairs every block's
// two parts and takes both points back.
void testAWindowPartAddedLaterJoinsItsCopy() {
  constexpr CacheWindowPolicy window{2, 1};
  Process earlier(true, nullptr, std::nullopt, window);
  earlier.storage.budgetPages = 16;
  const std::vector<uint32_t> prompt = promptOf(5000, 4);
  earlier.cache.beginRequest(1);
  require(admitTokens(earlier.cache, 1, prompt.size()).granted(), "fixture: the prompt was refused");
  earlier.cache.publishCommittedBlocks(1, prompt, static_cast<uint32_t>(prompt.size()), {});
  std::vector<uint64_t> blocks;
  for (uint32_t end = KvCache::pageTokens; end <= prompt.size(); end += KvCache::pageTokens)
    blocks.push_back(earlier.cache.blockAt(1, end));
  earlier.publish(blocks[3], false, 1);
  require(earlier.persist(blocks[3]) == PersistStatus::Durable && earlier.kvLabels().size() == 4 &&
              earlier.windowLabels().size() == 2,
          "fixture: the first point did not land with base parts and its tail's window parts");
  earlier.publish(blocks[1], false, 1);
  require(earlier.persist(blocks[1]) == PersistStatus::Durable && earlier.windowLabels().size() == 4 &&
              std::ranges::count_if(earlier.tier.writes, [](const auto &write) { return write.parts == 2; }) == 2,
          "the second point did not add the window parts its blocks' units gained to their copies");
  earlier.cache.endRequest(1);
  const auto labels = partLabels(earlier.kvLabels(), earlier.windowLabels());
  require(labels.size() == 4 && std::ranges::all_of(labels, [](const auto &each) {
            return decodeKvSlotLabel(each.second.first, true)->copy == decodeKvSlotLabel(each.second.second, true)->copy;
          }),
          "a window part added later did not take its copy's serial");
  Process next(true, nullptr, std::nullopt, window);
  next.storage.budgetPages = 16;
  const CacheAdoption adoption = next.adopt(earlier.kvLabels(), earlier.stateLabels(), earlier.windowLabels());
  require(adoption.blocks == 4 && adoption.states == 2 && adoption.dropped == 0,
          "the next process did not pair a window part added later with its copy's base part");
}

// A windowed restore point copies what each block holds, the base part of each and the window part of the
// two blocks of its tail, which hold their units, and a restore reads back what each block then holds: the
// window part of the tail alone; a tail block whose window part was lost is left behind with the restore
// point it breaks.
void testWindowedPointsComeBackWhole() {
  const std::vector<uint32_t> prompt = promptOf(3000, 6);
  constexpr CacheWindowPolicy window{2, 1};
  Process earlier(true, nullptr, std::nullopt, window);
  earlier.storage.budgetPages = 16;
  const std::vector<uint64_t> blocks = earlier.cachePrompt(1, prompt, {}, {5});
  const PersistStatus status = earlier.persist(blocks[5]);
  std::vector<std::vector<std::byte>> copies = earlier.kvLabels(), windows = earlier.windowLabels();
  require(status == PersistStatus::Durable && copies.size() == 6 && windows.size() == 2 &&
              std::ranges::count_if(earlier.tier.writes, [](const auto &write) { return write.parts == 3; }) == 2 &&
              std::ranges::count_if(earlier.tier.writes, [](const auto &write) { return write.parts == 1; }) == 4,
          "the restore point did not become durable with a base part of each copy and its tail's window parts");
  {
    Process next(true, nullptr, std::nullopt, window);
    next.storage.budgetPages = 16;
    const CacheAdoption adoption = next.adopt(copies, earlier.stateLabels(), windows);
    require(adoption.blocks == 6 && adoption.states == 1 && adoption.dropped == 0,
            "the next process did not take back the restore point");
    std::vector<uint32_t> resumed = prompt;
    resumed.push_back(1);
    next.cache.beginRequest(1);
    const CacheLookup lookup = next.cache.lookup(resumed, {});
    require(lookup.resumeBoundary() == 6 * KvCache::pageTokens && next.cache.restoreRequest(1, lookup).granted() &&
                next.cache.pageTable(1).pages.size() == 6 && next.tier.restores == 6 &&
                std::ranges::count(next.tier.restoredParts, kv::Components{3}) == 2 &&
                std::ranges::count(next.tier.restoredParts, kv::Components{1}) == 4,
            "the restore point did not restore the base part of each block and its tail's window parts");
    next.tier.complete();
    static_cast<void>(next.cache.pollTransfers());
    static_cast<void>(next.cache.cutPageTable(1));
    next.cache.checkInvariants();
    next.cache.endRequest(1);
  }
  std::erase_if(windows, [&](const auto &bytes) { return decodeKvSlotLabel(bytes, true)->block.id == blocks[4]; });
  Process next(true, nullptr, std::nullopt, window);
  next.storage.budgetPages = 16;
  const CacheAdoption adoption = next.adopt(copies, earlier.stateLabels(), windows);
  require(adoption.blocks == 0 && adoption.states == 0 && adoption.dropped == 7,
          "a restore point whose chain holds a block without its window part was taken back");
}

// A crash at any step of copying a point ahead leaves the durable point before it or the new one, whole: the next
// process takes back a state only with every base part of its chain and every window part of its tail that landed,
// and gives up the window parts no point it took back needs. Points at blocks 3 and 7 of eight, and between them a
// point at block 5 that was durable and then given up, its tail's window parts left behind; the point at 7 is copied
// a write at a time, and the process crashes after each landing.
void testACrashWhileAPointIsCopiedLeavesAPointWhole() {
  constexpr CacheWindowPolicy window{2, 1};
  const std::vector<uint32_t> prompt = promptOf(3300, 8);
  std::string failures;
  for (uint32_t crash = 0;; ++crash) {
    Process earlier(true, nullptr, std::nullopt, window);
    earlier.storage.budgetPages = 32;
    earlier.cache.beginRequest(1);
    require(admitTokens(earlier.cache, 1, prompt.size()).granted(), "fixture: the prompt was refused");
    earlier.cache.publishCommittedBlocks(1, prompt, static_cast<uint32_t>(prompt.size()), {});
    std::vector<uint64_t> blocks;
    for (uint32_t end = KvCache::pageTokens; end <= prompt.size(); end += KvCache::pageTokens)
      blocks.push_back(earlier.cache.blockAt(1, end));
    for (const uint32_t at : {3u, 5u}) {
      earlier.publish(blocks[at], false, 1);
      require(earlier.persist(blocks[at]) == PersistStatus::Durable, "fixture: a point was not made durable");
    }
    earlier.cache.invalidateState(blocks[5]);
    earlier.publish(blocks[7], false, 1);
    earlier.cache.endRequest(1);
    // The deeper point's copy, a write at a time, up to `crash` landings.
    earlier.tier.transferLimit = 1;
    earlier.shelf->ready = false;
    PersistStatus status = earlier.cache.persist(blocks[7]);
    bool stateLanded = false;
    for (uint32_t landing = 0; landing < crash && status != PersistStatus::Durable; ++landing) {
      if (!stateLanded) {
        earlier.shelf->ready = stateLanded = true;
      } else {
        for (const auto &transfer : earlier.tier.transfers)
          if (!transfer->finished && !transfer->ready) {
            transfer->ready = true;
            break;
          }
      }
      static_cast<void>(earlier.cache.pollTransfers());
      status = earlier.cache.persist(blocks[7]);
    }
    const bool whole = status == PersistStatus::Durable;
    std::vector<std::vector<std::byte>> states = earlier.stateLabels();
    if (!stateLanded)
      std::erase_if(states, [&](const auto &bytes) { return decodeLabel<StateLabel>(bytes)->block == blocks[7]; });
    Process next(true, nullptr, std::nullopt, window);
    next.storage.budgetPages = 32;
    const CacheAdoption adoption = next.adopt(earlier.tier.landedLabels(), states,
                                              earlier.tier.landedLabels(kv::Component::Window));
    std::vector<uint32_t> resumed = prompt;
    resumed.push_back(1);
    next.cache.beginRequest(1);
    const CacheLookup lookup = next.cache.lookup(resumed, {});
    const uint32_t expected = (whole ? 8 : 4) * KvCache::pageTokens;
    // The deepest point's chain, and the window parts of the points' tails.
    const uint64_t bytes = (whole ? 8 : 4) * 100 + (whole ? 4 : 2) * 100;
    if (lookup.resumeBoundary() != expected || next.cache.snapshot().kvTier.diskBytes != bytes ||
        !next.cache.restoreRequest(1, lookup).granted())
      failures += "crash " + std::to_string(crash) + ": resumed " + std::to_string(lookup.resumeBoundary()) +
                  " tokens of " + std::to_string(expected) + ", " +
                  std::to_string(next.cache.snapshot().kvTier.diskBytes) + " bytes kept of " +
                  std::to_string(bytes) + " (" + std::to_string(adoption.states) + " states); ";
    next.tier.complete();
    static_cast<void>(next.cache.pollTransfers());
    next.cache.checkInvariants();
    next.cache.endRequest(1);
    for (; earlier.tier.inFlight(); static_cast<void>(earlier.cache.pollTransfers()))
      earlier.tier.complete();
    if (whole)
      break;
    require(crash < 32, "the point's copy did not end");
  }
  require(failures.empty(), failures);
}

// Blocks recomputed where only their copies are cached take the writer's pages and get copies of
// them, so a state published there restores from what its writer computed, in this process and the
// next.
void testRecomputedBlocksGetCopiesOfTheirPages() {
  for (const bool persistent : {false, true}) {
    const std::vector<uint32_t> earlier = promptOf(5000, 6);
    constexpr CacheWindowPolicy window{2};
    Process process(persistent, nullptr, std::nullopt, window);
    process.storage.budgetPages = 16;
    const std::vector<uint64_t> blocks = process.cachePrompt(1, earlier);
    process.publish(blocks[5]);
    process.writeEverything();

    // A later prompt shares the first four blocks, and no state there: its
    // request computes them again.
    std::vector<uint32_t> later(earlier.begin(), earlier.begin() + 4 * KvCache::pageTokens);
    for (uint32_t row = 0; row < 2 * KvCache::pageTokens; ++row)
      later.push_back(90000 + row);
    const CacheLookup none = process.cache.lookup(later, {});
    require(!none.state && none.kvBoundary == 4 * KvCache::pageTokens, "the shared blocks were not cached alone");
    process.cache.beginRequest(2);
    require(admitTokens(process.cache, 2, later.size()).granted(), "the later prompt's KV was not admitted");
    process.cache.publishCommittedBlocks(2, later, static_cast<uint32_t>(later.size()), {});
    require(process.cache.blockAt(2, 4 * KvCache::pageTokens) == blocks[3],
            "the writer's blocks were not the cached ones");
    process.publish(blocks[3]);
    // The later conversation's own state: with one on each branch below it, the state where the
    // two part is no history of either, and a reclaim writes it.
    process.publish(process.cache.blockAt(2, 6 * KvCache::pageTokens));
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
    require(process.cache.pageTable(3).pages.size() == 4, "a restore did not find its chain");
    static_cast<void>(process.cache.cutPageTable(3));
    process.cache.checkInvariants();
    process.cache.endRequest(3);
    if (!persistent)
      continue;
    Process next(true, nullptr, std::nullopt, window);
    next.storage.budgetPages = 16;
    const CacheAdoption adoption = next.adopt(process.kvLabels(), process.stateLabels());
    require(adoption.states == 3 && adoption.dropped == 0,
            "the next process did not take back the three restore points");
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

// A copy ahead takes no resident conversation's point: on a full quota a new point is refused, the
// points on the SSD stay durable and whole, and a crash takes them back. Each stride of a new point
// then asks again and is refused again, writing nothing, where giving the oldest point up had its
// conversation's next stride write its whole chain again, and so on in turn.
void testAFullQuotaRefusesANewPointRatherThanRotate() {
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
  const std::vector<uint64_t> newest = process.cachePrompt(3, promptOf(3000, 2));
  process.publish(newest[1]);
  const uint32_t writes = process.tier.demotions;
  for (int stride = 0; stride < 4; ++stride)
    require(keep(newest[1]) == PersistStatus::Refused && keep(oldest[2]) == PersistStatus::Durable &&
                keep(older) == PersistStatus::Durable,
            "a new point took a resident conversation's point's room, or a point on the SSD lost it");
  require(process.tier.demotions == writes && process.tier.slots == 6 && process.shelf->disk.size() == 2,
          "the refused point's strides wrote " + std::to_string(process.tier.demotions - writes) + " pages");
  process.cache.checkInvariants();
  // A crash now takes back the two points whole, and nothing of the refused one.
  Process next(true);
  const CacheAdoption adoption = next.adopt(process.kvLabels(), process.stateLabels());
  require(adoption.states == 2 && adoption.blocks == 6 && adoption.dropped == 0,
          "the next process did not take the two durable points back whole");
  next.cache.checkInvariants();
  process.cache.checkInvariants();
}

// A copy ahead never takes the room of an only copy: a point the state file has no room for is
// refused, and so is one whose room only a prefix memory no longer holds could give. A point's chain
// is copied before its state, so a refused point leaves only copies of what memory holds, in room
// that was free, and no state copy; its next attempt goes on from them.
void testPointKeepsItsChainWhileItsStateIsWritten() {
  Process process(true);
  process.shelf->capacity = 0;
  const auto keep = [&](uint64_t point) {
    const PersistStatus status = process.cache.persist(point);
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
    return status;
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
  // The deeper point, on disk alone, holds the only copies of its prefix: a copy ahead, whose data
  // is resident, takes no room from it, here or on the point's own chain, and the tier keeps it.
  const uint64_t other = process.cachePrompt(2, promptOf(2000, 1))[0];
  process.publish(other);
  process.publish(chain[1]);
  const uint32_t writes = process.tier.demotions;
  require(keep(other) == PersistStatus::Refused && keep(chain[1]) == PersistStatus::Refused &&
              process.tier.demotions == writes + 1 && process.shelf->disk.size() == 1 && process.tier.slots == 5,
          "a copy ahead took the room of the only copies of a prefix memory no longer holds");
  require(keep(other) == PersistStatus::Refused && process.tier.demotions == writes + 1,
          "a refused point's next attempt wrote its chain again");
  process.cache.checkInvariants();
  Process next(true);
  const CacheAdoption adoption = next.adopt(process.kvLabels(), process.stateLabels());
  require(adoption.states == 1 && adoption.blocks == 4 && adoption.dropped == 1,
          "the next process did not take the deeper point back whole, or kept the refused point's copy");
  next.cache.checkInvariants();
}

// A point whose room only its own state could give is refused and keeps that state: a copy ahead
// takes no only copy, its own included, and its chain is not touched.
void testPointWithoutRoomIsRefused() {
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
  require(process.cache.persist(blocks[3]) == PersistStatus::Refused && process.shelf->disk.size() == 1,
          "a point that had no room was not refused, or gave its own state up");
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
  process.cache.closePersistentTier();
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
    kept.push_back(file.acquire(payload.size()));
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
    blocks.push_back({label, [&](std::shared_ptr<KvDiskSlot> into) { return next.tier.adopt(kv::Component::Base, std::move(into)); }});
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

// No gone point or checkpoint is copied. A temporary tier copies only the newest points: none a
// conversation went on from (to a point a request went on from in turn: a finished request's last
// point alone supersedes nothing, so the point before it stays a point too); one two conversations
// share, or only a checkpoint follows, is. A persistent tier copies a point its conversation went
// on from, a prefix of the next one's.
void testOnlyTheNewestPointsAreKept() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  for (const bool persistent : {true, false}) {
    Process process(persistent);
    const std::vector<uint64_t> blocks = process.cachePrompt(1, prompt);
    process.publish(blocks[0]);
    process.publish(blocks[1]);
    process.publish(blocks[2], true);
    process.publish(blocks[3]);
    require(process.cache.persist(blocks[2]) == PersistStatus::Unneeded &&
                process.cache.persist(blocks[3] + 100) == PersistStatus::Unneeded && !process.tier.demotions,
            "a checkpoint or a gone point was written");
    if (persistent) {
      require(process.cache.persist(blocks[0]) == PersistStatus::Started,
              "a persistent tier gave up a point its conversation went on from");
      continue;
    }
    require(process.cache.persist(blocks[0]) == PersistStatus::Unneeded && !process.tier.demotions,
            "a temporary tier wrote a point its conversation went on from");
    std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + KvCache::pageTokens);
    const std::vector<uint32_t> own = promptOf(5000, 3);
    branch.insert(branch.end(), own.begin(), own.end());
    const std::vector<uint64_t> other = process.cachePrompt(2, branch);
    process.publish(other[2]);
    process.publish(other[3]);
    require(process.cache.persist(blocks[0]) == PersistStatus::Started,
            "a point two conversations went on from was not kept");

    Process followed(persistent);
    const std::vector<uint64_t> chain = followed.cachePrompt(1, prompt);
    followed.publish(chain[1]);
    followed.publish(chain[3], true);
    require(followed.cache.persist(chain[1]) == PersistStatus::Started,
            "a point only a checkpoint follows was not kept");
  }
}

// /status attributes what the tier wrote to its class, counts what the quota gave up by kind, and counts
// a page written again after its copy was given up while it stayed in RAM: a conversation read back
// keeps its copies until the limit falls, and its pages are written again when they leave.
void testWritesAreCountedByClass() {
  auto budget = std::make_shared<model::DiskBudget>(uint64_t{1} << 20);
  Process process(false, budget);
  process.tier.budget = budget;
  const std::vector<uint32_t> prompt = promptOf(1000, 2);
  process.publish(process.cachePrompt(1, prompt)[1]);
  process.writeEverything();
  KvTierSnapshot tier = process.cache.snapshot().kvTier;
  require(tier.demotions == 2 && tier.demotedBytes == 200 && !tier.copiedBytes && !tier.rewrites &&
              process.cache.snapshot().stateCache.offloadBytes == 100,
          "the conversation's writes were not counted by class");
  std::vector<uint32_t> asked = prompt;
  asked.push_back(7);
  process.cache.beginRequest(2);
  require(process.cache.restoreRequest(2, process.cache.lookup(asked, {})).granted(), "the conversation was not read back");
  process.tier.complete();
  static_cast<void>(process.cache.pollTransfers());
  process.cache.endRequest(2);
  budget->setLimitBytes(100);
  process.cache.keepDiskLimit();
  // What the lower limit gave up, by kind (window parts, KV copies, state copies, KV leaves, states, states
  // in use): the copy of the page written again below.
  require(process.cache.snapshot().kvTier.quotaGivenUp == std::array<uint64_t, 5>{0, 1, 0, 0, 0},
          "the copy the quota gave up was not counted by its kind");
  budget->setLimitBytes(budget->capacityBytes());
  process.writeEverything();
  tier = process.cache.snapshot().kvTier;
  require(tier.demotions == 3 && tier.rewrites == 1 && tier.rewrittenBytes == 100 && tier.demotedBytes == 300,
          "a page written again after its copy was given up was not counted (" + std::to_string(tier.rewrites) + ")");
}

// While the write budget is spent, two conversations that take turns, each needing a page the other
// holds, resume each from the state nearest below the tip it lost: a page need takes a leaf and the
// state on it, and nothing that frees no page (a history state of a dense layout holds no tail).
// Resident blocks form a prefix subtree at the root, so the other's copied interior pages cannot
// leave before its uncopied tip: resuming at the newest state itself (192 at turn 5) is out of any
// walk's reach, the state below it (160) is the best a walk can keep.
void testASpentBudgetKeepsTheNewestStates() {
  test::WriteBudget budget;
  Process process(false, budget.budget);
  process.shelf->writeBudget = budget.budget;
  process.storage.budgetPages = 12;
  process.tier.capacity = 32;
  process.shelf->capacity = 32;
  // One turn: admitted the way the engine admits it, the ledger freeing
  // what it lacks; returns the tokens it resumed from.
  const auto take = [&](uint64_t id, const std::vector<uint32_t> &prompt) {
    std::vector<uint32_t> asked = prompt;
    asked.push_back(7);
    process.cache.beginRequest(id);
    uint32_t resumed = 0;
    for (int attempt = 0; attempt < 8; ++attempt) {
      const CacheLookup lookup = process.cache.lookup(asked, {});
      resumed = lookup.resumeBoundary();
      if (!lookup.state)
        break;
      const TokenAdmission admission = process.cache.restoreRequest(id, lookup);
      if (admission.granted())
        break;
      const uint32_t lacking =
          admission.additionalPages - std::min(admission.additionalPages, admission.availablePages);
      static_cast<void>(test::freeFromCache(process.cache, test::kInServiceGrowth, {.pages = lacking, .form = admission.form}));
      process.tier.complete();
      static_cast<void>(process.cache.pollTransfers());
    }
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
    for (int attempt = 0; attempt < 8; ++attempt) {
      const TokenAdmission admission = admitTokens(process.cache, id, prompt.size());
      if (admission.granted())
        break;
      const uint32_t lacking =
          admission.additionalPages - std::min(admission.additionalPages, admission.availablePages);
      static_cast<void>(test::freeFromCache(process.cache, test::kInServiceGrowth, {.pages = lacking, .form = admission.form}));
      process.tier.complete();
      static_cast<void>(process.cache.pollTransfers());
    }
    process.cache.publishCommittedBlocks(id, prompt, static_cast<uint32_t>(prompt.size()), {});
    const uint64_t block = process.cache.blockAt(id, static_cast<uint32_t>(prompt.size()));
    process.cache.endRequest(id);
    if (!process.cache.stateResident(block))
      process.publish(block);
    return resumed;
  };
  // Two conversations of four pages each, written to the SSD while the
  // budget had room; then the budget is spent.
  std::vector<uint32_t> first = promptOf(1000, 4), second = promptOf(5000, 4);
  static_cast<void>(take(1, first));
  static_cast<void>(take(2, second));
  process.writeEverything();
  budget.spend();
  uint64_t id = 10;
  for (uint32_t turn = 0; turn < 6; ++turn) {
    std::vector<uint32_t> &prompt = turn % 2 ? second : first;
    const uint32_t before = static_cast<uint32_t>(prompt.size());
    const std::vector<uint32_t> more = promptOf(9000 + 100 * turn, 1);
    prompt.insert(prompt.end(), more.begin(), more.end());
    const uint32_t resumed = take(++id, prompt);
    require(turn < 2 || resumed + KvCache::pageTokens >= before,
            "turn " + std::to_string(turn) + " resumed at token " + std::to_string(resumed) +
                ", not at the state nearest below its lost tip (" + std::to_string(before - KvCache::pageTokens) + ")");
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
    Process process(persistent, budget, CacheSpillPolicy{1, 1, true});
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
  Process process(false, budget, CacheSpillPolicy{1, 1, true});
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

// Pages of a stride of write-behind (WriteBehind::kMinimumTokens), and of half of one, a test turn.
constexpr uint32_t kStridePages = WriteBehind::kMinimumTokens / KvCache::pageTokens;
constexpr uint32_t kTurnPages = kStridePages / 2;

// One turn of a conversation: it resumes from the deepest cached state of its prompt, publishes
// its blocks and ends, and the state at its last block is published, as the engine publishes the
// replay point. Returns that block.
uint64_t converse(Process &process, uint64_t requestId, const std::vector<uint32_t> &prompt) {
  std::vector<uint32_t> asked = prompt;
  asked.push_back(7);
  process.cache.beginRequest(requestId);
  if (CacheLookup lookup = process.cache.lookup(asked, {}); lookup.state) {
    require(process.cache.restoreRequest(requestId, lookup).granted(), "the turn did not restore its prefix");
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
  }
  require(admitTokens(process.cache, requestId, prompt.size()).granted(), "the turn's KV was not admitted");
  process.cache.publishCommittedBlocks(requestId, prompt, static_cast<uint32_t>(prompt.size()), {});
  const uint64_t block = process.cache.blockAt(requestId, static_cast<uint32_t>(prompt.size()));
  process.cache.endRequest(requestId);
  if (!process.cache.stateResident(block))
    process.publish(block);
  return block;
}

// The conversation's next turn, half a stride, published to write-behind at `now`.
uint64_t nextTurn(Process &process, WriteBehind &writes, std::vector<uint32_t> &prompt, uint64_t id, double now) {
  const std::vector<uint32_t> more = promptOf(1000 * static_cast<uint32_t>(id), kTurnPages);
  prompt.insert(prompt.end(), more.begin(), more.end());
  const uint64_t block = converse(process, id, prompt);
  writes.published(block, static_cast<uint32_t>(prompt.size()), now);
  return block;
}

// Write-behind's steps at `now`, each write landing, until nothing moves.
void land(Process &process, WriteBehind &writes, double now) {
  for (int step = 0; step < 256; ++step) {
    const bool settled = writes.run(now, kCopiesDemanded, false);
    process.tier.complete();
    if (!process.cache.pollTransfers() && !settled)
      return;
  }
}

// Where a next process resumes the conversation from what this one left on the tier.
uint32_t resumedAfterRestart(const Process &process, std::vector<uint32_t> prompt) {
  Process next(true, 16 * kStridePages);
  static_cast<void>(next.adopt(process.kvLabels(), process.stateLabels()));
  prompt.push_back(7);
  return next.cache.lookup(prompt, {}).resumeBoundary();
}

// A persistent tier copies by progress, not by quiet: a point falls due at once when it is a
// stride deeper than its chain's deepest point on the SSD or being copied, a short one never, and
// its copy goes on though its conversation does; a point not due waits for a clean stop.
void testPointsFallDueByProgress() {
  Process process(true, 8 * kStridePages);
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompt;
  static_cast<void>(nextTurn(process, writes, prompt, 1, 0.0));
  require(!writes.snapshot().waiting, "a point shorter than a stride was queued");
  static_cast<void>(nextTurn(process, writes, prompt, 2, 0.0));
  require(!writes.run(0.0, kCopiesDemanded, false) && process.tier.demotions,
          "a point a stride deep did not fall due at once");
  // The conversation goes on before the copy lands: the copy goes on, and
  // the next point, half a stride past the one being copied, is not due.
  static_cast<void>(nextTurn(process, writes, prompt, 3, 0.0));
  land(process, writes, 0.0);
  require(writes.snapshot().durable == 1 && process.tier.demotions == kStridePages && writes.snapshot().waiting == 1 &&
              !writes.nextWakeup(),
          "a point its conversation went on from was given up, or one not due was copied");
  // A stride past the durable point: due. The one not due before it stays a point while no request went on from
  // the new one (it may be a side question's), and goes once the conversation goes on.
  static_cast<void>(nextTurn(process, writes, prompt, 4, 0.0));
  land(process, writes, 0.0);
  require(writes.snapshot().durable == 2 && writes.snapshot().unneeded == 0 &&
              process.tier.demotions == 2 * kStridePages && writes.snapshot().waiting == 1,
          "a point a stride past the durable one did not fall due, or copied what was copied (" +
              std::to_string(writes.snapshot().durable) + " durable, " + std::to_string(writes.snapshot().unneeded) +
              " unneeded, " + std::to_string(writes.snapshot().waiting) + " waiting)");
  // A clean stop copies the newest point not due.
  static_cast<void>(nextTurn(process, writes, prompt, 5, 0.0));
  land(process, writes, 0.0);
  require(process.tier.demotions == 2 * kStridePages && writes.snapshot().waiting == 1 &&
              writes.snapshot().unneeded == 1,
          "a point half a stride past the durable one was copied, or the one the conversation went on past stayed (" +
              std::to_string(writes.snapshot().unneeded) + " unneeded, " + std::to_string(writes.snapshot().waiting) +
              " waiting)");
  for (int step = 0; step < 64 && !writes.flush(); ++step) {
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
  }
  require(writes.snapshot().durable == 3 && !writes.snapshot().waiting &&
              resumedAfterRestart(process, prompt) == prompt.size(),
          "the clean stop did not leave the newest point durable");
}

// A crash keeps a conversation's durable point within a stride of its newest however quick its
// turns: eight turns of half a stride 2 s apart, the process killed right after the last, 15 s
// after it, or stopped cleanly, and with turns 12 s apart.
void testACrashLosesAtMostAStride() {
  struct Case final {
    double gap;
    double idle;
    bool clean;
  };
  for (const Case shape : {Case{2'000.0, 0.0, false}, Case{2'000.0, 15'000.0, false}, Case{2'000.0, 0.0, true},
                           Case{12'000.0, 0.0, false}}) {
    constexpr uint32_t kTurns = 8;
    Process process(true, 2 * kTurns * kTurnPages);
    WriteBehind writes(process.cache);
    std::vector<uint32_t> prompt;
    double now = 0.0;
    for (uint32_t turn = 1; turn <= kTurns; ++turn) {
      static_cast<void>(nextTurn(process, writes, prompt, turn, now));
      for (const double end = now + (turn < kTurns ? shape.gap : shape.idle); now < end;)
        land(process, writes, now = std::min(end, now + 1'000.0));
    }
    for (int step = 0; shape.clean && step < 64 && !writes.flush(); ++step) {
      process.tier.complete();
      static_cast<void>(process.cache.pollTransfers());
    }
    const uint32_t resumed = resumedAfterRestart(process, prompt);
    require(resumed + (shape.clean ? 0 : WriteBehind::kMinimumTokens) >= prompt.size(),
            "a restart resumed at token " + std::to_string(resumed) + " of " + std::to_string(prompt.size()));
  }
}

// A point is measured against what the tier holds when it is stepped: a durable point whose
// copies the tier gave up since (here the volume's limit fell and rose again; a full quota's
// replacement or a windowed lane's tail release take one the same way) is no base, so the
// conversation's next point, a stride and a half deep with nothing durable on its chain, falls due.
void testAPointGivenUpAfterItWasDurableIsNoBase() {
  const auto budget = std::make_shared<model::DiskBudget>(uint64_t{1} << 20);
  Process process(true, 8 * kStridePages, budget);
  process.tier.budget = budget;
  // States count in the quota, as on the volume.
  process.shelf->quota = budget;
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompt;
  static_cast<void>(nextTurn(process, writes, prompt, 1, 0.0));
  static_cast<void>(nextTurn(process, writes, prompt, 2, 0.0));
  land(process, writes, 0.0);
  require(writes.snapshot().durable == 1 && process.shelf->disk.size() == 1, "the first point did not become durable");
  budget->setLimitBytes(0);
  process.cache.keepDiskLimit();
  budget->setLimitBytes(budget->capacityBytes());
  require(process.shelf->disk.empty() && !process.tier.slots, "the tier kept the point's copies past its limit");
  static_cast<void>(nextTurn(process, writes, prompt, 3, 0.0));
  land(process, writes, 0.0);
  const uint32_t resumed = resumedAfterRestart(process, prompt);
  require(resumed + WriteBehind::kMinimumTokens >= prompt.size(),
          "with its durable point given up, a restart resumed at token " + std::to_string(resumed) + " of " +
              std::to_string(prompt.size()));
}

// A durable point's chain can lose a copy above it while its own copy and its state's copy stay:
// the quota gives up the copies of what memory holds by their conversation's use, the chain's root
// first, refreshed first by the conversation's last restore (here the volume's limit fell by a
// copy and rose again; a demotion's room takes one the same way). A restart then takes the point
// back no more, so it is no base: the conversation's next point, half a stride past it, falls due,
// and a restart takes that one back.
void testAPointWhoseChainLostACopyIsNoBase() {
  const auto budget = std::make_shared<model::DiskBudget>(uint64_t{1} << 20);
  Process process(true, 8 * kStridePages, budget);
  process.tier.budget = budget;
  // States count in the quota, as on the volume.
  process.shelf->quota = budget;
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompt;
  static_cast<void>(nextTurn(process, writes, prompt, 1, 0.0));
  static_cast<void>(nextTurn(process, writes, prompt, 2, 0.0));
  land(process, writes, 0.0);
  static_cast<void>(nextTurn(process, writes, prompt, 3, 0.0));
  land(process, writes, 0.0);
  require(writes.snapshot().durable == 1 && writes.snapshot().waiting == 1 &&
              resumedAfterRestart(process, prompt) == WriteBehind::kMinimumTokens,
          "the point a stride deep was not durable, or the one half a stride past it was copied");
  budget->setLimitBytes(budget->usedBytes() - 1);
  process.cache.keepDiskLimit();
  budget->setLimitBytes(budget->capacityBytes());
  require(process.shelf->disk.size() == 1 && resumedAfterRestart(process, prompt) == 0,
          "the quota gave up the durable point's state, or no copy of its chain");
  land(process, writes, 0.0);
  const uint32_t resumed = resumedAfterRestart(process, prompt);
  require(resumed == prompt.size(), "with a copy of its durable point's chain given up, a restart resumed at token " +
                                        std::to_string(resumed) + " of " + std::to_string(prompt.size()));
}

// A copy given up on another conversation's chain leaves a waiting point's base as it was: only a point whose chain
// held the copy is measured again, by a walk of its chain. At a full quota nearly every copy given up lies under a
// state, and a walk of every waiting point's chain at each step would cost the most where the SSD is full.
void testACopyGivenUpOnAnotherChainLeavesAPointsBase() {
  const auto budget = std::make_shared<model::DiskBudget>(uint64_t{1} << 20);
  Process process(true, 8 * kStridePages, budget);
  process.tier.budget = budget;
  // States count in the quota, as on the volume.
  process.shelf->quota = budget;
  WriteBehind writes(process.cache);
  // The other conversation: a durable point a stride deep, used before the first's.
  std::vector<uint32_t> other;
  static_cast<void>(nextTurn(process, writes, other, 11, 0.0));
  static_cast<void>(nextTurn(process, writes, other, 12, 0.0));
  land(process, writes, 0.0);
  // The conversation: a durable point a stride deep, and one half a stride past it waiting.
  std::vector<uint32_t> prompt;
  static_cast<void>(nextTurn(process, writes, prompt, 1, 0.0));
  static_cast<void>(nextTurn(process, writes, prompt, 2, 0.0));
  land(process, writes, 0.0);
  const uint64_t waiting = nextTurn(process, writes, prompt, 3, 0.0);
  land(process, writes, 0.0);
  require(writes.snapshot().durable == 2 && writes.snapshot().waiting == 1,
          "the two points a stride deep were not durable, or the one half a stride past was copied");
  // One copy goes, the oldest: the root of the other conversation's chain.
  budget->setLimitBytes(budget->usedBytes() - 1);
  process.cache.keepDiskLimit();
  budget->setLimitBytes(budget->capacityBytes());
  require(resumedAfterRestart(process, other) == 0 &&
              resumedAfterRestart(process, prompt) == WriteBehind::kMinimumTokens,
          "the quota gave up no copy of the other conversation's chain, or one of the conversation's");
  const uint64_t visits = process.cache.reachVisits();
  land(process, writes, 0.0);
  require(writes.snapshot().waiting == 1 && process.cache.reachVisits() - visits < process.cache.chainLength(waiting),
          "a copy given up on another conversation's chain had the waiting point measured again (" +
              std::to_string(process.cache.reachVisits() - visits) + " visits)");
}

// A point not due is measured again once the point it was measured against is given up: the first
// point waits for its copy while the write budget is spent, memory takes its state meanwhile
// (dropped, as the cap says), and once the budget refills the conversation's newest point, with
// nothing of its chain durable or being copied, falls due.
void testAPointNotDueFallsDueOnceItsBaseIsGivenUp() {
  test::WriteBudget budget;
  Process process(true, 8 * kStridePages, budget.budget);
  process.shelf->writeBudget = budget.budget;
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompt;
  static_cast<void>(nextTurn(process, writes, prompt, 1, budget.now));
  budget.spend();
  const uint64_t first = nextTurn(process, writes, prompt, 2, budget.now);
  land(process, writes, budget.now);
  require(!process.tier.demotions && process.shelf->disk.empty() && writes.snapshot().waiting == 1,
          "the first point did not wait for the write budget");
  const uint64_t second = nextTurn(process, writes, prompt, 3, budget.now);
  for (int step = 0; step < 4 && process.cache.stateResident(first); ++step)
    static_cast<void>(test::stateStep(process.cache));
  require(!process.cache.stateResident(first) && process.cache.stateResident(second) && process.shelf->disk.empty(),
          "memory did not take the first point's state, dropped while the budget is spent");
  budget.refill();
  land(process, writes, budget.now);
  const uint32_t resumed = resumedAfterRestart(process, prompt);
  require(resumed + WriteBehind::kMinimumTokens >= prompt.size(),
          "with the point it was measured against given up, a restart resumed at token " + std::to_string(resumed) +
              " of " + std::to_string(prompt.size()));
}

// A full quota keeps the only copies of a conversation that left memory before any copy ahead of
// one that is resident: two conversations on a windowed layout take turns of two strides each, memory
// holding both, every page of them whole, and the quota one and a little more; each turn resumes at
// the point its conversation's last turn published, as copies ahead give way first.
void testAFullQuotaKeepsTheOtherConversationsPrefix() {
  constexpr uint32_t kStride = WriteBehind::kMinimumTokens / KvCache::pageTokens;
  constexpr uint32_t kTurnPages = 2 * kStride;
  constexpr CacheWindowPolicy window{2};
  Process process(8 * kTurnPages, 4 * kTurnPages + 16, window);
  process.tier.capacity = 2 * kTurnPages + kStride;
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompts[2];
  for (uint32_t turn = 1; turn <= 6; ++turn) {
    std::vector<uint32_t> &prompt = prompts[turn % 2];
    const uint32_t before = static_cast<uint32_t>(prompt.size());
    const std::vector<uint32_t> more = promptOf(100'000 * (turn % 2 + 1) + 1000 * turn, kTurnPages);
    prompt.insert(prompt.end(), more.begin(), more.end());
    std::vector<uint32_t> asked = prompt;
    asked.push_back(7);
    process.cache.beginRequest(turn);
    uint32_t resumed = 0;
    bool restored = false;
    // An attempt lands one demotion: memory writes the pages of a refused
    // point as it lets them go, up to a turn's.
    for (uint32_t attempt = 0; attempt < 64 + kTurnPages; ++attempt) {
      TokenAdmission admission;
      if (!restored) {
        const CacheLookup lookup = process.cache.lookup(asked, {});
        if (!attempt)
          resumed = lookup.resumeBoundary();
        if (lookup.state)
          admission = process.cache.restoreRequest(turn, lookup);
        restored = !lookup.state || admission.granted();
        process.tier.complete();
        static_cast<void>(process.cache.pollTransfers());
      }
      if (restored)
        admission = admitTokens(process.cache, turn, prompt.size());
      if (restored && admission.granted())
        break;
      const uint32_t lacking =
          admission.additionalPages - std::min(admission.additionalPages, admission.availablePages);
      static_cast<void>(test::freeFromCache(process.cache, test::kInServiceGrowth,
                                            {.pages = std::max(lacking, 1u), .form = admission.form}));
      process.tier.complete();
      static_cast<void>(process.cache.pollTransfers());
    }
    require(resumed + KvCache::pageTokens >= before,
            "turn " + std::to_string(turn) + " resumed at token " + std::to_string(resumed) + " of the " +
                std::to_string(before) + " its conversation had cached (KV on the SSD " +
                std::to_string(process.cache.snapshot().kvTier.diskBlocks) + " pages, copies ahead " +
                std::to_string(process.cache.snapshot().kvTier.copies) + ")");
    process.cache.publishCommittedBlocks(turn, prompt, static_cast<uint32_t>(prompt.size()), {});
    const uint64_t block = process.cache.blockAt(turn, static_cast<uint32_t>(prompt.size()));
    process.cache.endRequest(turn);
    if (!process.cache.stateResident(block))
      process.publish(block);
    writes.published(block, static_cast<uint32_t>(prompt.size()), 0.0);
    for (int step = 0; step < 1024; ++step) {
      const bool settled = writes.run(0.0, kCopiesDemanded, false);
      process.tier.complete();
      if (!process.cache.pollTransfers() && !settled)
        break;
    }
    if (turn != 4)
      continue;
    // A crash with the quota full: the next process takes back each
    // conversation at the point its first turn copied, whole, and this
    // process still resumes both. Their points at four strides were refused
    // (the price the documents state): one of four strides beside the
    // other's of two cannot fit a quota of five unless a copy ahead takes the
    // resident conversation's point, the rotation. A restart takes back as
    // much as before, 4,096 + 4,096 tokens where it took 8,192 + 0.
    Process next(8 * kTurnPages, 8 * kTurnPages, window);
    static_cast<void>(next.adopt(process.kvLabels(), process.stateLabels()));
    const auto resumes = [&](Process &where, const std::vector<uint32_t> &cached) {
      std::vector<uint32_t> asked = cached;
      asked.push_back(7);
      return where.cache.lookup(asked, {}).resumeBoundary();
    };
    require(resumes(next, prompts[0]) == prompts[0].size() / 2 && resumes(next, prompts[1]) == prompts[1].size() / 2 &&
                resumes(process, prompts[0]) == prompts[0].size() && resumes(process, prompts[1]) == prompts[1].size(),
            "a crash with the quota full took back " + std::to_string(resumes(next, prompts[0])) + " and " +
                std::to_string(resumes(next, prompts[1])) + " tokens of the two conversations");
  }
}

// Conversations that take turns through a temporary tier, one in memory at a time, each turn
// resumed from its newest point on the SSD and adding a page. Turn after turn, a full quota then
// writes each page once: the SSD holds every conversation's newest point with room to spare, and
// what it gives up is old history (the states earlier turns superseded), not the copy of a page
// read back that memory holds, which would be written again as it leaves. Then each conversation
// is asked two questions from its last turn, one after another: the second resumes there too,
// though the first went on from that point and made it history. On a windowed layout, with memory
// for each page's window unit too, a point on the SSD holds its tail's window K/V as window parts or
// as units in memory, and the same holds: each newest point's window parts take room beside its chain.
void testTurnsThroughAFullQuotaWriteEachPageOnce(std::optional<CacheWindowPolicy> window = std::nullopt) {
  constexpr uint32_t kConversations = 3, kFirstPages = 4, kRounds = 6;
  const uint32_t tail = window ? window->windowPages : 0;
  const uint32_t kMemoryPages = (kFirstPages + kRounds + 2) * (window ? 1 + window->unitPages : 1);
  // Every newest point at its last turn, chain, tail and state, and six states more.
  const uint64_t kQuota = (kConversations * (kFirstPages + kRounds + tail) + kConversations + 6) * 100;
  const auto quota = std::make_shared<model::DiskBudget>(kQuota);
  Process process(false, 4 * kMemoryPages, quota, window);
  process.storage.budgetPages = kMemoryPages;
  process.tier.budget = quota;
  process.shelf->quota = quota;
  std::vector<uint32_t> prompts[kConversations];
  uint64_t request = 0;
  // Runs `prompt` to its end, resumed where its cache has it, and keeps a state there; returns
  // where it resumed.
  const auto run = [&](const std::vector<uint32_t> &prompt) {
    std::vector<uint32_t> asked = prompt;
    asked.push_back(7);
    const uint64_t id = ++request;
    process.cache.beginRequest(id);
    uint32_t resumed = 0;
    bool restored = false;
    for (int attempt = 0; attempt < 64; ++attempt) {
      TokenAdmission admission;
      if (!restored) {
        const CacheLookup lookup = process.cache.lookup(asked, {});
        if (!attempt)
          resumed = lookup.resumeBoundary();
        if (lookup.state)
          admission = process.cache.restoreRequest(id, lookup);
        restored = !lookup.state || admission.granted();
        process.tier.complete();
        static_cast<void>(process.cache.pollTransfers());
      }
      if (restored)
        admission = admitTokens(process.cache, id, prompt.size());
      if (restored && admission.granted())
        break;
      const uint32_t lacking =
          admission.additionalPages - std::min(admission.additionalPages, admission.availablePages);
      static_cast<void>(test::freeFromCache(process.cache, test::kInServiceGrowth,
                                            {.pages = std::max(lacking, 1u), .form = admission.form}));
      process.tier.complete();
      static_cast<void>(process.cache.pollTransfers());
    }
    process.cache.publishCommittedBlocks(id, prompt, static_cast<uint32_t>(prompt.size()), {});
    // Published from its lane, whose units its tail takes.
    process.publish(process.cache.blockAt(id, static_cast<uint32_t>(prompt.size())), false, id);
    process.cache.endRequest(id);
    return resumed;
  };
  for (uint32_t round = 0; round <= kRounds; ++round)
    for (uint32_t each = 0; each < kConversations; ++each) {
      std::vector<uint32_t> &prompt = prompts[each];
      const uint32_t before = static_cast<uint32_t>(prompt.size());
      const std::vector<uint32_t> more = promptOf(100'000 * (each + 1) + 1000 * round, round ? 1 : kFirstPages);
      prompt.insert(prompt.end(), more.begin(), more.end());
      const uint32_t resumed = run(prompt);
      require(resumed == before, "conversation " + std::to_string(each) + " resumed at token " +
                                     std::to_string(resumed) + " of the " + std::to_string(before) +
                                     " it had cached, round " + std::to_string(round));
    }
  const KvTierSnapshot tier = process.cache.snapshot().kvTier;
  require(tier.demotions > kConversations * kFirstPages && !tier.rewrites,
          std::to_string(tier.rewrites) + " of " + std::to_string(tier.demotions) +
              " pages written were written again after a full quota gave up their copy while memory held them");
  require(!window || tier.windowPartBytes, "no window part was written");
  for (uint32_t question = 0; question < 2; ++question)
    for (uint32_t each = 0; each < kConversations; ++each) {
      std::vector<uint32_t> asked = prompts[each];
      const std::vector<uint32_t> more = promptOf(900'000 + 10'000 * question + 1000 * each, 1);
      asked.insert(asked.end(), more.begin(), more.end());
      const uint32_t resumed = run(asked);
      require(resumed == prompts[each].size(),
              "question " + std::to_string(question) + " to conversation " + std::to_string(each) +
                  " resumed at token " + std::to_string(resumed) + " of the " +
                  std::to_string(prompts[each].size()) + " it had cached");
    }
}

// Conversations that take turns on a persistent tier, a stride each, memory holding all of them and
// their chains together far past the quota. Copies ahead fill the quota once; then a new point is
// refused rather than take a resident conversation's point, which that conversation's next stride
// would copy again whole, and so on in turn. So each page is written once, and once the quota is
// full the turns write nothing: the cache converges. Every turn resumes all it had cached, and a
// restart takes back each conversation's durable point whole.
void testAFullQuotaDoesNotRotate(std::optional<CacheWindowPolicy> window = std::nullopt) {
  constexpr uint32_t kConversations = 3, kRounds = 8;
  const uint32_t memoryPages = kConversations * (kRounds + 2) * kStridePages * (window ? 1 + window->unitPages : 1);
  Process process(true, 4 * memoryPages, nullptr, window);
  process.storage.budgetPages = memoryPages;
  process.tier.capacity = 4 * kStridePages;
  process.tier.transferLimit = 64;
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompts[kConversations];
  uint64_t request = 0;
  uint32_t written = 0;
  uint64_t states = 0;
  for (uint32_t round = 0; round < kRounds; ++round) {
    // Written by the rounds after the quota filled (the second round fills it).
    if (round == 3) {
      written = process.tier.demotions;
      states = process.cache.snapshot().stateCache.offloads;
    }
    for (uint32_t each = 0; each < kConversations; ++each) {
      std::vector<uint32_t> &prompt = prompts[each];
      const uint32_t before = static_cast<uint32_t>(prompt.size());
      const std::vector<uint32_t> more = promptOf(100'000 * (each + 1) + 1000 * round, kStridePages);
      prompt.insert(prompt.end(), more.begin(), more.end());
      std::vector<uint32_t> asked = prompt;
      asked.push_back(7);
      const uint32_t resumed = process.cache.lookup(asked, {}).resumeBoundary();
      require(resumed + KvCache::pageTokens >= before,
              "conversation " + std::to_string(each) + " resumed at token " + std::to_string(resumed) + " of the " +
                  std::to_string(before) + " it had cached, round " + std::to_string(round));
      // The turn, its state published from its lane, whose units its tail takes.
      process.cache.beginRequest(++request);
      if (const CacheLookup lookup = process.cache.lookup(asked, {}); lookup.state) {
        require(process.cache.restoreRequest(request, lookup).granted(), "the turn did not restore its prefix");
        process.tier.complete();
        static_cast<void>(process.cache.pollTransfers());
      }
      require(admitTokens(process.cache, request, prompt.size()).granted(), "the turn's KV was not admitted");
      process.cache.publishCommittedBlocks(request, prompt, static_cast<uint32_t>(prompt.size()), {});
      const uint64_t block = process.cache.blockAt(request, static_cast<uint32_t>(prompt.size()));
      process.publish(block, false, request);
      process.cache.endRequest(request);
      writes.published(block, static_cast<uint32_t>(prompt.size()), 0.0);
      land(process, writes, 0.0);
    }
  }
  const KvTierSnapshot tier = process.cache.snapshot().kvTier;
  require(process.tier.demotions == written && process.cache.snapshot().stateCache.offloads == states,
          "at a full quota with every conversation in memory, " + std::to_string(kRounds - 3) + " rounds wrote " +
              std::to_string(process.tier.demotions - written) + " pages and " +
              std::to_string(process.cache.snapshot().stateCache.offloads - states) + " states");
  require(tier.copies && !tier.rewrites && writes.snapshot().refused,
          std::to_string(tier.rewrites) + " of " + std::to_string(tier.copies) +
              " pages copied ahead were written again after the quota gave up their copy while memory held them");
  process.cache.checkInvariants();
  // A restart takes back each durable point whole: the first conversation's at two strides, the
  // others' at their first.
  Process next(true, 4 * memoryPages, nullptr, window);
  static_cast<void>(next.adopt(process.kvLabels(), process.stateLabels(), process.windowLabels()));
  for (uint32_t each = 0; each < kConversations; ++each) {
    std::vector<uint32_t> asked = prompts[each];
    asked.push_back(7);
    const uint32_t resumed = next.cache.lookup(asked, {}).resumeBoundary();
    require(resumed == (each ? 1u : 2u) * WriteBehind::kMinimumTokens,
            std::string(window ? "windowed: " : "dense: ") + "a restart at a full quota resumed conversation " +
                std::to_string(each) + " at token " + std::to_string(resumed) + " (" +
                std::to_string(process.cache.snapshot().kvTier.diskBlocks) + " pages and " +
                std::to_string(process.shelf->disk.size()) + " states on the SSD)");
  }
  next.cache.checkInvariants();
  // Memory then lets pages go, the first conversation's, used longest ago; theirs past its point
  // have no copy, and their demotions take the room of copies of what memory holds, the least
  // recently used first: that conversation's own chain (two strides), as a use refreshes the copies
  // on its chain, then the second's. A crash at any moment takes back every point the demotions
  // have not reached: the other two whole while the first's copies last, the third always.
  const auto takenBack = [&](uint32_t each) {
    Process after(true, 4 * memoryPages, nullptr, window);
    static_cast<void>(after.adopt(process.kvLabels(), process.stateLabels(), process.windowLabels()));
    std::vector<uint32_t> asked = prompts[each];
    asked.push_back(7);
    return after.cache.lookup(asked, {}).resumeBoundary();
  };
  // The state copies on the SSD of the conversations in use, by their blocks' first tokens (promptOf's seeds).
  const auto statesInUse = [&] {
    std::unordered_map<uint64_t, uint32_t> conversation;
    for (const auto &bytes : process.kvLabels())
      if (const auto label = decodeKvSlotLabel(bytes, process.windowed && window->unitPages))
        conversation[label->block.id] = label->block.tokens[0] / 100'000 - 1;
    std::set<uint64_t> held;
    for (const auto &bytes : process.stateLabels())
      if (const auto label = decodeLabel<StateLabel>(bytes); label && conversation[label->block] != 0)
        held.insert(label->block);
    return held;
  };
  const std::set<uint64_t> heldInUse = statesInUse();
  require(!heldInUse.empty(), "no state copy of a conversation in use is on the SSD");
  for (uint32_t freed = 0; freed < 3 * kStridePages; ++freed) {
    static_cast<void>(test::freeFromCache(process.cache, test::kInServiceGrowth, {.pages = 1}));
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
    // A state copy is as old as its conversation's last use, as the K/V copies of its chain are: the first
    // demotions give up the first conversation's copies, none of a state copy of the others.
    if (process.tier.demotions > written && process.tier.demotions - written <= kStridePages)
      require(std::ranges::includes(statesInUse(), heldInUse),
              std::string(window ? "windowed: " : "dense: ") + "after " +
                  std::to_string(process.tier.demotions - written) + " demotions of the conversation used longest " +
                  "ago, a state copy of a conversation in use was given up");
    if (process.tier.demotions == written || freed % 8)
      continue;
    for (uint32_t each = process.tier.demotions - written > 2 * kStridePages ? 2 : 1; each < kConversations; ++each)
      require(takenBack(each) == WriteBehind::kMinimumTokens,
              std::string(window ? "windowed: " : "dense: ") + "after " +
                  std::to_string(process.tier.demotions - written) + " demotions of the conversation used longest " +
                  "ago, a crash took back conversation " + std::to_string(each) + " at token " +
                  std::to_string(takenBack(each)));
  }
  require(process.tier.demotions > written, "memory let no page go that needed a write");
  process.cache.checkInvariants();
}

// A request's restore refreshes the copies on its chain, once, on the engine's thread: on a chain of
// 8,100 blocks (260K tokens of Inkling-2-Mini), every block with a copy, restoring it from memory
// stays within a bound a tick can afford (the refresh alone measured 0.17 ms at 3,100 blocks and
// 0.49 ms at 8,100 on the M6).
void testARestoreRefreshesItsChainWithinATick() {
  constexpr uint32_t kBlocks = 8100;
  Process process(true, 2 * kBlocks);
  process.tier.capacity = kBlocks;
  process.tier.transferLimit = 1u << 20;
  const std::vector<uint32_t> prompt = promptOf(1, kBlocks);
  const std::vector<uint64_t> chain = process.cachePrompt(1, prompt);
  process.publish(chain.back());
  for (uint32_t step = 0; step < kBlocks && process.cache.persist(chain.back()) != PersistStatus::Durable; ++step) {
    process.tier.complete();
    static_cast<void>(process.cache.pollTransfers());
  }
  require(process.cache.snapshot().kvTier.diskBlocks == kBlocks, "the chain was not copied whole");
  std::vector<uint32_t> asked = prompt;
  asked.push_back(7);
  process.cache.beginRequest(2);
  const CacheLookup lookup = process.cache.lookup(asked, {});
  const auto start = std::chrono::steady_clock::now();
  require(process.cache.restoreRequest(2, lookup).granted(), "the chain was not restored");
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  require(ms < test::timeBound(20.0),
          "restoring a chain of 8,100 blocks with its refresh took " + std::to_string(ms) + " ms");
  process.cache.endRequest(2);
  process.cache.checkInvariants();
}

// A crash in a loop of quick turns on a windowed layout, with memory short enough that a
// conversation's history leaves as it goes on (Cache::reclaimState), still leaves the next process
// a point within a stride (WriteBehind::kMinimumTokens) of the last reply: a conversation's deepest
// durable point is not history until a deeper point of it is durable. Turns of half a stride; the
// process is killed after each turn's writes had a moment, and the restart is checked every time.
void testAWindowedCrashLosesAtMostAStride() {
  constexpr uint32_t kStridePages = WriteBehind::kMinimumTokens / KvCache::pageTokens;
  constexpr uint32_t kTurnPages = kStridePages / 2;
  constexpr uint32_t kTurns = 10;
  constexpr CacheWindowPolicy window{2};
  // Memory for the conversation's pages, every one whole, and little more: its states' buffers run
  // short (below).
  Process process(2 * kTurns * kTurnPages, kTurns * kTurnPages + 8, window);
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompt;
  for (uint32_t turn = 1; turn <= kTurns; ++turn) {
    const std::vector<uint32_t> more = promptOf(1000 * turn, kTurnPages);
    prompt.insert(prompt.end(), more.begin(), more.end());
    std::vector<uint32_t> asked = prompt;
    asked.push_back(7);
    process.cache.beginRequest(turn);
    for (int attempt = 0; attempt < 16; ++attempt) {
      const CacheLookup lookup = process.cache.lookup(asked, {});
      TokenAdmission admission = lookup.state ? process.cache.restoreRequest(turn, lookup) : TokenAdmission{};
      if (admission.granted()) {
        process.tier.complete();
        static_cast<void>(process.cache.pollTransfers());
        admission = admitTokens(process.cache, turn, prompt.size());
      }
      if (admission.granted())
        break;
      const uint32_t lacking =
          admission.additionalPages - std::min(admission.additionalPages, admission.availablePages);
      static_cast<void>(test::freeFromCache(process.cache, test::kInServiceGrowth,
                                            {.pages = std::max(lacking, 1u), .form = admission.form}));
      process.tier.complete();
      static_cast<void>(process.cache.pollTransfers());
    }
    process.cache.publishCommittedBlocks(turn, prompt, static_cast<uint32_t>(prompt.size()), {});
    const uint64_t block = process.cache.blockAt(turn, static_cast<uint32_t>(prompt.size()));
    process.cache.endRequest(turn);
    if (!process.cache.stateResident(block))
      process.publish(block);
    writes.published(block, static_cast<uint32_t>(prompt.size()), 0.0);
    for (int step = 0; step < 256; ++step) {
      const bool settled = writes.run(0.0, kCopiesDemanded, false);
      process.tier.complete();
      if (!process.cache.pollTransfers() && !settled)
        break;
    }
    // Memory runs short once the conversation is a few turns deep: the
    // buffers of two states are claimed after each turn's copies.
    for (int claim = 0; turn >= 3 && claim < 2; ++claim) {
      static_cast<void>(test::stateStep(process.cache));
      process.tier.complete();
      static_cast<void>(process.cache.pollTransfers());
    }
    Process next(2 * kTurns * kTurnPages, 2 * kTurns * kTurnPages, window);
    static_cast<void>(next.adopt(process.kvLabels(), process.stateLabels()));
    const uint32_t resumed = next.cache.lookup(asked, {}).resumeBoundary();
    require(turn < 2 || resumed + WriteBehind::kMinimumTokens >= prompt.size(),
            "after turn " + std::to_string(turn) + " a restart resumed at token " + std::to_string(resumed) + " of " +
                std::to_string(prompt.size()) + " (durable " + std::to_string(writes.snapshot().durable) + ", states on the SSD " +
                std::to_string(process.shelf->disk.size()) + ", evictions " +
                std::to_string(process.cache.snapshot().stateCache.evictions) + ")");
  }
}

// A point is measured against what a restart would take back: a state a reclaim wrote, with a copy
// of the leaf it sits on and none of the pages before it, is no base, as the next start leaves it
// behind. Here the last point copied whole is a stride deep and the conversation goes on nearly
// two strides past it: the newest point falls due.
void testAStateWhoseChainIsNotOnTheSsdIsNoBase() {
  Process process(true, 8 * kStridePages);
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompt;
  const auto turn = [&](uint64_t id, uint32_t pages) {
    const std::vector<uint32_t> more = promptOf(1000 * static_cast<uint32_t>(id), pages);
    prompt.insert(prompt.end(), more.begin(), more.end());
    const uint64_t block = converse(process, id, prompt);
    writes.published(block, static_cast<uint32_t>(prompt.size()), 0.0);
    return block;
  };
  static_cast<void>(turn(1, kTurnPages));
  const uint64_t whole = turn(2, kTurnPages);
  land(process, writes, 0.0);
  require(writes.snapshot().durable == 1 && process.cache.stillTakenBack(whole), "the first point did not become durable");
  // Two pages short of a stride past it: not due.
  const uint64_t tip = turn(3, kStridePages - 2);
  land(process, writes, 0.0);
  require(writes.snapshot().durable == 1 && process.tier.demotions == kStridePages, "a point not due was copied");
  // Another conversation runs: memory takes the first one's states, written as they leave, and
  // then one page, the leaf its newest state sits on, demoted for that state.
  static_cast<void>(converse(process, 100, promptOf(900'000, 2)));
  for (int step = 0; step < 8 && process.cache.stateResident(tip); ++step) {
    static_cast<void>(test::stateStep(process.cache));
    static_cast<void>(process.cache.pollTransfers());
  }
  require(!process.cache.stateResident(tip) && process.shelf->disk.size() >= 2, "the newest state was not written out");
  require(test::evictStep(process.cache, ReclaimClass::Ordinary).gave(), "memory took no page");
  process.tier.complete();
  static_cast<void>(process.cache.pollTransfers());
  require(process.tier.demotions == kStridePages + 1, "the reclaim did not demote one leaf (" +
                                                          std::to_string(process.tier.demotions) + " writes)");
  // The conversation goes on, two pages short of a stride again.
  static_cast<void>(turn(4, kStridePages - 2));
  land(process, writes, 0.0);
  const uint32_t resumed = resumedAfterRestart(process, prompt);
  require(resumed + WriteBehind::kMinimumTokens >= prompt.size(),
          "measured against a state whose chain is not on the SSD, a restart resumed at token " +
              std::to_string(resumed) + " of " + std::to_string(prompt.size()));
}

// The cadence is a state a stride for the points published while copies wait too: a point that
// lands is a base for those waiting. A conversation durable a stride deep goes on four turns of
// half a stride while the write budget is spent; once it refills, the points two and three strides
// deep are copied and the one between them is not.
void testPointsThatWaitedAreCopiedAStrideApart() {
  test::WriteBudget budget;
  Process process(true, 8 * kStridePages, budget.budget);
  process.shelf->writeBudget = budget.budget;
  WriteBehind writes(process.cache);
  std::vector<uint32_t> prompt;
  static_cast<void>(nextTurn(process, writes, prompt, 1, budget.now));
  static_cast<void>(nextTurn(process, writes, prompt, 2, budget.now));
  land(process, writes, budget.now);
  require(writes.snapshot().durable == 1 && process.shelf->disk.size() == 1, "the first point did not become durable");
  budget.spend();
  for (uint64_t id = 3; id <= 6; ++id) {
    static_cast<void>(nextTurn(process, writes, prompt, id, budget.now));
    land(process, writes, budget.now);
  }
  require(process.shelf->disk.size() == 1 && process.tier.demotions == kStridePages,
          "a point was copied while the budget was spent");
  const uint32_t waited = writes.snapshot().waiting;
  budget.refill();
  land(process, writes, budget.now);
  require(resumedAfterRestart(process, prompt) == prompt.size(), "the newest point, a stride past the last, was not copied");
  require(process.shelf->disk.size() <= 3,
          std::to_string(process.shelf->disk.size()) + " states are on the SSD for a conversation three strides deep "
              "once the budget refilled (" + std::to_string(waited) + " points had waited, durable " +
              std::to_string(writes.snapshot().durable) + ", waiting " + std::to_string(writes.snapshot().waiting) + ")");
}

// A quota of one windowed conversation of four blocks, a state at its end: each copy a base part of 100 bytes,
// and a window part of 100 for the two blocks of the state's tail, which held their units. A second
// conversation's copies replace the first's, each copy giving up all its parts and freeing exactly what it
// counted, so that what the quota holds is always what the cache counts, a part each unit a block held.
void testAFullQuotaGivesUpBothPartsOfACopy() {
  constexpr CacheWindowPolicy window{2, 1};
  const auto budget = std::make_shared<model::DiskBudget>(4 * 100 + 2 * 100);
  Process process(true, budget, std::nullopt, window);
  process.storage.budgetPages = 16;
  process.tier.budget = budget;
  const auto counted = [&](const char *when) {
    const CacheSnapshot snapshot = process.cache.snapshot();
    require(budget->usedBytes() == snapshot.kvTier.diskBytes &&
                snapshot.kvTier.diskBytes == 100 * (snapshot.kvTier.diskBlocks + process.windowLabels().size()) &&
                process.kvLabels().size() == snapshot.kvTier.diskBlocks && process.windowLabels().size() == 2,
            std::string("the quota does not hold the parts of every copy the cache counts, and no other, ") + when +
                " (" + std::to_string(budget->usedBytes()) + " used, " + std::to_string(snapshot.kvTier.diskBytes) +
                " counted in " + std::to_string(snapshot.kvTier.diskBlocks) + " copies, " +
                std::to_string(process.kvLabels().size()) + " and " + std::to_string(process.windowLabels().size()) +
                " labels)");
  };
  const std::vector<uint64_t> first = process.cachePrompt(1, promptOf(1000, 4), {}, {3});
  process.writeEverything();
  counted("once the first conversation is written");
  require(process.cache.snapshot().kvTier.diskBlocks == 4, "the first conversation did not fill the quota");
  const std::vector<uint64_t> second = process.cachePrompt(2, promptOf(2000, 4), {}, {3});
  process.writeEverything();
  counted("once the second conversation replaced it");
  std::vector<uint32_t> again = promptOf(2000, 4);
  again.push_back(1);
  require(process.cache.snapshot().kvTier.diskBlocks == 4 && process.cache.lookup(again, {}).state.has_value(),
          "the second conversation's copies did not replace the first's");
}

// A windowed copy whose write fails, as one whose window part failed fails at the tier, leaves no copy: the
// cache gives the whole copy up, both parts' bytes return to the quota, and the page stays.
void testAFailedWriteGivesUpBothParts() {
  const auto budget = std::make_shared<model::DiskBudget>(16 * 200);
  Process process(true, budget, std::nullopt, CacheWindowPolicy{2, 1});
  process.storage.budgetPages = 16;
  process.tier.budget = budget;
  const std::vector<uint64_t> blocks = process.cachePrompt(1, promptOf(1000, 2), {}, {1});
  require(test::stateStep(process.cache).gave() && test::evictStep(process.cache, ReclaimClass::Ordinary).gave() &&
              budget->usedBytes() == 200 && process.cache.snapshot().kvTier.diskBlocks == 1,
          "the leaf's copy did not start with both parts");
  process.tier.fail(engine::WriteClass::Reclaim);
  static_cast<void>(process.cache.pollTransfers());
  require(budget->usedBytes() == 0 && process.cache.snapshot().kvTier.diskBlocks == 0 &&
              process.cache.snapshot().pool.pagesPrefix == 2 * 2,
          "a failed write left a part of its copy, or its page went");
  process.cache.checkInvariants();
}

// A spent write budget writes no part of a windowed copy; once it refills, a point's copies are written
// with both parts.
void testASpentBudgetWritesNoPartOfACopy() {
  test::WriteBudget budget;
  Process process(true, budget.budget, std::nullopt, CacheWindowPolicy{2, 1});
  process.storage.budgetPages = 16;
  process.tier.budget = budget.budget;
  process.shelf->writeBudget = budget.budget;
  WriteBehind writes(process.cache);
  const std::vector<uint64_t> blocks = process.cachePrompt(1, promptOf(1000, 2), {}, {1});
  writes.published(blocks[1], WriteBehind::kMinimumTokens, budget.now, true);
  budget.spend();
  budget.now = WriteBehind::kDelayMilliseconds;
  require(!writes.run(budget.now, kCopiesDemanded, false) && !process.tier.demotions && process.shelf->disk.empty() &&
              !budget.budget->usedBytes(),
          "a part of a copy was written or reserved while the write budget was spent");
  budget.refill();
  static_cast<void>(writes.run(budget.now, kCopiesDemanded, false));
  process.tier.complete();
  require(process.cache.pollTransfers() && writes.run(budget.now, kCopiesDemanded, false) &&
              writes.snapshot().durable == 1 && process.tier.demotions == 2 &&
              std::ranges::all_of(process.tier.writes, [](const auto &write) { return write.parts == 3; }) &&
              budget.budget->usedBytes() == 2 * 200,
          "once the write budget refilled, the point's copies were not written with both parts");
}

// A due point waits while the write budget is spent and wakes when it refills; a clean stop writes
// nothing while it is spent, nor past the volume's limit.
void testWritesKeepToTheWriteBudget() {
  constexpr double kDelay = WriteBehind::kDelayMilliseconds;
  test::WriteBudget budget;
  Process process(true, budget.budget);
  process.shelf->writeBudget = budget.budget;
  // Its points are heads, due at once, as the rule under test is the budget's.
  WriteBehind writes(process.cache);
  const std::vector<uint64_t> blocks = process.cachePrompt(1, promptOf(1000, 2));
  process.publish(blocks[1]);
  writes.published(blocks[1], WriteBehind::kMinimumTokens, budget.now, true);
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
  writes.published(next[0], WriteBehind::kMinimumTokens, budget.now, true);
  budget.spend();
  require(writes.flush() && process.tier.demotions == 2 && process.shelf->disk.size() == 1 &&
              writes.snapshot().refused == 1 && !writes.snapshot().waiting,
          "the clean stop wrote past a spent write budget, or waited for it");
  budget.refill();

  // Nothing at a stop raises a limit the volume holds below a point.
  const std::vector<uint64_t> last = process.cachePrompt(3, promptOf(3000, 2));
  process.publish(last[1]);
  writes.published(last[1], WriteBehind::kMinimumTokens, budget.now, true);
  budget.budget->setLimitBytes(250);
  require(writes.flush() && writes.snapshot().refused == 2 && process.tier.demotions == 2,
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
  Process process(true, budget, CacheSpillPolicy{1, 1, true});
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
      const PersistStatus started = process.cache.persist(point);
      static_cast<void>(process.cache.pollTransfers());
      require(started == PersistStatus::Started && process.tier.inFlight() == copies,
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
    // A persistent tier's points are heads here, due at once.
    writes.published(used[1], WriteBehind::kMinimumTokens, 0.0, persistent);
    const uint64_t unused = process.cachePrompt(2, promptOf(2000, 1))[0];
    process.publish(unused);
    writes.published(unused, WriteBehind::kMinimumTokens, 1.0, persistent);
    if (persistent) {
      require(writes.nextWakeup() == 0.0 && !writes.run(1.0, kCopiesDemanded, false) && process.tier.demotions == 3,
              "a persistent tier did not copy a point in use at once");
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
  Process earlier(true, std::make_shared<model::DiskBudget>(1600), CacheSpillPolicy{1, 1, true});
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

// A Spill request writes through every block it publishes, past its restore point too, as a reply
// completes a block past the prompt's: no state needs that copy, which stays behind after a crash,
// and a clean stop gives it up, but for one whose time for such copies has run out, which gives up
// none past it, the clock read for each copy (RuntimeResources::closePersistentCache,
// supervision::kDrain).
void testCleanStopGivesUpCopiesPastAPoint() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  const auto budget = std::make_shared<model::DiskBudget>(1600);
  Process earlier(true, budget, CacheSpillPolicy{1, 1, true});
  earlier.tier.budget = budget;
  // Room for an Index extent beside the eight Full pages.
  earlier.storage.budgetPages = 9;
  earlier.cache.beginRequest(1, KvMode::Spill);
  require(earlier.cache.commitSpill(1, 4) && admitTokens(earlier.cache, 1, prompt.size()).granted(),
          "the Spill request was not admitted");
  earlier.cache.publishCommittedBlocks(1, prompt, static_cast<uint32_t>(prompt.size()), {});
  const uint64_t point = earlier.cache.blockAt(1, 3 * KvCache::pageTokens);
  static_cast<void>(test::spillRound(earlier.cache, earlier.tier));
  earlier.publish(point);
  earlier.cache.endRequest(1);
  require(earlier.persist(point) == PersistStatus::Durable && earlier.kvLabels().size() == 4,
          "the point did not become durable beside the copy past it");
  const auto adopted = [&] {
    Process next(true);
    return next.adopt(earlier.kvLabels(), earlier.stateLabels());
  };
  require(adopted().dropped == 1, "the copy past the point did not stay behind after a crash");
  earlier.cache.closePersistentTier(AwakeClock::now());
  require(adopted().dropped == 1, "a clean stop gave up a copy past the point once its time for them had run out");
  earlier.cache.closePersistentTier(AwakeClock::now() + std::chrono::seconds(60));
  const CacheAdoption adoption = adopted();
  require(adoption.blocks == 3 && adoption.states == 1 && adoption.dropped == 0 && earlier.kvLabels().size() == 3,
          "a clean stop left the copy past the point behind");
  earlier.cache.checkInvariants();
}

// Recomputing a point's chain leaves its copies to the point, which stays whole, and they count as
// no KV of the request.
void testRecomputedChainsStayWhole() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  for (const KvMode mode : {KvMode::Resident, KvMode::Spill}) {
    const auto budget = std::make_shared<model::DiskBudget>(1600);
    Process process(true, budget, CacheSpillPolicy{1, 1, true});
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
  Process process(true, budget, CacheSpillPolicy{1, 1, true});
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
  engine::Cache cache(pool, &tier, budget, CacheSpillPolicy{1, 1, true});
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
    testWindowedCopiesAreTwoParts();
    testAFullQuotaGivesUpBothPartsOfACopy();
    testASpentBudgetWritesNoPartOfACopy();
    testAFailedWriteGivesUpBothParts();
    testAWindowPartAddedLaterJoinsItsCopy();
    testWindowedPointsComeBackWhole();
    testACrashWhileAPointIsCopiedLeavesAPointWhole();
    testRecomputedBlocksGetCopiesOfTheirPages();
    testRestorePointsComeBack();
    testSaltedPointsComeBackUnderTheirSalt();
    testBrokenChainsStayBehind();
    testOldestPointGoes();
    testAPointAQueuedRequestRestoresStays();
    testAFullQuotaRefusesANewPointRatherThanRotate();
    testPointKeepsItsChainWhileItsStateIsWritten();
    testAdoptionTrimsToTheLimit();
    testLowerLimitKeepsCopiesInUse();
    testCleanStopRecordsRecency();
    testReusedCheckpointsComeBackOrdinary();
    testPointWithoutRoomIsRefused();
    testPointLargerThanTheQuotaIsRefused();
    testPersistingStateStaysInRam();
    testDurablePointsStayInRam();
    testOnlyTheNewestPointsAreKept();
    testWritesAreCountedByClass();
    testASpentBudgetKeepsTheNewestStates();
    testRestoresGoFirst();
    testCopiesWaitForWritesNeededNow();
    testTemporaryCopiesTakeOnlyFreeRoom();
    testPointsFallDueByProgress();
    testACrashLosesAtMostAStride();
    testAFullQuotaKeepsTheOtherConversationsPrefix();
    testTurnsThroughAFullQuotaWriteEachPageOnce();
    testTurnsThroughAFullQuotaWriteEachPageOnce(CacheWindowPolicy{2, 1});
    testAFullQuotaDoesNotRotate();
    testAFullQuotaDoesNotRotate(CacheWindowPolicy{2, 1});
    testARestoreRefreshesItsChainWithinATick();
    testAPointGivenUpAfterItWasDurableIsNoBase();
    testAPointWhoseChainLostACopyIsNoBase();
    testACopyGivenUpOnAnotherChainLeavesAPointsBase();
    testAPointNotDueFallsDueOnceItsBaseIsGivenUp();
    testAWindowedCrashLosesAtMostAStride();
    testAStateWhoseChainIsNotOnTheSsdIsNoBase();
    testPointsThatWaitedAreCopiedAStrideApart();
    testWritesKeepToTheWriteBudget();
    testPointsWaitForTheTier();
    testRefusedDemotionTakesTheLeafBesideCopiesAhead();
    testCopiesAheadOfSpilledPagesArePending();
    testTemporaryPointsWaitForTheirRequests();
    testCopiesFollowDemandNotAWindow();
    testCopyDemandCountsOnlyWhatGoesAtOnce();
    testWriteThroughCopiesAreRestorePoints();
    testCleanStopGivesUpCopiesPastAPoint();
    testRecomputedChainsStayWhole();
    testCommittedPagesTakeTheSpilledChainOfAStateInRam();
    testAPrefixWithoutOwnCopiesHoldsBackNoSpill();
    std::cout << "Persistent cache tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
