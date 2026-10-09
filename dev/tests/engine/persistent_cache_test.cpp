#include "Checked.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Cache.hpp"
#include "engine/DiskLabels.hpp"
#include "engine/WriteBehind.hpp"
#include "model/SlotFile.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

// Where a persistent tier keeps state copies in these tests: each copy on
// disk keeps the label its write gave it while it lives, as a slot file's
// record does. Writes land when the shelf is ready.
class ShelvedState;
struct Shelf final {
  std::set<const ShelvedState *> disk;
  uint32_t capacity = 8;
  bool ready = true;
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
  bool canOffload() const noexcept override { return !disk_; }
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
  while (admission.failure == TokenAdmissionFailure::Denied &&
         cache.reclaimForPages(admission.additionalPages, CacheReclaimMode::KeepExtents,
                               ReclaimClass::InUse)
             .madeProgress)
    admission = cache.ensureTokens(requestId, tokens);
  return admission;
}

// One process's cache over eight pages, with a KV tier and a shelf for
// state copies, persistent or not.
struct Process final {
  test::TestKvStorage storage{9, 100, 1};
  KvPool pool{storage, 8};
  test::TestKvTier tier;
  std::shared_ptr<Shelf> shelf = std::make_shared<Shelf>();
  engine::Cache cache;

  explicit Process(bool persistent, std::shared_ptr<const model::DiskBudget> budget = nullptr)
      : tier(persistent), cache(pool, &tier, std::move(budget)) {
    storage.budgetPages = 8;
    tier.capacity = 16;
    tier.transferLimit = 8;
  }

  // Caches the prompt through a finished request and returns
  // its blocks, root first.
  std::vector<uint64_t> cachePrompt(uint64_t requestId, const std::vector<uint32_t> &prompt) {
    cache.beginRequest(requestId);
    require(admitTokens(cache, requestId, prompt.size()).granted(), "prompt KV was not admitted");
    cache.publishCommittedBlocks(requestId, prompt, static_cast<uint32_t>(prompt.size()), {});
    std::vector<uint64_t> blocks;
    for (uint32_t end = KvCache::pageTokens; end <= prompt.size(); end += KvCache::pageTokens)
      blocks.push_back(cache.blockAt(requestId, end));
    cache.endRequest(requestId);
    return blocks;
  }

  void publish(uint64_t block, bool checkpoint = false) {
    cache.publishCompositeState(block, std::make_shared<ShelvedState>(shelf, false), checkpoint);
  }

  // Everything leaves RAM: each state is written as it goes, and each KV
  // leaf a state needs is demoted, landing before the next step.
  void writeEverything() {
    for (;;) {
      const CacheReclaimResult step =
          cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse);
      tier.complete();
      const bool landed = cache.pollTransfers();
      if (!step.madeProgress && !landed)
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

// In a persistent tier every copy is labelled: a KV block with its id, its
// parent, its tokens and its recency, a state with its block, boundary and
// class. A temporary tier labels nothing.
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

// The next process takes back each restore point whole, under the ids and
// recency it had: a lookup finds the deepest one and restores its chain
// from disk, and what the process caches next continues after them.
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

// A restore point whose chain is broken stays behind, and so do KV no state
// needs and damaged or stale records; the next process takes back none of
// their slots.
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

// A full persistent tier gives up the oldest restore point: its state, then,
// as room is needed again, the copies of its chain no other state needs,
// before any other point. A prefix a newer point shares stays with it.
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

// Making room for a point can give up its own state, when it is the oldest
// copy: the point is refused then, and the chain it no longer holds is not
// touched.
void testPointGivenUpForItselfIsRefused() {
  Process process(true);
  process.tier.capacity = 2;
  const std::vector<uint64_t> blocks = process.cachePrompt(1, promptOf(1000, 4));
  process.publish(blocks[3]);
  // The state is written, and the two leaves under it are demoted: the
  // chain is two resident pages without copies over two disk-only blocks.
  while (process.tier.slots < 2) {
    require(process.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse)
                .madeProgress,
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

// A point larger than the whole quota is refused before anything else gives
// way to it.
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

// While a write reads a state's RAM copy, no reclaim frees that copy, a
// page scan's included: it frees nothing until the write lands.
void testPersistingStateStaysInRam() {
  Process process(true);
  const uint64_t block = process.cachePrompt(1, promptOf(1000, 2))[1];
  process.publish(block);
  process.shelf->ready = false;
  require(process.cache.persist(block) == PersistStatus::Started, "the point was not written");
  process.tier.complete();
  static_cast<void>(process.cache.pollTransfers());
  static_cast<void>(process.cache.reclaimOne(CacheReclaimMode::ReusePages, ReclaimClass::InUse));
  require(process.cache.stateResident(block), "a page scan freed a state its write still read");
  process.shelf->ready = true;
  static_cast<void>(process.cache.pollTransfers());
  require(process.cache.persist(block) == PersistStatus::Durable &&
              process.cache.reclaimOne(CacheReclaimMode::ReusePages, ReclaimClass::InUse)
                  .madeProgress &&
              !process.cache.stateResident(block),
          "the state's RAM copy did not go once its write landed");
}

// At a clean stop each state copy's label takes its state's current recency:
// a point used after its write is the newest for the next process. Until
// then the labels keep the recency of their writes.
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

// States of a slot file kept for the next process: each holds a slot of it.
class SlotState final : public CompositeState {
public:
  explicit SlotState(std::shared_ptr<model::SlotFile::Slot> slot) : slot_(std::move(slot)) {}
  uint64_t bytes() const noexcept override { return kHostPageBytes; }
  uint64_t residentBytes() const noexcept override { return 0; }

private:
  std::shared_ptr<model::SlotFile::Slot> slot_;
};

// Taken back under a lower quota than the earlier process had, the oldest
// copies go until the rest fits.
void testAdoptionTrimsToTheQuota() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / ("splash-adoption-" + std::to_string(::getpid()));
  std::filesystem::create_directories(directory);
  const model::SlotFile::Persistence persistence{
      directory / "state.slots", directory / "state.records", {}, sizeof(StateLabel)};
  std::vector<std::vector<std::byte>> kv;
  {
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
  }
  auto budget = std::make_shared<model::DiskBudget>(2 * kHostPageBytes);
  model::SlotFile file(kHostPageBytes, budget, persistence);
  Process next(true, budget);
  std::vector<PersistedKv> blocks;
  for (const auto &label : kv)
    blocks.push_back({label, [&] { return next.tier.adopt(); }});
  std::vector<PersistedState> states;
  for (const model::SlotRecord &record : file.records())
    states.push_back({record.label, [&file, record](uint32_t) -> std::shared_ptr<const CompositeState> {
                        return std::make_shared<SlotState>(file.adopt(record));
                      }});
  const CacheAdoption adoption = next.cache.adopt(std::move(blocks), std::move(states));
  file.finishAdoption();
  require(adoption.states == 2 && budget->usedBytes() == 2 * kHostPageBytes,
          "adoption past the quota was not trimmed to it");
  for (uint32_t block = 2; block <= 3; ++block) {
    const std::vector<uint32_t> prompt(KvCache::pageTokens + 1, block);
    require(next.cache.lookup(prompt, {}).state.has_value(), "a newer state was given up first");
  }
  require(!next.cache.lookup(std::vector<uint32_t>(KvCache::pageTokens + 1, 1), {}).state,
          "the oldest state was kept past the quota");
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
}

// A point made durable stays in RAM: the pages of its chain are copied and
// its state is written from its own buffers, and once every write has
// landed it is durable. The next process takes it back whole.
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

// A point its conversation went on from is not kept, nor one that is gone or
// only a checkpoint. A point two conversations went on from is kept, and so
// is one only a checkpoint follows.
void testOnlyTheNewestPointsAreKept() {
  const std::vector<uint32_t> prompt = promptOf(1000, 4);
  Process process(true);
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

  Process followed(true);
  const std::vector<uint64_t> chain = followed.cachePrompt(1, prompt);
  followed.publish(chain[1]);
  followed.publish(chain[3], true);
  require(followed.cache.persist(chain[1]) == PersistStatus::Started,
          "a point only a checkpoint follows was not kept");
}

// Nothing is written while a restore waits: the tier serves the request
// first.
void testRestoresGoFirst() {
  const std::vector<uint32_t> earlier = promptOf(1000, 2);
  Process process(true);
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

// A point is written once it has waited the delay, unless its conversation
// went on from it meanwhile; a short point never is. At a clean stop every
// waiting point is written, the newest first.
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
  require(writes.snapshot().waiting == 1 && !writes.run(kDelay - 1) && !process.tier.demotions &&
              writes.nextWakeup() == kDelay,
          "a point was written before its delay, or a short one was queued");
  require(!writes.run(kDelay) && process.tier.demotions == 2 && !writes.nextWakeup(),
          "a due point was not written, or its writes did not hold the queue");
  process.tier.complete();
  require(process.cache.pollTransfers() && writes.run(kDelay + 1) &&
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
  require(writes.run(3 * kDelay) && writes.snapshot().unneeded == 1 &&
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

// While the tier has written its hourly bytes in the last hour, a due point
// waits, and the limit is looked at again a minute later.
void testWritesKeepToTheHourlyLimit() {
  constexpr double kMinute = 60'000.0, kHour = 3'600'000.0;
  auto budget = std::make_shared<model::DiskBudget>(4 * kHostPageBytes);
  Process process(true, budget);
  WriteBehind writes(process.cache, kHostPageBytes);
  const std::vector<uint64_t> blocks = process.cachePrompt(1, promptOf(1000, 2));
  process.publish(blocks[1]);
  writes.published(blocks[1], WriteBehind::kMinimumTokens, 0.0);
  require(!writes.run(0.0), "a point was written before its delay");
  // Eviction writes the hour's bytes.
  model::SlotFile file(kHostPageBytes, budget);
  const std::vector<std::byte> payload(kHostPageBytes, std::byte{1});
  require(file.write(file.acquire(), {payload}, {})->wait(), "the eviction write failed");
  require(!writes.run(WriteBehind::kDelayMilliseconds) && !process.tier.demotions &&
              writes.nextWakeup() == kMinute,
          "a due point was written past the hourly limit");
  require(!writes.run(kMinute) && !writes.run(kHour) && !process.tier.demotions,
          "a due point was written while the hour's writes reached the limit");
  require(!writes.run(kHour + kMinute) && process.tier.demotions == 2,
          "a due point waited once the writes were an hour old");
}

} // namespace

int main() {
  try {
    testCopiesAreLabelled();
    testRestorePointsComeBack();
    testBrokenChainsStayBehind();
    testOldestPointGoes();
    testAdoptionTrimsToTheQuota();
    testCleanStopRecordsRecency();
    testPointGivenUpForItselfIsRefused();
    testPointLargerThanTheQuotaIsRefused();
    testPersistingStateStaysInRam();
    testDurablePointsStayInRam();
    testOnlyTheNewestPointsAreKept();
    testRestoresGoFirst();
    testPointsWaitTheirTurn();
    testWritesKeepToTheHourlyLimit();
    std::cout << "Persistent cache tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
