#include "Checked.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Cache.hpp"
#include "engine/DiskLabels.hpp"
#include "model/SlotFile.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
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
// record does.
class ShelvedState;
struct Shelf final {
  std::set<const ShelvedState *> disk;
  uint32_t capacity = 8;
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
    return std::make_unique<Written>(std::make_shared<ShelvedState>(shelf_, true));
  }
  void label(std::vector<std::byte> label) const override { label_ = std::move(label); }
  [[nodiscard]] const std::vector<std::byte> &labelled() const noexcept { return label_; }

private:
  struct Written final : StateOffload {
    explicit Written(std::shared_ptr<const CompositeState> copy) : disk(std::move(copy)) {}
    bool ready() const noexcept override { return true; }
    bool finish() override { return true; }
    const std::shared_ptr<const CompositeState> &state() const noexcept override { return disk; }
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

  void publish(uint64_t block) {
    cache.publishCompositeState(block, std::make_shared<ShelvedState>(shelf, false), false);
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

} // namespace

int main() {
  try {
    testCopiesAreLabelled();
    testRestorePointsComeBack();
    testBrokenChainsStayBehind();
    testOldestPointGoes();
    testAdoptionTrimsToTheQuota();
    std::cout << "Persistent cache tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
