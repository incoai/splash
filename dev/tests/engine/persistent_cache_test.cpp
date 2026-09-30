#include "TestKvPool.hpp"
#include "TestStateSnapshot.hpp"
#include "engine/Cache.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sqlite3.h>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace splash;
using namespace splash::engine;
using namespace splash::model;
namespace {
constexpr uint64_t unit = SlotFile::kAlignmentBytes;
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Directory {
  std::filesystem::path path;
  Directory() {
    auto name = (std::filesystem::temp_directory_path() /
                 "splash-persistence-test-XXXXXX")
                    .string();
    require(mkdtemp(name.data()), "mkdtemp failed");
    path = name;
  }
  ~Directory() { std::filesystem::remove_all(path); }
};

struct DiskSlot final : KvDiskSlot {
  std::shared_ptr<SlotFile::Slot> slot;
  explicit DiskSlot(std::shared_ptr<SlotFile::Slot> value)
      : slot(std::move(value)) {}
  std::shared_ptr<SlotFile::Slot> record() const override { return slot; }
  bool durable() const noexcept override { return slot && slot->durable(); }
};
struct FileTicket final : KvTransfer {
  std::vector<std::byte> data;
  std::shared_ptr<SlotFile::Operation> operation;
  ~FileTicket() override {
    if (operation)
      operation->drain();
  }
  bool ready() const noexcept override { return operation->ready(); }
  bool finish() override { return operation->wait(); }
};
struct FileTier final : KvTier {
  std::shared_ptr<SlotFile> file;
  bool failDemotions = false;
  bool pauseRestores = false;
  explicit FileTier(std::shared_ptr<SlotFile> value) : file(std::move(value)) {}
  uint64_t slotBytes() const noexcept override { return unit; }
  bool writable() const noexcept override { return file->writable(); }
  bool canDemote() const noexcept override { return true; }
  std::shared_ptr<KvDiskSlot> acquireSlot() override {
    auto slot = file->acquire();
    return slot ? std::make_shared<DiskSlot>(slot) : nullptr;
  }
  std::shared_ptr<KvDiskSlot>
  reopenSlot(std::shared_ptr<SlotFile::Slot> slot) override {
    return std::make_shared<DiskSlot>(std::move(slot));
  }
  std::unique_ptr<KvTransfer> demote(uint32_t page,
                                     std::shared_ptr<KvDiskSlot> slot,
                                     std::function<void()> done) override {
    auto ticket = std::make_unique<FileTicket>();
    ticket->data.resize(unit, std::byte(page + 1));
    ticket->operation =
        failDemotions
            ? file->metadata(
                  [] { throw std::runtime_error("injected IO failure"); },
                  std::move(done))
            : file->write(slot->record(), {ticket->data}, std::move(done));
    return ticket;
  }
  std::unique_ptr<KvTransfer> restore(std::shared_ptr<KvDiskSlot> slot,
                                      uint32_t,
                                      std::function<void()> done) override {
    if (pauseRestores)
      return {};
    require(bool(slot), "queued restore lost its disk slot");
    auto ticket = std::make_unique<FileTicket>();
    ticket->data.resize(unit);
    ticket->operation =
        file->read(slot->record(), {ticket->data}, std::move(done));
    return ticket;
  }
  bool copiesQueued() const noexcept override { return false; }
  void poll() override {}
};
struct State final : StatePayload {
  std::shared_ptr<SlotFile> file;
  std::shared_ptr<SlotFile::Slot> slot;
  uint64_t boundary;
  std::vector<std::shared_ptr<SlotFile::Slot>> components;
  State(std::shared_ptr<SlotFile> f, uint64_t b,
        std::shared_ptr<SlotFile::Slot> s = {},
        std::vector<std::shared_ptr<SlotFile::Slot>> c = {})
      : file(std::move(f)), slot(std::move(s)), boundary(b),
        components(std::move(c)) {}
  uint64_t bytes() const noexcept override {
    uint64_t result = unit;
    for (const auto &component : components)
      result += component->bytes();
    return result;
  }
  uint64_t residentBytes() const noexcept override {
    return slot ? 0 : bytes();
  }
  bool canOffload() const noexcept override {
    return file && !slot && file->writable() &&
           file->capacityBytes() >= file->slotBytes();
  }
  DiskStateRecord diskRecord() const override {
    return {slot, {boundary}, components};
  }
  bool durable() const noexcept override { return slot && slot->durable(); }
  std::unique_ptr<StateWrite<StatePayload>>
  offload(std::function<void()> done,
          DiskReservation *reservation = nullptr) const override {
    struct Ticket final : StateWrite<StatePayload> {
      std::vector<std::byte> data;
      std::shared_ptr<const StatePayload> snapshot;
      std::shared_ptr<SlotFile::Operation> operation;
      ~Ticket() override { operation->drain(); }
      bool ready() const noexcept override { return operation->ready(); }
      bool finish() override { return operation->wait(); }
      const std::shared_ptr<const StatePayload> &
      state() const noexcept override {
        return snapshot;
      }
    };
    auto held = reservation ? file->acquire(*reservation) : file->acquire();
    if (!held)
      return {};
    auto ticket = std::make_unique<Ticket>();
    ticket->data.resize(unit, std::byte{0x5a});
    ticket->snapshot =
        std::make_shared<State>(file, boundary, held, components);
    ticket->operation = file->write(held, {ticket->data}, std::move(done));
    return ticket;
  }
};
struct Storage final : StateStorage {
  std::shared_ptr<SlotFile> file;
  std::shared_ptr<SlotFile> componentFile;
  std::vector<CacheGroupSpec> groups{{0}};
  std::vector<CacheGroupSpec> cacheGroups() const override { return groups; }
  explicit Storage(std::shared_ptr<SlotFile> f)
      : file(std::move(f)), componentFile(file->sibling(unit)) {}
  SnapshotWritePlan
  prepareWrite(std::shared_ptr<const RestoreState> source) override {
    return {source,
            [source](std::function<void()> done, DiskReservation *reservation) {
              struct Batch final : StateWrite<RestoreState> {
                std::shared_ptr<const RestoreState> output;
                std::vector<std::unique_ptr<StateWrite<StatePayload>>> copies;
                bool ready() const noexcept override {
                  return std::all_of(
                      copies.begin(), copies.end(),
                      [](const auto &copy) { return copy->ready(); });
                }
                bool finish() override {
                  bool ok = true;
                  for (auto &copy : copies)
                    ok = copy->finish() && ok;
                  return ok;
                }
                const std::shared_ptr<const RestoreState> &
                state() const noexcept override {
                  return output;
                }
              };
              auto result = std::make_unique<Batch>();
              auto state = std::make_shared<RestoreState>(*source);
              for (auto &part : state->blocks) {
                if (!part.payload->residentBytes())
                  continue;
                auto copy = part.payload->offload(done, reservation);
                if (!copy)
                  return std::unique_ptr<Batch>{};
                part.payload = copy->state();
                result->copies.push_back(std::move(copy));
              }
              result->output = std::move(state);
              return result;
            }};
  }
  uint64_t actualAllocatedBytes() const noexcept override { return 0; }
  uint64_t releaseIdle(uint32_t, uint32_t) noexcept override { return 0; }
  std::shared_ptr<const StatePayload>
  reopenState(const CachedStateBlock &, StoredStateRecord record,
              DiskReservation &reservation) override {
    require(record.metadata.size() == 1 && !record.records.empty(),
            "invalid test state manifest");
    const auto boundary = record.metadata.front();
    auto slot = file->reopen(record.records.front().id, reservation);
    std::vector<std::shared_ptr<SlotFile::Slot>> components;
    for (size_t i = 1; i < record.records.size(); ++i)
      components.push_back(
          componentFile->reopen(record.records[i].id, reservation));
    return std::make_shared<State>(file, boundary, std::move(slot),
                                   std::move(components));
  }
};
struct Fixture {
  std::shared_ptr<CacheStore> store;
  std::shared_ptr<DiskBudget> budget, durableBudget;
  std::shared_ptr<SlotFile> kvFile, stateFile;
  Storage storage;
  FileTier tier;
  test::TestKvBacking backing{128, unit};
  KvPool pool{backing};
  Cache cache;
  Fixture(const std::filesystem::path &path, uint64_t durable,
          uint64_t temporary = 0, std::vector<CacheGroupSpec> groups = {{0}})
      : store(durable ? std::make_shared<CacheStore>(path, "test-model")
                      : nullptr),
        budget(std::make_shared<DiskBudget>(temporary)),
        durableBudget(std::make_shared<DiskBudget>(durable)),
        kvFile(std::make_shared<SlotFile>(unit, budget, path.parent_path(),
                                          store)),
        stateFile(std::make_shared<SlotFile>(unit, budget, path.parent_path(),
                                             store)),
        storage(stateFile), tier(kvFile), cache(pool, {}, &tier, budget) {
    storage.groups = std::move(groups);
    cache.configureGroups(storage.groups);
    if (durable)
      cache.enablePersistence({durableBudget, store, kvFile, &storage});
  }
  uint64_t
  publish(uint32_t seed, uint32_t count = 512, bool checkpoint = false,
          std::vector<std::shared_ptr<SlotFile::Slot>> components = {}) {
    std::vector<uint32_t> tokens(count);
    std::iota(tokens.begin(), tokens.end(), seed);
    cache.beginRequest(seed);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!cache.ensureTokens(seed, count).granted()) {
      static_cast<void>(cache.pollTransfers());
      static_cast<void>(cache.reclaimCache(unit, false, false));
      require(std::chrono::steady_clock::now() < deadline, "admission stalled");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    static_cast<void>(cache.publishCommittedBlocks(seed, tokens, count));
    const uint64_t block = cache.blockAt(seed, count);
    auto state = std::make_shared<RestoreState>(RestoreState{count, {}});
    for (const auto &group : storage.groups) {
      if (group.kind == CacheGroupKind::Checkpoint)
        state->blocks.push_back(
            {group.id, count, count,
             std::make_shared<State>(stateFile, count, nullptr, components)});
      else
        for (uint32_t end = count,
                      begin = count - std::min(count, group.windowTokens);
             end > begin; end -= 32)
          state->blocks.push_back({group.id, end - 32, end,
                                   std::make_shared<State>(stateFile, end)});
    }
    cache.publishRestoreState(block, std::move(state), checkpoint);
    cache.endRequest(seed);
    return block;
  }
  CacheLookup lookup(uint32_t seed, uint32_t count = 512) {
    std::vector<uint32_t> tokens(count + 1);
    std::iota(tokens.begin(), tokens.end(), seed);
    return cache.lookup(tokens);
  }
  void hit(uint32_t seed, uint32_t count = 512) {
    auto result = lookup(seed, count);
    cache.recordLookup(result);
  }
  void settle() {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
      const bool progress = cache.pollTransfers();
      if (!progress && !cache.transfersInFlight())
        break;
      require(std::chrono::steady_clock::now() < deadline,
              "persistent job stalled");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(stateFile->metadata([] {})->wait(), "metadata drain failed");
  }
};

void restartAndResize(const std::filesystem::path &path) {
  {
    Fixture f(path, 40 * unit);
    f.publish(1000);
    f.settle();
    f.publish(2000);
    f.settle();
    auto s = f.cache.snapshot();
    require(s.persistent.entries == 2 && s.persistent.saved == 2 &&
                s.persistent.usedBytes == 34 * unit && s.kvTier.usedBytes == 0,
            "RAM-resident prefixes did not persist with separate ownership");
    require(s.persistent.writtenBytes == 34 * unit,
            "publication wrote duplicate payloads");
  }
  {
    Fixture f(path, 17 * unit);
    require(
        f.cache.snapshot().persistent.restored == 1 &&
            f.lookup(2000).resumeBoundary() == 512 &&
            !f.lookup(1000).resumeBoundary(),
        "smaller restart quota did not retain most recent complete manifest");
    require(f.cache.snapshot().pool.pagesPrefix == 0,
            "startup allocated GPU pages");
  }
  {
    Fixture f(path, 40 * unit);
    require(f.lookup(2000).resumeBoundary() == 512,
            "larger quota lost existing payloads");
    f.publish(3000);
    f.settle();
    require(f.cache.snapshot().persistent.entries == 2,
            "enlarged quota did not admit new manifest");
  }
}
void queuedRestoresRetainSlots(const std::filesystem::path &path) {
  {
    Fixture f(path, 17 * unit);
    f.publish(1000);
    f.settle();
  }
  Fixture f(path, 17 * unit);
  auto hit = f.lookup(1000);
  require(hit.resumeBoundary() == 512, "restore fixture did not survive restart");
  f.cache.beginRequest(1);
  f.tier.pauseRestores = true;
  require(f.cache.restoreRequest(1, hit).granted(), "restore admission failed");
  f.cache.discardState(hit.state->kvBlock(), hit.state->state().get());
  require(f.cache.snapshot().persistent.entries == 0 &&
              f.durableBudget->usedBytes() >= 16 * unit,
          "manifest invalidation released slots still queued for reading");
  f.tier.pauseRestores = false;
  f.settle();
  require(f.cache.kvRestoreStatus(1) == KvRestoreStatus::None &&
              f.cache.snapshot().kvTier.restores == 16,
          "queued reads failed after manifest invalidation");
  f.cache.endRequest(1);
  hit = {};
  require(f.durableBudget->usedBytes() == 0,
          "completed reads retained retired durable quota");
}
void quotaAndLRU(const std::filesystem::path &path) {
  Fixture f(path, 17 * unit, 17 * unit);
  f.publish(1000);
  f.settle();
  f.publish(2000);
  f.settle();
  const auto s = f.cache.snapshot();
  require(s.persistent.entries == 1 && s.persistent.usedBytes == 17 * unit &&
              s.persistent.saved == 2 && s.kvTier.usedBytes == 0,
          "manifest LRU borrowed temporary capacity or leaked durable records");
  require(f.lookup(1000).resumeBoundary() == 512,
          "durable eviction removed RAM snapshot");
  f.publish(3000, 480);
  f.settle();
  require(f.cache.snapshot().persistent.saved == 2,
          "admission floor was bypassed");
}
void failedPublication(const std::filesystem::path &path, bool fillTemporary) {
  Fixture f(path, 17 * unit, unit);
  const auto block = f.publish(1000, 512, true);
  require(f.cache.reclaimOneState(CacheGroupId{0}).madeProgress,
          "temporary state offload failed");
  f.settle();
  require(f.budget->usedBytes() == unit,
          "temporary state did not own its quota");
  require(f.cache.reuseStoredState(block),
          "completed request did not upgrade checkpoint");
  f.tier.failDemotions = true;
  f.hit(1000);
  static_cast<void>(f.cache.pollTransfers());
  require(f.durableBudget->usedBytes() == 17 * unit && !f.budget->usedBytes(),
          "publication wrote before reserving all slots");
  auto competing = fillTemporary ? f.stateFile->acquire() : nullptr;
  f.settle();
  require(f.cache.snapshot().persistent.failures == 1 &&
              !f.durableBudget->usedBytes(),
          "failed job retained durable quota");
  require(f.lookup(1000).resumeBoundary() == (fillTemporary ? 0u : 512u),
          "borrowed state did not follow rollback ownership");
  require(f.budget->usedBytes() == unit,
          "rollback exceeded or lost temporary ownership");
  require(f.store->load().empty(), "failed publication committed manifest");
}
void zeroRewriteTransfer(const std::filesystem::path &path) {
  Fixture f(path, 17 * unit, 17 * unit);
  const auto block = f.publish(1000, 512, true);
  for (unsigned n = 0; n < 100 && f.cache.snapshot().kvCache.bytes; ++n) {
    static_cast<void>(f.cache.reclaimCache(17 * unit, false, false));
    f.settle();
  }
  require(f.budget->usedBytes() == 17 * unit, "temporary fixture incomplete");
  const auto written = f.budget->writtenBytes();
  require(f.cache.reuseStoredState(block),
          "completed request did not upgrade checkpoint");
  f.hit(1000);
  f.settle();
  require(f.cache.snapshot().persistent.entries == 1 &&
              !f.budget->usedBytes() &&
              f.durableBudget->usedBytes() == 17 * unit &&
              f.budget->writtenBytes() == written &&
              !f.durableBudget->writtenBytes(),
          "temporary transfer rewrote payloads or double charged slots");
  f.publish(2000);
  f.settle();
  require(f.budget->usedBytes() == 17 * unit &&
              f.durableBudget->usedBytes() == 17 * unit,
          "last manifest reference did not return borrowed slots");
  require(f.lookup(1000).resumeBoundary() == 512,
          "returned temporary prefix was lost");
}
void publicationWaitsForTemporaryIO(const std::filesystem::path &path) {
  Fixture f(path, 17 * unit, 17 * unit);
  const auto block = f.publish(1000, 512, true);
  require(f.cache.reclaimOneState(CacheGroupId{0}).madeProgress,
          "state offload failed");
  f.settle();
  f.tier.failDemotions = true;
  static_cast<void>(f.cache.reclaimOne(CacheReclaimMode::ReuseBacking, false));
  require(f.cache.snapshot().kvTier.pendingPages > 0,
          "temporary IO fixture missing");
  f.tier.failDemotions = false;
  require(f.cache.reuseStoredState(block), "checkpoint reuse failed");
  static_cast<void>(f.cache.pollTransfers());
  require(!f.durableBudget->usedBytes(),
          "publication borrowed an unresolved temporary write");
  f.settle();
  require(f.cache.snapshot().persistent.saved == 1 &&
              !f.cache.snapshot().persistent.failures &&
              f.cache.snapshot().kvTier.demotionFailures > 0,
          "publication did not recover from failed temporary demotion");
}
void sharedPrefixReservation(const std::filesystem::path &path) {
  {
    Fixture f(path, 33 * unit);
    f.publish(1000);
    f.settle();
    f.publish(1000, 1024);
    f.settle();
    const auto snapshot = f.cache.snapshot().persistent;
    require(
        snapshot.entries == 1 && snapshot.usedBytes == 33 * unit &&
            snapshot.saved == 2,
        "reservation evicted shared records needed by the incoming manifest");
    require(snapshot.writtenBytes == 34 * unit,
            "shared target prefix was rewritten");
    require(f.lookup(1000, 1024).resumeBoundary() == 1024,
            "incoming shared manifest was lost");
  }
  {
    Fixture f(path, 33 * unit);
    require(f.lookup(1000, 1024).resumeBoundary() == 1024,
            "shared records failed restart");
  }
}
void oversizedAdmission(const std::filesystem::path &path) {
  Fixture f(path, 16 * unit, 100 * unit);
  f.publish(1000);
  f.settle();
  const auto snapshot = f.cache.snapshot();
  require(
      !snapshot.persistent.entries && !snapshot.persistent.usedBytes &&
          !snapshot.persistent.writtenBytes && !snapshot.kvTier.usedBytes,
      "oversized manifest wrote partial data or borrowed temporary capacity");
  require(f.lookup(1000).resumeBoundary() == 512,
          "rejected manifest lost RAM cache");

  Fixture shared(path.string() + ".shared", 17 * unit);
  shared.publish(1000);
  shared.settle();
  shared.publish(1000, 1024);
  shared.settle();
  const auto kept = shared.cache.snapshot().persistent;
  require(kept.entries == 1 && kept.usedBytes == 17 * unit &&
              kept.writtenBytes == 17 * unit,
          "oversized extension evicted its smaller durable prefix");
}
[[noreturn]] void crashWriter(const std::filesystem::path &path, unsigned phase) {
  Fixture f(path, 17 * unit);
  f.publish(1000);
  if (phase == 1) {
    static_cast<void>(f.cache.pollTransfers());
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!f.stateFile->idle() || !f.kvFile->idle()) {
      require(std::chrono::steady_clock::now() < deadline,
              "payload IO did not finish");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } else if (phase == 2)
    f.settle();
  _exit(0); // No publisher, slot, or database destructors.
}
void abruptExit(const std::filesystem::path &directory, const char *executable) {
  for (unsigned phase = 0; phase < 3; ++phase) {
    const auto phaseText = std::to_string(phase);
    const auto path = directory / ("crash-" + phaseText + ".sqlite");
    const auto child = fork();
    require(child >= 0, "cannot fork crash test");
    if (!child) {
      // A fresh process avoids using SQLite or sanitizer runtime locks inherited
      // from the parent's threads. Only async-signal-safe calls follow fork.
      execl(executable, executable, "--crash-writer", path.c_str(),
            phaseText.c_str(), static_cast<char *>(nullptr));
      _exit(64);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                WEXITSTATUS(status) == 0,
            "crash writer failed before exit");
    Fixture f(path, 17 * unit);
    require(f.cache.snapshot().persistent.restored == (phase == 2 ? 1u : 0u),
            "restart retained an uncommitted manifest or lost a committed one");
    require(f.lookup(1000).resumeBoundary() == (phase == 2 ? 512u : 0u),
            "crash recovery exposed incomplete prefix");
  }
}
void directCaptureReusesPartialGroups(const std::filesystem::path &path) {
  const std::vector<CacheGroupSpec> groups{
      {0}, {1, CacheGroupKind::SlidingWindow, 64}};
  {
    Fixture f(path, 19 * unit, unit, groups);
    std::vector<uint32_t> tokens(512);
    std::iota(tokens.begin(), tokens.end(), 1000);
    f.cache.beginRequest(1);
    require(f.cache.ensureTokens(1, 512).granted(),
            "partial capture fixture failed");
    const auto block = f.cache.publishCommittedBlocks(1, tokens, 512);
    f.cache.publishRestoreState(
        block, test::checkpoint(512, std::make_shared<State>(f.stateFile, 512)),
        true);
    auto checkpoint = f.cache.checkpointState(block);
    f.cache.endRequest(1);
    require(f.cache.reclaimOneState(CacheGroupId{0}).madeProgress,
            "checkpoint offload failed");
    f.settle();
    std::shared_ptr<SlotFile::Slot> original;
    {
      auto partial = f.lookup(1000, 576);
      require(partial.resumeBoundary() == 512, "partial source is missing");
      original =
          partial.state->state()->blocks.front().payload->diskRecord().slot;
    }
    auto source = std::make_shared<RestoreState>(RestoreState{
        512,
        {{0, 512, 512, std::make_shared<State>(f.stateFile, 512)},
         {1, 448, 480, std::make_shared<State>(f.stateFile, 480)},
         {1, 480, 512, std::make_shared<State>(f.stateFile, 512)}}});
    require(f.cache.publishStateToDisk(block, f.storage.prepareWrite(source)),
            "direct capture failed");
    f.settle();
    require(f.cache.retireCheckpointState(checkpoint),
            "stale checkpoint retirement failed");
    auto hit = f.lookup(1000);
    require(hit.resumeBoundary() == 512 && original->durable() &&
                f.durableBudget->writtenBytes() == 18 * unit &&
                f.durableBudget->usedBytes() == 19 * unit,
            "partial capture duplicated an existing group record");
    const auto manifests = f.store->load();
    require(manifests.size() == 1 && std::any_of(manifests[0].records.begin(),
                                                 manifests[0].records.end(),
                                                 [&](const auto &record) {
                                                   return record.id ==
                                                          original->recordId();
                                                 }),
            "live group copy differs from committed manifest");
  }
  {
    Fixture f(path, 19 * unit, 0, groups);
    require(f.lookup(1000).resumeBoundary() == 512,
            "partial capture failed restart");
  }
}
void genericGroups(const std::filesystem::path &directory) {
  const std::vector<std::vector<CacheGroupSpec>> variants{
      {},
      {{19, CacheGroupKind::SlidingWindow, 64}},
      {{3},
       {9},
       {25, CacheGroupKind::SlidingWindow, 64},
       {80, CacheGroupKind::SlidingWindow, 32}}};
  for (size_t i = 0; i < variants.size(); ++i) {
    const auto path = directory / ("groups-" + std::to_string(i) + ".sqlite");
    {
      Fixture f(path, 40 * unit, 0, variants[i]);
      f.publish(1000);
      f.settle();
      require(f.cache.snapshot().persistent.entries == 1,
              "generic group manifest missing");
    }
    {
      Fixture f(path, 40 * unit, 0, variants[i]);
      auto hit = f.lookup(1000);
      require(hit.resumeBoundary() == 512 &&
                  f.cache.snapshot().persistent.restored == 1,
              "generic groups did not restore");
    }
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string_view(argv[1]) == "--crash-writer")
      crashWriter(argv[2], std::stoul(argv[3]));
    Directory directory;
    restartAndResize(directory.path / "restart.sqlite");
    quotaAndLRU(directory.path / "lru.sqlite");
    queuedRestoresRetainSlots(directory.path / "queued-reads.sqlite");
    genericGroups(directory.path);
    directCaptureReusesPartialGroups(directory.path / "partial-capture.sqlite");
    failedPublication(directory.path / "rollback.sqlite", false);
    failedPublication(directory.path / "full-rollback.sqlite", true);
    zeroRewriteTransfer(directory.path / "transfer.sqlite");
    sharedPrefixReservation(directory.path / "shared.sqlite");
    oversizedAdmission(directory.path / "oversized.sqlite");
    abruptExit(directory.path, argv[0]);
    publicationWaitsForTemporaryIO(directory.path / "pending-temp.sqlite");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "persistent cache: PASS\n";
}
