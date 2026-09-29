#include "TestStateSnapshot.hpp"
#include "TestKvPool.hpp"
#include "engine/Cache.hpp"

#include <chrono>
#include <csignal>
#include <iostream>
#include <numeric>
#include <sqlite3.h>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace splash;
using namespace splash::engine;
using namespace splash::model;
namespace {
constexpr uint64_t unit = SlotFile::kAlignmentBytes;
const char *executable = nullptr;
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
        file->write(slot->record(), {ticket->data}, std::move(done));
    return ticket;
  }
  std::unique_ptr<KvTransfer> restore(std::shared_ptr<KvDiskSlot> slot,
                                      uint32_t,
                                      std::function<void()> done) override {
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
  uint64_t offloadBytes() const noexcept override { return slot ? 0 : unit; }
  uint64_t residentBytes() const noexcept override {
    return slot ? 0 : bytes();
  }
  bool canOffload() const noexcept override { return file && !slot; }
  DiskStateRecord diskRecord() const override {
    return {slot, {boundary}, components};
  }
  bool durable() const noexcept override { return slot && slot->durable(); }
  std::unique_ptr<StateOffload>
  offload(std::function<void()> done) const override {
    struct Ticket final : StateOffload {
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
    auto held = file->acquire();
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
  uint64_t actualAllocatedBytes() const noexcept override { return 0; }
  uint64_t releaseIdle(uint32_t, uint32_t) noexcept override { return 0; }
  std::shared_ptr<const StatePayload>
  reopenState(const CachedStateBlock &, StoredStateRecord record) override {
    require(record.metadata.size() == 1 && !record.records.empty(), "invalid test state manifest");
    const auto boundary = record.metadata.front();
    auto slot = file->reopen(record.records.front().id);
    std::vector<std::shared_ptr<SlotFile::Slot>> components;
    for (size_t i = 1; i < record.records.size(); ++i)
      components.push_back(componentFile->reopen(record.records[i].id));
    return std::make_shared<State>(file, boundary, std::move(slot), std::move(components));
  }
};
struct Fixture {
  std::shared_ptr<CacheStore> store;
  std::shared_ptr<DiskBudget> budget;
  std::shared_ptr<SlotFile> kvFile, stateFile;
  Storage storage;
  FileTier tier;
  test::TestKvBacking backing{32, unit};
  KvPool pool{backing};
  Cache cache;
  Fixture(const std::filesystem::path &path, uint64_t durable,
          uint64_t temporary = 0, uint32_t minimumTokens = 0,
          uint64_t writeRate = UINT64_MAX, uint64_t writeBurst = 0,
          std::vector<CacheGroupSpec> groups = {{0}})
      : store(durable ? std::make_shared<CacheStore>(path, "test-model")
                      : nullptr),
        budget(std::make_shared<DiskBudget>(durable + temporary)),
        kvFile(std::make_shared<SlotFile>(unit, budget, path.parent_path(),
                                          store)),
        stateFile(std::make_shared<SlotFile>(unit, budget, path.parent_path(),
                                             store)),
        storage(stateFile), tier(kvFile), cache(pool, {}, &tier, budget) {
    storage.groups = std::move(groups);
    cache.configureGroups(storage.groups);
    // Small synthetic slots exercise storage mechanics independently of the
    // production admission floor. admissionFloor() exercises that policy.
    if (durable)
      cache.enablePersistence({durable, store, kvFile, &storage,
                               minimumTokens, writeRate, writeBurst},
                              temporary != 0);
  }
  uint64_t
  publish(uint32_t seed, uint32_t count = 64, bool checkpoint = false,
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
    auto state = std::make_shared<RestoreState>(RestoreState{count,{}});
    for (const auto &group : storage.groups) {
      if (group.kind == CacheGroupKind::Checkpoint)
        state->blocks.push_back({group.id,count,count,std::make_shared<State>(stateFile,count,nullptr,components)});
      else
        for (uint32_t end = count, begin = count - std::min(count,group.windowTokens); end > begin; end -= 32)
          state->blocks.push_back({group.id,end-32,end,std::make_shared<State>(stateFile,end)});
    }
    cache.publishCompositeState(block, std::move(state), checkpoint);
    cache.endRequest(seed);
    return block;
  }
  CacheLookup lookup(uint32_t seed, uint32_t count = 64) {
    std::vector<uint32_t> tokens(count + 1);
    std::iota(tokens.begin(), tokens.end(), seed);
    return cache.lookup(tokens);
  }
  void hit(uint32_t seed, uint32_t count = 64) {
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

void genericGroups(const std::filesystem::path &directory) {
  const std::vector<std::vector<CacheGroupSpec>> variants{
    {}, {{19,CacheGroupKind::SlidingWindow,64}},
    {{3},{9},{25,CacheGroupKind::SlidingWindow,64},{80,CacheGroupKind::SlidingWindow,32}}};
  for (size_t i = 0; i < variants.size(); ++i) {
    const auto path = directory / ("groups-" + std::to_string(i) + ".sqlite");
    {
      Fixture f(path, 16*unit, 0, 0, UINT64_MAX, 0, variants[i]);
      f.publish(100); f.settle();
      require(f.cache.snapshot().persistent.entries == 1, "generic groups were not persisted");
    }
    {
      Fixture f(path, 16*unit, 0, 0, UINT64_MAX, 0, variants[i]);
      auto hit = f.lookup(100);
      require(hit.resumeBoundary() == 64 && f.cache.snapshot().persistent.restored == 1,
              "generic groups failed restart without a distinguished recurrent record");
      const auto &parts = hit.state->state()->blocks;
      require(variants[i].empty() ? parts.empty() : !parts.empty(), "manifest lost model groups");
      for (const auto &part : parts) {
        std::vector<std::byte> bytes(unit);
        require(f.stateFile->read(part.payload->diskRecord().slot, {bytes}, {})->wait() &&
                bytes.front() == std::byte{0x5a}, "group payload failed verification after restart");
      }
    }
  }
}

void sharedComponents(const std::filesystem::path &path) {
  {
    Fixture f(path, 7 * unit);
    auto block = f.storage.componentFile->acquire();
    std::vector<std::byte> payload(unit, std::byte{0x71});
    require(f.storage.componentFile->write(block, {payload}, {})->wait(),
            "shared component write failed");
    f.publish(100, 64, false, {block});
    f.settle();
    f.publish(200, 64, false, {block});
    f.settle();
    require(f.cache.snapshot().persistent.usedBytes == 7 * unit &&
                f.cache.snapshot().persistent.entries == 2,
            "persistent component charged once per prefix");
  }
  {
    Fixture f(path, 7 * unit);
    require(f.cache.snapshot().persistent.restored == 2 &&
                f.budget->usedBytes() == 7 * unit,
            "shared components did not reopen within the unique quota");
    auto first = f.lookup(100), second = f.lookup(200);
    require(first.resumeBoundary() == 64 && second.resumeBoundary() == 64,
            "component-backed prefixes lost their restore points");
    auto a = first.state->state()->blocks.front().payload->diskRecord(),
         b = second.state->state()->blocks.front().payload->diskRecord();
    require(a.components.size() == 1 && a.components[0] == b.components[0],
            "reopened prefixes do not share component ownership");
  }
  {
    Fixture f(path, 4 * unit);
    require(f.cache.snapshot().persistent.restored == 1 &&
                f.budget->usedBytes() == 4 * unit,
            "quota resize did not retain a complete component-backed prefix");
    auto hit = f.lookup(200);
    require(hit.resumeBoundary() == 64,
            "resize evicted the newest complete prefix");
    auto record = hit.state->state()->blocks.front().payload->diskRecord();
    std::vector<std::byte> payload(unit);
    require(f.storage.componentFile->read(record.components[0], {payload}, {})
                    ->wait() &&
                payload.front() == std::byte{0x71} &&
                payload.back() == std::byte{0x71},
            "evicting one prefix retired a shared component still in use");
  }
}

void pressureAdmission(const std::filesystem::path &path) {
  {
    Fixture f(path, 6 * unit, 6 * unit);
    f.publish(100);
    f.settle();
    f.publish(200);
    f.settle();
    const auto before = f.budget->writtenBytes();
    for (uint32_t i = 0; i < 14; ++i) {
      f.publish(1000 + i * 100);
      f.settle();
    }
    require(f.budget->writtenBytes() == before,
            "one-use scan wrote payload under pressure");
    require(f.cache.snapshot().persistent.admissionSkips == 14,
            "scan skips were not observable");
    f.hit(2300);
    f.settle(); // real reuse adapts to the new working set
    require(f.budget->writtenBytes() > before,
            "reused tail was never admitted");
    f.hit(200);
    f.settle();
  }
  Fixture reopened(path, 6 * unit);
  require(reopened.lookup(200).state && reopened.lookup(2300).state,
          "pressure admission did not retain the reused working set");
}

void writePacing(const std::filesystem::path &path) {
  {
    Fixture f(path, 9 * unit, 0, 0, 0, 3 * unit);
    f.publish(100);
    f.settle();
    const auto before = f.budget->writtenBytes();
    f.publish(200);
    f.hit(200);
    f.settle();
    require(f.budget->writtenBytes() == before && f.store->load().size() == 1,
            "exhausted write credit still copied payload");
    require(f.cache.snapshot().persistent.writeThrottles == 1 &&
                !f.cache.snapshot().persistent.failures && f.lookup(200).state,
            "write pacing failed or disabled the RAM cache");
    f.hit(100);
    f.settle();
    require(f.budget->writtenBytes() == before,
            "durable hit spent payload credit");
  }
  Fixture reopened(path, 9 * unit);
  require(reopened.lookup(100).state && !reopened.lookup(200).state,
          "write pacing damaged an earlier durable prefix");
}

void admissionFloor(const std::filesystem::path &path) {
  const uint32_t minimum = PersistentCacheConfig{}.minimumTokens;
  {
    Fixture f(path, 64 * unit, 0, minimum);
    f.publish(1000, minimum - KvCache::pageTokens);
    f.settle();
    require(f.store->load().empty() && f.budget->writtenBytes() == 0,
            "short prefix consumed durable storage");
    require(f.lookup(1000, minimum - KvCache::pageTokens).resumeBoundary() ==
                minimum - KvCache::pageTokens,
            "admission floor disabled the RAM cache");
    f.publish(2000, minimum);
    f.settle();
    require(f.store->load().size() == 1,
            "minimum-length prefix was not persisted");
    const auto written = f.budget->writtenBytes();
    for (uint32_t i = 0; i < 14; ++i) {
      f.publish(10000 + i * 100, KvCache::pageTokens);
      f.settle();
      {
        auto hit = f.lookup(10000 + i * 100, KvCache::pageTokens);
        f.cache.recordLookup(hit);
      }
      f.settle();
    }
    require(f.budget->writtenBytes() == written && f.store->load().size() == 1,
            "short publications or RAM hits polluted durable storage");
  }
  Fixture reopened(path, 64 * unit, 0, minimum);
  require(reopened.lookup(2000, minimum).resumeBoundary() == minimum,
          "short scan displaced the reusable prefix across restart");
}

void runtimeAndRestart(const std::filesystem::path &path) {
  {
    Fixture first(path, 9 * unit);
    const auto block = first.publish(100);
    first.settle();
    const auto stats = first.cache.snapshot();
    require(stats.persistent.saved == 1 &&
                stats.persistent.usedBytes == 3 * unit,
            "RAM prefix not saved during runtime");
    require(first.cache.stateResident(block) && stats.kvCache.bytes == 2 * unit,
            "write-through evicted RAM");
    require(first.store->load().size() == 1,
            "publication was deferred to exit");
    first.publish(200, 64, true);
    first.settle();
    {
      auto checkpoint = first.lookup(200);
      first.cache.recordLookup(checkpoint);
    }
    first.settle();
    require(first.store->load().size() == 1,
            "disposable checkpoint became durable");
  }
  {
    Fixture second(path, 12 * unit);
    require(second.cache.snapshot().persistent.restored == 1,
            "growing quota lost prefix");
    require(second.cache.snapshot().kvCache.bytes == 0,
            "startup eagerly allocated GPU KV");
    auto hit = second.lookup(100);
    require(hit.resumeBoundary() == 64 && !hit.state->state()->resident(),
            "restart missed durable state");
    auto record = hit.state->state()->blocks.front().payload->diskRecord();
    std::vector<std::byte> state(unit);
    require(second.stateFile->read(record.slot, {state}, {})->wait(),
            "state payload failed to restore");
    require(state.front() == std::byte{0x5a} && state.back() == std::byte{0x5a},
            "state payload changed");
    second.cache.beginRequest(1);
    require(second.cache.restoreRequest(1, hit).granted(),
            "disk KV restore admission failed");
    second.settle();
    require(second.cache.kvRestoreStatus(1) != KvRestoreStatus::Failed,
            "disk KV restore failed");
    second.cache.endRequest(1);
    require(second.lookup(101).resumeBoundary() == 0,
            "nearby tokens falsely matched");
  }
}

void lruAndQuotas(const std::filesystem::path &path) {
  {
    Fixture f(path, 6 * unit, 6 * unit);
    f.publish(100);
    f.settle();
    f.publish(200);
    f.settle();
    {
      auto hit = f.lookup(100);
      f.cache.recordLookup(hit);
    }
    f.settle();
  }
  {
    Fixture f(path, 3 * unit);
    require(f.lookup(100).resumeBoundary() == 64, "shrink ignored last access");
    require(f.lookup(200).resumeBoundary() == 0, "shrink retained LRU victim");
    require(f.cache.snapshot().persistent.usedBytes == 3 * unit,
            "shrink quota incorrect");
    f.publish(300, 96);
    f.settle();
    require(f.lookup(100).resumeBoundary() == 64 && f.store->load().size() == 1,
            "oversized prefix evicted fitting entry");
  }
  // Temporary-only opens no durable store and does not erase the old file.
  {
    Fixture f(path, 0, 8 * unit);
    f.publish(400);
    f.settle();
  }
  {
    Fixture f(path, 6 * unit);
    require(f.lookup(100).resumeBoundary() == 64,
            "disabled persistence deleted data");
  }
}

void metadataWritingStatus(const std::filesystem::path &path) {
  Fixture f(path, 6 * unit);
  f.publish(100);
  f.settle();
  f.publish(200);
  f.settle();
  require(!f.cache.snapshot().persistent.writing, "idle cache reports writing");
  const auto before = f.budget->writtenBytes();
  f.hit(100);
  require(f.cache.snapshot().persistent.writing,
          "pending recency update was reported as settled");
  static_cast<void>(f.cache.pollTransfers());
  require(f.cache.snapshot().persistent.writing,
          "submitted recency update was reported as settled");
  f.settle();
  require(!f.cache.snapshot().persistent.writing,
          "completed recency update still reports writing");
  require(f.budget->writtenBytes() == before, "recency update copied payload");
}

void atomicAccessBatch(const std::filesystem::path &path) {
  std::vector<uint64_t> ids;
  {
    Fixture f(path, 9 * unit);
    for (uint32_t seed : {100u, 200u, 300u}) {
      f.publish(seed);
      f.settle();
    }
    for (const auto &prefix : f.store->load())
      ids.push_back(prefix.id);
    require(ids.size() == 3, "batch fixture did not persist three prefixes");
    const std::array<uint64_t, 2> accessed{ids[1], ids[0]};
    f.store->touch(accessed);
    const auto ordered = f.store->load();
    require(ordered[0].id == ids[2] && ordered[1].id == ids[1] &&
                ordered[2].id == ids[0],
            "access batch lost its order");
  }
  // Inject failure on the second UPDATE. An earlier UPDATE in the same
  // access batch must not survive on its own, including across restart.
  sqlite3 *db = nullptr;
  require(sqlite3_open(path.c_str(), &db) == SQLITE_OK,
          "batch fixture open failed");
  const std::string trigger =
      "CREATE TRIGGER fail_touch BEFORE UPDATE OF used ON prefixes WHEN "
      "NEW.id=" +
      std::to_string(ids[0]) +
      " BEGIN SELECT RAISE(ABORT,'injected touch failure'); END";
  const int created =
      sqlite3_exec(db, trigger.c_str(), nullptr, nullptr, nullptr);
  sqlite3_close(db);
  require(created == SQLITE_OK, "batch failure fixture failed");
  {
    CacheStore store(path, "test-model");
    // The trigger rejects any UPDATE to the newest prefix. Repeating its
    // access must preserve order without issuing a database write.
    store.touch({});
    const std::array<uint64_t, 4> unchanged{0, ids[0], UINT64_MAX, ids[0]};
    store.touch(unchanged);
    bool failed = false;
    try {
      const std::array<uint64_t, 2> accessed{ids[2], ids[0]};
      store.touch(accessed);
    } catch (const std::runtime_error &) {
      failed = true;
    }
    require(failed, "batch fixture did not fail the second update");
  }
  CacheStore reopened(path, "test-model");
  const auto ordered = reopened.load();
  require(ordered.size() == 3 && ordered[0].id == ids[2] &&
              ordered[1].id == ids[1] && ordered[2].id == ids[0],
          "failed metadata batch partially changed durable recency");
}

void sharedPrefixes(const std::filesystem::path &path) {
  {
    Fixture f(path, 5 * unit, 4 * unit);
    f.publish(100);
    f.settle();
    f.publish(100, 96);
    f.settle();
    require(f.cache.snapshot().persistent.entries == 2,
            "shared prefix publication was lost");
    require(f.cache.snapshot().persistent.usedBytes == 5 * unit,
            "shared KV pages were charged twice");
    const auto written = f.budget->writtenBytes();
    require(f.cache.reuseCompositeState(f.lookup(100, 96).state->kvBlock()),
            "RAM prefix disappeared");
    f.settle();
    require(f.budget->writtenBytes() == written,
            "hot prefix was written again");
  }
  {
    Fixture f(path, 5 * unit);
    require(f.cache.snapshot().persistent.restored == 2 &&
                f.budget->usedBytes() == 5 * unit,
            "restart duplicated shared slots");
    require(f.lookup(100, 64).resumeBoundary() == 64 &&
                f.lookup(100, 96).resumeBoundary() == 96,
            "shared-prefix restart lost a boundary");
  }
}

void badMetadata(const std::filesystem::path &path) {
  {
    Fixture f(path, 4 * unit);
    f.publish(100);
    f.settle();
  }
  sqlite3 *db = nullptr;
  require(sqlite3_open(path.c_str(), &db) == SQLITE_OK,
          "metadata fixture open failed");
  require(sqlite3_exec(
              db, "UPDATE prefixes SET metadata=zeroblob(length(metadata))",
              nullptr, nullptr, nullptr) == SQLITE_OK,
          "metadata fixture corruption failed");
  sqlite3_close(db);
  {
    Fixture f(path, 4 * unit);
    require(!f.cache.snapshot().persistent.entries && !f.lookup(100).state,
            "bad metadata was restored");
  }
}

void runChild(const std::filesystem::path &path, const char *mode) {
  const pid_t child = fork();
  require(child >= 0, "fork failed");
  if (!child) {
    // exec gives the crash writer a fresh process (and sanitizer runtime).
    // Running SQLite/CommonCrypto after fork without exec is not safe.
    execl(executable, executable, mode, path.c_str(), nullptr);
    _exit(127);
  }
  int status;
  require(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              !WEXITSTATUS(status),
          "crash fixture failed");
}

void crashAndIntegrity(const std::filesystem::path &path) {
  runChild(path, "--crash-writer");
  {
    Fixture f(path, 9 * unit);
    require(f.lookup(500).resumeBoundary() == 64,
            "abrupt exit lost committed prefix");
    require(f.store->load().size() == 1, "incomplete prefix was published");
    bool locked = false;
    try {
      CacheStore other(path, "test-model");
    } catch (...) {
      locked = true;
    }
    require(locked, "concurrent process opened same cache");
  }
  sqlite3 *db = nullptr;
  require(sqlite3_open(path.c_str(), &db) == SQLITE_OK,
          "cannot open corruption fixture");
  require(sqlite3_exec(db, "UPDATE slots SET data=zeroblob(length(data))",
                       nullptr, nullptr, nullptr) == SQLITE_OK,
          "cannot corrupt payload");
  sqlite3_close(db);
  {
    Fixture f(path, 9 * unit);
    auto hit = f.lookup(500);
    auto record = hit.state->state()->blocks.front().payload->diskRecord();
    std::vector<std::byte> data(unit);
    require(!f.stateFile->read(record.slot, {data}, {})->wait(),
            "damaged payload silently restored");
  }
  {
    CacheStore other(path, "wrong-model");
    require(other.load().empty(), "model namespace mismatch reused a prefix");
  }
}
void failedWrite(const std::filesystem::path &path) {
  runChild(path, "--failed-write");
  Fixture f(path, 9 * unit);
  require(f.lookup(500).resumeBoundary() == 64,
          "failed write lost an earlier committed prefix");
}

void churnAndRestart(const std::filesystem::path &path) {
  // More publications than either RAM or SSD can hold. Exercise graph pruning,
  // shared budget reclamation and index ownership across repeated lifetimes.
  uint32_t last = 0;
  for (uint32_t epoch = 0; epoch < 12; ++epoch) {
    Fixture f(path, (epoch % 2 ? 9 : 6) * unit, epoch % 3 ? 0 : 3 * unit);
    if (last)
      require(f.lookup(last).resumeBoundary() == 64,
              "churn lost newest prefix on restart");
    for (uint32_t i = 0; i < 80; ++i) {
      last = 1000 + (epoch * 80 + i) * 100;
      f.publish(last);
      f.hit(last);
      f.settle();
      auto status = f.cache.snapshot();
      require(!status.persistent.failures, "churn persistence failure");
      require(status.persistent.usedBytes <= status.persistent.capacityBytes,
              "churn exceeded durable quota");
      require(f.budget->usedBytes() <= f.budget->capacityBytes(),
              "churn exceeded shared quota");
      require(f.store->load().size() == status.persistent.entries,
              "churn index ownership diverged");
    }
  }
}

void queuedPublications(const std::filesystem::path &path) {
  Fixture f(path, 60 * unit);
  // Candidates are bounded and may disappear under pressure before copying.
  for (uint32_t i = 0; i < 40; ++i)
    f.publish(1000 + i * 100);
  f.settle();
  require(!f.cache.snapshot().persistent.failures, "burst persistence failed");
  require(f.store->load().size() <= 16, "publication queue grew without bound");
  require(f.cache.snapshot().persistent.saved,
          "burst did not publish anything");
}

void temporaryDuplicateBeforeDurable(const std::filesystem::path &path) {
  {
    // One durable prefix fills its own quota; one temporary state fills the
    // remaining slot. Refresh the latter in RAM, then offload another state.
    // Dropping the redundant temporary copy must precede losing crash recovery.
    Fixture f(path, 3 * unit, unit, 0, 0, 3 * unit);
    f.publish(100);
    f.settle();
    const auto duplicate = f.publish(200, 64, true);
    require(f.cache.reclaimOneState(), "temporary fixture did not offload");
    f.settle();
    test::publishCheckpoint(f.cache,
        duplicate, std::make_shared<State>(f.stateFile, 64), true);
    require(f.cache.reuseCompositeState(duplicate),
            "temporary RAM copy was not reusable");
    f.publish(300, 64, true);
    require(f.cache.reclaimOneState(), "new temporary state did not offload");
    f.settle();
    require(f.cache.snapshot().persistent.entries == 1,
            "redundant temporary copy displaced a committed durable prefix");
    require(f.cache.stateResident(duplicate),
            "discarding a duplicate lost its RAM state");
  }
  Fixture reopened(path, 3 * unit);
  require(
      reopened.lookup(100).resumeBoundary() == 64,
      "temporary replacement lost the previously committed prefix on restart");
}

void readmitHotRam(const std::filesystem::path &path) {
  {
    Fixture f(path, 3 * unit, 6 * unit);
    f.publish(100);
    f.settle();
    f.publish(200);
    f.hit(200);
    f.settle();
    require(f.cache.snapshot().persistent.entries == 1,
            "fixture did not evict durable ownership");
    {
      auto hit = f.lookup(100);
      require(hit.state && hit.state->state()->resident(),
              "fixture lost hot RAM state");
      f.cache.recordLookup(hit);
    }
    f.settle();
  }
  Fixture f(path, 3 * unit);
  require(f.lookup(100).resumeBoundary() == 64,
          "RAM hit did not readmit evicted durable prefix");
  require(!f.lookup(200).state,
          "less recent durable prefix survived readmission");
}

void physicalShrink(const std::filesystem::path &path) {
  {
    Fixture f(path, 60 * unit);
    for (uint32_t i = 0; i < 20; ++i) {
      f.publish(1000 + 100 * i);
      f.settle();
    }
  }
  const auto before = std::filesystem::file_size(path);
  {
    Fixture f(path, 3 * unit);
    require(f.cache.snapshot().persistent.entries == 1,
            "shrink did not keep exactly one entry");
    require(std::filesystem::file_size(path) < before / 2,
            "quota shrink retained physical high-water allocation");
  }
}

void foreignFile(const std::filesystem::path &path) {
  sqlite3 *db = nullptr;
  require(sqlite3_open(path.c_str(), &db) == SQLITE_OK,
          "foreign fixture open failed");
  require(sqlite3_exec(
              db,
              "CREATE TABLE unrelated(value); INSERT INTO unrelated VALUES(42)",
              nullptr, nullptr, nullptr) == SQLITE_OK,
          "foreign fixture failed");
  sqlite3_close(db);
  bool rejected = false;
  try {
    CacheStore store(path, "test-model");
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "foreign database accepted as prefix cache");
  require(sqlite3_open(path.c_str(), &db) == SQLITE_OK,
          "foreign fixture reopen failed");
  sqlite3_stmt *statement = nullptr;
  require(sqlite3_prepare_v2(db, "SELECT value FROM unrelated", -1, &statement,
                             nullptr) == SQLITE_OK,
          "foreign database was overwritten");
  require(sqlite3_step(statement) == SQLITE_ROW &&
              sqlite3_column_int(statement, 0) == 42,
          "foreign database content changed");
  sqlite3_finalize(statement);
  sqlite3_close(db);
}

void migrateWal(const std::filesystem::path &path) {
  {
    Fixture f(path, 6 * unit);
    f.publish(100);
    f.settle();
  }
  const pid_t child = fork();
  require(child >= 0, "WAL migration fork failed");
  if (!child) {
    execl(executable, executable, "--wal-writer", path.c_str(), nullptr);
    _exit(127);
  }
  int status;
  require(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0,
          "WAL fixture failed");
  require(std::filesystem::exists(path.string() + "-wal"),
          "WAL fixture was checkpointed early");
  Fixture reopened(path, 6 * unit);
  require(reopened.lookup(100).resumeBoundary() == 64,
          "WAL migration lost its prefix");
  std::atomic<bool> cancelled{false};
  for (const auto &prefix : reopened.store->load())
    for (const auto &record : prefix.records) {
      std::vector<std::byte> bytes(record.bytes);
      require(reopened.store->read(record.id, {bytes}, cancelled),
              "WAL migration changed payload");
    }
}

void checksumContract(const std::filesystem::path &path) {
  // The standard CRC32C check value also verifies incremental, unaligned
  // spans: storage writes need not share the reader's chunk boundaries.
  const std::string data = "123456789";
  const auto bytes = std::as_bytes(std::span(data));
  std::atomic<bool> cancelled{false};
  {
    CacheStore store(path, "checksum-test");
    const auto id = store.allocate();
    require(store.write(id, {bytes.first(3), bytes.subspan(3)}, cancelled),
            "split checksum write failed");
    store.save({id, {1}, {{id, data.size()}}});
    std::vector<std::byte> restored(data.size());
    require(store.read(id, {restored}, cancelled) &&
                std::equal(bytes.begin(), bytes.end(), restored.begin()),
            "checksum depended on span boundaries");
  }
  sqlite3 *db = nullptr;
  require(sqlite3_open(path.c_str(), &db) == SQLITE_OK,
          "checksum fixture open failed");
  sqlite3_stmt *statement = nullptr;
  require(sqlite3_prepare_v2(db, "SELECT checksum FROM slots", -1, &statement,
                             nullptr) == SQLITE_OK,
          "checksum fixture query failed");
  require(sqlite3_step(statement) == SQLITE_ROW &&
              sqlite3_column_int64(statement, 0) == 0xe3069283,
          "payload checksum is not standard CRC32C");
  sqlite3_finalize(statement);
  sqlite3_close(db);
}

void interruptedPublications(const std::filesystem::path &path) {
  for (uint32_t round = 0; round < 16; ++round) {
    int ready[2];
    require(pipe(ready) == 0, "crash notification pipe failed");
    const std::string descriptor = std::to_string(ready[1]);
    const pid_t child = fork();
    require(child >= 0, "fork failed");
    if (!child) {
      close(ready[0]);
      execl(executable, executable, "--churn-writer", path.c_str(),
            descriptor.c_str(), nullptr);
      _exit(127);
    }
    close(ready[1]);
    char notification;
    require(read(ready[0], &notification, 1) == 1,
            "crash writer did not publish");
    close(ready[0]);
    std::this_thread::sleep_for(std::chrono::microseconds(500 + round * 731));
    require(kill(child, SIGKILL) == 0, "cannot interrupt crash writer");
    int status;
    require(waitpid(child, &status, 0) == child && WIFSIGNALED(status),
            "writer was not interrupted");
    Fixture f(path, 9 * unit);
    const auto prefixes = f.store->load();
    require(!prefixes.empty(), "interrupted write lost all committed prefixes");
    std::atomic<bool> cancelled{false};
    for (const auto &prefix : prefixes) {
      size_t cursor = 3;
      for (uint64_t group = 0; group < prefix.metadata.at(2); ++group)
        cursor += 5 + prefix.metadata.at(cursor + 3) + 2 * prefix.metadata.at(cursor + 4);
      require(f.lookup(static_cast<uint32_t>(prefix.metadata.at(cursor + 4)))
                      .resumeBoundary() == 64,
              "interrupted publication lost its graph");
      for (const auto &record : prefix.records) {
        std::vector<std::byte> bytes(record.bytes);
        require(f.store->read(record.id, {bytes}, cancelled),
                "interrupted publication exposed partial payload");
      }
    }
  }
}

} // namespace
int main(int argc, char **argv) {
  executable = argv[0];
  try {
    if (argc == 3 && std::string_view(argv[1]) == "--wal-writer") {
      sqlite3 *db = nullptr;
      require(sqlite3_open(argv[2], &db) == SQLITE_OK,
              "WAL fixture open failed");
      require(
          sqlite3_exec(db,
                       "PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;"
                       "UPDATE prefixes SET used=used+1",
                       nullptr, nullptr, nullptr) == SQLITE_OK,
          sqlite3_errmsg(db));
      _exit(0); // Leave the committed WAL for the new store to recover.
    }
    if (argc == 4 && std::string_view(argv[1]) == "--churn-writer") {
      Fixture f(argv[2], 9 * unit);
      for (uint32_t i = 0; i < 10000; ++i) {
        f.publish(1000 + 100 * i);
        f.hit(1000 + 100 * i);
        f.settle();
        if (!i) {
          const char ready = '1';
          require(write(std::stoi(argv[3]), &ready, 1) == 1, "notify failed");
        }
      }
      _exit(0);
    }
    if (argc == 3 && std::string_view(argv[1]) == "--failed-write") {
      std::signal(SIGXFSZ, SIG_DFL);
      Fixture f(argv[2], 9 * unit);
      f.publish(500);
      f.settle();
      rlimit limit{0, 0};
      require(setrlimit(RLIMIT_FSIZE, &limit) == 0, "setrlimit failed");
      auto slot = f.kvFile->acquire();
      std::vector<std::byte> data(unit, std::byte{0x77});
      require(!f.kvFile->write(slot, {data}, {})->wait() &&
                  !f.kvFile->writable(),
              "persistent file-size failure did not stop writes safely");
      _exit(0);
    }
    if (argc == 3 && std::string_view(argv[1]) == "--crash-writer") {
      Fixture f(argv[2], 9 * unit);
      f.publish(500);
      f.settle();
      auto orphan = f.kvFile->acquire();
      std::vector<std::byte> data(unit, std::byte{0x22});
      require(f.kvFile->write(orphan, {data}, {})->wait(),
              "orphan write failed");
      _exit(0); // no destructors or exit export
    }
    Directory directory;
    genericGroups(directory.path);
    admissionFloor(directory.path / "admission.sqlite");
    sharedComponents(directory.path / "components.sqlite");
    pressureAdmission(directory.path / "pressure.sqlite");
    writePacing(directory.path / "pacing.sqlite");
    runtimeAndRestart(directory.path / "runtime.sqlite");
    lruAndQuotas(directory.path / "lru.sqlite");
    sharedPrefixes(directory.path / "shared.sqlite");
    atomicAccessBatch(directory.path / "atomic-access.sqlite");
    metadataWritingStatus(directory.path / "metadata-status.sqlite");
    badMetadata(directory.path / "metadata.sqlite");
    crashAndIntegrity(directory.path / "crash.sqlite");
    failedWrite(directory.path / "limited.sqlite");
    churnAndRestart(directory.path / "churn.sqlite");
    queuedPublications(directory.path / "burst.sqlite");
    temporaryDuplicateBeforeDurable(directory.path /
                                    "temporary-duplicate.sqlite");
    readmitHotRam(directory.path / "readmit.sqlite");
    physicalShrink(directory.path / "shrink.sqlite");
    foreignFile(directory.path / "foreign.sqlite");
    migrateWal(directory.path / "migration.sqlite");
    checksumContract(directory.path / "checksum.sqlite");
    interruptedPublications(directory.path / "interrupt.sqlite");
    std::cout << "Persistent cache tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
