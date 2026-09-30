#pragma once

#include "engine/StateCache.hpp"
#include "model/CacheStore.hpp"
#include <deque>
#include <map>
#include <unordered_map>

namespace splash::engine {
struct PersistentCacheConfig {
  std::shared_ptr<model::DiskBudget> budget;
  std::shared_ptr<model::CacheStore> store;
  std::shared_ptr<model::SlotFile> kvFile;
  model::StateStorage *stateStorage = nullptr;
};
struct PersistentCacheSnapshot {
  uint64_t capacityBytes = 0, usedBytes = 0, readBytes = 0, writtenBytes = 0;
  uint64_t saved = 0, restored = 0, failures = 0;
  uint32_t entries = 0;
  bool writing = false;
};

// A bounded publisher over the existing cache index. It owns manifests and
// durable quota; the model owns capture, and the cache owns matching/restores.
class PersistentCache final {
public:
  PersistentCache(PersistentCacheConfig config, KvCache &kv, StateCache &states,
                  model::KvTier &tier, CacheRecency &recency,
                  std::function<void(uint64_t)> discardKv,
                  const std::function<void()> &completion);
  ~PersistentCache();
  void publish(uint64_t block);
  bool capture(uint64_t block, const RestoreState &source);
  void touch(uint64_t block);
  void invalidate(uint64_t block);
  bool poll();
  bool busy() const noexcept;
  PersistentCacheSnapshot snapshot() const;

private:
  using Slot = model::SlotFile::Slot;
  struct Entry {
    uint64_t id = 0, block = 0, used = 0;
    std::vector<std::shared_ptr<Slot>> records;
    bool touched = false;
  };
  struct Location {
    uint64_t block;
    std::optional<CacheGroupId> group;
    bool temporary = false;
  };
  struct Reference {
    uint32_t count = 0;
    std::vector<Location> locations;
  };
  struct Copy {
    uint64_t block;
    std::unique_ptr<model::KvTransfer> transfer;
  };
  struct Job {
    Entry entry;
    std::optional<RestoreLease> lease;
    std::vector<uint64_t> blocks;
    std::optional<model::DiskReservation> reservation;
    std::shared_ptr<const RestoreState> disk;
    std::vector<Copy> copies;
    std::shared_ptr<model::SlotFile::Operation> commit;
    bool failed = false;
  };
  void load();
  void start(uint64_t block);
  bool begin(uint64_t block, const SnapshotWritePlan &plan,
             std::optional<RestoreLease> lease);
  void finish(bool committed = false);
  bool flushTouches();
  bool evictOldest();
  void forget(std::map<uint64_t, Entry>::iterator entry);
  void hold(Entry &entry, const std::shared_ptr<Slot> &slot, Location location);
  void release(Entry &entry);
  bool stored(const Slot &slot, const Location &location) const;
  model::CacheStore::Prefix describe(const Job &job) const;

  PersistentCacheConfig config_;
  KvCache &kv_;
  StateCache &states_;
  model::KvTier &tier_;
  CacheRecency &recency_;
  std::function<void(uint64_t)> discardKv_;
  const std::function<void()> &completion_;
  std::map<uint64_t, Entry> entries_;
  std::unordered_map<uint64_t, Reference> references_;
  std::deque<uint64_t> pending_;
  std::unique_ptr<Job> job_;
  std::shared_ptr<model::SlotFile::Operation> touches_;
  uint64_t saved_ = 0, restored_ = 0, failures_ = 0;
};
} // namespace splash::engine
