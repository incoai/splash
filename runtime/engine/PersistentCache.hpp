#pragma once

#include "engine/StateCache.hpp"
#include "model/CacheStore.hpp"

#include <chrono>
#include <deque>
#include <map>
#include <unordered_map>

namespace splash::engine {

struct PersistentCacheConfig {
  uint64_t capacityBytes = 0;
  std::shared_ptr<model::CacheStore> store;
  std::shared_ptr<model::SlotFile> kvFile;
  model::StateStorage *stateStorage = nullptr;
  uint32_t minimumTokens = 512;
  // Payload write pacing; the token bucket permits a bounded initial burst.
  uint64_t writeBytesPerSecond = (128ULL << 30) / 3600;
  uint64_t writeBurstBytes = 0; // zero selects one full cache capacity
};

struct PersistentCacheSnapshot {
  uint64_t capacityBytes = 0;
  uint64_t usedBytes = 0;
  uint64_t saved = 0;
  uint64_t restored = 0;
  uint64_t failures = 0;
  uint32_t entries = 0;
  bool writing = false;
  uint64_t admissionSkips = 0;
  uint64_t writeThrottles = 0;
};

// Admission/publication only. Matching, RAM residency and restores remain
// in KvCache/StateCache. One bounded job copies existing immutable snapshots;
// it never captures new model state or replays a prompt to manufacture one.
class PersistentCache final {
public:
  PersistentCache(PersistentCacheConfig config, KvCache &kv, StateCache &states,
                  model::KvTier &tier, CacheRecency &recency,
                  std::function<bool()> makeRoom,
                  const std::function<void()> &completion);
  ~PersistentCache();
  void publish(uint64_t block, bool reused = false);
  void touch(uint64_t block);
  void invalidate(uint64_t block);
  bool poll();
  bool evictOldest();
  std::optional<CacheEvictionCandidate> evictionCandidate() const noexcept;
  bool busy() const noexcept { return job_ || touches_; }
  PersistentCacheSnapshot snapshot() const;

private:
  struct Entry {
    uint64_t id = 0;
    uint64_t block = 0;
    uint64_t used = 0;
    std::vector<std::shared_ptr<model::SlotFile::Slot>> records;
    bool touched = false;
  };
  struct Copy {
    uint64_t block;
    std::unique_ptr<model::KvTransfer> transfer;
  };
  struct Job {
    RestoreLease state;
    std::vector<uint64_t> blocks;
    std::vector<Copy> copies;
    std::shared_ptr<model::SlotFile::Operation> commit;
    std::optional<Entry> entry;
    bool failed = false;
  };
  void load();
  bool flushTouches();
  void start(uint64_t block, bool reused);
  void finish();
  void remember(Entry entry);
  void forget(std::map<uint64_t, Entry>::iterator entry);
  model::CacheStore::Prefix describe(Entry &entry, const RestoreState &state,
                                     std::span<const uint64_t> blocks) const;
  uint64_t additionalBytes(const Entry &entry) const;

  PersistentCacheConfig config_;
  KvCache &kv_;
  StateCache &states_;
  model::KvTier &tier_;
  CacheRecency &recency_;
  std::function<bool()> makeRoom_;
  const std::function<void()> &completion_;
  std::map<uint64_t, Entry> entries_; // keyed by current process's leaf block
  struct RecordReference {
    uint32_t count = 0;
    uint64_t bytes = 0;
  };
  std::unordered_map<uint64_t, RecordReference> references_;
  uint64_t used_ = 0;
  struct Candidate {
    uint64_t block;
    bool reused;
  };
  std::deque<Candidate> pending_;
  std::chrono::steady_clock::time_point writeRefill_ =
      std::chrono::steady_clock::now();
  double writeCredit_ = 0;
  uint64_t admissionSkips_ = 0, writeThrottles_ = 0;
  std::unique_ptr<Job> job_;
  std::shared_ptr<model::SlotFile::Operation> touches_;
  uint64_t saved_ = 0, restored_ = 0, failures_ = 0;
};
} // namespace splash::engine
