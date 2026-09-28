#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace splash::model {

// Shared backing for durable and temporary slots. SQLite owns allocation,
// crash recovery and atomic publication; the engine owns admission and LRU.
// Payload IO runs only on SlotFile workers. Opening/loading happens at startup.
class CacheStore final {
public:
  struct Record {
    uint64_t id = 0;
    uint64_t bytes = 0;
  };
  struct Prefix {
    uint64_t id = 0;
    std::vector<uint64_t> metadata;
    std::vector<Record> records;
  };
  CacheStore(const std::filesystem::path &path, const std::string &identity);
  ~CacheStore();
  CacheStore(const CacheStore &) = delete;
  CacheStore &operator=(const CacheStore &) = delete;

  uint64_t allocate() noexcept;
  // Called when the last live handle leaves. Collection never removes a
  // record referenced by a committed prefix, even during shutdown.
  void retire(uint64_t id) noexcept;
  bool write(uint64_t id, const std::vector<std::span<const std::byte>> &source,
             const std::atomic<bool> &cancelled);
  bool read(uint64_t id, const std::vector<std::span<std::byte>> &destination,
            const std::atomic<bool> &cancelled);
  std::vector<Prefix> load(); // oldest first
  // Startup only, after quota trimming and before admitting new slots.
  void collectUnreferenced();
  void save(const Prefix &prefix);
  void erase(uint64_t id);
  // Oldest first: one transaction persists a batch of actual accesses.
  void touch(std::span<const uint64_t> ids);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
