#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace splash::flash {

inline constexpr uint64_t kFlashPLESSDRowBytes = 100;

// Original affine Q4/G32 planes are read on demand. The store neither maps the
// table nor creates GPU table buffers; source payloads remain unchanged.
class FlashPLESSDStore final {
public:
  struct Source final {
    std::filesystem::path path;
    uint64_t byteCount = 0;
  };
  struct Plane final {
    uint32_t source = 0;
    uint64_t offset = 0;
    uint64_t rowStride = 0;
  };
  struct Part final {
    uint64_t rows = 0;
    Plane weights;
    Plane scales;
    Plane biases;
  };
  struct Options final {
    uint64_t cacheBytes = 64 * 1024 * 1024;
    uint64_t maxCoalescedReadBytes = 1024 * 1024;
    // On macOS, F_NOCACHE avoids growing the unified file cache with the table.
    // It is an I/O policy, not a promise that the storage device is uncached.
    bool noCache = true;
  };
  struct Statistics final {
    uint64_t preparedBatches = 0;
    uint64_t requestedRows = 0;
    uint64_t uniqueMissRows = 0;
    uint64_t cacheHitRows = 0;
    uint64_t duplicateMissRows = 0;
    uint64_t readRequests = 0;
    uint64_t requestedReadBytes = 0;
    uint64_t logicalMissBytes = 0;
    uint64_t completedReadBytes = 0;
    uint64_t cacheEvictions = 0;
    uint64_t failedBatches = 0;
    uint64_t sourceValidationFailures = 0;
    uint64_t hostReadNanoseconds = 0;
    uint64_t cacheBudgetBytes = 0;
    uint64_t cacheAccountedBytes = 0;
    uint64_t cachedRows = 0;
    uint64_t readScratchLimitBytes = 0;
    bool fileCacheBypassEnabled = false;
    bool poisoned = false;
  };

  FlashPLESSDStore(std::vector<Source> sources, std::vector<Part> parts,
                   Options options);
  FlashPLESSDStore(std::vector<Source> sources, std::vector<Part> parts)
      : FlashPLESSDStore(std::move(sources), std::move(parts), Options{}) {}
  ~FlashPLESSDStore();
  FlashPLESSDStore(const FlashPLESSDStore &) = delete;
  FlashPLESSDStore &operator=(const FlashPLESSDStore &) = delete;

  // IDs are global physical table rows, in [0, tableRows()). Output has exactly
  // IDs.size()*100 bytes, preserving input order and duplicates. Input and
  // output spans must not overlap. A source/read
  // failure poisons this store and throws; the caller must not use its output.
  void lookupRows(std::span<const int64_t> globalIDs,
                  std::span<uint8_t> packedRows);
  [[nodiscard]] uint64_t tableRows() const noexcept;
  [[nodiscard]] uint64_t shardRows() const noexcept;
  [[nodiscard]] uint64_t partCount() const noexcept;
  [[nodiscard]] Statistics statistics() const;
  // Drops bounded CPU row cache only. It does not claim OS cache eviction.
  void clearCache();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash
