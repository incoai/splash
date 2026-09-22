#include "flash/FlashPLESSDStore.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::flash {
namespace {
[[noreturn]] void fail(const std::string &what) {
  throw std::runtime_error("Flash PLE SSD store: " + what);
}
uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > UINT64_MAX / b) fail("byte extent overflows");
  return a * b;
}
uint64_t add(uint64_t a, uint64_t b) {
  if (a > UINT64_MAX - b) fail("byte offset overflows");
  return a + b;
}
void accumulate(uint64_t &value, uint64_t amount) {
  value = amount > UINT64_MAX - value ? UINT64_MAX : value + amount;
}
bool sameVersion(const struct stat &a, const struct stat &b) {
#if defined(__APPLE__)
  const bool times = a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec &&
      a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
      a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec &&
      a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec;
#else
  const bool times = a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
      a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
      a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
      a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
#endif
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
      a.st_size == b.st_size && S_ISREG(b.st_mode) && times;
}
} // namespace

struct FlashPLESSDStore::Impl final {
  struct OpenSource final {
    Source specification;
    int fd = -1;
    struct stat version{};
    ~OpenSource() { if (fd >= 0) ::close(fd); }
    OpenSource() = default;
    OpenSource(const OpenSource &) = delete;
    OpenSource &operator=(const OpenSource &) = delete;
  };
  struct CacheSlot final {
    std::array<uint8_t, kFlashPLESSDRowBytes> row{};
    uint64_t id = 0;
    uint32_t previous = UINT32_MAX;
    uint32_t next = UINT32_MAX;
    bool occupied = false;
  };
  // Fixed, conservatively admitted open-addressed index; no allocator overhead
  // scales with a changing number of requests or cached rows.
  struct CacheIndex final {
    uint64_t id = 0;
    uint32_t slot = 0;
    uint8_t state = 0; // 0 empty, 1 present.
  };
  struct Miss final {
    uint64_t id = 0;
    std::array<uint8_t, kFlashPLESSDRowBytes> row{};
  };
  struct Segment final {
    uint32_t source = 0;
    uint64_t offset = 0;
    uint32_t bytes = 0;
    uint64_t miss = 0;
    uint32_t destinationOffset = 0;
  };

  std::vector<std::unique_ptr<OpenSource>> sources;
  std::vector<Part> parts;
  Options options;
  Statistics stats;
  uint64_t totalRows = 0;
  uint64_t rowsPerPart = 0;
  std::vector<CacheSlot> cache;
  std::vector<CacheIndex> index;
  uint32_t nextFree = 0;
  uint32_t newest = UINT32_MAX;
  uint32_t oldest = UINT32_MAX;
  std::vector<uint8_t> readScratch;
  mutable std::mutex mutex;

  Impl(std::vector<Source> specifications, std::vector<Part> inputParts,
       Options inputOptions) : parts(std::move(inputParts)), options(inputOptions) {
    if (specifications.empty() || specifications.size() > UINT32_MAX ||
        parts.empty() || parts.size() > UINT32_MAX)
      fail("empty or oversized source/part inventory");
    if (!options.maxCoalescedReadBytes || options.maxCoalescedReadBytes < 80 ||
        options.maxCoalescedReadBytes > 16 * 1024 * 1024)
      fail("read scratch limit must be 80 bytes through 16 MiB");
    if (options.cacheBytes > 1024ULL * 1024 * 1024)
      fail("CPU row cache limit exceeds 1 GiB");
    rowsPerPart = parts.front().rows;
    if (!rowsPerPart) fail("empty table part");
    totalRows = multiply(rowsPerPart, parts.size());
    if (totalRows > INT64_MAX) fail("table extent exceeds signed I64 IDs");
    for (const auto &part : parts) {
      if (part.rows != rowsPerPart) fail("parts must have homogeneous row counts");
      const std::array<const Plane *, 3> planes{&part.weights, &part.scales, &part.biases};
      const std::array<uint64_t, 3> byteCounts{80, 10, 10};
      for (size_t p = 0; p < planes.size(); ++p) {
        const auto &plane = *planes[p];
        if (plane.source >= specifications.size() || plane.rowStride < byteCounts[p])
          fail("invalid source index or affine row stride");
        const uint64_t end = add(add(plane.offset,
            multiply(part.rows - 1, plane.rowStride)), byteCounts[p]);
        if (end > specifications[plane.source].byteCount ||
            end > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
          fail("affine plane exceeds source file or native offset extent");
      }
    }
    bool bypass = options.noCache;
    for (auto &specification : specifications) {
      if (specification.path.empty() || !specification.byteCount ||
          specification.byteCount > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
        fail("invalid source path or byte count");
      auto source = std::make_unique<OpenSource>();
      source->specification = std::move(specification);
      source->fd = ::open(source->specification.path.c_str(),
                          O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
      if (source->fd < 0) fail("cannot open readonly source payload");
      if (::fstat(source->fd, &source->version) != 0 ||
          !S_ISREG(source->version.st_mode) || source->version.st_size < 0 ||
          static_cast<uint64_t>(source->version.st_size) != source->specification.byteCount)
        fail("source file byte count or type mismatch");
#if defined(__APPLE__)
      if (options.noCache && ::fcntl(source->fd, F_NOCACHE, 1) != 0)
        fail("cannot apply F_NOCACHE to readonly source");
#else
      // No portable equivalent guarantees the requested bypass policy. Report
      // its absence rather than claiming uncached physical device reads.
      bypass = false;
#endif
      sources.push_back(std::move(source));
    }
    // The hash table reserves four buckets per row. Round its bucket count up
    // to a power of two and reduce rows until BOTH fixed arrays fit the budget.
    uint64_t slots = options.cacheBytes /
        (sizeof(CacheSlot) + 8 * sizeof(CacheIndex));
    if (slots > UINT32_MAX) fail("cache slot extent exceeds native index width");
    uint64_t buckets = slots ? 1 : 0;
    while (buckets && buckets < slots * 4) buckets *= 2;
    while (slots && add(multiply(slots, sizeof(CacheSlot)),
                        multiply(buckets, sizeof(CacheIndex))) > options.cacheBytes) {
      --slots;
      if (!slots) buckets = 0;
      else while (buckets / 2 >= slots * 4) buckets /= 2;
    }
    cache.resize(static_cast<size_t>(slots));
    index.resize(static_cast<size_t>(buckets));
    readScratch.resize(static_cast<size_t>(options.maxCoalescedReadBytes));
    stats.cacheBudgetBytes = options.cacheBytes;
    stats.cacheAccountedBytes = add(multiply(cache.capacity(), sizeof(CacheSlot)),
                                    multiply(index.capacity(), sizeof(CacheIndex)));
    if (stats.cacheAccountedBytes > options.cacheBytes)
      fail("allocator capacity exceeds admitted row cache budget");
    stats.readScratchLimitBytes = options.maxCoalescedReadBytes;
    stats.fileCacheBypassEnabled = bypass;
    validateSources();
  }

  void validateSources() {
    for (const auto &source : sources) {
      struct stat current{}, named{};
      if (::fstat(source->fd, &current) != 0 ||
          ::stat(source->specification.path.c_str(), &named) != 0 ||
          !sameVersion(source->version, current) || !sameVersion(source->version, named)) {
        accumulate(stats.sourceValidationFailures, 1);
        stats.poisoned = true;
        fail("source payload identity or timestamps changed");
      }
    }
  }
  static uint64_t hash(uint64_t id) {
    id ^= id >> 30;
    id *= 0xbf58476d1ce4e5b9ULL;
    id ^= id >> 27;
    id *= 0x94d049bb133111ebULL;
    return id ^ (id >> 31);
  }
  uint32_t find(uint64_t id) const {
    if (index.empty()) return UINT32_MAX;
    const size_t mask = index.size() - 1;
    size_t bucket = static_cast<size_t>(hash(id)) & mask;
    for (size_t visited = 0; visited < index.size(); ++visited) {
      const auto &entry = index[bucket];
      if (!entry.state) return UINT32_MAX;
      if (entry.state == 1 && entry.id == id) return entry.slot;
      bucket = (bucket + 1) & mask;
    }
    return UINT32_MAX;
  }
  void erase(uint64_t id) {
    const size_t mask = index.size() - 1;
    size_t bucket = static_cast<size_t>(hash(id)) & mask;
    for (size_t visited = 0; visited < index.size(); ++visited) {
      auto &entry = index[bucket];
      if (!entry.state) break;
      if (entry.state == 1 && entry.id == id) {
        // Backward-shift deletion prevents tombstones from growing without
        // bound after many eviction cycles. Keep each probing chain reachable.
        size_t hole = bucket;
        size_t next = (hole + 1) & mask;
        while (index[next].state) {
          const size_t home = static_cast<size_t>(hash(index[next].id)) & mask;
          if (((next - home) & mask) >= ((hole - home) & mask)) {
            index[hole] = index[next];
            hole = next;
          }
          next = (next + 1) & mask;
        }
        index[hole] = CacheIndex{};
        return;
      }
      bucket = (bucket + 1) & mask;
    }
    fail("internal row cache index lost an occupied slot");
  }
  void insert(uint64_t id, uint32_t slot) {
    const size_t mask = index.size() - 1;
    size_t bucket = static_cast<size_t>(hash(id)) & mask;
    for (size_t visited = 0; visited < index.size(); ++visited) {
      auto &entry = index[bucket];
      if (!entry.state) {
        index[bucket] = {id, slot, 1};
        return;
      }
      bucket = (bucket + 1) & mask;
    }
    fail("internal row cache index is full");
  }
  void touch(uint32_t slot) {
    if (slot == newest) return;
    auto &entry = cache[slot];
    if (entry.previous != UINT32_MAX) cache[entry.previous].next = entry.next;
    else if (oldest == slot) oldest = entry.next;
    if (entry.next != UINT32_MAX) cache[entry.next].previous = entry.previous;
    entry.previous = newest;
    entry.next = UINT32_MAX;
    if (newest != UINT32_MAX) cache[newest].next = slot;
    newest = slot;
    if (oldest == UINT32_MAX) oldest = slot;
  }
  void cacheRow(const Miss &miss) {
    if (cache.empty()) return;
    uint32_t slot;
    if (nextFree < cache.size()) {
      slot = nextFree++;
      accumulate(stats.cachedRows, 1);
    } else {
      slot = oldest;
      erase(cache[slot].id);
      accumulate(stats.cacheEvictions, 1);
    }
    cache[slot].id = miss.id;
    cache[slot].row = miss.row;
    cache[slot].occupied = true;
    insert(miss.id, slot);
    touch(slot);
  }
  void read(uint32_t source, uint64_t offset, uint64_t bytes) {
    accumulate(stats.readRequests, 1);
    accumulate(stats.requestedReadBytes, bytes);
    const auto start = std::chrono::steady_clock::now();
    uint64_t completed = 0;
    while (completed < bytes) {
      const auto count = ::pread(sources[source]->fd, readScratch.data() + completed,
          static_cast<size_t>(bytes - completed), static_cast<off_t>(offset + completed));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) {
        stats.poisoned = true;
        accumulate(stats.hostReadNanoseconds, static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count()));
        fail(count == 0 ? "short source read" : "source pread failed");
      }
      completed += static_cast<uint64_t>(count);
      accumulate(stats.completedReadBytes, static_cast<uint64_t>(count));
    }
    accumulate(stats.hostReadNanoseconds, static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count()));
  }

  void lookup(std::span<const int64_t> ids, std::span<uint8_t> output) {
    if (stats.poisoned) fail("store is poisoned after source or I/O failure");
    if (multiply(ids.size(), kFlashPLESSDRowBytes) != output.size())
      fail("staged row byte count does not match ID count");
    if (!ids.empty()) {
      const uintptr_t inputBegin = reinterpret_cast<uintptr_t>(ids.data());
      const uintptr_t outputBegin = reinterpret_cast<uintptr_t>(output.data());
      const uint64_t inputBytes = multiply(ids.size(), sizeof(int64_t));
      if (!ids.data() || !output.data() || inputBytes > UINTPTR_MAX - inputBegin ||
          output.size() > UINTPTR_MAX - outputBegin)
        fail("input or output span exceeds address extent");
      if (inputBegin < outputBegin + output.size() &&
          outputBegin < inputBegin + inputBytes)
        fail("ID and staged row spans must not overlap");
    }
    for (const auto id : ids)
      if (id < 0 || static_cast<uint64_t>(id) >= totalRows)
        fail("global table row ID is out of bounds");
    validateSources();
    accumulate(stats.preparedBatches, 1);
    accumulate(stats.requestedRows, ids.size());
    std::vector<uint64_t> missIDs;
    missIDs.reserve(ids.size());
    for (size_t row = 0; row < ids.size(); ++row) {
      const auto id = static_cast<uint64_t>(ids[row]);
      const uint32_t slot = find(id);
      if (slot != UINT32_MAX) {
        std::memcpy(output.data() + row * kFlashPLESSDRowBytes,
                    cache[slot].row.data(), kFlashPLESSDRowBytes);
        touch(slot);
        accumulate(stats.cacheHitRows, 1);
      } else missIDs.push_back(id);
    }
    const uint64_t missedRequests = missIDs.size();
    std::sort(missIDs.begin(), missIDs.end());
    missIDs.erase(std::unique(missIDs.begin(), missIDs.end()), missIDs.end());
    accumulate(stats.uniqueMissRows, missIDs.size());
    accumulate(stats.logicalMissBytes, multiply(missIDs.size(), kFlashPLESSDRowBytes));
    accumulate(stats.duplicateMissRows, missedRequests - missIDs.size());
    std::vector<Miss> misses(missIDs.size());
    std::vector<Segment> segments;
    segments.reserve(multiply(misses.size(), 3));
    for (size_t m = 0; m < missIDs.size(); ++m) {
      misses[m].id = missIDs[m];
      const auto &part = parts[missIDs[m] / rowsPerPart];
      const uint64_t row = missIDs[m] % rowsPerPart;
      const std::array<const Plane *, 3> planes{&part.weights, &part.scales, &part.biases};
      const std::array<uint32_t, 3> counts{80, 10, 10};
      const std::array<uint32_t, 3> destinations{0, 80, 90};
      for (size_t p = 0; p < planes.size(); ++p) {
        const auto &plane = *planes[p];
        segments.push_back({plane.source, add(plane.offset, multiply(row, plane.rowStride)),
                            counts[p], m, destinations[p]});
      }
    }
    std::sort(segments.begin(), segments.end(), [](const Segment &a, const Segment &b) {
      return a.source < b.source || (a.source == b.source && a.offset < b.offset);
    });
    for (size_t begin = 0; begin < segments.size();) {
      size_t end = begin + 1;
      uint64_t readEnd = add(segments[begin].offset, segments[begin].bytes);
      while (end < segments.size() && segments[end].source == segments[begin].source &&
          segments[end].offset <= readEnd) {
        const uint64_t extended = std::max(readEnd,
            add(segments[end].offset, segments[end].bytes));
        if (extended - segments[begin].offset > options.maxCoalescedReadBytes) break;
        readEnd = extended;
        ++end;
      }
      read(segments[begin].source, segments[begin].offset,
           readEnd - segments[begin].offset);
      for (size_t s = begin; s < end; ++s) {
        const auto &segment = segments[s];
        std::memcpy(misses[segment.miss].row.data() + segment.destinationOffset,
                    readScratch.data() + segment.offset - segments[begin].offset,
                    segment.bytes);
      }
      begin = end;
    }
    // Detect a mutation during the read before making its rows cache-visible.
    validateSources();
    for (size_t row = 0; row < ids.size(); ++row) {
      const uint64_t id = static_cast<uint64_t>(ids[row]);
      const auto found = std::lower_bound(missIDs.begin(), missIDs.end(), id);
      if (found != missIDs.end() && *found == id)
        std::memcpy(output.data() + row * kFlashPLESSDRowBytes,
                    misses[static_cast<size_t>(found - missIDs.begin())].row.data(),
                    kFlashPLESSDRowBytes);
    }
    for (const auto &miss : misses) cacheRow(miss);
  }
};

FlashPLESSDStore::FlashPLESSDStore(std::vector<Source> sources,
    std::vector<Part> parts, Options options)
    : impl_(std::make_unique<Impl>(std::move(sources), std::move(parts), options)) {}
FlashPLESSDStore::~FlashPLESSDStore() = default;
void FlashPLESSDStore::lookupRows(std::span<const int64_t> ids,
                                std::span<uint8_t> output) {
  std::lock_guard lock(impl_->mutex);
  try { impl_->lookup(ids, output); }
  catch (...) { accumulate(impl_->stats.failedBatches, 1); throw; }
}
uint64_t FlashPLESSDStore::tableRows() const noexcept { return impl_->totalRows; }
uint64_t FlashPLESSDStore::shardRows() const noexcept { return impl_->rowsPerPart; }
uint64_t FlashPLESSDStore::partCount() const noexcept { return impl_->parts.size(); }
FlashPLESSDStore::Statistics FlashPLESSDStore::statistics() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->stats;
}
void FlashPLESSDStore::clearCache() {
  std::lock_guard lock(impl_->mutex);
  std::fill(impl_->cache.begin(), impl_->cache.end(), Impl::CacheSlot{});
  std::fill(impl_->index.begin(), impl_->index.end(), Impl::CacheIndex{});
  impl_->newest = impl_->oldest = UINT32_MAX;
  impl_->nextFree = 0;
  impl_->stats.cachedRows = 0;
}

} // namespace splash::flash
