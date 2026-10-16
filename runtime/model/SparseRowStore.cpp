#include "model/SparseRowStore.hpp"
#include "model/WeightSource.hpp"
#include "AwakeClock.hpp"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace splash::model {

void SparseRowTable::validate() const {
  if (!rows || !rowBytes) throw std::invalid_argument("streamed table " + name + " has no rows");
  if (rowBytes > kSparseRowMaximumBytes)
    throw std::invalid_argument("streamed table " + name + " has rows wider than the store takes");
  // Row ids travel as 32-bit values (the GPU's index buffers and the
  // staging slots); a wider table needs a wider ABI.
  if (rows > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("streamed table " + name + " has more rows than 32-bit ids address");
  if (rows > (std::numeric_limits<uint64_t>::max() - offset) / rowBytes)
    throw std::invalid_argument("streamed table " + name + " ends past the largest file offset");
}

namespace {

// Latencies count the time the Mac is awake, as the runtime's every duration.
using Clock = AwakeClock;
constexpr uint32_t kNoSlot = std::numeric_limits<uint32_t>::max();

class FileRowReader final : public RowReader {
public:
  FileRowReader(int descriptor, std::filesystem::path path) : path_(std::move(path)) {
    file_ = fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
    if (file_ < 0) fail("unable to open");
    // Rows are read once into the cache: through the page cache each 90-byte
    // row would hold a page of the file in memory.
    if (fstat(file_, &opened_) || fcntl(file_, F_NOCACHE, 1)) {
      const int error = errno;
      close(file_);
      errno = error;
      fail("unable to open");
    }
    if (!S_ISREG(opened_.st_mode)) {
      close(file_);
      throw SparseRowError("streamed table source is not a regular file: " + path_.string());
    }
  }
  ~FileRowReader() override { close(file_); }

  void read(uint64_t offset, std::span<uint8_t> bytes) override {
    if (bytes.size() > uint64_t(std::numeric_limits<off_t>::max()) ||
        offset > uint64_t(std::numeric_limits<off_t>::max()) - bytes.size())
      throw SparseRowError("row read offset overflow: " + path_.string());
    while (!bytes.empty()) {
      const ssize_t count = pread(file_, bytes.data(), bytes.size(), static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      if (count < 0)
        throw SparseRowError(std::string("unable to read rows: ") + std::strerror(errno) + ": " + path_.string());
      if (!count) throw SparseRowError("truncated streamed table: " + path_.string());
      offset += uint64_t(count);
      bytes = bytes.subspan(size_t(count));
    }
  }

  void checkUnchanged() override {
    struct stat current{};
    if (fstat(file_, &current)) fail("unable to stat");
    if (current.st_size != opened_.st_size || current.st_mtimespec.tv_sec != opened_.st_mtimespec.tv_sec ||
        current.st_mtimespec.tv_nsec != opened_.st_mtimespec.tv_nsec)
      throw SparseRowError("streamed table source was written while the model is loaded: " + path_.string());
  }

  [[nodiscard]] uint64_t size() const override { return uint64_t(opened_.st_size); }

private:
  [[noreturn]] void fail(const char *operation) const {
    throw SparseRowError(std::string(operation) + " " + path_.string() + ": " + std::strerror(errno));
  }
  std::filesystem::path path_;
  int file_ = -1;
  struct stat opened_{};
};

// Latencies in log-2 buckets of microseconds, updated from any thread.
class LatencyHistogram final {
public:
  void add(uint64_t nanoseconds) noexcept {
    const uint64_t micros = nanoseconds / 1000;
    const uint32_t bucket = std::min<uint32_t>(uint32_t(std::bit_width(micros)), kBuckets - 1);
    buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    total_.fetch_add(nanoseconds, std::memory_order_relaxed);
    uint64_t seen = maximum_.load(std::memory_order_relaxed);
    while (nanoseconds > seen && !maximum_.compare_exchange_weak(seen, nanoseconds, std::memory_order_relaxed)) {
    }
  }
  [[nodiscard]] SparseRowLatency snapshot() const noexcept {
    SparseRowLatency result;
    std::array<uint64_t, kBuckets> counts{};
    for (uint32_t b = 0; b < kBuckets; ++b) {
      counts[b] = buckets_[b].load(std::memory_order_relaxed);
      result.count += counts[b];
    }
    result.totalMilliseconds = double(total_.load(std::memory_order_relaxed)) / 1e6;
    result.maximumMilliseconds = double(maximum_.load(std::memory_order_relaxed)) / 1e6;
    const auto quantile = [&](double q) {
      if (!result.count) return 0.0;
      const uint64_t rank = uint64_t(q * double(result.count - 1)) + 1;
      uint64_t seen = 0;
      for (uint32_t b = 0; b < kBuckets; ++b) {
        seen += counts[b];
        // Bucket b holds microseconds of bit width b: below 2^b.
        if (seen >= rank) return double(uint64_t{1} << b) / 1000.0;
      }
      return result.maximumMilliseconds;
    };
    result.p50Milliseconds = quantile(0.50);
    result.p95Milliseconds = quantile(0.95);
    result.p99Milliseconds = quantile(0.99);
    return result;
  }

private:
  static constexpr uint32_t kBuckets = 40;
  std::array<std::atomic<uint64_t>, kBuckets> buckets_{};
  std::atomic<uint64_t> count_{0}, total_{0}, maximum_{0};
};

uint64_t nanosecondsSince(Clock::time_point start) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

uint64_t indexBuckets(uint64_t rows) noexcept { return std::bit_ceil(std::max<uint64_t>(2 * rows, 2)); }

} // namespace

uint64_t SparseRowStore::cacheBytesFor(uint64_t rows, uint32_t rowBytes) noexcept {
  if (!rows) return 0;
  // Row bytes, its id, two recency links; buckets of 32-bit slot numbers.
  const uint64_t perRow = uint64_t{rowBytes} + 3 * sizeof(uint32_t);
  if (rows > std::numeric_limits<uint64_t>::max() / 4 / perRow) return std::numeric_limits<uint64_t>::max();
  return rows * perRow + indexBuckets(rows) * sizeof(uint32_t);
}

uint64_t SparseRowStore::workingBytes(uint32_t rowBytes, uint32_t prefetchRows) noexcept {
  // A prefetched row's bytes, id and destination; a stage's dedupe map (three
  // words in each of twice as many buckets), distinct and missing ids, and
  // its read job's id and destination per row.
  const uint64_t prefetch = uint64_t{prefetchRows} * (uint64_t{rowBytes} + sizeof(uint32_t) + sizeof(uint8_t *));
  const uint64_t stage = uint64_t{kSparseRowStageRows} * (6 * sizeof(uint32_t) + 2 * sizeof(uint32_t) +
                                                          sizeof(uint32_t) + sizeof(uint8_t *));
  return prefetch + stage;
}

uint64_t SparseRowStore::cacheRowsFor(uint64_t cacheBytes, uint32_t rowBytes) noexcept {
  if (!rowBytes) return 0;
  // Slot numbers are 32-bit, one of them reserved.
  uint64_t low = 0, high = std::min<uint64_t>(cacheBytes / rowBytes, std::numeric_limits<uint32_t>::max() - 1);
  while (low < high) {
    const uint64_t middle = low + (high - low + 1) / 2;
    if (cacheBytesFor(middle, rowBytes) <= cacheBytes) low = middle;
    else high = middle - 1;
  }
  return low;
}

struct SparseRowStore::Impl {
  // Rows a reader reads: ids[i]'s row into destinations[i].
  struct ReadJob final {
    explicit ReadJob(std::atomic<uint64_t> &live) : live(live) { live.fetch_add(1); }
    ~ReadJob() { live.fetch_sub(1); }
    std::atomic<uint64_t> &live;
    std::vector<uint32_t> ids;
    std::vector<uint8_t *> destinations;
    std::vector<uint8_t> buffer; // a prefetch's own rows
    bool prefetch = false;
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    std::mutex mutex;
    std::condition_variable finished;
    bool complete = false; // every row read (and a prefetch's rows cached)
    std::exception_ptr failure;
  };

  SparseRowTable table;
  std::unique_ptr<RowReader> reader;
  SparseRowStoreOptions options;

  // The cache: slots of rows with their ids, a recency list (head most
  // recent) and an open-addressed index of slot + 1 (0 empty).
  mutable std::mutex cacheMutex;
  uint64_t capacity = 0;
  uint64_t used = 0;
  std::unique_ptr<uint8_t[]> rows;
  std::unique_ptr<uint32_t[]> ids, previous, next, buckets;
  uint64_t bucketMask = 0;
  uint32_t bucketShift = 0;
  uint32_t head = kNoSlot, tail = kNoSlot;
  uint64_t evictions = 0;

  // Readers.
  std::atomic<uint64_t> liveReadJobs{0};
  std::mutex poolMutex;
  std::condition_variable poolWake;
  std::condition_variable prefetchIdle;
  std::deque<std::shared_ptr<ReadJob>> critical, background;
  std::vector<std::shared_ptr<ReadJob>> prefetching;
  uint64_t prefetchingRows = 0;
  std::atomic<uint32_t> criticalWaiting{0};
  bool stopping = false;
  std::vector<std::thread> threads;

  // Staging: serialized; its dedupe map stamps entries with the stage's
  // epoch, so it is never cleared.
  std::mutex stageMutex;
  std::vector<uint32_t> mapIds, mapRows, mapEpochs;
  uint32_t epoch = 0;
  std::vector<uint32_t> unique, missing;
  std::mutex prefetchMutex;
  std::vector<uint32_t> prefetchIds, prefetchEpochs;
  uint32_t prefetchEpoch = 0;

  // Failure.
  std::atomic<bool> poisoned{false};
  mutable std::mutex failureMutex;
  std::string failure;

  // Statistics.
  struct UseCounters final {
    std::atomic<uint64_t> stages{0}, requested{0}, unique{0}, hits{0}, read{0};
    LatencyHistogram latency;
  };
  std::array<UseCounters, kSparseRowUses> uses;
  std::atomic<uint64_t> readBytes{0}, readRetries{0}, readErrors{0};
  std::atomic<uint64_t> prefetchRequested{0}, prefetchRead{0}, prefetchDropped{0};
  LatencyHistogram readLatency;

  Impl(SparseRowTable table, std::unique_ptr<RowReader> reader, SparseRowStoreOptions options)
      : table(std::move(table)), reader(std::move(reader)), options(options) {}

  [[nodiscard]] uint8_t *slotRow(uint32_t slot) const noexcept { return rows.get() + uint64_t{slot} * table.rowBytes; }
  [[nodiscard]] uint64_t bucketOf(uint32_t id) const noexcept {
    return (uint64_t{id} * 0x9E3779B97F4A7C15ull) >> bucketShift;
  }

  void allocate(uint64_t cacheBytes) {
    const uint64_t working = workingBytes(table.rowBytes, options.prefetchRows);
    capacity = cacheBytes > working ? cacheRowsFor(cacheBytes - working, table.rowBytes) : 0;
    if (!capacity) throw std::invalid_argument("the row cache holds no row of " + table.name);
    const uint64_t bucketCount = indexBuckets(capacity);
    bucketMask = bucketCount - 1;
    bucketShift = 64 - uint32_t(std::countr_zero(bucketCount));
    rows = std::make_unique_for_overwrite<uint8_t[]>(capacity * table.rowBytes);
    ids = std::make_unique_for_overwrite<uint32_t[]>(capacity);
    previous = std::make_unique_for_overwrite<uint32_t[]>(capacity);
    next = std::make_unique_for_overwrite<uint32_t[]>(capacity);
    buckets = std::make_unique<uint32_t[]>(bucketCount);
  }

  // Cache operations, under cacheMutex.
  [[nodiscard]] uint32_t find(uint32_t id) const noexcept {
    for (uint64_t b = bucketOf(id);; b = (b + 1) & bucketMask) {
      const uint32_t entry = buckets[b];
      if (!entry) return kNoSlot;
      if (ids[entry - 1] == id) return entry - 1;
    }
  }
  void unlink(uint32_t slot) noexcept {
    const uint32_t before = previous[slot], after = next[slot];
    (before == kNoSlot ? head : next[before]) = after;
    (after == kNoSlot ? tail : previous[after]) = before;
  }
  void pushFront(uint32_t slot) noexcept {
    previous[slot] = kNoSlot;
    next[slot] = head;
    (head == kNoSlot ? tail : previous[head]) = slot;
    head = slot;
  }
  void touch(uint32_t slot) noexcept {
    if (head == slot) return;
    unlink(slot);
    pushFront(slot);
  }
  // Removes `id`'s bucket, shifting the probe run after it back.
  void erase(uint32_t id) noexcept {
    uint64_t hole = bucketOf(id);
    while (ids[buckets[hole] - 1] != id) hole = (hole + 1) & bucketMask;
    for (uint64_t probe = (hole + 1) & bucketMask;; probe = (probe + 1) & bucketMask) {
      const uint32_t entry = buckets[probe];
      if (!entry) break;
      const uint64_t home = bucketOf(ids[entry - 1]);
      // The entry may move to the hole when its home is not in (hole, probe].
      if (((probe - home) & bucketMask) >= ((probe - hole) & bucketMask)) {
        buckets[hole] = entry;
        hole = probe;
      }
    }
    buckets[hole] = 0;
  }
  void insert(uint32_t id, const uint8_t *bytes) noexcept {
    if (const uint32_t slot = find(id); slot != kNoSlot) {
      touch(slot);
      return;
    }
    uint32_t slot;
    if (used < capacity) {
      slot = uint32_t(used++);
    } else {
      slot = tail;
      unlink(slot);
      erase(ids[slot]);
      ++evictions;
    }
    ids[slot] = id;
    std::memcpy(slotRow(slot), bytes, table.rowBytes);
    uint64_t b = bucketOf(id);
    while (buckets[b]) b = (b + 1) & bucketMask;
    buckets[b] = slot + 1;
    pushFront(slot);
  }

  void poison(const std::string &message) {
    {
      const std::lock_guard lock(failureMutex);
      if (failure.empty()) failure = message;
    }
    poisoned.store(true, std::memory_order_release);
  }
  [[noreturn]] void throwPoisoned() const {
    const std::lock_guard lock(failureMutex);
    throw SparseRowError("rows of " + table.name + " are unavailable: " + failure);
  }

  void readRow(uint32_t id, uint8_t *destination) {
    const uint64_t offset = table.offset + uint64_t{id} * table.rowBytes;
    const auto started = Clock::now();
    try {
      reader->read(offset, {destination, table.rowBytes});
    } catch (...) {
      // A transient error gets one more read; a second failure is final.
      readRetries.fetch_add(1, std::memory_order_relaxed);
      try {
        reader->read(offset, {destination, table.rowBytes});
      } catch (...) {
        readErrors.fetch_add(1, std::memory_order_relaxed);
        throw;
      }
    }
    readLatency.add(nanosecondsSince(started));
    readBytes.fetch_add(table.rowBytes, std::memory_order_relaxed);
  }

  // Reads the job's unclaimed rows on this thread. A reader working on a
  // prefetch returns once a stage waits, leaving the rest for later.
  void work(const std::shared_ptr<ReadJob> &job, bool yieldToStages) {
    const size_t count = job->ids.size();
    for (;;) {
      if (yieldToStages && criticalWaiting.load(std::memory_order_acquire)) return;
      const size_t index = job->next.fetch_add(1, std::memory_order_relaxed);
      if (index >= count) return;
      try {
        readRow(job->ids[index], job->destinations[index]);
      } catch (...) {
        const std::lock_guard lock(job->mutex);
        if (!job->failure) job->failure = std::current_exception();
      }
      if (job->done.fetch_add(1, std::memory_order_acq_rel) + 1 == count) complete(job);
    }
  }

  // Runs on the thread that read a job's last row.
  void complete(const std::shared_ptr<ReadJob> &job) {
    if (job->prefetch) finishPrefetch(*job);
    {
      const std::lock_guard lock(job->mutex);
      job->complete = true;
    }
    {
      const std::lock_guard lock(poolMutex);
      std::erase(critical, job);
      std::erase(background, job);
      if (job->prefetch) {
        std::erase(prefetching, job);
        prefetchingRows -= job->ids.size();
        if (prefetching.empty()) prefetchIdle.notify_all();
      }
    }
    job->finished.notify_all();
  }

  // A prefetch's rows enter the cache once the source is unchanged since
  // they were read; a failure poisons the store for the next stage.
  void finishPrefetch(ReadJob &job) {
    std::exception_ptr failed;
    {
      const std::lock_guard lock(job.mutex);
      failed = job.failure;
    }
    try {
      if (failed) std::rethrow_exception(failed);
      reader->checkUnchanged();
    } catch (const std::exception &error) {
      poison(error.what());
      return;
    } catch (...) {
      poison("unknown read failure");
      return;
    }
    if (poisoned.load(std::memory_order_acquire)) return;
    const std::lock_guard lock(cacheMutex);
    for (size_t index = 0; index < job.ids.size(); ++index) insert(job.ids[index], job.destinations[index]);
    prefetchRead.fetch_add(job.ids.size(), std::memory_order_relaxed);
  }

  static void waitComplete(ReadJob &job) {
    std::unique_lock lock(job.mutex);
    job.finished.wait(lock, [&] { return job.complete; });
  }

  // The first job of `queue` with unclaimed rows; drops the exhausted ones.
  static std::shared_ptr<ReadJob> frontWithWork(std::deque<std::shared_ptr<ReadJob>> &queue) {
    while (!queue.empty()) {
      if (queue.front()->next.load(std::memory_order_relaxed) < queue.front()->ids.size()) return queue.front();
      queue.pop_front();
    }
    return nullptr;
  }

  void readerLoop() {
    for (;;) {
      std::shared_ptr<ReadJob> job;
      bool fromBackground = false;
      {
        std::unique_lock lock(poolMutex);
        // Prefetches wait while a stage does.
        poolWake.wait(lock, [&] {
          if (stopping) return true;
          job = frontWithWork(critical);
          if (!job && !criticalWaiting.load(std::memory_order_relaxed)) {
            job = frontWithWork(background);
            fromBackground = job != nullptr;
          }
          return job != nullptr;
        });
        if (stopping) return;
      }
      work(job, fromBackground);
    }
  }

  // Reads a stage's or an adopted prefetch's rows here and on the readers,
  // and returns once every row was read.
  void readNow(const std::shared_ptr<ReadJob> &job) {
    {
      const std::lock_guard lock(poolMutex);
      criticalWaiting.fetch_add(1, std::memory_order_acq_rel);
      critical.push_back(job);
    }
    poolWake.notify_all();
    work(job, false);
    waitComplete(*job);
    {
      // Under the lock, so that a reader that saw a stage waiting and
      // passed the prefetches over is woken for them.
      const std::lock_guard lock(poolMutex);
      // Adoption may have queued a job after its completion retired it.
      std::erase(critical, job);
      std::erase(background, job);
      criticalWaiting.fetch_sub(1, std::memory_order_acq_rel);
    }
    poolWake.notify_all();
  }

  void adoptPrefetches() {
    std::vector<std::shared_ptr<ReadJob>> adopted;
    {
      const std::lock_guard lock(poolMutex);
      adopted = prefetching;
    }
    for (const std::shared_ptr<ReadJob> &job : adopted) readNow(job);
  }
};

SparseRowStore::SparseRowStore(SparseRowTable table, std::unique_ptr<RowReader> reader,
                               SparseRowStoreOptions options) {
  table.validate();
  if (!reader) throw std::invalid_argument("streamed table " + table.name + " has no reader");
  if (table.offset + table.rows * table.rowBytes > reader->size())
    throw std::invalid_argument("streamed table " + table.name + " ends past its file");
  impl_ = std::make_unique<Impl>(std::move(table), std::move(reader), options);
  impl_->allocate(options.cacheBytes);
  impl_->threads.reserve(options.readers);
  try {
    for (uint32_t index = 0; index < options.readers; ++index)
      impl_->threads.emplace_back([impl = impl_.get()] { impl->readerLoop(); });
  } catch (...) {
    {
      const std::lock_guard lock(impl_->poolMutex);
      impl_->stopping = true;
    }
    impl_->poolWake.notify_all();
    for (std::thread &thread : impl_->threads) thread.join();
    throw;
  }
}

SparseRowStore::~SparseRowStore() {
  {
    const std::lock_guard lock(impl_->poolMutex);
    impl_->stopping = true;
  }
  impl_->poolWake.notify_all();
  for (std::thread &thread : impl_->threads) thread.join();
}

const SparseRowTable &SparseRowStore::table() const noexcept { return impl_->table; }

uint64_t SparseRowStore::committedBytes() const noexcept {
  return cacheBytesFor(impl_->capacity, impl_->table.rowBytes) +
         workingBytes(impl_->table.rowBytes, impl_->options.prefetchRows);
}

SparseRowStageResult SparseRowStore::stage(std::span<const uint32_t> ids, std::span<uint8_t> rows,
                                           std::span<uint32_t> slots, SparseRowUse use, bool adoptPrefetch) {
  Impl &s = *impl_;
  const std::lock_guard stageLock(s.stageMutex);
  const auto started = Clock::now();
  const uint32_t rowBytes = s.table.rowBytes;
  if (slots.size() < ids.size() || rows.size() / rowBytes < ids.size())
    throw std::invalid_argument("staging of " + s.table.name + " is smaller than its rows");
  if (ids.size() > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("too many rows staged at once");
  if (s.poisoned.load(std::memory_order_acquire)) s.throwPoisoned();
  if (adoptPrefetch) s.adoptPrefetches();
  if (s.poisoned.load(std::memory_order_acquire)) s.throwPoisoned();
  try {
    s.reader->checkUnchanged();
  } catch (const std::exception &error) {
    s.poison(error.what());
    s.throwPoisoned();
  }

  // Distinct ids, in order of first use.
  const size_t mapSize = std::bit_ceil(std::max<size_t>(2 * ids.size(), 2));
  if (s.mapIds.size() < mapSize) {
    s.mapIds.assign(mapSize, 0);
    s.mapRows.assign(mapSize, 0);
    s.mapEpochs.assign(mapSize, 0);
    s.epoch = 0;
  }
  if (++s.epoch == 0) {
    std::fill(s.mapEpochs.begin(), s.mapEpochs.end(), 0);
    s.epoch = 1;
  }
  const size_t mask = mapSize - 1;
  const uint32_t shift = 64 - uint32_t(std::countr_zero(mapSize));
  s.unique.clear();
  for (size_t i = 0; i < ids.size(); ++i) {
    const uint32_t id = ids[i];
    if (id >= s.table.rows)
      throw SparseRowError("row " + std::to_string(id) + " is past the end of " + s.table.name);
    size_t b = size_t((uint64_t{id} * 0x9E3779B97F4A7C15ull) >> shift) & mask;
    while (s.mapEpochs[b] == s.epoch && s.mapIds[b] != id) b = (b + 1) & mask;
    if (s.mapEpochs[b] != s.epoch) {
      s.mapEpochs[b] = s.epoch;
      s.mapIds[b] = id;
      s.mapRows[b] = uint32_t(s.unique.size());
      s.unique.push_back(id);
    }
    slots[i] = s.mapRows[b];
  }

  // Cached rows are copied out while the cache cannot evict them.
  s.missing.clear();
  {
    const std::lock_guard lock(s.cacheMutex);
    for (uint32_t u = 0; u < s.unique.size(); ++u) {
      const uint32_t slot = s.find(s.unique[u]);
      if (slot == kNoSlot) {
        s.missing.push_back(u);
        continue;
      }
      std::memcpy(rows.data() + uint64_t{u} * rowBytes, s.slotRow(slot), rowBytes);
      s.touch(slot);
    }
  }

  if (!s.missing.empty()) {
    auto job = std::make_shared<Impl::ReadJob>(s.liveReadJobs);
    job->ids.reserve(s.missing.size());
    job->destinations.reserve(s.missing.size());
    for (const uint32_t u : s.missing) {
      job->ids.push_back(s.unique[u]);
      job->destinations.push_back(rows.data() + uint64_t{u} * rowBytes);
    }
    s.readNow(job);
    // No reader writes into `rows` from here on: every row of the job was
    // claimed and read.
    try {
      if (job->failure) std::rethrow_exception(job->failure);
      s.reader->checkUnchanged();
    } catch (const std::exception &error) {
      s.poison(error.what());
      s.throwPoisoned();
    }
    const std::lock_guard lock(s.cacheMutex);
    for (size_t index = 0; index < job->ids.size(); ++index) s.insert(job->ids[index], job->destinations[index]);
  }

  SparseRowStageResult result;
  result.uniqueRows = uint32_t(s.unique.size());
  result.readRows = uint32_t(s.missing.size());
  result.hitRows = result.uniqueRows - result.readRows;
  result.nanoseconds = nanosecondsSince(started);
  Impl::UseCounters &counters = s.uses[static_cast<uint32_t>(use)];
  counters.stages.fetch_add(1, std::memory_order_relaxed);
  counters.requested.fetch_add(ids.size(), std::memory_order_relaxed);
  counters.unique.fetch_add(result.uniqueRows, std::memory_order_relaxed);
  counters.hits.fetch_add(result.hitRows, std::memory_order_relaxed);
  counters.read.fetch_add(result.readRows, std::memory_order_relaxed);
  counters.latency.add(result.nanoseconds);
  return result;
}

void SparseRowStore::prefetch(std::span<const uint32_t> ids) {
  Impl &s = *impl_;
  if (ids.empty() || s.poisoned.load(std::memory_order_acquire)) return;
  const std::lock_guard prefetchLock(s.prefetchMutex);
  s.prefetchRequested.fetch_add(ids.size(), std::memory_order_relaxed);
  // Distinct ids within the table, not cached.
  const size_t mapSize = std::bit_ceil(std::max<size_t>(2 * ids.size(), 2));
  if (s.prefetchIds.size() < mapSize) {
    s.prefetchIds.assign(mapSize, 0);
    s.prefetchEpochs.assign(mapSize, 0);
    s.prefetchEpoch = 0;
  }
  if (++s.prefetchEpoch == 0) {
    std::fill(s.prefetchEpochs.begin(), s.prefetchEpochs.end(), 0);
    s.prefetchEpoch = 1;
  }
  const size_t mask = mapSize - 1;
  const uint32_t shift = 64 - uint32_t(std::countr_zero(mapSize));
  auto job = std::make_shared<Impl::ReadJob>(s.liveReadJobs);
  job->prefetch = true;
  {
    const std::lock_guard lock(s.cacheMutex);
    for (const uint32_t id : ids) {
      if (id >= s.table.rows) continue;
      size_t b = size_t((uint64_t{id} * 0x9E3779B97F4A7C15ull) >> shift) & mask;
      while (s.prefetchEpochs[b] == s.prefetchEpoch && s.prefetchIds[b] != id) b = (b + 1) & mask;
      if (s.prefetchEpochs[b] == s.prefetchEpoch) continue;
      s.prefetchEpochs[b] = s.prefetchEpoch;
      s.prefetchIds[b] = id;
      if (s.find(id) == kNoSlot) job->ids.push_back(id);
    }
  }
  {
    const std::lock_guard lock(s.poolMutex);
    const uint64_t room =
        s.options.prefetchRows > s.prefetchingRows ? s.options.prefetchRows - s.prefetchingRows : 0;
    if (job->ids.size() > room) {
      s.prefetchDropped.fetch_add(job->ids.size() - room, std::memory_order_relaxed);
      job->ids.resize(room);
    }
    if (job->ids.empty()) return;
    job->buffer.resize(job->ids.size() * s.table.rowBytes);
    job->destinations.reserve(job->ids.size());
    for (size_t index = 0; index < job->ids.size(); ++index)
      job->destinations.push_back(job->buffer.data() + index * s.table.rowBytes);
    s.prefetchingRows += job->ids.size();
    s.prefetching.push_back(job);
    s.background.push_back(job);
  }
  s.poolWake.notify_all();
}

void SparseRowStore::drainPrefetch() {
  Impl &s = *impl_;
  if (s.options.readers == 0) {
    for (;;) {
      {
        const std::lock_guard lock(s.poolMutex);
        if (s.prefetching.empty() || s.stopping) return;
      }
      // A concurrent prefetch may enqueue after the last adoption snapshot.
      s.adoptPrefetches();
    }
  }
  std::unique_lock lock(s.poolMutex);
  s.prefetchIdle.wait(lock, [&] { return s.prefetching.empty() || s.stopping; });
}

SparseRowStats SparseRowStore::stats() const {
  const Impl &s = *impl_;
  SparseRowStats result;
  result.table = s.table.name;
  result.tableBytes = s.table.rows * s.table.rowBytes;
  result.cacheBytes = cacheBytesFor(s.capacity, s.table.rowBytes) + workingBytes(s.table.rowBytes, s.options.prefetchRows);
  result.cacheCapacityRows = s.capacity;
  {
    const std::lock_guard lock(s.cacheMutex);
    result.cachedRows = s.used;
    result.evictions = s.evictions;
  }
  for (uint32_t use = 0; use < kSparseRowUses; ++use) {
    const Impl::UseCounters &counters = s.uses[use];
    result.uses[use] = {counters.stages.load(), counters.requested.load(), counters.unique.load(),
                        counters.hits.load(),   counters.read.load(),      counters.latency.snapshot()};
  }
  result.liveReadJobs = s.liveReadJobs.load();
  result.readBytes = s.readBytes.load();
  result.readRetries = s.readRetries.load();
  result.readErrors = s.readErrors.load();
  result.prefetchRequestedRows = s.prefetchRequested.load();
  result.prefetchReadRows = s.prefetchRead.load();
  result.prefetchDroppedRows = s.prefetchDropped.load();
  result.read = s.readLatency.snapshot();
  result.poisoned = s.poisoned.load();
  {
    const std::lock_guard lock(s.failureMutex);
    result.failure = s.failure;
  }
  return result;
}

std::unique_ptr<RowReader> openRowReader(int descriptor, const std::filesystem::path &path) {
  return std::make_unique<FileRowReader>(descriptor, path);
}

std::unique_ptr<RowReader> openRowReader(const WeightSource &source) {
  auto reader = openRowReader(source.descriptor(), source.path());
  // Recheck the original snapshot after the reader has taken its own. A
  // later write is then caught by the reader's checks before staging.
  source.checkUnchanged();
  return reader;
}

} // namespace splash::model
