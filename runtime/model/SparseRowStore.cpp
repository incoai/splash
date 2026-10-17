#include "model/SparseRowStore.hpp"
#include "model/WeightSource.hpp"
#include "AwakeClock.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <utility>
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

// Reads the source a table's metadata was parsed from, as it was opened
// (WeightSource): a new file at its path changes nothing, and a write to it
// is a change.
class SourceRowReader final : public RowReader {
public:
  explicit SourceRowReader(std::shared_ptr<const WeightSource> source) : source_(std::move(source)) {}
  void read(uint64_t offset, std::span<uint8_t> bytes) override { source_->readFile(offset, bytes); }
  void checkUnchanged() override { source_->checkUnchanged(); }
  [[nodiscard]] uint64_t size() const override { return source_->bytes(); }

private:
  std::shared_ptr<const WeightSource> source_;
};

uint64_t nanosecondsSince(Clock::time_point start) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

uint64_t indexBuckets(uint64_t rows) noexcept { return std::bit_ceil(std::max<uint64_t>(2 * rows, 2)); }

// Id's bucket of a table of 2^(64 - shift) buckets: the top bits of its
// product with 2^64 over the golden ratio (Fibonacci hashing).
uint64_t hashBucket(uint32_t id, uint32_t shift) noexcept { return (uint64_t{id} * 0x9E3779B97F4A7C15ull) >> shift; }

// The distinct ids of a stage or a prefetch, each with the order it came
// in: an open-addressed table of at least twice as many buckets as ids,
// whose entries carry the epoch they were written in, so begin() empties it
// without clearing it.
class DistinctIds final {
public:
  // Empties the set for up to `count` ids.
  void begin(size_t count) {
    const size_t buckets = std::bit_ceil(std::max<size_t>(2 * count, 2));
    if (ids_.size() < buckets) {
      ids_.assign(buckets, 0);
      orders_.assign(buckets, 0);
      epochs_.assign(buckets, 0);
      epoch_ = 0;
    }
    if (++epoch_ == 0) {
      std::fill(epochs_.begin(), epochs_.end(), 0);
      epoch_ = 1;
    }
    mask_ = buckets - 1;
    shift_ = 64 - uint32_t(std::countr_zero(buckets));
    size_ = 0;
  }
  // The order `id` was first added in, and whether this call added it.
  std::pair<uint32_t, bool> add(uint32_t id) {
    size_t b = size_t(hashBucket(id, shift_));
    while (epochs_[b] == epoch_ && ids_[b] != id) b = (b + 1) & mask_;
    if (epochs_[b] == epoch_) return {orders_[b], false};
    epochs_[b] = epoch_;
    ids_[b] = id;
    orders_[b] = size_++;
    return {orders_[b], true};
  }

private:
  std::vector<uint32_t> ids_, orders_, epochs_;
  uint32_t epoch_ = 0, size_ = 0, shift_ = 0;
  size_t mask_ = 0;
};

} // namespace

uint64_t SparseRowStore::cacheBytesFor(uint64_t rows, uint32_t rowBytes) noexcept {
  if (!rows) return 0;
  // Row bytes, its id, two recency links; buckets of 32-bit slot numbers.
  const uint64_t perRow = uint64_t{rowBytes} + 3 * sizeof(uint32_t);
  if (rows > std::numeric_limits<uint64_t>::max() / 4 / perRow) return std::numeric_limits<uint64_t>::max();
  return rows * perRow + indexBuckets(rows) * sizeof(uint32_t);
}

uint64_t SparseRowStore::workingBytes(uint32_t rowBytes, uint32_t prefetchRows, uint32_t stageRows) noexcept {
  // A prefetched row's bytes, id and destination; a stage's dedupe map (three
  // words in each of twice as many buckets), distinct and missing ids, and
  // its read job's id and destination per row.
  const uint64_t prefetch = uint64_t{prefetchRows} * (uint64_t{rowBytes} + sizeof(uint32_t) + sizeof(uint8_t *));
  const uint64_t stage = uint64_t{stageRows} * (6 * sizeof(uint32_t) + 2 * sizeof(uint32_t) +
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
  // A prefetch: ids[i]'s row into its own rows, read on the readers
  // (`batch`); `ended` once its rows were cached.
  struct Prefetch final {
    std::vector<uint32_t> ids;
    std::vector<uint8_t> rows;
    std::shared_ptr<WeightReaders::Batch> batch;
    bool ended = false; // under prefetchMutex
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

  // Stages, which are serialized, and prefetches: the distinct ids of the
  // last of each, and the prefetches not yet stopped or finished, whose rows
  // prefetchingRows count and prefetchingIds hold until they end.
  std::mutex stageMutex;
  DistinctIds distinct;
  std::vector<uint32_t> unique, missing;
  std::vector<uint8_t *> destinations;
  std::mutex prefetchMutex;
  DistinctIds prefetchDistinct;
  std::vector<std::shared_ptr<Prefetch>> prefetching;
  uint64_t prefetchingRows = 0;
  std::unordered_set<uint32_t> prefetchingIds;

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
      : table(std::move(table)), reader(std::move(reader)), options(std::move(options)) {}

  [[nodiscard]] uint8_t *slotRow(uint32_t slot) const noexcept { return rows.get() + uint64_t{slot} * table.rowBytes; }
  [[nodiscard]] uint64_t bucketOf(uint32_t id) const noexcept { return hashBucket(id, bucketShift); }

  void allocate(uint64_t cacheBytes) {
    const uint64_t working = workingBytes(table.rowBytes, options.prefetchRows, options.stageRows);
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
    const auto started = Clock::now();
    reader->read(table.offset + uint64_t{id} * table.rowBytes, {destination, table.rowBytes});
    readLatency.add(nanosecondsSince(started));
    readBytes.fetch_add(table.rowBytes, std::memory_order_relaxed);
  }
  void count(const WeightReaders::Tally &tally) {
    readRetries.fetch_add(tally.retries, std::memory_order_relaxed);
    readErrors.fetch_add(tally.failures, std::memory_order_relaxed);
  }

  // Runs on the thread that read a prefetch's last row: its rows enter the
  // cache once the source is unchanged since they were read, but a row that
  // could not be read, a hint, is dropped; a written source poisons the
  // store for the next stage.
  void endPrefetch(Prefetch &prefetch, const WeightReaders::Batch &batch) {
    count(batch.tally());
    try {
      reader->checkUnchanged();
    } catch (const std::exception &error) {
      poison(error.what());
    }
    uint64_t read = 0;
    if (!poisoned.load(std::memory_order_acquire)) {
      const std::lock_guard lock(cacheMutex);
      for (size_t index = 0; index < prefetch.ids.size(); ++index)
        if (!batch.failed(index)) {
          insert(prefetch.ids[index], prefetch.rows.data() + index * table.rowBytes);
          ++read;
        }
    }
    prefetchRead.fetch_add(read, std::memory_order_relaxed);
    prefetchDropped.fetch_add(prefetch.ids.size() - read, std::memory_order_relaxed);
    const std::lock_guard lock(prefetchMutex);
    prefetch.ended = true;
    prefetchingRows -= prefetch.ids.size();
    for (const uint32_t id : prefetch.ids) prefetchingIds.erase(id);
  }

  // The prefetches in flight; forgets those that ended, whose readers no
  // longer run them.
  std::vector<std::shared_ptr<Prefetch>> inFlight() {
    const std::lock_guard lock(prefetchMutex);
    std::erase_if(prefetching, [](const std::shared_ptr<Prefetch> &prefetch) { return prefetch->ended; });
    return prefetching;
  }

  // Finishes the prefetches in flight here and on the readers.
  void adoptPrefetches() {
    for (const std::shared_ptr<Prefetch> &prefetch : inFlight()) options.readers->finish(*prefetch->batch);
  }
};

SparseRowStore::SparseRowStore(SparseRowTable table, std::unique_ptr<RowReader> reader,
                               SparseRowStoreOptions options) {
  table.validate();
  if (!reader) throw std::invalid_argument("streamed table " + table.name + " has no reader");
  if (table.offset + table.rows * table.rowBytes > reader->size())
    throw std::invalid_argument("streamed table " + table.name + " ends past its file");
  if (!options.readers) throw std::invalid_argument("streamed table " + table.name + " has no readers");
  const uint64_t cacheBytes = options.cacheBytes;
  impl_ = std::make_unique<Impl>(std::move(table), std::move(reader), std::move(options));
  impl_->allocate(cacheBytes);
}

SparseRowStore::~SparseRowStore() {
  // A prefetch's rows enter the cache, which goes with the store.
  for (const std::shared_ptr<Impl::Prefetch> &prefetch : impl_->inFlight())
    impl_->options.readers->stop(*prefetch->batch);
}

const SparseRowTable &SparseRowStore::table() const noexcept { return impl_->table; }

uint64_t SparseRowStore::committedBytes() const noexcept {
  return cacheBytesFor(impl_->capacity, impl_->table.rowBytes) +
         workingBytes(impl_->table.rowBytes, impl_->options.prefetchRows, impl_->options.stageRows);
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
  s.distinct.begin(ids.size());
  s.unique.clear();
  for (size_t i = 0; i < ids.size(); ++i) {
    const uint32_t id = ids[i];
    if (id >= s.table.rows)
      throw SparseRowError("row " + std::to_string(id) + " is past the end of " + s.table.name);
    const auto [order, added] = s.distinct.add(id);
    if (added) s.unique.push_back(id);
    slots[i] = order;
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
    s.destinations.clear();
    for (const uint32_t u : s.missing) s.destinations.push_back(rows.data() + uint64_t{u} * rowBytes);
    // No reader writes into `rows` once the reads returned.
    WeightReaders::Tally tally;
    try {
      s.options.readers->run(
          s.missing.size(), [&](size_t index) { s.readRow(s.unique[s.missing[index]], s.destinations[index]); },
          &tally);
      s.count(tally);
      s.reader->checkUnchanged();
    } catch (const std::exception &error) {
      s.count(tally);
      s.poison(error.what());
      s.throwPoisoned();
    }
    const std::lock_guard lock(s.cacheMutex);
    for (size_t index = 0; index < s.missing.size(); ++index)
      s.insert(s.unique[s.missing[index]], s.destinations[index]);
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
  // Distinct ids within the table, neither cached nor being read ahead, up
  // to the bound.
  s.prefetchDistinct.begin(ids.size());
  auto prefetch = std::make_shared<Impl::Prefetch>();
  {
    const std::lock_guard lock(s.cacheMutex);
    for (const uint32_t id : ids)
      if (id < s.table.rows && s.prefetchDistinct.add(id).second && s.find(id) == kNoSlot &&
          !s.prefetchingIds.contains(id))
        prefetch->ids.push_back(id);
  }
  std::erase_if(s.prefetching, [](const std::shared_ptr<Impl::Prefetch> &each) { return each->ended; });
  const uint64_t room = s.options.prefetchRows > s.prefetchingRows ? s.options.prefetchRows - s.prefetchingRows : 0;
  if (prefetch->ids.size() > room) {
    s.prefetchDropped.fetch_add(prefetch->ids.size() - room, std::memory_order_relaxed);
    prefetch->ids.resize(room);
  }
  if (prefetch->ids.empty()) return;
  prefetch->rows.resize(prefetch->ids.size() * s.table.rowBytes);
  s.prefetchingRows += prefetch->ids.size();
  s.prefetchingIds.insert(prefetch->ids.begin(), prefetch->ids.end());
  s.prefetching.push_back(prefetch);
  // The batch ends its prefetch, which the store holds until then.
  Impl::Prefetch *const reading = prefetch.get();
  prefetch->batch = s.options.readers->start(
      reading->ids.size(),
      [&s, reading](size_t index) {
        s.readRow(reading->ids[index], reading->rows.data() + index * s.table.rowBytes);
      },
      std::max(1u, s.options.readers->readers()),
      [&s, reading](const WeightReaders::Batch &batch) { s.endPrefetch(*reading, batch); });
}

void SparseRowStore::drainPrefetch() { impl_->adoptPrefetches(); }

SparseRowStats SparseRowStore::stats() const {
  const Impl &s = *impl_;
  SparseRowStats result;
  result.table = s.table.name;
  result.tableBytes = s.table.rows * s.table.rowBytes;
  result.cacheBytes = committedBytes();
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

std::unique_ptr<RowReader> openRowReader(std::shared_ptr<const WeightSource> source) {
  return std::make_unique<SourceRowReader>(std::move(source));
}

} // namespace splash::model
