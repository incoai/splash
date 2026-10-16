#pragma once

// A large table of fixed-size native rows that stays in its source file and
// is read a row at a time (Qwen3.8-Flash-Next's PLE n-gram table with
// --ple-ssd-cache): a bounded host cache of rows, a pool of readers and the
// staging of one command's rows into memory the GPU reads. The store knows
// nothing of what the rows mean: the family hashes its row ids, and gathers
// the staged rows with its own kernel.
//
// stage() copies every distinct row a command reads once into the command's
// staging and, for each id, the index of its row there. It returns only once
// every row is in place, so the staging needs no protocol with the GPU: the
// caller stages before it submits the command that reads them, and no reader
// writes into it after stage() returned. The calling thread reads rows too,
// so a stage can finish while readers are blocked on unrelated rows.
// A row already claimed by a reader still waits for that reader.
//
// A row enters the cache only after the source file was checked unchanged
// after it was read. A failed read (after one retry), a short file, or a
// source written since it was opened poisons the store: that stage and every
// later one throw SparseRowError, and no row of the failing batch is cached.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace splash::model {

class WeightSource;

class SparseRowError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Where a table's rows are in its source file.
struct SparseRowTable final {
  std::string name;
  uint64_t rows = 0;
  uint32_t rowBytes = 0;
  uint64_t offset = 0; // of row 0, from the start of the file
  // Throws std::invalid_argument for a table without rows, with rows wider
  // than kSparseRowMaximumBytes, or with row ids that do not fit 32 bits.
  void validate() const;
};

// The widest row a store takes.
inline constexpr uint32_t kSparseRowMaximumBytes = 64 << 10;
// The most ids a stage's scratch is sized for (a 2,048-row prefill chunk's
// 16 PLE rows each); a larger stage grows it.
inline constexpr uint32_t kSparseRowStageRows = 1u << 15;

// How a store reads its source. Production stores read a WeightSource's file
// (openRowReader); tests inject failures and delays.
class RowReader {
public:
  virtual ~RowReader() = default;
  // Reads bytes.size() bytes at `offset` of the file; throws on failure.
  // Called from several threads at once.
  virtual void read(uint64_t offset, std::span<uint8_t> bytes) = 0;
  // Throws SparseRowError once the file was written since it was opened.
  virtual void checkUnchanged() = 0;
  // The file's size when it was opened.
  [[nodiscard]] virtual uint64_t size() const = 0;
};

// A reader of the file open at `descriptor`, the descriptor whose file the
// table's metadata was parsed from (a WeightSource's): its own duplicate,
// uncached (F_NOCACHE). It holds that file for as long as it lives, so a new
// file at `path` changes nothing (as a WeightSource); a write to the open
// file (its size or modification time) is a change. `path` names the file in
// errors only.
[[nodiscard]] std::unique_ptr<RowReader> openRowReader(int descriptor, const std::filesystem::path &path);
// Keeps the identity of the source the metadata was parsed from across the
// descriptor handoff; a write before the reader's fresh snapshot is refused.
[[nodiscard]] std::unique_ptr<RowReader> openRowReader(const WeightSource &source);

struct SparseRowStoreOptions final {
  // The cache and working-memory reservation (committedBytes): workingBytes
  // and, in the rest, rows with their ids, recency links and index. This is
  // not a bound on all host allocations (see workingBytes).
  uint64_t cacheBytes = 0;
  // Reader threads besides the staging thread.
  uint32_t readers = 16;
  // The most rows a prefetch may have queued or in flight; more are dropped.
  uint32_t prefetchRows = 1u << 16;
};

// What a stage or prefetch was for, which its statistics are kept by.
enum class SparseRowUse : uint8_t { Prefill, Decode };
inline constexpr uint32_t kSparseRowUses = 2;

struct SparseRowStageResult final {
  uint32_t uniqueRows = 0; // rows staged: distinct ids
  uint32_t hitRows = 0;    // of them, read from the cache
  uint32_t readRows = 0;   // of them, read from the source
  uint64_t nanoseconds = 0;
};

// Latency quantiles from a log-2 histogram of microseconds: each value is
// the upper bound of its bucket (so within a factor of two).
struct SparseRowLatency final {
  uint64_t count = 0;
  double totalMilliseconds = 0;
  double maximumMilliseconds = 0;
  double p50Milliseconds = 0, p95Milliseconds = 0, p99Milliseconds = 0;
};

struct SparseRowStats final {
  std::string table;
  uint64_t tableBytes = 0;
  uint64_t cacheBytes = 0;    // cache and working-memory reservation
  uint64_t cacheCapacityRows = 0;
  uint64_t cachedRows = 0;
  // Per SparseRowUse: stages, row ids they took, distinct rows, cache hits
  // and rows read.
  struct Use final {
    uint64_t stages = 0;
    uint64_t requestedRows = 0;
    uint64_t uniqueRows = 0;
    uint64_t hitRows = 0;
    uint64_t readRows = 0;
    SparseRowLatency stage;
  };
  Use uses[kSparseRowUses];
  uint64_t liveReadJobs = 0; // includes jobs still held by active readers
  uint64_t readBytes = 0;
  uint64_t readRetries = 0;
  uint64_t readErrors = 0;
  uint64_t evictions = 0;
  // Prefetch: rows asked for, read into the cache and dropped by the bound.
  uint64_t prefetchRequestedRows = 0;
  uint64_t prefetchReadRows = 0;
  uint64_t prefetchDroppedRows = 0;
  SparseRowLatency read; // one row's read, any thread
  bool poisoned = false;
  std::string failure;
};

class SparseRowStore final {
public:
  // The rows a cache of `cacheBytes` committed bytes holds: each row's bytes,
  // its id and two recency links, and an index of a power of two of buckets,
  // at least two per row. 0 when not even one row fits.
  [[nodiscard]] static uint64_t cacheRowsFor(uint64_t cacheBytes, uint32_t rowBytes) noexcept;
  // The committed bytes of a cache of `rows` rows.
  [[nodiscard]] static uint64_t cacheBytesFor(uint64_t rows, uint32_t rowBytes) noexcept;
  // Reservation beside the cache: prefetch row payloads and stage scratch
  // for kSparseRowStageRows ids. Larger stages grow the scratch. This does
  // not count prefetch dedupe scratch, excess vector capacity, job/queue
  // metadata, allocator overhead or reader thread stacks.
  [[nodiscard]] static uint64_t workingBytes(uint32_t rowBytes, uint32_t prefetchRows) noexcept;

  // Allocates the cache now. Throws std::invalid_argument when the table is
  // invalid, ends past the reader's file, or no row fits in
  // options.cacheBytes beside the working memory.
  SparseRowStore(SparseRowTable table, std::unique_ptr<RowReader> reader, SparseRowStoreOptions options);
  // Stops the readers; a read in progress finishes first.
  ~SparseRowStore();
  SparseRowStore(const SparseRowStore &) = delete;
  SparseRowStore &operator=(const SparseRowStore &) = delete;

  [[nodiscard]] const SparseRowTable &table() const noexcept;
  // Cache bytes plus the fixed working-memory reservation, at most
  // options.cacheBytes; not the store's measured allocation high-water mark.
  [[nodiscard]] uint64_t committedBytes() const noexcept;

  // Stages the rows of `ids` (any order, repeats allowed): row u of `rows`
  // (u * rowBytes) holds the u-th distinct id's row, and slots[i] = u for
  // ids[i]. rows must hold ids.size() rows and slots ids.size() values.
  // Blocks until every row is staged. With `adoptPrefetch`, it first
  // finishes the prefetches in flight (a prefill chunk's read-ahead holds
  // the rows the next chunk stages), helping read them; otherwise it leaves
  // them to the readers and may read a row they are reading too. Throws
  // SparseRowError when a row cannot be read (and poisons the store) or an
  // id is past the table (a defect of the caller), std::invalid_argument
  // when rows or slots are too small. Stages are serialized.
  SparseRowStageResult stage(std::span<const uint32_t> ids, std::span<uint8_t> rows, std::span<uint32_t> slots,
                             SparseRowUse use, bool adoptPrefetch = false);

  // Starts reading the rows of `ids` that are not cached into the cache, on
  // the readers, below any stage's reads. Returns at once. Rows beyond the
  // bound of queued rows are dropped and counted. Ids past the table are
  // ignored (a prefetch is only a hint). A failure poisons the store, which
  // the next stage reports.
  void prefetch(std::span<const uint32_t> ids);

  // Waits until no prefetch is queued or in flight (tests, shutdown).
  void drainPrefetch();

  [[nodiscard]] SparseRowStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
