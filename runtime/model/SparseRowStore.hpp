#pragma once

// A large table of fixed-size native rows that stays in its source file and
// is read a row at a time: a bounded host cache of rows, read on the model's
// weight readers (model::WeightReaders), and the staging of one command's
// rows into memory the GPU reads. Its user
// names the table (SparseRowTable) and sizes the working memory of its
// stages and prefetches (SparseRowStoreOptions). The store knows nothing of
// what the rows mean: its user computes the row ids, and gathers the staged
// rows with its own kernel.
//
// stage() copies every distinct row a command reads once into the command's
// staging and, for each id, the index of its row there. It returns only once
// every row is in place, so the staging needs no protocol with the GPU: the
// caller stages before it submits the command that reads them, and no reader
// writes into it after stage() returned. The calling thread reads rows too,
// so a stage can finish while readers are blocked on unrelated rows; a stage
// runs ahead of every read in the background, a prefetch's or a streamed
// expert's read-ahead.
//
// A row enters the cache only after the source file was checked unchanged
// after it was read. A read is tried once more when it fails (WeightReaders).
// A stage's failed read, a short file, or a source written since it was
// opened poisons the store: that stage and every later one throw
// SparseRowError, and no row of the failing batch is cached. A prefetch is a
// hint: a row it fails to read is dropped, and the stage that needs it reads
// it again.

#include "model/LatencyHistogram.hpp"
#include "model/WeightReaders.hpp"

#include <cstdint>
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

// How a store reads its source. Production stores read a WeightSource
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

// A reader of the source the table's metadata was parsed from, uncached, as
// it was opened: a new file at its path changes nothing, and a write to the
// open file since it was opened is a change. It holds the source.
[[nodiscard]] std::unique_ptr<RowReader> openRowReader(std::shared_ptr<const WeightSource> source);

struct SparseRowStoreOptions final {
  // The cache and working-memory reservation (committedBytes): workingBytes
  // and, in the rest, rows with their ids, recency links and index. This is
  // not a bound on all host allocations (see workingBytes).
  uint64_t cacheBytes = 0;
  // The readers its reads run on besides the staging thread, which a model's
  // streamed weights share.
  std::shared_ptr<WeightReaders> readers;
  // The most rows prefetches may have queued or in flight, whose bytes are
  // working memory; more are dropped, and with 0 every one.
  uint32_t prefetchRows = 0;
  // The ids of a stage whose scratch the working memory holds; a larger
  // stage grows its scratch past that.
  uint32_t stageRows = 0;
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
    LatencyStats stage;
  };
  Use uses[kSparseRowUses];
  uint64_t readBytes = 0;
  uint64_t readRetries = 0;
  uint64_t readErrors = 0;
  uint64_t evictions = 0;
  // Prefetch: rows asked for, read into the cache and dropped by the bound.
  uint64_t prefetchRequestedRows = 0;
  uint64_t prefetchReadRows = 0;
  uint64_t prefetchDroppedRows = 0;
  LatencyStats read; // one row's read, any thread
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
  // The working memory reserved beside the cache: the payloads of
  // prefetchRows prefetched rows of rowBytes each, and stage scratch for
  // stageRows ids (SparseRowStoreOptions). This does not count prefetch
  // dedupe scratch, the scratch of a larger stage, excess vector capacity,
  // job/queue metadata, allocator overhead or reader thread stacks.
  [[nodiscard]] static uint64_t workingBytes(uint32_t rowBytes, uint32_t prefetchRows, uint32_t stageRows) noexcept;

  // Allocates the cache now. Throws std::invalid_argument when the table is
  // invalid, ends past the reader's file, the options name no readers, or no
  // row fits in options.cacheBytes beside the working memory.
  SparseRowStore(SparseRowTable table, std::unique_ptr<RowReader> reader, SparseRowStoreOptions options);
  // Stops its prefetches; a read in progress finishes first.
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
  // finishes the prefetches in flight (a read-ahead of the rows this stage
  // takes), helping read them; otherwise it leaves them to the readers and
  // may read a row they are reading too. Throws
  // SparseRowError when a row cannot be read (and poisons the store) or an
  // id is past the table (a defect of the caller), std::invalid_argument
  // when rows or slots are too small. Stages are serialized.
  SparseRowStageResult stage(std::span<const uint32_t> ids, std::span<uint8_t> rows, std::span<uint32_t> slots,
                             SparseRowUse use, bool adoptPrefetch = false);

  // Starts reading the rows of `ids` that are not cached into the cache, on
  // the readers, below any stage's reads. Returns at once. Rows beyond the
  // bound of queued rows, and rows it fails to read, are dropped and
  // counted. Ids past the table are ignored (a prefetch is only a hint). A
  // source written since it was opened poisons the store, which the next
  // stage reports.
  void prefetch(std::span<const uint32_t> ids);

  // Finishes the prefetches queued or in flight, reading them here too:
  // tests wait for prefetched rows so.
  void drainPrefetch();

  [[nodiscard]] SparseRowStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
