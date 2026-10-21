// A sparse row store stages exactly the rows of the source a command reads,
// each distinct row once; caches them within its byte budget, least recently
// used first out; reads with its readers and the staging thread, so a stage
// completes while the readers are blocked; never caches a row of a failed
// read or of a written source, and fails every later stage once poisoned;
// and reads a prefetch's rows ahead, below stages, within its bound,
// skipping rows an earlier prefetch still reads and dropping a row it fails
// to read; and a store that goes while a prefetch reads waits for it.
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/SparseRowStore.hpp"
#include "model/WeightSource.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>

using namespace splash::model;
using splash::test::rejects;
using splash::test::require;

namespace {

// The row width the tests run at, which main sets to each width in turn: rows
// of 160 values, as Qwen3.8-Flash-Next's PLE table's, in IQ4_NL (90 bytes)
// and in Q8_0 (170 bytes, five 34-byte blocks).
uint32_t rowBytes = 0;
constexpr uint64_t kOffset = 77; // rows start past a header
// The ids every test store reserves stage scratch for.
constexpr uint32_t kStageRows = 1024;

std::vector<uint8_t> randomBytes(size_t count, uint32_t seed) {
  std::mt19937 random(seed);
  std::vector<uint8_t> bytes(count);
  for (uint8_t &byte : bytes) byte = uint8_t(random());
  return bytes;
}

// Rows held in memory; a hook runs before every read and may fail or block.
class MemoryReader final : public RowReader {
public:
  explicit MemoryReader(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}
  void read(uint64_t offset, std::span<uint8_t> out) override {
    reads.fetch_add(1);
    if (hook) hook((offset - kOffset) / rowBytes);
    if (offset + out.size() > bytes_.size()) throw SparseRowError("truncated streamed table: memory");
    std::memcpy(out.data(), bytes_.data() + offset, out.size());
  }
  void checkUnchanged() override {
    if (changed) throw SparseRowError("streamed table source was written while the model is loaded: memory");
  }
  [[nodiscard]] uint64_t size() const override { return bytes_.size(); }

  std::function<void(uint64_t row)> hook;
  std::atomic<bool> changed{false};
  std::atomic<uint64_t> reads{0};

private:
  std::vector<uint8_t> bytes_;
};

// A gate readers wait at until it opens.
struct Gate final {
  std::mutex mutex;
  std::condition_variable opened;
  bool open = false;
  std::atomic<uint32_t> waiting{0};
  void wait() {
    std::unique_lock lock(mutex);
    waiting.fetch_add(1);
    opened.wait(lock, [&] { return open; });
    waiting.fetch_sub(1);
  }
  void release() {
    {
      const std::lock_guard lock(mutex);
      open = true;
    }
    opened.notify_all();
  }
};

SparseRowTable tableOf(uint64_t rows) { return {"test-table", rows, rowBytes, kOffset}; }

// The bytes of a store whose cache holds `rows` rows.
uint64_t bytesFor(uint64_t rows, uint32_t prefetchRows) {
  return SparseRowStore::cacheBytesFor(rows, rowBytes) +
         SparseRowStore::workingBytes(rowBytes, prefetchRows, kStageRows);
}

// A store's options: its bytes, `readers` readers of its own, the prefetch
// bound and the stage scratch.
SparseRowStoreOptions optionsOf(uint64_t bytes, uint32_t readers, uint32_t prefetchRows) {
  return {bytes, std::make_shared<SsdReaders>(readers), prefetchRows, kStageRows};
}

struct Fixture final {
  std::vector<uint8_t> bytes;
  MemoryReader *reader = nullptr;
  std::unique_ptr<SparseRowStore> store;
  Fixture(uint64_t rows, uint64_t cacheRows, uint32_t readers = 4, uint32_t prefetchRows = 1u << 16) {
    bytes = randomBytes(kOffset + rows * rowBytes, uint32_t(rows));
    auto owned = std::make_unique<MemoryReader>(bytes);
    reader = owned.get();
    store = std::make_unique<SparseRowStore>(tableOf(rows), std::move(owned),
                                             optionsOf(bytesFor(cacheRows, prefetchRows), readers, prefetchRows));
    require(store->stats().cacheCapacityRows == cacheRows, "a store's cache holds other than its rows");
    require(store->committedBytes() <= bytesFor(cacheRows, prefetchRows), "a store holds more than its bytes");
  }
  const uint8_t *row(uint32_t id) const { return bytes.data() + kOffset + uint64_t{id} * rowBytes; }
  // Stages `ids` and checks every id's staged row against the source.
  SparseRowStageResult stage(const std::vector<uint32_t> &ids, bool adopt = false,
                             SparseRowUse use = SparseRowUse::Decode) {
    std::vector<uint8_t> rows(ids.size() * rowBytes);
    std::vector<uint32_t> slots(ids.size());
    const SparseRowStageResult result = store->stage(ids, rows, slots, use, adopt);
    std::set<uint32_t> distinct(ids.begin(), ids.end());
    require(result.uniqueRows == distinct.size(), "staged rows are not the distinct ids");
    for (size_t i = 0; i < ids.size(); ++i) {
      require(slots[i] < result.uniqueRows, "a slot is past the staged rows");
      require(std::memcmp(rows.data() + uint64_t{slots[i]} * rowBytes, row(ids[i]), rowBytes) == 0,
              "a staged row differs from its source row");
    }
    return result;
  }
};

void cacheBytesHoldTheirRows() {
  for (const uint64_t budget : {uint64_t{1} << 20, uint64_t{64} << 20, uint64_t{1} << 30, uint64_t{3} << 30}) {
    const uint64_t rows = SparseRowStore::cacheRowsFor(budget, rowBytes);
    require(rows > 0 && SparseRowStore::cacheBytesFor(rows, rowBytes) <= budget,
            "a cache's rows take more than its budget");
    require(SparseRowStore::cacheBytesFor(rows + 1, rowBytes) > budget, "a cache's budget holds another row");
  }
  require(SparseRowStore::cacheRowsFor(100, rowBytes) == 0, "a budget below one row holds a row");
  // The row, its id and two links, and at least two 4-byte buckets: 20 to 28
  // bytes a row more than its own.
  const uint64_t gib = SparseRowStore::cacheRowsFor(uint64_t{1} << 30, rowBytes);
  require(gib > (uint64_t{1} << 30) / (rowBytes + 28) && gib < (uint64_t{1} << 30) / (rowBytes + 20),
          "1 GiB holds an unexpected row count");
  rejects([] { Fixture(1000, 0); }, "holds no row", "a cache without a row was accepted");
}

void stagesExactRowsOnce() {
  Fixture f(100'000, 20'000);
  std::mt19937 random(7);
  for (int batch = 0; batch < 300; ++batch) {
    std::vector<uint32_t> ids(1 + random() % 4096);
    for (uint32_t &id : ids) id = random() % (batch % 3 ? 100'000 : 3'000); // some batches repeat rows
    const auto before = f.reader->reads.load();
    const SparseRowStageResult result = f.stage(ids);
    require(f.reader->reads.load() - before == result.readRows, "a stage read a row more than once");
  }
  const SparseRowStats stats = f.store->stats();
  require(stats.cachedRows <= stats.cacheCapacityRows, "the cache holds more rows than it has room for");
  require(stats.uses[1].stages == 300 && !stats.poisoned, "stage statistics are wrong");
}

void evictsLeastRecentlyUsed() {
  Fixture f(1000, 100, 2);
  std::vector<uint32_t> first(100), second(50), third(50);
  for (uint32_t i = 0; i < 100; ++i) first[i] = i;
  for (uint32_t i = 0; i < 50; ++i) second[i] = 100 + i, third[i] = 50 + i;
  require(f.stage(first).readRows == 100, "a cold stage hit");
  require(f.stage(first).hitRows == 100, "a warm stage missed");
  require(f.stage(second).readRows == 50, "new rows hit");
  // 0..49 were the least recently used, 50..99 stay.
  require(f.stage(third).hitRows == 50, "recently used rows were evicted");
  require(f.stage({0, 1, 2}).readRows == 3, "least recently used rows were kept");
  require(f.store->stats().evictions == 53, "evictions are miscounted");
}

void retriesOnceThenPoisons() {
  Fixture f(1000, 500, 2);
  std::atomic<int> failures{1};
  f.reader->hook = [&](uint64_t row) {
    if (row == 5 && failures.fetch_sub(1) > 0) throw SparseRowError("injected EIO");
  };
  f.stage({4, 5, 6});
  require(f.store->stats().readRetries == 1 && !f.store->stats().poisoned, "a transient error was not retried");
  failures = 2;
  f.reader->hook = [&](uint64_t row) {
    if (row == 7 && failures.fetch_sub(1) > 0) throw SparseRowError("injected EIO");
  };
  const uint64_t cached = f.store->stats().cachedRows;
  rejects([&] { f.stage({7, 8, 9}); }, "injected EIO", "a failed read was accepted");
  require(f.store->stats().poisoned && f.store->stats().cachedRows == cached,
          "a failed batch's rows were cached or the store was not poisoned");
  f.reader->hook = nullptr;
  rejects([&] { f.stage({4}); }, "unavailable", "a poisoned store staged a cached row");
  const uint64_t reads = f.reader->reads.load();
  f.store->prefetch(std::vector<uint32_t>{100, 101});
  f.store->drainPrefetch();
  require(f.reader->reads.load() == reads && !f.store->stats().prefetchRequestedRows,
          "a poisoned store took a prefetch");
}

void aWrittenSourcePoisons() {
  Fixture f(1000, 500, 2);
  f.stage({1, 2, 3});
  f.reader->changed = true;
  // Every row is cached, and the source is still checked.
  rejects([&] { f.stage({1, 2, 3}); }, "written while", "a written source was accepted");
  require(f.store->stats().poisoned, "a written source did not poison the store");
}

void aWriteDuringReadsCachesNothing() {
  Fixture f(1000, 500, 2);
  f.reader->hook = [&](uint64_t) { f.reader->changed = true; };
  rejects([&] { f.stage({10, 11}); }, "written while", "rows read while the source changed were accepted");
  require(f.store->stats().cachedRows == 0, "rows read while the source changed were cached");
}

void rejectsBadCallsWithoutPoison() {
  Fixture f(1000, 500, 2);
  rejects([&] { f.stage({1, 1000}); }, "past the end", "an id past the table was staged");
  std::vector<uint8_t> rows(rowBytes);
  std::vector<uint32_t> slots(2), ids{1, 2};
  rejects([&] { f.store->stage(ids, rows, slots, SparseRowUse::Decode); }, "smaller than its rows",
          "too small a staging was accepted");
  require(!f.store->stats().poisoned, "a caller's error poisoned the store");
  f.stage({1, 2});
  // A table past its file is refused.
  rejects([] {
    SparseRowStore(tableOf(10), std::make_unique<MemoryReader>(std::vector<uint8_t>(kOffset + 9 * rowBytes)),
                   optionsOf(bytesFor(10, 16), 1, 16));
  }, "ends past its file", "a table past its file was accepted");
  rejects([] { SparseRowTable{"wide", uint64_t{1} << 32, rowBytes, 0}.validate(); }, "32-bit",
          "a table past 32-bit ids was accepted");
}

void stagesWithoutReaders() {
  Fixture f(5000, 64, 0, 32);
  std::vector<uint32_t> ids(32);
  for (uint32_t batch = 0; batch < 300; ++batch) {
    for (uint32_t i = 0; i < ids.size(); ++i) ids[i] = (batch * 32 + i) % 5000;
    require(f.stage(ids).readRows == ids.size(), "a store without readers did not read cold rows");
  }
  require(f.store->stats().evictions > 9000, "zero-reader stages did not turn the cache over");
  // Each prefetch fills the bound of 32 rows: one kept past its stage or
  // drain would leave the next no room.
  for (uint32_t batch = 0; batch < 300; ++batch) {
    for (uint32_t i = 0; i < ids.size(); ++i) ids[i] = (batch * 32 + i + 2000) % 5000;
    const uint64_t reads = f.reader->reads.load();
    f.store->prefetch(ids);
    require(f.reader->reads.load() == reads, "a zero-reader prefetch was read before a stage or a drain");
    if (batch % 2) f.store->drainPrefetch(); // the caller helps without readers
    const auto result = f.stage(ids, true);
    require(result.readRows == 0 && result.hitRows == ids.size(), "zero-reader adoption read rows twice");
    require(f.reader->reads.load() - reads == ids.size(), "zero-reader prefetch read rows twice");
  }
  require(f.store->stats().prefetchReadRows == 300 * ids.size() && !f.store->stats().prefetchDroppedRows,
          "zero-reader prefetches stopped turning over");
}

// Every reader is held inside a prefetch's read: the stage reads its rows
// itself.
void stagesWhileReadersAreBlocked() {
  Fixture f(5000, 2000, 3);
  Gate gate;
  f.reader->hook = [&](uint64_t row) {
    if (row >= 4000) gate.wait();
  };
  std::vector<uint32_t> ahead(64);
  for (uint32_t i = 0; i < ahead.size(); ++i) ahead[i] = 4000 + i;
  f.store->prefetch(ahead);
  while (gate.waiting.load() < 3) std::this_thread::yield();
  require(f.stage({1, 2, 3, 2, 1}).readRows == 3, "a stage behind blocked readers failed");
  gate.release();
  f.store->drainPrefetch();
  require(f.stage(ahead).hitRows == ahead.size(), "prefetched rows were not cached");
  require(f.store->stats().prefetchReadRows == ahead.size(), "prefetched rows are miscounted");
}

// A prefill stage finishes the read-ahead of its own rows and reads none of
// them again.
void adoptsPrefetchedRows() {
  Fixture f(50'000, 40'000, 4);
  std::vector<uint32_t> chunk(8192);
  std::mt19937 random(3);
  for (uint32_t &id : chunk) id = random() % 50'000;
  f.reader->hook = [](uint64_t) { std::this_thread::sleep_for(std::chrono::microseconds(20)); };
  f.store->prefetch(chunk);
  const SparseRowStageResult result = f.stage(chunk, true, SparseRowUse::Prefill);
  require(result.readRows == 0 && result.hitRows == result.uniqueRows, "an adopted prefetch was read again");
  require(f.reader->reads.load() == result.uniqueRows, "rows were read twice");
}

void boundsQueuedPrefetch() {
  Fixture f(1000, 500, 1, 10);
  Gate gate;
  f.reader->hook = [&](uint64_t) { gate.wait(); };
  std::vector<uint32_t> ids(25);
  for (uint32_t i = 0; i < ids.size(); ++i) ids[i] = i;
  f.store->prefetch(ids);
  f.store->prefetch(ids); // the bound is full, and ten rows are being read
  require(f.store->stats().prefetchDroppedRows == 30, "rows past the prefetch bound were not dropped");
  gate.release();
  f.store->drainPrefetch();
  require(f.store->stats().prefetchReadRows == 10, "a bounded prefetch read the wrong rows");
}

// A prefetch skips the rows an earlier one is still reading.
void skipsRowsInFlight() {
  Fixture f(1000, 500, 1, 100);
  Gate gate;
  f.reader->hook = [&](uint64_t) { gate.wait(); };
  std::vector<uint32_t> first(10), second(20);
  for (uint32_t i = 0; i < first.size(); ++i) first[i] = i;
  for (uint32_t i = 0; i < second.size(); ++i) second[i] = i;
  f.store->prefetch(first);
  f.store->prefetch(second);
  gate.release();
  f.store->drainPrefetch();
  require(f.reader->reads.load() == 20 && f.store->stats().prefetchReadRows == 20,
          "a prefetch read again a row another was reading");
}

// A store destroyed while its readers read a prefetch waits for the rows they
// took and their end: nothing touches the store after it went (ASan, TSan).
void tearsDownWithAPrefetchInFlight() {
  Gate gate;
  {
    Fixture f(1000, 500, 2, 100);
    f.reader->hook = [&](uint64_t) { gate.wait(); };
    std::vector<uint32_t> ids(40);
    for (uint32_t i = 0; i < ids.size(); ++i) ids[i] = i;
    f.store->prefetch(ids);
    while (gate.waiting.load() < 2) std::this_thread::yield();
    std::thread opener([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      gate.release();
    });
    f.store.reset();
    opener.join();
  }
}

// A row a prefetch fails to read (twice) is dropped, a hint: the store stays
// usable, and the stage that needs the row reads it.
void dropsAFailedPrefetch() {
  Fixture f(1000, 500, 2);
  std::atomic<int> failures{2};
  f.reader->hook = [&](uint64_t row) {
    if (row == 21 && failures.fetch_sub(1) > 0) throw SparseRowError("injected EIO");
  };
  f.store->prefetch(std::vector<uint32_t>{20, 21, 22});
  f.store->drainPrefetch();
  const SparseRowStats stats = f.store->stats();
  require(!stats.poisoned && stats.prefetchReadRows == 2 && stats.prefetchDroppedRows == 1 &&
              stats.readRetries == 1 && stats.readErrors == 1,
          "a failed prefetch poisoned the store, or its rows are miscounted");
  require(f.stage({20, 21, 22}).readRows == 1, "a stage did not read the row its prefetch failed to");
}

// Stages, prefetches and statistics from several threads (also under TSan).
void concurrentUse() {
  Fixture f(200'000, 30'000, 6, 20'000);
  std::atomic<bool> stop{false};
  std::atomic<int> failures{0};
  std::thread prefetcher([&] {
    std::mt19937 random(11);
    while (!stop) {
      std::vector<uint32_t> ids(1 + random() % 2048);
      for (uint32_t &id : ids) id = random() % 200'000;
      f.store->prefetch(ids);
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  std::thread observer([&] {
    while (!stop) {
      static_cast<void>(f.store->stats());
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  std::vector<std::thread> stagers;
  for (uint32_t t = 0; t < 3; ++t)
    stagers.emplace_back([&, t] {
      std::mt19937 random(100 + t);
      try {
        for (int batch = 0; batch < 150; ++batch) {
          std::vector<uint32_t> ids(1 + random() % 1024);
          for (uint32_t &id : ids) id = random() % 200'000;
          f.stage(ids, batch % 2 == 0, batch % 2 ? SparseRowUse::Decode : SparseRowUse::Prefill);
        }
      } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        failures.fetch_add(1);
      }
    });
  for (std::thread &thread : stagers) thread.join();
  stop = true;
  prefetcher.join();
  observer.join();
  f.store->drainPrefetch();
  require(!failures.load() && !f.store->stats().poisoned, "concurrent stages failed");
}

// The production reader checks the identity its source was opened with: a
// same-size write after the source was opened is a change, and a new file at
// its path is not.
void keepsTheSourceIdentity(const std::filesystem::path &directory) {
  const auto path = directory / "handoff.bin";
  splash::test::writeFile(path, randomBytes(kOffset + 8 * rowBytes, 31));
  const auto source = std::make_shared<WeightSource>(path);
  source->checkUnchanged();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const int writer = open(path.c_str(), O_WRONLY);
  const uint8_t changed = 7;
  require(writer >= 0 && pwrite(writer, &changed, 1, kOffset) == 1, "write after the source was opened");
  close(writer);
  rejects([&] { openRowReader(source)->checkUnchanged(); }, "written while",
          "a write after the source was opened was accepted");
  const auto original = directory / "handoff-original.bin";
  splash::test::writeFile(original, randomBytes(kOffset + 8 * rowBytes, 32));
  const auto unchanged = std::make_shared<WeightSource>(original);
  const auto replacement = directory / "handoff-replacement.bin";
  splash::test::writeFile(replacement, randomBytes(kOffset + 8 * rowBytes, 33));
  std::filesystem::rename(replacement, original);
  openRowReader(unchanged)->checkUnchanged();
}

// The production reader reads the file it was given open, uncached: a write
// to it poisons, a new file at its path changes nothing, a truncation fails.
void readsAnOpenFile(const std::filesystem::path &directory) {
  const auto path = directory / "table.bin";
  const uint64_t rows = 4096;
  const std::vector<uint8_t> bytes = randomBytes(kOffset + rows * rowBytes, 5);
  splash::test::writeFile(path, bytes);
  SparseRowStore store(tableOf(rows), openRowReader(std::make_shared<WeightSource>(path)),
                       optionsOf(bytesFor(1024, 64), 2, 64));
  const auto check = [&](const std::vector<uint32_t> &ids) {
    std::vector<uint8_t> staged(ids.size() * rowBytes);
    std::vector<uint32_t> slots(ids.size());
    store.stage(ids, staged, slots, SparseRowUse::Decode);
    for (size_t i = 0; i < ids.size(); ++i)
      require(std::memcmp(staged.data() + uint64_t{slots[i]} * rowBytes,
                          bytes.data() + kOffset + uint64_t{ids[i]} * rowBytes, rowBytes) == 0,
              "the file's row was staged wrong");
  };
  check({0, 4095, 17, 17, 2048});
  const auto replacement = directory / "replacement.bin";
  splash::test::writeFile(replacement, randomBytes(bytes.size(), 6));
  std::filesystem::rename(replacement, path);
  check({1, 4094, 3000}); // still the file that was opened
  // Writing the opened file: reopen the old inode through the store's
  // reader is not possible now that the path names another file, so write a
  // second fixture and its store.
  const auto second = directory / "second.bin";
  splash::test::writeFile(second, bytes);
  SparseRowStore written(tableOf(rows), openRowReader(std::make_shared<WeightSource>(second)),
                         optionsOf(bytesFor(1024, 64), 2, 64));
  std::this_thread::sleep_for(std::chrono::milliseconds(10)); // a distinct modification time
  const int writer = open(second.c_str(), O_WRONLY);
  const uint8_t value = 1;
  require(writer >= 0 && pwrite(writer, &value, 1, 0) == 1, "write the table fixture");
  close(writer);
  std::vector<uint8_t> staged(rowBytes);
  std::vector<uint32_t> slots(1);
  const std::vector<uint32_t> one{3};
  rejects([&] { written.stage(one, staged, slots, SparseRowUse::Decode); }, "written while",
          "a written file was read");
  // A file cut short after it was opened: its size shows the write.
  const auto third = directory / "third.bin";
  splash::test::writeFile(third, bytes);
  SparseRowStore truncated(tableOf(rows), openRowReader(std::make_shared<WeightSource>(third)),
                           optionsOf(bytesFor(1024, 64), 2, 64));
  require(truncate(third.c_str(), off_t(kOffset + 10 * rowBytes)) == 0, "truncate the table fixture");
  const std::vector<uint32_t> late{4000};
  rejects([&] { truncated.stage(late, staged, slots, SparseRowUse::Decode); }, "written while",
          "a row past a truncated file was read");
  require(truncated.stats().poisoned, "a truncated file did not poison its store");
}

} // namespace

int main() {
  try {
    for (const uint32_t width : {90u, 170u}) {
      rowBytes = width;
      const splash::test::TemporaryDirectory directory("sparse-row-store");
      cacheBytesHoldTheirRows();
      stagesExactRowsOnce();
      evictsLeastRecentlyUsed();
      retriesOnceThenPoisons();
      aWrittenSourcePoisons();
      aWriteDuringReadsCachesNothing();
      rejectsBadCallsWithoutPoison();
      stagesWithoutReaders();
      stagesWhileReadersAreBlocked();
      adoptsPrefetchedRows();
      boundsQueuedPrefetch();
      skipsRowsInFlight();
      tearsDownWithAPrefetchInFlight();
      dropsAFailedPrefetch();
      concurrentUse();
      readsAnOpenFile(directory.path());
      keepsTheSourceIdentity(directory.path());
    }
  } catch (const std::exception &error) {
    std::cerr << "sparse row store test failed at " << rowBytes << "-byte rows: " << error.what() << '\n';
    return 1;
  }
  std::cout << "sparse row store tests passed (90- and 170-byte rows)\n";
  return 0;
}
