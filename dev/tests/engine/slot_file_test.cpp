#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "TestChecks.hpp"
#include "model/Crc32c.hpp"
#include "model/SlotFile.hpp"
#include "model/SsdReaders.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <csignal>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// While set, a write of this test's files fails with this error, as a full
// volume fails it: the slot file's pwrite is this one in the test binary,
// which otherwise passes the call on.
static std::atomic<int> failingWrites{0};

extern "C" ssize_t pwrite(int descriptor, const void *data, size_t bytes, off_t offset) {
  if (const int error = failingWrites.load()) {
    errno = error;
    return -1;
  }
  using Write = ssize_t (*)(int, const void *, size_t, off_t);
  static const auto next = reinterpret_cast<Write>(::dlsym(RTLD_NEXT, "pwrite"));
  return next(descriptor, data, bytes, offset);
}

using splash::kHostPageBytes;
using splash::model::DiskBudget;
using splash::model::SlotFile;
using splash::model::SlotRecord;
using splash::model::SsdReaders;

// A slot is its payload rounded up to whole host pages, as uncached IO wants.
static_assert(SlotFile::slotBytesFor(1) == kHostPageBytes &&
              SlotFile::slotBytesFor(kHostPageBytes) == kHostPageBytes &&
              SlotFile::slotBytesFor(kHostPageBytes + 1) == 2 * kHostPageBytes);

using splash::test::rejects;
using splash::test::require;

static void testFailedWriteStopsWriting() {
  const pid_t child = fork();
  require(child >= 0, "failed to isolate file limit test");
  if (!child) {
    int result = 0;
    try {
      constexpr size_t size = kHostPageBytes;
      // The engine starts with the default disposition, which kills the
      // process on a write past the limit, so the file has to change it.
      signal(SIGXFSZ, SIG_DFL);
      auto budget = std::make_shared<DiskBudget>(2 * size);
      SlotFile file(size, budget);
      auto complete = file.acquire();
      auto partial = file.acquire();
      std::vector<std::byte> source(size, std::byte{1}), output(size);
      require(file.write(complete, {source}, {})->wait(), "initial write failed");
      struct rlimit original;
      require(getrlimit(RLIMIT_FSIZE, &original) == 0, "file limit unavailable");
      auto limited = original;
      // The second slot starts at size: allow half its payload before failure.
      limited.rlim_cur = size + size / 2;
      require(setrlimit(RLIMIT_FSIZE, &limited) == 0, "file limit could not be set");
      std::fill(source.begin(), source.end(), std::byte{2});
      require(!file.write(partial, {source}, {})->wait(), "partial write reported success");
      require(!file.read(partial, {output}, {})->wait(), "partially written slot was readable");
      require(budget->writtenBytes() == size + size / 2 && budget->readBytes() == 0,
              "IO counters lost a partial write or counted an invalid read");
      require(setrlimit(RLIMIT_FSIZE, &original) == 0, "file limit restore failed");
      require(!file.writable() && budget->writesStopped() &&
                  file.write(partial, {source}, {}) == nullptr,
              "storage failure did not stop further writes");
      // A closed file still rejects a write that does not fit its slots.
      std::vector<std::byte> oversized(size + 1);
      rejects([&] { static_cast<void>(file.write(partial, {oversized}, {})); },
              "slot write does not match this file's slots",
              "a closed file absorbed a write of more than a slot");
      require(file.read(complete, {output}, {})->wait() && output.front() == std::byte{1},
              "complete slot became unreadable after a storage failure");
      require(budget->readBytes() == size, "successful read bytes were not counted");
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      result = 1;
    }
    _exit(result);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "file failure/integrity test failed");
}

// Callers' memory need not be aligned or fill a slot: spans of any size and
// alignment, from a few bytes to more than one of the worker's chunks, move
// through the worker's own buffer. A write zeros the slot past its spans,
// and a read may take less than was written, scattered differently.
static void testScatteredSpans() {
  // Three of the worker's 1 MiB chunks, the last one partial.
  constexpr size_t size = (2 << 20) + 3 * kHostPageBytes;
  auto budget = std::make_shared<DiskBudget>(size);
  SlotFile file(size, budget);
  auto slot = file.acquire();
  std::vector<std::byte> source(size);
  for (size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<std::byte>(index * 131 + 7);
  // Odd addresses and lengths, an empty span, and one span across the first
  // chunk's end, adding up to less than a slot.
  const std::span<const std::byte> bytes(source);
  const std::vector<std::span<const std::byte>> spans{
      bytes.subspan(1, 7), bytes.subspan(4099, 0), bytes.subspan(4099, (1 << 20) + 12345),
      bytes.subspan(33, 999'983)};
  std::vector<std::byte> expected;
  for (auto span : spans) expected.insert(expected.end(), span.begin(), span.end());
  require(expected.size() < size, "the scattered spans fill the slot");
  expected.resize(size);
  require(file.write(slot, spans, {})->wait() && budget->writtenBytes() == size,
          "a scattered write failed or did not store the whole slot");

  // The whole slot through one unaligned span: the spans in order, zeros after.
  std::vector<std::byte> whole(size + 1);
  require(file.read(slot, {std::span(whole).subspan(1)}, {})->wait() &&
              std::equal(expected.begin(), expected.end(), whole.begin() + 1),
          "a scattered write did not store its spans in order with zeros after");
  // Less than was written, across the first chunk's end into odd addresses:
  // the read takes the two chunks its spans reach and nothing around them.
  std::vector<std::byte> head(3 * 4096 + 3), tail((1 << 20) + 5);
  const uint64_t readBefore = budget->readBytes();
  require(file.read(slot, {std::span(head).subspan(3), std::span(tail).subspan(5)}, {})->wait() &&
              budget->readBytes() - readBefore == 2 << 20,
          "a partial read failed or did not take exactly the chunks it needs");
  require(std::all_of(head.begin(), head.begin() + 3, [](std::byte b) { return b == std::byte{0}; }) &&
              std::all_of(tail.begin(), tail.begin() + 5, [](std::byte b) { return b == std::byte{0}; }) &&
              std::equal(head.begin() + 3, head.end(), expected.begin()) &&
              std::equal(tail.begin() + 5, tail.end(), expected.begin() + (head.size() - 3)),
          "a partial read returned the wrong bytes or wrote outside its spans");
}

// Spans of the given sizes, carved from `storage`: each starts at an address
// aligned like slot offsets, as a Metal buffer does, or where `aligned` is
// false one byte past one.
using Shape = std::vector<std::pair<size_t, bool>>;
static std::vector<std::span<std::byte>> placeSpans(std::vector<std::byte> &storage,
                                                    const Shape &shape) {
  const auto room = [](size_t bytes) { return splash::alignUp(bytes + 1); };
  size_t total = kHostPageBytes;
  for (const auto &[bytes, aligned] : shape) total += room(bytes);
  storage.assign(total, std::byte{0});
  void *base = storage.data();
  size_t space = storage.size();
  auto *cursor = static_cast<std::byte *>(
      std::align(kHostPageBytes, total - kHostPageBytes, base, space));
  std::vector<std::span<std::byte>> spans;
  for (const auto &[bytes, aligned] : shape) {
    spans.emplace_back(cursor + (aligned ? 0 : 1), bytes);
    cursor += room(bytes);
  }
  return spans;
}

// A run of at least one chunk of aligned memory moves straight between the
// memory and the file; the rest of the spans, and a run the slot would hold
// at an unaligned offset, move through the worker's buffer. Either way a
// write stores the spans in order with zeros after them, counts the slot
// once, and a read brings them back into spans of any shape.
static void testAlignedRunsMoveDirectly() {
  constexpr size_t chunk = 1 << 20;
  constexpr size_t size = 3 * chunk;
  auto budget = std::make_shared<DiskBudget>(size);
  SlotFile file(size, budget);
  auto slot = file.acquire();
  const std::vector<Shape> shapes{
      // Two chunks straight, and the run's last half chunk through the buffer.
      {{5 * chunk / 2, true}},
      // The last run starts at an unaligned offset of the slot.
      {{3 * chunk / 2, true}, {100, false}, {chunk, true}},
      // The buffer takes only the host pages before a run, so the run starts
      // at an aligned offset of the slot.
      {{3 * kHostPageBytes, false}, {5 * chunk / 4, true}},
  };
  uint32_t state = 1;
  for (const Shape &shape : shapes) {
    std::vector<std::byte> sourceMemory;
    const auto sources = placeSpans(sourceMemory, shape);
    std::vector<std::byte> expected;
    for (const auto span : sources) {
      for (std::byte &value : span) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<std::byte>(state >> 24);
      }
      expected.insert(expected.end(), span.begin(), span.end());
    }
    expected.resize(size);
    const uint64_t writtenBefore = budget->writtenBytes();
    require(file.write(slot, {sources.begin(), sources.end()}, {})->wait() &&
                budget->writtenBytes() - writtenBefore == size,
            "a write of aligned runs failed or did not store the slot exactly once");

    // The whole slot into one unaligned span, all of it through the buffer.
    std::vector<std::byte> wholeMemory;
    const auto whole = placeSpans(wholeMemory, {{size, false}}).front();
    require(file.read(slot, {whole}, {})->wait() &&
                std::equal(expected.begin(), expected.end(), whole.begin()),
            "aligned runs did not land in order with zeros after them");
    // Back into spans of the same shape, the aligned runs straight.
    std::vector<std::byte> destinationMemory;
    const auto destinations = placeSpans(destinationMemory, shape);
    require(file.read(slot, destinations, {})->wait(), "a read into aligned runs failed");
    for (size_t index = 0; index < sources.size(); ++index) {
      require(std::equal(sources[index].begin(), sources[index].end(),
                         destinations[index].begin()),
              "a read into aligned runs returned the wrong bytes");
    }
  }
}

// A queued write of aligned runs, cancelled behind a parked worker, fails and
// leaves its slot unreadable. The worker sees the cancellation before the
// write's first chunk, so this covers a queued direct write only, not one
// cancelled between two of its chunks.
static void testCancelledDirectWrite() {
  constexpr size_t size = 3 << 20;
  SlotFile file(size, std::make_shared<DiskBudget>(size));
  auto slot = file.acquire();
  std::vector<std::byte> memory;
  const auto source = placeSpans(memory, {{size, true}}).front();
  require(file.write(slot, {source}, {})->wait(), "the first write of aligned runs failed");
  std::promise<void> reached, release;
  auto released = release.get_future().share();
  std::vector<std::byte> output(size);
  auto hold = file.read(slot, {output}, [&] { reached.set_value(); released.wait(); });
  reached.get_future().wait();
  auto cancelled = file.write(slot, {source}, {});
  cancelled->cancel();
  release.set_value();
  require(hold->wait() && !cancelled->wait(), "a cancelled write of aligned runs succeeded");
  require(file.writable() && !file.read(slot, {output}, {})->wait(),
          "a cancelled write of aligned runs left its slot readable or closed the file");
}

// A freed slot gives its blocks back: two files filling one quota in turn
// occupy what the quota holds on disk, not twice it.
static void testFreedSlotsReturnTheirBlocks() {
  constexpr size_t size = kHostPageBytes;
  auto budget = std::make_shared<DiskBudget>(4 * size);
  SlotFile first(size, budget), second(size, budget);
  std::vector<std::byte> source(size, std::byte{7}), output(size);
  const auto fill = [&](SlotFile &file) {
    std::vector<std::shared_ptr<SlotFile::Slot>> slots;
    for (int index = 0; index < 4; ++index) {
      slots.push_back(file.acquire());
      require(slots.back() && file.write(slots.back(), {source}, {})->wait(),
              "a slot of the quota could not be written");
    }
    return slots;
  };
  auto slots = fill(first);
  require(budget->fileBytes() == 4 * size, "written slots were not counted on disk");
  slots.clear();
  // A read of an unwritten slot completes behind the punches queued before it.
  require(!first.read(first.acquire(), {output}, {})->wait() && budget->fileBytes() == 0,
          "freed slots kept their blocks");
  slots = fill(second);
  require(budget->fileBytes() == 4 * size,
          "two files filling one quota in turn occupied more than it");
}

// A punched slot takes a new write like any other: the write lands in the
// hole and reads back. The first slot goes on the worker, which drops the
// write's hold on it last.
static void testPunchedSlotIsReusable() {
  constexpr size_t size = 2 * kHostPageBytes;
  auto budget = std::make_shared<DiskBudget>(size);
  SlotFile file(size, budget);
  std::vector<std::byte> first(size, std::byte{1}), second(size / 2, std::byte{2});
  require(file.write(file.acquire(), {first}, {})->wait(), "the first write failed");
  // The quota holds one slot, so this is the same one.
  auto slot = file.acquire();
  std::vector<std::byte> output(size), expected(size);
  std::copy(second.begin(), second.end(), expected.begin());
  require(slot && file.write(slot, {second}, {})->wait() &&
              file.read(slot, {output}, {})->wait() && output == expected &&
              budget->fileBytes() == size,
          "a punched slot did not take a new write");
}

// A directory of the temporary directory, removed with everything in it.
class Directory final {
public:
  Directory() {
    std::string name = (std::filesystem::temp_directory_path() / "splash-slots-XXXXXX").string();
    require(::mkdtemp(name.data()) != nullptr, "temporary directory could not be made");
    path_ = name;
  }
  ~Directory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  Directory(const Directory &) = delete;
  Directory &operator=(const Directory &) = delete;

  [[nodiscard]] std::filesystem::path slots() const { return path_ / "slots"; }
  [[nodiscard]] std::filesystem::path records() const { return path_ / "records"; }
  [[nodiscard]] SlotFile::Persistence persistence(std::string_view tag = "cache") const {
    const auto bytes = std::as_bytes(std::span(tag));
    return {slots(), records(), {bytes.begin(), bytes.end()}, sizeof(uint64_t)};
  }

private:
  std::filesystem::path path_;
};

static std::vector<std::byte> labelOf(uint64_t value) {
  std::vector<std::byte> label(sizeof(value));
  std::memcpy(label.data(), &value, sizeof(value));
  return label;
}

// Flips one byte of a file in place, as damage after the fact would.
static void flipByte(const std::filesystem::path &path, off_t offset) {
  const int descriptor = ::open(path.c_str(), O_RDWR);
  require(descriptor >= 0, "file to damage could not be opened");
  std::byte value{};
  require(::pread(descriptor, &value, 1, offset) == 1, "byte to damage could not be read");
  value ^= std::byte{0xff};
  require(::pwrite(descriptor, &value, 1, offset) == 1, "byte could not be damaged");
  ::close(descriptor);
}

static uint64_t sizeOf(const std::filesystem::path &path) {
  struct stat status {};
  require(::stat(path.c_str(), &status) == 0, "file could not be measured");
  return static_cast<uint64_t>(status.st_size);
}

// A persistent file keeps written, labelled slots for the next process; a
// slot dropped while the process runs gives up its record, one dropped once
// the file is sealed keeps it, and one never labelled leaves none. The next
// process adopts what it keeps and frees the rest.
static void testPersistentSlotsComeBack() {
  constexpr size_t size = kHostPageBytes;
  Directory directory;
  std::vector<std::byte> one(size / 2, std::byte{1}), two(size, std::byte{2}),
      three(size, std::byte{3});
  {
    auto budget = std::make_shared<DiskBudget>(4 * size);
    SlotFile file(size, budget, directory.persistence());
    require(file.persistent() && file.records().empty(), "a new persistent file listed records");
    file.finishAdoption();
    auto first = file.acquire(), second = file.acquire(), third = file.acquire(),
         unlabelled = file.acquire();
    require(file.write(first, {one}, {})->wait() && file.write(second, {two}, {})->wait() &&
                file.write(third, {three}, {})->wait() &&
                file.write(unlabelled, {three}, {})->wait(),
            "persistent writes failed");
    file.label(first, labelOf(1));
    file.label(second, labelOf(2));
    file.label(third, labelOf(3));
    third.reset();
    require(file.synchronize({})->wait(), "a persistent file did not synchronize");
    file.seal();
  }
  {
    auto budget = std::make_shared<DiskBudget>(4 * size);
    SlotFile file(size, budget, directory.persistence());
    const std::vector<SlotRecord> records = file.records();
    require(records.size() == 2 && records[0].index == 0 && records[0].label == labelOf(1) &&
                records[0].payloadBytes == size / 2 && records[1].label == labelOf(2) &&
                records[1].payloadBytes == size,
            "the next process did not find exactly the labelled slots it kept");
    auto first = file.adopt(records[0]);
    rejects([&] { static_cast<void>(file.adopt(records[0])); },
            "adopted slot is not one of this file's records", "a slot was adopted twice");
    file.finishAdoption();
    require(budget->usedBytes() == size && budget->fileBytes() == size &&
                sizeOf(directory.slots()) == size,
            "adoption did not keep exactly the adopted slot");
    std::vector<std::byte> output(size / 2);
    require(file.read(first, {output}, {})->wait() && output == one,
            "an adopted slot did not read back");
    rejects([&] { static_cast<void>(file.adopt(records[1])); },
            "slots are adopted only while a persistent slot file opens",
            "a slot was adopted after adoption finished");
    file.seal();
  }
  auto budget = std::make_shared<DiskBudget>(4 * size);
  SlotFile file(size, budget, directory.persistence());
  require(file.records().size() == 1 && file.records()[0].label == labelOf(1),
          "a slot the previous process did not adopt kept its record");
}

// The first read of an adopted slot checks its payload against the record:
// damage, or a read that does not cover the payload, fails it, and only it.
// A slot this process wrote is trusted.
static void testAdoptedSlotsCheckTheirFirstRead() {
  constexpr size_t size = kHostPageBytes;
  Directory directory;
  std::vector<std::byte> four(size, std::byte{4}), five(size, std::byte{5}), output(size);
  {
    SlotFile file(size, std::make_shared<DiskBudget>(3 * size), directory.persistence());
    file.finishAdoption();
    const std::vector<std::pair<const std::vector<std::byte> *, uint64_t>> payloads{
        {&four, 4}, {&five, 5}, {&five, 6}};
    std::vector<std::shared_ptr<SlotFile::Slot>> kept;
    for (const auto &[payload, label] : payloads) {
      kept.push_back(file.acquire());
      require(file.write(kept.back(), {*payload}, {})->wait(), "persistent write failed");
      file.label(kept.back(), labelOf(label));
    }
    require(file.synchronize({})->wait(), "a persistent file did not synchronize");
    file.seal();
  }
  flipByte(directory.slots(), 10);
  SlotFile file(size, std::make_shared<DiskBudget>(4 * size), directory.persistence());
  const std::vector<SlotRecord> records = file.records();
  require(records.size() == 3, "the records of three written slots did not come back");
  auto damaged = file.adopt(records[0]);
  auto intact = file.adopt(records[1]);
  auto partial = file.adopt(records[2]);
  file.finishAdoption();
  require(!file.read(damaged, {output}, {})->wait(), "a damaged slot read back");
  require(file.read(intact, {output}, {})->wait() && output == five &&
              file.read(intact, {output}, {})->wait(),
          "an intact adopted slot did not read back");
  std::vector<std::byte> half(size / 2);
  require(!file.read(partial, {half}, {})->wait(),
          "an adopted slot passed a read that did not cover its payload");
  auto fresh = file.acquire();
  require(file.write(fresh, {four}, {})->wait() && file.read(fresh, {half}, {})->wait(),
          "a slot this process wrote was checked against a record");
}

// A file written for another tag empties when it opens; so does a record
// whose bytes changed, alone.
static void testForeignFilesAndDamagedRecordsAreDropped() {
  constexpr size_t size = kHostPageBytes;
  Directory directory;
  std::vector<std::byte> payload(size, std::byte{6});
  const auto fill = [&](std::string_view tag, int slots) {
    SlotFile file(size, std::make_shared<DiskBudget>(4 * size), directory.persistence(tag));
    file.finishAdoption();
    std::vector<std::shared_ptr<SlotFile::Slot>> kept;
    for (int index = 0; index < slots; ++index) {
      kept.push_back(file.acquire());
      require(file.write(kept.back(), {payload}, {})->wait(), "persistent write failed");
      file.label(kept.back(), labelOf(static_cast<uint64_t>(index)));
    }
    require(file.synchronize({})->wait(), "a persistent file did not synchronize");
    file.seal();
  };
  fill("one", 2);
  {
    SlotFile other(size, std::make_shared<DiskBudget>(4 * size), directory.persistence("two"));
    require(other.records().empty() && sizeOf(directory.slots()) == 0,
            "a file written for another tag was taken back");
  }
  fill("one", 2);
  // The second record starts one record after the first, past the header.
  flipByte(directory.records(), 4096 + 64 + 3);
  SlotFile file(size, std::make_shared<DiskBudget>(4 * size), directory.persistence("one"));
  require(file.records().size() == 1 && file.records()[0].label == labelOf(0),
          "a damaged record was taken back, or took an intact one with it");
}

// Adopted slots count against the budget past its capacity, so the caller
// can give up what no longer fits; until it has, nothing new is granted.
// Nothing runs on a persistent file before adoption finishes.
static void testAdoptionCountsPastTheQuota() {
  constexpr size_t size = kHostPageBytes;
  Directory directory;
  std::vector<std::byte> payload(size, std::byte{7});
  {
    SlotFile file(size, std::make_shared<DiskBudget>(3 * size), directory.persistence());
    file.finishAdoption();
    std::vector<std::shared_ptr<SlotFile::Slot>> kept;
    for (int index = 0; index < 3; ++index) {
      kept.push_back(file.acquire());
      require(file.write(kept.back(), {payload}, {})->wait(), "persistent write failed");
      file.label(kept.back(), labelOf(static_cast<uint64_t>(index)));
    }
    require(file.synchronize({})->wait(), "a persistent file did not synchronize");
    file.seal();
  }
  auto budget = std::make_shared<DiskBudget>(size);
  SlotFile file(size, budget, directory.persistence());
  rejects([&] { static_cast<void>(file.acquire()); },
          "a persistent slot file acquires slots only after adoption",
          "a persistent file acquired a slot before adoption finished");
  rejects([&] { static_cast<void>(file.synchronize({})); },
          "a persistent slot file takes operations only after adoption",
          "a persistent file ran an operation before adoption finished");
  std::vector<std::shared_ptr<SlotFile::Slot>> adopted;
  for (const SlotRecord &record : file.records())
    adopted.push_back(file.adopt(record));
  file.finishAdoption();
  require(budget->usedBytes() == 3 * size && !file.acquire(),
          "adoption did not count past the quota, or a slot was granted beyond it");
  adopted.pop_back();
  adopted.pop_back();
  require(budget->usedBytes() == size && !file.acquire(), "a full quota granted a slot");
  adopted.pop_back();
  require(file.acquire() != nullptr, "the quota did not return once adoption gave it up");
}

// The limit bounds live slots below the quota; one that falls below what
// is live grants nothing until enough is freed, and one below a slot takes
// no write at all. A file's measures name its volume and the blocks it
// holds.
static void testTheLimitBoundsLiveSlots() {
  constexpr size_t size = kHostPageBytes;
  auto budget = std::make_shared<DiskBudget>(4 * size);
  SlotFile file(size, budget);
  budget->setLimitBytes(8 * size);
  require(budget->limitBytes() == 4 * size, "the limit passed the quota");
  budget->setLimitBytes(2 * size);
  auto first = file.acquire(), second = file.acquire();
  require(first && second && !file.acquire(), "the limit did not bound live slots");
  budget->setLimitBytes(size - 1);
  require(!file.takesWrite(false) && !file.takesWrite(true),
          "a file took a write under a limit that holds no slot");
  budget->setLimitBytes(size);
  require(file.takesWrite(true), "a file took no write under a limit that holds a slot");
  first.reset();
  require(!file.acquire(), "a slot was granted while more than the limit was live");
  second.reset();
  first = file.acquire();
  require(first != nullptr, "the limit granted nothing once enough was freed");
  const std::vector<std::byte> payload(size, std::byte{3});
  require(file.write(first, {payload}, {})->wait() && file.allocatedBytes() >= size &&
              file.volume() && file.volume()->f_bsize,
          "a written file did not measure its blocks and volume");
}

// Every write takes its whole slot from the write budget as it is
// submitted. Overdrawn, the budget holds writes not in use back until it
// refills, and when it will is known; a copy in use passes and is counted
// all the same; a clean stop lifts it. It refills up to a quarter of a
// day's.
static void testWritesDrawOnTheWriteBudget() {
  constexpr size_t size = kHostPageBytes;
  constexpr double kDay = 86'400'000.0;
  double now = 0.0;
  auto budget = std::make_shared<DiskBudget>(8 * size, [&now] { return now; });
  SlotFile file(size, budget);
  require(file.takesWrite(false) && !budget->admitsAgainAt() && !budget->writeBudgetBytes(),
          "a budget without a write budget held a write back");
  // Four slots a day, one at once.
  budget->setWriteBudget(4 * size);
  require(budget->writeBudgetBytes() == static_cast<int64_t>(size) && file.takesWrite(false),
          "a new write budget did not start full");
  const std::vector<std::byte> half(size / 2, std::byte{1});
  std::vector<std::shared_ptr<SlotFile::Slot>> slots;
  const auto write = [&] {
    slots.push_back(file.acquire());
    require(file.write(slots.back(), {half}, {})->wait(), "a slot write failed");
  };
  write();
  require(!budget->writeBudgetBytes() && file.takesWrite(false),
          "half a slot's payload did not take the whole slot");
  write();
  const auto again = budget->admitsAgainAt();
  require(!file.takesWrite(false) && file.takesWrite(true) && again &&
              std::abs(*again - kDay / 4) < 1.0,
          "an overdrawn budget did not hold back what is not in use alone, or say until when");
  write();
  require(std::abs(*budget->admitsAgainAt() - kDay / 2) < 1.0,
          "a copy in use was not counted");
  now = kDay / 2 + 1.0;
  require(file.takesWrite(false) && !budget->admitsAgainAt(),
          "the budget did not refill at its rate");
  now = 10 * kDay;
  require(budget->writeBudgetBytes() == static_cast<int64_t>(size),
          "the budget refilled past a quarter of a day's");
  write();
  write();
  budget->lift();
  require(file.takesWrite(false) && !budget->admitsAgainAt(), "a lifted budget held a write back");
}

// A write that found the volume full (ENOSPC, EDQUOT) only reports it, for
// the engine to measure the volume: the slot it wrote is not readable, and
// the file goes on taking writes.
static void testFullVolumeIsReported() {
  constexpr size_t size = kHostPageBytes;
  auto budget = std::make_shared<DiskBudget>(4 * size);
  SlotFile file(size, budget);
  const std::vector<std::byte> payload(size, std::byte{5});
  std::vector<std::byte> output(size);
  for (const int error : {ENOSPC, EDQUOT}) {
    auto slot = file.acquire();
    failingWrites = error;
    const bool written = file.write(slot, {payload}, {})->wait();
    failingWrites = 0;
    require(!written && budget->takeFullReport() && file.writable() && !budget->writesStopped() &&
                !file.read(slot, {output}, {})->wait(),
            "a write that found the volume full was not reported, or stopped the file");
    require(file.write(slot, {payload}, {})->wait() && file.read(slot, {output}, {})->wait() &&
                output == payload && !budget->fullReported(),
            "the file took no write after its volume was full");
  }
}

// Bytes that differ from one place and seed to the next.
static std::vector<std::byte> patterned(size_t size, uint32_t seed) {
  std::vector<std::byte> bytes(size);
  uint32_t state = seed;
  for (std::byte &value : bytes) {
    state = state * 1664525u + 1013904223u;
    value = static_cast<std::byte>(state >> 24);
  }
  return bytes;
}

// Memory aligned for reads beside the worker (SlotFile::readAt).
class AlignedBytes final {
public:
  explicit AlignedBytes(size_t size) : size_(size) {
    void *memory = nullptr;
    require(::posix_memalign(&memory, splash::kSlotReadAlignmentBytes, size) == 0,
            "aligned memory could not be allocated");
    memory_.reset(static_cast<std::byte *>(memory));
  }
  [[nodiscard]] std::span<std::byte> span() const noexcept { return {memory_.get(), size_}; }

private:
  struct Free {
    void operator()(std::byte *memory) const noexcept { std::free(memory); }
  };
  std::unique_ptr<std::byte, Free> memory_;
  size_t size_;
};

// Every byte of a file.
static std::vector<std::byte> contentsOf(const std::filesystem::path &path) {
  std::vector<std::byte> bytes(sizeOf(path));
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  require(descriptor >= 0 && ::pread(descriptor, bytes.data(), bytes.size(), 0) ==
                                 static_cast<ssize_t>(bytes.size()),
          "file could not be read");
  ::close(descriptor);
  return bytes;
}

// `value` at `offset` of `bytes`, as this host lays it out.
template <class Value> static void put(std::vector<std::byte> &bytes, size_t offset, Value value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

// A slot reads beside the worker once its write landed: the bytes asked for,
// from one pread of the aligned units around them into aligned memory that
// holds those; nothing for an unwritten slot, bytes past the payload, memory
// that is not aligned or too small, or another file's slot.
static void testReadAtReadsAlignedUnits() {
  constexpr size_t size = 4 * kHostPageBytes;
  constexpr uint64_t unit = splash::kSlotReadAlignmentBytes;
  auto budget = std::make_shared<DiskBudget>(4 * size);
  SlotFile file(size, budget), other(size, budget);
  auto slot = file.acquire(), unwritten = file.acquire(), foreign = other.acquire();
  const std::vector<std::byte> payload = patterned(size - 1000, 3);
  require(file.write(slot, {payload}, {})->wait() && other.write(foreign, {payload}, {})->wait(),
          "slot writes failed");
  const AlignedBytes buffer(size);
  const std::vector<std::pair<uint64_t, uint64_t>> ranges{
      {0, 1}, {unit - 1, 2}, {5000, 20000}, {0, payload.size()}, {payload.size(), 0}};
  for (const auto [offset, bytes] : ranges) {
    const uint64_t before = budget->readBytes();
    const auto read = file.readAt(*slot, offset, bytes, buffer.span());
    const uint64_t units = bytes ? (offset + bytes + unit - 1) / unit - offset / unit : 0;
    require(read && read->size() == bytes && std::equal(read->begin(), read->end(), payload.begin() + offset) &&
                budget->readBytes() - before == units * unit,
            "a read beside the worker did not return its bytes from the aligned units around them");
  }
  require(!file.readAt(*unwritten, 0, 1, buffer.span()), "an unwritten slot read back");
  require(!file.readAt(*slot, payload.size() - 1, 2, buffer.span()), "bytes past the payload read back");
  require(!file.readAt(*slot, 0, 1, buffer.span().subspan(1)), "a read landed in unaligned memory");
  require(!file.readAt(*slot, unit - 100, 200, buffer.span().first(unit)),
          "a read landed in memory smaller than its aligned units");
  require(!file.readAt(*foreign, 0, 1, buffer.span()), "another file's slot read back through this one");
}

// Reads beside the worker run on any thread: a slot whose write is queued
// reads nothing until the write lands; written slots read back while the
// worker writes other slots and punches freed ones. Built under TSan too.
static void testReadAtBesideTheWorker() {
  constexpr size_t size = 2 * kHostPageBytes;
  auto budget = std::make_shared<DiskBudget>(16 * size);
  SlotFile file(size, budget);
  const AlignedBytes buffer(size);
  const std::vector<std::byte> first = patterned(size, 1), second = patterned(size, 2);
  auto written = file.acquire();
  require(file.write(written, {first}, {})->wait(), "slot write failed");
  std::promise<void> reached, release;
  auto released = release.get_future().share();
  std::vector<std::byte> output(size);
  auto hold = file.read(written, {output}, [&] {
    reached.set_value();
    released.wait();
  });
  reached.get_future().wait();
  auto queued = file.acquire();
  auto write = file.write(queued, {second}, {});
  require(!file.readAt(*queued, 0, size, buffer.span()), "a slot read back before its write landed");
  release.set_value();
  require(hold->wait() && write->wait(), "queued operations failed");
  const auto landed = file.readAt(*queued, 0, size, buffer.span());
  require(landed && std::ranges::equal(*landed, second), "a slot did not read back once its write landed");

  std::atomic<bool> stop = false;
  std::atomic<int> wrong = 0;
  std::vector<std::thread> readers;
  for (const auto &target : {std::pair{written, &first}, std::pair{queued, &second}}) {
    readers.emplace_back([&, target] {
      const AlignedBytes mine(size);
      for (uint64_t round = 0; !stop.load(); ++round) {
        const uint64_t offset = round * 4099 % (size - 512);
        const auto read = file.readAt(*target.first, offset, 512, mine.span());
        if (!read || !std::equal(read->begin(), read->end(), target.second->begin() + offset)) ++wrong;
      }
    });
  }
  for (uint32_t round = 0; round < 64; ++round) {
    const std::vector<std::byte> payload = patterned(size, round);
    auto other = file.acquire();
    require(other && file.write(other, {payload}, {})->wait(), "a write beside reads failed");
  }
  stop = true;
  for (std::thread &reader : readers) reader.join();
  require(!wrong.load(), "a read beside the worker's writes and punches returned the wrong bytes");
}

// A file of one segment keeps version 1 records byte for byte, whatever its
// options: the header names the tag, label and slot size; a slot's record is
// its payload's size and CRC-32C, its label and a CRC-32C of those.
static void testOneSegmentKeepsVersionOneRecords() {
  constexpr size_t size = kHostPageBytes;
  const std::vector<std::byte> payload = patterned(size - 100, 5);
  for (const SlotFile::Options &options : {SlotFile::Options{}, SlotFile::Options{.segments = {size - 100}}}) {
    Directory directory;
    {
      SlotFile file(size, std::make_shared<DiskBudget>(2 * size), directory.persistence("tag"), options);
      file.finishAdoption();
      auto slot = file.acquire();
      require(file.write(slot, {payload}, {})->wait(), "persistent write failed");
      file.label(slot, labelOf(9));
      require(file.synchronize({})->wait(), "a persistent file did not synchronize");
      file.seal();
    }
    std::vector<std::byte> expected(4096 + 64);
    std::memcpy(expected.data(), "SPLASHSF", 8);
    put(expected, 8, uint32_t{1});
    put(expected, 12, uint32_t{sizeof(uint64_t)});
    put(expected, 16, uint64_t{size});
    put(expected, 24, uint32_t{3});
    std::memcpy(expected.data() + 32, "tag", 3);
    put(expected, 28, splash::model::crc32c(std::span(expected).first(35)));
    put(expected, 4096, uint32_t{0x53524543});
    put(expected, 4100, splash::model::crc32c(payload));
    put(expected, 4104, uint64_t{payload.size()});
    put(expected, 4112, uint64_t{9});
    put(expected, 4120, splash::model::crc32c(std::span(expected).subspan(4096, 24)));
    require(contentsOf(directory.records()) == expected, "a file of one segment did not keep version 1 records");
  }
}

// A file of several segments keeps a CRC-32C of each in version 2 records,
// whose header names the segments run by run after the tag, and takes only
// payloads of exactly their sum. The next process finds each segment's
// checksum; one opening it for other segments, or for one, starts empty, as
// one opening a file of one segment for several does.
static void testSegmentedRecords() {
  constexpr size_t size = 2 * kHostPageBytes;
  const std::vector<uint64_t> segments{8192, 4160, 4160, 4160, 4160};
  const SlotFile::Options options{.segments = segments};
  const std::vector<std::byte> payload = patterned(24832, 7);
  Directory directory;
  const auto fill = [&](const SlotFile::Options &with, size_t bytes) {
    SlotFile file(size, std::make_shared<DiskBudget>(2 * size), directory.persistence("tag"), with);
    file.finishAdoption();
    auto slot = file.acquire();
    require(file.write(slot, {std::span(payload).first(bytes)}, {})->wait(), "persistent write failed");
    file.label(slot, labelOf(4));
    require(file.synchronize({})->wait(), "a persistent file did not synchronize");
    file.seal();
  };
  fill(options, payload.size());
  std::vector<uint32_t> checksums;
  for (uint64_t start = 0; const uint64_t bytes : segments) {
    checksums.push_back(splash::model::crc32c(std::span(payload).subspan(start, bytes)));
    start += bytes;
  }
  std::vector<std::byte> expected(4096 + 64);
  std::memcpy(expected.data(), "SPLASHSF", 8);
  put(expected, 8, uint32_t{2});
  put(expected, 12, uint32_t{sizeof(uint64_t)});
  put(expected, 16, uint64_t{size});
  put(expected, 24, uint32_t{3 + 2 * 16});
  std::memcpy(expected.data() + 32, "tag", 3);
  for (size_t run = 0; const auto [bytes, count] : {std::pair<uint64_t, uint64_t>{8192, 1}, {4160, 4}}) {
    put(expected, 35 + 16 * run, bytes);
    put(expected, 43 + 16 * run++, count);
  }
  put(expected, 28, splash::model::crc32c(std::span(expected).first(67)));
  put(expected, 4096, uint32_t{0x53524543});
  put(expected, 4100, checksums.front());
  put(expected, 4104, uint64_t{payload.size()});
  std::memcpy(expected.data() + 4112, checksums.data() + 1, 16);
  put(expected, 4128, uint64_t{4});
  put(expected, 4136, splash::model::crc32c(std::span(expected).subspan(4096, 40)));
  require(contentsOf(directory.records()) == expected,
          "a file of several segments did not keep version 2 records naming them");
  {
    SlotFile file(size, std::make_shared<DiskBudget>(2 * size), directory.persistence("tag"), options);
    require(file.records().size() == 1 && file.records()[0].checksums == checksums &&
                file.records()[0].payloadBytes == payload.size(),
            "the next process did not find each segment's checksum");
    rejects([&] { static_cast<void>(file.adopt({0, payload.size(), {checksums.front()}, labelOf(4)})); },
            "adopted slot is not one of this file's records", "a record of other segments was adopted");
    file.finishAdoption();
    auto slot = file.acquire();
    rejects([&] { static_cast<void>(file.write(slot, {std::span(payload).first(100)}, {})); },
            "slot write does not match this file's slots", "a payload of other than the segments was written");
    file.seal();
  }
  for (const SlotFile::Options &other :
       {SlotFile::Options{.segments = {8192, 4160, 4160, 8320}}, SlotFile::Options{}}) {
    fill(options, payload.size());
    SlotFile file(size, std::make_shared<DiskBudget>(2 * size), directory.persistence("tag"), other);
    require(file.records().empty() && sizeOf(directory.slots()) == 0,
            "a file opened for other segments took slots of these back");
  }
  fill({}, payload.size());
  SlotFile file(size, std::make_shared<DiskBudget>(2 * size), directory.persistence("tag"), options);
  require(file.records().empty(), "a file opened for several segments took slots of one back");
  for (const std::vector<uint64_t> &bad : {std::vector<uint64_t>{size, 1}, std::vector<uint64_t>{8192, 0, 4096}})
    rejects([&] { SlotFile(size, std::make_shared<DiskBudget>(size), {.segments = bad}); },
            "slot file segments do not fit its slots", "segments past a slot, or empty, were accepted");
}

// A read of an adopted slot checks every segment it covers, widened to
// them whole: damage to one segment fails the reads that cover it and no
// other. The worker's first read checks the whole payload, after which the
// slot is trusted; a slot this process wrote is too.
static void testAdoptedReadsCheckTheirSegments() {
  constexpr size_t size = 2 * kHostPageBytes;
  constexpr uint64_t unit = splash::kSlotReadAlignmentBytes;
  const SlotFile::Options options{.segments = {8192, 4160, 4160, 4160, 4160}};
  const std::vector<std::byte> payload = patterned(24832, 9);
  Directory directory;
  {
    SlotFile file(size, std::make_shared<DiskBudget>(4 * size), directory.persistence(), options);
    file.finishAdoption();
    std::vector<std::shared_ptr<SlotFile::Slot>> kept;
    for (uint64_t label = 0; label < 2; ++label) {
      kept.push_back(file.acquire());
      require(file.write(kept.back(), {payload}, {})->wait(), "persistent write failed");
      file.label(kept.back(), labelOf(label));
    }
    require(file.synchronize({})->wait(), "a persistent file did not synchronize");
    file.seal();
  }
  // The second slot's third segment, its second record.
  flipByte(directory.slots(), static_cast<off_t>(size + 8192 + 4160 + 10));
  auto budget = std::make_shared<DiskBudget>(4 * size);
  SlotFile file(size, budget, directory.persistence(), options);
  require(file.records().size() == 2, "the records of two written slots did not come back");
  auto intact = file.adopt(file.records()[0]), damaged = file.adopt(file.records()[1]);
  file.finishAdoption();
  const AlignedBytes buffer(size);
  const auto reads = [&](const SlotFile::Slot &slot, uint64_t offset, uint64_t bytes) {
    const auto read = file.readAt(slot, offset, bytes, buffer.span());
    return read && std::equal(read->begin(), read->end(), payload.begin() + offset);
  };
  uint64_t before = budget->readBytes();
  require(reads(*intact, 8192 + 100, 50) && budget->readBytes() - before == 2 * unit,
          "a read of an adopted slot did not cover the segment it lies in");
  require(reads(*damaged, 0, 8192) && reads(*damaged, 8192, 4160) && reads(*damaged, 8192 + 2 * 4160, 4160) &&
              !reads(*damaged, 8192 + 4160 + 5, 1) && !reads(*damaged, 8192 + 4000, 300),
          "damage to one segment failed reads that do not cover it, or passed one that does");
  std::vector<std::byte> output(payload.size());
  require(!file.read(damaged, {output}, {})->wait() && file.read(intact, {output}, {})->wait() &&
              output == payload,
          "the worker's first read of an adopted slot did not check its whole payload");
  before = budget->readBytes();
  require(reads(*intact, 8192 + 100, 50) && budget->readBytes() - before == unit,
          "an adopted slot the worker checked was not trusted");
  auto fresh = file.acquire();
  require(file.write(fresh, {payload}, {})->wait() && reads(*fresh, 8192 + 4160 + 5, 1),
          "a slot this process wrote was checked against a record");
}

// Active KV's quota: a commitment locks room within the active limit beside
// the slots requests hold, without taking a slot; a committed page's slot
// draws on it within the active limit while a cached copy's takes the
// limit, each against what the files hold, and no draw takes more than is
// committed.
static void testCommitmentsDrawWithinTheActiveLimit() {
  constexpr size_t size = kHostPageBytes;
  auto budget = std::make_shared<DiskBudget>(10 * size);
  require(budget->activeLimitBytes() == 10 * size, "the active limit was not the quota before a measure");
  budget->setActiveLimitBytes(20 * size);
  require(budget->activeLimitBytes() == 10 * size, "the active limit passed the quota");
  budget->setActiveLimitBytes(8 * size);
  budget->setLimitBytes(2 * size);
  SlotFile file(size, budget);
  require(budget->commit(4 * size, 3 * size) && !budget->commit(2 * size, 3 * size) &&
              budget->commit(size, 3 * size) && budget->committedBytes() == 5 * size,
          "a commitment passed the active limit beside what is held and committed");
  budget->uncommit(size);
  std::vector<std::shared_ptr<SlotFile::Slot>> slots{file.acquire(), file.acquire()};
  require(slots[0] && slots[1] && !file.acquire(), "cached copies passed the limit");
  for (int page = 0; page < 4; ++page) {
    slots.push_back(file.acquire(DiskBudget::Draw::Commitment));
    require(slots.back() && budget->committedBytes() == (3 - page) * size,
            "a committed page did not draw its slot from the commitment within the active limit");
  }
  require(!file.acquire(DiskBudget::Draw::Commitment) && budget->usedBytes() == 6 * size,
          "a draw took more than was committed");
  require(budget->commit(3 * size, 0), "a commitment beside nothing held was refused");
  for (int page = 0; page < 2; ++page) {
    slots.push_back(file.acquire(DiskBudget::Draw::Commitment));
    require(slots.back() != nullptr, "a committed page did not draw its slot within the active limit");
  }
  require(!file.acquire(DiskBudget::Draw::Commitment) && budget->committedBytes() == size,
          "a committed draw passed what the files hold under the active limit");
  slots.pop_back();
  require(file.acquire(DiskBudget::Draw::Commitment) && !budget->committedBytes(),
          "a freed slot's room did not take the committed draw");
}

// A request may start to spill while the write budget covers what Spill
// requests committed: always without one, and once it is lifted.
static void testSpillsWithinTheWriteBudget() {
  constexpr size_t size = kHostPageBytes;
  double now = 0.0;
  auto budget = std::make_shared<DiskBudget>(8 * size, [&now] { return now; });
  require(budget->admitsSpill(uint64_t{1} << 50), "a budget without a write budget refused a spill");
  // Four slots a day, one at once.
  budget->setWriteBudget(4 * size);
  require(budget->admitsSpill(size) && !budget->admitsSpill(size + 1),
          "a write budget did not cover exactly its balance of committed writes");
  SlotFile file(size, budget);
  auto slot = file.acquire();
  const std::vector<std::byte> payload(size, std::byte{2});
  require(file.write(slot, {payload}, {})->wait(), "slot write failed");
  require(budget->admitsSpill(0) && !budget->admitsSpill(1), "a spent write budget covered a committed write");
  // It covers half a slot once it refilled an eighth of a day's, a slot
  // and a half once its refill passes what it holds at once.
  require(!budget->admitsSpillAt(0) && std::abs(*budget->admitsSpillAt(size / 2) - 86'400'000.0 / 8) < 1.0 &&
              std::abs(*budget->admitsSpillAt(3 * size / 2) - 3 * 86'400'000.0 / 8) < 1.0,
          "the write budget did not say when it covers committed writes again");
  now = 86'400'000.0 / 8;
  require(budget->admitsSpill(size / 2) && !budget->admitsSpill(size), "the write budget did not refill");
  budget->lift();
  require(budget->admitsSpill(uint64_t{1} << 50), "a lifted write budget refused a spill");
}

// While reads a step waits for run, each chunk of a write waits for them, at
// most kWriteDeferral, and the wait is counted; without them no chunk
// waits, nor while a read of the file waits behind the write, until the
// worker takes it. Either way the write lands.
static void testWritesDeferToReads() {
  constexpr size_t size = 3 << 20;
  constexpr uint64_t bound = std::chrono::microseconds(SlotFile::kWriteDeferral).count();
  std::atomic<bool> reading = true;
  std::atomic<int> asked = 0;
  auto budget = std::make_shared<DiskBudget>(2 * size);
  SlotFile file(size, budget, {.deferWrites = [&] {
                                 ++asked;
                                 return reading.load();
                               }});
  auto slot = file.acquire();
  const std::vector<std::byte> payload = patterned(size, 11);
  const auto started = splash::AwakeClock::now();
  require(file.write(slot, {payload}, {})->wait(), "a deferred write failed");
  const double waited = splash::millisecondsSince(started);
  const uint64_t deferred = budget->writesDeferredMicroseconds();
  // Three chunks, each held back its bound and at most a loaded machine's
  // oversleep more.
  require(deferred >= 3 * bound && waited * 1000 >= 3 * bound && deferred < 3 * (bound + 20'000),
          "a write's chunks did not wait their bound for reads, or waited longer");
  reading = false;
  asked = 0;
  require(file.write(slot, {payload}, {})->wait() && budget->writesDeferredMicroseconds() == deferred &&
              asked.load() == 3,
          "a write waited without reads to wait for, or did not ask once a chunk");
  std::vector<std::byte> output(size);
  require(file.read(slot, {output}, {})->wait() && output == payload, "a deferred write did not land");

  reading = true;
  std::promise<void> reached, release;
  auto released = release.get_future().share();
  auto hold = file.read(slot, {output}, [&] {
    reached.set_value();
    released.wait();
  });
  reached.get_future().wait();
  const std::vector<std::byte> later = patterned(size, 12);
  auto write = file.write(slot, {later}, {});
  auto behind = file.read(slot, {output}, {});
  release.set_value();
  require(hold->wait() && write->wait() && behind->wait() && output == later &&
              budget->writesDeferredMicroseconds() == deferred,
          "a write held back for other reads while a read of its file waited behind it");
  require(file.write(slot, {payload}, {})->wait() && budget->writesDeferredMicroseconds() >= deferred + 3 * bound,
          "a write did not hold back again once the read behind the last one was taken");
}

// A file's writes watch the readers without keeping them
// (SsdReaders::criticalBusyProbe), so the slots a reader's task holds never
// make it the readers' last owner: a task that holds a file's last slot ends,
// and lets the slot go, after the file and the readers' owner let go of them,
// and the readers stop. Built under TSan too.
static void testSlotsKeepNoReaders() {
  constexpr size_t size = kHostPageBytes;
  auto readers = std::make_shared<SsdReaders>(1);
  const std::weak_ptr<SsdReaders> watched = readers;
  auto file = std::make_unique<SlotFile>(size, std::make_shared<DiskBudget>(size),
                                         SlotFile::Options{.deferWrites = readers->criticalBusyProbe()});
  require(watched.use_count() == 1, "a slot file kept the readers its writes watch");
  std::promise<void> began, release;
  auto batch = readers->start(
      1,
      [slot = file->acquire(), &began, released = release.get_future().share()](size_t) {
        began.set_value();
        released.wait();
      },
      1);
  began.get_future().wait();
  readers->halt(*batch);
  batch.reset();
  file.reset();
  std::thread owner([&] {
    while (!watched.expired()) std::this_thread::yield();
    release.set_value();
  });
  readers.reset();
  owner.join();
}

int main() {
  try {
    testFailedWriteStopsWriting();
    testScatteredSpans();
    testAlignedRunsMoveDirectly();
    testCancelledDirectWrite();
    testFreedSlotsReturnTheirBlocks();
    testPunchedSlotIsReusable();
    testPersistentSlotsComeBack();
    testAdoptedSlotsCheckTheirFirstRead();
    testForeignFilesAndDamagedRecordsAreDropped();
    testAdoptionCountsPastTheQuota();
    testTheLimitBoundsLiveSlots();
    testWritesDrawOnTheWriteBudget();
    testFullVolumeIsReported();
    testReadAtReadsAlignedUnits();
    testReadAtBesideTheWorker();
    testOneSegmentKeepsVersionOneRecords();
    testSegmentedRecords();
    testAdoptedReadsCheckTheirSegments();
    testCommitmentsDrawWithinTheActiveLimit();
    testSpillsWithinTheWriteBudget();
    testWritesDeferToReads();
    testSlotsKeepNoReaders();
    constexpr size_t size = 4 * kHostPageBytes;
    auto budget = std::make_shared<DiskBudget>(size * 2 + 1);
    SlotFile file(size, budget);
    require(file.slotBytes() == size && budget->usedBytes() == 0,
            "file did not report its slot size or took quota");
    auto first = file.acquire();
    auto second = file.acquire();
    require(first && second && budget->usedBytes() == size * 2 && !file.acquire(),
            "a partial slot of the quota was granted");
    std::vector<std::byte> source(size, std::byte{0xa5}), restored(size);
    auto write = file.write(first, {std::span(source).first(128), std::span(source).subspan(128)}, {});
    require(write->wait(), "slot write failed");
    require(file.read(first, {restored}, {})->wait() && restored == source,
            "slot did not roundtrip across IO spans");
    require(!file.read(second, {restored}, {})->wait(), "unwritten slot was readable");
    require(write->ready(), "a write that finished did not report ready");

    // Park the worker in a completion so the queue behind it is deterministic:
    // a queued write, a cancelled read and a cancelled overwrite behind it.
    std::promise<void> reached, release;
    auto released = release.get_future().share();
    auto hold = file.read(first, {restored}, [&] { reached.set_value(); released.wait(); });
    reached.get_future().wait();
    std::vector<std::byte> later(size, std::byte{0x5a});
    write = file.write(second, {later}, {});
    std::vector<std::byte> untouched(size, std::byte{0});
    auto cancelled = file.read(second, {untouched}, {});
    cancelled->cancel();
    auto cancelledWrite = file.write(first, {later}, {});
    cancelledWrite->cancel();
    require(!write->ready() && !cancelled->ready() && !cancelledWrite->ready(),
            "an operation queued behind a parked worker reported ready");
    release.set_value();
    require(hold->wait() && write->wait() && !cancelled->wait() && !cancelledWrite->wait(),
            "queued operations misreported");
    require(hold->ready() && write->ready() && cancelled->ready() && cancelledWrite->ready(),
            "the queue drained but an operation did not report ready");
    require(untouched.front() == std::byte{0} && untouched.back() == std::byte{0},
            "cancelled queued read touched destination");
    require(file.read(second, {restored}, {})->wait() && restored == later,
            "queued write did not land in order");
    require(file.writable() && !file.read(first, {restored}, {})->wait(),
            "cancelled overwrite exposed stale data or disabled the file");
    require(file.write(first, {later}, {})->wait() &&
                file.read(first, {restored}, {})->wait() && restored == later,
            "slot did not recover after a cancelled overwrite");

    // A completion is the caller's wake-up: one that throws is not the
    // worker's to die of.
    require(file.write(second, {later}, [] { throw std::runtime_error("completion"); })->wait(),
            "write with a throwing completion failed");
    require(file.read(second, {restored}, {})->wait() && restored == later,
            "the worker did not survive a throwing completion");
    first.reset();
    second.reset();
    auto reused = file.acquire();
    second = file.acquire();
    require(reused && second && !file.acquire(), "released quota was not reusable");
    rejects([&] { SlotFile(size, std::make_shared<DiskBudget>(size - 1)); },
            "slot file quota holds no slot", "quota below one slot was accepted");
    rejects([&] { SlotFile(size + 1, std::make_shared<DiskBudget>(4 * size)); },
            "slot size is not aligned for uncached IO", "unaligned slot size was accepted");
    std::vector<std::byte> oversized(size + 1);
    rejects([&] { static_cast<void>(file.read(reused, {oversized}, {})); },
            "slot read does not match this file's slots",
            "a read of more than a slot was accepted");
    rejects(
        [&] {
          static_cast<void>(file.write(
              reused, {std::span<const std::byte>(source), std::span<const std::byte>(oversized).first(1)},
              {}));
        },
        "slot write does not match this file's slots", "a write of more than a slot was accepted");
    {
      // Two files of different slot sizes draw on one budget.
      auto shared = std::make_shared<DiskBudget>(4 * size);
      SlotFile small(size, shared);
      SlotFile large(2 * size, shared);
      auto one = large.acquire();
      auto two = small.acquire();
      require(one && two && shared->usedBytes() == 3 * size && !large.acquire(),
              "the shared budget did not bound the second file");
      std::vector<std::byte> largePayload(2 * size, std::byte{3});
      require(large.write(one, {largePayload}, {})->wait() &&
                  small.write(two, {source}, {})->wait(), "shared IO writes failed");
      require(shared->writtenBytes() == 3 * size,
              "shared IO counters did not include both slot files");
      auto three = small.acquire();
      require(three && !small.acquire() && shared->usedBytes() == 4 * size,
              "the last slot of the budget was not granted exactly once");
      one.reset();
      require(shared->usedBytes() == 2 * size && large.acquire() && !small.acquire(),
              "a released slot did not return its bytes to the budget");
      rejects([&] { SlotFile(8 * size, shared); }, "slot file quota holds no slot",
              "a file whose slot exceeds the shared budget was accepted");
    }
    std::cout << "Slot file tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
