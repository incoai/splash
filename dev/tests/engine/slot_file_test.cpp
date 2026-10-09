#include "Checked.hpp"
#include "TestChecks.hpp"
#include "model/SlotFile.hpp"

#include <algorithm>
#include <future>
#include <csignal>
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
#include <utility>
#include <vector>

using splash::kHostPageBytes;
using splash::model::DiskBudget;
using splash::model::SlotFile;
using splash::model::SlotRecord;

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
      require(!file.writable() && file.write(partial, {source}, {}) == nullptr,
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
