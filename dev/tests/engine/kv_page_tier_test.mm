#include "engine/KvPageTier.hpp"
#include "tests/engine/AllocationFailure.hpp"
#include "tests/engine/TestBuffers.hpp"
#include "tests/engine/TestChecks.hpp"
#include "tests/engine/TestDrafts.hpp"
#include "tests/engine/TestPageEntries.hpp"

#include "engine/MemoryGovernor.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/KvCopyRead.hpp"

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace splash;
using engine::KvPageTier;
using engine::KvTransfer;
using engine::WriteClass;
using model::DiskBudget;
using model::SlotFile;

constexpr auto cached = DiskBudget::Draw::Cache;

namespace {

using splash::test::rejects;
using splash::test::require;

std::vector<std::byte> pattern(uint64_t bytes, uint32_t seed) {
  std::vector<std::byte> result(bytes);
  uint32_t state = seed;
  for (std::byte &value : result) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    value = static_cast<std::byte>(state);
  }
  return result;
}

// A page's bytes through the extent's host memory, in the order the tier
// stores them.
std::vector<std::byte> contents(const kv::PageStorage &pages, uint32_t page) {
  std::vector<std::byte> bytes;
  for (const auto span : pages.spans(page)) bytes.insert(bytes.end(), span.begin(), span.end());
  return bytes;
}

// Writes a pattern to a page the same way and returns it.
std::vector<std::byte> fill(const kv::PageStorage &pages, uint32_t page, uint32_t seed) {
  const auto bytes = pattern(pages.bytesPerPage(), seed);
  auto next = bytes.begin();
  for (const auto span : pages.spans(page)) {
    std::copy_n(next, span.size(), span.begin());
    next += static_cast<std::ptrdiff_t>(span.size());
  }
  return bytes;
}

// What a slot holds of a page: its bytes in slot order (kv::SlotFormat).
std::vector<std::byte> payload(const kv::SlotFormat &format, const kv::PageStorage &pages, uint32_t page) {
  std::vector<std::byte> bytes;
  for (const auto span : format.transferSpans(pages, page))
    bytes.insert(bytes.end(), span.begin(), span.end());
  return bytes;
}

std::shared_ptr<SlotFile::Slot> fileSlot(const std::shared_ptr<engine::KvDiskSlot> &slot) {
  return static_cast<KvPageTier::DiskSlot &>(*slot).slot;
}

// The tier's slot file: its slots hold the format's payload, checksummed
// by its segments when persistent.
std::shared_ptr<SlotFile> slotFile(const kv::SlotFormat &format, uint32_t slots) {
  const uint64_t bytes = SlotFile::slotBytesFor(format.payloadBytes());
  return std::make_shared<SlotFile>(bytes, std::make_shared<DiskBudget>(slots * bytes),
                                    SlotFile::Options{.segments = format.segments()});
}

std::vector<std::byte> readSlot(SlotFile &file, const std::shared_ptr<engine::KvDiskSlot> &slot) {
  std::vector<std::byte> bytes(file.slotBytes());
  require(file.read(fileSlot(slot), {bytes}, {})->wait(), "reading a disk slot failed");
  return bytes;
}

// A disk slot holds its page's bytes and zeros up to its aligned size.
bool holdsPage(const std::vector<std::byte> &slot, const std::vector<std::byte> &page) {
  return slot.size() >= page.size() && std::equal(page.begin(), page.end(), slot.begin()) &&
         std::all_of(slot.begin() + static_cast<std::ptrdiff_t>(page.size()), slot.end(),
                     [](std::byte value) { return value == std::byte{0}; });
}

// Drives the tier the way the engine does: restores asked for go out
// together, and polls retire what the worker and the readers did.
void runUntilReady(KvPageTier &tier, KvTransfer &transfer) {
  tier.submitRestores();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (std::chrono::steady_clock::now() < deadline) {
    tier.poll();
    if (transfer.ready()) return;
    std::this_thread::yield();
  }
  throw std::runtime_error("KV transfer did not complete");
}

// Two pages in different extents go from their extents to disk and back.
// Between the two the host overwrites them, so what a restore leaves in a
// page is what the readers read; the pages beside them keep their own bytes.
// A slot holds its page as the layout's slot format lays it out.
void roundTrip(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
               const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, 2 * extent,
                        extent);
  const uint32_t pageA = extent - 1;
  const uint32_t pageB = extent;
  require(pages.allocateExtent(0) && pages.allocateExtent(1) && pages.isAllocated(pageA) &&
              pages.isAllocated(pageB),
          "test pages were not allocated");
  const kv::SlotFormat format(layout);
  const uint64_t slotBytes = SlotFile::slotBytesFor(format.payloadBytes());
  auto file = slotFile(format, 2);
  KvPageTier tier(pages, format, file, readers);
  require(tier.slotBytes() == slotBytes, "tier reports another slot size");

  const auto first = fill(pages, pageA, 0x1234567u);
  const auto second = fill(pages, pageB, 0x89abcdefu);
  const auto beforeA = fill(pages, pageA - 1, 0x600df00du);
  const auto afterB = fill(pages, pageB + 1, 0x0ddba11u);
  require(contents(pages, pageA) == first && contents(pages, pageB) == second,
          "a page does not hold what the host wrote through its spans");

  const auto slottedA = payload(format, pages, pageA);
  const auto slottedB = payload(format, pages, pageB);
  auto slotA = tier.acquireSlot(cached);
  auto slotB = tier.acquireSlot(cached);
  require(slotA && slotB && !tier.acquireSlot(cached), "disk quota was not enforced");
  auto demoteA = tier.demote(pageA, slotA, WriteClass::Reclaim, {});
  auto demoteB = tier.demote(pageB, slotB, WriteClass::Spill, {});
  require(demoteA && demoteB, "demotions within the tier's limit were refused");
  runUntilReady(tier, *demoteA);
  runUntilReady(tier, *demoteB);
  require(demoteA->finish() && demoteB->finish(), "demotions did not succeed");
  require(contents(pages, pageA) == first && contents(pages, pageB) == second,
          "a demotion changed its page");
  require(holdsPage(readSlot(*file, slotA), slottedA) && holdsPage(readSlot(*file, slotB), slottedB),
          "disk slots do not hold their pages as the slot format lays them out");

  static_cast<void>(fill(pages, pageA, 0xdeadbeefu));
  static_cast<void>(fill(pages, pageB, 0xfeedfaceu));
  auto restoreB = tier.restore(slotB, pageB, {});
  auto restoreA = tier.restore(slotA, pageA, {});
  require(restoreA && restoreB, "restores within the tier's limit were refused");
  runUntilReady(tier, *restoreA);
  runUntilReady(tier, *restoreB);
  require(restoreA->finish() && restoreB->finish(), "restores did not succeed");
  require(contents(pages, pageA) == first && contents(pages, pageB) == second,
          "pages did not come back from disk byte for byte");
  require(contents(pages, pageA - 1) == beforeA && contents(pages, pageB + 1) == afterB,
          "a transfer wrote outside its page");

  // A finished transfer holds neither room in the tier nor its slot.
  slotA.reset();
  require(tier.canDemote(WriteClass::Reclaim) && tier.acquireSlot(cached) != nullptr,
          "a released disk slot was not reusable");
  std::cout << "round trip " << kv::formatName(layout.format) << " "
            << layout.attentionLayers << "x" << layout.kvHeads << "x"
            << layout.headDimension << " slot format v" << format.version()
            << ": payload_bytes=" << format.payloadBytes() << " slot_bytes=" << slotBytes << '\n';
}

// A spillable layout's slot holds each block of each layer as a record that
// one aligned read returns through the slot's copy (model::KvCopy), as
// reads of spilled K/V take it; and an Index page's restore reads the
// slot's index alone into the page.
void blockRecords(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                  const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, 2 * extent, extent);
  require(pages.allocateExtent(0) && pages.allocateExtent(1, kv::PageForm::Index),
          "the test extents were not allocated");
  const kv::SlotFormat format(layout);
  KvPageTier tier(pages, format, slotFile(format, 2), readers);
  static_cast<void>(fill(pages, 3, 0x7e57u));
  const auto slotted = payload(format, pages, 3);
  auto slot = tier.acquireSlot(cached);
  auto written = tier.demote(3, slot, WriteClass::Spill, {});
  require(written != nullptr, "the page was not written");
  runUntilReady(tier, *written);
  require(written->finish(), "the page's write failed");
  const std::shared_ptr<const model::KvCopy> copy = tier.copy(slot);
  for (uint32_t layer : {0u, layout.attentionLayers - 1}) {
    for (uint32_t block = 0; block < kv::SlotFormat::kBlocksPerPage; ++block) {
      const kv::SlotRange range = format.recordRange(layer, block);
      const auto bytes = model::readKvCopy(*copy, range);
      require(bytes && std::equal(bytes->begin(), bytes->end(), slotted.begin() + range.offset),
              "a block's record did not come back in one aligned read");
    }
  }
  require(copy->id() == fileSlot(slot)->id(), "a copy did not name its slot's write");

  const uint32_t index = extent + 5;
  auto restore = tier.restore(slot, index, {});
  require(restore != nullptr, "the index restore was refused");
  runUntilReady(tier, *restore);
  const auto indexOf = [&](uint32_t page) {
    std::vector<std::byte> bytes;
    for (const auto span : pages.indexSpans(page)) bytes.insert(bytes.end(), span.begin(), span.end());
    return bytes;
  };
  require(restore->finish() && indexOf(index) == indexOf(3), "an Index page did not take its slot's index");
  rejects([&] { static_cast<void>(tier.demote(index, tier.acquireSlot(cached), WriteClass::Spill, {})); },
          "only a page that holds its K/V", "an Index page was written to the tier");
  std::cout << "block records " << kv::formatName(layout.format) << ": record_bytes=" << format.recordBytes()
            << " pitch=" << format.recordPitch() << '\n';
}

// Writes in flight are bounded, in all and by class: persistent copies take
// at most a quarter of the bound and pages written through at most half, so
// reclaim always keeps a quarter; a write counts until poll() has retired
// it. Restores go to the readers beside them: a read that fails, of a slot
// never written, fails its restore alone. A bound of no write is refused.
void limits(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
            const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  const kv::SlotFormat slots(layout);
  auto file = slotFile(slots, 16);
  rejects([&] { KvPageTier refused(pages, slots, file, readers, 0); }, "room for a write",
          "a tier with no room for a write was built");
  KvPageTier tier(pages, slots, file, readers, 8);
  const auto bytes = fill(pages, 9, 0x51a5e5u);
  std::vector<std::unique_ptr<KvTransfer>> writes;
  const auto write = [&](WriteClass kind) {
    auto started = tier.demote(9, tier.acquireSlot(cached), kind, {});
    if (started) writes.push_back(std::move(started));
    return writes.size();
  };
  require(write(WriteClass::Persist) == 1 && write(WriteClass::Persist) == 2 &&
              !tier.canDemote(WriteClass::Persist) && write(WriteClass::Persist) == 2,
          "persistent copies took more than a quarter of the bound");
  for (int index = 0; index < 4; ++index) static_cast<void>(write(WriteClass::Spill));
  require(writes.size() == 6 && !tier.canDemote(WriteClass::Spill) && tier.canDemote(WriteClass::Reclaim),
          "write-through took more than half of the bound, or left reclaim nothing");
  require(write(WriteClass::Reclaim) == 7 && write(WriteClass::Reclaim) == 8 &&
              !tier.canDemote(WriteClass::Reclaim) && write(WriteClass::Reclaim) == 8,
          "the tier took more writes than its bound");
  // An unwritten slot cannot be restored; the restores beside it can.
  auto written = tier.acquireSlot(cached);
  require(file->write(fileSlot(written), {bytes}, {})->wait(), "seeding failed");
  std::vector<std::unique_ptr<KvTransfer>> restores;
  restores.push_back(tier.restore(written, 0, {}));
  restores.push_back(tier.restore(tier.acquireSlot(cached), 1, {}));
  restores.push_back(tier.restore(written, 2, {}));
  for (auto &restore : restores) runUntilReady(tier, *restore);
  require(restores[0]->finish() && !restores[1]->finish() && restores[2]->finish() &&
              contents(pages, 0) == bytes && contents(pages, 2) == bytes,
          "a restore beside a failed one did not succeed, or an unwritten slot was restored");
  runUntilReady(tier, *writes.front());
  require(writes.front()->finish() && tier.canDemote(WriteClass::Persist),
          "a retired write did not give its room back");
  for (auto &each : writes) runUntilReady(tier, *each);
  std::cout << "transfer limit tests passed\n";
}

// A transfer that cannot be started leaves nothing behind: no room taken in
// the tier and no IO the tier does not track.
void allocationFailure(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                       const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  const uint64_t bytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  for (bool restoring : {false, true}) {
    bool completed = false;
    for (int failure = 0; failure < 64; ++failure) {
      auto budget = std::make_shared<DiskBudget>(bytes);
      auto file = std::make_shared<SlotFile>(bytes, budget);
      KvPageTier tier(pages, kv::SlotFormat(layout), file, readers, 1);
      auto slot = tier.acquireSlot(cached);
      const auto payload = fill(pages, 0, 123);
      require(file->write(fileSlot(slot), {payload}, {})->wait(), "fault slot seed failed");
      std::unique_ptr<KvTransfer> transfer;
      allocationFailureAfter = failure;
      try {
        if (restoring) {
          transfer = tier.restore(slot, 0, {});
          tier.submitRestores();
        } else {
          transfer = tier.demote(0, slot, WriteClass::Reclaim, {});
        }
        allocationFailureAfter = -1;
      } catch (const std::bad_alloc &) {
        allocationFailureAfter = -1;
        // A restore whose batch could not start goes out with the next; a
        // write or a restore that was never taken holds nothing.
        if (transfer) {
          runUntilReady(tier, *transfer);
          require(transfer->finish(), "a restore whose submission failed did not go out with the next");
        }
        transfer.reset();
        slot.reset();
        require(tier.canDemote(WriteClass::Reclaim) && tier.canRestore() && budget->usedBytes() == 0,
                "failed transfer submission took room or left untracked IO");
        continue;
      }
      require(transfer != nullptr, "transfer refused after allocation failures");
      runUntilReady(tier, *transfer);
      require(transfer->finish() && tier.canDemote(WriteClass::Reclaim) && contents(pages, 0) == payload,
              "transfer did not recover");
      completed = true;
      break;
    }
    require(completed, "transfer allocation sweep never reached success");
  }
}

// The write budget holds no KV write back: with it spent the tier still
// takes a demotion, and the demotion's slot is counted against it all the
// same.
void demotionsPassTheWriteBudget(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                                 const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  const uint64_t slotBytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  // The clock stands still: what the budget gives out does not come back.
  auto budget = std::make_shared<DiskBudget>(4 * slotBytes, [] { return 0.0; });
  budget->setWriteBudget(slotBytes);
  auto file = std::make_shared<SlotFile>(slotBytes, budget);
  KvPageTier tier(pages, kv::SlotFormat(layout), file, readers);
  auto first = tier.demote(0, tier.acquireSlot(cached), WriteClass::Reclaim, {});
  require(first != nullptr, "a demotion within the write budget was refused");
  runUntilReady(tier, *first);
  require(first->finish(), "the first demotion failed");
  const int64_t spent = budget->writeBudgetBytes();
  require(!budget->admits(false) && spent < 0 && tier.canDemote(WriteClass::Reclaim),
          "the write budget was not spent, or its being spent refused a demotion");
  auto second = tier.demote(1, tier.acquireSlot(cached), WriteClass::Reclaim, {});
  require(second != nullptr, "a demotion was refused while the write budget was spent");
  runUntilReady(tier, *second);
  require(second->finish() &&
              budget->writeBudgetBytes() == spent - static_cast<int64_t>(slotBytes),
          "a demotion past the write budget failed or was not counted");
  std::cout << "write budget tests passed\n";
}

// A write that fails for another reason than a full volume closes the file,
// and the tier refuses every demotion from then on without throwing. A
// file-size limit at the file's length fails the next slot's write with
// EFBIG. The tier refuses here in canDemote(): its refusal of a null write,
// for a file that closes between canDemote() and the write, has no test of
// its own, only slot-file's null write from a closed file.
void closedFileRefusesDemotion(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                               const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  const uint64_t slotBytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  auto file = std::make_shared<SlotFile>(slotBytes, std::make_shared<DiskBudget>(4 * slotBytes));
  KvPageTier tier(pages, kv::SlotFormat(layout), file, readers);
  auto first = tier.acquireSlot(cached);
  auto stored = tier.demote(0, first, WriteClass::Reclaim, {});
  require(stored != nullptr, "the first demotion was refused");
  runUntilReady(tier, *stored);
  require(stored->finish(), "the first demotion failed");

  // The file is one slot long now, and that slot stays taken. The limit
  // holds only until the failing write has landed, so nothing else of the
  // process meets it.
  rlimit original{};
  require(getrlimit(RLIMIT_FSIZE, &original) == 0, "file limit unavailable");
  rlimit limited = original;
  limited.rlim_cur = slotBytes;
  require(setrlimit(RLIMIT_FSIZE, &limited) == 0, "file limit could not be set");
  auto failing = tier.demote(1, tier.acquireSlot(cached), WriteClass::Reclaim, {});
  require(failing != nullptr, "a demotion into a new slot was refused");
  runUntilReady(tier, *failing);
  require(setrlimit(RLIMIT_FSIZE, &original) == 0, "file limit restore failed");
  require(!failing->finish(), "a write past the file limit succeeded");
  require(!tier.writable() && !tier.canDemote(WriteClass::Reclaim) &&
              tier.demote(2, tier.acquireSlot(cached), WriteClass::Reclaim, {}) == nullptr,
          "a closed file took a demotion");
  std::cout << "closed file tests passed\n";
}

// Kernels, the worker and the readers share an extent. Kernels write a page
// through its address, and the worker writes those bytes to disk. Then one
// command keeps writing and checking the pages around it while the readers
// read the page back and the worker writes another one out; the read lands
// while the command runs. The command sees its own data, the restored page
// holds the disk's bytes, and the next command reads them through the
// page's address.
void besideACommand(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                    const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{10, 2, 256, format}; // Qwen3.6-35B
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, extent, extent);
  require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  const uint64_t slotBytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  auto file = std::make_shared<SlotFile>(slotBytes, std::make_shared<DiskBudget>(2 * slotBytes));
  KvPageTier tier(pages, kv::SlotFormat(layout), file, readers);
  constexpr uint32_t restored = 16, demoted = 25, neighbours = 8;
  const uint64_t data = layout.dataBytesPerLayerPage();
  const uint32_t words = static_cast<uint32_t>(data / sizeof(uint32_t));

  // The GPU addresses of a page's key and value slabs in every layer. Page
  // 0 starts its extent, so its entry is the extent's address.
  std::byte *const base = pages.spans(0).front().data();
  const uint64_t address = test::entryOf(
      pages, 0, test::sharedBuffer(backend, sizeof(SplashKvPage)));
  const auto slabs = [&](uint32_t page, std::vector<uint64_t> &table) {
    for (const auto span : pages.spans(page)) {
      if (span.size() == data) table.push_back(address + static_cast<uint64_t>(span.data() - base));
    }
  };
  std::vector<uint64_t> own, around;
  slabs(restored, own);
  for (uint32_t page = restored - neighbours; page <= restored + neighbours; ++page) {
    if (page != restored) slabs(page, around);
  }
  const auto tableOf = [&](const std::vector<uint64_t> &entries) {
    metal::MetalBuffer table = test::sharedBuffer(backend, entries.size() * sizeof(uint64_t));
    std::memcpy(table.contents(), entries.data(), entries.size() * sizeof(uint64_t));
    return table;
  };
  const metal::MetalBuffer ownTable = tableOf(own), aroundTable = tableOf(around);
  metal::MetalBuffer mismatches = test::sharedBuffer(backend, sizeof(uint32_t));
  auto *mismatched = static_cast<uint32_t *>(mismatches.contents());
  // Writes seed ^ (131071 * entry + word) to every word of every slab of a
  // table, or counts the words that hold something else.
  const auto write = [&](const metal::MetalBuffer &table, size_t entries, const uint32_t &seed) {
    return metal::ComputeDispatch{"addressed_write_u32", {{0, table}},
        {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
        {words / 256, entries, 1}, {256, 1, 1}};
  };
  const auto check = [&](const metal::MetalBuffer &table, size_t entries, const uint32_t &seed) {
    return metal::ComputeDispatch{"addressed_check_u32", {{0, table}, {3, mismatches}},
        {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
        {words / 256, entries, 1}, {256, 1, 1}};
  };

  // Kernels write the page's slabs, the host its scales; the worker writes
  // what both left in the extent.
  const uint32_t ownSeed = 0x9e3779b9u;
  static_cast<void>(fill(pages, restored, 0x5ca1ab1eu));
  static_cast<void>(backend.submit(write(ownTable, own.size(), ownSeed)));
  const auto expected = contents(pages, restored);
  const auto leaving = fill(pages, demoted, 0xc0ffee11u);
  auto slot = tier.acquireSlot(cached);
  auto demotion = tier.demote(restored, slot, WriteClass::Reclaim, {});
  require(demotion != nullptr, "demotion was refused");
  runUntilReady(tier, *demotion);
  require(demotion->finish() && holdsPage(readSlot(*file, slot), expected),
          "the disk does not hold what kernels wrote to the page");
  static_cast<void>(fill(pages, restored, 0xdeadbeefu));

  // The command grows until it outlasts the read.
  uint32_t rounds = 16;
  for (bool overlapped = false; !overlapped;) {
    rounds *= 4;
    require(rounds <= 4096, "no restore landed while a command was running");
    std::vector<uint32_t> seeds(rounds);
    std::vector<metal::ComputeDispatch> command;
    for (uint32_t round = 0; round < rounds; ++round) {
      seeds[round] = 0x85ebca6bu * (round + 1);
      command.push_back(write(aroundTable, around.size(), seeds[round]));
      command.push_back(check(aroundTable, around.size(), seeds[round]));
    }
    *mismatched = 0;
    static_cast<void>(fill(pages, restored, 0xdeadbeefu + rounds));
    auto other = tier.acquireSlot(cached);
    auto ticket = backend.submitCommandAsync(command);
    auto restore = tier.restore(slot, restored, {});
    auto leave = tier.demote(demoted, other, WriteClass::Reclaim, {});
    require(restore && leave, "transfers beside a command were refused");
    runUntilReady(tier, *restore);
    overlapped = !ticket.ready();
    runUntilReady(tier, *leave);
    static_cast<void>(ticket.wait());
    require(restore->finish() && leave->finish(), "transfers beside a command failed");
    require(*mismatched == 0, "a command read other data than it wrote beside a transfer");
    require(contents(pages, restored) == expected,
            "a page restored beside a command does not hold the disk's bytes");
    require(holdsPage(readSlot(*file, other), leaving) && contents(pages, demoted) == leaving,
            "a page demoted beside a command reached the disk changed");
    for (size_t entry = 0; entry < around.size(); ++entry) {
      const auto *slab = reinterpret_cast<const uint32_t *>(base + (around[entry] - address));
      for (uint32_t word = 0; word < words; ++word) {
        require(slab[word] == (seeds.back() ^ (static_cast<uint32_t>(entry) * 131071u + word)),
                "a transfer changed a page the command was writing");
      }
    }
  }
  // The next command reads the restored page through its address.
  *mismatched = 0;
  static_cast<void>(backend.submit(check(ownTable, own.size(), ownSeed)));
  require(*mismatched == 0, "kernels read other data than the restore left in the page");
  std::cout << "transfers beside a command " << kv::formatName(format) << ": the restore landed within "
            << rounds << " rounds of writes and checks\n";
}

// A tier goes away with transfers still in flight. They read and write their
// pages in place, so the worker and the readers have to be done with them
// first.
void teardown(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
              const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{16, 4, 256, format}; // Qwen3.8-27B
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  const uint64_t slotBytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  auto budget = std::make_shared<DiskBudget>(16 * slotBytes);
  auto file = std::make_shared<SlotFile>(slotBytes, budget);
  {
    KvPageTier tier(pages, kv::SlotFormat(layout), file, readers, 16);
    std::vector<std::unique_ptr<KvTransfer>> transfers;
    std::vector<std::shared_ptr<engine::KvDiskSlot>> slots;
    for (uint32_t page = 0; page < 8; ++page) {
      slots.push_back(tier.acquireSlot(cached));
      require(slots.back() != nullptr, "the quota did not hold a demotion");
      transfers.push_back(tier.demote(page, slots.back(), WriteClass::Reclaim, {}));
      require(transfers.back() != nullptr, "a demotion within the share was refused");
    }
    for (uint32_t page = 8; page < 16; ++page)
      transfers.push_back(tier.restore(slots[page - 8], page, {}));
    tier.submitRestores();
  }
  require(budget->usedBytes() == 0, "the tier left its pages with disk IO still running");
  std::cout << "teardown tests passed\n";
}

// A persistent tier keeps a demoted page's slot and its label for the next
// process, whose tier takes the slot back: the page restores byte for byte,
// and a page whose slot changed after its record fails its restore. Of a
// spillable layout's slot, a block's record reads back alone, unless the
// change is in it.
void persistentRoundTrip(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                         const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, extent, extent);
  require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  const kv::SlotFormat format(layout);
  const uint64_t slotBytes = SlotFile::slotBytesFor(format.payloadBytes());
  const SlotFile::Options options{.segments = format.segments()};
  std::string name = (std::filesystem::temp_directory_path() / "splash-kv-tier-XXXXXX").string();
  require(::mkdtemp(name.data()) != nullptr, "temporary directory could not be made");
  const std::filesystem::path directory = name;
  const SlotFile::Persistence persistence{directory / "kv.slots", directory / "kv.records", {}, 8};
  const auto budget = std::make_shared<DiskBudget>(4 * slotBytes);
  std::vector<std::byte> first, second;
  {
    auto file = std::make_shared<SlotFile>(slotBytes, budget, persistence, options);
    file->finishAdoption();
    KvPageTier tier(pages, format, file, readers);
    require(tier.persistent(), "a tier over a persistent file is not persistent");
    first = fill(pages, 0, 0x1357u);
    second = fill(pages, 1, 0x2468u);
    std::vector<std::shared_ptr<engine::KvDiskSlot>> kept;
    for (uint32_t page = 0; page < 2; ++page) {
      kept.push_back(tier.acquireSlot(cached));
      auto demotion = tier.demote(page, kept.back(), WriteClass::Spill, {});
      require(demotion != nullptr, "a demotion was refused");
      tier.label(kept.back(), std::vector<std::byte>(8, std::byte{static_cast<uint8_t>(page + 1)}));
      runUntilReady(tier, *demotion);
      require(demotion->finish(), "a demotion failed");
    }
    require(file->synchronize({})->wait(), "the tier's file did not synchronize");
    file->seal();
  }
  // The second page's slot changes after its record was written: in its
  // last block's record for a spillable layout.
  const uint32_t lastLayer = layout.attentionLayers - 1, lastBlock = kv::SlotFormat::kBlocksPerPage - 1;
  const uint64_t corrupted = format.version() ? format.recordRange(lastLayer, lastBlock).offset : 5;
  {
    std::fstream slots(directory / "kv.slots", std::ios::in | std::ios::out | std::ios::binary);
    slots.seekp(static_cast<std::streamoff>(slotBytes + corrupted));
    slots.put('\x7f');
  }
  static_cast<void>(fill(pages, 0, 0xdeadu));
  static_cast<void>(fill(pages, 1, 0xbeefu));
  auto file = std::make_shared<SlotFile>(slotBytes, budget, persistence, options);
  KvPageTier tier(pages, format, file, readers);
  const std::vector<model::SlotRecord> records = file->records();
  require(records.size() == 2 && records[0].label == std::vector<std::byte>(8, std::byte{1}),
          "the next process did not find both labelled slots");
  auto intact = tier.adopt(records[0]);
  auto changed = tier.adopt(records[1]);
  file->finishAdoption();
  auto restoreIntact = tier.restore(intact, 0, {});
  auto restoreChanged = tier.restore(changed, 1, {});
  runUntilReady(tier, *restoreIntact);
  runUntilReady(tier, *restoreChanged);
  require(restoreIntact->finish() && contents(pages, 0) == first,
          "an adopted slot did not restore its page byte for byte");
  require(!restoreChanged->finish(), "a slot that changed after its record was restored");
  if (format.version()) {
    const kv::SlotRange whole = format.recordRange(0, 0);
    const kv::SlotRange broken = format.recordRange(lastLayer, lastBlock);
    require(model::readKvCopy(*tier.copy(changed), whole).has_value() &&
                !model::readKvCopy(*tier.copy(changed), broken).has_value(),
            "a record of an adopted slot was read against another's checksum");
  }
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
}

// A persistent cache's namespace names how its slots hold a page
// (KvPageTier::slotFormatToken): a dense layout keeps the namespace every
// cache before block records was written under, and a spillable layout's
// changed with its slots, so neither build adopts the other's.
void namespaceNamesTheSlotFormat() {
  const model::ModelDescriptor model = model::makeModelDescriptor(
      "namespace-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(), test::qwen3_8Vision(),
      model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  const auto name = [&](kv::Layout layout) {
    return engine::persistentCacheNamespace(
        engine::makeRuntimeCacheIdentity(std::string(64, 'a'), std::string(64, 'b'), "build", layout, {}),
        model.stateLayout);
  };
  const kv::Layout dense{16, 4, 256, kv::Format::Int8};
  const kv::Layout flashNext{12, 2, 256, kv::Format::Int8, 128, true};
  const kv::Layout flashNextBf16{12, 2, 256, kv::Format::BFloat16, 128, true};
  require(KvPageTier::slotFormatToken(dense).empty() &&
              KvPageTier::slotFormatToken(flashNext) == " slots block-records-v1",
          "the slot format's token changed");
  // What the stack's builds named these caches.
  require(name(dense) == "6365d687c8be103a3bbcc6d2a83b1543",
          "a dense layout's cache namespace changed: its caches would be emptied");
  require(name(flashNext) != "06d3cbb9e548c52d3d582ff9e5d16453" &&
              name(flashNextBf16) != "80a5feae1f55cd1b06749217aedfdae3",
          "a spillable layout's namespace did not change with its slots' format");
}

void run(const std::string &metallib) {
  metal::MetalBackend backend(metallib);
  engine::MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
      engine::queryHostMemory, 0);
  const auto readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);
  namespaceNamesTheSlotFormat();
  for (const auto format : {kv::Format::Int8, kv::Format::BFloat16}) {
    roundTrip(backend, governor, readers, kv::Layout{2, 2, 256, format});
    roundTrip(backend, governor, readers, kv::Layout{10, 2, 256, format}); // Qwen3.6-35B
    roundTrip(backend, governor, readers, kv::Layout{16, 4, 256, format}); // Qwen3.8-27B
    persistentRoundTrip(backend, governor, readers, kv::Layout{10, 2, 256, format});
    persistentRoundTrip(backend, governor, readers, kv::Layout{16, 4, 256, format});
    // Qwen3.8-Flash-Next's token-major values and QSA index: its slots hold
    // the index, then one record per block of each layer.
    const kv::Layout flashNext{12, 2, 256, format, 128, true};
    roundTrip(backend, governor, readers, flashNext);
    blockRecords(backend, governor, readers, flashNext);
    persistentRoundTrip(backend, governor, readers, flashNext);
    limits(backend, governor, readers, format);
    allocationFailure(backend, governor, readers, format);
    demotionsPassTheWriteBudget(backend, governor, readers, format);
    closedFileRefusesDemotion(backend, governor, readers, format);
    besideACommand(backend, governor, readers, format);
    teardown(backend, governor, readers, format);
  }
  std::cout << "kv page tier tests passed\n";
}

} // namespace

int main(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "usage: kv_page_tier_test METALLIB\n";
    return 2;
  }
  try {
    run(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
