#include "engine/KvPageTier.hpp"
#include "tests/engine/AllocationFailure.hpp"
#include "tests/engine/ClaimScope.hpp"
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
constexpr kv::Components kBase = kv::componentBit(kv::Component::Base);

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
  for (const auto span : format.transferSpans(pages, std::array<uint32_t, 1>{page}))
    bytes.insert(bytes.end(), span.begin(), span.end());
  return bytes;
}

std::shared_ptr<SlotFile::Slot> fileSlot(const std::shared_ptr<engine::KvDiskSlot> &slot) {
  return static_cast<KvPageTier::DiskSlot &>(*slot).slots[0]->slot;
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

// Two pages in different extents go to disk and back byte for byte, their neighbours untouched, in
// the layout's slot format.
void roundTrip(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
               const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, 2 * extent,
                        extent);
  const uint32_t pageA = extent - 1;
  const uint32_t pageB = extent;
  {
    // The extents are allocated under a Floor claim; the tier itself allocates nothing.
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(pages.allocateExtent(0) && pages.allocateExtent(1) && pages.isAllocated(pageA) &&
                pages.isAllocated(pageB),
            "test pages were not allocated");
  }
  const kv::SlotFormat format(layout);
  const uint64_t slotBytes = SlotFile::slotBytesFor(format.payloadBytes());
  auto file = slotFile(format, 2);
  KvPageTier tier(pages, format, {file}, readers);
  require(tier.slotBytes(kBase) == slotBytes, "tier reports another slot size");

  const auto first = fill(pages, pageA, 0x1234567u);
  const auto second = fill(pages, pageB, 0x89abcdefu);
  const auto beforeA = fill(pages, pageA - 1, 0x600df00du);
  const auto afterB = fill(pages, pageB + 1, 0x0ddba11u);
  require(contents(pages, pageA) == first && contents(pages, pageB) == second,
          "a page does not hold what the host wrote through its spans");

  const auto slottedA = payload(format, pages, pageA);
  const auto slottedB = payload(format, pages, pageB);
  auto slotA = tier.acquireSlot(kBase, cached, nullptr);
  auto slotB = tier.acquireSlot(kBase, cached, nullptr);
  require(slotA && slotB && !tier.acquireSlot(kBase, cached, nullptr), "disk quota was not enforced");
  auto demoteA = tier.demote(std::array<uint32_t, 1>{pageA}, slotA, kBase, WriteClass::Reclaim, {});
  auto demoteB = tier.demote(std::array<uint32_t, 1>{pageB}, slotB, kBase, WriteClass::Spill, {});
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
  auto restoreB = tier.restore(slotB, kBase, std::array<uint32_t, 1>{pageB}, {});
  auto restoreA = tier.restore(slotA, kBase, std::array<uint32_t, 1>{pageA}, {});
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
  require(tier.canDemote(WriteClass::Reclaim) && tier.acquireSlot(kBase, cached, nullptr) != nullptr,
          "a released disk slot was not reusable");
  std::cout << "round trip " << kv::formatName(layout.format) << " "
            << layout.attentionLayers << "x" << layout.kvHeads << "x"
            << layout.headDimension << " slot format v" << format.version()
            << ": payload_bytes=" << format.payloadBytes() << " slot_bytes=" << slotBytes << '\n';
}

// A spillable slot holds one record per block and layer that one aligned read returns; an Index
// restore reads the index alone.
void blockRecords(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                  const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, 2 * extent, extent);
  {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(pages.allocateExtent(0) && pages.allocateExtent(1, kv::PageForm::Index),
            "the test extents were not allocated");
  }
  const kv::SlotFormat format(layout);
  KvPageTier tier(pages, format, {slotFile(format, 2)}, readers);
  static_cast<void>(fill(pages, 3, 0x7e57u));
  const auto slotted = payload(format, pages, 3);
  auto slot = tier.acquireSlot(kBase, cached, nullptr);
  auto written = tier.demote(std::array<uint32_t, 1>{3}, slot, kBase, WriteClass::Spill, {});
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
  auto restore = tier.restore(slot, kBase, std::array<uint32_t, 1>{index}, {});
  require(restore != nullptr, "the index restore was refused");
  runUntilReady(tier, *restore);
  const auto indexOf = [&](uint32_t page) {
    std::vector<std::byte> bytes;
    for (const auto span : pages.indexSpans(page)) bytes.insert(bytes.end(), span.begin(), span.end());
    return bytes;
  };
  require(restore->finish() && indexOf(index) == indexOf(3), "an Index page did not take its slot's index");
  rejects(
      [&] {
        static_cast<void>(tier.demote(std::array<uint32_t, 1>{index}, tier.acquireSlot(kBase, cached, nullptr), kBase, WriteClass::Spill, {}));
      },
      "only a page that holds its K/V", "an Index page was written to the tier");
  std::cout << "block records " << kv::formatName(layout.format) << ": record_bytes=" << format.recordBytes()
            << " pitch=" << format.recordPitch() << '\n';
}

// Writes in flight are bounded in all and per class (reclaim always keeps a quarter); a failed
// restore fails alone.
void limits(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
            const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  }
  const kv::SlotFormat slots(layout);
  auto file = slotFile(slots, 16);
  rejects([&] { KvPageTier refused(pages, slots, {file}, readers, 0); }, "room for a write",
          "a tier with no room for a write was built");
  KvPageTier tier(pages, slots, {file}, readers, 8);
  const auto bytes = fill(pages, 9, 0x51a5e5u);
  std::vector<std::unique_ptr<KvTransfer>> writes;
  const auto write = [&](WriteClass kind) {
    auto started = tier.demote(std::array<uint32_t, 1>{9}, tier.acquireSlot(kBase, cached, nullptr), kBase, kind, {});
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
  runUntilReady(tier, *writes.front());
  require(writes.front()->finish() && tier.canDemote(WriteClass::Persist),
          "a retired write did not give its room back");
  for (auto &each : writes) runUntilReady(tier, *each);
  // An unwritten copy cannot be restored; a landed one can, twice at once.
  auto written = tier.acquireSlot(kBase, cached, nullptr);
  auto seeding = tier.demote(std::array<uint32_t, 1>{9}, written, kBase, WriteClass::Reclaim, {});
  runUntilReady(tier, *seeding);
  require(seeding->finish(), "seeding failed");
  rejects([&] { static_cast<void>(tier.restore(tier.acquireSlot(kBase, cached, nullptr), kBase, std::array<uint32_t, 1>{1}, {})); },
          "not as the tier's call needs", "an unwritten copy was restored");
  std::vector<std::unique_ptr<KvTransfer>> restores;
  restores.push_back(tier.restore(written, kBase, std::array<uint32_t, 1>{0}, {}));
  restores.push_back(tier.restore(written, kBase, std::array<uint32_t, 1>{2}, {}));
  for (auto &restore : restores) runUntilReady(tier, *restore);
  require(restores[0]->finish() && restores[1]->finish() && contents(pages, 0) == bytes && contents(pages, 2) == bytes,
          "a landed copy did not restore");
  std::cout << "transfer limit tests passed\n";
}

// A transfer that cannot be started leaves nothing behind: no room taken in
// the tier and no IO the tier does not track.
void allocationFailure(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                       const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  }
  const uint64_t bytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  for (bool restoring : {false, true}) {
    bool completed = false;
    for (int failure = 0; failure < 64; ++failure) {
      auto budget = std::make_shared<DiskBudget>(bytes);
      auto file = std::make_shared<SlotFile>(bytes, budget);
      KvPageTier tier(pages, kv::SlotFormat(layout), {file}, readers, 1);
      auto slot = tier.acquireSlot(kBase, cached, nullptr);
      const auto payload = fill(pages, 0, 123);
      if (restoring) {
        auto seeding = tier.demote(std::array<uint32_t, 1>{0}, slot, kBase, WriteClass::Reclaim, {});
        runUntilReady(tier, *seeding);
        require(seeding->finish(), "fault slot seed failed");
      }
      std::unique_ptr<KvTransfer> transfer;
      allocationFailureAfter = failure;
      try {
        if (restoring) {
          transfer = tier.restore(slot, kBase, std::array<uint32_t, 1>{0}, {});
          tier.submitRestores();
        } else {
          transfer = tier.demote(std::array<uint32_t, 1>{0}, slot, kBase, WriteClass::Reclaim, {});
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

// Each demotion's slot counts against the write budget. The cache asks the budget before a write
// (Cache::takesCacheWrites), the tier does not.
void demotionsDrawOnTheWriteBudget(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                                 const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  }
  const uint64_t slotBytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  // The clock stands still: what the budget gives out does not come back.
  auto budget = std::make_shared<DiskBudget>(4 * slotBytes, [] { return 0.0; });
  budget->setWriteBudget(slotBytes);
  auto file = std::make_shared<SlotFile>(slotBytes, budget);
  KvPageTier tier(pages, kv::SlotFormat(layout), {file}, readers);
  auto first = tier.demote(std::array<uint32_t, 1>{0}, tier.acquireSlot(kBase, cached, nullptr), kBase, WriteClass::Reclaim, {});
  require(first != nullptr, "a demotion within the write budget was refused");
  runUntilReady(tier, *first);
  require(first->finish(), "the first demotion failed");
  const int64_t spent = budget->writeBudgetBytes();
  require(!budget->admits() && spent < 0, "a demotion was not counted against the write budget");
  auto second = tier.demote(std::array<uint32_t, 1>{1}, tier.acquireSlot(kBase, cached, nullptr), kBase, WriteClass::Spill, {});
  require(second != nullptr, "the tier refused a Spill request's write");
  runUntilReady(tier, *second);
  require(second->finish() &&
              budget->writeBudgetBytes() == spent - static_cast<int64_t>(slotBytes),
          "a Spill request's write failed or was not counted");
  std::cout << "write budget tests passed\n";
}

// A write failing for a reason other than a full volume closes the file, and the tier then refuses
// demotions without throwing.
void closedFileRefusesDemotion(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                               const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  }
  const uint64_t slotBytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  auto file = std::make_shared<SlotFile>(slotBytes, std::make_shared<DiskBudget>(4 * slotBytes));
  KvPageTier tier(pages, kv::SlotFormat(layout), {file}, readers);
  auto first = tier.acquireSlot(kBase, cached, nullptr);
  auto stored = tier.demote(std::array<uint32_t, 1>{0}, first, kBase, WriteClass::Reclaim, {});
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
  auto failing = tier.demote(std::array<uint32_t, 1>{1}, tier.acquireSlot(kBase, cached, nullptr), kBase, WriteClass::Reclaim, {});
  require(failing != nullptr, "a demotion into a new slot was refused");
  runUntilReady(tier, *failing);
  require(setrlimit(RLIMIT_FSIZE, &original) == 0, "file limit restore failed");
  require(!failing->finish(), "a write past the file limit succeeded");
  require(!tier.writable() && !tier.canDemote(WriteClass::Reclaim) &&
              tier.demote(std::array<uint32_t, 1>{2}, tier.acquireSlot(kBase, cached, nullptr), kBase, WriteClass::Reclaim, {}) == nullptr,
          "a closed file took a demotion");
  std::cout << "closed file tests passed\n";
}

// The worker and readers move a page while a command writes the pages around it in the same extent;
// each side sees its own bytes.
void besideACommand(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                    const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{10, 2, 256, format}; // Qwen3.6-35B
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, extent, extent);
  {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  }
  const uint64_t slotBytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  auto file = std::make_shared<SlotFile>(slotBytes, std::make_shared<DiskBudget>(2 * slotBytes));
  KvPageTier tier(pages, kv::SlotFormat(layout), {file}, readers);
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
  // A table of slab addresses names the extent they are in as a page table
  // of the extent's page does (kv::PageStorage::writeEntries).
  const auto tableOf = [&](const std::vector<uint64_t> &entries) {
    metal::MetalBuffer table = test::sharedBuffer(backend, entries.size() * sizeof(uint64_t));
    pages.writeEntries(std::array<uint32_t, 1>{0}, 0, table);
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
  auto slot = tier.acquireSlot(kBase, cached, nullptr);
  auto demotion = tier.demote(std::array<uint32_t, 1>{restored}, slot, kBase, WriteClass::Reclaim, {});
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
    auto other = tier.acquireSlot(kBase, cached, nullptr);
    auto ticket = backend.submitCommandAsync(command);
    auto restore = tier.restore(slot, kBase, std::array<uint32_t, 1>{restored}, {});
    auto leave = tier.demote(std::array<uint32_t, 1>{demoted}, other, kBase, WriteClass::Reclaim, {});
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

// A tier goes away with transfers in flight only once the worker and readers are done with its
// pages. A restore reads a copy that has landed (an unwritten one is refused, limits), so the
// restores in flight read copies written before.
void teardown(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
              const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{16, 4, 256, format}; // Qwen3.8-27B
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  }
  const uint64_t slotBytes = SlotFile::slotBytesFor(pages.bytesPerPage());
  auto budget = std::make_shared<DiskBudget>(16 * slotBytes);
  auto file = std::make_shared<SlotFile>(slotBytes, budget);
  {
    KvPageTier tier(pages, kv::SlotFormat(layout), {file}, readers, 16);
    std::vector<std::unique_ptr<KvTransfer>> transfers;
    std::vector<std::shared_ptr<engine::KvDiskSlot>> landed, slots;
    for (uint32_t page = 8; page < 16; ++page) {
      landed.push_back(tier.acquireSlot(kBase, cached, nullptr));
      auto seeding = tier.demote(std::array<uint32_t, 1>{page}, landed.back(), kBase, WriteClass::Reclaim, {});
      require(seeding != nullptr, "a seeding demotion was refused");
      runUntilReady(tier, *seeding);
      require(seeding->finish(), "a seeding demotion failed");
    }
    for (uint32_t page = 0; page < 8; ++page) {
      slots.push_back(tier.acquireSlot(kBase, cached, nullptr));
      require(slots.back() != nullptr, "the quota did not hold a demotion");
      transfers.push_back(tier.demote(std::array<uint32_t, 1>{page}, slots.back(), kBase, WriteClass::Reclaim, {}));
      require(transfers.back() != nullptr, "a demotion within the share was refused");
    }
    for (uint32_t page = 8; page < 16; ++page)
      transfers.push_back(tier.restore(landed[page - 8], kBase, std::array<uint32_t, 1>{page}, {}));
    tier.submitRestores();
  }
  require(budget->usedBytes() == 0, "the tier left its pages with disk IO still running");
  std::cout << "teardown tests passed\n";
}

// A window part's read holds the unit pages it fills but not its block's base page
// (engine::Cache::startUnitRead), whose extent a reclaim may release while the read is in flight. The
// read then fails, or lands what it read if it found the base page's spans first; either way it writes
// nothing but its unit page. A release before the read runs fails it; releases from the moment it
// starts to past its landing race the reader's look-up of the base page's extent, which the storage's
// lock orders (ThreadSanitizer, test-sanitizers-metal).
void windowReadBesideARelease(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                              const std::shared_ptr<model::SsdReaders> &readers, kv::Format format) {
  const kv::Layout layout{6, 2, 256, format, 0, false, 512, 0b010110};
  require(layout.blockPages() == 2 && layout.pageSlots() == 3,
          "the test's block is not a base page and one full window page");
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, 2 * extent, extent);
  const auto allocate = [&](uint32_t index) {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(static_cast<bool>(pages.allocateExtent(index)), "a test extent was not allocated");
  };
  allocate(0);
  allocate(1);
  // The base page in the first extent; the window page in the second, between two neighbours.
  const std::array<uint32_t, 2> block{0, extent + 1};
  const kv::SlotFormat slotFormat(layout);
  constexpr kv::Components window = kv::componentBit(kv::Component::Window);
  uint64_t copyBytes = 0;
  for (const kv::Component part : kv::kAllComponents)
    copyBytes += SlotFile::slotBytesFor(slotFormat.payloadBytes(part));
  const auto budget = std::make_shared<DiskBudget>(copyBytes);
  std::vector<std::shared_ptr<SlotFile>> files;
  for (const kv::Component part : kv::kAllComponents)
    files.push_back(std::make_shared<SlotFile>(SlotFile::slotBytesFor(slotFormat.payloadBytes(part)), budget,
                                               SlotFile::Options{.segments = slotFormat.segments(part)}));
  KvPageTier tier(pages, slotFormat, files, readers);
  static_cast<void>(fill(pages, block[0], 0x2545f491u));
  const auto written = fill(pages, block[1], 0x9e3779b9u);
  const auto copy = tier.acquireSlot(kBase | window, cached, nullptr);
  require(copy != nullptr, "the quota did not hold the block's copy");
  auto write = tier.demote(block, copy, kBase | window, WriteClass::Reclaim, {});
  require(write != nullptr, "the block's write was refused");
  runUntilReady(tier, *write);
  require(write->finish(), "the block's write failed");

  constexpr uint32_t rounds = 16;
  uint32_t failed = 0;
  for (uint32_t round = 0; round < rounds; ++round) {
    if (!pages.isAllocated(block[0])) allocate(0);
    const auto stale = fill(pages, block[1], 0x5bd1e995u + round);
    const auto before = fill(pages, extent, 0x68e31da4u + round);
    const auto after = fill(pages, extent + 2, 0x1b873593u + round);
    auto read = tier.restore(copy, window, block, {});
    require(read != nullptr, "the window read was refused");
    // The first round releases the base page's extent before the read runs; the others after it
    // starts, from at once to 65 ms later.
    if (round) tier.submitRestores();
    if (round > 1) std::this_thread::sleep_for(std::chrono::microseconds(4 << (round - 2)));
    pages.releaseExtent(0);
    runUntilReady(tier, *read);
    const bool landed = read->finish();
    failed += !landed;
    require(round || !landed, "a window read whose base page's extent went first landed");
    require(contents(pages, block[1]) == (landed ? written : stale),
            "a window read left its unit page holding neither its part nor what it held");
    require(contents(pages, extent) == before && contents(pages, extent + 2) == after,
            "a window read wrote a page beside its unit page");
  }
  std::cout << "window reads beside a release " << kv::formatName(format) << ": " << failed << " of " << rounds
            << " failed\n";
}

// A persistent tier's slot comes back in the next process byte for byte; a slot changed after its
// record fails its restore.
void persistentRoundTrip(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                         const std::shared_ptr<model::SsdReaders> &readers, kv::Layout layout) {
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, extent, extent);
  {
    const test::ClaimScope start(governor, engine::ClaimClass::Floor);
    require(static_cast<bool>(pages.allocateExtent(0)), "the test extent was not allocated");
  }
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
    KvPageTier tier(pages, format, {file}, readers);
    require(tier.persistent(), "a tier over a persistent file is not persistent");
    first = fill(pages, 0, 0x1357u);
    second = fill(pages, 1, 0x2468u);
    std::vector<std::shared_ptr<engine::KvDiskSlot>> kept;
    for (uint32_t page = 0; page < 2; ++page) {
      kept.push_back(tier.acquireSlot(kBase, cached, nullptr));
      auto demotion = tier.demote(std::array<uint32_t, 1>{page}, kept.back(), kBase, WriteClass::Spill, {});
      require(demotion != nullptr, "a demotion was refused");
      tier.label(kept.back(), kBase, std::vector<std::byte>(8, std::byte{static_cast<uint8_t>(page + 1)}));
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
  KvPageTier tier(pages, format, {file}, readers);
  const std::vector<model::SlotRecord> records = file->records();
  require(records.size() == 2 && records[0].label == std::vector<std::byte>(8, std::byte{1}),
          "the next process did not find both labelled slots");
  auto intact = tier.adopt(records[0], kv::Component::Base, nullptr);
  auto changed = tier.adopt(records[1], kv::Component::Base, nullptr);
  file->finishAdoption();
  auto restoreIntact = tier.restore(intact, kBase, std::array<uint32_t, 1>{0}, {});
  auto restoreChanged = tier.restore(changed, kBase, std::array<uint32_t, 1>{1}, {});
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

// A persistent cache's namespace names its slot format, so dense, spillable and windowed builds never
// adopt each other's slots.
void namespaceNamesTheSlotFormat() {
  const model::ModelDescriptor model = model::makeModelDescriptor(
      "namespace-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(), test::qwen3_8Vision(),
      model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  const auto name = [&](kv::Layout layout, model::CompositeStateLayout states) {
    return engine::persistentCacheNamespace(
        engine::makeRuntimeCacheIdentity(std::string(64, 'a'), std::string(64, 'b'), "build", layout, {}), states);
  };
  const kv::Layout dense{16, 4, 256, kv::Format::Int8};
  const kv::Layout flashNext{12, 2, 256, kv::Format::Int8, 128, true, 0, 0, kv::Reads::SelectedBlocks};
  const kv::Layout flashNextBf16{12, 2, 256, kv::Format::BFloat16, 128, true, 0, 0, kv::Reads::SelectedBlocks};
  const kv::Layout windowed{6, 4, 128, kv::Format::Int8, 0, false, 512, 0b011010};
  require(KvPageTier::slotFormatToken(dense).empty() &&
              KvPageTier::slotFormatToken(flashNext) == " slots block-records-v1" &&
              KvPageTier::slotFormatToken(windowed) == " slots base-window-parts-v1",
          "the slot format's token changed");
  // What the stack's builds named these caches since cached GDN states took
  // int8 codes (0d770e276), which renamed every GDN model's caches once.
  const std::string denseName = name(dense, model.stateLayout);
  require(denseName == "88a6a28cadb13dc02b99998aa48baa72",
          "a dense layout's cache namespace changed to " + denseName + ": its caches would be emptied");
  require(name(flashNext, model.stateLayout) != "06d3cbb9e548c52d3d582ff9e5d16453" &&
              name(flashNextBf16, model.stateLayout) != "80a5feae1f55cd1b06749217aedfdae3",
          "a spillable layout's namespace did not change with its slots' format");
  // A cached state in the running format (--cached-states exact) names its
  // caches apart from the codes' and by its format, so neither mode reads the
  // other's states; the codes' name is the same whatever the running format.
  model::CompositeStateLayout exact = model.stateLayout, exactBf16 = model.stateLayout, codesBf16 = model.stateLayout;
  exact.target.cachedCodes = exactBf16.target.cachedCodes = false;
  exactBf16.target.format = codesBf16.target.format = ops::GdnStateFormat::BFloat16;
  require(name(dense, exact) != denseName && name(dense, exactBf16) != denseName &&
              name(dense, exact) != name(dense, exactBf16) && name(dense, codesBf16) == denseName,
          "a cached state's form or format did not name its caches apart");
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
    const kv::Layout flashNext{12, 2, 256, format, 128, true, 0, 0, kv::Reads::SelectedBlocks};
    roundTrip(backend, governor, readers, flashNext);
    blockRecords(backend, governor, readers, flashNext);
    persistentRoundTrip(backend, governor, readers, flashNext);
    limits(backend, governor, readers, format);
    allocationFailure(backend, governor, readers, format);
    demotionsDrawOnTheWriteBudget(backend, governor, readers, format);
    closedFileRefusesDemotion(backend, governor, readers, format);
    besideACommand(backend, governor, readers, format);
    teardown(backend, governor, readers, format);
    windowReadBesideARelease(backend, governor, readers, format);
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
