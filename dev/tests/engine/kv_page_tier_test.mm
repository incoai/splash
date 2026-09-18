#include "model/KvPageTier.hpp"
#include "tests/engine/AllocationFailure.hpp"

#include "engine/MemoryGovernor.hpp"
#include "metal/CommandGraph.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace splash;
using model::KvPageTier;
using model::KvTransfer;
using model::SlotFile;

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> pattern(uint64_t bytes, uint64_t payload, uint32_t seed) {
  std::vector<std::byte> result(bytes);
  uint32_t state = seed;
  for (uint64_t index = 0; index < payload; ++index) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    result[index] = static_cast<std::byte>(state);
  }
  return result;
}

std::shared_ptr<SlotFile::Slot> fileSlot(const std::shared_ptr<model::KvDiskSlot> &slot) {
  return static_cast<KvPageTier::DiskSlot &>(*slot).slot;
}

std::vector<std::byte> readSlot(SlotFile &file, const std::shared_ptr<model::KvDiskSlot> &slot) {
  std::vector<std::byte> bytes(file.slotBytes());
  require(file.read(fileSlot(slot), {bytes}, {})->wait(), "reading a disk slot failed");
  return bytes;
}

// Drives the tier the way the engine and runtime do: poll, ride the queued
// copies on a command, report its completion, poll again.
void runUntilReady(KvPageTier &tier, metal::MetalBackend &backend, KvTransfer &transfer) {
  for (int spin = 0; spin < 100000; ++spin) {
    tier.poll();
    if (transfer.ready()) return;
    if (tier.copiesQueued()) {
      metal::CommandGraph graph;
      const uint64_t batch = tier.encode(graph);
      require(batch != 0 && !graph.empty(), "queued copies were not encoded");
      static_cast<void>(backend.submitCommand(graph.dispatches()));
      tier.commandCompleted(batch);
    } else {
      std::this_thread::yield();
    }
  }
  throw std::runtime_error("KV transfer did not complete");
}

void allocationFailure(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
              kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                          std::max(16u, layout.backingExtentPages()));
  const uint64_t bytes = KvPageTier::slotBytesFor(pages);
  for (bool restoring : {false, true}) {
    bool completed = false;
    for (int failure = 0; failure < 64; ++failure) {
      auto file = std::make_shared<SlotFile>(bytes, bytes);
      KvPageTier tier(backend, pages, file, 1);
      auto slot = tier.acquireSlot();
      const auto payload = pattern(bytes, pages.bytesPerPage(), 123);
      require(file->write(fileSlot(slot), {payload}, {})->wait(), "fault slot seed failed");
      std::unique_ptr<KvTransfer> transfer;
      allocationFailureAfter = failure;
      try {
        transfer = restoring ? tier.restore(slot, 0, {}) : tier.demote(0, slot, {});
        allocationFailureAfter = -1;
      } catch (const std::bad_alloc &) {
        allocationFailureAfter = -1;
        require(tier.canDemote() && !tier.copiesQueued() && file->idle(),
                "failed transfer submission retained staging or untracked IO");
        continue;
      }
      require(transfer != nullptr, "transfer refused after allocation failures");
      runUntilReady(tier, backend, *transfer);
      require(transfer->finish() && tier.canDemote(), "transfer did not recover");
      completed = true;
      break;
    }
    require(completed, "transfer allocation sweep never reached success");
  }
}

// Two pages in different extents go disk -> page -> disk through a single
// staging slot, so every copy lands on staging that still holds the other
// page: a range the kernel skipped would surface as the wrong pattern.
void roundTrip(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
               kv::Layout layout) {
  const uint32_t extent = layout.backingExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, 2 * extent);
  const uint32_t pageA = extent - 1;
  const uint32_t pageB = extent;
  require(pages.isResident(pageA) && pages.ensureResident(pageB) && pages.isResident(pageB),
          "test pages were not mapped");
  const uint64_t slotBytes = KvPageTier::slotBytesFor(pages);
  const uint64_t payload = pages.bytesPerPage();
  require(slotBytes >= payload && slotBytes < payload + SlotFile::kAlignmentBytes &&
              slotBytes % SlotFile::kAlignmentBytes == 0,
          "KV slot size is not the page rounded up for uncached IO");
  auto file = std::make_shared<SlotFile>(slotBytes, 4 * slotBytes);
  KvPageTier tier(backend, pages, file, 1);
  require(tier.slotBytes() == slotBytes, "tier reports another slot size");

  const auto first = pattern(slotBytes, payload, 0x1234567u);
  const auto second = pattern(slotBytes, payload, 0x89abcdefu);
  auto slotA = tier.acquireSlot();
  auto slotB = tier.acquireSlot();
  require(slotA && slotB, "disk slots were not granted");
  require(file->write(fileSlot(slotA), {first}, {})->wait() &&
              file->write(fileSlot(slotB), {second}, {})->wait(),
          "seeding the disk slots failed");

  auto restoreA = tier.restore(slotA, pageA, {});
  require(restoreA != nullptr, "restore was refused with free staging");
  require(!tier.restore(slotB, pageB, {}), "a second transfer found staging that does not exist");
  runUntilReady(tier, backend, *restoreA);
  require(restoreA->finish(), "restore did not succeed");
  auto restoreB = tier.restore(slotB, pageB, {});
  require(restoreB != nullptr, "staging was not released after the restore");
  runUntilReady(tier, backend, *restoreB);
  require(restoreB->finish(), "second restore did not succeed");

  auto slotC = tier.acquireSlot();
  auto slotD = tier.acquireSlot();
  require(slotC && slotD && !tier.acquireSlot(), "disk quota was not enforced");
  auto demoteA = tier.demote(pageA, slotC, {});
  require(demoteA != nullptr, "demotion was refused with free staging");
  {
    metal::CommandGraph graph;
    const uint64_t batch = tier.encode(graph);
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    tier.commandCompleted(batch);
    tier.poll();
    require(!demoteA->ready() || demoteA->finish(),
            "demotion failed after its copy ran");
  }
  runUntilReady(tier, backend, *demoteA);
  require(demoteA->finish(), "demotion did not succeed");
  auto demoteB = tier.demote(pageB, slotD, {});
  require(demoteB != nullptr, "staging was not released after the write");
  runUntilReady(tier, backend, *demoteB);
  require(demoteB->finish(), "second demotion did not succeed");
  require(std::memcmp(readSlot(*file, slotC).data(), first.data(), payload) == 0 &&
              std::memcmp(readSlot(*file, slotD).data(), second.data(), payload) == 0,
          "pages did not round-trip through staging byte for byte");

  // A finished transfer holds neither staging nor its slot.
  slotA.reset();
  require(tier.acquireSlot() != nullptr, "a released disk slot was not reusable");
  std::cout << "round trip " << kv::formatName(layout.format) << " "
            << layout.attentionLayers << "x" << layout.kvHeads << "x"
            << layout.headDimension << ": page_bytes=" << payload
            << " slot_bytes=" << slotBytes << '\n';
}

// Several staging slots, a read that fails, demotions limited to half the
// ring, and a table entry that must not outlive its transfer.
void staging(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
              kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                          std::max(16u, layout.backingExtentPages()));
  const uint64_t slotBytes = KvPageTier::slotBytesFor(pages);
  const uint64_t payload = pages.bytesPerPage();
  auto file = std::make_shared<SlotFile>(slotBytes, 3 * slotBytes);
  KvPageTier tier(backend, pages, file, 2);
  const auto bytes = pattern(slotBytes, payload, 0x5eed5eedu);
  auto written = tier.acquireSlot();
  require(file->write(fileSlot(written), {bytes}, {})->wait(), "seeding failed");

  // An unwritten slot cannot be restored; the failure frees its staging.
  auto empty = tier.acquireSlot();
  auto failed = tier.restore(empty, 5, {});
  auto good = tier.restore(written, 6, {});
  require(failed && good, "two restores did not fit two staging slots");
  require(!tier.canDemote() && !tier.restore(written, 8, {}),
          "a full staging ring admitted another transfer");
  runUntilReady(tier, backend, *failed);
  require(!failed->finish(), "restoring an unwritten slot succeeded");
  runUntilReady(tier, backend, *good);
  require(good->finish(), "restore beside a failed one did not succeed");

  // Both staging slots are free again. Demotions may hold half the ring, so
  // the second waits for the first; a demotion of page 6 reproduces the
  // pattern, and the copy table is idle afterwards so a later command with
  // nothing queued moves nothing.
  require(tier.canDemote(), "drained restores did not restore demotion admission");
  auto target = tier.acquireSlot();
  auto demote = tier.demote(6, target, {});
  require(demote != nullptr, "staging was not released by finished transfers");
  require(!tier.canDemote() && !tier.demote(6, empty, {}),
          "demotions took the restores' half of the ring");
  runUntilReady(tier, backend, *demote);
  require(tier.canDemote(), "finished demotion did not restore admission");
  auto second = tier.demote(6, empty, {});
  require(second != nullptr, "a finished demotion did not free its staging");
  runUntilReady(tier, backend, *second);
  require(demote->finish() && second->finish() && tier.writable(), "demotions did not succeed");
  require(std::memcmp(readSlot(*file, target).data(), bytes.data(), payload) == 0 &&
              std::memcmp(readSlot(*file, empty).data(), bytes.data(), payload) == 0,
          "demoted slots do not hold the page");
  metal::CommandGraph graph;
  require(tier.encode(graph) == 0 && graph.empty() && !tier.copiesQueued(),
          "an idle tier encoded copies");
  std::cout << "staging tests passed\n";
}

// Shares of the ring: demotions hold at most half of it and restores at most
// three quarters, so a burst of restores leaves a demotion its slot and a
// burst of demotions leaves restores theirs.
void shares(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
              kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                          std::max(16u, layout.backingExtentPages()));
  const uint64_t slotBytes = KvPageTier::slotBytesFor(pages);
  auto file = std::make_shared<SlotFile>(slotBytes, 8 * slotBytes);
  KvPageTier tier(backend, pages, file, 4);
  const auto bytes = pattern(slotBytes, pages.bytesPerPage(), 0x51a5e5u);
  std::vector<std::shared_ptr<model::KvDiskSlot>> written;
  for (int index = 0; index < 3; ++index) {
    written.push_back(tier.acquireSlot());
    require(file->write(fileSlot(written.back()), {bytes}, {})->wait(), "seeding failed");
  }
  std::vector<std::unique_ptr<KvTransfer>> restores;
  for (uint32_t index = 0; index < 3; ++index) {
    restores.push_back(tier.restore(written[index], index, {}));
    require(restores.back() != nullptr, "a restore within the share was refused");
  }
  require(!tier.restore(written[0], 3, {}), "restores took more than three quarters of the ring");
  auto target = tier.acquireSlot();
  auto demotion = tier.demote(3, target, {});
  require(demotion != nullptr, "a burst of restores left no slot for a demotion");
  require(!tier.demote(3, tier.acquireSlot(), {}), "a demotion found staging that does not exist");
  for (auto &restore : restores) runUntilReady(tier, backend, *restore);
  runUntilReady(tier, backend, *demotion);
  require(demotion->finish(), "demotion beside restores did not succeed");
  for (auto &restore : restores) require(restore->finish(), "restore did not succeed");
  // With the ring idle, demotions hold half of it and no more.
  auto first = tier.demote(0, tier.acquireSlot(), {});
  auto second = tier.demote(1, tier.acquireSlot(), {});
  require(first && second && !tier.demote(2, tier.acquireSlot(), {}),
          "demotions took more than half the ring");
  runUntilReady(tier, backend, *first);
  runUntilReady(tier, backend, *second);
  require(first->finish() && second->finish(), "demotions did not succeed");
  std::cout << "ring share tests passed\n";
}

// A tier goes away with its writes still on the way out: staging is their
// source and dies with it, so the IO worker has to be done with it first.
void teardown(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
              kv::Format format) {
  const kv::Layout layout{16, 4, 256, format}; // Qwen3.8-27B
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                          std::max(16u, layout.backingExtentPages()));
  const uint64_t slotBytes = KvPageTier::slotBytesFor(pages);
  auto file = std::make_shared<SlotFile>(slotBytes, 16 * slotBytes);
  {
    KvPageTier tier(backend, pages, file, 16);
    std::vector<std::unique_ptr<KvTransfer>> demotions;
    for (uint32_t page = 0; page < 8; ++page) {
      auto slot = tier.acquireSlot();
      require(slot != nullptr, "the quota did not hold a demotion");
      demotions.push_back(tier.demote(page, std::move(slot), {}));
      require(demotions.back() != nullptr, "a demotion within the share was refused");
    }
    metal::CommandGraph graph;
    const uint64_t batch = tier.encode(graph);
    require(batch != 0, "queued demotions were not encoded");
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    tier.commandCompleted(batch);
    // One poll hands every copy to the worker, which is still writing when
    // the tier goes out of scope.
    tier.poll();
  }
  require(file->idle(), "the tier released staging with disk IO still running");
  std::cout << "teardown tests passed\n";
}

void run(const std::string &metallib) {
  metal::MetalBackend backend(metallib);
  engine::MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  for (const auto format : {kv::Format::Int8, kv::Format::BFloat16}) {
    roundTrip(backend, governor, kv::Layout{2, 2, 256, format});
    roundTrip(backend, governor, kv::Layout{10, 2, 256, format}); // Qwen3.6-35B
    roundTrip(backend, governor, kv::Layout{16, 4, 256, format}); // Qwen3.8-27B
    allocationFailure(backend, governor, format);
    staging(backend, governor, format);
    shares(backend, governor, format);
    teardown(backend, governor, format);
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
