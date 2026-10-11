#include "Checked.hpp"
#include "engine/MemoryGovernor.hpp"
#include "tests/engine/AllocationFailure.hpp"
#include "model/QwenState.hpp"
#include "tests/engine/TestChecks.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <chrono>
#include <new>
#include <iostream>
#include <limits>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr model::GdnStateLayout kTargetState{48, 3, 10'240, 48, 128, 128};
constexpr model::DraftStateLayout kDraftState{5, 8, 128, 5120};
constexpr model::CompositeStateLayout kStateLayout{kTargetState, kDraftState};
constexpr uint64_t kStateSlotBytes = model::SlotFile::slotBytesFor(kStateLayout.cachedBytes());

using splash::test::rejects;
using splash::test::require;

uint32_t &word(const metal::MetalBuffer &buffer, uint64_t byteOffset = 0) {
  require(byteOffset + sizeof(uint32_t) <= buffer.sizeBytes(),
          "test marker is outside buffer");
  auto *bytes = static_cast<uint8_t *>(buffer.contents());
  require(bytes != nullptr, "test buffer is not CPU-visible");
  return *reinterpret_cast<uint32_t *>(bytes + byteOffset);
}

void fill(const metal::MetalBuffer &buffer, uint64_t seed) {
  auto *bytes = static_cast<uint8_t *>(buffer.contents());
  require(bytes != nullptr, "test buffer is not CPU-visible");
  uint64_t x = seed | 1;
  for (uint64_t i = 0; i < buffer.sizeBytes(); ++i) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    bytes[i] = static_cast<uint8_t>(x);
  }
}

std::vector<uint8_t> bytesOf(const metal::MetalBuffer &buffer) {
  auto *bytes = static_cast<const uint8_t *>(buffer.contents());
  return std::vector<uint8_t>(bytes, bytes + buffer.sizeBytes());
}

bool sameBytes(const metal::MetalBuffer &buffer, const std::vector<uint8_t> &image) {
  return image.size() == buffer.sizeBytes() &&
         std::memcmp(buffer.contents(), image.data(), image.size()) == 0;
}

// Every byte of a lane's current state, in the order its disk copy holds
// them: its current GDN cell and its context window.
std::vector<std::vector<uint8_t>> stateImage(const model::QwenStateStorage &storage,
                                             uint32_t lane) {
  return {bytesOf(storage.current(lane).stateBase), bytesOf(storage.window(lane))};
}

// The lengths of a lane at `tokens` whose rings and window hold the last
// draft window, as a published state's are.
model::QwenLogicalLengths published(uint64_t tokens) {
  const uint32_t window = static_cast<uint32_t>(std::min<uint64_t>(tokens, 2048));
  return {tokens, tokens - window, window, tokens};
}

// Every pooled buffer the storage returns to macOS, one reclaim step at a
// time, as the engine releases them.
uint64_t releaseAllIdle(model::QwenStateStorage &storage, bool keepLane) {
  uint64_t released = 0;
  while (const uint64_t buffer = storage.releaseOneIdle(keepLane))
    released += buffer;
  return released;
}

template <typename Ticket> bool finishWhenReady(Ticket &ticket) {
  while (!ticket.ready()) std::this_thread::yield();
  return ticket.finish();
}

void testLayoutFormulas() {
  require(kTargetState.convolutionLayerBytes() == 65'536,
          "GDN convolution layer formula is wrong");
  require(kTargetState.convolutionBytes() == 3'145'728,
          "GDN convolution parity formula is wrong");
  require(kTargetState.recurrentLayerBytes() == 3'145'728,
          "GDN recurrent layer formula is wrong");
  require(kTargetState.recurrentBytes() == 150'994'944,
          "GDN recurrent parity formula is wrong");
  require(kDraftState.tensorBytes() == 4'194'304,
          "draft tensor formula is wrong");
  require(kDraftState.ringBytes() == 41'943'040,
          "draft state formula is wrong");
  // 2048 rows of 5120 4-bit codes and 80 fp16 scale-minimum pairs.
  require(kDraftState.windowBytes() == 5'898'240,
          "context window formula is wrong");
  require(kStateLayout.laneBytes() == 356'122'624,
          "per-lane byte formula is wrong");
  require(uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                  kStateLayout.laneBytes() ==
              1'424'490'496,
          "four-lane byte formula is wrong");
  require(kStateLayout.cachedBytes() == 160'038'912,
          "prefix byte formula is wrong");
}

void testOffloadAllocationFailure(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1,
                          queryHostAvailableMemory, 0);
  constexpr model::CompositeStateLayout layout{{1, 3, 128, 1, 128, 128},
                                               {1, 1, 4, 128}};
  const uint64_t slotBytes = model::SlotFile::slotBytesFor(layout.cachedBytes());
  auto budget = std::make_shared<model::DiskBudget>(3 * slotBytes);
  auto file = std::make_shared<model::SlotFile>(slotBytes, budget);
  model::QwenStateStorage storage(backend, governor.allocationAdmission(), layout, file);
  require(static_cast<bool>(storage.tryActivateLane(0, 1)), "fault source activation failed");
  storage.updateLengths(0, published(4096));
  auto source = storage.snapshot(0);
  auto held = file->acquire();
  std::vector<std::byte> bytes(layout.cachedBytes());
  struct Result {
    bool failed;
    std::unique_ptr<StateOffload> transfer;
  };
  for (int failure = 0; failure < 64; ++failure) {
    // Keep the worker behind a barrier so a submitted write cannot finish
    // before the failure path has either drained it or returned unsafely.
    auto reached = std::make_shared<std::promise<void>>();
    std::promise<void> release;
    auto released = release.get_future().share();
    auto barrier = file->read(held, {bytes}, [reached, released] {
      reached->set_value();
      released.wait();
    });
    reached->get_future().wait();
    auto attempt = std::async(std::launch::async, [&] {
      allocationFailureAfter = failure;
      try {
        auto transfer = source->offload({});
        allocationFailureAfter = -1;
        return Result{false, std::move(transfer)};
      } catch (const std::bad_alloc &) {
        allocationFailureAfter = -1;
        return Result{true, {}};
      }
    });
    const bool returned = attempt.wait_for(std::chrono::milliseconds(100)) ==
                          std::future_status::ready;
    // The offload's slot is still held: its write waits behind the barrier.
    const bool pending = budget->usedBytes() > slotBytes;
    release.set_value();
    auto result = attempt.get();
    require(!(result.failed && returned && pending),
            "allocation failure released staging before the submitted write drained");
    if (!result.failed) {
      require(result.transfer && result.transfer->finish(),
              "offload did not recover after allocation failures");
      return;
    }
    require(budget->usedBytes() == slotBytes,
            "failed offload leaked its disk quota");
  }
  throw std::runtime_error("offload allocation failure sweep never reached success");
}

void testDiskRestore(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1,
                          queryHostAvailableMemory, 0);
  model::QwenStateStorage storage(
      backend, governor.allocationAdmission(), kStateLayout,
      std::make_shared<model::SlotFile>(kStateSlotBytes,
                                        std::make_shared<model::DiskBudget>(kStateSlotBytes)));
  require(static_cast<bool>(storage.tryActivateLane(0, 123)), "disk source activation failed");
  const model::GdnParityBuffers &gdn = storage.current(0);
  const metal::MetalBuffer &window = storage.window(0);
  const metal::MetalBuffer &keys = storage.draft(0)[0].keys;
  // Every byte of the state travels through the file; markers alone would
  // not notice a misplaced or truncated span.
  fill(gdn.stateBase, 1);
  word(gdn.stateBase) = 0x12345678;
  fill(window, 2);
  word(window) = 0x0c0c0c0c;
  const auto images = stateImage(storage, 0);
  storage.updateLengths(0, published(4096));
  auto source = storage.snapshot(0);
  auto write = source->offload({});
  require(write != nullptr, "disk offload not admitted");
  auto disk = write->state();
  require(disk && !disk->residentBytes(), "disk state retained resident allocation");
  // The write owns its copy: the source buffers are free before it finishes.
  source.reset();
  require(storage.idleCells() == 1 && storage.idleWindows() == 1 && storage.idleRings() == 0,
          "demotion did not return the source buffers at once");
  releaseAllIdle(storage, false);
  require(finishWhenReady(*write), "disk write failed");
  write.reset();
  const auto beforeRestore = storage.actualAllocatedBytes();
  word(gdn.stateBase) = 0;
  // The rings are the runtime's to compute from the window again.
  word(keys) = 0x5a5a5a5a;
  storage.updateLengths(0, {});
  bool committed = false;
  auto read = storage.beginRestore(0, *disk, true, {}, [&] { committed = true; });
  require(read && !committed, "disk restore committed before IO was consumed");
  require(finishWhenReady(*read) && committed, "disk restore failed");
  read.reset();
  require(storage.actualAllocatedBytes() == beforeRestore, "restore allocated a second state");
  require(stateImage(storage, 0) == images, "disk restore did not reproduce every state byte");
  require(word(gdn.stateBase) == 0x12345678 &&
              storage.metadata(0).lengths.targetTokens == 4096,
          "disk state changed target values or metadata");
  require(word(window) == 0x0c0c0c0c && word(keys) == 0x5a5a5a5a,
          "disk state changed the window, or a restore wrote the rings");
  read = storage.beginRestore(0, *disk, false, {}, [] {});
  require(finishWhenReady(*read) && storage.metadata(0).lengths.draftLength == 0 &&
              storage.metadata(0).lengths.draftBase == 4096 &&
              storage.metadata(0).lengths.windowEnd == 4096,
          "skipped draft restore retained stale context");
  auto promoted = read->snapshot();
  require(promoted && promoted->residentBytes() == kStateLayout.cachedBytes(),
          "completed disk restore could not create a resident snapshot");
  require(storage.metadata(0).lengths.draftLength == 0,
          "promotion changed the executing request's draft plan");
  word(gdn.stateBase) = 0;
  word(window) = 0;
  committed = false;
  require(!storage.beginRestore(0, *promoted, true, {}, [&] { committed = true; }) &&
              committed,
          "a resident restore returned a read or did not commit");
  require(word(gdn.stateBase) == 0x12345678 &&
              storage.metadata(0).lengths.hasCompleteDraftWindow(2048),
          "promotion lost the original complete state when execution skipped draft");
  require(word(window) == 0x0c0c0c0c, "promotion aliased mutable active buffers");
  require(stateImage(storage, 0) == images,
          "promoted snapshot did not reproduce every state byte");
  read.reset();
  disk.reset();
  promoted.reset();
  storage.releaseLane(0, 123);
}

// A lane whose state no cache slot can hold writes it from its own cells: no
// cache buffer is taken, the disk copy restores every byte of the active
// parity, and a full quota refuses until a disk copy is dropped.
void testDirectDiskSnapshot(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1,
                          queryHostAvailableMemory, 0);
  model::QwenStateStorage storage(
      backend, governor.allocationAdmission(), kStateLayout,
      std::make_shared<model::SlotFile>(kStateSlotBytes,
                                        std::make_shared<model::DiskBudget>(kStateSlotBytes)));
  require(storage.canSnapshotToDisk(), "a state file that holds one state refuses writes");
  require(static_cast<bool>(storage.tryActivateLane(0, 321)), "lane activation failed");
  storage.swapParity(0);
  const model::GdnParityBuffers &current = storage.current(0);
  const model::GdnParityBuffers &next = storage.next(0);
  fill(next.stateBase, 8);
  fill(current.stateBase, 7);
  word(current.stateBase) = 0x0badf00d;
  fill(storage.window(0), 20);
  const auto inactive = bytesOf(next.stateBase);
  const auto images = stateImage(storage, 0);
  storage.updateLengths(0, published(4096));
  const uint64_t before = storage.actualAllocatedBytes();
  auto write = storage.snapshotToDisk(0, {});
  require(write != nullptr, "direct disk snapshot was not admitted");
  require(storage.actualAllocatedBytes() == before && storage.idleCells() == 0 &&
              storage.idleWindows() == 0,
          "direct disk snapshot took a cache slot");
  auto disk = write->state();
  require(disk && !disk->residentBytes() && disk->bytes() == kStateLayout.cachedBytes(),
          "the ticket does not carry a disk copy");
  // The write reads staging, so the lane may move on at once.
  word(current.stateBase) = 0;
  require(finishWhenReady(*write), "direct disk write failed");
  write.reset();
  require(storage.snapshotToDisk(0, {}) == nullptr, "a full quota admitted a second state");

  fill(current.stateBase, 99);
  fill(storage.window(0), 98);
  storage.updateLengths(0, {});
  bool committed = false;
  auto read = storage.beginRestore(0, *disk, true, {}, [&] { committed = true; });
  require(finishWhenReady(*read) && committed, "restore of the direct disk copy failed");
  read.reset();
  require(stateImage(storage, 0) == images &&
              word(current.stateBase) == 0x0badf00d &&
              storage.metadata(0).lengths.targetTokens == 4096,
          "the disk copy did not reproduce the lane's active state");
  require(sameBytes(next.stateBase, inactive), "the inactive parity was touched");
  disk.reset();
  auto again = storage.snapshotToDisk(0, {});
  require(again != nullptr, "the dropped disk copy did not free its quota");
  require(finishWhenReady(*again), "the second direct disk write failed");
  again.reset();
  storage.releaseLane(0, 321);
}

// A state need not fill its slot: the write zeros the slot past it and the
// read leaves the rest, in a slot one host page larger than an aligned state
// and in the rounded-up slot of a state that is not aligned.
void testStateSmallerThanSlot(metal::MetalBackend &backend) {
  constexpr model::GdnStateLayout target{1, 3, 128, 1, 128, 128};
  constexpr model::CompositeStateLayout aligned{target, {1, 1, 4, 128}};
  constexpr model::CompositeStateLayout unaligned{target, {1, 1, 1, 64}};
  static_assert(aligned.cachedBytes() % kHostPageBytes == 0 &&
                unaligned.cachedBytes() % kHostPageBytes != 0);
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1,
                          queryHostAvailableMemory, 0);
  for (const auto &[layout, slotBytes] :
       {std::pair{aligned, aligned.cachedBytes() + kHostPageBytes},
        std::pair{unaligned, model::SlotFile::slotBytesFor(unaligned.cachedBytes())}}) {
    model::QwenStateStorage storage(backend, governor.allocationAdmission(), layout,
                                    std::make_shared<model::SlotFile>(
                                        slotBytes, std::make_shared<model::DiskBudget>(slotBytes)));
    require(static_cast<bool>(storage.tryActivateLane(0, 77)), "lane activation failed");
    const model::GdnParityBuffers &gdn = storage.current(0);
    fill(gdn.stateBase, 31);
    fill(storage.window(0), 32);
    const auto images = stateImage(storage, 0);
    storage.updateLengths(0, published(4096));
    auto write = storage.snapshotToDisk(0, {});
    require(write && finishWhenReady(*write), "a state did not reach a larger slot");
    auto disk = write->state();
    write.reset();

    fill(gdn.stateBase, 34);
    fill(storage.window(0), 35);
    storage.updateLengths(0, {});
    auto read = storage.beginRestore(0, *disk, true, {}, [] {});
    require(read && finishWhenReady(*read) && stateImage(storage, 0) == images,
            "a state did not come back exactly from a larger slot");
    read.reset();
    disk.reset();
    storage.releaseLane(0, 77);
  }
}

// A persistent tier keeps a written state's slot and label for the next
// process, whose storage takes the state back from its boundary alone: it
// restores every byte, with the lengths of a cached state's complete draft
// window, and a state whose slot changed after its record fails its
// restore.
void testPersistentStateComesBack(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1,
                          queryHostAvailableMemory, 0);
  std::string name = (std::filesystem::temp_directory_path() / "splash-states-XXXXXX").string();
  require(::mkdtemp(name.data()) != nullptr, "temporary directory could not be made");
  const std::filesystem::path directory = name;
  const model::SlotFile::Persistence persistence{directory / "state.slots",
                                                 directory / "state.records", {}, 8};
  const auto budget = std::make_shared<model::DiskBudget>(2 * kStateSlotBytes);
  std::vector<std::vector<uint8_t>> images;
  {
    auto file = std::make_shared<model::SlotFile>(kStateSlotBytes, budget, persistence);
    file->finishAdoption();
    model::QwenStateStorage storage(backend, governor.allocationAdmission(), kStateLayout, file);
    require(static_cast<bool>(storage.tryActivateLane(0, 5)), "lane activation failed");
    std::vector<std::shared_ptr<const CompositeState>> kept;
    for (uint64_t seed : {40, 50}) {
      fill(storage.current(0).stateBase, seed);
      fill(storage.window(0), seed + 1);
      if (images.empty())
        images = stateImage(storage, 0);
      storage.updateLengths(0, published(4096));
      auto write = storage.snapshotToDisk(0, {});
      require(write != nullptr, "a state write was refused");
      kept.push_back(write->state());
      kept.back()->label(std::vector<std::byte>(8, std::byte{static_cast<uint8_t>(seed)}));
      require(finishWhenReady(*write), "a state write failed");
    }
    require(file->synchronize({})->wait(), "the state file did not synchronize");
    file->seal();
    storage.releaseLane(0, 5);
  }
  {
    // The second state's slot changes after its record was written.
    std::fstream slots(directory / "state.slots", std::ios::in | std::ios::out | std::ios::binary);
    slots.seekp(static_cast<std::streamoff>(kStateSlotBytes + 100));
    slots.put('\x7f');
  }
  auto file = std::make_shared<model::SlotFile>(kStateSlotBytes, budget, persistence);
  model::QwenStateStorage storage(backend, governor.allocationAdmission(), kStateLayout, file);
  const std::vector<model::SlotRecord> records = file->records();
  require(records.size() == 2 && records[0].label == std::vector<std::byte>(8, std::byte{40}),
          "the next process did not find both labelled states");
  rejects([&] { static_cast<void>(storage.adopt(records[0], 4100)); },
          "composite snapshot requires equal page-aligned committed lengths",
          "a state was taken back at a boundary off a page");
  auto intact = storage.adopt(records[0], 4096);
  auto changed = storage.adopt(records[1], 4096);
  file->finishAdoption();
  require(intact && !intact->residentBytes() && intact->bytes() == kStateLayout.cachedBytes(),
          "a state taken back is not a disk copy of the layout");
  require(static_cast<bool>(storage.tryActivateLane(0, 6)), "lane activation failed");
  fill(storage.current(0).stateBase, 99);
  auto read = storage.beginRestore(0, *intact, true, {}, [] {});
  require(read && finishWhenReady(*read) && stateImage(storage, 0) == images &&
              storage.metadata(0).lengths == published(4096),
          "a state taken back did not restore every byte and its lengths");
  read = storage.beginRestore(0, *changed, true, {}, [] {});
  require(read && !finishWhenReady(*read), "a state whose slot changed was restored");
  read.reset();
  storage.releaseLane(0, 6);
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
}

// A persistent tier's copy of a state that stays in RAM is written from the
// RAM copy's own buffers: no staging, and the RAM copy keeps them. The disk
// copy restores every byte.
void testPersistKeepsTheRamCopy(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1,
                          queryHostAvailableMemory, 0);
  model::QwenStateStorage storage(
      backend, governor.allocationAdmission(), kStateLayout,
      std::make_shared<model::SlotFile>(kStateSlotBytes,
                                        std::make_shared<model::DiskBudget>(kStateSlotBytes)));
  require(static_cast<bool>(storage.tryActivateLane(0, 7)), "lane activation failed");
  fill(storage.current(0).stateBase, 60);
  fill(storage.window(0), 61);
  const auto images = stateImage(storage, 0);
  storage.updateLengths(0, published(4096));
  const auto source = storage.snapshot(0);
  const uint64_t allocated = storage.actualAllocatedBytes();
  auto write = source->persist({});
  require(write && !write->state()->residentBytes() &&
              source->residentBytes() == kStateLayout.cachedBytes() &&
              storage.actualAllocatedBytes() == allocated,
          "a persisted state staged its write or gave up its RAM copy");
  require(!source->persist({}), "a second copy was admitted past the quota");
  require(finishWhenReady(*write), "the persisted state's write failed");
  fill(storage.current(0).stateBase, 99);
  auto read = storage.beginRestore(0, *write->state(), true, {}, [] {});
  require(read && finishWhenReady(*read) && stateImage(storage, 0) == images &&
              storage.metadata(0).lengths == published(4096),
          "the persisted copy did not restore every byte and its lengths");
  read.reset();
  storage.releaseLane(0, 7);
}

void run(const std::string &metallib) {
  using model::QwenCompositeState;
  using model::QwenLogicalLengths;
  using model::QwenStateStorage;

  testLayoutFormulas();
  metal::MetalBackend backend(metallib);
  testOffloadAllocationFailure(backend);
  testDiskRestore(backend);
  testDirectDiskSnapshot(backend);
  testStateSmallerThanSlot(backend);
  testPersistentStateComesBack(backend);
  testPersistKeepsTheRamCopy(backend);
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1,
      queryHostAvailableMemory, 0);
  // Switched off to prove that a pooled cache slot needs no new admission;
  // the count is of the admissions granted.
  bool admitNewAllocations = true;
  uint32_t admissions = 0;
  auto admitState = [admit = governor.allocationAdmission(), &admitNewAllocations,
                     &admissions](uint64_t bytes, const std::function<void()> &allocate)
      -> metal::AllocationResult {
    if (!admitNewAllocations)
      return metal::AllocationFailure::EngineBudget;
    const metal::AllocationResult result = admit(bytes, allocate);
    if (result)
      ++admissions;
    return result;
  };
  uint64_t beforeStorage = backend.memoryStats().allocatedBytes;
  uint64_t observedStorageActual = 0;
  uint64_t observedLaneActual = 0;
  uint64_t observedPrefixActual = 0;

  {
    QwenStateStorage storage(backend, admitState, kStateLayout, nullptr);
    require(storage.actualAllocatedBytes() == 0 &&
                backend.memoryStats().allocatedBytes == beforeStorage,
            "lane state was allocated eagerly");
    for (uint32_t lane = 0;
         lane < model::ExecutionLimits::maximumBatchWidth;
         ++lane)
      require(!storage.metadata(lane).assigned(), "lane was assigned eagerly");
    rejects([&] { static_cast<void>(storage.current(4)); }, "invalid Qwen lane",
            "storage exposed more than four lanes");

    require(static_cast<bool>(storage.tryActivateLane(0, 101)),
            "lane activation failed");
    observedLaneActual = storage.actualAllocatedBytes();
    require(observedLaneActual >= kStateLayout.laneBytes(),
            "activated lane allocation is below declared bytes");
    require(storage.tryActivateLane(1, 202) &&
                storage.actualAllocatedBytes() == 2 * observedLaneActual,
            "activated lane accounting is not incremental");
    observedStorageActual = storage.actualAllocatedBytes();

    // A lane's accessors read the cells it holds now: at parity p, current()
    // is its cell p and next() its cell p ^ 1.
    void *stableGdnBase = storage.current(0).stateBase.contents();
    void *stableDraftBase = storage.draft(0)[0].keys.contents();
    require(stableGdnBase && stableDraftBase,
            "stable lane buffers are not CPU-visible");
    require(stableGdnBase != storage.current(1).stateBase.contents(),
            "two lanes alias one GDN allocation");
    require(storage.current(0).convolutionLayers[1].contents() ==
                    static_cast<uint8_t *>(stableGdnBase) +
                        kTargetState.convolutionLayerBytes() &&
                storage.current(0).recurrentLayers[1].contents() ==
                    static_cast<uint8_t *>(stableGdnBase) +
                        kTargetState.convolutionBytes() +
                        kTargetState.recurrentLayerBytes(),
            "GDN layer view offset is wrong");
    require(storage.current(0).stateBase.sizeBytes() ==
                    kTargetState.cellBytes() &&
                storage.draft(0).size() == kDraftState.layers &&
                storage.draft(0)[0].keys.sizeBytes() == kDraftState.tensorBytes() &&
                storage.window(0).sizeBytes() == kDraftState.windowBytes(),
            "split state-buffer sizes are wrong");

    require(storage.metadata(0).requestId == 101 &&
                storage.metadata(0).activeParity == 0,
            "lane activation metadata is wrong");
    rejects([&] { static_cast<void>(storage.tryActivateLane(0, 303)); },
            "Qwen lane is already assigned",
            "double lane activation was accepted");

    // Hot metadata updates must leave every buffer and marker untouched.
    word(storage.current(0).convolutionLayers[0]) = 0x10101010;
    word(storage.next(0).convolutionLayers[0]) = 0x21212121;
    word(storage.next(0).recurrentLayers[0]) = 0x31313131;
    const uint64_t windowTail = kDraftState.windowBytes() - sizeof(uint32_t);
    word(storage.window(0)) = 0x41414141;
    word(storage.window(0), windowTail) = 0x51515151;
    QwenLogicalLengths lengths{2'048, 0, 2'048, 2'048};
    storage.updateLengths(0, lengths);
    require(storage.metadata(0).lengths.draftLength == 2'048,
            "draft resident length is wrong");
    require(word(storage.next(0).convolutionLayers[0]) == 0x21212121 &&
                word(storage.window(0)) == 0x41414141,
            "length update copied or cleared hot state");
    storage.swapParity(0);
    require(storage.metadata(0).activeParity == 1,
            "parity swap did not select parity one");
    require(word(storage.next(0).convolutionLayers[0]) == 0x10101010 &&
                word(storage.current(0).convolutionLayers[0]) == 0x21212121,
            "parity swap copied hot state or kept the current cell");

    // Publication copies the lane's active-parity GDN cell and its context
    // window into a cache slot the governor admits. The lane keeps its own
    // cells: no address changes, no aliasing, no parity handoff.
    const uint64_t beforePrefix = backend.memoryStats().allocatedBytes;
    const uint64_t storageBeforePrefix = storage.actualAllocatedBytes();
    void *laneActiveGdnBase = storage.current(0).stateBase.contents();
    std::shared_ptr<const QwenCompositeState> prefix = storage.snapshot(0);
    require(prefix != nullptr, "snapshot could not obtain a cache slot");
    observedPrefixActual = backend.memoryStats().allocatedBytes - beforePrefix;
    require(prefix->bytes() == kStateLayout.cachedBytes(),
            "cached state footprint is not the declared cached bytes");
    require(observedPrefixActual >= kStateLayout.cachedBytes() &&
                storage.actualAllocatedBytes() ==
                    storageBeforePrefix + observedPrefixActual,
            "cache slot allocation is below declared bytes or unaccounted");
    require(storage.current(0).stateBase.contents() == laneActiveGdnBase &&
                storage.next(0).stateBase.contents() == stableGdnBase &&
                storage.draft(0)[0].keys.contents() == stableDraftBase,
            "snapshot moved or aliased the lane's own cells");
    require(storage.metadata(0).activeParity == 1 &&
                storage.metadata(0).lengths == lengths,
            "snapshot changed the lane's metadata");

    // The cached copy is independent of the lane: writes to the lane's
    // active cell or window after publication never reach a restore, which
    // leaves the rings to the runtime.
    word(storage.current(0).convolutionLayers[0]) = 0xa1a1a1a1;
    word(storage.current(0).recurrentLayers[0]) = 0xa2a2a2a2;
    word(storage.window(0)) = 0xa3a3a3a3;
    word(storage.current(1).convolutionLayers[0]) = 0xb0b0b0b0;
    word(storage.next(1).convolutionLayers[0]) = 0xb1b1b1b1;
    word(storage.draft(1)[0].keys) = 0xb2b2b2b2;
    void *destinationGdnBase = storage.current(1).stateBase.contents();
    void *destinationDraftBase = storage.draft(1)[0].keys.contents();
    require(!storage.beginRestore(1, *prefix, true, {}, [] {}),
            "a resident restore returned a read");

    require(storage.metadata(1).requestId == 202 &&
                storage.metadata(1).activeParity == 0 &&
                storage.metadata(1).lengths == lengths,
            "restore lost owner, changed parity, or lengths");
    require(storage.current(1).stateBase.contents() == destinationGdnBase &&
                storage.draft(1)[0].keys.contents() == destinationDraftBase,
            "restore replaced the destination's buffers");
    require(word(storage.current(1).convolutionLayers[0]) == 0x21212121 &&
                word(storage.current(1).recurrentLayers[0]) == 0x31313131,
            "restore did not deliver the pre-mutation GDN snapshot");
    require(word(storage.next(1).convolutionLayers[0]) == 0xb1b1b1b1,
            "restore overwrote inactive parity");
    require(word(storage.window(1)) == 0x41414141 &&
                word(storage.window(1), windowTail) == 0x51515151,
            "restore did not deliver the pre-mutation context window");
    require(word(storage.draft(1)[0].keys) == 0xb2b2b2b2,
            "a restore wrote the rings, which the runtime computes from the window");
    require(word(storage.current(0).convolutionLayers[0]) == 0xa1a1a1a1 &&
                word(storage.current(0).recurrentLayers[0]) == 0xa2a2a2a2 &&
                word(storage.window(0)) == 0xa3a3a3a3,
            "restore wrote back into the source lane");

    // A suffix that will capture a full 2048-token window restores only GDN:
    // the window is not copied to be overwritten by the next prefill.
    storage.swapParity(1);
    word(storage.window(1)) = 0xd2d2d2d2;
    require(!storage.beginRestore(1, *prefix, false, {}, [] {}),
            "a resident restore returned a read");
    require(word(storage.current(1).convolutionLayers[0]) == 0x21212121,
            "GDN-only restore did not restore convolution state");
    require(word(storage.window(1)) == 0xd2d2d2d2,
            "GDN-only restore copied an obsolete context window");
    require(storage.metadata(1).lengths.targetTokens == 2048 &&
                storage.metadata(1).lengths.draftLength == 0 &&
                storage.metadata(1).lengths.draftBase == 2048 &&
                storage.metadata(1).lengths.windowEnd == 2048,
            "GDN-only restore exposed stale draft metadata");
    // Cancellation releases ownership and returns the lane's buffers to the
    // pool. Reactivation takes them back in the same order and as they are:
    // only a cold start clears the current cell, which a restore would
    // overwrite instead.
    const uint64_t tailWord = kTargetState.cellBytes() - sizeof(uint32_t);
    void *reusableGdnBase = storage.next(0).stateBase.contents();
    void *reusableDraftBase = storage.draft(0)[0].keys.contents();
    word(storage.next(0).convolutionLayers[0]) = 0xc1c1c1c1;
    word(storage.next(0).recurrentLayers[0]) = 0xc2c2c2c2;
    word(storage.next(0).stateBase, tailWord) = 0xc3c3c3c3;
    storage.releaseLane(0, 101);
    require(!storage.metadata(0).assigned(), "cancellation did not release metadata");
    require(storage.idleCells() == 2 && storage.idleRings() == 1 &&
                storage.idleWindows() == 1 && storage.statesToActivate() == 0,
            "released lane buffers did not return to the pool");
    rejects([&] { storage.swapParity(0); }, "Qwen lane is not assigned",
            "unassigned lane accepted a parity update");
    require(static_cast<bool>(storage.tryActivateLane(0, 303)), "lane reuse failed");
    require(storage.idleCells() == 0 && storage.idleRings() == 0 &&
                storage.idleWindows() == 0,
            "reactivation left pooled buffers behind");
    require(storage.current(0).stateBase.contents() == reusableGdnBase &&
                storage.draft(0)[0].keys.contents() == reusableDraftBase,
            "lane reuse changed stable buffer addresses");
    require(storage.metadata(0).requestId == 303 &&
                storage.metadata(0).activeParity == 0 &&
                storage.metadata(0).lengths == QwenLogicalLengths{},
            "lane reuse did not reset logical state");
    require(word(storage.current(0).convolutionLayers[0]) == 0xc1c1c1c1 &&
                word(storage.current(0).recurrentLayers[0]) == 0xc2c2c2c2 &&
                word(storage.current(0).stateBase, tailWord) == 0xc3c3c3c3,
            "activation touched the pooled cell it took");
    storage.clearForColdStart(0);
    require(word(storage.current(0).convolutionLayers[0]) == 0 &&
                word(storage.current(0).recurrentLayers[0]) == 0 &&
                word(storage.current(0).stateBase, tailWord) == 0,
            "a cold start left readable state in the current cell");

    // Rejected publications fail before any cache slot is taken or admitted.
    const uint64_t beforeRejected = storage.actualAllocatedBytes();
    storage.updateLengths(0, {128, 0, 127});
    rejects([&] { storage.clearForColdStart(0); }, "a cold start begins at logical length zero",
            "a lane past length zero was cleared for a cold start");
    rejects([&] { static_cast<void>(storage.snapshot(0)); },
            "composite snapshot requires equal page-aligned committed lengths",
            "snapshot accepted divergent target/draft lengths");
    rejects(
        [&] {
          storage.updateLengths(0, {129, 0, 129});
          static_cast<void>(storage.snapshot(0));
        },
        "composite snapshot requires equal page-aligned committed lengths",
        "unaligned prefix snapshot was accepted");
    // Decode writes the rings alone: a lane past that has no window to keep.
    rejects(
        [&] {
          storage.updateLengths(0, {2'048, 0, 2'048, 2'040});
          static_cast<void>(storage.snapshot(0));
        },
        "a composite snapshot needs the context window of its draft rings",
        "a snapshot was taken behind its context window");
    rejects([&] { storage.updateLengths(0, {2'048, 0, 2'040, 2'048}); }, "invalid draft ring metadata",
            "a window past the rings was accepted");
    require(storage.actualAllocatedBytes() == beforeRejected,
            "rejected snapshot allocated or consumed a cache slot");

    // A decode step leaves the rows it retained pending on the lane's tape:
    // until the lane's next decode step folds them in, its cell does not hold
    // its state, which neither a snapshot, a restore nor a prefill step reads.
    storage.updateLengths(0, {2'048, 0, 2'048, 2'048});
    const uint32_t parity = storage.metadata(0).activeParity;
    for (const uint32_t retained : {0U, model::ExecutionLimits::targetVerifyRows + 1})
      rejects([&] { storage.swapDecodeParity(0, retained); }, "a decode step retains one to all of its rows",
              "a decode step retained no rows or more than its rows");
    storage.swapDecodeParity(0, 3);
    require(storage.metadata(0).pendingRows == 3 && storage.metadata(0).activeParity != parity,
            "a decode step's swap lost its retained rows");
    rejects([&] { static_cast<void>(storage.snapshot(0)); }, "Qwen lane state has decoded rows pending",
            "a snapshot read a cell without its retained rows");
    rejects([&] { static_cast<void>(storage.snapshotToDisk(0, {})); }, "Qwen lane state has decoded rows pending",
            "a disk snapshot read a cell without its retained rows");
    rejects([&] { static_cast<void>(storage.beginRestore(0, *prefix, true, {}, [] {})); },
            "Qwen lane state has decoded rows pending", "a restore overwrote a cell with pending rows");
    rejects([&] { storage.swapParity(0); }, "Qwen lane state has decoded rows pending",
            "a prefill step followed a decode step's pending rows");
    storage.swapDecodeParity(0, model::ExecutionLimits::targetVerifyRows);
    require(storage.metadata(0).pendingRows == model::ExecutionLimits::targetVerifyRows &&
                storage.metadata(0).activeParity == parity,
            "the next decode step's swap did not take its own retained rows");
    rejects([&] { storage.releaseLane(0, 404); }, "Qwen lane owner mismatch",
            "lane release accepted the wrong owner");

    const uint64_t beforeSuspend = storage.actualAllocatedBytes();
    storage.releaseLane(0, 303);
    require(releaseAllIdle(storage, false) == observedLaneActual,
            "recomputation preemption retained active backing");
    require(!storage.metadata(0).assigned() &&
                storage.actualAllocatedBytes() == beforeSuspend - observedLaneActual,
            "preempted GDN or draft bytes remain outside the cache");
    require(storage.tryActivateLane(0, 303) &&
                storage.metadata(0).assigned() &&
                storage.metadata(0).lengths == QwenLogicalLengths{} &&
                !storage.metadata(0).pendingRows,
            "recomputation did not start from a fresh empty state");

    // Dropping a cached state returns its buffers to the storage's pool rather
    // than freeing them: accounting stays flat, and the next publication takes
    // the pooled buffers without a governor admission. Only releasing idle
    // buffers returns pooled bytes to macOS.
    const uint64_t beforeDrop = storage.actualAllocatedBytes();
    const uint64_t backendBeforeDrop = backend.memoryStats().allocatedBytes;
    prefix.reset();
    require(storage.actualAllocatedBytes() == beforeDrop &&
                backend.memoryStats().allocatedBytes == backendBeforeDrop,
            "dropped cached state freed its slot instead of pooling it");
    storage.updateLengths(0, lengths);
    word(storage.current(0).convolutionLayers[0]) = 0xe1e1e1e1;
    word(storage.current(0).recurrentLayers[0]) = 0xe2e2e2e2;
    word(storage.window(0)) = 0xe3e3e3e3;
    admitNewAllocations = false;
    std::shared_ptr<const QwenCompositeState> pooled = storage.snapshot(0);
    require(pooled != nullptr, "snapshot did not reuse the pooled cache slot");
    require(pooled->bytes() == kStateLayout.cachedBytes() &&
                storage.actualAllocatedBytes() == beforeDrop &&
                backend.memoryStats().allocatedBytes == backendBeforeDrop,
            "pooled cache slot reuse allocated new buffers");
    require(storage.snapshot(0) == nullptr,
            "snapshot with an empty pool bypassed the governor");
    require(storage.actualAllocatedBytes() == beforeDrop,
            "denied snapshot leaked cache slot bytes");
    admitNewAllocations = true;
    require(!storage.beginRestore(1, *pooled, true, {}, [] {}),
            "a resident restore returned a read");
    require(storage.metadata(1).activeParity == 1 &&
                storage.metadata(1).lengths == lengths &&
                word(storage.current(1).convolutionLayers[0]) == 0xe1e1e1e1 &&
                word(storage.current(1).recurrentLayers[0]) == 0xe2e2e2e2 &&
                word(storage.window(1)) == 0xe3e3e3e3,
            "reused cache slot served stale contents");
    require(word(storage.next(1).convolutionLayers[0]) == 0x21212121,
            "restore from the reused slot overwrote inactive parity");
    pooled.reset();
    require(storage.actualAllocatedBytes() == beforeDrop,
            "second dropped cached state was freed instead of pooled");
    require(releaseAllIdle(storage, false) == observedPrefixActual &&
                storage.actualAllocatedBytes() ==
                    beforeDrop - observedPrefixActual,
            "releasing idle buffers did not free the pooled cache slot");
    require(storage.metadata(0).assigned() && storage.metadata(1).assigned() &&
                storage.actualAllocatedBytes() == 2 * observedLaneActual,
            "pool reclaim touched active lane cells");

    // A live cached state survives its lane's release and reclaim; only its
    // drop plus a later reclaim frees the slot together with idle cells.
    std::shared_ptr<const QwenCompositeState> retained = storage.snapshot(1);
    require(retained != nullptr, "retained snapshot could not admit a slot");
    require(storage.actualAllocatedBytes() ==
                2 * observedLaneActual + observedPrefixActual,
            "retained snapshot accounting is wrong");
    storage.releaseLane(0, 303);
    storage.releaseLane(1, 202);
    require(storage.idleCells() == 4 && storage.idleRings() == 2 &&
                storage.idleWindows() == 2,
            "released lane buffers are missing from the pool");
    retained.reset();
    require(storage.idleCells() == 5 && storage.idleRings() == 2 &&
                storage.idleWindows() == 3,
            "dropped cached state did not return its cell and window to the pool");
    // Releasing down to one lane's worth keeps two cells, the rings and a
    // window warm.
    require(releaseAllIdle(storage, true) ==
                observedLaneActual + observedPrefixActual &&
                storage.idleCells() == 2 && storage.idleRings() == 1 &&
                storage.idleWindows() == 1,
            "partial idle release did not keep the requested buffers");
    require(releaseAllIdle(storage, false) == observedLaneActual,
            "idle lane buffers were not reclaimed");
    require(storage.idleCells() == 0 && storage.idleRings() == 0 &&
                storage.idleWindows() == 0,
            "reclaimed buffers remain pooled");
    // No cached state holds rings: without idle rings no number of states
    // makes up a lane.
    constexpr uint32_t kNoStatesSuffice = std::numeric_limits<uint32_t>::max();
    require(storage.actualAllocatedBytes() == 0 &&
                storage.statesToActivate() == kNoStatesSuffice,
            "reclaimed state cells remain accounted");

    // An activation asks the governor once for everything the pool lacks:
    // a refusal allocates nothing and leaves the pool as it was.
    admitNewAllocations = false;
    require(!storage.tryActivateLane(0, 505) && !storage.metadata(0).assigned() &&
                storage.actualAllocatedBytes() == 0 &&
                storage.statesToActivate() == kNoStatesSuffice,
            "a denied activation allocated part of its lane");
    admitNewAllocations = true;
    uint32_t admitted = admissions;
    require(static_cast<bool>(storage.tryActivateLane(0, 505)) &&
                admissions == admitted + 1 &&
                storage.actualAllocatedBytes() == observedLaneActual,
            "an activation asked the governor for its buffers one by one");
    // With one cell, the rings and a window in the pool the lane lacks a
    // cell: a refusal leaves them pooled, and the retry is admitted that cell
    // alone.
    storage.releaseLane(0, 505);
    require(storage.releaseOneIdle(false) != 0 && storage.idleCells() == 1 &&
                storage.idleRings() == 1 && storage.idleWindows() == 1 &&
                storage.statesToActivate() == 1,
            "fixture pool does not hold one cell, the rings and a window");
    const uint64_t pooledBytes = storage.actualAllocatedBytes();
    admitNewAllocations = false;
    require(!storage.tryActivateLane(0, 506) && storage.idleCells() == 1 &&
                storage.idleRings() == 1 && storage.idleWindows() == 1 &&
                storage.actualAllocatedBytes() == pooledBytes,
            "a denied activation took or dropped the pooled buffers");
    admitNewAllocations = true;
    admitted = admissions;
    require(static_cast<bool>(storage.tryActivateLane(0, 506)) &&
                admissions == admitted + 1 && storage.idleCells() == 0 &&
                storage.idleRings() == 0 && storage.idleWindows() == 0 &&
                storage.actualAllocatedBytes() == observedLaneActual,
            "the retry did not take the pooled buffers and one admission for the rest");
    storage.releaseLane(0, 506);
    require(releaseAllIdle(storage, false) == observedLaneActual &&
                storage.actualAllocatedBytes() == 0,
            "the lane's buffers were not reclaimed");
    // What else a request's start allocates joins the lane's admission: one
    // call for both, and a refusal builds neither.
    bool extraBuilt = false;
    const auto buildExtra = [&] { extraBuilt = true; };
    admitNewAllocations = false;
    require(!storage.tryActivateLane(0, 507, 4096, buildExtra) && !extraBuilt &&
                storage.actualAllocatedBytes() == 0,
            "a refused start built what came with its lane");
    admitNewAllocations = true;
    admitted = admissions;
    require(static_cast<bool>(storage.tryActivateLane(0, 507, 4096, buildExtra)) &&
                extraBuilt && admissions == admitted + 1 &&
                storage.actualAllocatedBytes() == observedLaneActual,
            "a start was not admitted in one piece");
    storage.releaseLane(0, 507);
    require(releaseAllIdle(storage, false) == observedLaneActual &&
                storage.actualAllocatedBytes() == 0,
            "the started lane's buffers were not reclaimed");
  }
  require(backend.memoryStats().allocatedBytes == beforeStorage,
          "destroyed state lanes remained in actual allocation count");

  std::cout << "qwen state storage tests passed: lane_declared="
            << kStateLayout.laneBytes()
            << " lane_actual=" << observedLaneActual << " four_lanes_declared="
            << uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                   kStateLayout.laneBytes()
            << " four_lanes_actual=" << observedStorageActual
            << " composite_declared=" << kStateLayout.cachedBytes()
            << " prefix_actual=" << observedPrefixActual << '\n';
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 2) {
    std::cerr << "usage: qwen_state_storage_test METALLIB\n";
    return EXIT_FAILURE;
  }
  @autoreleasepool {
    try {
      run(argv[1]);
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "qwen state storage test failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
