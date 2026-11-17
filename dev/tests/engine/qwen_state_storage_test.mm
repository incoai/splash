#include "ChainRows.hpp"
#include "Checked.hpp"
#include "engine/MemoryGovernor.hpp"
#include "tests/engine/AllocationFailure.hpp"
#include "tests/engine/ClaimScope.hpp"
#include "model/QwenState.hpp"
#include "tests/engine/StateCodecReference.hpp"
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
#include <deque>
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

// Fills a lane's cell: arbitrary bytes, but for its recurrent layers, which
// hold finite values in [-1, 1), in the layout's format, for a cached state to
// code.
void fillState(const model::GdnParityBuffers &cell, const model::GdnStateLayout &layout, uint64_t seed) {
  fill(cell.stateBase, seed);
  uint64_t x = seed * 2 + 1;
  const uint64_t values = uint64_t{layout.recurrentGroups} * layout.recurrentRows * layout.recurrentColumns;
  for (const metal::MetalBuffer &layer : cell.recurrentLayers) {
    for (uint64_t i = 0; i < values; ++i) {
      x ^= x << 13; x ^= x >> 7; x ^= x << 17;
      const float value = static_cast<float>(x % 2'000'001) / 1e6F - 1.0F;
      if (layout.format == ops::GdnStateFormat::BFloat16)
        static_cast<uint16_t *>(layer.contents())[i] = test::codecBf16(value);
      else
        static_cast<float *>(layer.contents())[i] = value;
    }
  }
}

// A recurrent layer's `count` values at `at`, widened where the layout runs
// them in bf16.
std::vector<float> recurrentValues(const uint8_t *at, uint64_t count, const model::GdnStateLayout &layout) {
  std::vector<float> values(count);
  for (uint64_t i = 0; i < count; ++i)
    values[i] = layout.format == ops::GdnStateFormat::BFloat16
                    ? test::codecWiden(reinterpret_cast<const uint16_t *>(at)[i])
                    : reinterpret_cast<const float *>(at)[i];
  return values;
}

// The values of a lane's recurrent layers, decoded from the codes a cached
// state stores of them (ops::StateCodec), in the layout's format.
void decodeRecurrent(std::vector<uint8_t> &cell, const model::GdnStateLayout &layout) {
  const uint32_t rows = layout.recurrentGroups * layout.recurrentRows, columns = layout.recurrentColumns;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    uint8_t *at = cell.data() + layout.convolutionBytes() + uint64_t{layer} * layout.recurrentLayerBytes();
    const std::vector<float> values = recurrentValues(at, uint64_t{rows} * columns, layout);
    const std::vector<uint8_t> codes = test::stateCodesReference(values.data(), rows, columns);
    const auto *scales = reinterpret_cast<const uint16_t *>(codes.data() + state_codec_scales_offset(rows, columns));
    for (uint64_t i = 0; i < values.size(); ++i) {
      const float decoded = static_cast<float>(static_cast<int8_t>(codes[i])) *
                            test::codecWiden(scales[i / SPLASH_STATE_CODEC_BLOCK]);
      if (layout.format == ops::GdnStateFormat::BFloat16)
        reinterpret_cast<uint16_t *>(at)[i] = test::codecBf16(decoded);
      else
        reinterpret_cast<float *>(at)[i] = decoded;
    }
  }
}

// What a restore of a lane's current state reproduces, in stateImage's
// order: its cell with its recurrent layers decoded from their codes, and its
// window.
std::vector<std::vector<uint8_t>> restoredImage(const model::QwenStateStorage &storage, uint32_t lane) {
  std::vector<std::vector<uint8_t>> image = stateImage(storage, lane);
  decodeRecurrent(image[0], storage.layout().target);
  return image;
}

// The cached cell a lane's current state stores: its convolution rows, each
// recurrent layer's codes and its auxiliary state.
std::vector<uint8_t> cachedImage(const model::QwenStateStorage &storage, uint32_t lane) {
  const model::GdnStateLayout layout = storage.layout().target;
  const std::vector<uint8_t> cell = bytesOf(storage.current(lane).stateBase);
  std::vector<uint8_t> image(layout.cachedCellBytes());
  std::copy_n(cell.begin(), layout.convolutionBytes(), image.begin());
  const uint32_t rows = layout.recurrentGroups * layout.recurrentRows;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const std::vector<float> values =
        recurrentValues(cell.data() + layout.convolutionBytes() + uint64_t{layer} * layout.recurrentLayerBytes(),
                        uint64_t{rows} * layout.recurrentColumns, layout);
    const std::vector<uint8_t> codes = test::stateCodesReference(values.data(), rows, layout.recurrentColumns);
    std::copy(codes.begin(), codes.end(),
              image.begin() + layout.convolutionBytes() + uint64_t{layer} * layout.cachedRecurrentLayerBytes());
  }
  return image;
}

// Whether a cached cell holds `image` (cachedImage) where a state is stored:
// its convolution rows, each layer's codes and scales and its auxiliary
// state, not the pages' padding after a layer's scales.
bool holdsCached(const metal::MetalBuffer &cell, const std::vector<uint8_t> &image, const model::GdnStateLayout &layout) {
  const auto *bytes = static_cast<const uint8_t *>(cell.contents());
  const auto same = [&](uint64_t offset, uint64_t count) {
    return std::memcmp(bytes + offset, image.data() + offset, count) == 0;
  };
  const uint64_t codes = state_codec_layer_bytes(layout.recurrentGroups * layout.recurrentRows, layout.recurrentColumns);
  bool held = cell.sizeBytes() == image.size() && same(0, layout.convolutionBytes()) &&
              same(layout.convolutionBytes() + layout.cachedRecurrentBytes(), layout.auxiliaryBytes);
  for (uint32_t layer = 0; layer < layout.layers; ++layer)
    held = held && same(layout.convolutionBytes() + uint64_t{layer} * layout.cachedRecurrentLayerBytes(), codes);
  return held;
}

// A block of a recurrent layer's first values that its codes hold exactly:
// multiples of 2^-10 up to 127 of them, which a block's scale of 2^-10
// codes as the multiples.
void markBlock(const metal::MetalBuffer &layer, int32_t seed) {
  auto *values = static_cast<float *>(layer.contents());
  for (int32_t i = 0; i < static_cast<int32_t>(SPLASH_STATE_CODEC_BLOCK); ++i)
    values[i] = static_cast<float>(i ? (seed + 7 * i) % 255 - 127 : 127) / 1024.0F;
}

bool marked(const metal::MetalBuffer &layer, int32_t seed) {
  const auto *values = static_cast<const float *>(layer.contents());
  for (int32_t i = 0; i < static_cast<int32_t>(SPLASH_STATE_CODEC_BLOCK); ++i)
    if (values[i] != static_cast<float>(i ? (seed + 7 * i) % 255 - 127 : 127) / 1024.0F)
      return false;
  return true;
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
  // A cached state's recurrent layer: a byte of code per value and a bf16
  // scale per 32 of them.
  require(kTargetState.cachedRecurrentLayerBytes() == 835'584 &&
              kTargetState.cachedCellBytes() == 43'253'760,
          "cached recurrent layer formula is wrong");
  require(kStateLayout.cachedBytes() == 49'152'000,
          "prefix byte formula is wrong");
  // A bf16 recurrent state takes half the bytes; the convolution rows keep
  // theirs, and a cached state its codes.
  model::CompositeStateLayout bfloat = kStateLayout;
  bfloat.target.format = ops::GdnStateFormat::BFloat16;
  require(bfloat.target.recurrentLayerBytes() == 1'572'864 &&
              bfloat.target.convolutionBytes() == 3'145'728 &&
              bfloat.laneBytes() == 205'127'680 && bfloat.cachedBytes() == 49'152'000,
          "bf16 state byte formula is wrong");
}

void testOffloadAllocationFailure(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  // What a case's storage allocates through the governor is a start's, in
  // its Floor claim (design/M.md L7).
  const test::ClaimScope start(governor, ClaimClass::Floor);
  constexpr model::CompositeStateLayout layout{{1, 3, 128, 1, 128, 128},
                                               {1, 1, 4, 128}};
  const uint64_t slotBytes = model::SlotFile::slotBytesFor(layout.cachedBytes());
  auto budget = std::make_shared<model::DiskBudget>(3 * slotBytes);
  auto file = std::make_shared<model::SlotFile>(slotBytes, budget);
  model::QwenStateStorage storage(backend, governor.allocationAdmission(), layout, file);
  require(static_cast<bool>(storage.tryActivateLane(0, 1)), "fault source activation failed");
  storage.updateLengths(0, published(4096));
  auto source = storage.snapshot(0);
  auto held = file->acquire(layout.cachedBytes());
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
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  model::QwenStateStorage storage(
      backend, governor.allocationAdmission(), kStateLayout,
      std::make_shared<model::SlotFile>(kStateSlotBytes,
                                        std::make_shared<model::DiskBudget>(kStateSlotBytes)));
  require(static_cast<bool>(storage.tryActivateLane(0, 123)), "disk source activation failed");
  const model::GdnParityBuffers &gdn = storage.current(0);
  const metal::MetalBuffer &window = storage.window(0);
  const metal::MetalBuffer &keys = storage.draft(0)[0].keys;
  // Every byte of the state travels through the file, its recurrent layers'
  // as codes; markers alone would not notice a misplaced or truncated span.
  fillState(gdn, kTargetState, 1);
  word(gdn.stateBase) = 0x12345678;
  fill(window, 2);
  word(window) = 0x0c0c0c0c;
  const auto images = restoredImage(storage, 0);
  storage.updateLengths(0, published(4096));
  auto source = storage.snapshot(0);
  auto write = source->offload({});
  require(write != nullptr, "disk offload not admitted");
  auto disk = write->state();
  require(disk && !disk->residentBytes(), "disk state retained resident allocation");
  // The write owns its copy: the source buffers are free before it finishes.
  source.reset();
  require(storage.idleCachedCells() == 1 && storage.idleWindows() == 1 && storage.idleRings() == 0,
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

// With --cached-states exact a cached state keeps the running format: its
// cell is a lane's, and a restore from its RAM copy or from the disk
// reproduces every byte of the lane's state, NaN patterns and all, in fp32
// and in bf16, where codes reproduce the values to half a step.
void testExactCachedStatesRestoreEveryByte(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  for (const ops::GdnStateFormat format : {ops::GdnStateFormat::Float32, ops::GdnStateFormat::BFloat16}) {
    const std::string label = std::string(ops::formatName(format)) + " exact cached state: ";
    model::CompositeStateLayout layout = kStateLayout;
    layout.target.format = format;
    layout.target.cachedCodes = false;
    require(layout.target.cachedCellBytes() == layout.target.cellBytes() &&
                layout.cachedBytes() == layout.target.cellBytes() + kDraftState.windowBytes(),
            label + "it is not a lane's cell and window");
    const uint64_t slotBytes = model::SlotFile::slotBytesFor(layout.cachedBytes());
    model::QwenStateStorage storage(
        backend, governor.allocationAdmission(), layout,
        std::make_shared<model::SlotFile>(slotBytes, std::make_shared<model::DiskBudget>(slotBytes)));
    require(static_cast<bool>(storage.tryActivateLane(0, 7)), label + "lane activation failed");
    fill(storage.current(0).stateBase, 11);
    fill(storage.window(0), 12);
    const auto image = stateImage(storage, 0);
    storage.updateLengths(0, published(4096));
    auto resident = storage.snapshot(0);
    require(resident && resident->residentBytes() == layout.cachedBytes(), label + "the RAM copy is not its size");
    fill(storage.current(0).stateBase, 13);
    require(!storage.beginRestore(0, *resident, true, {}, [] {}) && stateImage(storage, 0) == image,
            label + "a restore from RAM did not reproduce every byte");
    auto write = resident->offload({});
    require(write != nullptr, label + "the disk copy was refused");
    auto disk = write->state();
    resident.reset();
    require(finishWhenReady(*write), label + "the disk write failed");
    write.reset();
    fill(storage.current(0).stateBase, 14);
    auto read = storage.beginRestore(0, *disk, true, {}, [] {});
    require(read && finishWhenReady(*read) && stateImage(storage, 0) == image,
            label + "a restore from the disk did not reproduce every byte");
    read.reset();
    disk.reset();
    storage.releaseLane(0, 7);
  }
}

// A lane whose state no cache slot can hold writes it from its own cells: no
// cache buffer is taken, the disk copy restores every byte of the active
// parity, and a full quota refuses until a disk copy is dropped.
void testDirectDiskSnapshot(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  model::QwenStateStorage storage(
      backend, governor.allocationAdmission(), kStateLayout,
      std::make_shared<model::SlotFile>(kStateSlotBytes,
                                        std::make_shared<model::DiskBudget>(kStateSlotBytes)));
  require(storage.canSnapshotToDisk(false), "a state file that holds one state refuses writes");
  require(static_cast<bool>(storage.tryActivateLane(0, 321)), "lane activation failed");
  storage.swapParity(0);
  const model::GdnParityBuffers &current = storage.current(0);
  const model::GdnParityBuffers &next = storage.next(0);
  fill(next.stateBase, 8);
  fillState(current, kTargetState, 7);
  word(current.stateBase) = 0x0badf00d;
  fill(storage.window(0), 20);
  const auto images = restoredImage(storage, 0);
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
  // The read lands in the inactive parity, which the lane's next
  // transition writes before it reads it.
  require(std::memcmp(next.stateBase.contents(), current.stateBase.contents(), kTargetState.convolutionBytes()) == 0,
          "the read did not land in the lane's other cell");
  disk.reset();
  auto again = storage.snapshotToDisk(0, {});
  require(again != nullptr, "the dropped disk copy did not free its quota");
  require(finishWhenReady(*again), "the second direct disk write failed");
  again.reset();
  storage.releaseLane(0, 321);
}

// A state's write, from a lane or from a cached copy, takes its slot from
// the write budget. Overdrawn, the budget holds back writes not in use
// until it refills, and lets those in use pass.
void testStateWritesDrawOnTheWriteBudget(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  double now = 0.0;
  auto budget = std::make_shared<model::DiskBudget>(4 * kStateSlotBytes, [&now] { return now; });
  // Two slots a day, half a slot at once.
  budget->setWriteBudget(2 * kStateSlotBytes);
  model::QwenStateStorage storage(backend, governor.allocationAdmission(), kStateLayout,
                                  std::make_shared<model::SlotFile>(kStateSlotBytes, budget));
  require(static_cast<bool>(storage.tryActivateLane(0, 456)), "lane activation failed");
  storage.updateLengths(0, published(4096));
  auto cached = storage.snapshot(0);
  require(cached && storage.canSnapshotToDisk(false) && cached->canOffload(false),
          "a write budget with room held a state back");
  auto write = storage.snapshotToDisk(0, {});
  require(write && finishWhenReady(*write), "the lane's state was not written");
  require(!storage.canSnapshotToDisk(false) && storage.canSnapshotToDisk(true) &&
              !cached->canOffload(false) && cached->canOffload(true),
          "an overdrawn write budget did not hold back only what is not in use");
  now = 86'400'000.0 / 4 + 1.0;
  require(storage.canSnapshotToDisk(false) && cached->canOffload(false),
          "the write budget did not admit writes once it refilled");
  write.reset();
  cached.reset();
  storage.releaseLane(0, 456);
}

// A state need not fill its slot: it comes back exactly from a slot one host
// page larger than an aligned state and from the rounded-up slot of a state
// that is not aligned. slot_file_test checks that a write stores only the
// pages its state reaches.
void testStateSmallerThanSlot(metal::MetalBackend &backend) {
  constexpr model::GdnStateLayout target{1, 3, 128, 1, 128, 128};
  constexpr model::CompositeStateLayout aligned{target, {1, 1, 4, 128}};
  constexpr model::CompositeStateLayout unaligned{target, {1, 1, 1, 64}};
  static_assert(aligned.cachedBytes() % kHostPageBytes == 0 &&
                unaligned.cachedBytes() % kHostPageBytes != 0);
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  for (const auto &[layout, slotBytes] :
       {std::pair{aligned, aligned.cachedBytes() + kHostPageBytes},
        std::pair{unaligned, model::SlotFile::slotBytesFor(unaligned.cachedBytes())}}) {
    model::QwenStateStorage storage(backend, governor.allocationAdmission(), layout,
                                    std::make_shared<model::SlotFile>(
                                        slotBytes, std::make_shared<model::DiskBudget>(slotBytes)));
    require(static_cast<bool>(storage.tryActivateLane(0, 77)), "lane activation failed");
    const model::GdnParityBuffers &gdn = storage.current(0);
    fillState(gdn, target, 31);
    fill(storage.window(0), 32);
    const auto images = restoredImage(storage, 0);
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
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
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
      fillState(storage.current(0), kTargetState, seed);
      fill(storage.window(0), seed + 1);
      if (images.empty())
        images = restoredImage(storage, 0);
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
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  model::QwenStateStorage storage(
      backend, governor.allocationAdmission(), kStateLayout,
      std::make_shared<model::SlotFile>(kStateSlotBytes,
                                        std::make_shared<model::DiskBudget>(kStateSlotBytes)));
  require(static_cast<bool>(storage.tryActivateLane(0, 7)), "lane activation failed");
  fillState(storage.current(0), kTargetState, 60);
  fill(storage.window(0), 61);
  const auto images = restoredImage(storage, 0);
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

// A state a prefill command captures takes its cache slot before the command
// runs: the pool's buffers, one admission for what the pool lacks, nothing
// on a refusal, whose cause it reports. copyWindow() fills the slot's window
// alone, copyBeside() its convolution rows and auxiliary state too, and
// storeState() the lane's whole state, its recurrent layers as their codes.
// Sealing moves the slot into the cached state, which carries the tier's
// file and staging as a snapshot does, and leaves the reservation empty: a
// dropped reservation returns what it still holds, so each buffer goes back
// to the pool once, and nothing once its storage is gone.
void testReservations(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  bool admitNew = true;
  uint32_t admissions = 0;
  const metal::AllocationAdmission admit = [admit = governor.allocationAdmission(), &admitNew, &admissions](
                                               uint64_t bytes, const std::function<void()> &allocate)
      -> metal::AllocationResult {
    if (!admitNew)
      return metal::AllocationFailure::EngineBudget;
    const metal::AllocationResult result = admit(bytes, allocate);
    admissions += static_cast<bool>(result);
    return result;
  };
  constexpr model::CompositeStateLayout layout{{1, 3, 128, 1, 128, 128}, {1, 1, 4, 128}};
  const uint64_t slotBytes = model::SlotFile::slotBytesFor(layout.cachedBytes());
  const uint64_t beforeStorage = backend.memoryStats().allocatedBytes;
  std::shared_ptr<model::QwenStateReservation> outlives;
  {
    model::QwenStateStorage storage(
        backend, admit, layout,
        std::make_shared<model::SlotFile>(slotBytes, std::make_shared<model::DiskBudget>(slotBytes)));
    require(static_cast<bool>(storage.tryActivateLane(0, 9)), "lane activation failed");
    fillState(storage.current(0), layout.target, 70);
    fill(storage.window(0), 71);
    storage.updateLengths(0, published(4096));
    const auto images = restoredImage(storage, 0);
    const std::vector<uint8_t> cached = cachedImage(storage, 0);
    const uint64_t laneBytes = storage.actualAllocatedBytes();

    admitNew = false;
    metal::AllocationFailure refusal;
    require(!storage.reserve(refusal) && refusal == metal::AllocationFailure::EngineBudget &&
                storage.actualAllocatedBytes() == laneBytes,
            "a refused reservation allocated its slot or lost its cause");
    admitNew = true;
    auto reservation = storage.reserve(refusal);
    require(reservation && refusal == metal::AllocationFailure::None && admissions == 2 &&
                storage.actualAllocatedBytes() > laneBytes &&
                storage.idleCachedCells() == 0 && storage.idleWindows() == 0,
            "a reservation did not take one admission for its slot");
    require(reservation->buffers().gdn->buffers().stateBase.sizeBytes() == layout.target.cachedCellBytes(),
            "a reservation's cell is not a cached cell");
    const uint64_t slotted = storage.actualAllocatedBytes();
    reservation.reset();
    require(storage.idleCachedCells() == 1 && storage.idleCells() == 0 && storage.idleWindows() == 1 &&
                storage.actualAllocatedBytes() == slotted,
            "a dropped reservation did not return its slot to the pool");
    admitNew = false;
    reservation = storage.reserve(refusal);
    require(reservation && storage.idleCachedCells() == 0 && storage.idleWindows() == 0,
            "a reservation did not take the pooled slot without an admission");
    admitNew = true;

    // The window alone, then the rest beside the codes, then the whole state.
    const model::QwenCachedBuffers buffers = reservation->buffers();
    const model::GdnParityBuffers &slot = buffers.gdn->buffers();
    fill(slot.stateBase, 72);
    const auto cell = bytesOf(slot.stateBase);
    storage.copyWindow(0, *reservation);
    require(sameBytes(buffers.window->buffer(), images[1]) && sameBytes(slot.stateBase, cell),
            "copyWindow did not copy the lane's window alone");
    storage.copyBeside(0, *reservation);
    const std::vector<uint8_t> beside = bytesOf(slot.stateBase);
    require(std::equal(beside.begin(), beside.begin() + layout.target.convolutionBytes(), cached.begin()) &&
                std::equal(beside.begin() + layout.target.convolutionBytes(), beside.end(),
                           cell.begin() + layout.target.convolutionBytes()),
            "copyBeside did not copy the lane's convolution rows alone into the cell");
    storage.storeState(0, *reservation);
    require(holdsCached(slot.stateBase, cached, layout.target) && sameBytes(buffers.window->buffer(), images[1]),
            "storeState did not store the lane's convolution rows, codes and window");

    // A refused seal leaves the reservation as it was.
    rejects([&] { static_cast<void>(storage.seal(*reservation, {4096, 2048, 2048, 4000})); },
            "a composite snapshot needs the context window of its draft rings",
            "a state was sealed behind its context window");
    std::shared_ptr<const model::QwenCompositeState> state = storage.seal(*reservation, published(4096));
    rejects([&] { static_cast<void>(reservation->buffers()); }, "a sealed state reservation holds no buffers",
            "a sealed reservation still held its buffers");
    rejects([&] { static_cast<void>(storage.seal(*reservation, published(4096))); },
            "composite state buffers are empty", "a reservation was sealed twice");
    reservation.reset();
    require(storage.idleCachedCells() == 0 && storage.idleWindows() == 0,
            "a sealed reservation returned the buffers its state holds");
    require(state->canOffload(false), "a sealed state cannot be written to the disk tier");
    auto write = state->offload({});
    require(write && finishWhenReady(*write), "a sealed state's disk write failed");
    fill(storage.current(0).stateBase, 99);
    fill(storage.window(0), 98);
    auto read = storage.beginRestore(0, *write->state(), true, {}, [] {});
    require(read && finishWhenReady(*read) && stateImage(storage, 0) == images &&
                storage.metadata(0).lengths == published(4096),
            "a sealed state's disk copy did not restore every byte and its lengths");
    read.reset();
    write.reset();
    state.reset();
    require(storage.idleCachedCells() == 1 && storage.idleWindows() == 1 &&
                storage.actualAllocatedBytes() == slotted,
            "a dropped sealed state did not return its slot to the pool once");

    // Another storage's reservation is refused.
    model::QwenStateStorage other(backend, admit, layout, nullptr);
    auto foreign = other.reserve(refusal);
    rejects([&] { static_cast<void>(storage.seal(*foreign, published(4096))); },
            "a state reservation of another storage", "a foreign reservation was sealed");
    outlives = storage.reserve(refusal);
    storage.releaseLane(0, 9);
  }
  // Its storage went first: the reservation frees its buffers.
  outlives.reset();
  require(backend.memoryStats().allocatedBytes == beforeStorage,
          "a reservation that outlived its storage kept its buffers");
}

// A target without GDN layers (attention alone) keeps no GDN state: its
// layout has no GDN dimensions, its cells no bytes, and a lane and a state
// hold the draft's buffers alone. Every lane and state shares one empty cell,
// which Metal allocates nothing for and the pool never holds, so no idle
// cell is counted, a start lacks no cell a state could give and a reclaim
// releases rings and windows. A state copies, writes, reads and restores
// its window alone.
void testAttentionOnlyTarget(metal::MetalBackend &backend) {
  constexpr model::CompositeStateLayout layout{{}, kDraftState};
  static_assert(layout.valid() && layout.target.cellBytes() == 0 &&
                layout.laneBytes() == kDraftState.ringBytes() + kDraftState.windowBytes() &&
                layout.cachedBytes() == kDraftState.windowBytes());
  static_assert(model::GdnStateLayout{.format = ops::GdnStateFormat::BFloat16}.valid() &&
                !model::GdnStateLayout{.convolutionHistory = 3}.valid() &&
                !model::GdnStateLayout{48, 3, 10'240, 48, 128, 0}.valid());
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  uint32_t admissions = 0;
  const metal::AllocationAdmission admit = [admit = governor.allocationAdmission(), &admissions](
                                               uint64_t bytes, const std::function<void()> &allocate) {
    ++admissions;
    return admit(bytes, allocate);
  };
  const uint64_t slotBytes = model::SlotFile::slotBytesFor(layout.cachedBytes());
  const uint64_t beforeStorage = backend.memoryStats().allocatedBytes;
  {
    model::QwenStateStorage storage(
        backend, admit, layout,
        std::make_shared<model::SlotFile>(slotBytes, std::make_shared<model::DiskBudget>(slotBytes)));
    require(storage.actualAllocatedBytes() == 0 && storage.stagingBytes() == layout.cachedBytes(),
            "an attention-only storage allocated a cell or stages more than a window");
    require(static_cast<bool>(storage.tryActivateLane(0, 1)) && admissions == 1 &&
                storage.actualAllocatedBytes() >= layout.laneBytes(),
            "an attention-only lane did not take its rings and window in one admission");
    const model::GdnParityBuffers &current = storage.current(0);
    require(&current == &storage.next(0) && !current.stateBase && current.convolutionLayers.empty() &&
                current.recurrentLayers.empty() && !current.auxiliary,
            "an attention-only lane's cells are not the one empty cell");
    storage.clearForColdStart(0);
    fill(storage.window(0), 31);
    const std::vector<uint8_t> window = bytesOf(storage.window(0));
    storage.updateLengths(0, published(4096));
    storage.swapParity(0);

    // A state takes a window alone, and gives it back alone.
    auto state = storage.snapshot(0);
    require(state && state->bytes() == layout.cachedBytes() && admissions == 2,
            "an attention-only state did not take a window alone");
    require(static_cast<bool>(storage.tryActivateLane(1, 2)), "a second attention-only lane did not start");
    fill(storage.window(1), 32);
    bool committed = false;
    require(!storage.beginRestore(1, *state, true, {}, [&] { committed = true; }) && committed &&
                sameBytes(storage.window(1), window) && storage.metadata(1).lengths == published(4096),
            "an attention-only state did not restore its window and lengths");
    state.reset();
    require(storage.idleCells() == 0 && storage.idleWindows() == 1 && storage.idleRings() == 0,
            "a dropped attention-only state did not return its window alone");
    const uint64_t pooled = storage.actualAllocatedBytes();
    const uint32_t admitted = admissions;
    metal::AllocationFailure refusal;
    auto reservation = storage.reserve(refusal);
    require(reservation && admissions == admitted && storage.idleWindows() == 0 &&
                storage.actualAllocatedBytes() == pooled,
            "an attention-only reservation did not take the pooled window alone");
    storage.storeState(1, *reservation);
    require(sameBytes(reservation->buffers().window->buffer(), window),
            "an attention-only state's copy did not take the lane's window");
    state = storage.seal(*reservation, published(4096));
    reservation.reset();

    // On disk a state is its window.
    auto write = state->offload({});
    require(write && finishWhenReady(*write) && write->state()->bytes() == layout.cachedBytes(),
            "an attention-only state's disk write failed");
    fill(storage.window(0), 33);
    auto read = storage.beginRestore(0, *write->state(), true, {}, [] {});
    require(read && finishWhenReady(*read) && sameBytes(storage.window(0), window),
            "an attention-only state's disk copy did not restore its window");
    read.reset();
    write.reset();
    state.reset();

    // Released lanes leave rings and windows, and no cell, to reclaim.
    storage.releaseLane(0, 1);
    storage.releaseLane(1, 2);
    require(storage.idleCells() == 0 && storage.idleRings() == 2 && storage.idleWindows() == 3 &&
                storage.statesToActivate() == 0 &&
                storage.idleBytes(1, false, true) == kDraftState.ringBytes() + 3 * kDraftState.windowBytes(),
            "released attention-only lanes pooled a cell, lost a buffer or counted a cell's bytes");
    const uint64_t idle = storage.idleBytes(0, false, false);
    require(idle > 0 && releaseAllIdle(storage, false) == idle && storage.actualAllocatedBytes() == 0 &&
                storage.idleRings() == 0 && storage.idleWindows() == 0,
            "an attention-only pool kept buffers past reclaim, or released other than its idle bytes");
  }
  require(backend.memoryStats().allocatedBytes == beforeStorage,
          "an attention-only storage's buffers outlived it");
}

// A lane's start allocates startBytes, the lane's buffers the pool lacks,
// and the states its claims then evict leave idleBytes after the start to
// release: they refill the lane's buffers the start took from the pool
// first. From an empty pool and from pools holding none to two lanes'
// buffers and none to two states' cells and windows, with none to three
// states evicted.
void testStartBytes(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                          queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  for (uint32_t pooledLanes = 0; pooledLanes < 3; ++pooledLanes) {
    for (uint32_t pooledStates = 0; pooledStates < 3; ++pooledStates) {
      for (uint32_t evicted = 0; evicted < 4; ++evicted) {
        model::QwenStateStorage storage(backend, governor.allocationAdmission(), kStateLayout, nullptr);
        metal::AllocationFailure refusal;
        std::vector<std::shared_ptr<model::QwenStateReservation>> cached, pooled;
        for (uint32_t each = 0; each < evicted; ++each)
          cached.push_back(storage.reserve(refusal));
        for (uint32_t lane = 0; lane < pooledLanes; ++lane)
          require(static_cast<bool>(storage.tryActivateLane(lane, lane + 1)), "a pool lane did not start");
        for (uint32_t each = 0; each < pooledStates; ++each)
          pooled.push_back(storage.reserve(refusal));
        for (uint32_t lane = 0; lane < pooledLanes; ++lane)
          storage.releaseLane(lane, lane + 1);
        const bool reserved = std::ranges::all_of(cached, [](const auto &state) { return state != nullptr; }) &&
                              std::ranges::all_of(pooled, [](const auto &state) { return state != nullptr; });
        pooled.clear();
        const std::string name = std::to_string(evicted) + " evicted states beside " + std::to_string(pooledLanes) +
                                 " pooled lanes and " + std::to_string(pooledStates) + " pooled states";
        require(reserved && storage.idleRings() == pooledLanes && storage.idleWindows() == pooledLanes + pooledStates,
                "the fixture of " + name + " did not hold its states and pool");
        const uint64_t charged = storage.startBytes();
        const uint64_t credited = storage.idleBytes(evicted, true, true);
        const uint64_t before = storage.actualAllocatedBytes();
        require(static_cast<bool>(storage.tryActivateLane(0, 9)), "the start of " + name + " failed");
        require(storage.actualAllocatedBytes() == before + charged,
                "the start of " + name + " did not allocate what it was charged");
        cached.clear();
        require(releaseAllIdle(storage, true) == credited,
                "the states evicted after the start of " + name + " did not release what was counted");
        storage.releaseLane(0, 9);
      }
    }
  }
}

void testParkedLane(metal::MetalBackend &backend) {
  const uint64_t backendBefore = backend.memoryStats().allocatedBytes;
  for (const auto format : {ops::GdnStateFormat::Float32, ops::GdnStateFormat::BFloat16}) {
    const model::CompositeStateLayout layout{{2, 3, 16, 1, 8, SPLASH_STATE_CODEC_BLOCK, format, 37}, {1, 1, 8, 128}};
    MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                            queryHostMemory, 0);
    const test::ClaimScope start(governor, ClaimClass::Floor);
    bool allowed = true;
    auto admission = [&](uint64_t bytes, const std::function<void()> &allocate) -> metal::AllocationResult {
      return allowed ? governor.allocationAdmission()(bytes, allocate)
                     : metal::AllocationResult{metal::AllocationFailure::EngineBudget};
    };
    model::QwenStateStorage storage(backend, admission, layout, nullptr);
    require(static_cast<bool>(storage.tryActivateLane(0, 101)), "park fixture activation failed");
    fill(storage.current(0).stateBase, 19);
    fill(storage.next(0).stateBase, 23);
    fill(storage.window(0), 29);
    for (const auto &ring : storage.draft(0)) {
      fill(ring.keys, 31);
      fill(ring.values, 37);
    }
    storage.updateLengths(0, {37, 0, 37, 32});
    constexpr uint32_t pendingPath = 1u | (1u << 3) | (1u << 15);
    storage.swapDecodeParity(0, pendingPath);
    const auto metadata = storage.metadata(0);
    const auto current = bytesOf(storage.current(0).stateBase);
    const auto next = bytesOf(storage.next(0).stateBase);
    const auto window = bytesOf(storage.window(0));
    const auto keys = bytesOf(storage.draft(0).front().keys);
    const auto values = bytesOf(storage.draft(0).front().values);
    const uint64_t before = storage.actualAllocatedBytes();
    metal::AllocationFailure refusal;
    bool saved = false;
    const auto save = [&](const metal::MetalBuffer &scratch) { saved = true; fill(scratch, 41); };
    allowed = false;
    auto parked = storage.parkLane(0, 101, 19, save, refusal);
    require(!parked && !saved && refusal == metal::AllocationFailure::EngineBudget &&
                storage.metadata(0).requestId == 101 && storage.actualAllocatedBytes() == before,
            "refused parking detached a lane, copied scratch or changed its charge");
    allowed = true;
    rejects([&] {
      static_cast<void>(storage.parkLane(0, 101, 19, [](const auto &) {
        throw std::runtime_error("save refused");
      }, refusal));
    }, "save refused", "throwing scratch save was swallowed");
    require(storage.metadata(0).requestId == 101 && storage.actualAllocatedBytes() == before,
            "throwing scratch save lost lane ownership or leaked its charge");
    rejects([&] { static_cast<void>(storage.parkLane(0, 202, 0, {}, refusal)); },
            "owner mismatch", "another request parked the lane");
    rejects([&] {
      static_cast<void>(storage.parkLane(0, 101, std::numeric_limits<uint64_t>::max(), save, refusal));
    }, "overflows", "overflowing parked scratch was accepted");
    parked = storage.parkLane(0, 101, 19, save, refusal);
    require(parked && saved && refusal == metal::AllocationFailure::None &&
                !storage.metadata(0).assigned() && parked->metadata().pendingPath == pendingPath &&
                parked->metadata().activeParity == metadata.activeParity &&
                parked->metadata().lengths == metadata.lengths &&
                storage.actualAllocatedBytes() == before + parked->scratch().allocatedBytes() &&
                storage.idleCells() == 0 && storage.idleRings() == 0 && storage.idleWindows() == 0,
            "parking changed pending state, undercharged memory or pooled held buffers");
    const auto scratch = bytesOf(parked->scratch());
    require(static_cast<bool>(storage.tryActivateLane(0, 202)), "intruder activation failed");
    fill(storage.current(0).stateBase, 43);
    fill(storage.window(0), 47);
    for (const auto &ring : storage.draft(0)) fill(ring.keys, 53);
    rejects([&] { storage.restoreParkedLane(0, 101, parked); }, "destination lane is assigned",
            "restore overwrote an occupied destination");
    model::QwenStateStorage other(backend, admission, layout, nullptr);
    rejects([&] { other.restoreParkedLane(1, 101, parked); }, "another storage",
            "restore accepted a continuation from another storage");
    rejects([&] { storage.restoreParkedLane(1, 202, parked); }, "owner mismatch",
            "restore accepted another request's continuation");
    require(parked && sameBytes(parked->scratch(), scratch), "rejected restore consumed scratch");
    const uint64_t withScratch = storage.actualAllocatedBytes();
    const uint64_t scratchCharge = parked->scratch().allocatedBytes();
    storage.restoreParkedLane(1, 101, parked);
    require(!parked && storage.metadata(1).requestId == 101 &&
                storage.metadata(1).activeParity == metadata.activeParity &&
                storage.metadata(1).pendingPath == metadata.pendingPath &&
                storage.metadata(1).lengths == metadata.lengths &&
                sameBytes(storage.current(1).stateBase, current) &&
                sameBytes(storage.next(1).stateBase, next) && sameBytes(storage.window(1), window) &&
                sameBytes(storage.draft(1).front().keys, keys) &&
                sameBytes(storage.draft(1).front().values, values) &&
                storage.actualAllocatedBytes() == withScratch - scratchCharge,
            "rebinding changed bytes, metadata, parity or the scratch allocation charge");
    rejects([&] { storage.restoreParkedLane(2, 101, parked); }, "another storage",
            "a consumed continuation was restored twice");
    parked = storage.parkLane(1, 101, 0, {}, refusal);
    require(parked && !parked->scratch(), "scratch-free parking allocated scratch");
    parked.reset();
    require(storage.idleCells() == 0 && storage.idleRings() == 0 && storage.idleWindows() == 0 &&
                storage.actualAllocatedBytes() == before,
            "discarding a parked lane retained its state or allocation charge");
    storage.releaseLane(0, 202);
  }
  std::unique_ptr<model::QwenStateStorage::ParkedLane> late;
  {
    const model::CompositeStateLayout layout{{1, 3, 16, 1, 8, SPLASH_STATE_CODEC_BLOCK}, {1, 1, 8, 128}};
    MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                            queryHostMemory, 0);
    const test::ClaimScope start(governor, ClaimClass::Floor);
    model::QwenStateStorage storage(backend, governor.allocationAdmission(), layout, nullptr);
    require(static_cast<bool>(storage.tryActivateLane(0, 909)), "late discard fixture failed");
    metal::AllocationFailure refusal;
    late = storage.parkLane(0, 909, 0, {}, refusal);
  }
  require(late && backend.memoryStats().allocatedBytes > backendBefore,
          "storage destruction lost a parked lane's owned buffers");
  late.reset();
  require(backend.memoryStats().allocatedBytes == backendBefore, "parked lane tests leaked Metal buffers");
  std::cout << "parked_lane=PASS formats=fp32,bf16 pending_rows=3 rebind=1 refusal=1 rollback=1\n";
}

// The pool keeps the buffers the next lane starts from only while no lane
// holds its own: beside a lane in service, a released lane's cells, rings and
// window are idle bytes that keepLane releases, since the lane in service
// returns its own for the next start; with no lane in service, keepLane keeps
// one lane's again. A lane's cells serve no cached state, so kept beside a
// lane in service they would stay past every cached unit a KV admission
// evicts.
void testKeptLaneBesideALaneInService(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1, queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
  model::QwenStateStorage storage(backend, governor.allocationAdmission(), kStateLayout, nullptr);
  require(static_cast<bool>(storage.tryActivateLane(0, 1)), "the first lane did not start");
  const uint64_t lane = storage.actualAllocatedBytes();
  require(static_cast<bool>(storage.tryActivateLane(1, 2)), "the second lane did not start");
  storage.releaseLane(1, 2);
  require(storage.idleBytes(0, false, true) == kStateLayout.laneBytes() &&
              storage.idleBytes(0, false, false) == kStateLayout.laneBytes(),
          "beside a lane in service, the pool kept a released lane's buffers for the next start");
  require(releaseAllIdle(storage, true) == lane && storage.idleCells() == 0 && storage.idleRings() == 0 &&
              storage.idleWindows() == 0 && storage.actualAllocatedBytes() == lane,
          "beside a lane in service, keepLane did not release a released lane's buffers");
  storage.releaseLane(0, 1);
  require(storage.idleBytes(0, false, true) == 0 && releaseAllIdle(storage, true) == 0 && storage.idleCells() == 2 &&
              storage.idleRings() == 1 && storage.idleWindows() == 1,
          "with no lane in service, keepLane did not keep the buffers the next lane starts from");
  require(releaseAllIdle(storage, false) == lane && storage.actualAllocatedBytes() == 0,
          "the kept lane's buffers were not reclaimed");
}

// Captures, demotions, direct writes, RAM and SSD restores, promotions and
// drops interleaved on four lanes. Every restore reproduces the codes of the
// state it came from; at every step the storage tracks the lanes', the
// resident states' and the idle pool's bytes and nothing else; idleBytes()
// predicts what dropping the states leaves idle, releasing the pool frees
// exactly that, and every buffer returns once lanes and states are gone. Over
// a running state in fp32 and in bf16, and a cell whose auxiliary state no
// page divides.
void testAccountingAcrossCapturesAndRestores(metal::MetalBackend &backend) {
  constexpr model::GdnStateLayout kBf16State{48, 3, 10'240, 48, 128, 128, ops::GdnStateFormat::BFloat16};
  constexpr model::GdnStateLayout kAuxiliaryState{36, 3, 8192, 32, 128, 128, ops::GdnStateFormat::Float32,
                                                  2'100'003};
  for (const model::GdnStateLayout &target : {kTargetState, kBf16State, kAuxiliaryState}) {
    const model::CompositeStateLayout layout{target, kDraftState};
    const uint64_t slot = model::SlotFile::slotBytesFor(layout.cachedBytes());
    MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1, queryHostMemory, 0);
    const test::ClaimScope start(governor, ClaimClass::Floor);
    const uint64_t backendBefore = backend.memoryStats().allocatedBytes;
    {
      model::QwenStateStorage storage(
          backend, governor.allocationAdmission(), layout,
          std::make_shared<model::SlotFile>(slot, std::make_shared<model::DiskBudget>(12 * slot)));
      for (uint32_t lane = 0; lane < 4; ++lane)
        require(static_cast<bool>(storage.tryActivateLane(lane, 100 + lane)), "a lane's activation failed");
      const uint64_t lanes = storage.actualAllocatedBytes();
      require(lanes == 4 * layout.laneBytes(), "four lanes did not take four lanes' bytes");
      struct Kept {
        std::shared_ptr<const CompositeState> state;
        std::vector<std::vector<uint8_t>> image; // what restoring it reproduces
      };
      std::deque<Kept> ram, disk;
      uint32_t direct = 0, demotions = 0, ramRestores = 0, diskRestores = 0, promotions = 0;
      // 36 steps take every path and fill both lists past their bounds.
      for (uint64_t step = 0; step < 36; ++step) {
        const uint32_t lane = static_cast<uint32_t>(step % 4);
        fillState(storage.current(lane), target, step + 1);
        fill(storage.window(lane), step + 1000);
        storage.updateLengths(lane, published(4096 + 64 * step));
        const std::vector<std::vector<uint8_t>> image = restoredImage(storage, lane);
        if (step % 7 == 6) {
          // Straight from the lane to the SSD.
          auto write = storage.snapshotToDisk(lane, {});
          require(write && finishWhenReady(*write), "a direct disk write failed");
          disk.push_back({write->state(), image});
          ++direct;
        } else if (auto state = storage.snapshot(lane)) {
          ram.push_back({state, image});
        }
        if (step % 3 == 2 && !ram.empty() && ram.front().state->canOffload(false)) {
          auto write = ram.front().state->offload({});
          require(write && finishWhenReady(*write), "a demotion failed");
          disk.push_back({write->state(), ram.front().image});
          ram.pop_front();
          ++demotions;
        }
        if (step % 4 == 3 && !ram.empty()) {
          const uint32_t into = (lane + 1) % 4;
          const Kept &kept = ram[step % ram.size()];
          require(!storage.beginRestore(into, *kept.state, true, {}, [] {}), "a RAM restore returned a read");
          require(stateImage(storage, into) == kept.image, "a RAM restore did not reproduce its state's codes");
          ++ramRestores;
        }
        if (step % 5 == 4 && !disk.empty()) {
          const uint32_t into = (lane + 2) % 4;
          const Kept kept = disk[step % disk.size()];
          auto read = storage.beginRestore(into, *kept.state, step % 2 == 0, {}, [] {});
          require(read && finishWhenReady(*read), "an SSD restore failed");
          require(stateImage(storage, into) == kept.image, "an SSD restore did not reproduce its state's codes");
          ++diskRestores;
          if (auto promoted = read->snapshot()) {
            const uint32_t other = (into + 1) % 4;
            require(!storage.beginRestore(other, *promoted, true, {}, [] {}), "a promoted restore returned a read");
            require(stateImage(storage, other) == kept.image, "a promoted copy did not restore the disk copy's codes");
            ram.push_back({promoted, kept.image});
            ++promotions;
          }
        }
        while (ram.size() > 6)
          ram.pop_front();
        while (disk.size() > 8)
          disk.pop_front();
        uint64_t resident = 0;
        for (const Kept &kept : ram)
          resident += kept.state->residentBytes();
        require(storage.actualAllocatedBytes() == lanes + resident + storage.idleBytes(0, false, false),
                "step " + std::to_string(step) + ": the storage tracked bytes beside the lanes, states and pool");
      }
      require(direct && demotions && ramRestores && diskRestores && promotions,
              "the sequence left a capture or restore path untried");
      const uint64_t predicted = storage.idleBytes(static_cast<uint32_t>(ram.size()), false, false);
      ram.clear();
      disk.clear();
      require(storage.idleBytes(0, false, false) == predicted,
              "idleBytes() did not predict what dropping the states left idle");
      require(releaseAllIdle(storage, false) == predicted && storage.actualAllocatedBytes() == lanes,
              "releasing the pool did not free what idleBytes() predicted");
      for (uint32_t lane = 0; lane < 4; ++lane)
        storage.releaseLane(lane, 100 + lane);
      require(releaseAllIdle(storage, false) == lanes && storage.actualAllocatedBytes() == 0 &&
                  storage.idleCells() == 0 && storage.idleCachedCells() == 0 && storage.idleRings() == 0 &&
                  storage.idleWindows() == 0,
              "the lanes' buffers did not all return");
    }
    require(backend.memoryStats().allocatedBytes == backendBefore, "the storage left Metal allocations behind");
  }

}

void run(const std::string &metallib) {
  using model::QwenCompositeState;
  using model::QwenLogicalLengths;
  using model::QwenStateStorage;

  testLayoutFormulas();
  metal::MetalBackend backend(metallib);
  testParkedLane(backend);
  testOffloadAllocationFailure(backend);
  testDiskRestore(backend);
  testExactCachedStatesRestoreEveryByte(backend);
  testDirectDiskSnapshot(backend);
  testStateWritesDrawOnTheWriteBudget(backend);
  testStateSmallerThanSlot(backend);
  testPersistentStateComesBack(backend);
  testPersistKeepsTheRamCopy(backend);
  testReservations(backend);
  testAttentionOnlyTarget(backend);
  testKeptLaneBesideALaneInService(backend);
  testAccountingAcrossCapturesAndRestores(backend);
  testStartBytes(backend);
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
      queryHostMemory, 0);
  const test::ClaimScope start(governor, ClaimClass::Floor);
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
    markBlock(storage.next(0).recurrentLayers[0], 0x31);
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

    // Publication stores the lane's active-parity GDN cell, its recurrent
    // layers as codes, and its context window into a cache slot the governor
    // admits. The lane keeps its own cells: no address changes, no aliasing,
    // no parity handoff.
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
    markBlock(storage.current(0).recurrentLayers[0], 0xa2);
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
                marked(storage.current(1).recurrentLayers[0], 0x31),
            "restore did not deliver the pre-mutation GDN snapshot");
    require(word(storage.next(1).convolutionLayers[0]) == 0xb1b1b1b1,
            "restore overwrote inactive parity");
    require(word(storage.window(1)) == 0x41414141 &&
                word(storage.window(1), windowTail) == 0x51515151,
            "restore did not deliver the pre-mutation context window");
    require(word(storage.draft(1)[0].keys) == 0xb2b2b2b2,
            "a restore wrote the rings, which the runtime computes from the window");
    require(word(storage.current(0).convolutionLayers[0]) == 0xa1a1a1a1 &&
                marked(storage.current(0).recurrentLayers[0], 0xa2) &&
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
    // A lane whose request found a row that is not finite clears every
    // layer's ring keys and values before its release.
    const model::DFlashDraftRingLayer &lastRing = storage.draft(0).back();
    const uint64_t ringTail = lastRing.values.sizeBytes() - sizeof(uint32_t);
    word(storage.draft(0)[0].keys) = 0xe1e1e1e1;
    word(storage.draft(0)[0].values) = 0xe2e2e2e2;
    word(lastRing.keys) = 0xe3e3e3e3;
    word(lastRing.values, ringTail) = 0xe4e4e4e4;
    storage.clearDraftRings(0);
    require(word(storage.draft(0)[0].keys) == 0 && word(storage.draft(0)[0].values) == 0 &&
                word(lastRing.keys) == 0 && word(lastRing.values, ringTail) == 0,
            "the lane's draft rings were not cleared");

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
    // A path holds its anchor's row and rows of a block.
    for (const uint32_t path : {0U, 0b110U, test::chainPath(model::ExecutionLimits::targetVerifyRows + 1)})
      rejects([&] { storage.swapDecodeParity(0, path); }, "a decode step accepts a path of its rows from its anchor's",
              "a decode step accepted no rows, a path without its anchor or rows past a block's");
    storage.swapDecodeParity(0, 0b1011);
    require(storage.metadata(0).pendingPath == 0b1011 && storage.metadata(0).activeParity != parity,
            "a decode step's swap lost its accepted path");
    rejects([&] { static_cast<void>(storage.snapshot(0)); }, "Qwen lane state has decoded rows pending",
            "a snapshot read a cell without its retained rows");
    rejects([&] { static_cast<void>(storage.snapshotToDisk(0, {})); }, "Qwen lane state has decoded rows pending",
            "a disk snapshot read a cell without its retained rows");
    rejects([&] { static_cast<void>(storage.beginRestore(0, *prefix, true, {}, [] {})); },
            "Qwen lane state has decoded rows pending", "a restore overwrote a cell with pending rows");
    rejects([&] { storage.swapParity(0); }, "Qwen lane state has decoded rows pending",
            "a prefill step followed a decode step's pending rows");
    storage.swapDecodeParity(0, test::chainPath(model::ExecutionLimits::targetVerifyRows));
    require(storage.metadata(0).pendingPath == test::chainPath(model::ExecutionLimits::targetVerifyRows) &&
                storage.metadata(0).activeParity == parity,
            "the next decode step's swap did not take its own accepted path");
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
                !storage.metadata(0).pendingPath,
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
    markBlock(storage.current(0).recurrentLayers[0], 0xe2);
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
                marked(storage.current(1).recurrentLayers[0], 0xe2) &&
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
    // What releasing down to one lane's worth frees once the cached state is
    // dropped, counted before at the layout's sizes: a lane's buffers and the
    // state's cell and window.
    const uint64_t idle = storage.idleBytes(1, false, true);
    require(idle == kStateLayout.laneBytes() + kStateLayout.cachedBytes(),
            "the idle bytes a dropped state would leave are not the layout's");
    retained.reset();
    require(storage.idleCells() == 4 && storage.idleCachedCells() == 1 && storage.idleRings() == 2 &&
                storage.idleWindows() == 3 && storage.idleBytes(0, false, true) == idle,
            "dropped cached state did not return its cell and window to the pool");
    // Releasing down to one lane's worth keeps two cells, the rings and a
    // window warm, and frees no less than was counted.
    const uint64_t released = releaseAllIdle(storage, true);
    require(released == observedLaneActual + observedPrefixActual && released >= idle &&
                storage.idleCells() == 2 && storage.idleCachedCells() == 0 && storage.idleRings() == 1 &&
                storage.idleWindows() == 1 && storage.idleBytes(0, false, true) == 0,
            "partial idle release did not keep the requested buffers");
    require(releaseAllIdle(storage, false) == observedLaneActual,
            "idle lane buffers were not reclaimed");
    require(storage.idleCells() == 0 && storage.idleRings() == 0 &&
                storage.idleWindows() == 0,
            "reclaimed buffers remain pooled");
    // No cached state holds rings: without idle rings no number of states
    // makes up a lane.
    require(storage.actualAllocatedBytes() == 0 &&
                !storage.statesToActivate(),
            "reclaimed state cells remain accounted");

    // An activation asks the governor once for everything the pool lacks:
    // a refusal allocates nothing and leaves the pool as it was.
    admitNewAllocations = false;
    require(!storage.tryActivateLane(0, 505) && !storage.metadata(0).assigned() &&
                storage.actualAllocatedBytes() == 0 &&
                !storage.statesToActivate(),
            "a denied activation allocated part of its lane");
    admitNewAllocations = true;
    uint32_t admitted = admissions;
    require(static_cast<bool>(storage.tryActivateLane(0, 505)) &&
                admissions == admitted + 1 &&
                storage.actualAllocatedBytes() == observedLaneActual,
            "an activation asked the governor for its buffers one by one");
    // With one cell, the rings and a window in the pool the lane lacks a
    // cell, which no cached state holds: a refusal leaves them pooled, and
    // the retry is admitted that cell alone.
    storage.releaseLane(0, 505);
    require(storage.releaseOneIdle(false) != 0 && storage.idleCells() == 1 &&
                storage.idleRings() == 1 && storage.idleWindows() == 1 &&
                !storage.statesToActivate(),
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
  if (argc != 2 && !(argc == 3 && std::string_view(argv[2]) == "--park-only")) {
    std::cerr << "usage: qwen_state_storage_test METALLIB [--park-only]\n";
    return EXIT_FAILURE;
  }
  @autoreleasepool {
    try {
      if (argc == 3) {
        metal::MetalBackend backend(argv[1]);
        testParkedLane(backend);
      } else {
        run(argv[1]);
      }
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "qwen state storage test failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
