#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"
#include "tests/engine/AllocationFailure.hpp"

#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace splash;
using namespace splash::engine;

namespace {

splash::DraftContextPlan restorePlan(const splash::RestoreState &state,
                                     bool draft) {
  return splash::planDraftContext(
      state.boundary,
      state.boundary +
          (draft ? 0 : splash::model::ExecutionLimits::draftContextTokens),
      state.boundary, {},
      state.windowBegin(splash::model::kDraftWindowGroup,
                        splash::model::ExecutionLimits::draftContextTokens));
}

constexpr model::ModelStateLayout layout{{1, 3, 128, 1, 128, 128},
                                         {2, 1, 2048, 128}};
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

template <class Exception = std::exception, class Function>
void requireThrows(Function function, const char *message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}
template <class Ticket> bool finishWhenReady(Ticket &ticket) {
  while (!ticket.ready())
    std::this_thread::yield();
  return ticket.finish();
}
using Image = std::vector<std::vector<std::byte>>;
Image stateImage(const model::QwenSlotBuffers &buffers, uint32_t parity = 0) {
  Image result;
  auto append = [&](const metal::MetalBuffer &buffer) {
    const auto *begin = static_cast<const std::byte *>(buffer.contents());
    result.emplace_back(begin, begin + buffer.sizeBytes());
  };
  append(buffers.gdn[parity].stateBase);
  for (const auto &layer : buffers.draft) {
    for (const auto &page : layer.keyPages)
      append(page);
    for (const auto &page : layer.valuePages)
      append(page);
  }
  return result;
}
void fill(model::QwenStateStorage &storage, uint32_t slot, uint8_t value) {
  require(static_cast<bool>(storage.prepareDraftWrite(slot, 0, 2048)),
          "cannot prepare test pages");
  const auto &buffers = storage.buffers(slot);
  std::memset(
      buffers.gdn[storage.metadata(slot).activeParity].stateBase.contents(),
      value, layout.target.cellBytes());
  for (const auto &layer : buffers.draft) {
    for (const auto &page : layer.keyPages)
      std::memset(page.contents(), value++, page.sizeBytes());
    for (const auto &page : layer.valuePages)
      std::memset(page.contents(), value++, page.sizeBytes());
  }
}
void testLayoutFormulas() {
  constexpr model::GdnStateLayout kTargetState{48, 3, 10'240, 48, 128, 128};
  constexpr model::DraftStateLayout kDraftState{5, 8, 2'048, 128};
  constexpr model::ModelStateLayout kStateLayout{kTargetState, kDraftState};
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
  require(kDraftState.windowBytes() == 41'943'040,
          "draft state formula is wrong");
  require(kStateLayout.activeCellBytes() == 350'224'384,
          "per-slot byte formula is wrong");
  require(uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                  kStateLayout.activeCellBytes() ==
              1'400'897'536,
          "four-slot byte formula is wrong");
  require(kStateLayout.cachedBytes() == 196'083'712,
          "prefix byte formula is wrong");
}

void testSlotLifecycle(metal::MetalBackend &backend) {
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  uint32_t admissions = 0, maximumAdmissions = UINT32_MAX;
  auto admit =
      [&](uint64_t bytes,
          const std::function<void()> &allocate) -> metal::AllocationResult {
    if (admissions++ >= maximumAdmissions)
      return metal::AllocationFailure::EngineBudget;
    return governor.allocationAdmission()(bytes, allocate);
  };
  model::QwenStateStorage storage(backend, admit, layout);
  requireThrows<std::invalid_argument>(
      [&] { static_cast<void>(storage.tryActivateSlot(0, 0)); },
      "zero owner accepted");
  require(storage.tryActivateSlot(0, 1) && storage.tryActivateSlot(1, 2),
          "lifecycle activation failed");
  requireThrows<std::logic_error>(
      [&] { static_cast<void>(storage.tryActivateSlot(0, 3)); },
      "assigned lane reactivated");
  storage.swapParity(0);
  fill(storage, 0, 17);
  storage.updateLengths(0, {2048, 0, 2048, 0});
  const auto original = stateImage(storage.buffers(0), 1);
  auto snapshot = storage.snapshot(0);
  storage.swapParity(1);
  std::memset(storage.buffers(1).gdn[0].stateBase.contents(), 91,
              layout.target.cellBytes());
  storage.restore(1, *snapshot, true);
  require(stateImage(storage.buffers(1), 1) == original &&
              *static_cast<uint8_t *>(
                  storage.buffers(1).gdn[0].stateBase.contents()) == 91,
          "restore changed inactive GDN parity or missed active parity");
  // Fail after the first shared page was copied. The partially prepared table
  // must remain equivalent and retry without touching the snapshot.
  static_cast<void>(storage.releaseIdle(0, 0));
  maximumAdmissions = admissions + 1;
  const uint64_t beforeFailedCow = storage.actualAllocatedBytes();
  require(!storage.prepareDraftWrite(1, 2048, 2304),
          "partial COW fixture did not refuse");
  require(storage.actualAllocatedBytes() == beforeFailedCow,
          "failed COW retained new idle pages");
  require(stateImage(storage.buffers(1), 1) == original,
          "partial COW changed state");
  maximumAdmissions = UINT32_MAX;
  require(static_cast<bool>(storage.prepareDraftWrite(1, 2048, 2304)),
          "partial COW could not retry");
  fill(storage, 1, 38);
  storage.restore(1, *snapshot, false);
  require(storage.metadata(1).lengths.targetTokens == 2048 &&
              storage.metadata(1).lengths.draftLength == 0 &&
              storage.metadata(1).lengths.draftBase == 2048,
          "GDN-only restore exposed stale draft metadata");
  requireThrows<std::logic_error>([&] { storage.releaseSlot(0, 9); },
                                  "wrong owner released lane");
  storage.releaseSlot(0, 1);
  storage.releaseSlot(1, 2);
  static_cast<void>(storage.releaseIdle(0, 0));
  require(storage.actualAllocatedBytes() == snapshot->bytes(),
          "reclaim freed snapshot backing");
  requireThrows<std::logic_error>([&] { storage.swapParity(0); },
                                  "unassigned parity changed");
  require(static_cast<bool>(storage.tryActivateSlot(0, 3)),
          "reactivation failed");
  require(storage.metadata(0).lengths == model::QwenLogicalLengths{} &&
              *static_cast<uint8_t *>(
                  storage.buffers(0).gdn[0].stateBase.contents()) == 0,
          "reactivation retained logical or GDN state");
  storage.restore(0, *snapshot, true);
  require(stateImage(storage.buffers(0)) == original,
          "snapshot did not survive lane reclaim");
  const uint64_t beforeInvalid = storage.actualAllocatedBytes();
  storage.updateLengths(0, {128, 0, 127, 127});
  requireThrows<std::invalid_argument>(
      [&] { static_cast<void>(storage.snapshot(0)); },
      "divergent lengths accepted");
  require(storage.actualAllocatedBytes() == beforeInvalid,
          "invalid snapshot allocated backing");
  storage.releaseSlot(0, 3);
  snapshot.reset();
  static_cast<void>(storage.releaseIdle(
      2, layout.draft.tokens / model::DraftStateLayout::blockTokens));
  require(storage.idleCells() == 2 &&
              storage.idlePages() ==
                  layout.draft.tokens / model::DraftStateLayout::blockTokens,
          "warm backing retention changed");
  require(storage.reclaimIdle() == layout.target.cellBytes() &&
              storage.reclaimIdle() == layout.target.cellBytes(),
          "incremental reclaim did not release idle GDN cells");
  require(storage.reclaimIdle() == layout.draft.blockBytes() &&
              storage.idlePages() == SPLASH_DRAFT_PAGE_COUNT - 1 &&
              storage.reclaimIdle() == layout.draft.blockBytes(),
          "incremental reclaim stranded a partial draft window");
  static_cast<void>(storage.releaseIdle(0, 0));
  require(!storage.actualAllocatedBytes(), "lifecycle leaked backing");
}

void testCowCanDropOnlyTheSharedPage(metal::MetalBackend &backend) {
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  bool paused = false;
  auto admission = [&](uint64_t bytes, const std::function<void()> &allocate) {
    return paused
               ? metal::AllocationResult{metal::AllocationFailure::HostPressure}
               : governor.allocationAdmission()(bytes, allocate);
  };
  model::QwenStateStorage storage(backend, admission, layout);
  require(static_cast<bool>(storage.tryActivateSlot(0, 1)),
          "COW fixture activation failed");
  fill(storage, 0, 1);
  storage.updateLengths(0, {2048, 0, 2048, 0});
  auto snapshot = storage.snapshot(0);
  auto retained = std::make_shared<RestoreState>(*snapshot);
  snapshot.reset();
  const auto before = storage.actualAllocatedBytes();
  paused = true;
  require(!storage.prepareDraftWrite(0, 2048, 2080),
          "shared page ignored pressure");
  const auto page = std::find_if(
      retained->blocks.begin(), retained->blocks.end(), [](const auto &part) {
        return part.group == model::kDraftWindowGroup &&
               part.end == model::DraftStateLayout::blockTokens;
      });
  require(page != retained->blocks.end(),
          "COW fixture has no first draft page");
  retained->blocks.erase(page);
  require(storage.prepareDraftWrite(0, 2048, 2080) &&
              storage.actualAllocatedBytes() == before &&
              retained->blocks.front().group == model::kQwenRecurrentGroup,
          "dropping a shared draft page did not permit an in-place write");
  retained.reset();
  storage.releaseSlot(0, 1);
}

void testPages(metal::MetalBackend &backend) {
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  bool allow = true;
  auto admission =
      [&](uint64_t bytes,
          const std::function<void()> &allocate) -> metal::AllocationResult {
    if (!allow)
      return metal::AllocationFailure::EngineBudget;
    return governor.allocationAdmission()(bytes, allocate);
  };
  auto file = std::make_shared<model::SlotFile>(layout.target.cellBytes(),
                                                8 * layout.cachedBytes());
  model::QwenStateStorage storage(backend, admission, layout, file);
  require(!storage.actualAllocatedBytes(), "state storage allocated eagerly");
  require(storage.tryActivateSlot(0, 1) && storage.tryActivateSlot(1, 2),
          "activation failed");
  require(storage.actualAllocatedBytes() == 2 * layout.activeCellBytes(),
          "active page accounting is wrong");
  fill(storage, 0, 1);
  storage.updateLengths(0, {2048, 0, 2048, 0});
  const auto original = stateImage(storage.buffers(0));
  const auto before = storage.actualAllocatedBytes();
  auto first = storage.snapshot(0);
  require(first && storage.actualAllocatedBytes() - before ==
                       layout.target.cellBytes(),
          "snapshot copied draft pages instead of sharing them");
  require(first->blocks.front().payload->reclaimableBytes() ==
              layout.target.cellBytes(),
          "active draft pages counted as reclaimable");
  auto write = storage.prepareSnapshotToDisk(0).write({});
  require(write && finishWhenReady(*write), "initial offload failed");
  auto disk = write->state();
  write.reset();
  const auto beforeCow = storage.actualAllocatedBytes();
  allow = false;
  require(static_cast<bool>(storage.prepareDraftWrite(0, 2049, 2049)),
          "empty write range attempted COW");
  require(!storage.prepareDraftWrite(0, 2048, 2080), "COW bypassed admission");
  require(stateImage(storage.buffers(0)) == original,
          "denied COW changed active state");
  allow = true;
  require(static_cast<bool>(storage.prepareDraftWrite(0, 2048, 2080)),
          "COW admission failed");
  require(storage.actualAllocatedBytes() - beforeCow ==
              layout.draft.blockBytes(),
          "COW copied more than one page");
  for (const auto &layer : storage.buffers(0).draft) {
    std::memset(layer.keyPages[0].contents(), 99,
                32 * layout.draft.headDimension * 2);
    for (uint32_t channel = 0; channel < layout.draft.headDimension; ++channel)
      std::memset(static_cast<std::byte *>(layer.valuePages[0].contents()) +
                      channel * model::DraftStateLayout::blockTokens * 2,
                  99, 32 * 2);
  }
  storage.updateLengths(0, {2080, 32, 2048, 32});
  const auto nextImage = stateImage(storage.buffers(0));
  auto next = storage.snapshot(0);
  storage.restore(1, *first, true);
  require(stateImage(storage.buffers(1)) == original,
          "COW mutated a cached prefix");
  require(storage.buffers(1).draft[0].keyPages[0].sameView(
              storage.buffers(0).draft[0].keyPages[0]) == false,
          "changed page still aliases the old prefix");
  require(storage.buffers(1).draft[0].keyPages[1].sameView(
              storage.buffers(0).draft[0].keyPages[1]),
          "restoring a branch copied an unchanged page");
  const uint64_t written = file->writtenBytes();
  write = storage.prepareSnapshotToDisk(0).write({});
  require(write && finishWhenReady(*write), "overlap offload failed");
  auto nextDisk = write->state();
  write.reset();
  require(file->writtenBytes() - written ==
              layout.target.cellBytes() + layout.draft.blockBytes(),
          "overlap offload rewrote unchanged pages");
  const auto diskPages = [](const RestoreState &state) {
    std::array<const void *, SPLASH_DRAFT_PAGE_COUNT> pages;
    for (const auto &part : state.blocks)
      if (part.group == model::kDraftWindowGroup)
        pages[(part.begin / model::DraftStateLayout::blockTokens) %
              SPLASH_DRAFT_PAGE_COUNT] =
            part.payload->resources().front().identity;
    return pages;
  };
  auto a = diskPages(*disk), b = diskPages(*nextDisk);
  require(a[0] != b[0], "wrong disk page table");
  for (size_t i = 1; i < SPLASH_DRAFT_PAGE_COUNT; ++i)
    require(a[i] == b[i], "disk pages not shared");

  // The branch has its own mutable tail; rejected/uncommitted positions are
  // not written by this fixture, just as the context commit kernel behaves.
  fill(storage, 1, 71);
  storage.restore(0, *next, true);
  require(stateImage(storage.buffers(0)) == nextImage,
          "branch write polluted another prefix");
  const uint64_t readBytes = file->readBytes();
  auto restore = storage.beginRestore(1, *nextDisk,
                                      restorePlan(*nextDisk, false), {}, [] {});
  require(finishWhenReady(*restore) && !restore->snapshot(),
          "GDN-only restore promoted missing draft data");
  require(file->readBytes() - readBytes == layout.target.cellBytes(),
          "GDN-only restore read draft pages");
  restore.reset();
  restore = storage.beginRestore(1, *nextDisk, restorePlan(*nextDisk, true), {},
                                 [] {});
  require(finishWhenReady(*restore) &&
              stateImage(storage.buffers(1)) == nextImage,
          "disk page table restore changed state bytes");
  auto promoted = restore->snapshot();
  require(bool(promoted),
          "complete restore did not produce a promotable snapshot");
  const uint64_t beforeCopy = file->writtenBytes();
  for (const auto &part : promoted->blocks) {
    if (part.group != model::kDraftWindowGroup)
      continue;
    auto copy = part.payload->offload({});
    require(copy && finishWhenReady(*copy), "promoted page lost its disk copy");
  }
  require(file->writtenBytes() == beforeCopy,
          "promoted pages were written again");
  restore.reset();
  auto invalid = std::make_shared<RestoreState>(*next);
  invalid->blocks[1].begin += 1;
  requireThrows<std::invalid_argument>(
      [&] { storage.restore(1, *invalid, true); },
      "partial draft page accepted");
  invalid.reset();
  const auto beforeDeniedRestore = stateImage(storage.buffers(1));
  const auto beforeDeniedReads = file->readBytes();
  allow = false;
  requireThrows<metal::MetalAllocationError>(
      [&] {
        static_cast<void>(storage.beginRestore(
            1, *nextDisk, restorePlan(*nextDisk, true), {}, [] {}));
      },
      "disk restore bypassed allocation admission");
  require(file->readBytes() == beforeDeniedReads &&
              stateImage(storage.buffers(1)) == beforeDeniedRestore &&
              backend.healthy(),
          "denied restore started IO, committed state, or poisoned Metal");
  allow = true;
  restore = storage.beginRestore(1, *nextDisk, restorePlan(*nextDisk, true), {},
                                 [] {});
  require(restore && finishWhenReady(*restore) &&
              stateImage(storage.buffers(1)) == nextImage,
          "aligned disk pages did not restore correctly");
  restore.reset();
  // Reuse cannot inherit the previous request's draft bytes or page identity.
  storage.releaseSlot(1, 2);
  require(static_cast<bool>(storage.tryActivateSlot(1, 3)),
          "reactivation failed");
  for (const auto &layer : storage.buffers(1).draft)
    for (const auto &page : layer.keyPages) {
      const auto *bytes = static_cast<const std::byte *>(page.contents());
      require(std::all_of(bytes, bytes + page.sizeBytes(),
                          [](auto b) { return b == std::byte{}; }),
              "reactivated lane exposed another request's draft context");
    }
  storage.updateLengths(1, {32, 0, 32, 32});
  auto shortState = storage.snapshot(1);
  require(shortState->bytes() ==
              layout.target.cellBytes() + layout.draft.blockBytes(),
          "short snapshot retained the entire window");
  requireThrows<std::invalid_argument>(
      [&] {
        storage.updateLengths(1, {33, 0, 33, 33});
        static_cast<void>(storage.snapshot(1));
      },
      "unaligned snapshot accepted");
  storage.releaseSlot(0, 1);
  storage.releaseSlot(1, 3);
  first.reset();
  next.reset();
  promoted.reset();
  shortState.reset();
  static_cast<void>(storage.releaseIdle(0, 0));
  require(storage.actualAllocatedBytes() == 0,
          "released pages remained allocated");
}

void testDemandAwareDiskRestore(metal::MetalBackend &backend) {
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  auto file = std::make_shared<model::SlotFile>(layout.target.cellBytes(),
                                                layout.cachedBytes());
  model::QwenStateStorage storage(backend, governor.allocationAdmission(),
                                  layout, file);
  require(bool(storage.tryActivateSlot(0, 1)), "source activation failed");
  fill(storage, 0, 17);
  storage.updateLengths(0, {4096, 2048, 2048, 0});
  auto write = storage.prepareSnapshotToDisk(0).write({});
  require(write && finishWhenReady(*write),
          "partial-restore fixture write failed");
  auto disk = write->state();
  write.reset();
  for (bool oldPages : {false, true}) {
    require(bool(storage.tryActivateSlot(1, 2)),
            "restore lane activation failed");
    RestoreState available = *disk;
    std::erase_if(available.blocks, [oldPages](const auto &part) {
      return part.group == model::kDraftWindowGroup &&
             (!oldPages || part.end <= 3072);
    });
    const std::array<uint32_t, 1> optional{5120};
    const auto plan =
        planDraftContext(4096, 8192, 4096, optional,
                         available.windowBegin(model::kDraftWindowGroup, 2048));
    const uint64_t before = file->readBytes();
    auto restore = storage.beginRestore(1, available, plan, {}, [] {});
    require(restore && finishWhenReady(*restore),
            "demand-aware restore failed");
    require(file->readBytes() - before ==
                layout.target.cellBytes() +
                    (oldPages ? 32 * layout.draft.blockBytes() : 0),
            "restore read pages outside the shared window plan");
    require(storage.metadata(1).lengths.draftLength ==
                    (oldPages ? 1024u : 0u) &&
                !restore->snapshot(),
            "partial window was treated as a complete cache snapshot");
    for (size_t layer = 0; layer < storage.buffers(1).draft.size(); ++layer)
      for (uint32_t page = 0; page < SPLASH_DRAFT_PAGE_COUNT; ++page) {
        const auto &actual = storage.buffers(1).draft[layer].keyPages[page];
        if (oldPages && page >= 32) {
          const auto &expected = storage.buffers(0).draft[layer].keyPages[page];
          require(std::memcmp(actual.contents(), expected.contents(),
                              actual.sizeBytes()) == 0,
                  "partial restore loaded the wrong physical page");
        } else {
          const auto *data = static_cast<const std::byte *>(actual.contents());
          require(std::all_of(data, data + actual.sizeBytes(),
                              [](std::byte b) { return b == std::byte{}; }),
                  "unloaded draft page was not zero initialized");
        }
      }
    restore.reset();
    storage.releaseSlot(1, 2);
  }
  storage.releaseSlot(0, 1);
}

void testDirectDiskAndCancellation(metal::MetalBackend &backend) {
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  auto file = std::make_shared<model::SlotFile>(layout.target.cellBytes(),
                                                layout.cachedBytes());
  model::QwenStateStorage storage(backend, governor.allocationAdmission(),
                                  layout, file);
  require(static_cast<bool>(storage.tryActivateSlot(0, 1)),
          "direct lane activation failed");
  fill(storage, 0, 11);
  storage.updateLengths(0, {4096, 2048, 2048, 0});
  const auto original = stateImage(storage.buffers(0));
  const auto allocated = storage.actualAllocatedBytes();
  const auto used = file->usedBytes();
  auto plan = storage.prepareSnapshotToDisk(0);
  require(
      plan.source && plan.source->boundary == 4096 &&
          plan.source->bytes() == layout.cachedBytes() &&
          file->usedBytes() == used &&
          storage.actualAllocatedBytes() == allocated,
      "preparing direct snapshot allocated payload or misreported its cost");
  auto write = plan.write({});
  plan = {};
  require(write && storage.actualAllocatedBytes() == allocated,
          "direct write allocated GPU pages");
  auto disk = write->state();
  fill(storage, 0, 22);
  requireThrows<std::logic_error>(
      [&] { static_cast<void>(storage.prepareSnapshotToDisk(0).write({})); },
      "concurrent staging write admitted");
  require(finishWhenReady(*write), "direct write failed");
  write.reset();
  require(!storage.prepareSnapshotToDisk(0).write({}),
          "full quota admitted another checkpoint");
  bool committed = false;
  auto read = storage.beginRestore(0, *disk, restorePlan(*disk, true), {},
                                   [&] { committed = true; });
  read->cancel();
  // Cancellation may race completed IO; either way destruction drains it.
  read.reset();
  require(!committed, "cancelled restore committed without finish");
  read = storage.beginRestore(0, *disk, restorePlan(*disk, true), {},
                              [&] { committed = true; });
  require(finishWhenReady(*read) && committed &&
              stateImage(storage.buffers(0)) == original,
          "direct snapshot or cancelled read changed immutable payload");
  read.reset();
  disk.reset();
  require(file->usedBytes() == 0, "disk page handles leaked quota");
  storage.releaseSlot(0, 1);
}

void testActivationRollback(metal::MetalBackend &backend) {
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  uint32_t pageAllocations = 0;
  bool limited = false;
  auto admit =
      [&](uint64_t bytes,
          const std::function<void()> &allocate) -> metal::AllocationResult {
    if (limited && bytes == layout.draft.blockBytes() && pageAllocations++ >= 3)
      return metal::AllocationFailure::EngineBudget;
    return governor.allocationAdmission()(bytes, allocate);
  };
  model::QwenStateStorage storage(backend, admit, layout);
  for (bool warm : {false, true}) {
    limited = false;
    if (warm) {
      require(static_cast<bool>(storage.tryActivateSlot(0, 1)),
              "warm rollback setup failed");
      storage.releaseSlot(0, 1);
      static_cast<void>(storage.releaseIdle(2, 0));
    }
    const uint64_t before = storage.actualAllocatedBytes();
    limited = true;
    for (uint32_t attempt = 0; attempt < 32; ++attempt) {
      pageAllocations = 0;
      require(!storage.tryActivateSlot(0, 1),
              "partial draft activation unexpectedly fit");
      require(!storage.metadata(0).assigned &&
                  storage.actualSlotBytes(0) == 0 &&
                  storage.actualAllocatedBytes() == before,
              "failed activation retained new pages and created false reclaim "
              "progress");
    }
    static_cast<void>(storage.releaseIdle(0, 0));
  }
}

void testCowAllocationFailure(metal::MetalBackend &backend) {
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  model::QwenStateStorage storage(backend, governor.allocationAdmission(),
                                  layout);
  require(storage.tryActivateSlot(0, 1) && storage.tryActivateSlot(1, 2),
          "COW fault activation failed");
  fill(storage, 0, 7);
  storage.updateLengths(0, {2048, 0, 2048, 0});
  auto source = storage.snapshot(0);
  const auto original = stateImage(storage.buffers(0));
  for (int failure = 0; failure < 1024; ++failure) {
    storage.restore(1, *source, true);
    bool failed = false;
    allocationFailureAfter = failure;
    try {
      failed = !storage.prepareDraftWrite(1, 0, 256);
    } catch (const std::bad_alloc &) {
      failed = true;
    }
    allocationFailureAfter = -1;
    require(stateImage(storage.buffers(1)) == original,
            "COW preparation changed bytes before dispatch");
    require(static_cast<bool>(storage.prepareDraftWrite(1, 0, 256)),
            "COW fault retry failed");
    // Write only already-prepared pages: another COW call must not accidentally
    // repair a stale table and hide a failed preparation's aliasing bug.
    for (const auto &layer : storage.buffers(1).draft)
      for (uint32_t page = 0; page < 256 / model::DraftStateLayout::blockTokens;
           ++page) {
        std::memset(layer.keyPages[page].contents(), 91,
                    layer.keyPages[page].sizeBytes());
        std::memset(layer.valuePages[page].contents(), 92,
                    layer.valuePages[page].sizeBytes());
      }
    require(stateImage(storage.buffers(0)) == original,
            "COW failure or retry mutated a shared snapshot");
    if (!failed) {
      storage.releaseSlot(0, 1);
      storage.releaseSlot(1, 2);
      return;
    }
  }
  throw std::runtime_error(
      "COW allocation failure sweep never reached success");
}

void testOffloadAllocationFailure(metal::MetalBackend &backend) {
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  constexpr model::ModelStateLayout layout{{1, 3, 128, 1, 128, 128},
                                           {1, 1, 2048, 4}};
  auto file = std::make_shared<model::SlotFile>(layout.target.cellBytes(),
                                                3 * layout.cachedBytes());
  model::QwenStateStorage storage(backend, governor.allocationAdmission(), layout, file);
  require(static_cast<bool>(storage.tryActivateSlot(0, 1)), "fault source activation failed");
  storage.updateLengths(0, {4096, 2048, 2048, 0});
  auto source = storage.snapshot(0);
  auto held = file->acquire();
  std::vector<std::byte> bytes(layout.target.cellBytes());
  struct Result {
    bool failed;
    std::unique_ptr<StateWrite<RestoreState>> transfer;
  };
  for (int failure = 0; failure < 1024; ++failure) {
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
        auto transfer = storage.prepareSnapshotToDisk(0).write({});
        allocationFailureAfter = -1;
        return Result{false, std::move(transfer)};
      } catch (const std::bad_alloc &) {
        allocationFailureAfter = -1;
        return Result{true, {}};
      }
    });
    const bool returned = attempt.wait_for(std::chrono::milliseconds(1)) ==
                          std::future_status::ready;
    const bool pending = !file->idle();
    release.set_value();
    auto result = attempt.get();
    while (!file->idle()) std::this_thread::yield();
    require(!(result.failed && returned && pending),
            "allocation failure released staging before the submitted write "
            "drained");
    if (!result.failed) {
      require(result.transfer && result.transfer->finish(),
              "offload did not recover after allocation failures");
      return;
    }
    require(file->usedBytes() == layout.target.cellBytes(),
            "failed offload leaked its disk quota");
  }
  throw std::runtime_error("offload allocation failure sweep never reached success");
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 2)
    return EXIT_FAILURE;
  @autoreleasepool {
    try {
      metal::MetalBackend backend(argv[1]);
      const auto before = backend.memoryStats().allocatedBytes;
      testLayoutFormulas();
      testSlotLifecycle(backend);
      testPages(backend);
      testCowCanDropOnlyTheSharedPage(backend);
      testDemandAwareDiskRestore(backend);
      testDirectDiskAndCancellation(backend);
      testActivationRollback(backend);
      testCowAllocationFailure(backend);
      testOffloadAllocationFailure(backend);
      require(backend.memoryStats().allocatedBytes == before,
              "state storage leaked Metal allocations");
      std::cout << "Paged draft state storage tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "state storage test failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
