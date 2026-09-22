#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"
#include "ops/PageStorage.hpp"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr model::GdnStateLayout kTargetState{48, 3, 10'240, 48, 128, 128};
constexpr model::DraftStateLayout kDraftState{5, 8, 2'048, 128};
constexpr model::CompositeStateLayout kStateLayout{kTargetState, kDraftState};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Exception = std::exception>
void requireThrows(const std::function<void()> &operation,
                   const char *message) {
  try {
    operation();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

uint32_t &word(const metal::MetalBuffer &buffer, uint64_t byteOffset = 0) {
  require(byteOffset + sizeof(uint32_t) <= buffer.sizeBytes(),
          "test marker is outside buffer");
  auto *bytes = static_cast<uint8_t *>(buffer.contents());
  require(bytes != nullptr, "test buffer is not CPU-visible");
  return *reinterpret_cast<uint32_t *>(bytes + byteOffset);
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
  require(kStateLayout.activeCellBytes() == 350'224'384,
          "per-slot byte formula is wrong");
  require(uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                  kStateLayout.activeCellBytes() ==
              1'400'897'536,
          "four-slot byte formula is wrong");
  require(kStateLayout.cachedBytes() == 196'083'712,
          "prefix byte formula is wrong");
}

void run(const std::string &metallib) {
  using model::QwenCompositeState;
  using model::QwenLogicalLengths;
  using model::QwenStateStorage;

  testLayoutFormulas();
  metal::MetalBackend backend(metallib);
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  // Switched off to prove that a pooled cache slot needs no new admission.
  bool admitNewAllocations = true;
  auto admitState = [&governor, &admitNewAllocations](
                        uint64_t bytes,
                        const std::function<void()> &allocate) {
    if (!admitNewAllocations)
      return false;
    auto reservation = governor.tryReserve(bytes);
    if (!reservation)
      return false;
    allocate();
    reservation->commit();
    return true;
  };
  constexpr kv::Layout kvLayout{16, 4, 256};
  kv::PageStorage pageStorage(backend, governor.allocationAdmission(),
                                kvLayout,
                                kvLayout.sparseMappingBatchPages());
  uint64_t beforeStorage = backend.memoryStats().allocatedBytes;
  uint64_t observedStorageActual = 0;
  uint64_t observedSlotActual = 0;
  uint64_t observedPrefixActual = 0;

  {
    QwenStateStorage storage(backend, admitState, kStateLayout);
    observedStorageActual = storage.actualAllocatedBytes();
    observedSlotActual = storage.actualSlotBytes(0);
    require(storage.actualAllocatedBytes() == 0 &&
                backend.memoryStats().allocatedBytes == beforeStorage,
            "state cells were allocated eagerly");
    for (uint32_t slot = 0;
         slot < model::ExecutionLimits::maximumBatchWidth;
         ++slot) {
      require(!storage.metadata(slot).assigned, "slot was assigned eagerly");
      require(storage.actualSlotBytes(slot) == 0,
              "idle slot has physical backing before activation");
    }
    requireThrows<std::out_of_range>(
        [&] { static_cast<void>(storage.buffers(4)); },
        "storage exposed more than four slots");

    require(storage.tryActivateSlot(0, 101) && storage.tryActivateSlot(1, 202),
            "state cell activation failed");
    observedStorageActual = storage.actualAllocatedBytes();
    observedSlotActual = storage.actualSlotBytes(0);
    require(observedSlotActual >= kStateLayout.activeCellBytes(),
            "activated slot allocation is below declared bytes");
    require(storage.actualSlotBytes(1) == observedSlotActual &&
                storage.actualAllocatedBytes() == 2 * observedSlotActual,
            "activated slot accounting is not incremental");

    const auto &slot0 = storage.buffers(0);
    const auto &slot1 = storage.buffers(1);
    void *stableGdnBase = slot0.gdn[0].stateBase.contents();
    void *stableConvBase = slot0.gdn[0].convolutionBase.contents();
    void *stableDraftBase = slot0.draft[0].keys.contents();
    require(stableGdnBase && stableConvBase && stableDraftBase,
            "stable slot buffers are not CPU-visible");
    require(stableGdnBase != slot1.gdn[0].stateBase.contents(),
            "two slots alias one GDN allocation");
    require(slot0.gdn[0].convolutionLayers[1].contents() ==
                static_cast<uint8_t *>(stableConvBase) +
                    kTargetState.convolutionLayerBytes(),
            "GDN layer view offset is wrong");
    require(slot0.gdn[0].stateBase.sizeBytes() ==
                    kTargetState.cellBytes() &&
                slot0.draft.size() == kDraftState.layers &&
                slot0.draft[0].keys.sizeBytes() == kDraftState.tensorBytes(),
            "split state-buffer sizes are wrong");

    require(storage.metadata(0).assigned &&
                storage.metadata(0).requestId == 101 &&
                storage.metadata(0).activeParity == 0,
            "slot activation metadata is wrong");
    requireThrows<std::logic_error>(
        [&] { static_cast<void>(storage.tryActivateSlot(0, 303)); },
        "double slot activation was accepted");

    // Hot metadata updates must leave every buffer and marker untouched.
    word(slot0.gdn[0].convolutionBase) = 0x10101010;
    word(slot0.gdn[1].convolutionBase) = 0x21212121;
    word(slot0.gdn[1].recurrentBase) = 0x31313131;
    word(slot0.draft[0].keys) = 0x41414141;
    word(slot0.draft[4].values,
         kDraftState.tensorBytes() - sizeof(uint32_t)) = 0x51515151;
    QwenLogicalLengths lengths{2'048, 0, 2'048, 0};
    storage.updateLengths(0, lengths);
    require(storage.metadata(0).lengths.draftCommitCursor == 0,
            "draft ring cursor is wrong");
    require(storage.metadata(0).lengths.draftLength == 2'048,
            "draft resident length is wrong");
    require(word(slot0.gdn[1].convolutionBase) == 0x21212121 &&
                word(slot0.draft[0].keys) == 0x41414141,
            "length update copied or cleared hot state");
    storage.swapParity(0);
    require(storage.metadata(0).activeParity == 1,
            "parity swap did not select parity one");
    require(word(slot0.gdn[0].convolutionBase) == 0x10101010 &&
                word(slot0.gdn[1].convolutionBase) == 0x21212121,
            "parity swap copied hot state");

    // Publication copies the lane's active-parity GDN cell and its draft
    // ring into a cache slot the governor admits. The lane keeps its own
    // cells: no address changes, no aliasing, no parity handoff.
    const uint64_t beforePrefix = backend.memoryStats().allocatedBytes;
    const uint64_t storageBeforePrefix = storage.actualAllocatedBytes();
    void *laneActiveGdnBase = slot0.gdn[1].stateBase.contents();
    std::shared_ptr<const QwenCompositeState> prefix = storage.snapshot(0);
    require(prefix != nullptr, "snapshot could not obtain a cache slot");
    observedPrefixActual = backend.memoryStats().allocatedBytes - beforePrefix;
    require(prefix->bytes() == kStateLayout.cachedBytes(),
            "cached state footprint is not the declared cached bytes");
    require(observedPrefixActual >= kStateLayout.cachedBytes() &&
                storage.actualAllocatedBytes() ==
                    storageBeforePrefix + observedPrefixActual,
            "cache slot allocation is below declared bytes or unaccounted");
    require(slot0.gdn[1].stateBase.contents() == laneActiveGdnBase &&
                slot0.gdn[0].stateBase.contents() == stableGdnBase &&
                slot0.draft[0].keys.contents() == stableDraftBase,
            "snapshot moved or aliased the lane's own cells");
    require(storage.metadata(0).activeParity == 1 &&
                storage.metadata(0).lengths == lengths,
            "snapshot changed the lane's metadata");

    // The cached copy is independent of the lane: writes to the lane's
    // active cell or draft ring after publication never reach a restore.
    word(slot0.gdn[1].convolutionBase) = 0xa1a1a1a1;
    word(slot0.gdn[1].recurrentBase) = 0xa2a2a2a2;
    word(slot0.draft[0].keys) = 0xa3a3a3a3;
    word(slot1.gdn[0].convolutionBase) = 0xb0b0b0b0;
    word(slot1.gdn[1].convolutionBase) = 0xb1b1b1b1;
    word(slot1.draft[0].keys) = 0xb2b2b2b2;
    void *destinationGdnBase = slot1.gdn[0].stateBase.contents();
    void *destinationDraftBase = slot1.draft[0].keys.contents();
    storage.restore(1, *prefix, true);

    require(storage.metadata(1).requestId == 202 &&
                storage.metadata(1).activeParity == 0 &&
                storage.metadata(1).lengths == lengths,
            "restore lost owner, changed parity, or lengths");
    require(slot1.gdn[0].stateBase.contents() == destinationGdnBase &&
                slot1.draft[0].keys.contents() == destinationDraftBase,
            "restore replaced the destination's buffers");
    require(word(slot1.gdn[0].convolutionBase) == 0x21212121 &&
                word(slot1.gdn[0].recurrentBase) == 0x31313131,
            "restore did not deliver the pre-mutation GDN snapshot");
    require(word(slot1.gdn[1].convolutionBase) == 0xb1b1b1b1,
            "restore overwrote inactive parity");
    require(
        word(slot1.draft[0].keys) == 0x41414141 &&
            word(slot1.draft[4].values, kDraftState.tensorBytes() -
                                            sizeof(uint32_t)) == 0x51515151,
        "restore did not deliver the pre-mutation draft ring snapshot");
    require(word(slot0.gdn[1].convolutionBase) == 0xa1a1a1a1 &&
                word(slot0.gdn[1].recurrentBase) == 0xa2a2a2a2 &&
                word(slot0.draft[0].keys) == 0xa3a3a3a3,
            "restore wrote back into the source lane");

    // A suffix that will rebuild a full 2048-token window restores only GDN.
    // A cached draft ring must not consume a 40 MiB copy merely to be
    // overwritten by the next prefill commands.
    storage.swapParity(1);
    word(slot1.draft[0].keys) = 0xd2d2d2d2;
    storage.restore(1, *prefix, false);
    require(word(slot1.gdn[storage.metadata(1).activeParity].convolutionBase) ==
                0x21212121,
            "GDN-only restore did not restore convolution state");
    require(word(slot1.draft[0].keys) == 0xd2d2d2d2,
            "GDN-only restore copied an obsolete draft ring");
    require(storage.metadata(1).lengths.targetTokens == 2048 &&
                storage.metadata(1).lengths.draftLength == 0 &&
                storage.metadata(1).lengths.draftBase == 2048,
            "GDN-only restore exposed stale draft metadata");
    // Cancellation releases ownership and returns the lane's buffers to the
    // pool. Reactivation takes them back in the same order and initializes
    // every state that can be read at logical length zero.
    void *reusableGdnBase = slot0.gdn[0].stateBase.contents();
    void *reusableDraftBase = slot0.draft[0].keys.contents();
    word(slot0.gdn[0].convolutionBase) = 0xc1c1c1c1;
    word(slot0.gdn[0].recurrentBase) = 0xc2c2c2c2;
    storage.releaseSlot(0, 101);
    require(!storage.metadata(0).assigned && storage.metadata(0).requestId == 0,
            "cancellation did not release metadata");
    require(storage.idleCells() == 2 && storage.idleRings() == 1 &&
                storage.actualSlotBytes(0) == 0,
            "released lane buffers did not return to the pool");
    requireThrows<std::logic_error>([&] { storage.swapParity(0); },
                                    "unassigned slot accepted a parity update");
    require(static_cast<bool>(storage.tryActivateSlot(0, 303)), "state cell reuse failed");
    require(storage.idleCells() == 0 && storage.idleRings() == 0,
            "reactivation left pooled buffers behind");
    require(slot0.gdn[0].stateBase.contents() == reusableGdnBase &&
                slot0.draft[0].keys.contents() == reusableDraftBase,
            "slot reuse changed stable buffer addresses");
    require(storage.metadata(0).requestId == 303 &&
                storage.metadata(0).activeParity == 0 &&
                storage.metadata(0).lengths == QwenLogicalLengths{},
            "slot reuse did not reset logical state");
    require(word(slot0.gdn[0].convolutionBase) == 0 &&
                word(slot0.gdn[0].recurrentBase) == 0,
            "slot reuse did not initialize readable state");

    // Rejected publications fail before any cache slot is taken or admitted.
    const uint64_t beforeRejected = storage.actualAllocatedBytes();
    storage.updateLengths(0, {128, 0, 127, 127});
    requireThrows<std::invalid_argument>(
        [&] { static_cast<void>(storage.snapshot(0)); },
        "snapshot accepted divergent target/draft lengths");
    requireThrows<std::invalid_argument>(
        [&] {
          storage.updateLengths(0, {129, 0, 129, 129});
          static_cast<void>(storage.snapshot(0));
        },
        "unaligned prefix snapshot was accepted");
    require(storage.actualAllocatedBytes() == beforeRejected,
            "rejected snapshot allocated or consumed a cache slot");
    requireThrows<std::logic_error>([&] { storage.releaseSlot(0, 404); },
                                    "slot release accepted the wrong owner");

    const uint64_t beforeSuspend = storage.actualAllocatedBytes();
    const uint64_t releasedSlotBytes = storage.actualSlotBytes(0);
    storage.releaseSlot(0, 303);
    require(storage.releaseIdle(0, 0) == releasedSlotBytes,
            "recomputation preemption retained active backing");
    require(!storage.metadata(0).assigned && storage.actualSlotBytes(0) == 0 &&
                storage.actualAllocatedBytes() == beforeSuspend - releasedSlotBytes,
            "preempted GDN or draft bytes remain outside the cache");
    require(storage.tryActivateSlot(0, 303) &&
                storage.metadata(0).assigned &&
                storage.metadata(0).lengths == QwenLogicalLengths{} &&
                word(slot0.gdn[0].convolutionBase) == 0 &&
                word(slot0.gdn[0].recurrentBase) == 0,
            "recomputation did not start from a fresh empty state");

    // Dropping a cached state returns its buffers to the storage's pool rather
    // than freeing them: accounting stays flat, and the next publication takes
    // the pooled buffers without a governor admission. Only releaseIdle
    // returns pooled bytes to macOS.
    const uint64_t beforeDrop = storage.actualAllocatedBytes();
    const uint64_t backendBeforeDrop = backend.memoryStats().allocatedBytes;
    prefix.reset();
    require(storage.actualAllocatedBytes() == beforeDrop &&
                backend.memoryStats().allocatedBytes == backendBeforeDrop,
            "dropped cached state freed its slot instead of pooling it");
    storage.updateLengths(0, lengths);
    word(slot0.gdn[0].convolutionBase) = 0xe1e1e1e1;
    word(slot0.gdn[0].recurrentBase) = 0xe2e2e2e2;
    word(slot0.draft[0].keys) = 0xe3e3e3e3;
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
    storage.restore(1, *pooled, true);
    require(storage.metadata(1).activeParity == 1 &&
                storage.metadata(1).lengths == lengths &&
                word(slot1.gdn[1].convolutionBase) == 0xe1e1e1e1 &&
                word(slot1.gdn[1].recurrentBase) == 0xe2e2e2e2 &&
                word(slot1.draft[0].keys) == 0xe3e3e3e3,
            "reused cache slot served stale contents");
    require(word(slot1.gdn[0].convolutionBase) == 0x21212121,
            "restore from the reused slot overwrote inactive parity");
    pooled.reset();
    require(storage.actualAllocatedBytes() == beforeDrop,
            "second dropped cached state was freed instead of pooled");
    require(storage.releaseIdle(0, 0) == observedPrefixActual &&
                storage.actualAllocatedBytes() ==
                    beforeDrop - observedPrefixActual,
            "releaseIdle did not free the pooled cache slot");
    require(storage.metadata(0).assigned && storage.metadata(1).assigned &&
                storage.actualSlotBytes(0) == observedSlotActual &&
                storage.actualSlotBytes(1) == observedSlotActual,
            "pool reclaim touched active lane cells");

    // A live cached state survives its lane's release and reclaim; only its
    // drop plus a later reclaim frees the slot together with idle cells.
    std::shared_ptr<const QwenCompositeState> retained = storage.snapshot(1);
    require(retained != nullptr, "retained snapshot could not admit a slot");
    require(storage.actualAllocatedBytes() ==
                2 * observedSlotActual + observedPrefixActual,
            "retained snapshot accounting is wrong");
    storage.releaseSlot(0, 303);
    storage.releaseSlot(1, 202);
    require(storage.idleCells() == 4 && storage.idleRings() == 2,
            "released lane buffers are missing from the pool");
    retained.reset();
    require(storage.idleCells() == 5 && storage.idleRings() == 3,
            "dropped cached state did not return its buffers to the pool");
    // Releasing down to one lane's worth keeps two cells and one ring warm.
    require(storage.releaseIdle(2, 1) ==
                observedSlotActual + observedPrefixActual &&
                storage.idleCells() == 2 && storage.idleRings() == 1,
            "partial idle release did not keep the requested buffers");
    require(storage.releaseIdle(0, 0) == observedSlotActual,
            "idle lane buffers were not reclaimed");
    require(storage.idleCells() == 0 && storage.idleRings() == 0,
            "reclaimed buffers remain pooled");
    require(storage.actualAllocatedBytes() == 0,
            "reclaimed state cells remain accounted");
  }
  require(backend.memoryStats().allocatedBytes == beforeStorage,
          "destroyed state slots remained in actual allocation count");

  std::cout << "qwen state storage tests passed: slot_declared="
            << kStateLayout.activeCellBytes()
            << " slot_actual=" << observedSlotActual << " four_slots_declared="
            << uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                   kStateLayout.activeCellBytes()
            << " four_slots_actual=" << observedStorageActual
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
