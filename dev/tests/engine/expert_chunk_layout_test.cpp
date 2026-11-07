// How an expert store's size lays its slots out in chunks (model::
// streamedExpertChunkLayout and wholeExpertChunkLayout), at two layouts: one
// whose planes each fill whole regions, and one whose slots fill them only
// eight at a time. Every layout takes at most MOE_EXPERT_MAX_CHUNKS chunks of
// at most the largest buffer, every chunk but the last of a multiple of the
// slots that fill whole regions, so the chunks take the bytes of one chunk of
// all the slots. A streamed size's floor is the floor asked for,
// kExpertFloorShare of its slots by default, rounded down to such a multiple
// unless it is every slot, at least its minimum and at most its slots; its floor chunk holds
// it, or the largest buffer's slots and the elastic chunks after it the rest,
// all of which hold the floor; its elastic chunks take about the elastic bytes
// asked for
// (kExpertElasticChunkBytes by default), or more where the slots above the
// floor would otherwise take more than MOE_EXPERT_MAX_CHUNKS - 1 chunks, and
// a largest buffer that holds the size only in more than
// MOE_EXPERT_MAX_CHUNKS chunks is refused, the smallest that holds it in at
// most that many is not. A size of one block is its floor chunk alone,
// refused when the largest buffer does not hold it.
#include "TestChecks.hpp"
#include "model/ExpertChunkLayout.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

using splash::model::alignWeightOffset;
using splash::model::ExpertChunkLayout;
using splash::model::ExpertSlotPlanes;
using splash::model::kExpertElasticChunkBytes;
using splash::model::kExpertFloorShare;
using splash::model::kWeightFileAlignment;
using splash::model::streamedExpertChunkLayout;
using splash::model::wholeExpertChunkLayout;
using splash::test::rejects;
using splash::test::require;

namespace {

// The 35B's Q4_K experts, 512 x 2048 gate and up and 2048 x 512 down: every
// plane fills whole regions.
constexpr ExpertSlotPlanes kWhole{{{524288, 0, 65536}, {524288, 0, 65536}, {524288, 0, 65536}}};
// Q5_K gate and up of 640 x 256 and a Q8_0 down of 256 x 640: plane1 fills
// whole regions four slots at a time and meta eight at a time.
constexpr ExpertSlotPlanes kEighths{{{81920, 20480, 10240}, {81920, 20480, 10240}, {163840, 0, 10240}}};
constexpr std::pair<ExpertSlotPlanes, const char *> kLayouts[] = {{kWhole, "whole"}, {kEighths, "eighths"}};

// The fewest slots whose every plane fills whole regions.
uint64_t alignedSlotsOf(const ExpertSlotPlanes &planes) {
  for (uint64_t slots = 1;; ++slots) {
    bool whole = true;
    for (const auto &projection : planes)
      for (const uint64_t plane : projection) whole = whole && slots * plane % kWeightFileAlignment == 0;
    if (whole) return slots;
  }
}

uint64_t alignDown(uint64_t value, uint64_t alignment) { return value / alignment * alignment; }
uint64_t alignUp(uint64_t value, uint64_t alignment) { return alignDown(value + alignment - 1, alignment); }

// The bytes of a chunk of `slots` slots: each plane in a region of its own.
uint64_t chunkBytes(const ExpertSlotPlanes &planes, uint64_t slots) {
  uint64_t bytes = 0;
  for (const auto &projection : planes)
    for (const uint64_t plane : projection) bytes += alignWeightOffset(slots * plane);
  return bytes;
}

// A layout of `slots` slots, which must take 1 to MOE_EXPERT_MAX_CHUNKS
// chunks, each of at most `largest` bytes and of at least one slot, the floor
// first and then elastic chunks of chunkSlots slots, the last of the rest;
// every chunk but the last of a multiple of the aligned slots, together of
// the bytes of one chunk of all the slots.
ExpertChunkLayout checked(const ExpertChunkLayout &layout, const ExpertSlotPlanes &planes, uint32_t slots,
                          uint64_t largest, const std::string &what) {
  require(layout.slots == slots && layout.floorSlots > 0 && layout.chunkSlots > 0 && layout.chunks >= 1 &&
              layout.chunks <= MOE_EXPERT_MAX_CHUNKS && layout.floorChunks >= 1 &&
              layout.floorChunks <= layout.chunks,
          what + ": " + std::to_string(layout.chunks) + " chunks, " + std::to_string(layout.floorChunks) +
              " of the floor, or a floor or elastic chunks of no slots");
  require(layout.chunks > 1 || (layout.floorSlots == slots && layout.chunkSlots == slots),
          what + ": a size of one chunk has other floor or chunk slots than its own");
  const uint64_t group = alignedSlotsOf(planes);
  uint64_t held = 0, bytes = 0;
  for (uint32_t chunk = 0; chunk < layout.chunks; ++chunk) {
    const uint64_t slotsOfChunk = layout.held(chunk);
    require(slotsOfChunk > 0 && slotsOfChunk <= (chunk ? layout.chunkSlots : layout.floorSlots),
            what + ": chunk " + std::to_string(chunk) + " holds no slots or more than its share");
    require(chunk + 1 == layout.chunks || slotsOfChunk % group == 0,
            what + ": chunk " + std::to_string(chunk) + " of " + std::to_string(slotsOfChunk) +
                " slots does not fill whole regions");
    require(chunkBytes(planes, slotsOfChunk) <= largest,
            what + ": chunk " + std::to_string(chunk) + " takes more than the largest buffer");
    require(layout.first(chunk) == held, what + ": chunk " + std::to_string(chunk) + " starts elsewhere");
    held += slotsOfChunk;
    bytes += chunkBytes(planes, slotsOfChunk);
  }
  require(layout.first(layout.chunks) == slots, what + ": the chunks end elsewhere than at the slots");
  require(held == slots, what + ": the chunks hold " + std::to_string(held) + " slots");
  require(bytes == chunkBytes(planes, slots), what + ": the chunks take other bytes than one chunk of their slots");
  return layout;
}

// The default floor: kExpertFloorShare of the slots.
uint32_t floorOf(uint32_t slots) { return static_cast<uint32_t>(kExpertFloorShare * slots); }

ExpertChunkLayout streamed(const ExpertSlotPlanes &planes, uint32_t slots, uint32_t minimum, uint64_t largest,
                           const std::string &what, uint64_t elasticBytes = kExpertElasticChunkBytes) {
  return checked(
      streamedExpertChunkLayout(planes, slots, floorOf(slots), minimum, largest, elasticBytes, kExpertFloorShare),
      planes, slots, largest, what);
}

// A large size's floor is the floor share of its slots rounded down to whole
// regions, the rest in elastic chunks of about kExpertElasticChunkBytes, the
// last of fewer: the 35B's experts in about 13.5 GiB, and the eighths in
// about 12 GiB.
void floorShareAndElasticChunks() {
  for (const auto &[planes, name] : kLayouts) {
    const uint32_t slots = planes == kWhole ? 8193 : 32771;
    const uint64_t group = alignedSlotsOf(planes);
    const ExpertChunkLayout layout = streamed(planes, slots, 512, UINT64_MAX, name);
    require(layout.floorSlots == alignDown(static_cast<uint64_t>(kExpertFloorShare * slots), group) &&
                layout.floorChunks == 1,
            std::string(name) + ": the floor is not the floor share rounded down to whole regions in one chunk");
    require(chunkBytes(planes, layout.chunkSlots) <= kExpertElasticChunkBytes &&
                chunkBytes(planes, layout.chunkSlots + group) > kExpertElasticChunkBytes && layout.chunks >= 3 &&
                (slots - layout.floorSlots) % layout.chunkSlots,
            std::string(name) + ": the elastic chunks take other bytes than about the target, or the last is whole");
  }
}

// Elastic chunks take about the bytes asked for, beside a floor of the floor
// share: asked for the bytes of a sixteenth of its slots, a size takes a
// floor of the floor share and three elastic chunks of a sixteenth of its
// slots each, the last of fewer.
void elasticChunkBytes() {
  for (const auto &[planes, name] : kLayouts) {
    const uint32_t slots = 4099;
    const uint64_t group = alignedSlotsOf(planes);
    const ExpertChunkLayout layout =
        streamed(planes, slots, 512, UINT64_MAX, name, chunkBytes(planes, alignUp(slots / 16, group)));
    require(layout.floorSlots == alignDown(static_cast<uint64_t>(kExpertFloorShare * slots), group) &&
                layout.chunkSlots == alignUp(slots / 16, group) && layout.chunks == 4 &&
                (slots - layout.floorSlots) % layout.chunkSlots,
            std::string(name) + ": elastic chunks of a sixteenth of the slots are not what they hold, or the last "
                                "is whole");
  }
}

// A floor holds at least the size's minimum, rounded up to whole regions,
// and at most the slots; a size of fewer slots than fill whole regions is
// one chunk of them.
void floorMinimum() {
  for (const auto &[planes, name] : kLayouts) {
    const uint64_t group = alignedSlotsOf(planes);
    const ExpertChunkLayout raised = streamed(planes, 4096, 3999, UINT64_MAX, name);
    require(raised.floorSlots == alignUp(3999, group) && raised.chunks == 2 &&
                raised.chunkSlots == 4096 - raised.floorSlots,
            std::string(name) + ": a floor below the minimum, or the rest not in one elastic chunk");
    const ExpertChunkLayout all = streamed(planes, 515, 515, UINT64_MAX, name);
    require(all.chunks == 1, std::string(name) + ": a size of its minimum's slots takes elastic chunks");
  }
  require(streamed(kEighths, 3, 1, UINT64_MAX, "3 slots").chunks == 1 &&
              streamed(kWhole, 16, 16, UINT64_MAX, "16 slots").chunks == 1,
          "a size of fewer slots than a floor's group or minimum takes more than one chunk");
}

// A floor chunk and elastic chunks take at most the largest buffer, and as
// many whole regions as it holds: under a largest buffer below the floor
// share and kExpertElasticChunkBytes, the floor chunk and each elastic chunk
// but the last are what it holds, the floor chunk also below the size's
// minimum, and the fewest elastic chunks after it that hold the rest of the
// floor hold it too.
void floorCap() {
  for (const auto &[planes, name] : kLayouts) {
    const uint32_t slots = 4096;
    const uint64_t group = alignedSlotsOf(planes);
    const uint64_t largest = chunkBytes(planes, 300) + 1;
    const ExpertChunkLayout capped = streamed(planes, slots, 2048, largest, name);
    const uint64_t floor = alignDown(floorOf(slots), group);
    require(capped.floorSlots == alignDown(300, group) && capped.chunkSlots == capped.floorSlots,
            std::string(name) + ": a floor or elastic chunk under a largest buffer takes less than it holds");
    require(capped.first(capped.floorChunks) >= floor && capped.first(capped.floorChunks - 1) < floor,
            std::string(name) + ": " + std::to_string(capped.floorChunks) +
                " chunks are not the fewest that hold the floor");
  }
}

// A floor asked for is held as the default is: rounded down to whole
// regions, the elastic chunks above it; a floor of every slot is one chunk,
// and one below a block's experts is raised to them.
void floorsAskedFor() {
  for (const auto &[planes, name] : kLayouts) {
    const uint32_t slots = 4099;
    const uint64_t group = alignedSlotsOf(planes);
    const std::string what = std::string(name) + ", a floor of half the slots";
    const ExpertChunkLayout half = checked(
        streamedExpertChunkLayout(planes, slots, slots / 2, 512, UINT64_MAX, chunkBytes(planes, alignUp(256, group)),
                                  kExpertFloorShare),
        planes, slots, UINT64_MAX, what);
    require(half.floorSlots == alignDown(slots / 2, group) && half.floorChunks == 1 &&
                half.chunkSlots == alignUp(256, group) &&
                half.chunks == 1 + (slots - half.floorSlots + half.chunkSlots - 1) / half.chunkSlots,
            what + ": the floor or the elastic chunks above it are not what was asked for");
    const ExpertChunkLayout all = checked(
        streamedExpertChunkLayout(planes, slots, slots, 512, UINT64_MAX, kExpertElasticChunkBytes, kExpertFloorShare),
        planes,
        slots, UINT64_MAX, std::string(name) + ", a floor of every slot");
    require(all.chunks == 1 && all.floorChunks == 1,
            std::string(name) + ": a floor of every slot takes elastic chunks");
    const ExpertChunkLayout raised = checked(
        streamedExpertChunkLayout(planes, slots, 100, 512, UINT64_MAX, kExpertElasticChunkBytes, kExpertFloorShare),
        planes,
        slots, UINT64_MAX, std::string(name) + ", a floor below a block's experts");
    require(raised.floorSlots == alignUp(512, group) && raised.floorChunks == 1,
            std::string(name) + ": a floor below a block's experts is not raised to them");
  }
}

// The slots above a floor that chunks of about kExpertElasticChunkBytes hold
// only in more than 47 take larger chunks, 47 at most: the 35B's experts in
// about 330 GiB.
void largerElasticChunks() {
  const uint32_t slots = 200000;
  const ExpertChunkLayout layout = streamed(kWhole, slots, 512, UINT64_MAX, "330 GiB");
  const uint64_t rest = slots - layout.floorSlots;
  require(chunkBytes(kWhole, layout.chunkSlots) > kExpertElasticChunkBytes &&
              layout.chunkSlots == (rest + MOE_EXPERT_MAX_CHUNKS - 2) / (MOE_EXPERT_MAX_CHUNKS - 1) &&
              layout.chunks == MOE_EXPERT_MAX_CHUNKS,
          "330 GiB: the elastic chunks are not the fewest slots that take 47 chunks");
}

// The floor share of the model's plan sets the default floor: a share of 1.0
// keeps every slot in the floor's chunks, so the store never gives a chunk
// back; the default 0.85 keeps that share of the slots rounded down to whole
// regions, the elastic chunks above it; and a floor given holds whatever the
// share. Flash-Next's experts on a 24 GB M6, 5721 slots of 1.3 MiB at the cap.
void floorShareOfThePlan() {
  constexpr ExpertSlotPlanes kFlashNext{{{442368, 0, 16384}, {442368, 0, 16384}, {442368, 0, 16384}}};
  const uint64_t group = alignedSlotsOf(kFlashNext);
  const auto layout = [&](uint32_t floor, double share, const std::string &what) {
    return checked(
        streamedExpertChunkLayout(kFlashNext, 5721, floor, 512, UINT64_MAX, kExpertElasticChunkBytes, share),
        kFlashNext, 5721, UINT64_MAX, what);
  };
  const ExpertChunkLayout whole = layout(0, 1.0, "M6, a floor share of 1.0");
  require(whole.floorSlots == 5721 && whole.floorChunks == whole.chunks,
          "M6: a floor share of 1.0 left a chunk to give back");
  const ExpertChunkLayout share = layout(0, kExpertFloorShare, "M6, the default floor share");
  require(share.floorSlots == alignDown(floorOf(5721), group) && share.floorChunks == 1 && share.chunks > 1,
          "M6: the default floor share did not keep 85% of the slots below elastic chunks");
  const ExpertChunkLayout half = layout(0, 0.5, "M6, a floor share of 0.5");
  require(half.floorSlots == alignDown(5721 / 2, group) && half.floorChunks == 1,
          "M6: a floor share of 0.5 did not keep half the slots");
  for (const double any : {1.0, kExpertFloorShare, 0.5}) {
    const ExpertChunkLayout given = layout(3000, any, "M6, a floor of 3000 slots given");
    require(given.floorSlots == alignDown(3000, group) && given.floorChunks == 1,
            "M6: a floor given moved with the floor share " + std::to_string(any));
  }
}

// The smallest largest buffer that holds the size in at most 48 chunks is
// taken, a byte less refused, and one of a sixtieth of the size refused. The
// 20000-slot sizes take exactly 48 chunks under it, the 515-slot ones fewer.
void tooManyChunks() {
  for (const auto &[planes, name] : kLayouts)
    for (const uint32_t slots : {515u, 20000u}) {
      const std::string what = std::string(name) + ", " + std::to_string(slots) + " slots";
      const uint64_t fewest = alignUp((slots + MOE_EXPERT_MAX_CHUNKS - 1) / MOE_EXPERT_MAX_CHUNKS,
                                      alignedSlotsOf(planes));
      const uint64_t largest = chunkBytes(planes, fewest);
      const ExpertChunkLayout layout = streamed(planes, slots, 1, largest, what);
      require(layout.floorSlots == fewest && layout.chunkSlots == fewest &&
                  layout.chunks == (slots + fewest - 1) / fewest &&
                  (slots != 20000 || layout.chunks == MOE_EXPERT_MAX_CHUNKS),
              what + ": the floor and elastic chunks under the smallest largest buffer of at most 48 chunks are "
                     "not what it holds");
      const std::string expected = "takes more than " + std::to_string(MOE_EXPERT_MAX_CHUNKS) +
                                   " of the device's largest buffers";
      const auto under = [&](uint64_t buffer) {
        static_cast<void>(
            streamedExpertChunkLayout(planes, slots, floorOf(slots), 1, buffer, kExpertElasticChunkBytes,
                                      kExpertFloorShare));
      };
      rejects([&] { under(largest - 1); }, expected,
              what + ": a largest buffer a byte short of the smallest of at most 48 chunks was taken");
      rejects([&] { under(chunkBytes(planes, slots) / 60); }, expected,
              what + ": a largest buffer of a sixtieth of the size was taken");
    }
}

// A size of one block keeps every expert in its floor chunk alone, taken
// under a largest buffer of its bytes exactly and refused a byte less.
void oneBlockSizes() {
  for (const auto &[planes, name] : kLayouts)
    for (const uint32_t slots : {3u, 259u, 512u}) {
      const std::string what = std::string(name) + ", one block of " + std::to_string(slots) + " slots";
      const uint64_t bytes = chunkBytes(planes, slots);
      const ExpertChunkLayout layout =
          checked(wholeExpertChunkLayout(planes, slots, bytes), planes, slots, bytes, what);
      require(layout.chunks == 1, what + ": takes more than its floor chunk");
      rejects([&] { static_cast<void>(wholeExpertChunkLayout(planes, slots, bytes - 1)); },
              "more than the device's largest buffer", what + ": taken under a largest buffer a byte short of it");
    }
}

} // namespace

int main() {
  try {
    floorShareAndElasticChunks();
    elasticChunkBytes();
    floorMinimum();
    floorCap();
    floorsAskedFor();
    largerElasticChunks();
    floorShareOfThePlan();
    tooManyChunks();
    oneBlockSizes();
  } catch (const std::exception &error) {
    std::cerr << "expert_chunk_layout_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "expert_chunk_layout_test: PASS\n";
  return 0;
}
