#pragma once

// How a size of an expert store lays its slots out in chunks (model::
// ExpertStore, ops::ExpertChunks): the rule alone, so that it is checked
// without a GPU.
//
// A streamed size's floor chunk, chunk 0, holds its floor: the slots it
// keeps however short memory is (floorBytes, or kExpertFloorShare of its
// slots unless the model's plan sets another share), or as much of them as
// the largest buffer holds, the elastic chunks after it the rest of the
// floor. The slots above the floor lie in elastic chunks of about
// kExpertElasticChunkBytes each (ExpertStoreOptions::elasticChunkBytes), the
// last holding the rest, which the store gives back from the last on while
// memory is short and takes again once it is not; it starts with its floor
// alone, and an idle release releases all of them. A size of one block
// keeps every expert, in its floor chunk alone. Chunks are few because each
// chunk an encoder makes resident costs GPU time: about 0.065% of a
// streamed decode cycle per chunk.

#include "metal/abi/MoE.h"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>

namespace splash::model {

// The share of its slots a streamed size's floor holds where the model's
// plan sets no other (ExpertStoreOptions::floorShare).
inline constexpr double kExpertFloorShare = 0.85;
// About the most bytes an elastic chunk takes by default, unless the size
// needs larger ones to fit the expert kernels' table (MOE_EXPERT_MAX_CHUNKS).
inline constexpr uint64_t kExpertElasticChunkBytes = 1ULL << 30;

// A slot's bytes of each plane of a size's experts: gate's, up's and down's
// plane0, plane1 and meta, each plane in a region of its own in a chunk.
using ExpertSlotPlanes = std::array<std::array<uint64_t, 3>, 3>;

// Where a size's `slots` slots lie: the floor chunk, chunk 0, holds the first
// floorSlots of them, and each elastic chunk after it chunkSlots, the last
// the rest; `chunks` chunks in all. The first floorChunks of them hold the
// floor and are never given back: the floor chunk, and the elastic chunks
// after it that hold the rest of a floor larger than the largest buffer. A
// size without an elastic chunk has floorSlots = chunkSlots = slots. Every
// chunk but the last holds a multiple of the fewest slots whose every plane
// fills whole regions (kWeightFileAlignment), so that only the last chunk's
// regions pad and the chunks take the bytes of one chunk of all the slots.
struct ExpertChunkLayout final {
  uint32_t slots = 0;
  uint32_t floorSlots = 0;
  uint32_t chunkSlots = 0;
  uint32_t chunks = 0;
  uint32_t floorChunks = 1;

  // The first slot of chunk `chunk` (at most `chunks`), which is the slots
  // of the chunks before it: `slots` past the last.
  [[nodiscard]] uint32_t first(uint32_t chunk) const noexcept {
    if (chunk == 0) return 0;
    return static_cast<uint32_t>(std::min<uint64_t>(slots, floorSlots + uint64_t{chunk - 1} * chunkSlots));
  }
  // The slots chunk `chunk` (below `chunks`) holds.
  [[nodiscard]] uint32_t held(uint32_t chunk) const noexcept { return first(chunk + 1) - first(chunk); }
  // The chunk that holds `slot` (below `slots`).
  [[nodiscard]] uint32_t chunkOf(uint32_t slot) const noexcept {
    return slot < floorSlots ? 0 : 1 + (slot - floorSlots) / chunkSlots;
  }
};

namespace detail {

// The fewest slots whose every plane fills whole regions, and their bytes.
struct ExpertSlotGroup final {
  uint64_t slots = 1;
  uint64_t bytes = 0;
};

[[nodiscard]] inline ExpertSlotGroup expertSlotGroup(const ExpertSlotPlanes &planes) noexcept {
  ExpertSlotGroup group;
  uint64_t slotBytes = 0;
  for (const std::array<uint64_t, 3> &projection : planes)
    for (const uint64_t plane : projection) {
      if (plane) group.slots = std::lcm(group.slots, kWeightFileAlignment / std::gcd(plane, kWeightFileAlignment));
      slotBytes += plane;
    }
  group.bytes = group.slots * slotBytes;
  return group;
}

[[nodiscard]] inline std::invalid_argument tooManyExpertChunks(uint32_t slots) {
  return std::invalid_argument("a size of " + std::to_string(slots) + " expert slots takes more than " +
                               std::to_string(MOE_EXPERT_MAX_CHUNKS) + " of the device's largest buffers");
}

} // namespace detail

// The chunks of a streamed size of `slots` slots, each of `planes`' bytes, in
// buffers of at most `largest` bytes. The floor is `floor` slots, or
// `floorShare` of them where `floor` is zero, rounded down to a multiple of
// the fewest that fill whole regions unless it is every slot, but at least
// `minimum` (a block's experts) rounded up to one and at least one such
// group, and at most the slots. The floor chunk holds the floor, or what a
// buffer of `largest` bytes holds when that is less, and the fewest elastic
// chunks after it that hold the rest of the floor hold it too (floorChunks).
// An elastic chunk holds a multiple of those slots of about `elasticBytes`
// (kExpertElasticChunkBytes), or more where the slots above the floor would
// take more than MOE_EXPERT_MAX_CHUNKS - 1 of them, at most what a buffer of
// `largest` bytes holds, and at most the slots above the floor chunk. Throws
// std::invalid_argument when chunks of at most `largest` bytes would take
// more than MOE_EXPERT_MAX_CHUNKS of them.
[[nodiscard]] inline ExpertChunkLayout streamedExpertChunkLayout(const ExpertSlotPlanes &planes, uint32_t slots,
                                                                 uint32_t floor, uint32_t minimum,
                                                                 uint64_t largest, uint64_t elasticBytes,
                                                                 double floorShare) {
  if (!slots || minimum > slots) throw std::invalid_argument("invalid streamed expert size");
  const detail::ExpertSlotGroup group = detail::expertSlotGroup(planes);
  const auto alignUp = [&](uint64_t value) { return (value + group.slots - 1) / group.slots * group.slots; };
  const auto alignDown = [&](uint64_t value) { return value / group.slots * group.slots; };
  const uint64_t largestSlots = largest / group.bytes * group.slots;
  const uint64_t least = std::min<uint64_t>(std::max(alignUp(minimum), group.slots), slots);
  const uint64_t asked = floor ? floor : static_cast<uint64_t>(floorShare * slots);
  const uint64_t kept = std::min<uint64_t>(std::max(asked >= slots ? slots : alignDown(asked), least), slots);
  const uint64_t floorSlots = std::min(kept, largestSlots);
  if (!floorSlots) throw detail::tooManyExpertChunks(slots);
  if (floorSlots == slots) return {slots, slots, slots, 1, 1};
  const uint64_t fewest = alignUp((slots - floorSlots + MOE_EXPERT_MAX_CHUNKS - 2) / (MOE_EXPERT_MAX_CHUNKS - 1));
  const uint64_t chunkSlots = std::min(std::max(fewest, elasticBytes / group.bytes * group.slots), largestSlots);
  if (chunkSlots < fewest) throw detail::tooManyExpertChunks(slots);
  const uint64_t rest = slots - floorSlots, elastic = std::min(chunkSlots, rest);
  return {slots, static_cast<uint32_t>(floorSlots), static_cast<uint32_t>(elastic),
          static_cast<uint32_t>(1 + (rest + elastic - 1) / elastic),
          static_cast<uint32_t>(1 + (kept - floorSlots + elastic - 1) / elastic)};
}

// The one chunk of a size of one block, which keeps every one of its `slots`
// experts, each of `planes`' bytes, in its floor. Throws std::invalid_argument
// when they take more than `largest` bytes.
[[nodiscard]] inline ExpertChunkLayout wholeExpertChunkLayout(const ExpertSlotPlanes &planes, uint32_t slots,
                                                              uint64_t largest) {
  if (!slots) throw std::invalid_argument("invalid expert size");
  uint64_t bytes = 0;
  for (const std::array<uint64_t, 3> &projection : planes)
    for (const uint64_t plane : projection) bytes += alignWeightOffset(uint64_t{slots} * plane);
  if (bytes > largest)
    throw std::invalid_argument("a size of one block's " + std::to_string(slots) + " experts takes " +
                                std::to_string(bytes) + " bytes, more than the device's largest buffer");
  return {slots, slots, slots, 1, 1};
}

} // namespace splash::model
