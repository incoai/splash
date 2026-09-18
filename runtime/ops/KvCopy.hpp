#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Q8PageStorage.hpp"

#include <cstdint>

namespace splash::ops {

// Moves whole KV pages between the Q8 page pool and host-visible staging.
// A copy table holds one entry per staging slot: the page to move and the
// direction, none for an idle slot. One dispatch per attention layer serves
// every slot of the table.
class KvCopy final {
public:
  enum class Direction : uint8_t { None, ToStaging, ToPage };

  // Bytes of the copy table for a ring of this many slots.
  [[nodiscard]] static uint64_t tableBytes(uint32_t stagingSlots) noexcept;
  static void setEntry(const metal::MetalBuffer &table, uint32_t slot, uint32_t page,
                       Direction direction) noexcept;
  static void addPages(metal::CommandGraph &graph, const kv::Q8PageStorage &pages,
                       metal::MetalBuffer staging, metal::MetalBuffer table,
                       uint32_t stagingSlots, uint64_t slotBytes);
};

} // namespace splash::ops
