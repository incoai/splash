#pragma once

#include "flash/FlashPLE.hpp"

namespace splash::flash {

inline constexpr uint64_t kFlashPLEFusedMaximumArgumentBytes = 16 * 1024;

// Immutable direct access to the checkpoint's unchanged 128 Q4/G32 shards.
// The argument buffer owns source allocations through the backend. The class
// is safe to share across sequential and batched request graphs; all mutable
// IDs, token history, output and diagnostics are supplied by the caller.
class FlashPLEFused final {
public:
  FlashPLEFused(metal::MetalBackend &backend, FlashPLEWeights weights);

  // Zero on unsupported devices, where addHashGather/addGather return false
  // before validating or changing a graph so the caller can use the control.
  [[nodiscard]] static uint64_t plannedBytes(metal::MetalBackend &backend);
  [[nodiscard]] bool supported() const noexcept { return bool(sources_); }
  [[nodiscard]] uint64_t allocatedBytes() const noexcept {
    return sources_.sizeBytes();
  }

  // One direct gather dispatch from audited existing I64 IDs.
  [[nodiscard]] bool addGather(metal::CommandGraph &graph,
      metal::MetalBuffer ngramIDs, metal::MetalBuffer output,
      metal::MetalBuffer diagnostics, FlashPLEGeometry geometry) const;

  // One fused hash/gather dispatch plus one ordered history update, replacing
  // the qualified control's hash, update and sixteen gather dispatches. IDs
  // and all BF16 output values remain available for exact primitive auditing.
  [[nodiscard]] bool addHashGather(metal::CommandGraph &graph,
      metal::MetalBuffer tokenIDs, metal::MetalBuffer tokenHistory,
      metal::MetalBuffer ngramIDs, metal::MetalBuffer output,
      metal::MetalBuffer diagnostics, FlashPLEGeometry geometry) const;

private:
  friend class FlashForward;
  FlashPLEWeights weights_;
  metal::MetalBuffer sources_;
};

} // namespace splash::flash
