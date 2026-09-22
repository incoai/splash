#pragma once
#include "flash/FlashDenseCache.hpp"
#include <stdexcept>
#include <string_view>

namespace splash::flash {
inline constexpr const char *kFlashSharedExpertFusedSemantics =
    "shared-expert-cached-bf16-whole-k-mpp-f32accum-bf16-dots-compiled-bf16-swiglu-m32n128-min256-tail-m16n64-v1";

// Frozen by the owning trunk at construction. An absent flag defaults off;
// malformed values are rejected rather than silently selecting another route.
inline bool flashSharedExpertFusedFlag(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_SHARED_EXPERT_FUSED must be 0 or 1");
}
inline bool flashSharedExpertFusedEligible(bool enabled, bool denseOperands,
                                           uint32_t rows) noexcept {
  return enabled && denseOperands && rows >= 256 && rows <= 8192;
}
struct FlashSharedExpertFusedRowPlan {
  uint32_t fullRows = 0;
  uint32_t tailWholeRows = 0;
  uint32_t tailVectorRows = 0;
};
inline FlashSharedExpertFusedRowPlan flashSharedExpertFusedRows(uint32_t rows) {
  if (rows < 256 || rows > 8192)
    throw std::invalid_argument("Flash shared expert fused prefill rows must be256..8192");
  const uint32_t full = rows / 32 * 32, remaining = rows - full;
  return {full, remaining / 16 * 16, remaining % 16};
}
struct FlashSharedExpertFusedTail {
  // Borrow existing gate/up workspace; no new persistent or temporary planes.
  // Only the final≤31 rows are written when a main M32 tile is incomplete.
  metal::MetalBuffer gate, up;
};

// Checked BF16[640,2560] immutable operands only. No conversion, allocation,
// CPU matrix work or GPU submission occurs. Rows256..8192 preserve BF16 dot
// casts before compiled SwiGLU. Canonical M16/N64 and vector tail producers
// preserve the original traversal for every incomplete final M32 row tile.
void addSharedExpertFusedPrefill(metal::MetalBackend &backend,
    metal::CommandGraph &graph, const FlashTensor &gate, const FlashTensor &up,
    metal::MetalBuffer input, metal::MetalBuffer activated,
    metal::MetalBuffer diagnostics, uint32_t rows,
    const FlashSharedExpertFusedTail &tail = {});
} // namespace splash::flash
