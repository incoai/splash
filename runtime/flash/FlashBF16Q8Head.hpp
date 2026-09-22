#pragma once
#include "flash/FlashWeights.hpp"
#include "metal/CommandGraph.hpp"
#include <cstdint>

namespace splash::flash {
inline constexpr const char *kFlashBF16Q8HeadSemantics =
    "original-q8-exact-cached-bf16-coeff-register-m8n32-k64-simd1-f32accum-last-r2to4-v1";
[[nodiscard]] bool flashBF16Q8HeadEnabled();
[[nodiscard]] bool flashBF16Q8HeadGeometry(const FlashAffineProjection &, uint32_t rows) noexcept;
// Read-only source buffers remain owned by the caller. This object allocates
// no weights or scratch; the joint head's existing padding arena is reused.
class FlashBF16Q8Head final {
public:
  FlashBF16Q8Head(metal::MetalBackend &, const FlashWeights &, metal::MetalBuffer padding);
  void addProjection(metal::CommandGraph &, metal::MetalBuffer input,
      metal::MetalBuffer output, metal::MetalBuffer diagnostics, uint32_t rows) const;
  [[nodiscard]] static constexpr uint64_t plannedBytes() noexcept { return 0; }
  [[nodiscard]] constexpr uint64_t allocatedBytes() const noexcept { return 0; }
private:
  FlashAffineProjection source_;
  metal::MetalBuffer padded_;
};
} // namespace splash::flash
