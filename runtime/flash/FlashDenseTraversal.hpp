#pragma once

#include <cstdint>

namespace splash::flash {

// Traversal is an opt-in policy for measured, exact shapes. It changes only
// threadgroup order; operand layout, tile geometry and dot arithmetic stay fixed.
enum class FlashDenseTraversal : uint32_t {
  ColumnFast = 0,
  RowFast = 1,
  Swizzle2 = 2,
  Swizzle4 = 3,
  Swizzle8 = 4,
};

inline constexpr const char *kFlashDenseTraversalSemantics =
    ";dense-bf16-measured-shape-traversal-v1";
[[nodiscard]] bool flashDenseTraversalEnabled();

[[nodiscard]] constexpr FlashDenseTraversal flashDenseTraversalPolicy(
    uint32_t rows, uint32_t outputs, uint32_t inputs,
    uint32_t tileRows, uint32_t tileOutputs) noexcept {
  if (rows == 2048 && outputs == 2560 && inputs == 6144 &&
      tileRows == 64 && tileOutputs == 128)
    return FlashDenseTraversal::RowFast;
  if (rows == 512 && inputs == 2560 && tileRows == 32 && tileOutputs == 128) {
    if (outputs == 6144) return FlashDenseTraversal::Swizzle8;
    if (outputs == 2560) return FlashDenseTraversal::Swizzle4;
  }
  return FlashDenseTraversal::ColumnFast;
}

struct FlashDenseTraversalGrid final {
  uint64_t x, y;
};
[[nodiscard]] constexpr FlashDenseTraversalGrid flashDenseTraversalGrid(
    uint32_t rowTiles, uint32_t columnTiles, FlashDenseTraversal traversal) noexcept {
  const auto mode = static_cast<uint32_t>(traversal);
  if (mode == 0) return {columnTiles, rowTiles};
  if (mode == 1) return {rowTiles, columnTiles};
  if (mode > 4) return {0, 0};
  const uint32_t width = 1u << (mode - 1);
  return {uint64_t(columnTiles) * width, (uint64_t(rowTiles) + width - 1) / width};
}

} // namespace splash::flash
