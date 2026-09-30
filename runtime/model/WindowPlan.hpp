#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace splash {

struct WindowCapture final {
  uint32_t begin;
  uint32_t end;
  bool reset;
};

struct WindowBoundary final {
  uint32_t tokens;
  uint32_t captureBegin;
};

struct WindowPlan final {
  bool usable = false;
  // Old rows needed by the earliest retained boundary, or [B, B) for none.
  uint32_t loadBegin = 0;
  uint32_t loadEnd = 0;
  std::vector<WindowBoundary> boundaries;
  std::vector<WindowCapture> captures;
};

// Old rows are available in [availableBegin, restore). Only prompt is required;
// optional boundaries without enough old context are omitted. All new captures
// and old-row requirements derive from the retained boundaries here.
[[nodiscard]] WindowPlan
planWindow(uint32_t restore, uint32_t prompt, uint32_t window,
           uint32_t availableBegin,
           std::span<const uint32_t> optionalBoundaries = {});

} // namespace splash
