#include "model/WindowPlan.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash {

WindowPlan planWindow(uint32_t restore, uint32_t prompt, uint32_t window,
                      uint32_t availableBegin,
                      std::span<const uint32_t> optionalBoundaries) {
  if (!window || prompt < restore || availableBegin > restore)
    throw std::invalid_argument("invalid window planning range");
  uint32_t previous = restore;
  for (uint32_t boundary : optionalBoundaries) {
    if (boundary <= previous || boundary > prompt)
      throw std::invalid_argument(
          "window boundaries must be sorted within replay");
    previous = boundary;
  }
  const auto oldBegin = [=](uint32_t boundary) {
    return std::min(restore, boundary - std::min(boundary, window));
  };
  WindowPlan result;
  result.loadBegin = result.loadEnd = restore;
  if (oldBegin(prompt) < availableBegin)
    return result;
  result.usable = true;
  for (uint32_t boundary : optionalBoundaries)
    if (boundary < prompt && oldBegin(boundary) >= availableBegin)
      result.boundaries.push_back({boundary, 0});
  result.boundaries.push_back({prompt, 0});
  result.loadBegin = oldBegin(result.boundaries.front().tokens);

  uint32_t through = restore;
  bool haveContext = result.loadBegin < restore;
  for (auto &boundary : result.boundaries) {
    const bool extend = haveContext && boundary.tokens - through < window;
    boundary.captureBegin =
        extend ? through : boundary.tokens - std::min(boundary.tokens, window);
    // At an already restored endpoint all required rows come from old pages.
    if (boundary.tokens == restore)
      boundary.captureBegin = restore;
    if (boundary.captureBegin < boundary.tokens) {
      if (extend && !result.captures.empty() &&
          result.captures.back().end == boundary.captureBegin)
        result.captures.back().end = boundary.tokens;
      else
        result.captures.push_back(
            {boundary.captureBegin, boundary.tokens, !extend});
    }
    through = boundary.tokens;
    haveContext = true;
  }
  return result;
}

} // namespace splash
