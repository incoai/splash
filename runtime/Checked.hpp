#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash {

// On overflow, the result is left untouched.
[[nodiscard]] inline bool checkedAdd(uint64_t left, uint64_t right,
                                     uint64_t &result) {
  if (left > std::numeric_limits<uint64_t>::max() - right)
    return false;
  result = left + right;
  return true;
}

[[nodiscard]] inline bool checkedMultiply(uint64_t left, uint64_t right,
                                          uint64_t &result) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

// The sum, or an Error "<what> overflows".
template <class Error = std::overflow_error>
[[nodiscard]] uint64_t checkedAdd(uint64_t left, uint64_t right,
                                  std::string_view what) {
  uint64_t result = 0;
  if (!checkedAdd(left, right, result))
    throw Error(std::string(what) + " overflows");
  return result;
}

// The product, or an Error "<what> overflows".
template <class Error = std::overflow_error>
[[nodiscard]] uint64_t checkedMultiply(uint64_t left, uint64_t right,
                                       std::string_view what) {
  uint64_t result = 0;
  if (!checkedMultiply(left, right, result))
    throw Error(std::string(what) + " overflows");
  return result;
}

// The Apple Silicon VM page: arenas, state cells and slot files align to it.
inline constexpr uint64_t kHostPageBytes = 16 * 1024;

// bytes rounded up to a whole number of host pages.
[[nodiscard]] constexpr uint64_t alignUp(uint64_t bytes) noexcept {
  return (bytes + kHostPageBytes - 1) & ~(kHostPageBytes - 1);
}

} // namespace splash
