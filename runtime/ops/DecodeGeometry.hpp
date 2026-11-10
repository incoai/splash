#pragma once

#include "metal/abi/ExecutionGeometry.h"

#include <compare>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace splash::ops {

// Fixed execution geometry of one runtime. This is independent of Table16's
// physical eight-row packing and of any maximum-capacity ABI arrays.
class DecodeGeometry final {
public:
  static constexpr uint32_t maximumRows = SPLASH_MAXIMUM_VERIFY_ROWS;
  constexpr DecodeGeometry() noexcept = default;
  explicit constexpr DecodeGeometry(uint32_t rows) : rows_(rows) {
    if (rows != 8 && rows != 16)
      throw std::invalid_argument("decode geometry must have eight or sixteen rows");
  }
  [[nodiscard]] constexpr uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] constexpr uint32_t proposals() const noexcept { return rows_ - 1; }
  [[nodiscard]] constexpr uint32_t maximumStepTokens() const noexcept { return rows_ + 1; }
  [[nodiscard]] constexpr uint32_t proposalUniform() const noexcept { return 1; }
  [[nodiscard]] constexpr uint32_t acceptanceUniform() const noexcept { return 1 + proposals(); }
  [[nodiscard]] constexpr uint32_t correctionUniform() const noexcept { return 1 + 2 * proposals(); }
  [[nodiscard]] constexpr uint32_t uniformCount() const noexcept { return correctionUniform() + 1; }
  // Eight-row row-sensitive entry names use no geometry suffix. ABI
  // revisions are explicit in the base name. Sixteen-row row-sensitive
  // kernels are explicit compile-time variants in the same library.
  [[nodiscard]] std::string kernel(std::string name) const {
    if (rows_ == 16) name += "_v16";
    return name;
  }
  auto operator<=>(const DecodeGeometry &) const = default;

private:
  uint32_t rows_ = SPLASH_TARGET_VERIFY_ROWS;
};

} // namespace splash::ops
