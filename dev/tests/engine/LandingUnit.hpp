#pragma once

// A memory source whose one unit leaves memory later, as a write's landing
// does.

#include "engine/MemoryLedger.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

namespace splash::test {

// A unit of `bytes` that leaves memory once it lands, at no cost: until it
// has landed, each claim whose free() takes it waits for it.
class LandingUnit final : public engine::VictimSource {
public:
  explicit LandingUnit(uint64_t bytes) : bytes_(bytes) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "landing"; }
  [[nodiscard]] std::optional<engine::Victim> peek(const engine::VictimQuery &,
                                                   const engine::Victim *after) const override {
    if (after || landed)
      return std::nullopt;
    return engine::Victim{.key = 1, .bytes = bytes_, .write = true, .later = true};
  }
  engine::FreeResult take(const engine::VictimQuery &, const engine::Victim &) override {
    return {engine::FreeOutcome::Pending, 0, bytes_};
  }
  bool landed = false;

private:
  uint64_t bytes_;
};

} // namespace splash::test
