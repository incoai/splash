#pragma once

// A model's elastic memory (model::ElasticPool) over FakeMemory.

#include "FakeMemory.hpp"
#include "model/ElasticPool.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::test {

// A pool of `units` units of unitBytes, held in FakeMemory under its name,
// between floorUnits and capUnits; wholeUnits would hold every key it
// serves, which its residentShare() is the share of. It gives back whole
// units above its floor, and takes back one unit under the admission it is
// given.
class FakeElasticPool final : public model::ElasticPool {
public:
  FakeElasticPool(std::string name, FakeMemory &memory, uint64_t unitBytes, uint32_t floorUnits,
                  uint32_t capUnits, uint32_t wholeUnits, uint32_t units)
      : name_(std::move(name)), memory_(memory), unitBytes_(unitBytes), floorUnits_(floorUnits),
        capUnits_(capUnits), wholeUnits_(wholeUnits), units_(units) {
    if (!unitBytes || floorUnits > units || units > capUnits || capUnits > wholeUnits || !wholeUnits)
      throw std::invalid_argument("invalid fake elastic pool");
    memory_.hold(name_, uint64_t{units_} * unitBytes_);
  }
  FakeElasticPool(const FakeElasticPool &) = delete;
  FakeElasticPool &operator=(const FakeElasticPool &) = delete;

  [[nodiscard]] model::PoolState state() const override {
    const uint32_t cap = regrows ? capUnits_ : units_;
    return {.name = name_,
            .floorBytes = uint64_t{floorUnits_} * unitBytes_,
            .currentBytes = uint64_t{units_} * unitBytes_,
            .capBytes = uint64_t{cap} * unitBytes_,
            .releaseUnitBytes = units_ > floorUnits_ ? unitBytes_ : 0,
            .growUnitBytes = units_ < cap ? unitBytes_ : 0,
            .rebuildOnly = rebuildOnly};
  }
  [[nodiscard]] uint64_t release(uint64_t bytes) override {
    const uint64_t above = units_ - floorUnits_;
    const uint64_t units = bytes == std::numeric_limits<uint64_t>::max()
                               ? above
                               : std::min(above, bytes / unitBytes_ + (bytes % unitBytes_ != 0));
    if (!units)
      return 0;
    units_ -= static_cast<uint32_t>(units);
    shrinks += units;
    const uint64_t released = units * unitBytes_;
    memory_.release(name_, released);
    return released;
  }
  [[nodiscard]] metal::AllocationResult grow(const metal::AllocationAdmission &admit) override {
    if (units_ == capUnits_ || !regrows)
      return {};
    const metal::AllocationResult result = memory_.allocate(admit, name_, unitBytes_);
    if (result) {
      ++units_;
      ++grows;
    }
    return result;
  }

  [[nodiscard]] double residentShare() const noexcept { return double(units_) / wholeUnits_; }
  [[nodiscard]] uint32_t units() const noexcept { return units_; }

  // Whether its units cost only their rebuild (model::PoolState::rebuildOnly,
  // a cache the model builds on use such as the vision arena), which the
  // test sets.
  bool rebuildOnly = false;
  // Its units live requests hold from now on, its floor, as a refused image
  // start keeps the vision arena its image fits (StateAdmission::held).
  void hold(uint32_t units) {
    if (units > units_) throw std::invalid_argument("a fake elastic pool holds more units than it has");
    floorUnits_ = units;
  }
  // Units taken and given back.
  uint64_t grows = 0, shrinks = 0;
  // Cleared, the pool grows no further than it stands: its cap is its units.
  bool regrows = true;

private:
  std::string name_;
  FakeMemory &memory_;
  uint64_t unitBytes_;
  uint32_t floorUnits_, capUnits_, wholeUnits_, units_;
};

} // namespace splash::test
