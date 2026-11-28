#pragma once

#include "Checked.hpp"
#include "TestChecks.hpp"
#include "model/SlotFile.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace splash::test {

// Writes one page through a file of `budget`, as another writer of the tier
// would, overdrawing a budget that had less.
inline void spendWriteBudget(const std::shared_ptr<model::DiskBudget> &budget) {
  model::SlotFile file(kHostPageBytes, budget);
  const std::vector<std::byte> payload(kHostPageBytes, std::byte{1});
  require(file.write(file.acquire(payload.size()), {payload}, {})->wait(), "the budget's write failed");
}

// A disk budget whose write budget refills on a clock the test moves: two
// host pages a day, half a page at once, so that spend() overdraws a full
// budget by half a page, and it takes writes again once it holds an eighth of
// its half page, 9/32 of a day later. The test links model/SlotFile.cpp.
class WriteBudget final {
public:
  static constexpr double kRefillMilliseconds = 24'300'000.0;
  static constexpr double kDayMilliseconds = 86'400'000.0;

  explicit WriteBudget(uint64_t quotaBytes = uint64_t{1} << 30)
      : budget(std::make_shared<model::DiskBudget>(quotaBytes, [this] { return now; })) {
    budget->setWriteBudget(2 * kHostPageBytes);
  }
  WriteBudget(const WriteBudget &) = delete;
  WriteBudget &operator=(const WriteBudget &) = delete;

  // Moves the clock on until the budget is full again.
  void refill() { now += kDayMilliseconds; }
  void spend() { spendWriteBudget(budget); }

  double now = 0.0;
  std::shared_ptr<model::DiskBudget> budget;
};

} // namespace splash::test
