#pragma once

#include "engine/Cache.hpp"

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_set>

namespace splash::engine {

struct WriteBehindSnapshot final {
  // Points waiting to fall due, or the one being written.
  uint32_t waiting = 0;
  // Points durable when they fell due or once their writes landed.
  uint64_t durable = 0;
  // Points their conversation went on from, or gone, before they fell due.
  uint64_t unneeded = 0;
  // Points the tier could not keep.
  uint64_t refused = 0;
};

// Keeps a persistent tier's newest restore points on disk while they stay
// in RAM. An ordinary point published or reused at least kMinimumTokens
// deep is made durable (Cache::persist) once it has waited
// kDelayMilliseconds, unless its conversation has gone on from it by then:
// a point that is newest only briefly, like a step of a tool loop, costs no
// write, and a shorter one recomputes in less time than its state takes to
// write. Points become durable one at a time, in the order they were
// published. Like every write of the tier these draw on its write budget,
// when one is set (model::DiskBudget): a point waits while the budget holds
// its state back, which is written before its chain.
class WriteBehind final {
public:
  static constexpr double kDelayMilliseconds = 10'000.0;
  static constexpr uint32_t kMinimumTokens = 2048;

  explicit WriteBehind(Cache &cache);
  WriteBehind(const WriteBehind &) = delete;
  WriteBehind &operator=(const WriteBehind &) = delete;

  // A point published or reused at this block, `tokens` deep. Nothing in a
  // temporary tier.
  void published(uint64_t block, uint32_t tokens, double now);
  // Steps the points that are due. True when one became durable or was
  // given up; a write it starts wakes the engine when it lands.
  [[nodiscard]] bool run(double now);
  // At a clean stop: steps every waiting point, the newest first, due or
  // not, whatever the write budget (Cache::liftDiskWriteBudget). True once
  // none is left.
  [[nodiscard]] bool flush();
  // When run() next has something to do that no landing write will wake it
  // for: the next point falling due, or, for a step the write budget held
  // back, when the budget admits it again.
  [[nodiscard]] std::optional<double> nextWakeup() const;
  [[nodiscard]] WriteBehindSnapshot snapshot() const noexcept;

private:
  struct Point final {
    uint64_t block = 0;
    double due = 0.0;
  };
  // Counts a step that settled its point; false while it is in progress.
  bool settle(PersistStatus status);

  Cache &cache_;
  std::deque<Point> waiting_;
  std::unordered_set<uint64_t> queued_;
  // The front point is due and waits for its writes, or the tier's, to land,
  // or for the write budget.
  bool stalled_ = false;
  WriteBehindSnapshot counters_;
};

} // namespace splash::engine
