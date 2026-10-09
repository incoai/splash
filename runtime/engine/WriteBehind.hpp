#pragma once

#include "engine/Cache.hpp"

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_set>
#include <utility>

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
// published. While the tier has written hourlyBytes in the last hour these
// writes wait; the writes eviction needs never do.
class WriteBehind final {
public:
  static constexpr double kDelayMilliseconds = 10'000.0;
  static constexpr uint32_t kMinimumTokens = 2048;
  static constexpr uint64_t kHourlyBytes = uint64_t{128} << 30;

  explicit WriteBehind(Cache &cache, uint64_t hourlyBytes = kHourlyBytes);
  WriteBehind(const WriteBehind &) = delete;
  WriteBehind &operator=(const WriteBehind &) = delete;

  // A point published or reused at this block, `tokens` deep. Nothing in a
  // temporary tier.
  void published(uint64_t block, uint32_t tokens, double now);
  // Steps the points that are due. True when one became durable or was
  // given up; a write it starts wakes the engine when it lands.
  [[nodiscard]] bool run(double now);
  // At a clean stop: steps every waiting point, the newest first, due or
  // not, past the hourly limit. True once none is left.
  [[nodiscard]] bool flush();
  // When run() next has something to do that no landing write will wake it
  // for: the next point falling due, or the next look at the hourly limit.
  [[nodiscard]] std::optional<double> nextWakeup() const;
  [[nodiscard]] WriteBehindSnapshot snapshot() const noexcept;

private:
  struct Point final {
    uint64_t block = 0;
    double due = 0.0;
  };
  // Counts a step that settled its point; false while it is in progress.
  bool settle(PersistStatus status);
  // Samples the bytes the tier has written, at most once a minute, and
  // whether the last hour's reached the limit.
  void sampleWrites(double now);

  Cache &cache_;
  uint64_t hourlyBytes_;
  std::deque<Point> waiting_;
  std::unordered_set<uint64_t> queued_;
  // (time, bytes written) samples back to the latest one at least an hour
  // old. Every write starts in a tick, so the window they span counts the
  // last hour's writes and at most a minute's from before it.
  std::deque<std::pair<double, uint64_t>> written_;
  bool limited_ = false;
  // The front point is due and waits for its writes, or the tier's, to land.
  bool stalled_ = false;
  WriteBehindSnapshot counters_;
};

} // namespace splash::engine
