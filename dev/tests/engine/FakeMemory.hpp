#pragma once

// The bytes a test's engine holds and the Mac around it, as one budget.

#include "ClaimScope.hpp"
#include "TestMetalStatistics.hpp"
#include "engine/MemoryGovernor.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::test {

// A scripted change of the Mac around the engine, at a clock time: what
// other applications hold and macOS's pressure level from then on.
struct HostEvent final {
  double atMilliseconds = 0.0;
  uint64_t otherBytes = 0;
  engine::MemoryPressure pressure = engine::MemoryPressure::Normal;
};

// Every fake that allocates (TestKvStorage's extents; the Executor's states,
// per-request reads and pools; Weights) allocates through an admission (the
// real MemoryGovernor's where the test has one), inside which FakeMemory
// counts the bytes for their holder and sets metalStatistics(), which the
// stand-in backend reports to the governor, as MetalBackend allocates inside
// the reservation. It decides nothing. The host's available and spare memory
// are what other applications and the engine leave of it. Script events at
// or before the clock's start apply at construction.
class FakeMemory final {
public:
  struct Config final {
    uint64_t hostBytes = 64ULL << 30;
    uint64_t otherBytes = 8ULL << 30;
    std::vector<HostEvent> script;
    // How long the host's readings take to show what the engine allocates
    // and gives back, as macOS's counters lag a release.
    double readingLagMilliseconds = 0.0;
  };
  struct Change final {
    double atMilliseconds = 0.0;
    std::string holder;
    int64_t bytes = 0;
    uint64_t totalAfter = 0;
  };

  explicit FakeMemory(Config config) : config_(std::move(config)), other_(config_.otherBytes) {
    if (!std::ranges::is_sorted(config_.script, {}, &HostEvent::atMilliseconds))
      throw std::invalid_argument("a host script's events are out of order");
    metalStatistics() = {};
    advanceTo(0.0);
  }
  FakeMemory(const FakeMemory &) = delete;
  FakeMemory &operator=(const FakeMemory &) = delete;
  ~FakeMemory() { metalStatistics() = {}; }

  // Runs `admission` for `bytes`; admitted, `holder` holds them from inside
  // the admitted operation; refused, nothing changes and the result names the
  // cause.
  [[nodiscard]] metal::AllocationResult allocate(const metal::AllocationAdmission &admission,
                                                 std::string_view holder, uint64_t bytes) {
    return admission(bytes, [&] { change(holder, static_cast<int64_t>(bytes)); });
  }
  // Setup memory, unadmitted (arenas, weights loaded before serving).
  void hold(std::string_view holder, uint64_t bytes) { change(holder, static_cast<int64_t>(bytes)); }
  void release(std::string_view holder, uint64_t bytes) {
    if (bytes > held(holder))
      throw std::logic_error("fake memory released more than " + std::string(holder) + " holds");
    change(holder, -static_cast<int64_t>(bytes));
  }

  [[nodiscard]] uint64_t held(std::string_view holder) const noexcept {
    const auto found = holders_.find(holder);
    return found == holders_.end() ? 0 : found->second;
  }
  [[nodiscard]] uint64_t total() const noexcept { return total_; }
  [[nodiscard]] uint64_t peak() const noexcept { return peak_; }
  // The most other applications and the engine together ever held beyond
  // the host's memory.
  [[nodiscard]] uint64_t peakOverHost() const noexcept { return peakOverHost_; }
  [[nodiscard]] const std::vector<Change> &changes() const noexcept { return changes_; }

  // The host as the governor samples it (MemoryGovernor::HostMemoryProvider):
  // available = spare = what other applications and the engine leave, the
  // engine's bytes as they stood the reading lag ago.
  [[nodiscard]] engine::MemoryGovernor::HostMemoryProvider hostMemory() {
    return [this]() -> std::optional<engine::HostMemory> {
      const uint64_t used = other_ + totalAt(now_ - config_.readingLagMilliseconds);
      const uint64_t free = used < config_.hostBytes ? config_.hostBytes - used : 0;
      return engine::HostMemory{free, free};
    };
  }
  [[nodiscard]] engine::MemoryPressure pressure() const noexcept { return pressure_; }
  [[nodiscard]] uint64_t otherBytes() const noexcept { return other_; }
  // Applies the script's events up to `nowMilliseconds`; the clock never
  // goes back.
  void advanceTo(double nowMilliseconds) {
    if (nowMilliseconds < now_)
      throw std::logic_error("fake memory's clock went back");
    now_ = nowMilliseconds;
    for (; next_ < config_.script.size() && config_.script[next_].atMilliseconds <= now_; ++next_) {
      other_ = config_.script[next_].otherBytes;
      pressure_ = config_.script[next_].pressure;
      notePeak();
    }
  }
  [[nodiscard]] std::optional<double> nextEvent() const noexcept {
    if (next_ == config_.script.size())
      return std::nullopt;
    return config_.script[next_].atMilliseconds;
  }
  // Changes the Mac around the engine now, as a script event at the clock's
  // time would, for a test that moves it from its own steps.
  void setHost(uint64_t otherBytes, engine::MemoryPressure pressure) {
    other_ = otherBytes;
    pressure_ = pressure;
    notePeak();
  }
  [[nodiscard]] uint64_t hostBytes() const noexcept { return config_.hostBytes; }
  // Throws unless the backend's statistics count exactly what the fakes
  // allocated through FakeMemory: nothing wrote them behind its back.
  void check() const {
    if (metalStatistics().allocatedBytes != total_)
      throw std::logic_error("the backend's allocated bytes are not what fake memory counts");
  }

private:
  // What the engine held at the clock time: after the last change by then.
  [[nodiscard]] uint64_t totalAt(double milliseconds) const noexcept {
    for (auto change = changes_.rbegin(); change != changes_.rend(); ++change)
      if (change->atMilliseconds <= milliseconds)
        return change->totalAfter;
    return 0;
  }
  void change(std::string_view holder, int64_t bytes) {
    auto found = holders_.find(holder);
    if (found == holders_.end())
      found = holders_.emplace(std::string(holder), 0).first;
    found->second += bytes;
    total_ += bytes;
    metalStatistics().allocatedBytes = total_;
    metalStatistics().peakAllocatedBytes = std::max(metalStatistics().peakAllocatedBytes, total_);
    peak_ = std::max(peak_, total_);
    notePeak();
    changes_.push_back({now_, std::string(holder), bytes, total_});
  }
  void notePeak() {
    const uint64_t used = other_ + total_;
    if (used > config_.hostBytes)
      peakOverHost_ = std::max(peakOverHost_, used - config_.hostBytes);
  }

  Config config_;
  std::map<std::string, uint64_t, std::less<>> holders_;
  uint64_t total_ = 0;
  uint64_t peak_ = 0;
  uint64_t peakOverHost_ = 0;
  uint64_t other_ = 0;
  engine::MemoryPressure pressure_ = engine::MemoryPressure::Normal;
  double now_ = 0.0;
  size_t next_ = 0;
  std::vector<Change> changes_;
};

// The limits of the real MemoryGovernor a test runs over FakeMemory's host.
struct GovernorLimits final {
  uint64_t limitBytes = 40ULL << 30, hostReserveBytes = 2ULL << 30, hostSpareReserveBytes = 2ULL << 30,
           untrackedReserveBytes = 0;
};

// That governor over the stand-in backend (TestMetalMemory.hpp), sampling
// `memory`'s host.
[[nodiscard]] inline std::optional<engine::MemoryGovernor> governorOver(metal::MetalBackend &backend, FakeMemory &memory,
                                                                       const GovernorLimits &limits) {
  return std::optional<engine::MemoryGovernor>(std::in_place, backend, limits.limitBytes, limits.hostReserveBytes,
                                               limits.hostSpareReserveBytes, memory.hostMemory(),
                                               limits.untrackedReserveBytes);
}

// macOS's warning from the start for good, other applications holding
// `otherBytes`.
[[nodiscard]] inline std::vector<HostEvent> persistentWarning(uint64_t otherBytes) {
  return {{0.0, otherBytes, engine::MemoryPressure::Warning}};
}

// Other applications hold `otherBytes` from the start and keep only
// `laterBytes` from `releaseMilliseconds` on: an engine that starts short of
// host memory.
[[nodiscard]] inline std::vector<HostEvent> lowHostAtStart(uint64_t otherBytes, uint64_t laterBytes,
                                                           double releaseMilliseconds) {
  return {{0.0, otherBytes, engine::MemoryPressure::Normal},
          {releaseMilliseconds, laterBytes, engine::MemoryPressure::Normal}};
}

} // namespace splash::test
