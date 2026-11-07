#pragma once

#include "metal/Allocation.hpp"
#include "model/WeightMemory.hpp"

#include <cstdint>
#include <limits>
#include <optional>

namespace splash::engine {

// The model's weights as /status reports them (WeightResidency::snapshot,
// and Engine::weightsSnapshot for a restore a request waits for).
struct WeightsSnapshot {
  // --idle-release: infinite with `off`, when only a release request frees them.
  double idleReleaseSeconds = 0.0;
  // From a release until the restore has written the last image back.
  bool released = false;
  // From the request that found them released until the restore ends.
  bool restoring = false;
  // The times they were written back, each for a request.
  uint64_t restores = 0;
  // The restores that gave up for want of memory: their requests failed
  // retryably and the weights were released again.
  uint64_t restoreFailures = 0;
};

// What the engine gives back while idle and takes again before it runs the
// next request (EngineConfig::weights).
struct WeightResidencyConfig final {
  // The weights and what goes with them (ReleasableMemory).
  model::WeightMemory *weights = nullptr;
  // Admits the memory they take back (MemoryGovernor::allocationAdmission).
  metal::AllocationAdmission admit;
  // --idle-release: infinite with `off`, when only a release request frees
  // them.
  double idleReleaseMilliseconds = std::numeric_limits<double>::infinity();
};

// The weights' release and restore, on the engine's clock. A restore takes
// back a part per call (WeightMemory::restore), so that the engine answers
// frames and runs its timed checks between them; the engine runs no request
// until it ends.
class WeightResidency final {
public:
  // Throws std::invalid_argument without the weights or their admission, or
  // with an idle release that is not positive.
  explicit WeightResidency(WeightResidencyConfig config);
  // Neither released nor being restored.
  [[nodiscard]] bool held() const noexcept;
  // From the first restore() after a release until the restore ends.
  [[nodiscard]] bool restoring() const noexcept { return restoringSince_.has_value(); }
  struct Step final {
    enum class Kind : uint8_t { Progress, Restored, Refused } kind;
    metal::AllocationFailure failure = metal::AllocationFailure::None;
  };
  // Takes back the next part under the admission; the first call starts the
  // restore. Progress: a part came back and others remain. Restored: the
  // last came back, which begins an idle period. Refused: admission or the
  // driver refused the part, which stays released, for `failure`; what came
  // back stays.
  [[nodiscard]] Step restore(double nowMilliseconds);
  // A request waiting for the restore failed at its resource wait limit: the
  // restore counts a failure if it is abandoned.
  void waiterFailed() noexcept;
  // No request waits for the restore and memory refused its next part: gives
  // back what came back, for the next request's restore to start over.
  void abandon();
  // engineIdle: Engine::idle(). idleOnly: the idle release, after
  // idleReleaseMilliseconds without a request, else a release request.
  // Releases only held weights, and logs which release it was; true when it
  // released.
  [[nodiscard]] bool release(double nowMilliseconds, bool engineIdle, bool idleOnly);
  // The idle period begins: at Ready, when the engine's last request left,
  // and when the weights were restored.
  void idleFrom(double nowMilliseconds) noexcept;
  [[nodiscard]] WeightsSnapshot snapshot() const noexcept;

private:
  WeightResidencyConfig config_;
  double idleSince_ = 0.0;
  std::optional<double> restoringSince_;
  // Since the first refusal after the last part that came back.
  std::optional<double> refusedSince_;
  // A request waiting for this restore failed at its resource wait limit.
  bool waiterFailed_ = false;
  uint64_t restores_ = 0, failures_ = 0;
};

} // namespace splash::engine
