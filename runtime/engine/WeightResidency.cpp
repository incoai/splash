#include "engine/WeightResidency.hpp"

#include "StderrLine.hpp"

#include <iomanip>
#include <stdexcept>
#include <utility>

namespace splash::engine {

WeightResidency::WeightResidency(WeightResidencyConfig config) : config_(std::move(config)) {
  if (!config_.weights || !config_.admit)
    throw std::invalid_argument("the weights' residency lacks the weights or their admission");
  if (!(config_.idleReleaseMilliseconds > 0.0))
    throw std::invalid_argument("the idle release must be positive");
}

bool WeightResidency::held() const noexcept {
  return !config_.weights->released() && !restoringSince_;
}

WeightResidency::Step WeightResidency::restore(double now) {
  if (!restoringSince_)
    restoringSince_ = now;
  bool restored = false;
  try {
    restored = config_.weights->restore(config_.admit);
  } catch (const metal::MetalAllocationError &refusal) {
    if (!refusedSince_) {
      refusedSince_ = now;
      logLine("Weights wait for memory to be restored: ",
              metal::allocationFailureName(refusal.failure()));
    }
    return {Step::Kind::Refused, refusal.failure()};
  }
  refusedSince_.reset();
  if (!restored)
    return {Step::Kind::Progress};
  ++restores_;
  logLine("Weights restored in ", std::fixed, std::setprecision(2),
          (now - *restoringSince_) / 1000.0, " s");
  restoringSince_.reset();
  waiterFailed_ = false;
  idleFrom(now);
  return {Step::Kind::Restored};
}

void WeightResidency::waiterFailed() noexcept { waiterFailed_ = true; }

void WeightResidency::abandon() {
  config_.weights->release();
  restoringSince_.reset();
  refusedSince_.reset();
  if (!std::exchange(waiterFailed_, false)) {
    logLine("Weights not restored: no request waits for them");
    return;
  }
  ++failures_;
  logLine("Weights not restored: the requests waiting for them failed for want of memory; "
          "the next request restores them");
}

bool WeightResidency::release(double now, bool engineIdle, bool idleOnly) {
  if (!held() || !engineIdle || (idleOnly && now - idleSince_ < config_.idleReleaseMilliseconds))
    return false;
  config_.weights->release();
  if (idleOnly)
    logLine("Weights released after ", config_.idleReleaseMilliseconds / 1000.0,
            " s without a request; the next request restores them");
  else
    logLine("Weights released on request; the next request restores them");
  return true;
}

void WeightResidency::idleFrom(double now) noexcept { idleSince_ = now; }

WeightsSnapshot WeightResidency::snapshot() const noexcept {
  return {config_.idleReleaseMilliseconds / 1000.0, config_.weights->released(),
          restoringSince_.has_value(), restores_, failures_};
}

} // namespace splash::engine
