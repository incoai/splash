#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace splash::metal {

// Guarded by the backend's submission mutex. A completed GPU command stops
// this clock even if its model ticket continues waiting for CPU work.
class CommandWatchdog final {
public:
  explicit CommandWatchdog(double timeoutSeconds)
      : timeoutSeconds_(timeoutSeconds) {
    if (!std::isfinite(timeoutSeconds) || timeoutSeconds <= 0.0)
      throw std::invalid_argument("command timeout must be finite and positive");
  }
  void start(uint64_t sequence, double nowSeconds) noexcept {
    sequence_ = sequence;
    deadlineSeconds_ = nowSeconds + timeoutSeconds_;
  }
  void complete(uint64_t sequence) noexcept {
    if (sequence_ == sequence)
      sequence_ = 0;
  }
  [[nodiscard]] bool expired(double nowSeconds) const noexcept {
    return sequence_ && nowSeconds >= deadlineSeconds_;
  }
  [[nodiscard]] double timeoutSeconds() const noexcept { return timeoutSeconds_; }

private:
  double timeoutSeconds_;
  uint64_t sequence_ = 0;
  double deadlineSeconds_ = 0.0;
};

} // namespace splash::metal
