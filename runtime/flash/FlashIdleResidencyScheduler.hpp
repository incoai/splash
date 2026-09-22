#pragma once

#include <chrono>
#include <cstdint>

namespace splash::flash::idle_maintenance {
class Scheduler final {
public:
  using Clock = std::chrono::steady_clock;
  using Time = Clock::time_point;
  explicit Scheduler(uint32_t interval) : interval_(interval) {}
  bool armed() const noexcept { return armed_; }
  bool hasUserGPUCommand() const noexcept { return lastUserGPU_ != Time{}; }
  bool due(Time now) const noexcept { return armed_ && now >= due_; }
  Time lastUserGPUCommand() const noexcept { return lastUserGPU_; }
  void userGPUCompleted(Time now) noexcept {
    lastUserGPU_ = now; due_ = now + interval_;
  }
  void userFinished(Time now, bool emitted, bool cancelled) noexcept {
    if (emitted && !cancelled && hasUserGPUCommand()) {
      armed_ = true; due_ = now + interval_;
    }
  }
  void pressureSuspended(Time now) noexcept { due_ = now + interval_; }
  void maintenanceCompleted(Time now, double wallSeconds) noexcept {
    due_ = now + interval_;
    if (wallSeconds >= std::chrono::duration<double>(interval_).count()) armed_ = false;
  }
  void failed() noexcept { armed_ = false; }
private:
  const std::chrono::milliseconds interval_;
  Time lastUserGPU_{}, due_{};
  bool armed_ = false;
};
} // namespace splash::flash::idle_maintenance
