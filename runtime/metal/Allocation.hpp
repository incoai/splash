#pragma once

// The allocation vocabulary and the backend's errors, without Metal types, so
// that the engine and the model reach them without the backend.

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::metal {

// Why an allocation was refused: by the engine's memory budget, to protect
// the host's memory reserve, or by the Metal driver.
enum class AllocationFailure : uint8_t {
  None,
  EngineBudget,
  HostPressure,
  DriverRejected,
};

struct AllocationResult final {
  AllocationFailure failure = AllocationFailure::None;
  // Granted.
  AllocationResult() noexcept = default;
  AllocationResult(AllocationFailure reason) noexcept : failure(reason) {}
  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == AllocationFailure::None;
  }
};

[[nodiscard]] constexpr const char *allocationFailureName(
    AllocationFailure failure) noexcept {
  switch (failure) {
  case AllocationFailure::None: return "none";
  case AllocationFailure::EngineBudget: return "engine memory budget exceeded";
  case AllocationFailure::HostPressure: return "host memory reserve protected";
  case AllocationFailure::DriverRejected:
    return "Metal driver rejected allocation";
  }
  return "unknown allocation failure";
}

// Allocators use this callback to obtain engine-governed headroom
// without depending on the engine policy type. The operation runs while the
// caller's reservation is held; the callback returns the refusal's cause,
// with no side effects, when admission is denied.
using AllocationAdmission =
    std::function<AllocationResult(uint64_t, const std::function<void()> &)>;

// The admission of one of several allocations an admission granted at once,
// whose reservation holds their bytes: the operation runs as it is, and a
// driver's refusal throws for that admission to report.
inline AllocationResult admittedWithin(uint64_t, const std::function<void()> &allocate) {
  allocate();
  return {};
}

// A failure of the Metal backend; an allocation's refusal is one
// (MetalAllocationError), carrying its cause.
class MetalBackendError : public std::runtime_error {
public:
  explicit MetalBackendError(const std::string &message, bool outOfMemory = false)
      : std::runtime_error(message), outOfMemory_(outOfMemory) {}
  // A command failed for want of GPU memory (commandOutOfMemory): another
  // process holds what it needs, or the command asked more than the GPU has.
  [[nodiscard]] bool outOfMemory() const noexcept { return outOfMemory_; }

private:
  bool outOfMemory_ = false;
};

// A normal capacity failure. Callers may evict cache or return a retryable
// admission error; the Metal backend remains healthy. An admission
// (AllocationAdmission) returns its failure as a value (AllocationResult).
class MetalAllocationError final : public MetalBackendError {
public:
  explicit MetalAllocationError(
      std::string message,
      AllocationFailure failure = AllocationFailure::DriverRejected)
      : MetalBackendError(std::move(message)), failure_(failure) {}
  [[nodiscard]] AllocationFailure failure() const noexcept { return failure_; }
private:
  AllocationFailure failure_;
};

} // namespace splash::metal
