#pragma once

#include "metal/DeviceCapabilities.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::metal {

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

enum class BufferStorage {
  Shared,
  Private,
};

class MetalBackend;
class CommandTicket;

// A cheap, copyable reference to a backend-owned Metal allocation. Views keep
// the base allocation alive and do not increase the tracked allocation count.
class MetalBuffer final {
public:
  MetalBuffer();
  ~MetalBuffer();
  MetalBuffer(const MetalBuffer &);
  MetalBuffer &operator=(const MetalBuffer &);
  MetalBuffer(MetalBuffer &&) noexcept;
  MetalBuffer &operator=(MetalBuffer &&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] uint64_t sizeBytes() const noexcept;
  [[nodiscard]] BufferStorage storage() const noexcept;
  // Returns nullptr for private buffers and released memory
  // (MetalBackend::releaseMemory). The pointer covers this view only.
  [[nodiscard]] void *contents() const noexcept;
  // GPU address of the view's first byte, for kernels that reach a buffer
  // through an address another buffer holds.
  [[nodiscard]] uint64_t gpuAddress() const noexcept;
  // Allocation identity and exact view range, including Private storage.
  // This compares metadata only; it never maps or reads device contents.
  [[nodiscard]] bool sameView(const MetalBuffer &other) const noexcept;

private:
  struct Impl;
  explicit MetalBuffer(std::shared_ptr<Impl> impl);

  std::shared_ptr<Impl> impl_;

  friend class MetalBackend;
};

struct DispatchSize {
  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

struct BufferBinding {
  uint32_t index = 0;
  MetalBuffer buffer;
};

// The pointed-to data only needs to remain valid until submit() returns.
struct BytesBinding {
  uint32_t index = 0;
  const void *data = nullptr;
  uint64_t sizeBytes = 0;
};

struct ComputeDispatch {
  std::string pipelineName;
  std::vector<BufferBinding> buffers;
  std::vector<BytesBinding> bytes;
  DispatchSize threadgroups;
  DispatchSize threadsPerThreadgroup;
};

struct CommandTiming {
  double gpuSeconds = 0.0;
  double wallSeconds = 0.0;
};

// Move-only ownership of one submitted Metal command. Completion is signalled
// without blocking the submitting thread; wait() is normally called only
// after the host event loop receives the completion notification.
// Destroying or replacing an unfinished ticket waits for GPU completion and
// retains its allocations throughout that wait. Each of these waits runs the
// command watchdog every second: once the watchdog gives up on the command,
// wait() throws MetalBackendError and destruction or replacement returns,
// leaving the allocations to the command's completion handler.
class CommandTicket final {
public:
  CommandTicket();
  ~CommandTicket();
  CommandTicket(const CommandTicket &) = delete;
  CommandTicket &operator=(const CommandTicket &) = delete;
  CommandTicket(CommandTicket &&) noexcept;
  CommandTicket &operator=(CommandTicket &&) noexcept;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] CommandTiming wait();

private:
  struct State;
  explicit CommandTicket(std::shared_ptr<State> state);

  std::shared_ptr<State> state_;

  friend class MetalBackend;
};

using CommandCompletion = std::function<void()>;

// Bytes one allocation added between two memoryStats() readings.
[[nodiscard]] inline uint64_t allocationDelta(uint64_t before, uint64_t after) {
  if (after < before)
    throw std::logic_error("Metal allocation accounting moved backwards");
  return after - before;
}

struct MetalMemoryStats {
  // Sum of MTLResource.allocatedSize for live base buffers created through
  // this backend. Views share their base allocation and add no bytes.
  uint64_t allocatedBytes = 0;
  uint64_t peakAllocatedBytes = 0;

  // Most recently sampled Metal device-wide process counter. Allocation and
  // command lifecycle boundaries refresh it; status reads never synchronize
  // with an in-flight GPU command.
  uint64_t deviceCurrentAllocatedBytes = 0;
  // Highest sampled device.currentAllocatedSize. Sampled after allocations
  // and pipeline creation, on host-side retirement and whenever admission
  // refreshes the stats.
  uint64_t devicePeakAllocatedBytes = 0;
};

class MetalBackendError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// A normal capacity failure. Callers may evict cache or return a retryable
// admission error; the Metal backend remains healthy. MemoryGovernor's
// allocationAdmission is the one place it becomes a value (AllocationResult).
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

// The capabilities a backend reads, without loading kernels or allocating:
// enough to refuse an unsupported Mac before a model is downloaded.
[[nodiscard]] DeviceCapabilities probeDeviceCapabilities();

// How long a command may run before the backend gives up on it by default,
// in time the Mac is awake (AwakeClock).
inline constexpr double kCommandTimeoutSeconds = 120.0;
// How long every buffer stays wired after the last command by default (see
// allocateBuffer), also the default of the engine's idle release
// (RuntimeResourcesConfig::idleReleaseSeconds).
inline constexpr double kResidencyKeepAliveSeconds = 600.0;
static_assert(kCommandTimeoutSeconds > 0.0 && kResidencyKeepAliveSeconds > 0.0);

// Permits exactly one submitted-but-not-applied command on its command queue.
// One thread submits, allocates and looks up pipelines; checkHealth(),
// healthy(), unhealthyReason(), memoryStats() and commandInFlight() may be
// called from any thread.
class MetalBackend final {
public:
  // Buffers stay wired for residencyKeepAliveSeconds after the last command;
  // an infinite keep-alive holds them while the backend lives. The watchdog
  // gives up on a command that runs longer than commandTimeoutSeconds.
  explicit MetalBackend(std::string metallibPath,
                        double residencyKeepAliveSeconds = kResidencyKeepAliveSeconds,
                        double commandTimeoutSeconds = kCommandTimeoutSeconds);
  ~MetalBackend();
  // Invoked before allocations and submissions; may throw to stop bootstrap.
  void setOperationGuard(std::function<void()> guard);
  // Asked between the slices of a ticket's wait(); true gives up the wait as
  // the watchdog does and marks the backend unhealthy: the process is
  // shutting down. Destroying or replacing a ticket still waits for its
  // command.
  void setWaitInterrupt(std::function<bool()> shuttingDown);
  void checkOperation() const;
  // Stop new submissions before teardown. Commands already committed to the
  // GPU retain their normal lifetime.
  void stop() noexcept;

  MetalBackend(const MetalBackend &) = delete;
  MetalBackend &operator=(const MetalBackend &) = delete;
  MetalBackend(MetalBackend &&) = delete;
  MetalBackend &operator=(MetalBackend &&) = delete;

  [[nodiscard]] const DeviceCapabilities &capabilities() const noexcept;

  // Every buffer the backend allocates belongs to one residency set, attached
  // to the command queue, until its memory is released or its last view is
  // gone. Metal by itself wires a buffer only while a command uses it and a
  // few seconds after, so memory pressure could compress idle state and the
  // next request would wait to get it back. A member is wired from
  // its allocation on until the keep-alive passes without a command, and
  // again from the next command: memory goes back to macOS when the engine
  // releases it, not when macOS chooses. Kernels may also reach a Shared
  // buffer only through its GPU address held in another buffer, as they
  // reach KV extents: residency makes it resident for every command, so no
  // command names it.
  [[nodiscard]] MetalBuffer
  allocateBuffer(uint64_t bytes, BufferStorage storage,
                 std::string_view label);

  [[nodiscard]] MetalBuffer view(const MetalBuffer &base, uint64_t offsetBytes,
                                 uint64_t lengthBytes) const;

  // Frees the memory of a buffer from allocateBuffer while no command is in
  // flight: it leaves the residency set and the accounting. Its views stay
  // valid handles of no memory: a command that binds one fails, and their
  // contents are null and GPU addresses 0, until restoreMemory allocates the
  // buffer's memory again, at another GPU address, its contents undefined
  // until written.
  void releaseMemory(const MetalBuffer &buffer);
  void restoreMemory(const MetalBuffer &buffer);

  // Encodes exactly one compute dispatch, commits it, waits for completion,
  // and reports both GPU and end-to-end wall time.
  [[nodiscard]] CommandTiming submit(const ComputeDispatch &dispatch);

  // Encodes an ordered dispatch list into one command buffer and commits it
  // without waiting. The completion callback only notifies host control
  // flow; command results and errors are consumed from the returned ticket.
  // A second command is rejected until wait() consumes the first ticket,
  // preserving the one-in-flight runtime invariant.
  [[nodiscard]] CommandTicket
  submitCommandAsync(std::span<const ComputeDispatch> dispatches,
                     CommandCompletion completion = {});

  // Startup: compiles now the pipelines a later submission of these
  // dispatches would compile. Validates them as submission does and throws
  // as it would, but encodes and commits nothing.
  void preparePipelines(std::span<const ComputeDispatch> dispatches);

  [[nodiscard]] MetalMemoryStats memoryStats() const noexcept;
  // Explicit safe-point refresh for memory admission/reclamation code. A
  // control-plane status query must use memoryStats() so it can never wait
  // behind an active Metal command.
  [[nodiscard]] MetalMemoryStats refreshMemoryStats() const noexcept;
  // True from a submission until its ticket has been consumed: while memory
  // the command reaches through addresses must stay allocated.
  [[nodiscard]] bool commandInFlight() const noexcept;
  [[nodiscard]] bool healthy() const noexcept;
  // Serving-loop check of the command in flight. Terminal results may invoke
  // completion here if the driver callback is delayed.
  // Timeout marks the backend unhealthy without releasing in-flight resources.
  // A synchronous ticket wait runs the same command watchdog.
  void checkHealth();
  [[nodiscard]] std::string unhealthyReason() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class BackendInstrumentation;
};

} // namespace splash::metal
