#pragma once

#include "metal/DeviceCapabilities.hpp"

#include <array>
#include <cstddef>
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
  Capacity, // Unclassified capacity refusal from an alternate backing.
  EngineBudget,
  HostPressure,
  DriverRejected,
};

struct AllocationResult final {
  AllocationFailure failure;
  AllocationResult(bool granted)
      : failure(granted ? AllocationFailure::None
                        : AllocationFailure::Capacity) {}
  AllocationResult(AllocationFailure reason) : failure(reason) {}
  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == AllocationFailure::None;
  }
};

[[nodiscard]] constexpr const char *allocationFailureName(
    AllocationFailure failure) noexcept {
  switch (failure) {
  case AllocationFailure::None: return "none";
  case AllocationFailure::Capacity: return "allocation capacity unavailable";
  case AllocationFailure::EngineBudget: return "engine memory budget exceeded";
  case AllocationFailure::HostPressure: return "host memory reserve protected";
  case AllocationFailure::DriverRejected:
    return "Metal driver rejected allocation";
  }
  return "unknown allocation failure";
}

// Physical allocators use this callback to obtain engine-governed headroom
// without depending on the engine policy type. The operation runs while the
// caller's reservation is held and returns false without side effects when
// admission is denied.
using AllocationAdmission =
    std::function<AllocationResult(uint64_t, const std::function<void()> &)>;

enum class BufferStorage {
  Shared,
  Private,
};

class MetalBackend;
class CommandTicket;
class SparseHeap;

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
  // Returns nullptr for private buffers. The pointer covers this view only.
  [[nodiscard]] void *contents() const noexcept;
  // Allocation identity and exact view range, including Private storage.
  // This compares metadata only; it never maps or reads device contents.
  [[nodiscard]] bool sameView(const MetalBuffer &other) const noexcept;

private:
  struct Impl;
  explicit MetalBuffer(std::shared_ptr<Impl> impl);

  std::shared_ptr<Impl> impl_;

  friend class MetalBackend;
};

// Move-only ownership of one private placement heap. Destroying an empty heap
// is the operation that actually returns sparse KV backing to the OS; merely
// unmapping tiles is not sufficient on Apple Silicon.
class SparseHeap final {
public:
  SparseHeap();
  ~SparseHeap();
  SparseHeap(const SparseHeap &) = delete;
  SparseHeap &operator=(const SparseHeap &) = delete;
  SparseHeap(SparseHeap &&) noexcept;
  SparseHeap &operator=(SparseHeap &&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] uint64_t sizeBytes() const noexcept;
private:
  struct Impl;
  explicit SparseHeap(std::shared_ptr<Impl> impl);

  std::shared_ptr<Impl> impl_;

  friend class MetalBackend;
};

struct SparseMapping {
  MetalBuffer buffer;
  uint64_t bufferOffsetBytes = 0;
  uint64_t sizeBytes = 0;
  uint64_t heapOffsetBytes = 0;
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

// GPU time of one dispatch replayed as its own command while profiling.
struct DispatchTiming {
  std::string pipelineName;
  double gpuSeconds = 0.0;
};

// Move-only ownership of one submitted Metal command, including any resource
// dependency wait before GPU commitment. Completion is signalled
// without blocking the submitting thread; wait() is normally called only
// after the host event loop receives the completion notification.
// Destroying or replacing an unfinished ticket waits for GPU completion and
// retains its allocations throughout that wait.
class CommandTicket final {
public:
  CommandTicket();
  ~CommandTicket();
  CommandTicket(const CommandTicket &) = delete;
  CommandTicket &operator=(const CommandTicket &) = delete;
  CommandTicket(CommandTicket &&) noexcept;
  CommandTicket &operator=(CommandTicket &&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] uint64_t sequence() const noexcept;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] CommandTiming wait();

private:
  struct State;
  explicit CommandTicket(std::shared_ptr<State> state);

  std::shared_ptr<State> state_;

  friend class MetalBackend;
};

using CommandCompletion = std::function<void(uint64_t sequence)>;

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
  // Highest sampled device.currentAllocatedSize. Sampling occurs after
  // allocations and pipeline creation, and at the pre-commit, post-commit,
  // scheduled and completed command boundaries.
  uint64_t devicePeakAllocatedBytes = 0;

  // Placement-sparse buffers reserve virtual GPU address space without
  // committing it. Resident bytes count live placement heaps, which are the
  // reclaimable physical unit.
  uint64_t sparseVirtualBytes = 0;
  uint64_t sparseResidentBytes = 0;
  uint64_t peakSparseResidentBytes = 0;

  // Peak of the simultaneous dense + sparse physical allocations. The two
  // component peaks above can occur at different times and must not be added.
  uint64_t peakResidentBytes = 0;

  // Placement-sparse unmapping is asynchronous. The heap of an unmapped
  // extent stays retained, and counted resident, until the sparse queue
  // reports that unmap complete; at most one unmap is outstanding. Durations
  // are observed at the next backend safe point, not measured by the kernel.
  uint64_t sparseTileBytes = 0;
  uint64_t pendingSparseUnmaps = 0;
  uint64_t completedSparseUnmaps = 0;
  double lastSparseUnmapSeconds = 0.0;
  double maxSparseUnmapSeconds = 0.0;
  double pendingSparseUnmapSeconds = 0.0;
  // Host-side dependency wait before committing a compute command.
  uint64_t sparseMapWaitEvent = 0;
  double pendingSparseMapWaitSeconds = 0.0;
  double lastSparseMapWaitSeconds = 0.0;
  double maxSparseMapWaitSeconds = 0.0;
};

class MetalBackendError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// A normal capacity failure. Callers may evict cache or return a retryable
// admission error; the Metal backend remains healthy.
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

// Permits exactly one submitted-but-not-applied command on its command queue.
class MetalBackend final {
public:
  explicit MetalBackend(std::string metallibPath,
                        double commandTimeoutSeconds = 120.0);
  ~MetalBackend();
  // Invoked before allocations and submissions; may throw to stop bootstrap.
  void setOperationGuard(std::function<void()> guard);
  void checkOperation() const;
  // Stop new submissions and cancel dependency waits before teardown.
  // Commands already committed to the GPU retain their normal lifetime.
  void stop() noexcept;

  MetalBackend(const MetalBackend &) = delete;
  MetalBackend &operator=(const MetalBackend &) = delete;
  MetalBackend(MetalBackend &&) noexcept;
  MetalBackend &operator=(MetalBackend &&) noexcept;

  [[nodiscard]] const DeviceCapabilities &capabilities() const noexcept;
  // Digest of the immutable bytes used to create this backend's library,
  // independent of later replacement or removal of its original file path.
  [[nodiscard]] const std::array<uint8_t, 32> &metallibSha256() const noexcept;

  [[nodiscard]] MetalBuffer
  allocateBuffer(uint64_t bytes, BufferStorage storage = BufferStorage::Shared,
                 std::string_view label = {});

  // Private sparse buffers use the shared 64 KiB tile ABI. Per-layer scale
  // ranges must stay tile-aligned; fewer dirty tiles reduce unmap cost.
  [[nodiscard]] MetalBuffer
  allocatePlacementSparseBuffer(uint64_t virtualBytes, uint64_t sparsePageBytes,
                                std::string_view label = {});
  [[nodiscard]] SparseHeap allocatePlacementHeap(uint64_t physicalBytes,
                                                 uint64_t sparsePageBytes,
                                                 std::string_view label = {});

  // Mapping is ordered before the next compute command by an internal
  // Metal event. Unmapping is only legal with no submitted command. It is
  // asynchronous: the backend takes ownership of the mapped heap and keeps it
  // resident until the sparse queue reports the unmap complete, which is
  // observed at later safe points. Only one unmap may be outstanding; a
  // second call first waits for the previous unmap. Unmapping GPU-written
  // tiles is kernel work that can stall the whole GPU stack when issued in
  // bursts, so callers pace releases with sparseUnmapPending().
  void mapSparse(const SparseHeap &heap,
                 std::span<const SparseMapping> mappings);
  void unmapSparse(std::span<const SparseMapping> mappings, SparseHeap &&heap);
  // Reports whether an unmap is still outstanding, reaping a completed one
  // (releasing its heap) when no command is being encoded.
  // Never blocks. Marks the backend unhealthy once the outstanding unmap has
  // been pending longer than the drain's bounded wait.
  [[nodiscard]] bool sparseUnmapPending() noexcept;
  // Placement-sparse page (tile) size shared with the KV page layout.
  static constexpr uint64_t kPlacementSparsePageBytes = 64 * 1024;
  // Blocks until the outstanding unmap, if any, has completed. A bounded
  // timeout marks the backend unhealthy; use only at startup and shutdown.
  void drainSparseUnmaps();

  // Wraps page-aligned shared memory without copying it. The lifetime token
  // is retained by Metal's deallocator, including any internal buffer owners
  // that outlive our C++ views and completed tickets.
  [[nodiscard]] MetalBuffer wrapSharedMemory(void *address, uint64_t bytes,
                                             std::shared_ptr<void> lifetime,
                                             std::string_view label = {});
  [[nodiscard]] MetalBuffer view(const MetalBuffer &base, uint64_t offsetBytes,
                                 uint64_t lengthBytes) const;

  // Encodes exactly one compute dispatch, commits it, waits for completion,
  // and reports both GPU and end-to-end wall time.
  [[nodiscard]] CommandTiming submit(const ComputeDispatch &dispatch);

  // Encodes an ordered dispatch list into one command buffer and waits for it.
  [[nodiscard]] CommandTiming
  submitCommand(std::span<const ComputeDispatch> dispatches);

  // Encodes and commits without waiting. The completion callback only
  // notifies host control flow; command results and errors are consumed from
  // the returned ticket. A second command is rejected until wait() consumes
  // the first ticket, preserving the one-in-flight runtime invariant.
  [[nodiscard]] CommandTicket
  submitAsync(const ComputeDispatch &dispatch,
              CommandCompletion completion = {});
  [[nodiscard]] CommandTicket
  submitCommandAsync(std::span<const ComputeDispatch> dispatches,
                     CommandCompletion completion = {});

  // Development profiling replays a multi-dispatch command synchronously,
  // one dispatch per command buffer. Even submitCommandAsync() then blocks,
  // invokes completion inline and returns an already-completed ticket.
  // Production serving leaves this disabled. Benchmarks read and clear the
  // per-dispatch timings with takeDispatchProfile().
  void setDispatchProfiling(bool enabled) noexcept;
  [[nodiscard]] std::vector<DispatchTiming> takeDispatchProfile();

  [[nodiscard]] MetalMemoryStats memoryStats() const noexcept;
  // Explicit safe-point refresh for memory admission/reclamation code. A
  // control-plane status query must use memoryStats() so it can never wait
  // behind an active Metal command.
  [[nodiscard]] MetalMemoryStats refreshMemoryStats() const noexcept;
  [[nodiscard]] uint64_t submissionCount() const noexcept;
  [[nodiscard]] size_t pipelineCount() const noexcept;
  [[nodiscard]] bool healthy() const noexcept;
  // Nonblocking serving-loop check of actual GPU commands and pending unmaps.
  // Timeout marks the backend unhealthy without releasing in-flight resources.
  void checkHealth();
  [[nodiscard]] bool needsHealthCheck() const noexcept;
  [[nodiscard]] std::string unhealthyReason() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::metal
