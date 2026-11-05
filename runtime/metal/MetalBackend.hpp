#pragma once

#include "Supervision.hpp"
#include "metal/Allocation.hpp"
#include "metal/DeviceCapabilities.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef __OBJC__
@class NSError;
#endif

namespace splash::metal {

enum class BufferStorage {
  Shared,
  Private,
};

// How kernels reach a buffer: bound to the dispatches that use it, or only
// through its GPU address held in a table, another buffer. A Bound buffer
// may also be reached through a dispatch's address table (AddressTable),
// which the backend writes per dispatch. The table that holds an Addressed
// buffer's address names it (MetalBackend::setTableReach), and the backend
// declares it for each encoder that uses that table.
enum class BufferReach {
  Bound,
  Addressed,
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
  // The base allocation's MTLResource.allocatedSize, as memoryStats() counts
  // it; views of one allocation all report it.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;
  [[nodiscard]] BufferStorage storage() const noexcept;
  // Returns nullptr for private buffers and released memory
  // (MetalBackend::releaseMemory). The pointer covers this view only.
  [[nodiscard]] void *contents() const noexcept;
  // GPU address of the view's first byte, for kernels that reach a buffer
  // through an address a table holds; zero for released memory. Only a
  // buffer allocated BufferReach::Addressed has one, and only a table that
  // names it (MetalBackend::setTableReach) makes it resident for commands.
  [[nodiscard]] uint64_t gpuAddress() const;
  // Allocation identity and exact view range, including Private storage.
  // This compares metadata only; it never maps or reads device contents.
  [[nodiscard]] bool sameView(const MetalBuffer &other) const noexcept;

private:
  struct Impl;
  explicit MetalBuffer(std::shared_ptr<Impl> impl);

  std::shared_ptr<Impl> impl_;

  friend class MetalBackend;
};

// A shared event another agent, such as the Neural Engine, waits on or
// signals. Copies name the same event. nativeHandle() is its
// id<MTLSharedEvent> for Objective-C++ callers. Its value never decreases:
// Metal ignores a signal below it, from the CPU as from the GPU.
class SharedEvent final {
public:
  SharedEvent();
  ~SharedEvent();
  SharedEvent(const SharedEvent &);
  SharedEvent &operator=(const SharedEvent &);
  SharedEvent(SharedEvent &&) noexcept;
  SharedEvent &operator=(SharedEvent &&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] void *nativeHandle() const noexcept;
  // Raises the event to `value` from the CPU.
  void signal(uint64_t value) const noexcept;
  // Calls `callback` once, when the event reaches `value`, promptly if it
  // already has. Callbacks run on a serial dispatch queue of the backend
  // that created the event, one at a time and never within notify(); they
  // must not throw.
  void notify(uint64_t value, std::function<void()> callback) const;

private:
  struct Impl;
  explicit SharedEvent(std::shared_ptr<Impl> impl);

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

// A grid the GPU reads as the dispatch starts, so an earlier dispatch of the
// same command may write it: three uint32 threadgroup counts, x, y and z
// (MTLDispatchThreadgroupsIndirectArguments), at offsetBytes into buffer and
// at a multiple of 4 bytes into its allocation.
struct IndirectGrid {
  MetalBuffer buffer;
  uint64_t offsetBytes = 0;
};

// Buffers a dispatch reaches through their GPU addresses: the backend writes
// each one's address at encode time, in order, as a constant ulong table at
// `index` (setBytes, so at most 512 of them), and makes their memory resident
// for the encoder's dispatches from this one on (useResources, for reading,
// or reading and writing when `writes`), with what the bytes of any of them
// reach as a table (MetalBackend::setTableReach).
// Addresses are the buffers' when the command is encoded, so a buffer
// released and restored since the graph was built is reached where it is
// now; released memory is refused like a binding of it. Dispatches that
// reach the same buffers share one list, which the backend resolves once per
// command.
struct AddressTable {
  uint32_t index = 0;
  std::shared_ptr<const std::vector<MetalBuffer>> buffers; // none when null
  bool writes = false;
};

struct ComputeDispatch {
  std::string pipelineName;
  std::vector<BufferBinding> buffers;
  std::vector<BytesBinding> bytes;
  // With a buffer in indirectThreadgroups the dispatch runs the grid it
  // holds, which its writer keeps within threadgroups.
  DispatchSize threadgroups;
  DispatchSize threadsPerThreadgroup;
  IndirectGrid indirectThreadgroups = {};
  AddressTable addresses = {};
};

// A host step's round trip on the host clock (AwakeClock, the time base of
// the GPU's command buffer times): from the GPU end of the work before the
// step to the runner taking it up, its work, and from its return to the GPU
// start of the work after it.
struct HostStepTiming {
  double pickupSeconds = 0.0;
  double workSeconds = 0.0;
  double resumeSeconds = 0.0;
};

// What the GPU does through a host step (EventStep::Kind::Host): it idles,
// or one GPU thread spins through the step and keeps it awake.
enum class HostStepGpu : uint8_t {
  Idle,
  Awake,
};

// Orders a command against another agent, such as the Neural Engine, after
// the first `before` of its dispatches. A Signal raises the event to `value`
// once all earlier work has completed; a Wait holds all later work until the
// event reaches `value`. A command is split into Metal command buffers at its
// signals, so a signal is never held back behind the dispatches that follow
// it, and a signal is delivered even when the work before it fails.
//
// Metal fails a wait (a Wait, or the work after a host step) still unmet
// supervision::kGpuWaitLimit after its queue reached it ("Caused GPU Timeout
// Error"), and the command with it. The queue reaches a command's first wait
// as the command is submitted, whatever work runs ahead of it, and each later
// one once the wait before it was met: the work between two waits, with the
// agent's or the step's part, must end within it. Longer work, such as the
// encodes of a prefill command's images, runs as a command of its own.
//
// A Host step runs `work` on the CPU once all earlier work has completed and
// holds all later work until it returned, through two shared events the
// backend creates for the command's host steps, one the GPU raises and one the
// backend's runner raises, so that no write of one side puts back a value of
// the other (the step's event and value are unused): the work reads what the
// GPU wrote before it and writes Shared buffers the GPU reads after it. With
// HostStepGpu::Awake, one GPU
// thread spins through the step so the GPU stays awake and starts the work
// after it at once and at full clocks: for steps about as short as the GPU's
// wake-up (a streamed decode load), not for long ones, where the spin saves
// little. The spin, host_step_wait of host-step.metallib beside the
// backend's metallib, sees the CPU end it through a memory scope Metal's
// public headers leave out, so the backend checks, before the first command
// with an awake step, that it does; without that library, or if it does not,
// awake steps run as plain ones, with a warning. The steps of the committed
// commands run in order on one thread of the backend's while the submitting
// thread goes on, so `work` must not call the backend: it touches buffer
// contents and its own state only. It may also run after submitCommandAsync
// or wait() threw (its encoding failed after its first buffers were committed,
// or the watchdog or a shutdown gave the command up), so it owns what it
// touches. A step is skipped when the command buffer before it failed, and
// so is every later step of its command, as are those after a step whose
// work threw: a later step may build on what an earlier one did (an expert
// load's later rounds on its first), and a command that failed is given up.
// The GPU work after a skipped step still runs, on what its buffers held, as
// all work after a failed buffer runs on what it left, and the command fails
// with the first failure; a step whose work threw names it (its ticket's
// wait() throws its message). The failure also names, of the steps that ran,
// the longest work and the longest pickup (from the end of the GPU work
// before a step to its start), so that a buffer Metal timed out tells which
// step held its wait. Once a command completed without a failure, `timed`
// gets the round trip of each of its host steps (HostStepTiming) on Metal's
// completion thread; one that throws marks the backend unhealthy, as a
// completion callback that throws does. A command run under dispatch
// profiling reports no round trip. Once the work returned, `then` runs on the
// runner right after the step let the work after it go: short CPU work the
// GPU need not wait for and that would slow its start, such as reads begun in
// the background; the runner takes the next step once it returned. The
// command may have completed by then, so `then` must not throw: one that does
// marks the backend unhealthy.
struct EventStep {
  enum class Kind : uint8_t { Signal, Wait, Host };

  size_t before = 0;
  SharedEvent event;
  uint64_t value = 0;
  Kind kind = Kind::Signal;
  std::function<void()> work{};
  std::function<void(const HostStepTiming &)> timed{};
  HostStepGpu gpu = HostStepGpu::Idle;
  std::function<void()> then{};
};

// What one submission encodes: its dispatches in order, and the event steps
// between them in the order of their `before`.
struct Command {
  std::span<const ComputeDispatch> dispatches;
  std::span<const EventStep> events;
};

struct CommandTiming {
  // From the GPU start of a command's first Metal command buffer to the end
  // of its last: a command split at event signals (EventStep) also counts the
  // time its later buffers wait for the other agent or its host steps.
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
  // Its highest value since the backend was made or its peaks restarted
  // (MetalBackend::restartPeaks).
  uint64_t peakAllocatedBytes = 0;

  // Most recently sampled Metal device-wide process counter. Allocation and
  // command lifecycle boundaries refresh it; status reads never synchronize
  // with an in-flight GPU command.
  uint64_t deviceCurrentAllocatedBytes = 0;
  // Highest sampled device.currentAllocatedSize since then. Sampled after
  // allocations and pipeline creation, on host-side retirement and whenever
  // admission refreshes the stats.
  uint64_t devicePeakAllocatedBytes = 0;
};

#ifdef __OBJC__
// Whether a failed command buffer's error says the GPU had too little memory
// for it (MTLCommandBufferErrorDomain, MTLCommandBufferErrorOutOfMemory).
[[nodiscard]] bool commandOutOfMemory(NSError *error) noexcept;
#endif

// The capabilities a backend reads, without loading kernels or allocating:
// enough to refuse an unsupported Mac before a model is downloaded.
[[nodiscard]] DeviceCapabilities probeDeviceCapabilities();

// How long a command may go without progress before the backend gives up on
// it by default, in time the Mac is awake (AwakeClock): from its commit, and
// then from each of its host steps as the GPU work before it ends and as it
// returns (CommandWatchdog).
inline constexpr double kCommandTimeoutSeconds =
    std::chrono::duration<double>(supervision::kCommandTimeout).count();
// How long every buffer stays wired after the last command by default (see
// allocateBuffer), also the default of the engine's idle release
// (RuntimeResourcesConfig::idleReleaseSeconds).
inline constexpr double kResidencyKeepAliveSeconds = 600.0;
static_assert(kCommandTimeoutSeconds > 0.0 && kResidencyKeepAliveSeconds > 0.0);

// Permits exactly one submitted-but-not-applied command on its command queue.
// One thread submits, allocates, names tables' reach (setTableReach) and
// looks up pipelines; checkHealth(), healthy(), unhealthyReason(),
// memoryStats() and commandInFlight() may be called from any thread.
class MetalBackend final {
public:
  // Buffers stay wired for residencyKeepAliveSeconds after the last command;
  // an infinite keep-alive holds them while the backend lives. The watchdog
  // gives up on a command that goes longer than commandTimeoutSeconds without
  // progress (kCommandTimeoutSeconds).
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

  // Every buffer the backend allocates belongs to one residency set until its
  // memory is released or its last view is gone. Metal by itself wires a
  // buffer only while a command uses it and a few seconds after, so memory
  // pressure could compress idle state and the next request would wait to get
  // it back. A member is wired from its allocation on until the keep-alive
  // passes without a command, and again from the next command: memory goes
  // back to macOS when the engine releases it, not when macOS chooses. A
  // command makes resident only the memory it uses: the buffers its
  // dispatches bind or list (AddressTable), and the Addressed buffers the
  // tables among those reach (setTableReach), declared on each encoder that
  // uses them. Nothing is resident for the queue as a whole, so memory no
  // command uses costs no command buffer anything.
  [[nodiscard]] MetalBuffer
  allocateBuffer(uint64_t bytes, BufferStorage storage,
                 std::string_view label,
                 BufferReach reach = BufferReach::Bound);
  // A buffer as allocateBuffer makes it, its memory released from the start
  // (releaseMemory): its views are handles of no memory until restoreMemory
  // allocates it.
  [[nodiscard]] MetalBuffer
  reserveBuffer(uint64_t bytes, BufferStorage storage,
                std::string_view label,
                BufferReach reach = BufferReach::Bound);
  // A Shared buffer over whole pages of memory another agent also reads or
  // writes, such as an IOSurface of the Neural Engine, without a copy. Metal
  // keeps `owner` until it lets the buffer go; the memory is the owner's, so
  // releaseMemory refuses it.
  [[nodiscard]] MetalBuffer wrapSharedMemory(void *address, uint64_t bytes,
                                             std::shared_ptr<void> owner,
                                             std::string_view label);

  [[nodiscard]] MetalBuffer view(const MetalBuffer &base, uint64_t offsetBytes,
                                 uint64_t lengthBytes) const;
  // An event at value 0, whose notify() callbacks run on this backend's
  // event queue.
  [[nodiscard]] SharedEvent newSharedEvent();

  // Frees the memory of a buffer from allocateBuffer while no command is in
  // flight: it leaves the residency set and the accounting at once. Its
  // views stay valid handles of no memory: a command that binds one fails,
  // and their contents are null and GPU addresses 0, until restoreMemory
  // allocates the buffer's memory again, at another GPU address, its
  // contents undefined until written.
  void releaseMemory(const MetalBuffer &buffer);
  void restoreMemory(const MetalBuffer &buffer);
  // Names the Addressed buffers whose GPU addresses the bytes of `table`, a
  // view of a Shared buffer, hold: each encoder that uses the table makes
  // them resident (useResources, for reading and writing) as the command is
  // encoded. An encoder uses the table when its dispatches bind or list
  // (AddressTable) a view holding any of those bytes, or another table
  // reaches such a view: a kernel may follow an address to the entries of
  // the table it names, so what an encoder makes resident brings what it
  // reaches. It replaces what the same bytes named before, so the owner
  // calls it whenever it rewrites the table's entries, for every buffer they
  // reach. The table holds them weakly: memory released or freed since is
  // not declared, as no command may reach it then. MetalBackendError for
  // another backend's buffers, a table that is not Shared or a reached
  // buffer that is not Addressed.
  void setTableReach(const MetalBuffer &table, std::span<const MetalBuffer> reached);

  // Encodes exactly one compute dispatch, commits it, waits for completion,
  // and reports both GPU and end-to-end wall time.
  [[nodiscard]] CommandTiming submit(const ComputeDispatch &dispatch);

  // Encodes a command into Metal command buffers, one plus one per buffer its
  // event steps add (EventStep), and commits each as its encoding ends,
  // without waiting: the GPU runs the work before an event step while the
  // work after it is still encoded. The completion callback only notifies
  // host control flow; command results and errors are consumed from the
  // returned ticket, which reports the error of the first of its buffers that
  // failed. A second command is rejected until wait() consumes the first
  // ticket, preserving the one-in-flight runtime invariant.
  [[nodiscard]] CommandTicket
  submitCommandAsync(const Command &command, CommandCompletion completion = {});
  // A command of these dispatches without event steps: one command buffer.
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
  // Starts both peaks again from what is allocated now, sampling the device:
  // what memory held before, such as a load the caller discarded, no longer
  // counts. Called where nothing else allocates.
  void restartPeaks() noexcept;
  // True from a submission until its ticket has been consumed: while memory
  // the command reaches through addresses must stay allocated. It stays true
  // for a command whose committed buffers the backend gave up (an encoding
  // failure after a commit, or a watchdog timeout): they may still run.
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
