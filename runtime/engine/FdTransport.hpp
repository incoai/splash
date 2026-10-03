#pragma once

#include "engine/NativeRuntime.hpp"
#include "engine/Protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace splash::engine {

enum class NativeProcessExit : int {
  CleanEof = 0,
  ProtocolFailure = 64,
  EngineFailure = 70,
  IoFailure = 74,
};

// POSIX pipe transport for the native runtime child process. While run() is
// active a reader thread reads the input fd, which is nonblocking meanwhile,
// so the process keeps reading what the server writes while the loop runs a
// command or a control pass. Each iteration passes everything read so far to
// the loop before advancing the backend. Constraint-mask responses can
// therefore arrive while the scheduler-owned target-forward command is in
// flight; cancellation remains safe because provisional writes commit only
// after that command drains.
class FdTransport final {
public:
  // The reader stops reading once this much input waits for the loop: one
  // frame of the largest size the protocol accepts, so the reader can take a
  // whole request of any size while the loop runs. Until the loop takes it,
  // the pipe stops the writer. Tests set a smaller bound through TestConfig.
  static constexpr size_t kInputQueueBytes =
      protocol::kFrameHeaderBytes + protocol::kAbsoluteMaxFramePayloadBytes;

  explicit FdTransport(int inputFd, int outputFd);
  FdTransport(const FdTransport &) = delete;
  FdTransport &operator=(const FdTransport &) = delete;

  [[nodiscard]] NativeRuntime::ByteSink outputSink();
  // Runs between commands after a control notification. Returning true asks
  // for another run at the next command-free point after the loop wakes, so
  // a reclaim that transfers in flight held back continues when one of them
  // lands: its completion wakes the loop.
  using ControlHandler = std::function<bool()>;
  [[nodiscard]] std::function<void()> controlNotifier();
  void setControlHandler(ControlHandler handler);
  [[nodiscard]] NativeProcessExit run(NativeRuntime &loop);
  // Async-signal-safe. Asks run() to return CleanEof at its next iteration.
  // It does not wait for in-flight GPU work; the process owner bounds teardown.
  // An output write the signal interrupts fails instead of resuming.
  void requestShutdown() noexcept;
  [[nodiscard]] bool shutdownRequested() const noexcept;
  // The longest one control pass and one tick of run() have taken. Read on
  // the loop thread (status). The reader keeps reading input meanwhile, so a
  // stalled request write shows only that the process stopped reading; a
  // stuck loop is detected by the server's status round trip
  // (server/runtime.py `_probe_liveness`).
  [[nodiscard]] double maxTickMilliseconds() const noexcept;
  // Why run() ended with IoFailure, or with EngineFailure when the input
  // reader ran out of memory; empty when the loop chose the exit.
  [[nodiscard]] const std::string &failure() const noexcept;

private:
  struct LoopWake;
  class InputReader;
  void writeAll(std::span<const uint8_t> bytes) const;

  int inputFd_ = -1;
  int outputFd_ = -1;
  size_t inputQueueBytes_ = 0;
  std::shared_ptr<LoopWake> wake_;
  ControlHandler controlHandler_;
  double maxTickMilliseconds_ = 0.0;
  std::string failure_;
};

} // namespace splash::engine
