#pragma once

#include <functional>
#include <memory>

namespace splash {

class StatePayload;
struct RestoreState;

// Completion is consumed on the engine thread; cancellation never makes
// backing reusable until ready() is true. Destruction drains outstanding
// accesses before the caller releases their backing.
class StateRestore {
public:
  virtual ~StateRestore() = default;
  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual bool finish() = 0;
  virtual void cancel() noexcept = 0;
  // After successful finish, before executing against the restored buffers.
  // Copies the complete immutable payload, even if execution skips part of it.
  // A null result means that cache storage could not be admitted.
  [[nodiscard]] virtual std::shared_ptr<const RestoreState> snapshot() = 0;
};

// A demotion in flight. The disk copy is usable from the start: a read
// queued behind the write observes the write's outcome. The source may be
// released as soon as the ticket exists; destruction drains the write.
class StateOffload {
public:
  virtual ~StateOffload() = default;
  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual bool finish() = 0;
  [[nodiscard]] virtual const std::shared_ptr<const StatePayload> &
  state() const noexcept = 0;
};

// A direct snapshot can write several independent group blocks in one bounded
// staging transaction. Publication still waits for every component to finish.
class SnapshotOffload {
public:
  virtual ~SnapshotOffload() = default;
  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual bool finish() = 0;
  [[nodiscard]] virtual const std::shared_ptr<const RestoreState> &
  state() const noexcept = 0;
};

// Preparing a direct snapshot neither allocates disk records nor starts IO.
// The source describes its exact backing for admission; it may borrow the
// lane's committed buffers and must not outlive the next model execution.
// write() stages all borrowed data before returning a ticket and may be
// retried synchronously after freeing quota. Never queue the plan itself.
struct SnapshotWritePlan final {
  using Writer =
      std::function<std::unique_ptr<SnapshotOffload>(std::function<void()>)>;
  std::shared_ptr<const RestoreState> source;
  Writer write;
};

} // namespace splash
