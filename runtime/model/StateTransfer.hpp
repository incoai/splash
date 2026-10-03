#pragma once

#include <memory>

namespace splash {

class CompositeState;

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
  [[nodiscard]] virtual std::shared_ptr<const CompositeState> snapshot() = 0;
};

// A demotion in flight. The disk copy is usable from the start: a read
// queued behind the write observes the write's outcome. The source may be
// released as soon as the ticket exists; destruction drains the write.
class StateOffload {
public:
  virtual ~StateOffload() = default;
  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual bool finish() = 0;
  [[nodiscard]] virtual const std::shared_ptr<const CompositeState> &
  state() const noexcept = 0;
};

} // namespace splash
