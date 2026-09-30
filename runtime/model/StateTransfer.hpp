#pragma once

#include <functional>
#include <memory>

namespace splash {

namespace model {
class DiskReservation;
}

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

// A staged write owns its disk result and drains IO before releasing backing.
// Payload is either one group block or a set of blocks at a restore boundary.
template <typename Payload> class StateWrite {
public:
  virtual ~StateWrite() = default;
  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual bool finish() = 0;
  [[nodiscard]] virtual const std::shared_ptr<const Payload> &
  state() const noexcept = 0;
};

template <typename Payload>
using StateWriter = std::function<std::unique_ptr<StateWrite<Payload>>(
    std::function<void()>, model::DiskReservation *)>;

// Preparing a direct snapshot neither allocates disk records nor starts IO.
// The source describes its exact backing for admission; it may borrow the
// lane's committed buffers and must not outlive the next model execution.
// write() stages all borrowed data before returning a ticket and may be
// retried synchronously after freeing quota. Never queue the plan itself.
struct SnapshotWritePlan final {
  std::shared_ptr<const RestoreState> source;
  StateWriter<RestoreState> write;
};

} // namespace splash
