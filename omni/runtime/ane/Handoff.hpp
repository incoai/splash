#pragma once

#include "AwakeClock.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace splash::ane {

// The handoff of a Metal command's work to another agent, the Neural Engine,
// over one shared event, which only the GPU, the agent and this handoff
// raise. Each job of a command waits for the GPU to raise the event to its
// `ready`, and raises it to its `done`, which the GPU waits for in turn.
//
// A job has `bound` from the moment the event reaches its `ready` to report
// success. Any failure, the agent reporting one, failing to start the job or
// running past the bound, retires the handoff for its life: it raises the
// event at once to the highest value it has allocated, so that every wait of
// the GPU passes, and starts no job again. The event never decreases (Metal
// ignores a signal below its value, from the CPU, the GPU or the agent), so
// the release holds whatever the GPU or the agent signals after it; a job
// whose wait it meets runs on what its inputs hold, which nothing reads once
// its command's results are discarded.
//
// The engine thread calls every member; the event's notifications, the
// agent's reports and the bound's deadlines run on threads of their own and
// share the state this handoff owns, so that one arriving after it is gone
// finds nothing to do.
class Handoff final {
public:
  // An agent's report of a job: true once it ran, false if it failed. Only
  // its first call counts.
  using Report = std::function<void(bool success)>;
  // Starts a job on the agent: work that starts once `event` reaches `wait`
  // and raises it to `signal` when done, which reports through `report`. It
  // throws if it cannot start the job, whose report then does not count.
  using Start = std::function<void(const metal::SharedEvent &event, uint64_t wait, uint64_t signal,
                                   Report report)>;

  Handoff(metal::MetalBackend &backend, AwakeClock::duration bound);
  // Retires the handoff and waits at most one bound for the agent to report
  // the jobs it started. A job it never reports keeps what it references
  // itself: the Neural Engine's requests hold the surfaces they read and
  // write.
  ~Handoff();
  Handoff(const Handoff &) = delete;
  Handoff &operator=(const Handoff &) = delete;

  [[nodiscard]] const metal::SharedEvent &event() const noexcept;

  // A value of the event no step has taken, for a step of the command being
  // encoded; once retired, one the event has reached.
  [[nodiscard]] uint64_t next();

  // Starts a job of the command being encoded, before the command is
  // committed, through `start`, which this call does not keep. A job that
  // cannot start, or one queued once retired, has failed: this never throws
  // for the agent's faults.
  void queue(uint64_t ready, uint64_t done, const Start &start);
  // After the command has completed: if every job it queued succeeded, the
  // agent's time over them, each job's from the notice that the event reached
  // its `ready` to its report (none for a report that came first); none
  // otherwise. Waits at most one bound for their reports: a job that has not
  // reported by then has failed. Forgets the command's jobs either way.
  [[nodiscard]] std::optional<AwakeClock::duration> finish();
  // The command will not be committed: retires with `reason`, forgets its
  // jobs, and waits at most one bound for the agent to report those it
  // started.
  void cancel(std::string reason);
  // Retires with `reason` unless already retired.
  void retire(std::string reason);

  [[nodiscard]] bool retired() const;
  // Whether the agent has reported every job it started, so that none of
  // them still runs.
  [[nodiscard]] bool idle() const;
  // Why the handoff retired, the first failure's; empty until it does.
  [[nodiscard]] std::string reason() const;

private:
  struct State;
  std::shared_ptr<State> state_;
};

} // namespace splash::ane
