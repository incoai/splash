#import "ane/Handoff.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace splash::ane {
namespace {

// Where a job of the current command stands: queued until the event reaches
// its `ready`, started from then until it reports, then succeeded or failed.
enum class Phase : uint8_t { Queued, Started, Succeeded, Failed };
// A job of the current command: its phase, when the handoff learned that the
// event reached its `ready`, and once it succeeded, how long after that it
// reported.
struct Job final {
  Phase phase = Phase::Queued;
  AwakeClock::time_point started{};
  AwakeClock::duration ran{};
};

std::string seconds(AwakeClock::duration duration) {
  char text[32];
  std::snprintf(text, sizeof text, "%g", std::chrono::duration<double>(duration).count());
  return text;
}

} // namespace

struct Handoff::State final {
  State(metal::SharedEvent event, AwakeClock::duration bound) : event(std::move(event)), bound(bound) {}

  // Retires with `why` unless already retired, and raises the event to the
  // highest value allocated. Under `mutex`.
  void retire(std::string why) {
    if (!retired) {
      retired = true;
      reason = std::move(why);
    }
    event.signal(value);
    changed.notify_all();
  }
  // Fails the job of `phase` with `why`. Under `mutex`.
  void fail(Phase &phase, std::string why) {
    phase = Phase::Failed;
    retire(std::move(why));
  }
  // Fails the job of `phase`, which ran or reported past the bound, with
  // `why`; a handoff it retires is one restart() may start over. Under `mutex`.
  void overrun(Phase &phase, std::string why) {
    if (!retired) overran = true;
    fail(phase, std::move(why));
  }

  // The event reached the `ready` of job `id`: its bound starts.
  static void reached(std::shared_ptr<State> state, uint64_t id) noexcept {
    {
      std::lock_guard lock(state->mutex);
      const auto job = state->jobs.find(id);
      if (job == state->jobs.end() || job->second.phase != Phase::Queued) return;
      job->second.phase = Phase::Started;
      job->second.started = AwakeClock::now();
    }
    // dispatch_time(DISPATCH_TIME_NOW, ...) counts mach absolute time, which
    // stops while the Mac sleeps, as AwakeClock's CLOCK_UPTIME_RAW does.
    const auto bound = std::chrono::duration_cast<std::chrono::nanoseconds>(state->bound).count();
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, bound), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^{
                     expired(state, id);
                   });
  }
  // Job `id`'s bound passed.
  static void expired(const std::shared_ptr<State> &state, uint64_t id) noexcept {
    try {
      std::lock_guard lock(state->mutex);
      const auto job = state->jobs.find(id);
      if (job != state->jobs.end() && job->second.phase == Phase::Started)
        state->overrun(job->second.phase, "an evaluation did not complete within " + seconds(state->bound) + " s");
    } catch (...) {
    }
  }
  // The agent's report of job `id`; only its first counts.
  static void reported(const std::shared_ptr<State> &state, uint64_t id, bool success) noexcept {
    try {
      std::lock_guard lock(state->mutex);
      if (!state->outstanding.erase(id)) return;
      state->changed.notify_all();
      const auto found = state->jobs.find(id);
      if (found == state->jobs.end()) return;
      Job &job = found->second;
      if (job.phase == Phase::Succeeded || job.phase == Phase::Failed) return;
      if (!success) {
        state->fail(job.phase, "an evaluation failed");
        return;
      }
      if (job.phase == Phase::Started) job.ran = AwakeClock::now() - job.started;
      job.phase = Phase::Succeeded;
    } catch (...) {
    }
  }

  std::mutex mutex;
  std::condition_variable changed;
  const metal::SharedEvent event;
  const AwakeClock::duration bound;
  // The highest value of the event allocated.
  uint64_t value = 0;
  // The jobs of the command queued last, by id, until finish() or cancel().
  std::unordered_map<uint64_t, Job> jobs;
  // The jobs of any command the agent started and has not reported.
  std::unordered_set<uint64_t> outstanding;
  uint64_t lastJob = 0;
  bool retired = false;
  bool overran = false;
  std::string reason;
};

Handoff::Handoff(metal::MetalBackend &backend, AwakeClock::duration bound)
    : state_(std::make_shared<State>(backend.newSharedEvent(), bound)) {}

Handoff::~Handoff() {
  try {
    cancel("the handoff ended");
  } catch (...) {
  }
}

const metal::SharedEvent &Handoff::event() const noexcept { return state_->event; }

uint64_t Handoff::next() {
  std::lock_guard lock(state_->mutex);
  ++state_->value;
  if (state_->retired) state_->event.signal(state_->value);
  return state_->value;
}

void Handoff::queue(uint64_t ready, uint64_t done, const Start &start) {
  uint64_t id = 0;
  {
    std::lock_guard lock(state_->mutex);
    id = ++state_->lastJob;
    if (state_->retired) {
      state_->jobs.emplace(id, Job{Phase::Failed});
      return;
    }
    state_->jobs.emplace(id, Job{});
    state_->outstanding.insert(id);
  }
  std::string failure;
  try {
    start(state_->event, ready, done,
          [state = state_, id](bool success) { State::reported(state, id, success); });
    state_->event.notify(ready, [state = state_, id] { State::reached(state, id); });
    return;
  } catch (const std::exception &error) {
    failure = std::string("an evaluation could not start: ") + error.what();
  } catch (...) {
    failure = "an evaluation could not start";
  }
  std::lock_guard lock(state_->mutex);
  state_->outstanding.erase(id);
  state_->fail(state_->jobs.at(id).phase, std::move(failure));
}

std::optional<AwakeClock::duration> Handoff::finish() {
  std::optional<std::vector<AwakeClock::duration>> each = finishEach();
  if (!each) return std::nullopt;
  AwakeClock::duration ran{};
  for (const AwakeClock::duration &job : *each) ran += job;
  return ran;
}

std::optional<std::vector<AwakeClock::duration>> Handoff::finishEach() {
  State &state = *state_;
  std::unique_lock lock(state.mutex);
  const auto settled = [&] {
    bool succeeded = true;
    for (const auto &[id, job] : state.jobs) {
      if (job.phase == Phase::Failed) return true;
      succeeded &= job.phase == Phase::Succeeded;
    }
    return succeeded;
  };
  if (!state.changed.wait_until(lock, AwakeClock::now() + state.bound, settled))
    for (auto &[id, job] : state.jobs)
      if (job.phase != Phase::Succeeded)
        state.overrun(job.phase, "an evaluation did not report within " + seconds(state.bound) + " s of its command");
  // Job ids rise in the order queue() took the jobs.
  std::vector<std::pair<uint64_t, AwakeClock::duration>> jobs;
  std::optional<std::vector<AwakeClock::duration>> ran = std::vector<AwakeClock::duration>{};
  for (const auto &[id, job] : state.jobs) {
    if (job.phase != Phase::Succeeded) {
      ran.reset();
      break;
    }
    jobs.emplace_back(id, job.ran);
  }
  state.jobs.clear();
  if (!ran) return ran;
  std::ranges::sort(jobs);
  for (const auto &[id, duration] : jobs) ran->push_back(duration);
  return ran;
}

void Handoff::cancel(std::string reason) {
  State &state = *state_;
  std::unique_lock lock(state.mutex);
  state.retire(std::move(reason));
  state.jobs.clear();
  state.changed.wait_until(lock, AwakeClock::now() + state.bound, [&] { return state.outstanding.empty(); });
}

void Handoff::retire(std::string reason) {
  std::lock_guard lock(state_->mutex);
  state_->retire(std::move(reason));
}

bool Handoff::retired() const {
  std::lock_guard lock(state_->mutex);
  return state_->retired;
}

bool Handoff::idle() const {
  std::lock_guard lock(state_->mutex);
  return state_->outstanding.empty();
}

bool Handoff::overran() const {
  std::lock_guard lock(state_->mutex);
  return state_->overran;
}

bool Handoff::restart(metal::MetalBackend &backend) {
  {
    std::lock_guard lock(state_->mutex);
    if (!state_->retired || !state_->overran || !state_->outstanding.empty()) return false;
  }
  // Reports and notices still due on the old state find it alone.
  state_ = std::make_shared<State>(backend.newSharedEvent(), state_->bound);
  return true;
}

std::string Handoff::reason() const {
  std::lock_guard lock(state_->mutex);
  return state_->reason;
}

} // namespace splash::ane
