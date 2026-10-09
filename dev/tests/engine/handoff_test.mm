// ane::Handoff (runtime/ane/Handoff.mm) between Metal commands and an agent the CPU plays, under Metal's shader
// validation. For each way the agent can behave, a command of eight layers, each its GPU work, a signal of the layer's
// `ready`, more GPU work and a wait for its `done`, completes within the handoff's bound of the fault and far from
// Metal's own limit, and leaves the backend healthy; finish() reports whether every job succeeded, and how long the
// agent took over them; a failure stops the handoff for good, and the commands after it run on the GPU alone; while
// the agent behaves, the GPU never passes a layer's wait before the agent has seen its ready. cancel() and the
// destructor return within the bound, and a report after the handoff is gone does nothing. The premise the handoff
// stands on: a shared event never decreases, whether the CPU or the GPU signals below its value, and a GPU wait on a
// value it has passed completes at once.
#include "AwakeClock.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ane/Handoff.hpp"
#include "metal/MetalBackend.hpp"

#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace splash;
using metal::Command;
using metal::ComputeDispatch;
using metal::EventStep;
using metal::MetalBackend;
using metal::MetalBuffer;
using metal::SharedEvent;
using test::require;

constexpr auto kBound = std::chrono::milliseconds(100);
// How far past the bound a command may complete: GPU work, notifications and validation's overhead.
constexpr auto kSlack = std::chrono::milliseconds(250);
// The watchdog's limit: a command still waiting at the ticket's first watchdog check, after one second, fails the
// test there, well before Metal fails a command buffer whose wait stays unmet (5 s).
constexpr double kCommandTimeoutSeconds = 1.0;
constexpr uint32_t kLayers = 8;

double milliseconds(AwakeClock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

// The value `event` has reached.
uint64_t reached(const SharedEvent &event) {
  return [(__bridge id<MTLSharedEvent>)event.nativeHandle() signaledValue];
}

// Whether `event` reaches `value` within `limit`.
bool awaitValue(const SharedEvent &event, uint64_t value, std::chrono::milliseconds limit) {
  auto reachedValue = std::make_shared<std::promise<void>>();
  std::future<void> future = reachedValue->get_future();
  event.notify(value, [reachedValue] { reachedValue->set_value(); });
  return future.wait_for(limit) == std::future_status::ready;
}

// How the agent treats the faulty job: as every other one (run, raise the event to the job's `done`, report
// success), report failure from inside its start before the GPU's ready, report failure once the ready is reached,
// throw from its start, never run nor report, run three bounds late, or report success and then failure.
enum class Behaviour { Succeed, FailBeforeReady, FailAfterReady, ThrowAtStart, Never, Late, Twice };

const char *name(Behaviour behaviour) {
  switch (behaviour) {
  case Behaviour::Succeed: return "an agent that succeeds";
  case Behaviour::FailBeforeReady: return "a failure before the ready";
  case Behaviour::FailAfterReady: return "a failure after the ready";
  case Behaviour::ThrowAtStart: return "a start that throws";
  case Behaviour::Never: return "a job never run";
  case Behaviour::Late: return "a job run late";
  case Behaviour::Twice: return "a report twice";
  }
  return "";
}

// The agent: a thread that runs the jobs it was given in order, each once the event reaches its wait and taking
// `pace` from then, as the Neural Engine does. Its `faulty`-th job, counted from 0 over its life, takes `behaviour`.
// The GPU work it can see is the first word of `markers`, which each layer's work before its ready adds one to, and
// word 1 + k, which the work after layer k's wait sets.
class Agent final {
public:
  Agent(Behaviour behaviour, uint32_t faulty, const MetalBuffer &markers,
        std::chrono::milliseconds pace = std::chrono::milliseconds(0))
      : behaviour_(behaviour), faulty_(faulty), pace_(pace),
        markers_(static_cast<volatile uint32_t *>(markers.contents())), thread_([this] { run(); }) {}
  ~Agent() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_all();
    thread_.join();
  }

  [[nodiscard]] ane::Handoff::Start start() {
    return [this](const SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report report) {
      std::unique_lock lock(mutex_);
      const uint32_t index = started_++;
      if (index == faulty_ && behaviour_ == Behaviour::ThrowAtStart)
        throw std::runtime_error("the agent takes no job " + std::to_string(index));
      if (index == faulty_ && behaviour_ == Behaviour::FailBeforeReady) {
        lock.unlock();
        report(false);
        return;
      }
      jobs_.push_back({index, event, wait, signal, std::move(report)});
      changed_.notify_all();
    };
  }
  // The jobs it was started for, and those it reported.
  [[nodiscard]] uint32_t started() const {
    std::lock_guard lock(mutex_);
    return started_;
  }
  [[nodiscard]] uint32_t reported() const { return reported_; }
  // Whether, for every job before `count` it ran, the GPU had done the work before its ready and not passed its
  // wait when the agent saw the ready. Job k is layer k % kLayers of a command, whose markers start at 0.
  [[nodiscard]] bool ordered(uint32_t count) const {
    std::lock_guard lock(mutex_);
    for (const Seen &seen : seen_)
      if (seen.index < count && (seen.before != seen.index % kLayers + 1 || seen.passed)) return false;
    return true;
  }

private:
  struct Job {
    uint32_t index;
    SharedEvent event;
    uint64_t wait, signal;
    ane::Handoff::Report report;
  };
  // What the GPU had done when the agent saw job `index`'s ready.
  struct Seen {
    uint32_t index, before;
    bool passed;
  };

  void run() {
    for (;;) {
      Job job;
      {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
        if (jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      if (!awaitValue(job.event, job.wait, std::chrono::seconds(5))) continue;
      const uint32_t layer = job.index % kLayers;
      {
        std::lock_guard lock(mutex_);
        seen_.push_back({job.index, markers_[0], markers_[1 + layer] != 0});
      }
      if (job.index == faulty_) {
        switch (behaviour_) {
        case Behaviour::FailAfterReady:
          report(job, false);
          continue;
        case Behaviour::Never: continue;
        case Behaviour::Late: std::this_thread::sleep_for(3 * kBound); break;
        case Behaviour::Twice:
          job.event.signal(job.signal);
          report(job, true);
          report(job, false);
          continue;
        default: break;
        }
      }
      std::this_thread::sleep_for(pace_);
      job.event.signal(job.signal);
      report(job, true);
    }
  }
  // Counts the report first: the handoff may return from a wait for it at once.
  void report(Job &job, bool success) {
    ++reported_;
    job.report(success);
  }

  const Behaviour behaviour_;
  const uint32_t faulty_;
  const std::chrono::milliseconds pace_;
  volatile uint32_t *const markers_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Job> jobs_;
  std::vector<Seen> seen_;
  uint32_t started_ = 0;
  std::atomic<uint32_t> reported_{0};
  bool stopping_ = false;
  std::thread thread_;
};

// A dispatch of test_add_u32 that adds `increment` to the first `count` words of `buffer`.
ComputeDispatch addition(const MetalBuffer &buffer, const uint32_t &count, const uint32_t &increment) {
  return {"test_add_u32", {{0, buffer}}, {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
          {1, 1, 1}, {1, 1, 1}};
}

// A command of kLayers layers over `markers` (Agent): each adds one to word 0, signals its ready, adds one to a word
// past the markers, waits for its done and sets its marker. Its jobs start through `start` before it is committed.
struct Layers {
  Layers(MetalBackend &backend, const MetalBuffer &markers) {
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      words.push_back(backend.view(markers, 4 * (1 + layer), 4));
    }
    counter = backend.view(markers, 0, 4);
    scratch = backend.view(markers, 4 * (1 + kLayers), 4);
  }
  // Runs one command; the milliseconds it took to complete once committed.
  double run(MetalBackend &backend, ane::Handoff &handoff, const ane::Handoff::Start &start) {
    std::vector<ComputeDispatch> dispatches;
    std::vector<EventStep> events;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      const uint64_t ready = handoff.next(), done = handoff.next();
      dispatches.push_back(addition(counter, kOne, kOne));
      events.push_back({dispatches.size(), handoff.event(), ready, EventStep::Kind::Signal});
      dispatches.push_back(addition(scratch, kOne, kOne));
      events.push_back({dispatches.size(), handoff.event(), done, EventStep::Kind::Wait});
      dispatches.push_back(addition(words[layer], kOne, kOne));
      handoff.queue(ready, done, start);
    }
    const auto committed = AwakeClock::now();
    static_cast<void>(backend.submitCommandAsync(Command{dispatches, events}).wait());
    return milliseconds(AwakeClock::now() - committed);
  }

  static constexpr uint32_t kOne = 1;
  MetalBuffer counter, scratch;
  std::vector<MetalBuffer> words;
};

MetalBuffer zeroed(MetalBackend &backend) {
  MetalBuffer buffer = test::sharedBuffer(backend, 4 * (2 + kLayers));
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}

// Two commands on one handoff, the agent's `behaviour` at job `faulty` of the first.
void behaves(MetalBackend &backend, Behaviour behaviour, uint32_t faulty) {
  const std::string label = std::string(name(behaviour)) + " at job " + std::to_string(faulty);
  const bool fails = behaviour != Behaviour::Succeed && behaviour != Behaviour::Twice;
  ane::Handoff handoff(backend, kBound);
  const MetalBuffer markers = zeroed(backend);
  Agent agent(behaviour, faulty, markers);
  Layers layers(backend, markers);
  const double first = layers.run(backend, handoff, agent.start());
  require(first < milliseconds(kBound + kSlack),
          label + ": the command took " + std::to_string(first) + " ms, past the bound");
  require(backend.healthy(), label + ": the backend is unhealthy: " + backend.unhealthyReason());
  require(handoff.finish().has_value() == !fails, label + ": finish() reported " + (fails ? "success" : "failure"));
  require(handoff.retired() == fails && handoff.reason().empty() == !fails,
          label + ": retired " + std::to_string(handoff.retired()) + " for " + handoff.reason());
  // While the agent behaves, the GPU waits for each layer's done; a failure as the jobs start releases every wait
  // before the GPU runs.
  const bool atStart = behaviour == Behaviour::FailBeforeReady || behaviour == Behaviour::ThrowAtStart;
  require(agent.ordered(!fails ? kLayers : atStart ? 0 : faulty),
          label + ": the GPU passed a wait before the agent saw its ready");

  // The command after it: the agent's jobs as before, or none once the handoff stopped.
  const std::string reason = handoff.reason();
  std::memset(markers.contents(), 0, markers.sizeBytes());
  const uint32_t started = agent.started();
  const double second = layers.run(backend, handoff, agent.start());
  require(second < milliseconds(kBound + kSlack), label + ": the next command took " + std::to_string(second) + " ms");
  require(handoff.finish().has_value() == !fails, label + ": the next command's finish() changed");
  require(handoff.retired() == fails && handoff.reason() == reason, label + ": the handoff's state changed");
  require(agent.started() == started + (fails ? 0 : kLayers),
          label + ": the agent started " + std::to_string(agent.started() - started) + " jobs of the next command");
  require(fails || agent.ordered(2 * kLayers), label + ": the next command passed a wait before its ready");
  require(backend.healthy(), label + ": the next command left the backend unhealthy");
}

// finish() reports the agent's time over a command's jobs, each from the notice of its ready to its report: at least
// the time it took over each, less the notices' delays, and no longer than the command.
void times(MetalBackend &backend) {
  constexpr auto kPace = std::chrono::milliseconds(20), kNotice = std::chrono::milliseconds(5);
  ane::Handoff handoff(backend, kBound);
  const MetalBuffer markers = zeroed(backend);
  Agent agent(Behaviour::Succeed, kLayers, markers, kPace);
  Layers layers(backend, markers);
  const double command = layers.run(backend, handoff, agent.start());
  const std::optional<AwakeClock::duration> ran = handoff.finish();
  require(ran && milliseconds(*ran) >= kLayers * milliseconds(kPace - kNotice) &&
              milliseconds(*ran) <= command + milliseconds(kNotice),
          "finish() reported " + (ran ? std::to_string(milliseconds(*ran)) + " ms" : std::string("a failure")) +
              " for 8 jobs of " + std::to_string(kPace.count()) + " ms in a command of " + std::to_string(command) +
              " ms");
  std::cout << "PASS finish() reports the agent's time over a command's jobs\n";
}

// cancel() releases the jobs of a command never committed and waits for the agent's reports, at most a bound; the
// destructor too; and a report after the handoff is gone does nothing.
void cancels(MetalBackend &backend) {
  const MetalBuffer markers = zeroed(backend);
  const auto queued = [&](ane::Handoff &handoff, Agent &agent) {
    for (uint32_t job = 0; job < kLayers; ++job) {
      const uint64_t ready = handoff.next(), done = handoff.next();
      handoff.queue(ready, done, agent.start());
    }
  };
  {
    ane::Handoff handoff(backend, kBound);
    Agent agent(Behaviour::Succeed, kLayers, markers);
    queued(handoff, agent);
    const auto start = AwakeClock::now();
    handoff.cancel("the command was not committed");
    const double elapsed = milliseconds(AwakeClock::now() - start);
    require(agent.reported() == kLayers && elapsed < milliseconds(kBound),
            "cancel() returned in " + std::to_string(elapsed) + " ms with " + std::to_string(agent.reported()) +
                " of 8 reports");
    require(handoff.retired() && handoff.reason() == "the command was not committed",
            "cancel() did not stop the handoff");
    const uint64_t ready = handoff.next(), done = handoff.next();
    handoff.queue(ready, done, agent.start());
    require(!handoff.finish() && agent.started() == kLayers && reached(handoff.event()) >= done,
            "a job queued after cancel() started or was not released");
  }
  {
    ane::Handoff handoff(backend, kBound);
    Agent agent(Behaviour::Never, 3, markers);
    queued(handoff, agent);
    const auto start = AwakeClock::now();
    handoff.cancel("the command was not committed");
    const double elapsed = milliseconds(AwakeClock::now() - start);
    require(elapsed >= 0.9 * milliseconds(kBound) && elapsed < milliseconds(kBound + kSlack),
            "cancel() of a job never reported returned in " + std::to_string(elapsed) + " ms");
  }
  for (const Behaviour behaviour : {Behaviour::Never, Behaviour::Late}) {
    Agent agent(behaviour, 0, markers);
    double elapsed = 0.0;
    {
      auto handoff = std::make_unique<ane::Handoff>(backend, kBound);
      queued(*handoff, agent);
      const auto start = AwakeClock::now();
      handoff.reset();
      elapsed = milliseconds(AwakeClock::now() - start);
    }
    require(elapsed < milliseconds(kBound + kSlack),
            std::string("the destructor waiting on ") + name(behaviour) + " returned in " + std::to_string(elapsed) +
                " ms");
    // The late job reports after the handoff is gone.
    std::this_thread::sleep_for(4 * kBound);
  }
  std::cout << "PASS cancel() and the destructor return within the bound\n";
}

// A shared event never decreases: the CPU and the GPU signalling below its value leave it there, and a GPU wait on a
// value it has passed completes at once.
void eventsNeverDecrease(MetalBackend &backend) {
  const SharedEvent event = backend.newSharedEvent();
  event.signal(10);
  event.signal(5);
  require(reached(event) == 10, "a CPU signal below the event's value moved it to " + std::to_string(reached(event)));
  const MetalBuffer word = zeroed(backend);
  const uint32_t one = 1;
  const ComputeDispatch add = addition(word, one, one);
  const std::vector<EventStep> below{{1, event, 7, EventStep::Kind::Signal}};
  static_cast<void>(backend.submitCommandAsync(Command{{&add, 1}, below}).wait());
  require(reached(event) == 10, "a GPU signal below the event's value moved it to " + std::to_string(reached(event)));
  const std::vector<EventStep> passed{{0, event, 3, EventStep::Kind::Wait}};
  const auto start = AwakeClock::now();
  static_cast<void>(backend.submitCommandAsync(Command{{&add, 1}, passed}).wait());
  require(milliseconds(AwakeClock::now() - start) < 1000.0 && *static_cast<uint32_t *>(word.contents()) == 2,
          "a GPU wait on a value the event passed did not complete at once");
  require(backend.healthy(), "the event checks left the backend unhealthy");
  std::cout << "PASS a shared event never decreases\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: handoff METALLIB\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1], metal::kResidencyKeepAliveSeconds, kCommandTimeoutSeconds);
      eventsNeverDecrease(backend);
      for (const Behaviour behaviour :
           {Behaviour::Succeed, Behaviour::FailBeforeReady, Behaviour::FailAfterReady, Behaviour::ThrowAtStart,
            Behaviour::Never, Behaviour::Late, Behaviour::Twice}) {
        for (const uint32_t faulty : {0u, 3u, kLayers - 1})
          behaves(backend, behaviour, faulty);
        std::cout << "PASS " << name(behaviour) << " at jobs 0, 3 and 7 of 8\n";
      }
      times(backend);
      cancels(backend);
    } catch (const std::exception &error) {
      std::cout << "FAIL " << error.what() << '\n';
      return 1;
    }
    std::cout << "handoff: all checks passed\n";
  }
  return 0;
}
