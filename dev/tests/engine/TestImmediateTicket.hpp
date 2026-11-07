#pragma once

#include "engine/Types.hpp"
#include "model/CommandCost.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::test {

// The times a command takes: none unless a test sets them, as a fake runs
// no kernels, and its sample (ModelBatchTicket::sample), whose wall is
// wallMilliseconds. Of the wall, what a profile's command waits on its host
// steps, and of that on its history's reads (prefillTimes).
struct CommandTimes final {
  double wallMilliseconds = 0.0;
  double hostWaitMilliseconds = 0.0;
  double historyWaitMilliseconds = 0.0;
  model::CommandSample sample{};
};

[[nodiscard]] inline model::CommandSample sampleOf(const CommandTimes &times) noexcept {
  model::CommandSample sample = times.sample;
  sample.wallMilliseconds = times.wallMilliseconds;
  return sample;
}

class ImmediateTicket final : public ModelBatchTicket {
public:
  explicit ImmediateTicket(std::vector<ModelStepResult> results, CommandTimes times = {})
      : results_(std::move(results)), times_(times) {}

  [[nodiscard]] bool ready() const noexcept override { return true; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    return std::move(results_);
  }
  [[nodiscard]] model::CommandSample sample() const noexcept override { return sampleOf(times_); }

private:
  std::vector<ModelStepResult> results_;
  CommandTimes times_;
};

// A command that completes once the test's clock reaches its end.
class TimedTicket final : public ModelBatchTicket {
public:
  TimedTicket(std::vector<ModelStepResult> results, CommandTimes times, std::function<double()> clock,
              double endMilliseconds)
      : results_(std::move(results)), times_(times), clock_(std::move(clock)), end_(endMilliseconds) {}

  [[nodiscard]] bool ready() const noexcept override { return clock_() >= end_; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    if (!ready())
      throw std::logic_error("a timed command completed before its end");
    return std::move(results_);
  }
  [[nodiscard]] model::CommandSample sample() const noexcept override { return sampleOf(times_); }

private:
  std::vector<ModelStepResult> results_;
  CommandTimes times_;
  std::function<double()> clock_;
  double end_;
};

// A command that stays in flight until the test sets `ready`, from any
// thread.
class HeldTicket final : public ModelBatchTicket {
public:
  HeldTicket(std::vector<ModelStepResult> results,
             std::shared_ptr<std::atomic<bool>> ready, double wallMilliseconds)
      : results_(std::move(results)), ready_(std::move(ready)),
        wallMilliseconds_(wallMilliseconds) {}

  [[nodiscard]] bool ready() const noexcept override { return *ready_; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    return std::move(results_);
  }
  [[nodiscard]] model::CommandSample sample() const noexcept override {
    return {.wallMilliseconds = wallMilliseconds_};
  }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<std::atomic<bool>> ready_;
  double wallMilliseconds_ = 0.0;
};

// A command of two, as a prefill command whose images are encoded ahead of
// it (model::Runtime): the first runs until the test sets `first`, from any
// thread, and the second completes once the engine's advance() has run it
// after that.
class StagedTicket final : public ModelBatchTicket {
public:
  StagedTicket(std::vector<ModelStepResult> results, std::shared_ptr<std::atomic<bool>> first)
      : results_(std::move(results)), first_(std::move(first)) {}

  void advance() override { second_ = second_ || *first_; }
  [[nodiscard]] bool ready() const noexcept override { return second_; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    if (!second_)
      throw std::logic_error("a staged ticket completed before its second command ran");
    return std::move(results_);
  }
  [[nodiscard]] model::CommandSample sample() const noexcept override { return {.wallMilliseconds = 1.0}; }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<std::atomic<bool>> first_;
  bool second_ = false;
};

inline std::unique_ptr<ModelBatchTicket>
immediateTicket(std::vector<ModelStepResult> results,
                const std::function<void()> &completion, CommandTimes times = {}) {
  if (completion)
    completion();
  return std::make_unique<ImmediateTicket>(std::move(results), times);
}

} // namespace splash::test
