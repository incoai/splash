#pragma once

#include "engine/Types.hpp"

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

namespace splash::test {

class ImmediateTicket final : public ModelBatchTicket {
public:
  explicit ImmediateTicket(std::vector<ModelStepResult> results)
      : results_(std::move(results)) {}

  [[nodiscard]] bool ready() const noexcept override { return true; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    return std::move(results_);
  }
  [[nodiscard]] double wallMilliseconds() const noexcept override {
    return 0.0;
  }

private:
  std::vector<ModelStepResult> results_;
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
  [[nodiscard]] double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<std::atomic<bool>> ready_;
  double wallMilliseconds_ = 0.0;
};

inline std::unique_ptr<ModelBatchTicket>
immediateTicket(std::vector<ModelStepResult> results,
                const std::function<void()> &completion) {
  if (completion)
    completion();
  return std::make_unique<ImmediateTicket>(std::move(results));
}

} // namespace splash::test
