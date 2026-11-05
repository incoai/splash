#pragma once

#include "engine/Capture.hpp"
#include "engine/Scheduler.hpp"
#include "model/Model.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace splash::engine {

// A prefill command between its passes (BatchPlan::passes), the one owner of
// its life: its progress as planning reads it (PlanInputs::passes), its plan
// at its next pass, which each pass's completion advances; its items, with
// the states they capture, and their captures, which every pass runs and the
// last one commits.
//
// A lane of it that ends (`ended`, the engine's Request::pendingEnd) stays
// in it to its last pass; it is given up whole once every lane of it has
// ended, or when the engine yields it. The calls that give it up return the
// lanes of it that ended, for the caller to settle at once, or to leave to
// Engine::settleEndedBetweenPasses where it may not settle: no lane's end
// depends on who takes the command.
class PassCommand final {
public:
  PassCommand(Scheduler &scheduler, std::function<bool(uint64_t)> ended);

  [[nodiscard]] bool active() const noexcept { return command_.has_value(); }
  // Null without a command.
  [[nodiscard]] const PassProgress *progress() const noexcept;
  [[nodiscard]] bool holds(uint64_t requestId) const noexcept;
  [[nodiscard]] bool anyEnded() const;
  // The buffers of the states it would capture, which giving it up returns
  // to the model's pool.
  [[nodiscard]] uint32_t captureBuffers() const noexcept;

  // Its first pass, `plan`, was submitted with these items and captures;
  // only without a command. It keeps the items without their page tables,
  // which moves between its passes rewrite.
  void start(const BatchPlan &plan, std::vector<ModelBatchItem> items,
             std::vector<std::vector<LaneCapture>> captures);
  // Its items, without page tables, and captures for its next pass; the
  // caller cuts each item's page table again.
  void nextPass(std::vector<ModelBatchItem> &items, std::vector<std::vector<LaneCapture>> &captures) const;
  // Its pass `plan` completed. The last ends it, its lanes that ended ending
  // with that pass's results, so it returns none; a pass before moves it to
  // its next pass, or gives it up where every lane of it has ended.
  [[nodiscard]] std::vector<uint64_t> passCompleted(const BatchPlan &plan);
  // A lane of it ended: where every lane has and no pass of it is in flight,
  // it is given up; otherwise the lane ends at its last pass.
  [[nodiscard]] std::vector<uint64_t> laneEnded(uint64_t requestId, bool passInFlight);
  // Gives it up: its lanes count its passes' time as waited and run its rows
  // again later, nothing of them committed (Scheduler::abandoned, which
  // throws while a pass of it is in flight). Nothing without a command.
  [[nodiscard]] std::vector<uint64_t> yield();

private:
  struct Command final {
    PassProgress progress;
    std::vector<ModelBatchItem> items;
    std::vector<std::vector<LaneCapture>> captures;
  };
  [[nodiscard]] bool allEnded() const;

  Scheduler &scheduler_;
  std::function<bool(uint64_t)> ended_;
  std::optional<Command> command_;
};

} // namespace splash::engine
