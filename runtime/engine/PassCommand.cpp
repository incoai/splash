#include "engine/PassCommand.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::engine {

PassCommand::PassCommand(Scheduler &scheduler, std::function<bool(uint64_t)> ended)
    : scheduler_(scheduler), ended_(std::move(ended)) {}

const PassProgress *PassCommand::progress() const noexcept {
  return command_ ? &command_->progress : nullptr;
}

bool PassCommand::holds(uint64_t requestId) const noexcept {
  return command_ && std::ranges::any_of(command_->items, [requestId](const ModelBatchItem &item) {
           return item.requestId == requestId;
         });
}

bool PassCommand::anyEnded() const {
  return command_ && std::ranges::any_of(command_->items, [&](const ModelBatchItem &item) {
           return ended_(item.requestId);
         });
}

bool PassCommand::allEnded() const {
  return std::ranges::all_of(command_->items, [&](const ModelBatchItem &item) { return ended_(item.requestId); });
}

uint32_t PassCommand::captureBuffers() const noexcept {
  uint32_t buffers = 0;
  if (command_)
    for (const ModelBatchItem &item : command_->items)
      buffers += static_cast<uint32_t>(std::ranges::count_if(
          item.stateCaptures, [](const StateCapture &capture) { return capture.reservation != nullptr; }));
  return buffers;
}

void PassCommand::start(const BatchPlan &plan, std::vector<ModelBatchItem> items,
                        std::vector<std::vector<LaneCapture>> captures) {
  if (command_)
    throw std::logic_error("a command in passes started beside another");
  for (ModelBatchItem &item : items) {
    item.pageTable = {};
    item.pageTableRevision = 0;
    item.pageTableFirstChanged = 0;
    item.pageSources.reset();
  }
  command_ = Command{.progress = {.plan = plan},
                     .items = std::move(items),
                     .captures = std::move(captures)};
}

void PassCommand::nextPass(std::vector<ModelBatchItem> &items,
                           std::vector<std::vector<LaneCapture>> &captures) const {
  if (!command_)
    throw std::logic_error("a prefill command's pass has no command");
  items = command_->items;
  captures = command_->captures;
}

std::vector<uint64_t> PassCommand::passCompleted(const BatchPlan &plan) {
  // Every pass of a command runs its items, so its pass and layer name it.
  if (!command_ || plan.pass != command_->progress.plan.pass || plan.layer != command_->progress.plan.layer)
    throw std::logic_error("a pass completed that is not its command's next");
  if (plan.pass + 1 >= plan.passes) {
    command_.reset();
    return {};
  }
  command_->progress.plan = scheduler_.nextPass(plan);
  return allEnded() ? yield() : std::vector<uint64_t>{};
}

std::vector<uint64_t> PassCommand::laneEnded(uint64_t requestId, bool passInFlight) {
  if (!holds(requestId) || passInFlight || !allEnded())
    return {};
  return yield();
}

std::vector<uint64_t> PassCommand::yield() {
  if (!command_)
    return {};
  scheduler_.abandoned();
  std::vector<uint64_t> ended;
  for (const ModelBatchItem &item : command_->items)
    if (ended_(item.requestId))
      ended.push_back(item.requestId);
  // Its items go with it, and with them the buffers of the states they would
  // capture.
  command_.reset();
  return ended;
}

} // namespace splash::engine
