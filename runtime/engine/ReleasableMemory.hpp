#pragma once

#include "model/WeightMemory.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::engine {

// What the engine gives back while idle (NativeLoopConfig::weights): the
// weight memory's parts in the order they come back, the weights' images
// first, then any store of streamed weights, and, when the prefill FFN
// splits with the Neural Engine, the split's program in the ANE service
// (ops::AneFfn::release). release() releases every part, then the program;
// restore() takes back the first part released, a call at a time as the part
// does (the images an image per call, a store in one), then loads the
// program in one call more, so a release during a restore gives back what
// came back. Without a split it is the parts; a split that stopped gives its
// program back once more, and then nothing (ops::AneFfn::release).
class ReleasableMemory final : public model::WeightMemory {
public:
  // The split's part: `release` unloads its program between commands, true
  // when it did, and `restore` loads it again. `restore` never throws: a
  // program that does not load stops the split, and the GPU runs the prefill
  // FFN alone (ops::AneFfn::restore).
  struct Split final {
    std::function<bool()> release;
    std::function<void()> restore;
  };

  // `parts` in the order they come back, the images first; a null part, one
  // the model has not, is left out.
  ReleasableMemory(std::vector<model::WeightMemory *> parts, std::optional<Split> split)
      : parts_(std::move(parts)), split_(std::move(split)) {
    std::erase(parts_, nullptr);
    if (parts_.empty()) throw std::invalid_argument("releasable memory without the weights' images");
  }

  // The parts': the program comes back after them.
  [[nodiscard]] bool released() const noexcept override {
    return std::ranges::any_of(parts_, &model::WeightMemory::released);
  }
  void release() override {
    for (model::WeightMemory *part : parts_) part->release();
    if (!splitReleased_)
      splitReleased_ = split_ && split_->release();
  }
  [[nodiscard]] bool restore(const metal::AllocationAdmission &admit) override {
    // Every part, then the program. With nothing released, the first part
    // refuses the restore.
    auto part = std::ranges::find_if(parts_, &model::WeightMemory::released);
    if (part == parts_.end() && !splitReleased_) part = parts_.begin();
    if (part != parts_.end())
      return (*part)->restore(admit) && !released() && !splitReleased_;
    splitReleased_ = false;
    split_->restore();
    return true;
  }

private:
  std::vector<model::WeightMemory *> parts_;
  std::optional<Split> split_;
  // From a release() that unloaded the split's program until restore() loads
  // it again.
  bool splitReleased_ = false;
};

} // namespace splash::engine
