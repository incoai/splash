#pragma once

#include "model/WeightMemory.hpp"

#include <functional>
#include <optional>
#include <utility>

namespace splash::engine {

// What the engine gives back while idle (NativeLoopConfig::weights): the
// weights' images and, when the prefill FFN splits with the Neural Engine,
// the split's program in the ANE service (ops::AneFfn::release). release()
// releases the images, then the program; restore() writes back an image per
// call, as the images do, then loads the program in one call more, so a
// release during a restore gives back images alone. Without a split it is the
// images; a split that stopped gives its program back once more, and then
// nothing (ops::AneFfn::release).
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

  ReleasableMemory(model::WeightMemory &images, std::optional<Split> split)
      : images_(images), split_(std::move(split)) {}

  // The images': the program comes back after them.
  [[nodiscard]] bool released() const noexcept override { return images_.released(); }
  void release() override {
    images_.release();
    if (!splitReleased_)
      splitReleased_ = split_ && split_->release();
  }
  [[nodiscard]] bool restore(const metal::AllocationAdmission &admit) override {
    // Every image, then the program.
    if (images_.released() || !splitReleased_)
      return images_.restore(admit) && !splitReleased_;
    splitReleased_ = false;
    split_->restore();
    return true;
  }

private:
  model::WeightMemory &images_;
  std::optional<Split> split_;
  // From a release() that unloaded the split's program until restore() loads
  // it again.
  bool splitReleased_ = false;
};

} // namespace splash::engine
