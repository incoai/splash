#pragma once

#include "FakeMemory.hpp"
#include "TestAdmission.hpp"
#include "TestChecks.hpp"
#include "engine/Engine.hpp"
#include "model/WeightMemory.hpp"
#include "ops/PagedKv.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::test {

// A governor of its own for each engine a test builds without one, over an
// ample limit and host, so that its memory is the KV storage and the model
// the test gives it: set by the binary's stand-in backend
// (TestMetalMemory.hpp), none in a binary that links the Metal backend.
inline engine::MemoryGovernor *(*ampleGovernor)() = nullptr;

// An engine config completed as the bootstrap completes one, for tests: what
// the test set, and otherwise the whole logical context, every 32-bit token
// in the vocabulary and an ample governor.
[[nodiscard]] inline engine::EngineConfig engineConfig(engine::EngineConfig config = {}) {
  if (!config.maxContext)
    config.maxContext = kv::kMaximumLogicalTokens;
  if (!config.vocabularySize)
    config.vocabularySize = std::numeric_limits<uint32_t>::max();
  if (!config.governor && ampleGovernor)
    config.governor = ampleGovernor();
  return config;
}

// The weights of a native loop under test: three images of a gibibyte that
// count what the loop does with them, each restored under the admission the
// loop passes; with `memory`, which holds the images in memory as "weights",
// restored through it. With `writes`, each image restore() takes back is
// written on a thread of its own, as model::WeightImages writes it, until the
// test calls endWrite(), which another thread may call while the loop runs,
// as it may read writing() and `restores`.
class Weights final : public model::WeightMemory {
public:
  static constexpr uint32_t kImages = 3;
  static constexpr uint64_t kImageBytes = 1ULL << 30;
  [[nodiscard]] bool released() const noexcept override {
    return restored_ < kImages || taken_.load(std::memory_order_acquire);
  }
  void release() override {
    require(!writing(), "weights were released while an image was written");
    if (memory)
      memory->release("weights", uint64_t{restored_} * kImageBytes);
    restored_ = 0;
    taken_ = false;
  }
  bool restore(const metal::AllocationAdmission &admit) override {
    if (failRestore)
      throw std::runtime_error("weights restore test");
    require(released(), "held weights were restored");
    require(!writing(), "weights were restored while an image was written");
    taken_ = false;
    if (restored_ == kImages)
      return true;
    if (const metal::AllocationResult admitted =
            memory ? memory->allocate(admit, "weights", kImageBytes) : admit(kImageBytes, [] {});
        !admitted)
      throw metal::MetalAllocationError("weights restore test refused", admitted.failure);
    ++restores;
    ++restored_;
    if (writes) {
      ended_ = false;
      taken_.store(true, std::memory_order_release);
    }
    return !released();
  }
  [[nodiscard]] bool writing() const noexcept override {
    return taken_.load(std::memory_order_acquire) && !ended_.load(std::memory_order_acquire);
  }
  void setCompletionNotifier(std::function<void()> notifier) override { written_ = std::move(notifier); }
  // Ends the write of the image restore() took back last, and calls the
  // completion notifier.
  void endWrite() {
    require(writing(), "no image was being written");
    ended_.store(true, std::memory_order_release);
    if (written_)
      written_();
  }
  // The images in memory.
  [[nodiscard]] uint32_t held() const noexcept { return restored_; }
  bool failRestore = false;
  bool writes = false;
  FakeMemory *memory = nullptr;
  // The images written back.
  std::atomic<uint32_t> restores{0};

private:
  uint32_t restored_ = kImages;
  // The last image taken back is written until endWrite() while `taken_`
  // holds, which the next restore() clears; `written_` is the completion
  // notifier, set before the loop runs.
  std::atomic<bool> taken_{false}, ended_{false};
  std::function<void()> written_;
};

} // namespace splash::test
