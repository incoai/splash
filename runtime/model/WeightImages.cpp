#include "model/WeightImages.hpp"

#include "model/Bfloat16.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace splash::model {

WeightFile WeightImages::load(ImagePlan image) {
  if (released_) throw std::logic_error("weights load while their memory is released");
  metal::MetalBuffer buffer = backend_->allocateBuffer(image.bytes, metal::BufferStorage::Shared, image.component);
  image.write(contentsOf(buffer), buffer);
  WeightFile file(*backend_, buffer, image.component, image.magic, image.layer, image.type, contentIdentity_);
  images_.push_back({std::move(image.component), std::move(buffer), std::move(image.write)});
  return file;
}

void WeightImages::release() {
  // During a restore, only the images it wrote back hold memory.
  const size_t held = released_ ? restored_ : images_.size();
  for (size_t index = 0; index < held; ++index) backend_->releaseMemory(images_[index].buffer);
  released_ = true;
  restored_ = 0;
}

bool WeightImages::restore(const metal::AllocationAdmission &admit) {
  if (!released_) throw std::logic_error("weights are not released");
  if (restored_ < images_.size()) {
    Image &image = images_[restored_];
    const metal::AllocationResult admitted = admit(image.buffer.sizeBytes(), [&] {
      backend_->restoreMemory(image.buffer);
      try {
        image.write(contentsOf(image.buffer), image.buffer);
      } catch (const metal::MetalAllocationError &) {
        // A writer refused memory of its own leaves the image released too.
        backend_->releaseMemory(image.buffer);
        throw;
      }
    });
    if (!admitted)
      throw metal::MetalAllocationError("unable to restore " + image.component + ": " +
                                            metal::allocationFailureName(admitted.failure),
                                        admitted.failure);
    ++restored_;
  }
  released_ = restored_ < images_.size();
  return !released_;
}

std::vector<WeightImages::Contents> WeightImages::contents() const {
  std::vector<Contents> result;
  for (const Image &image : images_) {
    const auto *bytes = static_cast<const uint8_t *>(image.buffer.contents());
    result.push_back({image.component, {bytes, bytes ? image.buffer.sizeBytes() : 0}});
  }
  return result;
}

unsigned loadThreads() noexcept { return std::max(1u, std::thread::hardware_concurrency()); }

void parallelFor(size_t count, const std::function<void(size_t index, unsigned thread)> &task) {
  struct Work {
    Work(const std::function<void(size_t, unsigned)> &task, size_t count) : task(task), count(count) {}
    const std::function<void(size_t, unsigned)> &task;
    const size_t count;
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex failureMutex;
    std::exception_ptr failure;
  } work(task, count);
  // Each worker takes indices until none is left; libdispatch runs the
  // workers on its own threads.
  dispatch_apply_f(std::min<size_t>(count, loadThreads()), DISPATCH_APPLY_AUTO, &work, [](void *context, size_t thread) {
    auto &work = *static_cast<Work *>(context);
    for (size_t index; !work.failed.load(std::memory_order_relaxed) && (index = work.next++) < work.count;) {
      try {
        work.task(index, static_cast<unsigned>(thread));
      } catch (...) {
        std::lock_guard lock(work.failureMutex);
        if (!work.failure) work.failure = std::current_exception();
        work.failed = true;
      }
    }
  });
  if (work.failure) std::rethrow_exception(work.failure);
}

void writeGdnDecay(const uint8_t *values, uint64_t count, bool bfloat16, uint8_t *to) {
  for (uint64_t i = 0; i < count; ++i) {
    float logarithm;
    if (bfloat16) {
      uint16_t bits;
      std::memcpy(&bits, values + 2 * i, 2);
      logarithm = widenBfloat16(bits);
    } else {
      std::memcpy(&logarithm, values + 4 * i, 4);
    }
    const auto decay = static_cast<float>(-std::exp(static_cast<double>(logarithm)));
    if (!std::isfinite(decay)) throw std::runtime_error("non-finite GDN decay");
    std::memcpy(to + 4 * i, &decay, 4);
  }
}

void zeroUnwritten(std::span<uint8_t> image, std::vector<std::pair<uint64_t, uint64_t>> extents) {
  std::sort(extents.begin(), extents.end());
  uint64_t written = 0;
  for (const auto &[offset, bytes] : extents) {
    if (offset > image.size() || bytes > image.size() - offset)
      throw std::out_of_range("weight image extent is out of bounds");
    if (offset > written) std::memset(image.data() + written, 0, offset - written);
    written = std::max(written, offset + bytes);
  }
  std::memset(image.data() + written, 0, image.size() - written);
}

} // namespace splash::model
