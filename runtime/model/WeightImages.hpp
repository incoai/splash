#pragma once

#include "metal/MetalBackend.hpp"
#include "model/WeightMemory.hpp"
#include "model/WeightStore.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace splash::model {

// The source bytes a load task reads, or stages, at most.
inline constexpr uint64_t kLoadStepBytes = 4 << 20;

// Writes every byte of an image from the sources its loader planned it from.
// bytes is the image's memory and buffer the Metal buffer that holds it, of
// which a writer that runs GPU commands binds views; a writer that uses only
// the CPU ignores buffer, so a test can give it plain memory.
using ImageWriter = std::function<void(std::span<uint8_t> bytes, const metal::MetalBuffer &buffer)>;

// The bytes of a buffer an image is written into.
[[nodiscard]] inline std::span<uint8_t> contentsOf(const metal::MetalBuffer &image) {
  return {static_cast<uint8_t *>(image.contents()), image.sizeBytes()};
}

// An image of a model's weights as its loader plans it: the component it is
// (such as target/layer-0.bin), the header its file starts with, its size and
// its writer, and the size its record declares when not that one
// (gguf::Image::residentBytes: a layer whose routed experts stream).
struct ImagePlan final {
  std::string component;
  std::string magic;
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t bytes = 0;
  ImageWriter write;
  uint64_t declaredBytes = 0;
};

// A model's weights in memory: each image in a buffer of its own, written
// from its sources by its writer, which keeps them open. The memory can be
// released and restored, an image at a time in load order, while every view
// of it stays the same handle: a command that binds a view of released memory
// fails (MetalBackend::releaseMemory). Every image's file records
// contentIdentity, what the sources hold (ModelDescriptor::sourceIdentity).
//
// A restore writes each image on a thread of its own, so that the engine's
// loop goes on answering the server while one is written: Flash-Next's PLE
// table loaded whole is one image of 28.8 GB, whose write from a disk slower
// than about 1 GB/s outlasts the 30 s within which the loop must answer a
// status request (supervision::kLoopStall), past which the server stops the
// engine. The writer has the backend to itself meanwhile: the engine leaves
// memory and the backend alone until the write ends (WeightMemory::writing,
// engine::Engine::tick).
class WeightImages final : public WeightMemory {
public:
  WeightImages(metal::MetalBackend &backend, std::string contentIdentity) noexcept
      : backend_(&backend), contentIdentity_(std::move(contentIdentity)) {}
  WeightImages(const WeightImages &) = delete;
  WeightImages &operator=(const WeightImages &) = delete;
  // Waits for an image being written.
  ~WeightImages() override;

  // Allocates the image's buffer, writes it and returns its file.
  [[nodiscard]] WeightFile load(ImagePlan image);
  // What every image's file records the sources hold.
  [[nodiscard]] const std::string &contentIdentity() const noexcept { return contentIdentity_; }
  [[nodiscard]] bool released() const noexcept override { return released_; }
  // An image being written ends first: its memory goes back with the others.
  void release() override;
  // Ends the write of the last image it took back, then takes back the next
  // and starts writing it on a thread of its own.
  [[nodiscard]] bool restore(const metal::AllocationAdmission &admit) override;
  [[nodiscard]] bool writing() const noexcept override;
  // The writes restore() starts from then on call `notifier` on their thread
  // as they end.
  void setCompletionNotifier(std::function<void()> notifier) override { written_ = std::move(notifier); }

  struct Contents final {
    std::string_view component;
    std::span<const uint8_t> bytes;
  };
  // Each image's component and bytes in load order, its bytes empty while
  // released and still being written while writing(), valid until the next
  // load().
  [[nodiscard]] std::vector<Contents> contents() const;

private:
  struct Image final {
    std::string component;
    metal::MetalBuffer buffer;
    ImageWriter write;
  };
  // The write of the last image restore() took back, on its thread: whether
  // it ended, and how it failed.
  struct Writing final {
    std::thread thread;
    std::atomic<bool> ended{false};
    std::exception_ptr failure;
  };
  // Waits for the write in flight and forgets it; returns how it failed.
  std::exception_ptr endWrite();

  metal::MetalBackend *backend_;
  std::string contentIdentity_;
  std::vector<Image> images_;
  bool released_ = false;
  // The images restore() has taken back since release(), the last of them
  // written by writing_'s thread while that is set, until the next restore()
  // or release() ends it.
  size_t restored_ = 0;
  std::unique_ptr<Writing> writing_;
  // The completion notifier, which each write copies as it starts.
  std::function<void()> written_;
};

// The threads that read and convert weights while images are written: one
// per core.
[[nodiscard]] unsigned loadThreads() noexcept;

// Runs task(index, thread) for every index below count on up to
// loadThreads() workers, numbered below that so that each task can stage
// through its worker's buffers. Rethrows the first exception once every
// worker has stopped.
void parallelFor(size_t count, const std::function<void(size_t index, unsigned thread)> &task);

// Writes the F32 GDN decay of `count` BF16 or F32 values A_log at values,
// each float(-exp(double(A_log))), to `to`; throws on a non-finite decay.
void writeGdnDecay(const uint8_t *values, uint64_t count, bool bfloat16, uint8_t *to);

// Zeroes the bytes of image outside the extents, [offset, offset + bytes),
// that its writer writes: the alignment between sections and any padding.
void zeroUnwritten(std::span<uint8_t> image, std::vector<std::pair<uint64_t, uint64_t>> extents);

} // namespace splash::model
