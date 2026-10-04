#pragma once

#include "metal/MetalBackend.hpp"
#include "model/WeightMemory.hpp"
#include "model/WeightStore.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
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
// its writer.
struct ImagePlan final {
  std::string component;
  std::string magic;
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t bytes = 0;
  ImageWriter write;
};

// A model's weights in memory: each image in a buffer of its own, written
// from its sources by its writer, which keeps them open. The memory can be
// released and restored, an image at a time in load order, while every view
// of it stays the same handle: a command that binds a view of released memory
// fails (MetalBackend::releaseMemory). Every image's file records
// contentIdentity, what the sources hold (ModelDescriptor::sourceIdentity).
class WeightImages final : public WeightMemory {
public:
  explicit WeightImages(metal::MetalBackend &backend, std::string contentIdentity = {}) noexcept
      : backend_(&backend), contentIdentity_(std::move(contentIdentity)) {}
  WeightImages(const WeightImages &) = delete;
  WeightImages &operator=(const WeightImages &) = delete;

  // Allocates the image's buffer, writes it and returns its file.
  [[nodiscard]] WeightFile load(ImagePlan image);
  [[nodiscard]] bool released() const noexcept override { return released_; }
  void release() override;
  [[nodiscard]] bool restore() override;

  struct Contents final {
    std::string_view component;
    std::span<const uint8_t> bytes;
  };
  // Each image's component and bytes in load order, its bytes empty while
  // released, valid until the next load().
  [[nodiscard]] std::vector<Contents> contents() const;

private:
  struct Image final {
    std::string component;
    metal::MetalBuffer buffer;
    ImageWriter write;
  };
  metal::MetalBackend *backend_;
  std::string contentIdentity_;
  std::vector<Image> images_;
  bool released_ = false;
  // The images restore() has written back since release().
  size_t restored_ = 0;
};

// The image of a Splash package's packed file at path, read as it is.
[[nodiscard]] ImagePlan packedImage(const std::filesystem::path &path, std::string component,
                                    std::string_view magic, uint32_t layer, uint32_t type);

// The threads that read and convert weights while images are written: one
// per core.
[[nodiscard]] unsigned loadThreads() noexcept;

// Runs task(index, thread) for every index below count on up to
// loadThreads() workers, numbered below that so that each task can stage
// through its worker's buffers. Rethrows the first exception once every
// worker has stopped.
void parallelFor(size_t count, const std::function<void(size_t index, unsigned thread)> &task);

// Zeroes the bytes of image outside the extents, [offset, offset + bytes),
// that its writer writes: the alignment between sections and any padding.
void zeroUnwritten(std::span<uint8_t> image, std::vector<std::pair<uint64_t, uint64_t>> extents);

} // namespace splash::model
