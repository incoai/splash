#pragma once

#include "model/PreparedWeights.hpp"
#include "model/WeightStore.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::model {

// The prepared files of one source: each reused, or written now and
// published. check runs throughout, admitConversion on a cache miss and
// sourceUnchanged around every write, so no file of a modified source is
// published or used.
class PreparedFiles final {
public:
  PreparedFiles(PreparationCheck check, PreparationCheck admitConversion, PreparationCheck sourceUnchanged)
      : guards_{std::move(check), std::move(admitConversion), std::move(sourceUnchanged)} {}
  [[nodiscard]] std::filesystem::path prepare(const PreparedWeight &weight, const WeightWriter &write) const {
    return store_.prepare(weight, write, guards_);
  }
  // The prepared file mapped as a weight file of magic, layer and type.
  [[nodiscard]] WeightFile open(metal::MetalBackend &backend, const PreparedWeight &weight,
                                const WeightWriter &write, std::string_view magic, uint32_t layer,
                                uint32_t type) const {
    return WeightFile(backend, prepare(weight, write), weight.component, magic, layer, type, weight.key);
  }

private:
  PreparationGuards guards_;
  PreparedWeights store_;
};

// The planned images of one source and their prepared files, layers first.
// Every image is prepared before the first is mapped (ModelFactory.cpp). An
// Image has bytes, magic, layer and type.
template <class Image> class PreparedImages final {
public:
  // writer(image) writes one image; the image outlives the writer.
  PreparedImages(PreparedFiles files, std::function<WeightWriter(const Image &)> writer)
      : files_(std::move(files)), writer_(std::move(writer)) {}
  // Appends a planned image, already bound to its source, and its cache identity.
  void add(Image image, PreparedWeight weight) {
    images_.push_back(std::move(image));
    weights_.push_back(std::move(weight));
  }
  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept { return weights_; }
  [[nodiscard]] size_t size() const noexcept { return images_.size(); }
  // Writes every missing image and maps none.
  void prepare() const {
    for (size_t index = 0; index < images_.size(); ++index)
      static_cast<void>(files_.prepare(weights_[index], writer_(images_[index])));
  }
  // The prepared file of image `index`, mapped as a weight file.
  [[nodiscard]] WeightFile open(metal::MetalBackend &backend, size_t index) const {
    const Image &image = images_[index];
    return files_.open(backend, weights_[index], writer_(image), image.magic, image.layer, image.type);
  }
  [[nodiscard]] static uint64_t bytes(std::span<const Image> images) noexcept {
    uint64_t total = 0;
    for (const Image &image : images) total += image.bytes;
    return total;
  }

private:
  PreparedFiles files_;
  std::function<WeightWriter(const Image &)> writer_;
  std::vector<Image> images_;
  std::vector<PreparedWeight> weights_;
};

} // namespace splash::model
