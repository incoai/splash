#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"

namespace splash::model {

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry, ExpertStoreOptions experts)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)), experts_(experts) {
  planned_->images =
      safetensors::planImages(planned_->checkpoint, geometry, {.streamExperts = experts.memoryBytes != 0});
  float32Norms_ = safetensors::float32Norms(safetensors::moduleNames(planned_->checkpoint));
}

StreamedBlocks SafetensorsTargetLoader::streamedExperts() const {
  return {gguf::streamedExperts(std::span(planned_->images).first(planned_->images.size() - 2)),
          {[planned = planned_] { planned->checkpoint.checkUnchanged(); }}};
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->checkpoint.checkUnchanged();
                       }});
}

WeightFile SafetensorsTargetLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 2) throw WeightStoreError("target layer is out of range");
  return open(index);
}

WeightFile SafetensorsTargetLoader::head() { return open(planned_->images.size() - 2); }

WeightFile SafetensorsTargetLoader::embedding() { return open(planned_->images.size() - 1); }

uint64_t safetensorsRoutedExpertBytes(const std::filesystem::path &directory, const QwenTargetDimensions &geometry) {
  const SafetensorsCheckpoint checkpoint(directory);
  uint64_t bytes = 0;
  for (const gguf::Image &image : safetensors::planImages(checkpoint, geometry, {.streamExperts = true}))
    bytes += gguf::streamedExpertBytes(image);
  return bytes;
}

uint64_t safetensorsTargetImageBytes(const std::filesystem::path &directory, const QwenTargetDimensions &geometry,
                                     gguf::PlanOptions options) {
  const SafetensorsCheckpoint checkpoint(directory);
  uint64_t bytes = 0;
  for (const gguf::Image &image : safetensors::planImages(checkpoint, geometry, options)) bytes += image.bytes;
  return bytes;
}

} // namespace splash::model
