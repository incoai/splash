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
  planned_->layers = geometry.layers;
  float32Norms_ = safetensors::float32Norms(safetensors::moduleNames(planned_->checkpoint));
}

StreamedBlocks SafetensorsTargetLoader::streamedExperts() const {
  return {gguf::streamedExperts(std::span(planned_->images).first(planned_->layers)), planned_,
          {[planned = planned_] { planned->checkpoint.checkUnchanged(); }}};
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  return loadImage(backend_, images_, "target/", {planned_, &planned_->images[index]},
                   [planned = planned_] { planned->checkpoint.checkUnchanged(); });
}

WeightFile SafetensorsTargetLoader::layer(uint32_t index) {
  if (index >= planned_->layers) throw WeightStoreError("target layer is out of range");
  return open(index);
}

WeightFile SafetensorsTargetLoader::head() { return open(planned_->layers); }

WeightFile SafetensorsTargetLoader::embedding() {
  if (planned_->images.size() != planned_->layers + 2) throw WeightStoreError("the target's token table is its head");
  return open(planned_->layers + 1);
}

} // namespace splash::model
