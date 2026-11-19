#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"

namespace splash::model {

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry, std::string_view architecture,
                                                 ExpertStoreOptions experts)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)),
      layers_(geometry.layers), firstMoeLayer_(geometry.leadingDenseLayers), experts_(experts) {
  planned_->images = safetensors::planImages(planned_->checkpoint, geometry, architecture,
                                             {.streamExperts = experts.memoryBytes != 0});
  float32Norms_ = safetensors::float32Norms(safetensors::moduleNames(planned_->checkpoint), architecture);
}

StreamedBlocks SafetensorsTargetLoader::streamedExperts() const {
  const std::span layers = std::span(planned_->images).first(layers_);
  return {gguf::streamedExperts(layers.subspan(firstMoeLayer_)), planned_,
          {[planned = planned_] { planned->checkpoint.checkUnchanged(); }}};
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  return loadImage(backend_, images_, "target/", {planned_, &planned_->images[index]},
                   [planned = planned_] { planned->checkpoint.checkUnchanged(); });
}

WeightFile SafetensorsTargetLoader::layer(uint32_t index) {
  if (index >= layers_) throw WeightStoreError("target layer is out of range");
  return open(index);
}

WeightFile SafetensorsTargetLoader::head() { return open(layers_); }

WeightFile SafetensorsTargetLoader::embedding() { return open(layers_ + 1); }

} // namespace splash::model
