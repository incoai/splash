#pragma once

// Source adapter for a safetensors target (MLX's, Model Optimizer's or
// compressed-tensors'): its block images (model/SafetensorsImage.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/TargetImages.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace splash::model {

// How a family's checkpoints plan (SafetensorsTargetLoader): its images, from
// a checkpoint with what streams, and whether they hold its norms as F32, its
// layer images first, `layerImages` of them, the first MoE layer's
// `firstMoeLayer`.
struct SafetensorsTargetPlan final {
  struct Images final {
    std::vector<gguf::Image> images;
    bool float32Norms = true;
  };
  uint32_t layerImages = 0;
  uint32_t firstMoeLayer = 0;
  std::function<Images(const SafetensorsCheckpoint &, gguf::PlanOptions)> images;
};
// A family whose checkpoints safetensors::planImages plans, an image a layer,
// its norms as its checkpoint stores them (safetensors::normStorage); a
// family that plans its own declares its overload beside its layout.
template <class Layout> [[nodiscard]] SafetensorsTargetPlan safetensorsTargetPlan(const Layout &layout) {
  return {layout.layers, layout.leadingDenseLayers,
          [layout](const SafetensorsCheckpoint &checkpoint, gguf::PlanOptions options) {
            const safetensors::NormStorage norms = safetensors::normStorage(checkpoint, layout);
            return SafetensorsTargetPlan::Images{safetensors::planImages(checkpoint, layout, norms, options),
                                                 safetensors::float32Norms(norms, layout)};
          }};
}

// Whether --weights auto plans any image of `checkpoint` in another format
// than --weights original does (gguf::imageFormats of the two plans), from
// its headers alone: what the app offers the option by (model-check).
[[nodiscard]] inline bool weightsAutoChanges(const SafetensorsTargetPlan &plan,
                                             const SafetensorsCheckpoint &checkpoint) {
  const auto formats = [&](gguf::Float16Weights weights) {
    std::vector<std::string> result;
    for (const gguf::Image &image : plan.images(checkpoint, {.weights = weights}).images)
      result.push_back(gguf::imageFormats(image));
    return result;
  };
  return formats(gguf::Float16Weights::Q8_0) != formats(gguf::Float16Weights::Published);
}

class SafetensorsTargetLoader final : public TargetImages {
public:
  // Plans every image of the checkpoint in directory once, as its family's
  // `plan` does. With a table store (table.cacheBytes), the rows of a table
  // the family adds stream through it (streamTable()) instead of loading
  // (table()). With expert memory (experts.memoryBytes), a MoE target's
  // routed experts stream (streamedExperts()) instead of loading with their
  // layers. 16-bit projections load as `weights` says (gguf::float16Planes).
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                          const SafetensorsTargetPlan &plan, SparseRowStoreOptions table = {},
                          ExpertStoreOptions experts = {}, gguf::Float16Weights weights = gguf::Float16Weights::Published);

  // The streamed table, its store reading the table's native rows from the
  // checkpoint's open files (readRepackRows over its shards); none when the
  // table loads.
  [[nodiscard]] std::optional<StreamedTable> streamTable() const;
  // The checkpoint, for what a family reads beside its images (Flash-Next's
  // PLE hash).
  [[nodiscard]] const SafetensorsCheckpoint &checkpoint() const noexcept { return planned_->checkpoint; }
  // Whether its images hold F32 norms (safetensors::float32Norms).
  [[nodiscard]] bool float32Norms() const noexcept { return float32Norms_; }

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images;
  };
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, std::shared_ptr<Planned> planned,
                          const SafetensorsTargetPlan &plan, SparseRowStoreOptions table, ExpertStoreOptions experts,
                          gguf::Float16Weights weights);

  std::shared_ptr<Planned> planned_;
  bool float32Norms_ = false;
};

} // namespace splash::model
