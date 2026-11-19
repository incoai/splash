#pragma once

// Source adapter for a safetensors target (MLX's, Model Optimizer's or
// compressed-tensors'): its block images (model/SafetensorsImage.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/TargetImages.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace splash::model {

class SafetensorsTargetLoader final : public TargetImages {
public:
  // Plans every image of the checkpoint in directory once, of the family
  // whose GGUFs are of `architecture` (safetensors::planImages). With a
  // table store (table.cacheBytes), the rows of a table the family adds
  // stream through it (streamTable()) instead of loading (table()). With
  // expert memory (experts.memoryBytes), a MoE target's routed experts
  // stream (streamedExperts()) instead of loading with their layers.
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                          const QwenTargetDimensions &geometry, std::string_view architecture,
                          SparseRowStoreOptions table = {}, ExpertStoreOptions experts = {});

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
                          const QwenTargetDimensions &geometry, std::string_view architecture,
                          SparseRowStoreOptions table, ExpertStoreOptions experts);

  std::shared_ptr<Planned> planned_;
  bool float32Norms_ = false;
};

} // namespace splash::model
