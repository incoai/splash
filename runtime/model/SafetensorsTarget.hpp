#pragma once

// Source adapter for a safetensors target (MLX's, Model Optimizer's or
// compressed-tensors'): its block images (model/SafetensorsImage.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include "model/ExpertStore.hpp"
#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/SparseRowStore.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace splash::model {

class SafetensorsTargetLoader final {
public:
  // Plans every image of the checkpoint in directory once. With expert
  // memory (experts.memoryBytes), a MoE target's routed experts stream
  // (streamedExperts()) instead of loading with their layers. With table
  // memory (table.cacheBytes), PLE rows stream through that cache.
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                          const QwenTargetDimensions &geometry, ExpertStoreOptions experts = {},
                          SparseRowStoreOptions table = {});
  SafetensorsTargetLoader(const SafetensorsTargetLoader &) = delete;
  SafetensorsTargetLoader &operator=(const SafetensorsTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The PLE image, or its row store when it streams. tableRows() names the
  // same concatenated row count in either mode.
  [[nodiscard]] WeightFile table();
  [[nodiscard]] std::optional<StreamedTable> streamTable() const;
  [[nodiscard]] const SafetensorsCheckpoint &checkpoint() const noexcept { return planned_->checkpoint; }
  [[nodiscard]] uint64_t tableRows() const;
  // Whether its images hold F32 norms (Flash-Next and transformers names).
  [[nodiscard]] bool float32Norms() const noexcept { return float32Norms_; }
  // Whether the routed experts stream, then the store's options and every
  // layer's experts, which read the checkpoint's open files.
  [[nodiscard]] bool streamsExperts() const noexcept { return experts_.memoryBytes != 0; }
  [[nodiscard]] const ExpertStoreOptions &expertStoreOptions() const noexcept { return experts_; }
  [[nodiscard]] StreamedBlocks streamedExperts() const;

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images; // layers, head, embedding, optional PLE table
    uint32_t layers = 0;
    SparseRowStoreOptions table;
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  bool float32Norms_ = false;
  ExpertStoreOptions experts_;
};

} // namespace splash::model
