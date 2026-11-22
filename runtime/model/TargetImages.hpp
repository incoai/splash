#pragma once

// What both sources' loaders of a target (GgufTargetLoader,
// SafetensorsTargetLoader) hold alike: the images they planned, each weight
// layer's (QwenTargetDimensions::weightLayers) then the head's, the
// embedding's and a table's, which they write into memory as a family reads
// them, and the routed experts' blocks when those stream. The loaders differ
// in their source, their planner and how a streamed table's rows are read.

#include "model/ExpertStore.hpp"
#include "model/GgufImage.hpp"
#include "model/GgufPreparation.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SparseRowStore.hpp"
#include "model/WeightImages.hpp"

#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace splash::model {

class TargetImages {
public:
  // A table whose rows stream from the source (a loader's streamTable()):
  // the store that reads them, their format (GGUF_FMT_*), and the record of
  // its image, which the weights' fingerprint takes as when the table loads,
  // as the rows gathered are the same.
  struct StreamedTable final {
    std::shared_ptr<SparseRowStore> store;
    uint32_t format = 0;
    WeightFileRecord record;
  };

  TargetImages(const TargetImages &) = delete;
  TargetImages &operator=(const TargetImages &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index) {
    if (index >= layers_) throw WeightStoreError("target layer is out of range");
    return open(index);
  }
  [[nodiscard]] WeightFile head() { return open(layers_); }
  [[nodiscard]] WeightFile embedding() { return open(layers_ + 1); }
  // The image of the table the family adds after the embedding (Qwen3.8-
  // Flash-Next's PLE n-gram table), refused when its rows stream.
  [[nodiscard]] WeightFile table() {
    const gguf::Image *plan = tablePlan();
    if (!plan) throw WeightStoreError("the target has no table");
    if (plan->streamedRows) throw WeightStoreError("the target's table streams");
    return open(layers_ + 2);
  }
  // Whether the routed experts stream, then the store's options and every
  // MoE weight layer's experts (a block each, which every layer that runs it
  // shares; those of the leading dense layers' none), which read the
  // source's open files.
  [[nodiscard]] bool streamsExperts() const noexcept { return experts_.memoryBytes != 0; }
  [[nodiscard]] const ExpertStoreOptions &expertStoreOptions() const noexcept { return experts_; }
  [[nodiscard]] StreamedBlocks streamedExperts() const {
    const std::span layers = std::span(*images_).first(layers_);
    return {gguf::streamedExperts(layers.subspan(firstMoeLayer_)), images_, {checkUnchanged_}};
  }

protected:
  // The images `images` points to, which the loader plans once constructed,
  // owned with the sources they read; `checkUnchanged` finds those sources
  // unwritten since they were planned. With a table store
  // (table.cacheBytes), the rows of a table the family adds stream instead
  // of loading; with expert memory (experts.memoryBytes), a MoE target's
  // routed experts.
  TargetImages(metal::MetalBackend &backend, WeightImages &images, std::shared_ptr<std::vector<gguf::Image>> planned,
               std::function<void()> checkUnchanged, const QwenTargetDimensions &geometry,
               SparseRowStoreOptions table, ExpertStoreOptions experts)
      : backend_(backend), weightImages_(images), images_(std::move(planned)),
        checkUnchanged_(std::move(checkUnchanged)), layers_(geometry.weightLayers()),
        firstMoeLayer_(geometry.leadingDenseLayers), table_(std::move(table)), experts_(experts) {}
  ~TargetImages() = default;

  // What the planner takes: whether experts and a table stream.
  [[nodiscard]] gguf::PlanOptions planOptions() const noexcept {
    return {.streamExperts = experts_.memoryBytes != 0, .streamTable = table_.cacheBytes != 0};
  }
  // The plan of the table's image, or null for a family without one.
  [[nodiscard]] const gguf::Image *tablePlan() const noexcept {
    const size_t index = layers_ + 2;
    return index < images_->size() ? &(*images_)[index] : nullptr;
  }
  [[nodiscard]] const SparseRowStoreOptions &tableOptions() const noexcept { return table_; }
  // The record of a streamed table's image (StreamedTable::record).
  [[nodiscard]] WeightFileRecord tableRecord(const gguf::Image &plan) const {
    return imageRecord(weightImages_, "target/", plan);
  }

private:
  [[nodiscard]] WeightFile open(size_t index) {
    return loadImage(backend_, weightImages_, "target/", {images_, &(*images_)[index]}, checkUnchanged_);
  }

  metal::MetalBackend &backend_;
  WeightImages &weightImages_;
  std::shared_ptr<std::vector<gguf::Image>> images_;
  std::function<void()> checkUnchanged_;
  uint32_t layers_ = 0, firstMoeLayer_ = 0;
  SparseRowStoreOptions table_;
  ExpertStoreOptions experts_;
};

} // namespace splash::model
