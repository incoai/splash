#pragma once

// Source adapter for a Qwen GGUF: its images (model/GgufImageLayout.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "model/ExpertStore.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/TargetImages.hpp"

namespace splash::model {

// The .gguf files of a target directory: its one GGUF, or every part of a
// split one, the parts of one name (NAME-0000K-of-0000N.gguf, K = 1..N), in
// split order.
[[nodiscard]] std::vector<std::filesystem::path> findTargetGgufs(const std::filesystem::path &directory);

class GgufTargetLoader final : public TargetImages {
public:
  // Plans every image of the GGUF of `architecture` (its family's layout's
  // ggufArchitecture) from its metadata once. With a table store
  // (table.cacheBytes), the rows of a table the family's GGUF holds stream
  // through it (streamTable()) instead of loading (table()). With expert
  // memory (experts.memoryBytes), a MoE target's routed experts stream
  // (streamedExperts()) instead of loading with their layers.
  GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                   const std::vector<std::filesystem::path> &paths, std::string_view architecture,
                   const QwenTargetDimensions &geometry, SparseRowStoreOptions table = {},
                   ExpertStoreOptions experts = {});

  // The streamed table, its store reading the open file the metadata was
  // parsed from; none when the table loads.
  [[nodiscard]] std::optional<StreamedTable> streamTable() const;
  // The GGUF's header, for what a family reads beside its images (Flash-Next's
  // PLE hash).
  [[nodiscard]] const GgufFile &file() const noexcept { return planned_->source.file(); }
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return file().rotation(); }

private:
  // The GGUF's files and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::vector<std::filesystem::path> &paths) : source(paths) {}
    OpenGguf source;
    std::vector<gguf::Image> images;
  };
  GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images, std::shared_ptr<Planned> planned,
                   std::string_view architecture, const QwenTargetDimensions &geometry, SparseRowStoreOptions table,
                   ExpertStoreOptions experts);

  std::shared_ptr<Planned> planned_;
};

} // namespace splash::model
