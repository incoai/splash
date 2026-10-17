#pragma once

// Source adapter for a Qwen GGUF: its images (model/GgufImageLayout.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "model/ExpertStore.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SparseRowStore.hpp"
#include "model/WeightImages.hpp"

namespace splash::model {

// The .gguf files of a target directory: its one GGUF, or every part of a
// split one (NAME-0000K-of-0000N.gguf, K = 1..N) in split order.
[[nodiscard]] std::vector<std::filesystem::path> findTargetGgufs(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  // A table whose rows stream from the GGUF (streamTable()): the store that
  // reads them, their format (GGUF_FMT_*), and the record of its image, which
  // the weights' fingerprint takes as when the table loads, as the rows
  // gathered are the same.
  struct StreamedTable final {
    std::shared_ptr<SparseRowStore> store;
    uint32_t format = 0;
    WeightFileRecord record;
  };

  // Plans every image of the GGUF of `architecture` (its family's layout's
  // ggufArchitecture) from its metadata once. With a table store
  // (table.cacheBytes), the rows of a table the family's GGUF holds stream
  // through it (streamTable()) instead of loading (table()). With expert
  // memory (experts.memoryBytes), a MoE target's routed experts stream
  // (streamedExperts()) instead of loading with their layers.
  // With `draftTokens`, also the head an MTP head drafts from of them
  // (gguf::planDraftHeadImage, draftHead()).
  GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                   const std::vector<std::filesystem::path> &paths, std::string_view architecture,
                   const QwenTargetDimensions &geometry, SparseRowStoreOptions table = {},
                   ExpertStoreOptions experts = {}, std::span<const uint32_t> draftTokens = {});
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The image of the table the family's GGUF holds after the embedding
  // (gguf::planImages: Qwen3.8-Flash-Next's PLE n-gram table), refused when
  // its rows stream.
  [[nodiscard]] WeightFile table();
  // The streamed table, its store reading the open file the metadata was
  // parsed from; none when the table loads.
  [[nodiscard]] std::optional<StreamedTable> streamTable() const;
  // The draft head the loader was asked for (draftTokens).
  [[nodiscard]] WeightFile draftHead();
  // The GGUF's header, for what a family reads beside its images (Flash-Next's
  // PLE hash).
  [[nodiscard]] const GgufFile &file() const noexcept { return planned_->source.file(); }
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return file().rotation(); }
  // Whether the routed experts stream, then the store's options and every
  // layer's experts, which read the open files the metadata was parsed from.
  [[nodiscard]] bool streamsExperts() const noexcept { return planned_->experts.memoryBytes != 0; }
  [[nodiscard]] const ExpertStoreOptions &expertStoreOptions() const noexcept { return planned_->experts; }
  [[nodiscard]] StreamedBlocks streamedExperts() const;

private:
  // The GGUF's files and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::vector<std::filesystem::path> &paths) : source(paths) {}
    OpenGguf source;
    // Each layer's image, then the head's, the embedding's and a table's
    // (gguf::planImages), then a draft head's.
    std::vector<gguf::Image> images;
    uint32_t layers = 0;
    std::optional<size_t> draftHead;
    SparseRowStoreOptions table;
    ExpertStoreOptions experts;
  };
  [[nodiscard]] WeightFile open(size_t index);
  // The plan of the table's image, or null for a family without one.
  [[nodiscard]] const gguf::Image *tablePlan() const noexcept;

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
};

// Qwen3.8-Flash-Next's MTP head, the one GGUF in the model root's mtp/
// (install/upstream.py select_mtp): its image, written and read as a
// target's (gguf::planMtpImage).
class GgufMtpLoader final {
public:
  GgufMtpLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &path,
                const QwenTargetDimensions &geometry);
  GgufMtpLoader(const GgufMtpLoader &) = delete;
  GgufMtpLoader &operator=(const GgufMtpLoader &) = delete;
  [[nodiscard]] WeightFile open();

private:
  struct Planned {
    explicit Planned(const std::filesystem::path &path) : source({path}) {}
    OpenGguf source;
    gguf::Image image;
  };
  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
};

} // namespace splash::model
