#pragma once

// Source adapter for a Qwen GGUF: its images (model/GgufImageLayout.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SparseRowStore.hpp"
#include "model/WeightImages.hpp"

namespace splash::model {

// The .gguf files of a target directory: its one GGUF, or every part of a
// split one (NAME-0000K-of-0000N.gguf, K = 1..N) in split order.
[[nodiscard]] std::vector<std::filesystem::path> findTargetGgufs(const std::filesystem::path &directory);

// Reading a qwen4exp target's PLE table from its GGUF (--ple-ssd-cache)
// instead of writing ple.bin: a SparseRowStore of cacheBytes, 0 for none.
struct PleStreaming final {
  uint64_t cacheBytes = 0;
  uint32_t readers = 16;
};

class GgufTargetLoader final {
public:
  // Plans every image from the GGUF's metadata once. With streaming and a
  // PLE table, ple() is refused and openPleStore() reads the table instead.
  GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                   const std::vector<std::filesystem::path> &paths, const QwenTargetDimensions &geometry,
                   PleStreaming streaming = {});
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // A qwen4exp target's PLE n-gram table and its hash.
  [[nodiscard]] WeightFile ple();
  [[nodiscard]] const gguf::PleHash &pleHash() const noexcept { return planned_->pleHash; }
  // Whether the PLE table streams (PleStreaming), and then its rows' format,
  // its store, which reads the open file its metadata was parsed from, and
  // the record ple.bin's file would have: the weights' fingerprint is the
  // same streamed or not, as the gathered rows are.
  [[nodiscard]] bool pleStreamed() const noexcept { return planned_->pleRows.has_value(); }
  [[nodiscard]] uint32_t pleFormat() const;
  [[nodiscard]] std::shared_ptr<SparseRowStore> openPleStore() const;
  [[nodiscard]] WeightFileRecord pleRecord() const;
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

private:
  // The GGUF's files and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::vector<std::filesystem::path> &paths);
    void checkUnchanged() const;
    std::deque<WeightSource> files; // split order; tensor rows point into them
    std::vector<gguf::Image> images; // layers, head, embedding, then ple
    uint32_t layers = 0;
    gguf::PleHash pleHash;
    std::optional<gguf::StreamedRows> pleRows; // set when the table streams
    PleStreaming streaming;
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  std::optional<GgufRotation> rotation_;
};

// A qwen4exp MTP head's GGUF (mtp/ in the model root): one image, written
// and read as a target's (gguf::planMtpImage).
class GgufMtpLoader final {
public:
  GgufMtpLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &path,
                const QwenTargetDimensions &geometry);
  GgufMtpLoader(const GgufMtpLoader &) = delete;
  GgufMtpLoader &operator=(const GgufMtpLoader &) = delete;
  [[nodiscard]] WeightFile open();

private:
  struct Planned {
    explicit Planned(const std::filesystem::path &path) : source(path) {}
    WeightSource source;
    gguf::Image image;
  };
  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
};

} // namespace splash::model
