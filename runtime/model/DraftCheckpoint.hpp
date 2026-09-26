#pragma once

#include "model/PreparedFiles.hpp"

#include <memory>
#include <span>

namespace splash::model {

struct DFlashDraftLayout;

// A DFlash2 checkpoint as its repository releases it, config.json and BF16
// safetensors -> the packed draft files of a Splash package (DFlashDraft.cpp):
// every projection quantized to affine Q4 as those drafts were, every other
// tensor copied as stored. The checkpoint is planned once; each file is
// prepared when it is opened.
class DraftCheckpointLoader final {
public:
  DraftCheckpointLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                        const DFlashDraftLayout &layout, PreparationCheck admitConversion = {});
  ~DraftCheckpointLoader();
  // Every file's cache identity and size, layers first, for the model's disk
  // check before the first file is written.
  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept;
  // Writes every missing file and maps none.
  void prepare();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t preparedDraftBytes(const DFlashDraftLayout &layout);

} // namespace splash::model
