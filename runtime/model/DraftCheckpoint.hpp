#pragma once

#include "model/AffinePreparation.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

struct DFlashDraftLayout;

namespace affine {
struct PlannedCheckpoint;
}

// A DFlash2 checkpoint as its repository releases it, config.json and BF16
// safetensors -> the draft files of a Splash package (DFlashDraft.cpp):
// every projection quantized to affine Q4 as those drafts were, every other
// tensor copied as stored. The checkpoint is planned once; each file is
// written into memory when it is opened.
class DraftCheckpointLoader final {
public:
  DraftCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                        const DFlashDraftLayout &layout);
  ~DraftCheckpointLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, then model.bin
};

// Every planned file of a layout, its sections at their offsets: the layers,
// then model.bin.
[[nodiscard]] std::vector<affine::Image> draftCheckpointImages(const DFlashDraftLayout &layout);

} // namespace splash::model
