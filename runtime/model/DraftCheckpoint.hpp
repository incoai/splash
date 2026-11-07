#pragma once

#include "model/GgufImage.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

struct DFlashDraftLayout;

// A DFlash2 checkpoint as its repository releases it, config.json and BF16
// safetensors -> the draft's block images (DFlashDraft.cpp), laid out as a
// target's are (model/GgufImage.hpp): every projection quantized into af4g64
// planes as MLX's affine quantization rounds it, every other tensor copied as
// stored. The checkpoint is planned once; each image is written into memory
// when it is opened.
class DraftCheckpointLoader final {
public:
  DraftCheckpointLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                        const DFlashDraftLayout &layout);
  DraftCheckpointLoader(const DraftCheckpointLoader &) = delete;
  DraftCheckpointLoader &operator=(const DraftCheckpointLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();
  // Whether the checkpoint ships its mask token's embedding (kDraftMaskEmbedding),
  // which model.bin then holds after the selector's codebooks.
  [[nodiscard]] bool hasMaskEmbedding() const noexcept;

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images; // layers, then model.bin
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
};

// The BF16 [hidden] embedding of the mask token a draft's checkpoint may ship,
// in a file of its own beside its weights (mask_embedding.safetensors): its
// draft rows' input in place of the target's token table row, which a
// quantized table can lose (DFlashDraft::addMaskRows).
inline constexpr std::string_view kDraftMaskEmbedding = "mask_embedding";

// The images of a draft checkpoint, in the order DFlashDraft.cpp reads them:
// the layers, then model.bin. The checkpoint outlives them: their rows read
// its tensors.
[[nodiscard]] std::vector<gguf::Image> planDraftImages(const SafetensorsCheckpoint &checkpoint,
                                                       const DFlashDraftLayout &layout);

// The bytes of every image of the draft checkpoint in directory.
[[nodiscard]] uint64_t draftImageBytes(const std::filesystem::path &directory, const DFlashDraftLayout &layout);

} // namespace splash::model
