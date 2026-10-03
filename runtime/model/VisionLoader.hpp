#pragma once

#include "model/ModelDescriptor.hpp"
#include "model/PreparedFiles.hpp"
#include "ops/Vision.hpp"

#include <filesystem>
#include <memory>

namespace splash::model {

// Source adapter for the vision tower: the vision_tower.* tensors of an MLX
// checkpoint or a GGUF mmproj, both prepared into the packed vision/model.bin
// (model/VisionPreparation.hpp). Construction validates the source's
// metadata, plans the prepared file and computes its key, which hashes each
// source file's tensor data once (later starts reuse the digest remembered
// for the unchanged file); tensor values are converted only when preparing.
class VisionLoader final {
public:
  // check runs on every load, admitConversion on a cache miss.
  VisionLoader(const std::filesystem::path &directory, VisionSource source, const ops::VisionLayout &layout,
               PreparationCheck check, PreparationCheck admitConversion);
  ~VisionLoader();
  VisionLoader(const VisionLoader &) = delete;
  VisionLoader &operator=(const VisionLoader &) = delete;

  [[nodiscard]] const ops::VisionLayout &layout() const noexcept;
  // The prepared file's cache identity and size, for disk budgeting.
  [[nodiscard]] const PreparedWeight &weight() const noexcept;
  // The prepared file, reused or written now.
  [[nodiscard]] std::filesystem::path prepare() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Bytes of the packed layout, which every source prepares.
[[nodiscard]] uint64_t preparedVisionBytes(const ops::VisionLayout &layout);

// Throws unless the layout is one the packed file can hold: every size set,
// the heads covering the width, the merger's width the merged patches' and
// the patch embedding's width that of two frames of RGB patches.
void requireVisionLayout(const ops::VisionLayout &layout);

} // namespace splash::model
