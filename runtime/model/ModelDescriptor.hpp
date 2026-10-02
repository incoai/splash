#pragma once

#include "DFlashDraft.hpp"
#include "Model.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "Qwen4Exp.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>

namespace splash::model {

using TargetLayout = std::variant<Qwen3_8Layout, Qwen3_6MoeLayout, Qwen4ExpLayout>;

// Where a model's weights come from: files already in the packed layout, or
// an MLX or GGUF checkpoint, and the draft's DFlash2 checkpoint, prepared
// into cached files when it loads, or None for a family no DFlash2 draft was
// trained for (nullDraftLayout). The vision tower is None for a model
// installed with --language-only.
enum class TargetSource : uint8_t { Packed, Mlx, Gguf };
enum class DraftSource : uint8_t { Packed, Checkpoint, None };
enum class VisionSource : uint8_t { Packed, Mlx, Gguf, None };

// Package metadata validated before weight buffers are loaded. The engine
// consumes capabilities; model loading consumes the concrete layouts.
struct ModelDescriptor final {
  std::string name;
  TargetLayout target;
  DFlashDraftLayout draft;
  ops::VisionLayout vision;
  ModelCapabilities capabilities;
  kv::Layout targetKvLayout;
  CompositeStateLayout stateLayout;
  // Container selection belongs to loading; runtime dispatch follows each weight.
  TargetSource targetSource = TargetSource::Packed;
  DraftSource draftSource = DraftSource::Packed;
  VisionSource visionSource = VisionSource::Packed;

  // A model installed with --language-only has no vision tower: it loads no
  // vision weights and serves no image requests.
  [[nodiscard]] bool hasVision() const noexcept {
    return visionSource != VisionSource::None;
  }
  [[nodiscard]] bool hasDraft() const noexcept {
    return draftSource != DraftSource::None;
  }
  [[nodiscard]] bool valid() const noexcept;
};

// Derives the capabilities and cache layouts the engine consumes from the
// concrete target and draft layouts.
[[nodiscard]] ModelDescriptor makeModelDescriptor(std::string name,
                                                  TargetLayout target,
                                                  DFlashDraftLayout draft,
                                                  ops::VisionLayout vision);
[[nodiscard]] ModelDescriptor
inspectModelPackage(const std::filesystem::path &root);

// The draft layout of a target that decodes without a DFlash2 draft: no draft
// model runs, but the engine's composite state keeps its one-layer,
// eight-wide ring, so the state and cache bookkeeping stay those of a draft
// (a few tens of KiB per state).
[[nodiscard]] DFlashDraftLayout nullDraftLayout(uint32_t hiddenSize, uint32_t vocabularySize,
                                                uint32_t capturedHiddenSize);

} // namespace splash::model
