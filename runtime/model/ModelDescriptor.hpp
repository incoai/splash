#pragma once

#include "DFlashDraft.hpp"
#include "Model.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>

namespace splash::model {

using TargetLayout = std::variant<Qwen3_8Layout, Qwen3_6MoeLayout>;

// Where a model's weights come from: files already in the packed layout, or
// an MLX or GGUF checkpoint prepared into images when it loads. The vision
// tower is None for a model installed with --language-only.
enum class TargetSource : uint8_t { Packed, Mlx, Gguf };
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
  VisionSource visionSource = VisionSource::Packed;
  // The SHA-256 of the record that names the digest of every source file,
  // an assembly's model.json or a package's manifest.json, which the
  // installer verifies at every start (inspectModelPackage): what every
  // image is written from (WeightFileRecord).
  std::string sourceIdentity;

  // A source model's draft is a DFlash2 checkpoint; a packed package carries
  // its draft packed.
  [[nodiscard]] bool draftFromCheckpoint() const noexcept {
    return targetSource != TargetSource::Packed;
  }

  // A model installed with --language-only has no vision tower: it loads no
  // vision weights and serves no image requests.
  [[nodiscard]] bool hasVision() const noexcept {
    return visionSource != VisionSource::None;
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

} // namespace splash::model
