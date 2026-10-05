#pragma once

#include "DFlashDraft.hpp"
#include "Model.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
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
  [[nodiscard]] std::string_view family() const noexcept {
    return std::visit([](const auto &layout) { return layout.family; }, target);
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

// The family of an upstream model's configuration, checked by the rules
// inspectModelPackage holds its assembly to: config, the target's config.json
// or the one the installer derives from its GGUF; targetFormat and
// visionFormat, the source formats its record names (mlx-affine or gguf;
// none, safetensors or gguf); ggufMetadata, a GGUF target's scalar metadata
// as the installer read it from the header, which every start's planner
// checks in the GGUF (gguf::requireMetadata); and draft, once chosen, the
// DFlash2 draft's config.json. The installer runs it before any weight
// download (the engine's model-check command).
[[nodiscard]] std::string_view
inspectSourceConfiguration(std::string_view targetFormat, std::string_view visionFormat,
                           const std::filesystem::path &config,
                           const std::optional<std::filesystem::path> &ggufMetadata,
                           const std::optional<std::filesystem::path> &draft);

} // namespace splash::model
