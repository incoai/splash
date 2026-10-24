#pragma once

#include "DFlashDraft.hpp"
#include "Flint.hpp"
#include "Model.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "Qwen4Exp.hpp"
#include "RopeScaling.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>

namespace splash::model {

using TargetLayout = std::variant<Qwen3_8Layout, Qwen3_6MoeLayout, Qwen4ExpLayout, FlintLayout>;

// The families Splash serves, values of the layout types TargetLayout holds,
// one type for each architecture's code. A family of an architecture is one
// more value here. Refusals name them in this order, and a config's model type
// finds its family among those of its type in it (describeSourceModel).
inline constexpr std::tuple kFamilies{Qwen3_8Layout{}, Qwen3_6MoeLayout{}, kQwen3_5_122BLayout, Qwen4ExpLayout{},
                                      FlintLayout{}};

// Calls `use` with each family's layout, in kFamilies' order.
template <class Use> constexpr void forEachFamily(Use use) {
  std::apply([&](const auto &...family) { (use(family), ...); }, kFamilies);
}

// Where a model's weights come from: a safetensors checkpoint (MLX's, Model
// Optimizer's or compressed-tensors') or a GGUF, prepared into images when it
// loads. The vision tower is None for a model installed with --language-only.
// Zero is no source: a descriptor states its sources (makeModelDescriptor),
// and one that leaves either unset is not valid.
enum class TargetSource : uint8_t { Safetensors = 1, Gguf };
enum class VisionSource : uint8_t { Safetensors = 1, Gguf, None };

// A model's metadata, validated before its weight buffers are loaded. The
// engine consumes capabilities; model loading consumes the concrete layouts.
struct ModelDescriptor final {
  std::string name;
  TargetLayout target;
  DFlashDraftLayout draft;
  ops::VisionLayout vision;
  ModelCapabilities capabilities;
  kv::Layout targetKvLayout;
  CompositeStateLayout stateLayout;
  // Container selection belongs to loading; runtime dispatch follows each weight.
  TargetSource targetSource{};
  VisionSource visionSource{};
  // The SHA-256 of the record that names the digest of every source file,
  // an assembly's model.json, which the installer verifies at every start
  // (inspectModelRoot): what every image is written from
  // (WeightFileRecord).
  std::string sourceIdentity;
  // The target's RoPE scaling (servedFor), which stretches the window
  // capabilities states beyond the layout's native one.
  RopeScaling ropeScaling;

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
                                                  ops::VisionLayout vision,
                                                  TargetSource targetSource,
                                                  VisionSource visionSource);
[[nodiscard]] ModelDescriptor
inspectModelRoot(const std::filesystem::path &root);
// `model` served for a context of `contextTokens`, zero for auto: as it is
// within its native window, and past it with YaRN by the context over the
// native window (RopeScaling), whose window is that context
// (QwenTargetDimensions::servedContextTokens). std::invalid_argument past
// the family's YaRN window, or the native one of a family YaRN does not
// serve, or for a model that is already scaled.
[[nodiscard]] ModelDescriptor servedFor(ModelDescriptor model, uint32_t contextTokens);

// The family of an upstream model's configuration, checked by the rules
// inspectModelRoot holds its assembly to: config, the target's config.json
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
