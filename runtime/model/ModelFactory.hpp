#pragma once

#include "ops/Vision.hpp"
#include "DFlashDraft.hpp"
#include "ModelDescriptor.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "QwenVision.hpp"
#include "ops/PageStorage.hpp"
#include "ops/ExecutionPlans.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <variant>

namespace splash::model {

class QwenStateStorage;
class VisionLoader;

using TargetWeights = std::variant<Qwen3_8Weights, Qwen3_6MoeWeights>;

struct ModelPackage final {
  ModelDescriptor descriptor;
  // The memory of every image the weights below are views of.
  std::shared_ptr<WeightImages> images;
  TargetWeights target;
  DFlashDraftWeights draft;
  QwenVisionWeights vision;
  std::string manifestFingerprintSha256;

  [[nodiscard]] const std::string &name() const noexcept {
    return descriptor.name;
  }
  [[nodiscard]] kv::Layout targetKvLayout(kv::Format format) const noexcept {
    auto layout = descriptor.targetKvLayout;
    layout.format = format;
    return layout;
  }
  [[nodiscard]] CompositeStateLayout stateLayout() const noexcept {
    return descriptor.stateLayout;
  }
  [[nodiscard]] uint32_t maximumContextTokens() const noexcept {
    return descriptor.capabilities.maximumContextTokens;
  }
  [[nodiscard]] uint64_t targetActualAllocatedBytes() const noexcept {
    return std::visit([](const auto &weights) {
      return weights.actualAllocatedBytes;
    }, target);
  }
  [[nodiscard]] const std::string &targetManifestFingerprint() const noexcept {
    return std::visit([](const auto &weights) -> const std::string & {
      return weights.manifestFingerprintSha256;
    }, target);
  }
  [[nodiscard]] std::span<const WeightFileRecord> targetFiles() const noexcept {
    return std::visit([](const auto &weights) ->
                          std::span<const WeightFileRecord> {
      return weights.files;
    }, target);
  }
};

// Model execution resources, which the engine assembles. What a request's
// start allocates is admitted through the state storage.
struct RuntimeContext final {
  metal::MetalBackend &backend;
  const ModelPackage &package;
  kv::PageStorage &kvPages;
  QwenStateStorage &stateStorage;
  const ops::ExecutionPlans &operators;
};

// Validates only the interface between independently defined target and draft
// architectures. Each architecture validates its own tensor and state layout.
void requireCompatibleModelPackage(const ModelPackage &package);

// The bytes of every image the model's weights load into.
[[nodiscard]] uint64_t modelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor);

// The vision role's upstream source, planned; null for a package's vision
// file or a model without vision.
[[nodiscard]] std::unique_ptr<VisionLoader> planVisionLoader(const std::filesystem::path &root,
                                                             const ModelDescriptor &descriptor);
// The vision role: written by `loader` when there is one, else read from the
// package's vision file; empty weights for a model without vision.
[[nodiscard]] QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                                  const std::filesystem::path &root,
                                                  const ModelDescriptor &descriptor, const VisionLoader *loader);

// Production loading is selected by the validated package descriptor. There
// is one shared engine and DFlash controller; only model execution differs.
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const std::filesystem::path &root,
                 const ModelDescriptor &descriptor);

[[nodiscard]] ModelMemoryPlan
plannedRuntimeMemory(const ModelPackage &package,
                     const ops::ExecutionPlans &operators,
                     kv::Format format);
[[nodiscard]] std::unique_ptr<RuntimeModel>
createRuntime(RuntimeContext context);

} // namespace splash::model
