#pragma once

#include "ops/Vision.hpp"
#include "DFlashDraft.hpp"
#include "ModelDescriptor.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "QwenVision.hpp"
#include "VisionLoader.hpp"
#include "ops/AneFfn.hpp"
#include "ops/PageStorage.hpp"
#include "ops/ExecutionPlans.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <variant>

namespace splash::model {

class QwenStateStorage;

using TargetWeights = std::variant<Qwen3_8Weights, Qwen3_6MoeWeights>;

struct ModelPackage final {
  ModelDescriptor descriptor;
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
  // The share of the prefill FFN's Neural Engine split (calibrateAneFfn), or
  // 0 for none.
  double aneFfnShare = 0.0;
};

// Validates only the interface between independently defined target and draft
// architectures. Each architecture validates its own tensor and state layout.
void requireCompatibleModelPackage(const ModelPackage &package);

[[nodiscard]] uint64_t preparedModelWeightBytes(const std::filesystem::path &root,
                                                 const ModelDescriptor &descriptor);

// The vision role's upstream source, planned for preparation; null for a
// packed vision file or a model without vision.
[[nodiscard]] std::unique_ptr<VisionLoader>
planVisionLoader(metal::MetalBackend &backend, const std::filesystem::path &root,
                 const ModelDescriptor &descriptor, PreparationCheck admitConversion);
// The vision role: prepared by `loader` when there is one, else the packed
// file; empty weights for a model without vision.
[[nodiscard]] QwenVisionWeights
loadVisionWeights(metal::MetalBackend &backend, const std::filesystem::path &root,
                  const ModelDescriptor &descriptor, const VisionLoader *loader);

// Production loading is selected by the validated package descriptor. There
// is one shared engine and DFlash controller; only model execution differs.
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const std::filesystem::path &root,
                 const ModelDescriptor &descriptor, PreparationCheck admitConversion);

// The Neural Engine split of the package's prefill FFN calibrated on this
// device (ops::AneFfn::calibrate) on a prefill arena of its own; a target
// other than a dense one, or one whose FFN the split does not take, gets
// none.
[[nodiscard]] ops::AneFfn::Calibration calibrateAneFfn(metal::MetalBackend &backend, const ModelPackage &package,
                                                       const ops::ExecutionPlans &operators, kv::Format format);
[[nodiscard]] ModelMemoryPlan
plannedRuntimeMemory(const DeviceCapabilities &device,
                     const ModelPackage &package,
                     const ops::ExecutionPlans &operators,
                     kv::Format format, double aneFfnShare = 0.0);
[[nodiscard]] std::unique_ptr<RuntimeModel>
createRuntime(RuntimeContext context);

} // namespace splash::model
