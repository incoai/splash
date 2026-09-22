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
#include <string>
#include <variant>

namespace splash::model {

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
  [[nodiscard]] kv::Layout targetKvLayout(
      kv::Format format = kv::Format::Int8) const noexcept {
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

// Model execution resources; physical memory admission remains governed by
// the engine through admitAllocation.
struct RuntimeContext final {
  metal::MetalBackend &backend;
  metal::AllocationAdmission admitAllocation;
  const ModelPackage &package;
  kv::PageStorage &kvPages;
  StateStorage &stateStorage;
  const ops::ExecutionPlans &operators;
  uint32_t maximumImagePatches = ops::kMaximumImagePatches;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;
};

// Validates only the interface between independently defined target and draft
// architectures. Each architecture validates its own tensor and state layout.
void requireCompatibleModelPackage(const ModelPackage &package);

// Production loading is selected by the validated package descriptor. There
// is one shared engine and DFlash controller; only model execution differs.
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const std::filesystem::path &root);
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const std::filesystem::path &root,
                 const ModelDescriptor &descriptor);

[[nodiscard]] ModelMemoryPlan
plannedRuntimeMemory(const DeviceCapabilities &device,
                     const ModelPackage &package,
                     const ops::ExecutionPlans &operators,
                     kv::Format format = kv::Format::Int8);
[[nodiscard]] std::unique_ptr<StateStorage>
createStateStorage(metal::MetalBackend &backend,
                   metal::AllocationAdmission admitAllocation,
                   const ModelPackage &package);
[[nodiscard]] std::unique_ptr<RuntimeModel>
createRuntime(RuntimeContext context);

} // namespace splash::model
