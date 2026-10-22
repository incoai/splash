#include "ModelFactory.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/SparseRowStore.hpp"
#include "model/VisionLoader.hpp"

#include <sys/sysctl.h>

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::model {

void requireCompatibleModel(const LoadedModel &model) {
  if (!model.descriptor.valid() ||
      model.descriptor.draft != model.draft.layout ||
      !std::visit(
          [&](const auto &target) {
            return model.descriptor.target == TargetLayout{target.layout} &&
                   target.layout.vocabularySize ==
                       model.draft.layout.vocabularySize;
          },
          model.target)) {
    throw std::invalid_argument(
        "target and draft model interfaces are incompatible");
  }
}

std::unique_ptr<VisionLoader> planVisionLoader(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  if (descriptor.visionSource != VisionSource::Safetensors && descriptor.visionSource != VisionSource::Gguf)
    return nullptr;
  return std::make_unique<VisionLoader>(root / "vision", descriptor.visionSource, descriptor.vision);
}

VisionTowerWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                     const VisionLoader *loader) {
  return loader ? loadVisionTowerWeights(backend, images, *loader) : VisionTowerWeights{};
}

namespace {

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_8Layout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen4ExpLayout &layout,
                         const QwenTargetFiles &files) {
  return loadQwen4ExpWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const FlintLayout &layout, const QwenTargetFiles &files) {
  return loadFlintWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Inkling2LoopLayout &layout,
                         const QwenTargetFiles &files) {
  return loadInkling2LoopWeights(backend, layout, files);
}

} // namespace

SparseRowStoreOptions streamedTableRows(const Qwen3_8Layout &) { return {}; }

SparseRowStoreOptions streamedTableRows(const Qwen3_6MoeLayout &) { return {}; }

SparseRowStoreOptions streamedTableRows(const FlintLayout &) { return {}; }

StreamedParts StreamedParts::loaded(const LoadedModel &model, bool kv) noexcept {
  const SparseRowStore *table = model.streamedTable();
  return {.experts = model.expertStore() != nullptr, .tableRowBytes = table ? table->table().rowBytes : 0, .kv = kv};
}

namespace {

// The target of `descriptor` in `root`, its images loaded into `images` as
// `options` says.
TargetWeights loadTarget(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &root,
                         const ModelDescriptor &descriptor, const ModelLoadOptions &options) {
  // The weights that stream from their files read on the engine's readers.
  const ExpertStoreOptions experts{.memoryBytes = options.expertMemoryBytes,
                                   .floorBytes = options.expertFloorBytes,
                                   .floorShare = options.expertFloorShare,
                                   .readers = options.readers};
  return std::visit(
      [&](const auto &layout) -> TargetWeights {
        const std::filesystem::path directory = root / "target";
        // A streamed table's store: its memory and readers here, the rows a
        // step stages and prefetches from the family.
        SparseRowStoreOptions table = streamedTableRows(layout);
        table.cacheBytes = options.streamedTableBytes;
        table.readers = options.readers;
        switch (descriptor.targetSource) {
        case TargetSource::Safetensors: {
          SafetensorsTargetLoader loader(backend, images, directory, safetensorsTargetPlan(layout), table, experts);
          return readTarget(backend, layout, std::ref(loader));
        }
        case TargetSource::Gguf: {
          GgufTargetLoader loader(backend, images, findTargetGgufs(directory), ggufTargetPlan(layout), table,
                                  experts);
          return readTarget(backend, layout, std::ref(loader));
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      descriptor.target);
}

} // namespace

QwenTargetGeometry plannedTargetGeometry(metal::MetalBackend &backend, const std::filesystem::path &root,
                                         const ModelDescriptor &descriptor, uint64_t streamedTableBytes,
                                         std::shared_ptr<SsdReaders> readers) {
  if (!descriptor.valid()) throw std::invalid_argument("model descriptor is invalid");
  // Released before the first image, the images load reserved.
  WeightImages images(backend, descriptor.sourceIdentity);
  images.release();
  const TargetWeights target = loadTarget(backend, images, root, descriptor,
                                          {.streamedTableBytes = streamedTableBytes, .readers = std::move(readers)});
  return std::visit(
      [&](const auto &weights) { return qwenTargetGeometry(weights, descriptor.draft.targetLayerIds()); }, target);
}

LoadedModel loadModel(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor,
                              const ModelLoadOptions &options) {
  LoadedModel result;
  result.descriptor = descriptor;
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  // A control vector's file is read before any weight: the forward pass of
  // Qwen3.8-Flash-Next (its hyper-connection streams), Qwen3.8-27B and
  // Qwen3.6-35B-A3B (the residual), each its architecture's first family
  // (kFamilies), applies one, no other family's does.
  std::optional<ControlVectorTable> steering;
  if (options.controlVector) {
    const auto architecture = std::visit(
        [&](const auto &layout) -> std::pair<std::string_view, std::pair<uint32_t, uint32_t>> {
          using Layout = std::decay_t<decltype(layout)>;
          if constexpr (std::is_same_v<Layout, Qwen4ExpLayout> || std::is_same_v<Layout, Qwen3_8Layout> ||
                        std::is_same_v<Layout, Qwen3_6MoeLayout>)
            if (layout.family == Layout{}.family)
              return {Layout::ggufArchitecture, std::pair{layout.layers, layout.hiddenSize}};
          throw std::invalid_argument("--control-vector steers " + std::string(Qwen4ExpLayout{}.family) + ", " +
                                      std::string(Qwen3_8Layout{}.family) + " and " +
                                      std::string(Qwen3_6MoeLayout{}.family) + " only, not " +
                                      result.descriptor.name);
        },
        result.descriptor.target);
    steering = readControlVector(options.controlVector, architecture.first, architecture.second.first,
                                 architecture.second.second);
  }
  result.images = std::make_shared<WeightImages>(backend, result.descriptor.sourceIdentity);
  WeightImages &images = *result.images;
  // Every source's metadata is checked before the first image is written:
  // the vision tower's and the draft's here, the target's by its loader.
  const auto vision = planVisionLoader(root, result.descriptor);
  DraftCheckpointLoader draft(backend, images, root / "draft", result.descriptor.draft);
  result.target = loadTarget(backend, images, root, result.descriptor, options);
  if (steering) {
    // Its directions are the target's: its allocation counts with theirs. A
    // Prism-rotated target's residual is not the vector's space.
    std::visit(
        [&](auto &target) {
          if (target.inputsRotated)
            throw std::invalid_argument("--control-vector cannot steer " + result.descriptor.name +
                                        ": its GGUF's inputs are Prism-rotated");
          const uint64_t before = backend.memoryStats().allocatedBytes;
          target.controlVector = loadControlVector(backend, *steering);
          target.actualAllocatedBytes += metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
        },
        result.target);
  }
  result.draft = loadDFlashDraftWeights(backend, draft, result.descriptor.draft);
  result.vision = loadVisionWeights(backend, images, vision.get());

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  records.insert(records.end(), result.draft.files.begin(),
                 result.draft.files.end());
  records.insert(records.end(), result.vision.files.begin(),
                 result.vision.files.end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModel(result);
  return result;
}

std::string stepCostsKey(const LoadedModel &model, const kv::PageStorage &kvPages,
                         const QwenStateStorage &stateStorage, const ops::ExecutionPlans &operators,
                         const metal::MetalBackend &backend, std::string_view buildId) {
  // The macOS version and build, as the kernel names them.
  const auto kernel = [](const char *name) {
    std::array<char, 64> value{};
    size_t size = value.size();
    return sysctlbyname(name, value.data(), &size, nullptr, 0) ? std::string() : std::string(value.data());
  };
  const ExpertStoreStats experts = model.expertStore() ? model.expertStore()->stats() : ExpertStoreStats{};
  const SparseRowStore *table = model.streamedTable();
  return "verify budget step costs models " + model.manifestFingerprintSha256 + " kv " +
         std::string(kv::formatName(kvPages.layout().format)) + " state " +
         std::string(ops::formatName(stateStorage.layout().target.format)) + " int8 activations " +
         std::to_string(operators.linear().int8Activations() == ops::Int8Activations::On) + " experts " +
         std::to_string(experts.capBytes) + " floor " + std::to_string(experts.floorBytes) + " table " +
         std::to_string(table ? table->committedBytes() : 0) + " device " + backend.capabilities().deviceName +
         " macos " + kernel("kern.osproductversion") + " " + kernel("kern.osversion") + " build " +
         std::string(buildId) + " lanes ";
}

} // namespace splash::model
