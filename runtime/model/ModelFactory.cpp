#include "ModelFactory.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/VisionLoader.hpp"

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

QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                    const VisionLoader *loader) {
  return loader ? loadQwenVisionWeights(backend, images, *loader) : QwenVisionWeights{};
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

} // namespace

SparseRowStoreOptions streamedTableRows(const Qwen3_8Layout &) { return {}; }

SparseRowStoreOptions streamedTableRows(const Qwen3_6MoeLayout &) { return {}; }

StreamedParts StreamedParts::loaded(const LoadedModel &model, bool kv) noexcept {
  const SparseRowStore *table = model.streamedTable();
  return {.experts = model.expertStore() != nullptr, .tableRowBytes = table ? table->table().rowBytes : 0, .kv = kv};
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
  // Qwen3.8-Flash-Next (its hyper-connection streams) and of both Qwen3.5
  // families (the residual) applies one, no other family's does.
  std::optional<ControlVectorTable> steering;
  if (options.controlVector) {
    const auto architecture = std::visit(
        [&](const auto &layout) -> std::pair<std::string_view, std::pair<uint32_t, uint32_t>> {
          using Layout = std::decay_t<decltype(layout)>;
          if constexpr (std::is_same_v<Layout, Qwen4ExpLayout> || std::is_same_v<Layout, Qwen3_8Layout> ||
                        std::is_same_v<Layout, Qwen3_6MoeLayout>)
            return {Layout::ggufArchitecture, std::pair{layout.layers, layout.hiddenSize}};
          throw std::invalid_argument("--control-vector steers " + std::string(Qwen4ExpLayout::family) + ", " +
                                      std::string(Qwen3_8Layout::family) + " and " +
                                      std::string(Qwen3_6MoeLayout::family) + " only, not " +
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
  // The weights that stream from their files read on the engine's readers.
  const ExpertStoreOptions experts{.memoryBytes = options.expertMemoryBytes, .readers = options.readers};
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        const std::filesystem::path directory = root / "target";
        switch (result.descriptor.targetSource) {
        case TargetSource::Safetensors: {
          SafetensorsTargetLoader loader(backend, images, directory, layout, experts);
          return readTarget(backend, layout, std::ref(loader));
        }
        case TargetSource::Gguf: {
          // A streamed table's store: its memory and readers here, the rows
          // a step stages and prefetches from the family.
          SparseRowStoreOptions table = streamedTableRows(layout);
          table.cacheBytes = options.streamedTableBytes;
          table.readers = options.readers;
          GgufTargetLoader loader(backend, images, findTargetGgufs(directory), layout.ggufArchitecture, layout, table,
                                  experts);
          return readTarget(backend, layout, std::ref(loader));
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
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

} // namespace splash::model
