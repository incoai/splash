#include "ModelFactory.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/VisionLoader.hpp"

#include <functional>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::model {

void requireCompatibleModel(const LoadedModel &model) {
  if (!model.descriptor.valid() ||
      model.descriptor.draft != model.draft.layout || model.descriptor.draft.dspark() != model.dspark.has_value() ||
      !std::visit(
          [&](const auto &target) {
            return model.descriptor.target == TargetLayout{target.layout} &&
                   (!model.draft.layout.present() ||
                    target.layout.vocabularySize == model.draft.layout.vocabularySize);
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

TargetWeights readTarget(metal::MetalBackend &backend, const FlintLayout &layout, const QwenTargetFiles &files) {
  return loadFlintWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const DeepSeek41Layout &layout, const QwenTargetFiles &files) {
  return loadDeepSeek41Weights(backend, layout, files);
}

} // namespace

SparseRowStoreOptions streamedTableRows(const Qwen3_8Layout &) { return {}; }

SparseRowStoreOptions streamedTableRows(const Qwen3_6MoeLayout &) { return {}; }

SparseRowStoreOptions streamedTableRows(const FlintLayout &) { return {}; }

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
  result.images = std::make_shared<WeightImages>(backend, result.descriptor.sourceIdentity);
  WeightImages &images = *result.images;
  // Every source's metadata is checked before the first image is written:
  // the vision tower's and the draft's here, the target's by its loader.
  const auto vision = planVisionLoader(root, result.descriptor);
  std::optional<DraftCheckpointLoader> draft;
  if (result.descriptor.draft.dflash())
    draft.emplace(backend, images, root / "draft", result.descriptor.draft);
  // The weights that stream from their files read on the engine's readers.
  const ExpertStoreOptions experts{
      .memoryBytes = options.expertMemoryBytes, .floorBytes = options.expertFloorBytes, .readers = options.readers};
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
  if (draft)
    result.draft = loadDFlashDraftWeights(backend, *draft, result.descriptor.draft);
  if (result.descriptor.draft.dspark()) {
    // DSpark, every expert loading with its layer, its images the model's
    // draft/ components.
    const auto *target = std::get_if<DeepSeek41Layout>(&result.descriptor.target);
    const std::optional<std::filesystem::path> path = findDSparkGguf(root / "draft");
    if (!target || !path) throw std::invalid_argument("a DSpark draft serves DeepSeek-V4.1 from its GGUF in draft/");
    DSparkLayout layout;
    {
      const OpenGguf file(std::vector<std::filesystem::path>{*path});
      layout = readDSparkLayout(file.file(), *target);
    }
    if (dsparkDraftLayout(layout) != result.descriptor.draft)
      throw std::invalid_argument("the DSpark GGUF in draft/ is not the draft the model was described with");
    const uint64_t before = backend.memoryStats().allocatedBytes;
    result.dspark.emplace(loadDSparkWeights(backend, images, *path, layout));
    result.draft.layout = result.descriptor.draft;
    result.draft.files = result.dspark->files;
    result.draft.actualAllocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  }
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
