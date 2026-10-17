#include "ModelFactory.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/VisionLoader.hpp"

#include <functional>
#include <stdexcept>
#include <string_view>
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

// The rows of a family's streamed table that a step stages and prefetches
// (SparseRowStoreOptions); the Qwen3.5 families have no table to stream.
SparseRowStoreOptions tableRows(const Qwen3_8Layout &) { return {}; }
SparseRowStoreOptions tableRows(const Qwen3_6MoeLayout &) { return {}; }
SparseRowStoreOptions tableRows(const Qwen4ExpLayout &layout) { return streamedTableRows(layout); }

// The sum of `bytes` over the images of the GGUF target in root's target/,
// planned for a layout of `architecture` and `geometry`.
uint64_t ggufTargetBytes(const std::filesystem::path &root, std::string_view architecture,
                         const QwenTargetDimensions &geometry, gguf::PlanOptions options,
                         uint64_t (*bytes)(const gguf::Image &)) {
  const OpenGguf target(findTargetGgufs(root / "target"));
  uint64_t total = 0;
  for (const gguf::Image &image : gguf::planImages(target.file(), architecture, geometry, options))
    total += bytes(image);
  return total;
}

} // namespace

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
  DraftCheckpointLoader draft(backend, images, root / "draft", result.descriptor.draft);
  // The weights that stream from their files read on one set of readers.
  const auto readers = options.streamedTableBytes || options.expertMemoryBytes
                           ? std::make_shared<WeightReaders>(kWeightReaders)
                           : nullptr;
  const ExpertStoreOptions experts{.memoryBytes = options.expertMemoryBytes, .readers = readers};
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
          SparseRowStoreOptions table = tableRows(layout);
          table.cacheBytes = options.streamedTableBytes;
          table.readers = readers;
          GgufTargetLoader loader(backend, images, findTargetGgufs(directory), layout.ggufArchitecture, layout, table,
                                  experts);
          return readTarget(backend, layout, std::ref(loader));
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
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

uint64_t routedExpertBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  return std::visit(
      [&](const auto &layout) -> uint64_t {
        if (layout.ffnKind != QwenFfnKind::SparseMoe) return 0;
        if (descriptor.targetSource != TargetSource::Gguf) return safetensorsRoutedExpertBytes(root / "target", layout);
        return ggufTargetBytes(root, layout.ggufArchitecture, layout, {.streamExperts = true},
                               gguf::streamedExpertBytes);
      },
      descriptor.target);
}

uint64_t streamableTableBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  if (descriptor.targetSource != TargetSource::Gguf) return 0;
  return std::visit(
      [&](const auto &layout) {
        return ggufTargetBytes(root, layout.ggufArchitecture, layout, {.streamTable = true},
                               gguf::streamedTableBytes);
      },
      descriptor.target);
}

uint64_t modelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor,
                          const ModelLoadOptions &options) {
  const gguf::PlanOptions plan{.streamExperts = options.expertMemoryBytes != 0,
                               .streamTable = options.streamedTableBytes != 0};
  uint64_t bytes = std::visit(
      [&](const auto &layout) {
        return descriptor.targetSource == TargetSource::Gguf
                   ? ggufTargetBytes(root, layout.ggufArchitecture, layout, plan, gguf::loadedBytes)
                   : safetensorsTargetImageBytes(root / "target", layout, plan);
      },
      descriptor.target);
  // A store allocates at most its memory (ExpertStoreOptions::memoryBytes).
  if (plan.streamExperts &&
      std::visit([](const auto &layout) { return layout.ffnKind == QwenFfnKind::SparseMoe; }, descriptor.target))
    bytes += options.expertMemoryBytes;
  bytes += draftImageBytes(root / "draft", descriptor.draft);
  if (descriptor.hasVision())
    bytes += visionImageBytes(descriptor.vision);
  return bytes;
}

} // namespace splash::model
