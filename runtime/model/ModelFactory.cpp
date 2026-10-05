#include "ModelFactory.hpp"
#include "model/AffineTarget.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/VisionLoader.hpp"

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
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
  if (descriptor.visionSource != VisionSource::Mlx && descriptor.visionSource != VisionSource::Gguf)
    return nullptr;
  return std::make_unique<VisionLoader>(root / "vision", descriptor.visionSource, descriptor.vision);
}

QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                    const std::filesystem::path &root, const ModelDescriptor &descriptor,
                                    const VisionLoader *loader) {
  if (loader) return loadQwenVisionWeights(backend, images, *loader);
  if (descriptor.visionSource == VisionSource::Package)
    return loadQwenVisionWeights(backend, images, root / "vision", descriptor.vision);
  return {};
}

namespace {

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_8Layout &layout,
                         const QwenTargetFiles<Qwen3_8Layout> &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const QwenTargetFiles<Qwen3_6MoeLayout> &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

template <class Image> uint64_t imageBytes(const std::vector<Image> &images) {
  uint64_t total = 0;
  for (const Image &image : images) total += image.bytes;
  return total;
}

} // namespace

LoadedModel loadModel(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor) {
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
  if (result.descriptor.draftFromCheckpoint())
    draft.emplace(images, root / "draft", result.descriptor.draft);
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        using Layout = std::remove_cvref_t<decltype(layout)>;
        const std::filesystem::path directory = root / "target";
        switch (result.descriptor.targetSource) {
        case TargetSource::Package:
          return readTarget(backend, layout, PackageTargetFiles<Layout>{images, directory, layout});
        case TargetSource::Mlx: {
          AffineTargetLoader loader(images, directory, layout);
          return readTarget(backend, layout, std::ref(loader));
        }
        case TargetSource::Gguf: {
          GgufTargetLoader loader(backend, images, findTargetGguf(directory), layout);
          return readTarget(backend, layout, std::ref(loader));
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
  result.draft = loadDFlashDraftWeights(
      backend,
      draft ? DraftFiles(std::ref(*draft))
            : DraftFiles(PackageDraftFiles{images, root / "draft", result.descriptor.draft}),
      result.descriptor.draft);
  result.vision = loadVisionWeights(backend, images, root, result.descriptor, vision.get());

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

uint64_t modelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  uint64_t bytes = 0;
  if (descriptor.targetSource == TargetSource::Gguf) {
    WeightSource source(findTargetGguf(root / "target"));
    const GgufFile file(source);
    bytes = std::visit(
        [&](const auto &layout) { return imageBytes(gguf::planImages(file, layout)); },
        descriptor.target);
  } else if (descriptor.targetSource == TargetSource::Mlx) {
    bytes = std::visit([](const auto &layout) { return imageBytes(affineTargetImages(layout)); }, descriptor.target);
  }
  if (descriptor.draftFromCheckpoint())
    bytes += imageBytes(draftCheckpointImages(descriptor.draft));
  if (descriptor.visionSource == VisionSource::Mlx || descriptor.visionSource == VisionSource::Gguf)
    bytes += visionImageBytes(descriptor.vision);
  for (std::string_view directory : {"target", "draft", "vision"}) {
    if (directory == "vision" && descriptor.visionSource != VisionSource::Package) continue;
    if (directory == "draft" && descriptor.draftFromCheckpoint()) continue;
    if (directory == "target" && descriptor.targetSource != TargetSource::Package) continue;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root / directory)) {
      if (!entry.is_regular_file()) continue;
      const uint64_t size = entry.file_size();
      if (size > std::numeric_limits<uint64_t>::max() - bytes) throw std::overflow_error("model weight size overflows");
      bytes += size;
    }
  }
  if (!bytes) throw std::invalid_argument("the model root holds no weight files");
  return bytes;
}

} // namespace splash::model
