#include "ModelFactory.hpp"

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace splash::model {

void requireCompatibleModelPackage(const ModelPackage &package) {
  if (!package.descriptor.valid() ||
      package.descriptor.draft != package.draft.layout ||
      !std::visit(
          [&](const auto &target) {
            return package.descriptor.target == TargetLayout{target.layout} &&
                   target.layout.vocabularySize ==
                       package.draft.layout.vocabularySize;
          },
          package.target)) {
    throw std::invalid_argument(
        "target and draft model interfaces are incompatible");
  }
}

namespace {

ModelPackage loadPackage(metal::MetalBackend &backend,
                         const std::filesystem::path &root,
                         ModelDescriptor descriptor) {
  ModelPackage result;
  result.descriptor = std::move(descriptor);
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layout)>,
                                     Qwen3_8Layout>)
          return loadQwen3_8Weights(backend, root / "target", layout);
        else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layout)>,
                                          Qwen3_8Q8Layout>)
          return loadQwen3_8Q8Weights(backend, root / "target", layout);
        else
          return loadQwen3_6MoeWeights(backend, root / "target", layout);
      },
      result.descriptor.target);
  result.draft = loadDFlashDraftWeights(
      backend, root / "draft", result.descriptor.draft);
  result.vision = loadQwenVisionWeights(
      backend, root / "vision", result.descriptor.vision);

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  records.insert(records.end(), result.draft.files.begin(),
                 result.draft.files.end());
  records.insert(records.end(), result.vision.files.begin(),
                 result.vision.files.end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModelPackage(result);
  return result;
}

} // namespace

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root) {
  return loadPackage(backend, root, inspectModelPackage(root));
}

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor) {
  return loadPackage(backend, root, descriptor);
}

} // namespace splash::model
