#include "ModelFactory.hpp"
#include "model/AffineTarget.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"

#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <span>
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

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_8Layout &layout,
                         const QwenTargetFiles<Qwen3_8Layout> &files, GgufMtpLoader *) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const QwenTargetFiles<Qwen3_6MoeLayout> &files, GgufMtpLoader *) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen4ExpLayout &layout,
                         const QwenTargetFiles<Qwen4ExpLayout> &files, GgufMtpLoader *mtp) {
  return loadQwen4ExpWeights(backend, layout, files, mtp);
}

ModelPackage loadPackage(metal::MetalBackend &backend,
                         const std::filesystem::path &root,
                         ModelDescriptor descriptor, PreparationCheck admitConversion = {}) {
  ModelPackage result;
  result.descriptor = std::move(descriptor);
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  const PreparationCheck check = [&backend] { backend.checkOperation(); };
  // Every prepared file of the model, the vision tower's, the draft's and the
  // target's, is planned before the first is written, so one disk check
  // budgets them all.
  const auto vision = planVisionLoader(backend, root, result.descriptor, admitConversion);
  std::vector<PreparedWeight> prepared;
  if (vision) prepared.push_back(vision->weight());
  std::optional<DraftCheckpointLoader> draft;
  if (result.descriptor.draftSource == DraftSource::Checkpoint) {
    draft.emplace(backend, root / "draft", result.descriptor.draft, admitConversion);
    prepared.insert(prepared.end(), draft->weights().begin(), draft->weights().end());
  }
  // Qwen3.8-Flash-Next's MTP head, from mtp/ beside target/.
  std::unique_ptr<GgufMtpLoader> mtp;
  if (const auto *qwen4 = std::get_if<Qwen4ExpLayout>(&result.descriptor.target); qwen4 && qwen4->mtpLayers) {
    const auto files = findTargetGgufs(root / "mtp");
    if (files.size() != 1) throw std::invalid_argument("mtp/ must hold one GGUF");
    mtp = std::make_unique<GgufMtpLoader>(backend, files.front(), ggufTargetGeometry(*qwen4), admitConversion);
  }
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        using Layout = std::remove_cvref_t<decltype(layout)>;
        const std::filesystem::path directory = root / "target";
        // Every missing file is written before the first is mapped: conversion
        // needs normal memory pressure, which the residency of the files
        // mapped before it would take away on a Mac with little memory.
        const auto read = [&](const QwenTargetFiles<Layout> &files,
                              std::span<const PreparedWeight> target, const auto &prepareTarget) {
          prepared.insert(prepared.end(), target.begin(), target.end());
          if (mtp) prepared.insert(prepared.end(), mtp->weights().begin(), mtp->weights().end());
          if (!prepared.empty()) PreparedWeights().requireSpace(prepared, check);
          if (vision) static_cast<void>(vision->prepare());
          if (draft) draft->prepare();
          prepareTarget();
          if (mtp) mtp->prepare();
          return readTarget(backend, layout, files, mtp.get());
        };
        switch (result.descriptor.targetSource) {
        case TargetSource::Packed:
          return read(PackedTargetFiles<Layout>{backend, directory, layout}, {}, [] {});
        case TargetSource::Mlx: {
          if constexpr (std::is_same_v<Layout, Qwen4ExpLayout>) {
            throw std::invalid_argument("Qwen3.8-Flash-Next loads from a GGUF only");
          } else {
            AffineTargetLoader loader(backend, directory, layout, admitConversion);
            return read(loader, loader.weights(), [&] { loader.prepare(); });
          }
        }
        case TargetSource::Gguf: {
          GgufTargetLoader loader(backend, findTargetGgufs(directory), ggufTargetGeometry(layout),
                                  admitConversion);
          return read(loader, loader.weights(), [&] { loader.prepare(); });
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
  if (result.descriptor.hasDraft())
    result.draft = loadDFlashDraftWeights(
        backend,
        draft ? DraftFiles(std::ref(*draft))
              : DraftFiles(PackedDraftFiles{backend, root / "draft", result.descriptor.draft}),
        result.descriptor.draft);
  else
    result.draft.layout = result.descriptor.draft;
  result.vision = loadVisionWeights(backend, root, result.descriptor, vision.get());

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
                              const ModelDescriptor &descriptor, PreparationCheck admitConversion) {
  return loadPackage(backend, root, descriptor, std::move(admitConversion));
}

std::unique_ptr<VisionLoader> planVisionLoader(metal::MetalBackend &backend, const std::filesystem::path &root,
                                               const ModelDescriptor &descriptor,
                                               PreparationCheck admitConversion) {
  if (descriptor.visionSource != VisionSource::Mlx && descriptor.visionSource != VisionSource::Gguf)
    return nullptr;
  return std::make_unique<VisionLoader>(root / "vision", descriptor.visionSource, descriptor.vision,
                                        [&backend] { backend.checkOperation(); },
                                        std::move(admitConversion));
}

QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, const std::filesystem::path &root,
                                    const ModelDescriptor &descriptor, const VisionLoader *loader) {
  if (loader) return loadQwenVisionWeights(backend, *loader);
  if (descriptor.visionSource == VisionSource::Packed)
    return loadQwenVisionWeights(backend, root / "vision", descriptor.vision);
  return {};
}

uint64_t preparedModelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  uint64_t bytes = 0;
  if (descriptor.targetSource == TargetSource::Gguf) {
    std::deque<WeightSource> sources;
    std::vector<WeightSource *> files;
    for (const std::filesystem::path &path : findTargetGgufs(root / "target"))
      files.push_back(&sources.emplace_back(path));
    const GgufFile file(files);
    for (const gguf::Image &image :
         std::visit([&](const auto &layout) { return gguf::planImages(file, ggufTargetGeometry(layout)); },
                    descriptor.target))
      bytes += image.bytes;
  } else if (descriptor.targetSource == TargetSource::Mlx) {
    bytes = std::visit([](const auto &layout) -> uint64_t {
      if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layout)>, Qwen4ExpLayout>)
        throw std::invalid_argument("Qwen3.8-Flash-Next loads from a GGUF only");
      else
        return preparedAffineBytes(layout);
    }, descriptor.target);
  }
  if (descriptor.draftSource == DraftSource::Checkpoint) bytes += preparedDraftBytes(descriptor.draft);
  if (descriptor.visionSource == VisionSource::Mlx || descriptor.visionSource == VisionSource::Gguf)
    bytes += preparedVisionBytes(descriptor.vision);
  for (std::string_view directory : {"target", "draft", "vision"}) {
    if (directory == "vision" && descriptor.visionSource != VisionSource::Packed) continue;
    if (directory == "draft" && descriptor.draftSource != DraftSource::Packed) continue;
    if (directory == "target" && descriptor.targetSource != TargetSource::Packed) continue;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root / directory)) {
      if (!entry.is_regular_file()) continue;
      const uint64_t size = entry.file_size();
      if (size > std::numeric_limits<uint64_t>::max() - bytes) throw std::overflow_error("model weight size overflows");
      bytes += size;
    }
  }
  if (!bytes) throw std::invalid_argument("model package contains no regular files");
  return bytes;
}

} // namespace splash::model
