#include "ModelDescriptor.hpp"
#include "QwenVision.hpp"

#import <Foundation/Foundation.h>

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace splash::model {
namespace {

struct GeometryField final {
  const char *name;
  uint64_t value;
};

constexpr auto kExecutionGeometry = std::to_array<GeometryField>(
    {{"allocation_extent_target_bytes", kv::kAllocationExtentTargetBytes},
     {"draft_proposal_tokens", ExecutionLimits::draftProposalTokens},
     {"draft_query_rows", ExecutionLimits::draftQueryRows},
     {"draft_sliding_window", ExecutionLimits::draftContextTokens},
     {"maximum_batch_width", ExecutionLimits::maximumBatchWidth},
     {"prefill_token_budget", ExecutionLimits::prefillTokenBudget},
     {"target_kv_block_tokens", kv::kPageTokens},
     {"target_verify_rows", ExecutionLimits::targetVerifyRows}});

NSDictionary *readObject(const std::filesystem::path &path,
                         std::string_view label,
                         std::array<uint8_t, 32> *sha256 = nullptr) {
  NSString *nativePath = [NSString stringWithUTF8String:path.c_str()];
  if (!nativePath)
    throw std::invalid_argument(std::string(label) +
                                " path is not representable");
  NSError *readError = nil;
  NSData *data = [NSData dataWithContentsOfFile:nativePath
                                        options:0
                                          error:&readError];
  if (!data) {
    const char *description = readError.localizedDescription.UTF8String;
    throw std::invalid_argument("could not read " + std::string(label) +
                                ": " +
                                (description ? description
                                             : "unknown read error"));
  }
  if (sha256) {
    if (data.length > std::numeric_limits<CC_LONG>::max())
      throw std::overflow_error(std::string(label) + " is too large to hash");
    if (!CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), sha256->data()))
      throw std::runtime_error(std::string(label) + " SHA-256 failed");
  }
  NSError *parseError = nil;
  id value = [NSJSONSerialization JSONObjectWithData:data
                                             options:0
                                               error:&parseError];
  if (![value isKindOfClass:[NSDictionary class]]) {
    const char *description = parseError.localizedDescription.UTF8String;
    throw std::invalid_argument("could not parse " + std::string(label) +
                                ": " +
                                (description ? description
                                             : "expected a JSON object"));
  }
  return static_cast<NSDictionary *>(value);
}

NSDictionary *requireObject(NSDictionary *object, NSString *key,
                            std::string_view label) {
  id value = object[key];
  if (![value isKindOfClass:[NSDictionary class]])
    throw std::invalid_argument(std::string(label) + " must be an object");
  return static_cast<NSDictionary *>(value);
}

NSArray *requireArray(NSDictionary *object, NSString *key,
                      std::string_view label) {
  id value = object[key];
  if (![value isKindOfClass:[NSArray class]])
    throw std::invalid_argument(std::string(label) + " must be an array");
  return static_cast<NSArray *>(value);
}

std::string requireString(NSDictionary *object, NSString *key,
                          std::string_view label) {
  id value = object[key];
  if (![value isKindOfClass:[NSString class]])
    throw std::invalid_argument(std::string(label) + " must be a string");
  const char *text = static_cast<NSString *>(value).UTF8String;
  if (!text || !*text)
    throw std::invalid_argument(std::string(label) + " must not be empty");
  return text;
}

uint64_t requireUnsigned(NSDictionary *object, NSString *key,
                         std::string_view label) {
  id value = object[key];
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) {
    throw std::invalid_argument(std::string(label) +
                                " must be an unsigned integer");
  }
  NSNumber *number = static_cast<NSNumber *>(value);
  if (CFNumberIsFloatType((__bridge CFNumberRef)number) ||
      number.longLongValue <= 0 ||
      static_cast<uint64_t>(number.longLongValue) !=
          number.unsignedLongLongValue) {
    throw std::invalid_argument(std::string(label) +
                                " must be a positive unsigned integer");
  }
  return number.unsignedLongLongValue;
}

void requireEqual(uint64_t actual, uint64_t expected,
                  std::string_view label) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(label) + " mismatch: package " +
                                std::to_string(actual) + ", runtime " +
                                std::to_string(expected));
  }
}

void requireEqual(std::string_view actual, std::string_view expected,
                  std::string_view label) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(label) + " mismatch: package " +
                                std::string(actual) + ", runtime " +
                                std::string(expected));
  }
}

void validateExecutionGeometry(NSDictionary *manifest) {
  NSDictionary *geometry = requireObject(
      manifest, @"execution_geometry", "model execution geometry");
  // Packages may carry descriptive metadata, but every execution-semantic
  // field understood by this runtime must be present and match exactly.
  for (const GeometryField &field : kExecutionGeometry) {
    NSString *key = [NSString stringWithUTF8String:field.name];
    requireEqual(requireUnsigned(geometry, key, field.name), field.value,
                 field.name);
  }
}

void validateCommonFormat(NSDictionary *format, std::string_view targetMagic) {
  requireEqual(requireUnsigned(format, @"section_alignment_bytes",
                               "section_alignment_bytes"),
               kWeightFileAlignment, "section_alignment_bytes");
  requireEqual(requireString(format, @"target_layer_magic",
                             "target_layer_magic"),
               targetMagic, "target_layer_magic");
  requireEqual(requireString(format, @"draft_layer_magic",
                             "draft_layer_magic"),
               kDFlashLayerMagic, "draft_layer_magic");
  requireEqual(requireString(format, @"vision_magic", "vision_magic"),
               kVisionMagic, "vision_magic");
}

DFlashDraftLayout qwen36DraftLayout() {
  DFlashDraftLayout layout;
  layout.layers = 6;
  layout.hiddenSize = 2048;
  layout.dynamicSize = 512;
  layout.intermediateSize = 6144;
  layout.targetHiddenSize = 16384;
  return layout;
}

ModelDescriptor qwen38Descriptor(std::string name) {
  return makeModelDescriptor(std::move(name), Qwen3_8Layout{},
                             DFlashDraftLayout{}, ops::VisionLayout{});
}

ModelDescriptor qwen36Descriptor(std::string name) {
  constexpr Qwen3_6MoeLayout target;
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.hiddenSize;
  return makeModelDescriptor(std::move(name), target, qwen36DraftLayout(),
                             vision);
}

void validateTokenizer(const std::filesystem::path &root,
                       const ModelDescriptor &descriptor,
                       std::string_view expectedTextModelType) {
  NSDictionary *config = readObject(root / "tokenizer" / "config.json",
                                    "tokenizer model config");
  NSDictionary *text =
      requireObject(config, @"text_config", "text model config");
  requireEqual(requireString(text, @"model_type", "text model type"),
               expectedTextModelType, "text model type");
  requireEqual(requireUnsigned(text, @"hidden_size", "hidden_size"),
               std::visit([](const auto &layout) { return layout.hiddenSize; },
                          descriptor.target),
               "hidden_size");
  requireEqual(requireUnsigned(text, @"vocab_size", "vocab_size"),
               descriptor.capabilities.vocabularySize, "vocab_size");
  requireEqual(requireUnsigned(text, @"max_position_embeddings",
                               "max_position_embeddings"),
               descriptor.capabilities.maximumContextTokens,
               "max_position_embeddings");
}

void validateQwen38(NSDictionary *manifest,
                    const std::filesystem::path &root,
                    const ModelDescriptor &descriptor) {
  requireEqual(requireUnsigned(manifest, @"schema_version", "schema_version"),
               3, "schema_version");
  NSDictionary *format =
      requireObject(manifest, @"format", "model weight format");
  requireEqual(requireUnsigned(format, @"q4_bits", "q4_bits"), 4,
               "q4_bits");
  requireEqual(requireUnsigned(format, @"q4_group_size", "q4_group_size"),
               kQ4GroupElements, "q4_group_size");
  requireEqual(requireUnsigned(format, @"q4_storage_n", "q4_storage_n"),
               kQ4StorageN, "q4_storage_n");
  validateCommonFormat(format, Qwen3_8Layout::layerMagic);
  validateTokenizer(root, descriptor, "qwen3_5_text");
}

void validateLayerTypes(NSDictionary *target,
                        const Qwen3_6MoeLayout &layout) {
  NSArray *types = requireArray(target, @"layer_types", "target layer_types");
  requireEqual(types.count, layout.layers, "target layer_types count");
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    id value = types[layer];
    if (![value isKindOfClass:[NSString class]])
      throw std::invalid_argument("target layer type must be a string");
    const std::string expected =
        layout.isFullAttentionLayer(layer) ? "attention" : "gdn";
    const char *actual = static_cast<NSString *>(value).UTF8String;
    requireEqual(actual ? actual : "", expected,
                 "target layer " + std::to_string(layer));
  }
}

void validateCaptureLayers(NSDictionary *draft) {
  NSArray *layers =
      requireArray(draft, @"target_capture_layers", "target capture layers");
  requireEqual(layers.count, Qwen3_6MoeLayout::hiddenCaptureLayers.size(),
               "target capture layer count");
  for (uint32_t index = 0; index < layers.count; ++index) {
    id value = layers[index];
    if (![value isKindOfClass:[NSNumber class]])
      throw std::invalid_argument("target capture layer must be an integer");
    requireEqual(static_cast<NSNumber *>(value).unsignedLongLongValue,
                 Qwen3_6MoeLayout::hiddenCaptureLayers[index],
                 "target capture layer " + std::to_string(index));
  }
}

void validateQwen36(NSDictionary *manifest,
                    const std::filesystem::path &root,
                    const ModelDescriptor &descriptor) {
  requireEqual(requireUnsigned(manifest, @"schema_version", "schema_version"),
               4, "schema_version");
  NSDictionary *format =
      requireObject(manifest, @"format", "model weight format");
  requireEqual(requireUnsigned(format, @"q4_bits", "q4_bits"), 4,
               "q4_bits");
  requireEqual(requireUnsigned(format, @"q8_bits", "q8_bits"), 8,
               "q8_bits");
  requireEqual(requireUnsigned(format, @"quant_group_size",
                               "quant_group_size"),
               kQ4GroupElements, "quant_group_size");
  requireEqual(requireUnsigned(format, @"storage_n", "storage_n"),
               kQ4StorageN, "storage_n");
  validateCommonFormat(format, Qwen3_6MoeLayout::layerMagic);

  const auto &targetLayout = std::get<Qwen3_6MoeLayout>(descriptor.target);
  NSDictionary *target =
      requireObject(manifest, @"target", "target declaration");
  requireEqual(requireString(target, @"architecture", "target architecture"),
               "qwen3_5_moe", "target architecture");
  for (const GeometryField &field : std::to_array<GeometryField>(
           {{"layers", targetLayout.layers},
            {"hidden_size", targetLayout.hiddenSize},
            {"vocabulary_size", targetLayout.vocabularySize},
            {"gdn_actual_width", targetLayout.actualGdnWidth()},
            {"gdn_packed_width", targetLayout.packedGdnWidth},
            {"attention_packed_width", targetLayout.packedFullWidth},
            {"experts", targetLayout.experts},
            {"experts_per_token", targetLayout.expertsPerToken},
            {"moe_intermediate_size", targetLayout.expertIntermediateSize},
            {"shared_expert_intermediate_size",
             targetLayout.expertIntermediateSize}})) {
    requireEqual(requireUnsigned(target,
                                 [NSString stringWithUTF8String:field.name],
                                 field.name),
                 field.value, field.name);
  }
  validateLayerTypes(target, targetLayout);

  const DFlashDraftLayout &draftLayout = descriptor.draft;
  NSDictionary *draft =
      requireObject(manifest, @"draft", "draft declaration");
  requireEqual(requireString(draft, @"architecture", "draft architecture"),
               "DFlash2DraftModel", "draft architecture");
  for (const GeometryField &field : std::to_array<GeometryField>(
           {{"layers", draftLayout.layers},
            {"hidden_size", draftLayout.hiddenSize},
            {"intermediate_size", draftLayout.intermediateSize},
            {"sliding_window", ExecutionLimits::draftContextTokens},
            {"block_size", ExecutionLimits::draftQueryRows},
            {"dynamic_conv_group_size", 16},
            {"dynamic_conv_kernel_size", 2},
            {"selector_rank", draftLayout.selectorRank},
            {"selector_top_k", 16}})) {
    requireEqual(requireUnsigned(draft,
                                 [NSString stringWithUTF8String:field.name],
                                 field.name),
                 field.value, field.name);
  }
  validateCaptureLayers(draft);
  validateTokenizer(root, descriptor, "qwen3_5_moe_text");
}

} // namespace

ModelDescriptor makeModelDescriptor(std::string name, TargetLayout target,
                                    DFlashDraftLayout draft,
                                    ops::VisionLayout vision) {
  ModelDescriptor result;
  result.name = std::move(name);
  result.target = target;
  result.draft = draft;
  result.vision = vision;
  std::visit(
      [&](const auto &layout) {
        result.capabilities = {
            layout.vocabularySize,
            layout.maximumContextTokens,
            ExecutionLimits::maximumBatchWidth,
            ExecutionLimits::prefillTokenBudget,
            ExecutionLimits::draftQueryRows,
            ExecutionLimits::draftProposalTokens,
            ExecutionLimits::targetVerifyRows,
            ExecutionLimits::draftContextTokens,
        };
        result.targetKvLayout = layout.kvLayout();
        result.stateLayout = {layout.gdnStateLayout(), draft.stateLayout()};
      },
      target);
  return result;
}

bool ModelDescriptor::valid() const noexcept {
  if (name.empty() || !capabilities.vocabularySize ||
      !capabilities.maximumContextTokens ||
      capabilities.maximumBatchWidth != ExecutionLimits::maximumBatchWidth ||
      capabilities.prefillTokenBudget != ExecutionLimits::prefillTokenBudget ||
      capabilities.draftQueryRows != ExecutionLimits::draftQueryRows ||
      capabilities.draftProposalTokens != ExecutionLimits::draftProposalTokens ||
      capabilities.targetVerifyRows != ExecutionLimits::targetVerifyRows ||
      capabilities.draftContextTokens != ExecutionLimits::draftContextTokens ||
      !targetKvLayout.valid() || !stateLayout.valid() ||
      stateLayout.draft != draft.stateLayout() ||
      vision.outputHiddenSize != draft.hiddenSize) {
    return false;
  }
  return std::visit(
      [&](const auto &layout) {
        return layout.vocabularySize == capabilities.vocabularySize &&
               layout.maximumContextTokens ==
                   capabilities.maximumContextTokens &&
               layout.hiddenSize == draft.hiddenSize &&
               layout.capturedHiddenSize() == draft.targetHiddenSize &&
               layout.kvLayout() == targetKvLayout &&
               layout.gdnStateLayout() == stateLayout.target;
      },
      target);
}

ModelDescriptor inspectModelPackage(const std::filesystem::path &root) {
  @autoreleasepool {
    std::array<uint8_t, 32> packageManifestSha256{};
    NSDictionary *manifest =
        readObject(root / "manifest.json", "model manifest", &packageManifestSha256);
    validateExecutionGeometry(manifest);
    const std::string model = requireString(manifest, @"model", "model name");
    const std::string format = requireString(
        requireObject(manifest, @"format", "model weight format"),
        @"name", "weight format");
    ModelDescriptor descriptor;
    if (format == "splash-packed-q4") {
      descriptor = qwen38Descriptor(model);
      validateQwen38(manifest, root, descriptor);
    } else if (format == "splash-packed-q4-moe") {
      descriptor = qwen36Descriptor(model);
      validateQwen36(manifest, root, descriptor);
    } else {
      throw std::invalid_argument("unsupported weight format: " + format);
    }
    descriptor.packageManifestSha256 = packageManifestSha256;
    if (!descriptor.valid())
      throw std::logic_error("built-in model descriptor is inconsistent");
    return descriptor;
  }
}

} // namespace splash::model
