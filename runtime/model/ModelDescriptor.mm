#include "ModelDescriptor.hpp"
#include "QwenVision.hpp"
#include "WeightStore.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <cmath>
#include <span>
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
    {{"draft_proposal_tokens", ExecutionLimits::draftProposalTokens},
     {"draft_query_rows", ExecutionLimits::draftQueryRows},
     {"draft_sliding_window", ExecutionLimits::draftContextTokens},
     {"maximum_batch_width", ExecutionLimits::maximumBatchWidth},
     {"prefill_token_budget", ExecutionLimits::prefillTokenBudget},
     {"target_kv_block_tokens", kv::kPageTokens},
     {"target_verify_rows", ExecutionLimits::targetVerifyRows}});

// The JSON object of the file at path; with sha256, the SHA-256 of its bytes
// too.
NSDictionary *readObject(const std::filesystem::path &path,
                         std::string_view label, std::string *sha256 = nullptr) {
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
  if (sha256)
    *sha256 = weightDigest(std::span<const uint8_t>(static_cast<const uint8_t *>(data.bytes), data.length));
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
  if (CFNumberIsFloatType((__bridge CFNumberRef)number)) {
    // A config saved from Python writes a float that holds a whole number,
    // rope_theta=1e7 as 10000000.0, and the installer compares it equal to
    // that integer. A double holds every integer exactly up to 2^53 - 1.
    const double real = number.doubleValue;
    if (std::isfinite(real) && real >= 1 && real <= 9007199254740991.0 &&
        std::floor(real) == real)
      return static_cast<uint64_t>(real);
  } else if (number.longLongValue > 0 &&
             static_cast<uint64_t>(number.longLongValue) ==
                 number.unsignedLongLongValue) {
    return number.unsignedLongLongValue;
  }
  throw std::invalid_argument(std::string(label) +
                              " must be a positive unsigned integer");
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

void validateCaptureLayers(NSDictionary *draft, const Qwen3_6MoeLayout &layout) {
  NSArray *layers =
      requireArray(draft, @"target_capture_layers", "target capture layers");
  requireEqual(layers.count, layout.hiddenCaptureLayers.size(),
               "target capture layer count");
  for (uint32_t index = 0; index < layers.count; ++index) {
    id value = layers[index];
    if (![value isKindOfClass:[NSNumber class]])
      throw std::invalid_argument("target capture layer must be an integer");
    requireEqual(static_cast<NSNumber *>(value).unsignedLongLongValue,
                 layout.hiddenCaptureLayers[index],
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
  validateCaptureLayers(draft, targetLayout);
  validateTokenizer(root, descriptor, "qwen3_5_moe_text");
}

void requireNumbers(NSDictionary *object, std::initializer_list<GeometryField> fields) {
  for (const auto &field : fields)
    requireEqual(requireUnsigned(object, [NSString stringWithUTF8String:field.name], field.name), field.value, field.name);
}

ModelDescriptor inspectSourceModel(const std::filesystem::path &root) {
  std::string sourceIdentity;
  NSDictionary *record = readObject(root / "model.json", "resolved model", &sourceIdentity);
  requireEqual(requireUnsigned(record, @"version", "model record version"), 1, "model record version");
  NSDictionary *config = readObject(root / "config.json", "upstream model config");
  NSDictionary *text = requireObject(config, @"text_config", "text config");
  const auto type = requireString(text, @"model_type", "text model type");
  const auto name = requireString(record, @"model", "model name");
  ModelDescriptor result;
  if (type == "qwen3_5_moe_text") result = qwen36Descriptor(name);
  else if (type == "qwen3_5_text") result = qwen38Descriptor(name);
  else throw std::invalid_argument("unsupported model architecture: " + type);
  std::visit([&](const auto &layout) {
    requireNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
        {"vocab_size", layout.vocabularySize}, {"max_position_embeddings", layout.maximumContextTokens},
        {"num_attention_heads", layout.attentionQueryHeads}, {"num_key_value_heads", layout.attentionKvHeads},
        {"head_dim", layout.attentionHeadDimension}});
  }, result.target);
  const auto target = requireString(record, @"target_format", "target format");
  if (target == "mlx-affine") result.targetSource = TargetSource::Mlx;
  else if (target == "gguf") result.targetSource = TargetSource::Gguf;
  else throw std::invalid_argument("unsupported target source format: " + target);

  NSDictionary *draft = readObject(root / "draft" / "config.json", "draft config");
  NSArray *architectures = requireArray(draft, @"architectures", "draft architectures");
  if (architectures.count != 1 || ![architectures[0] isEqual:@"DFlash2DraftModel"])
    throw std::invalid_argument("draft is not a DFlash2 model");
  const auto &d = result.draft;
  requireNumbers(draft, {{"num_hidden_layers", d.layers}, {"hidden_size", d.hiddenSize},
      {"vocab_size", d.vocabularySize}, {"intermediate_size", d.intermediateSize},
      {"num_attention_heads", d.attentionSize / d.attentionHeadDimension},
      {"num_key_value_heads", d.kvHeads}, {"head_dim", d.attentionHeadDimension},
      {"sliding_window", ExecutionLimits::draftContextTokens}});
  if (![draft[@"is_causal"] isEqual:@NO] ||
      ![draft[@"attention_bias"] isEqual:@NO] ||
      ![draft[@"tie_word_embeddings"] isEqual:@NO] ||
      ![draft[@"rms_norm_eps"] isEqual:@(1e-6)] ||
      ![draft[@"hidden_act"] isEqual:@"silu"])
    throw std::invalid_argument("unsupported draft attention or normalization configuration");
  NSDictionary *rope = requireObject(draft, @"rope_parameters", "draft rotary configuration");
  requireEqual(requireString(rope, @"rope_type", "draft rope type"), "default", "draft rope type");
  requireEqual(requireUnsigned(rope, @"rope_theta", "draft rotary theta"), 10000000, "draft rotary theta");
  NSDictionary *flash = requireObject(draft, @"dflash_config", "draft configuration");
  requireNumbers(flash, {{"block_size", ExecutionLimits::draftQueryRows}, {"conv_group_size", 16},
      {"conv_kernel_size", 2}, {"selector_rank", d.selectorRank}, {"selector_top_k", 16}});
  NSArray *capture = requireArray(flash, @"target_layer_ids", "draft target layers");
  std::visit([&](const auto &layout) {
    requireEqual(requireUnsigned(flash, @"mask_token_id", "draft mask token"), layout.maskToken, "draft mask token");
    requireEqual(capture.count, layout.hiddenCaptureLayers.size(), "draft target layer count");
    for (size_t i = 0; i < layout.hiddenCaptureLayers.size(); ++i) {
      id value = capture[i];
      if (![value isKindOfClass:[NSNumber class]] || [value unsignedLongLongValue] != layout.hiddenCaptureLayers[i])
        throw std::invalid_argument("draft target capture layers do not match this model");
    }
  }, result.target);

  const auto vision = requireString(record, @"vision_format", "vision format");
  if (vision == "none") result.visionSource = VisionSource::None;
  else {
    if (vision == "safetensors") result.visionSource = VisionSource::Mlx;
    else if (vision == "gguf") result.visionSource = VisionSource::Gguf;
    else throw std::invalid_argument("unsupported vision source format: " + vision);
    NSDictionary *v = requireObject(config, @"vision_config", "vision config");
    const auto &l = result.vision;
    requireNumbers(v, {{"depth", l.depth}, {"hidden_size", l.hiddenSize}, {"num_heads", l.heads},
        {"intermediate_size", l.intermediateSize}, {"out_hidden_size", l.outputHiddenSize},
        {"patch_size", l.patchSize}, {"spatial_merge_size", l.spatialMerge},
        {"temporal_patch_size", 2}, {"in_channels", 3}, {"num_position_embeddings", l.positionGridSide * l.positionGridSide}});
    requireEqual(requireString(v, @"hidden_act", "vision activation"), "gelu_pytorch_tanh", "vision activation");
    NSArray *deepstack = requireArray(v, @"deepstack_visual_indexes", "vision deepstack layers");
    if (deepstack.count) throw std::invalid_argument("vision deepstack layers are unsupported");
  }
  result.sourceIdentity = std::move(sourceIdentity);
  if (!result.valid()) throw std::invalid_argument("incompatible target and draft model");
  return result;
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
        result.capabilities = {layout.vocabularySize,
                               layout.maximumContextTokens};
        result.targetKvLayout = layout.kvLayout();
        result.stateLayout = {layout.gdnStateLayout(), draft.stateLayout()};
      },
      target);
  return result;
}

bool ModelDescriptor::valid() const noexcept {
  if (name.empty() || !capabilities.vocabularySize ||
      !capabilities.maximumContextTokens ||
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
    if (std::filesystem::exists(root / "model.json")) return inspectSourceModel(root);
    std::string sourceIdentity;
    NSDictionary *manifest = readObject(root / "manifest.json", "model manifest", &sourceIdentity);
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
    descriptor.sourceIdentity = std::move(sourceIdentity);
    if (!descriptor.valid())
      throw std::logic_error("built-in model descriptor is inconsistent");
    return descriptor;
  }
}

} // namespace splash::model
