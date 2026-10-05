#include "ModelDescriptor.hpp"
#include "WeightStore.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/ExecutionGeometry.h"

#import <Foundation/Foundation.h>

#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace splash::model {
namespace {

// A model's records and configurations are kilobytes of JSON: a file past
// this is not one and is not read, as a checkpoint's config.json is not
// (SafetensorsCheckpoint.mm).
constexpr uint64_t kMaximumJsonBytes = 1 << 20;

// The JSON object of the file at path; with sha256, the SHA-256 of its bytes
// too.
NSDictionary *readObject(const std::filesystem::path &path,
                         std::string_view label, std::string *sha256 = nullptr) {
  std::error_code sizeError;
  const uintmax_t bytes = std::filesystem::file_size(path, sizeError);
  if (!sizeError && bytes > kMaximumJsonBytes)
    throw std::invalid_argument(std::string(label) + " exceeds " +
                                std::to_string(kMaximumJsonBytes) + " bytes");
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

void requireEqual(std::string_view actual, std::string_view expected,
                  std::string_view label) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(label) + " mismatch: package " +
                                std::string(actual) + ", runtime " +
                                std::string(expected));
  }
}

// The one rule every number of a record or configuration is checked by: a
// JSON number, never a boolean, equal to the value expected. A config saved
// from Python may write a whole number as a float, rope_theta=1e7 as
// 10000000.0, which equals that integer; every value expected is exact as a
// double.
void requireNumber(id value, double expected, std::string_view label) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
    throw std::invalid_argument(std::string(label) + " must be a number");
  if ([value doubleValue] != expected)
    throw std::invalid_argument(std::string(label) + " mismatch: package " +
                                [value description].UTF8String + ", runtime " +
                                @(expected).description.UTF8String);
}

// A key and the number it must hold.
struct ExpectedNumber final {
  template <class Number>
  ExpectedNumber(const char *key, Number value)
      : key(key), value(static_cast<double>(value)) {}
  const char *key;
  double value;
};

// The numbers object holds at the keys, which errors name after `where`.
void requireNumbers(NSDictionary *object, std::string_view where,
                    std::initializer_list<ExpectedNumber> fields) {
  for (const ExpectedNumber &field : fields)
    requireNumber(object[@(field.key)], field.value,
                  std::string(where) + " " + field.key);
}

// The numbers of an array, in order.
void requireNumbers(NSArray *values, std::span<const uint32_t> expected,
                    std::string_view label) {
  if (values.count != expected.size())
    throw std::invalid_argument(std::string(label) + " count mismatch: package " +
                                std::to_string(values.count) + ", runtime " +
                                std::to_string(expected.size()));
  for (size_t index = 0; index < expected.size(); ++index)
    requireNumber(values[index], expected[index],
                  std::string(label) + " " + std::to_string(index));
}

// The JSON booleans, never numbers, object holds at the keys.
void requireBooleans(NSDictionary *object, std::string_view where,
                     std::initializer_list<std::pair<const char *, bool>> fields) {
  for (const auto &[key, expected] : fields) {
    id value = object[@(key)];
    if (![value isKindOfClass:[NSNumber class]] ||
        CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() ||
        [value boolValue] != expected)
      throw std::invalid_argument(std::string(where) + " " + key + " must be " +
                                  (expected ? "true" : "false"));
  }
}

// Each layer's type in a layer_types array, as the array names a
// full-attention layer and a GDN layer.
void requireLayerTypes(NSArray *types, const QwenTargetDimensions &target,
                       NSString *attention, NSString *gdn,
                       std::string_view label) {
  if (types.count != target.layers)
    throw std::invalid_argument(std::string(label) + " count mismatch: package " +
                                std::to_string(types.count) + ", runtime " +
                                std::to_string(target.layers));
  for (uint32_t layer = 0; layer < target.layers; ++layer) {
    NSString *expected = target.isFullAttentionLayer(layer) ? attention : gdn;
    if (![types[layer] isEqual:expected])
      throw std::invalid_argument(std::string(label) + " " +
                                  std::to_string(layer) + " must be " +
                                  expected.UTF8String);
  }
}

// A package records the execution geometry it was published with. Its draft
// was trained for blocks of draft_query_rows rows, the anchor and
// draft_proposal_tokens proposals, over draft_sliding_window context tokens,
// which the draft kernels are built for. The batch width, prefill budget, KV
// page and verify rows it records were that runtime's choices; this runtime
// makes its own.
void validateExecutionGeometry(NSDictionary *manifest) {
  requireNumbers(requireObject(manifest, @"execution_geometry",
                               "model execution geometry"),
                 "execution_geometry",
                 {{"draft_proposal_tokens", ExecutionLimits::draftProposalTokens},
                  {"draft_query_rows", ExecutionLimits::draftQueryRows},
                  {"draft_sliding_window", ExecutionLimits::draftContextTokens}});
}

void validateCommonFormat(NSDictionary *format, std::string_view targetMagic) {
  requireNumbers(format, "format",
                 {{"section_alignment_bytes", kWeightFileAlignment}});
  requireEqual(requireString(format, @"target_layer_magic",
                             "target_layer_magic"),
               targetMagic, "target_layer_magic");
  requireEqual(requireString(format, @"draft_layer_magic",
                             "draft_layer_magic"),
               kDFlashLayerMagic, "draft_layer_magic");
  requireEqual(requireString(format, @"vision_magic", "vision_magic"),
               kVisionMagic, "vision_magic");
}

ModelDescriptor qwen38Descriptor(std::string name) {
  return makeModelDescriptor(std::move(name), Qwen3_8Layout{},
                             kQwen3_8DraftLayout, ops::VisionLayout{});
}

ModelDescriptor qwen36Descriptor(std::string name) {
  constexpr Qwen3_6MoeLayout target;
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.hiddenSize;
  return makeModelDescriptor(std::move(name), target, kQwen3_6MoeDraftLayout,
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
  requireNumbers(
      text, "tokenizer text config",
      {{"hidden_size",
        std::visit([](const auto &layout) { return layout.hiddenSize; },
                   descriptor.target)},
       {"vocab_size", descriptor.capabilities.vocabularySize},
       {"max_position_embeddings",
        descriptor.capabilities.maximumContextTokens}});
}

void validateQwen38(NSDictionary *manifest,
                    const std::filesystem::path &root,
                    const ModelDescriptor &descriptor) {
  requireNumbers(manifest, "manifest", {{"schema_version", 3}});
  NSDictionary *format =
      requireObject(manifest, @"format", "model weight format");
  requireNumbers(format, "format",
                 {{"q4_bits", 4},
                  {"q4_group_size", kQ4GroupElements},
                  {"q4_storage_n", kQ4StorageN}});
  validateCommonFormat(format, Qwen3_8Layout::layerMagic);
  validateTokenizer(root, descriptor, "qwen3_5_text");
}

void validateQwen36(NSDictionary *manifest,
                    const std::filesystem::path &root,
                    const ModelDescriptor &descriptor) {
  requireNumbers(manifest, "manifest", {{"schema_version", 4}});
  NSDictionary *format =
      requireObject(manifest, @"format", "model weight format");
  requireNumbers(format, "format",
                 {{"q4_bits", 4},
                  {"q8_bits", 8},
                  {"quant_group_size", kQ4GroupElements},
                  {"storage_n", kQ4StorageN}});
  validateCommonFormat(format, Qwen3_6MoeLayout::layerMagic);

  const auto &targetLayout = std::get<Qwen3_6MoeLayout>(descriptor.target);
  NSDictionary *target =
      requireObject(manifest, @"target", "target declaration");
  requireEqual(requireString(target, @"architecture", "target architecture"),
               "qwen3_5_moe", "target architecture");
  requireNumbers(target, "target",
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
                   targetLayout.expertIntermediateSize}});
  requireLayerTypes(requireArray(target, @"layer_types", "target layer_types"),
                    targetLayout, @"attention", @"gdn", "target layer_types");

  const DFlashDraftLayout &draftLayout = descriptor.draft;
  NSDictionary *draft =
      requireObject(manifest, @"draft", "draft declaration");
  requireEqual(requireString(draft, @"architecture", "draft architecture"),
               "DFlash2DraftModel", "draft architecture");
  requireNumbers(draft, "draft",
                 {{"layers", draftLayout.layers},
                  {"hidden_size", draftLayout.hiddenSize},
                  {"intermediate_size", draftLayout.intermediateSize},
                  {"sliding_window", ExecutionLimits::draftContextTokens},
                  {"block_size", ExecutionLimits::draftQueryRows},
                  {"dynamic_conv_group_size", SPLASH_DRAFT_CONVOLUTION_GROUP},
                  {"dynamic_conv_kernel_size", SPLASH_DRAFT_CONVOLUTION_TAPS},
                  {"selector_rank", draftLayout.selectorRank},
                  {"selector_top_k", SPLASH_DRAFT_CANDIDATES}});
  requireNumbers(requireArray(draft, @"target_capture_layers",
                              "draft target_capture_layers"),
                 targetLayout.hiddenCaptureLayers, "draft target_capture_layers");
  validateTokenizer(root, descriptor, "qwen3_5_moe_text");
}

// The target's text configuration. Every one holds the sizes the descriptor
// shares with it. An MLX target's config.json, which is target/config.json
// too and which its images are planned from, also holds the rest of what the
// kernels compute; a GGUF's is what the installer derived from the GGUF's
// metadata, which the GGUF planner checks.
void validateTextConfig(NSDictionary *text, const QwenTargetDimensions &target,
                        TargetSource source) {
  requireNumbers(text, "text config",
                 {{"hidden_size", target.hiddenSize},
                  {"num_hidden_layers", target.layers},
                  {"vocab_size", target.vocabularySize},
                  {"max_position_embeddings", target.maximumContextTokens},
                  {"num_attention_heads", target.attentionQueryHeads},
                  {"num_key_value_heads", target.attentionKvHeads},
                  {"head_dim", target.attentionHeadDimension}});
  if (source != TargetSource::Mlx) return;
  requireNumbers(text, "text config",
                 {{"linear_num_key_heads", target.gdnKeyHeads},
                  {"linear_num_value_heads", target.gdnValueHeads},
                  {"linear_key_head_dim", target.gdnHeadDimension},
                  {"linear_value_head_dim", target.gdnHeadDimension},
                  {"linear_conv_kernel_dim", kGdnConvolutionTaps},
                  {"full_attention_interval", target.fullAttentionPeriod},
                  {"rms_norm_eps", SPLASH_RMS_EPSILON}});
  if (target.ffnKind == QwenFfnKind::SparseMoe)
    requireNumbers(text, "text config",
                   {{"num_experts", target.experts},
                    {"num_experts_per_tok", target.expertsPerToken},
                    {"moe_intermediate_size", target.expertIntermediateSize},
                    {"shared_expert_intermediate_size",
                     target.expertIntermediateSize}});
  else
    requireNumbers(text, "text config",
                   {{"intermediate_size", target.intermediateSize}});
  requireBooleans(text, "text config",
                  {{"attention_bias", false},
                   {"attn_output_gate", true},
                   {"tie_word_embeddings", false}});
  requireEqual(requireString(text, @"hidden_act", "text config hidden_act"),
               "silu", "text config hidden_act");
  requireLayerTypes(requireArray(text, @"layer_types", "text config layer_types"),
                    target, @"full_attention", @"linear_attention",
                    "text config layer_types");
  NSDictionary *rope =
      requireObject(text, @"rope_parameters", "text config rope_parameters");
  requireNumbers(rope, "text config rope_parameters",
                 {{"rope_theta", target.rotaryTheta},
                  {"partial_rotary_factor", 2.0 * target.rotaryPairs /
                                                target.attentionHeadDimension}});
  // Transformers also reads the rope type from the older `type` key, which
  // fine-tunes such as Ornith 1.5 still write.
  NSString *typeKey = rope[@"rope_type"] ? @"rope_type" : @"type";
  const std::string typeLabel =
      std::string("text config rope_parameters ") + typeKey.UTF8String;
  requireEqual(requireString(rope, typeKey, typeLabel), "default", typeLabel);
}

// A DFlash2 checkpoint's config: the draft's layout; the block, window,
// convolutions and selector the draft kernels are built for; and the target's
// mask token and the layers the draft reads.
void validateDraftConfig(NSDictionary *draft, const DFlashDraftLayout &layout,
                         uint32_t maskToken,
                         std::span<const uint32_t> captureLayers) {
  NSArray *architectures =
      requireArray(draft, @"architectures", "draft architectures");
  if (architectures.count != 1 ||
      ![architectures[0] isEqual:@"DFlash2DraftModel"])
    throw std::invalid_argument("draft is not a DFlash2 model");
  requireNumbers(draft, "draft config",
                 {{"num_hidden_layers", layout.layers},
                  {"hidden_size", layout.hiddenSize},
                  {"vocab_size", layout.vocabularySize},
                  {"intermediate_size", layout.intermediateSize},
                  {"num_attention_heads",
                   layout.attentionSize / layout.attentionHeadDimension},
                  {"num_key_value_heads", layout.kvHeads},
                  {"head_dim", layout.attentionHeadDimension},
                  {"sliding_window", ExecutionLimits::draftContextTokens},
                  {"rms_norm_eps", SPLASH_RMS_EPSILON}});
  requireBooleans(draft, "draft config",
                  {{"is_causal", false},
                   {"attention_bias", false},
                   {"tie_word_embeddings", false}});
  requireEqual(requireString(draft, @"hidden_act", "draft config hidden_act"),
               "silu", "draft config hidden_act");
  NSDictionary *rope =
      requireObject(draft, @"rope_parameters", "draft config rope_parameters");
  requireEqual(requireString(rope, @"rope_type",
                             "draft config rope_parameters rope_type"),
               "default", "draft config rope_parameters rope_type");
  requireNumbers(rope, "draft config rope_parameters",
                 {{"rope_theta", layout.rotaryTheta}});
  NSDictionary *flash =
      requireObject(draft, @"dflash_config", "draft config dflash_config");
  requireNumbers(flash, "draft config dflash_config",
                 {{"block_size", ExecutionLimits::draftQueryRows},
                  {"conv_group_size", SPLASH_DRAFT_CONVOLUTION_GROUP},
                  {"conv_kernel_size", SPLASH_DRAFT_CONVOLUTION_TAPS},
                  {"selector_rank", layout.selectorRank},
                  {"selector_top_k", SPLASH_DRAFT_CANDIDATES},
                  {"mask_token_id", maskToken}});
  requireNumbers(requireArray(flash, @"target_layer_ids",
                              "draft config target_layer_ids"),
                 captureLayers, "draft config target_layer_ids");
}

ModelDescriptor inspectSourceModel(const std::filesystem::path &root) {
  std::string sourceIdentity;
  NSDictionary *record = readObject(root / "model.json", "resolved model", &sourceIdentity);
  requireNumbers(record, "model record", {{"version", 1}});
  NSDictionary *config = readObject(root / "config.json", "upstream model config");
  NSDictionary *text = requireObject(config, @"text_config", "text config");
  const auto type = requireString(text, @"model_type", "text model type");
  const auto name = requireString(record, @"model", "model name");
  ModelDescriptor result;
  if (type == "qwen3_5_moe_text") result = qwen36Descriptor(name);
  else if (type == "qwen3_5_text") result = qwen38Descriptor(name);
  else throw std::invalid_argument("unsupported model architecture: " + type);
  const auto target = requireString(record, @"target_format", "target format");
  if (target == "mlx-affine") result.targetSource = TargetSource::Mlx;
  else if (target == "gguf") result.targetSource = TargetSource::Gguf;
  else throw std::invalid_argument("unsupported target source format: " + target);
  NSDictionary *draft = readObject(root / "draft" / "config.json", "draft config");
  std::visit([&](const auto &layout) {
    validateTextConfig(text, layout, result.targetSource);
    validateDraftConfig(draft, result.draft, layout.maskToken, layout.hiddenCaptureLayers);
  }, result.target);

  const auto vision = requireString(record, @"vision_format", "vision format");
  if (vision == "none") result.visionSource = VisionSource::None;
  else {
    if (vision == "safetensors") result.visionSource = VisionSource::Mlx;
    else if (vision == "gguf") result.visionSource = VisionSource::Gguf;
    else throw std::invalid_argument("unsupported vision source format: " + vision);
    NSDictionary *v = requireObject(config, @"vision_config", "vision config");
    const auto &l = result.vision;
    requireNumbers(v, "vision config", {{"depth", l.depth}, {"hidden_size", l.hiddenSize}, {"num_heads", l.heads},
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
