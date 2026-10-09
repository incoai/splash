#include "ModelDescriptor.hpp"
#include "GgufImage.hpp"
#include "MlxImage.hpp"
#include "WeightStore.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/ExecutionGeometry.h"

#import <Foundation/Foundation.h>

#include <cmath>
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
// this is not one and is not read.
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

// A value that is not the runtime's is refused naming its source, what the
// model was installed from or the assembly's own record, beside the runtime:
// "mismatch: MLX 47, runtime 48".
void requireEqual(std::string_view actual, std::string_view expected,
                  std::string_view source, std::string_view label) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(label) + " mismatch: " + std::string(source) +
                                " " + std::string(actual) + ", runtime " +
                                std::string(expected));
  }
}

// The one rule every number of a record or configuration is read by: a JSON
// number, never a boolean.
NSNumber *requireNumber(id value, std::string_view label) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
    throw std::invalid_argument(std::string(label) + " must be a number");
  return value;
}

// A number equal to the value expected. A config saved from Python may write
// a whole number as a float, rope_theta=1e7 as 10000000.0, which equals that
// integer; every value expected is exact as a double.
void requireNumber(id value, double expected, std::string_view source,
                   std::string_view label) {
  NSNumber *number = requireNumber(value, label);
  if (number.doubleValue != expected)
    throw std::invalid_argument(std::string(label) + " mismatch: " + std::string(source) +
                                " " + number.description.UTF8String + ", runtime " +
                                @(expected).description.UTF8String);
}

// A whole number from 0 to 2^32 - 1, which may be written as a float.
uint32_t requireWhole(id value, std::string_view label) {
  const double number = requireNumber(value, label).doubleValue;
  if (!(number >= 0 && number <= UINT32_MAX) || number != std::floor(number))
    throw std::invalid_argument(std::string(label) + " must be a whole number");
  return static_cast<uint32_t>(number);
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
void requireNumbers(NSDictionary *object, std::string_view source, std::string_view where,
                    std::initializer_list<ExpectedNumber> fields) {
  for (const ExpectedNumber &field : fields)
    requireNumber(object[@(field.key)], field.value, source,
                  std::string(where) + " " + field.key);
}

// The numbers of an array, in order.
void requireNumbers(NSArray *values, std::span<const uint32_t> expected,
                    std::string_view source, std::string_view label) {
  if (values.count != expected.size())
    throw std::invalid_argument(std::string(label) + " count mismatch: " + std::string(source) +
                                " " + std::to_string(values.count) + ", runtime " +
                                std::to_string(expected.size()));
  for (size_t index = 0; index < expected.size(); ++index)
    requireNumber(values[index], expected[index], source,
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

// Each layer's type in a layer_types array: full_attention for a
// full-attention layer, linear_attention for a GDN layer.
void requireLayerTypes(NSArray *types, const QwenTargetDimensions &target,
                       std::string_view source, std::string_view label) {
  if (types.count != target.layers)
    throw std::invalid_argument(std::string(label) + " count mismatch: " + std::string(source) +
                                " " + std::to_string(types.count) + ", runtime " +
                                std::to_string(target.layers));
  for (uint32_t layer = 0; layer < target.layers; ++layer) {
    NSString *expected =
        target.isFullAttentionLayer(layer) ? @"full_attention" : @"linear_attention";
    if (![types[layer] isEqual:expected])
      throw std::invalid_argument(std::string(label) + " " +
                                  std::to_string(layer) + " must be " +
                                  expected.UTF8String);
  }
}

// Each family's native window, which its config's max_position_embeddings
// states, fits the runtime's KV ceiling.
static_assert(Qwen3_8Layout{}.maximumContextTokens <= kv::kMaximumLogicalTokens &&
                  Qwen3_6MoeLayout{}.maximumContextTokens <= kv::kMaximumLogicalTokens,
              "a family's native window exceeds the runtime's KV ceiling");

ModelDescriptor qwen38Descriptor(std::string name, TargetSource targetSource,
                                 VisionSource visionSource) {
  return makeModelDescriptor(std::move(name), Qwen3_8Layout{},
                             kQwen3_8DraftLayout, kQwen3_8VisionLayout,
                             targetSource, visionSource);
}

ModelDescriptor qwen36Descriptor(std::string name, TargetSource targetSource,
                                 VisionSource visionSource) {
  return makeModelDescriptor(std::move(name), Qwen3_6MoeLayout{},
                             kQwen3_6MoeDraftLayout, kQwen3_6MoeVisionLayout,
                             targetSource, visionSource);
}

// The name errors give an upstream target's source.
std::string_view sourceName(TargetSource source) {
  return source == TargetSource::Mlx ? "MLX" : "GGUF";
}

// The refusal of a model of none of the families Splash serves: what tells it
// from the family its config's model type names, or that it names none.
std::invalid_argument unsupportedModel(const std::string &difference) {
  return std::invalid_argument("no supported model has this architecture (" + difference + "); supported: " +
                               std::string(Qwen3_8Layout::family) + ", " + std::string(Qwen3_6MoeLayout::family));
}

// The target's text configuration, of the family its model type names. Every
// one holds the sizes the descriptor shares with it, which tell the family's
// target from other models of its architecture. An MLX target's config.json
// also holds the rest of what the kernels compute; a GGUF's is what the
// installer derived from the GGUF's metadata, which the GGUF planner checks
// (gguf::requireMetadata).
void validateTextConfig(NSDictionary *text, const QwenTargetDimensions &target, std::string_view family,
                        TargetSource source) {
  const std::string_view name = sourceName(source);
  for (const ExpectedNumber &size : std::initializer_list<ExpectedNumber>{
           {"hidden_size", target.hiddenSize},
           {"num_hidden_layers", target.layers},
           {"vocab_size", target.vocabularySize},
           {"max_position_embeddings", target.maximumContextTokens},
           {"num_attention_heads", target.attentionQueryHeads},
           {"num_key_value_heads", target.attentionKvHeads},
           {"head_dim", target.attentionHeadDimension}}) {
    const std::string label = std::string("text config ") + size.key;
    NSNumber *value = requireNumber(text[@(size.key)], label);
    if (value.doubleValue != size.value)
      throw unsupportedModel(label + ": " + std::string(name) + " " + value.description.UTF8String + ", " +
                             std::string(family) + " " + @(size.value).description.UTF8String);
  }
  if (source != TargetSource::Mlx) return;
  requireNumbers(text, name, "text config",
                 {{"linear_num_key_heads", target.gdnKeyHeads},
                  {"linear_num_value_heads", target.gdnValueHeads},
                  {"linear_key_head_dim", target.gdnHeadDimension},
                  {"linear_value_head_dim", target.gdnHeadDimension},
                  {"linear_conv_kernel_dim", kGdnConvolutionTaps},
                  {"full_attention_interval", target.fullAttentionPeriod},
                  {"rms_norm_eps", SPLASH_RMS_EPSILON}});
  if (target.ffnKind == QwenFfnKind::SparseMoe)
    requireNumbers(text, name, "text config",
                   {{"num_experts", target.experts},
                    {"num_experts_per_tok", target.expertsPerToken},
                    {"moe_intermediate_size", target.expertIntermediateSize},
                    {"shared_expert_intermediate_size",
                     target.expertIntermediateSize}});
  else
    requireNumbers(text, name, "text config",
                   {{"intermediate_size", target.intermediateSize}});
  requireBooleans(text, "text config",
                  {{"attention_bias", false},
                   {"attn_output_gate", true},
                   {"tie_word_embeddings", false}});
  requireEqual(requireString(text, @"hidden_act", "text config hidden_act"),
               "silu", name, "text config hidden_act");
  requireLayerTypes(requireArray(text, @"layer_types", "text config layer_types"),
                    target, name, "text config layer_types");
  NSDictionary *rope =
      requireObject(text, @"rope_parameters", "text config rope_parameters");
  requireNumbers(rope, name, "text config rope_parameters",
                 {{"rope_theta", target.rotaryTheta},
                  {"partial_rotary_factor", 2.0 * target.rotaryPairs /
                                                target.attentionHeadDimension}});
  // Transformers also reads the rope type from the older `type` key, which
  // fine-tunes such as Ornith 1.5 still write.
  NSString *typeKey = rope[@"rope_type"] ? @"rope_type" : @"type";
  const std::string typeLabel =
      std::string("text config rope_parameters ") + typeKey.UTF8String;
  requireEqual(requireString(rope, typeKey, typeLabel), "default", name, typeLabel);
}

// An MLX target's quantization, the "quantization" object of its config.json:
// the object and each module's own entry name a format the block kernels hold
// (metal/abi/QuantFormat.h), as MLX loads it: an entry's mode is MLX's
// default, affine, unless it names another. The object states its bits and
// group size; a module's entry that omits either takes its mode's default, as
// MLX's to_quantized does: affine 4 bits in groups of 64, mxfp4 4 bits in
// groups of 32. The images read each module's format from its tensors, as MLX
// does (model/MlxImage.hpp). An entry that is not an object, false as MLX
// writes for a module it leaves unquantized, is refused for a module the
// images read only quantized (mlx::quantizedModules: each projection, the head
// and the token table), so before any weight download; any other module's,
// such as the router's, the shared-expert gate's or GDN alpha's and beta's,
// which the images read unquantized too, is accepted.
void requireQuantization(NSDictionary *config, const QwenTargetDimensions &geometry) {
  // A checkpoint without it holds BF16 weights, or another method's that a
  // transformers quantization_config states (GPTQ, AWQ, ...).
  NSDictionary *quantization = config[@"quantization"];
  if (![quantization isKindOfClass:[NSDictionary class]])
    throw std::invalid_argument(
        "this model requires an MLX checkpoint (affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128, or "
        "mxfp4) or a supported GGUF");
  const auto require = [](NSDictionary *entry, const std::string &label, bool module) {
    NSString *mode = entry[@"mode"] ?: @"affine";
    if (![mode isKindOfClass:[NSString class]]) throw std::invalid_argument(label + " mode must be a string");
    const bool affine = [mode isEqual:@"affine"];
    const auto number = [&](NSString *key, uint32_t modeDefault) {
      return module && !entry[key] ? modeDefault : requireWhole(entry[key], label + " " + key.UTF8String);
    };
    const uint32_t bits = number(@"bits", 4), group = number(@"group_size", affine ? 64 : 32);
    if (affine ? quant_affine_format_of(bits, group) == GGUF_FMT_COUNT
               : ![mode isEqual:@"mxfp4"] || bits != 4 || group != 32)
      throw std::invalid_argument(label + " is " + mode.UTF8String + " " + std::to_string(bits) +
                                  "-bit in groups of " + std::to_string(group) +
                                  "; MLX weights load as affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or "
                                  "128, or as mxfp4");
  };
  require(quantization, "quantization", false);
  // A module's entry is an object; the object's other values are its own.
  for (NSString *module in quantization)
    if ([quantization[module] isKindOfClass:[NSDictionary class]])
      require(quantization[module], "quantization " + std::string(module.UTF8String ?: ""), true);
  for (const std::string &module : mlx::quantizedModules(geometry)) {
    id entry = quantization[@(module.c_str())];
    if (entry && ![entry isKindOfClass:[NSDictionary class]])
      throw std::invalid_argument("quantization " + module +
                                  " is unquantized; Splash loads quantized MLX projections and token tables");
  }
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
  requireNumbers(draft, "checkpoint", "draft config",
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
               "silu", "checkpoint", "draft config hidden_act");
  NSDictionary *rope =
      requireObject(draft, @"rope_parameters", "draft config rope_parameters");
  requireEqual(requireString(rope, @"rope_type",
                             "draft config rope_parameters rope_type"),
               "default", "checkpoint", "draft config rope_parameters rope_type");
  requireNumbers(rope, "checkpoint", "draft config rope_parameters",
                 {{"rope_theta", layout.rotaryTheta}});
  NSDictionary *flash =
      requireObject(draft, @"dflash_config", "draft config dflash_config");
  requireNumbers(flash, "checkpoint", "draft config dflash_config",
                 {{"block_size", ExecutionLimits::draftQueryRows},
                  {"conv_group_size", SPLASH_DRAFT_CONVOLUTION_GROUP},
                  {"conv_kernel_size", SPLASH_DRAFT_CONVOLUTION_TAPS},
                  {"selector_rank", layout.selectorRank},
                  {"selector_top_k", SPLASH_DRAFT_CANDIDATES},
                  {"mask_token_id", maskToken}});
  requireNumbers(requireArray(flash, @"target_layer_ids",
                              "draft config target_layer_ids"),
                 captureLayers, "checkpoint", "draft config target_layer_ids");
}

// The vision tower's config, config.json's vision_config, which the
// installer derives from a GGUF's mmproj too: the tower the layout describes,
// over RGB patches of two frames, without deepstack layers.
void validateVisionConfig(NSDictionary *vision, const ops::VisionLayout &layout,
                          TargetSource source) {
  requireNumbers(vision, sourceName(source), "vision config",
                 {{"depth", layout.depth},
                  {"hidden_size", layout.hiddenSize},
                  {"num_heads", layout.heads},
                  {"intermediate_size", layout.intermediateSize},
                  {"out_hidden_size", layout.outputHiddenSize},
                  {"patch_size", layout.patchSize},
                  {"spatial_merge_size", layout.spatialMerge},
                  {"temporal_patch_size", 2},
                  {"in_channels", ops::kImageChannels},
                  {"num_position_embeddings", layout.positionGridSide * layout.positionGridSide}});
  requireEqual(requireString(vision, @"hidden_act", "vision config hidden_act"), "gelu_pytorch_tanh",
               sourceName(source), "vision config hidden_act");
  if (requireArray(vision, @"deepstack_visual_indexes", "vision config deepstack_visual_indexes").count)
    throw std::invalid_argument("vision deepstack layers are unsupported");
}

// The descriptor, named name, of an upstream model of the family its text
// config's model type names, from the source formats its record names, with
// each config checked: the target's, as those formats require, and the
// draft's unless draft is nil.
ModelDescriptor describeSourceModel(std::string name, std::string_view targetFormat,
                                    std::string_view visionFormat, NSDictionary *config,
                                    NSDictionary *draft) {
  NSDictionary *text = config[@"text_config"];
  if (![text isKindOfClass:[NSDictionary class]]) throw unsupportedModel("its config has no text_config");
  const auto type = requireString(text, @"model_type", "text model type");
  const bool moe = type == "qwen3_5_moe_text";
  if (!moe && type != "qwen3_5_text") throw unsupportedModel("text model type " + type);
  TargetSource targetSource;
  if (targetFormat == "mlx-affine") targetSource = TargetSource::Mlx;
  else if (targetFormat == "gguf") targetSource = TargetSource::Gguf;
  else throw std::invalid_argument("unsupported target source format: " + std::string(targetFormat));
  VisionSource visionSource;
  if (visionFormat == "none") visionSource = VisionSource::None;
  else if (visionFormat == "safetensors") visionSource = VisionSource::Mlx;
  else if (visionFormat == "gguf") visionSource = VisionSource::Gguf;
  else throw std::invalid_argument("unsupported vision source format: " + std::string(visionFormat));
  ModelDescriptor result = moe ? qwen36Descriptor(std::move(name), targetSource, visionSource)
                               : qwen38Descriptor(std::move(name), targetSource, visionSource);
  std::visit([&](const auto &layout) {
    validateTextConfig(text, layout, layout.family, result.targetSource);
    if (result.targetSource == TargetSource::Mlx) requireQuantization(config, layout);
    if (draft) validateDraftConfig(draft, result.draft, layout.maskToken, layout.hiddenCaptureLayers);
  }, result.target);
  if (result.hasVision())
    validateVisionConfig(requireObject(config, @"vision_config", "vision config"), result.vision,
                         result.targetSource);
  return result;
}

// The text of a JSON string, which a lone surrogate escape leaves without one.
std::string requireText(id value, std::string_view label) {
  const char *text = [value isKindOfClass:[NSString class]] ? static_cast<NSString *>(value).UTF8String : nullptr;
  if (!text) throw std::invalid_argument(std::string(label) + " must be a string");
  return text;
}

// A GGUF's scalar metadata as the installer copies it from the header it read
// (install/gguf.py scalar_metadata): "unsigned", "float" and "string" objects
// of the values of each kind. An unsigned one is a whole number from 0 to
// 2^64 - 1, which NSJSONSerialization keeps exactly as a signed or, past
// 2^63 - 1, an unsigned 64-bit integer.
GgufMetadata readGgufMetadata(const std::filesystem::path &path) {
  NSDictionary *object = readObject(path, "GGUF metadata");
  GgufMetadata metadata;
  NSDictionary *unsigneds = requireObject(object, @"unsigned", "GGUF metadata unsigned");
  for (id key in unsigneds) {
    const std::string name = requireText(key, "GGUF metadata key");
    id value = unsigneds[key];
    const std::string_view type = [value isKindOfClass:[NSNumber class]] ? [value objCType] : "";
    if (type != "Q" && (type != "q" || [value longLongValue] < 0))
      throw std::invalid_argument("GGUF metadata " + name + " must be a whole number from 0 to 2^64 - 1");
    metadata.unsigneds.emplace(name, [value unsignedLongLongValue]);
  }
  NSDictionary *floats = requireObject(object, @"float", "GGUF metadata float");
  for (id key in floats) {
    const std::string name = requireText(key, "GGUF metadata key");
    metadata.floats.emplace(name, requireNumber(floats[key], "GGUF metadata " + name).doubleValue);
  }
  NSDictionary *strings = requireObject(object, @"string", "GGUF metadata string");
  for (id key in strings) {
    const std::string name = requireText(key, "GGUF metadata key");
    metadata.strings.emplace(name, requireText(strings[key], "GGUF metadata " + name));
  }
  return metadata;
}

} // namespace

ModelDescriptor makeModelDescriptor(std::string name, TargetLayout target,
                                    DFlashDraftLayout draft,
                                    ops::VisionLayout vision,
                                    TargetSource targetSource,
                                    VisionSource visionSource) {
  ModelDescriptor result;
  result.name = std::move(name);
  result.target = target;
  result.draft = draft;
  result.vision = vision;
  result.targetSource = targetSource;
  result.visionSource = visionSource;
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
  if (name.empty() || targetSource == TargetSource{} || visionSource == VisionSource{} ||
      !capabilities.vocabularySize ||
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

ModelDescriptor inspectModelRoot(const std::filesystem::path &root) {
  @autoreleasepool {
    std::string sourceIdentity;
    NSDictionary *record = readObject(root / "model.json", "resolved model", &sourceIdentity);
    requireNumbers(record, "assembly", "model record", {{"version", 1}});
    std::string name = requireString(record, @"model", "model name");
    const std::string targetFormat = requireString(record, @"target_format", "target format");
    const std::string visionFormat = requireString(record, @"vision_format", "vision format");
    NSDictionary *config = readObject(root / "config.json", "upstream model config");
    NSDictionary *draft = readObject(root / "draft" / "config.json", "draft config");
    ModelDescriptor result = describeSourceModel(std::move(name), targetFormat, visionFormat, config, draft);
    result.sourceIdentity = std::move(sourceIdentity);
    if (!result.valid()) throw std::invalid_argument("incompatible target and draft model");
    return result;
  }
}

std::string_view inspectSourceConfiguration(std::string_view targetFormat, std::string_view visionFormat,
                                            const std::filesystem::path &config,
                                            const std::optional<std::filesystem::path> &ggufMetadata,
                                            const std::optional<std::filesystem::path> &draft) {
  @autoreleasepool {
    NSDictionary *target = readObject(config, "upstream model config");
    NSDictionary *checkpoint = draft ? readObject(*draft, "draft config") : nil;
    // Unnamed: the record names an installation, which this precedes.
    const ModelDescriptor descriptor = describeSourceModel({}, targetFormat, visionFormat, target, checkpoint);
    if (descriptor.targetSource == TargetSource::Gguf) {
      if (!ggufMetadata) throw std::invalid_argument("a GGUF target is checked with its metadata");
      const GgufMetadata metadata = readGgufMetadata(*ggufMetadata);
      std::visit([&](const auto &layout) { gguf::requireMetadata(metadata, layout); }, descriptor.target);
    }
    return descriptor.family();
  }
}

} // namespace splash::model
