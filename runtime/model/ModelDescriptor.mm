#include "ModelDescriptor.hpp"
#include "GgufImage.hpp"
#include "SafetensorsImage.hpp"
#include "WeightStore.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/ExecutionGeometry.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fnmatch.h>
#include <initializer_list>
#include <iterator>
#include <regex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
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

// The text of a JSON string, which a lone surrogate escape leaves without one.
std::string requireText(id value, std::string_view label) {
  const char *text = [value isKindOfClass:[NSString class]] ? static_cast<NSString *>(value).UTF8String : nullptr;
  if (!text) throw std::invalid_argument(std::string(label) + " must be a string");
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
                  Qwen3_6MoeLayout{}.maximumContextTokens <= kv::kMaximumLogicalTokens &&
                  Qwen4ExpLayout{}.maximumContextTokens <= kv::kMaximumLogicalTokens,
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

// mtp: whether the MTP head drafts (Qwen4ExpLayout::mtpLayers).
ModelDescriptor qwen4Descriptor(std::string name, bool mtp, TargetSource targetSource,
                                VisionSource visionSource) {
  Qwen4ExpLayout target;
  target.mtpLayers = mtp ? 1 : 0;
  ModelDescriptor result = makeModelDescriptor(
      std::move(name), target,
      nullDraftLayout(target.hiddenSize, target.vocabularySize, target.capturedHiddenSize()),
      kQwen4ExpVisionLayout, targetSource, visionSource);
  result.draftModel = false;
  return result;
}

// The name errors give an upstream target's source.
std::string_view sourceName(TargetSource source) {
  return source == TargetSource::Safetensors ? "MLX" : "GGUF";
}

// The refusal of a model of none of the families Splash serves: what tells it
// from the family its config's model type names, or that it names none.
std::invalid_argument unsupportedModel(const std::string &difference) {
  return std::invalid_argument("no supported model has this architecture (" + difference + "); supported: " +
                               std::string(Qwen3_8Layout::family) + ", " + std::string(Qwen3_6MoeLayout::family) +
                               ", " + std::string(Qwen4ExpLayout::family));
}

// The target's text configuration, of the family its model type names. Every
// one holds the sizes the descriptor shares with it, which tell the family's
// target from other models of its architecture. A safetensors target's
// config.json also holds the rest of what the kernels compute; a GGUF's is
// what the installer derived from the GGUF's metadata, which the GGUF planner
// checks (gguf::requireMetadata).
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
  if (source != TargetSource::Safetensors) return;
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
// (metal/abi/QuantFormat.h), as MLX loads it: an entry's mode is MLX's default,
// affine, unless it names another. The object states its bits and group size; a
// module's entry that omits either takes its mode's default, as MLX's
// to_quantized does: affine 4 bits in groups of 64, mxfp4 4 bits in groups of
// 32, nvfp4 4 bits in groups of 16. The images read each module's format from
// its tensors, as MLX does (model/SafetensorsImage.hpp). An entry that is not
// an object, false as MLX writes for a module it leaves unquantized, is refused
// for a module the images read only quantized (safetensors::quantizedModules:
// each projection and the head), so before any weight download; any other
// module's, such as the router's, the shared-expert gate's, GDN alpha's and
// beta's or the token table's, which the images read unquantized too, is
// accepted.
void requireMlxQuantization(NSDictionary *quantization, const QwenTargetDimensions &geometry) {
  const auto require = [](NSDictionary *entry, const std::string &label, bool module) {
    NSString *mode = entry[@"mode"] ?: @"affine";
    if (![mode isKindOfClass:[NSString class]]) throw std::invalid_argument(label + " mode must be a string");
    const bool affine = [mode isEqual:@"affine"], nvfp4 = [mode isEqual:@"nvfp4"];
    const auto number = [&](NSString *key, uint32_t modeDefault) {
      return module && !entry[key] ? modeDefault : requireWhole(entry[key], label + " " + key.UTF8String);
    };
    const uint32_t bits = number(@"bits", 4), group = number(@"group_size", affine ? 64 : nvfp4 ? 16 : 32);
    if (affine ? quant_affine_format_of(bits, group) == GGUF_FMT_COUNT
               : nvfp4 ? bits != 4 || group != 16 : ![mode isEqual:@"mxfp4"] || bits != 4 || group != 32)
      throw std::invalid_argument(label + " is " + mode.UTF8String + " " + std::to_string(bits) +
                                  "-bit in groups of " + std::to_string(group) +
                                  "; MLX weights load as affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or "
                                  "128, or as mxfp4 or nvfp4");
  };
  require(quantization, "quantization", false);
  // A module's entry is an object; the object's other values are its own.
  for (NSString *module in quantization)
    if ([quantization[module] isKindOfClass:[NSDictionary class]])
      require(quantization[module], "quantization " + std::string(module.UTF8String ?: ""), true);
  for (const std::string &module : safetensors::quantizedModules(geometry, safetensors::ModuleNames::Mlx)) {
    id entry = quantization[@(module.c_str())];
    if (entry && ![entry isKindOfClass:[NSDictionary class]])
      throw std::invalid_argument("quantization " + module +
                                  " is unquantized; Splash loads quantized MLX projections");
  }
}

// The projections a configuration names for a layer's routed experts
// (safetensors::quantizedModules' mlp.experts by transformers names) as each
// expert's own modules: the first and the last expert's stand for every
// expert's, as matching all names of 256 experts in 40 layers takes half a
// second, and the images refuse any other expert left unquantized when they
// read its tensors.
std::vector<std::string> expertProjections(const std::string &experts, const QwenTargetDimensions &geometry) {
  std::vector<std::string> names;
  for (const uint32_t expert : {0u, geometry.experts - 1})
    for (const char *projection : {"gate_proj", "up_proj", "down_proj"})
      names.push_back(experts + "." + std::to_string(expert) + "." + projection);
  return names;
}

// A Model Optimizer target's quantization, its config.json's
// quantization_config of quant_method "modelopt": each layer it quantizes, by
// its quantized_layers or else by its quant_algo for every layer its ignore
// patterns leave, holds NVFP4 in groups of 16 (NVFP4, W4A16_NVFP4) or FP8 with
// a scale per tensor (FP8), which the block kernels read weights-only, and each
// module the images read only quantized (safetensors::quantizedModules by
// transformers names, a layer's routed experts as the one layer mlp.experts, as
// quantized_layers keys them) is such a layer: a key of quantized_layers, or
// else a module no ignore pattern names, nor its experts' projections
// (expertProjections). Its activation and KV cache quantization play no part:
// the kernels read activations in bf16 and keep Splash's own KV formats.
void requireModelOptimizerQuantization(NSDictionary *quantization, const QwenTargetDimensions &geometry) {
  const auto requireAlgorithm = [](NSDictionary *entry, const std::string &label) {
    const std::string algorithm = requireString(entry, @"quant_algo", label + " quant_algo");
    if (algorithm == "FP8") return;
    if (algorithm != "NVFP4" && algorithm != "W4A16_NVFP4")
      throw std::invalid_argument(label + " is " + algorithm +
                                  "; Model Optimizer weights load as NVFP4, W4A16_NVFP4 or FP8");
    if (entry[@"group_size"] && requireWhole(entry[@"group_size"], label + " group_size") != 16)
      throw std::invalid_argument(label + " is NVFP4 in groups of " +
                                  std::to_string(requireWhole(entry[@"group_size"], label + " group_size")) +
                                  "; Model Optimizer NVFP4 loads in groups of 16");
  };
  const std::vector<std::string> required =
      safetensors::quantizedModules(geometry, safetensors::ModuleNames::Transformers);
  NSDictionary *layers = quantization[@"quantized_layers"];
  if ([layers isKindOfClass:[NSDictionary class]]) {
    for (NSString *module in layers) {
      const std::string label = "quantization_config quantized_layers " + std::string(module.UTF8String ?: "");
      if (![layers[module] isKindOfClass:[NSDictionary class]])
        throw std::invalid_argument(label + " must be an object");
      requireAlgorithm(layers[module], label);
    }
    for (const std::string &module : required)
      if (!layers[@(module.c_str())])
        throw std::invalid_argument("quantization_config leaves " + module +
                                    " unquantized; Splash loads quantized projections");
    return;
  }
  requireAlgorithm(quantization, "quantization_config");
  NSArray *ignore = quantization[@"ignore"] ?: quantization[@"exclude_modules"];
  if (ignore && ![ignore isKindOfClass:[NSArray class]])
    throw std::invalid_argument("quantization_config ignore must be an array");
  for (id entry in ignore) {
    const std::string pattern = requireText(entry, "quantization_config ignore entry");
    for (const std::string &module : required) {
      std::vector<std::string> names{module};
      if (module.ends_with(".experts"))
        std::ranges::copy(expertProjections(module, geometry), std::back_inserter(names));
      for (const std::string &name : names)
        if (!fnmatch(pattern.c_str(), name.c_str(), 0))
          throw std::invalid_argument("quantization_config leaves " + name +
                                      " unquantized; Splash loads quantized projections");
    }
  }
}

// The modules compressed-tensors names by a configuration's targets and
// ignore entries: a module's own name, a regular expression after "re:"
// matched from the start of a name (as Python's re.match), or a class, of
// which Linear is every projection's. A pattern is bounded, as parsing it
// recurses on its nesting, and so is a match (libc++ throws error_complexity
// past its step bound), so no configuration exhausts the check.
class ModuleMatcher {
public:
  static constexpr size_t kMaxPatternBytes = 512;

  ModuleMatcher(NSArray *entries, std::string label) : label_(std::move(label)) {
    for (id entry in entries) {
      const std::string text = requireText(entry, label_ + " entry");
      if (text == "Linear") {
        linear_ = true;
      } else if (text.starts_with("re:")) {
        if (text.size() > kMaxPatternBytes)
          throw std::invalid_argument(label_ + " entry exceeds " + std::to_string(kMaxPatternBytes) + " bytes");
        try {
          patterns_.emplace_back(text.substr(3));
        } catch (const std::regex_error &) {
          throw std::invalid_argument(label_ + " entry " + text + " is not a regular expression Splash reads");
        }
      } else {
        names_.insert(text);
      }
    }
  }
  [[nodiscard]] bool matches(const std::string &module) const {
    if (linear_ || names_.contains(module)) return true;
    try {
      return std::ranges::any_of(patterns_, [&](const std::regex &pattern) {
        return std::regex_search(module, pattern, std::regex_constants::match_continuous);
      });
    } catch (const std::regex_error &) {
      throw std::invalid_argument(label_ + " holds a regular expression too complex to match " + module);
    }
  }

private:
  std::string label_;
  bool linear_ = false;
  std::unordered_set<std::string> names_;
  std::vector<std::regex> patterns_;
};

// A compressed-tensors target's quantization, its config.json's
// quantization_config of quant_method "compressed-tensors" (llm-compressor's,
// as unsloth's NVFP4 releases are): each config group's weights are NVFP4
// (nvfp4-pack-quantized: float 4-bit by tensor_group in groups of 16) or FP8
// (float-quantized: float 8-bit by channel or tensor), symmetric and in their
// stored column order, and each module the images read only quantized
// (safetensors::quantizedModules by transformers names, the routed experts by
// their projections, expertProjections) is a group's target that no ignore
// entry names (ModuleMatcher). Its activation and KV cache schemes play no
// part.
void requireCompressedTensorsQuantization(NSDictionary *quantization, const QwenTargetDimensions &geometry) {
  NSDictionary *groups = quantization[@"config_groups"];
  if (![groups isKindOfClass:[NSDictionary class]] || !groups.count)
    throw std::invalid_argument("quantization_config config_groups must be a non-empty object");
  const std::string format =
      quantization[@"format"] ? requireString(quantization, @"format", "quantization_config format") : "";
  std::vector<ModuleMatcher> targets;
  for (NSString *name in groups) {
    const std::string label = "quantization_config config_groups " + std::string(name.UTF8String ?: "");
    NSDictionary *group = groups[name], *weights = nil;
    if ([group isKindOfClass:[NSDictionary class]]) weights = group[@"weights"];
    if (![weights isKindOfClass:[NSDictionary class]])
      throw std::invalid_argument(label + " weights must be an object");
    const std::string groupFormat = group[@"format"] ? requireString(group, @"format", label + " format") : format;
    const std::string type = requireString(weights, @"type", label + " weights type");
    const std::string strategy = requireString(weights, @"strategy", label + " weights strategy");
    const uint64_t bits = requireWhole(weights[@"num_bits"], label + " weights num_bits");
    const uint64_t groupSize = weights[@"group_size"] && weights[@"group_size"] != [NSNull null]
                                   ? requireWhole(weights[@"group_size"], label + " weights group_size")
                                   : 0;
    const bool nvfp4 = groupFormat == "nvfp4-pack-quantized" && type == "float" && bits == 4 &&
                       strategy == "tensor_group" && groupSize == 16;
    const bool fp8 = groupFormat == "float-quantized" && type == "float" && bits == 8 &&
                     (strategy == "channel" || strategy == "tensor");
    if (!nvfp4 && !fp8)
      throw std::invalid_argument(label + " is " + groupFormat + ": " + type + " " + std::to_string(bits) + "-bit by " +
                                  strategy + (groupSize ? " in groups of " + std::to_string(groupSize) : "") +
                                  "; compressed-tensors weights load as NVFP4 (nvfp4-pack-quantized: float 4-bit by "
                                  "tensor_group in groups of 16) or FP8 (float-quantized: float 8-bit by channel or "
                                  "tensor)");
    if ([weights[@"symmetric"] isEqual:@NO] || [weights[@"actorder"] isEqual:@"group"] ||
        [weights[@"actorder"] isEqual:@YES])
      throw std::invalid_argument(label + " is asymmetric or ordered by activation groups; Splash loads symmetric "
                                  "compressed-tensors weights in their stored column order");
    targets.emplace_back(requireArray(group, @"targets", label + " targets"), label + " targets");
  }
  NSArray *ignoreEntries = quantization[@"ignore"] ?: @[];
  if (![ignoreEntries isKindOfClass:[NSArray class]])
    throw std::invalid_argument("quantization_config ignore must be an array");
  const ModuleMatcher ignore(ignoreEntries, "quantization_config ignore");
  for (const std::string &module : safetensors::quantizedModules(geometry, safetensors::ModuleNames::Transformers)) {
    const std::vector<std::string> names =
        module.ends_with(".experts") ? expertProjections(module, geometry) : std::vector<std::string>{module};
    for (const std::string &name : names)
      if (ignore.matches(name) ||
          std::ranges::none_of(targets, [&](const ModuleMatcher &target) { return target.matches(name); }))
        throw std::invalid_argument("quantization_config leaves " + name +
                                    " unquantized; Splash loads quantized projections");
  }
}

// A safetensors target's quantization: MLX's "quantization" object, or the
// quantization_config of Model Optimizer or compressed-tensors. A checkpoint
// with none of them holds BF16 weights, or another method's that a
// transformers quantization_config states (GPTQ, AWQ, ...).
void requireQuantization(NSDictionary *config, const QwenTargetDimensions &geometry) {
  NSDictionary *mlxQuantization = config[@"quantization"];
  if ([mlxQuantization isKindOfClass:[NSDictionary class]]) return requireMlxQuantization(mlxQuantization, geometry);
  NSDictionary *quantization = config[@"quantization_config"];
  if ([quantization isKindOfClass:[NSDictionary class]]) {
    if ([quantization[@"quant_method"] isEqual:@"modelopt"])
      return requireModelOptimizerQuantization(quantization, geometry);
    if ([quantization[@"quant_method"] isEqual:@"compressed-tensors"])
      return requireCompressedTensorsQuantization(quantization, geometry);
  }
  throw std::invalid_argument(
      "this model requires an MLX checkpoint (affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, 64 or 128, mxfp4 or "
      "nvfp4), an NVFP4 checkpoint of Model Optimizer or compressed-tensors, or a supported GGUF");
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
// draft's unless draft is nil. mtp says whether Qwen3.8-Flash-Next's MTP head
// drafts.
ModelDescriptor describeSourceModel(std::string name, std::string_view targetFormat,
                                    std::string_view visionFormat, NSDictionary *config,
                                    NSDictionary *draft, bool mtp) {
  NSDictionary *text = config[@"text_config"];
  if (![text isKindOfClass:[NSDictionary class]]) throw unsupportedModel("its config has no text_config");
  const auto type = requireString(text, @"model_type", "text model type");
  const bool moe = type == "qwen3_5_moe_text";
  const bool qwen4 = type == "qwen4_exp_text";
  if (!moe && !qwen4 && type != "qwen3_5_text") throw unsupportedModel("text model type " + type);
  TargetSource targetSource;
  // "mlx-affine" names every safetensors target, as installations record it.
  if (targetFormat == "mlx-affine") targetSource = TargetSource::Safetensors;
  else if (targetFormat == "gguf") targetSource = TargetSource::Gguf;
  else throw std::invalid_argument("unsupported target source format: " + std::string(targetFormat));
  VisionSource visionSource;
  if (visionFormat == "none") visionSource = VisionSource::None;
  else if (visionFormat == "safetensors") visionSource = VisionSource::Safetensors;
  else if (visionFormat == "gguf") visionSource = VisionSource::Gguf;
  else throw std::invalid_argument("unsupported vision source format: " + std::string(visionFormat));
  if (qwen4 && targetSource != TargetSource::Gguf)
    throw std::invalid_argument("Qwen3.8-Flash-Next loads from a GGUF only");
  ModelDescriptor result = moe     ? qwen36Descriptor(std::move(name), targetSource, visionSource)
                           : qwen4 ? qwen4Descriptor(std::move(name), mtp, targetSource, visionSource)
                                   : qwen38Descriptor(std::move(name), targetSource, visionSource);
  if (!result.hasDraft() && draft)
    throw std::invalid_argument("this model decodes without a DFlash2 draft, but a draft config was given");
  std::visit([&](const auto &layout) {
    validateTextConfig(text, layout, layout.family, result.targetSource);
    // Qwen3.8-Flash-Next loads from a GGUF only.
    if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(layout)>, Qwen4ExpLayout>)
      if (result.targetSource == TargetSource::Safetensors) requireQuantization(config, layout);
    if (draft) validateDraftConfig(draft, result.draft, layout.maskToken, layout.hiddenCaptureLayers);
  }, result.target);
  if (result.hasVision())
    validateVisionConfig(requireObject(config, @"vision_config", "vision config"), result.vision,
                         result.targetSource);
  return result;
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

DFlashDraftLayout nullDraftLayout(uint32_t hiddenSize, uint32_t vocabularySize, uint32_t capturedHiddenSize) {
  DFlashDraftLayout layout;
  layout.layers = 1;
  layout.hiddenSize = hiddenSize;
  layout.vocabularySize = vocabularySize;
  layout.kvHeads = 1;
  layout.attentionHeadDimension = 8;
  layout.targetHiddenSize = capturedHiddenSize;
  return layout;
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
    // A family no DFlash2 draft was trained for (Qwen3.8-Flash-Next) has no
    // draft/; its MTP head drafts when mtp/ holds it, unless SPLASH_MTP=0.
    const bool draftDirectory = std::filesystem::exists(root / "draft");
    NSDictionary *draft = draftDirectory ? readObject(root / "draft" / "config.json", "draft config") : nil;
    const char *mtpSetting = std::getenv("SPLASH_MTP");
    const bool mtp = std::filesystem::exists(root / "mtp") && !(mtpSetting && std::string_view(mtpSetting) == "0");
    ModelDescriptor result = describeSourceModel(std::move(name), targetFormat, visionFormat, config, draft, mtp);
    if (result.hasDraft() != draftDirectory)
      throw std::invalid_argument(result.hasDraft()
                                      ? "the model's root holds no draft/ for its DFlash2 draft"
                                      : "this model decodes without a DFlash2 draft, but its root holds one");
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
    const ModelDescriptor descriptor =
        describeSourceModel({}, targetFormat, visionFormat, target, checkpoint, false);
    if (descriptor.targetSource == TargetSource::Gguf) {
      if (!ggufMetadata) throw std::invalid_argument("a GGUF target is checked with its metadata");
      const GgufMetadata metadata = readGgufMetadata(*ggufMetadata);
      std::visit([&](const auto &layout) { gguf::requireMetadata(metadata, layout); }, descriptor.target);
    }
    return descriptor.family();
  }
}

} // namespace splash::model
