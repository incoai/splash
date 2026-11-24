#include "ModelDescriptor.hpp"
#include "GgufImage.hpp"
#include "GgufTarget.hpp"
#include "SafetensorsImage.hpp"
#include "WeightStore.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/PagedAttention.h"
#include "metal/abi/Vision.h"
#include "ops/MoE.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <optional>
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
// "mismatch: safetensors 47, runtime 48".
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

// Each layer's type in a layer_types array: linear_attention for a GDN
// layer; for an attention layer, sliding_attention where a window bounds it
// (kv::Layout::windowed), else full_attention.
void requireLayerTypes(NSArray *types, const QwenHybridLayout &target,
                       std::string_view source, std::string_view label) {
  if (types.count != target.layers)
    throw std::invalid_argument(std::string(label) + " count mismatch: " + std::string(source) +
                                " " + std::to_string(types.count) + ", runtime " +
                                std::to_string(target.layers));
  for (uint32_t layer = 0, attention = 0; layer < target.layers; ++layer) {
    NSString *expected = !target.isFullAttentionLayer(layer)       ? @"linear_attention"
                         : target.kvLayout().windowed(attention++) ? @"sliding_attention"
                                                                   : @"full_attention";
    if (![types[layer] isEqual:expected])
      throw std::invalid_argument(std::string(label) + " " +
                                  std::to_string(layer) + " must be " +
                                  expected.UTF8String);
  }
}

// Each family's native window, which its config's max_position_embeddings
// states, and the window YaRN stretches it to fit the runtime's KV ceiling.
template <class Layout> constexpr bool fitsKvCeiling(const Layout &layout) {
  return layout.nativeContextTokens <= kv::kMaximumLogicalTokens &&
         layout.yarnContextTokens <= kv::kMaximumLogicalTokens;
}
static_assert([] {
  bool fits = true;
  forEachFamily([&](const auto &layout) { fits = fits && fitsKvCeiling(layout); });
  return fits;
}(), "a family's window exceeds the runtime's KV ceiling");

// The name errors give an upstream target's source: its format, safetensors
// for MLX, Model Optimizer and compressed-tensors checkpoints alike.
std::string_view sourceName(TargetSource source) {
  return source == TargetSource::Safetensors ? "safetensors" : "GGUF";
}

// The refusal of a model of none of the families Splash serves: what tells it
// from the family its config's model type names, or that it names none.
std::invalid_argument unsupportedModel(const std::string &difference) {
  std::string supported;
  forEachFamily([&](const auto &layout) {
    supported += (supported.empty() ? "" : ", ") + std::string(layout.family);
  });
  return std::invalid_argument("no supported model has this architecture (" + difference + "); supported: " +
                               supported);
}

// How a text configuration differs from a family in the sizes every one
// holds, which tell the family's target from other models of its
// architecture: the first that differs, after `matched` that agree.
struct SizeDifference final {
  size_t matched;
  std::string difference;
};
std::optional<SizeDifference> sizeDifference(NSDictionary *text, std::string_view family, TargetSource source,
                                             std::initializer_list<ExpectedNumber> sizes) {
  size_t matched = 0;
  for (const ExpectedNumber &size : sizes) {
    const std::string label = std::string("text config ") + size.key;
    NSNumber *value = requireNumber(text[@(size.key)], label);
    if (value.doubleValue != size.value)
      return SizeDifference{matched, label + ": " + std::string(sourceName(source)) + " " +
                                         value.description.UTF8String + ", " + std::string(family) + " " +
                                         @(size.value).description.UTF8String};
    ++matched;
  }
  return std::nullopt;
}
std::optional<SizeDifference> sizeDifference(NSDictionary *text, const QwenHybridLayout &target,
                                             TargetSource source) {
  return sizeDifference(text, target.family, source,
                        {{"hidden_size", target.hiddenSize},
                         {"num_hidden_layers", target.layers},
                         {"vocab_size", target.vocabularySize},
                         {"max_position_embeddings", target.nativeContextTokens},
                         {"num_attention_heads", target.attentionQueryHeads},
                         {"num_key_value_heads", target.attentionKvHeads},
                         {"head_dim", target.attentionHeadDimension}});
}
// Inkling-2-Mini's sizes: its weight layers, which its steps run more
// applications of, and its own window (modelContextTokens, above the context
// it is served at) at model_max_length, its release config's key, which the
// config the installer derives states too.
std::optional<SizeDifference> sizeDifference(NSDictionary *text, const Inkling2LoopLayout &target,
                                             TargetSource source) {
  return sizeDifference(text, target.family, source,
                        {{"hidden_size", target.hiddenSize},
                         {"num_hidden_layers", target.weightLayers},
                         {"vocab_size", target.vocabularySize},
                         {"model_max_length", target.modelContextTokens},
                         {"num_attention_heads", target.attentionQueryHeads},
                         {"num_key_value_heads", target.attentionKvHeads},
                         {"head_dim", target.attentionHeadDimension}});
}

// The target's text configuration, of its family (describeSourceModel), whose
// sizes it holds (sizeDifference). A safetensors target's config.json also
// holds the rest of what the kernels compute; a GGUF's is what the installer
// derived from the GGUF's metadata, which the GGUF planner checks
// (gguf::planImages).
void validateTextConfig(NSDictionary *text, const QwenHybridLayout &target, TargetSource source,
                        bool statesAttentionGate = true) {
  const std::string_view name = sourceName(source);
  if (const auto difference = sizeDifference(text, target, source)) throw unsupportedModel(difference->difference);
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
                     target.sharedExpertSlices * target.expertIntermediateSize}});
  else
    requireNumbers(text, name, "text config",
                   {{"intermediate_size", target.intermediateSize}});
  requireBooleans(text, "text config",
                  {{"attention_bias", false},
                   {"tie_word_embeddings", false}});
  if (statesAttentionGate) requireBooleans(text, "text config", {{"attn_output_gate", true}});
  // The gate of the GDN's gated norm, where a config names it.
  if (text[@"output_gate_type"])
    requireEqual(requireString(text, @"output_gate_type", "text config output_gate_type"),
                 target.gdnSigmoidGate ? "sigmoid" : "swish", name, "text config output_gate_type");
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

// The whole numbers of a JSON array, which must be `expected`, in order.
void requireWholes(NSArray *values, std::span<const uint32_t> expected, std::string_view source,
                   const std::string &label) {
  std::vector<uint32_t> found;
  for (id value in values) found.push_back(requireWhole(value, label + " " + std::to_string(found.size())));
  if (std::ranges::equal(found, expected)) return;
  const auto text = [](std::span<const uint32_t> numbers) {
    std::string list = "[";
    for (const uint32_t number : numbers) list += (list.size() > 1 ? ", " : "") + std::to_string(number);
    return list + "]";
  };
  throw std::invalid_argument(label + " mismatch: " + std::string(source) + " " + text(found) + ", runtime " +
                              text(expected));
}

// Qwen3.8-Flash-Next's text configuration: the hybrid's (validateTextConfig
// above) but attn_output_gate, which it leaves unstated (its attention is
// gated: Qwen4ExpLayout's packedFullWidth), then, where a safetensors config
// states them (a GGUF's metadata
// does, which its planner checks), its hyper-connections, its QSA indexer
// (its heads, head width, one key head, the tokens of a block and the tokens
// its blocks select), its PLE n-gram embedding (n-grams, heads, convolution,
// width, its layer by its 1-based id), and the end-of-sequence token its
// hash reads (Qwen4ExpLayout::pleEosToken).
void validateTextConfig(NSDictionary *text, const Qwen4ExpLayout &target, TargetSource source) {
  validateTextConfig(text, static_cast<const QwenHybridLayout &>(target), source, false);
  if (source != TargetSource::Safetensors) return;
  requireNumbers(text, sourceName(source), "text config",
                 {{"hc_count", target.hyperConnections},
                  {"hc_lowrank", target.hyperRank},
                  {"indexer_n_heads", target.indexerHeads},
                  {"indexer_head_dim", target.indexerHeadDimension},
                  {"indexer_kv_heads", 1},
                  {"indexer_compress_ratio", target.indexerBlockTokens},
                  {"indexer_budget", target.indexerTokens()},
                  {"ngram_size", target.pleNgram},
                  {"heads_per_ngram", target.pleHeadsPerNgram},
                  {"ple_conv_kernel_size", target.pleConvolutionTaps},
                  {"ple_embed_dim", target.pleWidth()},
                  {"eos_token_id", target.pleEosToken}});
  const std::array<uint32_t, 1> pleLayers{target.pleLayer + 1};
  requireWholes(requireArray(text, @"ple_layer_ids", "text config ple_layer_ids"), pleLayers, sourceName(source),
                "text config ple_layer_ids");
}

// The layers of `mask` among `layers`, a bit each, in order.
std::vector<uint32_t> layersOf(uint64_t mask, uint32_t layers) {
  std::vector<uint32_t> result;
  for (uint32_t layer = 0; layer < layers && layer < 64; ++layer)
    if (mask >> layer & 1) result.push_back(layer);
  return result;
}

// Flint's text configuration, its config.json at the top level: the sizes
// every family's holds (sizeDifference), then the rest as the layout's
// values make them: its dense, routed and shared FFN widths, the experts a
// token takes and how its router weighs them, its leading dense layers
// (mlp_only_layers), its window, the layers that slide (layer_types) and
// those without RoPE (no_rope_layer_indices), and its stop tokens. A
// config the installer derives from a GGUF states those, as the header
// holds them; where a config states them too (the release's config.json
// does), its learned sinks, RoPE base, RMS epsilon, activation, bias-free
// attention and untied head.
void validateTextConfig(NSDictionary *text, const FlintLayout &target, TargetSource source) {
  const std::string_view name = sourceName(source);
  if (const auto difference = sizeDifference(text, target, source)) throw unsupportedModel(difference->difference);
  requireNumbers(text, name, "text config",
                 {{"intermediate_size", target.intermediateSize},
                  {"moe_intermediate_size", target.expertIntermediateSize},
                  {"shared_expert_intermediate_size", target.sharedExpertSlices * target.expertIntermediateSize},
                  {"num_experts_per_tok", target.expertsPerToken},
                  {"sliding_window", target.slidingWindow}});
  const bool sigmoid = target.moeRouter == ops::MoeRouter::SigmoidNormalized;
  requireEqual(requireString(text, @"router_score_function", "text config router_score_function"),
               sigmoid ? "sigmoid" : "softmax", name, "text config router_score_function");
  requireBooleans(text, "text config", {{"norm_topk_prob", true}});
  requireLayerTypes(requireArray(text, @"layer_types", "text config layer_types"), target, name,
                    "text config layer_types");
  std::vector<uint32_t> dense(target.leadingDenseLayers);
  std::iota(dense.begin(), dense.end(), 0u);
  requireWholes(requireArray(text, @"mlp_only_layers", "text config mlp_only_layers"), dense, name,
                "text config mlp_only_layers");
  requireWholes(requireArray(text, @"no_rope_layer_indices", "text config no_rope_layer_indices"),
                layersOf(target.unrotatedLayers, target.layers), name, "text config no_rope_layer_indices");
  requireWholes(requireArray(text, @"eos_token_id", "text config eos_token_id"), target.stopTokens.tokens(), name,
                "text config eos_token_id");
  if (text[@"softmax_type"])
    requireEqual(requireString(text, @"softmax_type", "text config softmax_type"), "learnable", name,
                 "text config softmax_type");
  if (text[@"rope_theta"]) requireNumbers(text, name, "text config", {{"rope_theta", target.rotaryTheta}});
  if (text[@"rms_norm_eps"]) requireNumbers(text, name, "text config", {{"rms_norm_eps", SPLASH_RMS_EPSILON}});
  if (text[@"hidden_act"])
    requireEqual(requireString(text, @"hidden_act", "text config hidden_act"), "silu", name, "text config hidden_act");
  for (const char *key : {"attention_bias", "tie_word_embeddings"})
    if (text[@(key)]) requireBooleans(text, "text config", {{key, false}});
}

// Inkling-2-Mini's text configuration, in its config.json's text_config (its
// release's or an MLX conversion's, or the one the installer derives from its
// GGUF, whose metadata its plan checks, ggufTargetPlan): the sizes
// (sizeDifference), then the rest as the layout's values make them: its
// tokens (unpadded_vocab_size), its window and sliding layers
// (local_layer_ids), its loop (loop_config: its layers, passes and gate
// rank, the same passes for every token), and the experts a token takes and
// their width (intermediate_size). Where a config states them (the release's
// states most, the derived one those its header holds), its sliding heads,
// relative bias, clips, log scaling, value norm, convolutions, block
// residual, softcaps, router bias and scale, shared experts (none), stop
// token, RMS epsilon and bias-free attention; its clip and log-scaling alpha
// are held as the layout's F32 values, the nearest F32 of the config's.
void validateTextConfig(NSDictionary *text, const Inkling2LoopLayout &target, TargetSource source) {
  const std::string_view name = sourceName(source);
  if (const auto difference = sizeDifference(text, target, source)) throw unsupportedModel(difference->difference);
  requireNumbers(text, name, "text config",
                 {{"unpadded_vocab_size", target.vocabularyTokens},
                  {"sliding_window_size", target.slidingWindow},
                  {"num_experts_per_tok", target.expertsPerToken},
                  {"intermediate_size", target.expertIntermediateSize}});
  std::vector<uint32_t> sliding;
  for (uint32_t layer = 0; layer < target.weightLayers; ++layer)
    if (target.slides(layer)) sliding.push_back(layer);
  requireWholes(requireArray(text, @"local_layer_ids", "text config local_layer_ids"), sliding, name,
                "text config local_layer_ids");
  NSDictionary *loop = requireObject(text, @"loop_config", "text config loop_config");
  requireNumbers(loop, name, "text config loop_config",
                 {{"core_start", target.loopFirstLayer},
                  {"core_end", target.loopEndLayer},
                  {"num_passes", target.loopPasses},
                  {"gate_rank", target.loopGateRank}});
  if (loop[@"is_varloop"]) requireBooleans(loop, "text config loop_config", {{"is_varloop", false}});
  for (const ExpectedNumber &stated : {ExpectedNumber{"swa_head_dim", target.attentionHeadDimension},
                                       ExpectedNumber{"swa_num_attention_heads", target.attentionQueryHeads},
                                       ExpectedNumber{"swa_num_key_value_heads", target.attentionKvHeads},
                                       ExpectedNumber{"d_rel", target.relativeRank},
                                       ExpectedNumber{"rel_extent", target.relativeExtent},
                                       ExpectedNumber{"rel_l2_clip_threshold", target.relativeClip},
                                       ExpectedNumber{"log_scaling_n_floor", target.logScalingFloor},
                                       ExpectedNumber{"sconv_kernel_size", target.convolutionTaps},
                                       ExpectedNumber{"n_layers_per_attn_res_block", target.blockResidualPeriod},
                                       ExpectedNumber{"mlp_gate_softcap", target.expertGateSoftcap},
                                       ExpectedNumber{"mlp_up_softcap", target.expertUpSoftcap},
                                       ExpectedNumber{"final_logit_softcapping", target.logitSoftcap},
                                       ExpectedNumber{"route_scale", target.expertRoutedScale},
                                       ExpectedNumber{"n_shared_experts", target.sharedExpertSlices},
                                       ExpectedNumber{"eos_token_id", target.stopTokens.ids[0]},
                                       ExpectedNumber{"rms_norm_eps", SPLASH_RMS_EPSILON}})
    if (text[@(stated.key)]) requireNumbers(text, name, "text config", {stated});
  for (const auto &[key, expected] : {std::pair{"qk_l2_clip_threshold", target.queryClip},
                                      std::pair{"log_scaling_alpha", target.logScalingAlpha}}) {
    if (!text[@(key)]) continue;
    const std::string label = std::string("text config ") + key;
    NSNumber *number = requireNumber(text[@(key)], label);
    if (number.floatValue != expected)
      throw std::invalid_argument(label + " mismatch: " + std::string(name) + " " + number.description.UTF8String +
                                  ", runtime " + @(expected).description.UTF8String);
  }
  for (const auto &[key, expected] : {std::pair{"use_value_norm_no_gamma", true},
                                      std::pair{"attn_res_normalize_values", true},
                                      std::pair{"use_gate_bias", true}, std::pair{"q_bias", false},
                                      std::pair{"o_bias", false}})
    if (text[@(key)]) requireBooleans(text, "text config", {{key, expected}});
}

// The mask token a DFlash2 draft of a family whose drafts choose theirs
// names (dflash_config.mask_token_id), which must be one of the target's
// reserved tokens; readDraftLayout then holds the draft to it.
uint32_t draftMaskToken(NSDictionary *draft, const QwenTargetDimensions &target) {
  NSDictionary *flash = requireObject(draft, @"dflash_config", "draft config dflash_config");
  const uint32_t token = requireWhole(flash[@"mask_token_id"], "draft config dflash_config mask_token_id");
  if (token < target.firstDraftMaskToken || token > target.lastDraftMaskToken)
    throw std::invalid_argument("draft config dflash_config mask_token_id " + std::to_string(token) +
                                " is not one of the target's reserved tokens " +
                                std::to_string(target.firstDraftMaskToken) + " to " +
                                std::to_string(target.lastDraftMaskToken));
  return token;
}

// The routed experts of a GGUF target of `target`'s family: its header's
// expert_count, which the installer derives into the config's num_experts
// (gguf.model_config), fewer than the family's in a pruned release. A count
// is served when it is at most the family's, a shape the MoE kernels route
// (ops::MoeShape: from the experts each token takes to the wide slots), and
// whole 8-column tiles of the router's float projection (GgufFloatParams);
// every start's planner holds the header and each expert tensor to it
// (gguf::planImages). A config without it, which an earlier release
// derived, states the family's.
uint32_t ggufRoutedExperts(NSDictionary *text, const QwenHybridLayout &target) {
  if (!text[@"num_experts"]) return target.experts;
  constexpr uint32_t routerTile = 8;
  const auto served = [&](uint32_t experts) {
    ops::MoeShape shape = target.moeBlockShape();
    shape.experts = experts;
    return experts <= target.experts && experts % routerTile == 0 && shape.valid();
  };
  const uint32_t experts = requireWhole(text[@"num_experts"], "text config num_experts");
  if (served(experts)) return experts;
  // The refusal names the counts served, from the fewest to the family's.
  uint32_t fewest = routerTile;
  while (fewest < target.experts && !served(fewest)) fewest += routerTile;
  throw unsupportedModel("text config num_experts: GGUF " + std::to_string(experts) + ", " +
                         std::string(target.family) + " a multiple of " + std::to_string(routerTile) + " from " +
                         std::to_string(fewest) + " to " + std::to_string(target.experts));
}

// An MLX target's quantization, the "quantization" object of its config.json:
// the object and each module's own entry name a format the block kernels hold
// (metal/abi/QuantFormat.h), as MLX loads it: an entry's mode is MLX's default,
// affine, unless it names another. The object states its bits and group size; a
// module's entry that omits either takes its mode's default, as MLX's
// to_quantized does: affine 4 bits in groups of 64, mxfp4 4 bits and mxfp8 8
// bits in groups of 32, nvfp4 4 bits in groups of 16. The images read each
// module's format from its tensors, as MLX does (model/SafetensorsImage.hpp).
// An entry that is not an object, false as MLX writes for a module it leaves
// unquantized, names nothing to check: the images load such a projection as
// Q8_0 and any other module as they read it unquantized.
void requireMlxQuantization(NSDictionary *quantization) {
  const auto require = [](NSDictionary *entry, const std::string &label, bool module) {
    NSString *mode = entry[@"mode"] ?: @"affine";
    if (![mode isKindOfClass:[NSString class]]) throw std::invalid_argument(label + " mode must be a string");
    const bool affine = [mode isEqual:@"affine"], nvfp4 = [mode isEqual:@"nvfp4"];
    const auto number = [&](NSString *key, uint32_t modeDefault) {
      return module && !entry[key] ? modeDefault : requireWhole(entry[key], label + " " + key.UTF8String);
    };
    const uint32_t bits = number(@"bits", [mode isEqual:@"mxfp8"] ? 8 : 4),
                   group = number(@"group_size", affine ? 64 : nvfp4 ? 16 : 32);
    if (safetensors::mlxFormat(mode.UTF8String ?: "", bits, group) == GGUF_FMT_COUNT)
      throw std::invalid_argument(label + " is " + mode.UTF8String + " " + std::to_string(bits) +
                                  "-bit in groups of " + std::to_string(group) + "; MLX weights load as " +
                                  std::string(safetensors::kMlxAffineFormats) + ", or as " +
                                  std::string(safetensors::kMlxFloatFormats));
  };
  require(quantization, "quantization", false);
  // A module's entry is an object; the object's other values are its own.
  for (NSString *module in quantization)
    if ([quantization[module] isKindOfClass:[NSDictionary class]])
      require(quantization[module], "quantization " + std::string(module.UTF8String ?: ""), true);
}

// A Model Optimizer target's quantization, its config.json's
// quantization_config of quant_method "modelopt": each layer it quantizes, by
// its quantized_layers or else by its quant_algo, holds NVFP4 in groups of 16
// (NVFP4, W4A16_NVFP4) or FP8 with a scale per tensor (FP8), which the block
// kernels read weights-only; a projection it leaves unquantized loads as Q8_0
// (model/SafetensorsImage.hpp). Its activation and KV cache quantization play
// no part: the kernels read activations in bf16 and keep Splash's own KV
// formats.
void requireModelOptimizerQuantization(NSDictionary *quantization) {
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
  NSDictionary *layers = quantization[@"quantized_layers"];
  if (![layers isKindOfClass:[NSDictionary class]]) return requireAlgorithm(quantization, "quantization_config");
  for (NSString *module in layers) {
    const std::string label = "quantization_config quantized_layers " + std::string(module.UTF8String ?: "");
    if (![layers[module] isKindOfClass:[NSDictionary class]]) throw std::invalid_argument(label + " must be an object");
    requireAlgorithm(layers[module], label);
  }
}

// A compressed-tensors target's quantization, its config.json's
// quantization_config of quant_method "compressed-tensors" (llm-compressor's,
// as unsloth's NVFP4 releases are): each config group's weights are NVFP4
// (nvfp4-pack-quantized: float 4-bit by tensor_group in groups of 16) or FP8
// (float-quantized: float 8-bit by channel or tensor), symmetric and in their
// stored column order; a projection no group quantizes loads as Q8_0
// (model/SafetensorsImage.hpp). Its activation and KV cache schemes play no
// part.
void requireCompressedTensorsQuantization(NSDictionary *quantization) {
  NSDictionary *groups = quantization[@"config_groups"];
  if (![groups isKindOfClass:[NSDictionary class]] || !groups.count)
    throw std::invalid_argument("quantization_config config_groups must be a non-empty object");
  const std::string format =
      quantization[@"format"] ? requireString(quantization, @"format", "quantization_config format") : "";
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
  }
}

// A safetensors target's quantization: MLX's "quantization" object, or the
// quantization_config of Model Optimizer or compressed-tensors. A checkpoint
// with none of them holds BF16 weights, or another method's that a
// transformers quantization_config states (GPTQ, AWQ, ...).
void requireQuantization(NSDictionary *config) {
  NSDictionary *mlxQuantization = config[@"quantization"];
  if ([mlxQuantization isKindOfClass:[NSDictionary class]]) return requireMlxQuantization(mlxQuantization);
  NSDictionary *quantization = config[@"quantization_config"];
  if ([quantization isKindOfClass:[NSDictionary class]]) {
    if ([quantization[@"quant_method"] isEqual:@"modelopt"]) return requireModelOptimizerQuantization(quantization);
    if ([quantization[@"quant_method"] isEqual:@"compressed-tensors"])
      return requireCompressedTensorsQuantization(quantization);
  }
  throw std::invalid_argument("this model requires an MLX checkpoint (" + std::string(safetensors::kMlxAffineFormats) +
                              ", " + std::string(safetensors::kMlxFloatFormats) +
                              "), an NVFP4 or FP8 checkpoint of Model Optimizer or compressed-tensors, or a supported "
                              "GGUF");
}

// A DFlash2 checkpoint's config: the draft's layout. Its attention widths,
// window, convolutions and selector must be those the draft kernels are built
// for, and its block at least their rows; its hidden size, vocabulary, mask
// token, the finish of its logits and how it reads its target the target's;
// and the target layers it reads (target_layer_ids) some of the target's,
// ascending. Its layers, intermediate size and rotary base are its own. The
// target's values that say how it reads its target (num_target_layers,
// target_hidden_reduction, embedding_normalization, capture_final_norm) and
// the target's tokens (valid_vocab_size) it states or leaves to its target,
// as Inkling-2-Mini-DFlash2's release leaves three of them to its serving
// adapter: a config without one takes its target's, and one that states
// another is refused.
DFlashDraftLayout readDraftLayout(NSDictionary *draft, const QwenTargetDimensions &target) {
  NSArray *architectures =
      requireArray(draft, @"architectures", "draft architectures");
  if (architectures.count != 1 ||
      ![architectures[0] isEqual:@"DFlash2DraftModel"])
    throw std::invalid_argument("draft is not a DFlash2 model");
  requireNumbers(draft, "checkpoint", "draft config",
                 {{"hidden_size", target.hiddenSize},
                  {"vocab_size", target.vocabularySize},
                  {"num_attention_heads", SPLASH_DRAFT_QUERY_HEADS},
                  {"num_key_value_heads", SPLASH_DRAFT_KV_HEADS},
                  {"head_dim", SPLASH_DRAFT_HEAD_DIMENSION},
                  {"sliding_window", ExecutionLimits::draftContextTokens},
                  {"rms_norm_eps", SPLASH_RMS_EPSILON}});
  requireBooleans(draft, "draft config",
                  {{"is_causal", false},
                   {"attention_bias", false},
                   {"tie_word_embeddings", false}});
  // The count of target layers its ids number, where its config states it
  // (num_target_layers): its target's, so that its ids name the target's
  // layers (a draft numbered by another count, such as Inkling 2's weight
  // layers beside its 48 applications, would read other rows).
  if (draft[@"num_target_layers"])
    requireNumbers(draft, "checkpoint", "draft config", {{"num_target_layers", target.layers}});
  requireEqual(requireString(draft, @"hidden_act", "draft config hidden_act"),
               "silu", "checkpoint", "draft config hidden_act");
  NSDictionary *rope =
      requireObject(draft, @"rope_parameters", "draft config rope_parameters");
  requireEqual(requireString(rope, @"rope_type",
                             "draft config rope_parameters rope_type"),
               "default", "checkpoint", "draft config rope_parameters rope_type");
  NSDictionary *flash =
      requireObject(draft, @"dflash_config", "draft config dflash_config");
  requireNumbers(flash, "checkpoint", "draft config dflash_config",
                 {{"conv_group_size", SPLASH_DRAFT_CONVOLUTION_GROUP},
                  {"conv_kernel_size", SPLASH_DRAFT_CONVOLUTION_TAPS},
                  {"selector_rank", SPLASH_DRAFT_SELECTOR_RANK},
                  {"selector_top_k", SPLASH_DRAFT_CANDIDATES},
                  {"mask_token_id", target.maskToken}});
  // Scales the draft kernels do not take, 1 where a config states them: its
  // logits' multiplier before any softcap (output_multiplier) and its token
  // rows' scale (input_embedding_scale).
  for (const char *scale : {"output_multiplier", "input_embedding_scale"})
    if (flash[@(scale)]) requireNumbers(flash, "checkpoint", "draft config dflash_config", {{scale, 1.0}});
  // The block it was trained to fill: the kernels' rows or more, of which a
  // cycle drafts the first (Ornith-1.5-122B's fills 16). Its weights have no
  // part of the block's length.
  if (requireWhole(flash[@"block_size"], "draft config dflash_config block_size") <
      ExecutionLimits::draftQueryRows)
    throw std::invalid_argument("draft config dflash_config block_size is less than the " +
                                std::to_string(ExecutionLimits::draftQueryRows) + " rows a cycle drafts");
  // How a target layer's output row becomes the hidden values the draft reads
  // of it, as the draft was trained (target_hidden_reduction): a
  // hyper-connected target's streams by their mean (Qwen4ExpTarget's
  // capture), another's row as it is. A config without it states its
  // target's.
  const std::string_view reduction = target.hyperConnected() ? "stream_mean" : "none";
  if (flash[@"target_hidden_reduction"])
    requireEqual(requireString(flash, @"target_hidden_reduction", "draft config target_hidden_reduction"),
                 reduction, "checkpoint", "draft config dflash_config target_hidden_reduction");
  // Its rows of its own tokens as they were made (embedding_normalization):
  // the token table's rows through their norm where its target's first layer
  // reads them so (QwenTargetDimensions::draftEmbeddingNorm), else as stored.
  if (flash[@"embedding_normalization"])
    requireEqual(requireString(flash, @"embedding_normalization", "draft config embedding_normalization"),
                 target.draftEmbeddingNorm ? "folded_rmsnorm" : "none", "checkpoint",
                 "draft config dflash_config embedding_normalization");
  // Its head's finish, its target's (QwenTargetDimensions::logitSoftcap and
  // vocabularyTokens): the softcap, which its selector's scores take after
  // their edges (selector_softcap_mode after_correction, the one placement
  // the selector runs), stated where its target has one; and its tokens
  // (valid_vocab_size).
  if (target.logitSoftcap != 0.0F || flash[@"final_logit_softcapping"]) {
    requireNumbers(flash, "checkpoint", "draft config dflash_config",
                   {{"final_logit_softcapping", target.logitSoftcap}});
    requireEqual(requireString(flash, @"selector_softcap_mode", "draft config selector_softcap_mode"),
                 "after_correction", "checkpoint", "draft config dflash_config selector_softcap_mode");
  }
  if (flash[@"valid_vocab_size"])
    requireNumbers(flash, "checkpoint", "draft config dflash_config", {{"valid_vocab_size", target.tokens()}});
  // The ids of the rows it reads (QwenTargetDimensions::draftReadsLayerInputs):
  // a target layer's output rows, or the rows passed into a layer, the id
  // past the target's layers its final norm's rows, whether it reads them
  // (capture_final_norm).
  NSArray *ids = requireArray(flash, @"target_layer_ids", "draft config target_layer_ids");
  std::vector<uint32_t> targetLayers;
  for (id value : ids) {
    targetLayers.push_back(
        requireWhole(value, "draft config target_layer_ids " + std::to_string(targetLayers.size())));
    if (targetLayers.back() >= target.draftReadableLayers())
      throw std::invalid_argument("draft config target_layer_ids names a layer past the target's");
  }
  const bool finalNorm = !targetLayers.empty() && targetLayers.back() == target.layers;
  if (flash[@"capture_final_norm"])
    requireBooleans(flash, "draft config dflash_config", {{"capture_final_norm", finalNorm}});
  const DFlashDraftLayout layout =
      draftLayout(target.hiddenSize, target.vocabularySize,
                  requireWhole(draft[@"num_hidden_layers"], "draft config num_hidden_layers"),
                  requireWhole(draft[@"intermediate_size"], "draft config intermediate_size"),
                  requireNumber(rope[@"rope_theta"], "draft config rope_parameters rope_theta").floatValue,
                  targetLayers);
  return layout;
}

// The vision tower's config, config.json's vision_config, which the
// installer derives from a GGUF's mmproj too: the tower the layout describes,
// a Qwen tower over RGB patches of two frames, without deepstack layers; or
// Inkling's (inkling2_vision, in transformers' names as its config.json
// states them: its patches' side and merge as native_patch_size and
// adapter_merge_size, its MLP as a ratio of its width), over RGB patches of
// one frame into a projector without biases, of the exact GELU transformers'
// default activation is.
void validateVisionConfig(NSDictionary *vision, const ops::VisionLayout &layout,
                          TargetSource source) {
  if (layout.tower == ops::VisionTower::Inkling) {
    requireEqual(requireString(vision, @"model_type", "vision config model_type"), "inkling2_vision",
                 sourceName(source), "vision config model_type");
    requireNumbers(vision, sourceName(source), "vision config",
                   {{"num_hidden_layers", layout.depth},
                    {"hidden_size", layout.hiddenSize},
                    {"num_attention_heads", layout.heads},
                    {"mlp_ratio", double(layout.intermediateSize) / layout.hiddenSize},
                    {"text_hidden_size", layout.outputHiddenSize},
                    {"projection_hidden_dim", layout.mergedHiddenSize},
                    {"native_patch_size", layout.patchSize},
                    {"adapter_merge_size", layout.spatialMerge},
                    {"posemb_grid_size", layout.positionGridSide},
                    {"layer_norm_eps", SPLASH_INKLING_VISION_NORM_EPSILON}});
    requireBooleans(vision, "vision config", {{"projector_bias", false}});
    if (vision[@"hidden_act"])
      requireEqual(requireString(vision, @"hidden_act", "vision config hidden_act"), "gelu", sourceName(source),
                   "vision config hidden_act");
    // Its RoPE, axial over the encoder's base, which a config states only to
    // set transformers' defaults.
    if (vision[@"rope_parameters"]) {
      NSDictionary *rope = requireObject(vision, @"rope_parameters", "vision config rope_parameters");
      if (rope[@"rope_type"])
        requireEqual(requireString(rope, @"rope_type", "vision config rope_type"), "axial", sourceName(source),
                     "vision config rope_type");
      if (rope[@"rope_theta"])
        requireNumbers(rope, sourceName(source), "vision config rope_parameters",
                       {{"rope_theta", SPLASH_INKLING_VISION_ROPE_THETA}});
    }
    return;
  }
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
// each config checked: the target's, as those formats require, its vision
// tower's, and its DFlash2 draft's, whose layout it reads (readDraftLayout),
// unless draft is nil. A GGUF MoE target has the routed experts its config
// states (ggufRoutedExperts).
ModelDescriptor describeSourceModel(std::string name, std::string_view targetFormat,
                                    std::string_view visionFormat, NSDictionary *config,
                                    NSDictionary *draft) {
  // A family whose config is flat (flatConfig) holds its text model's keys
  // in the config itself, which then names its model type.
  bool flat = false;
  forEachFamily([&](const auto &layout) {
    flat = flat || (layout.flatConfig && [config[@"model_type"] isEqual:@(std::string(layout.modelType).c_str())]);
  });
  NSDictionary *text = flat ? config : config[@"text_config"];
  if (![text isKindOfClass:[NSDictionary class]]) throw unsupportedModel("its config has no text_config");
  const auto type = requireString(text, @"model_type", "text model type");
  TargetSource targetSource;
  // "mlx-affine" names every safetensors target, as installations record it.
  if (targetFormat == "mlx-affine") targetSource = TargetSource::Safetensors;
  else if (targetFormat == "gguf") targetSource = TargetSource::Gguf;
  else throw std::invalid_argument("unsupported target source format: " + std::string(targetFormat));
  // The family of the model type whose sizes the config holds, as
  // Qwen3.6-35B-A3B's and Qwen3.5-122B-A10B's tell those two apart; of none,
  // the refusal names how it differs from the one it matches furthest.
  std::optional<TargetLayout> target;
  std::optional<SizeDifference> nearest;
  forEachFamily([&](const auto &layout) {
    if (layout.modelType != type || layout.flatConfig != flat || target) return;
    auto difference = sizeDifference(text, layout, targetSource);
    if (!difference) target = layout;
    else if (!nearest || difference->matched > nearest->matched) nearest = std::move(difference);
  });
  if (!target) throw unsupportedModel(nearest ? nearest->difference : "text model type " + type);
  VisionSource visionSource;
  if (visionFormat == "none") visionSource = VisionSource::None;
  else if (visionFormat == "safetensors") visionSource = VisionSource::Safetensors;
  else if (visionFormat == "gguf") visionSource = VisionSource::Gguf;
  else throw std::invalid_argument("unsupported vision source format: " + std::string(visionFormat));
  // Without a draft config the descriptor names the family but has no draft
  // (inspectSourceConfiguration), and is not valid; a model root always
  // holds its draft's.
  DFlashDraftLayout dflashLayout;
  std::visit([&](auto &layout) {
    validateTextConfig(text, layout, targetSource);
    if (targetSource == TargetSource::Safetensors) requireQuantization(config);
    if (targetSource == TargetSource::Gguf && layout.ffnKind == QwenFfnKind::SparseMoe)
      layout.experts = ggufRoutedExperts(text, layout);
    if (draft && layout.draftChoosesMaskToken()) layout.maskToken = draftMaskToken(draft, layout);
    if (draft) dflashLayout = readDraftLayout(draft, layout);
  }, *target);
  // The family's tower, projecting into its text model's width.
  const ops::VisionLayout vision = std::visit(
      [](const auto &layout) { return ops::visionLayout(layout.visionTower, layout.hiddenSize); }, *target);
  ModelDescriptor result =
      makeModelDescriptor(std::move(name), *target, dflashLayout, vision, targetSource, visionSource);
  if (result.hasVision())
    validateVisionConfig(requireObject(config, @"vision_config", "vision config"), result.vision,
                         result.targetSource);
  return result;
}

// The attention of a target whose full-attention layers select blocks (a
// block indexer): a verify row attends its top blocks, and a row whose
// history holds no more than those attends it all, through its page table.
// None for a target whose rows attend every token.
std::optional<SparseKvShape> sparseKvShape(const QwenHybridLayout &layout) {
  if (layout.kvLayout().reads != kv::Reads::SelectedBlocks) return std::nullopt;
  return SparseKvShape{.blockTokens = layout.indexerBlockTokens,
                       .selectedBlocks = layout.indexerTopBlocks,
                       .verifyRows = ExecutionLimits::targetVerifyRows,
                       .denseHistoryTokens = layout.indexerTokens()};
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
                               layout.nativeContextTokens, sparseKvShape(layout), vision.patchSize};
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
               layout.servedContextTokens(ropeScaling) ==
                   capabilities.maximumContextTokens &&
               // YaRN stretches the native window, at most to the family's
               // YaRN window.
               (!ropeScaling.yarn() ||
                (ropeScaling.originalContextTokens == layout.nativeContextTokens &&
                 capabilities.maximumContextTokens <= layout.yarnContextTokens)) &&
               capabilities.sparseKv == sparseKvShape(layout) &&
               (!capabilities.sparseKv || capabilities.sparseKv->blockTokens == SPLASH_ATTENTION_BLOCK_TOKENS) &&
               vision.tower == layout.visionTower && vision.outputHiddenSize == layout.hiddenSize &&
               capabilities.imagePatchPixels == vision.patchSize &&
               layout.hiddenSize == draft.hiddenSize &&
               draft.targetLayerCount && draft.targetLayerCount <= ExecutionLimits::maximumDraftTargetLayers &&
               draft.targetLayers[draft.targetLayerCount - 1] < layout.draftReadableLayers() &&
               // A family whose drafts choose the mask token holds its draft's.
               (!layout.draftChoosesMaskToken() ||
                (layout.maskToken >= layout.firstDraftMaskToken && layout.maskToken <= layout.lastDraftMaskToken)) &&
               layout.kvLayout() == targetKvLayout &&
               layout.gdnStateLayout() == stateLayout.target;
      },
      target);
}

ModelDescriptor servedFor(ModelDescriptor model, uint32_t contextTokens) {
  if (model.ropeScaling.yarn())
    throw std::invalid_argument("the target's RoPE is already scaled");
  std::visit(
      [&](const auto &layout) {
        if (contextTokens <= layout.nativeContextTokens)
          return;
        if (contextTokens > layout.yarnContextTokens)
          throw std::invalid_argument(
              std::string(layout.family) + " serves at most " +
              std::to_string(layout.maximumContextTokens()) + " tokens of context");
        model.ropeScaling = {.factor = static_cast<double>(contextTokens) / layout.nativeContextTokens,
                             .originalContextTokens = layout.nativeContextTokens};
        model.capabilities.maximumContextTokens = layout.servedContextTokens(model.ropeScaling);
      },
      model.target);
  return model;
}

std::vector<FamilyFacts> familyFacts() {
  std::vector<FamilyFacts> facts;
  forEachFamily([&](const auto &layout) {
    facts.push_back({.family = layout.family,
                     .experts = layout.ffnKind == QwenFfnKind::SparseMoe,
                     .recurrentState = layout.gdnStateLayout().layers > 0,
                     .pleTable = layout.pleNgram > 0,
                     .sparseKv = sparseKvShape(layout).has_value(),
                     .denseFfn = layout.ffnKind == QwenFfnKind::Dense,
                     .contextTokens = layout.nativeContextTokens,
                     .maximumContextTokens = layout.maximumContextTokens()});
  });
  return facts;
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
    // Every family decodes with its DFlash2 draft, in draft/.
    if (!std::filesystem::exists(root / "draft"))
      throw std::invalid_argument("the model's root holds no draft/ for its DFlash2 draft");
    NSDictionary *draft = readObject(root / "draft" / "config.json", "draft config");
    ModelDescriptor result = describeSourceModel(std::move(name), targetFormat, visionFormat, config, draft);
    result.sourceIdentity = std::move(sourceIdentity);
    if (!result.valid()) throw std::invalid_argument("incompatible target and draft model");
    return result;
  }
}

std::string_view inspectSourceConfiguration(std::string_view targetFormat, std::string_view visionFormat,
                                            const std::filesystem::path &config,
                                            const std::optional<std::filesystem::path> &ggufHeaders,
                                            const std::optional<std::filesystem::path> &draft) {
  @autoreleasepool {
    NSDictionary *target = readObject(config, "upstream model config");
    NSDictionary *checkpoint = draft ? readObject(*draft, "draft config") : nil;
    // Unnamed: the record names an installation, which this precedes.
    const ModelDescriptor descriptor = describeSourceModel({}, targetFormat, visionFormat, target, checkpoint);
    if (descriptor.targetSource == TargetSource::Gguf) {
      if (!ggufHeaders) throw std::invalid_argument("a GGUF target is checked with its headers");
      const OpenGguf headers(findTargetGgufs(*ggufHeaders), GgufContents::Headers);
      std::visit(
          [&](const auto &layout) {
            static_cast<void>(ggufTargetPlan(layout).images(headers.file(), {}));
          },
          descriptor.target);
    }
    return descriptor.family();
  }
}

} // namespace splash::model
