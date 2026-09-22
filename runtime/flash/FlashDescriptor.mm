#include "FlashDescriptor.hpp"

#import <Foundation/Foundation.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash {
namespace {

struct IntegerField final {
  const char *key;
  uint32_t FlashDescriptor::*member;
  uint32_t expected;
};

constexpr IntegerField kTextFields[] = {
    {"num_hidden_layers", &FlashDescriptor::layers, 48},
    {"hidden_size", &FlashDescriptor::hiddenSize, 2560},
    {"vocab_size", &FlashDescriptor::vocabularySize, 248320},
    {"max_position_embeddings", &FlashDescriptor::maximumContextTokens, 262144},
    {"hc_count", &FlashDescriptor::hcCount, 4},
    {"hc_lowrank", &FlashDescriptor::hcLowRank, 320},
    {"num_experts", &FlashDescriptor::experts, 512},
    {"num_experts_per_tok", &FlashDescriptor::expertsPerToken, 10},
    {"moe_intermediate_size", &FlashDescriptor::expertIntermediateSize, 640},
    {"shared_expert_intermediate_size", &FlashDescriptor::sharedIntermediateSize,
     640},
    {"linear_num_key_heads", &FlashDescriptor::linearKeyHeads, 16},
    {"linear_num_value_heads", &FlashDescriptor::linearValueHeads, 48},
    {"linear_key_head_dim", &FlashDescriptor::linearKeyDimension, 128},
    {"linear_value_head_dim", &FlashDescriptor::linearValueDimension, 128},
    {"linear_conv_kernel_dim", &FlashDescriptor::linearConvolutionTaps, 4},
    {"num_attention_heads", &FlashDescriptor::attentionHeads, 24},
    {"num_key_value_heads", &FlashDescriptor::attentionKvHeads, 2},
    {"head_dim", &FlashDescriptor::attentionHeadDimension, 256},
    {"indexer_n_heads", &FlashDescriptor::indexerHeads, 4},
    {"indexer_kv_heads", &FlashDescriptor::indexerKvHeads, 1},
    {"indexer_head_dim", &FlashDescriptor::indexerHeadDimension, 128},
    {"indexer_budget", &FlashDescriptor::indexerBudget, 2048},
    {"indexer_compress_ratio", &FlashDescriptor::indexerCompression, 4},
    {"ple_embed_dim", &FlashDescriptor::pleEmbeddingSize, 2560},
    {"ngram_size", &FlashDescriptor::pleNgramSize, 3},
    {"heads_per_ngram", &FlashDescriptor::pleHeadsPerNgram, 8},
    {"ngram_vocab_size_base", &FlashDescriptor::pleVocabularyBase, 20000000},
    {"make_ngram_vocab_size_divisible_by",
     &FlashDescriptor::pleVocabularyAlignment, 128},
    {"split_ngram_parts", &FlashDescriptor::pleParts, 128},
    {"ple_conv_kernel_size", &FlashDescriptor::pleConvolutionTaps, 4},
    {"eos_token_id", &FlashDescriptor::pleHistoryEos, 248044},
    {"mtp_num_hidden_layers", &FlashDescriptor::mtpLayers, 1},
};

constexpr IntegerField kVisionFields[] = {
    {"depth", &FlashDescriptor::visionLayers, 27},
    {"hidden_size", &FlashDescriptor::visionHiddenSize, 1152},
    {"intermediate_size", &FlashDescriptor::visionIntermediateSize, 4304},
    {"num_heads", &FlashDescriptor::visionHeads, 16},
    {"patch_size", &FlashDescriptor::visionPatchSize, 16},
    {"temporal_patch_size", &FlashDescriptor::visionTemporalPatchSize, 2},
    {"spatial_merge_size", &FlashDescriptor::visionSpatialMerge, 2},
    {"num_position_embeddings", &FlashDescriptor::visionPositionCount, 2304},
};

constexpr IntegerField kRootFields[] = {
    {"image_token_id", &FlashDescriptor::imageToken, 248056},
    {"video_token_id", &FlashDescriptor::videoToken, 248057},
    {"vision_start_token_id", &FlashDescriptor::visionStartToken, 248053},
    {"vision_end_token_id", &FlashDescriptor::visionEndToken, 248054},
};

[[noreturn]] void fail(std::string_view label, std::string_view reason) {
  throw std::invalid_argument("unsupported Flash config: " + std::string(label) +
                              " " + std::string(reason));
}

NSString *nativeKey(const char *key) {
  return [NSString stringWithUTF8String:key];
}

NSDictionary *readObject(const std::filesystem::path &path) {
  NSString *nativePath = [NSString stringWithUTF8String:path.c_str()];
  if (!nativePath)
    fail("path", "is not representable as UTF-8");
  NSError *error = nil;
  NSData *data = [NSData dataWithContentsOfFile:nativePath options:0 error:&error];
  if (!data) {
    const char *description = error.localizedDescription.UTF8String;
    throw std::invalid_argument("could not read Flash config: " +
                                std::string(description ? description
                                                        : "unknown read error"));
  }
  id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  if (![value isKindOfClass:[NSDictionary class]]) {
    const char *description = error.localizedDescription.UTF8String;
    throw std::invalid_argument("could not parse Flash config: " +
                                std::string(description ? description
                                                        : "expected an object"));
  }
  return static_cast<NSDictionary *>(value);
}

NSDictionary *object(id value, std::string_view label) {
  if (![value isKindOfClass:[NSDictionary class]])
    fail(label, "must be an object");
  return static_cast<NSDictionary *>(value);
}

NSArray *array(id value, std::string_view label) {
  if (![value isKindOfClass:[NSArray class]])
    fail(label, "must be an array");
  return static_cast<NSArray *>(value);
}

NSNumber *number(id value, std::string_view label) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
    fail(label, "must be a number");
  return static_cast<NSNumber *>(value);
}

uint32_t integer(id value, std::string_view label) {
  NSNumber *numeric = number(value, label);
  const int64_t signedValue = numeric.longLongValue;
  if (CFNumberIsFloatType((__bridge CFNumberRef)numeric) || signedValue <= 0 ||
      static_cast<uint64_t>(signedValue) >
          std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(signedValue) != numeric.unsignedLongLongValue)
    fail(label, "must be a positive uint32 integer");
  return static_cast<uint32_t>(signedValue);
}

double real(id value, std::string_view label) {
  const double result = number(value, label).doubleValue;
  if (!std::isfinite(result))
    fail(label, "must be finite");
  return result;
}

void requireEqual(uint32_t actual, uint32_t expected, std::string_view label) {
  if (actual != expected)
    fail(label, "must equal " + std::to_string(expected) + ", got " +
                    std::to_string(actual));
}

void requireEqual(double actual, double expected, std::string_view label) {
  if (!std::isfinite(actual) || actual != expected)
    fail(label, "has unsupported numeric geometry");
}

void stringEqual(id value, NSString *expected, std::string_view label) {
  if (![value isKindOfClass:[NSString class]] ||
      ![static_cast<NSString *>(value) isEqualToString:expected])
    fail(label, "must equal " + std::string(expected.UTF8String));
}

void booleanEqual(id value, bool expected, std::string_view label) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() ||
      static_cast<NSNumber *>(value).boolValue != expected)
    fail(label, expected ? "must be true" : "must be false");
}

void requireNull(id value, std::string_view label) {
  if (![value isKindOfClass:[NSNull class]])
    fail(label, "must be null");
}

template <size_t N>
void readFields(FlashDescriptor &result, NSDictionary *source,
                const IntegerField (&fields)[N], std::string_view prefix) {
  for (const IntegerField &field : fields) {
    result.*field.member =
        integer(source[nativeKey(field.key)], std::string(prefix) + field.key);
  }
}

template <size_t N>
void validateFields(const FlashDescriptor &descriptor,
                    const IntegerField (&fields)[N], std::string_view prefix) {
  for (const IntegerField &field : fields)
    requireEqual(descriptor.*field.member, field.expected,
          std::string(prefix) + field.key);
}

void validateQuantization(NSDictionary *quantization) {
  stringEqual(quantization[@"mode"], @"affine", "quantization.mode");
  requireEqual(integer(quantization[@"bits"], "quantization.bits"), uint32_t{4},
        "quantization.bits");
  requireEqual(integer(quantization[@"group_size"], "quantization.group_size"),
        uint32_t{64}, "quantization.group_size");
  for (id rawKey in quantization) {
    if (![rawKey isKindOfClass:[NSString class]])
      fail("quantization key", "must be a string");
    NSString *key = static_cast<NSString *>(rawKey);
    if ([key isEqualToString:@"mode"] || [key isEqualToString:@"bits"] ||
        [key isEqualToString:@"group_size"])
      continue;
    const char *utf8 = key.UTF8String;
    if (!utf8 || !*utf8)
      fail("quantization module", "must have a nonempty UTF-8 name");
    const std::string label = "quantization." + std::string(utf8);
    NSDictionary *format = object(quantization[key], label);
    if (format.count != 3)
      fail(label, "must contain only bits, group_size, and mode");
    stringEqual(format[@"mode"], @"affine", label + ".mode");
    const uint32_t bits = integer(format[@"bits"], label + ".bits");
    const uint32_t group = integer(format[@"group_size"], label + ".group_size");
    if ((bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
        (group != 32 && group != 64 && group != 128))
      fail(label, "has unsupported affine bits or group_size");
  }
}

} // namespace

void FlashDescriptor::validate() const {
  validateFields(*this, kTextFields, "text_config.");
  validateFields(*this, kVisionFields, "vision_config.");
  validateFields(*this, kRootFields, "");
  requireEqual(rotaryDimensions, uint32_t{64}, "rotaryDimensions");
  requireEqual(rotaryTheta, 10000000.0, "rotaryTheta");
  requireEqual(normEpsilon, 1e-6, "normEpsilon");
  if (mropeSections != std::array<uint32_t, 3>{11, 11, 10})
    fail("mropeSections", "must equal [11,11,10]");
  if (stopTokens != std::array<uint32_t, 2>{248046, 248044})
    fail("stopTokens", "must equal [248046,248044]");
  if (pleLayerIndices.size() != 1 || pleLayerIndices.front() != 1)
    fail("pleLayerIndices", "must contain zero-based layer 1 only");
  for (size_t layer = 0; layer < layerKinds.size(); ++layer) {
    const FlashLayerKind expected = (layer % 4 == 3)
                                       ? FlashLayerKind::SparseAttention
                                       : FlashLayerKind::GatedDeltaNet;
    if (layerKinds[layer] != expected)
      fail("layerKinds[" + std::to_string(layer) + "]",
           "does not match the three-GDN/one-sparse-attention schedule");
  }
}

FlashDescriptor FlashDescriptor::fromConfig(const std::filesystem::path &path) {
  @autoreleasepool {
    NSDictionary *root = readObject(path);
    NSArray *architectures = array(root[@"architectures"], "architectures");
    if (architectures.count != 1)
      fail("architectures", "must contain exactly one architecture");
    stringEqual(architectures[0], @"Qwen4ExpForConditionalGeneration",
                "architectures[0]");
    stringEqual(root[@"model_type"], @"qwen4_exp", "model_type");
    booleanEqual(root[@"language_model_only"], false, "language_model_only");
    booleanEqual(root[@"tie_word_embeddings"], false, "tie_word_embeddings");

    NSDictionary *text = object(root[@"text_config"], "text_config");
    NSDictionary *vision = object(root[@"vision_config"], "vision_config");
    FlashDescriptor result;
    readFields(result, text, kTextFields, "text_config.");
    readFields(result, vision, kVisionFields, "vision_config.");
    readFields(result, root, kRootFields, "");
    stringEqual(text[@"model_type"], @"qwen4_exp_text", "text_config.model_type");
    stringEqual(text[@"dtype"], @"bfloat16", "text_config.dtype");
    stringEqual(text[@"mamba_ssm_dtype"], @"float32", "text_config.mamba_ssm_dtype");
    stringEqual(text[@"hidden_act"], @"silu", "text_config.hidden_act");
    stringEqual(text[@"output_gate_type"], @"sigmoid", "text_config.output_gate_type");
    booleanEqual(text[@"attention_bias"], false, "text_config.attention_bias");
    booleanEqual(text[@"tie_word_embeddings"], false, "text_config.tie_word_embeddings");
    booleanEqual(text[@"use_cache"], true, "text_config.use_cache");
    booleanEqual(text[@"output_router_logits"], false, "text_config.output_router_logits");
    requireNull(text[@"pad_token_id"], "text_config.pad_token_id");
    requireEqual(real(text[@"attention_dropout"], "text_config.attention_dropout"),
          0.0, "text_config.attention_dropout");
    requireEqual(integer(text[@"full_attention_interval"], "text_config.full_attention_interval"),
          uint32_t{4}, "text_config.full_attention_interval");
    requireEqual(integer(text[@"bos_token_id"], "text_config.bos_token_id"),
          uint32_t{248044}, "text_config.bos_token_id");
    result.normEpsilon = real(text[@"rms_norm_eps"], "text_config.rms_norm_eps");

    NSArray *types = array(text[@"layer_types"], "text_config.layer_types");
    if (types.count != result.layerKinds.size())
      fail("text_config.layer_types", "must contain 48 entries");
    for (size_t layer = 0; layer < result.layerKinds.size(); ++layer) {
      id type = types[layer];
      if ([type isKindOfClass:[NSString class]] &&
          [static_cast<NSString *>(type) isEqualToString:@"linear_attention"])
        result.layerKinds[layer] = FlashLayerKind::GatedDeltaNet;
      else if ([type isKindOfClass:[NSString class]] &&
               [static_cast<NSString *>(type) isEqualToString:@"full_attention"])
        result.layerKinds[layer] = FlashLayerKind::SparseAttention;
      else
        fail("text_config.layer_types[" + std::to_string(layer) + "]",
             "has an unsupported layer type");
    }

    NSDictionary *rope = object(text[@"rope_parameters"], "text_config.rope_parameters");
    stringEqual(rope[@"type"], @"default", "rope_parameters.type");
    booleanEqual(rope[@"mrope_interleaved"], true, "rope_parameters.mrope_interleaved");
    requireEqual(real(text[@"partial_rotary_factor"], "text_config.partial_rotary_factor"),
          0.25, "text_config.partial_rotary_factor");
    requireEqual(real(rope[@"partial_rotary_factor"], "rope_parameters.partial_rotary_factor"),
          0.25, "rope_parameters.partial_rotary_factor");
    if (result.attentionHeadDimension % 4 != 0)
      fail("text_config.head_dim", "must support an integral rotary dimension");
    result.rotaryDimensions = result.attentionHeadDimension / 4;
    result.rotaryTheta = real(rope[@"rope_theta"], "rope_parameters.rope_theta");
    NSArray *sections = array(rope[@"mrope_section"], "rope_parameters.mrope_section");
    if (sections.count != result.mropeSections.size())
      fail("rope_parameters.mrope_section", "must contain three entries");
    for (size_t i = 0; i < result.mropeSections.size(); ++i)
      result.mropeSections[i] = integer(sections[i], "rope_parameters.mrope_section");

    NSArray *pleLayers = array(text[@"ple_layer_ids"], "text_config.ple_layer_ids");
    if (pleLayers.count != 1)
      fail("text_config.ple_layer_ids", "must contain one layer");
    // Qwen4 stores PLe IDs as one-based positions; tensor names use layer 1.
    result.pleLayerIndices = {integer(pleLayers[0], "text_config.ple_layer_ids[0]") - 1};

    NSDictionary *mtp = object(text[@"mtp"], "text_config.mtp");
    requireEqual(integer(mtp[@"num_hidden_layers"], "mtp.num_hidden_layers"),
          result.mtpLayers, "mtp.num_hidden_layers");
    booleanEqual(mtp[@"hybrid"], true, "mtp.hybrid");
    booleanEqual(text[@"mtp_use_dedicated_embeddings"], false,
                 "text_config.mtp_use_dedicated_embeddings");
    requireNull(mtp[@"mtp_use_hidden_state_from_layer"], "mtp.mtp_use_hidden_state_from_layer");
    requireEqual(real(mtp[@"rope_theta"], "mtp.rope_theta"), 10000000.0, "mtp.rope_theta");
    NSArray *mtpTypes = array(mtp[@"layer_types"], "mtp.layer_types");
    if (mtpTypes.count != 1)
      fail("mtp.layer_types", "must contain one full-attention layer");
    stringEqual(mtpTypes[0], @"full_attention", "mtp.layer_types[0]");

    stringEqual(vision[@"model_type"], @"qwen4_exp", "vision_config.model_type");
    stringEqual(vision[@"hidden_act"], @"gelu_pytorch_tanh", "vision_config.hidden_act");
    requireEqual(integer(vision[@"in_channels"], "vision_config.in_channels"),
          uint32_t{3}, "vision_config.in_channels");
    requireEqual(integer(vision[@"out_hidden_size"], "vision_config.out_hidden_size"),
          result.hiddenSize, "vision_config.out_hidden_size");
    if (array(vision[@"deepstack_visual_indexes"], "vision_config.deepstack_visual_indexes").count != 0)
      fail("vision_config.deepstack_visual_indexes", "must be empty");

    NSArray *stopTokens = array(root[@"eos_token_id"], "eos_token_id");
    if (stopTokens.count != result.stopTokens.size())
      fail("eos_token_id", "must contain two stop tokens");
    for (size_t i = 0; i < result.stopTokens.size(); ++i)
      result.stopTokens[i] = integer(stopTokens[i], "eos_token_id");

    id first = root[@"quantization_config"];
    id second = root[@"quantization"];
    if (!first && !second)
      fail("quantization", "must be present");
    if (first)
      validateQuantization(object(first, "quantization_config"));
    if (second)
      validateQuantization(object(second, "quantization"));
    if (first && second && ![first isEqual:second])
      fail("quantization", "must equal quantization_config");

    result.validate();
    return result;
  }
}

} // namespace splash::flash
