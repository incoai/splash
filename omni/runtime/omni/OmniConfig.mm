#include "omni/OmniConfig.hpp"

#import <Foundation/Foundation.h>

#include <stdexcept>
#include <string>

namespace splash::omni {
namespace {

[[noreturn]] void fail(const std::string &what) { throw std::runtime_error("config.json: " + what); }

NSDictionary *object(NSDictionary *parent, const char *key, bool required = true) {
  id value = parent[@(key)];
  if (![value isKindOfClass:[NSDictionary class]]) {
    if (required) fail(std::string("missing object ") + key);
    return nil;
  }
  return value;
}

bool has(NSDictionary *parent, const char *key) { return parent[@(key)] != nil && parent[@(key)] != [NSNull null]; }

double number(NSDictionary *parent, const char *key) {
  id value = parent[@(key)];
  if (![value isKindOfClass:[NSNumber class]]) fail(std::string("missing number ") + key);
  return [value doubleValue];
}

uint32_t integer(NSDictionary *parent, const char *key) {
  const double value = number(parent, key);
  if (value < 0 || value > 4294967295.0 || value != static_cast<double>(static_cast<uint64_t>(value)))
    fail(std::string("not a whole number: ") + key);
  return static_cast<uint32_t>(value);
}

std::string string(NSDictionary *parent, const char *key) {
  id value = parent[@(key)];
  if (![value isKindOfClass:[NSString class]]) fail(std::string("missing string ") + key);
  return [value UTF8String];
}

void expect(NSDictionary *parent, const char *key, uint32_t expected) {
  if (integer(parent, key) != expected)
    fail(std::string(key) + " is " + std::to_string(integer(parent, key)) + ", expected " + std::to_string(expected));
}

bool quantization(NSDictionary *root) {
  NSDictionary *q = object(root, "quantization", false);
  if (!q) return false;
  if (integer(q, "bits") != 4 || integer(q, "group_size") != 64) fail("only affine 4-bit, group 64 quantization is supported");
  if (has(q, "mode") && string(q, "mode") != "affine") fail("only affine quantization is supported");
  return true;
}

// A Qwen3 decoder's dimensions: 16 query heads over 8 KV heads of 128.
Qwen3Dims decoderDims(NSDictionary *c, uint32_t hidden, uint32_t intermediate, uint32_t layers) {
  Qwen3Dims dims;
  dims.layers = integer(c, "num_hidden_layers");
  dims.hidden = integer(c, "hidden_size");
  dims.intermediate = integer(c, "intermediate_size");
  if (dims.layers != layers || dims.hidden != hidden || dims.intermediate != intermediate)
    fail("decoder shape " + std::to_string(dims.layers) + "x" + std::to_string(dims.hidden) + "x" +
         std::to_string(dims.intermediate) + " is not a supported model");
  expect(c, "num_attention_heads", 16);
  expect(c, "num_key_value_heads", 8);
  if (has(c, "head_dim")) expect(c, "head_dim", 128);
  dims.rmsEpsilon = static_cast<float>(number(c, "rms_norm_eps"));
  dims.ropeTheta = number(c, "rope_theta");
  if (has(c, "hidden_act") && string(c, "hidden_act") != "silu") fail("the decoder activation is not silu");
  NSDictionary *scaling = object(c, "rope_scaling", false);
  if (scaling && has(scaling, "mrope_section")) {
    // Interleaved M-RoPE with equal positions on every axis is 1-D RoPE.
    if (!has(scaling, "interleaved") || ![scaling[@"interleaved"] boolValue]) fail("only interleaved M-RoPE is supported");
  }
  return dims;
}

OmniConfig readTts(NSDictionary *root) {
  OmniConfig config;
  config.kind = ModelKind::Tts;
  config.quantized = quantization(root);
  TtsConfig tts;
  // mlx-audio's ModelConfig defaults the type to base, as the Base
  // checkpoints name it.
  const std::string type = has(root, "tts_model_type") ? string(root, "tts_model_type") : "base";
  if (type == "voice_design") tts.type = TtsModelType::VoiceDesign;
  else if (type == "custom_voice") tts.type = TtsModelType::CustomVoice;
  else if (type != "base") fail("tts_model_type " + type + " is not a supported model (base, voice_design, custom_voice)");
  NSDictionary *talker = object(root, "talker_config");
  tts.talker = decoderDims(talker, 2048, 6144, 28);
  tts.talker.vocabulary = integer(talker, "vocab_size");
  tts.talker.headRows = tts.talker.vocabulary; // 3072, whole Q4 tiles
  if (tts.talker.vocabulary != 3072) fail("talker vocabulary is not 3072");
  NSDictionary *predictor = object(talker, "code_predictor_config");
  tts.predictor = decoderDims(predictor, 1024, 3072, 5);
  tts.predictor.vocabulary = integer(predictor, "vocab_size");
  tts.predictor.headRows = tts.predictor.vocabulary;
  if (tts.predictor.vocabulary != 2048) fail("code predictor vocabulary is not 2048");
  expect(talker, "num_code_groups", 16);
  expect(talker, "text_vocab_size", 151936);
  expect(talker, "text_hidden_size", 2048);
  tts.codecPad = integer(talker, "codec_pad_id");
  tts.codecBos = integer(talker, "codec_bos_id");
  tts.codecEos = integer(talker, "codec_eos_token_id");
  tts.codecThink = integer(talker, "codec_think_id");
  tts.codecNoThink = integer(talker, "codec_nothink_id");
  tts.codecThinkBos = integer(talker, "codec_think_bos_id");
  tts.codecThinkEos = integer(talker, "codec_think_eos_id");
  NSDictionary *languages = object(talker, "codec_language_id", false);
  if (languages)
    for (NSString *name in languages) tts.languages[[name UTF8String]] = integer(languages, [name UTF8String]);
  if (tts.type == TtsModelType::CustomVoice) {
    NSDictionary *speakers = object(talker, "spk_id");
    for (NSString *name in speakers) {
      const uint32_t id = integer(speakers, [name UTF8String]);
      if (id >= tts.talker.vocabulary) fail(std::string("speaker ") + [name UTF8String] + " is outside the codec vocabulary");
      tts.speakers[[name UTF8String]] = id;
    }
    if (tts.speakers.empty()) fail("a custom_voice model has no speakers (talker_config.spk_id)");
  }
  tts.imStart = integer(root, "im_start_token_id");
  tts.imEnd = integer(root, "im_end_token_id");
  tts.assistant = integer(root, "assistant_token_id");
  tts.ttsPad = integer(root, "tts_pad_token_id");
  tts.ttsBos = integer(root, "tts_bos_token_id");
  tts.ttsEos = integer(root, "tts_eos_token_id");
  config.tts = tts;
  return config;
}

OmniConfig readAsr(NSDictionary *root) {
  OmniConfig config;
  config.kind = ModelKind::Asr;
  config.quantized = quantization(root);
  AsrConfig asr;
  NSDictionary *thinker = object(root, "thinker_config");
  NSDictionary *text = object(thinker, "text_config");
  // Qwen3-ASR-1.7B, or 0.6B (served as the 1.7B's speculative draft).
  const bool small = has(text, "hidden_size") && integer(text, "hidden_size") == 1024;
  asr.decoder = small ? decoderDims(text, 1024, 3072, 28) : decoderDims(text, 2048, 6144, 28);
  asr.vocabulary = integer(text, "vocab_size");
  if (asr.vocabulary != 151936) fail("text vocabulary is not 151936");
  asr.decoder.vocabulary = asr.vocabulary;
  // The tied embedding as a head: whole 256-row Q4 tiles when quantized.
  asr.decoder.headRows = config.quantized ? (asr.vocabulary + 255) / 256 * 256 : asr.vocabulary;
  if (has(text, "tie_word_embeddings") && ![text[@"tie_word_embeddings"] boolValue]) fail("the ASR head must be tied");
  NSDictionary *audio = object(thinker, "audio_config");
  asr.encoder = small ? ops::omni::AudioEncoderDims{896, 18, 3584, 1024} : ops::omni::AudioEncoderDims{};
  expect(audio, "d_model", asr.encoder.hidden);
  expect(audio, "encoder_layers", asr.encoder.layers);
  expect(audio, "encoder_attention_heads", asr.encoder.heads());
  expect(audio, "encoder_ffn_dim", asr.encoder.ffn);
  expect(audio, "num_mel_bins", 128);
  expect(audio, "output_dim", asr.encoder.output);
  if (has(audio, "downsample_hidden_size")) expect(audio, "downsample_hidden_size", OMNI_AUDIO_CONV_CHANNELS);
  asr.audioStart = integer(thinker, "audio_start_token_id");
  asr.audioEnd = integer(thinker, "audio_end_token_id");
  asr.audioPad = integer(thinker, "audio_token_id");
  config.asr = asr;
  return config;
}

} // namespace

namespace {

NSDictionary *readConfigJson(const std::filesystem::path &root) {
  NSError *error = nil;
  NSData *data = [NSData dataWithContentsOfFile:@((root / "config.json").c_str()) options:0 error:&error];
  if (!data) fail("cannot read " + (root / "config.json").string());
  id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  if (![parsed isKindOfClass:[NSDictionary class]]) fail("not a JSON object");
  return parsed;
}

bool speculatorsFormat(NSDictionary *root) {
  return has(root, "speculators_model_type") && [root[@"speculators_model_type"] isEqual:@"dflash"];
}

bool describesDFlash(NSDictionary *root) {
  id architectures = root[@"architectures"];
  return [architectures isKindOfClass:[NSArray class]] && [architectures count] == 1 &&
         [architectures[0] isEqual:@"DFlashDraftModel"] &&
         (object(root, "dflash_config", false) != nil || speculatorsFormat(root));
}

// Target layer ids, ascending and within the target; `inputs` when an id
// names a layer's input (Speculators), so the layer before it is read.
std::vector<uint32_t> targetLayers(id ids, bool inputs, uint32_t layers) {
  if (![ids isKindOfClass:[NSArray class]] || ![ids count] || [ids count] > 8)
    fail("a DFlash draft must name 1 to 8 target layers");
  std::vector<uint32_t> result;
  for (id value in ids) {
    if (![value isKindOfClass:[NSNumber class]]) fail("a DFlash draft's target layer ids hold a non-number");
    uint32_t layer = [value unsignedIntValue];
    if (inputs) {
      if (!layer) fail("a DFlash draft reading the target's embeddings is not supported");
      --layer;
    }
    if (layer >= layers) fail("a DFlash draft names a layer past the target's");
    if (!result.empty() && layer <= result.back()) fail("a DFlash draft's target layer ids must ascend");
    result.push_back(layer);
  }
  return result;
}

} // namespace

bool isDFlashDraft(const std::filesystem::path &root) {
  @autoreleasepool {
    if (!std::filesystem::exists(root / "config.json")) return false;
    return describesDFlash(readConfigJson(root));
  }
}

DFlashDraftConfig readDFlashDraftConfig(const std::filesystem::path &root, const AsrConfig &target) {
  @autoreleasepool {
    NSDictionary *c = readConfigJson(root);
    if (!describesDFlash(c)) fail("not a DFlash draft (architectures DFlashDraftModel)");
    const bool speculators = speculatorsFormat(c);
    // The layers' configuration: top level (z-lab) or transformer_layer_config (Speculators).
    NSDictionary *layers = speculators ? object(c, "transformer_layer_config") : c;
    DFlashDraftConfig config;
    const uint32_t count = integer(layers, "num_hidden_layers");
    if (!count || count > 16) fail("a DFlash draft takes 1 to 16 layers");
    // RoPE: the draft's own base, written as rope_theta or in rope_parameters.
    NSMutableDictionary *dims = [layers mutableCopy];
    NSDictionary *rope = object(layers, "rope_parameters", false);
    if (!has(layers, "rope_theta") && rope) dims[@"rope_theta"] = rope[@"rope_theta"];
    config.decoder = decoderDims(dims, target.decoder.hidden, target.decoder.intermediate, count);
    if (has(layers, "vocab_size") && integer(layers, "vocab_size") != target.vocabulary)
      fail("the draft's vocabulary is not the target's");
    if (speculators) {
      // Speculators: one greedy proposal method, a causal block unless the
      // window is non-causal; its trained sliding window is longer than
      // any prompt the engine takes (1200 s of audio, about 16k tokens) or
      // refused.
      config.targetLayers = targetLayers(c[@"aux_hidden_state_layer_ids"], true, target.decoder.layers);
      config.blockSize = integer(c, "block_size");
      config.maskToken = integer(c, "mask_token_id");
      // Speculators' masks: an explicit is_causal; else by layer type,
      // sliding layers causal unless sliding_window_non_causal, full
      // attention layers bidirectional. One mask for every layer.
      if (has(layers, "is_causal")) {
        config.causal = [layers[@"is_causal"] boolValue];
      } else {
        id types = layers[@"layer_types"];
        bool sliding = false, full = false;
        if ([types isKindOfClass:[NSArray class]])
          for (id type in types) {
            sliding = sliding || [type isEqual:@"sliding_attention"];
            full = full || [type isEqual:@"full_attention"];
          }
        else sliding = has(layers, "use_sliding_window") && [layers[@"use_sliding_window"] boolValue];
        if (sliding && full) fail("a DFlash draft mixing sliding and full attention layers is not supported");
        config.causal = sliding && !(has(c, "sliding_window_non_causal") && [c[@"sliding_window_non_causal"] boolValue]);
      }
      if (has(c, "sample_from_anchor") && [c[@"sample_from_anchor"] boolValue])
        fail("a DFlash draft sampling from its anchor is not supported");
      if (has(layers, "sliding_window") && [layers[@"use_sliding_window"] boolValue] &&
          integer(layers, "sliding_window") < 20000)
        fail("a DFlash draft's sliding window is shorter than the longest prompt");
      config.draftVocabulary = has(c, "draft_vocab_size") ? integer(c, "draft_vocab_size") : 0;
      if (config.draftVocabulary == target.vocabulary) config.draftVocabulary = 0;
      if (config.draftVocabulary > target.vocabulary) fail("the draft's vocabulary exceeds the target's");
    } else {
      if (has(c, "use_sliding_window") && [c[@"use_sliding_window"] boolValue])
        fail("a DFlash draft with a sliding window is not supported");
      NSDictionary *flash = object(c, "dflash_config");
      config.targetLayers = targetLayers(flash[@"target_layer_ids"], false, target.decoder.layers);
      config.blockSize = has(flash, "block_size") ? integer(flash, "block_size") : integer(c, "block_size");
      config.maskToken = integer(flash, "mask_token_id");
      config.causal = has(c, "is_causal") && [c[@"is_causal"] boolValue];
    }
    if (config.blockSize < 2 || config.blockSize > 16) fail("a DFlash draft's block_size must be 2 to 16");
    if (config.maskToken >= target.vocabulary) fail("the draft's mask_token_id is outside the target's vocabulary");
    config.quantized = quantization(c);
    return config;
  }
}

OmniConfig readOmniConfig(const std::filesystem::path &root) {
  @autoreleasepool {
    NSDictionary *dictionary = readConfigJson(root);
    const std::string type = string(dictionary, "model_type");
    if (type == "qwen3_tts") return readTts(dictionary);
    if (type == "qwen3_asr") return readAsr(dictionary);
    fail("model_type " + type + " is not a supported model (qwen3_tts, qwen3_asr)");
  }
}

} // namespace splash::omni
