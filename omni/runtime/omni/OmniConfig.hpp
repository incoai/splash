#pragma once

// The configurations of the Omni models, read from a checkpoint's
// config.json (OmniConfig.mm): the decoder dimensions and the token ids the
// pipelines need. Only the supported checkpoints' shapes are accepted: the
// served models (Qwen3-TTS-12Hz-1.7B Base, VoiceDesign and CustomVoice,
// Qwen3-ASR-1.7B) and Qwen3-ASR-0.6B, the ASR draft.

#include "omni/Qwen3Decoder.hpp"
#include "ops/OmniAudioEncoder.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace splash::omni {

enum class ModelKind { Tts, Asr };

struct TtsGenerationDefaults final {
  float temperature = 0.9F;
  uint32_t topK = 50;
  float topP = 1.0F;
  float repetitionPenalty = 1.05F;
  float predictorTemperature = 0.9F;
  uint32_t predictorTopK = 50;
  float predictorTopP = 1.0F;
  uint32_t maxNewTokens = 8192;
};

// The checkpoint's tts_model_type: the same talker, predictor and codec,
// prompted differently (TtsEngine.cpp, ttsPrompt). VoiceDesign takes an
// instruction describing the voice; CustomVoice a preset speaker and an
// optional instruction.
enum class TtsModelType { Base, VoiceDesign, CustomVoice };

struct TtsConfig final {
  TtsModelType type = TtsModelType::Base;
  Qwen3Dims talker;
  Qwen3Dims predictor;
  uint32_t codecVocabulary = 3072;
  uint32_t predictorVocabulary = 2048;
  uint32_t codeGroups = 16;
  uint32_t textVocabulary = 151936;
  uint32_t textHidden = 2048;
  // Whether the talker's hidden state reaches the code predictor through
  // small_to_mtp: only where the two widths differ (1.7B).
  [[nodiscard]] bool predictorProjection() const noexcept { return talker.hidden != predictor.hidden; }
  // Codec-vocabulary ids.
  uint32_t codecPad = 2148, codecBos = 2149, codecEos = 2150;
  uint32_t codecThink = 2154, codecNoThink = 2155, codecThinkBos = 2156, codecThinkEos = 2157;
  std::map<std::string, uint32_t> languages;
  // CustomVoice: the preset speakers' codec ids by name (talker_config.spk_id).
  std::map<std::string, uint32_t> speakers;
  // Text-vocabulary ids.
  uint32_t imStart = 151644, imEnd = 151645, assistant = 77091;
  uint32_t ttsPad = 151671, ttsBos = 151672, ttsEos = 151673;
  TtsGenerationDefaults generation;
};

struct AsrConfig final {
  Qwen3Dims decoder;
  // The audio tower, whose output is the decoder's width.
  ops::omni::AudioEncoderDims encoder;
  uint32_t vocabulary = 151936;
  uint32_t audioStart = 151669, audioEnd = 151670, audioPad = 151676;
  std::vector<uint32_t> stopTokens{151645, 151643};
  uint32_t imStart = 151644, imEnd = 151645;
};

struct OmniConfig final {
  ModelKind kind = ModelKind::Tts;
  // The checkpoint's projections: affine Q4 (group 64) or bf16.
  bool quantized = false;
  std::optional<TtsConfig> tts;
  std::optional<AsrConfig> asr;
};

// Reads and checks `root`/config.json. Throws std::runtime_error naming
// what does not match a supported model.
[[nodiscard]] OmniConfig readOmniConfig(const std::filesystem::path &root);

// A DFlash speculative draft of Qwen3-ASR-1.7B: Qwen3 layers of the
// target's width that read the outputs of some target layers (projected by
// fc and hidden_norm) as context keys and values, and propose blockSize - 1
// tokens from a block of [anchor, mask x (blockSize - 1)] embedded with the
// target's table. Two checkpoint formats:
//   - z-lab/dflash (jburtoft/Qwen3-ASR-1.7B-DFlash): dflash_config with
//     target_layer_ids (layer outputs), the block attends bidirectionally,
//     the target's head reads the proposals after the draft's final norm;
//   - Speculators (wilsonzheng0327/Qwen3-ASR-1.7B-dflash):
//     aux_hidden_state_layer_ids (layer inputs: id l is layer l - 1's
//     output) and transformer_layer_config, a causal block unless
//     sliding_window_non_causal, and its own lm_head over draft_vocab_size
//     tokens that d2t maps to the target's.
struct DFlashDraftConfig final {
  Qwen3Dims decoder;                  // the draft's layers (no head)
  std::vector<uint32_t> targetLayers; // 0-based target layers whose outputs it reads, ascending
  uint32_t blockSize = 0;
  uint32_t maskToken = 0;
  bool causal = false;                // the block attends causally
  uint32_t draftVocabulary = 0;       // its own head's tokens (0: the target's head)
  bool quantized = false;
};

// Whether `root`/config.json describes a DFlash draft (architectures
// DFlashDraftModel with a dflash_config).
[[nodiscard]] bool isDFlashDraft(const std::filesystem::path &root);
// Reads and checks a DFlash draft's config against its target's decoder
// (`target`: its width, layers and vocabulary). Throws std::runtime_error
// naming what does not fit.
[[nodiscard]] DFlashDraftConfig readDFlashDraftConfig(const std::filesystem::path &root, const AsrConfig &target);

} // namespace splash::omni
