// The Omni engine's side of the speech request (protocol version 11): the
// frame server/audio_protocol.py writes (dev/tests/server/test_audio_protocol.py,
// SPEECH_FRAME) decodes field by field, the Codes event has its layout, the
// configuration's tts_model_type and speakers are read and checked, each
// model type's talker prompt (ttsPrompt) has mlx-audio's layout (qwen3_tts.py,
// _prepare_generation_inputs), a continuation's prompt replays its
// reference's frames, and a DFlash draft's config is read in both formats.

#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniProtocol.hpp"
#include "omni/TtsEngine.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace splash::omni;
using splash::test::rejects;
using splash::test::require;

namespace {

void put32(std::vector<uint8_t> &bytes, uint32_t value) {
  for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

void putFloat(std::vector<uint8_t> &bytes, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, 4);
  put32(bytes, bits);
}

// The golden speech request's payload: text (0, 1, 42, 0xFFFFFFFF), the
// instruction 151644, 872, preset speaker 3065, one reference frame.
std::vector<uint8_t> speechPayload(uint32_t instructionCount = 2, uint32_t flags = 1) {
  std::vector<uint8_t> bytes;
  put32(bytes, 0x89ABCDEF);
  put32(bytes, 0x01234567);
  put32(bytes, flags); // flags: the codes
  put32(bytes, 2050); // language id
  putFloat(bytes, 0.9F);
  put32(bytes, 50);
  putFloat(bytes, 0.95F);
  putFloat(bytes, 1.05F);
  putFloat(bytes, 0.8F);
  put32(bytes, 40);
  putFloat(bytes, 1.0F);
  put32(bytes, 0x76543210);
  put32(bytes, 0xFEDCBA98);
  put32(bytes, 100); // max frames
  put32(bytes, 25);  // chunk frames
  put32(bytes, 4);   // text count
  put32(bytes, 1);   // reference text count
  put32(bytes, 1);   // reference frame count
  put32(bytes, 0);   // speaker present
  put32(bytes, instructionCount);
  put32(bytes, 3065); // preset speaker
  for (uint32_t token : {0u, 1u, 42u, 0xFFFFFFFFu}) put32(bytes, token);
  for (uint32_t i = 0; i < instructionCount; ++i) put32(bytes, i < 2 ? (i ? 872u : 151644u) : 7u);
  put32(bytes, 7);
  for (uint32_t code = 0; code < 16; ++code) put32(bytes, code);
  return bytes;
}

void checkSpeechRequest() {
  const std::vector<uint8_t> payload = speechPayload();
  require(payload.size() == 176, "the golden payload is 176 bytes");
  const protocol::SpeechRequestFrame f = protocol::decodeSpeechRequest(payload);
  require(f.requestId == 0x0123456789ABCDEFull && f.languageId == 2050 && f.topK == 50 && f.predictorTopK == 40,
          "speech request header fields");
  require(f.seed == 0xFEDCBA9876543210ull && f.maxFrames == 100 && f.chunkFrames == 25, "speech request limits");
  require(f.flags == protocol::kSpeechFlagCodes, "speech request flags");
  require(f.text == std::vector<uint32_t>({0, 1, 42, 0xFFFFFFFF}), "text tokens");
  require(f.instruction == std::vector<uint32_t>({151644, 872}), "instruction tokens");
  require(f.presetSpeaker == 3065, "preset speaker");
  require(f.referenceText == std::vector<uint32_t>({7}) && f.referenceCodes.size() == 16 && f.speaker.empty(),
          "reference fields");
  rejects([] { (void)protocol::decodeSpeechRequest(speechPayload(513)); }, "the instruction exceeds 512 tokens",
          "a 513-token instruction accepted");
  rejects([] { (void)protocol::decodeSpeechRequest(speechPayload(2, 2)); }, "reserved bits",
          "a reserved flag accepted");
  std::vector<uint8_t> trailing = payload;
  trailing.push_back(0);
  rejects([&] { (void)protocol::decodeSpeechRequest(trailing); }, "trailing bytes", "trailing bytes accepted");
  std::vector<uint8_t> header(protocol::kHeaderBytes);
  protocol::encodeHeader(protocol::FrameType::SpeechRequest, 176, std::span<uint8_t, protocol::kHeaderBytes>(header));
  require(header[4] == 11 && header[5] == 0, "the header names version 11");
  require(protocol::decodeHeader(std::span<const uint8_t, protocol::kHeaderBytes>(header)).length == 176, "header length");
  header[4] = 10;
  rejects([&] { (void)protocol::decodeHeader(std::span<const uint8_t, protocol::kHeaderBytes>(header)); },
          "protocol version 10, expected 11", "version 10 accepted");
}

// The Codes event: u64 request id, u32 frames, then 16 u32 codes a frame.
void checkCodesFrame() {
  std::vector<uint32_t> codes(32);
  for (uint32_t i = 0; i < 32; ++i) codes[i] = 2047 - i;
  const std::vector<uint8_t> frame = protocol::codesFrame(0x0123456789ABCDEFull, codes);
  require(frame.size() == protocol::kHeaderBytes + 12 + 128, "the codes frame's length");
  const protocol::FrameHeader header =
      protocol::decodeHeader(std::span<const uint8_t, protocol::kHeaderBytes>(frame.data(), protocol::kHeaderBytes));
  require(header.type == protocol::FrameType::Codes && header.length == 140, "the codes frame's header");
  uint64_t id = 0;
  uint32_t frames = 0, last = 0;
  std::memcpy(&id, frame.data() + 16, 8);
  std::memcpy(&frames, frame.data() + 24, 4);
  std::memcpy(&last, frame.data() + frame.size() - 4, 4);
  require(id == 0x0123456789ABCDEFull && frames == 2 && last == 2047 - 31, "the codes frame's fields");
}

// A Qwen3-TTS-12Hz-1.7B config.json of the given type and extra talker fields.
std::string ttsConfig(const std::string &type, const std::string &talkerExtra) {
  return R"({"model_type":"qwen3_tts",)" + type +
         R"("im_start_token_id":151644,"im_end_token_id":151645,"assistant_token_id":77091,
  "tts_pad_token_id":151671,"tts_bos_token_id":151672,"tts_eos_token_id":151673,
  "talker_config":{"num_hidden_layers":28,"hidden_size":2048,"intermediate_size":6144,
    "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"rms_norm_eps":1e-6,
    "rope_theta":1000000,"vocab_size":3072,"num_code_groups":16,"text_vocab_size":151936,
    "text_hidden_size":2048,"codec_pad_id":2148,"codec_bos_id":2149,"codec_eos_token_id":2150,
    "codec_think_id":2154,"codec_nothink_id":2155,"codec_think_bos_id":2156,"codec_think_eos_id":2157,
    "codec_language_id":{"english":2050,"chinese":2055,"beijing_dialect":2074},)" +
         talkerExtra +
         R"("code_predictor_config":{"num_hidden_layers":5,"hidden_size":1024,"intermediate_size":3072,
      "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"rms_norm_eps":1e-6,
      "rope_theta":1000000,"vocab_size":2048}}})";
}

OmniConfig readConfig(const std::filesystem::path &root, const std::string &json) {
  std::ofstream(root / "config.json") << json;
  return readOmniConfig(root);
}

void checkModelTypes(const std::filesystem::path &root) {
  const std::string speakers = R"("spk_id":{"vivian":3065,"dylan":2878},)";
  require(readConfig(root, ttsConfig("", R"("spk_id":{},)")).tts->type == TtsModelType::Base,
          "a config without tts_model_type is a Base model");
  const OmniConfig base = readConfig(root, ttsConfig(R"("tts_model_type":"base",)", R"("spk_id":{},)"));
  require(base.tts->type == TtsModelType::Base && base.tts->speakers.empty(), "a Base model has no speakers");
  require(readConfig(root, ttsConfig(R"("tts_model_type":"voice_design",)", "")).tts->type == TtsModelType::VoiceDesign,
          "voice_design");
  const OmniConfig custom = readConfig(root, ttsConfig(R"("tts_model_type":"custom_voice",)", speakers));
  require(custom.tts->type == TtsModelType::CustomVoice &&
              custom.tts->speakers == std::map<std::string, uint32_t>({{"dylan", 2878}, {"vivian", 3065}}),
          "a CustomVoice model's speakers");
  rejects([&] { (void)readConfig(root, ttsConfig(R"("tts_model_type":"voice_clone",)", "")); },
          "tts_model_type voice_clone is not a supported model", "an unknown model type accepted");
  rejects([&] { (void)readConfig(root, ttsConfig(R"("tts_model_type":"custom_voice",)", R"("spk_id":{},)")); },
          "has no speakers", "a CustomVoice model without speakers accepted");
  rejects([&] {
    (void)readConfig(root, ttsConfig(R"("tts_model_type":"custom_voice",)", R"("spk_id":{"giant":4000},)"));
  }, "outside the codec vocabulary", "a speaker outside the codec vocabulary accepted");
}

// A DFlash draft's config.json in both checkpoint formats, read against
// Qwen3-ASR-1.7B's decoder: the layers it reads (outputs; Speculators names
// inputs), its block, mask token, causality and vocabulary, and what is
// refused.
std::string layers(const std::string &extra) {
  return R"("num_hidden_layers":5,"hidden_size":2048,"intermediate_size":6144,"num_attention_heads":16,)"
         R"("num_key_value_heads":8,"head_dim":128,"rms_norm_eps":1e-6,"hidden_act":"silu",)" +
         extra;
}
std::string zlab(const std::string &flash, const std::string &extra = "") {
  return R"({"architectures":["DFlashDraftModel"],"model_type":"qwen3",)" +
         layers(R"("rope_theta":1000000.0,"vocab_size":151936,"block_size":16,)" + extra) + R"("dflash_config":)" +
         flash + "}";
}
std::string speculators(const std::string &extra) {
  return R"({"architectures":["DFlashDraftModel"],"speculators_model_type":"dflash","block_size":8,)"
         R"("mask_token_id":151705,"draft_vocab_size":48000,"sample_from_anchor":false,)" +
         extra + R"("transformer_layer_config":{)" +
         layers(R"("rope_parameters":{"rope_theta":1000000},"vocab_size":151936,"sliding_window":32768,)"
                R"("use_sliding_window":true})") +
         "}";
}

void checkDFlashConfigs(const std::filesystem::path &root) {
  AsrConfig target;
  target.decoder.layers = 28;
  target.decoder.hidden = 2048;
  target.decoder.intermediate = 6144;
  const auto read = [&](const std::string &json) {
    std::ofstream(root / "config.json") << json;
    return readDFlashDraftConfig(root, target);
  };
  const std::string flash = R"({"mask_token_id":151705,"target_layer_ids":[1,7,13,19,25]})";
  std::ofstream(root / "config.json") << zlab(flash);
  require(isDFlashDraft(root), "a z-lab DFlash config is a DFlash draft");
  const DFlashDraftConfig z = read(zlab(flash));
  require(z.targetLayers == std::vector<uint32_t>{1, 7, 13, 19, 25} && z.blockSize == 16 && z.maskToken == 151705 &&
              !z.causal && !z.draftVocabulary && z.decoder.layers == 5 && z.decoder.ropeTheta == 1e6,
          "a z-lab DFlash draft's config");
  const DFlashDraftConfig w = read(speculators(R"("aux_hidden_state_layer_ids":[2,14,25],"sliding_window_non_causal":false,)"));
  require(w.targetLayers == std::vector<uint32_t>{1, 13, 24} && w.blockSize == 8 && w.causal &&
              w.draftVocabulary == 48000 && w.decoder.ropeTheta == 1e6,
          "a Speculators DFlash draft's config: input layer ids, a causal block and its own vocabulary");
  require(isDFlashDraft(root), "a Speculators DFlash config is a DFlash draft");
  require(!read(speculators(R"("aux_hidden_state_layer_ids":[2],"sliding_window_non_causal":true,)")).causal,
          "a non-causal Speculators block");
  // Full attention layers attend bidirectionally (unless is_causal says
  // otherwise); a mix of layer types is refused.
  const auto withLayers = [&](const std::string &fields) {
    std::string json = speculators(R"("aux_hidden_state_layer_ids":[2],)");
    json.insert(json.find(R"("num_hidden_layers")"), fields);
    return read(json);
  };
  require(!withLayers(R"("layer_types":["full_attention","full_attention"],"use_sliding_window":false,)").causal,
          "full attention Speculators layers attend bidirectionally");
  require(withLayers(R"("layer_types":["full_attention"],"is_causal":true,)").causal, "an explicit is_causal");
  rejects([&] { (void)withLayers(R"("layer_types":["full_attention","sliding_attention"],)"); }, "mixing",
          "mixed attention layer types accepted");
  rejects([&] { (void)read(speculators(R"("aux_hidden_state_layer_ids":[0,14],)")); }, "embeddings",
          "a draft reading the embeddings accepted");
  rejects([&] { (void)read(zlab(R"({"mask_token_id":151705,"target_layer_ids":[7,1]})")); }, "ascend",
          "descending target layers accepted");
  rejects([&] { (void)read(zlab(R"({"mask_token_id":151705,"target_layer_ids":[28]})")); }, "past the target",
          "a target layer past the target's accepted");
  rejects([&] { (void)read(zlab(R"({"mask_token_id":151936,"target_layer_ids":[1]})")); }, "mask_token_id",
          "a mask token outside the vocabulary accepted");
  rejects([&] { (void)read(zlab(flash, R"("use_sliding_window":true,)")); }, "sliding window",
          "a z-lab draft with a sliding window accepted");
  std::ofstream(root / "config.json") << R"({"model_type":"qwen3_asr"})";
  require(!isDFlashDraft(root), "a Qwen3-ASR checkpoint is not a DFlash draft");
}

void checkPrompt(const TtsPrompt &prompt, const std::vector<uint32_t> &tokens,
                 const std::vector<std::pair<int32_t, int32_t>> &positions, uint32_t trailingBase,
                 const std::string &what) {
  require(prompt.tokens == tokens, what + ": text rows");
  require(prompt.positions.size() == positions.size(), what + ": prompt length");
  for (size_t i = 0; i < positions.size(); ++i) {
    const OmniPromptPosition &p = prompt.positions[i];
    require(p.text_row == positions[i].first && p.codec == positions[i].second && p.frame == -1 && p.speaker == 0,
            what + ": position " + std::to_string(i));
  }
  require(prompt.trailingBase == trailingBase, what + ": trailing rows");
}

void checkPrompts() {
  const TtsConfig c;
  const uint32_t pad = c.ttsPad, bos = c.ttsBos, eos = c.ttsEos;
  TtsRequest request;
  request.text = {9707, 11};
  // Base: role, tts_pad over [nothink, think_bos, think_eos], tts_bos over
  // codec pad, the first text token over codec bos.
  checkPrompt(ttsPrompt(c, request), {c.imStart, c.assistant, 198, pad, pad, pad, bos, 9707, 11, eos, pad},
              {{0, -1}, {1, -1}, {2, -1}, {3, 2155}, {4, 2156}, {5, 2157}, {6, 2148}, {7, 2149}}, 8, "Base");
  // VoiceDesign: the instruction's turn leads, its rows text alone.
  request.instruction = {151644, 872, 198, 32, 151645, 198};
  checkPrompt(ttsPrompt(c, request),
              {151644, 872, 198, 32, 151645, 198, c.imStart, c.assistant, 198, pad, pad, pad, bos, 9707, 11, eos, pad},
              {{0, -1}, {1, -1}, {2, -1}, {3, -1}, {4, -1}, {5, -1}, {6, -1}, {7, -1}, {8, -1},
               {9, 2155}, {10, 2156}, {11, 2157}, {12, 2148}, {13, 2149}},
              14, "VoiceDesign");
  // CustomVoice: the speaker's row follows the language's think prefix, and
  // the whole text is in the prompt (the official non-streaming layout, as
  // mlx-audio 0.6 builds it): t0, t1 and tts_eos over codec pad, tts_pad
  // over codec bos, which every frame then reads.
  TtsConfig custom;
  custom.type = TtsModelType::CustomVoice;
  request.instruction.clear();
  request.speaker = 3065;
  request.languageId = 2050;
  const TtsPrompt whole = ttsPrompt(custom, request);
  checkPrompt(whole, {c.imStart, c.assistant, 198, pad, pad, pad, pad, pad, bos, 9707, 11, eos, pad},
              {{0, -1}, {1, -1}, {2, -1}, {3, 2154}, {4, 2156}, {5, 2050}, {6, 2157}, {7, 3065}, {8, 2148},
               {9, 2148}, {10, 2148}, {11, 2148}, {12, 2149}},
              12, "CustomVoice");
  require(whole.trailingCount == 0, "CustomVoice: frames read tts_pad alone");
  // A CustomVoice continuation's reference frames read tts_pad too.
  request.referenceText = {77};
  request.referenceCodes.assign(2 * OMNI_CODE_GROUPS, 5);
  const TtsPrompt continued = ttsPrompt(custom, request);
  const size_t frames = continued.positions.size() - 2;
  require(continued.tokens.size() == whole.tokens.size() + 1 && continued.referenceFrames == 2 &&
              continued.positions[frames].text_row == int32_t(continued.trailingBase) &&
              continued.positions[frames + 1].text_row == int32_t(continued.trailingBase) &&
              continued.positions[frames].frame == 0 && continued.positions[frames + 1].frame == 1,
          "CustomVoice continuation: the reference's frames read tts_pad");
}

// A continuation speaks the reference's text then the request's, and its
// prompt ends with the reference's frames, each over the trailing row the
// step that sampled it read: t1.., then tts_eos, then tts_pad.
void checkContinuationPrompt() {
  const TtsConfig c;
  const uint32_t pad = c.ttsPad, bos = c.ttsBos, eos = c.ttsEos;
  TtsRequest request;
  request.text = {9707, 11};
  request.referenceText = {100, 101, 102};
  request.referenceCodes.assign(7 * 16, 5);
  const TtsPrompt prompt = ttsPrompt(c, request);
  require(prompt.tokens == std::vector<uint32_t>({c.imStart, c.assistant, 198, pad, pad, pad, bos, 100, 101, 102,
                                                   9707, 11, eos, pad}),
          "continuation: the reference's text leads");
  require(prompt.trailingBase == 8 && prompt.trailingCount == 5 && prompt.referenceFrames == 7,
          "continuation: trailing rows and reference frames");
  require(prompt.positions.size() == 8 + 7, "continuation: the prompt holds the reference's frames");
  for (uint32_t j = 0; j < 7; ++j) {
    const OmniPromptPosition &p = prompt.positions[8 + j];
    // Frames 0..3 over t1..t4, frame 4 over tts_eos, then tts_pad.
    require(p.text_row == int32_t(8 + std::min(j, 5u)) && p.codec == -1 && p.frame == int32_t(j) && p.speaker == 0,
            "continuation: reference frame " + std::to_string(j));
  }
  // The prompt before the frames is the plain request's for the joined text.
  TtsRequest joined;
  joined.text = {100, 101, 102, 9707, 11};
  const TtsPrompt plain = ttsPrompt(c, joined);
  require(plain.tokens == prompt.tokens && plain.positions.size() == 8, "continuation: the joined text's prompt");
}

} // namespace

int main() {
  try {
    checkSpeechRequest();
    checkCodesFrame();
    const splash::test::TemporaryDirectory directory("splash-omni-protocol");
    checkModelTypes(directory.path());
    checkDFlashConfigs(directory.path());
    checkPrompts();
    checkContinuationPrompt();
  } catch (const std::exception &error) {
    std::cerr << "omni protocol: " << error.what() << "\n";
    return 1;
  }
  std::cout << "omni protocol: all checks passed\n";
  return 0;
}
