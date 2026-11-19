#include "AwakeClock.hpp"
#include "omni/TtsEngine.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "ops/Omni.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace splash::omni {

using Clock = AwakeClock;

struct TtsEngine::Lane final {
  explicit Lane(KvPagePool &pool, LanePageTables &talker, LanePageTables &predictor, uint32_t index)
      : index(index), talkerKv(pool, talker, index), predictorKv(pool, predictor, index) {}
  uint32_t index;
  LaneKv talkerKv, predictorKv;
  bool active = false;
  bool hasLogits = false;      // the talker's logits of this lane are in the decode arena (or the prefill one)
  bool fromPrefill = false;    // which arena holds them
  uint32_t prefillRow = 0;     // logit row in the prefill arena
  TtsRequest request;
  uint32_t promptTokens = 0;   // talker positions prefilled
  uint32_t committed = 0;      // talker tokens in KV
  uint32_t trailingBase = 0;   // first trailing row in the lane's text region
  uint32_t trailingCount = 0;  // rows t1..tN-1, eos, then pad
  uint32_t referenceFrames = 0; // a continuation's replayed frames, before the new ones
  uint32_t minimumFrames = 0;  // EOS is masked until this many new frames
  uint32_t step = 0;           // frames generated so far
  uint32_t queuedFrames = 0;   // frames handed to the codec
  uint32_t decodedFrames = 0;  // frames turned into audio
  uint32_t sampleOffset = 0;
  // Where the codec stack's histories are: the GPU's stream, or the Neural
  // Engine lane's surfaces.
  bool stackOnAne = false;
  // The lane's Neural Engine chunk: due (its front for the next command),
  // encoded in the command being built, in flight, or failed (the GPU
  // decodes its stack from the lane's input).
  enum class AneState : uint8_t { Idle, Due, Encoded, InFlight, Failed };
  struct AneChunk final {
    AneState state = AneState::Idle;
    uint32_t first = 0, frames = 0, function = 0;
    uint64_t wait = 0, serial = 0;
    Clock::time_point queued;
  } ane;
  uint32_t maxFrames = 0;
  uint32_t chunkFrames = 25;
  uint32_t nextChunk = 4;
  std::vector<uint32_t> code0History;
  std::mt19937_64 rng;
  Clock::time_point started, prefillDone;
  std::chrono::microseconds decodeMicros{0};
  bool finished = false;
  DoneReason reason = DoneReason::Stop;
};

namespace {

constexpr uint32_t kDefaultFramesPerToken = 16;
constexpr uint32_t kMinimumDefaultFrames = 100;
// A continuation's codec stream starts from the reference's last frames, as
// if it had decoded them: their audio would come out as a whole decode's.
// The transformer's windows reach far back (its first second after 25
// frames of context is 13-18 dB from a whole decode, after 100 37-41 dB,
// after 200 61 dB), the convolution stack only a few frames (4 do), so
// the codec's front alone (transformer and upsampling, about a fifth of a
// frame's decode) takes the first frames of the context and only the last
// kReferenceStackFrames decode whole; their audio is dropped.
constexpr uint32_t kReferenceContextFrames = 100;
constexpr uint32_t kReferenceStackFrames = 8;
// A continuation masks EOS for its first frames: after the reference's
// closing pause is trimmed the talker still tends to end at once, having
// heard a whole utterance. Two frames per text token (half of normal
// speech's four) up to 40 (3.2 s), at least the usual two: a short text may
// still end early, a long one cannot end before it has begun.
uint32_t continuationMinimumFrames(size_t textTokens) {
  return static_cast<uint32_t>(std::clamp<size_t>(2 * textTokens, 2, 40));
}

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }

metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes, uint64_t &total, std::string_view label) {
  total += bytes;
  metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

template <class T> T *at(const metal::MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

} // namespace

TtsEngine::TtsEngine(metal::MetalBackend &backend, TtsModel &model, ops::omni::CodecDecoder &codec,
                     TtsEngineConfig config)
    : backend_(backend), model_(model), codec_(codec), config_(config), debug_(std::getenv("OMNI_TTS_DEBUG") != nullptr) {
  const TtsConfig &c = model.config();
  const uint32_t lanes = config.lanes;
  if (!lanes || lanes > model.maxLanes()) throw std::invalid_argument("TTS lanes exceed the model's lanes");
  if (config.textRows() > model.maxTextRows()) throw std::invalid_argument("TTS text exceeds the model's text rows");
  if (config.maxChunkFrames > codec.maximumFrames()) throw std::invalid_argument("codec chunk exceeds the decoder's frames");
  samplesPerFrame_ = codec.weights().dims.samplesPerFrame();
  textRows_ = config.textRows();
  historyStride_ = config.maxReferenceFrames + config.maxFrames;
  const uint32_t talkerBlocks =
      (textRows_ + config.maxReferenceFrames + config.maxFrames + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS;
  talkerTables_ = std::make_unique<LanePageTables>(backend, lanes, c.talker.layers, talkerBlocks);
  predictorTables_ = std::make_unique<LanePageTables>(backend, lanes, c.predictor.layers, 1);
  const Qwen3Decoder &talker = model.talker(), &predictor = model.predictor();
  talkerPrefill_ = std::make_unique<Qwen3Arena>(backend, talker.linear(), talker.weights(), SPLASH_PREFILL_TOKEN_BUDGET,
                                                lanes, lanes, false);
  talkerDecode_ = std::make_unique<Qwen3Arena>(backend, talker.linear(), talker.weights(), lanes, lanes, lanes, true);
  // Sub-step 0 runs two rows per lane as a decode step (the fused attention's
  // two-row form), so its arena is a decode arena of 2 * lanes rows.
  predictorPrefill_ = std::make_unique<Qwen3Arena>(backend, predictor.linear(), predictor.weights(), 2 * lanes, lanes,
                                                   lanes, true);
  predictorDecode_ = std::make_unique<Qwen3Arena>(backend, predictor.linear(), predictor.weights(), lanes, lanes, lanes,
                                                  true);
  memoryBytes_ += talkerPrefill_->bytes() + talkerDecode_->bytes() + predictorPrefill_->bytes() + predictorDecode_->bytes();
  for (uint32_t i = 0; i < lanes; ++i) {
    streams_.push_back(std::make_unique<ops::omni::CodecStream>(backend, codec.weights(), config.maxChunkFrames));
    memoryBytes_ += streams_.back()->bytes();
  }
  const uint32_t hidden = c.talker.hidden;
  textPool_ = zeroed(backend, uint64_t{lanes} * roundUp(textRows_, 64) * hidden * 2, memoryBytes_, "tts text pool");
  positions_ = zeroed(backend, uint64_t{SPLASH_PREFILL_TOKEN_BUDGET} * sizeof(OmniPromptPosition), memoryBytes_, "tts positions");
  stepCodes_ = zeroed(backend, uint64_t{lanes} * OMNI_CODE_GROUPS * 4, memoryBytes_, "tts step codes");
  trailingRows_ = zeroed(backend, uint64_t{lanes} * 4, memoryBytes_, "tts trailing rows");
  sampled_ = zeroed(backend, uint64_t{lanes} * OMNI_CODE_GROUPS * 4, memoryBytes_, "tts sampled");
  uniforms_ = zeroed(backend, uint64_t{lanes} * OMNI_CODE_GROUPS * 4, memoryBytes_, "tts uniforms");
  history_ = zeroed(backend, uint64_t{lanes} * historyStride_ * 4, memoryBytes_, "tts history");
  historyCounts_ = zeroed(backend, uint64_t{lanes} * 4, memoryBytes_, "tts history counts");
  predictorInput_ = zeroed(backend, uint64_t{2 * lanes + 64} * hidden * 2, memoryBytes_, "tts predictor input");
  predictorLogits_ = zeroed(backend, uint64_t{roundUp(lanes, 8)} * c.predictorVocabulary * 4, memoryBytes_, "tts predictor logits");
  rowIndex_ = zeroed(backend, uint64_t{lanes} * 4, memoryBytes_, "tts row index");
  for (uint32_t i = 0; i < lanes; ++i) at<uint32_t>(rowIndex_)[i] = i;
  codeColumn_ = zeroed(backend, uint64_t{lanes} * 4, memoryBytes_, "tts code column");
  frames_ = zeroed(backend, uint64_t{lanes} * config.maxFrames * OMNI_CODE_GROUPS * 4, memoryBytes_, "tts frames");
  references_ = zeroed(backend, uint64_t{lanes} * std::max(config.maxReferenceFrames, 1u) * OMNI_CODE_GROUPS * 4,
                       memoryBytes_, "tts reference frames");
  audio_ = zeroed(backend, uint64_t{lanes} * config.maxChunkFrames * samplesPerFrame_ * 4, memoryBytes_, "tts audio");

  // The codec's convolution stack on the Neural Engine, one function per
  // chunk size of the schedule (4, 8, 16, 25 frames).
  reports_ = std::make_shared<Reports>();
  codecPlacement_ = "codec convolutions on the GPU";
  const char *aneSwitch = std::getenv("OMNI_CODEC_ANE");
  if (aneSwitch) config_.aneCodec = std::string_view(aneSwitch) != "0";
  if (const char *disabled = std::getenv("OMNI_DISABLE_ANE"); disabled && std::string_view(disabled) == "1")
    config_.aneCodec = false;
  if (!config_.aneCodec) {
    codecPlacement_ += " (--disable-ane, OMNI_DISABLE_ANE=1 or OMNI_CODEC_ANE=0)";
  } else if (const auto reason = ane::unavailable()) {
    codecPlacement_ += " (the Neural Engine is unavailable: " + *reason + ")";
  } else {
    std::vector<uint32_t> functions;
    for (const uint32_t frames : {4u, 8u, 16u, 25u})
      if (frames <= config_.maxChunkFrames) functions.push_back(frames);
    try {
      const auto t0 = Clock::now();
      ane_ = std::make_unique<ops::omni::CodecAneStack>(backend, codec.weights(), functions, lanes,
                                                        ane::Program::Limits{std::chrono::minutes(10), {}});
      aneEvent_ = backend.newSharedEvent();
      memoryBytes_ += ane_->bytes();
      char line[160];
      std::snprintf(line, sizeof line, "codec convolutions on the Neural Engine (program ready in %.1f s; OMNI_CODEC_ANE=0 "
                    "keeps them on the GPU)", std::chrono::duration<double>(Clock::now() - t0).count());
      codecPlacement_ = line;
    } catch (const std::exception &error) {
      ane_.reset();
      codecPlacement_ += std::string(" (the Neural Engine refused the codec: ") + error.what() + ")";
    }
  }
  // The KV pool last, from what the budget leaves (kvPagesFor), and the
  // lanes over it.
  config_.kvPages = config.kvPagesFor ? config.kvPagesFor(backend.memoryStats().allocatedBytes) : config.kvPages;
  pool_ = std::make_unique<KvPagePool>(backend, config_.kvPages);
  memoryBytes_ += uint64_t{config_.kvPages} * OMNI_PAGE_BYTES;
  for (uint32_t i = 0; i < lanes; ++i)
    lanes_.push_back(std::make_unique<Lane>(*pool_, *talkerTables_, *predictorTables_, i));
}

TtsEngine::~TtsEngine() {
  // The Neural Engine's program unloads once no evaluation is in flight.
  const auto inFlight = [&] {
    for (const auto &lane : lanes_)
      if (lane->ane.state == Lane::AneState::InFlight) return true;
    return false;
  };
  const auto deadline = Clock::now() + kAneBound;
  while (ane_ && inFlight() && Clock::now() < deadline) {
    std::unique_lock lock(reports_->mutex);
    reports_->changed.wait_until(lock, deadline, [&] { return !reports_->reports.empty(); });
    for (const Reports::Report &report : std::exchange(reports_->reports, {}))
      if (lanes_[report.lane]->ane.serial == report.serial) lanes_[report.lane]->ane.state = Lane::AneState::Idle;
  }
}

bool TtsEngine::idle() const noexcept {
  return queue_.empty() && activeLanes() == 0;
}

uint32_t TtsEngine::activeLanes() const noexcept {
  uint32_t count = 0;
  for (const auto &lane : lanes_) count += lane->active;
  return count;
}

std::vector<uint32_t> TtsEngine::activeOrder() const {
  std::vector<uint32_t> order;
  for (const auto &lane : lanes_)
    if (lane->active) order.push_back(lane->index);
  return order;
}

void TtsEngine::submit(TtsRequest request, EventSink &sink) {
  if (request.text.empty() || request.text.size() > config_.maxTextTokens) {
    sink.onError(request.id, "invalid_request", "text must hold 1.." + std::to_string(config_.maxTextTokens) + " tokens");
    return;
  }
  const TtsConfig &c = model_.config();
  for (uint32_t token : request.text)
    if (token >= c.textVocabulary) {
      sink.onError(request.id, "invalid_request", "text token outside the vocabulary");
      return;
    }
  if (request.languageId && request.languageId >= c.codecVocabulary) {
    sink.onError(request.id, "invalid_request", "language id outside the codec vocabulary");
    return;
  }
  // The model type's prompt: Base takes no instruction and no speaker,
  // VoiceDesign an instruction, CustomVoice one of its speakers and an
  // instruction or none.
  if (!request.instruction.empty() && c.type == TtsModelType::Base) {
    sink.onError(request.id, "invalid_request", "a Base model takes no instruction");
    return;
  }
  if (request.instruction.size() > config_.maxInstructionTokens) {
    sink.onError(request.id, "invalid_request",
                 "the instruction must hold at most " + std::to_string(config_.maxInstructionTokens) + " tokens");
    return;
  }
  for (uint32_t token : request.instruction)
    if (token >= c.textVocabulary) {
      sink.onError(request.id, "invalid_request", "instruction token outside the vocabulary");
      return;
    }
  if (c.type == TtsModelType::VoiceDesign && request.instruction.empty()) {
    sink.onError(request.id, "invalid_request", "a VoiceDesign model takes an instruction describing the voice");
    return;
  }
  const bool presetSpeaker = std::any_of(c.speakers.begin(), c.speakers.end(),
                                         [&](const auto &speaker) { return speaker.second == request.speaker; });
  if (c.type == TtsModelType::CustomVoice ? !presetSpeaker : request.speaker != 0) {
    sink.onError(request.id, "invalid_request",
                 c.type == TtsModelType::CustomVoice ? "a CustomVoice model takes one of its preset speakers"
                                                     : "only a CustomVoice model takes a speaker");
    return;
  }
  // A continuation's reference: its text and its frames of audio codes.
  if (!request.referenceText.empty() || !request.referenceCodes.empty()) {
    const size_t frames = request.referenceCodes.size() / OMNI_CODE_GROUPS;
    if (request.referenceText.empty() || !frames || request.referenceCodes.size() % OMNI_CODE_GROUPS) {
      sink.onError(request.id, "invalid_request", "a reference takes its text and whole frames of its codes");
      return;
    }
    if (frames > config_.maxReferenceFrames) {
      sink.onError(request.id, "limit_exceeded",
                   "the reference holds " + std::to_string(frames) + " frames; a reference holds at most " +
                       std::to_string(config_.maxReferenceFrames) + " (" +
                       std::to_string(config_.maxReferenceFrames * 2 / 25) + " s)");
      return;
    }
    if (request.referenceText.size() > config_.maxReferenceTextTokens) {
      sink.onError(request.id, "limit_exceeded",
                   "the reference's text holds " + std::to_string(request.referenceText.size()) +
                       " tokens; a reference's text holds at most " + std::to_string(config_.maxReferenceTextTokens));
      return;
    }
    for (uint32_t token : request.referenceText)
      if (token >= c.textVocabulary) {
        sink.onError(request.id, "invalid_request", "reference text token outside the vocabulary");
        return;
      }
    // Audio codes only: code 0's special ids (2048 on) are not frames.
    for (uint32_t code : request.referenceCodes)
      if (code >= c.predictorVocabulary) {
        sink.onError(request.id, "invalid_request", "reference code outside the codebooks");
        return;
      }
  }
  queue_.push_back(std::move(request));
}

void TtsEngine::cancel(uint64_t requestId, EventSink &sink) {
  for (auto it = queue_.begin(); it != queue_.end(); ++it)
    if (it->id == requestId) {
      queue_.erase(it);
      sink.onDone(requestId, DoneReason::Cancelled, {});
      return;
    }
  for (auto &lane : lanes_)
    if (lane->active && lane->request.id == requestId) {
      finish(*lane, DoneReason::Cancelled, sink);
      return;
    }
}

void TtsEngine::finish(Lane &lane, DoneReason reason, EventSink &sink) {
  if (!lane.active) return;
  lane.active = false;
  lane.finished = false;
  lane.talkerKv.release();
  lane.predictorKv.release();
  // The new frames (not the reference's), which a later request may take as
  // its reference.
  if (reason != DoneReason::Cancelled && lane.request.returnCodes)
    sink.onCodes(lane.request.id,
                 {at<const uint32_t>(frames_) + uint64_t{lane.index} * config_.maxFrames * OMNI_CODE_GROUPS,
                  size_t(lane.step) * OMNI_CODE_GROUPS});
  DoneTimings timings;
  timings.promptTokens = lane.promptTokens;
  timings.completion = lane.step;
  const auto now = Clock::now();
  timings.prefillMicros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(lane.prefillDone - lane.started).count());
  timings.decodeMicros = static_cast<uint64_t>(lane.decodeMicros.count());
  timings.wallMicros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - lane.started).count());
  sink.onDone(lane.request.id, reason, timings);
}

bool TtsEngine::tick(EventSink &sink) {
  bool worked = false;
  if (!queue_.empty()) {
    admit(sink);
    worked = true;
  }
  if (activeLanes()) {
    step(sink);
    worked = true;
  }
  return worked;
}

// mlx-audio's qwen3_tts.py, _prepare_generation_inputs: the codec prefix
// [nothink, think_bos, think_eos] or [think, think_bos, lang, think_eos],
// the speaker's row, then [pad, bos]; tts_pad over every codec row but the
// last two, tts_bos over pad and the first text token over bos; the
// instruction's turn before the role, its text rows alone. A continuation
// speaks the reference's text then the request's, and replays the
// reference's frames after the prompt as the steps that sampled them would
// have read them: frame j's codes over trailing row j (t_{j+1}, then
// tts_eos, then tts_pad).
TtsPrompt ttsPrompt(const TtsConfig &c, const TtsRequest &request) {
  std::vector<uint32_t> joined;
  if (!request.referenceText.empty()) {
    joined = request.referenceText;
    joined.insert(joined.end(), request.text.begin(), request.text.end());
  }
  const std::vector<uint32_t> &text = joined.empty() ? request.text : joined;
  const uint32_t n = static_cast<uint32_t>(text.size());
  std::vector<uint32_t> codecPrefix{c.codecNoThink, c.codecThinkBos, c.codecThinkEos};
  if (request.languageId) codecPrefix = {c.codecThink, c.codecThinkBos, request.languageId, c.codecThinkEos};
  if (request.speaker) codecPrefix.push_back(request.speaker);
  codecPrefix.push_back(c.codecPad);
  const uint32_t padCount = static_cast<uint32_t>(codecPrefix.size()) - 1;
  const auto instruction = static_cast<int32_t>(request.instruction.size());
  TtsPrompt prompt;
  // Text rows: instruction, role (3), tts_pad x padCount, tts_bos, t0, then
  // t1..tN-1, tts_eos, tts_pad.
  std::vector<uint32_t> &tokens = prompt.tokens;
  tokens = request.instruction;
  tokens.insert(tokens.end(), {c.imStart, c.assistant, 198});
  for (uint32_t i = 0; i < padCount; ++i) tokens.push_back(c.ttsPad);
  tokens.push_back(c.ttsBos);
  const auto first = static_cast<uint32_t>(tokens.size()); // t0
  tokens.insert(tokens.end(), text.begin(), text.end());
  tokens.push_back(c.ttsEos);
  tokens.push_back(c.ttsPad);
  std::vector<OmniPromptPosition> &positions = prompt.positions;
  for (int32_t i = 0; i < instruction + 3; ++i) positions.push_back({i, -1, -1, 0});
  const int32_t pads = instruction + 3;
  for (uint32_t i = 0; i <= padCount; ++i) positions.push_back({pads + int32_t(i), int32_t(codecPrefix[i]), -1, 0});
  // CustomVoice reads the whole text in its prompt (the official
  // implementation's non-streaming mode, which it always uses for these
  // checkpoints): t0..tN-1 and tts_eos over codec_pad, tts_pad over
  // codec_bos, then tts_pad under every frame. The other models read t0
  // over codec_bos and one more text row under each frame (mlx-audio
  // 0.5's streaming layout): t1..tN-1, tts_eos, then tts_pad.
  const bool whole = c.type == TtsModelType::CustomVoice;
  if (whole) {
    for (uint32_t i = 0; i <= n; ++i) positions.push_back({int32_t(first + i), int32_t(c.codecPad), -1, 0});
    prompt.trailingBase = first + n + 1;
    prompt.trailingCount = 0;
  } else {
    prompt.trailingBase = first + 1;
    prompt.trailingCount = n; // t1..tN-1 and eos: N rows; row N is pad
  }
  positions.push_back({int32_t(whole ? prompt.trailingBase : first), int32_t(c.codecBos), -1, 0});
  prompt.referenceFrames = static_cast<uint32_t>(request.referenceCodes.size() / OMNI_CODE_GROUPS);
  for (uint32_t j = 0; j < prompt.referenceFrames; ++j)
    positions.push_back({int32_t(prompt.trailingBase + std::min(j, prompt.trailingCount)), -1, int32_t(j), 0});
  return prompt;
}

// The prompt of a request (docs/omni-design.md, TTS pipeline): the text rows
// projected in one go per lane, then the positions over them.
namespace {
// The pages of a lane's talker positions and its predictor's block.
uint32_t lanePages(const TtsConfig &model, uint32_t positions) {
  return (positions + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS * model.talker.layers + model.predictor.layers;
}
} // namespace

uint32_t TtsEngine::minimumPages(const TtsConfig &model, const TtsEngineConfig &config) noexcept {
  return lanePages(model, config.textRows() + config.maxReferenceFrames + std::min(config.maxFrames, kMinimumPoolFrames) + 1);
}

void TtsEngine::admit(EventSink &sink) {
  const TtsConfig &c = model_.config();
  const uint32_t hidden = c.talker.hidden;
  const uint32_t textStride = roundUp(textRows_, 64);
  std::vector<Lane *> admitted, continuations;
  std::vector<OmniPromptPosition> positions;
  std::vector<ops::omni::AttentionSequence> descriptors;
  uint32_t rows = 0;
  metal::CommandGraph graph;
  while (!queue_.empty() && admitted.size() < lanes_.size()) {
    Lane *free = nullptr;
    for (auto &lane : lanes_)
      // A lane whose last evaluation has not reported keeps its surfaces.
      if (!lane->active && lane->ane.state != Lane::AneState::InFlight &&
          std::find(admitted.begin(), admitted.end(), lane.get()) == admitted.end()) {
        free = lane.get();
        break;
      }
    if (!free) break;
    TtsRequest request = std::move(queue_.front());
    queue_.pop_front();
    const TtsPrompt prompt = ttsPrompt(c, request);
    const std::vector<uint32_t> &tokens = prompt.tokens;
    const uint32_t promptRows = static_cast<uint32_t>(prompt.positions.size());
    if (rows + promptRows > SPLASH_PREFILL_TOKEN_BUDGET) {
      queue_.push_front(std::move(request));
      break;
    }
    Lane &lane = *free;
    lane.request = std::move(request);
    // Without a limit the request may run to kDefaultFramesPerToken frames
    // per text token (at least kMinimumDefaultFrames): normal speech takes
    // about four, a slow speaker's up to ten (CustomVoice 4-bit's `vivian`
    // reading long Mandarin or English passages; eight cut them short), and
    // a sampled talker that misses EOS would otherwise run to the engine's
    // maximum and hold its batch. A continuation
    // keeps its reference's pace, which may be slower (a slow speaker's
    // Mandarin takes about six): it may run to twice the reference's frames
    // per text token if that is more.
    const uint64_t textTokens = lane.request.text.size();
    uint64_t paced = kDefaultFramesPerToken * textTokens;
    if (!lane.request.referenceText.empty())
      paced = std::max<uint64_t>(paced, (2 * prompt.referenceFrames * textTokens + lane.request.referenceText.size() - 1) /
                                            lane.request.referenceText.size());
    const uint32_t textFrames =
        static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(paced, kMinimumDefaultFrames), config_.maxFrames));
    lane.maxFrames = std::min(lane.request.maxFrames ? lane.request.maxFrames : textFrames, config_.maxFrames);
    // A pool below one request of this prompt and these frames (a tight
    // --max-memory) shortens the request to what the pool holds alone.
    const uint32_t poolPositions =
        (pool_->pageCount() - c.predictor.layers) / c.talker.layers * OMNI_PAGE_TOKENS;
    if (poolPositions <= promptRows + 1) {
      sink.onError(lane.request.id, "engine_busy", "the request's text needs more KV memory than this server's --max-memory leaves");
      continue;
    }
    lane.maxFrames = std::min(lane.maxFrames, poolPositions - promptRows - 1);
    // The pages it may need to its end, taken now so that no admitted request
    // runs out of them: it waits for them while others run.
    if (lanePages(c, promptRows + lane.maxFrames + 1) > pool_->freePages()) {
      // An idle engine has its whole pool free; it cannot wait for pages.
      if (!activeLanes() && admitted.empty()) {
        sink.onError(lane.request.id, "engine_busy", "no KV pages left");
        continue;
      }
      queue_.push_front(std::move(lane.request));
      break;
    }
    lane.chunkFrames = std::clamp<uint32_t>(lane.request.chunkFrames ? lane.request.chunkFrames : 25, 1, config_.maxChunkFrames);
    lane.nextChunk = std::min(std::max(config_.firstChunkFrames, 1u), lane.chunkFrames);
    lane.promptTokens = promptRows;
    lane.committed = 0;
    lane.step = 0;
    lane.queuedFrames = 0;
    lane.decodedFrames = 0;
    lane.sampleOffset = 0;
    lane.stackOnAne = false; // the stream's reset histories, which a first Neural Engine chunk takes
    lane.ane = {};
    lane.trailingBase = prompt.trailingBase;
    lane.trailingCount = prompt.trailingCount;
    lane.referenceFrames = prompt.referenceFrames;
    lane.minimumFrames = prompt.referenceFrames ? continuationMinimumFrames(lane.request.text.size())
                         : config_.minimumFrames ? 2u
                                                 : 0u;
    // The repetition penalty's history starts with the reference's code 0s.
    lane.code0History.clear();
    for (uint32_t j = 0; j < prompt.referenceFrames; ++j)
      lane.code0History.push_back(lane.request.referenceCodes[size_t(j) * OMNI_CODE_GROUPS]);
    streams_[lane.index]->reset();
    lane.rng.seed(lane.request.seed ? lane.request.seed : 0x9E3779B97F4A7C15ull ^ lane.request.id);
    lane.started = Clock::now();
    lane.decodeMicros = std::chrono::microseconds{0};
    if (!lane.talkerKv.reserve(promptRows + lane.maxFrames + 1) || !lane.predictorKv.reserve(OMNI_CODE_GROUPS)) {
      sink.onError(lane.request.id, "engine_busy", "no KV pages left");
      lane.talkerKv.release();
      continue;
    }
    lane.active = true;
    sink.onStart(lane.request.id, lane.index);
    // Text tokens into the lane's region and their projection.
    const uint32_t base = lane.index * textStride;
    model_.addTextProjection(graph, lane.index, tokens,
                             backend_.view(textPool_, uint64_t{base} * hidden * 2, uint64_t{textStride} * hidden * 2));
    const uint64_t referenceBase = uint64_t{lane.index} * config_.maxReferenceFrames;
    for (OmniPromptPosition position : prompt.positions) {
      position.text_row += int32_t(base);
      if (position.frame >= 0) position.frame += int32_t(referenceBase);
      positions.push_back(position);
    }
    if (prompt.referenceFrames) {
      std::memcpy(at<uint32_t>(references_) + referenceBase * OMNI_CODE_GROUPS, lane.request.referenceCodes.data(),
                  lane.request.referenceCodes.size() * 4);
      continuations.push_back(&lane);
    }
    descriptors.push_back({talkerTables_->offset(lane.index), talkerTables_->stride(), 0, promptRows, rows, 0});
    lane.prefillRow = rows + promptRows - 1;
    rows += promptRows;
    admitted.push_back(&lane);
  }
  if (admitted.empty()) return;
  addAneFronts(graph);
  // A continuation's codec stream takes the reference's last frames first:
  // the front of all but the last few, then those whole, into the lane's
  // audio rows (free until decodeAudio), which are dropped.
  const uint64_t laneAudioBytes = uint64_t{config_.maxChunkFrames} * samplesPerFrame_ * 4;
  for (Lane *lane : continuations) {
    const uint32_t context = std::min({lane->referenceFrames, kReferenceContextFrames, config_.maxChunkFrames});
    const uint32_t whole = std::min(context, kReferenceStackFrames);
    const uint64_t frameBytes = OMNI_CODE_GROUPS * 4;
    const uint64_t first = uint64_t{lane->index} * config_.maxReferenceFrames + lane->referenceFrames - context;
    ops::omni::CodecStream &stream = *streams_[lane->index];
    const auto frames = [&](uint64_t from, uint32_t count) {
      return backend_.view(references_, from * frameBytes, uint64_t{count} * frameBytes);
    };
    if (context > whole) codec_.addFrontStream(graph, stream, frames(first, context - whole), context - whole);
    codec_.addDecodeStream(graph, stream, frames(first + context - whole, whole), whole,
                           backend_.view(audio_, lane->index * laneAudioBytes, laneAudioBytes));
  }
  // Assemble every prompt row, prefill the talker, then the first frame.
  std::memcpy(positions_.contents(), positions.data(), positions.size() * sizeof(OmniPromptPosition));
  ops::omni::Embedding::addPromptAssemble(graph, positions_, textPool_, lanes_.size() * textStride, model_.tables(), {},
                                          references_, lanes_.size() * std::max(config_.maxReferenceFrames, 1u),
                                          talkerPrefill_->hidden, rows);
  const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
  Qwen3Decoder::uploadTables(*talkerPrefill_, attention);
  std::vector<uint32_t> order;
  auto *logitRows = at<uint32_t>(talkerPrefill_->logitRows);
  for (size_t i = 0; i < admitted.size(); ++i) {
    logitRows[i] = admitted[i]->prefillRow;
    order.push_back(admitted[i]->index);
  }
  model_.talker().addForward(graph, *talkerPrefill_, {talkerTables_->buffer(), talkerTables_->entries(), pool_->pages(),
                                                     pool_->pageCount()}, {&attention, static_cast<uint32_t>(admitted.size())});
  const auto rowsCount = static_cast<uint32_t>(admitted.size());
  writeUniforms(order);
  writeHistory(order);
  addCodeSampling(graph, talkerPrefill_->logits, rowsCount, 0, order, true);
  // The predictor reads the talker's final hidden of the selected rows.
  ops::omni::Embedding::addGatherRows(graph, rowIndex_, talkerPrefill_->selected, predictorInput_, rowsCount, rowsCount,
                                      hidden, 2);
  addPredictor(graph, rowsCount, order);
  (void)runCommand(graph, sink);
  const auto t1 = Clock::now();
  for (Lane *lane : admitted) {
    lane->committed = lane->promptTokens;
    lane->prefillDone = t1;
  }
  collectFrames(order, sink);
  decodeAudio(sink, false);
}

// This step's uniforms, one per code group per row, from the lanes' RNGs.
void TtsEngine::writeUniforms(const std::vector<uint32_t> &order) {
  auto *uniforms = at<float>(uniforms_);
  for (size_t r = 0; r < order.size(); ++r) {
    Lane &lane = *lanes_[order[r]];
    std::uniform_real_distribution<float> draw(0.0F, 1.0F);
    for (uint32_t g = 0; g < OMNI_CODE_GROUPS; ++g) uniforms[r * OMNI_CODE_GROUPS + g] = draw(lane.rng);
  }
}

// The code 0 history of each row, for the repetition penalty.
void TtsEngine::writeHistory(const std::vector<uint32_t> &order) {
  auto *history = at<uint32_t>(history_);
  auto *counts = at<uint32_t>(historyCounts_);
  for (size_t r = 0; r < order.size(); ++r) {
    const Lane &lane = *lanes_[order[r]];
    const uint32_t count = static_cast<uint32_t>(std::min<size_t>(lane.code0History.size(), historyStride_));
    std::memcpy(history + r * historyStride_, lane.code0History.data(), count * 4);
    counts[r] = count;
  }
}

// Samples code group `group` of every row into sampled_[row][group]. Group 0
// reads the talker's logits with suppression and the penalty; the others the
// predictor's logits with its policy. The rows' policies must agree
// (one request's per dispatch would be exact); the first row's are used.
void TtsEngine::addCodeSampling(metal::CommandGraph &graph, metal::MetalBuffer logits, uint32_t rows, uint32_t group,
                                const std::vector<uint32_t> &order, bool first) {
  const TtsConfig &c = model_.config();
  const Lane &lead = *lanes_[order.front()];
  OmniSampleParams params{};
  params.rows = rows;
  params.output_stride = OMNI_CODE_GROUPS;
  params.uniform_stride = OMNI_CODE_GROUPS;
  if (group == 0) {
    params.vocabulary = c.codecVocabulary;
    params.suppress_from = c.predictorVocabulary;
    params.eos = c.codecEos;
    params.top_k = lead.request.topK;
    params.top_p = lead.request.topP;
    params.temperature = lead.request.temperature;
    params.repetition_penalty = lead.request.repetitionPenalty;
    params.history_stride = historyStride_;
    // EOS is masked row by row, for the lanes short of their minimum.
    uint32_t maskEos = 0;
    for (size_t r = 0; r < order.size(); ++r)
      if (lanes_[order[r]]->step < lanes_[order[r]]->minimumFrames) maskEos |= 1u << r;
    params.mask_eos = maskEos;
  } else {
    params.vocabulary = c.predictorVocabulary;
    params.suppress_from = 0;
    params.eos = 0xffffffffu;
    params.top_k = lead.request.predictorTopK;
    params.top_p = lead.request.predictorTopP;
    params.temperature = lead.request.predictorTemperature;
    params.repetition_penalty = 1.0F;
    params.history_stride = historyStride_;
    params.mask_eos = 0;
  }
  (void)first;
  ops::omni::Sampling::addSampleCodes(graph, std::move(logits), history_, historyCounts_,
                                      backend_.view(uniforms_, uint64_t{group} * 4,
                                                    uniforms_.sizeBytes() - uint64_t{group} * 4),
                                      backend_.view(sampled_, uint64_t{group} * 4, sampled_.sizeBytes() - uint64_t{group} * 4),
                                      params);
}

// The 15 predictor sub-steps over `rows` rows whose talker hidden sits in
// the even rows of predictorInput_ and whose code 0 was just sampled.
void TtsEngine::addPredictor(metal::CommandGraph &graph, uint32_t rows, const std::vector<uint32_t> &order) {
  const TtsConfig &c = model_.config();
  const uint32_t hidden = c.talker.hidden;
  const KvBinding kv{predictorTables_->buffer(), predictorTables_->entries(), pool_->pages(), pool_->pageCount()};
  // Sub-step 0: [hidden, codec_embedding(code 0)] per row, two rows each.
  ops::omni::Embedding::addGather(graph, sampled_, model_.tables().codec,
                                  backend_.view(predictorInput_, uint64_t{hidden} * 2, predictorInput_.sizeBytes() - uint64_t{hidden} * 2),
                                  rows, c.codecVocabulary, hidden, 2, OMNI_CODE_GROUPS);
  model_.addPredictorInput(graph, predictorInput_, predictorPrefill_->hidden, 2 * rows);
  std::vector<ops::omni::AttentionSequence> descriptors;
  for (uint32_t r = 0; r < rows; ++r)
    descriptors.push_back({predictorTables_->offset(order[r]), predictorTables_->stride(), 0, 2, 2 * r, 0});
  const ops::omni::AttentionTables prefill = ops::omni::attentionTables(descriptors, true);
  Qwen3Decoder::uploadTables(*predictorPrefill_, prefill);
  auto *logitRows = at<uint32_t>(predictorPrefill_->logitRows);
  for (uint32_t r = 0; r < rows; ++r) logitRows[r] = 2 * r + 1;
  // At few rows the heads fold the final norm over the raw selected rows.
  const bool fold = model_.predictorHeadFoldsNorm(*predictorDecode_, rows);
  model_.predictor().addForward(graph, *predictorPrefill_, kv, {.tables = &prefill, .logitRows = rows, .rawSelected = fold});
  model_.addPredictorHead(graph, *predictorPrefill_, 0, rows, predictorLogits_,
                          fold ? predictorPrefill_->selectedRaw : metal::MetalBuffer{});
  addCodeSampling(graph, predictorLogits_, rows, 1, order, false);
  // Sub-steps 1..14: codec_embedding[i-1](code i) through the predictor.
  for (uint32_t i = 1; i + 1 < OMNI_CODE_GROUPS; ++i) {
    const bool qkvReady =
        model_.addPredictorCodeInput(graph, backend_.view(sampled_, uint64_t{i} * 4, sampled_.sizeBytes() - uint64_t{i} * 4),
                                     i, predictorDecode_->hidden, predictorDecode_->qkv, rows);
    std::vector<ops::omni::AttentionSequence> step;
    for (uint32_t r = 0; r < rows; ++r)
      step.push_back({predictorTables_->offset(order[r]), predictorTables_->stride(), 1 + i, 1, r, 1 + i});
    const ops::omni::AttentionTables decode = ops::omni::attentionTables(step, true);
    Qwen3Decoder::uploadTables(*predictorDecode_, decode);
    auto *decodeLogitRows = at<uint32_t>(predictorDecode_->logitRows);
    for (uint32_t r = 0; r < rows; ++r) decodeLogitRows[r] = r;
    model_.predictor().addForward(graph, *predictorDecode_, kv,
                                  {.tables = &decode, .logitRows = rows, .firstQkvReady = qkvReady, .rawSelected = fold});
    model_.addPredictorHead(graph, *predictorDecode_, i, rows, predictorLogits_,
                            fold ? predictorDecode_->hidden : metal::MetalBuffer{});
    addCodeSampling(graph, predictorLogits_, rows, i + 1, order, false);
  }
}

// Reads this step's codes: EOS ends a lane; a frame joins its lane's frames.
void TtsEngine::collectFrames(const std::vector<uint32_t> &order, EventSink &sink) {
  const TtsConfig &c = model_.config();
  const auto *sampled = at<const uint32_t>(sampled_);
  for (size_t r = 0; r < order.size(); ++r) {
    Lane &lane = *lanes_[order[r]];
    const uint32_t *codes = sampled + r * OMNI_CODE_GROUPS;
    if (codes[0] == c.codecEos) {
      lane.finished = true;
      lane.reason = DoneReason::Stop;
      continue;
    }
    std::memcpy(at<uint32_t>(frames_) + (uint64_t{lane.index} * config_.maxFrames + lane.step) * OMNI_CODE_GROUPS, codes,
                OMNI_CODE_GROUPS * 4);
    if (debug_) {
      std::fprintf(stderr, "lane %u frame %u:", lane.index, lane.step);
      for (uint32_t g = 0; g < OMNI_CODE_GROUPS; ++g) std::fprintf(stderr, " %u", codes[g]);
      std::fprintf(stderr, "\n");
    }
    lane.code0History.push_back(codes[0]);
    ++lane.step;
    if (lane.step >= lane.maxFrames) {
      lane.finished = true;
      lane.reason = DoneReason::Length;
    }
  }
  (void)sink;
}

void TtsEngine::step(EventSink &sink) {
  const auto begun = Clock::now();
  const TtsConfig &c = model_.config();
  const uint32_t hidden = c.talker.hidden;
  const uint32_t textStride = roundUp(textRows_, 64);
  std::vector<uint32_t> order;
  std::vector<ops::omni::AttentionSequence> descriptors;
  auto *stepCodes = at<uint32_t>(stepCodes_);
  auto *trailing = at<uint32_t>(trailingRows_);
  for (auto &lanePointer : lanes_) {
    Lane &lane = *lanePointer;
    if (!lane.active || lane.finished) continue;
    if (!lane.talkerKv.reserve(lane.committed + 1)) {
      lane.finished = true;
      lane.reason = DoneReason::Length;
      continue;
    }
    const uint32_t r = static_cast<uint32_t>(order.size());
    order.push_back(lane.index);
    std::memcpy(stepCodes + r * OMNI_CODE_GROUPS,
                at<uint32_t>(frames_) + (uint64_t{lane.index} * config_.maxFrames + lane.step - 1) * OMNI_CODE_GROUPS,
                OMNI_CODE_GROUPS * 4);
    // Trailing row of this step: t_{step}.. then eos, then pad for ever (a
    // continuation's steps go on from its reference's).
    const uint32_t trailingIndex = std::min(lane.referenceFrames + lane.step - 1, lane.trailingCount);
    trailing[r] = lane.index * textStride + lane.trailingBase + trailingIndex;
    descriptors.push_back({talkerTables_->offset(lane.index), talkerTables_->stride(), lane.committed, 1, r, lane.committed});
  }
  if (!order.empty()) {
    const auto rows = static_cast<uint32_t>(order.size());
    metal::CommandGraph graph;
    const size_t fronts = addAneFronts(graph);
    ops::omni::Embedding::addTtsStepInput(graph, stepCodes_, model_.tables(), trailingRows_, textPool_,
                                          static_cast<uint32_t>(lanes_.size()) * textStride, talkerDecode_->hidden, rows);
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, true);
    Qwen3Decoder::uploadTables(*talkerDecode_, attention);
    auto *logitRows = at<uint32_t>(talkerDecode_->logitRows);
    for (uint32_t r = 0; r < rows; ++r) logitRows[r] = r;
    model_.talker().addForward(graph, *talkerDecode_,
                               {talkerTables_->buffer(), talkerTables_->entries(), pool_->pages(), pool_->pageCount()},
                               {&attention, rows});
    writeUniforms(order);
    writeHistory(order);
    addCodeSampling(graph, talkerDecode_->logits, rows, 0, order, false);
    ops::omni::Embedding::addGatherRows(graph, rowIndex_, talkerDecode_->selected, predictorInput_, rows, rows, hidden, 2);
    // The predictor's sub-steps read only what the talker's command writes
    // on the GPU, and their host tables and logit rows belong to the
    // predictor's arenas, which no command in flight reads: they are built
    // while the talker runs and queued behind it.
    size_t dispatches = graph.dispatches().size();
    const auto addSubSteps = [&](metal::CommandGraph &next) {
      addPredictor(next, rows, order);
      dispatches += next.dispatches().size();
    };
    const auto t0 = Clock::now();
    const metal::CommandTiming timing = runCommand(graph, sink, addSubSteps);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0);
    if (debug_)
      std::fprintf(stderr, "step rows %u (+%zu codec fronts): %zu dispatches, built %.2f ms, gpu %.2f ms, command wall "
                   "%.2f ms, total %.2f ms\n", rows, fronts, dispatches,
                   std::chrono::duration<double, std::milli>(t0 - begun).count(), timing.gpuSeconds * 1e3,
                   timing.wallSeconds * 1e3, elapsed.count() / 1e3);
    for (uint32_t lane : order) {
      lanes_[lane]->committed += 1;
      lanes_[lane]->decodeMicros += elapsed;
    }
    collectFrames(order, sink);
  }
  decodeAudio(sink, false);
  // Every lane done generating: wait for the Neural Engine's last chunks.
  if (order.empty()) {
    awaitAneReport(sink);
    decodeAudio(sink, false);
  }
}

// Decodes the new frames of every lane whose chunk is due, or that
// finished, through the lane's codec stream (which carries the decoder's
// state across chunks): on the Neural Engine, whose chunks' fronts go to the
// head of the next command, when a function holds the chunk (only a
// stream's last chunk is padded up to one); on the GPU now otherwise, and
// for a chunk whose evaluation failed.
void TtsEngine::decodeAudio(EventSink &sink, bool final) {
  drainAneReports(sink);
  struct Due {
    Lane *lane;
    uint32_t frames;
    bool recover; // the stack of a chunk whose evaluation failed
  };
  std::vector<Due> due;
  const bool aneOpen = ane_ && !aneRetired_;
  for (auto &lanePointer : lanes_) {
    Lane &lane = *lanePointer;
    if (!lane.active) continue;
    if (lane.ane.state == Lane::AneState::Failed) {
      due.push_back({&lane, lane.ane.frames, true});
      continue;
    }
    // One chunk of a lane at a time on the Neural Engine keeps them in order.
    if (lane.ane.state != Lane::AneState::Idle) continue;
    const uint32_t pending = lane.step - lane.queuedFrames;
    if (!pending) continue;
    const bool last = lane.finished || final;
    if (pending < lane.nextChunk && !last) continue;
    if (aneOpen && lane.queuedFrames > 0) {
      const uint32_t frames = std::min(pending, last ? ane_->functionFrames().back() : lane.nextChunk);
      const uint32_t function = ane_->functionFor(frames);
      if (function && (function == frames || (last && frames == pending))) {
        lane.ane = {Lane::AneState::Due, lane.queuedFrames, frames, function, 0, 0, {}};
        lane.queuedFrames += frames;
        lane.nextChunk = std::min(lane.nextChunk * 2, lane.chunkFrames);
        continue;
      }
    }
    due.push_back({&lane, std::min(pending, config_.maxChunkFrames), false});
  }
  if (!due.empty()) {
    metal::CommandGraph graph;
    const uint64_t laneAudioBytes = uint64_t{config_.maxChunkFrames} * samplesPerFrame_ * 4;
    for (size_t i = 0; i < due.size(); ++i) {
      const Due &d = due[i];
      Lane &lane = *d.lane;
      ops::omni::CodecStream &stream = *streams_[lane.index];
      const metal::MetalBuffer audio = backend_.view(audio_, i * laneAudioBytes, laneAudioBytes);
      if (d.recover) {
        // The front advanced the stream; the stack reads the half before it.
        const uint32_t phase = stream.phase() ^ 1u;
        ane_->addHistoriesToStream(graph, lane.index, stream, phase);
        ane_->addInputToRows(graph, lane.index, codec_.stackInput(d.frames));
        codec_.addStackStream(graph, stream, d.frames, phase, audio);
      } else {
        if (lane.stackOnAne) ane_->addHistoriesToStream(graph, lane.index, stream, stream.phase());
        const uint64_t frameBytes = OMNI_CODE_GROUPS * 4;
        const uint64_t first = uint64_t{lane.index} * config_.maxFrames + lane.queuedFrames;
        codec_.addDecodeStream(graph, stream, backend_.view(frames_, first * frameBytes, uint64_t{d.frames} * frameBytes),
                               d.frames, audio);
      }
      lane.stackOnAne = false;
    }
    const metal::CommandTiming timing = runCommand(graph, sink);
    if (debug_) {
      uint32_t fresh = 0;
      for (const Due &d : due) fresh += d.frames;
      std::fprintf(stderr, "codec: %zu lanes, %u frames: gpu %.2f ms\n", due.size(), fresh, timing.gpuSeconds * 1e3);
    }
    for (size_t i = 0; i < due.size(); ++i) {
      const Due &d = due[i];
      Lane &lane = *d.lane;
      if (d.recover) {
        lane.ane.state = Lane::AneState::Idle;
      } else {
        lane.queuedFrames += d.frames;
        lane.nextChunk = std::min(lane.nextChunk * 2, lane.chunkFrames);
      }
      const float *samples = at<const float>(audio_) + i * (laneAudioBytes / 4);
      emitAudio(lane, {samples, size_t(d.frames) * samplesPerFrame_}, sink);
    }
  }
  // Fronts due with no step to carry them: a command of their own.
  bool stepping = !queue_.empty(), dueFronts = false;
  for (const auto &lane : lanes_) {
    stepping = stepping || (lane->active && !lane->finished);
    dueFronts = dueFronts || lane->ane.state == Lane::AneState::Due;
  }
  if (dueFronts && !stepping) {
    metal::CommandGraph graph;
    if (addAneFronts(graph)) (void)runCommand(graph, sink);
  }
  finishDecoded(sink);
}

void TtsEngine::emitAudio(Lane &lane, std::span<const float> samples, EventSink &sink) {
  std::vector<int16_t> pcm(samples.size());
  for (size_t s = 0; s < samples.size(); ++s)
    pcm[s] = static_cast<int16_t>(std::lrint(std::clamp(samples[s], -1.0F, 1.0F) * 32767.0F));
  sink.onAudio(lane.request.id, lane.sampleOffset, pcm);
  lane.sampleOffset += static_cast<uint32_t>(samples.size());
  lane.decodedFrames += static_cast<uint32_t>(samples.size() / samplesPerFrame_);
}

void TtsEngine::finishDecoded(EventSink &sink) {
  for (auto &lanePointer : lanes_) {
    Lane &lane = *lanePointer;
    if (lane.active && lane.finished && lane.decodedFrames == lane.step && lane.ane.state == Lane::AneState::Idle)
      finish(lane, lane.reason, sink);
  }
}

size_t TtsEngine::addAneFronts(metal::CommandGraph &graph) {
  size_t count = 0;
  for (auto &lanePointer : lanes_) {
    Lane &lane = *lanePointer;
    if (lane.ane.state != Lane::AneState::Due) continue;
    if (!lane.active) {
      lane.ane.state = Lane::AneState::Idle;
      continue;
    }
    ops::omni::CodecStream &stream = *streams_[lane.index];
    if (!lane.stackOnAne) {
      ane_->addHistoriesFromStream(graph, lane.index, stream, stream.phase());
      lane.stackOnAne = true;
    }
    const uint64_t frameBytes = OMNI_CODE_GROUPS * 4;
    const uint64_t first = uint64_t{lane.index} * config_.maxFrames + lane.ane.first;
    codec_.addFrontStream(graph, stream, backend_.view(frames_, first * frameBytes, uint64_t{lane.ane.frames} * frameBytes),
                          lane.ane.frames);
    ane_->addInput(graph, lane.index, codec_.stackInput(lane.ane.frames), lane.ane.function);
    lane.ane.state = Lane::AneState::Encoded;
    ++count;
  }
  if (!count) return 0;
  graph.signal(aneEvent_, ++aneValue_);
  for (auto &lane : lanes_)
    if (lane->ane.state == Lane::AneState::Encoded) lane->ane.wait = aneValue_;
  return count;
}

void TtsEngine::enqueueAneChunks() {
  for (auto &lanePointer : lanes_) {
    Lane &lane = *lanePointer;
    if (lane.ane.state != Lane::AneState::Encoded) continue;
    if (aneRetired_) {
      lane.ane.state = Lane::AneState::Failed;
      continue;
    }
    lane.ane.serial = ++aneSerial_;
    lane.ane.queued = Clock::now();
    lane.ane.state = Lane::AneState::InFlight;
    try {
      ane_->enqueue(lane.index, lane.ane.function, aneEvent_, lane.ane.wait, lane.ane.wait,
                    [reports = reports_, index = lane.index, serial = lane.ane.serial](bool success) {
                      std::lock_guard lock(reports->mutex);
                      reports->reports.push_back({index, serial, success});
                      reports->changed.notify_all();
                    });
    } catch (const std::exception &error) {
      lane.ane.state = Lane::AneState::Failed;
      retireAne(std::string("an evaluation could not be queued: ") + error.what());
    }
  }
}

metal::CommandTiming TtsEngine::runCommand(metal::CommandGraph &graph, EventSink &sink,
                                           const std::function<void(metal::CommandGraph &)> &behind) {
  const std::shared_ptr<Reports> reports = reports_;
  uint64_t target = 0;
  {
    std::lock_guard lock(reports->mutex);
    target = reports->completedCommands + (behind ? 2 : 1);
  }
  const auto completed = [reports] {
    std::lock_guard lock(reports->mutex);
    ++reports->completedCommands;
    reports->changed.notify_all();
  };
  metal::CommandTicket ticket = backend_.submitCommandAsync(graph.command(), completed);
  enqueueAneChunks();
  metal::CommandTicket queued;
  if (behind) {
    metal::CommandGraph next;
    behind(next);
    queued = backend_.submitCommandBehind(next.dispatches(), completed);
  }
  // The Neural Engine's chunks are emitted as they report while the GPU
  // works; after a second the ticket's wait, which runs the command
  // watchdog, takes over.
  const auto inFlight = [&] {
    for (const auto &lane : lanes_)
      if (lane->ane.state == Lane::AneState::InFlight) return true;
    return false;
  };
  const auto deadline = Clock::now() + std::chrono::seconds(1);
  while (inFlight()) {
    drainAneReports(sink);
    std::unique_lock lock(reports->mutex);
    if (reports->completedCommands >= target || Clock::now() >= deadline) break;
    reports->changed.wait_until(lock, deadline,
                                [&] { return !reports->reports.empty() || reports->completedCommands >= target; });
  }
  metal::CommandTiming timing = ticket.wait();
  if (behind) {
    const metal::CommandTiming second = queued.wait();
    timing.gpuSeconds += second.gpuSeconds;
    timing.wallSeconds += std::max(0.0, second.gpuEndSeconds - timing.gpuEndSeconds);
    timing.gpuEndSeconds = second.gpuEndSeconds;
  }
  drainAneReports(sink);
  return timing;
}

void TtsEngine::drainAneReports(EventSink &sink) {
  if (!ane_) return;
  std::vector<Reports::Report> reports;
  {
    std::lock_guard lock(reports_->mutex);
    reports.swap(reports_->reports);
  }
  for (const Reports::Report &report : reports) {
    Lane &lane = *lanes_[report.lane];
    if (lane.ane.state != Lane::AneState::InFlight || lane.ane.serial != report.serial) continue;
    if (!report.success) {
      lane.ane.state = Lane::AneState::Failed;
      retireAne("an evaluation failed");
      continue;
    }
    if (debug_)
      std::fprintf(stderr, "codec ane: lane %u, %u frames (function %u): %.2f ms from queue to report\n", lane.index,
                   lane.ane.frames, lane.ane.function,
                   std::chrono::duration<double, std::milli>(Clock::now() - lane.ane.queued).count());
    ane_->advance(lane.index);
    lane.ane.state = Lane::AneState::Idle;
    if (!lane.active) continue;
    std::vector<float> samples(size_t(lane.ane.frames) * samplesPerFrame_);
    ane_->samples(lane.index, lane.ane.function, samples);
    emitAudio(lane, samples, sink);
  }
  const auto now = Clock::now();
  for (auto &lane : lanes_)
    if (lane->ane.state == Lane::AneState::InFlight && now - lane->ane.queued > kAneBound) {
      lane->ane.state = Lane::AneState::Failed;
      retireAne("an evaluation did not report within " +
                std::to_string(std::chrono::duration_cast<std::chrono::seconds>(kAneBound).count()) + " s");
    }
}

void TtsEngine::awaitAneReport(EventSink &sink) {
  std::optional<Clock::time_point> deadline;
  for (const auto &lane : lanes_)
    if (lane->ane.state == Lane::AneState::InFlight)
      deadline = std::min(deadline.value_or(Clock::time_point::max()), lane->ane.queued + kAneBound);
  if (!deadline) return;
  {
    std::unique_lock lock(reports_->mutex);
    reports_->changed.wait_until(lock, *deadline, [&] { return !reports_->reports.empty(); });
  }
  drainAneReports(sink);
}

void TtsEngine::retireAne(const std::string &reason) {
  if (aneRetired_) return;
  aneRetired_ = true;
  codecPlacement_ = "codec convolutions on the GPU (the Neural Engine retired: " + reason + ")";
  std::fprintf(stderr, "Codec on the Neural Engine retired (%s); the GPU decodes the codec from here.\n",
               reason.c_str());
  for (auto &lanePointer : lanes_) {
    Lane &lane = *lanePointer;
    if (lane.ane.state == Lane::AneState::Due) {
      // Not encoded yet: the GPU takes these frames.
      lane.queuedFrames -= lane.ane.frames;
      lane.ane.state = Lane::AneState::Idle;
    } else if (lane.ane.state == Lane::AneState::Encoded) {
      lane.ane.state = Lane::AneState::Failed;
    }
  }
}

} // namespace splash::omni
