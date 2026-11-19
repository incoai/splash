#include "AwakeClock.hpp"
#include "omni/AsrEngine.hpp"

#include "StderrLine.hpp"
#include "ane/Program.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "omni/MelSpectrogram.hpp"
#include "ops/Embedding.hpp"
#include "ops/Omni.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <string_view>
#include <unistd.h>

namespace splash::omni {

using Clock = AwakeClock;

// A decoder with its tower, page tables and arenas: the target, or the
// draft. The verify arena runs the target's verification, or the draft's
// catch-up over the tokens it has not taken yet.
struct AsrEngine::Model final {
  AsrConfig config;
  AsrTextWeights weights;
  std::unique_ptr<Qwen3Decoder> decoder;
  std::unique_ptr<ops::omni::AudioEncoder> encoder;
  std::unique_ptr<LanePageTables> tables;
  std::unique_ptr<Qwen3Arena> prefill, decode, verify;
  metal::MetalBuffer audioRows; // bf16 [audio rows][hidden]
  uint32_t maxMelFrames = 0;
};

// A DFlash draft: its layers with the target's head (a decoder), its page
// tables on the shared pool, an arena for block forwards (up to
// kMaximumSpeculativeLanes lanes of a block each) and one for context rows
// (up to a prefill chunk), and the target forwards' captured layer outputs.
struct AsrEngine::DFlash final {
  DFlashDraftConfig config;
  Qwen3Projection fc;
  ops::NormWeights hiddenNorm;
  std::unique_ptr<Qwen3Decoder> decoder;
  std::unique_ptr<LanePageTables> tables;
  std::unique_ptr<Qwen3Arena> block, context;
  metal::MetalBuffer captured;          // bf16 [2048][target layers][hidden]
  std::vector<metal::MetalBuffer> slots; // views of `captured` from each layer's slot
  HiddenCapture capture;
  metal::MetalBuffer blockTokens; // u32 [kMaximumSpeculativeLanes * blockSize]
  // With the draft's own head: u32 [its vocabulary], each token's target token.
  metal::MetalBuffer tokenMap;
};

struct AsrEngine::Lane final {
  Lane(KvPagePool &pool, LanePageTables &tables, LanePageTables *draftTables, uint32_t index)
      : index(index), kv(pool, tables, index) {
    if (draftTables) draftKv = std::make_unique<LaneKv>(pool, *draftTables, index);
  }
  uint32_t index;
  LaneKv kv;
  bool active = false;
  AsrRequest request;
  uint32_t promptTokens = 0;
  uint32_t committed = 0;
  uint32_t generated = 0;
  uint32_t maxTokens = 0;
  uint32_t lastToken = 0;
  bool finished = false;
  DoneReason reason = DoneReason::Stop;
  Clock::time_point started, prefillDone;
  std::chrono::microseconds decodeMicros{0};
  std::vector<uint32_t> tokens; // emitted so far
  // The request's log-mel features, kept from admission while the draft
  // may still take the lane (its tower reads them).
  std::vector<float> features;
  // The draft's view of the lane while it follows it: the inputs its KV
  // holds, and the emitted tokens it has not taken yet (the last always
  // lastToken), which it takes at the start of the next round. A lane the
  // draft stopped following is not taken again.
  std::unique_ptr<LaneKv> draftKv;
  bool drafted = false, draftDropped = false;
  uint32_t draftCommitted = 0;
  std::vector<uint32_t> draftPending;
};

namespace {

// The most emitted tokens a lane's draft may fall behind by (plain steps
// while too many lanes are active) before it stops following the lane: the
// catch-up of eight speculating lanes stays within one 32-row forward.
constexpr uint32_t kMaximumDraftPending = 4;
constexpr uint32_t kMaximumSpeculativeLanes = 8;
// The tokens a lane's audio likely yields per second (Qwen3-ASR on English
// LibriSpeech: 3.4, the language tag included), to judge whether enough
// remain to repay the draft's tower and prefill, which take about 7 ms plus
// 0.115 ms per prompt row (Qwen3-ASR-0.6B-4bit: 16 ms for 86 rows, 56 for
// 427).
constexpr double kTokensPerAudioSecond = 3.0;
constexpr double kAdoptMilliseconds = 7.0, kAdoptMillisecondsPerRow = 0.115;
// The acceptance estimate's prior for a proposal's acceptance given the one
// before it held, and the lane rounds' weight it carries; each lane round's
// evidence decays by kAcceptanceDecay, so the estimate follows the audio.
constexpr double kAcceptancePrior = 0.9, kAcceptancePriorWeight = 4.0, kAcceptanceDecay = 0.995;
// The draft takes lanes only while the active lanes, averaged over about
// half a second, stay within the speculating lanes: at two concurrent
// requests a lane is often alone for a tick or two between one request's
// end and the next's arrival, and a draft taken then is paid for and
// dropped (on the subset over HTTP, 14 lanes taken for 19 rounds).
constexpr double kLoadSeconds = 0.5, kLoadMargin = 0.25;

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }

metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes, uint64_t &total, std::string_view label) {
  total += bytes;
  metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

template <class T> T *at(const metal::MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

// An instant as seconds on AwakeClock's base, which Metal's GPU times and
// the server's time.monotonic() share.
double seconds(Clock::time_point instant) noexcept {
  return std::chrono::duration<double>(instant.time_since_epoch()).count();
}

} // namespace

// The prefill FFN's Neural Engine split (docs/omni-performance.md, measured
// with dev/benchmarks/omni_ane_ffn_bench.mm on an M5 Pro): the ANE takes 3
// of the 12 units of 512 intermediate channels of chunks from 160 rows, an
// FFN layer 15-20% faster than on the GPU alone; below 160 rows the GPU alone
// is faster. 4 units gain 27-29% from 448 rows, but their int8 error took the
// real-weight parity's prefill logits below a cosine of 0.999. The GPU's part
// takes longer than the ANE's at every chunk, so a chunk pads up to the next
// function's rows at no cost.
const AneFfnPolicy &asrAneFfnPolicy() noexcept {
  static const AneFfnPolicy policy{3, {160, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1280, 1536, 2048}};
  return policy;
}

// Measured on an M5 Pro (dev/benchmarks/omni_asr_verify_bench.mm, the 200
// LibriSpeech files through omni-asr-run and over HTTP; docs/omni-performance.md
// rounds 7, 8 and 16): a Qwen3-ASR-0.6B draft forward of one row costs about
// 1.8 ms, against a 4-bit 1.7B step of 3.95 and a bf16 one of 10.2. A 4-bit
// target forward of 3 to 8 rows costs about 4.95 ms (one tile lane), of 9 to
// 16 rows 6.3; a bf16 one 12.1 ms to 8 rows (the MMA tiles), 15.3 to 16 and
// 19.8 to 32. A lone 4-bit lane's round of 5 proposals emits about 5.4 tokens
// in 13.9 ms, saving about 1.4 ms a token (4 to 7 proposals within 1%); at
// two lanes the draft's tower and prefill per request (about 7 ms plus 0.115
// a prompt row) cost more than the rounds save, so 4-bit lanes speculate
// alone. A bf16 round of 7 proposals verifies 8 rows (one MMA tile) and
// emits about 7.0 tokens in 24 ms, 3-4% faster than 11 proposals (12 rows,
// 9.5 tokens in 34 ms) one or two lanes at a time; at four lanes its rounds
// hold 32 rows. A lane without a draft riding in a round gets one token per
// round: a 4-bit round of one drafted lane and one rider emits about 6.4
// tokens in 15 ms, where the two lanes' plain steps emit 7, so 4-bit lanes
// speculate only while every active lane has a draft; a bf16 rider keeps
// pace with the others' rounds instead of stalling them.
//
// What a token saves, against which a lane's draft tower and prefill are
// weighed, comes from the acceptance the rounds observe and the forwards'
// costs (savedPerToken) where lanes do not ride: a lane the draft does not
// take decodes plainly. Where they ride, a lane the draft does not take
// gets one token a round, which that estimate does not price; a bf16 lane
// saves a fixed 7.5 ms a token there (about a lone lane's saving, and less
// than what riding costs a lane beside others), and the estimate instead
// decides whether the active lanes' rounds pay for their drafts at all
// (speculationPays): at eight short requests they do not (HTTP 89 against
// 92 audio s/s plain before that rule).
AsrSpeculationDefaults asrSpeculationDefaults(bool affine) noexcept {
  if (affine) return {5, 1, 16, false, 0.0};
  return {7, kMaximumSpeculativeLanes, OMNI_MAXIMUM_LANES, true, 7.5};
}

// The GPU milliseconds of a target forward of `rows` rows (a decode step of
// that many lanes, or a verification; omni-asr-verify-bench, history 200).
// 4-bit: one row on the Qmv tile, two sharing it, then the same per 8-row
// lane of the decode tiles up to 8 rows and more per lane beyond. bf16: the
// GEMV at one row, the packed MMA tile to 8, the bf16 one to 16 and the MPP
// tiles beyond.
double AsrEngine::targetMilliseconds(bool affine, uint32_t rows) noexcept {
  if (affine) {
    if (rows <= 2) return rows == 1 ? 3.95 : 4.3;
    static constexpr double kLanes[] = {4.95, 6.3, 9.9, 10.5};
    return kLanes[std::min<uint32_t>((rows + 7) / 8, 4) - 1];
  }
  if (rows <= 2) return rows == 1 ? 10.2 : 10.6;
  return rows <= 4 ? 11.5 : rows <= 8 ? 12.1 : rows <= 16 ? 15.3 : 19.8;
}

// The same for the 0.6B draft (a decode step of `rows` lanes, or a
// catch-up of `rows` rows).
double AsrEngine::draftMilliseconds(uint32_t rows) noexcept {
  if (rows <= 2) return rows == 1 ? 1.8 : 1.95;
  return rows <= 4 ? 2.5 : rows <= 8 ? 3.0 : 3.9;
}

// The draft tokens a round of `lanes` speculating lanes and `riders` riding
// proposes per lane: the configured tokens, fewer to keep its rows within
// the rows a round verifies.
uint32_t AsrEngine::roundDraftTokens(uint32_t lanes, uint32_t riders) const noexcept {
  return std::min(draftTokens_, (verifyRows_ - riders) / lanes - 1);
}

// The GPU milliseconds a token of `lanes` 4-bit lanes costs plainly less
// what it costs in their rounds: a round costs a draft forward per proposal
// and the target's verification of every lane's pending token and
// proposals, and emits each lane's accepted proposals and one more, the
// chance that a round's first j proposals all hold being the product of
// each depth's acceptance given the ones before it held. Those come from
// the decayed counts of lane rounds that reached and passed each depth,
// each depth's estimate drawn toward the one before it.
double AsrEngine::savedPerToken(uint32_t lanes) const noexcept {
  const uint32_t k = roundDraftTokens(lanes, 0);
  double expected = 1.0, survival = 1.0, conditional = kAcceptancePrior;
  for (uint32_t j = 1; j <= k; ++j) {
    conditional = (survived_[j] + kAcceptancePriorWeight * conditional) / (reached_[j] + kAcceptancePriorWeight);
    survival *= conditional;
    expected += survival;
  }
  const bool affine = target_->decoder->affine();
  const double round = k * draftMilliseconds(lanes) + targetMilliseconds(affine, lanes * (k + 1));
  return std::max(0.0, targetMilliseconds(affine, lanes) / lanes - round / (lanes * expected));
}

// A DFlash round's draft work in GPU milliseconds, beside the target's
// verification (targetMilliseconds): the draft's forward over `lanes`
// blocks (its five bf16 layers, about 0.5 GB of weights read once, then
// rows), the target's head over the proposals, and the verified rows'
// context (fc and five key-value projections). Estimates to calibrate with
// OMNI_ASR_DEBUG's round timings.
double AsrEngine::dflashMilliseconds(bool affine, uint32_t lanes, uint32_t proposals) noexcept {
  const double forward = 2.2 + 0.3 * (lanes - 1);
  const double head = (affine ? 0.8 : 2.6) + 0.01 * lanes * proposals;
  return forward + head + 0.5;
}

// The proposals per lane (1..limit) that save the most GPU time per token
// in rounds of `lanes` lanes, and that saving against plain steps (each of
// which also appends a row of context per lane): a round emits each lane's
// accepted proposals and one token more, the chance that the first j hold
// being the product of each depth's acceptance given the ones before it
// held, as savedPerToken estimates them.
std::pair<uint32_t, double> AsrEngine::dflashRound(uint32_t lanes, uint32_t limit) const noexcept {
  const bool affine = target_->decoder->affine();
  const double plain = (targetMilliseconds(affine, lanes) + 0.5) / lanes;
  std::pair<uint32_t, double> best{1, -1e9};
  double expected = 1.0, survival = 1.0, conditional = kAcceptancePrior;
  for (uint32_t k = 1; k <= limit && k < reached_.size(); ++k) {
    conditional = (survived_[k] + kAcceptancePriorWeight * conditional) / (reached_[k] + kAcceptancePriorWeight);
    survival *= conditional;
    expected += survival;
    const double round = dflashMilliseconds(affine, lanes, k) + targetMilliseconds(affine, lanes * (k + 1));
    const double saved = plain - round / (lanes * expected);
    if (saved > best.second) best = {k, saved};
  }
  return best;
}

AsrEngine::AsrEngine(metal::MetalBackend &backend, const AsrConfig &config, AsrTextWeights weights,
                     ops::omni::AudioEncoderWeights towerWeights, AsrEngineConfig engineConfig,
                     std::optional<AsrDraftModel> draft, std::optional<AsrDFlashDraft> dflash)
    : backend_(backend), config_(config), engineConfig_(engineConfig), debug_(std::getenv("OMNI_ASR_DEBUG") != nullptr) {
  const char *tracing = std::getenv("OMNI_ASR_TRACE");
  trace_ = tracing && std::string_view(tracing) == "1";
  const uint32_t lanes = engineConfig.lanes;
  if (!lanes || lanes > OMNI_MAXIMUM_LANES) throw std::invalid_argument("ASR takes 1..32 lanes");
  maxMelFrames_ = engineConfig.maxAudioSeconds * (kMelSampleRate / kMelHop);
  maxAudioTokens_ = ops::omni::audioTokens(maxMelFrames_);
  // Each lane's KV holds its prompt, its tokens and a round's unverified
  // rows beyond them.
  const uint32_t maxContext = maxAudioTokens_ + 256 + engineConfig.maxTokens + 2 * OMNI_MAXIMUM_LANES;
  const uint32_t blocks = (maxContext + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS;
  // A model's decoder, tower and arenas; its audio rows hold the longest
  // audio it encodes, or a batch's rows.
  const auto build = [&](const AsrConfig &modelConfig, AsrTextWeights modelWeights,
                         ops::omni::AudioEncoderWeights tower, uint32_t melFrames, bool withVerify) {
    if (tower.dims.output != modelConfig.decoder.hidden)
      throw std::invalid_argument("the audio tower's output is not the decoder's width");
    auto model = std::make_unique<Model>();
    model->config = modelConfig;
    model->weights = std::move(modelWeights);
    model->maxMelFrames = melFrames;
    model->decoder = std::make_unique<Qwen3Decoder>(backend, model->weights.decoder);
    model->encoder = std::make_unique<ops::omni::AudioEncoder>(backend, std::move(tower), melFrames);
    model->tables = std::make_unique<LanePageTables>(backend, lanes, modelConfig.decoder.layers, blocks);
    // Prefill chunks take one request's prompt, or several requests' prompts
    // admitted together (admitBatch), each selecting its last row's logits.
    const ops::Linear &linear = model->decoder->linear();
    const Qwen3Weights &w = model->decoder->weights();
    model->prefill = std::make_unique<Qwen3Arena>(backend, linear, w, SPLASH_PREFILL_TOKEN_BUDGET, lanes, lanes, false);
    model->decode = std::make_unique<Qwen3Arena>(backend, linear, w, lanes, lanes, lanes, true);
    memoryBytes_ += model->prefill->bytes() + model->decode->bytes();
    if (withVerify) {
      model->verify = std::make_unique<Qwen3Arena>(backend, linear, w, OMNI_MAXIMUM_LANES, lanes, OMNI_MAXIMUM_LANES,
                                                   true, true);
      memoryBytes_ += model->verify->bytes();
    }
    const uint32_t audioRows =
        roundUp(std::max<uint32_t>(ops::omni::audioTokens(melFrames), SPLASH_PREFILL_TOKEN_BUDGET), 64);
    model->audioRows = zeroed(backend, uint64_t{audioRows} * modelConfig.decoder.hidden * 2, memoryBytes_, "asr audio rows");
    return model;
  };
  if (draft && dflash) throw std::invalid_argument("an engine takes one draft");
  target_ = build(config, std::move(weights), std::move(towerWeights), maxMelFrames_, draft || dflash);
  if (draft) {
    const AsrConfig &d = draft->config;
    if (d.vocabulary != config.vocabulary || d.audioStart != config.audioStart || d.audioEnd != config.audioEnd ||
        d.stopTokens != config.stopTokens)
      throw std::invalid_argument("the draft's vocabulary or special tokens differ from the target's");
    const uint32_t draftSeconds = std::min(engineConfig.maxAudioSeconds, std::max(engineConfig.draftMaxAudioSeconds, 1u));
    draft_ = build(d, std::move(draft->weights), std::move(draft->tower), draftSeconds * (kMelSampleRate / kMelHop), true);
    const AsrSpeculationDefaults defaults = asrSpeculationDefaults(target_->decoder->affine());
    draftTokens_ = std::clamp<uint32_t>(engineConfig.draftTokens ? engineConfig.draftTokens : defaults.draftTokens, 1,
                                        Qwen3Arena::kTableSlots - 1);
    verifyRows_ = defaults.verifyRows;
    riders_ = defaults.riders;
    savedMilliseconds_ = defaults.savedMilliseconds;
    // Every speculating lane verifies at least two rows.
    const uint32_t speculating = engineConfig.speculativeLanes ? engineConfig.speculativeLanes : defaults.speculativeLanes;
    engineConfig_.speculativeLanes = std::min({speculating, kMaximumSpeculativeLanes, verifyRows_ / 2, lanes});
  }
  if (dflash) {
    const DFlashDraftConfig &c = dflash->config;
    if (c.decoder.hidden != config.decoder.hidden || c.maskToken >= config.vocabulary ||
        c.targetLayers.empty() || c.targetLayers.back() >= config.decoder.layers)
      throw std::invalid_argument("the DFlash draft does not fit the target");
    if (c.quantized) throw std::invalid_argument("a quantized DFlash draft is not supported yet");
    auto d = std::make_unique<DFlash>();
    d->config = c;
    d->fc = std::move(dflash->weights.fc);
    d->hiddenNorm = std::move(dflash->weights.hiddenNorm);
    // The draft's layers and final norm, then its own head or the target's.
    Qwen3Weights w = std::move(dflash->weights.decoder);
    if (c.draftVocabulary) {
      const std::vector<uint32_t> &map = dflash->weights.tokenMap;
      if (!w.head || map.size() != c.draftVocabulary ||
          std::any_of(map.begin(), map.end(), [&](uint32_t token) { return token >= config.vocabulary; }))
        throw std::invalid_argument("the DFlash draft's head or token map does not fit the target");
      d->tokenMap = zeroed(backend, map.size() * 4, memoryBytes_, "asr dflash token map");
      std::memcpy(d->tokenMap.contents(), map.data(), map.size() * 4);
    } else {
      w.head = target_->weights.decoder.head;
      w.dims.headRows = config.decoder.headRows;
      w.dims.vocabulary = config.decoder.vocabulary;
    }
    d->decoder = std::make_unique<Qwen3Decoder>(backend, std::move(w));
    d->tables = std::make_unique<LanePageTables>(backend, lanes, c.decoder.layers, blocks);
    const ops::Linear &linear = d->decoder->linear();
    const Qwen3Weights &dw = d->decoder->weights();
    const uint32_t blockRows = kMaximumSpeculativeLanes * c.blockSize;
    // A block row is a sequence of its own in the visibility tables.
    d->block = std::make_unique<Qwen3Arena>(backend, linear, dw, blockRows, blockRows, OMNI_MAXIMUM_LANES, false);
    d->context = std::make_unique<Qwen3Arena>(backend, linear, dw, SPLASH_PREFILL_TOKEN_BUDGET, lanes, 1, false);
    memoryBytes_ += d->block->bytes() + d->context->bytes();
    const auto width = static_cast<uint64_t>(c.targetLayers.size()) * config.decoder.hidden * 2;
    // Whole tiles past a chunk's last row: the fc projection reads its input
    // in tiles (ops::omni::densePaddedRows).
    d->captured = zeroed(backend, uint64_t{SPLASH_PREFILL_TOKEN_BUDGET + 128} * width, memoryBytes_, "asr dflash capture");
    for (size_t j = 0; j < c.targetLayers.size(); ++j) {
      const uint64_t offset = j * config.decoder.hidden * 2;
      d->slots.push_back(backend.view(d->captured, offset, d->captured.sizeBytes() - offset));
    }
    d->capture = {d->slots, {}, d->config.targetLayers, SPLASH_PREFILL_TOKEN_BUDGET}; // rowIndex below
    d->blockTokens = zeroed(backend, uint64_t{blockRows} * 4, memoryBytes_, "asr dflash block tokens");
    dflash_ = std::move(d);
    // A round proposes up to block - 1 tokens per lane, as few as its lanes
    // and riders leave rows for in the verify arena (dflashRound).
    draftTokens_ = std::clamp<uint32_t>(engineConfig.draftTokens ? engineConfig.draftTokens : c.blockSize - 1, 1,
                                        c.blockSize - 1);
    verifyRows_ = OMNI_MAXIMUM_LANES;
    riders_ = true;
    const uint32_t speculating = engineConfig.speculativeLanes ? engineConfig.speculativeLanes : kMaximumSpeculativeLanes;
    engineConfig_.speculativeLanes = std::min({speculating, kMaximumSpeculativeLanes, verifyRows_ / 2, lanes});
  }
  mel_ = zeroed(backend, uint64_t{maxMelFrames_} * kMelBins * 4, memoryBytes_, "asr mel");
  tokens_ = zeroed(backend, uint64_t{SPLASH_PREFILL_TOKEN_BUDGET} * 4, memoryBytes_, "asr tokens");
  // The identity over every audio row, so a prefill chunk can gather any
  // range of the audio embeddings.
  const uint32_t indexRows = std::max<uint32_t>(maxAudioTokens_, SPLASH_PREFILL_TOKEN_BUDGET);
  rowIndex_ = zeroed(backend, uint64_t{indexRows} * 4, memoryBytes_, "asr row index");
  for (uint32_t i = 0; i < indexRows; ++i) at<uint32_t>(rowIndex_)[i] = i;
  if (dflash_) dflash_->capture.rowIndex = rowIndex_;
  stepTokens_ = zeroed(backend, uint64_t{lanes} * 4, memoryBytes_, "asr step tokens");
  sampled_ = zeroed(backend, uint64_t{lanes} * 4, memoryBytes_, "asr sampled");
  chainedSampled_ = zeroed(backend, uint64_t{lanes} * 4, memoryBytes_, "asr chained sampled");
  draftInputs_ = zeroed(backend, OMNI_MAXIMUM_LANES * 4, memoryBytes_, "asr draft inputs");
  proposals_ = zeroed(backend, OMNI_MAXIMUM_LANES * 4, memoryBytes_, "asr proposals");
  verified_ = zeroed(backend, OMNI_MAXIMUM_LANES * 4, memoryBytes_, "asr verified");
  // The log-mel spectrogram's tables and Accelerate's first GEMM, here
  // rather than in the first request (about 3 ms).
  (void)logMelSpectrogram(std::vector<float>(kMelSampleRate / 10, 0.0F));
  const char *disabled = std::getenv("OMNI_DISABLE_ANE");
  if (engineConfig.neuralEngine && !(disabled && std::string_view(disabled) == "1") && target_->decoder->affine())
    enableNeuralEngine();
  // The KV pool last, from what the budget leaves (kvPagesFor), and the
  // lanes over it.
  engineConfig_.kvPages =
      engineConfig.kvPagesFor ? engineConfig.kvPagesFor(backend.memoryStats().allocatedBytes) : engineConfig.kvPages;
  pool_ = std::make_unique<KvPagePool>(backend, engineConfig_.kvPages);
  memoryBytes_ += uint64_t{engineConfig_.kvPages} * OMNI_PAGE_BYTES;
  LanePageTables *draftTables = draft_ ? draft_->tables.get() : dflash_ ? dflash_->tables.get() : nullptr;
  for (uint32_t i = 0; i < lanes; ++i) lanes_.push_back(std::make_unique<Lane>(*pool_, *target_->tables, draftTables, i));
}

void AsrEngine::enableNeuralEngine() {
  if (const std::optional<std::string> reason = ane::unavailable()) {
    logLine("The Neural Engine is unavailable (", *reason, "); the GPU runs the prefill alone.");
    return;
  }
  logLine("Setting up the Neural Engine FFN split (OMNI_DISABLE_ANE=1 or serve --disable-ane keeps the prefill on "
          "the GPU)");
  const auto start = Clock::now();
  double error = 0.0;
  Qwen3Decoder &decoder = *target_->decoder;
  try {
    error = decoder.enableAneFfn(backend_, asrAneFfnPolicy(), *target_->prefill);
  } catch (const std::exception &failure) {
    logWarning("Neural Engine FFN split not set up (", failure.what(), "); the GPU runs the prefill alone.");
    return;
  }
  const ops::AneFfn &split = *decoder.aneFfn();
  memoryBytes_ += split.allocatedBytes();
  logLine("Neural Engine FFN split: ", split.aneChannels(), " of ", decoder.dims().intermediate,
          " channels of prefill chunks from ", split.minimumRows(), " rows; set up in ", std::fixed,
          std::setprecision(1), millisecondsSince(start) / 1000.0, " s, ", split.allocatedBytes() >> 20,
          " MiB, checked within ", 100.0 * error, "% of its part.");
}

AsrEngine::~AsrEngine() = default;

void AsrEngine::trace(const char *format, ...) {
  char line[512];
  va_list arguments;
  va_start(arguments, format);
  const int length = std::vsnprintf(line, sizeof line, format, arguments);
  va_end(arguments);
  if (length <= 0) return;
  traceText_.append(line, std::min<size_t>(static_cast<size_t>(length), sizeof line - 1));
  traceText_.push_back('\n');
}

void AsrEngine::traceCommand(const char *kind, uint64_t request, uint32_t rows, Clock::time_point build,
                             Clock::time_point submit, Clock::time_point committed, const metal::CommandTiming &timing) {
  trace("asr trace %s id=%llu rows=%u build=%.6f submit=%.6f commit=%.6f gpu0=%.6f gpu1=%.6f woke=%.6f", kind,
        static_cast<unsigned long long>(request), rows, seconds(build), seconds(submit), seconds(committed),
        timing.gpuStartSeconds, timing.gpuEndSeconds, seconds(Clock::now()));
}

metal::CommandTiming AsrEngine::submitTraced(metal::CommandGraph &graph, const char *kind, uint64_t request,
                                             uint32_t rows) {
  const auto submitted = Clock::now();
  metal::CommandTicket ticket = backend_.submitCommandAsync(graph.command());
  const auto committed = Clock::now();
  const metal::CommandTiming timing = ticket.wait();
  if (trace_) traceCommand(kind, request, rows, submitted, submitted, committed, timing);
  return timing;
}

bool AsrEngine::idle() const noexcept { return queue_.empty() && activeLanes() == 0; }

uint32_t AsrEngine::activeLanes() const noexcept {
  uint32_t count = 0;
  for (const auto &lane : lanes_) count += lane->active;
  return count;
}

KvBinding AsrEngine::binding(const Model &model) const {
  return {model.tables->buffer(), model.tables->entries(), pool_->pages(), pool_->pageCount()};
}

KvBinding AsrEngine::dflashBinding() const {
  return {dflash_->tables->buffer(), dflash_->tables->entries(), pool_->pages(), pool_->pageCount()};
}

const HiddenCapture *AsrEngine::capture() const noexcept { return dflash_ ? &dflash_->capture : nullptr; }

// The draft's context from captured rows, contiguous in the capture from
// context.front().captureRow on: fc over each row's captured layer outputs,
// hidden_norm, then each draft layer's keys and values stored at the rows'
// positions in the lanes' draft pages.
void AsrEngine::addDFlashContext(metal::CommandGraph &graph, std::span<const ContextRows> context) {
  if (context.empty()) return;
  DFlash &d = *dflash_;
  const uint32_t first = context.front().captureRow;
  std::vector<ops::omni::AttentionSequence> descriptors;
  uint32_t rows = 0;
  for (const ContextRows &c : context) {
    if (c.captureRow != first + rows || !c.rows) throw std::logic_error("DFlash context rows must be contiguous");
    descriptors.push_back({d.tables->offset(c.lane->index), d.tables->stride(), c.position, c.rows, rows, c.position});
    rows += c.rows;
  }
  const uint32_t hidden = config_.decoder.hidden;
  const uint64_t rowBytes = uint64_t{hidden} * d.config.targetLayers.size() * 2;
  const metal::MetalBuffer input =
      backend_.view(d.captured, first * rowBytes, uint64_t{ops::omni::densePaddedRows(rows)} * rowBytes);
  ops::omni::DenseLinear::add(graph, std::get<ops::omni::DenseWeights>(d.fc), {.input = input, .output = d.context->residual},
                              rows);
  ops::Normalization::addRms(graph, d.context->residual, d.hiddenNorm, d.context->hidden, hidden, rows);
  const ops::omni::AttentionTables tables = ops::omni::attentionTables(descriptors, false);
  Qwen3Decoder::uploadTables(*d.context, tables);
  d.decoder->addKeysValues(graph, *d.context, dflashBinding(), tables);
}

void AsrEngine::submit(AsrRequest request, EventSink &sink) {
  if (request.prefix.empty() || request.suffix.empty()) {
    sink.onError(request.id, "invalid_request", "prefix and suffix tokens are required");
    return;
  }
  if (request.samples.size() > uint64_t{engineConfig_.maxAudioSeconds} * kMelSampleRate) {
    sink.onError(request.id, "invalid_request", "audio exceeds " + std::to_string(engineConfig_.maxAudioSeconds) + " seconds");
    return;
  }
  for (uint32_t token : request.prefix)
    if (token >= config_.vocabulary) {
      sink.onError(request.id, "invalid_request", "prefix token outside the vocabulary");
      return;
    }
  for (uint32_t token : request.suffix)
    if (token >= config_.vocabulary) {
      sink.onError(request.id, "invalid_request", "suffix token outside the vocabulary");
      return;
    }
  if (trace_)
    trace("asr trace received id=%llu read=%.6f queued=%.6f samples=%zu", static_cast<unsigned long long>(request.id),
          seconds(request.received), seconds(Clock::now()), request.samples.size());
  queue_.push_back(std::move(request));
}

void AsrEngine::cancel(uint64_t requestId, EventSink &sink) {
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

void AsrEngine::dropDraft(Lane &lane) {
  lane.draftDropped = lane.draftDropped || lane.drafted;
  lane.drafted = false;
  lane.draftPending.clear();
  lane.features = {};
  if (lane.draftKv) lane.draftKv->release();
}

void AsrEngine::finish(Lane &lane, DoneReason reason, EventSink &sink) {
  if (!lane.active) return;
  lane.active = false;
  lane.kv.release();
  dropDraft(lane);
  DoneTimings timings;
  timings.promptTokens = lane.promptTokens;
  timings.completion = lane.generated;
  const auto now = Clock::now();
  timings.prefillMicros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(lane.prefillDone - lane.started).count());
  timings.decodeMicros = static_cast<uint64_t>(lane.decodeMicros.count());
  timings.wallMicros = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - lane.started).count());
  lane.request.samples.clear();
  lane.request.samples.shrink_to_fit();
  sink.onDone(lane.request.id, reason, timings);
  if (trace_) {
    trace("asr trace done id=%llu tokens=%u written=%.6f", static_cast<unsigned long long>(lane.request.id),
          lane.generated, seconds(Clock::now()));
    (void)!::write(STDERR_FILENO, traceText_.data(), traceText_.size());
    traceText_.clear();
  }
}

bool AsrEngine::emit(Lane &lane, std::vector<uint32_t> tokens, EventSink &sink) {
  // Through the first stop token, and no further than the lane's limit.
  bool stop = false;
  for (size_t i = 0; i < tokens.size(); ++i)
    if (std::find(config_.stopTokens.begin(), config_.stopTokens.end(), tokens[i]) != config_.stopTokens.end()) {
      tokens.resize(i + 1);
      stop = true;
      break;
    }
  if (tokens.size() > lane.maxTokens - lane.generated) {
    tokens.resize(lane.maxTokens - lane.generated);
    stop = false;
  }
  // How often each token was followed by a stop token (chainable()).
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i == 0 && lane.tokens.empty()) continue;
    StopCounts &counts = followers_[i ? tokens[i - 1] : lane.tokens.back()];
    ++counts.followed;
    counts.stopped += stop && i + 1 == tokens.size();
  }
  const uint32_t offset = lane.generated;
  lane.generated += static_cast<uint32_t>(tokens.size());
  lane.tokens.insert(lane.tokens.end(), tokens.begin(), tokens.end());
  sink.onTokens(lane.request.id, offset, tokens);
  if (stop) finish(lane, DoneReason::Stop, sink);
  else if (lane.generated >= lane.maxTokens) finish(lane, DoneReason::Length, sink);
  return lane.active;
}

bool AsrEngine::tick(EventSink &sink) {
  bool worked = false;
  if (!queue_.empty()) {
    if (!admitBatch(sink)) admit(sink);
    worked = true;
  }
  if (activeLanes()) {
    // Lanes speculate while few are active; above that, every lane steps
    // plainly and the drafts fall behind (dropping out past a few tokens).
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - loadTime_).count();
    load_ += (1.0 - std::exp(-seconds / kLoadSeconds)) * (activeLanes() - load_);
    loadTime_ = now;
    if (dflash_) {
      // Lanes the DFlash draft follows round together while rounds save
      // time over plain steps; the others ride.
      uint32_t followed = 0;
      for (const auto &lane : lanes_) followed += lane->active && lane->drafted;
      const auto riders = activeLanes() - followed;
      const bool rounds = followed && followed <= engineConfig_.speculativeLanes &&
                          followed * 2 + riders <= verifyRows_ &&
                          dflashRound(followed, std::min(draftTokens_, (verifyRows_ - riders) / followed - 1)).second > 0;
      if (rounds) {
        if (pendingStep_) completeStep(sink);
        speculateDFlash(sink);
      } else {
        step(sink);
      }
      return true;
    }
    bool speculating = draft_ && activeLanes() <= engineConfig_.speculativeLanes && speculationPays();
    if (speculating) {
      // The step in flight completes before the draft takes lanes (the
      // lanes' tokens must be current), but only then: a lane the draft
      // will not take keeps its steps queued behind each other.
      if (load_ <= engineConfig_.speculativeLanes + kLoadMargin && !draftCandidates().empty()) {
        if (pendingStep_) completeStep(sink);
        adoptDrafts(draftCandidates());
      }
      uint32_t drafted = 0, riders = 0;
      for (const auto &lane : lanes_)
        if (lane->active) ++(lane->drafted ? drafted : riders);
      speculating = drafted && (riders_ || !riders);
    }
    if (speculating) {
      if (pendingStep_) completeStep(sink);
      speculate(sink);
    } else {
      step(sink);
    }
    worked = true;
  } else if (pendingStep_) {
    // A step queued past the last lane's stop token, completed once the
    // engine has nothing else to do.
    completeStep(sink);
    worked = true;
  }
  return worked;
}

void AsrEngine::addEmbedTokens(metal::CommandGraph &graph, const Model &model, metal::MetalBuffer tokens,
                               metal::MetalBuffer output, uint32_t rows, uint32_t tokenStride) {
  const auto &embedding = model.weights.embedding;
  if (std::holds_alternative<ops::EmbeddingWeights>(embedding))
    ops::Embedding::add(graph, std::move(tokens), std::get<ops::EmbeddingWeights>(embedding), std::move(output), rows,
                        tokenStride);
  else
    ops::omni::Embedding::addGather(graph, std::move(tokens), std::get<metal::MetalBuffer>(embedding), std::move(output),
                                    rows, model.config.vocabulary, model.config.decoder.hidden, 1, tokenStride);
}

void AsrEngine::addPromptRows(metal::CommandGraph &graph, const Model &model, Qwen3Arena &arena,
                              const AsrRequest &request, uint32_t audioTokens, uint32_t audioOffset, uint32_t first,
                              uint32_t rows, uint32_t outputRow, std::span<const uint32_t> emitted) {
  const uint32_t hidden = model.config.decoder.hidden;
  const auto prefixCount = static_cast<uint32_t>(request.prefix.size());
  const auto hiddenRows = [&](uint32_t row, uint32_t count) {
    return backend_.view(arena.hidden, uint64_t{row} * hidden * 2, uint64_t{count} * hidden * 2);
  };
  // The prompt's rows, in order: prefix tokens, audio rows, suffix tokens,
  // then the emitted tokens.
  const auto suffixEnd = static_cast<uint32_t>(prefixCount + audioTokens + request.suffix.size());
  uint32_t filled = 0;
  while (filled < rows) {
    const uint32_t position = first + filled, row = outputRow + filled;
    uint32_t count = 0;
    if (position < prefixCount || position >= prefixCount + audioTokens) {
      const bool prefix = position < prefixCount, suffix = !prefix && position < suffixEnd;
      const std::span<const uint32_t> tokens = prefix ? std::span<const uint32_t>(request.prefix)
                                               : suffix ? std::span<const uint32_t>(request.suffix)
                                                        : emitted;
      const uint32_t index = prefix ? position : suffix ? position - prefixCount - audioTokens : position - suffixEnd;
      count = std::min(rows - filled, static_cast<uint32_t>(tokens.size()) - index);
      std::memcpy(at<uint32_t>(tokens_) + row, tokens.data() + index, count * 4);
      addEmbedTokens(graph, model, backend_.view(tokens_, uint64_t{row} * 4, uint64_t{count} * 4), hiddenRows(row, count),
                     count);
    } else {
      const uint32_t audioIndex = position - prefixCount;
      count = std::min(rows - filled, audioTokens - audioIndex);
      ops::omni::Embedding::addGatherRows(
          graph, backend_.view(rowIndex_, uint64_t{audioOffset + audioIndex} * 4, uint64_t{count} * 4), model.audioRows,
          hiddenRows(row, count), count, audioOffset + audioTokens, hidden);
    }
    filled += count;
  }
}

// The prompt tokens of a request: its prefix, its audio's tower rows and
// its suffix.
static uint32_t promptTokensOf(const AsrRequest &request) {
  const uint32_t frames = melFrames(request.samples.size());
  const uint32_t audio = frames ? ops::omni::audioTokens(frames) : 0;
  return static_cast<uint32_t>(request.prefix.size() + audio + request.suffix.size());
}

// A request's output is at most 64 tokens and 16 per second of its audio:
// speech transcribes to about 3 a second (Mandarin about 5), so a transcript
// ends well within it, and a decoding loop stops there rather than at the
// engine's maximum.
constexpr uint32_t kOutputTokens = 64, kOutputTokensPerSecond = 16;
static uint32_t outputLimit(uint64_t samples, uint32_t maxTokens) {
  // A request without audio is no transcript: the engine's maximum.
  if (!samples) return maxTokens;
  const uint64_t limit = kOutputTokens + (kOutputTokensPerSecond * samples + kMelSampleRate - 1) / kMelSampleRate;
  return static_cast<uint32_t>(std::min<uint64_t>(limit, maxTokens));
}
static uint32_t requestMaxTokens(const AsrRequest &request, uint32_t engineMaxTokens) {
  const uint32_t asked = request.maxTokens ? std::min(request.maxTokens, engineMaxTokens) : engineMaxTokens;
  return std::min(asked, outputLimit(request.samples.size(), engineMaxTokens));
}
// The positions a lane holds at most: its prompt, its output and the rows of
// a round past its last token (the slack the page tables keep).
constexpr uint32_t kRoundRows = 2 * OMNI_MAXIMUM_LANES;
static uint32_t lanePages(uint32_t tokens, uint32_t layers) {
  return (tokens + kRoundRows + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS * layers;
}
static uint32_t requestPages(const AsrRequest &request, uint32_t engineMaxTokens, uint32_t layers) {
  return lanePages(promptTokensOf(request) + requestMaxTokens(request, engineMaxTokens), layers);
}

uint32_t AsrEngine::minimumPages(const AsrConfig &config, const AsrEngineConfig &engineConfig,
                                 uint32_t draftLayers) noexcept {
  const uint64_t samples = uint64_t{engineConfig.maxAudioSeconds} * kMelSampleRate;
  // The chat prefix and suffix take at most 256 positions (maxContext).
  const uint32_t tokens = ops::omni::audioTokens(engineConfig.maxAudioSeconds * (kMelSampleRate / kMelHop)) + 256 +
                          outputLimit(samples, engineConfig.maxTokens);
  return lanePages(tokens, config.decoder.layers + draftLayers);
}

// The features of a lane the draft may take later: none without a draft or
// for audio longer than the draft's tower takes.
void AsrEngine::keepFeatures(Lane &lane, std::vector<float> features) {
  if (draft_ && melFrames(lane.request.samples.size()) <= draft_->maxMelFrames) lane.features = std::move(features);
  else lane.features = {};
}

// The clips that pack (ops::omni::packableClip) share tower passes of up to
// the tower's most tokens, their features side by side in the mel buffer;
// any other clip runs alone. Every clip's rows are the bits of its own
// encode. A pass is a command: the tower's workspace and attention tables
// serve one at a time.
std::vector<uint32_t> AsrEngine::encodeAudio(Model &model, std::span<const TowerClip> clips, double &gpuSeconds) {
  ops::omni::AudioEncoder &encoder = *model.encoder;
  const uint64_t rowBytes = uint64_t{model.config.decoder.hidden} * 2, frameBytes = uint64_t{kMelBins} * 4;
  std::vector<uint32_t> offsets(clips.size(), 0);
  uint32_t rows = 0;
  const auto reserve = [&](size_t i) {
    offsets[i] = rows;
    rows += ops::omni::audioTokens(clips[i].frames);
    if (rows * rowBytes > model.audioRows.sizeBytes()) throw std::logic_error("encoded audio exceeds the audio rows");
  };
  const auto submit = [&](metal::CommandGraph &graph) {
    gpuSeconds += submitTraced(graph, "towers", 0, static_cast<uint32_t>(clips.size())).gpuSeconds;
  };
  std::vector<ops::omni::AudioClip> pass;
  uint32_t passRow = 0, passTokens = 0, passFrames = 0;
  const auto flush = [&] {
    if (pass.empty()) return;
    metal::CommandGraph graph;
    encoder.addEncode(graph, pass, backend_.view(model.audioRows, passRow * rowBytes, passTokens * rowBytes));
    submit(graph);
    pass.clear();
    passRow = rows;
    passTokens = passFrames = 0;
  };
  for (size_t i = 0; i < clips.size(); ++i) {
    const uint32_t frames = clips[i].frames, tokens = frames ? ops::omni::audioTokens(frames) : 0;
    if (!frames || !ops::omni::packableClip(frames)) continue;
    if (passTokens + tokens > encoder.maximumTokens() || passFrames + frames > maxMelFrames_) flush();
    reserve(i);
    std::memcpy(at<float>(mel_) + uint64_t{passFrames} * kMelBins, clips[i].features.data(), frames * frameBytes);
    pass.push_back({backend_.view(mel_, passFrames * frameBytes, frames * frameBytes), frames});
    passTokens += tokens;
    passFrames += frames;
  }
  flush();
  for (size_t i = 0; i < clips.size(); ++i) {
    const uint32_t frames = clips[i].frames;
    if (!frames || ops::omni::packableClip(frames)) continue;
    reserve(i);
    std::memcpy(mel_.contents(), clips[i].features.data(), frames * frameBytes);
    metal::CommandGraph graph;
    encoder.addEncode(graph, mel_, frames,
                      backend_.view(model.audioRows, offsets[i] * rowBytes, (rows - offsets[i]) * rowBytes));
    submit(graph);
  }
  return offsets;
}

// The tokens a lane's audio likely has left (a request without audio may
// run to its limit), what the draft's tower and prefill for it would cost,
// and whether the draft may take it at all.
double AsrEngine::remainingTokens(const Lane &lane) const noexcept {
  const auto samples = lane.request.samples.size();
  const double likely = samples ? kTokensPerAudioSecond * double(samples) / kMelSampleRate : lane.maxTokens;
  return std::max(0.0, std::min<double>(likely, lane.maxTokens) - lane.generated);
}
double AsrEngine::adoptionMilliseconds(const Lane &lane) const noexcept {
  return kAdoptMilliseconds + kAdoptMillisecondsPerRow * (lane.promptTokens + lane.generated);
}
bool AsrEngine::adoptable(const Lane &lane) const noexcept {
  return !lane.draftDropped && melFrames(lane.request.samples.size()) <= draft_->maxMelFrames;
}

// Where lanes ride, whether rounds of the active lanes are worth their
// drafts: what a token saves in rounds of that many lanes (savedPerToken,
// every lane speculating) over the tokens the lanes likely have left
// against the towers and prefills of the lanes the draft would still take.
// Lanes it follows already cost nothing more, so rounds go on once they
// have started. Without riders each lane is weighed alone (adoptDrafts).
bool AsrEngine::speculationPays() const noexcept {
  if (!riders_) return true;
  const uint32_t active = activeLanes();
  double remaining = 0.0, cost = 0.0;
  for (const auto &lane : lanes_) {
    if (!lane->active || !(lane->drafted || adoptable(*lane))) continue;
    remaining += remainingTokens(*lane);
    if (!lane->drafted) cost += adoptionMilliseconds(*lane);
  }
  return remaining * savedPerToken(std::max(active, 1u)) >= cost;
}

// While few lanes are active (and, where lanes ride, rounds pay:
// speculationPays), the draft takes each active lane it does not
// follow, unless it stopped following it before, its audio exceeds the
// draft tower's, or the tokens likely left would not repay the draft's
// tower and prefill at what rounds of the active lanes save a token;
// without riders, only when it can take every such lane (one it cannot
// keeps every lane from speculating).
std::vector<AsrEngine::Lane *> AsrEngine::draftCandidates() const {
  std::vector<Lane *> candidates;
  bool blocked = false;
  const double saved = riders_ ? savedMilliseconds_ : savedPerToken(std::max(activeLanes(), 1u));
  for (const auto &lane : lanes_) {
    if (!lane->active || lane->drafted) continue;
    if (!adoptable(*lane) || remainingTokens(*lane) * saved < adoptionMilliseconds(*lane)) blocked = true;
    else candidates.push_back(lane.get());
  }
  if (blocked && !riders_) candidates.clear();
  return candidates;
}

void AsrEngine::adoptDrafts(std::span<Lane *const> candidates) {
  if (candidates.empty()) return;
  uint32_t rows = 0;
  for (Lane *lane : candidates) rows += lane->promptTokens + lane->generated - 1;
  if (candidates.size() > 1 && rows <= SPLASH_PREFILL_TOKEN_BUDGET) adoptDraftsTogether(candidates);
  else
    for (Lane *lane : candidates) adoptDraft(*lane);
}

// The draft's tower over the lane's audio, then its prefill of the prompt
// and the tokens the lane emitted before its pending one, in chunks; the
// pending token comes with the next round. A lane whose draft gets no KV
// pages is not taken.
void AsrEngine::adoptDraft(Lane &lane) {
  Model &draft = *draft_;
  const AsrRequest &request = lane.request;
  const auto start = Clock::now();
  const uint32_t frames = melFrames(request.samples.size());
  const uint32_t audioTokens = frames ? ops::omni::audioTokens(frames) : 0;
  const std::span<const uint32_t> emitted = std::span<const uint32_t>(lane.tokens).first(lane.tokens.size() - 1);
  const uint32_t rows = lane.promptTokens + static_cast<uint32_t>(emitted.size());
  if (!lane.draftKv->reserve(rows + 1)) {
    lane.draftKv->release();
    lane.draftDropped = true;
    ++stats_.dropped;
    return;
  }
  double gpu = 0.0;
  if (audioTokens) {
    if (lane.features.empty()) lane.features = logMelSpectrogram(request.samples);
    std::memcpy(mel_.contents(), lane.features.data(), lane.features.size() * 4);
    lane.features = {};
    metal::CommandGraph graph;
    draft.encoder->addEncode(graph, mel_, frames, draft.audioRows);
    gpu += submitTraced(graph, "draft-tower", request.id, frames).gpuSeconds;
  }
  for (uint32_t done = 0; done < rows; done += SPLASH_PREFILL_TOKEN_BUDGET) {
    const uint32_t chunk = std::min<uint32_t>(SPLASH_PREFILL_TOKEN_BUDGET, rows - done);
    metal::CommandGraph graph;
    addPromptRows(graph, draft, *draft.prefill, request, audioTokens, 0, done, chunk, 0, emitted);
    const std::vector<ops::omni::AttentionSequence> descriptors{
        {draft.tables->offset(lane.index), draft.tables->stride(), done, chunk, 0, done}};
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
    Qwen3Decoder::uploadTables(*draft.prefill, attention);
    draft.decoder->addForward(graph, *draft.prefill, binding(draft), {&attention, 0});
    gpu += submitTraced(graph, "draft-prefill", request.id, chunk).gpuSeconds;
  }
  lane.drafted = true;
  lane.draftCommitted = rows;
  lane.draftPending = {lane.lastToken};
  ++stats_.adopted;
  if (debug_)
    std::fprintf(stderr, "asr draft adopted a lane: %u frames, %u rows, gpu %.2f ms, wall %.2f ms\n", frames, rows,
                 gpu * 1e3, millisecondsSince(start));
}

// The draft's towers over the lanes' audio, packed, then one ragged prefill
// of each lane's prompt and the tokens it emitted before its pending one
// (their rows fit one chunk). A lane whose draft gets no KV pages is not
// taken.
void AsrEngine::adoptDraftsTogether(std::span<Lane *const> candidates) {
  Model &draft = *draft_;
  const auto start = Clock::now();
  std::vector<Lane *> lanes;
  std::vector<TowerClip> clips;
  for (Lane *lane : candidates) {
    if (!lane->draftKv->reserve(lane->promptTokens + lane->generated)) {
      lane->draftKv->release();
      lane->draftDropped = true;
      ++stats_.dropped;
      continue;
    }
    const uint32_t frames = melFrames(lane->request.samples.size());
    if (frames && lane->features.empty()) lane->features = logMelSpectrogram(lane->request.samples);
    lanes.push_back(lane);
    clips.push_back({lane->features, frames});
  }
  if (lanes.empty()) return;
  double gpu = 0.0;
  const std::vector<uint32_t> audioOffsets = encodeAudio(draft, clips, gpu);
  metal::CommandGraph graph;
  std::vector<ops::omni::AttentionSequence> descriptors;
  uint32_t row = 0;
  for (size_t i = 0; i < lanes.size(); ++i) {
    Lane &lane = *lanes[i];
    lane.features = {};
    const std::span<const uint32_t> emitted = std::span<const uint32_t>(lane.tokens).first(lane.tokens.size() - 1);
    const uint32_t rows = lane.promptTokens + static_cast<uint32_t>(emitted.size());
    const uint32_t audioTokens = clips[i].frames ? ops::omni::audioTokens(clips[i].frames) : 0;
    addPromptRows(graph, draft, *draft.prefill, lane.request, audioTokens, audioOffsets[i], 0, rows, row, emitted);
    descriptors.push_back({draft.tables->offset(lane.index), draft.tables->stride(), 0, rows, row, 0});
    lane.drafted = true;
    lane.draftCommitted = rows;
    lane.draftPending = {lane.lastToken};
    ++stats_.adopted;
    row += rows;
  }
  const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
  Qwen3Decoder::uploadTables(*draft.prefill, attention);
  draft.decoder->addForward(graph, *draft.prefill, binding(draft), {&attention, 0});
  gpu += submitTraced(graph, "draft-prefill", lanes.front()->request.id, row).gpuSeconds;
  if (debug_)
    std::fprintf(stderr, "asr draft adopted %zu lanes together: %u rows, gpu %.2f ms, wall %.2f ms\n", lanes.size(),
                 row, gpu * 1e3, millisecondsSince(start));
}

bool AsrEngine::admitBatch(EventSink &sink) {
  // The queue's head and the requests after it while free lanes remain and
  // their prompts fit one chunk; a single request takes admit()'s path.
  std::vector<Lane *> free;
  for (auto &lane : lanes_)
    if (!lane->active) free.push_back(lane.get());
  uint32_t total = 0, pages = 0;
  size_t count = 0;
  while (count < queue_.size() && count < free.size()) {
    const uint32_t tokens = promptTokensOf(queue_[count]);
    if (total + tokens > SPLASH_PREFILL_TOKEN_BUDGET) break;
    // Each takes the pages it may need to its end now (admit).
    pages += requestPages(queue_[count], engineConfig_.maxTokens, config_.decoder.layers);
    if (pages > pool_->freePages()) break;
    total += tokens;
    ++count;
  }
  if (count < 2) return false;
  if (pendingStep_) completeStep(sink);
  Model &target = *target_;
  struct Admitted {
    Lane *lane;
    uint32_t audioTokens, audioOffset, rowOffset;
  };
  std::vector<Admitted> admitted;
  // Features are kept until the towers have read them.
  std::vector<std::vector<float>> features;
  std::vector<TowerClip> clips;
  features.reserve(count);
  uint32_t rowOffset = 0;
  for (size_t i = 0; i < count; ++i) {
    Lane &lane = *free[i];
    lane.request = std::move(queue_.front());
    queue_.pop_front();
    const AsrRequest &request = lane.request;
    lane.started = Clock::now();
    const uint32_t frames = melFrames(request.samples.size());
    const uint32_t audioTokens = frames ? ops::omni::audioTokens(frames) : 0;
    lane.promptTokens = static_cast<uint32_t>(request.prefix.size() + audioTokens + request.suffix.size());
    lane.committed = 0;
    lane.generated = 0;
    lane.maxTokens = requestMaxTokens(request, engineConfig_.maxTokens);
    lane.finished = false;
    lane.decodeMicros = std::chrono::microseconds{0};
    lane.tokens.clear();
    lane.draftDropped = false;
    if (!lane.kv.reserve(lane.promptTokens + lane.maxTokens + kRoundRows)) {
      sink.onError(request.id, "engine_busy", "no KV pages left");
      lane.kv.release();
      continue;
    }
    lane.active = true;
    // A DFlash draft follows the lane from its prompt, where it has pages.
    lane.drafted = dflash_ && lane.draftKv->reserve(lane.promptTokens + dflash_->config.blockSize);
    if (dflash_ && !lane.drafted) {
      lane.draftKv->release();
      lane.draftDropped = true;
      ++stats_.dropped;
    }
    sink.onStart(request.id, lane.index);
    features.push_back(audioTokens ? logMelSpectrogram(request.samples) : std::vector<float>{});
    if (trace_)
      trace("asr trace admit id=%llu start=%.6f mel=%.6f frames=%u", static_cast<unsigned long long>(request.id),
            seconds(lane.started), seconds(Clock::now()), frames);
    clips.push_back({features.back(), audioTokens ? frames : 0});
    admitted.push_back({&lane, audioTokens, 0, rowOffset});
    rowOffset += lane.promptTokens;
  }
  if (admitted.empty()) return true;
  // The towers in one packed pass, into the batch's audio rows.
  const auto towerStart = Clock::now();
  double towerGpu = 0.0;
  const std::vector<uint32_t> audioOffsets = encodeAudio(target, clips, towerGpu);
  for (size_t i = 0; i < admitted.size(); ++i) {
    admitted[i].audioOffset = audioOffsets[i];
    keepFeatures(*admitted[i].lane, std::move(features[i]));
  }
  if (debug_)
    std::fprintf(stderr, "asr batched towers: %zu requests, gpu %.2f ms, wall %.2f ms\n", admitted.size(),
                 towerGpu * 1e3, millisecondsSince(towerStart));
  const uint32_t rows = rowOffset;
  const auto requests = static_cast<uint32_t>(admitted.size());
  const auto encode = [&](metal::CommandGraph &graph, ops::AneFfn *split) {
    std::vector<ops::omni::AttentionSequence> descriptors;
    for (uint32_t i = 0; i < requests; ++i) {
      const Admitted &a = admitted[i];
      addPromptRows(graph, target, *target.prefill, a.lane->request, a.audioTokens, a.audioOffset, 0,
                    a.lane->promptTokens, a.rowOffset);
      descriptors.push_back({target.tables->offset(a.lane->index), target.tables->stride(), 0, a.lane->promptTokens,
                             a.rowOffset, 0});
      at<uint32_t>(target.prefill->logitRows)[i] = a.rowOffset + a.lane->promptTokens - 1;
    }
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
    Qwen3Decoder::uploadTables(*target.prefill, attention);
    ForwardStep forward{&attention, requests, split};
    forward.capture = capture();
    target.decoder->addForward(graph, *target.prefill, binding(target), forward);
    ops::omni::Sampling::addArgmax(graph, target.prefill->logits, sampled_, requests, config_.vocabulary,
                                   config_.decoder.headRows);
    // The draft's context, a command part per run of lanes it follows.
    std::vector<ContextRows> run;
    for (uint32_t i = 0; dflash_ && i <= requests; ++i) {
      if (i < requests && admitted[i].lane->drafted) {
        run.push_back({admitted[i].lane, 0, admitted[i].lane->promptTokens, admitted[i].rowOffset});
        continue;
      }
      addDFlashContext(graph, run);
      run.clear();
    }
  };
  const auto build = Clock::now();
  ops::AneFfn *split = target.decoder->aneFfnFor(rows);
  if (split && !split->begin()) split = nullptr;
  metal::CommandGraph graph;
  encode(graph, split);
  const auto submitted = Clock::now();
  metal::CommandTicket ticket = split ? split->commit(graph) : backend_.submitCommandAsync(graph.command());
  const auto committed = Clock::now();
  metal::CommandTiming timing = ticket.wait();
  if (trace_) traceCommand("batched-prefill", admitted.front().lane->request.id, rows, build, submitted, committed, timing);
  if (split && !split->finish()) {
    metal::CommandGraph again;
    encode(again, nullptr);
    timing = backend_.submitCommandAsync(again.command()).wait();
  }
  if (debug_)
    std::fprintf(stderr, "asr batched prefill: %u requests, %u rows, gpu %.2f ms, wall %.2f ms%s\n", requests, rows,
                 timing.gpuSeconds * 1e3, timing.wallSeconds * 1e3, split ? " (Neural Engine split)" : "");
  const auto now = Clock::now();
  for (uint32_t i = 0; i < requests; ++i) {
    Lane &lane = *admitted[i].lane;
    lane.committed = lane.promptTokens;
    if (dflash_ && lane.drafted) lane.draftCommitted = lane.promptTokens;
    lane.prefillDone = now;
    const uint32_t token = at<const uint32_t>(sampled_)[i];
    lane.lastToken = token;
    (void)emit(lane, {token}, sink);
  }
  return true;
}

// One request: features, the tower, then the prompt in prefill chunks of
// up to 2048 rows, each a command; its first token comes from the last.
void AsrEngine::admit(EventSink &sink) {
  Lane *free = nullptr;
  for (auto &lane : lanes_)
    if (!lane->active) {
      free = lane.get();
      break;
    }
  if (!free) return;
  // The pages the request may need to its end, taken at admission so that no
  // admitted request runs out of them: it waits for them while others run,
  // and is refused only where the pool cannot hold it at all.
  if (requestPages(queue_.front(), engineConfig_.maxTokens, config_.decoder.layers) > pool_->freePages()) {
    if (activeLanes()) return;
    const AsrRequest refused = std::move(queue_.front());
    queue_.pop_front();
    sink.onError(refused.id, "engine_busy",
                 "the request's audio needs more KV memory than this server's --max-memory leaves");
    return;
  }
  // Features and audio tokens. Audio shorter than one hop yields no frames;
  // such a request has no audio rows. The CPU computes the features while
  // a step queued past the last request's end may still run, which then
  // completes before KV pages change hands.
  const auto started = Clock::now();
  std::vector<float> features = logMelSpectrogram(queue_.front().samples);
  if (pendingStep_) completeStep(sink);
  Lane &lane = *free;
  lane.request = std::move(queue_.front());
  queue_.pop_front();
  const AsrRequest &request = lane.request;
  Model &target = *target_;
  lane.started = started;
  const uint32_t frames = melFrames(request.samples.size());
  const uint32_t audioTokens = frames ? ops::omni::audioTokens(frames) : 0;
  const uint32_t promptTokens = static_cast<uint32_t>(request.prefix.size() + audioTokens + request.suffix.size());
  lane.promptTokens = promptTokens;
  lane.committed = 0;
  lane.generated = 0;
  lane.maxTokens = requestMaxTokens(request, engineConfig_.maxTokens);
  lane.finished = false;
  lane.decodeMicros = std::chrono::microseconds{0};
  lane.tokens.clear();
  lane.draftDropped = false;
  if (!lane.kv.reserve(promptTokens + lane.maxTokens + kRoundRows)) {
    sink.onError(request.id, "engine_busy", "no KV pages left");
    lane.kv.release();
    return;
  }
  lane.active = true;
  lane.drafted = dflash_ && lane.draftKv->reserve(promptTokens + dflash_->config.blockSize);
  if (dflash_ && !lane.drafted) {
    lane.draftKv->release();
    lane.draftDropped = true;
    ++stats_.dropped;
  }
  sink.onStart(request.id, lane.index);
  const auto melDone = Clock::now();
  if (trace_)
    trace("asr trace admit id=%llu start=%.6f mel=%.6f frames=%u", static_cast<unsigned long long>(request.id),
          seconds(lane.started), seconds(melDone), frames);
  if (audioTokens) {
    std::memcpy(mel_.contents(), features.data(), features.size() * 4);
    keepFeatures(lane, std::move(features));
    const auto build = Clock::now();
    metal::CommandGraph graph;
    target.encoder->addEncode(graph, mel_, frames, target.audioRows);
    const auto submitted = Clock::now();
    metal::CommandTicket ticket = backend_.submitCommandAsync(graph.command());
    const auto committed = Clock::now();
    const metal::CommandTiming timing = ticket.wait();
    if (trace_) traceCommand("tower", request.id, audioTokens, build, submitted, committed, timing);
    if (debug_)
      std::fprintf(stderr, "asr admit: mel %.2f ms, encoder gpu %.2f ms, wall %.2f ms (%u frames, %u tokens)\n",
                   std::chrono::duration<double, std::milli>(melDone - lane.started).count(), timing.gpuSeconds * 1e3,
                   timing.wallSeconds * 1e3, frames, audioTokens);
  }
  uint32_t done = 0;
  while (done < promptTokens) {
    const uint32_t rows = std::min<uint32_t>(SPLASH_PREFILL_TOKEN_BUDGET, promptTokens - done);
    const bool last = done + rows == promptTokens;
    // The chunk's command, its FFN split with the Neural Engine or on the
    // GPU alone.
    const auto encode = [&](metal::CommandGraph &graph, ops::AneFfn *split) {
      addPromptRows(graph, target, *target.prefill, request, audioTokens, 0, done, rows, 0);
      std::vector<ops::omni::AttentionSequence> descriptors{
          {target.tables->offset(lane.index), target.tables->stride(), done, rows, 0, done}};
      const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
      Qwen3Decoder::uploadTables(*target.prefill, attention);
      at<uint32_t>(target.prefill->logitRows)[0] = rows - 1;
      ForwardStep forward{&attention, last ? 1u : 0u, split};
      forward.capture = capture();
      target.decoder->addForward(graph, *target.prefill, binding(target), forward);
      if (dflash_ && lane.drafted) {
        const ContextRows context{&lane, done, rows, 0};
        addDFlashContext(graph, {&context, 1});
      }
      if (last)
        ops::omni::Sampling::addArgmax(graph, target.prefill->logits, sampled_, 1, config_.vocabulary,
                                       config_.decoder.headRows);
    };
    // A split whose Neural Engine work failed leaves the command's results
    // unusable and stops; the chunk runs again on the GPU alone, which every
    // later chunk runs too. The chunk writes its KV rows and hidden rows
    // again from the embeddings.
    const auto build = Clock::now();
    ops::AneFfn *split = target.decoder->aneFfnFor(rows);
    if (split && !split->begin()) split = nullptr;
    metal::CommandGraph graph;
    encode(graph, split);
    const auto submitted = Clock::now();
    metal::CommandTicket ticket = split ? split->commit(graph) : backend_.submitCommandAsync(graph.command());
    const auto committed = Clock::now();
    metal::CommandTiming timing = ticket.wait();
    if (trace_) traceCommand("prefill", request.id, rows, build, submitted, committed, timing);
    bool retried = false;
    if (split && !split->finish()) {
      metal::CommandGraph again;
      encode(again, nullptr);
      timing = backend_.submitCommandAsync(again.command()).wait();
      retried = true;
    }
    if (debug_)
      std::fprintf(stderr, "asr prefill: %u rows, %zu dispatches, gpu %.2f ms, wall %.2f ms%s\n", rows,
                   graph.dispatches().size(), timing.gpuSeconds * 1e3, timing.wallSeconds * 1e3,
                   retried ? " (Neural Engine failed; GPU alone)"
                   : split ? (", Neural Engine " + std::to_string(split->aneChannels()) + " channels").c_str()
                           : "");
    done += rows;
  }
  lane.committed = promptTokens;
  if (dflash_ && lane.drafted) lane.draftCommitted = promptTokens;
  lane.prefillDone = Clock::now();
  const uint32_t token = at<const uint32_t>(sampled_)[0];
  lane.lastToken = token;
  (void)emit(lane, {token}, sink);
}

// A plain decode step's command from submission until its tokens are
// emitted: its lanes in row order, their requests and KV positions, and the
// buffer its argmax writes the rows' tokens to.
struct AsrEngine::PlainStep final {
  metal::CommandTicket ticket;
  std::vector<uint32_t> lanes;
  std::vector<uint64_t> requests;
  std::vector<uint32_t> positions;
  metal::MetalBuffer sampled;
  // Rows 0..followed-1: lanes a DFlash draft follows, whose rows become its context.
  uint32_t followed = 0;
  Clock::time_point build, submitted, committed;
  size_t dispatches = 0;
};

// Whether the next step can queue behind the step in flight: the same lanes
// are active, none of them can reach its limit with the step in flight's
// token, no request waits for admission (which needs the GPU's results
// first), and some lane will likely go on. A lane's stop token in the step
// in flight makes the next step's row for it wasted work, whose KV row lands
// past its end; when every lane's is wasted, the whole step (about 4.8 ms of
// a 4-bit 1.7B) delays the next request's admission. A lane likely stops
// when the token the step in flight takes (its last) was followed by a stop
// token more than a fifth of the time so far (a sentence's last period on
// English LibriSpeech: about 0.7); skipping the queue costs about 0.3 ms.
bool AsrEngine::chainable() const noexcept {
  if (!pendingStep_ || !queue_.empty()) return false;
  const PlainStep &step = *pendingStep_;
  if (activeLanes() != step.lanes.size()) return false;
  bool continues = false;
  for (size_t r = 0; r < step.lanes.size(); ++r) {
    const Lane &lane = *lanes_[step.lanes[r]];
    if (!lane.active || lane.request.id != step.requests[r] || lane.generated + 2 > lane.maxTokens) return false;
    const auto counts = followers_.find(lane.lastToken);
    continues = continues || counts == followers_.end() || 5 * counts->second.stopped <= counts->second.followed + 1;
  }
  return continues;
}

// Encodes and submits a step: of every active lane from its last token, or,
// behind `previous`, of the same lanes one position on, their tokens read on
// the GPU from where `previous`'s argmax writes them. A lane without KV for
// its row finishes (null when no lane steps); a chained step without KV for
// a row is not submitted.
std::unique_ptr<AsrEngine::PlainStep> AsrEngine::submitStep(EventSink &sink, const PlainStep *previous) {
  Model &target = *target_;
  auto step = std::make_unique<PlainStep>();
  step->build = Clock::now();
  std::vector<ops::omni::AttentionSequence> descriptors;
  const auto add = [&](const Lane &lane, uint32_t position) {
    const auto r = static_cast<uint32_t>(step->lanes.size());
    descriptors.push_back({target.tables->offset(lane.index), target.tables->stride(), position, 1, r, position});
    step->lanes.push_back(lane.index);
    step->requests.push_back(lane.request.id);
    step->positions.push_back(position);
  };
  // With a DFlash draft, the lanes it follows step first: their captured
  // rows, 0 to followed - 1, become its context in the same command.
  uint32_t followed = 0;
  if (previous) {
    for (size_t r = 0; r < previous->lanes.size(); ++r) {
      Lane &lane = *lanes_[previous->lanes[r]];
      const uint32_t position = previous->positions[r] + 1;
      if (!lane.kv.reserve(position + 1)) return nullptr;
      if (r < previous->followed && !lane.draftKv->reserve(position + dflash_->config.blockSize)) return nullptr;
      add(lane, position);
    }
    followed = previous->followed;
    step->sampled = previous->sampled.sameView(sampled_) ? chainedSampled_ : sampled_;
  } else {
    const auto stepLane = [&](Lane &lane) {
      at<uint32_t>(stepTokens_)[step->lanes.size()] = lane.lastToken;
      add(lane, lane.committed);
    };
    std::vector<Lane *> others;
    for (auto &lanePointer : lanes_) {
      Lane &lane = *lanePointer;
      if (!lane.active) continue;
      if (!lane.kv.reserve(lane.committed + 1)) {
        finish(lane, DoneReason::Length, sink);
        continue;
      }
      if (dflash_ && lane.drafted && !lane.draftKv->reserve(lane.committed + dflash_->config.blockSize)) {
        dropDraft(lane);
        ++stats_.dropped;
      }
      if (dflash_ && lane.drafted) {
        stepLane(lane);
        ++followed;
      } else {
        others.push_back(&lane);
      }
    }
    for (Lane *lane : others) stepLane(*lane);
    step->sampled = sampled_;
  }
  step->followed = followed;
  if (step->lanes.empty()) return nullptr;
  const auto rows = static_cast<uint32_t>(step->lanes.size());
  metal::CommandGraph graph;
  addEmbedTokens(graph, target, previous ? previous->sampled : stepTokens_, target.decode->hidden, rows);
  const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, true);
  Qwen3Decoder::uploadTables(*target.decode, attention);
  for (uint32_t r = 0; r < rows; ++r) at<uint32_t>(target.decode->logitRows)[r] = r;
  ForwardStep forward{&attention, rows};
  if (followed) forward.capture = capture();
  target.decoder->addForward(graph, *target.decode, binding(target), forward);
  ops::omni::Sampling::addArgmax(graph, target.decode->logits, step->sampled, rows, config_.vocabulary,
                                 config_.decoder.headRows);
  std::vector<ContextRows> context;
  for (uint32_t r = 0; r < followed; ++r) context.push_back({lanes_[step->lanes[r]].get(), step->positions[r], 1, r});
  addDFlashContext(graph, context);
  step->dispatches = graph.dispatches().size();
  step->submitted = Clock::now();
  step->ticket = previous ? backend_.submitCommandBehind(graph.dispatches()) : backend_.submitCommandAsync(graph.command());
  step->committed = Clock::now();
  return step;
}

// Waits for the step in flight and emits its rows' tokens; a row of a lane
// that ended meanwhile (a stop token in the step before, or a cancel) is
// dropped.
void AsrEngine::completeStep(EventSink &sink) {
  const std::unique_ptr<PlainStep> step = std::move(pendingStep_);
  const metal::CommandTiming timing = step->ticket.wait();
  const auto now = Clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(now - std::max(step->submitted, lastStepDone_));
  lastStepDone_ = now;
  const auto rows = static_cast<uint32_t>(step->lanes.size());
  if (trace_) traceCommand("step", step->requests[0], rows, step->build, step->submitted, step->committed, timing);
  if (debug_)
    std::fprintf(stderr, "asr step: %u rows, %zu dispatches, gpu %.2f ms, wall %.2f ms\n", rows, step->dispatches,
                 timing.gpuSeconds * 1e3, elapsed.count() / 1e3);
  for (uint32_t r = 0; r < rows; ++r) {
    Lane &lane = *lanes_[step->lanes[r]];
    if (!lane.active || lane.request.id != step->requests[r]) {
      ++wastedRows_;
      continue;
    }
    lane.committed = step->positions[r] + 1;
    lane.decodeMicros += elapsed;
    const uint32_t token = at<const uint32_t>(step->sampled)[r];
    lane.lastToken = token;
    // The step's row is the DFlash draft's context already.
    if (dflash_ && lane.drafted) lane.draftCommitted = lane.committed;
    if (!emit(lane, {token}, sink) || !lane.drafted || dflash_) continue;
    // The draft falls behind by the token; past a few it stops following.
    lane.draftPending.push_back(token);
    if (lane.draftPending.size() > kMaximumDraftPending) {
      dropDraft(lane);
      ++stats_.dropped;
    }
  }
  if (trace_) trace("asr trace emitted t=%.6f", seconds(Clock::now()));
}

// A token for every active lane. While the same lanes keep decoding, the
// next step is submitted behind the one in flight before its tokens are
// read, so the GPU starts it as soon as the step before ends instead of
// after the host has read, emitted and encoded (docs/omni-performance.md,
// round 10); a step queued past a lane's stop token runs to no use, and
// the engine completes it before KV pages change hands.
void AsrEngine::step(EventSink &sink) {
  if (pendingStep_ && !chainable()) completeStep(sink);
  if (!pendingStep_) {
    pendingStep_ = submitStep(sink, nullptr);
    if (!pendingStep_) return;
  }
  std::unique_ptr<PlainStep> next = chainable() ? submitStep(sink, pendingStep_.get()) : nullptr;
  completeStep(sink);
  pendingStep_ = std::move(next);
}

// One speculative round of the lanes the draft follows, in one command:
// (1) the draft takes each lane's pending tokens (its catch-up, ragged) and
// proposes from the last; (2) k - 1 draft decode steps propose the rest,
// each step's tokens read from where the previous step's argmax wrote them;
// (3) the target verifies [pending, d1..dk] of every lane in one forward of
// the verify arena (decode projections, ragged attention), taking
// every row's greedy token. A lane accepts the proposals that match the
// target's tokens and emits them and the target's token after them (its
// correction, or a bonus token when all match). Rollback is only the
// committed counts: KV rows past them are written again later. Active lanes
// the draft does not follow ride in the verify forward with their pending
// token alone, a plain step's token, so they keep pace with the round.
void AsrEngine::speculate(EventSink &sink) {
  const auto build = Clock::now();
  Model &target = *target_, &draft = *draft_;
  std::vector<Lane *> round, riders;
  for (auto &lane : lanes_)
    if (lane->active) (lane->drafted ? round : riders).push_back(lane.get());
  if (round.empty()) return;
  // Every lane verifies at least two rows: n + riders <= speculativeLanes
  // <= verifyRows_ / 2.
  const uint32_t k = roundDraftTokens(static_cast<uint32_t>(round.size()), static_cast<uint32_t>(riders.size()));
  // KV for the round's rows: the target's k + 1 (a rider's one), the
  // draft's catch-up and k - 1 steps. A lane whose draft has no pages left
  // rides along.
  std::vector<Lane *> lanes, plain;
  for (Lane *lane : round) {
    if (!lane->kv.reserve(lane->committed + k + 1)) {
      finish(*lane, DoneReason::Length, sink);
      continue;
    }
    const auto pending = static_cast<uint32_t>(lane->draftPending.size());
    if (!lane->draftKv->reserve(lane->draftCommitted + pending + k - 1)) {
      dropDraft(*lane);
      ++stats_.dropped;
      plain.push_back(lane);
      continue;
    }
    lanes.push_back(lane);
  }
  for (Lane *lane : riders) {
    if (!lane->kv.reserve(lane->committed + 1)) finish(*lane, DoneReason::Length, sink);
    else plain.push_back(lane);
  }
  if (lanes.empty()) {
    if (!plain.empty()) step(sink);
    return;
  }
  const auto count = static_cast<uint32_t>(lanes.size());
  const uint32_t stride = k + 1;
  const uint32_t draftVocabulary = draft.config.vocabulary;
  metal::CommandGraph graph;

  // (1) The draft's catch-up: a decode step when every lane has one pending
  // token, else a ragged verify-arena forward; its last rows propose d1.
  std::vector<ops::omni::AttentionSequence> descriptors;
  uint32_t rows = 0;
  bool single = true;
  for (uint32_t i = 0; i < count; ++i) {
    const Lane &lane = *lanes[i];
    const auto pending = static_cast<uint32_t>(lane.draftPending.size());
    std::memcpy(at<uint32_t>(draftInputs_) + rows, lane.draftPending.data(), pending * 4);
    descriptors.push_back({draft.tables->offset(lane.index), draft.tables->stride(), lane.draftCommitted, pending, rows,
                           lane.draftCommitted});
    rows += pending;
    single = single && pending == 1;
  }
  Qwen3Arena &catchUp = single ? *draft.decode : *draft.verify;
  addEmbedTokens(graph, draft, draftInputs_, catchUp.hidden, rows);
  const ops::omni::AttentionTables catchUpTables = ops::omni::attentionTables(descriptors, single);
  Qwen3Decoder::uploadTables(catchUp, catchUpTables);
  for (uint32_t i = 0; i < count; ++i)
    at<uint32_t>(catchUp.logitRows)[i] = descriptors[i].rowOffset + descriptors[i].rows - 1;
  draft.decoder->addForward(graph, catchUp, binding(draft), {&catchUpTables, count});
  ops::omni::Sampling::addArgmax(graph, catchUp.logits, backend_.view(proposals_, 4, (OMNI_MAXIMUM_LANES - 1) * 4), count,
                                 draftVocabulary, draft.config.decoder.headRows, stride);

  // (2) Draft steps j = 2..k, each lane's input the proposal before.
  for (uint32_t i = 0; i < count; ++i) at<uint32_t>(draft.decode->logitRows)[i] = i;
  for (uint32_t j = 2; j <= k; ++j) {
    std::vector<ops::omni::AttentionSequence> steps;
    for (uint32_t i = 0; i < count; ++i) {
      const Lane &lane = *lanes[i];
      const uint32_t position = lane.draftCommitted + static_cast<uint32_t>(lane.draftPending.size()) + j - 2;
      steps.push_back({draft.tables->offset(lane.index), draft.tables->stride(), position, 1, i, position});
    }
    addEmbedTokens(graph, draft, backend_.view(proposals_, uint64_t{j - 1} * 4, uint64_t{OMNI_MAXIMUM_LANES - j + 1} * 4),
                   draft.decode->hidden, count, stride);
    const ops::omni::AttentionTables stepTables = ops::omni::attentionTables(steps, true);
    Qwen3Decoder::uploadTables(*draft.decode, stepTables);
    draft.decoder->addForward(graph, *draft.decode, binding(draft), {&stepTables, count});
    ops::omni::Sampling::addArgmax(graph, draft.decode->logits,
                                   backend_.view(proposals_, uint64_t{j} * 4, uint64_t{OMNI_MAXIMUM_LANES - j} * 4), count,
                                   draftVocabulary, draft.config.decoder.headRows, stride);
  }

  // (3) The target's verification of every lane's k + 1 rows, then the
  // riders' rows.
  std::vector<ops::omni::AttentionSequence> verify;
  for (uint32_t i = 0; i < count; ++i) {
    const Lane &lane = *lanes[i];
    at<uint32_t>(proposals_)[i * stride] = lane.lastToken;
    verify.push_back({target.tables->offset(lane.index), target.tables->stride(), lane.committed, stride, i * stride,
                      lane.committed});
  }
  const uint32_t riderRow = count * stride;
  for (uint32_t p = 0; p < plain.size(); ++p) {
    const Lane &lane = *plain[p];
    at<uint32_t>(proposals_)[riderRow + p] = lane.lastToken;
    verify.push_back({target.tables->offset(lane.index), target.tables->stride(), lane.committed, 1, riderRow + p,
                      lane.committed});
  }
  const uint32_t verifyRows = riderRow + static_cast<uint32_t>(plain.size());
  addEmbedTokens(graph, target, proposals_, target.verify->hidden, verifyRows);
  const ops::omni::AttentionTables verifyTables = ops::omni::attentionTables(verify, false);
  Qwen3Decoder::uploadTables(*target.verify, verifyTables);
  for (uint32_t r = 0; r < verifyRows; ++r) at<uint32_t>(target.verify->logitRows)[r] = r;
  target.decoder->addForward(graph, *target.verify, binding(target), {&verifyTables, verifyRows});
  ops::omni::Sampling::addArgmax(graph, target.verify->logits, verified_, verifyRows, config_.vocabulary,
                                 config_.decoder.headRows);

  const auto t0 = Clock::now();
  metal::CommandTicket ticket = backend_.submitCommandAsync(graph.command());
  const auto committed = Clock::now();
  const metal::CommandTiming timing = ticket.wait();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0);
  if (trace_) traceCommand("round", lanes.front()->request.id, verifyRows, build, t0, committed, timing);
  uint32_t acceptedTotal = 0;
  for (uint32_t i = 0; i < count; ++i) {
    Lane &lane = *lanes[i];
    const uint32_t *proposed = at<const uint32_t>(proposals_) + i * stride; // [pending, d1..dk]
    const uint32_t *greedy = at<const uint32_t>(verified_) + i * stride;    // the target's token after each
    uint32_t accepted = 0;
    while (accepted < k && proposed[accepted + 1] == greedy[accepted]) ++accepted;
    std::vector<uint32_t> tokens(proposed + 1, proposed + 1 + accepted);
    tokens.push_back(greedy[accepted]);
    ++stats_.rounds;
    stats_.proposed += k;
    stats_.accepted += accepted;
    stats_.emitted += tokens.size();
    ++stats_.acceptedRounds[accepted];
    acceptedTotal += accepted;
    // Depth j was verified when the proposals before it held.
    for (uint32_t j = 1; j < reached_.size(); ++j) {
      reached_[j] *= kAcceptanceDecay;
      survived_[j] *= kAcceptanceDecay;
      if (j <= k && accepted + 1 >= j) {
        reached_[j] += 1.0;
        survived_[j] += accepted >= j;
      }
    }
    lane.decodeMicros += elapsed;
    const auto pending = static_cast<uint32_t>(lane.draftPending.size());
    if (!emit(lane, tokens, sink)) continue;
    // The target keeps its pending token and the accepted proposals; the
    // draft keeps its pending tokens and the accepted proposals it took
    // (d1..d(k-1)), then takes what it has not: the target's token, after
    // dk when every proposal held.
    lane.committed += accepted + 1;
    lane.lastToken = greedy[accepted];
    lane.draftCommitted += pending + std::min(accepted, k - 1);
    if (accepted == k) lane.draftPending = {proposed[k], greedy[k]};
    else lane.draftPending = {greedy[accepted]};
  }
  for (uint32_t p = 0; p < plain.size(); ++p) {
    Lane &lane = *plain[p];
    const uint32_t token = at<const uint32_t>(verified_)[riderRow + p];
    lane.decodeMicros += elapsed;
    if (!emit(lane, {token}, sink)) continue;
    lane.committed += 1;
    lane.lastToken = token;
  }
  if (trace_) trace("asr trace emitted t=%.6f", seconds(Clock::now()));
  if (debug_)
    std::fprintf(stderr, "asr speculative round: %u lanes, k %u, %zu riders, %u catch-up rows, accepted %u of %u, "
                 "%zu dispatches, gpu %.2f ms, wall %.2f ms\n", count, k, plain.size(), rows, acceptedTotal, count * k,
                 graph.dispatches().size(), timing.gpuSeconds * 1e3, elapsed.count() / 1e3);
}

// One DFlash round of every active lane the draft follows, in one command:
// (1) the draft's forward over each lane's block [pending, mask x (B - 1)]
// at the lane's next positions, the block's keys and values stored past its
// context and every block row seeing the context and the whole block, then
// the target's head over the first k rows after the pending one and their
// argmax: the proposals; (2) the target's verification of [pending,
// d1..dk] of every lane, then the riders' pending tokens, keeping the layer
// outputs the draft reads; (3) the verified rows' context at their
// positions. Rows past what a lane accepts land past its context, which the
// next block or verification writes again: rollback is the committed counts.
void AsrEngine::speculateDFlash(EventSink &sink) {
  const auto build = Clock::now();
  Model &target = *target_;
  DFlash &d = *dflash_;
  const uint32_t block = d.config.blockSize;
  std::vector<Lane *> round, plain;
  for (auto &lane : lanes_)
    if (lane->active) (lane->drafted ? round : plain).push_back(lane.get());
  if (round.empty()) return;
  const auto riders = static_cast<uint32_t>(plain.size());
  const uint32_t limit = std::min(draftTokens_, (verifyRows_ - riders) / static_cast<uint32_t>(round.size()) - 1);
  const uint32_t k = dflashRound(static_cast<uint32_t>(round.size()), std::max(limit, 1u)).first;
  std::vector<Lane *> lanes;
  for (Lane *lane : round) {
    // Without pages for its round, a lane rides with its pending token (the
    // draft no longer follows it); without a page for that row, it ends.
    if (!lane->kv.reserve(lane->committed + k + 1)) {
      dropDraft(*lane);
      ++stats_.dropped;
      plain.push_back(lane);
      continue;
    }
    if (!lane->draftKv->reserve(lane->committed + block)) {
      dropDraft(*lane);
      ++stats_.dropped;
      plain.push_back(lane);
      continue;
    }
    lanes.push_back(lane);
  }
  std::vector<Lane *> riding;
  for (Lane *lane : plain) {
    if (!lane->kv.reserve(lane->committed + 1)) finish(*lane, DoneReason::Length, sink);
    else riding.push_back(lane);
  }
  if (lanes.empty()) {
    if (!riding.empty()) step(sink);
    return;
  }
  const auto count = static_cast<uint32_t>(lanes.size());
  const uint32_t stride = k + 1;
  const uint32_t headRows = config_.decoder.headRows;
  const Qwen3Dims &draftDims = d.decoder->dims();
  metal::CommandGraph graph;

  // (1) The draft's blocks.
  std::vector<ops::omni::AttentionSequence> stored, seen;
  for (uint32_t i = 0; i < count; ++i) {
    const Lane &lane = *lanes[i];
    uint32_t *tokens = at<uint32_t>(d.blockTokens) + i * block;
    tokens[0] = lane.lastToken;
    std::fill(tokens + 1, tokens + block, d.config.maskToken);
    const uint32_t p = lane.committed;
    stored.push_back({d.tables->offset(lane.index), d.tables->stride(), p, block, i * block, p});
    for (uint32_t j = 0; j < block; ++j)
      seen.push_back({d.tables->offset(lane.index), d.tables->stride(), p + block - 1, 1, i * block + j, p + j});
    for (uint32_t j = 1; j <= k; ++j) at<uint32_t>(d.block->logitRows)[i * k + j - 1] = i * block + j;
  }
  addEmbedTokens(graph, target, d.blockTokens, d.block->hidden, count * block);
  const ops::omni::AttentionTables blockTables = ops::omni::attentionTables(stored, false);
  const ops::omni::AttentionTables visibility = ops::omni::attentionTables(seen, false);
  Qwen3Decoder::uploadTables(*d.block, blockTables);
  Qwen3Decoder::uploadVisibility(*d.block, visibility);
  ForwardStep draftForward{&blockTables, count * k};
  // A causal block attends as a prefill chunk does.
  if (!d.config.causal) draftForward.visibility = &visibility;
  d.decoder->addForward(graph, *d.block, dflashBinding(), draftForward);
  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t rowBytes = uint64_t{draftDims.headRows} * 4;
    const metal::MetalBuffer logits = backend_.view(d.block->logits, i * k * rowBytes, k * rowBytes);
    const metal::MetalBuffer output = backend_.view(proposals_, uint64_t{i * stride + 1} * 4, uint64_t{k} * 4);
    if (d.tokenMap)
      ops::omni::Sampling::addArgmaxMapped(graph, logits, output, d.tokenMap, k, draftDims.vocabulary,
                                           draftDims.headRows);
    else
      ops::omni::Sampling::addArgmax(graph, logits, output, k, draftDims.vocabulary, draftDims.headRows);
  }

  // (2) The target's verification, keeping the draft's layer outputs.
  std::vector<ops::omni::AttentionSequence> verify;
  for (uint32_t i = 0; i < count; ++i) {
    const Lane &lane = *lanes[i];
    at<uint32_t>(proposals_)[i * stride] = lane.lastToken;
    verify.push_back({target.tables->offset(lane.index), target.tables->stride(), lane.committed, stride, i * stride,
                      lane.committed});
  }
  const uint32_t riderRow = count * stride;
  for (uint32_t p = 0; p < riding.size(); ++p) {
    const Lane &lane = *riding[p];
    at<uint32_t>(proposals_)[riderRow + p] = lane.lastToken;
    verify.push_back({target.tables->offset(lane.index), target.tables->stride(), lane.committed, 1, riderRow + p,
                      lane.committed});
  }
  const uint32_t verifyRows = riderRow + static_cast<uint32_t>(riding.size());
  addEmbedTokens(graph, target, proposals_, target.verify->hidden, verifyRows);
  const ops::omni::AttentionTables verifyTables = ops::omni::attentionTables(verify, false);
  Qwen3Decoder::uploadTables(*target.verify, verifyTables);
  for (uint32_t r = 0; r < verifyRows; ++r) at<uint32_t>(target.verify->logitRows)[r] = r;
  ForwardStep verifyForward{&verifyTables, verifyRows};
  verifyForward.capture = capture();
  target.decoder->addForward(graph, *target.verify, binding(target), verifyForward);
  ops::omni::Sampling::addArgmax(graph, target.verify->logits, verified_, verifyRows, config_.vocabulary, headRows);

  // (3) The verified rows' context.
  std::vector<ContextRows> context;
  for (uint32_t i = 0; i < count; ++i) context.push_back({lanes[i], lanes[i]->committed, stride, i * stride});
  addDFlashContext(graph, context);

  const auto t0 = Clock::now();
  metal::CommandTicket ticket = backend_.submitCommandAsync(graph.command());
  const auto committed = Clock::now();
  const metal::CommandTiming timing = ticket.wait();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0);
  if (trace_) traceCommand("dflash-round", lanes.front()->request.id, verifyRows, build, t0, committed, timing);
  uint32_t acceptedTotal = 0;
  for (uint32_t i = 0; i < count; ++i) {
    Lane &lane = *lanes[i];
    const uint32_t *proposed = at<const uint32_t>(proposals_) + i * stride; // [pending, d1..dk]
    const uint32_t *greedy = at<const uint32_t>(verified_) + i * stride;    // the target's token after each
    uint32_t accepted = 0;
    while (accepted < k && proposed[accepted + 1] == greedy[accepted]) ++accepted;
    std::vector<uint32_t> tokens(proposed + 1, proposed + 1 + accepted);
    tokens.push_back(greedy[accepted]);
    ++stats_.rounds;
    stats_.proposed += k;
    stats_.accepted += accepted;
    stats_.emitted += tokens.size();
    ++stats_.acceptedRounds[accepted];
    acceptedTotal += accepted;
    for (uint32_t j = 1; j < reached_.size(); ++j) {
      reached_[j] *= kAcceptanceDecay;
      survived_[j] *= kAcceptanceDecay;
      if (j <= k && accepted + 1 >= j) {
        reached_[j] += 1.0;
        survived_[j] += accepted >= j;
      }
    }
    lane.decodeMicros += elapsed;
    if (!emit(lane, tokens, sink)) continue;
    lane.committed += accepted + 1;
    lane.draftCommitted = lane.committed;
    lane.lastToken = greedy[accepted];
  }
  for (uint32_t p = 0; p < riding.size(); ++p) {
    Lane &lane = *riding[p];
    const uint32_t token = at<const uint32_t>(verified_)[riderRow + p];
    lane.decodeMicros += elapsed;
    if (!emit(lane, {token}, sink)) continue;
    lane.committed += 1;
    lane.lastToken = token;
  }
  if (trace_) trace("asr trace emitted t=%.6f", seconds(Clock::now()));
  if (debug_)
    std::fprintf(stderr, "asr dflash round: %u lanes, k %u, %zu riders, accepted %u of %u, %zu dispatches, gpu %.2f ms, "
                 "wall %.2f ms\n", count, k, riding.size(), acceptedTotal, count * k, graph.dispatches().size(),
                 timing.gpuSeconds * 1e3, elapsed.count() / 1e3);
}

} // namespace splash::omni
