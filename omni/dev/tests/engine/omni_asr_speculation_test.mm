// The ASR engine's speculative decoding (runtime/omni/AsrEngine) at a small
// shape, in both projection formats: requests without audio (prompt tokens
// only) decoded greedily by a plain engine are the reference, and engines
// with a draft must emit exactly the same tokens: with the target as its
// own draft (every proposal accepted), with a draft that disagrees on every
// fourth token (rejections at every position of a round, and full
// acceptance), across KV page boundaries, with a stop token inside rounds,
// token limits shorter than a round, lanes that stop speculating while too
// many are active and catch up after, and cancellations; and with a DFlash
// draft (testDFlash).
//
// The models are decisive: random layers of small residual weight under a
// head that maps each token to its successor in a permutation of the
// vocabulary, by a margin far above the bf16 noise between a decode step
// and a verify forward (whose numerics differ: a random model's near-ties
// would flip; omni_decoder_test checks the verify forward's logits, the
// real-weight runs its transcripts). argv[1] names the production metallib.

#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "omni/AsrEngine.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::omni;
using namespace splash::test;
using splash::ops::tuning::floatToBf16;

namespace {

constexpr uint32_t kHidden = 256, kIntermediate = 512, kLayers = 2, kHeadRows = 512, kVocabulary = 500;
constexpr uint32_t kQkv = OMNI_DECODER_QKV_WIDTH, kAttn = OMNI_DECODER_ATTENTION_WIDTH, kD = OMNI_DECODER_HEAD_DIM;

std::vector<float> randomValues(std::mt19937 &rng, size_t count, float bound) {
  std::vector<float> v(count);
  for (float &x : v) x = std::uniform_real_distribution<float>(-bound, bound)(rng);
  return v;
}

// The token a model predicts after `token`: a permutation of the
// vocabulary, which the draft follows except after a pseudo-random fourth
// of the tokens (a multiplicative hash's top two bits).
uint32_t successor(uint32_t token, bool draft) {
  const uint32_t next = (token * 7 + 13) % kVocabulary;
  return draft && (token * 2654435761u) >> 30 == 0 ? (next + 1) % kVocabulary : next;
}

metal::MetalBuffer bf16Buffer(metal::MetalBackend &backend, const std::vector<float> &values) {
  metal::MetalBuffer buffer = sharedBuffer(backend, values.size() * 2);
  auto *bits = static_cast<uint16_t *>(buffer.contents());
  for (size_t i = 0; i < values.size(); ++i) bits[i] = floatToBf16(values[i]);
  return buffer;
}

// MLX affine Q4 (group 64) in the kernels' 256-row storage tiles.
ops::Projection affineOf(metal::MetalBackend &backend, const std::vector<float> &values, uint32_t n, uint32_t k) {
  const uint32_t groups = k / 64;
  const uint64_t parameters = uint64_t{n} * groups;
  ops::Projection p(n, k, ops::AffineWeights{sharedBuffer(backend, parameters * 32), sharedBuffer(backend, parameters * 2),
                                             sharedBuffer(backend, parameters * 2)});
  auto *codes = static_cast<uint8_t *>(p.affine().weights.contents());
  auto *scales = static_cast<uint16_t *>(p.affine().scales.contents());
  auto *biases = static_cast<uint16_t *>(p.affine().biases.contents());
  std::memset(codes, 0, parameters * 32);
  for (uint32_t row = 0; row < n; ++row)
    for (uint32_t g = 0; g < groups; ++g) {
      const float *group = &values[size_t(row) * k + g * 64];
      const auto [low, high] = std::minmax_element(group, group + 64);
      const float scale = std::max((*high - *low) / 15.0F, 1e-7F);
      const uint64_t at = (uint64_t(row / 256) * groups + g) * 256 + row % 256;
      scales[at] = floatToBf16(scale);
      biases[at] = floatToBf16(*low);
      for (uint32_t i = 0; i < 64; ++i) {
        const int q = std::clamp(int(std::lround((group[i] - *low) / scale)), 0, 15);
        codes[at * 32 + i / 2] |= uint8_t(q << (4 * (i % 2)));
      }
    }
  return p;
}

AsrConfig testConfig(std::vector<uint32_t> stopTokens) {
  AsrConfig config;
  config.decoder = {kLayers, kHidden, kIntermediate, kHeadRows, kVocabulary, 1e-6F, 1e6};
  config.vocabulary = kVocabulary;
  config.audioStart = 1;
  config.audioEnd = 2;
  config.audioPad = 3;
  config.stopTokens = std::move(stopTokens);
  config.encoder = {128, 1, 128, kHidden};
  return config;
}

// The target's or the draft's decoder: the same random layers, whose output
// and down projections add a tenth of a standard layer to the residual, and
// a head whose row for v sums the embeddings of the tokens v succeeds.
AsrTextWeights decisiveText(metal::MetalBackend &backend, bool affine, bool draft) {
  std::mt19937 rng(5);
  AsrTextWeights w;
  w.decoder.dims = testConfig({}).decoder;
  const auto project = [&](const std::vector<float> &values, uint32_t n, uint32_t k) -> Qwen3Projection {
    if (affine) return affineOf(backend, values, n, k);
    return ops::omni::DenseWeights{bf16Buffer(backend, values), {}, n, k};
  };
  const auto random = [&](uint32_t n, uint32_t k, float scale) {
    return project(randomValues(rng, size_t(n) * k, scale / std::sqrt(float(k))), n, k);
  };
  const auto norm = [&](uint32_t width) {
    std::vector<float> values = randomValues(rng, width, 0.5F);
    for (float &v : values) v += 1.0F;
    return ops::NormWeights{bf16Buffer(backend, values)};
  };
  for (uint32_t l = 0; l < kLayers; ++l) {
    Qwen3LayerWeights layer;
    layer.inputNorm = norm(kHidden);
    layer.postAttentionNorm = norm(kHidden);
    layer.queryNorm = norm(kD).buffer;
    layer.keyNorm = norm(kD).buffer;
    layer.qkv = random(kQkv, kHidden, 1.0F);
    layer.output = random(kHidden, kAttn, 0.1F);
    layer.gate = random(kIntermediate, kHidden, 1.0F);
    layer.up = random(kIntermediate, kHidden, 1.0F);
    layer.down = random(kHidden, kIntermediate, 0.1F);
    w.decoder.layers.push_back(std::move(layer));
  }
  std::vector<float> ones(kHidden, 1.0F);
  w.decoder.finalNorm = ops::NormWeights{bf16Buffer(backend, ones)};
  const std::vector<float> embedding = randomValues(rng, size_t(kVocabulary) * kHidden, 1.0F);
  std::vector<float> head(size_t(kHeadRows) * kHidden, 0.0F);
  for (uint32_t token = 0; token < kVocabulary; ++token)
    for (uint32_t i = 0; i < kHidden; ++i)
      head[size_t(successor(token, draft)) * kHidden + i] += 0.05F * embedding[size_t(token) * kHidden + i];
  w.decoder.head = project(head, kHeadRows, kHidden);
  w.embedding = bf16Buffer(backend, embedding);
  return w;
}

// A tower of the right shapes that never runs (the requests carry no audio).
ops::omni::AudioEncoderWeights idleTower(metal::MetalBackend &backend) {
  const ops::omni::AudioEncoderDims dims = testConfig({}).encoder;
  const auto dense = [&](uint32_t n, uint32_t k, bool bias) {
    return ops::omni::DenseWeights{sharedBuffer(backend, uint64_t{n} * k * 2), bias ? sharedBuffer(backend, n * 4) : metal::MetalBuffer{},
                                   n, k};
  };
  const auto norm = [&] { return ops::omni::AudioLayerNorm{sharedBuffer(backend, dims.hidden * 2), sharedBuffer(backend, dims.hidden * 2)}; };
  ops::omni::AudioEncoderWeights w;
  w.dims = dims;
  w.conv1Weight = sharedBuffer(backend, OMNI_AUDIO_CONV_CHANNELS * 9 * 2);
  w.conv1Bias = sharedBuffer(backend, OMNI_AUDIO_CONV_CHANNELS * 4);
  w.conv2 = dense(OMNI_AUDIO_CONV_PADDED_CHANNELS, OMNI_AUDIO_CONV_PADDED_TAPS, true);
  w.conv3 = dense(OMNI_AUDIO_CONV_PADDED_CHANNELS, OMNI_AUDIO_CONV_PADDED_TAPS, true);
  w.convOut = dense(dims.hidden, OMNI_AUDIO_CONV_OUT_INPUT, false);
  ops::omni::AudioEncoderLayerWeights layer{norm(), dense(3 * dims.hidden, dims.hidden, true), dense(dims.hidden, dims.hidden, true),
                                            norm(), dense(dims.ffn, dims.hidden, true), dense(dims.hidden, dims.ffn, true)};
  w.layers.push_back(std::move(layer));
  w.postNorm = norm();
  w.proj1 = dense(dims.hidden, dims.hidden, true);
  w.proj2 = dense(dims.output, dims.hidden, true);
  return w;
}

struct Collector final : EventSink {
  std::map<uint64_t, std::vector<uint32_t>> tokens;
  std::map<uint64_t, DoneReason> done;
  void onStart(uint64_t, uint32_t) override {}
  void onAudio(uint64_t, uint32_t, std::span<const int16_t>) override {}
  void onTokens(uint64_t id, uint32_t offset, std::span<const uint32_t> generated) override {
    require(!done.count(id), "tokens after a request's end");
    require(offset == tokens[id].size(), "token offsets are not contiguous");
    require(!generated.empty(), "an empty token event");
    tokens[id].insert(tokens[id].end(), generated.begin(), generated.end());
  }
  void onDone(uint64_t id, DoneReason reason, const DoneTimings &) override {
    require(!done.count(id), "a request ended twice");
    done[id] = reason;
  }
  void onError(uint64_t id, const std::string &code, const std::string &message) override {
    throw std::runtime_error("request " + std::to_string(id) + " failed: " + code + " " + message);
  }
};

// Prompt lengths around the 32-token KV pages, and token limits from one
// token to more than a page.
struct RequestSpec {
  uint32_t prefix, maxTokens;
};
const std::vector<RequestSpec> kRequests{{29, 100}, {31, 1}, {32, 2}, {33, 12}, {12, 64}, {61, 90}, {3, 3}, {44, 128}};

AsrRequest makeRequest(uint64_t id, const RequestSpec &spec) {
  std::mt19937 rng(static_cast<uint32_t>(1000 + id));
  AsrRequest request;
  request.id = id;
  for (uint32_t i = 0; i < spec.prefix; ++i) request.prefix.push_back(4 + rng() % (kVocabulary - 4));
  request.suffix = {2, 9, 11 + uint32_t(id) * 37};
  request.maxTokens = spec.maxTokens;
  return request;
}

struct Event {
  uint32_t tick;
  bool cancel;
  uint64_t id;
};

// Runs the requests through `engine`, submitting or cancelling each at its
// tick, until every request ended.
Collector run(AsrEngine &engine, const std::vector<Event> &events) {
  Collector sink;
  size_t next = 0;
  for (uint32_t tick = 0; tick < 5000; ++tick) {
    for (; next < events.size() && events[next].tick <= tick; ++next) {
      const Event &e = events[next];
      if (e.cancel) engine.cancel(e.id, sink);
      else engine.submit(makeRequest(e.id, kRequests[e.id - 1]), sink);
    }
    if (next == events.size() && engine.idle()) return sink;
    engine.tick(sink);
  }
  throw std::runtime_error("the engine did not finish");
}

std::vector<Event> allAtOnce() {
  std::vector<Event> events;
  for (uint64_t id = 1; id <= kRequests.size(); ++id) events.push_back({0, false, id});
  return events;
}

struct Fixture {
  metal::MetalBackend &backend;
  bool affine;
  std::vector<uint32_t> stopTokens;

  // An engine without a draft, or with the target as its draft (an exact
  // one) or a draft that disagrees on every fourth token.
  std::unique_ptr<AsrEngine> engine(bool draft, bool exact, uint32_t draftTokens, uint32_t speculativeLanes) {
    AsrEngineConfig config;
    config.lanes = 4;
    config.maxAudioSeconds = 1;
    config.maxTokens = 128;
    config.kvPages = 1024;
    config.neuralEngine = false;
    config.draftTokens = draftTokens;
    config.speculativeLanes = speculativeLanes;
    std::optional<AsrDraftModel> model;
    if (draft) model = AsrDraftModel{testConfig(stopTokens), decisiveText(backend, affine, !exact), idleTower(backend)};
    return std::make_unique<AsrEngine>(backend, testConfig(stopTokens), decisiveText(backend, affine, false),
                                       idleTower(backend), config, std::move(model));
  }
};

std::string describe(const std::vector<uint32_t> &tokens) {
  std::string text;
  for (uint32_t t : tokens) text += " " + std::to_string(t);
  return text;
}

// Every request `actual` ran emitted what it did in `expected`.
void requireSame(const Collector &expected, const Collector &actual, const std::string &what) {
  require(!actual.done.empty(), what + ": no request ran");
  for (const auto &[id, reason] : actual.done) {
    const std::vector<uint32_t> &tokens = expected.tokens.at(id);
    const auto found = actual.tokens.find(id);
    require(found != actual.tokens.end() && found->second == tokens,
            what + ": request " + std::to_string(id) + " emitted" +
                describe(found == actual.tokens.end() ? std::vector<uint32_t>{} : found->second) + ", plain decoding" +
                describe(tokens));
    require(reason == expected.done.at(id), what + ": request " + std::to_string(id) + " ended differently");
  }
}

void testFormat(metal::MetalBackend &backend, bool affine) {
  const std::string format = affine ? "affine Q4" : "dense bf16";
  Fixture fixture{backend, affine, {kVocabulary + 1}}; // no stop token yet
  const Collector plain = run(*fixture.engine(false, false, 0, 0), allAtOnce());
  for (uint64_t id = 1; id <= kRequests.size(); ++id) {
    require(plain.tokens.at(id).size() == kRequests[id - 1].maxTokens, format + ": plain decoding stopped early");
    require(plain.done.at(id) == DoneReason::Length, format + ": plain decoding did not end at its limit");
  }

  // The target's own weights as the draft: every proposal holds.
  {
    std::unique_ptr<AsrEngine> engine = fixture.engine(true, true, 4, 4);
    requireSame(plain, run(*engine, allAtOnce()), format + " identical draft");
    const AsrEngine::SpeculationStats &stats = engine->speculationStats();
    require(stats.rounds > 0 && stats.accepted == stats.proposed,
            format + " identical draft: accepted " + std::to_string(stats.accepted) + " of " + std::to_string(stats.proposed));
  }

  // A draft that disagrees on every fourth token: rounds that reject at
  // every position and that accept every proposal, one lane alone (k = 5)
  // and four together (k shrinks to fit the verify rows).
  for (const uint32_t lanes : {1u, 4u}) {
    std::unique_ptr<AsrEngine> engine = fixture.engine(true, false, 5, 4);
    std::vector<Event> events;
    if (lanes == 1)
      for (uint64_t id = 1; id <= kRequests.size(); ++id) events.push_back({uint32_t(id - 1) * 200, false, id});
    else events = allAtOnce();
    requireSame(plain, run(*engine, events), format + " disagreeing draft, " + std::to_string(lanes) + " lanes");
    const AsrEngine::SpeculationStats &stats = engine->speculationStats();
    std::string histogram;
    for (uint32_t accepted = 0; accepted <= 5; ++accepted) histogram += " " + std::to_string(stats.acceptedRounds[accepted]);
    std::cout << format << " disagreeing draft, " << lanes << " lanes: accepted " << stats.accepted << " of " << stats.proposed
              << ", rounds by accepted proposals" << histogram << ", " << stats.adopted << " lanes drafted\n";
    const uint32_t k = engine->draftTokens();
    if (lanes == 1) {
      for (uint32_t accepted = 0; accepted <= k; ++accepted)
        require(stats.acceptedRounds[accepted] > 0,
                format + ": no round accepted exactly " + std::to_string(accepted) + " proposals:" + histogram);
      // An affine target weighs a lane's draft against the acceptance its
      // rounds observed: the disagreeing draft stops taking lanes that the
      // exact one takes.
      if (affine) {
        std::unique_ptr<AsrEngine> exact = fixture.engine(true, true, 5, 4);
        requireSame(plain, run(*exact, events), format + " identical draft, one lane");
        require(stats.adopted < exact->speculationStats().adopted,
                format + ": the disagreeing draft took as many lanes as the exact one");
      }
    }
  }

  // A stop token the plain decoding emits mid-sequence ends requests inside
  // a round.
  {
    const uint32_t stop = plain.tokens.at(1)[10];
    Fixture stopping{backend, affine, {stop}};
    const Collector stopped = run(*stopping.engine(false, false, 0, 0), allAtOnce());
    require(stopped.done.at(1) == DoneReason::Stop && stopped.tokens.at(1).size() <= 11, format + ": the stop token did not stop");
    requireSame(stopped, run(*stopping.engine(true, false, 5, 4), allAtOnce()), format + " stop token");
  }

  // Two speculating lanes at most: a lane speculating alone, then two short
  // requests make three active, the first steps plainly (its draft falling
  // behind) and catches up once the shorter ends, the other (no draft)
  // riding in its rounds; later a lane speculating while two long requests
  // arrive falls too far behind and its draft drops. The exact draft keeps
  // the affine engine's estimate of what a token saves above the drafts'
  // cost (the disagreeing one accepts too few to repay them).
  {
    std::unique_ptr<AsrEngine> engine = fixture.engine(true, true, 4, 2);
    const std::vector<Event> events{{0, false, 1}, {6, false, 3}, {6, false, 7}, {40, false, 6}, {44, false, 5},
                                    {44, false, 8}};
    requireSame(plain, run(*engine, events), format + " speculative lanes");
    const AsrEngine::SpeculationStats &stats = engine->speculationStats();
    require(stats.dropped > 0, format + ": no draft stopped following a lane");
  }

  // A lane that starts among too many to speculate: once the others end,
  // the draft takes its prompt and the tokens it already emitted.
  {
    std::unique_ptr<AsrEngine> engine = fixture.engine(true, false, 4, 2);
    requireSame(plain, run(*engine, {{0, false, 6}, {0, false, 7}, {0, false, 2}}), format + " late adoption");
    const AsrEngine::SpeculationStats &stats = engine->speculationStats();
    require(stats.adopted >= 1 && stats.rounds > 0, format + ": the draft did not take the lane late");
  }

  // Cancellations: of a speculating lane, of a queued request, and a new
  // request on a cancelled lane.
  {
    std::unique_ptr<AsrEngine> engine = fixture.engine(true, false, 4, 4);
    const std::vector<Event> events{{0, false, 1}, {0, false, 6}, {3, true, 1}, {3, false, 8}, {3, false, 5},
                                    {3, false, 4}, {3, true, 4}, {9, false, 2}};
    const Collector cancelled = run(*engine, events);
    for (uint64_t id : {1, 4}) {
      require(cancelled.done.at(id) == DoneReason::Cancelled, format + ": a cancelled request did not end cancelled");
      const std::vector<uint32_t> &got = cancelled.tokens.count(id) ? cancelled.tokens.at(id) : std::vector<uint32_t>{};
      const std::vector<uint32_t> &all = plain.tokens.at(id);
      require(got.size() < all.size() && std::equal(got.begin(), got.end(), all.begin()),
              format + ": a cancelled request's tokens are not a prefix of plain decoding's");
    }
    for (uint64_t id : {2, 5, 6, 8})
      require(cancelled.tokens.at(id) == plain.tokens.at(id), format + ": request " + std::to_string(id) +
                                                                   " differs after a cancellation");
  }
  std::cout << format << " speculative decoding ok\n";
}

// A DFlash draft of random dense layers reading both target layers' outputs:
// with the target's head (a bidirectional block, as z-lab's format), or with
// a head of its own over `draftVocabulary` tokens that a map sends into the
// target's (a causal block, as Speculators'). Its proposals mostly miss.
AsrDFlashDraft randomDFlash(metal::MetalBackend &backend, bool causal, uint32_t draftVocabulary) {
  std::mt19937 rng(11);
  AsrDFlashDraft draft;
  DFlashDraftConfig &c = draft.config;
  c.decoder = {kLayers, kHidden, kIntermediate, draftVocabulary, draftVocabulary, 1e-6F, 1e6};
  c.targetLayers = {0, 1};
  c.blockSize = 8;
  c.maskToken = kVocabulary - 1;
  c.causal = causal;
  c.draftVocabulary = draftVocabulary;
  const auto dense = [&](uint32_t n, uint32_t k, float scale) -> Qwen3Projection {
    return ops::omni::DenseWeights{bf16Buffer(backend, randomValues(rng, size_t(n) * k, scale / std::sqrt(float(k)))), {}, n, k};
  };
  const auto norm = [&](uint32_t width) {
    std::vector<float> values = randomValues(rng, width, 0.5F);
    for (float &v : values) v += 1.0F;
    return ops::NormWeights{bf16Buffer(backend, values)};
  };
  DFlashDraftWeights &w = draft.weights;
  w.decoder.dims = c.decoder;
  for (uint32_t l = 0; l < kLayers; ++l) {
    Qwen3LayerWeights layer;
    layer.inputNorm = norm(kHidden);
    layer.postAttentionNorm = norm(kHidden);
    layer.queryNorm = norm(kD).buffer;
    layer.keyNorm = norm(kD).buffer;
    layer.qkv = dense(kQkv, kHidden, 1.0F);
    layer.output = dense(kHidden, kAttn, 0.5F);
    layer.gate = dense(kIntermediate, kHidden, 1.0F);
    layer.up = dense(kIntermediate, kHidden, 1.0F);
    layer.down = dense(kHidden, kIntermediate, 0.5F);
    w.decoder.layers.push_back(std::move(layer));
  }
  w.decoder.finalNorm = norm(kHidden);
  w.fc = dense(kHidden, kHidden * 2, 1.0F);
  w.hiddenNorm = norm(kHidden);
  if (draftVocabulary) {
    w.decoder.head = dense(draftVocabulary, kHidden, 1.0F);
    for (uint32_t i = 0; i < draftVocabulary; ++i) w.tokenMap.push_back((i * 7 + 3) % kVocabulary);
  }
  return draft;
}

// Engines with a DFlash draft emit what plain decoding does: its rounds
// reject at most positions and accept some, roll the target's and the
// draft's context back by their counts, append the context of prompts,
// plain steps and verified rows at the right positions across page
// boundaries, and stop inside a round.
void testDFlash(metal::MetalBackend &backend, bool affine) {
  const std::string format = affine ? "affine Q4" : "dense bf16";
  Fixture fixture{backend, affine, {kVocabulary + 1}};
  const Collector plain = run(*fixture.engine(false, false, 0, 0), allAtOnce());
  const auto engine = [&](const std::vector<uint32_t> &stops, bool causal, uint32_t vocabulary) {
    AsrEngineConfig config;
    config.lanes = 4;
    config.maxAudioSeconds = 1;
    config.maxTokens = 128;
    config.kvPages = 1024;
    config.neuralEngine = false;
    return std::make_unique<AsrEngine>(backend, testConfig(stops), decisiveText(backend, affine, false), idleTower(backend),
                                       config, std::nullopt, randomDFlash(backend, causal, vocabulary));
  };
  std::vector<Event> oneByOne;
  for (uint64_t id = 1; id <= kRequests.size(); ++id) oneByOne.push_back({uint32_t(id - 1) * 200, false, id});
  for (const bool causal : {false, true}) {
    const uint32_t vocabulary = causal ? 192 : 0;
    const std::string what = format + (causal ? " DFlash draft (own head, causal)" : " DFlash draft (target head)");
    for (const std::vector<Event> &events : {allAtOnce(), oneByOne}) {
      std::unique_ptr<AsrEngine> e = engine({kVocabulary + 1}, causal, vocabulary);
      requireSame(plain, run(*e, events), what);
      const AsrEngine::SpeculationStats &stats = e->speculationStats();
      require(e->dflash() && stats.rounds > 0, what + ": no round ran");
      std::cout << what << ": " << stats.rounds << " lane rounds, accepted " << stats.accepted << " of " << stats.proposed
                << "\n";
    }
  }
  const uint32_t stop = plain.tokens.at(1)[10];
  Fixture stopping{backend, affine, {stop}};
  const Collector stopped = run(*stopping.engine(false, false, 0, 0), allAtOnce());
  requireSame(stopped, run(*engine({stop}, true, 192), allAtOnce()), format + " DFlash draft, stop token");
  std::cout << format << " DFlash speculative decoding ok\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: omni-asr-speculation METALLIB\n";
    return 2;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    testFormat(backend, true);
    testFormat(backend, false);
    testDFlash(backend, true);
    testDFlash(backend, false);
  } catch (const std::exception &error) {
    std::cerr << "omni asr speculation: " << error.what() << "\n";
    return 1;
  }
  std::cout << "omni asr speculation ok\n";
  return 0;
}
