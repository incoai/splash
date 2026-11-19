// Real-weight parity of the Omni decoders against mlx-audio
// (dev/tools/omni_parity_reference.py): loads a checkpoint through the Omni
// loader and compares the logits of a fixed prompt.
//   omni-model-parity METALLIB asr|tts|tts-prompt|tts-force MODEL_DIR REFERENCE.bin [OUT.bin]
// tts-prompt takes a reference of mlx-audio's own prompt for a text with an
// instruction, a preset speaker or a language (VoiceDesign, CustomVoice),
// which the engine's prompt assembly (ttsPrompt) must match. tts-force
// teacher-forces a long generation's frames through the engine's step and
// checks the logits at every history length (runTtsForce); OUT.bin takes
// the engine's logits in the reference's format.
// Prefills the prompt as one sequence, then replays its last token as a
// decode step, and compares both against the reference; for ASR, again with
// the prefill FFN split with the Neural Engine (OMNI_DISABLE_ANE=1 skips it).

#include "TestChecks.hpp"
#include "ane/Program.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/WeightImages.hpp"
#include "omni/AsrEngine.hpp"
#include "omni/KvPages.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniLoader.hpp"
#include "omni/Qwen3Decoder.hpp"
#include "omni/TtsEngine.hpp"
#include "omni/TtsModel.hpp"
#include "ops/Embedding.hpp"
#include "ops/Omni.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::omni;
using namespace splash::test;

namespace {

const std::vector<uint32_t> kTokens{151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198,
                                    3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198, 785, 6722, 315,
                                    9625, 374, 12095, 13, 1084, 374, 264, 6662, 3283, 448, 1657, 1251, 13};

struct Reference {
  uint32_t length = 0, vocabulary = 0;
  std::vector<float> last, prefix;
};

Reference readReference(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  Reference r;
  file.read(reinterpret_cast<char *>(&r.length), 4);
  file.read(reinterpret_cast<char *>(&r.vocabulary), 4);
  r.last.resize(r.vocabulary);
  file.read(reinterpret_cast<char *>(r.last.data()), r.vocabulary * 4);
  if (!file) throw std::runtime_error("reference file is short");
  // The second array runs to the end of the file (the ASR prefix logits of
  // V, or the TTS predictor logits of 2048).
  std::vector<char> rest((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  r.prefix.resize(rest.size() / 4);
  std::memcpy(r.prefix.data(), rest.data(), r.prefix.size() * 4);
  return r;
}

void compare(const float *got, const std::vector<float> &expected, uint32_t vocabulary, const std::string &what) {
  double maxError = 0.0, scale = 0.0, dot = 0.0, normA = 0.0, normB = 0.0;
  uint32_t argmaxGot = 0, argmaxExpected = 0;
  for (uint32_t v = 0; v < vocabulary; ++v) {
    maxError = std::max(maxError, std::fabs(double(got[v]) - expected[v]));
    scale = std::max(scale, std::fabs(double(expected[v])));
    dot += double(got[v]) * expected[v];
    normA += double(got[v]) * got[v];
    normB += double(expected[v]) * expected[v];
    if (got[v] > got[argmaxGot]) argmaxGot = v;
    if (expected[v] > expected[argmaxExpected]) argmaxExpected = v;
  }
  const double cosine = dot / std::sqrt(normA * normB);
  std::cout << what << ": max |diff| " << maxError << " (scale " << scale << "), cosine " << cosine << ", argmax "
            << argmaxGot << " vs " << argmaxExpected << "\n";
  require(argmaxGot == argmaxExpected, what + ": argmax differs");
  require(cosine > 0.999, what + ": logits diverge");
  require(maxError <= 0.05 * scale + 0.5, what + ": logits differ beyond bf16 noise");
}

void runAsr(metal::MetalBackend &backend, const std::string &modelDir, const Reference &reference) {
  const OmniConfig config = readOmniConfig(modelDir);
  require(config.kind == ModelKind::Asr, "not an ASR model");
  model::WeightImages images(backend, "omni-parity");
  const auto start = std::chrono::steady_clock::now();
  AsrTextWeights weights = loadAsrTextWeights(backend, images, modelDir, *config.asr, config.quantized);
  std::cout << "weights loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
  const uint32_t hidden = config.asr->decoder.hidden, vocabulary = config.asr->vocabulary;
  require(reference.vocabulary == vocabulary && reference.length == kTokens.size(), "reference shape mismatch");
  Qwen3Decoder decoder(backend, weights.decoder);
  const uint32_t L = static_cast<uint32_t>(kTokens.size());
  KvPagePool pool(backend, 64);
  LanePageTables tables(backend, 1, config.asr->decoder.layers, 4);
  LaneKv lane(pool, tables, 0);
  const KvBinding kv{tables.buffer(), tables.entries(), pool.pages(), pool.pageCount()};
  Qwen3Arena prefillArena(backend, decoder.linear(), weights.decoder, 128, 1, 2, false);
  Qwen3Arena decodeArena(backend, decoder.linear(), weights.decoder, 8, 8, 8, true);

  const auto embed = [&](metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer output, uint32_t rows) {
    if (std::holds_alternative<ops::EmbeddingWeights>(weights.embedding))
      ops::Embedding::add(graph, tokens, std::get<ops::EmbeddingWeights>(weights.embedding), output, rows);
    else
      ops::omni::Embedding::addGather(graph, tokens, std::get<metal::MetalBuffer>(weights.embedding), output, rows,
                                      vocabulary, hidden);
  };
  // Prefill the first L-1 tokens, taking logits at the prefix position and
  // the last prefilled row.
  const uint32_t prefill = L - 1;
  require(lane.reserve(L), "reserve pages");
  // The prompt's prefill, its FFN on the GPU alone or split with the Neural
  // Engine; the logits of its last row.
  const auto prefillOnce = [&](ops::AneFfn *split, const std::string &what) {
    metal::MetalBuffer tokens = backend.allocateBuffer(prefill * 4, metal::BufferStorage::Shared, "tokens");
    std::memcpy(tokens.contents(), kTokens.data(), prefill * 4);
    std::vector<ops::omni::AttentionSequence> descriptors{{tables.offset(0), tables.stride(), 0, prefill, 0, 0}};
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
    Qwen3Decoder::uploadTables(prefillArena, attention);
    auto *logitRows = static_cast<uint32_t *>(prefillArena.logitRows.contents());
    logitRows[0] = L / 2 - 1;
    logitRows[1] = prefill - 1;
    if (split) require(split->begin(), what + ": the split stopped (" + split->reason() + ")");
    metal::CommandGraph graph;
    embed(graph, tokens, prefillArena.hidden, prefill);
    decoder.addForward(graph, prefillArena, kv, {&attention, 2, split});
    const auto t0 = std::chrono::steady_clock::now();
    metal::CommandTicket ticket = split ? split->commit(graph) : backend.submitCommandAsync(graph.command());
    (void)ticket.wait();
    if (split) require(split->finish(), what + ": the Neural Engine's work failed (" + split->reason() + ")");
    std::cout << what << " of " << prefill << " rows: "
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << " ms\n";
    const auto *logits = static_cast<const float *>(prefillArena.logits.contents());
    compare(logits, reference.prefix, vocabulary, what + " prefix logits");
    return std::vector<float>(logits + vocabulary, logits + 2 * vocabulary);
  };
  // A decode step of the last token over the prefill's KV.
  metal::MetalBuffer lastToken = backend.allocateBuffer(4, metal::BufferStorage::Shared, "token");
  std::memcpy(lastToken.contents(), &kTokens[L - 1], 4);
  const auto decodeOnce = [&](const std::string &what) {
    std::vector<ops::omni::AttentionSequence> descriptors{{tables.offset(0), tables.stride(), prefill, 1, 0, prefill}};
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, true);
    Qwen3Decoder::uploadTables(decodeArena, attention);
    static_cast<uint32_t *>(decodeArena.logitRows.contents())[0] = 0;
    metal::CommandGraph graph;
    embed(graph, lastToken, decodeArena.hidden, 1);
    decoder.addForward(graph, decodeArena, kv, {&attention, 1});
    const auto t0 = std::chrono::steady_clock::now();
    metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
    (void)ticket.wait();
    std::cout << "decode step: "
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << " ms\n";
    compare(static_cast<const float *>(decodeArena.logits.contents()), reference.last, vocabulary, what);
  };
  const std::vector<float> gpuLast = prefillOnce(nullptr, "prefill");
  // Decode the last token.
  {
    const metal::MetalBuffer &tokens = lastToken;
    decodeOnce("decode logits");
    // Steady-state decode timing over a few more steps of the same token.
    for (int i = 0; i < 3; ++i) {
      const uint32_t committed = prefill + 1 + i;
      require(lane.reserve(committed + 1), "reserve");
      std::vector<ops::omni::AttentionSequence> more{{tables.offset(0), tables.stride(), committed, 1, 0, committed}};
      const ops::omni::AttentionTables t = ops::omni::attentionTables(more, true);
      Qwen3Decoder::uploadTables(decodeArena, t);
      metal::CommandGraph g;
      embed(g, tokens, decodeArena.hidden, 1);
      decoder.addForward(g, decodeArena, kv, {&t, 1});
      const auto t1 = std::chrono::steady_clock::now();
      metal::CommandTicket ticket2 = backend.submitCommandAsync(g.command());
      (void)ticket2.wait();
      std::cout << "decode step: "
                << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() << " ms\n";
    }
  }
  // The prefill again with its FFN split with the Neural Engine as the
  // engine splits it (asrAneFfnPolicy), taken down to the prompt's rows,
  // which run its smallest function padded: the prefix logits against the
  // reference, the last row's against the GPU alone's, and the decode step
  // over the KV the split wrote.
  const char *disabled = std::getenv("OMNI_DISABLE_ANE");
  if (disabled && std::string_view(disabled) == "1") {
    std::cout << "Neural Engine split: skipped (OMNI_DISABLE_ANE=1)\n";
    return;
  }
  if (const std::optional<std::string> reason = ane::unavailable()) {
    std::cout << "Neural Engine split: skipped (" << *reason << ")\n";
    return;
  }
  Qwen3Arena fullArena(backend, decoder.linear(), weights.decoder, SPLASH_PREFILL_TOKEN_BUDGET, 1, 2, false);
  decoder.enableAneFfn(backend, asrAneFfnPolicy(), fullArena);
  {
    ops::AneFfn *split = decoder.aneFfn();
    split->setMinimumRows(1);
    const std::string what = "split prefill (" + std::to_string(split->aneChannels()) + " ANE channels)";
    const std::vector<float> last = prefillOnce(split, what);
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (uint32_t v = 0; v < vocabulary; ++v) {
      dot += double(last[v]) * gpuLast[v];
      normA += double(last[v]) * last[v];
      normB += double(gpuLast[v]) * gpuLast[v];
    }
    const double cosine = dot / std::sqrt(normA * normB);
    std::cout << what << " last-row logits against the GPU alone's: cosine " << cosine << "\n";
    require(cosine > 0.999, what + ": last-row logits diverge from the GPU alone's");
    decodeOnce("decode logits after the " + what);
  }
}


// The talker logits of the prompt of `request`, as the engine assembles it,
// and the code predictor's of its first two groups, against the reference.
void checkTts(metal::MetalBackend &backend, const std::string &modelDir, const TtsRequest &request, uint32_t length,
              const std::vector<float> &last, const std::vector<float> &heads) {
  const OmniConfig config = readOmniConfig(modelDir);
  require(config.kind == ModelKind::Tts, "not a TTS model");
  const TtsConfig &c = *config.tts;
  model::WeightImages images(backend, "omni-parity");
  const auto start = std::chrono::steady_clock::now();
  TtsWeights weights = loadTtsWeights(backend, images, modelDir, c, config.quantized);
  std::cout << "weights loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
  TtsModel model(backend, std::move(weights), c, 128, 4);
  const Qwen3Decoder &talker = model.talker(), &predictor = model.predictor();
  const TtsPrompt prompt = ttsPrompt(c, request);
  const std::vector<uint32_t> &textTokens = prompt.tokens;
  const std::vector<OmniPromptPosition> &positions = prompt.positions;
  const uint32_t L = static_cast<uint32_t>(positions.size());
  require(L == length && textTokens.size() <= 128, "reference shape mismatch: engine " + std::to_string(L) + " rows, reference " + std::to_string(length));
  metal::MetalBuffer textRows = backend.allocateBuffer(128 * 2048 * 2, metal::BufferStorage::Shared, "text rows");
  metal::MetalBuffer positionBuffer = backend.allocateBuffer(L * sizeof(OmniPromptPosition), metal::BufferStorage::Shared, "positions");
  std::memcpy(positionBuffer.contents(), positions.data(), L * sizeof(OmniPromptPosition));

  KvPagePool pool(backend, 64);
  LanePageTables talkerTables(backend, 1, c.talker.layers, 4), predictorTables(backend, 1, c.predictor.layers, 1);
  LaneKv talkerLane(pool, talkerTables, 0), predictorLane(pool, predictorTables, 0);
  const KvBinding talkerKv{talkerTables.buffer(), talkerTables.entries(), pool.pages(), pool.pageCount()};
  const KvBinding predictorKv{predictorTables.buffer(), predictorTables.entries(), pool.pages(), pool.pageCount()};
  Qwen3Arena talkerArena(backend, talker.linear(), talker.weights(), 128, 1, 1, false);
  Qwen3Arena predictorArena(backend, predictor.linear(), predictor.weights(), 8, 1, 1, false);
  require(talkerLane.reserve(L) && predictorLane.reserve(2), "reserve pages");
  // Talker prefill.
  std::vector<ops::omni::AttentionSequence> descriptors{{talkerTables.offset(0), talkerTables.stride(), 0, L, 0, 0}};
  const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
  Qwen3Decoder::uploadTables(talkerArena, attention);
  static_cast<uint32_t *>(talkerArena.logitRows.contents())[0] = L - 1;
  metal::CommandGraph graph;
  model.addTextProjection(graph, 0, textTokens, textRows);
  ops::omni::Embedding::addPromptAssemble(graph, positionBuffer, textRows, static_cast<uint32_t>(textTokens.size()),
                                          model.tables(), {}, {}, 0, talkerArena.hidden, L);
  talker.addForward(graph, talkerArena, talkerKv, {&attention, 1});
  // Code predictor: [talker final hidden, codec_embedding(100)] -> lm_head[0].
  metal::MetalBuffer predictorInput = backend.allocateBuffer(64 * 2048 * 2, metal::BufferStorage::Shared, "predictor input");
  metal::MetalBuffer rowIndex = backend.allocateBuffer(4, metal::BufferStorage::Shared, "row");
  static_cast<uint32_t *>(rowIndex.contents())[0] = 0;
  metal::MetalBuffer code = backend.allocateBuffer(4, metal::BufferStorage::Shared, "code");
  static_cast<uint32_t *>(code.contents())[0] = 100;
  ops::omni::Embedding::addGatherRows(graph, rowIndex, talkerArena.selected, predictorInput, 1, 1, c.talker.hidden);
  ops::omni::Embedding::addGather(graph, code, model.tables().codec, backend.view(predictorInput, uint64_t{c.talker.hidden} * 2, uint64_t{c.talker.hidden} * 2), 1,
                                  c.codecVocabulary, c.talker.hidden);
  model.addPredictorInput(graph, predictorInput, predictorArena.hidden, 2);
  std::vector<ops::omni::AttentionSequence> predictorDescriptors{{predictorTables.offset(0), predictorTables.stride(), 0, 2, 0, 0}};
  const ops::omni::AttentionTables predictorAttention = ops::omni::attentionTables(predictorDescriptors, false);
  Qwen3Decoder::uploadTables(predictorArena, predictorAttention);
  static_cast<uint32_t *>(predictorArena.logitRows.contents())[0] = 1;
  predictor.addForward(graph, predictorArena, predictorKv, {&predictorAttention, 1});
  metal::MetalBuffer predictorLogits = backend.allocateBuffer(8 * 2048 * 4, metal::BufferStorage::Shared, "predictor logits");
  model.addPredictorHead(graph, predictorArena, 0, 1, predictorLogits);
  const auto t0 = std::chrono::steady_clock::now();
  metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
  (void)ticket.wait();
  std::cout << "talker prefill + predictor: "
            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << " ms\n";
  require(last.size() == c.codecVocabulary, "talker reference width");
  compare(static_cast<const float *>(talkerArena.logits.contents()), last, c.codecVocabulary, "talker logits");
  require(heads.size() == 2 * c.predictorVocabulary, "predictor reference width");
  const std::vector<float> head0(heads.begin(), heads.begin() + c.predictorVocabulary);
  const std::vector<float> head1(heads.begin() + c.predictorVocabulary, heads.end());
  compare(static_cast<const float *>(predictorLogits.contents()), head0, c.predictorVocabulary, "predictor logits");
  // Sub-step 1 through the decode arena: codec_embedding[0](argmax) at position 2.
  uint32_t c1 = 0;
  const auto *p0 = static_cast<const float *>(predictorLogits.contents());
  for (uint32_t v = 1; v < c.predictorVocabulary; ++v) if (p0[v] > p0[c1]) c1 = v;
  Qwen3Arena predictorDecode(backend, predictor.linear(), predictor.weights(), 8, 8, 8, true);
  metal::MetalBuffer codeBuffer = backend.allocateBuffer(4, metal::BufferStorage::Shared, "c1");
  static_cast<uint32_t *>(codeBuffer.contents())[0] = c1;
  metal::MetalBuffer embed = backend.allocateBuffer(64 * 2048 * 2, metal::BufferStorage::Shared, "embed");
  const uint64_t tableBytes = uint64_t{c.predictorVocabulary} * c.talker.hidden * 2;
  metal::CommandGraph graph2;
  ops::omni::Embedding::addGather(graph2, codeBuffer, backend.view(model.tables().predictor, 0, tableBytes), embed, 1,
                                  c.predictorVocabulary, c.talker.hidden);
  model.addPredictorInput(graph2, embed, predictorDecode.hidden, 1);
  std::vector<ops::omni::AttentionSequence> stepDescriptors{{predictorTables.offset(0), predictorTables.stride(), 2, 1, 0, 2}};
  const ops::omni::AttentionTables stepAttention = ops::omni::attentionTables(stepDescriptors, true);
  Qwen3Decoder::uploadTables(predictorDecode, stepAttention);
  static_cast<uint32_t *>(predictorDecode.logitRows.contents())[0] = 0;
  predictor.addForward(graph2, predictorDecode, predictorKv, {&stepAttention, 1});
  model.addPredictorHead(graph2, predictorDecode, 1, 1, predictorLogits);
  metal::CommandTicket ticket2 = backend.submitCommandAsync(graph2.command());
  (void)ticket2.wait();
  compare(static_cast<const float *>(predictorLogits.contents()), head1, c.predictorVocabulary, "predictor sub-step 1 logits");
}


// The Base model's prompt of "Hello, world." (first token 9707) without a
// speaker, the reference's hand-built one.
void runTts(metal::MetalBackend &backend, const std::string &modelDir, const Reference &reference) {
  require(reference.vocabulary == 3072, "reference shape mismatch");
  TtsRequest request;
  request.text = {9707};
  checkTts(backend, modelDir, request, reference.length, reference.last, reference.prefix);
}

// A reference of mlx-audio's own prompt (omni_parity_reference.py
// tts-prompt): u32 L, V, language id, speaker, text count, instruction
// count, the text and instruction tokens, then the talker's last-row logits
// [V] and the predictor's [2][2048].
void runTtsPrompt(metal::MetalBackend &backend, const std::string &modelDir, const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  uint32_t header[6] = {};
  file.read(reinterpret_cast<char *>(header), sizeof header);
  TtsRequest request;
  request.languageId = header[2];
  request.speaker = header[3];
  request.text.resize(header[4]);
  request.instruction.resize(header[5]);
  file.read(reinterpret_cast<char *>(request.text.data()), request.text.size() * 4);
  file.read(reinterpret_cast<char *>(request.instruction.data()), request.instruction.size() * 4);
  std::vector<float> last(header[1]), heads(2 * 2048);
  file.read(reinterpret_cast<char *>(last.data()), last.size() * 4);
  file.read(reinterpret_cast<char *>(heads.data()), heads.size() * 4);
  if (!file || request.text.empty()) throw std::runtime_error("reference file is short");
  std::cout << "prompt: " << header[0] << " rows, " << request.instruction.size() << " instruction tokens, speaker "
            << request.speaker << ", language " << request.languageId << "\n";
  checkTts(backend, modelDir, request, header[0], last, heads);
}

// Teacher forcing over a long generation (omni_parity_reference.py
// tts-force): the reference's frames are fed back through the engine's own
// step (TtsEngine::step and addPredictor at one lane: the step input of the
// previous frame and the next trailing text row, a one-row talker decode
// over the lane's pages, the predictor's two-row sub-step 0 and its 14 decode
// sub-steps over the frame's own codes), and the talker's code 0 logits of
// every frame, and the predictor's every E frames, are compared with the
// reference's by history length. The engine's logits are written to `out`,
// when given, in the reference's format.
struct ForceStats final {
  double kl = 0.0, maxKl = 0.0, cosine = 0.0, minCosine = 1.0, maxDiff = 0.0;
  uint32_t count = 0, top1 = 0;
};

// KL(reference || ours) of the softmax over the first `vocabulary` ids,
// plus `extra` (EOS), the cosine of the logits over them, top-1 agreement.
void accumulate(ForceStats &s, const float *ours, const float *reference, uint32_t vocabulary, uint32_t extra) {
  const auto value = [&](const float *x, uint32_t i) { return double(x[i < vocabulary ? i : extra]); };
  const uint32_t n = vocabulary + (extra < 0xffffffffu ? 1 : 0);
  double maxA = -1e300, maxB = -1e300;
  uint32_t argA = 0, argB = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (value(ours, i) > maxA) maxA = value(ours, i), argA = i;
    if (value(reference, i) > maxB) maxB = value(reference, i), argB = i;
  }
  double sumA = 0.0, sumB = 0.0, dot = 0.0, normA = 0.0, normB = 0.0, diff = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    sumA += std::exp(value(ours, i) - maxA);
    sumB += std::exp(value(reference, i) - maxB);
    dot += value(ours, i) * value(reference, i);
    normA += value(ours, i) * value(ours, i);
    normB += value(reference, i) * value(reference, i);
    diff = std::max(diff, std::fabs(value(ours, i) - value(reference, i)));
  }
  double kl = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    const double logP = value(reference, i) - maxB - std::log(sumB), logQ = value(ours, i) - maxA - std::log(sumA);
    kl += std::exp(logP) * (logP - logQ);
  }
  const double cosine = dot / std::sqrt(normA * normB);
  s.kl += kl;
  s.maxKl = std::max(s.maxKl, kl);
  s.cosine += cosine;
  s.minCosine = std::min(s.minCosine, cosine);
  s.maxDiff = std::max(s.maxDiff, diff);
  s.top1 += argA == argB;
  ++s.count;
}

void runTtsForce(metal::MetalBackend &backend, const std::string &modelDir, const std::string &path,
                 const std::string &out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  uint32_t header[8] = {};
  file.read(reinterpret_cast<char *>(header), sizeof header);
  TtsRequest request;
  request.languageId = header[2];
  request.speaker = header[3];
  request.text.resize(header[4]);
  request.instruction.resize(header[5]);
  const uint32_t frames = header[6], every = std::max(header[7], 1u), vocabulary = header[1];
  const uint32_t picked = (frames + every - 1) / every;
  file.read(reinterpret_cast<char *>(request.text.data()), request.text.size() * 4);
  file.read(reinterpret_cast<char *>(request.instruction.data()), request.instruction.size() * 4);
  std::vector<uint32_t> codes(size_t(frames) * OMNI_CODE_GROUPS);
  file.read(reinterpret_cast<char *>(codes.data()), codes.size() * 4);
  std::vector<float> talkerReference(size_t(frames) * vocabulary), predictorReference(size_t(picked) * 15 * 2048);
  file.read(reinterpret_cast<char *>(talkerReference.data()), talkerReference.size() * 4);
  file.read(reinterpret_cast<char *>(predictorReference.data()), predictorReference.size() * 4);
  if (!file || request.text.empty() || !frames) throw std::runtime_error("reference file is short");

  const OmniConfig config = readOmniConfig(modelDir);
  require(config.kind == ModelKind::Tts, "not a TTS model");
  const TtsConfig &c = *config.tts;
  require(vocabulary == c.codecVocabulary, "reference vocabulary");
  model::WeightImages images(backend, "omni-parity");
  TtsWeights weights = loadTtsWeights(backend, images, modelDir, c, config.quantized);
  const TtsPrompt prompt = ttsPrompt(c, request);
  const uint32_t L = static_cast<uint32_t>(prompt.positions.size());
  require(L == header[0], "prompt length differs from the reference's");
  const auto textRows = static_cast<uint32_t>(prompt.tokens.size());
  const uint32_t poolRows = (textRows + 63) / 64 * 64;
  constexpr uint32_t kLanes = 8;
  TtsModel model(backend, std::move(weights), c, poolRows, kLanes);
  const Qwen3Decoder &talker = model.talker(), &predictor = model.predictor();
  const uint32_t hidden = c.talker.hidden;
  std::cout << "prompt " << L << " rows, " << textRows << " text rows, " << frames << " frames, predictor every "
            << every << "\n";

  const auto shared = [&](uint64_t bytes, const char *label) {
    metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    std::memset(buffer.contents(), 0, bytes);
    return buffer;
  };
  metal::MetalBuffer textPool = shared(uint64_t{poolRows} * hidden * 2, "text pool");
  metal::MetalBuffer positions = shared(uint64_t{L} * sizeof(OmniPromptPosition), "positions");
  std::memcpy(positions.contents(), prompt.positions.data(), L * sizeof(OmniPromptPosition));
  metal::MetalBuffer stepCodes = shared(OMNI_CODE_GROUPS * 4, "step codes");
  metal::MetalBuffer trailingRow = shared(4, "trailing row");
  metal::MetalBuffer forced = shared(OMNI_CODE_GROUPS * 4, "forced codes");
  metal::MetalBuffer rowIndex = shared(4, "row index");
  metal::MetalBuffer predictorInput = shared(uint64_t{2 * kLanes + 64} * hidden * 2, "predictor input");
  std::vector<metal::MetalBuffer> heads;
  for (uint32_t g = 0; g < 15; ++g) heads.push_back(shared(uint64_t{kLanes} * c.predictorVocabulary * 4, "head"));

  const uint32_t blocks = (L + frames + OMNI_PAGE_TOKENS) / OMNI_PAGE_TOKENS + 1;
  KvPagePool pool(backend, blocks * c.talker.layers + c.predictor.layers + 8);
  LanePageTables talkerTables(backend, 1, c.talker.layers, blocks), predictorTables(backend, 1, c.predictor.layers, 1);
  LaneKv talkerLane(pool, talkerTables, 0), predictorLane(pool, predictorTables, 0);
  require(talkerLane.reserve(L + frames) && predictorLane.reserve(OMNI_CODE_GROUPS), "reserve pages");
  const KvBinding talkerKv{talkerTables.buffer(), talkerTables.entries(), pool.pages(), pool.pageCount()};
  const KvBinding predictorKv{predictorTables.buffer(), predictorTables.entries(), pool.pages(), pool.pageCount()};
  // The engine's arenas at eight lanes (TtsEngine's constructor).
  Qwen3Arena talkerPrefill(backend, talker.linear(), talker.weights(), std::max(L, 128u), kLanes, kLanes, false);
  Qwen3Arena talkerDecode(backend, talker.linear(), talker.weights(), kLanes, kLanes, kLanes, true);
  Qwen3Arena predictorPrefill(backend, predictor.linear(), predictor.weights(), 2 * kLanes, kLanes, kLanes, true);
  Qwen3Arena predictorDecode(backend, predictor.linear(), predictor.weights(), kLanes, kLanes, kLanes, true);

  // TtsEngine::addPredictor at one row over the forced frame.
  const auto addPredictor = [&](metal::CommandGraph &graph, metal::MetalBuffer selected) {
    ops::omni::Embedding::addGatherRows(graph, rowIndex, selected, predictorInput, 1, 1, hidden, 2);
    ops::omni::Embedding::addGather(graph, forced, model.tables().codec,
                                    backend.view(predictorInput, uint64_t{hidden} * 2,
                                                 predictorInput.sizeBytes() - uint64_t{hidden} * 2),
                                    1, c.codecVocabulary, hidden, 2, OMNI_CODE_GROUPS);
    model.addPredictorInput(graph, predictorInput, predictorPrefill.hidden, 2);
    std::vector<ops::omni::AttentionSequence> first{{predictorTables.offset(0), predictorTables.stride(), 0, 2, 0, 0}};
    const ops::omni::AttentionTables firstTables = ops::omni::attentionTables(first, true);
    Qwen3Decoder::uploadTables(predictorPrefill, firstTables);
    static_cast<uint32_t *>(predictorPrefill.logitRows.contents())[0] = 1;
    const bool fold = model.predictorHeadFoldsNorm(predictorDecode, 1);
    predictor.addForward(graph, predictorPrefill, predictorKv, {.tables = &firstTables, .logitRows = 1, .rawSelected = fold});
    model.addPredictorHead(graph, predictorPrefill, 0, 1, heads[0], fold ? predictorPrefill.selectedRaw : metal::MetalBuffer{});
    for (uint32_t i = 1; i + 1 < OMNI_CODE_GROUPS; ++i) {
      const bool qkvReady = model.addPredictorCodeInput(
          graph, backend.view(forced, uint64_t{i} * 4, forced.sizeBytes() - uint64_t{i} * 4), i, predictorDecode.hidden,
          predictorDecode.qkv, 1);
      std::vector<ops::omni::AttentionSequence> step{{predictorTables.offset(0), predictorTables.stride(), 1 + i, 1, 0, 1 + i}};
      const ops::omni::AttentionTables stepTables = ops::omni::attentionTables(step, true);
      Qwen3Decoder::uploadTables(predictorDecode, stepTables);
      static_cast<uint32_t *>(predictorDecode.logitRows.contents())[0] = 0;
      predictor.addForward(graph, predictorDecode, predictorKv,
                           {.tables = &stepTables, .logitRows = 1, .firstQkvReady = qkvReady, .rawSelected = fold});
      model.addPredictorHead(graph, predictorDecode, i, 1, heads[i], fold ? predictorDecode.hidden : metal::MetalBuffer{});
    }
  };

  std::vector<float> talkerOurs(size_t(frames) * vocabulary), predictorOurs(predictorReference.size());
  const auto collect = [&](uint32_t frame, const Qwen3Arena &arena) {
    std::memcpy(talkerOurs.data() + size_t(frame) * vocabulary, arena.logits.contents(), vocabulary * 4);
    if (frame % every == 0)
      for (uint32_t g = 0; g < 15; ++g)
        std::memcpy(predictorOurs.data() + (size_t(frame / every) * 15 + g) * 2048, heads[g].contents(), 2048 * 4);
  };
  const auto t0 = std::chrono::steady_clock::now();
  {
    metal::CommandGraph graph;
    model.addTextProjection(graph, 0, std::span<const uint32_t>(prompt.tokens.data(), textRows), textPool);
    ops::omni::Embedding::addPromptAssemble(graph, positions, textPool, poolRows, model.tables(), {}, {}, 0,
                                            talkerPrefill.hidden, L);
    std::vector<ops::omni::AttentionSequence> descriptors{{talkerTables.offset(0), talkerTables.stride(), 0, L, 0, 0}};
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
    Qwen3Decoder::uploadTables(talkerPrefill, attention);
    static_cast<uint32_t *>(talkerPrefill.logitRows.contents())[0] = L - 1;
    talker.addForward(graph, talkerPrefill, talkerKv, {&attention, 1});
    std::memcpy(forced.contents(), codes.data(), OMNI_CODE_GROUPS * 4);
    addPredictor(graph, talkerPrefill.selected);
    (void)backend.submitCommandAsync(graph.command()).wait();
    collect(0, talkerPrefill);
  }
  const uint32_t trailingCount = static_cast<uint32_t>(request.text.size()); // t1..tN-1 and eos; then pad
  for (uint32_t frame = 1; frame < frames; ++frame) {
    const uint32_t committed = L + frame - 1;
    std::memcpy(stepCodes.contents(), codes.data() + size_t(frame - 1) * OMNI_CODE_GROUPS, OMNI_CODE_GROUPS * 4);
    static_cast<uint32_t *>(trailingRow.contents())[0] = prompt.trailingBase + std::min(frame - 1, trailingCount);
    metal::CommandGraph graph;
    ops::omni::Embedding::addTtsStepInput(graph, stepCodes, model.tables(), trailingRow, textPool, poolRows,
                                          talkerDecode.hidden, 1);
    std::vector<ops::omni::AttentionSequence> descriptors{
        {talkerTables.offset(0), talkerTables.stride(), committed, 1, 0, committed}};
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, true);
    Qwen3Decoder::uploadTables(talkerDecode, attention);
    static_cast<uint32_t *>(talkerDecode.logitRows.contents())[0] = 0;
    talker.addForward(graph, talkerDecode, talkerKv, {&attention, 1});
    if (frame % every == 0) {
      std::memcpy(forced.contents(), codes.data() + size_t(frame) * OMNI_CODE_GROUPS, OMNI_CODE_GROUPS * 4);
      addPredictor(graph, talkerDecode.selected);
    }
    (void)backend.submitCommandAsync(graph.command()).wait();
    collect(frame, talkerDecode);
  }
  std::cout << frames << " forced steps in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";

  if (!out.empty()) {
    std::ofstream o(out, std::ios::binary);
    o.write(reinterpret_cast<const char *>(header), sizeof header);
    o.write(reinterpret_cast<const char *>(request.text.data()), request.text.size() * 4);
    o.write(reinterpret_cast<const char *>(request.instruction.data()), request.instruction.size() * 4);
    o.write(reinterpret_cast<const char *>(codes.data()), codes.size() * 4);
    o.write(reinterpret_cast<const char *>(talkerOurs.data()), talkerOurs.size() * 4);
    o.write(reinterpret_cast<const char *>(predictorOurs.data()), predictorOurs.size() * 4);
    require(bool(o), "cannot write " + out);
    std::cout << "wrote " << out << "\n";
  }
  // By history: code 0 over the codec ids and EOS; the predictor's 15 groups.
  constexpr uint32_t kBin = 200;
  std::cout << "frames       code 0: mean KL   max KL  cosine(min)        top-1  max|diff|   predictor: mean KL  top-1\n";
  // A short last bin joins the one before.
  for (uint32_t begin = 0, end = 0; begin < frames; begin = end) {
    end = frames - begin < 3 * kBin / 2 ? frames : begin + kBin;
    ForceStats talkerBin, predictorBin;
    for (uint32_t f = begin; f < end; ++f) {
      accumulate(talkerBin, talkerOurs.data() + size_t(f) * vocabulary, talkerReference.data() + size_t(f) * vocabulary,
                 c.predictorVocabulary, c.codecEos);
      if (f % every == 0)
        for (uint32_t g = 0; g < 15; ++g) {
          const size_t at = (size_t(f / every) * 15 + g) * 2048;
          accumulate(predictorBin, predictorOurs.data() + at, predictorReference.data() + at, 2048, 0xffffffffu);
        }
    }
    char line[256];
    std::snprintf(line, sizeof line, "%5u-%-5u  %12.5f %8.4f  %.6f(%.5f)  %5.3f  %8.3f   %18.5f  %5.3f\n", begin,
                  end - 1, talkerBin.kl / talkerBin.count, talkerBin.maxKl,
                  talkerBin.cosine / talkerBin.count, talkerBin.minCosine, double(talkerBin.top1) / talkerBin.count,
                  talkerBin.maxDiff, predictorBin.count ? predictorBin.kl / predictorBin.count : 0.0,
                  predictorBin.count ? double(predictorBin.top1) / predictorBin.count : 0.0);
    std::cout << line;
    // Bounds by history, well above the bf16 noise (against mlx-audio's bf16
    // reference about 6e-4 mean KL and 0.98 top-1 at every history; against
    // its fp32 reference 2e-4 and 0.99) and far below what an error that
    // grows with the position gives.
    const std::string range = "frames " + std::to_string(begin) + "+";
    require(talkerBin.kl / talkerBin.count < 2e-3, range + ": code 0 logits diverge");
    require(double(talkerBin.top1) / talkerBin.count >= 0.95, range + ": code 0 argmax differs too often");
    if (predictorBin.count) {
      require(predictorBin.kl / predictorBin.count < 5e-3, range + ": predictor logits diverge");
      require(double(predictorBin.top1) / predictorBin.count >= 0.85, range + ": predictor argmax differs too often");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: omni-model-parity METALLIB asr|tts|tts-prompt|tts-force MODEL_DIR REFERENCE.bin [OUT.bin]\n";
    return 2;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    const std::string kind = argv[2];
    if (kind == "tts-prompt") runTtsPrompt(backend, argv[3], argv[4]);
    else if (kind == "tts-force") runTtsForce(backend, argv[3], argv[4], argc > 5 ? argv[5] : "");
    else if (kind == "asr") runAsr(backend, argv[3], readReference(argv[4]));
    else if (kind == "tts") runTts(backend, argv[3], readReference(argv[4]));
    else throw std::runtime_error("unknown kind " + kind);
  } catch (const std::exception &error) {
    std::cerr << "omni model parity: " << error.what() << "\n";
    return 1;
  }
  std::cout << "omni model parity ok\n";
  return 0;
}
