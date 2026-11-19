// Runs the TTS engine on real weights without the server:
//   omni-tts-run METALLIB MODEL_DIR OUT.wav TOKEN [TOKEN...] [--seed N] [--greedy] [--lanes N] [--repeat N] [--first-chunk F]
//                [--instruction T,T,...] [--speaker ID] [--language ID] [--continue T,T,... [--reference-frames K]]
// The tokens are the plain text's token ids (tokenize with the model's
// tokenizer). --repeat N submits the same text N times (concurrency).
// --instruction takes the instruction turn's tokens
// ("<|im_start|>user\n...<|im_end|>\n" tokenized whole), --speaker a
// CustomVoice speaker's codec id, --language a codec language id.
// --continue B checks a continuation (docs/omni-design.md, "Reference
// continuation"): it speaks the text A (the tokens) then B as one request,
// then B alone as the continuation of A with the first K (default 40)
// frames of that speech as its reference, and compares the continuation's
// frames with the whole request's from frame K on, and its audio (OUT.wav)
// with the whole request's (OUT.wav.whole.wav) from sample K * 1920. With
// --greedy the frames match until the prefill's and the decode steps'
// numerics part.

#include "metal/MetalBackend.hpp"
#include "model/WeightImages.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniEvents.hpp"
#include "omni/OmniLoader.hpp"
#include "omni/TtsEngine.hpp"
#include "omni/TtsModel.hpp"
#include "ops/OmniCodec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::omni;

namespace {

struct Collector final : EventSink {
  std::map<uint64_t, std::vector<int16_t>> audio;
  std::map<uint64_t, std::vector<uint32_t>> codes;
  std::map<uint64_t, std::chrono::steady_clock::time_point> firstAudio;
  std::map<uint64_t, DoneTimings> done;
  std::chrono::steady_clock::time_point started;
  void onStart(uint64_t, uint32_t) override {}
  void onAudio(uint64_t id, uint32_t, std::span<const int16_t> samples) override {
    if (!firstAudio.count(id)) firstAudio[id] = std::chrono::steady_clock::now();
    audio[id].insert(audio[id].end(), samples.begin(), samples.end());
  }
  void onCodes(uint64_t id, std::span<const uint32_t> frames) override {
    codes[id].assign(frames.begin(), frames.end());
  }
  void onTokens(uint64_t, uint32_t, std::span<const uint32_t>) override {}
  void onDone(uint64_t id, DoneReason reason, const DoneTimings &timings) override {
    done[id] = timings;
    std::cout << "request " << id << " done: reason " << int(reason) << ", frames " << timings.completion << ", prefill "
              << timings.prefillMicros / 1000.0 << " ms, decode " << timings.decodeMicros / 1000.0 << " ms, wall "
              << timings.wallMicros / 1000.0 << " ms\n";
  }
  void onError(uint64_t id, const std::string &code, const std::string &message) override {
    std::cout << "request " << id << " error " << code << ": " << message << "\n";
  }
};

void writeWav(const std::string &path, const std::vector<int16_t> &samples, uint32_t rate) {
  std::ofstream file(path, std::ios::binary);
  const uint32_t dataBytes = static_cast<uint32_t>(samples.size() * 2);
  const auto u32 = [&](uint32_t v) { file.write(reinterpret_cast<const char *>(&v), 4); };
  const auto u16 = [&](uint16_t v) { file.write(reinterpret_cast<const char *>(&v), 2); };
  file.write("RIFF", 4);
  u32(36 + dataBytes);
  file.write("WAVEfmt ", 8);
  u32(16);
  u16(1);
  u16(1);
  u32(rate);
  u32(rate * 2);
  u16(2);
  u16(16);
  file.write("data", 4);
  u32(dataBytes);
  file.write(reinterpret_cast<const char *>(samples.data()), dataBytes);
}

std::vector<uint32_t> tokenList(const std::string &list) {
  std::vector<uint32_t> tokens;
  for (size_t at = 0; at < list.size();) {
    const size_t comma = std::min(list.find(',', at), list.size());
    tokens.push_back(std::stoul(list.substr(at, comma - at)));
    at = comma + 1;
  }
  return tokens;
}

void runToIdle(TtsEngine &engine, Collector &sink) {
  while (!engine.idle()) engine.tick(sink);
}

// How many leading frames of `y` equal those of `x` from frame `from`: all
// 16 codes, and code 0 alone.
std::pair<uint32_t, uint32_t> matching(const std::vector<uint32_t> &x, uint32_t from, const std::vector<uint32_t> &y) {
  const uint32_t count =
      std::min<uint32_t>(static_cast<uint32_t>(x.size() / 16) - from, static_cast<uint32_t>(y.size() / 16));
  uint32_t same = 0, code0 = 0;
  while (same < count && std::equal(y.begin() + size_t(same) * 16, y.begin() + size_t(same + 1) * 16,
                                    x.begin() + (size_t(from) + same) * 16))
    ++same;
  while (code0 < count && y[size_t(code0) * 16] == x[(size_t(from) + code0) * 16]) ++code0;
  return {same, code0};
}

// The audio of frames [from, from + count) decoded through a stream that
// first took the `context` frames before them, the codec's front alone
// (through its transformer and upsampling) for all but the last `stack`
// and whole for those, against the same frames' audio in a decode of all
// frames from 0 (in chunks of 25): the SNR of the first 12 frames and of
// all `count`.
std::pair<double, double> codecContextSnr(metal::MetalBackend &backend, ops::omni::CodecDecoder &codec,
                                          const std::vector<uint32_t> &frames, uint32_t from, uint32_t context,
                                          uint32_t stack, uint32_t count) {
  const uint32_t spf = 1920, chunk = 25;
  metal::MetalBuffer codes = backend.allocateBuffer(frames.size() * 4, metal::BufferStorage::Shared, "check codes");
  std::memcpy(codes.contents(), frames.data(), frames.size() * 4);
  metal::MetalBuffer audio =
      backend.allocateBuffer(uint64_t{codec.maximumFrames()} * spf * 4, metal::BufferStorage::Shared, "check audio");
  const auto decode = [&](ops::omni::CodecStream &stream, uint32_t first, uint32_t n, std::vector<float> *into,
                          bool front = false) {
    for (uint32_t at = first; at < first + n; at += chunk) {
      const uint32_t step = std::min(chunk, first + n - at);
      metal::CommandGraph graph;
      const metal::MetalBuffer view = backend.view(codes, uint64_t{at} * 64, uint64_t{step} * 64);
      if (front) codec.addFrontStream(graph, stream, view, step);
      else codec.addDecodeStream(graph, stream, view, step, audio);
      (void)backend.submitCommandAsync(graph.command()).wait();
      const auto *samples = static_cast<const float *>(audio.contents());
      if (into) into->insert(into->end(), samples, samples + size_t(step) * spf);
    }
  };
  ops::omni::CodecStream whole(backend, codec.weights(), codec.maximumFrames());
  std::vector<float> reference, continued;
  decode(whole, 0, from + count, &reference);
  ops::omni::CodecStream stream(backend, codec.weights(), codec.maximumFrames());
  stack = std::min(stack, context);
  if (context > stack) decode(stream, from - context, context - stack, nullptr, true);
  if (stack) decode(stream, from - stack, stack, nullptr);
  decode(stream, from, count, &continued);
  const auto snr = [&](size_t samples) {
    double signal = 0.0, noise = 0.0;
    for (size_t i = 0; i < samples; ++i) {
      const double x = reference[size_t(from) * spf + i], d = continued[i] - x;
      signal += x * x;
      noise += d * d;
    }
    return noise > 0 ? 10.0 * std::log10(signal / noise) : INFINITY;
  };
  return {snr(size_t(std::min(12u, count)) * spf), snr(size_t(count) * spf)};
}

// The continuation check of --continue: request 1 speaks A then B, request
// 2 speaks B continuing A from request 1's first `referenceFrames` frames.
// Requests 3-5 speak A then B again as one batch, whose steps run other
// tiles than one request's: how long their frames stay request 1's is the
// numerics' own horizon, which the continuation's is read against.
int checkContinuation(metal::MetalBackend &backend, ops::omni::CodecDecoder &codec, TtsEngine &engine,
                      const TtsRequest &base, const std::vector<uint32_t> &a, const std::vector<uint32_t> &b,
                      uint32_t referenceFrames, const std::string &out) {
  Collector sink;
  TtsRequest whole = base;
  whole.id = 1;
  whole.text = a;
  whole.text.insert(whole.text.end(), b.begin(), b.end());
  whole.returnCodes = true;
  engine.submit(whole, sink);
  runToIdle(engine, sink);
  const std::vector<uint32_t> &frames = sink.codes[1];
  const uint32_t total = static_cast<uint32_t>(frames.size() / 16);
  if (total <= referenceFrames) {
    std::cerr << "the whole request spoke " << total << " frames, not more than the reference's " << referenceFrames
              << "\n";
    return 1;
  }
  TtsRequest continuation = base;
  continuation.id = 2;
  continuation.text = b;
  continuation.referenceText = a;
  continuation.referenceCodes.assign(frames.begin(), frames.begin() + size_t(referenceFrames) * 16);
  continuation.returnCodes = true;
  engine.submit(continuation, sink);
  runToIdle(engine, sink);
  for (uint64_t id = 3; id <= 5; ++id) {
    TtsRequest again = whole;
    again.id = id;
    engine.submit(again, sink);
  }
  runToIdle(engine, sink);
  const auto [same, code0] = matching(frames, referenceFrames, sink.codes[2]);
  const auto [batchSame, batchCode0] = matching(frames, 0, sink.codes[3]);
  std::cout << "continuation: whole " << total << " frames, reference " << referenceFrames << ", continuation "
            << sink.codes[2].size() / 16 << " frames; " << same << " frames identical to the whole request's (code 0: "
            << code0 << ")\n";
  std::cout << "the whole request in a batch of three: " << batchSame << " frames identical to it alone (code 0: "
            << batchCode0 << ")\n";
  // The codec stream's start: the reference's last frames as its context.
  const uint32_t count = std::min(total - referenceFrames, 100u);
  const std::pair<uint32_t, uint32_t> contexts[] = {{0, 0},   {25, 25},  {72, 72},  {25, 8},  {72, 8},
                                                     {100, 8}, {200, 8}, {500, 8}, {500, 4}, {500, 16}};
  for (const auto [context, stack] : contexts) {
    if (context > referenceFrames) continue;
    const auto [first, all] = codecContextSnr(backend, codec, frames, referenceFrames, context, stack, count);
    std::cout << "codec from frame " << referenceFrames << " after " << context << " context frames (" << stack
              << " whole): SNR " << first << " dB over the first second, " << all << " dB over " << count
              << " frames, against the whole decode\n";
  }
  writeWav(out, sink.audio[2], 24000);
  writeWav(out + ".whole.wav", sink.audio[1], 24000);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: omni-tts-run METALLIB MODEL_DIR OUT.wav TOKEN... [--seed N] [--greedy] [--lanes N] [--repeat N] "
                 "[--instruction T,T,...] [--speaker ID] [--language ID]\n";
    return 2;
  }
  try {
    const std::string modelDir = argv[2], out = argv[3];
    std::vector<uint32_t> tokens;
    uint64_t seed = 1;
    bool greedy = false;
    uint32_t lanes = 4, repeat = 1, firstChunk = 0, speaker = 0, language = 0, referenceFrames = 40;
    std::vector<uint32_t> instruction, then;
    for (int i = 4; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--seed") seed = std::stoull(argv[++i]);
      else if (arg == "--greedy") greedy = true;
      else if (arg == "--lanes") lanes = std::stoul(argv[++i]);
      else if (arg == "--repeat") repeat = std::stoul(argv[++i]);
      else if (arg == "--first-chunk") firstChunk = std::stoul(argv[++i]);
      else if (arg == "--speaker") speaker = std::stoul(argv[++i]);
      else if (arg == "--language") language = std::stoul(argv[++i]);
      else if (arg == "--instruction") instruction = tokenList(argv[++i]);
      else if (arg == "--continue") then = tokenList(argv[++i]);
      else if (arg == "--reference-frames") referenceFrames = std::stoul(argv[++i]);
      else tokens.push_back(std::stoul(arg));
    }
    metal::MetalBackend backend(argv[1]);
    const OmniConfig config = readOmniConfig(modelDir);
    if (config.kind != ModelKind::Tts) throw std::runtime_error("not a TTS model");
    model::WeightImages images(backend, "omni-tts-run");
    auto t0 = std::chrono::steady_clock::now();
    TtsWeights weights = loadTtsWeights(backend, images, modelDir, *config.tts, config.quantized);
    if (const uint64_t packed = packDenseWeights(backend, weights, UINT64_MAX))
      std::cout << "packed bf16 copies " << packed / (1024.0 * 1024.0) << " MiB\n";
    ops::omni::CodecDecoderWeights codecWeights =
        ops::omni::loadCodecDecoderWeights(backend, std::filesystem::path(modelDir) / "speech_tokenizer");
    std::cout << "weights loaded in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    TtsEngineConfig engineConfig;
    engineConfig.lanes = lanes;
    if (config.tts->type != TtsModelType::Base) engineConfig.maxInstructionTokens = 512;
    TtsModel model(backend, std::move(weights), *config.tts, engineConfig.textRows(), lanes);
    engineConfig.kvPages = 4096;
    if (firstChunk) engineConfig.firstChunkFrames = firstChunk;
    ops::omni::CodecDecoder codec(backend, std::move(codecWeights), engineConfig.maxChunkFrames);
    TtsEngine engine(backend, model, codec, engineConfig);
    std::cout << "engine memory " << engine.memoryBytes() / (1024.0 * 1024.0) << " MiB\n";
    if (!then.empty()) {
      TtsRequest base;
      base.seed = seed;
      base.instruction = instruction;
      base.speaker = speaker;
      base.languageId = language;
      if (greedy) {
        base.temperature = 0.0F;
        base.predictorTemperature = 0.0F;
      }
      return checkContinuation(backend, codec, engine, base, tokens, then, referenceFrames, out);
    }
    Collector sink;
    for (uint32_t r = 0; r < repeat; ++r) {
      TtsRequest request;
      request.id = r + 1;
      request.text = tokens;
      request.seed = seed + r;
      request.instruction = instruction;
      request.speaker = speaker;
      request.languageId = language;
      if (greedy) {
        request.temperature = 0.0F;
        request.predictorTemperature = 0.0F;
      }
      engine.submit(request, sink);
    }
    sink.started = std::chrono::steady_clock::now();
    uint32_t ticks = 0;
    while (!engine.idle()) {
      engine.tick(sink);
      ++ticks;
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - sink.started).count();
    double audioSeconds = 0.0;
    for (const auto &[id, samples] : sink.audio) audioSeconds += samples.size() / 24000.0;
    std::cout << "ticks " << ticks << ", wall " << wall << " s, audio " << audioSeconds << " s, RTF " << wall / std::max(audioSeconds, 1e-9)
              << ", throughput " << audioSeconds / wall << " audio s / s\n";
    for (const auto &[id, when] : sink.firstAudio)
      std::cout << "request " << id << " first audio after "
                << std::chrono::duration<double, std::milli>(when - sink.started).count() << " ms\n";
    if (sink.audio.count(1)) writeWav(out, sink.audio[1], 24000);
  } catch (const std::exception &error) {
    std::cerr << "omni tts run: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
