// Runs the ASR engine on real weights without the server:
//   omni-asr-run METALLIB MODEL_DIR FILE.wav [FILE.wav...] [--sequential | --concurrency N]
//                [--draft DRAFT_DIR] [--draft-tokens K] [--speculative-lanes N]
//                [--lanes N] [--language-suffix TOKEN...]
// WAV files must be 16 kHz 16-bit mono. Prints the generated token ids per
// file (decode them with the model's tokenizer) and timings; files are
// submitted together and decode in lock step, one after another with
// --sequential, or at most N at a time with --concurrency N. --draft decodes speculatively with a Qwen3-ASR-0.6B draft.

#include "metal/MetalBackend.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"
#include "omni/AsrEngine.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniEvents.hpp"
#include "omni/OmniLoader.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::omni;

namespace {

std::vector<float> readWav(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0) throw std::runtime_error("not a WAV file: " + path);
  size_t offset = 12;
  uint16_t channels = 1, bits = 16;
  uint32_t rate = 16000;
  while (offset + 8 <= bytes.size()) {
    const std::string id(bytes.data() + offset, 4);
    uint32_t size = 0;
    std::memcpy(&size, bytes.data() + offset + 4, 4);
    if (id == "fmt ") {
      std::memcpy(&channels, bytes.data() + offset + 10, 2);
      std::memcpy(&rate, bytes.data() + offset + 12, 4);
      std::memcpy(&bits, bytes.data() + offset + 22, 2);
    } else if (id == "data") {
      if (rate != 16000 || channels != 1 || bits != 16) throw std::runtime_error("WAV must be 16 kHz 16-bit mono");
      const size_t count = std::min<size_t>(size, bytes.size() - offset - 8) / 2;
      std::vector<float> samples(count);
      const auto *pcm = reinterpret_cast<const int16_t *>(bytes.data() + offset + 8);
      for (size_t i = 0; i < count; ++i) samples[i] = pcm[i] / 32768.0F;
      return samples;
    }
    offset += 8 + size + (size & 1);
  }
  throw std::runtime_error("no data chunk in " + path);
}

struct Collector final : EventSink {
  std::map<uint64_t, std::vector<uint32_t>> tokens;
  std::map<uint64_t, std::chrono::steady_clock::time_point> first;
  std::chrono::steady_clock::time_point started;
  void onStart(uint64_t, uint32_t) override {}
  void onAudio(uint64_t, uint32_t, std::span<const int16_t>) override {}
  void onTokens(uint64_t id, uint32_t, std::span<const uint32_t> generated) override {
    if (!first.count(id)) first[id] = std::chrono::steady_clock::now();
    tokens[id].insert(tokens[id].end(), generated.begin(), generated.end());
  }
  size_t finished = 0;
  void onDone(uint64_t id, DoneReason reason, const DoneTimings &timings) override {
    ++finished;
    std::cout << "request " << id << " done: reason " << int(reason) << ", prompt " << timings.promptTokens << " tokens, generated "
              << timings.completion << ", prefill " << timings.prefillMicros / 1000.0 << " ms, decode "
              << timings.decodeMicros / 1000.0 << " ms, wall " << timings.wallMicros / 1000.0 << " ms\n";
  }
  void onError(uint64_t id, const std::string &code, const std::string &message) override {
    ++finished;
    std::cout << "request " << id << " error " << code << ": " << message << "\n";
  }
};

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: omni-asr-run METALLIB MODEL_DIR FILE.wav [FILE.wav...] [--language-suffix TOKEN...]\n";
    return 2;
  }
  try {
    const std::string modelDir = argv[2];
    std::vector<std::string> files;
    std::vector<uint32_t> languageSuffix;
    bool suffixMode = false;
    size_t concurrency = 0; // 0: every file at once
    std::string draftDir;
    AsrEngineConfig engineConfig;
    engineConfig.lanes = 8;
    engineConfig.maxAudioSeconds = 330;
    engineConfig.kvPages = 8192;
    for (int i = 3; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(arg + " takes a value");
        return argv[++i];
      };
      if (arg == "--language-suffix") suffixMode = true;
      else if (arg == "--sequential") concurrency = 1;
      else if (arg == "--concurrency") concurrency = std::stoul(value());
      else if (arg == "--draft") draftDir = value();
      else if (arg == "--draft-tokens") engineConfig.draftTokens = std::stoul(value());
      else if (arg == "--speculative-lanes") engineConfig.speculativeLanes = std::stoul(value());
      else if (arg == "--lanes") engineConfig.lanes = std::stoul(value());
      else if (suffixMode) languageSuffix.push_back(std::stoul(arg));
      else files.push_back(arg);
    }
    metal::MetalBackend backend(argv[1]);
    const OmniConfig config = readOmniConfig(modelDir);
    if (config.kind != ModelKind::Asr) throw std::runtime_error("not an ASR model");
    model::WeightImages images(backend, "omni-asr-run");
    const auto t0 = std::chrono::steady_clock::now();
    AsrTextWeights text = loadAsrTextWeights(backend, images, modelDir, *config.asr, config.quantized);
    if (const uint64_t packed = packDenseWeights(backend, text, UINT64_MAX))
      std::cout << "packed bf16 copies " << packed / (1024.0 * 1024.0) << " MiB\n";
    const model::SafetensorsCheckpoint checkpoint(modelDir);
    ops::omni::AudioEncoderWeights tower = ops::omni::loadAudioEncoderWeights(backend, checkpoint, config.asr->encoder);
    std::optional<AsrDraftModel> draft;
    if (!draftDir.empty()) {
      const OmniConfig draftConfig = readOmniConfig(draftDir);
      if (draftConfig.kind != ModelKind::Asr) throw std::runtime_error("the draft is not an ASR model");
      AsrTextWeights draftText = loadAsrTextWeights(backend, images, draftDir, *draftConfig.asr, draftConfig.quantized);
      const model::SafetensorsCheckpoint draftCheckpoint(draftDir);
      draft = AsrDraftModel{*draftConfig.asr, std::move(draftText),
                            ops::omni::loadAudioEncoderWeights(backend, draftCheckpoint, draftConfig.asr->encoder)};
    }
    std::cout << "weights loaded in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    AsrEngine engine(backend, *config.asr, std::move(text), std::move(tower), engineConfig, std::move(draft));
    std::cout << "engine memory " << engine.memoryBytes() / (1024.0 * 1024.0) << " MiB\n";
    Collector sink;
    const AsrConfig &c = *config.asr;
    const std::vector<uint32_t> prefix{c.imStart, 8948, 198, c.imEnd, 198, c.imStart, 872, 198, c.audioStart};
    std::vector<uint32_t> suffix{c.audioEnd, c.imEnd, 198, c.imStart, 77091, 198};
    suffix.insert(suffix.end(), languageSuffix.begin(), languageSuffix.end());
    double audioSeconds = 0.0;
    std::vector<AsrRequest> requests;
    for (size_t i = 0; i < files.size(); ++i) {
      AsrRequest request;
      request.id = i + 1;
      request.prefix = prefix;
      request.suffix = suffix;
      request.samples = readWav(files[i]);
      audioSeconds += request.samples.size() / 16000.0;
      requests.push_back(std::move(request));
    }
    sink.started = std::chrono::steady_clock::now();
    if (!concurrency) concurrency = requests.size();
    size_t submitted = 0;
    while (sink.finished < requests.size()) {
      while (submitted < requests.size() && submitted - sink.finished < concurrency)
        engine.submit(std::move(requests[submitted++]), sink);
      engine.tick(sink);
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - sink.started).count();
    for (const auto &[id, tokens] : sink.tokens) {
      std::cout << "tokens " << id << ":";
      for (uint32_t token : tokens) std::cout << " " << token;
      std::cout << "\n";
    }
    if (engine.speculative()) {
      const AsrEngine::SpeculationStats &stats = engine.speculationStats();
      std::cout << "speculation: " << stats.rounds << " lane rounds, accepted " << stats.accepted << " of " << stats.proposed
                << " proposals, " << double(stats.emitted) / std::max<uint64_t>(stats.rounds, 1) << " tokens per round\n";
    }
    std::cout << "plain-step rows run past their lane's end: " << engine.wastedStepRows() << "\n";
    std::cout << "wall " << wall << " s for " << audioSeconds << " s of audio: RTFx " << audioSeconds / wall << "\n";
  } catch (const std::exception &error) {
    std::cerr << "omni asr run: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
