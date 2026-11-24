// Profiles every dispatch of a real TTS or ASR run, kernel by kernel:
//   omni-pipeline-profile METALLIB MODEL_DIR [--lanes N] [--repeat N] [--seed N] TOKEN...     (TTS)
//   omni-pipeline-profile METALLIB MODEL_DIR [--lanes N] [--repeat N] FILE.wav...             (ASR)
// With dispatch profiling (BackendInstrumentation) each dispatch runs as its
// own command, so the GPU times are per kernel and exclude the overlap a
// real command gets; the sum over a step therefore bounds the step from
// above, and the dispatch count tells how much is overhead. Prints the
// kernels sorted by total GPU time, grouped by the pipeline stage their
// names belong to, and the run's own wall time (which profiling inflates).

#include "metal/BackendInstrumentation.hpp"
#include "metal/MetalBackend.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"
#include "omni/AsrEngine.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniEvents.hpp"
#include "omni/OmniLoader.hpp"
#include "omni/TtsEngine.hpp"
#include "omni/TtsModel.hpp"
#include "ops/OmniCodec.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
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
  while (offset + 8 <= bytes.size()) {
    const std::string id(bytes.data() + offset, 4);
    uint32_t size = 0;
    std::memcpy(&size, bytes.data() + offset + 4, 4);
    if (id == "data") {
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

struct Sink final : EventSink {
  double audioSeconds = 0.0;
  uint32_t tokens = 0, frames = 0;
  void onStart(uint64_t, uint32_t) override {}
  void onAudio(uint64_t, uint32_t, std::span<const int16_t> samples) override { audioSeconds += samples.size() / 24000.0; }
  void onTokens(uint64_t, uint32_t, std::span<const uint32_t> generated) override { tokens += generated.size(); }
  void onDone(uint64_t, DoneReason, const DoneTimings &timings) override { frames += timings.completion; }
  void onError(uint64_t id, const std::string &code, const std::string &message) override {
    std::cout << "request " << id << " error " << code << ": " << message << "\n";
  }
};

struct Total {
  size_t count = 0;
  double seconds = 0.0;
};

const char *stageOf(const std::string &name) {
  if (name.starts_with("omni_codec")) return "codec";
  if (name.starts_with("omni_audio") || name.starts_with("omni_encoder")) return "audio tower";
  if (name.starts_with("prefill_") || name.starts_with("omni_prompt") || name.starts_with("omni_gemm") ||
      name.starts_with("omni_attention_prefill") || name.starts_with("omni_attention_prepare_prefill"))
    return "prefill";
  if (name.starts_with("omni_sample") || name.starts_with("omni_argmax")) return "sampling";
  return "decode";
}

void report(const char *title, std::map<std::string, Total> &totals, double runSeconds) {
  std::vector<std::pair<std::string, Total>> rows(totals.begin(), totals.end());
  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.second.seconds > b.second.seconds; });
  std::map<std::string, Total> stages;
  double sum = 0.0;
  size_t dispatches = 0;
  for (const auto &[name, t] : rows) {
    sum += t.seconds;
    dispatches += t.count;
    Total &s = stages[stageOf(name)];
    s.count += t.count;
    s.seconds += t.seconds;
  }
  std::printf("\n== %s: %zu dispatches, %.2f ms of GPU time in %.2f ms of wall (profiled)\n", title, dispatches, sum * 1e3,
              runSeconds * 1e3);
  std::printf("%-12s %9s %12s %8s\n", "stage", "dispatch", "gpu ms", "share");
  for (const auto &[stage, t] : stages)
    std::printf("%-12s %9zu %12.3f %7.1f%%\n", stage.c_str(), t.count, t.seconds * 1e3, 100.0 * t.seconds / sum);
  std::printf("%-48s %9s %12s %10s %8s\n", "kernel", "dispatch", "gpu ms", "mean us", "share");
  for (const auto &[name, t] : rows)
    std::printf("%-48s %9zu %12.3f %10.2f %7.1f%%\n", name.c_str(), t.count, t.seconds * 1e3, t.seconds * 1e6 / t.count,
                100.0 * t.seconds / sum);
  totals.clear();
}

void collect(metal::MetalBackend &backend, std::map<std::string, Total> &totals) {
  for (const metal::DispatchTiming &timing : metal::BackendInstrumentation::takeDispatchProfile(backend)) {
    Total &t = totals[timing.pipelineName];
    ++t.count;
    t.seconds += timing.gpuSeconds;
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: omni-pipeline-profile METALLIB MODEL_DIR [--lanes N] [--repeat N] [--seed N] TOKEN... | FILE.wav...\n";
    return 2;
  }
  try {
    const std::string modelDir = argv[2];
    std::vector<uint32_t> tokens;
    std::vector<std::string> files;
    uint32_t lanes = 8, repeat = 1;
    uint64_t seed = 1;
    for (int i = 3; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--lanes") lanes = std::stoul(argv[++i]);
      else if (arg == "--repeat") repeat = std::stoul(argv[++i]);
      else if (arg == "--seed") seed = std::stoull(argv[++i]);
      else if (arg.ends_with(".wav")) files.push_back(arg);
      else tokens.push_back(std::stoul(arg));
    }
    metal::MetalBackend backend(argv[1]);
    const OmniConfig config = readOmniConfig(modelDir);
    model::WeightImages images(backend, "omni-pipeline-profile");
    std::map<std::string, Total> totals;
    Sink sink;
    if (config.kind == ModelKind::Tts) {
      if (tokens.empty()) throw std::runtime_error("a TTS profile takes the text's token ids");
      TtsWeights weights = loadTtsWeights(backend, images, modelDir, *config.tts, config.quantized);
      (void)packDenseWeights(backend, weights, UINT64_MAX);
      ops::omni::CodecDecoderWeights codecWeights =
          ops::omni::loadCodecDecoderWeights(backend, std::filesystem::path(modelDir) / "speech_tokenizer");
      TtsModel model(backend, std::move(weights), *config.tts, 1024 + 8, lanes);
      TtsEngineConfig engineConfig;
      engineConfig.lanes = lanes;
      engineConfig.kvPages = 4096;
      ops::omni::CodecDecoder codec(backend, std::move(codecWeights), engineConfig.maxChunkFrames);
      TtsEngine engine(backend, model, codec, engineConfig);
      // One unprofiled warm-up request compiles the pipelines.
      {
        TtsRequest request;
        request.id = 1;
        request.text = tokens;
        request.seed = seed;
        request.maxFrames = 8;
        engine.submit(request, sink);
        while (!engine.idle()) engine.tick(sink);
      }
      sink = Sink{};
      for (uint32_t r = 0; r < repeat; ++r) {
        TtsRequest request;
        request.id = 2 + r;
        request.text = tokens;
        request.seed = seed + r;
        engine.submit(request, sink);
      }
      metal::BackendInstrumentation::setDispatchProfiling(backend, true);
      const auto t0 = std::chrono::steady_clock::now();
      uint32_t ticks = 0;
      while (!engine.idle()) {
        engine.tick(sink);
        collect(backend, totals);
        ++ticks;
      }
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("TTS: %u requests on %u lanes, %u ticks, %u frames, %.2f s of audio\n", repeat, lanes, ticks, sink.frames,
                  sink.audioSeconds);
      report("TTS run", totals, wall);
    } else {
      if (files.empty()) throw std::runtime_error("an ASR profile takes WAV files (16 kHz mono)");
      AsrTextWeights text = loadAsrTextWeights(backend, images, modelDir, *config.asr, config.quantized);
      (void)packDenseWeights(backend, text, UINT64_MAX);
      const model::SafetensorsCheckpoint checkpoint(modelDir);
      ops::omni::AudioEncoderWeights tower = ops::omni::loadAudioEncoderWeights(backend, checkpoint, config.asr->encoder);
      AsrEngineConfig engineConfig;
      engineConfig.lanes = lanes;
      engineConfig.maxAudioSeconds = 330;
      engineConfig.kvPages = 8192;
      AsrEngine engine(backend, *config.asr, std::move(text), std::move(tower), engineConfig);
      const AsrConfig &c = *config.asr;
      const std::vector<uint32_t> prefix{c.imStart, 8948, 198, c.imEnd, 198, c.imStart, 872, 198, c.audioStart};
      const std::vector<uint32_t> suffix{c.audioEnd, c.imEnd, 198, c.imStart, 77091, 198};
      const auto submit = [&](uint64_t id, const std::string &file) {
        AsrRequest request;
        request.id = id;
        request.prefix = prefix;
        request.suffix = suffix;
        request.samples = readWav(file);
        sink.audioSeconds += request.samples.size() / 16000.0;
        engine.submit(std::move(request), sink);
      };
      submit(1, files[0]);
      while (!engine.idle()) engine.tick(sink);
      sink = Sink{};
      uint64_t id = 2;
      for (uint32_t r = 0; r < repeat; ++r)
        for (const std::string &file : files) submit(id++, file);
      metal::BackendInstrumentation::setDispatchProfiling(backend, true);
      const auto t0 = std::chrono::steady_clock::now();
      uint32_t ticks = 0;
      while (!engine.idle()) {
        engine.tick(sink);
        collect(backend, totals);
        ++ticks;
      }
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("ASR: %zu requests on %u lanes, %u ticks, %u tokens, %.2f s of audio\n", files.size() * repeat, lanes, ticks,
                  sink.tokens, sink.audioSeconds);
      report("ASR run", totals, wall);
    }
  } catch (const std::exception &error) {
    std::cerr << "omni pipeline profile: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
