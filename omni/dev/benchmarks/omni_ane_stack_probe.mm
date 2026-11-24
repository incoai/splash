// Compiles the speech tokenizer decoder's convolution stack on the Neural
// Engine (ops::omni::CodecAneStack) from the cached Qwen3-TTS checkpoint
// with the given functions, and times each function's evaluation: alone
// (the event raised before it is queued), and back to back.
//   omni-ane-stack-probe METALLIB [--frames 4,8,16,25] [--repeat N] [--print-mil]

#include "metal/MetalBackend.hpp"
#include "ops/OmniCodec.hpp"
#include "ops/OmniCodecAne.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace splash;

namespace {

std::filesystem::path findSpeechTokenizer() {
  if (const char *model = std::getenv("SPLASH_OMNI_TTS_MODEL")) return std::filesystem::path(model) / "speech_tokenizer";
  const char *home = std::getenv("HOME");
  if (!home) return {};
  const std::filesystem::path snapshots =
      std::filesystem::path(home) / ".cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-4bit/snapshots";
  std::error_code error;
  if (!std::filesystem::is_directory(snapshots, error)) return {};
  for (const auto &entry : std::filesystem::directory_iterator(snapshots, error))
    if (std::filesystem::exists(entry.path() / "speech_tokenizer/model.safetensors", error))
      return entry.path() / "speech_tokenizer";
  return {};
}

struct Report final {
  std::mutex mutex;
  std::condition_variable changed;
  uint32_t reported = 0;
  bool failed = false;
};

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: omni-ane-stack-probe METALLIB [--frames 4,8,16,25] [--repeat N] [--print-mil]\n");
    return 2;
  }
  std::vector<uint32_t> frames{4, 8, 16, 25};
  uint32_t repeat = 9;
  bool printMil = false;
  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--print-mil") {
      printMil = true;
    } else if (flag == "--frames" && i + 1 < argc) {
      frames.clear();
      std::stringstream list(argv[++i]);
      std::string item;
      while (std::getline(list, item, ',')) frames.push_back(static_cast<uint32_t>(std::stoul(item)));
    } else if (flag == "--repeat" && i + 1 < argc) {
      repeat = static_cast<uint32_t>(std::stoul(argv[++i]));
    }
  }
  try {
    if (const auto reason = ane::unavailable()) {
      std::printf("Neural Engine unavailable: %s\n", reason->c_str());
      return 1;
    }
    const std::filesystem::path tokenizer = findSpeechTokenizer();
    if (tokenizer.empty()) throw std::runtime_error("no cached Qwen3-TTS speech tokenizer");
    metal::MetalBackend backend(argv[1]);
    ops::omni::CodecDecoderWeights weights = ops::omni::loadCodecDecoderWeights(backend, tokenizer);
    const uint32_t perFrame = weights.dims.samplesPerFrame();
    const auto start = std::chrono::steady_clock::now();
    ops::omni::CodecAneStack ane(backend, weights, frames, 1, {std::chrono::seconds(1200), [] { return false; }});
    std::printf("compiled and loaded %zu functions in %.1f s; %.1f MB of surfaces\n", frames.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), double(ane.bytes()) / 1e6);
    if (printMil) std::printf("%s", ane.source().c_str());
    metal::SharedEvent event = backend.newSharedEvent();
    uint64_t value = 0;
    for (const uint32_t function : ane.functionFrames()) {
      // Alone: each evaluation queued after the last reported.
      std::vector<double> times;
      for (uint32_t pass = 0; pass < repeat; ++pass) {
        event.signal(++value);
        auto report = std::make_shared<Report>();
        const auto t0 = std::chrono::steady_clock::now();
        ane.enqueue(0, function, event, value, value, [report](bool success) {
          std::lock_guard lock(report->mutex);
          ++report->reported;
          report->failed = report->failed || !success;
          report->changed.notify_all();
        });
        std::unique_lock lock(report->mutex);
        if (!report->changed.wait_for(lock, std::chrono::seconds(60), [&] { return report->reported > 0; }))
          throw std::runtime_error("the Neural Engine did not report");
        if (report->failed) throw std::runtime_error("an evaluation failed");
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(times.begin(), times.end());
      // Back to back: `repeat` evaluations queued at once.
      auto report = std::make_shared<Report>();
      event.signal(++value);
      const auto t0 = std::chrono::steady_clock::now();
      for (uint32_t pass = 0; pass < repeat; ++pass)
        ane.enqueue(0, function, event, value, value, [report](bool success) {
          std::lock_guard lock(report->mutex);
          ++report->reported;
          report->failed = report->failed || !success;
          report->changed.notify_all();
        });
      {
        std::unique_lock lock(report->mutex);
        if (!report->changed.wait_for(lock, std::chrono::seconds(120), [&] { return report->reported >= repeat; }))
          throw std::runtime_error("the Neural Engine did not report");
        if (report->failed) throw std::runtime_error("an evaluation failed");
      }
      const double pipelined = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / repeat;
      std::printf("%2u frames (%5u samples): alone median %.2f ms (min %.2f), back to back %.2f ms = %.3f ms per frame\n",
                  function, function * perFrame, times[times.size() / 2], times.front(), pipelined, pipelined / function);
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "omni ane stack probe: %s\n", error.what());
    return 1;
  }
  return 0;
}
