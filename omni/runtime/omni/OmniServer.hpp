#pragma once

// The Omni engine process (docs/omni-design.md): loads one model (TTS or
// ASR), then serves frames over a pipe pair. A reader thread turns the input
// into frames; the loop thread submits requests to the engine, ticks it
// while it has work and writes its events as frames.

#include "metal/MetalBackend.hpp"
#include "omni/AsrEngine.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/TtsEngine.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace splash::omni {

struct ServerOptions final {
  std::filesystem::path modelRoot;
  std::filesystem::path metallib;
  uint32_t lanes = 0;            // 0: the model's default (TTS 8, ASR 8)
  uint64_t maxMemoryBytes = 0;   // 0: automatic
  double idleReleaseSeconds = 0; // 0: never
  bool disableAne = false;       // the ASR prefill on the GPU alone (AsrEngineConfig::neuralEngine)
  // An ASR draft model for speculative decoding (Qwen3-ASR-0.6B beside the
  // 1.7B), and its tokens per round and speculating lanes (0: the engine's).
  std::filesystem::path draftModel;
  uint32_t draftTokens = 0;
  uint32_t speculativeLanes = 0;
};

struct LoadedEngine final {
  OmniConfig config;
  std::unique_ptr<metal::MetalBackend> backend;
  std::unique_ptr<model::WeightImages> images;
  std::unique_ptr<TtsModel> ttsModel;
  std::unique_ptr<ops::omni::CodecDecoder> codec;
  std::unique_ptr<TtsEngine> tts;
  std::unique_ptr<AsrEngine> asr;
  uint32_t lanes = 0;
  uint32_t maxContext = 0;
};

// Loads the model named by options into an engine; logs to stderr.
[[nodiscard]] LoadedEngine loadEngine(const ServerOptions &options);

// Serves until the input closes or shutdown is requested; returns the
// process exit code.
int serve(LoadedEngine &engine, int inputFd, int outputFd, std::atomic<bool> &shutdown);

// The family name of the model at root, as the installer checks it:
// "qwen3-tts" or "qwen3-asr", with its quantization.
[[nodiscard]] std::string checkModelJson(const std::filesystem::path &root);

} // namespace splash::omni
