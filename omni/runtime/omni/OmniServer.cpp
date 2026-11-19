#include "AwakeClock.hpp"
#include "omni/OmniServer.hpp"

#include "model/SafetensorsCheckpoint.hpp"
#include "omni/OmniLoader.hpp"
#include "omni/OmniProtocol.hpp"
#include "ops/OmniAudioEncoder.hpp"

#include <sys/select.h>
#include <sysexits.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace splash::omni {
namespace {

using Clock = AwakeClock;

void log(const std::string &line) {
  const std::string text = line + "\n";
  (void)!::write(STDERR_FILENO, text.data(), text.size());
}

uint64_t availableMemory() {
  const uint64_t physical = metal::probeDeviceCapabilities().physicalMemoryBytes;
  // Leave the rest of the system a quarter, at least 8 GiB.
  const uint64_t reserve = std::max<uint64_t>(physical / 4, uint64_t{8} << 30);
  return physical > reserve ? physical - reserve : physical / 2;
}

// What packed bf16 copies (packDenseWeights) may take of the budget beyond
// the weights: what is left after 7 GiB for arenas, workspace and the KV
// pages of a few long requests.
uint64_t packedLimit(uint64_t budget, uint64_t weightBytes) {
  const uint64_t reserved = weightBytes + (uint64_t{7} << 30);
  return budget > reserved ? budget - reserved : 0;
}

// Frames read from the input, complete.
struct InputFrame final {
  protocol::FrameType type;
  std::vector<uint8_t> payload;
  Clock::time_point read{}; // when its last byte was read
};

class Input final {
public:
  explicit Input(int fd) : fd_(fd), thread_([this] { run(); }) {}
  ~Input() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
  }
  // Waits up to `wait` for a frame; empty when none and the input is not closed.
  std::optional<InputFrame> next(std::chrono::milliseconds wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    // A wall-clock deadline: this is a poll interval, not a measured duration.
    ready_.wait_until(lock, std::chrono::system_clock::now() + wait, [&] { return !frames_.empty() || closed_; });
    if (frames_.empty()) return std::nullopt;
    InputFrame frame = std::move(frames_.front());
    frames_.pop_front();
    return frame;
  }
  [[nodiscard]] bool closed() const noexcept { return closed_; }
  [[nodiscard]] const std::string &failure() const noexcept { return failure_; }
  [[nodiscard]] bool pending() {
    std::lock_guard<std::mutex> lock(mutex_);
    return !frames_.empty();
  }

private:
  bool readAll(std::span<uint8_t> bytes) {
    size_t done = 0;
    while (done < bytes.size() && !stop_) {
      fd_set set;
      FD_ZERO(&set);
      FD_SET(fd_, &set);
      timeval timeout{0, 200000};
      const int ready = ::select(fd_ + 1, &set, nullptr, nullptr, &timeout);
      if (ready < 0 && errno != EINTR) return false;
      if (ready <= 0) continue;
      const ssize_t got = ::read(fd_, bytes.data() + done, bytes.size() - done);
      if (got == 0) return false;
      if (got < 0) {
        if (errno == EINTR || errno == EAGAIN) continue;
        return false;
      }
      done += static_cast<size_t>(got);
    }
    return done == bytes.size();
  }
  void run() {
    try {
      while (!stop_) {
        std::array<uint8_t, protocol::kHeaderBytes> header{};
        if (!readAll(header)) break;
        const protocol::FrameHeader decoded = protocol::decodeHeader(header);
        InputFrame frame{decoded.type, std::vector<uint8_t>(decoded.length)};
        if (decoded.length && !readAll(frame.payload)) break;
        frame.read = Clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        frames_.push_back(std::move(frame));
        ready_.notify_one();
      }
    } catch (const std::exception &error) {
      failure_ = error.what();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    ready_.notify_all();
  }

  int fd_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::string failure_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<InputFrame> frames_;
  std::thread thread_;
};

class Output final : public EventSink {
public:
  explicit Output(int fd) : fd_(fd) {}
  void write(const std::vector<uint8_t> &frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t done = 0;
    while (done < frame.size()) {
      const ssize_t wrote = ::write(fd_, frame.data() + done, frame.size() - done);
      if (wrote < 0) {
        if (errno == EINTR) continue;
        throw std::system_error(errno, std::generic_category(), "writing the output");
      }
      done += static_cast<size_t>(wrote);
    }
  }
  void onStart(uint64_t requestId, uint32_t lane) override { write(protocol::startFrame(requestId, lane)); }
  void onAudio(uint64_t requestId, uint32_t sampleOffset, std::span<const int16_t> samples) override {
    write(protocol::audioFrame(requestId, sampleOffset, samples));
  }
  void onCodes(uint64_t requestId, std::span<const uint32_t> codes) override {
    write(protocol::codesFrame(requestId, codes));
  }
  void onTokens(uint64_t requestId, uint32_t sequenceOffset, std::span<const uint32_t> tokens) override {
    write(protocol::tokensFrame(requestId, sequenceOffset, tokens));
  }
  void onDone(uint64_t requestId, DoneReason reason, const DoneTimings &timings) override {
    write(protocol::doneFrame(requestId, reason, timings));
  }
  void onError(uint64_t requestId, const std::string &code, const std::string &message) override {
    write(protocol::errorFrame(protocol::FailureClass::RequestError, false, requestId, code, message));
  }

private:
  int fd_;
  std::mutex mutex_;
};

std::string quoted(std::string_view text) {
  std::string out = "\"";
  for (char c : text) {
    if (c == '"' || c == '\\') out += '\\';
    if (static_cast<unsigned char>(c) < 0x20) continue;
    out += c;
  }
  return out + "\"";
}

} // namespace

std::string checkModelJson(const std::filesystem::path &root) {
  const OmniConfig config = readOmniConfig(root);
  return std::string("{\"family\":\"") + (config.kind == ModelKind::Tts ? "qwen3-tts" : "qwen3-asr") +
         "\",\"quantized\":" + (config.quantized ? "true" : "false") + "}";
}

// An installed model (install/assembly.py) keeps config.json at its root
// and links the checkpoint's files under target/; a checkpoint directory
// holds everything at its root.
std::filesystem::path weightsDirectory(const std::filesystem::path &root) {
  std::error_code error;
  if (std::filesystem::is_directory(root / "target", error)) return root / "target";
  return root;
}

LoadedEngine loadEngine(const ServerOptions &options) {
  LoadedEngine engine;
  engine.config = readOmniConfig(options.modelRoot);
  const std::filesystem::path weights = weightsDirectory(options.modelRoot);
  engine.backend = std::make_unique<metal::MetalBackend>(options.metallib.string());
  // Every kernel's pipeline compiles while the weights load, so no request
  // waits for one (docs/omni-performance.md, round 15): on the first start
  // after the library changes Metal compiles its 275 kernels in about 1.5 s
  // in parallel (one at a time, as requests would, about 13 s); later
  // starts load them from Metal's cache in a few milliseconds.
  engine.backend->compileLibraryPipelines();
  engine.images = std::make_unique<model::WeightImages>(*engine.backend, "omni");
  const uint64_t budget = options.maxMemoryBytes ? options.maxMemoryBytes : availableMemory();
  const auto t0 = Clock::now();
  if (!options.draftModel.empty() && engine.config.kind != ModelKind::Asr)
    throw std::runtime_error("a draft model takes a transcription model");
  if (engine.config.kind == ModelKind::Tts) {
    const TtsConfig &c = *engine.config.tts;
    engine.lanes = options.lanes ? std::min<uint32_t>(options.lanes, 16) : 8;
    TtsWeights ttsWeights = loadTtsWeights(*engine.backend, *engine.images, weights, c, engine.config.quantized);
    ops::omni::CodecDecoderWeights codecWeights =
        ops::omni::loadCodecDecoderWeights(*engine.backend, weights / "speech_tokenizer");
    (void)packDenseWeights(*engine.backend, ttsWeights, packedLimit(budget, engine.backend->memoryStats().allocatedBytes));
    const uint64_t weightBytes = engine.backend->memoryStats().allocatedBytes;
    TtsEngineConfig config;
    config.lanes = engine.lanes;
    config.maxTextTokens = 1024;
    // VoiceDesign and CustomVoice prompts lead with an instruction.
    if (c.type != TtsModelType::Base) config.maxInstructionTokens = protocol::kMaximumInstructionTokens;
    config.maxFrames = 4096;
    // KV pages from what the budget leaves after the weights and ~1.5 GiB
    // of arenas and codec workspace, at most 8 GiB.
    const uint64_t reserved = weightBytes + (uint64_t{3} << 30);
    const uint64_t kvBytes = budget > reserved ? std::min<uint64_t>(budget - reserved, uint64_t{8} << 30) : uint64_t{1} << 30;
    config.kvPages = static_cast<uint32_t>(std::max<uint64_t>(kvBytes / OMNI_PAGE_BYTES, 2048));
    engine.ttsModel = std::make_unique<TtsModel>(*engine.backend, std::move(ttsWeights), c, config.textRows(), engine.lanes);
    engine.codec = std::make_unique<ops::omni::CodecDecoder>(*engine.backend, std::move(codecWeights), config.maxChunkFrames);
    config.aneCodec = !options.disableAne;
    engine.tts = std::make_unique<TtsEngine>(*engine.backend, *engine.ttsModel, *engine.codec, config);
    log("Speech: " + engine.tts->codecPlacement() + ".");
    engine.maxContext = config.textRows() + config.maxReferenceFrames + config.maxFrames;
  } else {
    const AsrConfig &c = *engine.config.asr;
    engine.lanes = options.lanes ? std::min<uint32_t>(options.lanes, OMNI_MAXIMUM_LANES) : 8;
    AsrTextWeights textWeights = loadAsrTextWeights(*engine.backend, *engine.images, weights, c, engine.config.quantized);
    const model::SafetensorsCheckpoint checkpoint(weights);
    ops::omni::AudioEncoderWeights tower = ops::omni::loadAudioEncoderWeights(*engine.backend, checkpoint, c.encoder);
    // The Neural Engine split's resident int8 weights and surfaces take about
    // 0.3 GiB of an affine checkpoint (ops::AneFfn); a draft's arenas, tower
    // workspace and audio rows about 0.5 GiB.
    const uint64_t aneBytes = !options.disableAne && engine.config.quantized ? uint64_t{1} << 29 : 0;
    const uint64_t draftWorkspaceBytes = uint64_t{1} << 29;
    // The draft's weights (for Qwen3-ASR-0.6B-4bit about 0.7 GiB with its
    // bf16 tower), read like the target's.
    std::optional<AsrDraftModel> draft;
    // Or a DFlash draft (about 0.55 GB of bf16 layers), which reads the
    // target's hidden states and needs no tower.
    std::optional<AsrDFlashDraft> dflash;
    if (!options.draftModel.empty() && isDFlashDraft(options.draftModel)) {
      const DFlashDraftConfig dflashConfig = readDFlashDraftConfig(options.draftModel, c);
      dflash = AsrDFlashDraft{dflashConfig, loadDFlashDraftWeights(*engine.backend, *engine.images,
                                                                   weightsDirectory(options.draftModel), dflashConfig)};
      const uint64_t needed =
          engine.backend->memoryStats().allocatedBytes + (uint64_t{4} << 30) + aneBytes + draftWorkspaceBytes;
      if (needed > budget) {
        dflash.reset();
        char line[256];
        std::snprintf(line, sizeof line,
                      "Speculative decoding: off; the memory budget of %.2f GiB cannot fit the DFlash draft (%.2f GiB "
                      "needed).",
                      budget / double(uint64_t{1} << 30), needed / double(uint64_t{1} << 30));
        log(line);
      }
    } else if (!options.draftModel.empty()) {
      const OmniConfig draftConfig = readOmniConfig(options.draftModel);
      if (draftConfig.kind != ModelKind::Asr) throw std::runtime_error("the draft model is not a transcription model");
      const std::filesystem::path draftWeights = weightsDirectory(options.draftModel);
      AsrTextWeights draftText =
          loadAsrTextWeights(*engine.backend, *engine.images, draftWeights, *draftConfig.asr, draftConfig.quantized);
      const model::SafetensorsCheckpoint draftCheckpoint(draftWeights);
      draft = AsrDraftModel{*draftConfig.asr, std::move(draftText),
                            ops::omni::loadAudioEncoderWeights(*engine.backend, draftCheckpoint, draftConfig.asr->encoder)};
      // The draft serves only where the budget fits it beside the weights,
      // the ~3 GiB of arenas and workspace, its own workspace and the
      // smallest KV pool the engine takes (1 GiB); otherwise lanes decode
      // plainly, as without one.
      const uint64_t needed =
          engine.backend->memoryStats().allocatedBytes + (uint64_t{4} << 30) + aneBytes + draftWorkspaceBytes;
      if (needed > budget) {
        draft.reset();
        char line[256];
        std::snprintf(line, sizeof line,
                      "Speculative decoding: off; the memory budget of %.2f GiB cannot fit the draft (%.2f GiB "
                      "needed).",
                      budget / double(uint64_t{1} << 30), needed / double(uint64_t{1} << 30));
        log(line);
      }
    }
    (void)packDenseWeights(*engine.backend, textWeights, packedLimit(budget, engine.backend->memoryStats().allocatedBytes));
    const uint64_t weightBytes = engine.backend->memoryStats().allocatedBytes;
    AsrEngineConfig config;
    config.lanes = engine.lanes;
    config.maxAudioSeconds = 1200;
    config.maxTokens = 4096;
    config.neuralEngine = !options.disableAne;
    config.draftTokens = options.draftTokens;
    config.speculativeLanes = options.speculativeLanes;
    const uint64_t reserved =
        weightBytes + (uint64_t{3} << 30) + aneBytes + (draft || dflash ? draftWorkspaceBytes : 0);
    const uint64_t kvBytes = budget > reserved ? std::min<uint64_t>(budget - reserved, uint64_t{12} << 30) : uint64_t{1} << 30;
    config.kvPages = static_cast<uint32_t>(std::max<uint64_t>(kvBytes / OMNI_PAGE_BYTES, 4096));
    engine.asr =
        std::make_unique<AsrEngine>(*engine.backend, c, std::move(textWeights), std::move(tower), config, std::move(draft),
                                    std::move(dflash));
    if (engine.asr->speculative())
      log(std::string("Speculative decoding: ") + (engine.asr->dflash() ? "DFlash draft " : "draft ") +
          options.draftModel.string() + ", " +
          std::to_string(engine.asr->draftTokens()) + " tokens per round while at most " +
          std::to_string(engine.asr->speculativeLanes()) + " lanes are active.");
    engine.maxContext = ops::omni::audioTokens(config.maxAudioSeconds * 100) + 256 + config.maxTokens;
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  char line[256];
  std::snprintf(line, sizeof line, "Weights loaded in %.1f s; %.2f GiB allocated; %u lanes.", seconds,
                engine.backend->memoryStats().allocatedBytes / double(uint64_t{1} << 30), engine.lanes);
  log(line);
  const auto waited = Clock::now();
  const metal::LibraryPipelines pipelines = engine.backend->waitForLibraryPipelines();
  std::snprintf(line, sizeof line, "Pipelines compiled: %zu in %.2f s, %.2f s after the weights.", pipelines.compiled,
                pipelines.seconds, std::chrono::duration<double>(Clock::now() - waited).count());
  log(line);
  for (const std::string &name : pipelines.failed) log("Pipeline " + name + " did not compile at startup.");
  return engine;
}

int serve(LoadedEngine &engine, int inputFd, int outputFd, std::atomic<bool> &shutdown) {
  Input input(inputFd);
  Output output(outputFd);
  const bool tts = engine.tts != nullptr;
  output.write(protocol::readyFrame(engine.lanes, engine.maxContext,
                                    tts ? protocol::ModelKindCode::Tts : protocol::ModelKindCode::Asr));
  uint64_t requests = 0, completed = 0, failed = 0;
  const auto started = Clock::now();
  const auto status = [&](uint64_t correlation) {
    const uint32_t active = tts ? engine.tts->activeLanes() : engine.asr->activeLanes();
    const uint32_t queued = tts ? engine.tts->queued() : engine.asr->queued();
    const metal::MetalMemoryStats memory = engine.backend->memoryStats();
    std::string json = "{\"schema_version\":1,\"ready\":true,\"kind\":" + quoted(tts ? "tts" : "asr") +
                       ",\"lanes\":{\"active\":" + std::to_string(active) + ",\"total\":" + std::to_string(engine.lanes) +
                       "},\"queued\":" + std::to_string(queued) + ",\"requests\":{\"received\":" + std::to_string(requests) +
                       ",\"completed\":" + std::to_string(completed) + ",\"failed\":" + std::to_string(failed) +
                       "},\"memory\":{\"allocated_bytes\":" + std::to_string(memory.allocatedBytes) +
                       ",\"peak_allocated_bytes\":" + std::to_string(memory.peakAllocatedBytes) +
                       "},\"uptime_seconds\":" +
                       std::to_string(std::chrono::duration<double>(Clock::now() - started).count()) + "}";
    output.write(protocol::statusJsonFrame(correlation, json));
  };
  // Counts completions through the sink.
  struct Counting final : EventSink {
    Output &out;
    uint64_t &completed, &failed;
    Counting(Output &o, uint64_t &c, uint64_t &f) : out(o), completed(c), failed(f) {}
    void onStart(uint64_t id, uint32_t lane) override { out.onStart(id, lane); }
    void onAudio(uint64_t id, uint32_t offset, std::span<const int16_t> samples) override { out.onAudio(id, offset, samples); }
    void onCodes(uint64_t id, std::span<const uint32_t> codes) override { out.onCodes(id, codes); }
    void onTokens(uint64_t id, uint32_t offset, std::span<const uint32_t> tokens) override { out.onTokens(id, offset, tokens); }
    void onDone(uint64_t id, DoneReason reason, const DoneTimings &timings) override {
      ++completed;
      out.onDone(id, reason, timings);
    }
    void onError(uint64_t id, const std::string &code, const std::string &message) override {
      ++failed;
      out.onError(id, code, message);
    }
  } sink(output, completed, failed);

  while (!shutdown) {
    const bool busy = tts ? !engine.tts->idle() : !engine.asr->idle();
    // Take every pending frame, waiting only when there is no work and
    // nothing arrived yet: a request an idle engine receives is ticked at
    // once, not after the poll interval.
    std::chrono::milliseconds wait = busy ? std::chrono::milliseconds(0) : std::chrono::milliseconds(100);
    for (;;) {
      std::optional<InputFrame> frame = input.next(wait);
      if (!frame) break;
      wait = std::chrono::milliseconds(0);
      try {
        switch (frame->type) {
        case protocol::FrameType::SpeechRequest: {
          const protocol::SpeechRequestFrame f = protocol::decodeSpeechRequest(frame->payload);
          ++requests;
          if (!tts) {
            sink.onError(f.requestId, "wrong_model", "this engine serves transcription");
            break;
          }
          // A reference's text and codes continue earlier speech; a speaker
          // embedding (voice cloning) is a later phase.
          if (!f.speaker.empty()) {
            sink.onError(f.requestId, "unsupported", "speaker embeddings are not supported yet");
            break;
          }
          TtsRequest request;
          request.id = f.requestId;
          request.text = f.text;
          request.instruction = f.instruction;
          request.speaker = f.presetSpeaker;
          request.languageId = f.languageId;
          request.temperature = f.temperature;
          request.topK = f.topK;
          request.topP = f.topP;
          request.repetitionPenalty = f.repetitionPenalty;
          request.predictorTemperature = f.predictorTemperature;
          request.predictorTopK = f.predictorTopK;
          request.predictorTopP = f.predictorTopP;
          request.seed = f.seed;
          request.maxFrames = f.maxFrames;
          request.chunkFrames = f.chunkFrames;
          request.referenceText = f.referenceText;
          request.referenceCodes = f.referenceCodes;
          request.returnCodes = (f.flags & protocol::kSpeechFlagCodes) != 0;
          engine.tts->submit(std::move(request), sink);
          break;
        }
        case protocol::FrameType::TranscribeRequest: {
          protocol::TranscribeRequestFrame f = protocol::decodeTranscribeRequest(frame->payload);
          ++requests;
          if (tts) {
            sink.onError(f.requestId, "wrong_model", "this engine serves speech");
            break;
          }
          AsrRequest request;
          request.id = f.requestId;
          request.prefix = std::move(f.prefix);
          request.suffix = std::move(f.suffix);
          request.samples = std::move(f.samples);
          request.maxTokens = f.maxTokens;
          request.received = frame->read;
          engine.asr->submit(std::move(request), sink);
          break;
        }
        case protocol::FrameType::Cancel: {
          const uint64_t id = protocol::decodeCancel(frame->payload);
          if (tts) engine.tts->cancel(id, sink);
          else engine.asr->cancel(id, sink);
          break;
        }
        case protocol::FrameType::StatusRequest:
          status(protocol::decodeStatusRequest(frame->payload));
          break;
        default:
          throw protocol::ProtocolError("unexpected frame from the client");
        }
      } catch (const protocol::ProtocolError &error) {
        output.write(protocol::errorFrame(protocol::FailureClass::ProtocolFatal, false, 0, "protocol", error.what()));
        log(std::string("protocol error: ") + error.what());
        return EX_PROTOCOL;
      }
    }
    if (input.closed() && !(tts ? !engine.tts->idle() : !engine.asr->idle())) break;
    if (tts) engine.tts->tick(sink);
    else engine.asr->tick(sink);
  }
  if (!input.failure().empty()) {
    log("input failed: " + input.failure());
    return EX_PROTOCOL;
  }
  return EX_OK;
}

} // namespace splash::omni
