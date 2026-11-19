#pragma once

// The TTS engine (docs/omni-design.md): lanes of requests generating codec
// frames in lock step. A tick admits queued requests (one ragged talker
// prefill, then their first frame), runs one step over every active lane
// (talker decode, code 0 sampling, the 15 code predictor sub-steps, all in
// one GPU command), and decodes due chunks of frames into audio with the
// speech tokenizer decoder, streamed per lane (ops::omni::CodecStream).
//
// The decoder's convolution stack runs on the Neural Engine when it can
// (ops::omni::CodecAneStack, OMNI_CODEC_ANE=0 keeps it on the GPU): a due
// chunk's front (codes through the ConvNeXt stages) is encoded at the head
// of the next GPU command, which then signals a shared event the chunk's
// evaluation waits for; the GPU goes on with the decode step, and the
// engine emits the chunk's audio once the Neural Engine reports, while it
// waits for the GPU. A lane has at most one evaluation in flight, so its
// chunks stay in order. An evaluation that fails or does not report within
// kAneBound retires the Neural Engine for the engine's life: the GPU decodes
// that chunk from the lane's input and histories, and every later chunk.

#include "AwakeClock.hpp"
#include "metal/MetalBackend.hpp"
#include "omni/KvPages.hpp"
#include "omni/OmniEvents.hpp"
#include "omni/TtsModel.hpp"
#include "ops/OmniCodec.hpp"
#include "ops/OmniCodecAne.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace splash::omni {

struct TtsRequest final {
  uint64_t id = 0;
  std::vector<uint32_t> text; // plain text tokens
  uint32_t languageId = 0;    // codec language id, 0 = automatic
  // VoiceDesign and CustomVoice: the instruction's user turn,
  // "<|im_start|>user\n{instruction}<|im_end|>\n" tokenized whole, or none.
  std::vector<uint32_t> instruction;
  uint32_t speaker = 0; // CustomVoice: the preset speaker's codec id, 0 = none
  float temperature = 0.9F;
  uint32_t topK = 50;
  float topP = 1.0F;
  float repetitionPenalty = 1.05F;
  float predictorTemperature = 0.9F;
  uint32_t predictorTopK = 50;
  float predictorTopP = 1.0F;
  uint64_t seed = 0;
  uint32_t maxFrames = 0;   // 0: eight per text token (at least 100), within the engine's maximum
  uint32_t chunkFrames = 0; // 0: 25
  // A continuation: the text and the frames (16 codes each) of earlier
  // speech in the same voice, which this request goes on from
  // (docs/omni-design.md, "Reference continuation"). Both or neither.
  std::vector<uint32_t> referenceText;
  std::vector<uint32_t> referenceCodes;
  bool returnCodes = false; // report the generated frames before Done (EventSink::onCodes)
};

struct TtsEngineConfig final {
  uint32_t lanes = 8;
  uint32_t maxTextTokens = 1024;
  // The instruction's tokens a request may carry; 0 for a Base model, which
  // takes none.
  uint32_t maxInstructionTokens = 0;
  uint32_t maxFrames = 2048;      // frames per request (12.5 per second)
  uint32_t kvPages = 16384;       // 128 KiB each: 2 GiB
  // Or the pool's pages from the bytes allocated once everything else is:
  // the engine creates the pool last, from what its budget leaves.
  std::function<uint32_t(uint64_t allocatedBytes)> kvPagesFor;
  uint32_t maxChunkFrames = 100;  // frames decoded per codec pass (the stream's capacity)
  // The first chunks of a request are small so audio starts early: a chunk
  // is due at firstChunkFrames frames, the next at twice that, and so on up
  // to the request's chunk size. One frame (80 ms of audio) starts playback
  // after the prompt and a single step: 32 ms against 70 ms at four frames
  // on an M5 Pro, at the same real-time factor.
  uint32_t firstChunkFrames = 1;
  bool minimumFrames = true;      // mask EOS for the first two frames (min_new_tokens 2)
  // A continuation's reference: its frames (40 s), replayed in its prefill,
  // and its text, which leads the request's in the prompt. The text rows of
  // an instruction, a text and a reference (textRows) fit one 2048-row
  // projection.
  uint32_t maxReferenceFrames = 500;
  uint32_t maxReferenceTextTokens = 480;
  // The codec's convolution stack on the Neural Engine, when it is
  // available (OMNI_CODEC_ANE=0 turns it off). A request's first chunk
  // (4 frames) decodes on the GPU, where it reaches the client 3-4 ms sooner
  // at one stream (71 against 75 ms), and the stream's histories then move
  // to the Neural Engine.
  bool aneCodec = true;
  // The text rows a lane's prompt and trailing text take at most: the
  // instruction, the reference's text, the text and up to 16 more (role,
  // codec prefix, speaker, tts_bos, tts_eos, tts_pad). TtsModel's
  // maxTextRows.
  [[nodiscard]] uint32_t textRows() const noexcept {
    return maxInstructionTokens + maxReferenceTextTokens + maxTextTokens + 16;
  }
};

// The talker prompt of a request (docs/omni-design.md, TTS pipeline), as
// mlx-audio's Qwen3-TTS _prepare_generation_inputs builds it: the text
// tokens of the lane's text region (the instruction's, the role's
// "<|im_start|>assistant\n", tts_pad over the codec prefix, tts_bos, the
// first text token, then the trailing rows t1..tN-1, tts_eos, tts_pad) and
// the prompt's positions over them, text rows counted from the region's
// start. A continuation's text is the reference's then the request's, and
// its prompt ends with the reference's frames, each over the trailing row
// the step that sampled it read (frame indices from 0).
struct TtsPrompt final {
  std::vector<uint32_t> tokens;
  std::vector<OmniPromptPosition> positions;
  // The text rows frames read, one per frame: t1..tN-1 and tts_eos from
  // trailingBase (trailingCount = N), then tts_pad at row N; a CustomVoice
  // prompt holds its whole text, and trailingBase is its last tts_pad
  // (trailingCount 0).
  uint32_t trailingBase = 0;
  uint32_t trailingCount = 0;
  uint32_t referenceFrames = 0; // the reference's frames, the prompt's last rows
};
[[nodiscard]] TtsPrompt ttsPrompt(const TtsConfig &config, const TtsRequest &request);

class TtsEngine final {
public:
  TtsEngine(metal::MetalBackend &backend, TtsModel &model, ops::omni::CodecDecoder &codec, TtsEngineConfig config);
  ~TtsEngine();
  TtsEngine(const TtsEngine &) = delete;
  TtsEngine &operator=(const TtsEngine &) = delete;

  [[nodiscard]] const TtsEngineConfig &config() const noexcept { return config_; }
  // Queues a request; invalid requests fail at once through the sink.
  void submit(TtsRequest request, EventSink &sink);
  void cancel(uint64_t requestId, EventSink &sink);
  // One tick of work. Returns false when there was nothing to do.
  bool tick(EventSink &sink);
  [[nodiscard]] bool idle() const noexcept;
  [[nodiscard]] uint32_t activeLanes() const noexcept;
  // The smallest KV pool: one request of the longest prompt and
  // kMinimumPoolFrames frames. A pool below one request of the longest
  // prompt and maxFrames frames (a tight --max-memory) shortens what a
  // request may speak to what it holds.
  static constexpr uint32_t kMinimumPoolFrames = 1024; // 82 s
  [[nodiscard]] static uint32_t minimumPages(const TtsConfig &model, const TtsEngineConfig &config) noexcept;
  [[nodiscard]] uint32_t queued() const noexcept { return static_cast<uint32_t>(queue_.size()); }
  [[nodiscard]] uint64_t memoryBytes() const noexcept { return memoryBytes_; }
  // Where the codec's convolutions run, and why not on the Neural Engine.
  [[nodiscard]] const std::string &codecPlacement() const noexcept { return codecPlacement_; }

private:
  struct Lane;
  // What the Neural Engine's evaluations report, from the service's threads,
  // and the GPU command's completion, which the engine thread waits for.
  struct Reports final {
    struct Report final {
      uint32_t lane = 0;
      uint64_t serial = 0;
      bool success = false;
    };
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Report> reports;
    uint64_t completedCommands = 0;
  };
  static constexpr AwakeClock::duration kAneBound = std::chrono::seconds(5);

  void admit(EventSink &sink);
  void step(EventSink &sink);
  void decodeAudio(EventSink &sink, bool final);
  void finish(Lane &lane, DoneReason reason, EventSink &sink);
  // Neural Engine chunks: the fronts of those due, at the head of `graph`
  // with the event's signal after them; their evaluations once the graph's
  // command is committed.
  size_t addAneFronts(metal::CommandGraph &graph);
  void enqueueAneChunks();
  // Submits `graph` and waits for it, emitting the Neural Engine's chunks
  // as they report. With `behind`, builds a second graph while the GPU runs
  // the first and queues it behind (MetalBackend::submitCommandBehind); the
  // timing then spans both.
  metal::CommandTiming runCommand(metal::CommandGraph &graph, EventSink &sink,
                                  const std::function<void(metal::CommandGraph &)> &behind = {});
  // Emits reported chunks; marks failed or late ones for the GPU.
  void drainAneReports(EventSink &sink);
  // Waits for a report while evaluations are in flight and the GPU idles.
  void awaitAneReport(EventSink &sink);
  void retireAne(const std::string &reason);
  void emitAudio(Lane &lane, std::span<const float> samples, EventSink &sink);
  void finishDecoded(EventSink &sink);
  void addPredictor(metal::CommandGraph &graph, uint32_t rows, const std::vector<uint32_t> &laneOrder);
  void addCodeSampling(metal::CommandGraph &graph, metal::MetalBuffer logits, uint32_t rows, uint32_t group,
                       const std::vector<uint32_t> &laneOrder, bool first);
  void writeUniforms(const std::vector<uint32_t> &laneOrder);
  void writeHistory(const std::vector<uint32_t> &laneOrder);
  void collectFrames(const std::vector<uint32_t> &laneOrder, EventSink &sink);
  [[nodiscard]] std::vector<uint32_t> activeOrder() const;

  metal::MetalBackend &backend_;
  TtsModel &model_;
  ops::omni::CodecDecoder &codec_;
  TtsEngineConfig config_;
  uint64_t memoryBytes_ = 0;
  std::unique_ptr<KvPagePool> pool_;
  std::unique_ptr<LanePageTables> talkerTables_, predictorTables_;
  std::unique_ptr<Qwen3Arena> talkerPrefill_, talkerDecode_, predictorPrefill_, predictorDecode_;
  std::vector<std::unique_ptr<Lane>> lanes_;
  std::vector<std::unique_ptr<ops::omni::CodecStream>> streams_; // one per lane
  std::deque<TtsRequest> queue_;
  // Step buffers (rows in active-lane order).
  metal::MetalBuffer textPool_;        // bf16 [lanes][textRows][2048]: prompt text rows and trailing rows
  metal::MetalBuffer positions_;       // OmniPromptPosition [prefill rows]
  metal::MetalBuffer stepCodes_;       // u32 [lanes][16]: last frame per row
  metal::MetalBuffer trailingRows_;    // u32 [lanes]
  metal::MetalBuffer sampled_;         // u32 [lanes][16]: this step's codes
  metal::MetalBuffer uniforms_;        // f32 [lanes][16]
  metal::MetalBuffer history_;         // u32 [lanes][historyStride_]: code 0 of the reference's and the new frames
  metal::MetalBuffer historyCounts_;   // u32 [lanes]
  metal::MetalBuffer predictorInput_;  // bf16 [2 * lanes][2048]
  metal::MetalBuffer predictorLogits_; // f32 [lanes][2048]
  metal::MetalBuffer rowIndex_;        // u32 [lanes]
  metal::MetalBuffer codeColumn_;      // u32 [lanes]: code0 of each row (for the codec embedding gather)
  metal::MetalBuffer frames_;          // u32 [lanes][maxFrames][16]
  metal::MetalBuffer references_;      // u32 [lanes][maxReferenceFrames][16]: continuations' reference frames
  metal::MetalBuffer audio_;           // f32 [lanes][(context + chunk) * 1920]
  uint32_t textRows_ = 0;
  uint32_t historyStride_ = 0;
  uint32_t samplesPerFrame_ = 0;
  bool debug_ = false;
  // The Neural Engine's convolution stack; none on the GPU alone, and once
  // retired no lane starts an evaluation on it.
  std::unique_ptr<ops::omni::CodecAneStack> ane_;
  bool aneRetired_ = false;
  metal::SharedEvent aneEvent_;
  uint64_t aneValue_ = 0;
  uint64_t aneSerial_ = 0;
  std::shared_ptr<Reports> reports_;
  std::string codecPlacement_;
};

} // namespace splash::omni
