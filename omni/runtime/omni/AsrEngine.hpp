#pragma once

// The ASR engine (docs/omni-design.md): a request's audio is turned into
// log-mel features on the CPU, encoded by the audio tower, spliced between
// the prompt's prefix and suffix embeddings and prefilled in chunks; lanes
// then decode greedily in lock step, one token per tick, streaming tokens
// as they come.
//
// With a draft model (Qwen3-ASR-0.6B beside the 1.7B), lanes decode
// speculatively while at most AsrEngineConfig::speculativeLanes lanes are
// active: the draft first runs its tower and prefill for a lane (at the
// first tick that speculates while the lane likely has enough tokens left),
// then each tick it proposes k tokens per lane, the target
// verifies every lane's pending token and proposals in one forward, and a
// lane emits the proposals the target's greedy tokens confirm and the
// target's token after them (docs/omni-performance.md, "Speculative ASR").
//
// With a DFlash draft instead (AsrDFlashDraft: a few Qwen3 layers that read
// the target's own hidden states), the draft follows every lane from its
// admission: each target forward (prefill, plain step, verification) keeps
// the outputs of the layers the draft reads, and the same command projects
// them into the draft's context keys and values. A round is one command: the
// draft's forward over each lane's block [pending, mask x (block - 1)]
// proposes up to block - 1 tokens at once, the target verifies them, and
// the verified rows become context.

#include "AwakeClock.hpp"
#include "metal/MetalBackend.hpp"
#include "omni/KvPages.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniEvents.hpp"
#include "omni/OmniLoader.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/Qwen3Decoder.hpp"
#include "ops/OmniAudioEncoder.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace splash::omni {

struct AsrRequest final {
  uint64_t id = 0;
  std::vector<uint32_t> prefix; // the chat prefix through <|audio_start|>
  std::vector<uint32_t> suffix; // from <|audio_end|> to the assistant prefix (and a forced language)
  std::vector<float> samples;   // 16 kHz mono
  uint32_t maxTokens = 0;       // 0: 4096
  // When the server had read its frame, for OMNI_ASR_TRACE; unset elsewhere.
  std::chrono::time_point<AwakeClock> received{};
};

struct AsrEngineConfig final {
  uint32_t lanes = 8;
  uint32_t maxAudioSeconds = 1200;
  uint32_t maxTokens = 4096;
  uint32_t kvPages = 32768; // 128 KiB each: 4 GiB
  // Whether prefill chunks of an affine checkpoint split their FFN with the
  // Neural Engine (ops::AneFfn); OMNI_DISABLE_ANE=1 also turns it off.
  bool neuralEngine = true;
  // Speculative decoding, when the engine has a draft model: the tokens
  // the draft proposes per lane and round, the most active lanes that still
  // speculate (at most 8), each 0 for the target format's default
  // (asrSpeculationDefaults), and the longest audio a lane speculates on
  // (the draft tower's workspace).
  uint32_t draftTokens = 0;
  uint32_t speculativeLanes = 0;
  uint32_t draftMaxAudioSeconds = 120;
};

// A draft model for speculative decoding: Qwen3-ASR-0.6B's decoder and tower.
struct AsrDraftModel final {
  AsrConfig config;
  AsrTextWeights weights;
  ops::omni::AudioEncoderWeights tower;
};

// A DFlash draft for speculative decoding (OmniConfig.hpp, OmniLoader.hpp).
struct AsrDFlashDraft final {
  DFlashDraftConfig config;
  DFlashDraftWeights weights;
};

// The ASR decoder's prefill FFN split with the Neural Engine (AsrEngine.cpp).
[[nodiscard]] const AneFfnPolicy &asrAneFfnPolicy() noexcept;
// The default draft tokens per round and speculating lanes of an affine
// (true) or bf16 target, the most rows a round verifies (k shrinks to keep
// a round's lanes within them), whether lanes without a draft ride in
// rounds (else lanes speculate only while every active lane has a draft),
// and, where they ride, the milliseconds a speculating lane saves per
// token, against which the draft's tower and prefill for a lane are
// weighed (else the engine estimates it, AsrEngine.cpp).
struct AsrSpeculationDefaults final {
  uint32_t draftTokens = 0;
  uint32_t speculativeLanes = 0;
  uint32_t verifyRows = 0;
  bool riders = false;
  double savedMilliseconds = 0.0;
};
[[nodiscard]] AsrSpeculationDefaults asrSpeculationDefaults(bool affine) noexcept;

class AsrEngine final {
public:
  AsrEngine(metal::MetalBackend &backend, const AsrConfig &config, AsrTextWeights weights,
            ops::omni::AudioEncoderWeights towerWeights, AsrEngineConfig engineConfig,
            std::optional<AsrDraftModel> draft = std::nullopt, std::optional<AsrDFlashDraft> dflash = std::nullopt);
  ~AsrEngine();
  AsrEngine(const AsrEngine &) = delete;
  AsrEngine &operator=(const AsrEngine &) = delete;

  void submit(AsrRequest request, EventSink &sink);
  void cancel(uint64_t requestId, EventSink &sink);
  bool tick(EventSink &sink);
  [[nodiscard]] bool idle() const noexcept;
  [[nodiscard]] uint32_t activeLanes() const noexcept;
  [[nodiscard]] uint32_t queued() const noexcept { return static_cast<uint32_t>(queue_.size()); }
  [[nodiscard]] uint64_t memoryBytes() const noexcept { return memoryBytes_; }
  [[nodiscard]] bool speculative() const noexcept { return draft_ != nullptr || dflash_ != nullptr; }
  [[nodiscard]] bool dflash() const noexcept { return dflash_ != nullptr; }
  // The draft tokens a round proposes per lane, at most, and the most
  // active lanes that speculate (0 without a draft).
  [[nodiscard]] uint32_t draftTokens() const noexcept { return speculative() ? draftTokens_ : 0; }
  [[nodiscard]] uint32_t speculativeLanes() const noexcept { return speculative() ? engineConfig_.speculativeLanes : 0; }

  // Speculation's counters since the engine started.
  struct SpeculationStats final {
    uint64_t rounds = 0;   // lane rounds: one lane verified once
    uint64_t proposed = 0; // draft tokens verified
    uint64_t accepted = 0; // of them, the target's
    uint64_t emitted = 0;  // tokens the rounds emitted
    uint64_t adopted = 0;  // lanes the draft took
    uint64_t dropped = 0;  // active lanes the draft stopped following, or could not take
    // Lane rounds by their accepted proposals (0..15).
    std::array<uint64_t, 16> acceptedRounds{};
  };
  [[nodiscard]] const SpeculationStats &speculationStats() const noexcept { return stats_; }
  // Plain-step rows the GPU ran past their lane's end: the step queued
  // behind the one that emitted a stop token.
  [[nodiscard]] uint64_t wastedStepRows() const noexcept { return wastedRows_; }

private:
  struct Model;
  struct Lane;
  struct DFlash;
  // A DFlash draft's context from a target forward's captured rows: rows
  // [captureRow, captureRow + rows) of the capture, a lane's positions from
  // `position` on (AsrEngine.cpp).
  struct ContextRows final {
    Lane *lane = nullptr;
    uint32_t position = 0, rows = 0, captureRow = 0;
  };
  [[nodiscard]] const HiddenCapture *capture() const noexcept;
  void addDFlashContext(metal::CommandGraph &graph, std::span<const ContextRows> context);
  [[nodiscard]] KvBinding dflashBinding() const;
  // A DFlash round of every lane the draft follows, riders beside them.
  void speculateDFlash(EventSink &sink);
  // The proposals per lane a DFlash round of `lanes` lanes verifies, at
  // most `limit`, and what a token saves in such rounds against plain steps
  // (AsrEngine.cpp), at the acceptance the rounds observe.
  [[nodiscard]] static double dflashMilliseconds(bool affine, uint32_t lanes, uint32_t proposals) noexcept;
  [[nodiscard]] std::pair<uint32_t, double> dflashRound(uint32_t lanes, uint32_t limit) const noexcept;
  void admit(EventSink &sink);
  // Admits the queued requests whose prompts fit one prefill chunk
  // together: their audio towers in one packed pass, then one ragged
  // prefill of all their prompts; false when the queue's head needs the
  // chunked path alone.
  bool admitBatch(EventSink &sink);
  // One token for every active lane, the next step queued behind the one in
  // flight while the same lanes keep decoding (AsrEngine.cpp).
  struct PlainStep;
  void step(EventSink &sink);
  [[nodiscard]] bool chainable() const noexcept;
  std::unique_ptr<PlainStep> submitStep(EventSink &sink, const PlainStep *previous);
  void completeStep(EventSink &sink);
  // One speculative round for every active lane the draft follows, and a
  // token for every other active lane.
  void speculate(EventSink &sink);
  // Emits a lane's new tokens, finishing it on a stop token or its limit;
  // returns whether it is still active.
  bool emit(Lane &lane, std::vector<uint32_t> tokens, EventSink &sink);
  void finish(Lane &lane, DoneReason reason, EventSink &sink);
  // The draft stops following the lane, which decodes plainly from then on.
  void dropDraft(Lane &lane);
  // Keeps a lane's log-mel features from admission for the draft's tower.
  void keepFeatures(Lane &lane, std::vector<float> features);
  // The draft takes the lanes it should follow (AsrEngine.cpp): several
  // whose prompts fit one prefill chunk together in one packed tower pass
  // and one ragged prefill, any other alone.
  [[nodiscard]] std::vector<Lane *> draftCandidates() const;
  void adoptDrafts(std::span<Lane *const> candidates);
  void adoptDraft(Lane &lane);
  void adoptDraftsTogether(std::span<Lane *const> lanes);
  // A request's audio for a tower: its log-mel features and frames (none
  // without audio).
  struct TowerClip final {
    std::span<const float> features;
    uint32_t frames = 0;
  };
  // Encodes the clips with `model`'s tower into its audio rows, each clip's
  // rows after the rows of the clips before it, and returns each clip's
  // first row; adds the commands' GPU seconds to `gpuSeconds`.
  std::vector<uint32_t> encodeAudio(Model &model, std::span<const TowerClip> clips, double &gpuSeconds);
  // The forwards' GPU milliseconds, a round's draft tokens per lane, and
  // what a token saves in rounds of `lanes` lanes (AsrEngine.cpp).
  [[nodiscard]] static double targetMilliseconds(bool affine, uint32_t rows) noexcept;
  [[nodiscard]] static double draftMilliseconds(uint32_t rows) noexcept;
  [[nodiscard]] uint32_t roundDraftTokens(uint32_t lanes, uint32_t riders) const noexcept;
  [[nodiscard]] double savedPerToken(uint32_t lanes) const noexcept;
  // A lane's likely tokens left, the draft's cost of taking it, whether it
  // may, and (where lanes ride) whether rounds are worth their drafts.
  [[nodiscard]] double remainingTokens(const Lane &lane) const noexcept;
  [[nodiscard]] double adoptionMilliseconds(const Lane &lane) const noexcept;
  [[nodiscard]] bool adoptable(const Lane &lane) const noexcept;
  [[nodiscard]] bool speculationPays() const noexcept;
  void addEmbedTokens(metal::CommandGraph &graph, const Model &model, metal::MetalBuffer tokens,
                      metal::MetalBuffer output, uint32_t rows, uint32_t tokenStride = 1);
  // Fills `rows` rows of `arena`'s hidden from `outputRow` on with the
  // prompt positions from `first` on of a request whose audio rows start
  // at `audioOffset` of the model's audio rows; positions past the prompt
  // take `emitted`.
  void addPromptRows(metal::CommandGraph &graph, const Model &model, Qwen3Arena &arena, const AsrRequest &request,
                     uint32_t audioTokens, uint32_t audioOffset, uint32_t first, uint32_t rows, uint32_t outputRow,
                     std::span<const uint32_t> emitted = {});
  [[nodiscard]] KvBinding binding(const Model &model) const;
  // Sets up the prefill FFN's Neural Engine split, or logs why the GPU runs
  // the prefill alone.
  void enableNeuralEngine();

  metal::MetalBackend &backend_;
  AsrConfig config_; // the target's
  AsrEngineConfig engineConfig_;
  std::unique_ptr<Model> target_, draft_;
  std::unique_ptr<DFlash> dflash_;
  uint64_t memoryBytes_ = 0;
  std::unique_ptr<KvPagePool> pool_;
  std::vector<std::unique_ptr<Lane>> lanes_;
  std::deque<AsrRequest> queue_;
  metal::MetalBuffer mel_;        // f32 [maxMelFrames][128]
  metal::MetalBuffer tokens_;     // u32 [2048]: a prefill chunk's text tokens / the step's tokens
  metal::MetalBuffer rowIndex_;   // u32 [2048]: identity
  metal::MetalBuffer stepTokens_; // u32 [lanes]
  metal::MetalBuffer sampled_;    // u32 [lanes]
  // A step queued behind another writes its tokens here, then the next to
  // sampled_ again.
  metal::MetalBuffer chainedSampled_; // u32 [lanes]
  // The plain step submitted and not yet completed (the next may queue
  // behind it: chainable), when the last completed, and the rows completed
  // past their lane's end.
  std::unique_ptr<PlainStep> pendingStep_;
  std::chrono::time_point<AwakeClock> lastStepDone_{};
  uint64_t wastedRows_ = 0;
  // Per emitted token, how often a token followed it and how often that was
  // a stop token.
  struct StopCounts final {
    uint32_t followed = 0, stopped = 0;
  };
  std::unordered_map<uint32_t, StopCounts> followers_;
  // A round's tokens: the draft's catch-up inputs; the verify inputs, k + 1
  // per lane (its pending token, then the proposals the draft's argmax
  // writes in place); the target's greedy token after each.
  metal::MetalBuffer draftInputs_, proposals_, verified_; // u32 [32]
  uint32_t maxMelFrames_ = 0;
  uint32_t maxAudioTokens_ = 0;
  uint32_t draftTokens_ = 0;
  // The most rows a round verifies over all its lanes (k shrinks to fit),
  // and whether lanes without a draft ride in rounds.
  uint32_t verifyRows_ = 0;
  bool riders_ = false;
  double savedMilliseconds_ = 0.0;
  // The decayed counts of lane rounds whose proposal at depth j was
  // verified (the ones before it held) and held (savedPerToken).
  std::array<double, Qwen3Arena::kTableSlots> reached_{}, survived_{};
  // Active lanes averaged over time, as of loadTime_ (AsrEngine.cpp).
  double load_ = 0.0;
  std::chrono::time_point<AwakeClock> loadTime_;
  SpeculationStats stats_;
  bool debug_ = false;
  // OMNI_ASR_TRACE=1: each request's events and commands (build, submit,
  // commit, GPU start and end, wake) as instants on AwakeClock's base, kept
  // here and written to stderr in one piece when a request ends
  // (dev/tools/omni_request_trace.py joins them with the server's).
  bool trace_ = false;
  std::string traceText_;
  void trace(const char *format, ...) __attribute__((format(printf, 2, 3)));
  void traceCommand(const char *kind, uint64_t request, uint32_t rows, std::chrono::time_point<AwakeClock> build,
                    std::chrono::time_point<AwakeClock> submit, std::chrono::time_point<AwakeClock> committed,
                    const metal::CommandTiming &timing);
  // Submits a command and waits for it, traced as `kind`.
  metal::CommandTiming submitTraced(metal::CommandGraph &graph, const char *kind, uint64_t request, uint32_t rows);
};

} // namespace splash::omni
