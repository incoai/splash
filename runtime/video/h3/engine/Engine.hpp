#pragma once

// One request: device and kernel library, request geometry, and the phases that run on them (DiT denoising, video
// VAE decode, TAEH3 preview, audio VAE decode). Each phase loads its weights on first use; the DiT phase is released
// explicitly so a single process can decode afterwards.
#include "video/h3/engine/MemoryPlan.hpp"
#include "video/h3/engine/Scheduler.hpp"
#include "video/h3/engine/Status.hpp"
#include "video/h3/engine/Types.hpp"
#include "video/h3/engine/WeightStream.hpp"
#include "video/h3/engine/WindowTable.hpp"
#include "video/h3/metal/MetalBackend.hpp"
#include "video/h3/model/DitWeights.hpp"
#include "video/h3/model/Geometry.hpp"
#include "video/h3/model/Prep.hpp"
#include "video/h3/ops/AudioDecoder.hpp"
#include "video/h3/ops/PreviewDecoder.hpp"
#include "video/h3/ops/VaeDecoder.hpp"

#include <functional>
#include <memory>
#include <set>
#include <string>

namespace h3 {

struct EngineConfig {
  std::string metallib;
  std::string weights;  // root: dit-turbo/, dit-default/, vae/, preview/
};

// Prepared layout of the packed sequence (borrowed until releaseDit).
struct DitInputs {
  int32_t textRows = 0;
  const uint8_t *tags = nullptr;
  const uint16_t *rope = nullptr;
  const uint16_t *text = nullptr;
};

struct Schedule {
  int32_t steps = 0;
  const uint16_t *adaBlocks = nullptr, *adaFinal = nullptr;
  const float *videoT = nullptr, *audioT = nullptr, *videoSigmas = nullptr, *audioSigmas = nullptr;
};

struct DitInfo {
  int32_t rows = 0, paddedRows = 0, windowItems = 0, chunkedGroups = 0, partialSlots = 0, streamSlots = 0;
  double density = 0, weightsMb = 0, activationsMb = 0, deviceMb = 0;
};

using StepCallback = std::function<void(int32_t step, const ForwardStats &, const float *video, const float *audio)>;

class Engine {
public:
  static Status create(const EngineConfig &config, const Request &request, std::unique_ptr<Engine> &out);
  ~Engine();

  [[nodiscard]] const model::VideoGeometry &geometry() const { return geometry_; }
  [[nodiscard]] const Request &request() const { return request_; }
  [[nodiscard]] const Logger &log() const { return log_; }
  [[nodiscard]] metal::MetalBackend &backend() { return *backend_; }

  // ---- prep: request-constant DiT inputs from the text hidden states (model/Prep.hpp); dump: op-level intermediates
  Status prep(const model::PrepInputs &inputs, model::Prepared &out, const std::string &dump = "");

  // ---- DiT
  Status loadDit(const DitInputs &inputs);
  // Deterministic synthetic weights and request inputs (model/Synthetic.hpp; layout and RoPE of the request), no
  // weight files: kernel and speed checks (h3 bench --synthetic, dev/tests/synthetic_check.sh). Forwards then run
  // the first `blocks` blocks only (1: quick check), and only those blocks get weights.
  Status loadDitSynthetic(int32_t textRows, int32_t blocks = H3_DIT_BLOCKS);
  Status forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn, const float *audioIn,
                 float *videoOut, float *audioOut, ForwardStats &stats);
  Status denoise(const Schedule &schedule, const float *videoNoise, const float *audioNoise, float *videoOut,
                 float *audioOut, const StepCallback &onStep);
  [[nodiscard]] Status ditInfo(DitInfo &out) const;
  void releaseDit();
  // Internals for the block check harness (dev/tests).
  [[nodiscard]] Scheduler *scheduler() { return dit_ ? dit_->scheduler.get() : nullptr; }
  [[nodiscard]] const model::SequenceLayout *layout() const { return dit_ ? &dit_->layout : nullptr; }

  // ---- video VAE and preview (rows: final video latents [video_rows][96])
  Status decode(const float *videoRows, uint8_t *frames, DecodeStats &stats);
  [[nodiscard]] int32_t decodeFrames() const { return vaePlan_.outputFrames(); }
  // Per-stage GPU profile of the first `clips` temporal clips (one command buffer per dispatch; slow).
  Status profileDecode(const float *videoRows, int32_t clips, DecodeStats &stats,
                       std::vector<std::pair<std::string, double>> &stageMs);
  Status preview(const float *videoRows, uint8_t *frames, uint8_t *yuv, PreviewStats &stats);

  // ---- audio VAE (rows: final audio latents [audio_rows][32]) -> waveform [2][audioSamples()] at 32 kHz
  Status decodeAudio(const float *audioRows, float *wav, AudioStats &stats);
  [[nodiscard]] int64_t audioSamples() const { return ops::AudioDecoder::samples(geometry_.audioRows / 2); }
  [[nodiscard]] int32_t previewFrames() const { return model::previewFrames(geometry_.latentFrames); }

  // Key-range table of a request (no device needed).
  static Status windowTable(const Request &request, int32_t textRows, WindowTable &out);

private:
  struct DitState {
    model::SequenceLayout layout;
    std::unique_ptr<model::DitWeights> weights;
    std::unique_ptr<WeightStream> stream;
    WindowTable table;
    MemoryPlan plan;
    std::unique_ptr<Scheduler> scheduler;
  };

  Engine() = default;
  Status loadDitWith(const DitInputs &inputs, std::unique_ptr<model::DitWeights> synthetic);
  [[nodiscard]] std::string ditDirectory() const;
  Status loadVae();
  // de-normalised channels-last latents, the last frame repeated over the plan's pad tokens
  void vaeLatents(const float *videoRows, std::vector<float> &z) const;

  Logger log_;
  EngineConfig config_;
  Request request_;
  model::VideoGeometry geometry_;
  model::VaePlan vaePlan_;
  std::unique_ptr<metal::MetalBackend> backend_;
  std::unique_ptr<DitState> dit_;
  std::unique_ptr<ops::VaeDecoder> vae_;
  std::unique_ptr<ops::PreviewDecoder> preview_;
  std::unique_ptr<ops::AudioDecoder> audio_;
};

} // namespace h3
