#pragma once

// One request: Splash's Metal backend (device, queue, kernel library), request geometry, and the phases that run on
// them (DiT denoising, video VAE decode, TAEH3 preview, audio VAE decode). Each phase loads its weights on first use;
// the DiT phase is released explicitly so a single process can decode afterwards. Failures throw (Errors.hpp).
#include "video/h3/engine/Encoder.hpp"
#include "video/h3/engine/MemoryPlan.hpp"
#include "video/h3/engine/Scheduler.hpp"
#include "video/h3/engine/Types.hpp"
#include "video/h3/engine/WeightStream.hpp"
#include "video/h3/engine/WindowTable.hpp"
#include "video/h3/model/DitWeights.hpp"
#include "video/h3/model/Geometry.hpp"
#include "video/h3/model/Prep.hpp"
#include "video/h3/ops/AudioDecoder.hpp"
#include "video/h3/ops/PreviewDecoder.hpp"
#include "video/h3/ops/VaeDecoder.hpp"

#include <functional>
#include <memory>
#include <optional>
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
  Engine(const EngineConfig &config, const Request &request);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  [[nodiscard]] const model::VideoGeometry &geometry() const { return geometry_; }
  [[nodiscard]] const Request &request() const { return request_; }
  [[nodiscard]] metal::MetalBackend &backend() { return *backend_; }

  // ---- prep: request-constant DiT inputs from the text hidden states (model/Prep.hpp)
  void prep(const model::PrepInputs &inputs, model::Prepared &out);

  // ---- DiT
  void loadDit(const DitInputs &inputs);
  // Deterministic synthetic weights and request inputs (model/Synthetic.hpp; layout and RoPE of the request), no
  // weight files: the kernel check of dev/tests/engine/video_synthetic_metal_test.mm. Forwards then run the first
  // `blocks` blocks only (1: quick check), and only those blocks get weights.
  void loadDitSynthetic(int32_t textRows, int32_t blocks = H3_DIT_BLOCKS);
  void forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn, const float *audioIn,
               float *videoOut, float *audioOut, ForwardStats &stats);
  void denoise(const Schedule &schedule, const float *videoNoise, const float *audioNoise, float *videoOut,
               float *audioOut, const StepCallback &onStep);
  [[nodiscard]] DitInfo ditInfo() const;
  void releaseDit();
  [[nodiscard]] const model::SequenceLayout *layout() const { return dit_ ? &dit_->layout : nullptr; }

  // ---- video VAE and preview (rows: final video latents [video_rows][96])
  void decode(const float *videoRows, uint8_t *frames, DecodeStats &stats);
  [[nodiscard]] int32_t decodeFrames() const { return vaePlan_.outputFrames(); }
  void preview(const float *videoRows, uint8_t *frames, uint8_t *yuv, PreviewStats &stats);

  // ---- audio VAE (rows: final audio latents [audio_rows][32]) -> waveform [2][audioSamples()] at 32 kHz
  void decodeAudio(const float *audioRows, float *wav, AudioStats &stats);
  [[nodiscard]] int64_t audioSamples() const { return ops::AudioDecoder::samples(geometry_.audioRows / 2); }
  [[nodiscard]] int32_t previewFrames() const { return model::previewFrames(geometry_.latentFrames); }

  // Key-range table of a request (no device needed).
  [[nodiscard]] static WindowTable windowTable(const Request &request, int32_t textRows);

private:
  struct DitState {
    model::SequenceLayout layout;
    std::unique_ptr<model::DitWeights> weights;
    std::unique_ptr<WeightStream> stream;
    std::optional<WindowTable> table;
    std::optional<MemoryPlan> plan;
    std::unique_ptr<Scheduler> scheduler;
  };

  void loadDitWith(const DitInputs &inputs, std::unique_ptr<model::DitWeights> synthetic);
  [[nodiscard]] std::string ditDirectory() const;
  void loadVae();
  // de-normalised channels-last latents, the last frame repeated over the plan's pad tokens
  void vaeLatents(const float *videoRows, std::vector<float> &z) const;

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
