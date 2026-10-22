#pragma once

// The engine process of a video model: `serve-native MODEL_DIRECTORY` of an assembly whose record names an h3-video
// model. It speaks the native protocol on stdin and stdout as a language model's engine does, announcing
// FeatureVideo, and takes VideoRequests one at a time, and Cancels; every other client frame ends the stream.
//
// A request runs through the VideoRuntime, a composed pipeline (SGLang Diffusion's ComposedPipeline) of stages
// (stages/), every one on the GPU of this process, in order: the text encoder (TextEncodingStage), request
// preparation, with the video VAE encoder for the request's first and last frames when it carries them (PrepStage),
// the DiT, streamed from SSD (DenoisingStage), the TAEH3 preview when asked (PreviewStage), the video VAE
// (DecodingStage), the audio VAE when asked (AudioDecodingStage) and the mp4 writer (MuxStage). Each stage reports
// VideoProgress, the denoising each step; the preview and the clip arrive as VideoOutputs. A Cancel stops the request
// at the next step or stage boundary. The stages' log lines go to stderr.

#include "metal/MetalBackend.hpp"
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace splash::video {

// What a request's client hears from the runtime: each stage's phase as it starts and as it completes, besides what
// the stage reports.
class PipelineEvents : public StageEvents {
public:
  virtual void start(protocol::VideoPhase phase, uint32_t steps) = 0;
  virtual void complete() = 0;
};

using Stages = std::vector<std::unique_ptr<PipelineStage>>;

class VideoRuntime final {
public:
  explicit VideoRuntime(Stages stages) : stages_(std::move(stages)) {}

  // The model's stages, of the assembly at modelRoot: encoder/ and weights/.
  [[nodiscard]] static Stages stages(metal::MetalBackend &backend, const std::filesystem::path &modelRoot);

  // Runs the request's stages in order, each that runs for it; throws Cancelled at the first stage boundary or
  // denoising step after its Cancel.
  void forward(VideoBatch &batch, PipelineEvents &events) const;

private:
  Stages stages_;
};

// Whether the assembly at modelRoot holds a video model: its model.json's target_format.
[[nodiscard]] bool servesVideo(const std::filesystem::path &modelRoot);

// The engine process on an Apple10 GPU (M5) or later, of the stages `compose` makes on its backend: serves until the
// client ends the stream or a stop signal. The process's exit status (engine::NativeProcessExit).
[[nodiscard]] int serve(const std::filesystem::path &metallib,
                        const std::function<Stages(metal::MetalBackend &)> &compose);
// argv: serve-native MODEL_DIRECTORY, served with the model's stages.
[[nodiscard]] int serve(const std::filesystem::path &metallib, int argc, char **argv);

// argv: video-adaln WEIGHTS VARIANT TRANSFORMER, a conversion step of the installer (install/video.py):
// WEIGHTS/dit-VARIANT/adaln.safetensors, the adaLN tables of the variant (turbo or default), from FL2VA's transformer
// shards in TRANSFORMER (model::Prep::tabulateAdaLn). EX_OK, or EX_SOFTWARE after the error on stderr.
[[nodiscard]] int tabulateAdaLn(const std::filesystem::path &metallib, int argc, char **argv);

} // namespace splash::video
