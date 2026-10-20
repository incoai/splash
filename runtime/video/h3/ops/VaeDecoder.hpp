#pragma once

// Video VAE ViT decoder (36 layers, dim 2048) with VideoVAE.decode semantics: temporal clips of 5 latent frames + 2
// overlap tokens, 256 px tiles with >= 64 px overlaps. Per clip, batches of tile-clips (16 x 16 latent, 1797 rows each)
// run embed -> 36 x [rms -> qkv (+ qk-norm, RoPE) -> attention -> out (+ scale1, residual) -> rms -> fc1 + SwiGLU ->
// fc2 (+ scale2, residual)] -> LayerNorm -> proj_out in one command buffer; one finish kernel stitches the clip's
// tiles, cross-fades with the previous clip and writes uint8 frames. The host copies clip c - 1's frames while the GPU
// runs clip c. Precision per linear: W8A8 (Hadamard + per-token int8) or fp16; fp32 residual, norms and statistics.
#import <Metal/Metal.h>

#include "video/h3/engine/Types.hpp"
#include "video/h3/metal/MetalBackend.hpp"
#include "video/h3/model/Geometry.hpp"
#include "video/h3/model/VaeWeights.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace h3::ops {

class VaeDecoder {
public:
  enum Stage { Embed, Norm, Qkv, Attention, Quantize, Out, Fc1, Fc2, LayerNorm, Proj, Finish, kStages };
  static const char *stageName(int stage);

  static Status create(metal::MetalBackend &backend, std::unique_ptr<model::VaeWeights> weights,
                       std::unique_ptr<VaeDecoder> &out);

  // latents: de-normalised, time-padded fp32 [T][LH][LW][24]; frames: plan.outputFrames() x [H][W][3] uint8.
  Status decode(const float *latents, int32_t latentFrames, const model::VideoGeometry &geometry,
                const model::VaePlan &plan, uint8_t *frames, DecodeStats &stats);

  // Profile mode: one command buffer per dispatch, GPU time per stage (slow; for dev/benchmarks/vae_profile.py).
  void setProfile(bool on);
  [[nodiscard]] const std::array<double, kStages> &stageMs() const { return stageMs_; }
  [[nodiscard]] const std::array<int64_t, kStages> &stageDispatches() const { return stageDispatches_; }

private:
  class Recorder;
  explicit VaeDecoder(metal::MetalBackend &backend) : b_(backend) {}
  Status allocateActivations();
  Status ensure(size_t bytes, __strong id<MTLBuffer> &buffer, size_t &have, const char *what);
  void producer(Recorder &r, int stage, const char *kernel, id<MTLBuffer> x, int32_t rows, int32_t K, int32_t groups,
                metal::BufferView normWeight, metal::BufferView normBias, metal::BufferView signs, id<MTLBuffer> y);
  void gemm(Recorder &r, int stage, const char *kernel, int32_t tm, id<MTLBuffer> a, int32_t rows, int32_t N,
            int32_t K, metal::BufferView w, metal::BufferView scale, metal::BufferView bias, id<MTLBuffer> y,
            size_t yOffset, int32_t ldo, metal::BufferView aux, metal::BufferView aux2);
  void glu(Recorder &r, bool int8, id<MTLBuffer> a, int32_t rows, const model::VaeLayerWeights &w);
  void attention(Recorder &r, int32_t tiles);
  void embed(Recorder &r, int32_t tiles, const int32_t *origins, int32_t latentFrames, int32_t LH, int32_t LW);
  void layer(Recorder &r, int l, int32_t tiles);
  void head(Recorder &r, int32_t tiles, size_t outRow0);

  metal::MetalBackend &b_;
  std::unique_ptr<model::VaeWeights> w_;
  int32_t S_ = H3_VAE_TILE_ROWS, Spad_ = (H3_VAE_TILE_ROWS + 31) / 32 * 32;
  id<MTLBuffer> x_ = nil, a16_ = nil, o_ = nil, hidden_ = nil, aq_ = nil, sa_ = nil, q_ = nil, k_ = nil, v_ = nil;
  id<MTLBuffer> out_ = nil, latents_ = nil, bad_ = nil, dummy_ = nil;
  std::array<id<MTLBuffer>, 2> overlap_{}, stage_{};
  size_t outBytes_ = 0, latentBytes_ = 0, overlapBytes_ = 0, stageBytes_ = 0;
  bool profile_ = false;
  std::array<double, kStages> stageMs_{};
  std::array<int64_t, kStages> stageDispatches_{};
};

} // namespace h3::ops
