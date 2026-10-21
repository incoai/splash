#pragma once

// Video VAE ViT decoder (36 layers, dim 2048) with VideoVAE.decode semantics: temporal clips of 5 latent frames + 2
// overlap tokens, 256 px tiles with >= 64 px overlaps. Per clip, batches of tile-clips (16 x 16 latent, 1797 rows each)
// run embed -> 36 x [rms -> qkv (+ qk-norm, RoPE) -> attention -> out (+ scale1, residual) -> rms -> fc1 + SwiGLU ->
// fc2 (+ scale2, residual)] -> LayerNorm -> proj_out, and one finish kernel stitches the clip's tiles, cross-fades with
// the previous clip and writes uint8 frames, all in one command. The host copies clip c - 1's frames while the GPU
// runs clip c. Precision per linear: W8A8 (Hadamard + per-token int8) or fp16; fp32 residual, norms and statistics.
#include "video/h3/engine/Encoder.hpp"
#include "video/h3/engine/Types.hpp"
#include "video/h3/model/Geometry.hpp"
#include "video/h3/model/VaeWeights.hpp"

#include <array>
#include <memory>

namespace h3::ops {

class VaeDecoder {
public:
  VaeDecoder(metal::MetalBackend &backend, std::unique_ptr<model::VaeWeights> weights);

  // latents: de-normalised, time-padded fp32 [T][LH][LW][24]; frames: plan.outputFrames() x [H][W][3] uint8.
  void decode(const float *latents, int32_t latentFrames, const model::VideoGeometry &geometry,
              const model::VaePlan &plan, uint8_t *frames, DecodeStats &stats);

private:
  // A buffer of at least `bytes`, reallocated when smaller.
  void ensure(size_t bytes, metal::MetalBuffer &buffer, const char *what);
  void producer(Encoder &e, const char *kernel, const metal::MetalBuffer &x, int32_t rows, int32_t K, int32_t groups,
                BufferView normWeight, BufferView normBias, BufferView signs, const metal::MetalBuffer &y);
  void gemm(Encoder &e, const char *kernel, int32_t tm, const metal::MetalBuffer &a, int32_t rows, int32_t N, int32_t K,
            BufferView w, BufferView scale, BufferView bias, const metal::MetalBuffer &y, size_t yOffset, int32_t ldo,
            BufferView aux, BufferView aux2);
  void glu(Encoder &e, bool int8, const metal::MetalBuffer &a, int32_t rows, const model::VaeLayerWeights &w);
  void attention(Encoder &e, int32_t tiles);
  void embed(Encoder &e, int32_t tiles, const int32_t *origins, int32_t latentFrames, int32_t LH, int32_t LW);
  void layer(Encoder &e, int l, int32_t tiles);
  void head(Encoder &e, int32_t tiles, size_t outRow0);

  metal::MetalBackend &b_;
  std::unique_ptr<model::VaeWeights> w_;
  int32_t S_ = H3_VAE_TILE_ROWS, Spad_ = (H3_VAE_TILE_ROWS + 31) / 32 * 32;
  metal::MetalBuffer x_, a16_, o_, hidden_, aq_, sa_, q_, k_, v_;
  metal::MetalBuffer out_, latents_, bad_, dummy_;
  std::array<metal::MetalBuffer, 2> overlap_{}, stage_{};
};

} // namespace h3::ops
