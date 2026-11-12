#pragma once

// Ref2VA reference latents: diffusers' encode_vae_condition (its MiniMax-H3 modular pipeline) after the VAE
// encoder (video/ops/VaeEncoder.hpp): the posterior sampled under a fresh CPU torch generator, rounded to float16, then
// normalized with the VAE's latents_mean / latents_std (Latents.hpp).
#include <cstddef>
#include <cstdint>
#include <vector>

namespace splash::video::model {

// torch.randn(count, generator=torch.Generator().manual_seed(seed)) for a contiguous fp32 tensor of count >= 16
// elements (ATen normal_fill: mt19937 24-bit uniforms, Box-Muller over blocks of 16, the tail block redrawn).
void torchRandn(uint64_t seed, size_t count, float *out);

// encode_vae_condition after the encoder: moments [F][h][w][48] (mean 24 | logvar 24) -> normalized latents
// [24][F][h][w] (sampled with torch CPU seed `seed` noise in (1, 24, F, h, w) order, fp16 round trip,
// (x - mean) / std) — the layout of model::ReferenceInput::video in Prep.hpp.
void referenceLatents(const std::vector<float> &moments, int32_t F, int32_t h, int32_t w, uint64_t seed,
                      std::vector<float> &out);

// The same with given noise [24][F][h][w]; null noise: the posterior mode (no sampling).
void referenceLatentsWithNoise(const std::vector<float> &moments, int32_t F, int32_t h, int32_t w, const float *noise,
                               std::vector<float> &out);

} // namespace splash::video::model
