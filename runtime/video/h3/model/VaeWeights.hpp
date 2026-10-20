#pragma once

// Converted video VAE decoder (tools/convert_vae.py): one flat blob vae_dec_<tag>.bin + manifest_<tag>.json with the
// byte offsets of the globals and of every layer tensor (256-byte aligned). Each layer's byte range is loaded into its
// own Metal buffer. A layer linear with a scale tensor is W8A8, otherwise fp16.
#import <Metal/Metal.h>

#include "video/h3/engine/Status.hpp"
#include "video/h3/metal/MetalBackend.hpp"
#include "metal/abi/video/Layout.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace h3::model {

struct VaeLayerWeights {
  // qkv / out / fc1 (gate | up) / fc2: weight, int8 scale (empty = fp16 linear), bias
  metal::BufferView qkv, qkvScale, qkvBias, out, outScale, outBias, fc1, fc1Scale, fc1Bias, fc2, fc2Scale, fc2Bias;
  metal::BufferView scale1, scale2, norm1, norm2;  // residual scales, RMSNorm weights
  [[nodiscard]] bool int8Qkv() const { return static_cast<bool>(qkvScale); }
  [[nodiscard]] bool int8Out() const { return static_cast<bool>(outScale); }
  [[nodiscard]] bool int8Fc1() const { return static_cast<bool>(fc1Scale); }
  [[nodiscard]] bool int8Fc2() const { return static_cast<bool>(fc2Scale); }
};

struct VaeGlobalWeights {
  // folded post_quant_conv + x_embedder, register tokens, final LayerNorm, proj_out, RoPE table of a tile-clip,
  // Hadamard signs (2048, 8192)
  metal::BufferView embed, embedBias, registers, lnWeight, lnBias, proj, projBias, cos, sin, signs2048, signs8192;
};

class VaeWeights {
public:
  static Status load(metal::MetalBackend &backend, const std::string &dir, const std::string &tag,
                     std::unique_ptr<VaeWeights> &out);

  [[nodiscard]] const VaeGlobalWeights &globals() const { return globals_; }
  [[nodiscard]] const VaeLayerWeights &layer(int l) const { return layers_[l]; }
  [[nodiscard]] bool anyInt8() const { return anyInt8_; }

private:
  std::vector<id<MTLBuffer>> regions_;
  std::vector<std::pair<int64_t, int64_t>> ranges_;
  VaeGlobalWeights globals_;
  std::array<VaeLayerWeights, H3_VAE_LAYERS> layers_;
  bool anyInt8_ = false;
};

} // namespace h3::model
