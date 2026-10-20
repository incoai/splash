#pragma once

// The text encoder of the video model: MiniMax-H3 conditions on
// hidden_states[50] of Qwen3-VL-32B, the unnormalized residual stream after
// decoder layer 50 of 64, of every prompt token. It is the language model of
// an MLX affine checkpoint (4 bits in groups of 64), run as a prefill-only
// encoder through those layers: the later layers, the final norm and the head
// are never read. Its token table, norms and projections load as the
// engine's block images of the checkpoint's own quantization and run the
// engine's token gather, RMS norms and prefill linears (ops/); only the
// attention is its own (kernels/video/text_encoder.metal): Q/K RMS norms,
// the rotary embedding of a text prompt, whose M-RoPE positions are the same
// on every axis and so rotate as one, and causal GQA attention over the
// prompt in one pass.

#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace splash::video {

class TextEncoder final {
public:
  // hidden_states[kLayers] rows of kHiddenSize values.
  static constexpr uint32_t kLayers = 50;
  static constexpr uint32_t kHiddenSize = 5120;
  // One prefill pass: the engine's prefill linears' row bound.
  static constexpr uint32_t kMaximumTokens = SPLASH_PREFILL_TOKEN_BUDGET;

  // Plans the checkpoint in `directory` (its config.json and safetensors
  // shards) and checks its configuration; loads no weight. Throws
  // std::invalid_argument for a checkpoint of another model.
  TextEncoder(metal::MetalBackend &backend, const std::filesystem::path &directory);
  ~TextEncoder();
  TextEncoder(const TextEncoder &) = delete;
  TextEncoder &operator=(const TextEncoder &) = delete;

  // hidden_states[kLayers] of a prompt of 1 to kMaximumTokens token ids,
  // f32 [tokens][kHiddenSize] of bf16 values. The weights load for the call
  // and are freed when it returns: the encoder holds memory only while it
  // runs. Throws std::invalid_argument for a prompt it cannot encode.
  [[nodiscard]] std::vector<float> encode(std::span<const uint32_t> tokens);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::video
