#pragma once

// Loads the Omni models' weights from an MLX checkpoint directory into
// weight images (model/WeightImages.hpp): affine Q4 projections reordered
// into the kernels' storage tiles (model/AffinePreparation.hpp), bf16
// projections and tables copied as stored, norms and biases copied, the
// biases then converted to fp32 on the GPU.

#include "metal/MetalBackend.hpp"
#include "model/WeightImages.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/Qwen3Decoder.hpp"
#include "ops/Weights.hpp"

#include <filesystem>
#include <variant>
#include <vector>

namespace splash::omni {

struct TtsWeights final {
  Qwen3Weights talker;    // head: codec_head [3072][2048]
  Qwen3Weights predictor; // no head: the 15 lm_heads below
  Qwen3Projection textFc1, textFc2; // text_projection: fc2(silu(fc1 x + b1)) + b2, 2048 -> 2048 -> 2048
  metal::MetalBuffer textFc1Bias, textFc2Bias; // fp32 [2048]
  Qwen3Projection smallToMtp;                  // [1024][2048]
  metal::MetalBuffer smallToMtpBias;           // fp32 [1024]
  metal::MetalBuffer textEmbedding;            // bf16 [151936][2048]
  metal::MetalBuffer codecEmbedding;           // bf16 [3072][2048]
  metal::MetalBuffer predictorEmbeddings;      // bf16 [15][2048][2048]
  std::vector<Qwen3Projection> predictorHeads; // 15 x [2048][1024]
};

struct AsrTextWeights final {
  Qwen3Weights decoder; // head: the tied embedding
  // The token table the gather reads: Q4 rows (embedding_q4_h2048) or the
  // bf16 table (omni_embed_bf16).
  std::variant<ops::EmbeddingWeights, metal::MetalBuffer> embedding;
};

// A DFlash draft (OmniConfig.hpp DFlashDraftConfig): its layers and final
// norm (no head; the engine gives it the target's), and the projection of
// the captured target layer outputs, [hidden][layers x hidden], with its
// norm.
struct DFlashDraftWeights final {
  Qwen3Weights decoder; // its own head when it has one (draftVocabulary rows)
  Qwen3Projection fc;
  ops::NormWeights hiddenNorm;
  // With its own head: each draft token's target token (d2t's offsets added).
  std::vector<uint32_t> tokenMap;
};

[[nodiscard]] TtsWeights loadTtsWeights(metal::MetalBackend &backend, model::WeightImages &images,
                                        const std::filesystem::path &root, const TtsConfig &config, bool quantized);
[[nodiscard]] DFlashDraftWeights loadDFlashDraftWeights(metal::MetalBackend &backend, model::WeightImages &images,
                                                        const std::filesystem::path &root,
                                                        const DFlashDraftConfig &config);
[[nodiscard]] AsrTextWeights loadAsrTextWeights(metal::MetalBackend &backend, model::WeightImages &images,
                                                const std::filesystem::path &root, const AsrConfig &config,
                                                bool quantized);

// Packed copies (ops::omni::DenseWeights::packed) of a bf16 model's
// projections and heads, which decode steps of one or two rows stream
// instead of the bf16 weights, bit for bit the same: about 0.78 of their
// bytes, which the copies add to memory (the bf16 weights stay for the
// tiles of more rows and the embedding gathers). Like the fused bf16 QKV
// they are allocated outside the weight images and stay through an idle
// release. Packs nothing for quantized weights, under OMNI_PACKED_BF16=0, or
// when the copies would pass `limit` bytes; returns the bytes allocated.
uint64_t packDenseWeights(metal::MetalBackend &backend, TtsWeights &weights, uint64_t limit);
uint64_t packDenseWeights(metal::MetalBackend &backend, AsrTextWeights &weights, uint64_t limit);

} // namespace splash::omni
