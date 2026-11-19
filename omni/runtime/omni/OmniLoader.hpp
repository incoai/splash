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
#include <memory>
#include <span>
#include <variant>
#include <vector>

namespace splash::omni {

// A bf16 table [rows][width] read through a read-only mapping of its
// checkpoint file rather than held in Metal memory: only the pages of the
// rows read are resident, as file cache the system may evict. For a table
// of which a request reads a few rows once, such as Qwen3-TTS's text
// embedding (151936 x 2048, 0.6 GB, of which a request reads its text).
class HostTable final {
public:
  HostTable() = default;
  // The checkpoint tensor `name` under `directory`, which must be bf16
  // [rows][width].
  HostTable(const std::filesystem::path &directory, std::string_view name, uint32_t rows, uint32_t width);
  // Copies rows `indices` to destination, one after another, as stored.
  void copyRows(std::span<const uint32_t> indices, void *destination) const;
  [[nodiscard]] uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] uint32_t width() const noexcept { return width_; }

private:
  std::shared_ptr<const uint8_t> mapping_;
  const uint8_t *data_ = nullptr;
  uint32_t rows_ = 0, width_ = 0;
};

struct TtsWeights final {
  Qwen3Weights talker;    // head: codec_head [3072][2048]
  Qwen3Weights predictor; // no head: the 15 lm_heads below
  Qwen3Projection textFc1, textFc2; // text_projection: fc2(silu(fc1 x + b1)) + b2, 2048 -> 2048 -> 2048
  metal::MetalBuffer textFc1Bias, textFc2Bias; // fp32 [2048]
  Qwen3Projection smallToMtp;                  // [1024][2048]; none at 0.6B (TtsConfig::predictorProjection)
  metal::MetalBuffer smallToMtpBias;           // fp32 [1024]
  HostTable textEmbedding;                     // bf16 [151936][2048], read on the host
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
