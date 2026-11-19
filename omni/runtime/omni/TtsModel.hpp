#pragma once

// The Qwen3-TTS model's parts beyond its two decoders (docs/omni-design.md):
// the text projection (ResizeMLP) of text tokens into talker rows, the code
// predictor's input projection (small_to_mtp) and its 15 heads, and the
// embedding tables the prompt assembly and the per-step input read.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "omni/OmniConfig.hpp"
#include "omni/OmniLoader.hpp"
#include "omni/Qwen3Decoder.hpp"
#include "ops/Linear.hpp"
#include "ops/Omni.hpp"

#include <cstdint>
#include <memory>

namespace splash::omni {

// A projection with a bias, in either format: Q4 through the prefill tiles
// (rows padded to 32) then the bias and activation, or dense with the bias
// in the GEMM epilogue.
class BiasedProjection final {
public:
  BiasedProjection(metal::MetalBackend &backend, const ops::Linear &linear, Qwen3Projection projection,
                   metal::MetalBuffer bias, uint32_t maxRows, std::string_view label);
  [[nodiscard]] uint32_t maxRows() const noexcept { return maxRows_; }
  [[nodiscard]] uint32_t outputSize() const noexcept { return outputSize_; }
  [[nodiscard]] uint32_t inputSize() const noexcept { return inputSize_; }
  // Input and output buffers hold whole tiles of 64 rows past `rows`.
  void add(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer output, uint32_t rows,
           ops::omni::Activation activation) const;

private:
  const ops::Linear *linear_;
  Qwen3Projection projection_;
  metal::MetalBuffer bias_;
  metal::MetalBuffer sums_, raw_, biasRows_;
  ops::LinearScratch scratch_;
  uint32_t maxRows_, outputSize_, inputSize_;
};

class TtsModel final {
public:
  // maxTextRows bounds the text tokens projected per command (a prompt's
  // and its trailing text); maxLanes the decode lanes.
  TtsModel(metal::MetalBackend &backend, TtsWeights weights, const TtsConfig &config, uint32_t maxTextRows,
           uint32_t maxLanes);
  [[nodiscard]] const TtsConfig &config() const noexcept { return config_; }
  [[nodiscard]] const TtsWeights &weights() const noexcept { return weights_; }
  [[nodiscard]] const Qwen3Decoder &talker() const noexcept { return *talker_; }
  [[nodiscard]] const Qwen3Decoder &predictor() const noexcept { return *predictor_; }
  [[nodiscard]] const ops::omni::Embedding::TtsTables &tables() const noexcept { return tables_; }
  [[nodiscard]] uint32_t maxTextRows() const noexcept { return maxTextRows_; }
  [[nodiscard]] uint32_t maxLanes() const noexcept { return maxLanes_; }

  // tokens (u32 [rows]) -> output (bf16 [rows][2048], capacity whole 64-row
  // tiles): fc2(silu(fc1(text_embedding[token]) + b1)) + b2.
  void addTextProjection(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer output,
                         uint32_t rows) const;
  // input (bf16 [rows][2048]) -> output (bf16 [rows][1024]): the code
  // predictor's input projection with its bias. Rows at most 2 * maxLanes.
  void addPredictorInput(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer output,
                         uint32_t rows) const;
  // codes (u32, row r's at r * OMNI_CODE_GROUPS) -> output (bf16 [rows]
  // [1024]): the predictor input of sub-step `step` (1..14), the projection
  // of predictor embedding table step - 1 at each row's code, gathered from
  // the projected tables built at load. For a row count whose layer-0 QKV
  // the first-layer tables match, it also gathers the rows' layer-0 QKV
  // into `qkv` (bf16 [rows][4096]) and returns true
  // (ForwardStep::firstQkvReady).
  bool addPredictorCodeInput(metal::CommandGraph &graph, metal::MetalBuffer codes, uint32_t step,
                             metal::MetalBuffer output, metal::MetalBuffer qkv, uint32_t rows) const;
  // The predictor head of `group` (0..14) over the selected rows of a
  // predictor arena (its final-normed hidden) into fp32 logits [rows][2048].
  void addPredictorHead(metal::CommandGraph &graph, Qwen3Arena &arena, uint32_t group, uint32_t rows,
                        metal::MetalBuffer logits, metal::MetalBuffer raw = {}) const;
  // Whether the predictor head of `rows` rows folds the final norm: the
  // forward then leaves its selected rows raw (ForwardStep::rawSelected)
  // and addPredictorHead takes them as `raw`.
  [[nodiscard]] bool predictorHeadFoldsNorm(const Qwen3Arena &arena, uint32_t rows) const;

private:
  metal::MetalBackend *backend_;
  TtsConfig config_;
  TtsWeights weights_;
  std::unique_ptr<Qwen3Decoder> talker_, predictor_;
  ops::omni::Embedding::TtsTables tables_;
  std::unique_ptr<BiasedProjection> fc1_, fc2_, smallToMtp_;
  metal::MetalBuffer textEmbedded_, fc1Out_;
  // small_to_mtp(predictor embedding table g) for g in 0..13, bf16 [14]
  // [predictorVocabulary][1024].
  metal::MetalBuffer projectedTables_;
  // The predictor's layer-0 input norm and QKV projection of every
  // projected table row, bf16 [14][predictorVocabulary][4096], as a decode
  // step of qkvTableRows_ rows computes them.
  metal::MetalBuffer qkvTables_;
  uint32_t qkvTableRows_ = 0;
  ops::LinearScratch headScratch_;
  uint32_t maxTextRows_, maxLanes_;
};

} // namespace splash::omni
