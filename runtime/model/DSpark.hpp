#pragma once

// DSpark, DeepSeek-V4.1-Flash's own block drafter: the checkpoint's three
// mtp.* stages, which unsloth's dspark-*.gguf holds as a GGUF of
// architecture dflash (DeepSeek's inference/model.py DSparkBlock). Its layers
// are the family's (model/DeepSeek41.hpp) of window attention alone, whose
// window holds, for each position the target ran, the latent of the target's
// hidden row there: the mean of the four streams at the input of each of
// its target layers, joined, projected (fc) and normed (enc.output_norm).
// From the target's last token and blockSize - 1 noise tokens after it, one
// pass of its layers drafts a block of blockSize tokens, whose rows attend
// each other and the window; the target's head writes their logits, which
// its Markov head corrects token by token from the token drafted before
// (rank markovRank), and its confidence head scores each.

#include "model/DFlashDraft.hpp"
#include "model/DeepSeek41.hpp"
#include "model/GgufFile.hpp"
#include "model/WeightImages.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace splash::model {

struct DSparkLayout final {
  // Its layers, as a layout of the family: the target's widths, its own
  // experts, no compressor, indexer or Engram layer, every layer a window's.
  DeepSeek41Layout layers;
  uint32_t blockSize = 0;
  uint32_t noiseToken = 0;
  uint32_t markovRank = 0;
  // The target layers whose input rows it reads, ascending.
  std::array<uint32_t, 4> targetLayerValues{};
  uint32_t targetLayerCount = 0;

  [[nodiscard]] std::span<const uint32_t> targetLayers() const noexcept {
    return {targetLayerValues.data(), targetLayerCount};
  }
};

// The DSpark of the GGUF `file` (its dflash.* metadata, its mask token and
// its Markov head's rank) for `target`, whose widths, attention and
// vocabulary it shares and whose layers it reads. Throws GgufError past them.
[[nodiscard]] DSparkLayout readDSparkLayout(const GgufFile &file, const DeepSeek41Layout &target);

// What the runtime sizes DSpark of `layout` by (DFlashDraftLayout): its
// layers, its target's widths and vocabulary, its latent as one KV head, the
// target layers it reads, its rotary base and its block.
[[nodiscard]] DFlashDraftLayout dsparkDraftLayout(const DSparkLayout &layout);

// The DSpark GGUF of a model's draft/ directory, if it holds one: the one
// .gguf file there.
[[nodiscard]] std::optional<std::filesystem::path> findDSparkGguf(const std::filesystem::path &directory);

struct DSparkWeights final {
  DSparkLayout layout;
  std::vector<DeepSeek41LayerWeights> layers;
  // The target rows' norm and projection [hidden, target layers * hidden].
  ops::NormWeights mainNorm;
  ops::Projection mainProjection;
  // The norm its head reads the streams through, before the target's head.
  ops::NormWeights outputNorm;
  // The Markov head: its embedding rows [vocabulary][rank] and their
  // projection [vocabulary, rank] into the next token's logits' bias, fp32.
  ops::EmbeddingWeights markovEmbedding;
  ops::Projection markovHead;
  // The confidence head's weights, F32 [hidden + rank]: the score of a
  // drafted token from its row's output and the Markov embedding of the
  // token before it.
  metal::MetalBuffer confidence;
  // The identity table of its routed experts' slots, which load with their
  // layers.
  metal::MetalBuffer expertSlots;
  std::vector<WeightFileRecord> files;
};

// Loads the DSpark of `layout` from its GGUF at `path`
// (gguf::planDSparkImages) into `images`, every expert with its layer.
[[nodiscard]] DSparkWeights loadDSparkWeights(metal::MetalBackend &backend, WeightImages &images,
                                              const std::filesystem::path &path, const DSparkLayout &layout);

} // namespace splash::model
