#pragma once

// Qwen3.8-Flash-Next (llama.cpp's qwen4exp, transformers' qwen4_exp): the
// Qwen3.5 hybrid of GDN and gated full attention with a sparse MoE FFN, and
// three additions.
//  - Hyper-connections replace every norm: the residual is hcCount parallel
//    streams of hiddenSize values. Before each mixer and each FFN a mix
//    (Qwen4HyperWeights) normalizes every stream, gates it by a low-rank
//    projection of all of them and averages the streams into the block's
//    input; the block's output is added back to every stream scaled by its
//    own injection weight. A last mix without injection feeds the head; the
//    model has no output norm.
//  - QSA sparse attention: the full-attention layers attend, past the first
//    indexerTopBlocks blocks of indexerBlockTokens tokens, to the
//    indexerTopBlocks blocks an indexer scores highest and the current block.
//  - A PLE n-gram embedding feeds one GDN layer: the 2- and 3-gram hashes of
//    every token gather rows of a table of hundreds of millions of rows,
//    which gate a value added to every stream through a dilated convolution.
// Only GGUF sources (unsloth's) are read.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace splash::model {

struct Qwen4ExpLayout final : QwenHybridLayout<1> {
  // Never packed; the names keep PackedTargetFiles well-formed.
  static constexpr std::string_view layerMagic = "MDQ40001";
  static constexpr std::string_view headMagic = "MDQ40002";
  static constexpr QwenFfnKind ffnKind = QwenFfnKind::SparseMoe;

  uint32_t experts = 512;
  uint32_t expertsPerToken = 10;
  uint32_t expertIntermediateSize = 640;
  uint32_t hyperConnections = 4;
  uint32_t hyperRank = 320;
  // The PLE n-gram embedding: its layer, n-gram size, heads per n-gram, row
  // width (per head), the convolution's taps and the hash constants.
  uint32_t pleLayer = 1;
  uint32_t pleNgram = 3;
  uint32_t pleHeadsPerNgram = 8;
  uint32_t pleHeadDimension = 160;
  uint32_t pleConvolutionTaps = 4;
  uint32_t pleEosToken = 248044;
  // QSA indexer.
  uint32_t indexerHeads = 4;
  uint32_t indexerHeadDimension = 128;
  uint32_t indexerTopBlocks = 512;
  uint32_t indexerBlockTokens = 4;
  // The MTP head's full-attention blocks (0 or 1); its KV is the last
  // attention layer of the KV pages.
  uint32_t mtpLayers = 0;

  constexpr Qwen4ExpLayout()
      : QwenHybridLayout{.layers = 48,
                         .hiddenSize = 2560,
                         .vocabularySize = 248320,
                         // qkv 10240 | z 6144 | beta, alpha 96, to whole 64-column tiles.
                         .packedGdnWidth = 16512,
                         .packedFullWidth = 13312,
                         .convolutionDimension = 10240,
                         .gdnKeyHeads = 16,
                         .gdnValueHeads = 48,
                         .gdnHeadDimension = 128,
                         .attentionWidth = 6144,
                         .attentionQueryHeads = 24,
                         .attentionKvHeads = 2,
                         .attentionHeadDimension = 256,
                         .rotaryPairs = 32,
                         .rotaryTheta = 10'000'000.0F,
                         .fullAttentionPeriod = 4,
                         .maskToken = 248077,
                         .stopTokens = {248044, 248046},
                         .hiddenCaptureLayers = {47}} {}

  [[nodiscard]] constexpr uint32_t streamWidth() const noexcept { return hyperConnections * hiddenSize; }
  [[nodiscard]] constexpr uint32_t pleHeads() const noexcept { return (pleNgram - 1) * pleHeadsPerNgram; }
  [[nodiscard]] constexpr uint32_t pleWidth() const noexcept { return pleHeads() * pleHeadDimension; }
  // The PLE convolution's history: (taps - 1) * dilation rows, the dilation
  // being the n-gram size.
  [[nodiscard]] constexpr uint32_t pleHistory() const noexcept { return (pleConvolutionTaps - 1) * pleNgram; }
  // The KV pages also hold the QSA indexer's keys.
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    kv::Layout layout = QwenHybridLayout::kvLayout();
    layout.attentionLayers += mtpLayers;
    layout.indexDimension = indexerHeadDimension;
    return layout;
  }
  // The GDN state and, beside it, the PLE convolution's history (bf16 rows
  // of the streams) and, with the MTP head, the streams of the position
  // before the next one it reads (fp32).
  [[nodiscard]] constexpr uint64_t pleHistoryBytes() const noexcept {
    return uint64_t{pleHistory()} * streamWidth() * GdnStateLayout::bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint64_t mtpCarriedBytes() const noexcept {
    return mtpLayers ? uint64_t{streamWidth()} * sizeof(float) : 0;
  }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    GdnStateLayout layout = QwenHybridLayout::gdnStateLayout();
    layout.auxiliaryBytes = pleHistoryBytes() + mtpCarriedBytes();
    return layout;
  }
  bool operator==(const Qwen4ExpLayout &) const = default;
};

// The hash of the PLE n-gram embedding (llama.cpp qwen4exp.cpp): each
// head's table rows and the layer's multipliers, from the GGUF's metadata.
struct Qwen4PleHash final {
  std::array<uint64_t, 4> multipliers{};
  std::vector<uint32_t> headOffsets;
  std::vector<uint32_t> headVocabularies;
  uint64_t tableRows = 0;
};

// One hyper-connection mix: the streams' norm (hcCount * hidden F32
// multipliers), the low-rank gate's down [rank, hc * hidden] and up
// [hc * hidden, rank] projections, and, for a mix that injects its block's
// output back, the injection weights [hc][hc * hidden] F32.
struct Qwen4HyperWeights final {
  ops::NormWeights norm;
  ops::Projection down;
  ops::Projection up;
  metal::MetalBuffer inject;
};

// The QSA indexer of a full-attention layer: its query [heads * dim, hidden]
// and key [dim, hidden] projections (F32, widened from the GGUF's BF16) and
// their norms.
struct Qwen4IndexerWeights final {
  ops::Projection query;
  ops::Projection key;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
};

// The PLE layer's projections of the gathered n-gram rows, its norms (each
// hc * hidden F32 multipliers) and its dilated convolution's taps
// ([taps][hc * hidden] F32).
struct Qwen4PleWeights final {
  ops::Projection key;
  ops::Projection value;
  ops::NormWeights keyNorm;
  ops::NormWeights queryNorm;
  ops::NormWeights convolutionNorm;
  metal::MetalBuffer convolution;
};

struct Qwen4ExpLayerWeights final {
  Qwen4HyperWeights mixerHyper;
  QwenMixerWeights mixer;
  std::optional<Qwen4IndexerWeights> indexer;
  Qwen4HyperWeights ffnHyper;
  ops::MoeWeights ffn;
  std::optional<Qwen4PleWeights> ple;
};

// The MTP head (llama.cpp graph_mtp): a position's input is the trunk's
// residual streams at the position before (hiddenNorm per stream) and the
// position's token embedding (embedNorm), each stream through embedHidden
// ([e ; h_s] -> hidden); then one full-attention block of the trunk's form,
// whose output streams feed the next draft step, and the head's mix, whose
// output the trunk's vocabulary projection reads.
struct Qwen4MtpWeights final {
  Qwen4ExpLayerWeights block;
  ops::Projection embedHidden;
  ops::NormWeights embedNorm;
  ops::NormWeights hiddenNorm;
  Qwen4HyperWeights head;
};

struct Qwen4ExpWeights final : QwenTargetWeightsBase {
  Qwen4ExpLayout layout;
  std::vector<Qwen4ExpLayerWeights> layers;
  // The last mix, which feeds the head (finalNorm is unused).
  Qwen4HyperWeights finalHyper;
  // The PLE table's native rows of pleHeadDimension values.
  ops::EmbeddingWeights pleTable;
  Qwen4PleHash pleHash;
  std::optional<Qwen4MtpWeights> mtp;
};

class GgufMtpLoader;
// Loads the target and, with `mtp`, its MTP head (layout.mtpLayers).
[[nodiscard]] Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend, Qwen4ExpLayout layout,
                                                  const QwenTargetFiles<Qwen4ExpLayout> &files,
                                                  GgufMtpLoader *mtp = nullptr);

} // namespace splash::model
