#pragma once

// Qwen3.8-Flash-Next (llama.cpp's qwen4exp, transformers' qwen4_exp): the
// Qwen3.5 hybrid of GDN and gated full attention with a sparse MoE FFN, and
// three additions.
//  - Hyper-connections replace every norm: the residual is hyperConnections
//    parallel streams of hiddenSize values. Before each mixer and each FFN a mix
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
// Its weights load from a GGUF only. Its MTP head (Qwen4MtpWeights), one more
// block of the trunk's form, drafts the decode's proposals when no DFlash2
// draft is installed.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/Qwen4.h"
#include "metal/abi/RoPE.h"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace splash::model {

class SparseRowStore;

struct Qwen4ExpLayout final : QwenHybridLayout {
  // The family's name, as the installer pairs its draft (install/families.py).
  static constexpr std::string_view family = "Qwen3.8-Flash-Next";
  // Its config's text model type (transformers') and its GGUFs' architecture
  // (llama.cpp's); no safetensors checkpoint of it loads.
  static constexpr std::string_view modelType = "qwen4_exp_text";
  static constexpr std::string_view ggufArchitecture = "qwen4exp";
  static constexpr bool loadsSafetensors = false;
  // The PLE n-gram hash's end-of-sequence token, the first stop token, which
  // cuts an n-gram short; and the token it reads at every position of an
  // image, which a prompt holds there (<|image_pad|>, server/frontend.py). A
  // GGUF states both (ple.eos_token_id, ple.image_token_id).
  uint32_t pleEosToken = stopTokens[0];
  uint32_t pleImageToken = 248056;
  // The MTP head's full-attention blocks, 1 when the head drafts
  // (ModelDescriptor.mm: a model root with mtp/ and no draft/), else 0. Its
  // KV is the last attention layer of the KV pages.
  uint32_t mtpLayers = 0;
  // The tokens the MTP head drafts from (model/Qwen4DraftVocabulary.hpp): 0
  // for the whole vocabulary, else its draft vocabulary's, whose rows of the
  // vocabulary projection it reads alone. The verify reads them all.
  uint32_t mtpDraftTokens = 0;

  constexpr Qwen4ExpLayout()
      : QwenHybridLayout{{.maximumContextTokens = 262'144,
                          .layers = 48,
                          .hiddenSize = 2560,
                          .vocabularySize = 248320,
                          // qkv 10240 | z 6144 | beta, alpha 96, to whole 64-column tiles.
                          .packedGdnWidth = 16512,
                          .packedFullWidth = 13312,
                          .convolutionDimension = 10240,
                          .gdnKeyHeads = 16,
                          .gdnValueHeads = 48,
                          .gdnHeadDimension = 128,
                          .gdnSigmoidGate = true,
                          .attentionWidth = 6144,
                          .attentionQueryHeads = 24,
                          .attentionKvHeads = 2,
                          .attentionHeadDimension = 256,
                          .rotaryPairs = SPLASH_TARGET_ROPE_PAIRS,
                          .rotaryTheta = 10'000'000.0F,
                          .fullAttentionPeriod = 4,
                          .maskToken = 248077,
                          .stopTokens = {248044, 248046},
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .experts = 512,
                          .expertsPerToken = 10,
                          .expertIntermediateSize = 640,
                          .hyperConnections = 4,
                          .hyperRank = 320,
                          .indexerHeads = QWEN4_QSA_HEADS,
                          .indexerHeadDimension = QWEN4_QSA_DIMENSION,
                          .indexerTopBlocks = 512,
                          .indexerBlockTokens = QWEN4_QSA_BLOCK,
                          .pleLayer = 1,
                          .pleNgram = 3,
                          .pleHeadsPerNgram = 8,
                          .pleHeadDimension = 160,
                          .pleConvolutionTaps = 4}} {}

  // The MTP head's KV after the trunk's attention layers.
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    kv::Layout layout = QwenHybridLayout::kvLayout();
    layout.attentionLayers += mtpLayers;
    return layout;
  }
  // With the MTP head, each state also carries the fp32 streams of the
  // position before the next one the head reads (QwenTarget::mtpCarried),
  // after the PLE history.
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

// The longest n-gram the PLE hash mixes (metal/abi/Qwen4.h).
inline constexpr uint32_t kPleMaximumNgram = QWEN4_PLE_MAXIMUM_NGRAM;

// The hash of the PLE n-gram embedding, from the GGUF's metadata: the multiplier of each n-gram position (0 past
// pleNgram), each head's table rows and the table's rows.
struct Qwen4PleHash final {
  std::array<uint64_t, kPleMaximumNgram> multipliers{};
  std::vector<uint32_t> headOffsets;
  std::vector<uint32_t> headVocabularies;
  uint64_t tableRows = 0;
};

// One hyper-connection mix of the streamWidth() values of the streams: their
// norm (streamWidth() F32 multipliers), the low-rank gate's down
// [hyperRank, streamWidth()] and up [streamWidth(), hyperRank] projections,
// and, for a mix that injects its block's output back, the injection weights
// [hyperConnections][streamWidth()] F32.
struct Qwen4HyperWeights final {
  ops::NormWeights norm;
  ops::Projection down;
  ops::Projection up;
  metal::MetalBuffer inject;
};

// The QSA indexer of a full-attention layer: its query [heads * dim, hidden]
// and key [dim, hidden] projections (float tensors as the GGUF stores them,
// BF16 or F32) and their norms.
struct Qwen4IndexerWeights final {
  ops::Projection query;
  ops::Projection key;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
};

// The PLE layer's projections of the gathered n-gram rows, its norms (each
// streamWidth() F32 multipliers) and its dilated convolution's taps
// ([taps][streamWidth()] F32).
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
  ops::BlockMoeWeights ffn;
  std::optional<Qwen4PleWeights> ple;
};

// The MTP head (blk.<layers> of the model's MTP GGUF, GgufImage.cpp
// planMtpImage): a position's input is the trunk's residual streams at the
// position before (hiddenNorm per stream) and the position's token embedding
// (embedNorm), each stream through embedHidden ([e ; h_s] -> hidden); then
// one full-attention block of the trunk's form, whose output streams feed the
// next draft step, and the head's mix, whose output the trunk's vocabulary
// projection reads.
// The vocabulary projection's rows of the draft vocabulary's tokens
// (draft-head.bin): [mtpDraftTokens, hidden] in whole tiles, and the tokens,
// ascending (uint32 each).
struct Qwen4DraftHead final {
  ops::Projection logits;
  metal::MetalBuffer tokens;
};

struct Qwen4MtpWeights final {
  Qwen4ExpLayerWeights block;
  ops::Projection embedHidden;
  ops::NormWeights embedNorm;
  ops::NormWeights hiddenNorm;
  Qwen4HyperWeights head;
  // With a draft vocabulary (Qwen4ExpLayout::mtpDraftTokens), the head the
  // draft steps project with in place of the target's.
  std::optional<Qwen4DraftHead> draftHead;
};

struct Qwen4ExpWeights final : QwenTargetWeightsBase {
  Qwen4ExpLayout layout;
  std::vector<Qwen4ExpLayerWeights> layers;
  // The last mix, which feeds the head (finalNorm is unused).
  Qwen4HyperWeights finalHyper;
  // The PLE table's native rows of pleHeadDimension values; empty when the
  // table streams from its file (--ple-ssd-cache), where pleStore reads the
  // rows of pleFormat a step gathers.
  ops::EmbeddingWeights pleTable;
  std::shared_ptr<SparseRowStore> pleStore;
  uint32_t pleFormat = 0;
  Qwen4PleHash pleHash;
  // The MTP head, with layout.mtpLayers.
  std::optional<Qwen4MtpWeights> mtp;
};

class GgufMtpLoader;
// Loads the target and, with `mtp`, its MTP head (layout.mtpLayers).
[[nodiscard]] Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend, Qwen4ExpLayout layout,
                                                  const QwenTargetFiles &files, GgufMtpLoader *mtp = nullptr);

} // namespace splash::model
