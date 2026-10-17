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
// Its weights load from a GGUF only.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/BlockIndexer.h"
#include "metal/abi/HyperConnection.h"
#include "metal/abi/NgramEmbedding.h"
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

struct SparseRowStoreOptions;

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

  constexpr Qwen4ExpLayout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
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
                          .indexerHeads = 4,
                          .indexerHeadDimension = 128,
                          .indexerTopBlocks = 512,
                          .indexerBlockTokens = 4,
                          .pleLayer = 1,
                          .pleNgram = 3,
                          .pleHeadsPerNgram = 8,
                          .pleHeadDimension = 160,
                          .pleConvolutionTaps = 4}} {}
  bool operator==(const Qwen4ExpLayout &) const = default;
};

// The family's kernels are compiled for its shape: the hyper-connection norm's
// streams and width, the n-gram hash's n-grams and heads, and the QSA
// indexer's heads, head width and blocks, which are the sparse attention's.
static_assert(Qwen4ExpLayout{}.hyperConnections == HYPER_CONNECTION_MAXIMUM_STREAMS &&
                  Qwen4ExpLayout{}.hiddenSize == HYPER_CONNECTION_NORM_HIDDEN,
              "the hyper-connection norm kernels are compiled for Flash-Next's streams");
static_assert(Qwen4ExpLayout{}.pleNgram <= NGRAM_EMBEDDING_MAXIMUM_NGRAM &&
                  Qwen4ExpLayout{}.pleHeads() <= NGRAM_EMBEDDING_MAXIMUM_HEADS,
              "the n-gram hash holds Flash-Next's n-grams and heads");
static_assert(Qwen4ExpLayout{}.indexerHeads == BLOCK_INDEXER_HEADS &&
                  Qwen4ExpLayout{}.indexerHeadDimension == BLOCK_INDEXER_DIMENSION &&
                  Qwen4ExpLayout{}.indexerBlockTokens == BLOCK_INDEXER_BLOCK,
              "the QSA kernels are compiled for Flash-Next's indexer");

// The longest n-gram the PLE hash mixes (metal/abi/NgramEmbedding.h).
inline constexpr uint32_t kPleMaximumNgram = NGRAM_EMBEDDING_MAXIMUM_NGRAM;

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
// ([streamWidth()][taps] F32: a channel's taps are adjacent).
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

struct Qwen4ExpWeights final : QwenTargetWeightsBase {
  Qwen4ExpLayout layout;
  std::vector<Qwen4ExpLayerWeights> layers;
  // The last mix, which feeds the head in place of a final norm.
  Qwen4HyperWeights finalHyper;
  // The PLE table's native rows of pleHeadDimension values; empty when the
  // table streams from its file (--ple-memory), where streamedTable reads
  // the rows of pleFormat a step gathers.
  ops::EmbeddingWeights pleTable;
  uint32_t pleFormat = 0;
  Qwen4PleHash pleHash;
};

[[nodiscard]] Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend, Qwen4ExpLayout layout,
                                                  const QwenTargetFiles &files);

// The family's target (model/Qwen4ExpTarget.hpp), whose prefill commands run
// up to `prefillChunks` chunks; the step scratch it keeps itself; and the
// projections its hyper-connection mixes, QSA indexers and PLE layer
// dispatch (qwenTargetGeometry).
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const Qwen4ExpWeights &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);
[[nodiscard]] QwenTargetScratchBytes qwenTargetScratchBytes(const Qwen4ExpWeights &weights,
                                                            const QwenTargetGeometry &geometry,
                                                            uint32_t prefillChunks);
void includeFamilyProjections(QwenTargetGeometry &geometry, const Qwen4ExpWeights &weights);
// The rows of its PLE table, when it streams, that a step stages and the
// read-ahead may have queued (SparseRowStoreOptions::stageRows and
// prefetchRows; Qwen4Steps).
[[nodiscard]] SparseRowStoreOptions streamedTableRows(const Qwen4ExpLayout &layout);

} // namespace splash::model
