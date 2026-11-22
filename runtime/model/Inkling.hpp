#pragma once

// Inkling 2 Mini (thinkingmachines/Inkling-2-Mini; GGUF architecture
// inkling2_loop, text model type inkling2_loop_text): a decoder of 24 weight
// layers whose layers 6 to 17 run three times in a row, so a row runs
// through 48 layers, each with its K/V and short-convolution state of its
// own: layers 0 to 5 (the prelude), the three passes of weight layers 6 to
// 17 (layers 6 to 41), and weight layers 18 to 23 (the coda, layers 42 to
// 47). Every pass starts again from the prelude's output; passes 1 and 2
// gate each core layer's inputs with messages of the pass before
// (QwenTargetDimensions::weightLayer, loopPass). What its layers compute:
//  - Attention of 24 query heads over 8 KV heads of 128, without RoPE: an
//    additive bias of each query's 16 relative values against a bank of
//    1,024 key distances, a sink logit per row and head from a projection,
//    L2-clipped queries (each scaled per head and channel first) and keys,
//    RMS-normed values, and a sigmoid gate on its output. Every sixth weight
//    layer (5, 11, 17, 23; layers 5, 11, ..., 47) attends every key and
//    takes separate query, key and value inputs, with a log position scale
//    on queries past 8,191 tokens; the others a window of the 1,024 keys that
//    end at a row, from one input.
//  - Causal short convolutions of 4 taps on its keys and values (before the
//    clip and norm), on its attention output and on its FFN output.
//  - Its residual is a depth attention over the outputs of every block of 6
//    layers ("block attention residuals"): each layer mixes its inputs from
//    them with queries of its own, and the head its last.
//  - Every layer's FFN a MoE of 128 experts of 1,152, 8 a token, selected by
//    sigmoid score plus a bias and weighted by sigmoid scores normalized and
//    times 8, with a soft-capped SwiGLU ("SiTU"), and no shared expert.
//  - An untied head whose logits are soft-capped to 30, of a vocabulary of
//    201,024 rows of which 200,058 are tokens; one token ends a turn.
// It has no draft. Its weights load from a GGUF only (GgufImage.cpp
// InklingGguf), each weight layer's tensors once, which every layer that
// runs it shares.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/QuantFormat.h"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace splash::model {

// The loop and windows of Inkling's 48 layers.
inline constexpr QwenTargetDimensions kInklingLoop{.layers = 48, .loopStart = 6, .loopEnd = 18, .loopPasses = 3};

// Its layers over a window, a bit per layer in run order: those that run a
// weight layer other than every sixth (5, 11, 17 and 23), whose every layer
// (5, 11, 17, 23, 29, 35, 41 and 47) attends every key.
inline constexpr uint64_t kInklingWindowedLayers = [] {
  uint64_t layers = 0;
  for (uint32_t layer = 0; layer < kInklingLoop.layers; ++layer)
    if ((kInklingLoop.weightLayer(layer) + 1) % 6 != 0) layers |= uint64_t{1} << layer;
  return layers;
}();
static_assert(kInklingWindowedLayers == 0x7df7df7df7df, "Inkling's global layers are every sixth in run order too");

struct InklingLayout final : QwenHybridLayout {
  // Its text config's model type (under text_config), which states its
  // window as model_max_length, and its GGUFs' architecture; its weights
  // load from a GGUF only (ModelDescriptor.mm validateTextConfig).
  static constexpr std::string_view modelType = "inkling2_loop_text";
  static constexpr std::string_view contextKey = "model_max_length";
  static constexpr std::string_view ggufArchitecture = "inkling2_loop";

  // What its kernels compute, which its GGUFs state (InklingGguf): the L2
  // clip of queries and keys (16 sqrt(2)) and of relative values; the log
  // position scale tau(p) = 1 + alpha ln(max(1, (p + 1) / floor)) of a
  // global layer's queries and relative values; the layers of a block of
  // the depth attention; the rank of a loop gate; and the messages of a
  // pass, to the query, key, value and FFN inputs, in this order.
  static constexpr float queryKeyClip = 22.627416997969522F;
  static constexpr float relativeClip = 8.0F;
  static constexpr uint32_t logScaleFloor = 8192;
  static constexpr float logScaleAlpha = 0.1F;
  static constexpr uint32_t residualBlockLayers = 6;
  static constexpr uint32_t loopGateRank = 64;
  static constexpr uint32_t loopMessages = 4;
  // Its MoE's and head's values: the routing weights' scale, the SiTU
  // activation's gate and up softcaps (4 tanh(g / 4) sigmoid(g) 25 tanh(u /
  // 25)), the logits' softcap and the tokens of its vocabulary, the rows
  // before its head's padding.
  static constexpr float expertWeightsScale = 8.0F;
  static constexpr float gateSoftcap = 4.0F;
  static constexpr float upSoftcap = 25.0F;
  static constexpr float finalLogitSoftcap = 30.0F;
  static constexpr uint32_t tokenVocabulary = 200'058;

  // Inkling 2 Mini, its one family. Its attention's q, gate, k and v rows
  // are those of its sides (InklingSides), not a packed projection. Its
  // router weights the experts it selects by their sigmoid scores over the
  // sum of theirs, which expertWeightsScale then scales; it selects them by
  // those scores plus a bias (ops::BlockMoeWeights::routerBias). Its
  // attention holds no q or k norm, whose storage in a safetensors
  // checkpoint attentionNormMean tells: it states their identity, 1, as no
  // such checkpoint of it loads.
  constexpr InklingLayout()
      : QwenHybridLayout{{.nativeContextTokens = 1'048'576,
                          .layers = kInklingLoop.layers,
                          .hiddenSize = 2304,
                          .vocabularySize = 201'024,
                          .logitSoftcap = finalLogitSoftcap,
                          .validVocabularySize = tokenVocabulary,
                          .packedFullWidth = 8192,
                          .attentionWidth = 3072,
                          .attentionQueryHeads = 24,
                          .attentionKvHeads = 8,
                          .attentionHeadDimension = 128,
                          .attentionGate = true,
                          .attentionNormMean = 1.0F,
                          .slidingWindow = 1024,
                          .slidingLayers = kInklingWindowedLayers,
                          .unrotatedLayers = (uint64_t{1} << kInklingLoop.layers) - 1,
                          .fullAttentionPeriod = 1,
                          .stopTokens = {200006},
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .experts = 128,
                          .expertsPerToken = 8,
                          .expertIntermediateSize = 1152,
                          .moeRouter = ops::MoeRouter::SigmoidBiased,
                          .sharedExpertSlices = 0,
                          .sharedExpertGate = false,
                          .routeScale = expertWeightsScale,
                          .moeActivation = ops::MoeActivation::Situ,
                          .loopStart = kInklingLoop.loopStart,
                          .loopEnd = kInklingLoop.loopEnd,
                          .loopPasses = kInklingLoop.loopPasses,
                          .relativeDimension = 16,
                          .relativeExtent = 1024,
                          .shortConvolutionTaps = 4},
                         "Inkling-2-Mini"} {}
  bool operator==(const InklingLayout &) const = default;
};

// Whether weight layer `weightLayer` attends every key, as each layer that
// runs it does: it then takes separate query, key and value inputs, and its
// keys and values take dynamic convolution taps.
[[nodiscard]] constexpr bool inklingGlobalLayer(const QwenTargetDimensions &g, uint32_t weightLayer) noexcept {
  const uint32_t layer = g.firstLayer(weightLayer);
  return layer >= 64 || !(g.slidingLayers >> layer & 1);
}

// The rows of a layer's three input projections ("sides"), each written as
// one fp32 row per token row, in this order:
//  - query side: the queries [0, gate), head-major (head * 128 + channel);
//    the output gate before its sigmoid [gate, relative); the relative values
//    [relative, sink), head * 16 + value, before their clip; the sink logit
//    of each query head [sink, sink + heads); then zero rows to `query`, a
//    whole 64-row tile.
//  - key side: the keys before their convolution [0, lags), KV-head-major;
//    on a global layer the dynamic convolution taps of each KV head
//    [lags, lags + kvHeads * taps), head * taps + lag, lag 0 the current
//    token's; then zero rows to a whole tile (`globalKey`), or none on a
//    sliding layer (`key`).
//  - value side: the key side's rows for the values.
struct InklingSides final {
  uint32_t gate = 0, relative = 0, sink = 0, query = 0;
  uint32_t lags = 0, key = 0, globalKey = 0;

  [[nodiscard]] static constexpr InklingSides of(const QwenTargetDimensions &g) noexcept {
    InklingSides sides;
    sides.gate = g.attentionWidth;
    sides.relative = 2 * g.attentionWidth;
    sides.sink = sides.relative + g.attentionQueryHeads * g.relativeDimension;
    sides.query = tiled(sides.sink + g.attentionQueryHeads);
    sides.lags = g.attentionKvHeads * g.attentionHeadDimension;
    sides.key = tiled(sides.lags);
    sides.globalKey = tiled(sides.lags + g.attentionKvHeads * g.shortConvolutionTaps);
    return sides;
  }
  // The rows of a weight layer's key or value side.
  [[nodiscard]] constexpr uint32_t keyRows(bool global) const noexcept { return global ? globalKey : key; }

private:
  [[nodiscard]] static constexpr uint32_t tiled(uint32_t rows) noexcept {
    return (rows + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
  }
};
static_assert(InklingSides::of(InklingLayout{}).query == 6592 && InklingSides::of(InklingLayout{}).sink == 6528 &&
                  InklingSides::of(InklingLayout{}).key == 1024 && InklingSides::of(InklingLayout{}).globalKey == 1088,
              "Inkling's sides are the rows its ops read");

// A weight layer's attention, which every layer that runs it reads. F32
// tensors are their values as the GGUF stores them, row-major:
//  - residualQueries: the depth-attention queries of the layer's inputs,
//    [hidden], or on a global layer [3][hidden] of its query, key and value
//    inputs; none on weight layer 0, whose input is the embedding.
//  - inputNorms: the RMS norms of its inputs, one on a sliding layer
//    (attn_norm), the query, key and value inputs' on a global one.
//  - querySide, keySide, valueSide: its projections of InklingSides' rows
//    from hidden inputs, which write fp32 rows; segments of their GGUF
//    tensors' formats, a segment for each run of tensors of one type
//    (columnOffset at each run's first row).
//  - queryClip [heads][128] and relativeClip [heads][16]: the scales of each
//    query's and relative value's channels before their L2 clip.
//  - relativeBias [relativeExtent][16]: the bias bank, row d the profile of
//    key distance d (a query head's bias at distance d is its relative
//    values' dot product with it, 0 from relativeExtent on).
//  - keyConvolution, valueConvolution [kvHeads * 128][4] and
//    outputConvolution [hidden][4]: each channel's convolution taps, tap 3
//    the current token's input, tap 0 the input three tokens before.
//  - outputProjection [hidden, attentionWidth] of the gated attention.
struct InklingAttentionWeights final {
  metal::MetalBuffer residualQueries;
  std::array<ops::NormWeights, 3> inputNorms;
  ops::Projection querySide;
  ops::Projection keySide;
  ops::Projection valueSide;
  metal::MetalBuffer queryClip;
  metal::MetalBuffer relativeClip;
  metal::MetalBuffer relativeBias;
  metal::MetalBuffer keyConvolution;
  metal::MetalBuffer valueConvolution;
  ops::Projection outputProjection;
  metal::MetalBuffer outputConvolution;
};

// One loop gate of a core weight layer, of one input stream: down
// [loopGateRank, 2 * hidden] of the concatenated RMS of the stream and of its
// message, and gain and forget [hidden, loopGateRank], float projections of
// the GGUF's F32 or BF16 values, which write fp32 rows.
struct InklingLoopGateWeights final {
  ops::Projection down;
  ops::Projection gain;
  ops::Projection forget;
};

// What a core weight layer adds for the loop's passes after its first: the
// depth-attention queries of its four messages [loopMessages][hidden] F32,
// and the gates of its query, key, value and FFN inputs, in this order.
struct InklingLoopWeights final {
  metal::MetalBuffer messageQueries;
  std::array<InklingLoopGateWeights, InklingLayout::loopMessages> gates;
};

// A weight layer: its attention (mixer), the depth-attention query of its
// FFN's input [hidden] F32 and the norm after it, its MoE block (ffn: no
// shared expert; the router's F32 rows and its selection bias [experts] F32,
// BlockMoeWeights::routerBias), its FFN output's convolution [hidden][4] F32,
// and on a core layer what the loop adds.
struct InklingLayerWeights final {
  InklingAttentionWeights mixer;
  metal::MetalBuffer ffnResidualQuery;
  ops::NormWeights ffnNorm;
  ops::BlockMoeWeights ffn;
  metal::MetalBuffer ffnConvolution;
  std::optional<InklingLoopWeights> loop;
};

// Its weights: a weight layer's for each of its weight layers (24), which
// the layers that run it share, the norm of each token's embedding row, the
// head's depth-attention query [hidden] F32 and its final norm, and the
// logits projection of every row of the vocabulary (201,024).
struct InklingWeights final : QwenTargetWeightsBase {
  InklingLayout layout;
  std::vector<InklingLayerWeights> layers;
  ops::NormWeights embeddingNorm;
  metal::MetalBuffer outputQuery;
  ops::NormWeights finalNorm;
};

[[nodiscard]] InklingWeights loadInklingWeights(metal::MetalBackend &backend, InklingLayout layout,
                                                const QwenTargetFiles &files);

// What the shared geometry takes of its weights (qwenTargetGeometry): the
// three sides and output projection of each weight layer's attention, and
// its loop gates.
void includeMixer(QwenTargetGeometry &geometry, const InklingAttentionWeights &attention);
void includeFamilyProjections(QwenTargetGeometry &geometry, const InklingWeights &weights);

// Its target (model/InklingTarget.hpp), which the runtime makes of its
// weights, and the step scratch it keeps itself (model/InklingSteps.hpp):
// its residual's states, its layers' work and a decode step's convolution
// tape; it streams no table.
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const InklingWeights &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);
[[nodiscard]] QwenTargetScratchBytes qwenTargetScratchBytes(const InklingLayout &layout,
                                                            const QwenTargetGeometry &geometry,
                                                            uint32_t prefillChunks, uint64_t tableRowBytes);

} // namespace splash::model
