#pragma once

// DeepSeek-V4.1-Flash (GGUF architecture deepseek41, model type
// deepseek_v41_text; ISTA-DASLab's GSQ-RCO GGUFs): 40 layers, each a latent
// attention and a sparse MoE on a manifold hyper-connected residual, as
// DeepSeek's model.py computes them (dev/tools/deepseek41_reference.py; its
// dimensions in QwenTargetDimensions::deepseek, DeepSeek41Dimensions).
//  - Attention of 64 query heads over one latent per token of 512 values,
//    its key and its value: queries from a low-rank latent of 1280 values,
//    the latent from the row's input, both RMS normalized and rotated by
//    RoPE on their last 32 pairs; a learned sink per head; the heads'
//    outputs rotated back and projected in 8 low-rank groups of 1024 values,
//    then to the hidden width.
//  - Every layer reads the 128 latents that end at its row (its window).
//    Layers 2 to 39 also read compressed latents of groups of earlier tokens,
//    of 2 tokens on layers 2 to 19 and of 1 on the others, which the
//    compressors of layers 2, 8, 14 and 20 make for the layers up to the
//    next; the indexers of layers 2, 8, 14, 20, 24, 28, 32 and 36 select the
//    512 groups a row of them reads, from layer 20 on only among the 2048
//    blocks of 8 groups that layer's scores rank highest.
//  - Four residual streams (ops::ManifoldHyperConnection): each sublayer's
//    mix weighs all of them, and the sublayer's output enters each stream by
//    that mix's post weights and comb matrix. A sublayer's input is the
//    streams summed by the pre weights of the mix before it (stream 0 alone
//    for the first), as the GGUF's conversion shifts them; the head's by the
//    last mix's.
//  - A MoE of 384 routed experts, 6 a token, weighted by sqrt(softplus) over
//    their sum times 1.5 after a selection that adds a bias per expert,
//    beside one ungated shared expert; every SwiGLU clamps at 10.
//  - Engram on layers 1 and 14: each token's n-gram hashes gather rows of a
//    table of their own in a GGUF of its own, which gate a value into each
//    stream (ops::Engram).
// Its weights load from a GGUF only. Norms take its epsilon of 1e-20.

#include "EngramHash.hpp"
#include "GgufFile.hpp"
#include "QwenHybridLayout.hpp"
#include "SparseRowStore.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/LatentAttention.h"
#include "metal/abi/LatentIndex.h"
#include "metal/abi/LatentRotation.h"
#include "metal/abi/ManifoldHyperConnection.h"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/Weights.hpp"

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace splash::model {

// A bit for each of `layers` (QwenTargetDimensions' layer masks).
consteval uint64_t layerBits(std::initializer_list<uint32_t> layers) {
  uint64_t bits = 0;
  for (const uint32_t layer : layers) bits |= uint64_t{1} << layer;
  return bits;
}
// A bit for each layer from `first` up to `end`.
consteval uint64_t layerRange(uint32_t first, uint32_t end) {
  uint64_t bits = 0;
  for (uint32_t layer = first; layer < end; ++layer) bits |= uint64_t{1} << layer;
  return bits;
}

struct DeepSeek41Layout final : QwenHybridLayout {
  // Its config's text model type and its GGUFs' architecture; no safetensors
  // checkpoint of it loads.
  static constexpr std::string_view modelType = "deepseek_v41_text";
  static constexpr std::string_view ggufArchitecture = "deepseek41";
  static constexpr bool loadsSafetensors = false;

  // DeepSeek-V4.1-Flash's, its first family (kFamilies). Its native window is
  // YaRN's already (DeepSeek41Dimensions), which servedFor does not stretch.
  constexpr DeepSeek41Layout()
      : QwenHybridLayout{{.nativeContextTokens = 1'048'576,
                          .layers = 40,
                          .hiddenSize = 5120,
                          .vocabularySize = 129280,
                          .attentionWidth = 64 * 512,
                          .attentionQueryHeads = 64,
                          .attentionKvHeads = 1,
                          .attentionHeadDimension = 512,
                          .attentionGate = false,
                          .rotaryPairs = 32,
                          .rotaryTheta = 10'000.0F,
                          .slidingWindow = 128,
                          .slidingLayers = layerRange(0, 40),
                          .fullAttentionPeriod = 1,
                          .stopTokens = {1},
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .experts = 384,
                          .expertsPerToken = 6,
                          .expertIntermediateSize = 2304,
                          .moeRouter = ops::MoeRouter::SqrtSoftplusNormalized,
                          .sharedExpertSlices = 1,
                          .sharedExpertGate = false,
                          .deepseek = {.queryRank = 1280,
                                       .outputRank = 1024,
                                       .outputGroups = 8,
                                       .compressedLayers = layerRange(2, 40),
                                       .pairedLayers = layerRange(2, 20),
                                       .compressorLayers = layerBits({2, 8, 14, 20}),
                                       .indexerLayers = layerBits({2, 8, 14, 20, 24, 28, 32, 36}),
                                       .indexerHeads = 32,
                                       .indexerHeadDimension = 128,
                                       .indexerTopGroups = 512,
                                       .candidateLayer = 20,
                                       .candidateBlockGroups = 8,
                                       .candidateTopBlocks = 2048,
                                       .engramLayers = layerBits({1, 14}),
                                       .engramHeads = 8,
                                       .engramHeadDimension = 256,
                                       .engramNgram = 4,
                                       .engramPadToken = 2,
                                       .sinkhornIterations = 20,
                                       .hcEpsilon = 1e-6F,
                                       .normEpsilon = 1e-20F,
                                       .swigluLimit = 10.0F,
                                       .routeScale = 1.5F,
                                       .compressRopeTheta = 160'000.0F,
                                       .yarnFactor = 16.0F,
                                       .yarnOriginalContextTokens = 65'536,
                                       .yarnBetaFast = 32.0F,
                                       .yarnBetaSlow = 1.0F}},
                         "DeepSeek-V4.1-Flash"} {}
  // A family of the architecture.
  constexpr explicit DeepSeek41Layout(const QwenHybridLayout &family) : QwenHybridLayout(family) {}
  bool operator==(const DeepSeek41Layout &) const = default;
};

// The family's kernels are compiled for its shape: the latent width, the
// residual's streams, the indexer's head width and heads, and the widest
// vector a rotation normalizes alone (the query latent).
static_assert(DeepSeek41Layout{}.attentionHeadDimension == LATENT_ATTENTION_DIM &&
                  DeepSeek41Layout{}.attentionQueryHeads <= LATENT_ATTENTION_MAX_HEADS,
              "the latent attention kernels are compiled for DeepSeek-V4.1's latents");
static_assert(DeepSeek41Layout{}.deepseek.indexerHeadDimension == LATENT_INDEX_DIM &&
                  DeepSeek41Layout{}.deepseek.indexerHeads <= LATENT_INDEX_MAX_HEADS,
              "the latent index kernels are compiled for DeepSeek-V4.1's indexer");
static_assert(DeepSeek41Layout{}.deepseek.queryRank <= LATENT_ROTATION_MAX_WIDTH &&
                  DeepSeek41Layout{}.hiddenSize <= MHC_MAX_HIDDEN,
              "the rotation and hyper-connection kernels hold DeepSeek-V4.1's widths");

// A sublayer's manifold hyper-connection mix (ops::ManifoldHyperConnection::
// addMix), all F32: its projection fn [MHC_MIXES][MHC_STREAMS * hidden], its
// scale (pre, post, comb) and base [MHC_MIXES].
struct DeepSeek41MixWeights final {
  metal::MetalBuffer fn;
  metal::MetalBuffer scale;
  metal::MetalBuffer base;
};

// A compressor layer's projections of its inputs into a group's latent, BF16
// as the GGUF stores them, which the float projection runs in fp32: the
// latents [head dimension, hidden], on a layer of groups of 2 tokens the
// gate whose softmax over a group weighs them, the latent's norm; then the
// indexer's key [indexer head dimension, head dimension] of a group's
// latent and its norm.
struct DeepSeek41CompressorWeights final {
  ops::Projection kv;
  std::optional<ops::Projection> gate;
  ops::NormWeights norm;
  ops::Projection indexKey;
  ops::NormWeights indexKeyNorm;
};

// An indexer layer's queries [heads * head dimension, query rank] of the
// query latent and the weights [heads, hidden] of its heads' scores, BF16.
struct DeepSeek41IndexerWeights final {
  ops::Projection query;
  ops::Projection weights;
};

// An Engram layer's projection [(MHC_STREAMS + 1) * hidden, gathered width] of
// its gathered rows into a key per stream and a value, and its gate's
// weights, engram_gate F32 [MHC_STREAMS][hidden] (the product of DeepSeek's
// gate query and key), which ops::Engram::addGate reads.
struct DeepSeek41EngramWeights final {
  ops::Projection kv;
  metal::MetalBuffer gate;
};

// A layer, in its image's order (GgufImage.cpp DeepSeek41Gguf::layer).
struct DeepSeek41LayerWeights final {
  std::optional<DeepSeek41EngramWeights> engram;
  DeepSeek41MixWeights attentionMix;
  ops::NormWeights attentionNorm;
  ops::Projection queryA;
  ops::NormWeights queryNorm;
  ops::Projection queryB;
  ops::Projection kv;
  ops::NormWeights kvNorm;
  metal::MetalBuffer sinks;
  std::optional<DeepSeek41CompressorWeights> compressor;
  std::optional<DeepSeek41IndexerWeights> indexer;
  // The output's groups [output rank, group width] each, then their
  // projection [hidden, groups * output rank].
  std::vector<ops::Projection> outputA;
  ops::Projection outputB;
  DeepSeek41MixWeights ffnMix;
  ops::NormWeights ffnNorm;
  ops::BlockMoeWeights ffn;
};

struct DeepSeek41Weights final : QwenTargetWeightsBase {
  DeepSeek41Layout layout;
  std::vector<DeepSeek41LayerWeights> layers;
  // The norm the head reads the streams through (ops::ManifoldHyperConnection::addPreNorm).
  ops::NormWeights finalNorm;
  // The Engram hash's tables, which the host (engramHash) and a verify
  // step's hash read, from the GGUF's metadata (readDeepSeek41EngramHash):
  // each token's compressed id (I32 [vocabulary]); and per Engram layer its
  // multipliers (I64 [ngram]), primes and offsets (I64 [(ngram - 1) * heads]
  // each), its offsets from its first row in the tables.
  metal::MetalBuffer engramTokenMap;
  metal::MetalBuffer engramMultipliers;
  metal::MetalBuffer engramPrimes;
  metal::MetalBuffer engramOffsets;
  // With the Engram's tables streaming (streamedTable, their one row store),
  // where each Engram layer's rows start among the store's rows, and their
  // format (GGUF_FMT_*).
  std::vector<uint32_t> engramFirstRows;
  uint32_t engramFormat = 0;

  [[nodiscard]] EngramHashTables engramHash() const noexcept;
};

// The Engram's tables in the target's GGUF: per_layer_token_embd, the IQ4_NL
// rows of a head's values of every Engram layer's heads, back to back as one
// table in the split GGUF's last part (the file `file` of the target's),
// which one row store streams (DeepSeek41Weights::streamedTable).
struct DeepSeek41EngramTables final {
  SparseRowTable table;
  uint32_t file = 0;
  uint32_t format = 0;
};
// The Engram's tables the target GGUF `target` holds. Throws unless they are
// rows of IQ4_NL of the Engram's head width.
[[nodiscard]] DeepSeek41EngramTables deepSeek41EngramTables(const GgufFile &target, const DeepSeek41Layout &layout);
// The Engram hash the target GGUF's metadata states for the `tableRows` rows
// of its tables (deepseek41.engram.token_map and the ple.* arrays), in
// buffers of `backend`, each Engram layer's offsets from its first row
// (engramFirstRows), where its first head's lie. Throws unless every value
// is a whole number of its range and each head's rows lie inside the tables.
void readDeepSeek41EngramHash(metal::MetalBackend &backend, const GgufFile &target, uint64_t tableRows,
                              DeepSeek41Weights &weights);
// The bytes of the Engram hash's buffers (readDeepSeek41EngramHash).
[[nodiscard]] uint64_t deepSeek41EngramHashBytes(const DeepSeek41Layout &layout);

struct BlockTargetFormat;
struct MoeExperts;
class WeightFile;

// A layer of `layout` from its image (GgufImage.cpp DeepSeek41Gguf::layer):
// the target's, or a DSpark draft's (model/DSpark.hpp), whose layers are the
// family's of window attention alone.
[[nodiscard]] DeepSeek41LayerWeights readDeepSeek41Layer(WeightFile &file, const BlockTargetFormat &format,
                                                         const DeepSeek41Layout &layout, const MoeExperts &experts,
                                                         uint32_t index);
// Loads the target from its GGUF's images (GgufImage.cpp DeepSeek41Gguf), its
// routed experts through a store when they stream, and its Engram's tables
// through a row store of the loader's table options where they give it memory
// (GgufTargetLoader::tableOptions), from the file the target's metadata
// names (deepSeek41EngramPath). Throws unless the layout is one the family's
// kernels and blocks serve (requireDeepSeek41Layout).
[[nodiscard]] DeepSeek41Weights loadDeepSeek41Weights(metal::MetalBackend &backend, DeepSeek41Layout layout,
                                                      const QwenTargetFiles &files);
// The family's target (model/DeepSeek41Target.hpp), whose prefill commands
// run up to `prefillChunks` chunks; the step scratch it keeps itself, with
// the Engram's rows of `tableRowBytes` each; and the projections its layers
// and head dispatch through ops::Linear (its float projections, the
// compressors', indexers' and routers', run on the float tile).
[[nodiscard]] QwenTargetGeometry qwenTargetGeometry(const DeepSeek41Weights &weights,
                                                    std::span<const uint32_t> captureLayers);
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const DeepSeek41Weights &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);
[[nodiscard]] QwenTargetScratchBytes qwenTargetScratchBytes(const DeepSeek41Weights &weights,
                                                            const QwenTargetGeometry &geometry,
                                                            uint32_t prefillChunks, uint64_t tableRowBytes);
// The Engram rows a step stages and the read-ahead may have queued
// (SparseRowStoreOptions::stageRows and prefetchRows).
[[nodiscard]] SparseRowStoreOptions streamedTableRows(const DeepSeek41Layout &layout);

// Throws unless the layout's dimensions are set and agree: latents of the
// width the kernels are compiled for, one KV head, whole output groups, a
// MoE on every layer with one ungated shared expert, an Engram of 2-grams at
// least, and each layer that attends compressed groups at or after a
// compressor and an indexer layer, all of them among its layers.
void requireDeepSeek41Layout(const DeepSeek41Layout &layout);

// RoPE's inverse frequencies of the layout's pairs: of a compressed layer,
// base compress_rope_theta stretched by YaRN, else rope_theta (the
// reference's rope_angles).
[[nodiscard]] std::vector<float> deepSeek41RopeFrequencies(const DeepSeek41Layout &layout, bool compressed);

} // namespace splash::model
