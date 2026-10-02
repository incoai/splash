#pragma once

#include "Model.hpp"
#include "QwenHybridLayout.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace splash::model {

struct Qwen3_8Layout;
struct Qwen3_8LayerWeights;
struct Qwen3_6MoeLayout;
struct Qwen3_6MoeLayerWeights;
struct Qwen4ExpWeights;
struct Qwen4ExpLayerWeights;
struct Qwen4HyperWeights;

// Both supported targets bind the same mixer tensors per hybrid layer; only
// the FFN differs between them.
struct QwenGdnWeights final {
  ops::Projection inputProjection;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer decay;
  metal::MetalBuffer timeBias;
  ops::NormWeights mixerNorm;
  ops::Projection outputProjection;
  // The value-head order of outputProjection's input columns, in which the
  // GDN writes its output.
  ops::GdnHeadOrder outputHeadOrder = ops::GdnHeadOrder::Grouped;
};

struct QwenAttentionWeights final {
  ops::Projection inputProjection;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
  ops::Projection outputProjection;
};

using QwenMixerWeights = std::variant<QwenGdnWeights, QwenAttentionWeights>;

// A Qwen target's weights outside its layers and the record of every file
// its weights were read from.
struct QwenTargetWeightsBase {
  ops::NormWeights finalNorm;
  ops::Projection logitsProjection;
  ops::EmbeddingWeights tokenEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

// The weights of a target of Layout, whose layers the family keeps in Layer.
template <class Layout, class Layer> struct QwenTargetWeights final : QwenTargetWeightsBase {
  Layout layout;
  std::vector<Layer> layers;
};

// Runtime-visible tensor geometry shared by the supported Qwen hybrid
// targets. It describes semantics only; operators remain responsible for
// choosing device-specific Metal pipelines and compute tiles.
struct QwenTargetGeometry final {
  static constexpr uint32_t maximumCaptureLayers = 8;

  uint32_t maximumContextTokens = 0;
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionQueryHeads = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  uint32_t denseIntermediateSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  QwenFfnKind ffnKind = QwenFfnKind::Dense;
  // The weight layout every sparse MoE block of the target shares, and in a
  // GGUF the format of most of its routed expert weights.
  ops::WeightLayout moeLayout = ops::WeightLayout::Affine64;
  uint32_t moeExpertFormat = GGUF_FMT_COUNT;
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens{};
  std::array<uint32_t, maximumCaptureLayers> captureLayerValues{};
  uint32_t captureLayerCount = 0;
  kv::Layout kvLayout{};
  GdnStateLayout stateLayout{};
  // Qwen3.8-Flash-Next (model/Qwen4Exp.hpp) when hyperConnections is set:
  // the residual's streams and the mixes' rank, and the PLE layer (layers
  // for none), the width of its gathered rows (pleHeads rows of
  // pleHeadDimension) and its convolution.
  uint32_t hyperConnections = 0;
  uint32_t hyperRank = 0;
  uint32_t pleLayer = 0;
  uint32_t pleHeads = 0;
  uint32_t pleHeadDimension = 0;
  uint32_t pleTaps = 0;
  uint32_t pleDilation = 0;
  // The MTP head's attention layers, the last of kvLayout's.
  uint32_t mtpLayers = 0;
  // Distinct operator requirements, collected from the loaded weights.
  std::vector<ops::ProjectionShape> prefillProjections;
  std::vector<ops::ProjectionShape> decodeProjections;
  std::vector<ops::ProjectionShape> gateUpProjections;

  [[nodiscard]] constexpr bool qwen4() const noexcept { return hyperConnections != 0; }
  [[nodiscard]] constexpr uint32_t streamWidth() const noexcept { return hyperConnections * hiddenSize; }
  [[nodiscard]] constexpr uint32_t pleWidth() const noexcept { return pleHeads * pleHeadDimension; }
  [[nodiscard]] constexpr uint32_t pleHistory() const noexcept { return pleTaps ? (pleTaps - 1) * pleDilation : 0; }
  [[nodiscard]] constexpr uint32_t gdnKeyWidth() const noexcept {
    return gdnKeyHeads * gdnHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * captureLayerCount;
  }
  [[nodiscard]] constexpr ops::MoeShape moeShape() const noexcept {
    return {hiddenSize, experts, expertsPerToken, expertIntermediateSize, moeLayout, moeExpertFormat};
  }
  [[nodiscard]] constexpr uint32_t ffnScratchWidth() const noexcept {
    return ffnKind == QwenFfnKind::Dense ? denseIntermediateSize
                                         : expertIntermediateSize;
  }
  [[nodiscard]] constexpr std::span<const uint32_t>
  captureLayers() const noexcept {
    return {captureLayerValues.data(), captureLayerCount};
  }
  // The capture slot of `layer`, whose output the draft reads.
  [[nodiscard]] constexpr std::optional<uint32_t> captureSlot(uint32_t layer) const noexcept {
    const auto layers = captureLayers();
    const auto found = std::find(layers.begin(), layers.end(), layer);
    if (found == layers.end()) return std::nullopt;
    return static_cast<uint32_t>(found - layers.begin());
  }
  [[nodiscard]] constexpr ops::GdnShape gdnShape() const noexcept {
    return {gdnKeyHeads, gdnValueHeads, gdnHeadDimension,
            convolutionDimension, packedGdnWidth, qwen4()};
  }
  // The projection lists hold every projection the weights dispatch, which
  // each have sizes.
  [[nodiscard]] bool valid() const noexcept {
    const auto sized = [](const std::vector<ops::ProjectionShape> &shapes) {
      return !shapes.empty() && std::all_of(shapes.begin(), shapes.end(), [](const auto &shape) {
        return shape.outputSize && shape.inputSize;
      });
    };
    return maximumContextTokens && layers && hiddenSize && vocabularySize &&
           packedGdnWidth && packedFullWidth && convolutionDimension &&
           gdnKeyHeads && gdnValueHeads && gdnHeadDimension &&
           attentionWidth && attentionQueryHeads && attentionKvHeads &&
           attentionHeadDimension && rotaryPairs && rotaryTheta > 0.0F &&
           captureLayerCount && captureLayerCount <= maximumCaptureLayers &&
           kvLayout.valid() && stateLayout.valid() &&
           stateLayout.layers + kvLayout.attentionLayers == layers + mtpLayers &&
           gdnShape().valid() &&
           // The GDN value rows are sized with attentionWidth throughout.
           gdnValueHeads * gdnHeadDimension == attentionWidth &&
           attentionWidth == attentionQueryHeads * attentionHeadDimension &&
           kvLayout.kvHeads == attentionKvHeads &&
           kvLayout.headDimension == attentionHeadDimension &&
           sized(prefillProjections) && sized(decodeProjections) &&
           ((ffnKind == QwenFfnKind::Dense && denseIntermediateSize && sized(gateUpProjections)) ||
            (ffnKind == QwenFfnKind::SparseMoe && moeShape().valid()));
  }
};

struct QwenTargetPrefillCapture final {
  uint32_t sourceStart = 0;
  uint32_t destinationStart = 0;
  uint32_t rows = 0;
};

struct QwenTargetPrefillSequence final {
  uint32_t rowBegin = 0;
  uint32_t rows = 0;
  uint32_t attentionStride = 0;
  uint64_t queryOffset = 0;
  uint64_t kvOffset = 0;
  kv::Q8ChunkedPrefillParams q8;
  metal::MetalBuffer pageTable;
  std::span<const metal::MetalBuffer> convolutionIn;
  std::span<const metal::MetalBuffer> convolutionOut;
  std::span<const metal::MetalBuffer> recurrentIn;
  std::span<const metal::MetalBuffer> recurrentOut;
  std::array<QwenTargetPrefillCapture, 2> captures{};
  uint32_t captureCount = 0;
  // Qwen3.8-Flash-Next: the PLE convolution's history before and after the
  // sequence's rows, and the QSA blocks its rows complete (Qwen4QsaBlock
  // records the runtime writes).
  metal::MetalBuffer pleHistoryIn;
  metal::MetalBuffer pleHistoryOut;
  metal::MetalBuffer qsaBlocks;
  uint32_t qsaBlockCount = 0;
};

// Qwen3.8-Flash-Next's scratch of a step of `rows` rows (prefill or verify):
// the fp32 residual streams, a mix's normalized streams, low-rank and gate
// rows and injection weights, the zero rows the MoE adds its output to, and
// the PLE n-gram rows (indices the runtime hashes on the CPU), projections,
// gated values and convolution inputs. Empty for the other families.
struct QwenHyperBuffers final {
  metal::MetalBuffer streams;
  metal::MetalBuffer normalized;
  metal::MetalBuffer low;
  metal::MetalBuffer gate;
  metal::MetalBuffer weights;
  metal::MetalBuffer branch;
  metal::MetalBuffer zeros;
  metal::MetalBuffer pleIndices;
  metal::MetalBuffer pleRows;
  metal::MetalBuffer pleValue;
  metal::MetalBuffer pleGated;
  metal::MetalBuffer pleConvolution;
  // QSA: the indexer's projections (fp32), its prepared queries, the score
  // scratch of qsaScoreRows rows, the step's bitmaps of qsaMaskWords words
  // per row, the RoPE inverse frequencies and each full-attention layer's
  // index store (raw and pooled keys by physical KV page).
  metal::MetalBuffer indexerQuery;
  metal::MetalBuffer indexerKey;
  metal::MetalBuffer qsaQueries;
  metal::MetalBuffer qsaScores;
  metal::MetalBuffer qsaMask;
  metal::MetalBuffer inverseFrequencies;
  std::span<const metal::MetalBuffer> qsaRawKeys;
  std::span<const metal::MetalBuffer> qsaPooledKeys;
  uint32_t qsaScoreRows = 0;
  uint32_t qsaMaskWords = 0;
  // The MTP head: its fp32 streams, its eh projection's bf16 inputs (rows of
  // streams * 2 * hidden) and a draft step's tokens.
  metal::MetalBuffer mtpStreams;
  metal::MetalBuffer mtpInput;
  metal::MetalBuffer mtpTokens;
};

struct QwenTargetPrefillBuffers final {
  // Split projections of chunks of up to 32 rows (LinearGguf.cpp).
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer captured;
  metal::MetalBuffer gdnPacked;
  metal::MetalBuffer gdnQueries;
  metal::MetalBuffer gdnKeys;
  metal::MetalBuffer gdnValues;
  metal::MetalBuffer gdnDecay;
  metal::MetalBuffer gdnBeta;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseGateScratch;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer downProjectionSums;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  ops::MoeScratch moe;
  QwenHyperBuffers hyper;
};

struct QwenTargetVerifyBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer arrived;
  metal::MetalBuffer generation;
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer denseGateScratch;
  std::span<const metal::MetalBuffer> gdnPacked;
  std::span<const metal::MetalBuffer> gdnMixed;
  std::span<const metal::MetalBuffer> gdnDecay;
  std::span<const metal::MetalBuffer> gdnBeta;
  std::span<const metal::MetalBuffer> chunkKeys;
  std::span<const metal::MetalBuffer> chunkValues;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      pageTables;
  ops::MoeScratch moe;
  QwenHyperBuffers hyper;
  // Qwen3.8-Flash-Next: each lane's PLE history before the step (the
  // commit writes the one after it), and the QSA blocks its rows complete.
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> pleHistories;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> qsaBlocks;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> qsaBlockCounts{};
  // The MTP head's drafts (addMtpDraft): each lane's rows' tokens (row 0
  // its anchor), and the proposals and sparse candidates it writes.
  metal::MetalBuffer anchors;
  metal::MetalBuffer proposals;
  metal::MetalBuffer candidates;
  uint32_t proposalsPerLane = 0;
  uint32_t candidatesPerProposal = 0;
};

struct QwenTargetCommitBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextStates;
  metal::MetalBuffer retainedCounts;
  // Qwen3.8-Flash-Next: the step's PLE convolution inputs and each lane's
  // history before and after it.
  metal::MetalBuffer pleConvolution;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> pleHistoriesIn;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> pleHistoriesOut;
  // With the MTP head: the step's residual streams, whose row before each
  // lane's next position it carries.
  metal::MetalBuffer streams;
};

template <class Layout, class Layer>
[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const QwenTargetWeights<Layout, Layer> &weights);
[[nodiscard]] QwenTargetGeometry qwenTargetGeometry(const Qwen4ExpWeights &weights);

// Builds the shared Qwen GDN/attention layer graph with the target's dense
// or sparse-MoE FFN. Architecture-specific loaders supply the package tensors.
class QwenTarget final {
public:
  template <class Layout, class Layer>
  QwenTarget(const QwenTargetWeights<Layout, Layer> &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators,
             kv::Format format = kv::Format::Int8);
  QwenTarget(const Qwen4ExpWeights &weights, metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
             kv::Format format = kv::Format::Int8);
  // Qwen3.8-Flash-Next's weights, or null for another family.
  [[nodiscard]] const Qwen4ExpWeights *qwen4() const noexcept;

  [[nodiscard]] const QwenTargetGeometry &geometry() const noexcept {
    return geometry_;
  }
  [[nodiscard]] const ops::Projection &
  vocabularyProjection() const noexcept;
  // Lanes of storage the tensors of a decode step of `lanes` lanes bind: a
  // linear tile may hold more rows than the step (LinearPlan::storageRows;
  // a three-lane GGUF step on the staged tile runs its 32-row tile over four
  // lanes). Every op still processes the step's lanes; padding rows read
  // stale activations and write results no active row reads.
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;

  // Returns the hidden buffer that holds the last layer's output rows.
  [[nodiscard]] metal::MetalBuffer addPrefill(
      metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const kv::LayerStorage> kvLayers) const;
  void addVerify(
      metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
      std::span<const kv::LayerStorage> kvLayers,
      std::span<const kv::Q8ChunkedPrefillParams> q8,
      std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
      ops::LinearDispatchStats &stats) const;
  void addHead(metal::CommandGraph &graph, metal::MetalBuffer hidden,
               metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
               uint32_t normalizedRows, ops::LinearScratch scratch) const;
  void addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    metal::MetalBuffer hidden, uint32_t rows) const;
  void addStateCommit(metal::CommandGraph &graph,
                      QwenTargetCommitBuffers buffers, uint32_t lanes) const;

  // Qwen3.8-Flash-Next with its MTP head (geometry().mtpLayers).
  [[nodiscard]] bool hasMtp() const noexcept { return geometry_.mtpLayers != 0; }
  // `steps` draft steps over each lane's verify rows before the verify: step
  // j writes proposal j - 1, the last one also every later proposal. The
  // verify then writes the head's KV at its rows (addVerify).
  void addMtpDraft(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
                   std::span<const kv::LayerStorage> kvLayers, std::span<const kv::Q8ChunkedPrefillParams> q8,
                   std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes, uint32_t steps,
                   ops::LinearDispatchStats &stats) const;
  // The PLE rows of each lane's verify rows from their tokens and the two
  // tokens before each lane's rows (QWEN4_PLE_NONE for none).
  void addPleHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer before,
                  metal::MetalBuffer indices, uint32_t lanes) const;

private:
  using WeightView =
      std::variant<const QwenTargetWeights<Qwen3_8Layout, Qwen3_8LayerWeights> *,
                   const QwenTargetWeights<Qwen3_6MoeLayout, Qwen3_6MoeLayerWeights> *,
                   const Qwen4ExpWeights *>;
  struct PrefillStep;
  struct VerifyStep;

  // A layer's parts in dispatch order: the mixer normalizes its input and
  // returns the residual rows the FFN normalizes and adds to into `output`.
  void addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm,
                      ops::WeightLayout consumer) const;
  void addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                        metal::MetalBuffer input, metal::MetalBuffer output) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer, const ops::NormWeights &norm,
                                     metal::MetalBuffer input) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer,
                                     const ops::NormWeights &norm, metal::MetalBuffer input) const;
  void addPrefillFfn(PrefillStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                     metal::MetalBuffer output) const;
  void addPrefillFfn(PrefillStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                     metal::MetalBuffer output) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer, const ops::NormWeights &norm,
                                    metal::MetalBuffer input) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                    const ops::NormWeights &norm, metal::MetalBuffer input) const;
  void addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const;
  void addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const;

  // Qwen3.8-Flash-Next's graphs (QwenTarget.cpp addQwen4*).
  [[nodiscard]] metal::MetalBuffer addQwen4Prefill(PrefillStep &step, const Qwen4ExpWeights &weights) const;
  void addQwen4Verify(VerifyStep &step, const Qwen4ExpWeights &weights) const;
  void addQwen4VerifyMix(VerifyStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer streams,
                         metal::MetalBuffer output) const;
  // A full-attention block over the step's rows into hyper.branch.
  void addQwen4VerifyAttention(VerifyStep &step, const Qwen4ExpLayerWeights &layer,
                               const QwenAttentionWeights &mixer, uint32_t attentionLayer) const;
  // The MTP head over each lane's rows: its inputs (`embedding` rows paired
  // with the carried streams and `history`), its attention block (writing
  // its KV) and, when `full`, its FFN, its head mix and logits.
  void addQwen4MtpVerify(VerifyStep &step, const Qwen4ExpWeights &weights, metal::MetalBuffer embedding,
                         metal::MetalBuffer history, uint32_t limit, bool full) const;
  // A state cell's carried MTP streams within its auxiliary state.
  [[nodiscard]] metal::MetalBuffer mtpCarried(const metal::MetalBuffer &auxiliary) const;

  WeightView weights_;
  const QwenTargetWeightsBase &weightsBase_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
};

} // namespace splash::model
