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
#include "ops/Qwen4.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace splash::ops {
class AneFfn;
} // namespace splash::ops

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
// targets: the target's dimensions, the layers the draft reads and what its
// loaded weights add. It describes semantics only; operators remain
// responsible for choosing device-specific Metal pipelines and compute tiles.
struct QwenTargetGeometry final : QwenTargetDimensions {
  static constexpr uint32_t maximumCaptureLayers = 8;

  QwenTargetGeometry() = default;
  explicit QwenTargetGeometry(const QwenTargetDimensions &dimensions) : QwenTargetDimensions(dimensions) {}

  // The format of most of the sparse MoE blocks' routed expert weights.
  uint32_t moeExpertFormat = GGUF_FMT_COUNT;
  std::array<uint32_t, maximumCaptureLayers> captureLayerValues{};
  uint32_t captureLayerCount = 0;
  // The target's KV layout in the format the runtime stores KV in, and its
  // GDN state layout.
  kv::Layout kvLayout{};
  GdnStateLayout stateLayout{};
  // Qwen3.8-Flash-Next's MTP head's attention layers, the last of
  // kvLayout's.
  uint32_t mtpLayers = 0;
  // Distinct operator requirements, collected from the loaded weights.
  std::vector<ops::ProjectionShape> prefillProjections;
  std::vector<ops::ProjectionShape> decodeProjections;
  std::vector<ops::ProjectionShape> gateUpProjections;

  [[nodiscard]] constexpr uint32_t gdnKeyWidth() const noexcept {
    return gdnKeyHeads * gdnHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * captureLayerCount;
  }
  [[nodiscard]] constexpr ops::MoeShape moeShape() const noexcept {
    return {hiddenSize, experts, expertsPerToken, expertIntermediateSize, moeExpertFormat};
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
  // The layout itself was checked by requireQwenLayout when the target
  // loaded. The projection lists hold every projection the weights dispatch,
  // which each have sizes.
  [[nodiscard]] bool valid() const noexcept {
    const auto sized = [](const std::vector<ops::ProjectionShape> &shapes) {
      return !shapes.empty() && std::all_of(shapes.begin(), shapes.end(), [](const auto &shape) {
        return shape.outputSize && shape.inputSize;
      });
    };
    return captureLayerCount && captureLayerCount <= maximumCaptureLayers &&
           stateLayout.layers + kvLayout.attentionLayers == layers + mtpLayers &&
           gdnShape().valid() &&
           kvLayout.kvHeads == attentionKvHeads &&
           kvLayout.headDimension == attentionHeadDimension &&
           sized(prefillProjections) && sized(decodeProjections) &&
           ((ffnKind == QwenFfnKind::Dense && intermediateSize && sized(gateUpProjections)) ||
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
  kv::ChunkedPrefillParams chunk;
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
  // per row, the RoPE inverse frequencies and where each full-attention
  // layer's index tensors sit in the KV extents.
  metal::MetalBuffer indexerQuery;
  metal::MetalBuffer indexerKey;
  metal::MetalBuffer qsaQueries;
  metal::MetalBuffer qsaScores;
  metal::MetalBuffer qsaMask;
  metal::MetalBuffer inverseFrequencies;
  std::span<const ops::QsaIndex> qsaIndex;
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
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  ops::MoeScratch moe;
  QwenHyperBuffers hyper;

  // The dense FFN's buffers among these.
  [[nodiscard]] ops::PrefillFfnBuffers ffn() const {
    return {normalized, denseGateScratch, denseIntermediate, linearScratch};
  }
};

struct QwenTargetVerifyBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
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
// or sparse-MoE FFN. Architecture-specific loaders supply the model's tensors.
class QwenTarget final {
public:
  template <class Layout, class Layer>
  QwenTarget(const QwenTargetWeights<Layout, Layer> &weights, const QwenTargetGeometry &geometry,
             metal::MetalBackend &backend, const ops::ExecutionPlans &operators);
  QwenTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators);
  // Qwen3.8-Flash-Next's weights, or null for another family.
  [[nodiscard]] const Qwen4ExpWeights *qwen4() const noexcept;

  [[nodiscard]] const ops::Projection &
  vocabularyProjection() const noexcept;
  // Lanes of storage the tensors of a decode step of `lanes` lanes bind: a
  // linear tile may hold more rows than the step (LinearPlan::storageRows;
  // a three-lane GGUF step on the staged tile runs its 32-row tile over four
  // lanes). Every op still processes the step's lanes; padding rows read
  // stale activations and write results no active row reads.
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;

  // Returns the hidden buffer that holds the last layer's output rows. The
  // dense FFN of a chunk the split takes (AneFfn::splits) runs split with the
  // Neural Engine on `aneFfn`, when given.
  [[nodiscard]] metal::MetalBuffer addPrefill(
      metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const SplashKvLayer> kvLayers, ops::AneFfn *aneFfn = nullptr) const;
  void addVerify(
      metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
      std::span<const SplashKvLayer> kvLayers,
      std::span<const kv::ChunkedPrefillParams> chunks,
      uint32_t lanes) const;
  // The final norm and LM head over `lanes` lanes of targetVerifyRows rows,
  // as verify ends: one sweep of the vocabulary projection for every lane.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                    metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
                    uint32_t lanes, ops::LinearScratch scratch) const;
  // The verify input tokens addEmbedding then gathers: each lane's anchor,
  // row 0 of its draft input, and the draft's proposals.
  void addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput,
                      metal::MetalBuffer proposals,
                      metal::MetalBuffer verifyInput, uint32_t lanes) const;
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
                   std::span<const SplashKvLayer> kvLayers, std::span<const kv::ChunkedPrefillParams> chunks,
                   uint32_t lanes, uint32_t steps) const;
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
  void addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm) const;
  void addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                        metal::MetalBuffer input, metal::MetalBuffer output) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer, const ops::NormWeights &norm,
                                     metal::MetalBuffer input) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer,
                                     const ops::NormWeights &norm, metal::MetalBuffer input) const;
  // The FFN of layer `index`.
  void addPrefillFfn(PrefillStep &step, uint32_t index, const Qwen3_8LayerWeights &layer,
                     metal::MetalBuffer residual, metal::MetalBuffer output) const;
  void addPrefillFfn(PrefillStep &step, uint32_t index, const Qwen3_6MoeLayerWeights &layer,
                     metal::MetalBuffer residual, metal::MetalBuffer output) const;
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
