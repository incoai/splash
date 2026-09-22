#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace splash::model {

struct Qwen3_8Weights;
struct Qwen3_6MoeWeights;

enum class QwenFfnKind : uint8_t { Dense, SparseMoe };

// Both supported targets bind the same mixer tensors per hybrid layer; only
// the FFN differs between them.
struct QwenGdnWeights final {
  ops::Q4Projection inputProjection;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer decay;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer mixerNorm;
  ops::Q4Projection outputProjection;
};

struct QwenAttentionWeights final {
  ops::Q4Projection inputProjection;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Q4Projection outputProjection;
};

using QwenMixerWeights = std::variant<QwenGdnWeights, QwenAttentionWeights>;

// Sizes of the mixer sections in a packed layer file.
struct QwenMixerGeometry final {
  uint32_t hiddenSize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedAttentionWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionHeadDimension = 0;
};

// Reads the mixer sections that follow a layer's input norm, in file order.
[[nodiscard]] QwenMixerWeights readQwenMixer(WeightFile &file,
                                             metal::MetalBackend &backend,
                                             const QwenMixerGeometry &geometry,
                                             bool fullAttention);

inline constexpr std::string_view kEmbeddingMagic = "MDFE0001";

// Reads a packed target directory: one file per hybrid layer (input norm,
// mixer, post-attention norm, then the architecture's FFN through readFfn),
// head.bin and embedding.bin. Weights is the architecture's weight struct.
template <class Weights, class Layout, class ReadFfn>
[[nodiscard]] Weights
loadQwenTargetWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      const Layout &layout, std::string_view headMagic,
                      ReadFfn readFfn) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Weights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  const uint64_t hiddenBytes = checkedWeightMultiply(
      layout.hiddenSize, kBFloat16Bytes, "Qwen norm bytes");
  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    const std::string filename =
        "layer-" + std::to_string(layerIndex) + ".bin";
    WeightFile file(backend, directory / filename, "target/" + filename,
                    Layout::layerMagic, layerIndex, fullAttention ? 1U : 0U);
    auto &layer = result.layers.emplace_back();
    layer.inputNorm = file.section(hiddenBytes, "input-norm");
    layer.mixer =
        readQwenMixer(file, backend, layout.mixerGeometry(), fullAttention);
    layer.postAttentionNorm =
        file.section(hiddenBytes, "post-attention-norm");
    readFfn(file, layer);
    file.finish();
    result.files.push_back(file.record());
  }

  {
    WeightFile file(backend, directory / "head.bin", "target/head.bin",
                    headMagic, layout.layers, 2);
    result.finalNorm = file.section(hiddenBytes, "final-norm");
    result.logitsProjection = readQ4Projection(
        file, backend, layout.vocabularySize, layout.hiddenSize, "logits");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file(backend, directory / "embedding.bin",
                    "target/embedding.bin", kEmbeddingMagic,
                    layout.vocabularySize, layout.hiddenSize);
    result.tokenEmbedding = readQ4ProjectionComponents(
        file, layout.vocabularySize, layout.hiddenSize, "embedding");
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

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
  uint32_t packedAttentionWidth = 0;
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
  ops::MoeShape moe{};
  QwenFfnKind ffnKind = QwenFfnKind::Dense;
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens{};
  std::array<uint32_t, maximumCaptureLayers> captureLayerValues{};
  uint32_t captureLayerCount = 0;
  kv::Layout kvLayout{};
  GdnStateLayout stateLayout{};

  [[nodiscard]] constexpr uint32_t gdnKeyWidth() const noexcept {
    return gdnKeyHeads * gdnHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * captureLayerCount;
  }
  [[nodiscard]] constexpr uint32_t ffnScratchWidth() const noexcept {
    return ffnKind == QwenFfnKind::Dense ? denseIntermediateSize
                                         : moe.expertIntermediateSize;
  }
  [[nodiscard]] constexpr std::span<const uint32_t>
  captureLayers() const noexcept {
    return {captureLayerValues.data(), captureLayerCount};
  }
  [[nodiscard]] constexpr ops::GdnShape gdnShape() const noexcept {
    return {gdnKeyHeads, gdnValueHeads, gdnHeadDimension,
            convolutionDimension, packedGdnWidth};
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return maximumContextTokens && layers && hiddenSize && vocabularySize &&
           packedGdnWidth && packedAttentionWidth && convolutionDimension &&
           gdnKeyHeads && gdnValueHeads && gdnHeadDimension &&
           attentionWidth && attentionQueryHeads && attentionKvHeads &&
           attentionHeadDimension && rotaryPairs && rotaryTheta > 0.0F &&
           captureLayerCount && captureLayerCount <= maximumCaptureLayers &&
           kvLayout.valid() && stateLayout.valid() &&
           stateLayout.layers + kvLayout.attentionLayers == layers &&
           gdnKeyWidth() * 2 + attentionWidth <= packedGdnWidth &&
           attentionWidth == attentionQueryHeads * attentionHeadDimension &&
           kvLayout.kvHeads == attentionKvHeads &&
           kvLayout.headDimension == attentionHeadDimension &&
           ((ffnKind == QwenFfnKind::Dense && denseIntermediateSize) ||
            (ffnKind == QwenFfnKind::SparseMoe && moe.valid()));
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
};

struct QwenTargetPrefillBuffers final {
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
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
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
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
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
};

[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen3_8Weights &weights);
[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen3_6MoeWeights &weights);

// Builds the shared Qwen GDN/attention layer graph with the target's dense
// or sparse-MoE FFN. Architecture-specific loaders supply the package tensors.
class QwenTarget final {
public:
  QwenTarget(const Qwen3_8Weights &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators,
             kv::Format format = kv::Format::Int8);
  QwenTarget(const Qwen3_6MoeWeights &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators,
             kv::Format format = kv::Format::Int8);

  [[nodiscard]] const QwenTargetGeometry &geometry() const noexcept {
    return geometry_;
  }
  [[nodiscard]] const ops::Q4Projection &
  vocabularyProjection() const noexcept;

  void addPrefill(
      metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const kv::LayerStorage> kvLayers) const;
  void addVerify(
      metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
      std::span<const kv::LayerStorage> kvLayers,
      std::span<const kv::Q8ChunkedPrefillParams> q8,
      std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
      ops::Q4DispatchStats &stats) const;
  void addHead(metal::CommandGraph &graph, metal::MetalBuffer hidden,
               metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
               uint32_t normalizedRows, ops::LinearScratch scratch = {}) const;
  void addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    metal::MetalBuffer hidden, uint32_t rows) const;
  void addStateCommit(metal::CommandGraph &graph,
                      QwenTargetCommitBuffers buffers, uint32_t lanes) const;

private:
  using WeightView =
      std::variant<const Qwen3_8Weights *, const Qwen3_6MoeWeights *>;

  template <class Weights>
  void addPrefillImpl(
      const Weights &weights, metal::CommandGraph &graph,
      QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const kv::LayerStorage> kvLayers) const;
  template <class Weights>
  void addVerifyImpl(
      const Weights &weights, metal::CommandGraph &graph,
      QwenTargetVerifyBuffers buffers,
      std::span<const kv::LayerStorage> kvLayers,
      std::span<const kv::Q8ChunkedPrefillParams> q8,
      std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
      ops::Q4DispatchStats &stats) const;

  WeightView weights_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
};

} // namespace splash::model
