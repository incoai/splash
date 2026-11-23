#include "model/Inkling2Loop.hpp"

#include "model/GgufImage.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/SparseRowStore.hpp"

#include <array>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

// The bytes of `count` F32 values.
constexpr uint64_t floatBytes(uint64_t count) noexcept { return count * kFloat32Bytes; }

// A full layer's taps projection: its one segment of fewer rows than a tile
// as its image holds it (gguf::Builder::smallProjection), F32 or BF16 rows as
// stored or a quantized tensor of whole tiles, its rows past `rows` zero; the
// projection's columns to a whole tile, those past its rows padding; bf16, as
// the convolution reads its taps.
ops::Projection readTaps(WeightFile &file, uint32_t rows, uint32_t inputs, std::string_view label) {
  ops::QuantizedSegment segment = readQuantizedSegment(file, label);
  if ((segment.outputSize != rows && segment.outputSize != gguf::tiledRows(rows)) || segment.inputSize != inputs)
    throw WeightStoreError("GGUF tensor does not match the layout: " + std::string(label));
  return {static_cast<uint32_t>(gguf::tiledRows(rows)), inputs, ops::BlockWeights{{std::move(segment)}}};
}

// A tensor [rows, inputs] the float projection runs in fp32 (a layer's sinks,
// a loop layer's gates) as its image holds it (gguf::Builder::floatTensor):
// F32 or BF16 rows.
ops::QuantizedSegment readFloatTensor(WeightFile &file, uint32_t rows, uint32_t inputs, std::string_view label) {
  ops::QuantizedSegment segment = readFloatSegment(file, label);
  if (segment.outputSize != rows || segment.inputSize != inputs)
    throw WeightStoreError("GGUF tensor does not match the layout: " + std::string(label));
  return segment;
}

// The packed projection of a q input: queries | gate | relative scores (one
// segment each, of a tile's rows each) then the sinks, the columns past them
// to a whole tile padding. The sinks' float segment runs apart from the three
// quantized ones, which run fused (ops::Linear).
ops::Projection readQueryProjection(WeightFile &file, const BlockTargetFormat &format,
                                    const Inkling2LoopLayout &layout) {
  const uint32_t hidden = layout.hiddenSize;
  ops::BlockWeights weights;
  uint32_t offset = 0;
  for (const auto &[rows, label] : {std::pair{layout.attentionWidth, "attn-q"},
                                    std::pair{layout.attentionWidth, "attn-gate"},
                                    std::pair{layout.relativeWidth(), "attn-relative"}}) {
    ops::Projection projection = format.projection(file, rows, hidden, label);
    ops::QuantizedSegment segment = projection.blocks().segments.front();
    segment.columnOffset = offset;
    offset += segment.outputSize;
    weights.segments.push_back(std::move(segment));
  }
  ops::QuantizedSegment sinks = readFloatTensor(file, layout.attentionQueryHeads, hidden, "attn-sinks");
  sinks.columnOffset = offset;
  weights.segments.push_back(std::move(sinks));
  return {layout.queryProjectionWidth(), hidden, std::move(weights)};
}

// A float projection [outputs, inputs] of F32 or BF16 rows as stored, run in
// fp32 into fp32 (a loop layer's gates).
ops::Projection readFloatProjection(WeightFile &file, uint32_t outputs, uint32_t inputs, std::string_view label) {
  ops::Projection projection(outputs, inputs, ops::BlockWeights{{readFloatTensor(file, outputs, inputs, label)}});
  projection.destination = ops::FloatOutput::Float32;
  return projection;
}

// A message gate's projections, in the image's order.
Inkling2LoopGateWeights readGate(WeightFile &file, const Inkling2LoopLayout &layout, const std::string &label) {
  const uint32_t hidden = layout.hiddenSize, rank = layout.loopGateRank;
  return {readFloatProjection(file, rank, 2 * hidden, label + "-down"),
          readFloatProjection(file, hidden, rank, label + "-forget"),
          readFloatProjection(file, hidden, rank, label + "-gain")};
}

// A weight layer's image, in its order (model/Inkling2LoopGguf.cpp).
Inkling2LoopLayerWeights readLayer(WeightFile &file, const Inkling2LoopLayout &layout, const BlockTargetFormat &format,
                                   const MoeExperts &experts, uint32_t index) {
  const uint32_t hidden = layout.hiddenSize, taps = layout.convolutionTaps, kvWidth = layout.kvWidth();
  Inkling2LoopLayerWeights layer;
  if (const uint32_t queries = layout.mixerQueries(index))
    layer.mixerQueries = file.section(floatBytes(uint64_t{queries} * hidden), "mixer-queries");
  if (layout.slides(index)) {
    layer.inputs = Inkling2LoopSlidingInputs{format.norm(file, hidden, "input-norm")};
  } else {
    Inkling2LoopFullInputs inputs;
    inputs.queryNorm = format.norm(file, hidden, "query-input-norm");
    inputs.keyNorm = format.norm(file, hidden, "key-input-norm");
    inputs.valueNorm = format.norm(file, hidden, "value-input-norm");
    inputs.keyTaps = readTaps(file, layout.convolutionTapsWidth(), hidden, "key-taps");
    inputs.valueTaps = readTaps(file, layout.convolutionTapsWidth(), hidden, "value-taps");
    layer.inputs = std::move(inputs);
  }
  Inkling2LoopAttentionWeights &attention = layer.mixer;
  attention.inputProjection = readQueryProjection(file, format, layout);
  attention.key = format.projection(file, kvWidth, hidden, "attn-k");
  attention.value = format.projection(file, kvWidth, hidden, "attn-v");
  attention.queryScale = file.section(floatBytes(layout.attentionWidth), "query-scale");
  attention.relativeScale = file.section(floatBytes(layout.relativeWidth()), "relative-scale");
  attention.relativeBias =
      file.section(floatBytes(uint64_t{layout.relativeExtent} * layout.relativeRank), "relative-bias");
  attention.keyConvolution = file.section(floatBytes(uint64_t{kvWidth} * taps), "key-convolution");
  attention.valueConvolution = file.section(floatBytes(uint64_t{kvWidth} * taps), "value-convolution");
  attention.outputProjection = format.projection(file, hidden, layout.attentionWidth, "attention-output");
  attention.outputConvolution = file.section(floatBytes(uint64_t{hidden} * taps), "output-convolution");
  layer.ffnQuery = file.section(floatBytes(hidden), "ffn-query");
  layer.ffnNorm = format.norm(file, hidden, "ffn-norm");
  layer.ffn = readMoeBlock(file, layout, experts, index);
  layer.ffnConvolution = file.section(floatBytes(uint64_t{hidden} * taps), "ffn-convolution");
  if (layout.schedule().isLoopLayer(index)) {
    Inkling2LoopMessageWeights messages;
    messages.queries = file.section(floatBytes(uint64_t{messages.gates.size()} * hidden), "message-queries");
    constexpr std::array<std::string_view, 4> kInputs{"q", "k", "v", "ffn"};
    for (size_t gate = 0; gate < kInputs.size(); ++gate)
      messages.gates[gate] = readGate(file, layout, std::string(kInputs[gate]) + "-gate");
    layer.messages = std::move(messages);
  }
  return layer;
}

} // namespace

void requireInkling2Layout(const Inkling2LoopLayout &layout) {
  const kv::Layout kv = layout.kvLayout();
  const Inkling2LoopSchedule schedule = layout.schedule();
  if (!schedule.valid() || layout.layers != schedule.applications() || layout.weightLayers > 64 ||
      layout.loopPasses < 2 || layout.fullAttentionPeriod != 1 || layout.gdnLayerCount() || !kv.valid() ||
      layout.attentionWidth != layout.attentionQueryHeads * layout.attentionHeadDimension ||
      layout.packedFullWidth != layout.queryProjectionWidth() || layout.slidingWeightLayers >> layout.weightLayers ||
      !layout.loopGateRank || !layout.matchesKernels() || layout.ffnKind != QwenFfnKind::SparseMoe ||
      !layout.experts || layout.expertsPerToken > layout.experts || layout.sharedExpertSlices ||
      layout.sharedExpertGate || layout.leadingDenseLayers || !layout.vocabularyTokens ||
      layout.vocabularyTokens > layout.vocabularySize || layout.vocabularySize % QUANT_TILE_ROWS)
    throw WeightStoreError("Inkling-2-Mini layout is inconsistent");
}

// The layer images in weight layer order, then the head's (the token table's
// norm, the head's block mix query, the final norm and the head) and the
// embedding's (the token table), of a GGUF's or a checkpoint's images alike.
// Its routed experts open first, as loadQwenTarget opens them; its
// allocations are all that loading it allocates, and its fingerprint is its
// files'.
Inkling2LoopWeights loadInkling2LoopWeights(metal::MetalBackend &backend, Inkling2LoopLayout layout,
                                            const QwenTargetFiles &files) {
  requireInkling2Layout(layout);
  TargetImages &loader = std::visit([](const auto &source) -> TargetImages & { return source.get(); }, files);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  const MoeExperts experts = openMoeExperts(backend, files, layout);
  const BlockTargetFormat format;
  Inkling2LoopWeights result;
  result.layout = layout;
  result.layers.reserve(layout.weightLayers);
  const uint32_t hidden = layout.hiddenSize;
  for (uint32_t index = 0; index < layout.weightLayers; ++index) {
    WeightFile file = loader.layer(index);
    result.layers.push_back(readLayer(file, layout, format, experts, index));
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = loader.head();
    result.embeddingNorm = format.norm(file, hidden, "embedding-norm");
    result.outputQuery = file.section(floatBytes(hidden), "output-query");
    result.finalNorm = format.norm(file, hidden, "final-norm");
    result.logitsProjection = format.projection(file, layout.vocabularySize, hidden, "logits");
    // bf16 logits would round near-ties together (readQwenTargetHead).
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = loader.embedding();
    result.tokenEmbedding = format.embedding(file, layout.vocabularySize, hidden);
    file.finish();
    result.files.push_back(file.record());
  }
  result.experts = experts.store;
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

void includeFamilyProjections(QwenTargetGeometry &geometry, const Inkling2LoopWeights &weights) {
  for (const Inkling2LoopLayerWeights &layer : weights.layers) {
    geometry.include(layer.mixer.key);
    geometry.include(layer.mixer.value);
    if (const std::optional<ops::Projection> keyValues = keyValueProjection(layer)) geometry.include(*keyValues);
    if (const auto *full = std::get_if<Inkling2LoopFullInputs>(&layer.inputs)) {
      geometry.include(full->keyTaps);
      geometry.include(full->valueTaps);
    }
    if (layer.messages)
      for (const Inkling2LoopGateWeights &gate : layer.messages->gates) {
        geometry.include(gate.down);
        geometry.include(gate.forget);
        geometry.include(gate.gain);
      }
  }
}

std::optional<ops::Projection> keyValueProjection(const Inkling2LoopLayerWeights &layer) {
  if (!std::holds_alternative<Inkling2LoopSlidingInputs>(layer.inputs)) return std::nullopt;
  const std::array<const ops::Projection *, 2> both{&layer.mixer.key, &layer.mixer.value};
  return ops::Projection::sideBySide(both);
}

SparseRowStoreOptions streamedTableRows(const Inkling2LoopLayout &) { return {}; }

} // namespace splash::model
