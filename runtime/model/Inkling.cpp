#include "model/Inkling.hpp"

#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"

#include <stdexcept>
#include <string>

namespace splash::model {
namespace {

// The F32 values of a section of `values` of them.
metal::MetalBuffer floatValues(WeightFile &file, uint64_t values, std::string_view label) {
  return file.section(values * kFloat32Bytes, label);
}

// One side (InklingSides) of `rows` rows of hidden inputs: the segments the
// planner wrote for its runs of tensors of one type, in row order, until they
// hold its rows, each at its first row; its rows written fp32.
ops::Projection readSide(WeightFile &file, uint32_t rows, uint32_t inputs, const std::string &label) {
  ops::BlockWeights weights;
  uint32_t offset = 0;
  while (offset < rows) {
    ops::QuantizedSegment segment = readQuantizedSegment(file, label + "-" + std::to_string(weights.segments.size()));
    if (segment.inputSize != inputs || segment.outputSize > rows - offset)
      throw WeightStoreError("GGUF tensor does not match the layout: " + label);
    segment.columnOffset = offset;
    offset += segment.outputSize;
    weights.segments.push_back(std::move(segment));
  }
  ops::Projection side(rows, inputs, std::move(weights));
  side.destination = ops::FloatOutput::Float32;
  return side;
}

// A float projection of the GGUF's F32 or BF16 rows as stored, which writes
// fp32 rows.
ops::Projection readFloatProjection(WeightFile &file, uint32_t rows, uint32_t inputs, std::string_view label) {
  ops::Projection projection = readBlockProjection(file, rows, inputs, label);
  projection.destination = ops::FloatOutput::Float32;
  return projection;
}

// A weight layer's attention, in its image's order (GgufImage.cpp
// InklingGguf).
InklingAttentionWeights readAttention(WeightFile &file, const BlockTargetFormat &format, const InklingLayout &layout,
                                      uint32_t index) {
  const bool global = inklingGlobalLayer(layout, index);
  const InklingSides sides = InklingSides::of(layout);
  const uint32_t hidden = layout.hiddenSize, heads = layout.attentionQueryHeads;
  const uint32_t kvRows = layout.attentionKvHeads * layout.attentionHeadDimension;
  InklingAttentionWeights attention;
  if (index) attention.residualQueries = floatValues(file, (global ? 3ull : 1ull) * hidden, "attention-queries");
  for (uint32_t input = 0; input < (global ? 3u : 1u); ++input)
    attention.inputNorms[input] = format.norm(file, hidden, "input-norm-" + std::to_string(input));
  attention.querySide = readSide(file, sides.query, hidden, "query-side");
  attention.keySide = readSide(file, sides.keyRows(global), hidden, "key-side");
  attention.valueSide = readSide(file, sides.keyRows(global), hidden, "value-side");
  attention.queryClip = floatValues(file, uint64_t{heads} * layout.attentionHeadDimension, "query-clip");
  attention.relativeClip = floatValues(file, uint64_t{heads} * layout.relativeDimension, "relative-clip");
  attention.relativeBias =
      floatValues(file, uint64_t{layout.relativeExtent} * layout.relativeDimension, "relative-bias");
  attention.keyConvolution = floatValues(file, uint64_t{kvRows} * layout.shortConvolutionTaps, "key-convolution");
  attention.valueConvolution =
      floatValues(file, uint64_t{kvRows} * layout.shortConvolutionTaps, "value-convolution");
  attention.outputProjection = format.projection(file, hidden, layout.attentionWidth, "attention-output");
  attention.outputConvolution =
      floatValues(file, uint64_t{hidden} * layout.shortConvolutionTaps, "output-convolution");
  return attention;
}

// A core weight layer's loop weights, after its FFN's.
InklingLoopWeights readLoop(WeightFile &file, const InklingLayout &layout) {
  const uint32_t hidden = layout.hiddenSize, rank = InklingLayout::loopGateRank;
  InklingLoopWeights loop;
  loop.messageQueries = floatValues(file, uint64_t{InklingLayout::loopMessages} * hidden, "message-queries");
  for (InklingLoopGateWeights &gate : loop.gates) {
    gate.down = readFloatProjection(file, rank, 2 * hidden, "loop-gate-down");
    gate.gain = readFloatProjection(file, hidden, rank, "loop-gate-gain");
    gate.forget = readFloatProjection(file, hidden, rank, "loop-gate-forget");
  }
  return loop;
}

} // namespace

// Each weight layer's image holds its attention (readAttention), then its
// FFN's depth-attention query, norm, MoE block (readMoeBlock, of no shared
// expert) and its router's selection bias, its FFN output's convolution and,
// on a core layer, its loop weights; the head's image the head's
// depth-attention query, the final norm, the embedding's norm and the
// logits; the embedding's the token rows (GgufImage.cpp InklingGguf). The
// MoE blocks are the weight layers', which the passes of the loop share.
InklingWeights loadInklingWeights(metal::MetalBackend &backend, InklingLayout layout, const QwenTargetFiles &files) {
  return loadQwenTarget<InklingWeights>(
      backend, layout, files, [&](auto &loader, const BlockTargetFormat &format, const MoeExperts &experts) {
        InklingWeights result;
        result.layout = layout;
        result.layers.reserve(layout.weightLayers());
        const uint32_t hidden = layout.hiddenSize;
        for (uint32_t index = 0; index < layout.weightLayers(); ++index) {
          WeightFile file = loader.layer(index);
          InklingLayerWeights &layer = result.layers.emplace_back();
          layer.mixer = readAttention(file, format, layout, index);
          layer.ffnResidualQuery = floatValues(file, hidden, "ffn-query");
          layer.ffnNorm = format.norm(file, hidden, "ffn-norm");
          layer.ffn = readMoeBlock(file, layout, experts, layout.moeBlock(index));
          layer.ffnConvolution = floatValues(file, uint64_t{hidden} * layout.shortConvolutionTaps, "ffn-convolution");
          if (index >= layout.loopStart && index < layout.loopEnd) layer.loop = readLoop(file, layout);
          file.finish();
          result.files.push_back(file.record());
        }
        {
          WeightFile file = loader.head();
          result.outputQuery = floatValues(file, hidden, "output-query");
          result.finalNorm = format.norm(file, hidden, "final-norm");
          result.embeddingNorm = format.norm(file, hidden, "embedding-norm");
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
        return result;
      });
}

void includeMixer(QwenTargetGeometry &geometry, const InklingAttentionWeights &attention) {
  geometry.include(attention.querySide);
  geometry.include(attention.keySide);
  geometry.include(attention.valueSide);
  geometry.include(attention.outputProjection);
}

void includeFamilyProjections(QwenTargetGeometry &geometry, const InklingWeights &weights) {
  for (const InklingLayerWeights &layer : weights.layers)
    if (layer.loop)
      for (const InklingLoopGateWeights &gate : layer.loop->gates) {
        geometry.include(gate.down);
        geometry.include(gate.gain);
        geometry.include(gate.forget);
      }
}

} // namespace splash::model
