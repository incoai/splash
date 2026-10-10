#include "Qwen4Exp.hpp"
#include "model/QwenTargetLoader.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

constexpr uint64_t kFloat32Bytes = 4;

// The sections of one mix, in the image's order (GgufImage.cpp hyperImage).
Qwen4HyperWeights readHyper(WeightFile &file, const Qwen4ExpLayout &layout, bool inject, const std::string &label) {
  const BlockTargetFormat format;
  const uint32_t streams = layout.streamWidth();
  Qwen4HyperWeights hyper;
  hyper.norm = format.norm(file, streams, label + "-norm");
  hyper.down = format.projection(file, layout.hyperRank, streams, label + "-down");
  hyper.up = format.projection(file, streams, layout.hyperRank, label + "-up");
  if (inject)
    hyper.inject = file.section(uint64_t{layout.hyperConnections} * streams * kFloat32Bytes, label + "-inject");
  return hyper;
}

Qwen4IndexerWeights readIndexer(WeightFile &file, const Qwen4ExpLayout &layout) {
  const BlockTargetFormat format;
  Qwen4IndexerWeights indexer;
  indexer.query = format.projection(file, layout.indexerHeads * layout.indexerHeadDimension, layout.hiddenSize,
                                    "indexer-query");
  indexer.key = format.projection(file, layout.indexerHeadDimension, layout.hiddenSize, "indexer-key");
  // The indexer scores in fp32 (ops::Qwen4 QSA).
  indexer.query.destination = ops::FloatOutput::Float32;
  indexer.key.destination = ops::FloatOutput::Float32;
  indexer.queryNorm = format.norm(file, layout.indexerHeadDimension, "indexer-query-norm");
  indexer.keyNorm = format.norm(file, layout.indexerHeadDimension, "indexer-key-norm");
  return indexer;
}

void readFfn(WeightFile &file, Qwen4ExpLayerWeights &layer) {
  ops::BlockMoeWeights ffn;
  ffn.router = readQuantizedSegment(file, "router");
  ffn.gate.routed = readQuantizedSegment(file, "experts-gate");
  ffn.up.routed = readQuantizedSegment(file, "experts-up");
  ffn.down.routed = readQuantizedSegment(file, "experts-down");
  ffn.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
  ffn.up.shared = readQuantizedSegment(file, "shared-expert-up");
  ffn.down.shared = readQuantizedSegment(file, "shared-expert-down");
  ffn.sharedScalarGate = readQuantizedSegment(file, "shared-expert-scalar-gate");
  layer.ffn = std::move(ffn);
}

// The hybrid layout's consistency (requireQwenLayout without its affine
// StorageN checks: GGUF tiles span 64 columns) and Flash-Next's own fields.
void requireQwen4Layout(const Qwen4ExpLayout &layout) {
  if (layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension != (2 * layout.gdnKeyHeads + layout.gdnValueHeads) * layout.gdnHeadDimension ||
      layout.attentionWidth != layout.attentionQueryHeads * layout.attentionHeadDimension ||
      layout.gdnValueHeads * layout.gdnHeadDimension != layout.attentionWidth ||
      layout.packedFullWidth != 2 * layout.attentionWidth + 2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.packedGdnWidth < layout.actualGdnWidth() || layout.packedGdnWidth % 64 ||
      layout.expertsPerToken > layout.experts || layout.expertIntermediateSize % 64 || layout.hiddenSize % 256 ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("Qwen3.8-Flash-Next layout is inconsistent");
  if (!layout.hyperConnections || !layout.hyperRank || layout.hyperRank % 64 ||
      !layout.indexerHeads || !layout.indexerHeadDimension || !layout.indexerTopBlocks ||
      !layout.indexerBlockTokens || layout.pleNgram < 2 || layout.pleNgram > 4 || !layout.pleHeadsPerNgram ||
      !layout.pleHeadDimension || !layout.pleConvolutionTaps || layout.pleLayer >= layout.layers ||
      layout.isFullAttentionLayer(layout.pleLayer) || layout.mtpLayers > 1)
    throw WeightStoreError("Qwen3.8-Flash-Next layout is inconsistent");
}

} // namespace

Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend, Qwen4ExpLayout layout,
                                    const QwenTargetFiles &files, GgufMtpLoader *mtp) {
  requireQwen4Layout(layout);
  const auto *source = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files);
  if (!source) throw WeightStoreError("Qwen3.8-Flash-Next loads from a GGUF only");
  GgufTargetLoader &gguf = source->get();
  const BlockTargetFormat format;
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Qwen4ExpWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint32_t streams = layout.streamWidth();
  for (uint32_t index = 0; index < layout.layers; ++index) {
    const bool fullAttention = layout.isFullAttentionLayer(index);
    WeightFile file = gguf.layer(index);
    Qwen4ExpLayerWeights &layer = result.layers.emplace_back();
    layer.mixerHyper = readHyper(file, layout, true, "mixer-hyper");
    layer.mixer = readQwenMixer(file, format, layout, fullAttention);
    if (fullAttention) layer.indexer = readIndexer(file, layout);
    layer.ffnHyper = readHyper(file, layout, true, "ffn-hyper");
    readFfn(file, layer);
    if (index == layout.pleLayer) {
      Qwen4PleWeights ple;
      ple.key = format.projection(file, streams, layout.pleWidth(), "ple-key");
      ple.value = format.projection(file, layout.hiddenSize, layout.pleWidth(), "ple-value");
      ple.keyNorm = format.norm(file, streams, "ple-key-norm");
      ple.queryNorm = format.norm(file, streams, "ple-query-norm");
      ple.convolutionNorm = format.norm(file, streams, "ple-convolution-norm");
      ple.convolution =
          file.section(uint64_t{streams} * layout.pleConvolutionTaps * kFloat32Bytes, "ple-convolution");
      layer.ple = std::move(ple);
    }
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = gguf.head();
    result.finalHyper = readHyper(file, layout, false, "final-hyper");
    result.logitsProjection = format.projection(file, layout.vocabularySize, layout.hiddenSize, "logits");
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = gguf.embedding();
    result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }
  {
    const gguf::PleHash &hash = gguf.pleHash();
    if (hash.multipliers.size() != layout.pleNgram || hash.headOffsets.size() != layout.pleHeads() ||
        hash.eosToken != layout.pleEosToken || hash.tableRows > UINT32_MAX)
      throw WeightStoreError("the GGUF's PLE hash does not match the layout");
    WeightFile file = gguf.ple();
    result.pleTable =
        readBlockEmbedding(file, static_cast<uint32_t>(hash.tableRows), layout.pleHeadDimension, "ple-table");
    file.finish();
    result.files.push_back(file.record());
    std::copy(hash.multipliers.begin(), hash.multipliers.end(), result.pleHash.multipliers.begin());
    result.pleHash.headOffsets = hash.headOffsets;
    result.pleHash.headVocabularies = hash.headVocabularies;
    result.pleHash.tableRows = hash.tableRows;
  }
  if ((mtp != nullptr) != (layout.mtpLayers == 1))
    throw WeightStoreError("the MTP head does not match the layout");
  if (mtp) {
    // The image's order (GgufImage.cpp planMtpImage).
    WeightFile file = mtp->open();
    Qwen4MtpWeights head;
    head.block.mixerHyper = readHyper(file, layout, true, "mtp-mixer-hyper");
    head.block.mixer = readQwenMixer(file, format, layout, true);
    head.block.indexer = readIndexer(file, layout);
    head.block.ffnHyper = readHyper(file, layout, true, "mtp-ffn-hyper");
    readFfn(file, head.block);
    head.embedHidden = format.projection(file, layout.hiddenSize, 2 * layout.hiddenSize, "mtp-embed-hidden");
    head.embedNorm = format.norm(file, layout.hiddenSize, "mtp-embed-norm");
    head.hiddenNorm = format.norm(file, streams, "mtp-hidden-norm");
    head.head = readHyper(file, layout, false, "mtp-head");
    file.finish();
    result.files.push_back(file.record());
    result.mtp = std::move(head);
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes =
      metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
