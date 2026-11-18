#include "NemotronH.hpp"

#include "model/ExpertStore.hpp"
#include "model/GgufTarget.hpp"
#include "model/NemotronHTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SparseRowStore.hpp"

#include <algorithm>

namespace splash::model {

namespace {

// A layer's weights; a MoE layer's routed experts are block `block` of
// `experts` (openMoeExperts).
NemotronHLayerWeights readLayer(WeightFile &file, const NemotronHLayout &layout, uint32_t index,
                                const MoeExperts &experts, uint32_t block) {
  const auto f32 = [&](uint32_t width, std::string_view label) { return readNorm(file, width, true, label).buffer; };
  NemotronHLayerWeights layer;
  layer.norm = readNorm(file, layout.hiddenSize, true, "block-norm");
  layer.norm.epsilon = layout.rmsEpsilon;
  switch (NemotronHLayout::block(index)) {
  case NemotronBlock::Mamba: {
    NemotronMambaWeights m;
    m.in = readBlockProjection(file, layout.mambaInProjectionRows, layout.hiddenSize, "ssm-in");
    m.mixer.convolution = f32(layout.convolutionDimension * kGdnConvolutionTaps, "ssm-conv");
    m.mixer.convolutionBias = f32(layout.convolutionDimension, "ssm-conv-bias");
    m.mixer.dtBias = f32(layout.gdnValueHeads, "ssm-dt-bias");
    m.mixer.a = f32(layout.gdnValueHeads, "ssm-a");
    m.mixer.d = f32(layout.gdnValueHeads, "ssm-d");
    m.mixer.norm = f32(layout.mambaInnerSize, "ssm-norm");
    m.out = readBlockProjection(file, layout.hiddenSize, layout.mambaInnerSize, "ssm-out");
    layer.block = std::move(m);
    break;
  }
  case NemotronBlock::Attention: {
    ops::BlockWeights qkv;
    uint32_t offset = 0;
    for (const std::string_view label : {"attn-q", "attn-k", "attn-v"}) {
      ops::QuantizedSegment segment = readQuantizedSegment(file, label);
      segment.columnOffset = offset;
      offset += segment.outputSize;
      qkv.segments.push_back(std::move(segment));
    }
    if (offset != layout.packedFullWidth) throw WeightStoreError("Nemotron-H attention q | k | v has unexpected rows");
    QwenAttentionWeights a;
    a.inputProjection = {offset, layout.hiddenSize, std::move(qkv)};
    a.outputProjection = readBlockProjection(file, layout.hiddenSize, layout.attentionWidth, "attn-output");
    layer.block = std::move(a);
    break;
  }
  case NemotronBlock::Moe: {
    NemotronMoeWeights m;
    const uint32_t count = NemotronHLayout::layerExperts(layout, index);
    m.experts.router = readQuantizedSegment(file, "router");
    m.experts.selectionBias = f32(count, "exp-probs-bias");
    m.latentDown = readBlockProjection(file, layout.moeLatentSize, layout.hiddenSize, "latent-down");
    if (!experts.store) {
      m.experts.up.routed = readQuantizedSegment(file, "experts-up");
      m.experts.down.routed = readQuantizedSegment(file, "experts-down");
      m.experts.slotTable = experts.resident;
      m.experts.slots = count;
      m.experts.chunks = ops::residentExpertChunks(m.experts);
    }
    m.latentUp = readBlockProjection(file, layout.hiddenSize, layout.moeLatentSize, "latent-up");
    m.sharedUp = readBlockProjection(file, layout.sharedIntermediateSize, layout.hiddenSize, "shared-up");
    m.sharedDown = readBlockProjection(file, layout.hiddenSize, layout.sharedIntermediateSize, "shared-down");
    // The store keeps the block's router and selection bias, which predict
    // its routes.
    if (experts.store) experts.store->bind(block, m.experts);
    m.shape = layout.moeBlockShape();
    m.shape.experts = count;
    m.shape.expertIntermediateSize = NemotronHLayout::layerExpertWidth(layout, index);
    if (!m.shape.valid()) throw WeightStoreError("Nemotron-H MoE block has no compiled shape");
    layer.block = std::move(m);
    break;
  }
  }
  return layer;
}

} // namespace

NemotronHWeights loadNemotronHWeights(metal::MetalBackend &backend, const NemotronHLayout &layout,
                                      const QwenTargetFiles &files) {
  const auto *loader = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files);
  if (!loader) throw WeightStoreError(std::string(layout.family) + " loads from a GGUF only");
  GgufTargetLoader &gguf = loader->get();
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  NemotronHWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  // Its MoE layers' experts, in a store when they stream: its blocks those
  // layers, in order (gguf::streamedExperts).
  MoeExperts experts{.experts = layout.experts};
  if (gguf.streamsExperts())
    experts.store = std::make_shared<ExpertStore>(backend, gguf.streamedExperts(), layout.moeBlockShape(),
                                                  gguf.expertStoreOptions());
  else
    experts.resident = residentExpertSlots(backend, layout.experts);
  result.experts = experts.store;
  for (uint32_t index = 0, block = 0; index < layout.layers; ++index) {
    WeightFile file = gguf.layer(index);
    result.layers.push_back(readLayer(file, layout, index, experts, block));
    block += NemotronHLayout::block(index) == NemotronBlock::Moe;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = gguf.head();
    result.finalNorm = readNorm(file, layout.hiddenSize, true, "final-norm");
    result.finalNorm.epsilon = layout.rmsEpsilon;
    result.logitsProjection = readBlockProjection(file, layout.vocabularySize, layout.hiddenSize, "logits");
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = gguf.embedding();
    result.tokenEmbedding = readBlockEmbedding(file, layout.vocabularySize, layout.hiddenSize, "embedding");
    file.finish();
    result.files.push_back(file.record());
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

QwenTargetGeometry qwenTargetGeometry(const NemotronHWeights &weights, std::span<const uint32_t> captureLayers) {
  QwenTargetGeometry geometry(weights.layout, captureLayers);
  geometry.kvLayout = weights.layout.kvLayout();
  geometry.stateLayout = weights.layout.gdnStateLayout();
  // The runtime sizes its MoE scratch for the geometry's shape, of the most
  // experts and the widest, and its formats, which every block's plans take
  // (NemotronHTarget) so that a block of other formats needs no more.
  std::vector<const ops::BlockMoeWeights *> moeBlocks;
  for (const NemotronHLayerWeights &layer : weights.layers)
    std::visit(
        [&](const auto &block) {
          using Block = std::decay_t<decltype(block)>;
          if constexpr (std::is_same_v<Block, NemotronMoeWeights>) {
            for (const ops::Projection *p : {&block.latentDown, &block.latentUp, &block.sharedUp, &block.sharedDown})
              geometry.include(*p);
            moeBlocks.push_back(&block.experts);
          } else if constexpr (std::is_same_v<Block, NemotronMambaWeights>) {
            geometry.include(block.in);
            geometry.include(block.out);
          } else {
            geometry.include(block.inputProjection);
            geometry.include(block.outputProjection);
          }
        },
        layer.block);
  geometry.moeRouterFormat = ops::moeRouterFormat(moeBlocks);
  geometry.prefillProjections = geometry.decodeProjections;
  geometry.include(weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

std::unique_ptr<QwenTarget> makeQwenTarget(const NemotronHWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t) {
  return std::make_unique<NemotronHTarget>(weights, geometry, backend, operators);
}

SparseRowStoreOptions streamedTableRows(const NemotronHLayout &) { return {}; }

} // namespace splash::model
