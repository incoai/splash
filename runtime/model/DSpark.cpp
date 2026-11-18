#include "model/DSpark.hpp"

#include "model/GgufImage.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"

#include <algorithm>
#include <functional>
#include <string>

namespace splash::model {

DSparkLayout readDSparkLayout(const GgufFile &file, const DeepSeek41Layout &target) {
  const GgufMetadata &metadata = file.metadata();
  if (metadata.architecture() != "dflash")
    throw GgufError("GGUF architecture is " + metadata.architecture() + ", but a DSpark draft's is dflash");
  const auto value = [&](const std::string &key) {
    const std::optional<uint64_t> found = metadata.unsignedValue(key);
    if (!found) throw GgufError("a DSpark GGUF states no " + key);
    return *found;
  };
  DSparkLayout layout;
  DeepSeek41Layout &layers = layout.layers;
  layers = target;
  layers.layers = static_cast<uint32_t>(value("dflash.block_count"));
  layers.experts = static_cast<uint32_t>(value("dflash.expert_count"));
  layers.expertsPerToken = static_cast<uint32_t>(value("dflash.expert_used_count"));
  layers.expertIntermediateSize = static_cast<uint32_t>(value("dflash.expert_feed_forward_length"));
  if (const std::optional<double> base = metadata.floatValue("dflash.rope.freq_base"))
    layers.rotaryTheta = static_cast<float>(*base);
  if (!layers.layers || layers.layers >= 64) throw GgufError("a DSpark GGUF's block_count is out of range");
  const uint64_t every = (uint64_t{1} << layers.layers) - 1;
  layers.slidingLayers = every;
  DeepSeek41Dimensions &d = layers.deepseek;
  d.compressedLayers = d.pairedLayers = d.compressorLayers = d.indexerLayers = d.engramLayers = 0;
  requireDeepSeek41Layout(layers);

  layout.blockSize = static_cast<uint32_t>(value("dflash.block_size"));
  layout.noiseToken = static_cast<uint32_t>(value("tokenizer.ggml.mask_token_id"));
  const GgufTensor *markov = file.find("markov_w1.weight");
  if (!markov) throw GgufError("a DSpark GGUF holds no markov_w1.weight");
  layout.markovRank = static_cast<uint32_t>(markov->columns());
  const auto targets = file.numericArray("dflash.target_layers");
  if (!targets || targets->empty() || targets->size() > layout.targetLayerValues.size())
    throw GgufError("a DSpark GGUF's target_layers are out of range");
  for (const double layer : *targets) {
    if (layer < 0 || layer >= target.layers || layer != static_cast<uint32_t>(layer) ||
        (layout.targetLayerCount && layer <= layout.targetLayerValues[layout.targetLayerCount - 1]))
      throw GgufError("a DSpark GGUF's target_layers are not ascending layers of its target");
    layout.targetLayerValues[layout.targetLayerCount++] = static_cast<uint32_t>(layer);
  }
  if (!layout.blockSize || layout.blockSize > ExecutionLimits::draftProposalTokens ||
      layout.noiseToken >= target.vocabularySize || !layout.markovRank)
    throw GgufError("a DSpark GGUF's block size, noise token or Markov rank is out of range");
  return layout;
}

DFlashDraftLayout dsparkDraftLayout(const DSparkLayout &layout) {
  const DeepSeek41Layout &layers = layout.layers;
  DFlashDraftLayout result{.kind = DraftKind::DSpark,
                           .layers = layers.layers,
                           .hiddenSize = layers.hiddenSize,
                           .vocabularySize = layers.vocabularySize,
                           .attentionHeadDimension = layers.attentionHeadDimension,
                           .rotaryTheta = layers.rotaryTheta,
                           .kvHeads = layers.attentionKvHeads,
                           .targetLayerCount = layout.targetLayerCount,
                           .blockSize = layout.blockSize};
  std::ranges::copy(layout.targetLayers(), result.targetLayers.begin());
  return result;
}

std::optional<std::filesystem::path> findDSparkGguf(const std::filesystem::path &directory) {
  std::optional<std::filesystem::path> found;
  if (!std::filesystem::is_directory(directory)) return found;
  for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() != ".gguf") continue;
    if (found) throw GgufError("a draft directory holds more than one GGUF: " + directory.string());
    found = entry.path();
  }
  return found;
}

DSparkWeights loadDSparkWeights(metal::MetalBackend &backend, WeightImages &images,
                                const std::filesystem::path &path, const DSparkLayout &layout) {
  const DeepSeek41Layout &layers = layout.layers;
  GgufTargetLoader loader(
      backend, images, {path},
      [&](const GgufFile &file) {
        return gguf::planDSparkImages(file, layers, layout.targetLayerCount, layout.markovRank);
      },
      layers.layers, "draft/");
  const MoeExperts experts = openMoeExperts(backend, QwenTargetFiles{std::ref(loader)}, layers);
  const BlockTargetFormat format;
  DSparkWeights result;
  result.layout = layout;
  result.expertSlots = experts.resident;
  for (uint32_t index = 0; index < layers.layers; ++index) {
    WeightFile file = loader.layer(index);
    result.layers.push_back(readDeepSeek41Layer(file, format, layers, experts, index));
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = loader.head();
    result.mainNorm = format.norm(file, layers.hiddenSize, "main-norm");
    result.mainProjection =
        format.projection(file, layers.hiddenSize, layout.targetLayerCount * layers.hiddenSize, "main-projection");
    result.outputNorm = format.norm(file, layers.hiddenSize, "output-norm");
    result.markovHead = format.projection(file, layers.vocabularySize, layout.markovRank, "markov-head");
    result.markovHead.destination = ops::FloatOutput::Float32;
    result.confidence =
        file.section(uint64_t{layers.hiddenSize + layout.markovRank} * kFloat32Bytes, "confidence");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = loader.embedding();
    result.markovEmbedding = format.embedding(file, layers.vocabularySize, layout.markovRank);
    file.finish();
    result.files.push_back(file.record());
  }
  return result;
}

} // namespace splash::model
