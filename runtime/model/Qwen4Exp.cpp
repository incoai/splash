#include "Qwen4Exp.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SparseRowStore.hpp"

#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace splash::model {
namespace {

// The PLE metadata's integers, which the parser keeps as doubles
// (GgufFile::numericArray), are below these: a multiplier below 2^53, past
// which a double does not hold every integer, and a head's offset or row
// count below 2^31, a 32-bit signed value.
constexpr double kMultiplierLimit = 0x1p53, kHeadValueLimit = 0x1p31;

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
  // The indexer scores in fp32 (ops::BlockIndexer).
  indexer.query.destination = ops::FloatOutput::Float32;
  indexer.key.destination = ops::FloatOutput::Float32;
  indexer.queryNorm = format.norm(file, layout.indexerHeadDimension, "indexer-query-norm");
  indexer.keyNorm = format.norm(file, layout.indexerHeadDimension, "indexer-key-norm");
  return indexer;
}

// Throws unless Flash-Next's own dimensions are set: the hyper-connections,
// the QSA indexer, and a PLE n-gram embedding of 2 to kPleMaximumNgram tokens
// that feeds a GDN layer. loadQwenTarget checks the hybrid's
// (requireQwenLayout).
void requireQwen4Layout(const Qwen4ExpLayout &layout) {
  if (!layout.hyperConnections || !layout.hyperRank || !layout.indexerHeads || !layout.indexerHeadDimension ||
      !layout.indexerTopBlocks || !layout.indexerBlockTokens || layout.pleNgram < 2 ||
      layout.pleNgram > kPleMaximumNgram || !layout.pleHeadsPerNgram || !layout.pleHeadDimension ||
      !layout.pleConvolutionTaps || layout.pleLayer >= layout.layers || layout.isFullAttentionLayer(layout.pleLayer))
    throw WeightStoreError("Qwen3.8-Flash-Next layout is inconsistent");
}

// The PLE hash the GGUF's metadata states: every multiplier the array holds a whole
// number, the first pleNgram the hash's; each head's rows inside the table
// (per_layer_token_embd), whose rows are 32-bit ids; and its end-of-sequence
// token and the image token it may name the layout's.
Qwen4PleHash readPleHash(const GgufFile &file, const Qwen4ExpLayout &layout) {
  const std::string key = std::string(Qwen4ExpLayout::ggufArchitecture) + ".ple.";
  const auto array = [&](const char *name, size_t count) {
    const std::optional<std::span<const double>> values = file.numericArray(key + name);
    if (!values || values->size() < count) throw GgufError("GGUF PLE metadata is missing: ple." + std::string(name));
    return *values;
  };
  const auto whole = [](double value, double limit) {
    if (!(value >= 0 && value < limit) || value != std::floor(value))
      throw GgufError("GGUF PLE metadata holds an invalid value");
    return value;
  };
  Qwen4PleHash hash;
  const std::span<const double> multipliers = array("layer_multipliers", layout.pleNgram);
  for (size_t position = 0; position < multipliers.size(); ++position) {
    const auto multiplier = static_cast<uint64_t>(whole(multipliers[position], kMultiplierLimit));
    if (position < layout.pleNgram) hash.multipliers[position] = multiplier;
  }
  const uint32_t heads = layout.pleHeads();
  const std::span<const double> offsets = array("head_offsets", heads), sizes = array("head_vocab_sizes", heads);
  hash.tableRows = file.require("per_layer_token_embd.weight").rows();
  for (uint32_t head = 0; head < heads; ++head) {
    hash.headOffsets.push_back(static_cast<uint32_t>(whole(offsets[head], kHeadValueLimit)));
    hash.headVocabularies.push_back(static_cast<uint32_t>(whole(sizes[head], kHeadValueLimit)));
    if (!hash.headVocabularies.back() ||
        uint64_t{hash.headOffsets.back()} + hash.headVocabularies.back() > hash.tableRows)
      throw GgufError("a PLE head's rows lie outside per_layer_token_embd");
  }
  const std::optional<uint64_t> eos = file.metadata().unsignedValue(key + "eos_token_id");
  const std::optional<uint64_t> image = file.metadata().unsignedValue(key + "image_token_id");
  if (!eos) throw GgufError("GGUF PLE metadata is missing: ple.eos_token_id");
  if (*eos != layout.pleEosToken || (image && *image != layout.pleImageToken) || hash.tableRows > UINT32_MAX)
    throw WeightStoreError("the GGUF's PLE hash does not match the layout");
  return hash;
}

} // namespace

Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend, Qwen4ExpLayout layout,
                                    const QwenTargetFiles &files) {
  requireQwen4Layout(layout);
  return loadQwenTarget<Qwen4ExpWeights>(
      backend, layout, files, [&](GgufTargetLoader &gguf, const BlockTargetFormat &format, const MoeExperts &experts) {
        Qwen4ExpWeights result;
        result.layout = layout;
        // The PLE hash first, from the GGUF's metadata, before any image loads.
        result.pleHash = readPleHash(gguf.file(), layout);
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
          layer.ffn = readMoeBlock(file, layout, experts, index);
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
        if (std::optional<GgufTargetLoader::StreamedTable> table = gguf.streamTable()) {
          if (table->store->table().rows != result.pleHash.tableRows)
            throw WeightStoreError("the GGUF's PLE table does not match its hash");
          result.streamedTable = std::move(table->store);
          result.pleFormat = table->format;
          result.files.push_back(std::move(table->record));
        } else {
          WeightFile file = gguf.table();
          result.pleTable = readBlockEmbedding(file, static_cast<uint32_t>(result.pleHash.tableRows),
                                               layout.pleHeadDimension, "ple-table");
          file.finish();
          result.files.push_back(file.record());
        }
        return result;
      });
}

} // namespace splash::model
