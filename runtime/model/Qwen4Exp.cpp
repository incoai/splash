#include "Qwen4Exp.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SparseRowStore.hpp"

#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

// The PLE metadata's integers, which the parser keeps as doubles
// (GgufFile::numericArray), are below these: a multiplier below 2^53, past
// which a double does not hold every integer, and a head's offset or row
// count below 2^31, a 32-bit signed value.
constexpr double kMultiplierLimit = 0x1p53, kHeadValueLimit = 0x1p31;

// The sections of one mix, in the image's order (GgufImage.cpp hyperImage).
Qwen4HyperWeights readHyper(WeightFile &file, const BlockTargetFormat &format, const Qwen4ExpLayout &layout,
                            bool inject, const std::string &label) {
  const uint32_t streams = layout.streamWidth();
  Qwen4HyperWeights hyper;
  hyper.norm = format.norm(file, streams, label + "-norm");
  hyper.down = format.projection(file, layout.hyperRank, streams, label + "-down");
  hyper.up = format.projection(file, streams, layout.hyperRank, label + "-up");
  if (inject)
    hyper.inject = file.section(uint64_t{layout.hyperConnections} * streams * kFloat32Bytes, label + "-inject");
  return hyper;
}

// The indexer's projections are float segments (Qwen4ExpTarget.cpp
// addIndexerProjections): a GGUF's F32 or BF16 values, an MLX checkpoint's
// native rows.
Qwen4IndexerWeights readIndexer(WeightFile &file, const BlockTargetFormat &format, const Qwen4ExpLayout &layout) {
  const auto floatProjection = [&](uint32_t outputSize, std::string_view label) {
    ops::QuantizedSegment segment = readFloatSegment(file, label);
    if (segment.outputSize != outputSize || segment.inputSize != layout.hiddenSize)
      throw WeightStoreError("GGUF tensor does not match the layout: " + std::string(label));
    return ops::Projection{outputSize, layout.hiddenSize, ops::BlockWeights{{std::move(segment)}}};
  };
  Qwen4IndexerWeights indexer;
  indexer.query = floatProjection(layout.indexerHeads * layout.indexerHeadDimension, "indexer-query");
  indexer.key = floatProjection(layout.indexerHeadDimension, "indexer-key");
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

// The PLE hash of the multipliers, heads' offsets and sizes and table rows
// a source states, each already a whole number below its limit
// (kMultiplierLimit, kHeadValueLimit): the first pleNgram multipliers the
// hash's, each small enough that a token times it stays below 2^63, so the
// unsigned hash equals the reference's signed one (qwen4PleRows; the
// released multipliers are below 2.4e13 of the 3.7e13 this allows); each
// head's rows, at least one, inside the table, whose rows are 32-bit ids.
Qwen4PleHash checkedPleHash(const Qwen4ExpLayout &layout, std::span<const uint64_t> multipliers,
                            std::span<const uint64_t> offsets, std::span<const uint64_t> sizes, uint64_t tableRows) {
  Qwen4PleHash hash;
  for (uint32_t position = 0; position < layout.pleNgram; ++position) {
    if (multipliers[position] > uint64_t{INT64_MAX} / layout.vocabularySize)
      throw WeightStoreError("a PLE multiplier is too large for the vocabulary");
    hash.multipliers[position] = multipliers[position];
  }
  hash.tableRows = tableRows;
  if (tableRows > UINT32_MAX) throw WeightStoreError("the PLE table has more rows than 32-bit ids reach");
  for (uint32_t head = 0; head < layout.pleHeads(); ++head) {
    hash.headOffsets.push_back(static_cast<uint32_t>(offsets[head]));
    hash.headVocabularies.push_back(static_cast<uint32_t>(sizes[head]));
    if (!sizes[head] || offsets[head] + sizes[head] > tableRows)
      throw WeightStoreError("a PLE head's rows lie outside the PLE table");
  }
  return hash;
}

// The PLE hash the GGUF's metadata states (checkedPleHash): every value of its
// arrays a whole number below its limit, its table per_layer_token_embd; and
// its end-of-sequence token and the image token it may name the layout's.
Qwen4PleHash readPleHash(const GgufFile &file, const Qwen4ExpLayout &layout) {
  const std::string key = std::string(Qwen4ExpLayout::ggufArchitecture) + ".ple.";
  const auto array = [&](const char *name, size_t count, double limit) {
    const std::optional<std::span<const double>> values = file.numericArray(key + name);
    if (!values || values->size() < count) throw GgufError("GGUF PLE metadata is missing: ple." + std::string(name));
    std::vector<uint64_t> result;
    for (const double value : *values) {
      if (!(value >= 0 && value < limit) || value != std::floor(value))
        throw GgufError("GGUF PLE metadata holds an invalid value");
      result.push_back(static_cast<uint64_t>(value));
    }
    return result;
  };
  const uint32_t heads = layout.pleHeads();
  const Qwen4PleHash hash = checkedPleHash(layout, array("layer_multipliers", layout.pleNgram, kMultiplierLimit),
                                           array("head_offsets", heads, kHeadValueLimit),
                                           array("head_vocab_sizes", heads, kHeadValueLimit),
                                           file.require("per_layer_token_embd.weight").rows());
  const std::optional<uint64_t> eos = file.metadata().unsignedValue(key + "eos_token_id");
  const std::optional<uint64_t> image = file.metadata().unsignedValue(key + "image_token_id");
  if (!eos) throw GgufError("GGUF PLE metadata is missing: ple.eos_token_id");
  if (*eos != layout.pleEosToken || (image && *image != layout.pleImageToken))
    throw WeightStoreError("the GGUF's PLE hash does not match the layout");
  return hash;
}

// The PLE hash an MLX checkpoint holds as its ngram embedding's buffers
// (I64 layer_multipliers [pleNgram], ngram_heads_offsets and
// ngram_heads_vocab_sizes [heads]), which mlx-community's equal its GGUF's
// metadata (the 48 GB M5 Pro), checked as a GGUF's (checkedPleHash); its
// table's rows are its shards' (shards.N). Its end-of-sequence token is the
// config's, which the configuration check holds to the layout's
// (model/ModelDescriptor.mm); its image token is the layout's
// (pleImageToken), as no buffer states one and the check reads only the text
// config, which holds none.
Qwen4PleHash readPleHash(const SafetensorsCheckpoint &checkpoint, const Qwen4ExpLayout &layout) {
  const std::string embedding = safetensors::layerPrefix(safetensors::moduleNames(checkpoint), layout.pleLayer) +
                                "ple.ple_embedding.";
  const auto values = [&](const char *name, uint64_t count, double limit) {
    const SourceTensor &tensor = checkpoint.require(embedding + name);
    if (tensor.dtype != "I64" || tensor.shape != std::vector<uint64_t>{count})
      throw WeightStoreError("source tensor type or shape does not match: " + embedding + name);
    std::vector<int64_t> read(count);
    tensor.read(0, std::span(reinterpret_cast<uint8_t *>(read.data()), count * sizeof(int64_t)));
    std::vector<uint64_t> result;
    for (const int64_t value : read) {
      if (value < 0 || double(value) >= limit) throw WeightStoreError("the checkpoint's PLE hash holds an invalid value");
      result.push_back(uint64_t(value));
    }
    return result;
  };
  uint64_t tableRows = 0;
  for (uint32_t shard = 0;; ++shard) {
    const SourceTensor *codes = checkpoint.find(embedding + "ngram_embedding.shards." + std::to_string(shard) + ".weight");
    if (!codes) break;
    tableRows += codes->shape.empty() ? 0 : codes->shape.front();
  }
  return checkedPleHash(layout, values("layer_multipliers", layout.pleNgram, kMultiplierLimit),
                        values("ngram_heads_offsets", layout.pleHeads(), kHeadValueLimit),
                        values("ngram_heads_vocab_sizes", layout.pleHeads(), kHeadValueLimit), tableRows);
}

Qwen4PleHash readPleHash(const GgufTargetLoader &gguf, const Qwen4ExpLayout &layout) {
  return readPleHash(gguf.file(), layout);
}
Qwen4PleHash readPleHash(const SafetensorsTargetLoader &safetensors, const Qwen4ExpLayout &layout) {
  return readPleHash(safetensors.checkpoint(), layout);
}

} // namespace

Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend, Qwen4ExpLayout layout,
                                    const QwenTargetFiles &files) {
  requireQwen4Layout(layout);
  return loadQwenTarget<Qwen4ExpWeights>(
      backend, layout, files, [&](auto &loader, const BlockTargetFormat &format, const MoeExperts &experts) {
        Qwen4ExpWeights result;
        result.layout = layout;
        // The PLE hash first, from the GGUF's metadata or the checkpoint's
        // buffers, before any image loads.
        result.pleHash = readPleHash(loader, layout);
        result.layers.reserve(layout.layers);
        const uint32_t streams = layout.streamWidth();
        for (uint32_t index = 0; index < layout.layers; ++index) {
          const bool fullAttention = layout.isFullAttentionLayer(index);
          WeightFile file = loader.layer(index);
          Qwen4ExpLayerWeights &layer = result.layers.emplace_back();
          layer.mixerHyper = readHyper(file, format, layout, true, "mixer-hyper");
          layer.mixer = readQwenMixer(file, format, layout, fullAttention);
          if (fullAttention) layer.indexer = readIndexer(file, format, layout);
          layer.ffnHyper = readHyper(file, format, layout, true, "ffn-hyper");
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
          WeightFile file = loader.head();
          result.finalHyper = readHyper(file, format, layout, false, "final-hyper");
          result.logitsProjection = format.projection(file, layout.vocabularySize, layout.hiddenSize, "logits");
          result.logitsProjection.destination = ops::FloatOutput::Float32;
          file.finish();
          result.files.push_back(file.record());
        }
        {
          WeightFile file = loader.embedding();
          result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
          file.finish();
          result.files.push_back(file.record());
        }
        if (std::optional<TargetImages::StreamedTable> table = loader.streamTable()) {
          if (table->store->table().rows != result.pleHash.tableRows)
            throw WeightStoreError("the PLE table does not match its hash");
          result.streamedTable = std::move(table->store);
          result.pleFormat = table->format;
          result.files.push_back(std::move(table->record));
        } else {
          WeightFile file = loader.table();
          result.pleTable = readBlockEmbedding(file, static_cast<uint32_t>(result.pleHash.tableRows),
                                               layout.pleHeadDimension, "ple-table");
          file.finish();
          result.files.push_back(file.record());
        }
        return result;
      });
}

} // namespace splash::model
