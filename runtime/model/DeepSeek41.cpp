#include "model/DeepSeek41.hpp"

#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/WeightSource.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <string>

namespace splash::model {
namespace {

// A sublayer's mix, in the image's order (GgufImage.cpp mixImage): its
// projection's F32 rows, its scale and its base.
DeepSeek41MixWeights readMix(WeightFile &file, const DeepSeek41Layout &layout, const std::string &label) {
  DeepSeek41MixWeights mix;
  mix.fn = file.section(uint64_t{MHC_MIXES} * MHC_STREAMS * layout.hiddenSize * kFloat32Bytes, label + "-fn");
  mix.scale = file.section(3 * kFloat32Bytes, label + "-scale");
  mix.base = file.section(MHC_MIXES * kFloat32Bytes, label + "-base");
  return mix;
}

// A float projection (BF16 or F32 as the GGUF stores it) that writes fp32,
// as the compressor, the indexer and their scores read it.
ops::Projection floatProjection(WeightFile &file, const BlockTargetFormat &format, uint32_t outputSize,
                                uint32_t inputSize, std::string_view label) {
  ops::Projection projection = format.projection(file, outputSize, inputSize, label);
  projection.destination = ops::FloatOutput::Float32;
  return projection;
}

} // namespace

DeepSeek41LayerWeights readDeepSeek41Layer(WeightFile &file, const BlockTargetFormat &format,
                                           const DeepSeek41Layout &layout, const MoeExperts &experts, uint32_t index) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint32_t hidden = layout.hiddenSize, latent = layout.attentionHeadDimension;
  const auto has = [&](uint64_t layers) { return DeepSeek41Dimensions::has(layers, index); };
  DeepSeek41LayerWeights layer;
  if (has(d.engramLayers)) {
    DeepSeek41EngramWeights engram;
    engram.kv = format.projection(file, (MHC_STREAMS + 1) * hidden, d.engramWidth(), "engram-kv");
    engram.gate = file.section(uint64_t{MHC_STREAMS} * hidden * kFloat32Bytes, "engram-gate");
    layer.engram = std::move(engram);
  }
  layer.attentionMix = readMix(file, layout, "attention-mix");
  layer.attentionNorm = format.norm(file, hidden, "attention-norm");
  layer.queryA = format.projection(file, d.queryRank, hidden, "query-a");
  layer.queryNorm = format.norm(file, d.queryRank, "query-norm");
  layer.queryB = format.projection(file, layout.attentionWidth, d.queryRank, "query-b");
  layer.kv = format.projection(file, latent, hidden, "kv");
  layer.kvNorm = format.norm(file, latent, "kv-norm");
  layer.sinks = file.section(uint64_t{layout.attentionQueryHeads} * kFloat32Bytes, "attention-sinks");
  if (has(d.compressorLayers)) {
    DeepSeek41CompressorWeights compressor;
    compressor.kv = floatProjection(file, format, latent, hidden, "compressor-kv");
    if (has(d.pairedLayers)) compressor.gate = floatProjection(file, format, latent, hidden, "compressor-gate");
    compressor.norm = format.norm(file, latent, "compressor-norm");
    compressor.indexKey = floatProjection(file, format, d.indexerHeadDimension, latent, "index-key");
    compressor.indexKeyNorm = format.norm(file, d.indexerHeadDimension, "index-key-norm");
    layer.compressor = std::move(compressor);
  }
  if (has(d.indexerLayers)) {
    DeepSeek41IndexerWeights indexer;
    indexer.query = floatProjection(file, format, d.indexerHeads * d.indexerHeadDimension, d.queryRank,
                                    "indexer-query");
    indexer.weights = floatProjection(file, format, d.indexerHeads, hidden, "indexer-weights");
    layer.indexer = std::move(indexer);
  }
  const uint32_t groupWidth = d.groupWidth(layout.attentionWidth);
  for (uint32_t group = 0; group < d.outputGroups; ++group)
    layer.outputA.push_back(format.projection(file, d.outputRank, groupWidth, "output-a"));
  layer.outputB = format.projection(file, hidden, d.outputGroups * d.outputRank, "output-b");
  layer.ffnMix = readMix(file, layout, "ffn-mix");
  layer.ffnNorm = format.norm(file, hidden, "ffn-norm");
  layer.ffn = readMoeBlock(file, layout, experts, index);
  return layer;
}

std::vector<float> deepSeek41RopeFrequencies(const DeepSeek41Layout &layout, bool compressed) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint32_t dim = 2 * layout.rotaryPairs;
  const float base = compressed ? d.compressRopeTheta : layout.rotaryTheta;
  std::vector<float> freqs(dim / 2);
  for (uint32_t i = 0; i < dim / 2; ++i) freqs[i] = 1.0F / std::pow(base, float(2 * i) / float(dim));
  if (!compressed) return freqs;
  const double original = d.yarnOriginalContextTokens, factor = d.yarnFactor;
  const auto corrected = [&](double rotations) {
    return dim * std::log(original / (rotations * 2 * M_PI)) / (2 * std::log(double(base)));
  };
  const double low = std::max(std::floor(corrected(d.yarnBetaFast)), 0.0);
  const double high = std::min(std::ceil(corrected(d.yarnBetaSlow)), double(dim - 1));
  for (uint32_t i = 0; i < dim / 2; ++i) {
    const float ramp = float(std::clamp((double(i) - low) / std::max(high - low, 1e-3), 0.0, 1.0));
    const float smooth = 1.0F - ramp;
    freqs[i] = freqs[i] / float(factor) * (1.0F - smooth) + freqs[i] * smooth;
  }
  return freqs;
}


void requireDeepSeek41Layout(const DeepSeek41Layout &layout) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint64_t layers = layout.layers >= 64 ? ~uint64_t{0} : (uint64_t{1} << layout.layers) - 1;
  const auto among = [](uint64_t bits, uint64_t within) { return !(bits & ~within); };
  // The first layer that attends compressed groups reads its own compressor's
  // and indexer's.
  const uint64_t first = d.compressedLayers & -d.compressedLayers;
  if (!d.present() || !layout.layers || !layout.hiddenSize || !layout.vocabularySize ||
      layout.attentionHeadDimension != LATENT_ATTENTION_DIM || layout.attentionKvHeads != 1 ||
      !layout.attentionQueryHeads || layout.attentionQueryHeads > LATENT_ATTENTION_MAX_HEADS ||
      layout.attentionWidth != layout.attentionQueryHeads * layout.attentionHeadDimension ||
      2 * layout.rotaryPairs > layout.attentionHeadDimension || !layout.rotaryPairs ||
      !d.outputRank || !d.outputGroups || layout.attentionWidth % d.outputGroups || !d.queryRank ||
      d.queryRank > LATENT_ROTATION_MAX_WIDTH || layout.hiddenSize > MHC_MAX_HIDDEN ||
      d.indexerHeadDimension != LATENT_INDEX_DIM || !d.indexerHeads || d.indexerHeads > LATENT_INDEX_MAX_HEADS ||
      !d.indexerTopGroups || d.engramNgram < 2 || !d.engramHeads || !d.engramHeadDimension ||
      !d.sinkhornIterations || layout.ffnKind != QwenFfnKind::SparseMoe ||
      layout.moeRouter != ops::MoeRouter::SqrtSoftplusNormalized || layout.sharedExpertSlices != 1 ||
      layout.sharedExpertGate || layout.leadingDenseLayers || !layout.experts || !layout.expertsPerToken ||
      layout.expertsPerToken > layout.experts || !layout.expertIntermediateSize ||
      !among(d.compressedLayers, layers) || !among(d.pairedLayers, d.compressedLayers) ||
      !among(d.compressorLayers, d.compressedLayers) || !among(d.indexerLayers, d.compressedLayers) ||
      !among(d.engramLayers, layers) || (first && (!(d.compressorLayers & first) || !(d.indexerLayers & first))))
    throw WeightStoreError("DeepSeek-V4.1 layout is inconsistent");
}

DeepSeek41EngramTables deepSeek41EngramTables(const GgufFile &target, const DeepSeek41Layout &layout) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  DeepSeek41EngramTables result{.table = {.name = "engram", .rowBytes = d.engramHeadDimension / 32 * 18},
                                .format = GGUF_FMT_IQ4NL};
  const GgufTensor &table = target.require("per_layer_token_embd.weight");
  if (table.type != ggml::kIQ4_NL || table.columns() != d.engramHeadDimension ||
      table.bytes != table.rows() * result.table.rowBytes)
    throw WeightStoreError("the Engram's tables are no IQ4_NL rows of the Engram's head width");
  result.file = table.file;
  result.table.offset = target.source(table.file).dataOffset() + table.offset;
  result.table.rows = table.rows();
  result.table.validate();
  return result;
}

uint64_t deepSeek41EngramHashBytes(const DeepSeek41Layout &layout) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint64_t engrams = std::popcount(d.engramLayers), hashes = uint64_t{d.engramNgram - 1} * d.engramHeads;
  return uint64_t{layout.vocabularySize} * sizeof(int32_t) + engrams * (d.engramNgram + 2 * hashes) * sizeof(int64_t);
}

void readDeepSeek41EngramHash(metal::MetalBackend &backend, const GgufFile &target, uint64_t tableRows,
                              DeepSeek41Weights &weights) {
  // Whole numbers below 2^53, which the metadata's doubles hold exactly.
  constexpr double kWholeLimit = 9007199254740992.0;
  const DeepSeek41Dimensions &d = weights.layout.deepseek;
  const uint32_t engrams = static_cast<uint32_t>(std::popcount(d.engramLayers));
  const uint64_t hashes = uint64_t{d.engramNgram - 1} * d.engramHeads;
  const auto array = [&](const char *key, uint64_t count, double limit) {
    const std::optional<std::span<const double>> values = target.numericArray("deepseek41." + std::string(key));
    if (!values || values->size() != count) throw GgufError("GGUF Engram metadata is missing: " + std::string(key));
    std::vector<uint64_t> whole;
    whole.reserve(count);
    for (const double value : *values) {
      if (!(value >= 0 && value < limit) || value != std::floor(value))
        throw GgufError("GGUF Engram metadata holds an invalid value: " + std::string(key));
      whole.push_back(static_cast<uint64_t>(value));
    }
    return whole;
  };
  const std::vector<uint64_t> tokens = array("engram.token_map", weights.layout.vocabularySize, INT32_MAX);
  const std::vector<uint64_t> multipliers = array("ple.layer_multipliers", engrams * d.engramNgram, kWholeLimit);
  const std::vector<uint64_t> primes = array("ple.head_vocab_sizes", engrams * hashes, double(tableRows) + 1);
  const std::vector<uint64_t> offsets = array("ple.head_offsets", engrams * hashes, double(tableRows));
  std::vector<uint64_t> relative;
  weights.engramFirstRows.clear();
  for (uint32_t table = 0; table < engrams; ++table) {
    // The offsets count from the tables' first row; the hash counts each
    // Engram layer's from its own, where its first head's rows start.
    const uint64_t first = offsets[table * hashes];
    weights.engramFirstRows.push_back(static_cast<uint32_t>(first));
    for (uint64_t head = table * hashes; head < (table + 1) * hashes; ++head) {
      if (offsets[head] < first || !primes[head] || offsets[head] + primes[head] > tableRows)
        throw WeightStoreError("an Engram head's rows lie outside the Engram's tables");
      relative.push_back(offsets[head] - first);
    }
  }
  // Each in a buffer of the type the GGUF's tensors stored it in before.
  const auto upload = [&](const std::vector<uint64_t> &values, auto type, const char *label) {
    using T = decltype(type);
    metal::MetalBuffer buffer =
        backend.allocateBuffer(std::max<uint64_t>(values.size() * sizeof(T), 16), metal::BufferStorage::Shared, label);
    T *out = static_cast<T *>(buffer.contents());
    for (size_t i = 0; i < values.size(); ++i) out[i] = static_cast<T>(values[i]);
    return buffer;
  };
  weights.engramTokenMap = upload(tokens, int32_t{}, "deepseek41/engram-token-map");
  weights.engramMultipliers = upload(multipliers, int64_t{}, "deepseek41/engram-multipliers");
  weights.engramPrimes = upload(primes, int64_t{}, "deepseek41/engram-primes");
  weights.engramOffsets = upload(relative, int64_t{}, "deepseek41/engram-offsets");
}

EngramHashTables DeepSeek41Weights::engramHash() const noexcept {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint64_t engrams = std::popcount(d.engramLayers), hashes = uint64_t{d.engramNgram - 1} * d.engramHeads;
  const auto values = [](const metal::MetalBuffer &buffer, uint64_t count, auto *type) {
    using T = std::remove_pointer_t<decltype(type)>;
    return std::span(static_cast<const T *>(buffer.contents()), count);
  };
  return {.tokenMap = values(engramTokenMap, layout.vocabularySize, static_cast<int32_t *>(nullptr)),
          .multipliers = values(engramMultipliers, engrams * d.engramNgram, static_cast<int64_t *>(nullptr)),
          .primes = values(engramPrimes, engrams * hashes, static_cast<int64_t *>(nullptr)),
          .offsets = values(engramOffsets, engrams * hashes, static_cast<int64_t *>(nullptr)),
          .layers = static_cast<uint32_t>(std::popcount(d.engramLayers)),
          .ngram = d.engramNgram,
          .heads = d.engramHeads,
          .padToken = d.engramPadToken};
}

DeepSeek41Weights loadDeepSeek41Weights(metal::MetalBackend &backend, DeepSeek41Layout layout,
                                        const QwenTargetFiles &files) {
  requireDeepSeek41Layout(layout);
  const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files);
  if (!gguf) throw WeightStoreError(std::string(layout.family) + " loads from a GGUF only");
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  const MoeExperts experts = openMoeExperts(backend, files, layout);
  const BlockTargetFormat format;
  GgufTargetLoader &loader = gguf->get();
  DeepSeek41Weights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    WeightFile file = loader.layer(index);
    result.layers.push_back(readDeepSeek41Layer(file, format, layout, experts, index));
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = loader.head();
    result.finalNorm = format.norm(file, layout.hiddenSize, "final-norm");
    // fp32 logits, as every target's (readQwenTargetHead).
    result.logitsProjection = floatProjection(file, format, layout.vocabularySize, layout.hiddenSize, "logits");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = loader.embedding();
    result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }
  result.experts = experts.store;
  // The Engram's hash comes from the metadata; its tables stream from the
  // target's part that holds them through the table's row store, where the
  // loader's options give it memory.
  const DeepSeek41EngramTables tables = deepSeek41EngramTables(loader.file(), layout);
  readDeepSeek41EngramHash(backend, loader.file(), tables.table.rows, result);
  if (const SparseRowStoreOptions &rows = loader.tableOptions(); rows.cacheBytes) {
    const auto source = std::make_shared<WeightSource>(loader.file().source(tables.file).path());
    result.streamedTable = std::make_shared<SparseRowStore>(tables.table, openRowReader(source), rows);
    result.engramFormat = tables.format;
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
