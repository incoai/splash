#include "flash/FlashPLE.hpp"

#include "flash/FlashHC.hpp"
#include "flash/FlashPLEPostFused.hpp"
#include "metal/abi/FlashPLE.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {
constexpr uint32_t kThreads = 256;
constexpr uint32_t kStateRows = 9;
constexpr uint32_t kTaps = 4;
constexpr uint32_t kDilation = 3;

bool readPostFusionSwitch() {
  const char *value = std::getenv("SPLASH_FLASH_PLE_POST_FUSED");
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_PLE_POST_FUSED must be 0 or 1");
}

uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > UINT64_MAX / b)
    throw std::invalid_argument("Flash PLE byte extent overflows");
  return a * b;
}

uint64_t add(uint64_t a, uint64_t b) {
  if (a > UINT64_MAX - b)
    throw std::invalid_argument("Flash PLE row extent overflows");
  return a + b;
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!bytes || !buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash PLE insufficient ") + name);
}

void requireTensor(const FlashTensor &tensor, FlashDType dtype,
                   std::span<const uint64_t> shape, const char *name) {
  if (tensor.dtype != dtype || tensor.shape.size() != shape.size())
    throw std::invalid_argument(std::string("Flash PLE invalid ") + name);
  uint64_t elements = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (tensor.shape[i] != shape[i] || !shape[i])
      throw std::invalid_argument(std::string("Flash PLE wrong shape: ") + name);
    elements = multiply(elements, shape[i]);
  }
  const uint64_t elementBytes = dtype == FlashDType::I64 ? 8 :
                               dtype == FlashDType::BF16 ? 2 : 4;
  const uint64_t bytes = multiply(elements, elementBytes);
  if (tensor.logicalBytes < bytes)
    throw std::invalid_argument(std::string("Flash PLE truncated ") + name);
  requireBytes(tensor.buffer, bytes, name);
}

uint32_t rowCount(FlashPLEGeometry g) {
  if (!g.lanes || !g.rows || !g.width || !g.streams || g.streams > 8 ||
      !g.vocabularySize || g.eosToken >= g.vocabularySize ||
      !std::isfinite(g.epsilon) || g.epsilon <= 0 ||
      multiply(g.lanes, g.rows) > UINT32_MAX ||
      multiply(g.width, g.streams) > UINT32_MAX)
    throw std::invalid_argument("Flash PLE invalid geometry");
  return static_cast<uint32_t>(multiply(g.lanes, g.rows));
}

uint64_t hyperBytes(FlashPLEGeometry g) {
  return multiply(multiply(rowCount(g), multiply(g.width, g.streams)), 2);
}

void requireHashWeights(const FlashPLEWeights &w) {
  requireTensor(w.multipliers, FlashDType::I64, std::array<uint64_t, 1>{3},
                "stored multipliers");
  requireTensor(w.headVocabularySizes, FlashDType::I64,
                std::array<uint64_t, 1>{kFlashPLEHeads}, "stored head sizes");
  requireTensor(w.headOffsets, FlashDType::I64,
                std::array<uint64_t, 1>{kFlashPLEHeads}, "stored head offsets");
}

void requireShards(const FlashPLEWeights &w) {
  if (w.shards.empty() || w.shards.size() > 128 ||
      w.tableRows() > INT64_MAX)
    throw std::invalid_argument("Flash PLE invalid source shard count");
  const auto &first = w.shards.front();
  if (!first.rows || first.weightRowStrideBytes < kFlashPLEHeadWidth / 2 ||
      first.parameterRowStrideBytes < (kFlashPLEHeadWidth / 32) * 2 ||
      first.parameterRowStrideBytes % 2)
    throw std::invalid_argument("Flash PLE invalid Q4/G32 shard geometry");
  for (const auto &shard : w.shards) {
    if (shard.rows != first.rows ||
        shard.weightRowStrideBytes != first.weightRowStrideBytes ||
        shard.parameterRowStrideBytes != first.parameterRowStrideBytes)
      throw std::invalid_argument("Flash PLE source shards must be homogeneous");
    requireBytes(shard.weights, multiply(shard.rows,
                                        shard.weightRowStrideBytes),
                 "source packed rows");
    requireBytes(shard.scales, multiply(shard.rows,
                                       shard.parameterRowStrideBytes),
                 "source scales");
    requireBytes(shard.biases, multiply(shard.rows,
                                       shard.parameterRowStrideBytes),
                 "source biases");
  }
  requireTensor(w.sharedScale, FlashDType::BF16,
                std::array<uint64_t, 1>{1}, "shared weight_scale");
}

FlashPLEPostParams postParams(FlashPLEGeometry g, bool mask) {
  return {g.lanes, g.rows, g.width, g.streams, mask ? 1u : 0u,
          kStateRows, kTaps, kDilation};
}

uint32_t gateThreads(uint32_t width) {
  // Match the installed MLX row-reduction partition. The <=64 route folds
  // the row in one logical thread; extra threads only write broadcast values.
  if (width <= 64) return kThreads;
  if (width <= 512) return 32;
  if (width <= 1024) return 128;
  return static_cast<uint32_t>(std::min<uint64_t>(
      1024, ((uint64_t(width) + 3) / 4 + 31) / 32 * 32));
}

bool tokenValid(int64_t token, FlashPLEGeometry g) {
  return token >= 0 && uint64_t(token) < g.vocabularySize;
}
} // namespace

bool flashPLEPostFusedEnabled() {
  static const bool enabled = readPostFusionSwitch();
  return enabled;
}

const char *flashPLEPostRouteSemantics() {
  return flashPLEPostFusedEnabled() ? kFlashPLEPostFusedSemantics :
      "source-seven-dispatch-ple-post-inject-bf16-v1";
}

uint64_t FlashPLEWeights::tableRows() const {
  if (diskTableRows) {
    if (!shards.empty())
      throw std::invalid_argument("Flash PLE disk and GPU table routes are mutually exclusive");
    return diskTableRows;
  }
  uint64_t rows = 0;
  for (const auto &shard : shards)
    rows = add(rows, shard.rows);
  return rows;
}

FlashPLEWeights FlashPLEWeights::fromWeights(const FlashWeights &weights,
                                             std::string_view prefix) {
  const auto &d = weights.descriptor();
  if (d.pleNgramSize != 3 || d.pleHeadsPerNgram != 8 ||
      d.pleHeadDimension() != kFlashPLEHeadWidth || d.pleParts != 128 ||
      d.pleConvolutionTaps != kTaps)
    throw std::invalid_argument("Flash PLE unsupported checkpoint geometry");
  const std::string root(prefix);
  FlashPLEWeights result;
  result.multipliers = weights.tensor(root + ".ple_embedding.layer_multipliers");
  result.headVocabularySizes =
      weights.tensor(root + ".ple_embedding.ngram_heads_vocab_sizes");
  result.headOffsets = weights.tensor(root + ".ple_embedding.ngram_heads_offsets");
  result.sharedScale =
      weights.tensor(root + ".ple_embedding.ngram_embedding.weight_scale");
  result.normKey = weights.tensor(root + ".norm_key.weight");
  result.normQuery = weights.tensor(root + ".norm_query.weight");
  result.normConvolution = weights.tensor(root + ".norm_conv.weight");
  result.convolution = weights.tensor(root + ".conv1d.weight");
  result.normConvention = weights.normConvention(root + ".norm_key.weight");
  if (weights.normConvention(root + ".norm_query.weight") !=
          result.normConvention ||
      weights.normConvention(root + ".norm_conv.weight") !=
          result.normConvention)
    throw std::invalid_argument("Flash PLE inconsistent norm audit");
  if (weights.pleSSDStreamingEnabled()) {
    if (!d.pleTableRows || d.pleTableRows > INT64_MAX || !weights.pleSSDStore())
      throw std::invalid_argument("Flash PLE requires checked disk-only Q4/G32 tables");
    result.diskTableRows = d.pleTableRows;
  } else {
    result.shards.reserve(d.pleParts);
    for (uint32_t index = 0; index < d.pleParts; ++index) {
      const auto &p = weights.projection(
          root + ".ple_embedding.ngram_embedding.shards." +
          std::to_string(index));
      if (p.experts != 1 || p.bits != 4 || p.groupSize != 32 ||
          p.inputSize != kFlashPLEHeadWidth || p.outputSize != 2500012 ||
          !p.weights || !p.scales || !p.biases)
        throw std::invalid_argument("Flash PLE requires original Q4/G32 tables");
      if (p.weights->dtype != FlashDType::U32 ||
          p.scales->dtype != FlashDType::BF16 ||
          p.biases->dtype != FlashDType::BF16)
        throw std::invalid_argument("Flash PLE invalid source table dtypes");
      result.shards.push_back({p.weights->buffer, p.scales->buffer,
                              p.biases->buffer, p.outputSize,
                              p.weightRowStrideBytes,
                              p.parameterRowStrideBytes});
    }
  }
  requireHashWeights(result);
  if (result.diskTableRows) {
    requireTensor(result.sharedScale, FlashDType::BF16,
                  std::array<uint64_t, 1>{1}, "shared weight_scale");
  } else requireShards(result);
  const std::array<uint64_t, 1> normShape{d.hyperHiddenSize()};
  requireTensor(result.normKey, FlashDType::BF16, normShape, "key norm");
  requireTensor(result.normQuery, FlashDType::BF16, normShape, "query norm");
  requireTensor(result.normConvolution, FlashDType::BF16, normShape, "conv norm");
  requireTensor(result.convolution, FlashDType::BF16,
                std::array<uint64_t, 3>{d.hyperHiddenSize(), kTaps, 1},
                "depthwise convolution");
  return result;
}

void addPLENgramIDs(metal::CommandGraph &graph, const FlashPLEWeights &w,
                   metal::MetalBuffer tokenIDs, metal::MetalBuffer history,
                   metal::MetalBuffer ids, metal::MetalBuffer diagnostics,
                   FlashPLEGeometry g) {
  const auto rows = rowCount(g);
  requireHashWeights(w);
  if (multiply(rows, kFlashPLEHeads) > UINT32_MAX)
    throw std::invalid_argument("Flash PLE hash grid exceeds native index width");
  const uint64_t tableRows = w.tableRows();
  if (!tableRows || tableRows > INT64_MAX)
    throw std::invalid_argument("Flash PLE invalid table row extent");
  requireBytes(tokenIDs, multiply(rows, 8), "input token IDs");
  requireBytes(history, multiply(g.lanes, 16), "token history");
  requireBytes(ids, multiply(multiply(rows, kFlashPLEHeads), 8), "ngram IDs");
  requireBytes(diagnostics, 4, "diagnostics");
  if (tokenIDs.sameView(history) || tokenIDs.sameView(ids) || history.sameView(ids))
    throw std::invalid_argument("Flash PLE hash views must be distinct");
  const FlashPLEHashParams params{g.lanes, g.rows, 8, g.eosToken,
                                  g.vocabularySize, 0, tableRows};
  graph.add("flash_ple_hash", {tokenIDs, history, w.multipliers.buffer,
                              w.headVocabularySizes.buffer, w.headOffsets.buffer,
                              ids, diagnostics}, params,
            {(multiply(rows, kFlashPLEHeads) - 1) / kThreads + 1, 1, 1},
            {kThreads, 1, 1});
  graph.add("flash_ple_update_history", {tokenIDs, history, diagnostics}, params,
            {(g.lanes - 1) / kThreads + 1, 1, 1}, {kThreads, 1, 1});
}

void addPLEGather(metal::CommandGraph &graph, const FlashPLEWeights &w,
                  metal::MetalBuffer ids, metal::MetalBuffer output,
                  metal::MetalBuffer diagnostics, FlashPLEGeometry g) {
  const auto rows = rowCount(g);
  requireShards(w);
  requireBytes(ids, multiply(multiply(rows, kFlashPLEHeads), 8), "ngram IDs");
  requireBytes(output, multiply(multiply(rows, kFlashPLEHeads * kFlashPLEHeadWidth),
                                2), "sparse embedding output");
  requireBytes(diagnostics, 4, "diagnostics");
  if (ids.sameView(output))
    throw std::invalid_argument("Flash PLE gather views must be distinct");
  const auto &first = w.shards.front();
  const uint64_t tableRows = w.tableRows();
  for (size_t begin = 0; begin < w.shards.size(); begin += 8) {
    const auto count = static_cast<uint32_t>(
        std::min<size_t>(8, w.shards.size() - begin));
    std::vector<metal::MetalBuffer> buffers{ids, w.sharedScale.buffer, output,
                                           diagnostics};
    buffers.reserve(28);
    for (size_t slot = 0; slot < 8; ++slot) {
      const auto &shard = w.shards[begin + std::min<size_t>(slot, count - 1)];
      buffers.push_back(shard.weights);
      buffers.push_back(shard.scales);
      buffers.push_back(shard.biases);
    }
    const FlashPLEGatherParams params{rows, kFlashPLEHeads, kFlashPLEHeadWidth,
                                      count, multiply(begin, first.rows),
                                      first.rows, first.weightRowStrideBytes,
                                      first.parameterRowStrideBytes,
                                      tableRows};
    graph.add("flash_ple_gather8", std::move(buffers), params,
              {1, rows, kFlashPLEHeads}, {kThreads, 1, 1});
  }
}

void addPLEPostProject(metal::CommandGraph &graph, const FlashPLEWeights &w,
                       metal::MetalBuffer hyperInput,
                       metal::MetalBuffer keys, metal::MetalBuffer values,
                       const FlashPLEPostScratch &scratch,
                       metal::MetalBuffer state, metal::MetalBuffer output,
                       metal::MetalBuffer diagnostics, FlashPLEGeometry g,
                       metal::MetalBuffer mask) {
  const auto rows = rowCount(g);
  const auto bytes = hyperBytes(g);
  requireBytes(hyperInput, bytes, "hyper input");
  requireBytes(keys, bytes, "projected keys");
  requireBytes(values, multiply(multiply(rows, g.width), 2), "projected values");
  const std::array<metal::MetalBuffer, 4> scratchViews{
      scratch.normalizedKeys, scratch.normalizedQueries, scratch.gatedValues,
      scratch.normalizedConvolution};
  for (size_t i = 0; i < scratchViews.size(); ++i) {
    requireBytes(scratchViews[i], bytes, "post scratch");
    if (scratchViews[i].sameView(hyperInput) || scratchViews[i].sameView(keys) ||
        scratchViews[i].sameView(values) || scratchViews[i].sameView(state) ||
        scratchViews[i].sameView(output))
      throw std::invalid_argument("Flash PLE scratch overlaps another input/output");
    for (size_t j = 0; j < i; ++j)
      if (scratchViews[i].sameView(scratchViews[j]))
        throw std::invalid_argument("Flash PLE scratch views must be distinct");
  }
  requireBytes(state, multiply(multiply(multiply(g.lanes, kStateRows),
                                       multiply(g.width, g.streams)), 2),
               "convolution state");
  requireBytes(output, bytes, "PLE output");
  requireBytes(diagnostics, 4, "diagnostics");
  if (state.sameView(output) || hyperInput.sameView(output))
    throw std::invalid_argument("Flash PLE output/state views must be distinct");
  if (state.sameView(hyperInput) || state.sameView(keys) || state.sameView(values))
    throw std::invalid_argument("Flash PLE state overlaps a projected input");
  const bool useMask = static_cast<bool>(mask);
  if (useMask)
    requireBytes(mask, multiply(rows, 4), "mask");
  const std::array<uint64_t, 1> normShape{multiply(g.width, g.streams)};
  const std::array<const FlashTensor *, 3> normWeights{
      &w.normKey, &w.normQuery, &w.normConvolution};
  for (const auto *norm : normWeights) {
    if (norm->dtype != FlashDType::BF16 && norm->dtype != FlashDType::F32)
      throw std::invalid_argument("Flash PLE invalid grouped norm dtype");
    requireTensor(*norm, norm->dtype, normShape, "grouped norm");
  }
  if (w.normConvention != NormConvention::OnePlusWeight &&
      w.normConvention != NormConvention::DirectGamma)
    throw std::invalid_argument("Flash PLE invalid norm convention");
  requireTensor(w.convolution, FlashDType::BF16,
                std::array<uint64_t, 3>{multiply(g.width, g.streams), kTaps, 1},
                "depthwise convolution");
  // Every tensor remains the checkpoint's raw gamma or raw zero-centered
  // weight. HC grouped norm already preserves F32(1+w) formation.
  const FlashHCGeometry normGeometry{rows, g.width, g.streams, g.epsilon};
  addHCGroupedNorm(graph, keys, w.normKey, scratch.normalizedKeys,
                    normGeometry, w.normConvention);
  addHCGroupedNorm(graph, hyperInput, w.normQuery, scratch.normalizedQueries,
                    normGeometry, w.normConvention);
  const auto params = postParams(g, useMask);
  graph.add("flash_ple_gate", {scratch.normalizedKeys,
                              scratch.normalizedQueries, values,
                              useMask ? mask : diagnostics,
                              scratch.gatedValues, diagnostics}, params,
            {rows, g.streams, 1}, {gateThreads(g.width), 1, 1});
  addHCGroupedNorm(graph, scratch.gatedValues, w.normConvolution,
                    scratch.normalizedConvolution, normGeometry,
                    w.normConvention);
  graph.add("flash_ple_convolution", {scratch.normalizedConvolution,
                                      scratch.gatedValues, state,
                                      w.convolution.buffer, output,
                                      diagnostics}, params,
            {(multiply(g.width, g.streams) - 1) / kThreads + 1, rows, 1},
            {kThreads, 1, 1});
  graph.add("flash_ple_update_convolution_state",
            {scratch.normalizedConvolution, state}, params,
            {(multiply(g.width, g.streams) - 1) / kThreads + 1, g.lanes, 1},
            {kThreads, 1, 1});
}

void addPLEInject(metal::CommandGraph &graph, metal::MetalBuffer hyperInput,
                  metal::MetalBuffer pleOutput, metal::MetalBuffer output,
                  FlashPLEGeometry g) {
  const auto rows = rowCount(g);
  const auto bytes = hyperBytes(g);
  requireBytes(hyperInput, bytes, "inject hyper input");
  requireBytes(pleOutput, bytes, "inject PLE output");
  requireBytes(output, bytes, "inject output");
  if (pleOutput.sameView(output))
    throw std::invalid_argument("Flash PLE injection may not overwrite PLE output");
  graph.add("flash_ple_inject", {hyperInput, pleOutput, output},
            postParams(g, false),
            {(multiply(g.width, g.streams) - 1) / kThreads + 1, rows, 1},
            {kThreads, 1, 1});
}

void addPLEPostProjectAndInject(
    metal::CommandGraph &graph, const FlashPLEWeights &weights,
    metal::MetalBuffer hyperInput, metal::MetalBuffer keyProjected,
    metal::MetalBuffer valueProjected, const FlashPLEPostScratch &scratch,
    metal::MetalBuffer state, metal::MetalBuffer pleOutput,
    metal::MetalBuffer diagnostics, FlashPLEGeometry geometry,
    metal::MetalBuffer mask) {
  if (flashPLEPostFusedEnabled()) {
    addPLEPostProjectFused(graph, weights, hyperInput, keyProjected,
                          valueProjected, scratch, state, pleOutput, hyperInput,
                          diagnostics, geometry, mask);
  } else {
    addPLEPostProject(graph, weights, hyperInput, keyProjected, valueProjected,
                      scratch, state, pleOutput, diagnostics, geometry, mask);
    addPLEInject(graph, hyperInput, pleOutput, hyperInput, geometry);
  }
}

void addPLERestorePrefix(metal::CommandGraph &graph,
                         metal::MetalBuffer beforeHistory,
                         metal::MetalBuffer tokens,
                         metal::MetalBuffer beforeState,
                         metal::MetalBuffer normalized,
                         metal::MetalBuffer keptTokens,
                         metal::MetalBuffer outputHistory,
                         metal::MetalBuffer outputState,
                         metal::MetalBuffer diagnostics, FlashPLEGeometry g) {
  const uint64_t rows = rowCount(g);
  const uint64_t stateBytes = multiply(
      multiply(multiply(g.lanes, kStateRows), multiply(g.width, g.streams)), 2);
  const uint64_t historyBytes = multiply(g.lanes, 16);
  requireBytes(beforeHistory, historyBytes, "prefix history snapshot");
  requireBytes(tokens, multiply(rows, 8), "prefix token IDs");
  requireBytes(beforeState, stateBytes, "prefix convolution snapshot");
  requireBytes(normalized, hyperBytes(g), "prefix normalized inputs");
  requireBytes(keptTokens, multiply(g.lanes, 4), "prefix kept-token counts");
  requireBytes(outputHistory, historyBytes, "restored history");
  requireBytes(outputState, stateBytes, "restored convolution state");
  requireBytes(diagnostics, 4, "diagnostics");
  const std::array<metal::MetalBuffer, 5> inputs{
      beforeHistory, tokens, beforeState, normalized, keptTokens};
  for (const auto &input : inputs)
    if (input.sameView(outputHistory) || input.sameView(outputState))
      throw std::invalid_argument("Flash PLE prefix output overlaps its snapshot/input");
  if (outputHistory.sameView(outputState))
    throw std::invalid_argument("Flash PLE prefix outputs must be distinct");
  const FlashPLEPrefixParams params{postParams(g, false), g.vocabularySize,
                                    0, 0, 0};
  graph.add("flash_ple_restore_prefix",
            {beforeHistory, tokens, beforeState, normalized, keptTokens,
             outputHistory, outputState, diagnostics}, params,
            {(multiply(g.width, g.streams) - 1) / kThreads + 1, g.lanes, 1},
            {kThreads, 1, 1});
}

std::vector<int64_t> computePLENgramIDs(
    std::span<const int64_t> tokens, std::span<int64_t> history,
    std::span<const int64_t> multipliers, std::span<const int64_t> sizes,
    std::span<const int64_t> offsets, FlashPLEGeometry g, uint64_t tableRows) {
  const auto rows = rowCount(g);
  if (tokens.size() != rows || history.size() != multiply(g.lanes, 2) ||
      multipliers.size() != 3 || sizes.size() != kFlashPLEHeads ||
      offsets.size() != kFlashPLEHeads || !tableRows || tableRows > INT64_MAX)
    throw std::invalid_argument("Flash PLE CPU hash invalid input extents");
  for (const auto token : tokens)
    if (!tokenValid(token, g))
      throw std::invalid_argument("Flash PLE CPU hash invalid token");
  for (const auto token : history)
    if (!tokenValid(token, g))
      throw std::invalid_argument("Flash PLE CPU hash invalid history token");
  for (size_t head = 0; head < kFlashPLEHeads; ++head)
    if (sizes[head] <= 0 || offsets[head] < 0 ||
        uint64_t(offsets[head]) >= tableRows ||
        uint64_t(sizes[head]) > tableRows - uint64_t(offsets[head]))
      throw std::invalid_argument("Flash PLE CPU hash invalid checkpoint arrays");
  std::vector<int64_t> ids(multiply(rows, kFlashPLEHeads));
  for (uint32_t lane = 0; lane < g.lanes; ++lane) {
    int64_t older = history[uint64_t(lane) * 2];
    int64_t previous = history[uint64_t(lane) * 2 + 1];
    for (uint32_t row = 0; row < g.rows; ++row) {
      const uint64_t flat = uint64_t(lane) * g.rows + row;
      const int64_t current = tokens[flat];
      const int64_t previous2 = previous == g.eosToken ? g.eosToken : older;
      uint64_t mixed = uint64_t(current) * uint64_t(multipliers[0]);
      mixed ^= uint64_t(previous) * uint64_t(multipliers[1]);
      for (uint32_t head = 0; head < kFlashPLEHeads; ++head) {
        const uint64_t bits = head < 8 ? mixed :
            mixed ^ (uint64_t(previous2) * uint64_t(multipliers[2]));
        const int64_t signedMixed = std::bit_cast<int64_t>(bits);
        int64_t remainder = signedMixed % sizes[head];
        if (remainder < 0)
          remainder += sizes[head];
        ids[flat * kFlashPLEHeads + head] = remainder + offsets[head];
      }
      older = previous;
      previous = current;
    }
    history[uint64_t(lane) * 2] = older;
    history[uint64_t(lane) * 2 + 1] = previous;
  }
  return ids;
}

} // namespace splash::flash
