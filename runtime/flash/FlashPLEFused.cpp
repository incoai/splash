#include "flash/FlashPLEFused.hpp"

#include "metal/abi/FlashPLE.h"
#include "metal/abi/FlashPLEFused.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {
constexpr uint32_t kSourceShards = 128;
constexpr uint32_t kGatherThreads = 256;

uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > UINT64_MAX / b)
    throw std::invalid_argument("Flash PLE fused byte extent overflows");
  return a * b;
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!bytes || !buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash PLE fused insufficient ") + name);
}

void requireTensor(const FlashTensor &tensor, FlashDType dtype,
                   uint64_t elements, const char *name) {
  const uint64_t elementBytes = dtype == FlashDType::I64 ? 8 : 2;
  if (tensor.dtype != dtype || tensor.shape != std::vector<uint64_t>{elements} ||
      tensor.logicalBytes < multiply(elements, elementBytes))
    throw std::invalid_argument(std::string("Flash PLE fused invalid ") + name);
  requireBytes(tensor.buffer, multiply(elements, elementBytes), name);
}

uint32_t rowCount(FlashPLEGeometry g) {
  if (!g.lanes || !g.rows || !g.width || !g.streams || g.streams > 8 ||
      !g.vocabularySize || g.eosToken >= g.vocabularySize ||
      !std::isfinite(g.epsilon) || g.epsilon <= 0 ||
      multiply(g.lanes, g.rows) > UINT32_MAX ||
      multiply(g.width, g.streams) > UINT32_MAX)
    throw std::invalid_argument("Flash PLE fused invalid geometry");
  return static_cast<uint32_t>(multiply(g.lanes, g.rows));
}

FlashPLEFusedParams params(const FlashPLEWeights &w, FlashPLEGeometry g) {
  const auto &first = w.shards.front();
  return {g.lanes, g.rows, g.eosToken, g.vocabularySize, kSourceShards, 0,
          first.rows, first.weightRowStrideBytes,
          first.parameterRowStrideBytes, w.tableRows()};
}

void requireViews(metal::MetalBuffer ids, metal::MetalBuffer output,
                  metal::MetalBuffer diagnostics, FlashPLEGeometry g) {
  const uint64_t rows = rowCount(g);
  if (multiply(rows, kFlashPLEHeads) > UINT32_MAX)
    throw std::invalid_argument("Flash PLE fused index grid exceeds native width");
  requireBytes(ids, multiply(multiply(rows, kFlashPLEHeads), 8), "ngram IDs");
  requireBytes(output, multiply(multiply(rows,
      kFlashPLEHeads * kFlashPLEHeadWidth), 2), "sparse embedding output");
  requireBytes(diagnostics, 4, "diagnostics");
  if (ids.sameView(output) || ids.sameView(diagnostics) ||
      output.sameView(diagnostics))
    throw std::invalid_argument("Flash PLE fused writable views must be distinct");
}
} // namespace

uint64_t FlashPLEFused::plannedBytes(metal::MetalBackend &backend) {
  if (!backend.supportsArgumentBuffersTier2()) return 0;
  const auto bytes = backend.readOnlyArgumentBufferByteCount("flash_ple_hash_gather128", 0);
  if (bytes > kFlashPLEFusedMaximumArgumentBytes)
    throw std::invalid_argument("Flash PLE fused argument buffer exceeds its admission bound");
  return bytes;
}

FlashPLEFused::FlashPLEFused(metal::MetalBackend &backend, FlashPLEWeights weights)
    : weights_(std::move(weights)) {
  if (!backend.supportsArgumentBuffersTier2()) return;
  // Preflight the reflected byte extent before creating any source metadata.
  static_cast<void>(plannedBytes(backend));
  if (weights_.shards.size() != kSourceShards ||
      !weights_.tableRows() || weights_.tableRows() > INT64_MAX)
    throw std::invalid_argument("Flash PLE fused requires 128 original source shards");
  requireTensor(weights_.multipliers, FlashDType::I64, 3, "stored multipliers");
  requireTensor(weights_.headVocabularySizes, FlashDType::I64, 16, "stored head sizes");
  requireTensor(weights_.headOffsets, FlashDType::I64, 16, "stored head offsets");
  requireTensor(weights_.sharedScale, FlashDType::BF16, 1, "shared weight_scale");
  const auto &first = weights_.shards.front();
  if (!first.rows || first.weightRowStrideBytes < 80 ||
      first.parameterRowStrideBytes < 10 || first.parameterRowStrideBytes % 2)
    throw std::invalid_argument("Flash PLE fused invalid Q4/G32 source strides");
  std::vector<metal::BufferBinding> bindings;
  bindings.reserve(kSourceShards * 3);
  for (uint32_t shard = 0; shard < kSourceShards; ++shard) {
    const auto &s = weights_.shards[shard];
    if (s.rows != first.rows || s.weightRowStrideBytes != first.weightRowStrideBytes ||
        s.parameterRowStrideBytes != first.parameterRowStrideBytes)
      throw std::invalid_argument("Flash PLE fused source shards must be homogeneous");
    requireBytes(s.weights, multiply(s.rows, s.weightRowStrideBytes), "source Q4 rows");
    requireBytes(s.scales, multiply(s.rows, s.parameterRowStrideBytes), "source scales");
    requireBytes(s.biases, multiply(s.rows, s.parameterRowStrideBytes), "source biases");
    bindings.push_back({shard, s.weights});
    bindings.push_back({kSourceShards + shard, s.scales});
    bindings.push_back({kSourceShards * 2 + shard, s.biases});
  }
  sources_ = backend.makeReadOnlyArgumentBuffer("flash_ple_hash_gather128", 0,
      bindings, "Flash PLE immutable original source pointers");
}

bool FlashPLEFused::addGather(metal::CommandGraph &graph,
    metal::MetalBuffer ids, metal::MetalBuffer output,
    metal::MetalBuffer diagnostics, FlashPLEGeometry g) const {
  if (!supported()) return false;
  requireViews(ids, output, diagnostics, g);
  graph.add("flash_ple_gather128",
      {sources_, ids, weights_.sharedScale.buffer, output, diagnostics},
      params(weights_, g), {1, rowCount(g), kFlashPLEHeads},
      {kGatherThreads, 1, 1});
  return true;
}

bool FlashPLEFused::addHashGather(metal::CommandGraph &graph,
    metal::MetalBuffer tokens, metal::MetalBuffer history,
    metal::MetalBuffer ids, metal::MetalBuffer output,
    metal::MetalBuffer diagnostics, FlashPLEGeometry g) const {
  if (!supported()) return false;
  requireViews(ids, output, diagnostics, g);
  const uint64_t rows = rowCount(g);
  requireBytes(tokens, multiply(rows, 8), "input token IDs");
  requireBytes(history, multiply(g.lanes, 16), "token history");
  if (tokens.sameView(history) || tokens.sameView(ids) || tokens.sameView(output) ||
      tokens.sameView(diagnostics) || history.sameView(ids) ||
      history.sameView(output) || history.sameView(diagnostics))
    throw std::invalid_argument("Flash PLE fused input and writable views must be distinct");
  graph.add("flash_ple_hash_gather128",
      {sources_, tokens, history, weights_.multipliers.buffer,
       weights_.headVocabularySizes.buffer, weights_.headOffsets.buffer,
       weights_.sharedScale.buffer, ids, output, diagnostics}, params(weights_, g),
      {1, rows, kFlashPLEHeads}, {kGatherThreads, 1, 1});
  const FlashPLEHashParams hash{g.lanes, g.rows, 8, g.eosToken,
                                g.vocabularySize, 0, weights_.tableRows()};
  graph.add("flash_ple_update_history", {tokens, history, diagnostics}, hash,
      {(g.lanes - 1) / 256 + 1, 1, 1}, {256, 1, 1});
  return true;
}

} // namespace splash::flash
