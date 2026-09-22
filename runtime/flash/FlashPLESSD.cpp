#include "flash/FlashPLESSD.hpp"

#include "metal/abi/FlashPLESSD.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;

uint64_t rounded(uint64_t bytes) {
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}

void validateGeometry(uint32_t lanes, uint32_t rows) {
  if (!lanes || lanes > 4 || !rows || rows > 2048)
    throw std::invalid_argument("Flash PLE SSD staging requires lanes1..4 and rows1..2048");
}

std::span<const int64_t> hashArray(const FlashTensor &tensor, uint64_t count,
                                 const char *name) {
  if (tensor.dtype != FlashDType::I64 || tensor.shape != std::vector<uint64_t>{count} ||
      tensor.logicalBytes < count * 8 || !tensor.buffer ||
      tensor.buffer.sizeBytes() < count * 8 || !tensor.buffer.contents())
    throw std::invalid_argument(std::string("Flash PLE SSD requires shared stored ") + name);
  return {static_cast<const int64_t *>(tensor.buffer.contents()), static_cast<size_t>(count)};
}
} // namespace

FlashPLESSD::FlashPLESSD(metal::MetalBackend &backend,
                       std::shared_ptr<FlashPLESSDStore> store,
                       const FlashPLEWeights &weights, uint32_t lanes,
                       uint32_t rows)
    : backend_(backend), store_(std::move(store)),
      maximumLanes_(lanes), maximumRows_(rows) {
  validateGeometry(lanes, rows);
  if (!store_ || !weights.diskTableRows || !weights.shards.empty() ||
      weights.diskTableRows != store_->tableRows())
    throw std::invalid_argument("Flash PLE SSD staging requires checked disk-only tables");
  (void)hashArray(weights.multipliers, 3, "multipliers");
  (void)hashArray(weights.headVocabularySizes, 16, "head vocabulary sizes");
  (void)hashArray(weights.headOffsets, 16, "head offsets");
  const uint64_t selections = uint64_t{lanes} * rows * kFlashPLEHeads;
  expectedIDs_ = backend.allocateBuffer(rounded(selections * 8),
      metal::BufferStorage::Shared, "flash-ple-ssd expected canonical IDs");
  rows_ = backend.allocateBuffer(rounded(selections * kFlashPLESSDRowBytes),
      metal::BufferStorage::Shared, "flash-ple-ssd bounded original Q4 rows");
  if (!expectedIDs_.contents() || !rows_.contents())
    throw std::logic_error("Flash PLE SSD staging is not Shared");
}

FlashPLESSD::~FlashPLESSD() = default;

void FlashPLESSD::prepare(std::span<const int64_t> tokens,
                         std::span<const int64_t> histories,
                         const FlashPLEWeights &weights, FlashPLEGeometry geometry) {
  preparedLanes_ = preparedRows_ = 0;
  validateGeometry(geometry.lanes, geometry.rows);
  if (geometry.lanes > maximumLanes_ || geometry.rows > maximumRows_ ||
      tokens.size() != uint64_t{geometry.lanes} * geometry.rows ||
      histories.size() != uint64_t{geometry.lanes} * 2 ||
      weights.diskTableRows != store_->tableRows() || !weights.shards.empty())
    throw std::invalid_argument("Flash PLE SSD preparation exceeds arena or has changed tables");
  std::vector<int64_t> privateHistory(histories.begin(), histories.end());
  const auto ids = computePLENgramIDs(tokens, privateHistory,
      hashArray(weights.multipliers, 3, "multipliers"),
      hashArray(weights.headVocabularySizes, 16, "head vocabulary sizes"),
      hashArray(weights.headOffsets, 16, "head offsets"), geometry, store_->tableRows());
  store_->lookupRows(ids, {static_cast<uint8_t *>(rows_.contents()),
                           ids.size() * kFlashPLESSDRowBytes});
  std::memcpy(expectedIDs_.contents(), ids.data(), ids.size() * sizeof(int64_t));
  preparedLanes_ = geometry.lanes;
  preparedRows_ = geometry.rows;
}

void FlashPLESSD::addHashGather(metal::CommandGraph &graph,
                              const FlashPLEWeights &weights,
                              metal::MetalBuffer tokens, metal::MetalBuffer history,
                              metal::MetalBuffer ids, metal::MetalBuffer output,
                              metal::MetalBuffer diagnostics,
                              FlashPLEGeometry geometry) {
  if (!preparedRows_ || geometry.lanes != preparedLanes_ ||
      geometry.rows != preparedRows_ || weights.tableRows() != store_->tableRows())
    throw std::logic_error("Flash PLE SSD gather has no matching completed preparation");
  const uint32_t flat = geometry.lanes * geometry.rows;
  const uint64_t selections = uint64_t{flat} * kFlashPLEHeads;
  if (!output || output.sizeBytes() < selections * kFlashPLEHeadWidth * 2 ||
      !diagnostics || diagnostics.sizeBytes() < 4 ||
      weights.sharedScale.dtype != FlashDType::BF16 ||
      weights.sharedScale.shape != std::vector<uint64_t>{1} ||
      weights.sharedScale.logicalBytes < 2 || !weights.sharedScale.buffer ||
      weights.sharedScale.buffer.sizeBytes() < 2)
    throw std::invalid_argument("Flash PLE SSD gather has invalid output or diagnostics");
  const std::array<metal::MetalBuffer, 3> inputs{tokens, history, ids};
  for (const auto &input : inputs)
    if (input.sameView(output) || input.sameView(diagnostics))
      throw std::invalid_argument("Flash PLE SSD input and writable views must be distinct");
  if (output.sameView(diagnostics))
    throw std::invalid_argument("Flash PLE SSD writable views must be distinct");
  for (const auto &arena : {expectedIDs_, rows_})
    if (output.sameView(arena) || diagnostics.sameView(arena))
      throw std::invalid_argument("Flash PLE SSD output must not alias its staging");
  // A preparation belongs to one graph; every later graph must fetch its own
  // incoming token/history IDs, including after partial speculative restores.
  preparedLanes_ = preparedRows_ = 0;
  addPLENgramIDs(graph, weights, tokens, history, ids, diagnostics, geometry);
  const FlashPLESSDParams params{flat, kFlashPLEHeads, kFlashPLEHeadWidth,
                                 static_cast<uint32_t>(kFlashPLESSDRowBytes), store_->tableRows()};
  graph.add("flash_ple_ssd_gather", {ids,
      backend_.view(expectedIDs_, 0, selections * 8),
      backend_.view(rows_, 0, selections * kFlashPLESSDRowBytes),
      weights.sharedScale.buffer, output, diagnostics}, params,
      {1, flat, kFlashPLEHeads}, {kFlashPLEHeadWidth, 1, 1});
}

uint64_t FlashPLESSD::allocatedBytes() const noexcept {
  return expectedIDs_.sizeBytes() + rows_.sizeBytes();
}

std::vector<metal::MetalBuffer> FlashPLESSD::scratchBuffers() const {
  return {expectedIDs_, rows_};
}

uint64_t FlashPLESSD::plannedBytes(uint32_t lanes, uint32_t rows) {
  validateGeometry(lanes, rows);
  const uint64_t selections = uint64_t{lanes} * rows * kFlashPLEHeads;
  return rounded(selections * 8) + rounded(selections * kFlashPLESSDRowBytes);
}

} // namespace splash::flash
