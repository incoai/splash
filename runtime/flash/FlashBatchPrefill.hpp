#pragma once
#include "flash/FlashForward.hpp"
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::flash {
inline constexpr const char *kFlashBatchPrefillSemantics =
    "native-flash-uniform-real-lane-prefill-shared-source-project-v1";
inline constexpr uint32_t kFlashBatchPrefillMaximumRowsPerLane = 2048;
inline constexpr uint32_t kFlashBatchPrefillMaximumPhysicalRows = 8192;
// CPU-only copy metadata checks. No buffer contents are read. Shared feature
// destinations must cover whole U32 words, with no wrapping address range.
inline uint64_t flashBatchPrefillHiddenCopyBytes(uint32_t lanes, uint32_t rows) {
  if (!lanes || lanes > 4 || !rows || rows > kFlashBatchPrefillMaximumRowsPerLane)
    throw std::invalid_argument("Flash batch prefill feature copy geometry is invalid");
  return uint64_t{lanes} * rows * 10240 * sizeof(uint16_t);
}
inline void validateFlashBatchPrefillCopyRange(const void *address, uint64_t availableBytes,
                                              uint64_t copiedBytes) {
  const auto begin = reinterpret_cast<uintptr_t>(address);
  if (!address || !copiedBytes || copiedBytes % sizeof(uint32_t) ||
      begin % alignof(uint32_t) || availableBytes < copiedBytes ||
      availableBytes > std::numeric_limits<uintptr_t>::max() - begin)
    throw std::invalid_argument("Flash batch prefill feature copy range is invalid");
}
inline bool flashBatchPrefillCopyRangesOverlap(const void *first, uint64_t firstBytes,
                                               const void *second, uint64_t secondBytes) {
  if (!first || !second || !firstBytes || !secondBytes) return false;
  const auto a = reinterpret_cast<uintptr_t>(first), b = reinterpret_cast<uintptr_t>(second);
  if (firstBytes > std::numeric_limits<uintptr_t>::max() - a ||
      secondBytes > std::numeric_limits<uintptr_t>::max() - b)
    throw std::invalid_argument("Flash batch prefill feature copy address extent wraps");
  return a < b + secondBytes && b < a + firstBytes;
}
struct FlashBatchPrefillStatePlane final {
  std::string name;
  metal::MetalBuffer buffer;
  FlashDType dtype = FlashDType::BF16;
  uint64_t activeBytes = 0;
};
struct FlashBatchPrefillResult final {
  metal::CommandTiming timing;
  // Borrowed BF16[lanes,vocabulary], one LAST incoming prediction per lane.
  metal::MetalBuffer logitsBF16;
  // Optional BF16[lanes,rows,10240], ALL real premixer features. Borrowed from
  // this arena unless delivered into the caller's retained destination.
  metal::MetalBuffer hiddenBF16;
  std::vector<uint64_t> logicalLengths;
  uint32_t lanes = 0, rows = 0, capacity = 0;
  metal::MetalBuffer greedyResultsU32{};
  uint32_t greedyRows = 0;
  bool hiddenDeliveredToDestination = false;
};
// One 48-layer graph with 1..4 independent request lanes, uniform real rows
// 1..2048 per lane, total <=8192. Default arena remains 512 rows per lane.
// No padding, prefix snapshots, alternate states,
// persistent cache publication or cross-request token reuse. Source owns shared
// immutable dense/hot-expert operands; this arena owns only temporary workspace.
// All calls/request-state moves/destruction are serialized on the GPU owner.
class FlashBatchPrefill final {
public:
  FlashBatchPrefill(metal::MetalBackend &backend, const FlashWeights &weights,
      FlashForward &trunk, uint32_t capacity, uint32_t maximumLanes = 4,
      uint32_t maximumRowsPerLane = 512);
  ~FlashBatchPrefill();
  FlashBatchPrefill(const FlashBatchPrefill &) = delete;
  FlashBatchPrefill &operator=(const FlashBatchPrefill &) = delete;
  FlashBatchPrefill(FlashBatchPrefill &&) noexcept;
  FlashBatchPrefill &operator=(FlashBatchPrefill &&) noexcept;
  // hiddenDestination is optional retained Shared caller storage, requiring
  // captureHidden. Only its real feature prefix is written by a copy dispatch
  // in the producer graph. Invalid identity/extent/alias fails before mutation.
  [[nodiscard]] FlashBatchPrefillResult forwardBatch(
      std::span<FlashRequestState *const> states,
      std::span<const uint32_t> laneMajorTokens, uint32_t rows,
      bool captureHidden = false, metal::MetalBuffer hiddenDestination = {});
  [[nodiscard]] static uint64_t workspacePlannedBytes(uint32_t capacity,
      uint32_t maximumLanes = 4, uint32_t maximumRowsPerLane = 512);
  [[nodiscard]] uint64_t workspaceBytes() const noexcept;
  [[nodiscard]] uint64_t pleSSDStagingBytes() const noexcept;
  [[nodiscard]] std::string kernelRoutes() const;
  // Development oracle only: borrowed, read-only request storage descriptions.
  // Calls are serialized with the source trunk. Capacity tails are never live.
  [[nodiscard]] std::vector<FlashBatchPrefillStatePlane>
  inspectState(const FlashRequestState &state) const;
  [[nodiscard]] bool canariesIntact() const noexcept;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::flash
