#pragma once

#include "flash/FlashForward.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace splash::flash {

inline constexpr const char *kFlashBatchForwardExecution =
    "native-flash-real-batch-one-token-per-request-v1";

struct FlashBatchResult final {
  metal::CommandTiming timing;
  // Borrowed contiguous BF16[lanes,vocabularySize] and [lanes,10240]. Each
  // caller extracts its own lane before the next batch call reuses scratch.
  metal::MetalBuffer logitsBF16;
  metal::MetalBuffer hiddenBF16;
  std::vector<uint64_t> logicalLengths;
  uint32_t lanes = 0;
  uint32_t capacity = 0;
  metal::MetalBuffer greedyResultsU32{};
  uint32_t greedyRows = 0;
};

// Real shared-weight decode: one incoming token for each of1..4 existing
// request states, in one48-layer command graph. Projections/HC/router/MoE/head
// execute rows=lanes. QSA retains each request's separate cache and position;
// GDN/PLE state is packed/scattered exactly, with no new request allocations.
class FlashBatchForward final {
public:
  FlashBatchForward(metal::MetalBackend &backend, const FlashWeights &weights,
                    FlashForward &trunk, uint32_t capacity,
                    uint32_t maximumLanes = 4);
  ~FlashBatchForward();
  FlashBatchForward(const FlashBatchForward &) = delete;
  FlashBatchForward &operator=(const FlashBatchForward &) = delete;
  FlashBatchForward(FlashBatchForward &&) noexcept;
  FlashBatchForward &operator=(FlashBatchForward &&) noexcept;

  [[nodiscard]] FlashBatchResult
  forwardBatch(std::span<FlashRequestState *const> states,
               std::span<const uint32_t> tokens);
  [[nodiscard]] static uint64_t
  workspacePlannedBytes(uint32_t capacity, uint32_t maximumLanes = 4);
  [[nodiscard]] uint64_t workspaceBytes() const noexcept;
  [[nodiscard]] uint64_t pleSSDStagingBytes() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash
