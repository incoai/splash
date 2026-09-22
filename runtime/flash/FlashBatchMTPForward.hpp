#pragma once

#include "flash/FlashMTP.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace splash::flash {

inline constexpr const char *kFlashBatchMTPSemantics =
    "native-lightning-head-compact-real-lane-spans-independent-qsa-v1";

struct FlashBatchMTPResult final {
  metal::CommandTiming timing;
  // Last: one BF16 vocabulary row per lane. All: one per real input row.
  // None: absent. Borrowed until the next batch head call.
  metal::MetalBuffer logitsBF16;
  uint32_t logitRows = 0;
  uint32_t lanes = 0;
  // All real premixer features BF16[sum(counts),10240], compact lane-major.
  metal::MetalBuffer hiddenBF16;
  std::vector<uint32_t> laneOffsets; // count+1 offsets, final offset is rows.
  std::vector<uint64_t> logicalLengths;
  // Optional exact GPU argmax, one FlashGreedyGPURowResult per logit row in
  // the same order as logitsBF16. Borrowed until the next batch head call.
  metal::MetalBuffer greedyResultsU32 = {};
  uint32_t greedyRows = 0;
};

// Joint trained head graph for B1..4 and 1..128 REAL folded pairs per lane.
// Ownership/caches stay with the sequential head. It may be used for creating
// states and truncating speculative histories between batch calls. Keep that
// owner alive and unmoved for the batch object's lifetime. No fake padding
// token or recurrent update is introduced; every lane uses its own QSA offset.
// None supports joint priming; Last returns only B vocabulary rows. All is
// bounded to at most16 total real rows to keep vocabulary storage constant.
class FlashBatchMTPForward final {
public:
  explicit FlashBatchMTPForward(FlashMTPForward &owner,
      uint32_t maximumLanes = 4, uint32_t maximumRowsPerLane = 4,
      const FlashTensor *cachedVocabulary = nullptr);
  ~FlashBatchMTPForward();
  FlashBatchMTPForward(const FlashBatchMTPForward &) = delete;
  FlashBatchMTPForward &operator=(const FlashBatchMTPForward &) = delete;
  FlashBatchMTPForward(FlashBatchMTPForward &&) noexcept;
  FlashBatchMTPForward &operator=(FlashBatchMTPForward &&) noexcept;

  [[nodiscard]] FlashBatchMTPResult forward(
      std::span<FlashMTPState *const> states,
      metal::MetalBuffer previousHiddenBF16,
      std::span<const uint32_t> compactNextTokens,
      std::span<const uint32_t> laneCounts,
      FlashMTPLogits logits = FlashMTPLogits::Last);
  [[nodiscard]] static uint64_t workspacePlannedBytes(uint32_t capacity,
      uint32_t maximumLanes = 4, uint32_t maximumRowsPerLane = 4,
      bool includeCachedVocabulary = false);
  [[nodiscard]] uint64_t workspaceBytes() const noexcept;
  [[nodiscard]] const char *attentionRouteSemantics() const noexcept;
  [[nodiscard]] const char *projectionRouteSemantics() const noexcept;
  [[nodiscard]] bool vocabularyRegisterEnabled() const noexcept;
  [[nodiscard]] const char *vocabularyRouteSemantics() const noexcept;
  [[nodiscard]] uint64_t vocabularyRegisterCommands() const noexcept;
  [[nodiscard]] uint64_t vocabularyRegisterRows() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::flash
