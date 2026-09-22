#pragma once

#include "FlashMoEBlocked.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace splash::flash {

enum class FlashExpertCachePlane : uint32_t { Gate, Up, Down };
inline constexpr const char *kFlashExpertDenseCacheOperandFormat =
    "immutable-hot-expert-q4-f32coef-once-rounded-bf16-v1";
inline constexpr const char *kFlashExpertDenseCacheSemantics =
    "sparse-expert-bf16-whole-k-full-jobs-q4x8-k64-misses-and-tails-v1";

// Isolated prefill candidate. The caller reserves plannedBytes BEFORE
// construction; construction submits one conversion graph. Selection freezes
// for this cache's lifetime and is independent of request tokens/routing.
// Original planes are retained read-only; no request-time conversion occurs.
class FlashExpertDenseCache final {
public:
  FlashExpertDenseCache(metal::MetalBackend &backend, const FlashWeights &weights,
                       std::string expertPrefix, std::span<const uint32_t> selectedIDs);
  ~FlashExpertDenseCache();
  FlashExpertDenseCache(const FlashExpertDenseCache &) = delete;
  FlashExpertDenseCache &operator=(const FlashExpertDenseCache &) = delete;
  FlashExpertDenseCache(FlashExpertDenseCache &&) noexcept;
  FlashExpertDenseCache &operator=(FlashExpertDenseCache &&) noexcept;

  [[nodiscard]] static uint64_t plannedBytes(const FlashWeights &weights,
      std::string_view expertPrefix, std::span<const uint32_t> selectedIDs);
  [[nodiscard]] const std::string &identitySha256() const;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] metal::CommandTiming initializationTiming() const noexcept;
  [[nodiscard]] std::span<const uint32_t> selectedExpertIDs() const;
  [[nodiscard]] const metal::MetalBuffer &expertRanks() const;
  // BF16[hotCount,N,K], sorted IDs determine the compact expert rank.
  [[nodiscard]] const FlashTensor &cachedProjection(FlashExpertCachePlane plane) const;
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableWeightBuffers() const;
  [[nodiscard]] bool canariesIntact() const noexcept;

  // Same stable GPU jobs/scratch as FlashMoEBlocked. Full cached jobs use
  // direct device whole-K BF16 tensors/F32 dots; misses and partial cached
  // jobs retain the qualified Q4x8 K64 producer. Canonical BF16 combine stays
  // with the caller. Whole-K reduction needs separate numerical qualification.
  void addGateUp(metal::CommandGraph &graph, const FlashMoEBlockedScratch &scratch,
                metal::MetalBuffer diagnostics, uint32_t rows,
                FlashMoEBlockedTile tile, uint32_t selections = 10) const;
  void addDownScatter(metal::CommandGraph &graph, const FlashMoEBlockedScratch &scratch,
                     metal::MetalBuffer diagnostics, uint32_t rows,
                     FlashMoEBlockedTile tile, uint32_t selections = 10) const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::flash
