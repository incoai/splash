#pragma once

#include "FlashMoEBlocked.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace splash::flash {
inline constexpr const char *kFlashInt8ExpertStoreSemantics =
    "saved-selected-i8-row-f32scale-bf16activation-f32accum-whole-k-q4-misses-v1";

// A numerical alternative for large-row prefill only. The immutable signed
// INT8 coefficients are converted offline, checksum verified and mapped
// readonly. One store serves the target trunk and its batched prefills.
// Original Q4 coefficients remain the miss fallback; small-row decode and
// trained MTP weights are not selected by this class.
class FlashInt8ExpertStore final {
public:
  FlashInt8ExpertStore(metal::MetalBackend &backend, const FlashWeights &weights,
                      const std::filesystem::path &directory);
  ~FlashInt8ExpertStore();
  FlashInt8ExpertStore(const FlashInt8ExpertStore &) = delete;
  FlashInt8ExpertStore &operator=(const FlashInt8ExpertStore &) = delete;
  FlashInt8ExpertStore(FlashInt8ExpertStore &&) noexcept;
  FlashInt8ExpertStore &operator=(FlashInt8ExpertStore &&) noexcept;

  // CPU metadata, source geometry and file-size checks only; no payload read,
  // mapping or backend allocation occurs before the caller reserves this.
  [[nodiscard]] static uint64_t plannedBytes(const FlashWeights &weights,
      const std::filesystem::path &directory);
  [[nodiscard]] const std::string &identitySha256() const;
  [[nodiscard]] const std::string &planSha256() const;
  [[nodiscard]] uint64_t mappedBytes() const noexcept;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] std::span<const uint32_t> selectedExpertIDs(uint32_t layer) const;
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableWeightBuffers() const;

  void addGateUp(metal::CommandGraph &graph, uint32_t layer,
      const FlashMoEBlockedScratch &scratch, metal::MetalBuffer diagnostics,
      uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections = 10) const;
  void addDownScatter(metal::CommandGraph &graph, uint32_t layer,
      const FlashMoEBlockedScratch &scratch, metal::MetalBuffer diagnostics,
      uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections = 10) const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::flash
