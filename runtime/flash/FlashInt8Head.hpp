#pragma once

#include "FlashWeights.hpp"
#include "metal/CommandGraph.hpp"

#include <memory>
#include <string>
#include <vector>

namespace splash::flash {

inline constexpr const char *kFlashInt8HeadOperandFormat =
    "original-q8-unsigned-codes-bf16-scale-bias-g64-v1";
inline constexpr const char *kFlashInt8HeadSemantics =
    "head-only-bf16-uint8-g64-mpp-f32-scale-dot-bias-xsum-bf16-r2to16-v1";

// Read-only original Q8 byte view for the checked Qwen4 vocabulary head; a
// strided source retains a lossless in-memory UINT8 layout copy.
// The group-factored F32 dot is a declared numerical alternative to explicit
// per-coefficient F32 reconstruction. Activations retain their BF16 words.
// Caller reserves plannedBytes before construction; no checkpoint is changed.
class FlashInt8Head final {
public:
  FlashInt8Head(metal::MetalBackend &backend, const FlashWeights &weights,
               metal::MetalBuffer sharedPadding = {});
  ~FlashInt8Head();
  FlashInt8Head(const FlashInt8Head &) = delete;
  FlashInt8Head &operator=(const FlashInt8Head &) = delete;
  FlashInt8Head(FlashInt8Head &&) noexcept;
  FlashInt8Head &operator=(FlashInt8Head &&) noexcept;

  [[nodiscard]] static uint64_t plannedBytes(const FlashWeights &weights,
                                            bool reusePadding = false);
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;
  [[nodiscard]] bool usesOriginalCodeStorage() const noexcept;
  [[nodiscard]] const char *codeStorageSemantics() const noexcept;
  [[nodiscard]] const std::string &identitySha256() const;
  [[nodiscard]] metal::CommandTiming initializationTiming() const noexcept;
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableWeightBuffers() const;
  [[nodiscard]] std::vector<metal::MetalBuffer> scratchBuffers() const;

  // Only real rows2..16. Row1 and all larger windows retain the caller route.
  // Shared BF16 input/output and sticky U32 diagnostics must not overlap the
  // original source, exact code cache, padding, sums, or each other.
  void addProjection(metal::CommandGraph &graph, metal::MetalBuffer input,
                     metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                     uint32_t rows) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash
