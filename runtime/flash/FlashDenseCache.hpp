#pragma once

#include "FlashAffineMPP.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash {

inline constexpr const char *kFlashDenseCacheOperandFormat =
    "mlx-affine-f32-reconstruct-rounded-bf16-row-major-v1";
inline constexpr const char *kFlashDenseCacheExecutionSemantics =
    "cached-bf16-weights-whole-k-mpp-f32accum-bf16-v1";
inline constexpr const char *kFlashDenseM64OutSemantics =
    ";dense-bf16-output-roles-n2560-k6144-r512to8192-m64n128-simd8-v1";
[[nodiscard]] bool flashDenseM64OutEnabled();
[[nodiscard]] bool flashDenseM64OutGeometry(std::string_view prefix, uint32_t rows,
                                            uint32_t outputSize, uint32_t inputSize) noexcept;
[[nodiscard]] const char *flashDenseCacheExecutionSemantics();
inline constexpr const char *kFlashDenseCacheHCUpMixSemantics =
    "cached-bf16-hc-up-whole-k-precise-sigmoid-bf16-stream-mean-v1";

// Original BF16 matrices (including routers) can share the whole-K primitive
// without any cache allocation or conversion. Same Shared/bounds/tail contract
// as FlashDenseCache::addProjection. Weight dimensions must be N%64==0/K%32==0.
void addDenseBF16WholeK(metal::MetalBackend &backend, metal::CommandGraph &graph,
                        metal::MetalBuffer input, const FlashTensor &weights,
                        metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                        uint32_t rows, FlashAffineMPPTile tile);

// A derived, immutable cache. SPLASH_FLASH_OPERAND_STORE optionally supplies
// hash-verified readonly Shared mappings with the exact same cached bits. The
// constructor allocates only missing selected matrices, submits their conversion graph, and
// rejects nonfinite reconstruction before publishing an initialized cache.
// Originals, PLE lookup shards, embedding tables and expert matrices are never
// modified or expanded. Admission is the owner's responsibility: plannedBytes
// is a conservative estimate; actualAllocatedBytes is the backend ledger delta.
class FlashDenseCache final {
public:
  FlashDenseCache(metal::MetalBackend &backend, const FlashWeights &weights,
                  std::span<const std::string> prefixes);
  ~FlashDenseCache();
  FlashDenseCache(const FlashDenseCache &) = delete;
  FlashDenseCache &operator=(const FlashDenseCache &) = delete;
  FlashDenseCache(FlashDenseCache &&) noexcept;
  FlashDenseCache &operator=(FlashDenseCache &&) noexcept;

  [[nodiscard]] static std::vector<std::string>
  defaultPrefixes(const FlashWeights &weights,
                  bool includeVocabularyHead = true);
  [[nodiscard]] static uint64_t
  plannedBytes(const FlashWeights &weights,
               std::span<const std::string> prefixes);
  [[nodiscard]] bool contains(std::string_view prefix) const noexcept;
  [[nodiscard]] const FlashTensor &tensor(std::string_view prefix) const;
  [[nodiscard]] const std::string &identitySha256() const;
  [[nodiscard]] const char *executionSemantics() const {
    return flashDenseCacheExecutionSemantics();
  }
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] uint64_t persistedTensorCount() const noexcept;
  [[nodiscard]] uint64_t persistedPayloadBytes() const noexcept;
  [[nodiscard]] const std::string &operandStoreIdentitySha256() const;
  [[nodiscard]] metal::CommandTiming initializationTiming() const noexcept;
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableWeightBuffers() const;
  // Only verified saved-derived operand mappings; excludes runtime conversion
  // buffers and every original model tensor/PLE allocation.
  [[nodiscard]] std::vector<metal::MetalBuffer> persistedWeightBuffers() const;

  // Whole-K BF16 matrix operands and F32 accumulation. Full row tiles use MPP;
  // a final shorter row tail uses the BF16 vector primitive with the same
  // cached operand bits. N128 tiles dispatch a final N64 tile for N%128==64.
  // Input/output are Shared BF16[rows,K]/[rows,N], 1..8192 rows, never padded
  // or aliased. Shared addresses allow complete partial-overlap validation.
  // Root may retain raw vector decoding and select this only for prefill.
  void addProjection(metal::CommandGraph &graph, std::string_view prefix,
                     metal::MetalBuffer input, metal::MetalBuffer output,
                     metal::MetalBuffer diagnostics, uint32_t rows,
                     FlashAffineMPPTile tile) const;

  // Candidate fusion for the inspected HC up matrix BF16[10240,320]. It reuses
  // activatedDown across four stream dots, applies standalone precise sigmoid
  // and the qualified sequential BF16 stream products/mean, and writes mixed
  // directly. No rawUp tensor is materialized. Complete M tiles only; callers
  // use addProjection + addHCMix for an unsupported final row window.
  [[nodiscard]] bool supportsHCUpMix(std::string_view upPrefix, uint32_t rows,
                                     FlashAffineMPPTile tile) const noexcept;
  void addHCUpMix(metal::CommandGraph &graph, std::string_view upPrefix,
                  metal::MetalBuffer activatedDown,
                  metal::MetalBuffer normalizedHyper, metal::MetalBuffer mixed,
                  metal::MetalBuffer diagnostics, uint32_t rows,
                  FlashAffineMPPTile tile) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash
