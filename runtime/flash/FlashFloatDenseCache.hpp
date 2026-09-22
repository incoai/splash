#pragma once

#include "FlashAffine.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash {

enum class FlashFloatDenseSmallRowsTile : uint32_t {
  M8N64 = 0, M8N128 = 1, M16N64 = 2, M16N128 = 3,
};
inline constexpr const char *kFlashFloatDenseCacheOperandFormat =
    "original-affine-contractoff-f32-coefficients-row-major-v1";
inline constexpr const char *kFlashFloatDenseSmallRowsSemantics =
    "zero-padded-bf16-input-f32-weight-whole-k-mpp-f32accum-bf16-v1";
inline constexpr const char *kFlashFloatDenseSmallRowsPolicySemantics =
    "m5-ultra-source-qualified-f32cache-r4-r8-r16-padding-floors-v1";

// Opt-in, source-qualified QSA output tile. It changes cooperative reduction
// geometry only; original F32 coefficients, BF16 stages and storage are retained.
inline constexpr const char *kFlashQSAOutF32N32Semantics =
    "qsa-output-original-f32coeff-m8n32-simd4-bf16-bounded-reduction-v1";
[[nodiscard]] bool flashQSAOutF32N32Enabled();
[[nodiscard]] bool flashQSAOutF32N32Geometry(std::string_view prefix, uint32_t rows,
    uint32_t outputSize, uint32_t inputSize, uint32_t bits, uint32_t groupSize) noexcept;

// Source-qualified selection against the active F32-xsum quantized control.
// Unknown roles, geometry, quantization, and rows retain the caller's raw route.
// Intermediate rows use a qualified lower bound only when tile padding stays
// fixed; notably a winning M8 row-8 route does not imply a row-9 route wins.
[[nodiscard]] std::optional<FlashFloatDenseSmallRowsTile>
flashFloatDenseSmallRowsPolicy(std::string_view prefix, uint32_t rows,
                              uint32_t outputSize, uint32_t inputSize,
                              uint32_t bits, uint32_t groupSize) noexcept;

class FlashFloatDenseSmallRowsWorkspace final {
public:
  explicit FlashFloatDenseSmallRowsWorkspace(metal::MetalBackend &backend,
                                             uint32_t maximumInputSize = 32768);
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }
  [[nodiscard]] uint32_t maximumInputSize() const noexcept { return maximumInputSize_; }
  [[nodiscard]] metal::MetalBuffer paddedInput() const { return paddedInput_; }
  [[nodiscard]] bool belongsTo(const metal::MetalBackend &backend) const noexcept {
    return backend_ == &backend;
  }
private:
  metal::MetalBackend *backend_;
  uint32_t maximumInputSize_;
  uint64_t allocatedBytes_ = 0;
  metal::MetalBuffer paddedInput_;
  friend void addFloatDenseSmallRows(metal::MetalBackend &, metal::CommandGraph &,
      metal::MetalBuffer, const FlashTensor &, metal::MetalBuffer,
      metal::MetalBuffer, uint32_t, FlashFloatDenseSmallRowsWorkspace &,
      FlashFloatDenseSmallRowsTile);
};

// Standalone mixed-type primitive, also exposed for direct precision traps.
// F32 weights remain F32 through the declared MPP operand type. Real BF16 input
// words copy exactly into positive-zero padded M8/M16 rows; only real BF16
// output rows are written. Shared buffers permit complete overlap validation.
void addFloatDenseSmallRows(metal::MetalBackend &backend, metal::CommandGraph &graph,
                            metal::MetalBuffer input, const FlashTensor &weights,
                            metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                            uint32_t rows, FlashFloatDenseSmallRowsWorkspace &workspace,
                            FlashFloatDenseSmallRowsTile tile);

// Selected dense cache. Constructor submits one runtime GPU conversion;
// compiling or CPU-only oracle tests never instantiate it. Admission belongs
// to the caller. Original quantized coefficients are never modified or rounded
// to BF16: float(code)*float(stored scale)+float(stored bias), contract off.
class FlashFloatDenseCache final {
public:
  explicit FlashFloatDenseCache(metal::MetalBackend &backend,
                                const FlashWeights &weights,
                                std::string_view prefix = "language_model.lm_head");
  FlashFloatDenseCache(metal::MetalBackend &backend, const FlashWeights &weights,
                       std::span<const std::string> prefixes);
  ~FlashFloatDenseCache();
  FlashFloatDenseCache(const FlashFloatDenseCache &) = delete;
  FlashFloatDenseCache &operator=(const FlashFloatDenseCache &) = delete;
  FlashFloatDenseCache(FlashFloatDenseCache &&) noexcept;
  FlashFloatDenseCache &operator=(FlashFloatDenseCache &&) noexcept;
  [[nodiscard]] static uint64_t plannedBytes(const FlashWeights &weights,
      std::string_view prefix = "language_model.lm_head");
  [[nodiscard]] static uint64_t plannedBytes(const FlashWeights &weights,
                                             std::span<const std::string> prefixes);
  [[nodiscard]] static std::vector<std::string>
  defaultPrefixes(const FlashWeights &weights, bool includeVocabularyHead = true);
  [[nodiscard]] bool contains(std::string_view prefix) const noexcept;
  [[nodiscard]] const FlashTensor &tensor(std::string_view prefix) const;
  [[nodiscard]] const std::vector<std::string> &prefixes() const;
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableWeightBuffers() const;
  [[nodiscard]] std::vector<metal::MetalBuffer> persistedWeightBuffers() const;
  // Legacy accessors require a single selected projection.
  [[nodiscard]] const FlashTensor &tensor() const;
  [[nodiscard]] const std::string &prefix() const;
  [[nodiscard]] const std::string &identitySha256() const;
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;
  [[nodiscard]] uint64_t persistedTensorCount() const noexcept;
  [[nodiscard]] uint64_t persistedPayloadBytes() const noexcept;
  [[nodiscard]] const std::string &operandStoreIdentitySha256() const;
  [[nodiscard]] metal::CommandTiming initializationTiming() const noexcept;
  // CPU graph-construction counters, excluding GPU timing/profiling overhead.
  [[nodiscard]] uint64_t qsaOutF32N32Dispatches() const noexcept;
  [[nodiscard]] uint64_t qsaOutF32N32RealRows() const noexcept;
  void addSmallRows(metal::CommandGraph &graph, metal::MetalBuffer input,
                    metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                    uint32_t rows, FlashFloatDenseSmallRowsWorkspace &workspace,
                    FlashFloatDenseSmallRowsTile tile) const;
  void addSmallRows(metal::CommandGraph &graph, std::string_view prefix,
                    metal::MetalBuffer input, metal::MetalBuffer output,
                    metal::MetalBuffer diagnostics, uint32_t rows,
                    FlashFloatDenseSmallRowsWorkspace &workspace,
                    FlashFloatDenseSmallRowsTile tile) const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash
