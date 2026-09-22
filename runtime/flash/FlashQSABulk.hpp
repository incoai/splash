#pragma once
#include "FlashQSABulkPrepare.hpp"

namespace splash::flash {

inline constexpr char kFlashQSABulkPrefillRoute[] =
    ";qsa-bulk-prefill-begin0-r2048-w128-m16p1-p4-m32p4-v2";
inline constexpr char kFlashQSABulkPrefillSG8Route[] =
    ";qsa-bulk-prefill-temporal-sg8-w4to15-m32-p4-t256-v1";

// Both flags accept absent/0/1 and default off. Forward freezes the validated
// policy when constructing its workspace. Validation must run before admission.
[[nodiscard]] bool qsaBulkPrefillEnabled();
[[nodiscard]] bool qsaBulkPrefillSG8Enabled(bool bulkEnabled);
[[nodiscard]] bool qsaBulkPrefillGeometry(uint32_t begin, uint32_t rows,
                                         uint32_t capacity, bool verification) noexcept;

struct FlashQSABulkCounters final {
  uint64_t completedPrefillCalls = 0;
  uint64_t completedPrefillTokens = 0;
  uint64_t completedLayerCalls = 0;
  uint64_t completedSG8LayerCalls = 0;
};

struct FlashQSABulkWorkspace final {
  FlashQSABulkPreparedWorkspace prepared;
  FlashQSAFastWorkspace partials;
};
[[nodiscard]] uint64_t qsaBulkWorkspacePlannedBytes();
[[nodiscard]] FlashQSABulkWorkspace allocateQSABulkWorkspace(metal::MetalBackend &backend);
void addQSABulkPrefill(metal::MetalBackend &backend,metal::CommandGraph &graph,
    const FlashQSAFastInputs &input,FlashQSAState &state,FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &ordinaryFast,FlashQSABulkWorkspace &bulk,uint32_t begin,uint32_t rows,bool sg8=false);
} // namespace splash::flash
