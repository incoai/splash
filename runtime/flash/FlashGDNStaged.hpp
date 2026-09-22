#pragma once

#include "FlashGDN.hpp"

namespace splash::flash {

enum class FlashGDNStageTile : uint8_t {
  Values8Time16,
  Values8Time32,
  Values16Time16,
  Values16Time32,
};

inline constexpr const char *kFlashGDNStagedNumericalPolicy =
    "qwen4-gdn-staged-sequential-f32-simd32-v1";

// Exact qualified recurrence order: each SIMD32 lane owns four adjacent
// key dimensions, every token updates F32 state sequentially, and output
// rounds BF16 at the same boundary. Short time blocks cooperatively stage
// q/k/v and gates; no decay clipping, temporal reassociation, approximation,
// reduced-precision state/intermediate, or prefix-capture semantics.
// Existing input/output/state contracts and caller-cleared diagnostics apply.
void addGDNStagedPrefill(metal::CommandGraph &graph,
                        const FlashGDNWeights &weights,
                        const FlashGDNBuffers &buffers,
                        const FlashGDNState &state, uint32_t rows,
                        uint32_t lanes, FlashGDNStageTile tile,
                        float normEpsilon = 1e-6f);

} // namespace splash::flash
