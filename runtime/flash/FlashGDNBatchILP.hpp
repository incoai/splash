#pragma once
#include "FlashGDNStaged.hpp"

#include <span>
#include <string>

namespace splash::flash {
inline constexpr const char *kFlashGDNBatchILPRoute =
    ";gdn-batch-prefill-uniform-b2to4-r512plus-v32-t32-sg8-f32-separate-states";
[[nodiscard]] bool flashGDNBatchILPEnabled();
[[nodiscard]] bool flashGDNBatchILPEligible(uint32_t lanes, uint32_t rows) noexcept;
struct FlashGDNBatchILPTile { uint32_t values = 32, time = 32, simds = 8; };
struct FlashGDNBatchILPLane {
  splash::flash::FlashGDNState state;
  uint32_t rows = 0;
};

void validateGDNBatchILPTile(FlashGDNBatchILPTile tile);
[[nodiscard]] std::string gdnBatchILPPipelineName(FlashGDNBatchILPTile tile);
// Pure scalar and missing-live-state preflight; constructs no backend.
[[nodiscard]] uint32_t validateGDNBatchILPGeometry(std::span<const FlashGDNBatchILPLane> lanes,
    uint32_t inputRowsStride, FlashGDNBatchILPTile tile = {}, float epsilon = 1e-6f);

// Batched full GDN graph. Each live state remains a separate writable F32
// allocation. Prepare, direct-gamma output normalization and convolution carry
// keep the current per-lane kernels; only recurrence is merged across slots.
// No state copy, temporal reassociation or precision conversion occurs.
void addGDNBatchILP(splash::metal::MetalBackend &backend,
    splash::metal::CommandGraph &graph, const splash::flash::FlashGDNWeights &weights,
    const splash::flash::FlashGDNBuffers &flatBuffers,
    std::span<const FlashGDNBatchILPLane> lanes, uint32_t inputRowsStride,
    FlashGDNBatchILPTile tile = {}, float epsilon = 1e-6f);

} // namespace splash::flash
