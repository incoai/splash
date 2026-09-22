#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashGreedyGPU.h"

namespace splash::flash {

inline constexpr const char *kFlashGreedyGPUNumericalPolicy =
    "finite-bf16-integer-rank-lower-token-ties-exact-prefix-v1";

struct FlashGreedyGPUWorkspace final {
  uint32_t rowCapacity = 0;
  uint32_t vocabularyCapacity = 0;
  uint32_t partitionCapacity = 0;
  metal::MetalBuffer partials;
};

[[nodiscard]] FlashGreedyGPUWorkspace allocateGreedyGPUWorkspace(
    metal::MetalBackend &backend, uint32_t rows, uint32_t vocabulary,
    metal::BufferStorage storage = metal::BufferStorage::Private);

// Conservative admission estimate for partials plus a separate compact result
// allocation, each rounded to the native executors' 16 KiB allocation bound.
[[nodiscard]] uint64_t greedyGPUWorkspacePlannedBytes(uint32_t rows,
                                                    uint32_t vocabulary);
// Validate one completed compact record without reading vocabulary storage.
// Nonfinite errors retain the existing CPU greedy exception message.
[[nodiscard]] uint32_t greedyGPUResultToken(
    const FlashGreedyGPURowResult &result, uint32_t vocabulary);

// Appends both reductions to the caller's existing command graph. The result
// records are fully overwritten; no CPU scratch clearing or vocabulary readback
// is required. Nonfinite values anywhere in a row produce an error and invalid
// token, even when another logit is the largest finite value.
void addGreedyGPU(metal::CommandGraph &graph, metal::MetalBuffer logitsBF16,
                  const FlashGreedyGPUWorkspace &workspace,
                  metal::MetalBuffer results, uint32_t rows,
                  uint32_t vocabulary, uint32_t rowStride = 0);

// Uniform real verification windows, lane-major, with at most sixteen target
// rows altogether. Inputs contain the anchor followed by real drafts. Budgets
// contain remaining output tokens per lane. An inactive lane returns zero
// retained rows; active lanes with invalid logits/input IDs/budgets also retain
// zero. Callers must reject errors before restoring/committing model state.
// Exact EOS tokens are 248044 and 248046; EOS wins at the quota boundary.
void addGreedyGPUPrefix(metal::CommandGraph &graph,
                        metal::MetalBuffer logitsBF16,
                        const FlashGreedyGPUWorkspace &workspace,
                        metal::MetalBuffer inputsU32,
                        metal::MetalBuffer remainingU32,
                        metal::MetalBuffer results,
                        uint32_t lanes, uint32_t rowsPerLane,
                        uint32_t vocabulary, uint32_t activeLaneMask,
                        uint32_t rowStride = 0);

} // namespace splash::flash
