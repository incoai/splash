#pragma once
#include "flash/FlashQSAMPP.hpp"

namespace splash::flash {
// Dedicated bounded workspace: callers must reserve plannedBytes before
// construction before allocating this optional route.
struct FlashQSABulkPreparedWorkspace final {
  uint32_t maximumRows = 0;
  metal::MetalBuffer queries;
  metal::MetalBuffer indexQueries;
  metal::MetalBuffer selectedBlocks;
};
[[nodiscard]] uint64_t qsaBulkPreparedWorkspacePlannedBytes(uint32_t maximumRows);
[[nodiscard]] FlashQSABulkPreparedWorkspace allocateQSABulkPreparedWorkspace(
    metal::MetalBackend &backend, uint32_t maximumRows);
[[nodiscard]] bool qsaBulkPreparedGeometry(uint32_t begin, uint32_t rows,
                                         uint32_t capacity) noexcept;
// Bulk preparation/pooling/dense chronological selection only. Attention
// retains the existing per128-row route, partition policy and arithmetic.
void addQSABulkPrepared(metal::MetalBackend &backend,
    metal::CommandGraph &graph, const FlashQSAFastInputs &input,
    FlashQSAState &state, FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &fast, FlashQSABulkPreparedWorkspace &bulk,
    uint32_t begin, uint32_t rows);
// Actual current route, with ordinary128-row scratch reused after attention.
void addQSAChronologicalChunks(metal::MetalBackend &backend,
    metal::CommandGraph &graph, const FlashQSAFastInputs &input,
    FlashQSAState &state, FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &fast, uint32_t begin, uint32_t rows);
void addQSAChronologicalChunk(metal::CommandGraph &graph,
    const FlashQSAFastInputs &input, FlashQSAState &state,
    FlashQSAWorkspace &ordinary, FlashQSAFastWorkspace &fast,
    uint32_t begin, uint32_t rows);
[[nodiscard]] FlashQSAFastInputs sliceQSABulkInputs(
    metal::MetalBackend &backend, const FlashQSAFastInputs &input,
    uint32_t offset, uint32_t rows);
} // namespace splash::flash
