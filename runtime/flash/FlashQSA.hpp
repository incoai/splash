#pragma once

#include "FlashWeights.hpp"
#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::flash {

inline constexpr uint32_t kFlashQSABlockBudget = 512;
inline constexpr uint32_t kFlashQSATokenWidth = 2051;
inline constexpr uint32_t kFlashQSAInvalidNumeric = 1u << 8;
inline constexpr uint32_t kFlashQSAInvalidPosition = 1u << 9;

// One unpadded request/layer cache, token-major. begin is supplied by the
// caller, so rollback simply restores its logical length; the next append
// overwrites stale rows and recomputes every newly completed four-token block.
// Physical cache indexing and RoPE positions are deliberately distinct.
struct FlashQSAState final {
  uint32_t capacity = 0;
  metal::MetalBuffer keys;          // BF16[capacity, 2, 256], normalized + RoPE
  metal::MetalBuffer values;        // BF16[capacity, 2, 256]
  metal::MetalBuffer rawIndexKeys;  // BF16[capacity, 128], before pooling/norm
  metal::MetalBuffer pooledKeys;    // BF16[ceil(capacity/4),128], norm + RoPE
  metal::MetalBuffer indexPositions;// I64[capacity], first row of block rotates it
};

// Temporary buffers may be shared between ordered layers in one CommandGraph.
// This stores compact scores/indices, never gathered K/V tensors.
struct FlashQSAWorkspace final {
  uint32_t maximumRows = 0;
  uint32_t capacity = 0;
  metal::MetalBuffer queries;       // BF16[maximumRows,24,256]
  metal::MetalBuffer indexQueries;  // BF16[maximumRows,4,128]
  metal::MetalBuffer blockScores;   // F32[maximumRows,ceil(capacity/4)]
  metal::MetalBuffer selectedBlocks;// U32[maximumRows,512], chronological
  metal::MetalBuffer attentionScores; // F32[maximumRows,24,2051]
  metal::MetalBuffer probabilities; // BF16[maximumRows,24,2051]
};

[[nodiscard]] FlashQSAState allocateQSAState(metal::MetalBackend &backend,
                                            uint32_t capacity);
[[nodiscard]] FlashQSAWorkspace
allocateQSAWorkspace(metal::MetalBackend &backend, uint32_t maximumRows,
                     uint32_t capacity);

// qProjection BF16[rows,24,512] contains [query256,gate256] PER HEAD.
// k/vProjection BF16[rows,2,256]; indexProjection BF16[rows,5,128] contains
// four query heads followed by one raw key. All four norm weights accept
// BF16 or F32[dimension]; each uses its audited convention independently.
// output BF16[rows,6144] includes the sigmoid query gate. Optional
// positions I64[rows] supplies text RoPE positions, default begin + row.
// diagnostics is caller-cleared sticky U32 and must be checked after wait.
// Completed-block selection uses FP32 products, highest-ID cutoff ties, and
// always includes the causal incomplete tail. <=2048 visible tokens use
// complete causal attention; longer rows genuinely select up to512 blocks.
void addQSA(metal::CommandGraph &graph, metal::MetalBuffer qProjection,
            metal::MetalBuffer kProjection, metal::MetalBuffer vProjection,
            metal::MetalBuffer indexProjection, const FlashTensor &qNorm,
            const FlashTensor &kNorm, const FlashTensor &indexQNorm,
            const FlashTensor &indexKNorm, FlashQSAState &state,
            FlashQSAWorkspace &workspace, metal::MetalBuffer output,
            metal::MetalBuffer diagnostics, uint32_t begin, uint32_t rows,
            NormConvention qConvention, NormConvention kConvention,
            NormConvention indexQConvention, NormConvention indexKConvention,
            double epsilon = 1e-6, double theta = 10000000.0,
            metal::MetalBuffer positions = {});

} // namespace splash::flash
