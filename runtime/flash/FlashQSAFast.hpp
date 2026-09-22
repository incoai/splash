#pragma once

#include "FlashQSA.hpp"

namespace splash::flash {

enum class FlashQSAFastMode : uint8_t {
  // Identical BF16 probability/storage boundaries and ascending value-dot
  // order as the qualified QSA primitive; a candidate for exact comparison.
  CanonicalBF16Probabilities,
  // Explicit numerical alternative: probabilities/numerators remain F32
  // within each partition, then a stable F32 reduction precedes BF16 output.
  // This matches the optimized oMLX attention arithmetic family; it requires
  // tolerance AND full-model coherence qualification before adoption.
  PartitionedF32Probabilities,
};

struct FlashQSAFastWorkspace final {
  uint32_t maximumRows = 0;
  uint32_t maximumPartitions = 0;
  metal::MetalBuffer partitionStatistics; // F32[maxRows,24,maxParts,2]: max,sum
  metal::MetalBuffer partitionValues; // F32[maxRows,24,maxParts,256]: numerator
};

[[nodiscard]] FlashQSAFastWorkspace
allocateQSAFastWorkspace(metal::MetalBackend &backend, uint32_t maximumRows,
                         uint32_t maximumPartitions = 8);

// Attention-only candidate on prepared qualified queries/caches/selection.
// Does not change discrete selection, caches, or normalization. Canonical
// uses one fused dispatch, partitioned F32 uses two. Both include the existing
// precise staged BF16 sigmoid and typed BF16 product.
// Supported partitions are1,2,4,8, with appropriately sized threadgroup state.
void addQSAAttentionFast(metal::CommandGraph &graph,
                         metal::MetalBuffer qProjection,
                         const FlashQSAState &state,
                         const FlashQSAWorkspace &workspace,
                         FlashQSAFastWorkspace &fastWorkspace,
                         metal::MetalBuffer output,
                         metal::MetalBuffer diagnostics, uint32_t begin,
                         uint32_t rows, FlashQSAFastMode mode,
                         uint32_t partitions = 4);

// Same full projection/cache contract as addQSA, packaged to keep experimental
// call sites clear. Norm+RoPE of Q/K/indexQ and auxiliary append are fused;
// qualified pooling, FP32 indexer scores and top512 selection are copied into
// the SAME CommandGraph and retain their original pipeline/parameter contracts.
struct FlashQSAFastInputs final {
  metal::MetalBuffer qProjection, kProjection, vProjection, indexProjection;
  const FlashTensor *qNorm = nullptr;
  const FlashTensor *kNorm = nullptr;
  const FlashTensor *indexQNorm = nullptr;
  const FlashTensor *indexKNorm = nullptr;
  metal::MetalBuffer output, diagnostics, positions;
  NormConvention qConvention = NormConvention::OnePlusWeight;
  NormConvention kConvention = NormConvention::OnePlusWeight;
  NormConvention indexQConvention = NormConvention::OnePlusWeight;
  NormConvention indexKConvention = NormConvention::OnePlusWeight;
  double epsilon = 1e-6, theta = 10000000.0;
};

void addQSAFast(metal::CommandGraph &graph, const FlashQSAFastInputs &input,
                FlashQSAState &state, FlashQSAWorkspace &workspace,
                FlashQSAFastWorkspace &fastWorkspace, uint32_t begin,
                uint32_t rows, FlashQSAFastMode mode,
                uint32_t partitions = 4, bool fusePreparation = true);

} // namespace splash::flash
