#pragma once

#include "FlashWeights.hpp"
#include "metal/CommandGraph.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace splash::flash {

inline constexpr const char *kFlashAffineSemantics =
    "mlx-affine-f32coeff-f32accum-bf16-v1";
inline constexpr const char *kFlashAffineFastSemantics =
    "mlx-affine-mixed-f32coeff-qmv-f32xsum-v1";
inline constexpr const char *kFlashAffineExpertSemantics =
    "mlx-affine-f32coeff-expert-contig-gateup-k16-sg4-c2-down-k8-sg2-c4-v1";
inline constexpr const char *kFlashAffineCombinedSemantics =
    "mlx-affine-dense-f32xsum-expert-f32coeff-contig-gateup-k16-sg4-c2-down-k8-sg2-c4-v1";
inline constexpr uint32_t kFlashAffineInvalidIndex = 1;
inline constexpr uint32_t kFlashAffineInvalidParameters = 2;
inline constexpr uint32_t kFlashAffineInvalidNumerics = 4;

// Opt-in profiles, frozen on first validated read per process. Both
// SPLASH_FLASH_QMV_F32 and SPLASH_FLASH_EXPERT_QMV must be absent, "0" or "1";
// other values fail instead of silently changing math.
// Call the semantics getter when constructing model/cache identity so the
// off-default grouped-affine profile cannot reuse the explicit control's KV.
[[nodiscard]] bool flashAffineFastEnabled();
[[nodiscard]] bool flashAffineExpertEnabled();
[[nodiscard]] const char *flashAffineSemantics();

// Correctness-first native primitives for the original mixed MLX affine
// weights. Input and output are contiguous BF16; accumulation is F32. These
// do not quantize activations, repack weights, or materialize expert matrices.
// diagnostics is a caller-cleared uint32 sticky status buffer. The caller must
// check it after completed commands: bit 0 invalid ID, bit 1 invalid parameters,
// bit 2 nonfinite output arithmetic.
void addAffine(metal::CommandGraph &graph, metal::MetalBuffer input,
               const FlashAffineProjection &projection,
               metal::MetalBuffer output, metal::MetalBuffer diagnostics,
               uint32_t rows);

// expertIDs is contiguous I64[rows, selections]. Output is
// BF16[rows, selections, N]. With inputPerSelection=false the input is
// BF16[rows, K]; true accepts BF16[rows, selections, K], used by expert-down.
// Invalid device-written IDs set diagnostics and produce NaN output.
void addGatheredAffine(metal::CommandGraph &graph, metal::MetalBuffer input,
                       const FlashAffineProjection &projection,
                       metal::MetalBuffer expertIDs,
                       metal::MetalBuffer output,
                       metal::MetalBuffer diagnostics, uint32_t rows,
                       uint32_t selections, bool inputPerSelection = false);

// weights must be contiguous BF16[N, K]. The result is BF16[rows, N].
void addDenseBF16(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const FlashTensor &weights, metal::MetalBuffer output,
                  metal::MetalBuffer diagnostics, uint32_t rows);

// Gather original affine embedding rows directly; tokenIDs is I64[rows],
// output is BF16[rows, K]. No vocabulary-wide dequantization is performed.
void addAffineEmbedding(metal::CommandGraph &graph,
                        const FlashAffineProjection &projection,
                        metal::MetalBuffer tokenIDs,
                        metal::MetalBuffer output,
                        metal::MetalBuffer diagnostics, uint32_t rows);

// CPU-only integer reference for independent packing oracles. This extracts
// directly from the byte bitstream, including cross-byte 5/6-bit codes.
[[nodiscard]] uint32_t unpackAffineCode(std::span<const std::byte> packedRow,
                                       uint32_t bits, uint64_t channel);

} // namespace splash::flash
