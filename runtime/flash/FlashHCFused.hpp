#pragma once

#include "flash/FlashHC.hpp"

namespace splash::flash {
class FlashFloatDenseSmallRowsWorkspace;
inline constexpr const char *kFlashHCUpF32MPPSemantics =
    "hc-up-original-f32coeff-m8n32-simd4-bf16-stream-stages-reduction-alternative-v1";
[[nodiscard]] bool flashHCUpF32MPPEnabled();
[[nodiscard]] bool flashHCUpF32MPPGeometry(std::string_view prefix, uint32_t rows,
                                          uint32_t outputSize, uint32_t inputSize) noexcept;

// Source-verified original F32 coefficients, BF16 activated inputs and all
// original BF16 sigmoid/product/stream-add/mean stages. Reuses the owner's
// existing F32-cache padding workspace; no weight or row plane is allocated.
// F32 reduction order differs from the raw vector control and is explicit.
void addHCFusedUpMixF32Cache(metal::MetalBackend &backend, metal::CommandGraph &graph,
    metal::MetalBuffer normalized, metal::MetalBuffer activatedDown,
    const FlashTensor &originalF32Weights, metal::MetalBuffer mixed,
    metal::MetalBuffer diagnostics, FlashHCGeometry geometry,
    FlashFloatDenseSmallRowsWorkspace &workspace, metal::MetalBuffer rawUpDebug = {});

enum class FlashHCFusedArithmetic : uint32_t {
  ExplicitCoefficients = 0,
  GroupedAffineExperimental = 1,
};

struct FlashHCFusedConfig final {
  FlashHCFusedArithmetic arithmetic = FlashHCFusedArithmetic::ExplicitCoefficients;
  uint32_t simdgroups = 4;
};

inline constexpr const char *kFlashHCFusedExplicitSemantics =
    "native-hc-literal-f32coeff-lane32-bf16-stages-fused-v1";
inline constexpr const char *kFlashHCFusedGroupedSemantics =
    "experimental-hc-group-affine-f32-bf16-stages-fused-v1";

// First candidate targets the checked H2560/S4/R320 HC geometry, rows1..32.
// All affine bits4/5/6/8 and groups32/64/128 retain their original storage.
// The first route requires Shared, addressable views to validate full ranges.
// Both modes preserve BF16 boundaries. Grouped mode changes F32 affine
// arithmetic and requires a separately identified full-model qualification.
[[nodiscard]] bool supportsHCFused(const FlashAffineProjection &down,
                                    const FlashAffineProjection &up,
                                    const FlashAffineProjection *injection,
                                    FlashHCGeometry geometry) noexcept;

// Replaces down projection, its BF16 /S + compiled-fast SiLU, and the separate
// injection projection/gates with one dispatch. normalized is BF16[rows,S,H],
// activatedDown BF16[rows,320], gates BF16[rows,S]. Injection may be absent for
// the final mixer. No outputs may overlap inputs or each other.
void addHCFusedDown(metal::CommandGraph &graph, metal::MetalBuffer normalized,
                    const FlashAffineProjection &down,
                    const FlashAffineProjection *injection,
                    metal::MetalBuffer activatedDown,
                    metal::MetalBuffer injectionWeights,
                    metal::MetalBuffer diagnostics,
                    FlashHCGeometry geometry, FlashHCFusedConfig config = {});

// Each SIMD computes one hidden dimension's S up-projection rows, reusing
// activations. Raw projections round BF16 before unary-precise sigmoid; each
// product and each stream sum rounds BF16. rawUpDebug optionally receives all
// BF16[rows,S,H] raw projections for the numerical oracle; normal execution
// avoids this plane. normalized remains separate from activatedDown/mixed.
void addHCFusedUpMix(metal::CommandGraph &graph,
                     metal::MetalBuffer normalized,
                     metal::MetalBuffer activatedDown,
                     const FlashAffineProjection &up,
                     metal::MetalBuffer mixed,
                     metal::MetalBuffer diagnostics,
                     FlashHCGeometry geometry, FlashHCFusedConfig config = {},
                     metal::MetalBuffer rawUpDebug = {});

// Fuses a branch's BF16 residual injection with the immediately following
// width-grouped HC norm. Updated hyper may equal the input's exact view;
// normalized must be disjoint. Call only when no PLE/other update lies between
// injection and the next norm. The norm's audited convention is explicit.
void addHCFusedInjectNorm(metal::CommandGraph &graph,
                          metal::MetalBuffer hyperInput,
                          metal::MetalBuffer branch,
                          metal::MetalBuffer injectionWeights,
                          const FlashTensor &normWeight,
                          metal::MetalBuffer hyperOutput,
                          metal::MetalBuffer normalized,
                          metal::MetalBuffer diagnostics,
                          FlashHCGeometry geometry, NormConvention convention);

} // namespace splash::flash
