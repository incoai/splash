#pragma once
#include "FlashQSA.hpp"
#include "FlashQSAFast.hpp"

namespace splash::flash {

inline constexpr char kFlashQSAOnlineMPPRoute[] =
    ";qsa-f32-online-grouped-mpp-v4-prefill-begin0-r32to63-scalar-r64to128-p1-other-p4-decode-r1to4-begin1024-p32-r5to12-begin512-p16-r13to16-begin512-p8";
inline constexpr char kFlashQSARowTilesRoute[] =
    ";qsa-temporal-m32-dense-begin512-r32to128-end2048-p4-f32prob-v1";

// Opt-in flag is absent/0/1 and freezes lazily on first use. The dynamic
// identifier retains kFlashQSAOnlineMPPRoute as a prefix for batch validation.
[[nodiscard]] bool qsaOnlineMPPRowTilesEnabled();
[[nodiscard]] const char *qsaOnlineMPPRouteSemantics();
[[nodiscard]] bool qsaOnlineMPPRowTilesGeometry(uint32_t begin, uint32_t rows,
                                               uint32_t partitions) noexcept;

// Explicit numerical alternative: BF16 Q/K matrix operands, F32 MPP dot,
// F32 global softmax with one BF16 probability cast, BF16 P/V matrix operands,
// F32 MPP value reduction, BF16 attention and the qualified BF16 gate. The
// optional F32 probability policy skips the probability cast and uses mixed
// F32/BF16 MPP operands for PV. Discrete
// selection/cache/normalization semantics remain those of FlashQSA.
struct FlashQSAMPPWorkspace final {
  uint32_t maximumRows = 0;
  metal::MetalBuffer scores;        // F32[maxRows,24,2051]
  metal::MetalBuffer probabilities; // BF16[maxRows,24,2051]
};

[[nodiscard]] FlashQSAMPPWorkspace
allocateQSAMPPWorkspace(metal::MetalBackend &backend, uint32_t maximumRows);

// Online MPP can use more token partitions than the scalar attention route to
// supply enough groups for short decode batches. The returned scratch retains
// FlashQSAFast's statistics/numerator ABI; scalar controls may use it at p<=8.
[[nodiscard]] uint64_t
qsaOnlineMPPWorkspacePlannedBytes(uint32_t maximumRows,
                                  uint32_t maximumPartitions = 32);

[[nodiscard]] FlashQSAFastWorkspace
allocateQSAOnlineMPPWorkspace(metal::MetalBackend &backend, uint32_t maximumRows,
                              uint32_t maximumPartitions = 32);

// Measured M5 Ultra policy. Zero selects the qualified scalar F32 p4 route.
// At begin0 large prefill chunks use p1 and medium chunks use scalar attention.
// Later prefill chunks retain p4; short decode batches need more partitions once
// enough cached tokens amortize MPP staging and final partition reduction.
[[nodiscard]] uint32_t
qsaOnlineMPPRoutePartitions(uint32_t begin, uint32_t rows) noexcept;

void addQSAAttentionMPP(metal::CommandGraph &graph,
                        metal::MetalBuffer qProjection,
                        const FlashQSAState &state,
                        const FlashQSAWorkspace &workspace,
                        FlashQSAMPPWorkspace &mppWorkspace,
                        metal::MetalBuffer output,
                        metal::MetalBuffer diagnostics,
                        uint32_t begin, uint32_t rows,
                        uint32_t scoreTile = 64,
                        bool f32Probabilities = false);

// Fused grouped-query online MPP route: QK, stable F32 softmax, and PV share
// one selected tile in threadgroup memory. Existing F32 partition scratch and
// the qualified partition reducer are reused. No global score sheet is used.
void addQSAAttentionOnlineMPP(metal::CommandGraph &graph,
                              metal::MetalBuffer qProjection,
                              const FlashQSAState &state,
                              const FlashQSAWorkspace &workspace,
                              FlashQSAFastWorkspace &fastWorkspace,
                              metal::MetalBuffer output,
                              metal::MetalBuffer diagnostics,
                              uint32_t begin, uint32_t rows,
                              uint32_t partitions = 4);

// Full prepared-input wrapper. The qualified fused prepare/cache/indexer and
// exact chronological selection prefix are copied unchanged, replacing only
// the final attention dispatches with online MPP.
void addQSAOnlineMPP(metal::CommandGraph &graph,
                     const FlashQSAFastInputs &input,
                     FlashQSAState &state, FlashQSAWorkspace &workspace,
                     FlashQSAFastWorkspace &fastWorkspace,
                     uint32_t begin, uint32_t rows,
                     uint32_t partitions = 4,
                     bool fusePreparation = true);

} // namespace splash::flash
