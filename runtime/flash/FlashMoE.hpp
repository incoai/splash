#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::flash {

// Canonical, fixed BF16 semantics for every row count and phase. Source MLX
// argpartition does not specify selection order/ties; this route orders the
// rounded BF16 probabilities descending, then expert ID ascending. Switching
// semantics requires a different model/cache identity, never a phase switch.
inline constexpr const char *kFlashMoESemantics =
    "mlx-metal-bf16-prob-topk-idtie-scores-serial-col8-swiglu-fast-shared-precise-v3";
inline constexpr uint32_t kFlashMoEInvalidIndex = 1;
inline constexpr uint32_t kFlashMoEInvalidParameters = 2;
inline constexpr uint32_t kFlashMoEInvalidNumerics = 4;
inline constexpr uint32_t kFlashMoEMaxRows = 8192;
inline constexpr uint32_t kFlashMoEMaxExperts = 512;
inline constexpr uint32_t kFlashMoEMaxSelections = 10;
inline constexpr uint32_t kFlashMoEMaxWidth = 2560;

// logits is contiguous BF16[rows, experts]. The outputs are contiguous
// I64[rows, selections] IDs and BF16[rows, selections] weights. The full
// softmax accumulates in F32 and rounds probabilities to BF16 before selecting;
// every selected-score sum addition and normalized score round to BF16. When false,
// normalizeTopK returns the selected probabilities without renormalization.
// The current Qwen4Exp source always uses true. No activation conversion is
// introduced. Nonfinite logits set sticky diagnostics and emit -1/NaN routes.
void addRoute(metal::CommandGraph &graph, metal::MetalBuffer logits,
              metal::MetalBuffer expertIDs, metal::MetalBuffer routeWeights,
              metal::MetalBuffer diagnostics, uint32_t rows,
              uint32_t experts, uint32_t selections,
              bool normalizeTopK = true);

// Contiguous BF16[rows, selections, width]. Compiled SwiGLU sigmoid uses fast
// exp; exp, denominator, reciprocal, and complement each round to BF16. The
// gate*sigmoid and multiplication by up each round to BF16. Shared experts
// pass one selection. Actual MLX Metal compiled-SwiGLU goldens qualify these
// boundaries; the precision route stays fixed across row counts and phases.
void addSiLUMultiply(metal::CommandGraph &graph, metal::MetalBuffer gate,
                     metal::MetalBuffer up, metal::MetalBuffer intermediate,
                     metal::MetalBuffer diagnostics, uint32_t rows,
                     uint32_t width, uint32_t selections = 1);

// expertDown is BF16[rows,selections,width], weights is BF16[rows,selections],
// sharedDown/output is BF16[rows,width], sharedGate is BF16[rows,1]. Each
// weighted expert term rounds to BF16, then eight BF16 partials pair slots
// (0,8), (1,9), and 2..7; the partials merge sequentially in BF16. This matches
// the actual MLX Metal K10/W2560 column reduction and stays fixed for all
// phases and test widths. Shared sigmoid uses precise exp with BF16
// exp/denominator/reciprocal/complement stages, then its product rounds to BF16,
// followed by BF16(routed + shared). An input ID buffer checks all
// selections before any combine read; malformed IDs set sticky diagnostics.
// Every output view must be disjoint from every input view.
void addCombine(metal::CommandGraph &graph, metal::MetalBuffer expertDown,
                metal::MetalBuffer expertIDs, metal::MetalBuffer routeWeights,
                metal::MetalBuffer sharedDown, metal::MetalBuffer sharedGate,
                metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                uint32_t rows, uint32_t width, uint32_t experts,
                uint32_t selections);

} // namespace splash::flash
