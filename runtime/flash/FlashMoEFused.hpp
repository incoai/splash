#pragma once

#include "flash/FlashWeights.hpp"
#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::flash {

// Offline candidates only. The forward graph does not select any of these
// until exact checkpoint qualification and balanced timing identify a winner.
// Original unsigned LE Q4/G64 expert rows are read directly. Each projection
// keeps the qualified lane K order/F32 affine coefficients and BF16 dot cast;
// only then does fast-exp, BF16-staged compiled SwiGLU run in the same kernel.
// Input BF16[rows,2560], IDs I64[rows,selections], output
// BF16[rows,selections,640]. No gate/up planes or quantized activations are
// materialized. Diagnostics is a caller-cleared/checkable uint32 sticky word
// with bits 1 invalid ID, 2 malformed shape, 4 nonfinite arithmetic.
// An invalid ID ORs both 1 and 4, matching the complete separate chain.
void addFusedExpertGateUp(metal::CommandGraph &graph, metal::MetalBuffer input,
                          const FlashAffineProjection &gate,
                          const FlashAffineProjection &up,
                          metal::MetalBuffer expertIDs,
                          metal::MetalBuffer output,
                          metal::MetalBuffer diagnostics, uint32_t rows,
                          uint32_t selections = 10,
                          uint32_t columnsPerSimd = 1,
                          bool packedWordLoads = true,
                          uint32_t simdGroupsPerThreadgroup = 8);

// Down projection plus BF16 route-weight multiplication. Input is already
// gathered BF16[rows,selections,640]; output is BF16 weighted terms
// [rows,selections,2560]. The final reduction uses addWeightedExpertCombine.
void addFusedExpertDownTerms(metal::CommandGraph &graph,
                             metal::MetalBuffer input,
                             const FlashAffineProjection &down,
                             metal::MetalBuffer expertIDs,
                             metal::MetalBuffer routeWeights,
                             metal::MetalBuffer output,
                             metal::MetalBuffer diagnostics, uint32_t rows,
                             uint32_t selections = 10,
                             uint32_t simdGroupsPerThreadgroup = 8);

// Complete down projection, BF16 route weighting, source col8 BF16 reduction,
// and precise BF16 shared sigmoid/product/add in one kernel. This avoids the
// expert-down scratch plane. Its ten-expert loop preserves exact dot order;
// only balanced timing can determine whether lower launch count wins.
void addFusedExpertDownCombine(metal::CommandGraph &graph,
                               metal::MetalBuffer input,
                               const FlashAffineProjection &down,
                               metal::MetalBuffer expertIDs,
                               metal::MetalBuffer routeWeights,
                               metal::MetalBuffer sharedDown,
                               metal::MetalBuffer sharedGate,
                               metal::MetalBuffer output,
                               metal::MetalBuffer diagnostics, uint32_t rows,
                               uint32_t selections = 10,
                               uint32_t simdGroupsPerThreadgroup = 8);

// Reduce preweighted terms from addFusedExpertDownTerms with the same fixed
// BF16 col8 order and precise shared-expert gate as the qualified MoE v3.
void addWeightedExpertCombine(metal::CommandGraph &graph,
                              metal::MetalBuffer terms,
                              metal::MetalBuffer expertIDs,
                              metal::MetalBuffer sharedDown,
                              metal::MetalBuffer sharedGate,
                              metal::MetalBuffer output,
                              metal::MetalBuffer diagnostics, uint32_t rows,
                              uint32_t selections = 10);

} // namespace splash::flash
