#pragma once

#include "flash/FlashWeights.hpp"
#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::flash {

// Contiguous HC rows are [rows, streams, width], and mixed/branch rows are
// [rows, width]. Geometry comes from the checked model descriptor. The first
// route preserves the source's small, sequential stream reduction (1..8).
struct FlashHCGeometry final {
  uint32_t rows = 0;
  uint32_t width = 0;
  uint32_t streams = 0;
  float epsilon = 1e-6f;
};

// BF16[rows,width] -> BF16[rows,streams,width]. Inputs and outputs must not
// overlap. These pointwise operators never pad logical rows or dimensions.
void addHCExpand(metal::CommandGraph &graph, metal::MetalBuffer input,
                 metal::MetalBuffer output, FlashHCGeometry geometry);

// Normalize each width-sized stream in F32, apply raw F32(1+w) or direct gamma
// according to the loader's audit, then cast to BF16. normWeight is contiguous
// BF16 or F32[streams*width]. Exact input/output aliasing is supported.
void addHCGroupedNorm(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const FlashTensor &normWeight,
                      metal::MetalBuffer output, FlashHCGeometry geometry,
                      NormConvention convention);

// rawUp is the BF16 up-projection result [rows,streams,width]. Sigmoid returns
// BF16; each product, each successive stream sum, and the final /streams are
// BF16 operations. There is no division before this sigmoid. All output views
// must be disjoint from the inputs; partial overlaps are never supported.
void addHCMix(metal::CommandGraph &graph, metal::MetalBuffer normalized,
              metal::MetalBuffer rawUp, metal::MetalBuffer mixed,
              FlashHCGeometry geometry);

// Also forms BF16[rows,streams] injection weights in this same dispatch:
// BF16(2 * sigmoid(BF16(rawInjection / streams))). The sigmoid itself returns
// BF16. rawInjection is the separate BF16 projection output [rows,streams].
void addHCMixWithInjection(metal::CommandGraph &graph,
                           metal::MetalBuffer normalized,
                           metal::MetalBuffer rawUp,
                           metal::MetalBuffer rawInjection,
                           metal::MetalBuffer mixed,
                           metal::MetalBuffer injectionWeights,
                           FlashHCGeometry geometry);

// injectionWeights is already BF16[rows,streams]. The branch product is BF16
// before adding the original hyper state and rounding that add to BF16.
// output may equal hyperInput's exact view. Other overlaps are unsupported.
void addHCInject(metal::CommandGraph &graph, metal::MetalBuffer hyperInput,
                 metal::MetalBuffer branch,
                 metal::MetalBuffer injectionWeights,
                 metal::MetalBuffer output, FlashHCGeometry geometry);

} // namespace splash::flash
