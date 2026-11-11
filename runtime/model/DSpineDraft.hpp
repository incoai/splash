#pragma once

#include "model/DFlashDraft.hpp"

namespace splash::model {

// DSpine's backbone and bridge, beside DFlash2. The controller shares target
// capture, context K/V, and target-side verification with both architectures.
class DSpineDraft final {
public:
  static void addDecode(metal::CommandGraph &graph, const DFlashDraftWeights &weights,
                        const ops::ExecutionPlans &operators, DFlashDecodeBuffers buffers,
                        const ops::Projection &head, std::span<const uint32_t> cacheLengths);
};

} // namespace splash::model
