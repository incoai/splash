#pragma once

#include "metal/abi/DSpine.h"
#include "ops/DraftSelector.hpp"
#include "ops/Normalization.hpp"

#include <vector>

namespace splash::ops {

struct DSpineWeights final {
  std::vector<Projection> read;
  std::vector<Projection> write;
  Projection packed;
  metal::MetalBuffer codebook;
  metal::MetalBuffer gateBias;
  metal::MetalBuffer predecessorGate;
  metal::MetalBuffer previous;
};

struct DSpineBuffers final {
  metal::MetalBuffer raw;
  metal::MetalBuffer rms;
  metal::MetalBuffer codes;
  metal::MetalBuffer packed;
  metal::MetalBuffer message;
  metal::MetalBuffer candidateHead;
};

class DSpine final {
public:
  static void addResidual(metal::CommandGraph &graph, metal::MetalBuffer input,
                          metal::MetalBuffer residual, metal::MetalBuffer output,
                          uint32_t width, uint32_t rows);
  static void addReadInput(metal::CommandGraph &graph, metal::MetalBuffer input,
                           metal::MetalBuffer normalized, const DSpineBuffers &buffers, DSpineParams params);
  static void addCodes(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                       const DSpineWeights &weights, const DSpineBuffers &buffers, DSpineParams params);
  static void addMessage(metal::CommandGraph &graph, const DSpineWeights &weights,
                         const DSpineBuffers &buffers, DSpineParams params);
  static void addWrite(metal::CommandGraph &graph, metal::MetalBuffer delta,
                       metal::MetalBuffer output, const DSpineBuffers &buffers, DSpineParams params);
  static void addRefinement(metal::CommandGraph &graph, const DSpineWeights &weights,
                            const NormWeights &norm, const Projection &head,
                            const DSpineBuffers &buffers, const DraftSelectorBuffers &selection,
                            DSpineParams shape, std::span<const uint32_t> anchors,
                            std::span<const SamplingPolicy> policies);
};

} // namespace splash::ops
