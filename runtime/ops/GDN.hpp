#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// Tensor geometry mapped to a compiled Metal variant during graph construction.
struct GdnShape final {
  uint32_t keyHeads = 0;
  uint32_t valueHeads = 0;
  uint32_t headDimension = 0;
  uint32_t convolutionDimension = 0;
  uint32_t packedWidth = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return keyHeads && valueHeads && valueHeads % keyHeads == 0 &&
           headDimension && convolutionDimension && packedWidth &&
           convolutionDimension ==
               (uint64_t{2} * keyHeads + valueHeads) * headDimension &&
           packedWidth >= convolutionDimension +
                              uint64_t{valueHeads} * headDimension +
                              uint64_t{2} * valueHeads;
  }

  bool operator==(const GdnShape &) const = default;
};

// The value-head order of the GDN output, the out_proj input columns. Grouped
// keeps a key head's value heads adjacent; Tiled is llama.cpp's GGUF order,
// value head h at (h % heads per key) * key heads + h / heads per key.
enum class GdnHeadOrder : uint8_t { Grouped, Tiled };

struct GdnStateStrides final {
  uint64_t convolutionLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convolutionStateBytes = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return convolutionLayerBytes && recurrentLayerBytes &&
           convolutionStateBytes;
  }
};

struct GdnPrefillBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer convolutionIn;
  metal::MetalBuffer convolutionOut;
  metal::MetalBuffer queries;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  metal::MetalBuffer recurrentIn;
  metal::MetalBuffer recurrentOut;
  metal::MetalBuffer recurrentRows;
  metal::MetalBuffer mixerNorm;
  metal::MetalBuffer hidden;
};

struct GdnDecodeBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer mixerNorm;
  metal::MetalBuffer hidden;
  metal::MetalBuffer arrived;
  metal::MetalBuffer generation;
  LinearScratch linearScratch{};
};

struct GdnCommitBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer retainedCounts;
};

class GDN final {
public:
  static void addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                         GdnShape shape, uint32_t tokens,
                         GdnHeadOrder order = GdnHeadOrder::Grouped);
  static void addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                        GdnShape shape, uint32_t lanes, uint32_t layer,
                        GdnStateStrides state,
                        GdnHeadOrder order = GdnHeadOrder::Grouped);
  static void addCommit(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                        GdnShape shape, uint32_t layers, uint32_t lanes,
                        GdnStateStrides state);
};

} // namespace splash::ops
