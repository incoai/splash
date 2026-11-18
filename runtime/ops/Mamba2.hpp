#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/GDN.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace splash::ops {

// A Mamba-2 mixer's geometry (kernels/shared/mamba2.metal): `heads` heads of
// `headDimension` state rows over the `stateDimension` columns of their
// group's B and C, in `groups` groups. Its in-projection rows are
// z | x B C | dt: z and x one value per head row, B and C one row of N per
// group, dt one value per head; x B C go through the 4-tap convolution.
struct Mamba2Shape final {
  uint32_t groups = 0;
  uint32_t heads = 0;
  uint32_t headDimension = 0;
  uint32_t stateDimension = 0;

  [[nodiscard]] constexpr uint32_t innerSize() const noexcept { return heads * headDimension; }
  [[nodiscard]] constexpr uint32_t convolutionDimension() const noexcept {
    return innerSize() + 2 * groups * stateDimension;
  }
  [[nodiscard]] constexpr uint32_t packedWidth() const noexcept {
    return innerSize() + convolutionDimension() + heads;
  }
  // The grouped norm's width: the gated rows' groups.
  [[nodiscard]] constexpr uint32_t normWidth() const noexcept { return groups ? innerSize() / groups : 0; }
  // The kernels' name suffix of a compiled geometry, or none.
  [[nodiscard]] std::string_view compiledSuffix() const noexcept;
  bool operator==(const Mamba2Shape &) const = default;
};

// A layer's tape (metal/abi/Mamba2.h), which GdnTapeLane slots hold.
[[nodiscard]] uint64_t mamba2TapeLayerBytes(const Mamba2Shape &shape);

// A layer's F32 tensors: the convolution's taps (conv dim x 4) and bias, A,
// the dt bias and D (a value per head), and the grouped norm's weights
// (inner size).
struct Mamba2Weights final {
  metal::MetalBuffer convolution;
  metal::MetalBuffer convolutionBias;
  metal::MetalBuffer a;
  metal::MetalBuffer dtBias;
  metal::MetalBuffer d;
  metal::MetalBuffer norm;
};

// A prefill step's states, each a layer's view of its cell.
struct Mamba2StepStates final {
  metal::MetalBuffer convolutionIn;
  metal::MetalBuffer recurrentIn;
  metal::MetalBuffer convolutionOut;
  metal::MetalBuffer recurrentOut;
};

class Mamba2 final {
public:
  // Each lane's rows of the step's `rows` rows of `packed` (its block in
  // `laneRows`, SplashLaneRows, a chain of up to a verify block's rows) from
  // its state cell's layer `layer` to the next cell's, folding the path its
  // tape lane retained first (a chain's, its first rows), as GDN's decode
  // does: the state written is the one before the step's rows, which the
  // step leaves on its tape. Writes the gated rows y (fp32, inner size a
  // row) of the lanes' rows. The state is fp32.
  static void addDecode(metal::CommandGraph &graph, metal::MetalBuffer packed, const Mamba2Weights &weights,
                        std::span<const metal::MetalBuffer> currentStates,
                        std::span<const metal::MetalBuffer> nextStates, metal::MetalBuffer tape,
                        std::span<const GdnTapeLane> tapeLanes, metal::MetalBuffer laneRows, uint32_t rows,
                        metal::MetalBuffer y, Mamba2Shape shape, uint32_t layer, GdnStateStrides state);
  // One prefill step of eight rows of `packed` from `states`' inputs to
  // their outputs, folding the `pendingRows` rows of the previous step's
  // tape at `pendingTape` first and leaving its rows' at `stepTape`; rows
  // past a sequence's are computed and their outputs dropped.
  static void addPrefillStep(metal::CommandGraph &graph, metal::MetalBuffer packed, const Mamba2Weights &weights,
                             const Mamba2StepStates &states, metal::MetalBuffer tape, uint64_t pendingTape,
                             uint64_t stepTape, uint32_t pendingRows, metal::MetalBuffer y, Mamba2Shape shape);
  // The grouped RMS norm of `rows` gated rows y into bf16 `hidden`, the out
  // projection's input.
  static void addGroupNorm(metal::CommandGraph &graph, metal::MetalBuffer y, const metal::MetalBuffer &norm,
                           metal::MetalBuffer hidden, double epsilon, Mamba2Shape shape, uint32_t rows);
};

} // namespace splash::ops
