#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/KernelCalibration.hpp"

#include <cstdint>
#include <string>

namespace splash::ops {

// A dense projection as its checkpoint stores it: bf16 weights
// [outputs][inputs] and a bf16 bias of every output, zeros for a projection
// stored without one; and, for one whose output a LayerScale multiplies
// (Inkling 2's tower), that scale of every output, bf16, which the
// ScaledResidual epilogue reads.
struct DenseWeights final {
  metal::MetalBuffer weight;
  metal::MetalBuffer bias;
  metal::MetalBuffer scale{};
};

// What a projection does with each sum plus its bias: stores it, or its
// GELU in the tanh or erf form, as bf16; or adds it, as it is or times its
// output's scale (ScaledResidual), to the fp32 rows it stores into, a
// residual stream updated in place.
enum class DenseEpilogue : uint8_t { None, GeluTanh, GeluErf, Residual, ScaledResidual };

// Dense projections on the tile the device's KernelCalibration picks for
// each one's grid (DenseLinear.cpp).
class DenseLinear final {
public:
  // Inputs are whole multiples of the kernels' 128-wide K chunks, outputs of
  // their widest tile.
  static constexpr uint32_t kInputMultiple = 128;
  static constexpr uint32_t kOutputMultiple = 128;

  explicit DenseLinear(const DeviceCapabilities &device) noexcept;

  // The rows of input a projection of `rows` rows reads: whole row tiles.
  [[nodiscard]] static uint32_t paddedRows(uint32_t rows) noexcept;
  // The projection's contract, which its owner checks once, where it takes
  // the weights: an output size of a whole kOutputMultiple and an input size
  // of a whole kInputMultiple (the kernels compute whole tiles and K chunks
  // and would leave the rest out), outputSize x inputSize bf16 weights and
  // outputSize bf16 biases, and scales if it has any. Throws, naming `name`,
  // otherwise.
  static void requireWeights(const DenseWeights &weights, uint32_t outputSize, uint32_t inputSize,
                             const std::string &name);
  // output[r] = epilogue(input[r] weight^T + bias) for the rows r < rows of
  // bf16 inputs of inputSize values into outputs of outputSize, of weights
  // requireWeights took. An output's sum does not depend on the rows beside
  // it or on the tile, so a row's bits are the same at any offset in its
  // buffer and in a projection of any rows. Throws unless each buffer holds
  // what the kernel reaches: paddedRows(rows) rows of input, `rows` of
  // output.
  void add(metal::CommandGraph &graph, const DenseWeights &weights, uint32_t outputSize, uint32_t inputSize,
           const metal::MetalBuffer &input, const metal::MetalBuffer &output, uint32_t rows,
           DenseEpilogue epilogue = DenseEpilogue::None) const;

private:
  KernelCalibration calibration_;
};

} // namespace splash::ops
