#include "DenseLinear.hpp"

#include "metal/abi/DenseLinear.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

// A tile of rows x columns outputs on `simdgroups` simdgroups over static
// 128-wide input chunks, the kernel `name` and its epilogues'
// (kernels/shared/dense_linear.metal).
struct DenseTile final {
  uint32_t rows, columns, simdgroups;
  const char *name;
};
constexpr DenseTile kEight{64, 64, 8, "dense_linear_m64n64s8"};
constexpr DenseTile kSquare{64, 64, 4, "dense_linear_m64n64s4"};
constexpr DenseTile kWide{64, 128, 4, "dense_linear_m64n128s4"};
constexpr uint32_t kRowTile = 64;

// The rule, its thresholds the device's DenseTileCalibration: a projection of
// many inputs whose grid gives each core enough 64 x 64 output tiles runs
// 64 x 128 tiles, which read each input row once for 128 outputs, the traffic
// a long sum is bound by; one of a single row tile, which reads each weight
// once, or of few tiles per core runs 64 x 64 tiles on eight simdgroups,
// keeping the most simdgroups in flight; the rest run them on four. Each
// device class's thresholds are measured (KernelCalibration.hpp).
const DenseTile &tileOf(const KernelCalibration &calibration, uint32_t rows, uint32_t outputs,
                        uint32_t inputs) noexcept {
  const DenseTileCalibration &d = calibration.denseTile;
  // The grid's half 64 x 64 tiles, against a threshold's per core.
  const uint64_t halves = 2 * uint64_t{(rows + kRowTile - 1) / kRowTile} * (outputs / 64);
  const auto atLeast = [&](uint32_t perCore) { return halves >= uint64_t{perCore} * calibration.cores; };
  if (inputs >= d.wideInputs && atLeast(d.wideFromHalves)) return kWide;
  return rows <= kRowTile || !atLeast(d.eightBelowHalves) ? kEight : kSquare;
}

const char *epilogueSuffix(DenseEpilogue epilogue) noexcept {
  switch (epilogue) {
  case DenseEpilogue::None:
    return "";
  case DenseEpilogue::GeluTanh:
    return "_gelu_tanh";
  case DenseEpilogue::GeluErf:
    return "_gelu_erf";
  case DenseEpilogue::Residual:
    return "_residual";
  case DenseEpilogue::ScaledResidual:
    return "_scaled_residual";
  }
  return "";
}

} // namespace

DenseLinear::DenseLinear(const DeviceCapabilities &device) noexcept : calibration_(KernelCalibration::of(device)) {}

uint32_t DenseLinear::paddedRows(uint32_t rows) noexcept { return (rows + kRowTile - 1) / kRowTile * kRowTile; }

void DenseLinear::requireWeights(const DenseWeights &weights, uint32_t outputSize, uint32_t inputSize,
                                 const std::string &name) {
  static_assert(kOutputMultiple % kWide.columns == 0 && kOutputMultiple % kEight.columns == 0 &&
                    kOutputMultiple % kSquare.columns == 0,
                "an output multiple fills every tile's columns");
  if (!outputSize || outputSize % kOutputMultiple || !inputSize || inputSize % kInputMultiple)
    throw std::invalid_argument(name + " is " + std::to_string(outputSize) + " x " + std::to_string(inputSize) +
                                ", not whole multiples of the dense tiles' " + std::to_string(kOutputMultiple) +
                                " x " + std::to_string(kInputMultiple));
  requireBytes(weights.weight, uint64_t{outputSize} * inputSize * 2, name + " weight");
  requireBytes(weights.bias, uint64_t{outputSize} * 2, name + " bias");
  if (weights.scale) requireBytes(weights.scale, uint64_t{outputSize} * 2, name + " scale");
}

void DenseLinear::add(metal::CommandGraph &graph, const DenseWeights &weights, uint32_t outputSize,
                      uint32_t inputSize, const metal::MetalBuffer &input, const metal::MetalBuffer &output,
                      uint32_t rows, DenseEpilogue epilogue) const {
  // The weights and their sizes are requireWeights', which their owner checks once.
  requireBytes(input, uint64_t{paddedRows(rows)} * inputSize * 2, "dense input");
  const bool scaled = epilogue == DenseEpilogue::ScaledResidual;
  const uint64_t outputBytes = scaled || epilogue == DenseEpilogue::Residual ? 4 : 2;
  requireBytes(output, uint64_t{rows} * outputSize * outputBytes, "dense output");
  const DenseTile &tile = tileOf(calibration_, rows, outputSize, inputSize);
  std::vector<metal::MetalBuffer> buffers{input, weights.weight, weights.bias, output};
  if (scaled) buffers.push_back(weights.scale);
  graph.add(std::string(tile.name) + epilogueSuffix(epilogue), std::move(buffers),
            DenseLinearParams{rows, outputSize, inputSize, 0},
            {(rows + tile.rows - 1) / tile.rows, outputSize / tile.columns, 1}, {tile.simdgroups * 32, 1, 1});
}

} // namespace splash::ops
