#include "ops/ResidualSteer.hpp"

#include "metal/abi/ResidualSteer.h"
#include "ops/BufferExtent.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = RESIDUAL_STEER_THREADS;

}  // namespace

void addResidualSteer(metal::CommandGraph &graph, metal::MetalBuffer residual, metal::MetalBuffer directions,
                      ResidualSteer steer, uint32_t hiddenSize, uint32_t rows) {
  if (!rows || !hiddenSize || !std::isfinite(steer.scale))
    throw std::invalid_argument("invalid control vector step");
  requireBytes(residual, uint64_t{rows} * hiddenSize * sizeof(uint16_t), "steered residual");
  requireBytes(directions, (uint64_t{steer.layer} + 1) * hiddenSize * sizeof(float), "control vector directions");
  const uint32_t mode = steer.project ? RESIDUAL_STEER_PROJECT : RESIDUAL_STEER_ADD;
  const ResidualSteerParams params{rows, hiddenSize, steer.layer, mode, steer.scale};
  graph.add("residual_steer", {std::move(residual), std::move(directions)}, params, {rows, 1, 1}, {kThreads, 1, 1});
}

}  // namespace splash::ops
