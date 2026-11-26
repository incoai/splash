#include "ops/Steering.hpp"

#include "metal/abi/Steering.h"
#include "ops/BufferExtent.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace splash::ops {

void Steering::addProject(metal::CommandGraph &graph, metal::MetalBuffer rows, metal::MetalBuffer direction,
                          float alpha, uint32_t rowsCount, uint32_t hidden) {
  if (!rowsCount || !hidden) throw std::invalid_argument("invalid steer shape");
  if (!std::isfinite(alpha)) throw std::invalid_argument("steer alpha is not finite");
  requireBytes(rows, uint64_t{rowsCount} * hidden * sizeof(float), "steer rows");
  requireBytes(direction, uint64_t{hidden} * sizeof(float), "steer direction");
  const SteerParams params{rowsCount, hidden, alpha, 0.0F};
  // A threadgroup a row.
  graph.add("steer_project_f32", {std::move(rows), std::move(direction)}, params, {rowsCount, 1, 1},
            {STEER_THREADS, 1, 1});
}

} // namespace splash::ops
