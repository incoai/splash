#include "ops/LatentRotation.hpp"

#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <utility>

namespace splash::ops {

void LatentRotation::add(metal::CommandGraph &graph, metal::MetalBuffer input, bool floatInput,
                         metal::MetalBuffer norm, metal::MetalBuffer frequencies, metal::MetalBuffer positions,
                         metal::MetalBuffer output, LatentRotationShape shape, uint32_t rows) {
  // Each lane holds whole pairs of a vector's values, and a rounding block
  // whole lanes or a lane whole blocks.
  const uint32_t perLane = shape.width / 32;
  if (!rows || !shape.vectors || !shape.width || shape.width % 64 || shape.width > LATENT_ROTATION_MAX_WIDTH ||
      2 * shape.ropePairs > shape.width)
    throw std::invalid_argument("invalid latent rotation shape");
  const uint32_t block = shape.rounding == LatentRounding::Fp4E4m3Scale ? 16 : 32;
  if (shape.rounding != LatentRounding::None && perLane % block && block % perLane)
    throw std::invalid_argument("a latent rotation's rounding blocks do not fit its lanes");
  const uint64_t values = uint64_t{rows} * shape.vectors * shape.width;
  requireBytes(input, values * (floatInput ? sizeof(float) : sizeof(uint16_t)), "latent rotation input");
  if (shape.normalize) requireBytes(norm, uint64_t{shape.width} * sizeof(float), "latent rotation norm");
  requireBytes(frequencies, uint64_t{shape.ropePairs} * sizeof(float), "latent rotation frequencies");
  requireBytes(positions, uint64_t{rows} * sizeof(uint32_t), "latent rotation positions");
  requireBytes(output, values * sizeof(uint16_t), "latent rotation output");
  const LatentRotationParams params{rows,
                                    shape.vectors,
                                    shape.width,
                                    shape.ropePairs,
                                    shape.inverse ? 1u : 0u,
                                    static_cast<uint32_t>(shape.rounding),
                                    shape.normalize ? 1u : 0u,
                                    shape.normEpsilon};
  // An unnormalized rotation binds its frequencies in the norm's slot.
  metal::MetalBuffer weights = shape.normalize ? std::move(norm) : frequencies;
  graph.add(floatInput ? "latent_rotation_f32" : "latent_rotation_bf16",
            {std::move(input), std::move(weights), std::move(frequencies), std::move(positions), std::move(output)},
            params, {rows, shape.vectors, 1}, {32, 1, 1});
}

} // namespace splash::ops
