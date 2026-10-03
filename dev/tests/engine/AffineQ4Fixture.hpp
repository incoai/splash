#pragma once

// Affine Q4 test weights in the layout the kernels read: per output and
// 64-input group one parameter, in the packed 256-column storage order, with
// 32 bytes of packed nibbles and a bf16 scale and bias. A packed slab of
// model::q4PackedBytes holds the nibbles, then the scales, then the biases
// (model::readAffineProjection). Shared by the Linear and MoE tests.

#include "TestBuffers.hpp"
#include "metal/MetalBackend.hpp"
#include "model/WeightStore.hpp"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"

#include <cstdint>
#include <cstring>

namespace splash::test {

// A 32-bit integer hash for deterministic fixture values.
inline uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352d;
  value ^= value >> 15;
  value *= 0x846ca68b;
  return value ^ (value >> 16);
}

struct AffineQ4Planes final {
  uint8_t *weights;
  uint16_t *scales;
  uint16_t *biases;
};

// The Linear tests' projection: packed bytes mix(i + seed), and scales from
// 0.004 to 0.0056 with biases of -7.5 scales, so the weights centre on zero.
inline ops::Projection deterministicQ4Projection(metal::MetalBackend &backend, ops::LinearMatrix matrix,
                                                 uint32_t seed) {
  const uint64_t parameters = uint64_t{matrix.outputSize} * (matrix.inputSize / 64);
  ops::Projection p(matrix.outputSize, matrix.inputSize,
                    ops::AffineWeights{sharedBuffer(backend, parameters * 32), sharedBuffer(backend, parameters * 2),
                                       sharedBuffer(backend, parameters * 2)});
  const AffineQ4Planes planes{static_cast<uint8_t *>(p.affine().weights.contents()),
                              static_cast<uint16_t *>(p.affine().scales.contents()),
                              static_cast<uint16_t *>(p.affine().biases.contents())};
  for (uint64_t i = 0; i < parameters * 32; ++i) planes.weights[i] = mix(uint32_t(i) + seed);
  for (uint64_t i = 0; i < parameters; ++i) {
    const float scale = 0.004f + float(mix(uint32_t(i) + seed) % 17) * 0.0001f;
    planes.scales[i] = ops::tuning::floatToBf16(scale);
    planes.biases[i] = ops::tuning::floatToBf16(-7.5f * scale);
  }
  return p;
}

// `experts` zeroed outputSize x inputSize slabs at the packed stride, each
// written by fill(expert, planes).
template <class Fill>
ops::ExpertProjection expertSlabs(metal::MetalBackend &backend, uint32_t experts, uint32_t outputSize,
                                  uint32_t inputSize, const char *label, Fill fill) {
  const uint64_t stride = model::q4PackedBytes(outputSize, inputSize);
  const uint64_t weightBytes = uint64_t{outputSize} * inputSize / 2;
  const uint64_t parameters = uint64_t{outputSize} * (inputSize / 64);
  metal::MetalBuffer packed = backend.allocateBuffer(experts * stride, metal::BufferStorage::Shared, label);
  auto *bytes = static_cast<uint8_t *>(packed.contents());
  std::memset(bytes, 0, experts * stride);
  for (uint32_t expert = 0; expert < experts; ++expert) {
    uint8_t *slab = bytes + expert * stride;
    auto *scales = reinterpret_cast<uint16_t *>(slab + weightBytes);
    fill(expert, AffineQ4Planes{slab, scales, scales + parameters});
  }
  return {packed, experts, outputSize, inputSize, stride};
}

} // namespace splash::test
