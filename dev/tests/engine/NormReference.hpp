#pragma once

// Norm weights as a model stores them, bf16 or F32 as a GGUF does, and the
// fp64 RMS norm the kernels that read them are checked against. Shared by the
// Metal tests of those kernels.

#include "TestBuffers.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Normalization.hpp"
#include "tuning/LinearNumerics.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace splash::test {

// `size` weights holding value(0), value(1), ... in that order, stored as
// F32 when `float32` and rounded to bf16 otherwise.
template <class Value>
ops::NormWeights makeNormWeights(metal::MetalBackend &backend, uint32_t size, bool float32,
                                 Value value) {
  ops::NormWeights norm{{}, float32};
  norm.buffer = sharedBuffer(backend, norm.bytes(size));
  for (uint32_t index = 0; index < size; ++index) {
    const float weight = value(index);
    if (float32)
      static_cast<float *>(norm.buffer.contents())[index] = weight;
    else
      static_cast<uint16_t *>(norm.buffer.contents())[index] = ops::tuning::floatToBf16(weight);
  }
  return norm;
}

// The weight the kernels read at `index`.
inline double normWeight(const ops::NormWeights &norm, uint32_t index) {
  return norm.float32
      ? static_cast<const float *>(norm.buffer.contents())[index]
      : ops::tuning::bf16ToFloat(static_cast<const uint16_t *>(norm.buffer.contents())[index]);
}

// The fp64 RMS norm (epsilon 1e-6) of a bf16 row, scaled by the weights.
inline std::vector<double> rmsNorm(const uint16_t *row, const ops::NormWeights &norm,
                                   uint32_t size) {
  double squares = 0;
  for (uint32_t index = 0; index < size; ++index) {
    const double value = ops::tuning::bf16ToFloat(row[index]);
    squares += value * value;
  }
  const double inverse = 1 / std::sqrt(squares / size + 1e-6);
  std::vector<double> normalized(size);
  for (uint32_t index = 0; index < size; ++index)
    normalized[index] = ops::tuning::bf16ToFloat(row[index]) * inverse * normWeight(norm, index);
  return normalized;
}

// Whether `got` is `exact` rounded once to bf16, up to the fp32 arithmetic's
// noise (well below 2^-8 of half an ulp). A norm whose weights were rounded
// to bf16 anywhere on the way misses this for about a quarter of its values.
inline bool roundedOnceToBf16(uint16_t got, double exact) {
  return std::fabs(ops::tuning::bf16ToFloat(got) - exact) <=
      0.5 * ops::tuning::ulpBf16(float(exact)) * (1 + 1.0 / 256);
}

} // namespace splash::test
