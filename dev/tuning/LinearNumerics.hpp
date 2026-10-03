#pragma once

#include "ops/Linear.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

// bf16 arithmetic and the split-K tolerance shared by the tuner's candidate
// qualification and the Q4 kernel tests. Operand bounds use CPU-visible
// buffers and do not submit GPU work.
namespace splash::ops::tuning {

inline float bf16ToFloat(uint16_t value) noexcept {
  return std::bit_cast<float>(uint32_t{value} << 16);
}
// Round to nearest even, as the kernels' bfloat conversion does.
inline uint16_t floatToBf16(float value) noexcept {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}

// Spacing of the bf16 binade holding value: 2^(e - 7) for a normal value in
// [2^e, 2^(e+1)), the seven fraction bits. Zero and denormals take the
// smallest normal spacing so a reference that rounds to zero still admits
// the noise of the other summation order.
inline float ulpBf16(float value) noexcept {
  const float magnitude = std::fabs(value);
  if (!(magnitude >= std::ldexp(1.0f, -126)) || !std::isfinite(magnitude))
    return std::ldexp(1.0f, -133);
  int exponent = 0;
  (void)std::frexp(magnitude, &exponent); // magnitude = m * 2^exponent, m in [0.5, 1)
  return std::ldexp(1.0f, exponent - 8);
}

inline float silu(float value) noexcept { return value / (1.0f + std::exp(-value)); }

// The sequential kernel's values an element of a split-K output is compared
// against: the output itself and, per epilogue, the residual it added or the
// exact gate and up projections it combined.
struct SplitReference final {
  float value = 0;
  float residual = 0;
  float gate = 0;
  float up = 0;
};

// fp32 reassociation noise. The split kernel adds the same per-group fp32
// terms as the sequential kernel in a different association (four range sums
// instead of one running sum), so before the bf16 rounding the projections
// differ by the rounding of at most G = K / 64 additions, each within 2^-24
// of the partial sum it produced. Partial sums are not observable, so the
// tensor's largest |reference| stands in for their magnitude. This term only
// matters where |value| is far below that magnitude; elsewhere the bf16 ulp
// dominates it by orders of magnitude.
inline float reassociationSlack(uint32_t inputSize, float maxAbsReference) noexcept {
  return float(inputSize / 64) * std::ldexp(std::fabs(maxAbsReference), -24);
}

// Bound the magic-offset kernel's fp32 error using operand magnitudes, not
// the (possibly cancelling) output. For each group, gamma(72)*143 covers
// 64 products plus row-sum/correction rounding; gamma(2G+8) covers the two
// affine FMAs per group and up to eight split additions. Maxima over columns
// keep this qualification pass linear in packed metadata, not matrix FLOPs.
inline float simdgroupSlack(LinearWorkload w, const metal::MetalBuffer &input,
                           const Projection &projection) {
  const uint32_t groups = w.matrix.inputSize / 64;
  std::vector<double> scale(groups), bias(groups);
  const auto *sc = static_cast<const uint16_t *>(projection.affine().scales.contents());
  const auto *bi = static_cast<const uint16_t *>(projection.affine().biases.contents());
  for (uint32_t n = 0; n < w.matrix.outputSize; ++n)
    for (uint32_t g = 0; g < groups; ++g) {
      const uint64_t i = (uint64_t(n / 256) * groups + g) * 256 + n % 256;
      scale[g] = std::max(scale[g], double(std::fabs(bf16ToFloat(sc[i]))));
      bias[g] = std::max(bias[g], double(std::fabs(bf16ToFloat(bi[i]))));
    }
  constexpr double u = 0x1p-24;
  const auto gamma = [&](double n) { return n * u / (1 - n * u); };
  const auto *x = static_cast<const uint16_t *>(input.contents());
  double bound = 0;
  for (uint32_t row = 0; row < w.rows; ++row) {
    double quant = 0, affine = 0;
    for (uint32_t g = 0; g < groups; ++g) {
      double absolute = 0;
      for (uint32_t k = 0; k < 64; ++k)
        absolute += std::fabs(bf16ToFloat(x[uint64_t(row) * w.matrix.inputSize + g * 64 + k]));
      quant += absolute * scale[g];
      affine += absolute * (15 * scale[g] + bias[g]);
    }
    bound = std::max(bound, gamma(72) * 143 * quant + gamma(2 * groups + 8) * affine);
  }
  return float(bound);
}

// Largest |actual - reference.value| a split-K output may show. Two roundings
// of values that differ by fp32 noise straddle at most one bf16 spacing, so
// the projection p differs by at most ulp(p) + slack. The epilogue then
// propagates that step:
//   None:     out = p.
//   Residual: out = bf16(p + r). The inputs differ by ulp(p) + slack with
//             p = out - r, and the output rounding adds ulp(out).
//   GateUp:   out = bf16(silu(g) * u), with g and u each within one ulp plus
//             slack: |d(silu(g) u)| <= |u| max|silu'| dg + |silu(g)| du,
//             max|silu'| = 1.0998 < 1.1, and the output rounding adds
//             ulp(out).
inline float splitTolerance(LinearEpilogue epilogue, SplitReference reference,
                            float slack) noexcept {
  float bound = ulpBf16(reference.value) + slack;
  if (epilogue == LinearEpilogue::Residual)
    bound += ulpBf16(reference.value - reference.residual);
  if (epilogue == LinearEpilogue::GateUp)
    bound += 1.1f * (std::fabs(reference.up) + ulpBf16(reference.up) + slack) * (ulpBf16(reference.gate) + slack) +
        std::fabs(silu(reference.gate)) * (ulpBf16(reference.up) + slack);
  return bound;
}

inline bool withinSplitTolerance(float actual, LinearEpilogue epilogue,
                                 SplitReference reference, float slack) noexcept {
  return std::isfinite(actual) &&
      std::fabs(actual - reference.value) <= splitTolerance(epilogue, reference, slack);
}

} // namespace splash::ops::tuning
