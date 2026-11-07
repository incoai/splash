#pragma once

// The sampling penalties' arithmetic on the host, as the penalty kernels
// apply it to each logit (penalize_logit in sampling.metal), and how closely
// a logit they rewrote must match it. Shared by the Metal test of those
// kernels and the model runtime oracle, which checks penalized decisions
// against the logits of the forward that made them.

#include "ops/Sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace splash::test {

// Whether no token can be selected for a logit: NaN has no order and -inf
// no probability.
inline bool unselectableLogit(float value) {
  return std::isnan(value) || value == -std::numeric_limits<float>::infinity();
}

// The penalty kernel's arithmetic on the host: the logit of a token the
// output holds count times, or the prompt holds. An unselectable logit stays
// as it is.
inline float referencePenalty(float value, uint32_t count, bool prompt,
                              const ops::SamplingPenalties &penalties) {
  constexpr float kFloatMax = std::numeric_limits<float>::max();
  if ((!count && !prompt) || unselectableLogit(value))
    return value;
  value *= value > 0.0F
               ? std::min(1.0F / penalties.repetition, kFloatMax)
               : penalties.repetition;
  if (count)
    value -= penalties.frequency * float(count) + penalties.presence;
  return std::clamp(value, -kFloatMax, kFloatMax);
}

// A penalized logit as the kernel must write it. The scaling is one IEEE
// product on both sides, so a prompt-only token matches bitwise, as do a
// saturated and an unselectable one; the GPU may fuse frequency * count +
// presence, which leaves the difference within an ulp of the larger operand
// of the final subtraction.
inline bool penaltyMatches(float actual, float value, uint32_t count, bool prompt,
                           const ops::SamplingPenalties &penalties) {
  constexpr float kFloatMax = std::numeric_limits<float>::max();
  const float expected = referencePenalty(value, count, prompt, penalties);
  if (!count || unselectableLogit(value) || std::fabs(expected) == kFloatMax)
    return std::memcmp(&actual, &expected, sizeof(float)) == 0;
  const float subtrahend =
      penalties.frequency * float(count) + penalties.presence;
  const float magnitude = std::max(
      {std::fabs(expected + subtrahend), std::fabs(subtrahend),
       std::fabs(expected)});
  const float ulp = std::nextafter(magnitude, INFINITY) - magnitude;
  return std::fabs(actual - expected) <= 2.0F * ulp;
}

} // namespace splash::test
