#pragma once

#include "ops/Sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace splash::test {

// An independent fp64 policy reference over the logits the same forward
// actually produced. The allowance bounds fp32 policy arithmetic, not the
// unrelated drift between prefill and incremental target forward graphs.
inline double requirePenaltyLogits(std::span<const float> raw,
                                   std::span<const float> actual,
                                   std::span<const uint8_t> prompt,
                                   std::span<const uint32_t> counts,
                                   const ops::SamplingPenalties &penalties,
                                   uint32_t selected) {
  if (raw.empty() || actual.size() != raw.size() || prompt.size() != raw.size() ||
      counts.size() != raw.size() || selected >= raw.size())
    throw std::runtime_error("penalty oracle geometry mismatch");
  if (!std::isfinite(penalties.repetition) || penalties.repetition <= 0.0F ||
      !std::isfinite(penalties.presence) || !std::isfinite(penalties.frequency))
    throw std::runtime_error("penalty oracle needs finite valid penalties");
  double maximumRelativeError = 0.0;
  float best = -std::numeric_limits<float>::infinity();
  for (size_t token = 0; token < raw.size(); ++token) {
    const float input = raw[token], output = actual[token];
    if (std::isnan(input) || input == -std::numeric_limits<float>::infinity()) {
      if (!(input == output || (std::isnan(input) && std::isnan(output))))
        throw std::runtime_error("penalties changed a nonfinite logit");
      continue;
    }
    const bool applies = penalties.active() &&
        (counts[token] || prompt[token]);
    if (input == std::numeric_limits<float>::infinity()) {
      const float expected = applies ? std::numeric_limits<float>::max() : input;
      if (output != expected)
        throw std::runtime_error("positive infinite penalty logit was not saturated correctly");
      best = std::max(best, output);
      continue;
    }
    double expected = input;
    if (applies)
      expected = input > 0.0F ? expected * std::min(1.0 / double(penalties.repetition),
                                                  double(std::numeric_limits<float>::max()))
                              : expected * double(penalties.repetition);
    const double scaled = expected;
    if (applies && counts[token])
      expected -= double(penalties.frequency) * counts[token] + double(penalties.presence);
    expected = std::clamp(expected, -double(std::numeric_limits<float>::max()),
                                    double(std::numeric_limits<float>::max()));
    const double scale = std::min(double(std::numeric_limits<float>::max()),
        std::max({1.0, std::abs(double(input)), std::abs(scaled), std::abs(expected),
                                   std::abs(double(penalties.frequency) * counts[token]),
                                   std::abs(double(penalties.presence))}));
    const double relative = std::abs(double(output) - expected) / scale;
    if (!std::isfinite(output) || relative > 8.0 * std::numeric_limits<float>::epsilon())
      throw std::runtime_error("penalty logits differ from the same-forward fp64 reference at token " +
                               std::to_string(token));
    maximumRelativeError = std::max(maximumRelativeError, relative);
    best = std::max(best, output);
  }
  if (std::isnan(actual[selected]) || actual[selected] == -std::numeric_limits<float>::infinity() ||
      actual[selected] != best)
    throw std::runtime_error("a greedy token is not the best of its actual penalized logits");
  return maximumRelativeError;
}

} // namespace splash::test
