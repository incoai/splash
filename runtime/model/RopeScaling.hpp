#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <stdexcept>

namespace splash::model {

// YaRN, the RoPE scaling that stretches a target's native window by a factor
// (servedFor). The rotary frequencies whose wavelengths fit the native
// window beta_fast (32) times or more keep their value, those that fit it
// beta_slow (1) times or less are divided by the factor, and those between
// blend linearly; the rotated dimensions of queries and keys are then scaled
// by the attention factor, 0.1 ln(factor) + 1. These are YaRN's defaults, as
// a rope_parameters of type yarn with only a factor and
// original_max_position_embeddings configures them, which is how Qwen
// describes stretching these models. The scaling is static: every request,
// short or long, takes the same factor. The draft keeps its own RoPE. A
// family YaRN serves states the most context it stretches the window to
// (QwenTargetDimensions::yarnContextTokens).
struct RopeScaling final {
  // 1 without YaRN; with it, the context served over the native window, as
  // Qwen's model cards take it.
  double factor = 1.0;
  // The window YaRN stretches, original_max_position_embeddings; zero
  // without YaRN.
  uint32_t originalContextTokens = 0;

  [[nodiscard]] bool yarn() const noexcept { return factor != 1.0; }
  [[nodiscard]] float attentionFactor() const noexcept {
    return yarn() ? static_cast<float>(0.1 * std::log(factor) + 1.0) : 1.0F;
  }
  bool operator==(const RopeScaling &) const = default;
};

// The inverse frequencies of `pairs` rotated pairs of base `theta`, each
// theta^(-i / pairs), as YaRN scales them where `scaling` asks for it.
inline void rotaryInverseFrequencies(float theta, uint32_t pairs, const RopeScaling &scaling,
                                     std::span<float> frequencies) {
  if (!pairs || frequencies.size() < pairs)
    throw std::invalid_argument("invalid rotary inverse frequency table");
  for (uint32_t pair = 0; pair < pairs; ++pair)
    frequencies[pair] = std::pow(theta, -static_cast<float>(pair) / pairs);
  if (!scaling.yarn())
    return;
  if (!(scaling.factor > 1.0) || !scaling.originalContextTokens)
    throw std::invalid_argument("invalid YaRN scaling");
  // The correction range over the rotated dimensions, from the rotations
  // the native window holds at each wavelength, and the linear ramp across
  // it over the pairs.
  const double dimensions = 2.0 * pairs;
  const auto correction = [&](double rotations) {
    return dimensions *
           std::log(scaling.originalContextTokens / (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * std::log(double{theta}));
  };
  const double low = std::max(std::floor(correction(32.0)), 0.0);
  double high = std::min(std::ceil(correction(1.0)), dimensions - 1.0);
  if (low == high)
    high += 0.001;
  for (uint32_t pair = 0; pair < pairs; ++pair) {
    const float ramp = std::clamp(static_cast<float>((static_cast<double>(pair) - low) / (high - low)), 0.0F, 1.0F);
    const float extrapolation = frequencies[pair];
    const float interpolation = extrapolation / static_cast<float>(scaling.factor);
    frequencies[pair] = interpolation * ramp + extrapolation * (1.0F - ramp);
  }
}

} // namespace splash::model
