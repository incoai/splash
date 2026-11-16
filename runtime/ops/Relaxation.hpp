#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>

namespace splash::ops {

// RFlash, relaxed speculative decoding with a drift constraint
// (metal/abi/Sampling.h RelaxParams): the two hyperparameters of the
// relaxation of a sampled lane's verify, set once per server. The price
// `lambda` > 0 of the divergence a verify block spends, and the drift
// allowance `alpha` >= 0, the drift toward the draft a block may add per nat
// of divergence (0 allows none). A larger price stays closer to lossless
// verification. The other parameters of the kernel are those of the
// reference implementation.
struct RelaxationSetting final {
  float lambda = 1.0F;
  float alpha = 0.0F;
  // The kernel's fixed parameters (the reference implementation's): the
  // smoothing of the draft, the tail floor, the price search's cap as a
  // fraction of lambda / alpha (below which every row's program stays
  // concave) and the search's iterations.
  static constexpr float eps = 1e-3F;
  // The floor sits well above the fp32 rounding of 1 - sum(p) (about 1e-6
  // when a row's whole nucleus lies on the draft's support): a rounding-born
  // tail priced as a live rest bucket absorbs the row's mass near the
  // concavity limit and leaves nothing to draw.
  static constexpr float tailFloor = 1e-4F;
  static constexpr float priceCap = 0.99F;
  // The search's reach in units of lambda where the concavity limit does not
  // bound it: the price of a block the constraint binds is of the order of
  // lambda.
  static constexpr float priceReach = 16.0F;
  static constexpr uint32_t iterations = 24;
  // The settings' range: the price from `least` to `most`, the allowance 0
  // or up to `most`. Within it the search's upper end, the rows' prices and
  // their reciprocals are finite floats, far from either end of a float.
  static constexpr double least = 1e-6;
  static constexpr double most = 1e6;

  // Whether the number an option gives is in the range, as a price
  // (`positive`) or an allowance, and a float holds it as what it is: an
  // allowance not as zero unless it is 0. The server refuses the same
  // numbers (serve_options.py), so none reaches the engine's start.
  [[nodiscard]] static bool holds(double value, bool positive) noexcept {
    return value <= most && (positive ? value >= least : value >= 0.0) &&
           (value == 0.0 || static_cast<float>(value) != 0.0F);
  }
  // None unless lambda and alpha are in the range. Inline: the status and
  // argument code checks settings without the Metal side of Sampling.
  [[nodiscard]] static std::optional<RelaxationSetting> validated(float lambda, float alpha) noexcept {
    if (!(lambda >= static_cast<float>(least) && lambda <= static_cast<float>(most) && alpha >= 0.0F &&
          alpha <= static_cast<float>(most)))
      return std::nullopt;
    RelaxationSetting setting;
    setting.lambda = lambda;
    setting.alpha = alpha;
    return setting;
  }
  // Reach-weight passes: two at alpha 0, else three.
  [[nodiscard]] uint32_t passes() const noexcept { return alpha == 0.0F ? 2 : 3; }
  // Whether the price search ends at priceReach lambda, below the concavity
  // limit's fraction: at alpha 0, where every price is concave, and at an
  // alpha so small that the limit lies further out. The search resolves a
  // price to a fixed fraction of its upper end, so an end at lambda / alpha
  // for a tiny alpha would leave the price unresolved.
  [[nodiscard]] bool reachBound() const noexcept { return alpha * priceReach <= priceCap; }
  // The upper end of the price search.
  [[nodiscard]] float muMax() const noexcept { return reachBound() ? priceReach * lambda : priceCap * lambda / alpha; }
  // The first probe of the search as a fraction of muMax: where priceReach
  // bounds it the cap is far above the usual root, so lambda is probed
  // first; under the concavity limit roots sit near the limit.
  [[nodiscard]] float firstProbe() const noexcept { return reachBound() ? 1.0F / priceReach : 1.0F; }

  // A setting's float as a report shows it, the double its seven
  // significant digits read as: 0.7 for a float 0.7, not the digits of its
  // binary expansion.
  [[nodiscard]] static double shown(float value) noexcept {
    char text[32];
    std::snprintf(text, sizeof text, "%.7g", static_cast<double>(value));
    return std::strtod(text, nullptr);
  }

  bool operator==(const RelaxationSetting &) const = default;
};

} // namespace splash::ops
