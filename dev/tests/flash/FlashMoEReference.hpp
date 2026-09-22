#pragma once

// Independent CPU oracle. It deliberately includes no production headers or
// helpers. These are the fixed native semantics, rather than an assertion of
// identical MLX argpartition tie order or of CPU/GPU reduction order.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::flash::reference {

using Bits = uint16_t;
inline constexpr uint32_t kMaxRows = 2048;
inline constexpr uint32_t kMaxExperts = 512;
inline constexpr uint32_t kMaxSelections = 10;
inline constexpr uint32_t kMaxWidth = 2560;

inline Bits toBF16(float value) {
  uint32_t word = std::bit_cast<uint32_t>(value);
  if ((word & 0x7f800000U) == 0x7f800000U) {
    if (word & 0x007fffffU)
      return static_cast<Bits>((word >> 16) | 0x0040U);
    return static_cast<Bits>(word >> 16);
  }
  word += 0x7fffU + ((word >> 16) & 1U);
  return static_cast<Bits>(word >> 16);
}

inline float fromBF16(Bits value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

namespace detail {

inline void bounded(uint32_t value, uint32_t maximum, const char *name) {
  if (value == 0 || value > maximum)
    throw std::invalid_argument(std::string(name) + " is outside native bounds");
}

inline void shape(std::span<const Bits> values, uint64_t expected,
                  const char *name) {
  if (values.size() != expected)
    throw std::invalid_argument(std::string(name) + " has the wrong size");
  for (Bits value : values)
    if (!std::isfinite(fromBF16(value)))
      throw std::invalid_argument(std::string(name) + " contains nonfinite data");
}

inline Bits finiteRound(float value, const char *name) {
  const Bits result = toBF16(value);
  if (!std::isfinite(value) || !std::isfinite(fromBF16(result)))
    throw std::overflow_error(std::string(name) + " has a nonfinite result");
  return result;
}

} // namespace detail

inline Bits sigmoidBF16(Bits input) {
  const float value = fromBF16(input);
  if (!std::isfinite(value))
    throw std::invalid_argument("sigmoid input is nonfinite");
  // Installed MLX Metal Sigmoid<bfloat> rounds each operation. Its integer
  // literals keep add/div/sub arithmetic in BF16, unlike the CPU F32-tail
  // implementation. Exp/denominator overflow to +inf is valid saturation.
  const Bits absolute = toBF16(std::abs(value));
  const Bits exponent = toBF16(std::exp(fromBF16(absolute)));
  const Bits denominator = toBF16(1.0F + fromBF16(exponent));
  const Bits tail = detail::finiteRound(1.0F / fromBF16(denominator), "sigmoid tail");
  const Bits complement = detail::finiteRound(1.0F - fromBF16(tail), "sigmoid complement");
  return value < 0.0F ? tail : complement;
}

struct RouteResult {
  uint32_t rows = 0;
  uint32_t experts = 0;
  uint32_t selections = 0;
  std::vector<int64_t> indices;
  std::vector<Bits> scores;
  // Exposed to prove that selection occurs after BF16 softmax rounding.
  std::vector<Bits> probabilities;
};

// Actual MLX GPU top-10 score reduction is sequential BF16. This helper also
// permits isolated discriminatory score fixtures, without rerunning softmax.
inline std::vector<Bits> normalizeSelectedScores(std::span<const Bits> selected) {
  if (selected.empty() || selected.size() > kMaxSelections)
    throw std::invalid_argument("selections is outside native bounds");
  detail::shape(selected, selected.size(), "selected scores");
  Bits total = 0;
  for (Bits score : selected) {
    const float value = fromBF16(score);
    if (value < 0.0F || value > 1.0F)
      throw std::invalid_argument("selected score is outside [0,1]");
    total = detail::finiteRound(fromBF16(total) + value, "selected-score sum");
  }
  const float denominator = fromBF16(total);
  if (denominator <= 0.0F)
    throw std::overflow_error("selected-score sum is nonpositive");
  std::vector<Bits> result;
  result.reserve(selected.size());
  for (Bits score : selected)
    result.push_back(detail::finiteRound(fromBF16(score) / denominator, "normalized score"));
  return result;
}

// F32 max-subtracted softmax, sequential F32 denominator, BF16 probabilities.
// Canonical sort is probability descending, exact ties by ascending expert ID.
// The selected-score sum rounds after every BF16 addition; each normalized
// score rounds to BF16. Actual MLX GPU K=10 qualified these storage boundaries.
inline RouteResult route(std::span<const Bits> logits, uint32_t rows,
                         uint32_t experts = 512, uint32_t selections = 10,
                         bool normalizeTopK = true) {
  detail::bounded(rows, kMaxRows, "rows");
  detail::bounded(experts, kMaxExperts, "experts");
  detail::bounded(selections, kMaxSelections, "selections");
  if (selections > experts)
    throw std::invalid_argument("selections exceeds the number of experts");
  detail::shape(logits, uint64_t{rows} * experts, "logits");

  RouteResult result{rows, experts, selections, {}, {}, {}};
  result.indices.resize(uint64_t{rows} * selections);
  result.scores.resize(uint64_t{rows} * selections);
  result.probabilities.resize(uint64_t{rows} * experts);
  std::vector<float> exponentials(experts);
  std::vector<int64_t> ids(experts);

  for (uint32_t row = 0; row < rows; ++row) {
    const uint64_t inputBase = uint64_t{row} * experts;
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t expert = 0; expert < experts; ++expert)
      maximum = std::max(maximum, fromBF16(logits[inputBase + expert]));
    float denominator = 0.0F;
    for (uint32_t expert = 0; expert < experts; ++expert) {
      // Subtracting finite extremes may produce -inf. Its exp is valid zero.
      const float shifted = fromBF16(logits[inputBase + expert]) - maximum;
      exponentials[expert] = std::exp(shifted);
      denominator += exponentials[expert];
    }
    if (!std::isfinite(denominator) || denominator <= 0.0F)
      throw std::overflow_error("softmax denominator is invalid");
    for (uint32_t expert = 0; expert < experts; ++expert)
      result.probabilities[inputBase + expert] =
          detail::finiteRound(exponentials[expert] / denominator, "probability");
    std::iota(ids.begin(), ids.end(), int64_t{0});
    std::sort(ids.begin(), ids.end(), [&](int64_t left, int64_t right) {
      const float a = fromBF16(result.probabilities[inputBase + left]);
      const float b = fromBF16(result.probabilities[inputBase + right]);
      return a == b ? left < right : a > b;
    });

    const uint64_t outputBase = uint64_t{row} * selections;
    for (uint32_t slot = 0; slot < selections; ++slot) {
      result.indices[outputBase + slot] = ids[slot];
      result.scores[outputBase + slot] =
          result.probabilities[inputBase + ids[slot]];
    }
    if (normalizeTopK) {
      const auto normalized = normalizeSelectedScores(
          std::span<const Bits>(result.scores).subspan(outputBase, selections));
      std::copy(normalized.begin(), normalized.end(), result.scores.begin() + outputBase);
    }
  }
  return result;
}

// Sigmoid's internal exp/add/div/sub each round to BF16, followed by BF16
// gate*sigmoid and BF16 silu*up. CPU libm exp is an independent precise
// approximation: compiled MLX Metal SwiGLU uses fast exp and can differ at a
// BF16 rounding threshold (for example gate=-6.84375). Actual MLX GPU golden
// comparisons, rather than this CPU helper, qualify those threshold cases.
inline std::vector<Bits> swiglu(std::span<const Bits> gate,
                               std::span<const Bits> up, uint32_t rows,
                               uint32_t width, uint32_t selections = 1) {
  detail::bounded(rows, kMaxRows, "rows");
  detail::bounded(width, kMaxWidth, "width");
  detail::bounded(selections, kMaxSelections, "selections");
  const uint64_t elements = uint64_t{rows} * selections * width;
  detail::shape(gate, elements, "gate");
  detail::shape(up, elements, "up");
  std::vector<Bits> result(elements);
  for (uint64_t index = 0; index < elements; ++index) {
    const Bits sigmoid = sigmoidBF16(gate[index]);
    const Bits silu = detail::finiteRound(
        fromBF16(gate[index]) * fromBF16(sigmoid), "silu");
    result[index] = detail::finiteRound(
        fromBF16(silu) * fromBF16(up[index]), "swiglu");
  }
  return result;
}

// BF16 per-expert products. Fixed eight-partial BF16 column reduction:
// (0+8),(1+9),2..7, then BF16 merge partials 0..7. Actual MLX GPU K=10,
// width=2560 qualified this order; the native oracle retains that fixed order
// for every width/phase rather than selecting an accumulation policy by shape.
// BF16 shared sigmoid/product and BF16 routed+shared follow. IDs validate route
// layout, but expertDown is already gathered in slot order (no ID reindexing).
inline std::vector<Bits>
combine(std::span<const Bits> expertDown, std::span<const int64_t> expertIDs,
        std::span<const Bits> scores, std::span<const Bits> sharedDown,
        std::span<const Bits> sharedGate, uint32_t rows, uint32_t width,
        uint32_t experts = 512, uint32_t selections = 10) {
  detail::bounded(rows, kMaxRows, "rows");
  detail::bounded(width, kMaxWidth, "width");
  detail::bounded(experts, kMaxExperts, "experts");
  detail::bounded(selections, kMaxSelections, "selections");
  if (selections > experts)
    throw std::invalid_argument("selections exceeds the number of experts");
  const uint64_t routes = uint64_t{rows} * selections;
  detail::shape(expertDown, routes * width, "expert down");
  detail::shape(scores, routes, "scores");
  detail::shape(sharedDown, uint64_t{rows} * width, "shared down");
  detail::shape(sharedGate, rows, "shared gate");
  if (expertIDs.size() != routes)
    throw std::invalid_argument("expert IDs has the wrong size");
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t slot = 0; slot < selections; ++slot) {
      const uint64_t index = uint64_t{row} * selections + slot;
      const int64_t id = expertIDs[index];
      if (id < 0 || uint64_t(id) >= experts)
        throw std::invalid_argument("expert ID is out of bounds");
      for (uint32_t previous = 0; previous < slot; ++previous)
        if (expertIDs[uint64_t{row} * selections + previous] == id)
          throw std::invalid_argument("expert IDs repeat within a row");
      const float score = fromBF16(scores[index]);
      if (score < 0.0F || score > 1.0F)
        throw std::invalid_argument("route score is outside [0,1]");
    }
  }

  std::vector<Bits> result(uint64_t{rows} * width);
  for (uint32_t row = 0; row < rows; ++row) {
    const Bits sharedScale = sigmoidBF16(sharedGate[row]);
    for (uint32_t column = 0; column < width; ++column) {
      std::array<Bits, 8> partials{};
      for (uint32_t partial = 0; partial < partials.size(); ++partial) {
        for (uint32_t slot = partial; slot < selections; slot += partials.size()) {
          const uint64_t routeIndex = uint64_t{row} * selections + slot;
          const Bits term = detail::finiteRound(
              fromBF16(expertDown[routeIndex * width + column]) *
                  fromBF16(scores[routeIndex]),
              "weighted expert term");
          partials[partial] = detail::finiteRound(
              fromBF16(partials[partial]) + fromBF16(term), "expert partial sum");
        }
      }
      Bits routed = partials[0];
      for (uint32_t partial = 1; partial < partials.size(); ++partial)
        routed = detail::finiteRound(fromBF16(routed) + fromBF16(partials[partial]), "expert sum");
      const uint64_t outputIndex = uint64_t{row} * width + column;
      const Bits shared = detail::finiteRound(
          fromBF16(sharedDown[outputIndex]) * fromBF16(sharedScale),
          "weighted shared expert");
      result[outputIndex] = detail::finiteRound(
          fromBF16(routed) + fromBF16(shared), "combined experts");
    }
  }
  return result;
}

// Computational oracle convenience for valid routes; malformed-ID tests use
// the overload above. The gathered expert outputs are independent of ID labels.
inline std::vector<Bits>
combine(std::span<const Bits> expertDown, std::span<const Bits> scores,
        std::span<const Bits> sharedDown, std::span<const Bits> sharedGate,
        uint32_t rows, uint32_t width, uint32_t selections = 10) {
  detail::bounded(rows, kMaxRows, "rows");
  detail::bounded(selections, kMaxSelections, "selections");
  std::vector<int64_t> ids(uint64_t{rows} * selections);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t slot = 0; slot < selections; ++slot)
      ids[uint64_t{row} * selections + slot] = slot;
  return combine(expertDown, ids, scores, sharedDown, sharedGate, rows, width,
                 kMaxExperts, selections);
}

} // namespace splash::flash::reference
