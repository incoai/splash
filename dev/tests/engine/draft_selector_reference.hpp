#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace splash::test::draft_selector {

// Look-ahead selection in fp32 or double precision. score(p, s, c) is the selector
// score of candidate c at position p after predecessor s; position 0 has
// only the anchor (s = 0). q_p(c | s) is the softmax over c, and a path's
// expected accepted length is the sum of its prefix products of q.
template <typename Real>
using Score = std::function<Real(uint32_t, uint32_t, uint32_t)>;

template <typename Real>
std::vector<Real> conditional(const Score<Real> &score, uint32_t position,
                                uint32_t predecessor, uint32_t candidates) {
  std::vector<Real> q(candidates);
  Real maximum = -INFINITY;
  for (uint32_t c = 0; c < candidates; ++c)
    maximum = std::max(maximum, score(position, predecessor, c));
  if (maximum == -INFINITY)
    return std::vector<Real>(candidates, 0.0);
  Real sum = 0.0;
  for (uint32_t c = 0; c < candidates; ++c) {
    q[c] = std::exp(score(position, predecessor, c) - maximum);
    sum += q[c];
  }
  for (Real &value : q)
    value /= sum;
  return q;
}

template <typename Real>
Real expectedLength(const Score<Real> &score, std::span<const uint32_t> path,
                      uint32_t candidates) {
  Real total = 0.0, prefix = 1.0;
  uint32_t predecessor = 0;
  for (uint32_t position = 0; position < path.size(); ++position) {
    prefix *= conditional(score, position, predecessor, candidates)[path[position]];
    total += prefix;
    predecessor = path[position];
  }
  return total;
}

// Backward dynamic programming: the best expected length and its path.
template <typename Real>
std::pair<Real, std::vector<uint32_t>>
referenceLookahead(const Score<Real> &score, uint32_t positions, uint32_t candidates) {
  std::vector<std::vector<uint32_t>> choice(positions,
                                            std::vector<uint32_t>(candidates));
  std::vector<Real> value(candidates, 0.0);
  for (uint32_t step = 0; step < positions; ++step) {
    const uint32_t position = positions - 1 - step;
    std::vector<Real> next(candidates, 0.0);
    for (uint32_t s = 0; s < (position ? candidates : 1); ++s) {
      const auto q = conditional(score, position, s, candidates);
      Real best = -1.0;
      for (uint32_t c = 0; c < candidates; ++c) {
        const Real candidateValue = q[c] * (Real{1} + value[c]);
        if (candidateValue > best) {
          best = candidateValue;
          choice[position][s] = c;
        }
      }
      next[s] = std::max(best, Real{0});
    }
    value = next;
  }
  std::vector<uint32_t> path(positions);
  uint32_t predecessor = 0;
  for (uint32_t position = 0; position < positions; ++position) {
    path[position] = choice[position][predecessor];
    predecessor = path[position];
  }
  return {value[0], path};
}

} // namespace splash::test::draft_selector
