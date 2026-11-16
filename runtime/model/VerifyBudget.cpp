#include "model/VerifyBudget.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kParameters = 1 + VerifyBudget::kPositions;
constexpr uint32_t kIterations = 16;

using Parameters = std::array<double, kParameters>;

// x with h x = g for a positive definite h, by Gaussian elimination with
// partial pivoting: the Newton step of the acceptance fit.
Parameters solve(std::array<Parameters, kParameters> h, Parameters g) noexcept {
  for (size_t column = 0; column < kParameters; ++column) {
    size_t pivot = column;
    for (size_t row = column + 1; row < kParameters; ++row)
      if (std::fabs(h[row][column]) > std::fabs(h[pivot][column]))
        pivot = row;
    std::swap(h[column], h[pivot]);
    std::swap(g[column], g[pivot]);
    for (size_t row = column + 1; row < kParameters; ++row) {
      const double factor = h[row][column] / h[column][column];
      for (size_t k = column; k < kParameters; ++k)
        h[row][k] -= factor * h[column][k];
      g[row] -= factor * g[column];
    }
  }
  Parameters x{};
  for (size_t row = kParameters; row-- > 0;) {
    double sum = g[row];
    for (size_t k = row + 1; k < kParameters; ++k)
      sum -= h[row][k] * x[k];
    x[row] = sum / h[row][row];
  }
  return x;
}

double logit(double p) {
  p = std::clamp(p, 1.0e-6, 1.0 - 1.0e-6);
  return std::log(p / (1.0 - p));
}

bool valid(const VerifyBudget::Label &label) {
  return label.position < VerifyBudget::kPositions && std::isfinite(label.feature) && label.cap >= 0.0F &&
         label.cap <= 1.0F;
}

Parameters parametersOf(const VerifyBudget::Acceptance &acceptance) {
  Parameters parameters{acceptance.slope};
  for (uint32_t position = 0; position < VerifyBudget::kPositions; ++position)
    parameters[1 + position] = acceptance.intercepts[position];
  return parameters;
}

// Whether a label's cap bounds its chance at logit z: then the chance is the
// cap's, whatever the parameters.
bool capped(const VerifyBudget::Label &label, double z) { return label.cap < 1.0F && z > logit(label.cap); }

// The negative log posterior, up to a constant.
double objective(std::span<const VerifyBudget::Label> labels, const Parameters &theta,
                 const Parameters &prior) {
  double value = 0.0;
  for (const VerifyBudget::Label &label : labels) {
    const double z = theta[0] * label.feature + theta[1 + label.position];
    if (capped(label, z)) {
      const double cap = std::clamp(double(label.cap), 1.0e-6, 1.0 - 1.0e-6);
      value -= label.accepted ? std::log(cap) : std::log1p(-cap);
      continue;
    }
    // log(1 + e^-margin), the label's negative log-likelihood, stably.
    const double margin = label.accepted ? z : -z;
    value += margin > 0.0 ? std::log1p(std::exp(-margin)) : -margin + std::log1p(std::exp(margin));
  }
  for (uint32_t i = 0; i < kParameters; ++i)
    value += 0.5 * VerifyBudget::priorPrecision(i) * (theta[i] - prior[i]) * (theta[i] - prior[i]);
  return value;
}

} // namespace

std::vector<VerifyBudget::Label> VerifyBudget::treeLabels(std::span<const uint32_t> tokens,
                                                         std::span<const ops::TreeRow> rows,
                                                         std::span<const float> scores, std::span<const float> caps,
                                                         std::span<const uint32_t> selections) {
  if (rows.empty() || tokens.size() != rows.size() || scores.size() != 2 * rows.size() || caps.size() != rows.size())
    throw std::invalid_argument("invalid verify budget tree");
  // Each row's place on the walk's path, the index of its selection (and its
  // depth); none for a row off it. A row under the path's row at place i is a
  // proposal at position i.
  constexpr uint32_t kOff = UINT32_MAX;
  std::vector<uint32_t> place(rows.size(), kOff);
  place[0] = 0;
  std::vector<Label> labels;
  for (size_t row = 1; row < rows.size(); ++row) {
    const uint32_t parent = rows[row].parent;
    if (parent >= row)
      throw std::invalid_argument("invalid verify budget tree");
    const uint32_t at = place[parent];
    if (at == kOff || at >= selections.size())
      continue;
    const bool accepted = tokens[row] == selections[at];
    if (accepted)
      place[row] = at + 1;
    if (const float probability = scores[2 * row], cap = caps[row];
        probability >= 0.0F && probability <= 1.0F && cap >= 0.0F && cap <= 1.0F)
      labels.push_back({at, static_cast<float>(logit(probability)), accepted, cap});
  }
  return labels;
}

void VerifyBudget::observe(std::span<const Lane> lanes) {
  if (lanes.empty() || lanes.size() > kLanes)
    throw std::invalid_argument("invalid verify budget cycle");
  for (const Lane &lane : lanes)
    for (const Label &label : lane.labels)
      if (!valid(label))
        throw std::invalid_argument("invalid verify budget label");
  for (const Lane &lane : lanes) {
    std::vector<Label> &kept = labels_[lane.samples];
    size_t &next = next_[lane.samples];
    for (const Label &label : lane.labels) {
      if (kept.size() < kLabels) {
        kept.push_back(label);
      } else {
        kept[next] = label;
        next = (next + 1) % kLabels;
      }
    }
  }
  if (++cycles_ % kRefitCycles == 0)
    for (const bool samples : {false, true})
      if (!labels_[samples].empty())
        acceptance_[samples] = fit(labels(samples), acceptance_[samples]);
}

std::vector<VerifyBudget::Label> VerifyBudget::labels(bool samples) const {
  const std::vector<Label> &kept = labels_[samples];
  const auto split = kept.begin() + static_cast<std::ptrdiff_t>(next_[samples]);
  std::vector<Label> ordered(split, kept.end());
  ordered.insert(ordered.end(), kept.begin(), split);
  return ordered;
}

VerifyBudget::Acceptance VerifyBudget::fit(std::span<const Label> labels, const Acceptance &start) {
  for (const Label &label : labels)
    if (!valid(label))
      throw std::invalid_argument("invalid verify budget label");
  const Parameters prior = parametersOf(kPrior);
  Parameters theta = parametersOf(start);
  double current = objective(labels, theta, prior);
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    Parameters gradient{};
    std::array<Parameters, kParameters> hessian{};
    for (const Label &label : labels) {
      const uint32_t b = 1 + label.position;
      const double z = theta[0] * label.feature + theta[b];
      if (capped(label, z))
        continue;
      const double p = 1.0 / (1.0 + std::exp(-z));
      const double residual = p - (label.accepted ? 1.0 : 0.0);
      const double weight = p * (1.0 - p);
      gradient[0] += residual * label.feature;
      gradient[b] += residual;
      hessian[0][0] += weight * label.feature * label.feature;
      hessian[0][b] += weight * label.feature;
      hessian[b][0] += weight * label.feature;
      hessian[b][b] += weight;
    }
    for (uint32_t i = 0; i < kParameters; ++i) {
      gradient[i] += priorPrecision(i) * (theta[i] - prior[i]);
      hessian[i][i] += priorPrecision(i);
    }
    const Parameters step = solve(hessian, gradient);
    // Damped: the Newton step, halved until the objective does not grow.
    Parameters next{};
    double value = current, scale = 1.0;
    bool moved = false;
    for (uint32_t halving = 0; halving < 30 && !moved; ++halving, scale *= 0.5) {
      for (uint32_t i = 0; i < kParameters; ++i)
        next[i] = theta[i] - scale * step[i];
      value = objective(labels, next, prior);
      moved = value <= current;
    }
    if (!moved)
      break;
    double largest = 0.0;
    for (uint32_t i = 0; i < kParameters; ++i)
      largest = std::max(largest, std::fabs(next[i] - theta[i]));
    theta = next;
    current = value;
    if (largest < 1.0e-9)
      break;
  }
  Acceptance result{static_cast<float>(theta[0]), {}};
  for (uint32_t position = 0; position < kPositions; ++position)
    result.intercepts[position] = static_cast<float>(theta[1 + position]);
  return result;
}

std::vector<float> VerifyBudget::laneChances(std::span<const float> tree, std::span<const float> chain,
                                             uint32_t depth) {
  std::vector<float> chances(tree.size() + chain.size());
  std::merge(tree.begin(), tree.end(), chain.begin(), chain.end(), chances.begin(), std::greater<>());
  chances.resize(std::min<size_t>({chances.size(), depth, kBlockRows - 1}));
  return chances;
}

} // namespace splash::model
