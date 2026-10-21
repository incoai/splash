#include "model/VerifyBudget.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kParameters = 1 + VerifyBudget::kPositions;
constexpr uint32_t kIterations = 16;

// A cycle's cost at 1, 2, 4 and 8 rows a lane, for one to four lanes (every
// row costs 1): the mean shape of Qwen3.8-Flash-Next's cycles (resident and
// with 12 GB of experts) and Qwen3.6-35B-A3B's (resident and with 4 GB).
constexpr std::array<std::array<double, 4>, VerifyBudget::kLanes> kSeed{{
    {0.60, 0.69, 0.85, 1.0},
    {0.50, 0.58, 0.74, 1.0},
    {0.45, 0.52, 0.70, 1.0},
    {0.40, 0.47, 0.67, 1.0},
}};

using Parameters = std::array<double, kParameters>;

double logit(double p) {
  p = std::clamp(p, 1.0e-6, 1.0 - 1.0e-6);
  return std::log(p / (1.0 - p));
}

Parameters parametersOf(const VerifyBudget::Acceptance &acceptance) {
  Parameters parameters{acceptance.slope};
  for (uint32_t position = 0; position < VerifyBudget::kPositions; ++position)
    parameters[1 + position] = acceptance.intercepts[position];
  return parameters;
}

// The negative log posterior, up to a constant.
double objective(std::span<const VerifyBudget::Label> labels, const Parameters &theta,
                 const Parameters &prior) {
  double value = 0.0;
  for (const VerifyBudget::Label &label : labels) {
    const double z = theta[0] * label.feature + theta[1 + label.position];
    // log(1 + e^-margin), the label's negative log-likelihood, stably.
    const double margin = label.accepted ? z : -z;
    value += margin > 0.0 ? std::log1p(std::exp(-margin)) : -margin + std::log1p(std::exp(margin));
  }
  for (uint32_t i = 0; i < kParameters; ++i)
    value += 0.5 * VerifyBudget::priorPrecision(i) * (theta[i] - prior[i]) * (theta[i] - prior[i]);
  return value;
}

// x with h x = g, h positive definite: Gaussian elimination.
Parameters solve(std::array<Parameters, kParameters> h, Parameters g) {
  for (uint32_t column = 0; column < kParameters; ++column) {
    uint32_t pivot = column;
    for (uint32_t row = column + 1; row < kParameters; ++row)
      if (std::fabs(h[row][column]) > std::fabs(h[pivot][column]))
        pivot = row;
    std::swap(h[column], h[pivot]);
    std::swap(g[column], g[pivot]);
    for (uint32_t row = column + 1; row < kParameters; ++row) {
      const double factor = h[row][column] / h[column][column];
      for (uint32_t k = column; k < kParameters; ++k)
        h[row][k] -= factor * h[column][k];
      g[row] -= factor * g[column];
    }
  }
  Parameters x{};
  for (uint32_t row = kParameters; row-- > 0;) {
    double sum = g[row];
    for (uint32_t k = row + 1; k < kParameters; ++k)
      sum -= h[row][k] * x[k];
    x[row] = sum / h[row][row];
  }
  return x;
}

} // namespace

double VerifyBudget::seed(uint32_t lanes, uint32_t rows) {
  if (!lanes || lanes > kLanes || rows < lanes || rows > lanes * kBlockRows)
    throw std::invalid_argument("invalid verify budget rows");
  const double perLane = double(rows) / lanes;
  const std::array<double, 4> knots{1.0, 2.0, 4.0, 8.0};
  const std::array<double, 4> &shape = kSeed[lanes - 1];
  uint32_t upper = 1;
  while (upper < 3 && perLane > knots[upper])
    ++upper;
  const double t = (perLane - knots[upper - 1]) / (knots[upper] - knots[upper - 1]);
  return shape[upper - 1] + t * (shape[upper] - shape[upper - 1]);
}

const VerifyBudget::Cost &VerifyBudget::table(uint32_t lanes, bool constrained) const {
  if (!lanes || lanes > kLanes)
    throw std::invalid_argument("invalid verify budget lane count");
  return tables_[constrained ? 1 : 0][lanes - 1];
}

std::optional<std::pair<double, double>> VerifyBudget::line(uint32_t lanes, bool constrained) const {
  const Cost &cost = table(lanes, constrained);
  if (cost.weight <= 0.0)
    return std::nullopt;
  const uint32_t first = lanes, last = lanes * kBlockRows;
  const double meanRows = cost.rows / cost.weight, meanTime = cost.time / cost.weight;
  const double spread = std::max(cost.rowsSquared / cost.weight - meanRows * meanRows, 0.0);
  const double covariance = cost.rowsTime / cost.weight - meanRows * meanTime;
  // The seed's slope at the cycles' level, worth kPriorCycles cycles whose
  // rows spread by a row a lane.
  const double seedSlope = (seed(lanes, last) - seed(lanes, first)) / double(last - first);
  const double priorSlope = seedSlope * meanTime / seed(lanes, static_cast<uint32_t>(std::lround(meanRows)));
  const double priorSpread = double(lanes) * lanes;
  const double slope = std::max(
      (cost.weight * covariance + kPriorCycles * priorSpread * priorSlope) /
          (cost.weight * spread + kPriorCycles * priorSpread),
      0.0);
  return std::pair{meanTime - slope * meanRows, slope};
}

std::array<float, VerifyBudget::kBlockRows * VerifyBudget::kLanes + 1>
VerifyBudget::costs(uint32_t lanes, bool constrained) const {
  const uint32_t first = lanes, last = lanes * kBlockRows;
  std::array<float, kBlockRows * kLanes + 1> result{};
  const auto fit = line(lanes, constrained);
  if (!fit) {
    for (uint32_t rows = first; rows <= last; ++rows)
      result[rows] = static_cast<float>(seed(lanes, rows));
    return result;
  }
  const Cost &cost = table(lanes, constrained);
  // Nondecreasing and positive: a negative intercept leaves the fewest rows
  // a twentieth of the mean.
  for (uint32_t rows = first; rows <= last; ++rows)
    result[rows] = static_cast<float>(std::max(fit->first + fit->second * rows, 0.05 * cost.time / cost.weight));
  return result;
}

bool VerifyBudget::explores(uint32_t lanes, bool constrained) const {
  static_cast<void>(table(lanes, constrained));
  return tableCycles_[constrained ? 1 : 0][lanes - 1] % kExploreCycles == kExploreCycles - 1;
}

void VerifyBudget::observe(std::span<const Lane> lanes, bool constrained, double milliseconds) {
  if (lanes.empty() || lanes.size() > kLanes)
    throw std::invalid_argument("invalid verify budget cycle");
  uint32_t rows = 0;
  for (const Lane &lane : lanes) {
    if (!lane.verifyRows || lane.verifyRows > kBlockRows || lane.accepted > kPositions)
      throw std::invalid_argument("invalid verify budget lane");
    rows += lane.verifyRows;
  }
  const uint32_t count = static_cast<uint32_t>(lanes.size());
  ++tableCycles_[constrained ? 1 : 0][count - 1];
  if (std::isfinite(milliseconds) && milliseconds > 0.0) {
    if (const auto fit = line(count, constrained)) {
      const double expected = std::max(fit->first + fit->second * rows, 1.0e-3);
      milliseconds = std::clamp(milliseconds, 0.5 * expected, 2.0 * expected);
    }
    Cost &cost = tables_[constrained ? 1 : 0][count - 1];
    cost.weight = kDecay * cost.weight + 1.0;
    cost.rows = kDecay * cost.rows + rows;
    cost.rowsSquared = kDecay * cost.rowsSquared + double(rows) * rows;
    cost.time = kDecay * cost.time + milliseconds;
    cost.rowsTime = kDecay * cost.rowsTime + rows * milliseconds;
  }
  // Proposal j is checked against the target's row j, so a lane that
  // verified r rows labels its proposals up to r - 1, to the first rejection.
  for (const Lane &lane : lanes) {
    if (!lane.labels)
      continue;
    const uint32_t last = std::min({lane.accepted, lane.verifyRows - 1, kPositions - 1});
    for (uint32_t position = 0; position <= last; ++position) {
      const float confidence = lane.confidence[position];
      if (!(confidence >= 0.0F && confidence <= 1.0F))
        continue;
      const Label label{position, static_cast<float>(logit(confidence)), position < lane.accepted};
      if (labels_.size() < kLabels) {
        labels_.push_back(label);
      } else {
        labels_[next_] = label;
        next_ = (next_ + 1) % kLabels;
      }
    }
  }
  if (++cycles_ % kRefitCycles == 0 && !labels_.empty())
    acceptance_ = fit(labels(), acceptance_);
}

std::vector<VerifyBudget::Label> VerifyBudget::labels() const {
  std::vector<Label> ordered(labels_.begin() + static_cast<std::ptrdiff_t>(next_), labels_.end());
  ordered.insert(ordered.end(), labels_.begin(), labels_.begin() + static_cast<std::ptrdiff_t>(next_));
  return ordered;
}

VerifyBudget::Acceptance VerifyBudget::fit(std::span<const Label> labels, const Acceptance &start) {
  for (const Label &label : labels)
    if (label.position >= kPositions || !std::isfinite(label.feature))
      throw std::invalid_argument("invalid verify budget label");
  const Parameters prior = parametersOf(kPrior);
  Parameters theta = parametersOf(start);
  double current = objective(labels, theta, prior);
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    Parameters gradient{};
    std::array<Parameters, kParameters> hessian{};
    for (const Label &label : labels) {
      const uint32_t b = 1 + label.position;
      const double p = 1.0 / (1.0 + std::exp(-(theta[0] * label.feature + theta[b])));
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

} // namespace splash::model
