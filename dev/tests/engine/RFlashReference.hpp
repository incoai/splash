#pragma once

// A double-precision solver of RFlash's block program, the oracle of the
// relaxation kernel (decode_relax_dflash, metal/abi/Sampling.h RelaxParams).
// Per row the closed form r = p_hat / p = min(max(q / p, A x), A x
// e^{1 / Lam}), A = exp(-mu_j z / Lam_j), Lam_j = lam_j - alpha mu_j,
// lam_j = lam / R_j, mu_j = mu / R_j, with x fixed by
// sum p r = 1 over the candidates and the rest bucket (q = 0, z = log eps);
// the block's price mu is 0 where the constraint is slack and else the root
// of its violation on [0, mu_max], found by bisection; the reach weights
// w_j = prod_{i<j} a_i and runs R_j = 1 + a_{j+1} R_{j+1} start from the
// lossless acceptances and are refreshed after every pass (two at alpha 0,
// else three); a block without a root up to mu_max keeps the cap's solution
// and its constraint residual is accepted. The remaining constants (mu_max,
// the lesser of 0.99 lambda / alpha and 16 lambda; the tail floor; the
// smoothing eps) follow ops::RelaxationSetting.

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Sampling.h"
#include "ops/Relaxation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace splash::test {

constexpr uint32_t kRelaxCandidates = SPLASH_DRAFT_CANDIDATES;

// One drafted row on the draft's support: the target probability and the
// draft probability of each candidate slot, 0 both for a dead slot.
struct RelaxReferenceRow final {
  std::array<double, kRelaxCandidates> p{};
  std::array<double, kRelaxCandidates> q{};
};

struct RelaxReferenceResult final {
  // The relaxed probability of each candidate slot of each row, and each
  // row's rest ratio (1 without a tail); a block without a root up to the
  // cap keeps the cap's solution (status capped).
  std::vector<std::array<double, kRelaxCandidates>> relaxed;
  std::vector<double> rest;
  uint32_t status = SPLASH_RELAX_STATUS_SLACK;
};

namespace relax_detail {

// A row's live entries: its candidates with p > 0 and, last, its rest
// bucket when the tail is above the floor.
struct Row final {
  std::vector<double> p;
  std::vector<double> q;
  std::vector<double> z;
  std::vector<uint32_t> slots;
  bool tail = false;
};

inline Row makeRow(const RelaxReferenceRow &input, const ops::RelaxationSetting &setting) {
  Row row;
  double sum = 0.0;
  for (uint32_t k = 0; k < kRelaxCandidates; ++k) {
    if (!(input.p[k] > 0.0)) continue;
    row.p.push_back(input.p[k]);
    row.q.push_back(input.q[k]);
    row.slots.push_back(k);
    sum += input.p[k];
  }
  const double eps = setting.eps;
  for (size_t i = 0; i < row.p.size(); ++i) {
    const double z = std::log(((1.0 - eps) * row.q[i] + eps * row.p[i]) / std::max(row.p[i], 1e-30));
    row.z.push_back(std::clamp(z, -20.0, 20.0));
  }
  const double rest = std::max(1.0 - sum, 0.0);
  if (rest > setting.tailFloor) {
    row.p.push_back(rest);
    row.q.push_back(0.0);
    row.z.push_back(std::max(std::log(eps), -20.0));
    row.tail = true;
  }
  return row;
}

// One row's ratios r = p_hat / p at its prices, the normaliser by bisection
// in log x. The mass is compared in logs and the bracket widened until it
// holds the root: near the price cap the rest bucket's log A reaches the
// thousands, and the root lies as far below zero.
inline std::vector<double> solveRow(const Row &row, double lamR, double muR, double alpha) {
  const double Lam = lamR - alpha * muR;
  const size_t n = row.p.size();
  std::vector<double> lA(n), lk(n);
  double total = 0.0;
  for (size_t i = 0; i < n; ++i) {
    lA[i] = -muR * row.z[i] / Lam;
    lk[i] = row.q[i] > 0.0 ? std::log(std::max(row.q[i], 1e-300) / row.p[i]) : -std::numeric_limits<double>::infinity();
    total += row.p[i];
  }
  const auto ratios = [&](double lx) {
    std::vector<double> lr(n);
    for (size_t i = 0; i < n; ++i) lr[i] = std::min(std::max(lk[i], lA[i] + lx), lA[i] + lx + 1.0 / Lam);
    return lr;
  };
  const auto logMass = [&](double lx) {
    const std::vector<double> lr = ratios(lx);
    double top = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; ++i) top = std::max(top, std::log(row.p[i]) + lr[i]);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += std::exp(std::log(row.p[i]) + lr[i] - top);
    return top + std::log(sum);
  };
  const double target = std::log(total);
  // A row whose candidates' draft mass alone reaches the total has no root
  // below: the bracket then ends at its floor, where every candidate sits
  // on q / p.
  double a = -60.0, b = 60.0;
  while (a > -1e7 && logMass(a) >= target) a *= 2.0;
  while (b < 1e7 && logMass(b) < target) b *= 2.0;
  for (int iteration = 0; iteration < 200; ++iteration) {
    const double m = 0.5 * (a + b);
    (logMass(m) < target ? a : b) = m;
  }
  std::vector<double> r = ratios(0.5 * (a + b));
  for (double &value : r) value = std::exp(value);
  return r;
}

// A row's acceptance probability and slack n_j = alpha KL - drift at ratios r.
inline std::pair<double, double> rowStats(const Row &row, const std::vector<double> &r, double alpha) {
  double accept = 0.0, kl = 0.0, drift = 0.0;
  for (size_t i = 0; i < row.p.size(); ++i) {
    const double ph = row.p[i] * r[i];
    accept += std::min(ph, row.q[i]);
    kl += ph * std::log(std::max(r[i], 1e-300));
    drift += (ph - row.p[i]) * row.z[i];
  }
  return {accept, alpha * kl - drift};
}

} // namespace relax_detail

// The block program of `rows` under `setting`, as RFlash solves it.
inline RelaxReferenceResult relaxReference(std::span<const RelaxReferenceRow> inputs,
                                           const ops::RelaxationSetting &setting, int searchIterations = 45) {
  using relax_detail::Row;
  const size_t M = inputs.size();
  std::vector<Row> rows;
  for (const RelaxReferenceRow &input : inputs) rows.push_back(relax_detail::makeRow(input, setting));
  const double alpha = setting.alpha, lambda = setting.lambda, muMax = setting.muMax();
  std::vector<double> accepts(M);
  for (size_t j = 0; j < M; ++j) {
    double a = 0.0;
    for (size_t i = 0; i < rows[j].p.size(); ++i) a += std::min(rows[j].p[i], rows[j].q[i]);
    accepts[j] = a;
  }
  uint32_t status = SPLASH_RELAX_STATUS_SLACK;
  std::vector<std::vector<double>> solution(M);
  for (uint32_t pass = 0; pass < setting.passes(); ++pass) {
    std::vector<double> R(M, 1.0), w(M, 1.0);
    for (size_t j = M - 1; j-- > 0;) R[j] = 1.0 + accepts[j + 1] * R[j + 1];
    for (size_t j = 1; j < M; ++j) w[j] = w[j - 1] * accepts[j - 1];
    const auto solveAll = [&](double mu) {
      std::vector<std::vector<double>> all(M);
      for (size_t j = 0; j < M; ++j) all[j] = relax_detail::solveRow(rows[j], lambda / R[j], mu / R[j], alpha);
      return all;
    };
    const auto violation = [&](const std::vector<std::vector<double>> &all) {
      double f = 0.0;
      for (size_t j = 0; j < M; ++j) f -= w[j] * relax_detail::rowStats(rows[j], all[j], alpha).second;
      return f;
    };
    std::vector<std::vector<double>> at0 = solveAll(0.0);
    // A block without acceptance to gain (no candidate the draft favors
    // over the target) adds no drift: its violation is 0 up to the
    // normalisers' rounding, and it is slack.
    if (violation(at0) <= 1e-7) {
      status = SPLASH_RELAX_STATUS_SLACK;
      solution = std::move(at0);
    } else {
      std::vector<std::vector<double>> atCap = solveAll(muMax);
      if (violation(atCap) > 0.0) {
        // The next pass reweights from the cap's solution; a block still
        // without a root at the end keeps it, residual accepted.
        status = SPLASH_RELAX_STATUS_CAPPED;
        solution = std::move(atCap);
      } else {
        double lo = 0.0, hi = muMax;
        std::vector<std::vector<double>> feasible = std::move(atCap);
        for (int iteration = 0; iteration < searchIterations; ++iteration) {
          const double mid = 0.5 * (lo + hi);
          std::vector<std::vector<double>> atMid = solveAll(mid);
          if (violation(atMid) > 0.0) {
            lo = mid;
          } else {
            hi = mid;
            feasible = std::move(atMid);
          }
        }
        status = SPLASH_RELAX_STATUS_BOUND;
        solution = std::move(feasible);
      }
    }
    for (size_t j = 0; j < M; ++j) accepts[j] = relax_detail::rowStats(rows[j], solution[j], alpha).first;
  }
  RelaxReferenceResult result;
  result.status = status;
  result.relaxed.resize(M);
  result.rest.assign(M, 1.0);
  for (size_t j = 0; j < M; ++j) {
    const Row &row = rows[j];
    const size_t live = row.slots.size();
    for (size_t i = 0; i < live; ++i) {
      const double r = solution[j][i];
      result.relaxed[j][row.slots[i]] = row.p[i] * r;
    }
    if (row.tail) result.rest[j] = solution[j][live];
  }
  return result;
}

} // namespace splash::test
