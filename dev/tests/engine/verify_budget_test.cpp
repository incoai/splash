// The verify budget's online estimates (model::VerifyBudget). Its acceptance
// models, the greedy lanes' and the sampled lanes', are the prior before any
// label. A lane's tree labels each row whose parent its walk reached and
// selected at, with whether that selection is the row's token, a row past
// the verify rows too, but no row whose draft probability is not one (a row
// a host chain made, a draft row that is not finite) and none under a row
// past the selections, each carrying its row's cap; a cycle keeps the last
// kLabels of each policy apart.
// Every kRefitCycles-th cycle refits each model, which must equal an
// independent undamped Newton solution of the same posterior over its labels,
// from the online warm start and from the prior alike, and recover the model
// the labels were drawn from; a refit from the model a fit reached keeps it,
// and a model whose lanes labeled nothing stays as it was; a label its cap
// bounds tells the fit nothing, one under its cap moves it. The
// costs are the seed's shape before any cycle, then the affine fit of the
// latest cycles of each lane count and kind, positive and nondecreasing, the
// seed's slope where their rows do not spread; every kExploreCycles-th cycle
// of a lane count explores; a stall over rows that cluster leaves the cycles
// after it their tokens per millisecond. A cycle's rows (stepRows) are the
// most expected tokens per cost of every assignment of rows to its lanes'
// chances that its rows and their caps allow, the fewest among equal ones.
#include "TestChecks.hpp"
#include "model/VerifyBudget.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using splash::model::VerifyBudget;
using splash::test::rejects;
using splash::test::require;

namespace {

constexpr uint32_t kPositions = VerifyBudget::kPositions;
constexpr uint32_t kParameters = 1 + kPositions;
using Parameters = std::array<double, kParameters>;

double logitOf(double p) {
  p = std::clamp(p, 1.0e-6, 1.0 - 1.0e-6);
  return std::log(p / (1.0 - p));
}

Parameters parametersOf(const VerifyBudget::Acceptance &a) {
  Parameters p{a.slope};
  for (uint32_t j = 0; j < kPositions; ++j)
    p[1 + j] = a.intercepts[j];
  return p;
}

// The posterior's mode by plain Newton steps to convergence: each label's
// design row is (feature, one-hot position), the Hessian dense, the system
// solved by Gauss-Jordan elimination.
Parameters referenceFit(const std::vector<VerifyBudget::Label> &labels) {
  const Parameters prior = parametersOf(VerifyBudget::kPrior);
  Parameters theta = prior;
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::array<std::array<double, kParameters + 1>, kParameters> system{};
    for (const auto &label : labels) {
      Parameters x{};
      x[0] = label.feature;
      x[1 + label.position] = 1.0;
      double z = 0.0;
      for (uint32_t i = 0; i < kParameters; ++i)
        z += theta[i] * x[i];
      const double p = 1.0 / (1.0 + std::exp(-z));
      for (uint32_t i = 0; i < kParameters; ++i) {
        system[i][kParameters] += (p - (label.accepted ? 1.0 : 0.0)) * x[i];
        for (uint32_t k = 0; k < kParameters; ++k)
          system[i][k] += p * (1.0 - p) * x[i] * x[k];
      }
    }
    for (uint32_t i = 0; i < kParameters; ++i) {
      system[i][kParameters] += VerifyBudget::priorPrecision(i) * (theta[i] - prior[i]);
      system[i][i] += VerifyBudget::priorPrecision(i);
    }
    for (uint32_t column = 0; column < kParameters; ++column) {
      const double pivot = system[column][column];
      for (uint32_t k = 0; k <= kParameters; ++k)
        system[column][k] /= pivot;
      for (uint32_t row = 0; row < kParameters; ++row)
        if (row != column) {
          const double factor = system[row][column];
          for (uint32_t k = 0; k <= kParameters; ++k)
            system[row][k] -= factor * system[column][k];
        }
    }
    double largest = 0.0;
    for (uint32_t i = 0; i < kParameters; ++i) {
      theta[i] -= system[i][kParameters];
      largest = std::max(largest, std::fabs(system[i][kParameters]));
    }
    if (largest < 1.0e-12)
      break;
  }
  return theta;
}

double distance(const VerifyBudget::Acceptance &a, const Parameters &b) {
  const Parameters p = parametersOf(a);
  double largest = 0.0;
  for (uint32_t i = 0; i < kParameters; ++i)
    largest = std::max(largest, std::fabs(p[i] - b[i]));
  return largest;
}

// A refit of `labels` from the model `fitted` a fit of them reached keeps
// it, to its parameters' float rounding: there a Newton step changes the
// objective by no more than its rounding, and one that no halving lets lower
// it ends the fit where it stands.
void refitKeeps(const std::vector<VerifyBudget::Label> &labels, const VerifyBudget::Acceptance &fitted) {
  const double moved = distance(VerifyBudget::fit(labels, fitted), parametersOf(fitted));
  require(moved < 1e-6, "a refit from the model a fit reached moved it by " + std::to_string(moved));
}

// Greedy lanes drawn from a known model: every row verified in three lanes
// of four and 1 to 8 rows in the others, each with six labels at positions
// drawn alike, their draft probabilities spread over (0, 1), accepted with
// the model's probabilities.
struct Generator final {
  std::mt19937_64 random{0x7e51f1};
  Parameters truth;
  VerifyBudget::Lane lane() {
    std::normal_distribution<double> feature(1.5, 2.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    VerifyBudget::Lane lane;
    lane.verifyRows = random() % 4 ? 8 : 1 + static_cast<uint32_t>(random() % 8);
    for (uint32_t label = 0; label < 6; ++label) {
      const uint32_t position = static_cast<uint32_t>(random() % kPositions);
      const float logit = static_cast<float>(logitOf(1.0 / (1.0 + std::exp(-feature(random)))));
      const double z = truth[0] * logit + truth[1 + position];
      lane.labels.push_back({position, logit, unit(random) < 1.0 / (1.0 + std::exp(-z))});
    }
    return lane;
  }
};

void priorAndExploration() {
  VerifyBudget budget;
  require(budget.acceptance(false) == VerifyBudget::kPrior && budget.acceptance(true) == VerifyBudget::kPrior,
          "a new budget is not the prior");
  for (uint32_t lanes = 1; lanes <= VerifyBudget::kLanes; ++lanes) {
    const auto costs = budget.costs(lanes, false);
    for (uint32_t rows = 0; rows <= VerifyBudget::kBlockRows * VerifyBudget::kLanes; ++rows) {
      const bool inside = rows >= lanes && rows <= lanes * VerifyBudget::kBlockRows;
      require(inside ? std::fabs(costs[rows] - VerifyBudget::seed(lanes, rows)) < 1e-6 : costs[rows] == 0.0F,
              "a new budget's costs are not the seed's shape");
      require(!inside || rows == lanes || costs[rows] >= costs[rows - 1], "the seed's costs decrease");
    }
    require(std::fabs(costs[lanes * VerifyBudget::kBlockRows] - 1.0F) < 1e-6, "every row does not cost 1");
  }
  VerifyBudget::Lane lane;
  lane.verifyRows = 8;
  for (uint32_t cycle = 0; cycle < 3 * VerifyBudget::kExploreCycles; ++cycle) {
    require(budget.explores(2, false) == (cycle % VerifyBudget::kExploreCycles == VerifyBudget::kExploreCycles - 1),
            "two-lane cycles explore at another cadence");
    require(!budget.explores(1, false) && !budget.explores(2, true), "other cycles' kinds explored");
    const std::array lanes{lane, lane};
    budget.observe(lanes, false, 10.0);
  }
}

// A tree in its selector's order, after a walk that selected 11 at the
// anchor's row, 14 at row 1 and 16 at row 4, and nothing it kept after that,
// each label carrying its row's cap:
//   row 1 (11, under the anchor, 0.6)  on the path, accepted at position 0;
//   row 2 (12, under the anchor, 0.3, cap 0.45) rejected at position 0;
//   row 3 (13, under row 1, 0.5, cap 0.7) rejected at position 1;
//   row 4 (14, under row 1, a host chain's: no probability) on the path;
//   row 5 (15, under row 2, 0.4)       its parent off the path: no label;
//   row 6 (16, under row 4, 0.7)       accepted at position 2, a row past
//                                      the lane's verify rows or not alike;
//   row 7 (17, under row 6, 0.2)       under a row past the selections;
//   rows 8 and 9 (under row 4: NaN and infinity) not probabilities;
//   row 10 (20, under row 4, 0.25, a NaN cap) no cap.
void treeLabelsOfAWalk() {
  constexpr uint16_t kRoot = SPLASH_TREE_ROOT;
  const std::vector<uint32_t> tokens{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  const std::vector<splash::ops::TreeRow> rows{{1, kRoot, 0}, {3, 0, 1},   {5, 0, 1},   {11, 1, 2},
                                               {19, 1, 2},    {37, 2, 2},  {83, 4, 3},  {211, 6, 4},
                                               {275, 4, 3},   {531, 4, 3}, {1043, 4, 3}};
  const std::vector<float> probabilities{1.0F, 0.6F, 0.3F, 0.5F, -1.0F, 0.4F, 0.7F, 0.2F, NAN, INFINITY, 0.25F};
  const std::vector<float> caps{1.0F, 1.0F, 0.45F, 0.7F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, NAN};
  std::vector<float> scores;
  for (const float probability : probabilities)
    scores.insert(scores.end(), {probability, 0.5F});
  const std::vector<uint32_t> selections{11, 14, 16};
  const auto labels = VerifyBudget::treeLabels(tokens, rows, scores, caps, selections);
  const std::vector<VerifyBudget::Label> expected{{0, float(logitOf(0.6F)), true},
                                                  {0, float(logitOf(0.3F)), false, 0.45F},
                                                  {1, float(logitOf(0.5F)), false, 0.7F},
                                                  {2, float(logitOf(0.7F)), true}};
  require(labels == expected, "a walk's tree labeled " + std::to_string(labels.size()) + " rows, not its own four");
  // The walk's first selection alone labels the anchor's children only.
  require(VerifyBudget::treeLabels(tokens, rows, scores, caps, std::span(selections).first(1)) ==
              std::vector<VerifyBudget::Label>(expected.begin(), expected.begin() + 2),
          "a walk of one selection labeled rows below its anchor's children");
  require(VerifyBudget::treeLabels(tokens, rows, scores, caps, {}).empty(), "a walk of no selection labeled rows");

  // A cycle keeps a greedy lane's labels and a sampled lane's apart, and
  // without a valid time still labels, leaving the costs alone.
  VerifyBudget budget;
  const VerifyBudget::Lane greedy{8, false, labels}, sampled{3, true, {expected[1], expected[3]}};
  budget.observe(std::array{greedy, sampled, greedy}, false, 0.0);
  std::vector<VerifyBudget::Label> both = labels;
  both.insert(both.end(), labels.begin(), labels.end());
  require(budget.labels(false) == both && budget.labels(true) == sampled.labels,
          "a cycle did not keep its greedy and sampled lanes' labels apart");
  require(budget.costs(4, false) == VerifyBudget().costs(4, false), "a cycle without a time changed the costs");
}

// Greedy lanes whose rows the target always accepts and sampled lanes whose
// rows it never does: each policy's refit follows its own labels alone, and
// one whose lanes labeled nothing keeps its model.
void policiesFitApart() {
  VerifyBudget budget;
  VerifyBudget::Lane greedy{8, false, {}}, sampled{8, true, {}};
  for (uint32_t position = 0; position < kPositions; ++position) {
    greedy.labels.push_back({position, 0.5F, true});
    sampled.labels.push_back({position, 0.5F, false});
  }
  for (uint32_t cycle = 0; cycle < VerifyBudget::kRefitCycles; ++cycle)
    budget.observe(std::array{greedy, sampled}, false, 10.0);
  const VerifyBudget::Acceptance up = budget.acceptance(false), down = budget.acceptance(true);
  require(up == VerifyBudget::fit(budget.labels(false), VerifyBudget::kPrior) &&
              down == VerifyBudget::fit(budget.labels(true), VerifyBudget::kPrior),
          "a refit did not fit each policy's labels");
  for (uint32_t position = 0; position < kPositions; ++position)
    require(up.intercepts[position] > VerifyBudget::kPrior.intercepts[position] &&
                down.intercepts[position] < VerifyBudget::kPrior.intercepts[position],
            "a policy's refit did not follow its own lanes at position " + std::to_string(position));
  // Greedy lanes alone: the sampled model stays.
  for (uint32_t cycle = 0; cycle < VerifyBudget::kRefitCycles; ++cycle)
    budget.observe(std::array{greedy}, false, 10.0);
  require(budget.acceptance(true) == down && budget.acceptance(false) != up,
          "a refit of the greedy lanes' labels moved the sampled lanes' model");
}

void fitsTheLabels() {
  Generator generator;
  generator.truth = {0.85, 0.8, 0.6, 0.4, 0.4, 0.3, 0.3, 0.2};
  VerifyBudget budget;
  // Enough cycles that the window drops the oldest labels; the last refits.
  for (uint32_t cycle = 0; cycle < 40 * VerifyBudget::kRefitCycles; ++cycle) {
    std::vector<VerifyBudget::Lane> lanes(1 + cycle % VerifyBudget::kLanes);
    for (auto &lane : lanes)
      lane = generator.lane();
    budget.observe(lanes, cycle % 3 == 0, 50.0);
  }
  const auto labels = budget.labels(false);
  require(labels.size() == VerifyBudget::kLabels && budget.labels(true).empty(),
          "the windows hold " + std::to_string(labels.size()) + " greedy labels and " +
              std::to_string(budget.labels(true).size()) + " sampled ones");
  const VerifyBudget::Acceptance &online = budget.acceptance(false);
  require(budget.acceptance(true) == VerifyBudget::kPrior, "greedy lanes moved the sampled lanes' model");
  const Parameters reference = referenceFit(labels);
  require(distance(online, reference) < 2e-4,
          "the online fit differs from the reference by " + std::to_string(distance(online, reference)));
  const VerifyBudget::Acceptance cold = VerifyBudget::fit(labels, VerifyBudget::kPrior);
  require(distance(cold, reference) < 2e-4,
          "the fit from the prior differs from the reference by " + std::to_string(distance(cold, reference)));
  refitKeeps(labels, cold);
  require(std::fabs(online.slope - generator.truth[0]) < 0.1, "the fit missed the slope");
  // Each intercept within three standard errors of its labels (whose p(1 - p)
  // is at least about 0.2), from at least 100 of them.
  for (uint32_t j = 0; j < kPositions; ++j) {
    const auto count = std::count_if(labels.begin(), labels.end(), [&](const auto &label) { return label.position == j; });
    const double error = std::fabs(online.intercepts[j] - generator.truth[1 + j]);
    require(count >= 100 && error < 3.0 / std::sqrt(0.2 * double(count)),
            "the fit missed the intercept at position " + std::to_string(j) + " by " + std::to_string(error) +
                " over " + std::to_string(count) + " labels");
  }
  // A handful of labels stays near the prior.
  const std::vector<VerifyBudget::Label> few{{0, 2.0F, false}, {1, -1.0F, true}, {0, 0.5F, true}};
  const VerifyBudget::Acceptance shrunk = VerifyBudget::fit(few, VerifyBudget::kPrior);
  require(distance(shrunk, referenceFit(few)) < 2e-4, "the fit of a few labels differs from the reference");
  require(std::fabs(shrunk.slope - VerifyBudget::kPrior.slope) < 0.3, "a few labels moved the slope far");
  refitKeeps(few, shrunk);
  // Labels whose caps bound their chances (rows a likelier sibling left a
  // thousandth of its parent's) tell the fit nothing, accepted or not; those
  // under their caps (a cap of 0.9 over a chance near the fit's 0.5 at no
  // feature) move it like any.
  std::vector<VerifyBudget::Label> capped = labels;
  for (uint32_t position = 0; position < kPositions; ++position)
    for (const bool accepted : {false, true})
      capped.push_back({position, 1.0F, accepted, 1.0e-3F});
  require(distance(VerifyBudget::fit(capped, VerifyBudget::kPrior), parametersOf(cold)) < 1e-9,
          "labels their caps bound moved the fit");
  std::vector<VerifyBudget::Label> under = labels;
  for (int repeat = 0; repeat < 50; ++repeat)
    under.push_back({0, 0.0F, false, 0.9F});
  const VerifyBudget::Acceptance lowered = VerifyBudget::fit(under, VerifyBudget::kPrior);
  require(lowered.intercepts[0] < cold.intercepts[0] - 0.05F,
          "labels under their caps did not move the fit (intercept " + std::to_string(lowered.intercepts[0]) +
              " against " + std::to_string(cold.intercepts[0]) + ")");
}

void learnsTheCosts() {
  VerifyBudget budget;
  VerifyBudget::Lane lane;
  const auto cycle = [&](uint32_t lanes, uint32_t rows, bool constrained, double milliseconds) {
    std::vector<VerifyBudget::Lane> batch(lanes, lane);
    for (uint32_t i = 0; i < lanes; ++i)
      batch[i].verifyRows = rows / lanes + (i < rows % lanes ? 1 : 0);
    budget.observe(batch, constrained, milliseconds);
  };
  const auto affine = [](double a, double b) { return [=](uint32_t rows) { return a + b * rows; }; };
  const auto follows = [&](uint32_t lanes, bool constrained, const auto &time, const char *what) {
    const auto costs = budget.costs(lanes, constrained);
    for (uint32_t rows = lanes; rows <= lanes * VerifyBudget::kBlockRows; ++rows)
      require(std::fabs(costs[rows] - time(rows)) < 0.02 * time(rows),
              std::string(what) + " at " + std::to_string(rows) + " rows: " + std::to_string(costs[rows]) +
                  " against " + std::to_string(time(rows)));
  };
  // Cycles over spread rows give back their affine time.
  const auto shallow = affine(20.0, 2.5);
  for (int repeat = 0; repeat < 40; ++repeat)
    for (const uint32_t rows : {3U, 6U, 9U, 12U, 16U})
      cycle(2, rows, false, shallow(rows));
  follows(2, false, shallow, "the costs missed an affine time");
  // The latest requests price a row: a steeper time takes over.
  const auto steep = affine(30.0, 8.0);
  for (int repeat = 0; repeat < 40; ++repeat)
    for (const uint32_t rows : {3U, 6U, 9U, 12U, 16U})
      cycle(2, rows, false, steep(rows));
  follows(2, false, steep, "the costs kept an older request's price");
  require(budget.costs(2, true) == VerifyBudget().costs(2, true) &&
              budget.costs(3, false) == VerifyBudget().costs(3, false),
          "measured cycles changed another lane count's or kind's costs");
  // Without a spread of rows, the slope is the seed's at the cycles' level.
  for (int repeat = 0; repeat < 100; ++repeat)
    cycle(1, 8, false, 50.0);
  const auto flat = budget.costs(1, false);
  for (uint32_t rows = 1; rows <= VerifyBudget::kBlockRows; ++rows) {
    const double expected = 50.0 * (VerifyBudget::seed(1, 1) + (rows - 1) * (1.0 - VerifyBudget::seed(1, 1)) / 7.0);
    require(std::fabs(flat[rows] - expected) < 1e-3 * expected, "the costs of unspread rows left the seed's slope");
  }
  // A stall counts as at most twice the larger of its fitted price and the
  // cycles' mean.
  const auto before = budget.costs(2, false);
  cycle(2, 9, false, 50.0 * steep(9));
  const auto after = budget.costs(2, false);
  require(after[9] < before[9] * 1.04F, "a stall moved the costs by more than its bounded share");
  // A time that falls with the rows prices every total alike; a negative
  // intercept leaves the fewest rows positive.
  for (int repeat = 0; repeat < 60; ++repeat) {
    cycle(4, 10, true, 50.0);
    cycle(4, 12, true, 40.0);
  }
  const auto falling = budget.costs(4, true);
  for (uint32_t rows = 4; rows < 32; ++rows)
    require(falling[rows + 1] == falling[rows] && falling[rows] > 0.0F, "a falling time gave unequal costs");
  for (int repeat = 0; repeat < 60; ++repeat)
    for (const uint32_t rows : {16U, 24U, 32U})
      cycle(4, rows, false, -100.0 + 10.0 * rows);
  const auto negative = budget.costs(4, false);
  for (uint32_t rows = 4; rows <= 32; ++rows)
    require(negative[rows] > 0.0F && (rows == 4 || negative[rows] >= negative[rows - 1]),
            "a negative intercept gave a cost that is not positive or decreases");
}

// Two lanes whose proposals each survive 0.75 of the time, over cycles of the
// seed's shape, verify the total of the most expected tokens per cost, which
// draft_verify_rows takes: 10 rows in all, so the fit's rows cluster. A stall
// that doubles the second exploring cycle tilts the fit's line below zero at
// the anchors; the 64 cycles after it keep at least 0.9 of the tokens per
// millisecond the cycles of a run without the stall give.
void stallOverClusteredRows() {
  constexpr uint32_t kLanes = 2, kEvery = kLanes * VerifyBudget::kBlockRows, kAfter = 64;
  constexpr double kSurvival = 0.75;
  const auto time = [](uint32_t rows) { return 50.0 * VerifyBudget::seed(kLanes, rows); };
  // The anchors, then each lane's next proposal in turn.
  const auto expected = [&](uint32_t rows) {
    double tokens = kLanes;
    for (uint32_t row = kLanes; row < rows; ++row)
      tokens += std::pow(kSurvival, 1 + (row - kLanes) / kLanes);
    return tokens;
  };
  const auto choice = [&](const auto &cost) {
    uint32_t best = kLanes;
    for (uint32_t rows = kLanes + 1; rows <= kEvery; ++rows)
      if (expected(rows) / cost(rows) > expected(best) / cost(best))
        best = rows;
    return best;
  };
  require(choice(time) == 10, "the cycles' times did not make 10 rows the best total");
  const auto tokensPerMillisecond = [&](bool stall) {
    VerifyBudget budget;
    VerifyBudget::Lane lane;
    double tokens = 0.0, milliseconds = 0.0;
    for (uint32_t explored = 0, after = 0; after < kAfter;) {
      const auto costs = budget.costs(kLanes, false);
      const bool explores = budget.explores(kLanes, false);
      const uint32_t rows = explores ? kEvery : choice([&](uint32_t total) { return double(costs[total]); });
      const bool second = explores && ++explored == 2;
      if (explored >= 2 && !second) {
        ++after;
        tokens += expected(rows);
        milliseconds += time(rows);
      }
      std::array batch{lane, lane};
      batch[0].verifyRows = rows - rows / 2;
      batch[1].verifyRows = rows / 2;
      budget.observe(batch, false, (stall && second ? 2.0 : 1.0) * time(rows));
    }
    return tokens / milliseconds;
  };
  const double ratio = tokensPerMillisecond(true) / tokensPerMillisecond(false);
  require(ratio >= 0.9, "the cycles after a stall gave " + std::to_string(ratio) +
                            " of the tokens per millisecond of a run without it");
}

void invalidInputs() {
  VerifyBudget budget;
  VerifyBudget::Lane lane;
  lane.verifyRows = 8;
  rejects([&] { budget.observe({}, false, 1.0); }, "invalid verify budget cycle", "an empty cycle was observed");
  const std::vector<VerifyBudget::Lane> five(5, lane);
  rejects([&] { budget.observe(five, false, 1.0); }, "invalid verify budget cycle", "five lanes were observed");
  for (const uint32_t rows : {0U, 9U}) {
    VerifyBudget::Lane bad = lane;
    bad.verifyRows = rows;
    rejects([&] { budget.observe(std::array{bad}, false, 1.0); }, "invalid verify budget lane",
            "an impossible lane was observed");
  }
  for (const VerifyBudget::Label &label : {VerifyBudget::Label{kPositions, 0.0F, true}, {0, NAN, true}}) {
    VerifyBudget::Lane bad = lane;
    bad.labels = {label};
    rejects([&] { budget.observe(std::array{bad}, false, 1.0); }, "invalid verify budget label",
            "an impossible label was observed");
  }
  require(budget.labels(false).empty(), "a refused cycle kept labels");
  // Trees of no row, of rows short of their tokens or scores, and of a row
  // before its parent.
  const std::vector<splash::ops::TreeRow> two{{1, SPLASH_TREE_ROOT, 0}, {3, 0, 1}}, backwards{{1, SPLASH_TREE_ROOT, 0},
                                                                                          {3, 1, 1}};
  const std::vector<uint32_t> tokens{1, 2}, one{1};
  const std::vector<float> scores{1.0F, 1.0F, 0.5F, 0.5F}, caps{1.0F, 1.0F};
  using Tree = std::tuple<std::span<const splash::ops::TreeRow>, std::span<const uint32_t>, std::span<const float>,
                          std::span<const float>>;
  for (const auto &[rows, ids, pairs, rowCaps] :
       {Tree{}, Tree{two, one, scores, caps}, Tree{two, tokens, std::span(scores).first(3), caps},
        Tree{two, tokens, scores, std::span(caps).first(1)}, Tree{backwards, tokens, scores, caps}})
    rejects([&] { static_cast<void>(VerifyBudget::treeLabels(ids, rows, pairs, rowCaps, tokens)); },
            "invalid verify budget tree", "an impossible tree was labeled");
  for (const uint32_t lanes : {0U, 5U}) {
    rejects([&] { static_cast<void>(budget.costs(lanes, false)); }, "invalid verify budget lane count",
            "costs of an impossible lane count were given");
    rejects([&] { static_cast<void>(budget.explores(lanes, false)); }, "invalid verify budget lane count",
            "exploration of an impossible lane count was given");
  }
  for (const auto &[lanes, rows] : {std::pair{2U, 1U}, std::pair{2U, 17U}, std::pair{0U, 0U}})
    rejects([&] { static_cast<void>(VerifyBudget::seed(lanes, rows)); }, "invalid verify budget rows",
            "a seed outside a batch's rows was given");
  const std::vector<VerifyBudget::Label> past{{kPositions, 0.0F, true}};
  rejects([&] { static_cast<void>(VerifyBudget::fit(past, VerifyBudget::kPrior)); }, "invalid verify budget label",
          "a label past the proposals was fit");
  const std::vector<VerifyBudget::Label> infinite{{0, INFINITY, true}};
  rejects([&] { static_cast<void>(VerifyBudget::fit(infinite, VerifyBudget::kPrior)); }, "invalid verify budget label",
          "a label without a finite feature was fit");
  for (const float cap : {-0.5F, 1.5F, NAN}) {
    const std::vector<VerifyBudget::Label> uncapped{{0, 0.0F, true, cap}};
    rejects([&] { static_cast<void>(VerifyBudget::fit(uncapped, VerifyBudget::kPrior)); },
            "invalid verify budget label", "a label of a cap past [0, 1] was fit");
  }
}

// The most expected tokens past the anchors of any assignment of at most e
// rows to the lanes, for every e, each lane's a prefix of its curve and at
// most cap - 1 rows, by brute force.
std::vector<double> bestAssignments(const std::vector<std::vector<float>> &curves, uint32_t cap) {
  const size_t lanes = curves.size();
  std::vector<uint32_t> taken(lanes, 0);
  std::vector<double> best(lanes * cap + 1, 0.0);
  for (bool more = true; more;) {
    uint32_t total = 0;
    double expected = 0.0;
    for (size_t lane = 0; lane < lanes; ++lane) {
      total += taken[lane];
      for (uint32_t row = 0; row < taken[lane]; ++row)
        expected += curves[lane][row];
    }
    best[total] = std::max(best[total], expected);
    more = false;
    for (size_t lane = 0; lane < lanes && !more; ++lane) {
      if (taken[lane] < std::min<size_t>(curves[lane].size(), cap - 1)) {
        ++taken[lane];
        more = true;
      } else {
        taken[lane] = 0;
      }
    }
  }
  for (size_t extra = 1; extra < best.size(); ++extra)
    best[extra] = std::max(best[extra], best[extra - 1]);
  return best;
}

void cycleRowsChoice() {
  constexpr uint32_t kBlock = SPLASH_TARGET_VERIFY_ROWS;
  std::mt19937_64 random(0x57e9a1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (uint32_t trial = 0; trial < 240; ++trial) {
    const uint32_t lanes = 1 + trial % VerifyBudget::kLanes;
    // Curves of 0 to a block's rows past the anchor, largest first, an exact
    // 1 or 0 now and then.
    std::vector<std::vector<float>> curves(lanes);
    for (std::vector<float> &curve : curves) {
      curve.resize(random() % kBlock);
      for (float &chance : curve) {
        const uint64_t kind = random() % 8;
        chance = kind == 0 ? 1.0F : kind == 1 ? 0.0F : static_cast<float>(unit(random));
      }
      std::ranges::sort(curve, std::greater<>());
    }
    // Flat, linear and step-shaped costs over 16-row and 32-row tiles, and
    // random nondecreasing ones; now and then a table that ends early.
    const uint32_t ends = trial % 7 == 6 ? lanes + static_cast<uint32_t>(random() % (lanes * (kBlock - 1) + 1))
                                          : lanes * kBlock;
    std::vector<float> costs(ends + 1, 0.0F);
    float level = 1.0F;
    for (uint32_t rows = lanes; rows <= ends; ++rows) {
      switch (trial % 4) {
      case 0: costs[rows] = 1.0F; break;
      case 1: costs[rows] = 10.0F + 0.4F * float(rows); break;
      case 2: costs[rows] = rows <= 16 ? 1.0F : rows <= 32 ? 1.1F : 1.6F; break;
      default: level += static_cast<float>(unit(random)) * 0.1F; costs[rows] = level; break;
      }
    }
    const splash::ops::StepRows got = VerifyBudget::stepRows(curves, costs);
    double best = 0.0;
    uint32_t chosen = lanes;
    std::vector<double> values(ends + 1, 0.0);
    const std::vector<double> narrow = bestAssignments(curves, splash::ops::verifyLaneCap(lanes, lanes)),
                              wide = bestAssignments(curves, splash::ops::verifyLaneCap(lanes * kBlock, lanes));
    for (uint32_t rows = lanes; rows <= ends; ++rows) {
      const uint32_t cap = splash::ops::verifyLaneCap(rows, lanes);
      const std::vector<double> &assigned = cap == splash::ops::verifyLaneCap(lanes, lanes) ? narrow : wide;
      values[rows] = (lanes + assigned[std::min<size_t>(rows - lanes, assigned.size() - 1)]) / costs[rows];
      if (values[rows] > best * (1.0 + 1e-9)) {
        best = values[rows];
        chosen = rows;
      }
    }
    const std::string label = "cycle rows trial " + std::to_string(trial);
    require(got.lanes == lanes && got.laneCap == splash::ops::verifyLaneCap(got.rows, lanes),
            label + ": the step's lanes or cap are not its rows'");
    require(got.rows >= lanes && got.rows <= ends && std::fabs(values[got.rows] - best) <= 1e-9 * best &&
                got.rows <= chosen,
            label + ": chose " + std::to_string(got.rows) + " rows where " + std::to_string(chosen) + " give " +
                std::to_string(best) + " tokens per cost");
  }
  // One lane whose tree would take a block's rows, all certain, at a flat
  // cost: every row of a block.
  const std::vector<std::vector<float>> certain{std::vector<float>(kBlock - 1, 1.0F)};
  require(VerifyBudget::stepRows(certain, std::vector<float>(kBlock + 1, 1.0F)).rows == kBlock,
          "a lane of certain rows at a flat cost did not take a block's rows");
  // Four lanes of 0.9 a step, whose rows past 32 cost 60% more than those
  // up to it: 32 rows, eight a lane.
  std::vector<std::vector<float>> steady(4);
  for (std::vector<float> &curve : steady)
    for (uint32_t row = 1; row < kBlock; ++row)
      curve.push_back(static_cast<float>(std::pow(0.9, row)));
  std::vector<float> stepped(4 * kBlock + 1);
  for (uint32_t rows = 4; rows <= 4 * kBlock; ++rows)
    stepped[rows] = rows <= 32 ? 1.0F : 1.6F;
  const splash::ops::StepRows steps = VerifyBudget::stepRows(steady, stepped);
  require(steps.rows == 32 && steps.laneCap == 8, "a cost step past 32 rows did not stop the rows at 32");
  // A lane's chances: its tree's and its lookup chain's, largest first, at
  // most a block's rows past its anchor.
  require(VerifyBudget::laneChances(std::vector<float>{0.9F, 0.5F, 0.2F}, std::vector<float>{0.8F, 0.6F}) ==
              std::vector<float>{0.9F, 0.8F, 0.6F, 0.5F, 0.2F},
          "a lane's chances did not merge its tree's and its lookup's");
  require(VerifyBudget::laneChances(std::vector<float>(kBlock, 0.5F), std::vector<float>(kBlock, 0.7F)).size() ==
              kBlock - 1,
          "a lane's chances ran past a block's rows");
  const std::vector<std::vector<float>> none, five(5);
  rejects([&] { static_cast<void>(VerifyBudget::stepRows(none, stepped)); }, "invalid cycle rows lanes",
          "a cycle of no lanes chose rows");
  rejects([&] { static_cast<void>(VerifyBudget::stepRows(five, stepped)); }, "invalid cycle rows lanes",
          "a cycle of five lanes chose rows");
  rejects([&] { static_cast<void>(VerifyBudget::stepRows(steady, std::vector<float>(4, 1.0F))); },
          "invalid cycle rows cost", "costs that end before the lanes' anchors were taken");
  std::vector<float> holed = stepped;
  holed[20] = 0.0F;
  rejects([&] { static_cast<void>(VerifyBudget::stepRows(steady, holed)); }, "invalid cycle rows cost",
          "a cost that is not positive was taken");
}

} // namespace

int main() {
  try {
    priorAndExploration();
    treeLabelsOfAWalk();
    policiesFitApart();
    fitsTheLabels();
    learnsTheCosts();
    stallOverClusteredRows();
    invalidInputs();
    cycleRowsChoice();
  } catch (const std::exception &error) {
    std::cerr << "verify_budget_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "verify_budget_test: PASS\n";
  return 0;
}
