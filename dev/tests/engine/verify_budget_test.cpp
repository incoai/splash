// The verify budget's online estimates (model::VerifyBudget) and the step
// costs a cycle's rows follow (model::StepCosts). The budget's acceptance
// models, the greedy lanes' and the sampled lanes', are the prior before any
// label. A lane's tree labels each row whose parent its walk reached and
// selected at, with whether that selection is the row's token, a row past
// the verify rows too, but no row whose draft probability is not one (a row
// a host chain made, a draft row that is not finite) and none under a row
// past the selections; a cycle keeps the last kLabels of each policy apart.
// Every kRefitCycles-th cycle refits each model, which must equal an
// independent undamped Newton solution of the same posterior over its labels,
// from the online warm start and from the prior alike, and recover the model
// the labels were drawn from; a refit from the model a fit reached keeps it,
// and a model whose lanes labeled nothing stays as it was; a label its cap
// bounds tells the fit nothing, one under its cap moves it. A step table's
// point costs the median of its replays; a
// cycle's step is, of every step the lanes' chances fill, the most expected
// tokens per cost by its class's point and the wide lane cap's price where
// its allocation gives a lane more than a draft's block, the fewest rows
// among equals, against a brute force; before its lane count is measured it
// is a draft's block a lane; a table's line recalls its costs, and a line of
// another table's points none.
#include "TestChecks.hpp"
#include "model/StepCosts.hpp"
#include "model/VerifyBudget.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using splash::model::StepCosts;
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

// Greedy lanes drawn from a known model, each with six labels at positions
// drawn alike, their draft probabilities spread over (0, 1), accepted with
// the model's probabilities.
struct Generator final {
  std::mt19937_64 random{0x7e51f1};
  Parameters truth;
  VerifyBudget::Lane lane() {
    std::normal_distribution<double> feature(1.5, 2.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    VerifyBudget::Lane lane;
    for (uint32_t label = 0; label < 6; ++label) {
      const uint32_t position = static_cast<uint32_t>(random() % kPositions);
      const float logit = static_cast<float>(logitOf(1.0 / (1.0 + std::exp(-feature(random)))));
      const double z = truth[0] * logit + truth[1 + position];
      lane.labels.push_back({position, logit, unit(random) < 1.0 / (1.0 + std::exp(-z))});
    }
    return lane;
  }
};

void startsAtThePrior() {
  const VerifyBudget budget;
  require(budget.acceptance(false) == VerifyBudget::kPrior && budget.acceptance(true) == VerifyBudget::kPrior,
          "a new budget is not the prior");
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

  // A cycle keeps a greedy lane's labels and a sampled lane's apart.
  VerifyBudget budget;
  const VerifyBudget::Lane greedy{false, labels}, sampled{true, {expected[1], expected[3]}};
  budget.observe(std::array{greedy, sampled, greedy});
  std::vector<VerifyBudget::Label> both = labels;
  both.insert(both.end(), labels.begin(), labels.end());
  require(budget.labels(false) == both && budget.labels(true) == sampled.labels,
          "a cycle did not keep its greedy and sampled lanes' labels apart");
}

// Greedy lanes whose rows the target always accepts and sampled lanes whose
// rows it never does: each policy's refit follows its own labels alone, and
// one whose lanes labeled nothing keeps its model.
void policiesFitApart() {
  VerifyBudget budget;
  VerifyBudget::Lane greedy{false, {}}, sampled{true, {}};
  for (uint32_t position = 0; position < kPositions; ++position) {
    greedy.labels.push_back({position, 0.5F, true});
    sampled.labels.push_back({position, 0.5F, false});
  }
  for (uint32_t cycle = 0; cycle < VerifyBudget::kRefitCycles; ++cycle)
    budget.observe(std::array{greedy, sampled});
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
    budget.observe(std::array{greedy});
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
    budget.observe(lanes);
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

void invalidInputs() {
  VerifyBudget budget;
  const VerifyBudget::Lane lane;
  rejects([&] { budget.observe({}); }, "invalid verify budget cycle", "an empty cycle was observed");
  const std::vector<VerifyBudget::Lane> five(5, lane);
  rejects([&] { budget.observe(five); }, "invalid verify budget cycle", "five lanes were observed");
  for (const VerifyBudget::Label &label : {VerifyBudget::Label{kPositions, 0.0F, true}, {0, NAN, true}}) {
    VerifyBudget::Lane bad = lane;
    bad.labels = {label};
    rejects([&] { budget.observe(std::array{bad}); }, "invalid verify budget label",
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

// Spans of each lane's chances.
std::vector<std::span<const float>> spansOf(const std::vector<std::vector<float>> &curves) {
  return {curves.begin(), curves.end()};
}

// A table measured at every lane count, its classes rising where `rising`,
// each point at `cost(lanes, point)` milliseconds in each of its replays.
template <class Cost>
StepCosts measuredTable(std::span<const uint32_t> classes, uint32_t mostRows, bool rising, const Cost &cost) {
  StepCosts costs(classes, mostRows, rising);
  for (uint32_t lanes = 1; lanes <= StepCosts::kLanes; ++lanes) {
    std::vector<double> replay;
    for (const StepCosts::Point &point : costs.points(lanes))
      replay.push_back(cost(lanes, point));
    const std::vector<std::vector<double>> replays(StepCosts::kReplays, replay);
    costs.measure(lanes, replays);
  }
  return costs;
}

// The classes of the projection tiles and heights of Qwen3.8-27B's plans
// (ExecutionPlans::decodeRowClasses on an Apple9 GPU).
constexpr uint32_t kClasses[] = {1, 3, 9, 17, 25, 33, 49};

// A rising table measures each class of more than one row at its first rows
// and its last, the rows past the first rising from it; a dense table at its
// last rows alone.
void risingPoints() {
  struct Expected final {
    uint32_t first, last, cap;
    bool rises;
  };
  const std::vector<Expected> rising{{3, 3, 8, false},   {4, 8, 8, true},   {9, 9, 8, false},  {10, 16, 8, true},
                                     {16, 16, 16, false}, {17, 17, 16, false}, {18, 24, 16, true}, {25, 25, 16, false},
                                     {26, 32, 16, true}},
      dense{{3, 8, 8, false}, {9, 16, 8, false}, {16, 16, 16, false}, {17, 24, 16, false}, {25, 32, 16, false}};
  for (const auto &[rises, expected] : {std::pair{true, rising}, std::pair{false, dense}}) {
    const StepCosts costs(kClasses, 64, rises);
    const std::span<const StepCosts::Point> points = costs.points(2);
    bool same = points.size() == expected.size();
    for (size_t i = 0; same && i < points.size(); ++i)
      same = points[i].first == expected[i].first && points[i].last == expected[i].last &&
             points[i].cap == expected[i].cap && points[i].rises == expected[i].rises;
    require(same, std::string(rises ? "a rising" : "a dense") + " table of two lanes holds other points");
  }
}

// A point's cost is the median of its replays: a replay five times as slow
// moves no point, an even count's is the mean of its middle two, and a lane
// count is measured once every point has a cost.
void replaysMedian() {
  StepCosts costs(kClasses, 64, false);
  for (uint32_t lanes = 1; lanes <= StepCosts::kLanes; ++lanes) {
    require(!costs.measured(lanes), "a table measured nothing and was measured");
    const size_t points = costs.points(lanes).size();
    std::vector<std::vector<double>> replays(StepCosts::kReplays, std::vector<double>(points));
    for (size_t point = 0; point < points; ++point)
      for (uint32_t replay = 0; replay < StepCosts::kReplays; ++replay)
        replays[replay][point] = (10.0 + double(point)) * (replay == point % StepCosts::kReplays ? 5.0 : 1.0);
    costs.measure(lanes, replays);
    require(costs.measured(lanes), "a measured lane count was not measured");
    for (size_t point = 0; point < points; ++point)
      require(costs.points(lanes)[point].milliseconds == 10.0 + double(point),
              "a slow replay moved the cost of a point of " + std::to_string(lanes) + " lanes");
    costs.measure(lanes,
                  std::vector<std::vector<double>>{std::vector<double>(points, 4.0), std::vector<double>(points, 6.0)});
    require(costs.points(lanes).front().milliseconds == 5.0, "two replays' median is not their mean");
  }
}

// The step of a measured table, by brute force over every T the lanes'
// chances fill: T's expected tokens are the best assignment of T - lanes
// rows to the lanes' chances (each lane a prefix, at most a verify block's
// rows), its lane cap the wide one where that assignment gives a lane more
// than a draft's block of rows, and its cost its class's point (no class of
// the anchors alone), with what the wide cap adds at a draft's block a lane
// where a narrow class's step takes it, none where a noisy measurement put
// the wide cap below the narrow one. The choice is the step of the most
// expected tokens per cost, the fewest rows among equals. Random distinct
// chances, a lane in three copying, over random rising costs, every other
// trial with a step that may make a larger class the cheaper, the wide cap
// within 0.2 ms of the narrow one either way.
void choiceOverTheTable() {
  constexpr uint32_t kBlock = StepCosts::kWideCap, kNarrow = StepCosts::kNarrowCap;
  std::mt19937_64 random(0x57e9a1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (uint32_t trial = 0; trial < 400; ++trial) {
    const uint32_t lanes = 1 + trial % StepCosts::kLanes, mostRows = trial % 5 == 4 ? 32 : 64;
    // Every third trial's classes rise with their rows, as a MoE target's.
    const bool rising = trial % 3 == 1;
    std::vector<std::vector<float>> curves(lanes);
    for (std::vector<float> &curve : curves) {
      const bool copying = random() % 3 == 0;
      curve.resize(random() % kBlock);
      for (float &chance : curve)
        chance = static_cast<float>(copying ? 0.9 + 0.1 * unit(random) : unit(random));
      std::ranges::sort(curve, std::greater<>());
    }
    std::map<std::pair<uint32_t, uint32_t>, std::pair<double, double>> table;
    const auto cost = [&](uint32_t laneCount, uint32_t last, uint32_t cap) {
      auto [at, fresh] = table.try_emplace({laneCount, last});
      if (fresh)
        at->second = {10.0 + 0.2 * last + (trial % 2 ? 6.0 * unit(random) : 0.0), 0.4 * unit(random) - 0.2};
      return at->second.first + (cap == kBlock ? at->second.second : 0.0);
    };
    const StepCosts costs = measuredTable(kClasses, mostRows, rising, [&](uint32_t laneCount, const StepCosts::Point &p) {
      return cost(laneCount, p.last, p.cap);
    });
    const StepCosts::Choice got = costs.choose(spansOf(curves));
    const uint32_t most = std::min(lanes * kBlock, mostRows), narrow = lanes * kNarrow;
    std::vector<uint32_t> starts(std::begin(kClasses), std::end(kClasses));
    starts.push_back(narrow + 1);
    std::ranges::sort(starts);
    // The last rows of the class of a step of `rows` rows, and its first.
    const auto lastOf = [&](uint32_t rows) {
      uint32_t last = most;
      for (const uint32_t start : starts)
        if (start > rows) {
          last = std::min(last, start - 1);
          break;
        }
      return last;
    };
    const auto firstOf = [&](uint32_t rows) {
      uint32_t first = lanes;
      for (const uint32_t start : starts)
        if (start <= rows)
          first = std::max(first, start);
      return first;
    };
    // A step's cost in its class at `cap`: its last rows' point, or in a
    // rising class the line from its first rows' point to its last's.
    const auto classCost = [&](uint32_t rows, uint32_t cap) {
      const uint32_t first = firstOf(rows), last = lastOf(rows);
      if (!rising || first == last)
        return cost(lanes, last, cap);
      return cost(lanes, first, cap) +
             (cost(lanes, last, cap) - cost(lanes, first, cap)) * (rows - first) / double(last - first);
    };
    // The pooled chances, largest first, and the lanes' rows of each T.
    std::vector<std::pair<float, uint32_t>> pooled;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      for (const float chance : curves[lane]) pooled.push_back({chance, lane});
    std::ranges::sort(pooled, std::greater<>());
    const std::vector<double> assigned = bestAssignments(curves, kBlock);
    const double wider = std::max(cost(lanes, narrow, kBlock) - cost(lanes, narrow, kNarrow), 0.0);
    double best = 0.0;
    std::vector<uint32_t> held(lanes, 1);
    std::map<uint32_t, std::pair<uint32_t, double>> steps;
    for (uint32_t rows = lanes; rows <= std::min<size_t>(most, lanes + pooled.size()); ++rows) {
      if (rows > lanes)
        ++held[pooled[rows - lanes - 1].second];
      const uint32_t last = lastOf(rows);
      if (last == lanes)
        continue;
      const uint32_t cap = *std::ranges::max_element(held) > kNarrow ? kBlock : kNarrow;
      const double milliseconds =
          classCost(rows, last > narrow ? kBlock : kNarrow) + (last <= narrow && cap == kBlock ? wider : 0.0);
      steps[rows] = {cap, milliseconds};
      best = std::max(best, (lanes + assigned[rows - lanes]) / milliseconds);
    }
    const std::string label = "step choice trial " + std::to_string(trial);
    if (steps.empty()) {
      require(got.step.rows == lanes, label + ": lanes of no chance took more than their anchors");
      continue;
    }
    const auto at = steps.find(got.step.rows);
    const double value = got.expected / got.milliseconds;
    require(got.step.lanes == lanes && at != steps.end() && got.step.laneCap == at->second.first &&
                std::fabs(got.milliseconds - at->second.second) < 1e-9 * at->second.second &&
                std::fabs(got.expected - (lanes + assigned[got.step.rows - lanes])) < 1e-6,
            label + ": chose " + std::to_string(got.step.rows) + " rows at cap " + std::to_string(got.step.laneCap) +
                " weighing another cap, cost or expected tokens than the step's");
    require(std::fabs(value - best) <= 1e-9 * best,
            label + ": chose " + std::to_string(got.step.rows) + " rows for " + std::to_string(value) +
                " tokens per millisecond where the best gives " + std::to_string(best));
    for (const auto &[rows, step] : steps)
      require(rows >= got.step.rows || (lanes + assigned[rows - lanes]) / step.second < best * (1.0 - 1e-9),
              label + ": chose " + std::to_string(got.step.rows) + " rows where " + std::to_string(rows) +
                  " give as many tokens per millisecond");
  }
  // Two lanes, one copying (15 certain chances) beside one of none, where
  // 9-16 rows cost what 3-8 do and the wide cap adds 1%: the wide cap's 16
  // rows take its copy, which a cap of eight rows a lane would stop at 9.
  const std::vector<std::vector<float>> copying{std::vector<float>(kBlock - 1, 1.0F), {}};
  const StepCosts flat = measuredTable(kClasses, 64, false, [](uint32_t, const StepCosts::Point &p) {
    return (p.last <= 16 ? 10.0 : 20.0) * (p.cap == kBlock && p.last == 16 ? 1.01 : 1.0);
  });
  const StepCosts::Choice wide = flat.choose(spansOf(copying));
  require(wide.step.rows == 16 && wide.step.laneCap == kBlock && wide.expected == 16.0 &&
              std::fabs(wide.milliseconds - 10.1) < 1e-9,
          "a copying lane beside an empty one did not take 16 rows at the wide cap and its cost");
  // Two lanes of seven certain chances each: 16 rows at the narrow cap, at
  // its class's cost alone.
  const std::vector<std::vector<float>> even(2, std::vector<float>(kNarrow - 1, 1.0F));
  const StepCosts::Choice block = flat.choose(spansOf(even));
  require(block.step.rows == 16 && block.step.laneCap == kNarrow && block.milliseconds == 10.0,
          "two lanes of a draft's block each did not take it at the narrow cap");
  // A lane whose chances past two are none: its class's cost alike, the
  // fewest rows of the most tokens, three.
  const std::vector<std::vector<float>> two{{1.0F, 1.0F, 0.0F, 0.0F, 0.0F}};
  require(flat.choose(spansOf(two)).step.rows == 3, "rows of no chance were taken at no gain");
  // One lane whose thinking end forces five tokens, four certain rows past
  // its anchor, over classes 1-2, 3-8 and 9-16 that cost 9, 10, 12, 30, 33
  // and 45 ms at 1, 2, 3, 8, 9 and 16 rows: where the classes rise, a step of
  // five rows costs the line from 3 rows to 8, 19.2 ms, and takes the end
  // whole; at a class's last rows alone it would cost 30 ms and lose to two
  // rows. A dense table prices it at its class's end.
  const auto end = [](uint32_t lanes, const StepCosts::Point &p) {
    return lanes > 1 ? 10.0 + p.last
                     : std::map<uint32_t, double>{{1, 9.0}, {2, 10.0}, {3, 12.0}, {8, 30.0}, {9, 33.0}, {16, 45.0}}.at(
                           p.last);
  };
  const std::vector<std::vector<float>> forcedEnd{std::vector<float>(4, 1.0F)};
  const StepCosts::Choice rose = measuredTable(kClasses, 64, true, end).choose(spansOf(forcedEnd));
  require(rose.step.rows == 5 && std::fabs(rose.milliseconds - 19.2) < 1e-9,
          "a thinking end's four certain rows over rising classes took " + std::to_string(rose.step.rows) +
              " rows at " + std::to_string(rose.milliseconds) + " ms");
  const StepCosts::Choice flatEnd = measuredTable(kClasses, 64, false, [](uint32_t lanes, const StepCosts::Point &p) {
    return lanes > 1 ? 10.0 + p.last : std::map<uint32_t, double>{{2, 10.0}, {8, 30.0}, {16, 45.0}}.at(p.last);
  }).choose(spansOf(forcedEnd));
  require(flatEnd.step.rows == 2 && flatEnd.milliseconds == 10.0,
          "a dense class did not price a step inside it at its end");
  // A forced chain's 15 certain rows beside a first cycle's 7, in either lane
  // order: their equal chances split eight rows a lane, so 16 rows keep the
  // narrow cap at its class's cost, where piling the chain's rows on its lane
  // first takes the wide cap, and then its 24 rows.
  const StepCosts stepped = measuredTable(kClasses, 64, false, [](uint32_t, const StepCosts::Point &p) {
    return p.last <= 8 ? 40.0 : p.last <= 16 ? (p.cap == kBlock ? 55.0 : 50.0) : p.last <= 24 ? 80.0 : 85.0;
  });
  const std::vector<float> forced(kBlock - 1, 1.0F), first(kNarrow - 1, 1.0F);
  for (const std::vector<std::vector<float>> &order : {std::vector{forced, first}, std::vector{first, forced}}) {
    const StepCosts::Choice even = stepped.choose(spansOf(order));
    require(even.step.rows == 16 && even.step.laneCap == kNarrow && even.milliseconds == 50.0,
            "equal chances piled on one lane took " + std::to_string(even.step.rows) + " rows at cap " +
                std::to_string(even.step.laneCap));
  }
}

// A measured lane count's line gives its costs back to another table of the
// same classes; a line of other rows or caps, of a cost that is not above
// none, of a point short or one more, gives none and changes nothing; an
// unmeasured lane count has no line.
void recallsTheTable() {
  const StepCosts measured = measuredTable(kClasses, 64, false, [](uint32_t lanes, const StepCosts::Point &p) {
    return 10.0 + lanes + 0.25 * p.last + (p.cap == StepCosts::kWideCap ? 0.125 : 0.0);
  });
  StepCosts recalled(kClasses, 64, false);
  for (uint32_t lanes = 1; lanes <= StepCosts::kLanes; ++lanes) {
    recalled.recall(lanes, measured.text(lanes));
    require(recalled.measured(lanes), "a measured line did not recall its costs");
    for (size_t point = 0; point < measured.points(lanes).size(); ++point)
      require(recalled.points(lanes)[point].milliseconds == measured.points(lanes)[point].milliseconds,
              "a recalled line gave other costs than its measurement's");
  }
  StepCosts other(kClasses, 64, false);
  const std::string line = measured.text(2);
  const std::string shifted = "2:8:5" + line.substr(line.find(' '));
  for (const std::string &bad : {std::string(), std::string("8:8:x"), shifted, line + " 64:16:9",
                                 line.substr(0, line.rfind(' ')), line.substr(0, line.rfind(':')) + ":-1",
                                 line.substr(0, line.rfind(':')) + ":nan"}) {
    other.recall(2, bad);
    require(!other.measured(2), "a line of no table's costs was recalled: " + bad);
  }
  rejects([&] { static_cast<void>(other.text(2)); }, "an unmeasured lane count has no costs to keep",
          "an unmeasured lane count gave a line");
}

// Before its lane count is measured, a cycle takes a draft's block a lane at
// the narrow cap, no more than the lanes' chances fill nor the table's rows;
// lanes without a chance past their anchors take their anchors, measured or
// not; a table holds no step of more rows than it runs in one pass.
void unmeasuredAndBounds() {
  const StepCosts unmeasured(kClasses, 32, false);
  for (uint32_t lanes = 1; lanes <= StepCosts::kLanes; ++lanes) {
    const std::vector<std::vector<float>> sure(lanes, std::vector<float>(StepCosts::kWideCap - 1, 1.0F)),
        three(lanes, std::vector<float>(3, 0.5F)), none(lanes);
    const StepCosts::Choice block = unmeasured.choose(spansOf(sure));
    require(block.step.rows == std::min(8 * lanes, 32U) && block.step.laneCap == StepCosts::kNarrowCap &&
                block.milliseconds == 0.0,
            "an unmeasured table did not take a draft's block a lane at " + std::to_string(lanes) + " lanes");
    require(unmeasured.choose(spansOf(three)).step.rows == 4 * lanes,
            "an unmeasured table took rows the lanes' chances do not fill");
    require(unmeasured.choose(spansOf(none)).step.rows == lanes, "lanes of no chance took more than their anchors");
    for (const StepCosts::Point &point : unmeasured.points(lanes))
      require(point.last <= 32 && point.first >= lanes, "a point lies outside its lanes' rows in one pass");
  }
  const StepCosts measured = measuredTable(kClasses, 64, false, [](uint32_t, const StepCosts::Point &p) {
    return 10.0 + p.last;
  });
  for (uint32_t lanes = 1; lanes <= StepCosts::kLanes; ++lanes)
    require(measured.choose(spansOf(std::vector<std::vector<float>>(lanes))).step.rows == lanes,
            "measured lanes of no chance took more than their anchors");
}

// A lane's chances: its tree's and its lookup chain's, largest first, at
// most a block's rows past its anchor and its tree's depth. A lane of two
// tokens left, one past its anchor, expects at most two whatever the rows.
void laneChancesBound() {
  constexpr uint32_t kBlock = SPLASH_TARGET_VERIFY_ROWS;
  require(VerifyBudget::laneChances(std::vector<float>{0.9F, 0.5F, 0.2F}, std::vector<float>{0.8F, 0.6F},
                                    kBlock - 1) == std::vector<float>{0.9F, 0.8F, 0.6F, 0.5F, 0.2F},
          "a lane's chances did not merge its tree's and its lookup's");
  require(VerifyBudget::laneChances(std::vector<float>(kBlock, 0.5F), std::vector<float>(kBlock, 0.7F), kBlock)
                  .size() == kBlock - 1,
          "a lane's chances ran past a block's rows");
  const std::vector<std::vector<float>> last{
      VerifyBudget::laneChances(std::vector<float>(kBlock - 1, 0.9F), std::vector<float>{}, 1)};
  const StepCosts flat = measuredTable(kClasses, 64, false, [](uint32_t, const StepCosts::Point &) { return 1.0; });
  require(last[0] == std::vector<float>{0.9F} && flat.choose(spansOf(last)).expected <= 2.0,
          "a lane's chances ran past its tree's depth");
}

// The steps no costs choose: under --verify-rows all, a draft's block a lane,
// no more than each lane holds, at the narrow cap; and a forced step, a row a
// lane to a verify block's, at the cap it names, else at the one its rows
// need.
void stepsOfNoCost() {
  const std::vector<uint32_t> holds{3, 16, 1, 9};
  const splash::ops::StepRows all = StepCosts::block(holds);
  require(all.lanes == 4 && all.rows == 3 + 8 + 1 + 8 && all.laneCap == StepCosts::kNarrowCap,
          "a step of --verify-rows all took " + std::to_string(all.rows) + " rows, not a draft's block a lane held");
  for (const auto &[lanes, rows, cap, expected] :
       {std::tuple{2U, 16U, std::optional<uint32_t>(StepCosts::kWideCap), splash::ops::StepRows{2, 16, 16}},
        std::tuple{2U, 16U, std::optional<uint32_t>(), splash::ops::StepRows{2, 16, 8}},
        std::tuple{2U, 40U, std::optional<uint32_t>(), splash::ops::StepRows{2, 32, 16}},
        std::tuple{3U, 1U, std::optional<uint32_t>(), splash::ops::StepRows{3, 3, 8}},
        std::tuple{4U, 40U, std::optional<uint32_t>(StepCosts::kNarrowCap), splash::ops::StepRows{4, 32, 8}}}) {
    const splash::ops::StepRows got = StepCosts::forced(lanes, rows, cap);
    require(got.lanes == expected.lanes && got.rows == expected.rows && got.laneCap == expected.laneCap,
            "a forced step of " + std::to_string(rows) + " rows over " + std::to_string(lanes) + " lanes took " +
                std::to_string(got.rows) + " at cap " + std::to_string(got.laneCap));
  }
}

// Tables of no classes, of classes not from one row or not rising, or of no
// rows; replays of no point, of a point short, of a time that is not
// positive; and lane counts outside a batch.
void invalidSteps() {
  for (const std::vector<uint32_t> &classes :
       {std::vector<uint32_t>{}, std::vector<uint32_t>{3, 9}, std::vector<uint32_t>{1, 9, 3}})
    rejects([&] { StepCosts costs(classes, 64, false); }, "invalid step classes", "impossible classes made a table");
  rejects([&] { StepCosts costs(kClasses, 0, false); }, "invalid step classes", "a table of no rows was made");
  // Four lanes of four rows hold their anchors alone.
  rejects([&] { StepCosts costs(kClasses, StepCosts::kLanes, false); }, "hold nothing past its lanes' anchors",
          "a table of the anchors alone was made");
  StepCosts costs(kClasses, 64, false);
  const size_t points = costs.points(2).size();
  for (const std::vector<std::vector<double>> &replays :
       {std::vector<std::vector<double>>{}, {std::vector<double>(points - 1, 1.0)},
        {std::vector<double>(points, 1.0), std::vector<double>(points, 0.0)},
        {std::vector<double>(points, NAN)}})
    rejects([&] { costs.measure(2, replays); }, "invalid step replays", "impossible replays were measured");
  require(!costs.measured(2), "refused replays measured a lane count");
  for (const uint32_t lanes : {0U, 5U}) {
    rejects([&] { static_cast<void>(costs.points(lanes)); }, "invalid step lane count",
            "points of an impossible lane count were given");
    rejects([&] { costs.measure(lanes, std::vector<std::vector<double>>{std::vector<double>(points, 1.0)}); },
            "invalid step lane count", "an impossible lane count was measured");
    rejects([&] { costs.recall(lanes, ""); }, "invalid step lane count",
            "an impossible lane count's line was recalled");
    rejects([&] { static_cast<void>(costs.choose(spansOf(std::vector<std::vector<float>>(lanes)))); },
            "invalid step lane count", "a step of an impossible lane count was chosen");
  }
}

} // namespace

int main() {
  try {
    startsAtThePrior();
    treeLabelsOfAWalk();
    policiesFitApart();
    fitsTheLabels();
    invalidInputs();
    replaysMedian();
    risingPoints();
    choiceOverTheTable();
    unmeasuredAndBounds();
    laneChancesBound();
    recallsTheTable();
    stepsOfNoCost();
    invalidSteps();
  } catch (const std::exception &error) {
    std::cerr << "verify_budget_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "verify_budget_test: PASS\n";
  return 0;
}
