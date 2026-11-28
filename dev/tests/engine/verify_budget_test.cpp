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
#include "model/Model.hpp"
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
#include <utility>
#include <vector>

using splash::model::CostLine;
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
  const std::vector<VerifyBudget::Lane> tooMany(VerifyBudget::kLanes + 1, lane);
  rejects([&] { budget.observe(tooMany); }, "invalid verify budget cycle", "more lanes than a step holds were observed");
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
// moves no point; an even count's is the mean of its middle two; and a lane
// count is measured once every point has a cost. A measured table is stale
// where its largest step costs more than StepCosts::kStale past the median
// of that step's replays now (a table measured on a busy Mac), and stays
// where it costs that or less (now slower: other work only slows a step);
// forgetting leaves every lane count unmeasured.
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
    const double largest = 10.0 + double(points - 1);
    for (const auto &[now, stale] : std::initializer_list<std::pair<double, bool>>{
             {largest, false}, {largest / 1.19, false}, {largest / 1.21, true}, {largest * 3.0, false}}) {
      // One fast replay and one five times as slow leave the median at `now`.
      const std::array<double, StepCosts::kReplays> check{now * 5.0, now, now / 2.0};
      require(StepCosts::stale(costs.level(lanes, check)) == stale, "a table " + std::to_string(largest / now) +
                                                                       " times its largest step now was judged " +
                                                                       (stale ? "current" : "stale"));
    }
    costs.measure(lanes,
                  std::vector<std::vector<double>>{std::vector<double>(points, 4.0), std::vector<double>(points, 6.0)});
    require(costs.points(lanes).front().milliseconds == 5.0, "two replays' median is not their mean");
  }
  costs.forget();
  for (uint32_t lanes = 1; lanes <= StepCosts::kLanes; ++lanes)
    require(!costs.measured(lanes), "a forgotten table stayed measured");
  rejects([&] { static_cast<void>(costs.level(1, std::array<double, 1>{1.0})); }, "invalid level check",
          "an unmeasured table's level was checked");
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
    const uint32_t most = std::min(lanes * kBlock, mostRows), narrow = std::min(lanes * kNarrow, most);
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

// The cost of passes over history: k, the least-squares slope through the
// origin of cycles' wall less their price on their passes over history (a
// lane's history in thousands of tokens times the passes its own rows take,
// a pass a draft's block), each cycle's weight decaying by a 64th a cycle,
// none below none, one for the model, a stall counted at twice its predicted
// cost; a step costs its price plus k times its own passes over history, the
// anchors alone too, each lane's passes over its own history, so at a long
// history a lane's second pass can make fewer rows the better step where one
// level would not, and beside a lane of a long history a lane of a short one
// still takes it. Impossible samples are refused; an unmeasured lane count's
// cycle teaches nothing.
void contextPerPass() {
  const auto table = [] {
    return measuredTable(kClasses, 64, false, [](uint32_t, const StepCosts::Point &p) {
      return p.last <= 8 ? 10.0 : 12.0;
    });
  };
  StepCosts costs = table();
  // One lane whose rows past 8 expect 2.2 tokens more: 10.2 tokens in 12 ms
  // beat 8 in 10 at no history.
  std::vector<float> curve(StepCosts::kWideCap - 1, 0.0F);
  for (size_t row = 0; row < curve.size(); ++row)
    curve[row] = row < 7 ? 1.0F : 0.275F;
  const std::vector<std::vector<float>> lane{curve};
  const std::vector<uint64_t> history{30000};
  require(costs.contextCost() == 0.0 && costs.choose(spansOf(lane), history).step.rows == 16,
          "with no cost of passes over history the table's choice moved");
  const std::vector<uint32_t> sixteen{16}, eight{8};
  // 16 rows, two passes over 30K: 60; 8 rows, one: 30.
  costs.observeContext({1, 16, StepCosts::kWideCap}, sixteen, history, 12.0 + 6.0);
  costs.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, 10.0 + 3.0);
  require(std::fabs(costs.contextCost() - 0.1) < 1e-9, "k is not the slope of wall less price on passes over history: " +
                                                           std::to_string(costs.contextCost()));
  // At 30K, 8 rows cost 10 + 3 and 16 cost 12 + 6: 8 in 13 beat 10.2 in 18
  // (one level of 3 for both would have kept 16: 10.2 in 15).
  const StepCosts::Choice choice = costs.choose(spansOf(lane), history);
  require(choice.step.rows == 8 && std::fabs(choice.milliseconds - 13.0) < 1e-9,
          "a lane's second pass over 30K did not move the choice to 8 rows");
  require(costs.choose(spansOf(lane), std::vector<uint64_t>{0}).step.rows == 16,
          "a lane of no history paid for passes over history");
  // The anchors alone pay their pass too: a lane whose one row past its
  // anchor has a chance of 0.2 takes it, 1.2 tokens in 13 ms against 1 in 13
  // (the anchors at their table's 10 ms would have kept it out: 1 in 10).
  const StepCosts::Choice faint = costs.choose(spansOf(std::vector<std::vector<float>>{{0.2F}}), history);
  require(faint.step.rows == 2 && std::fabs(faint.milliseconds - 13.0) < 1e-9,
          "the anchors alone did not pay their pass over history");
  // Each lane's passes are over its own history. The lane above beside one
  // of a draft's block of certain rows, one of them at 2K and the other at
  // 30K, in either order: 16 rows cost 12 + 3.2. Where the lane past its
  // block is the one at 2K, its second pass costs 0.2, and 18.2 tokens in
  // 15.4 ms beat 16 in 15.2 (a pass more for every lane would cost 3.2: 18.2
  // in 18.4); where it is the one at 30K, 3, and 16 in 15.2 beat 18.2 in 18.2.
  const std::vector<float> block(StepCosts::kNarrowCap - 1, 1.0F);
  for (const bool pastFirst : {true, false})
    for (const bool pastLong : {false, true}) {
      const uint64_t past = pastLong ? 30000 : 2000, beside = pastLong ? 2000 : 30000;
      const std::vector<std::vector<float>> lanes = pastFirst ? std::vector{curve, block} : std::vector{block, curve};
      const std::vector<uint64_t> histories = pastFirst ? std::vector{past, beside} : std::vector{beside, past};
      const StepCosts::Choice both = costs.choose(spansOf(lanes), histories);
      require(pastLong ? both.step.rows == 16 && both.step.laneCap == StepCosts::kNarrowCap &&
                             std::fabs(both.milliseconds - 15.2) < 1e-9
                       : both.step.rows == 24 && both.step.laneCap == StepCosts::kWideCap &&
                             std::fabs(both.milliseconds - 15.4) < 1e-9,
              std::string("a lane's second pass was not priced over its own history: beside a lane at ") +
                  (pastLong ? "2K, a lane at 30K" : "30K, a lane at 2K") + " took " + std::to_string(both.step.rows) +
                  " rows at " + std::to_string(both.milliseconds) + " ms");
    }
  // A cycle's passes are its lanes' own rows' over their own histories: two
  // lanes of 4 and 16 rows at 30K and 2K pass over 30 + 2 * 2, 34 (their 20
  // rows split evenly would pass over 64, their histories exchanged over 62).
  StepCosts uneven = table();
  uneven.observeContext({2, 20, StepCosts::kWideCap}, std::vector<uint32_t>{4, 16}, std::vector<uint64_t>{30000, 2000},
                        12.0 + 0.1 * 34.0);
  require(std::fabs(uneven.contextCost() - 0.1) < 1e-9,
          "a cycle's passes over history are not its lanes' own rows' over their own histories: " +
              std::to_string(uneven.contextCost()));
  // At a slope of 0.2 for 64 cycles, then of 0.05 (two lanes) for 300: the
  // decayed fit follows the latest.
  for (uint32_t cycle = 0; cycle < 64; ++cycle)
    costs.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, 10.0 + 6.0);
  require(std::fabs(costs.contextCost() - 0.2) < 0.02, "k did not follow a slope of 0.2");
  for (uint32_t cycle = 0; cycle < 300; ++cycle)
    costs.observeContext({2, 16, StepCosts::kNarrowCap}, std::vector<uint32_t>{8, 8},
                         std::vector<uint64_t>{20000, 40000}, 12.0 + 0.05 * 60.0);
  require(std::fabs(costs.contextCost() - 0.05) < 0.003,
          "k is not one decayed fit for the model across lane counts: " + std::to_string(costs.contextCost()));
  // Over a long run each cycle is a 64th of the fit, one in the acceptance
  // fit's refit period: after 2048 cycles at a slope of 0.2, one at 0.05
  // moves k a 64th of the way.
  StepCosts steady = table();
  for (uint32_t cycle = 0; cycle < 2048; ++cycle)
    steady.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, 10.0 + 0.2 * 30.0);
  steady.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, 10.0 + 0.05 * 30.0);
  require(std::fabs(steady.contextCost() - (0.2 - 0.15 / VerifyBudget::kRefitCycles)) < 1e-9,
          "a cycle is not a 64th of a long run's fit: " + std::to_string(steady.contextCost()));
  // A stall counts at twice its predicted cost: from a k of none, a cycle of
  // 100 ms whose price is 10 and passes 30 counts at 20, so k is a third, and
  // the next at 40, so k grows to the cost the table lacks (k of 2: 70 ms).
  StepCosts stalls = table();
  stalls.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, 100.0);
  require(std::fabs(stalls.contextCost() - 10.0 / 30.0) < 1e-9,
          "a stall did not count at twice its predicted cost: " + std::to_string(stalls.contextCost()));
  for (uint32_t cycle = 0; cycle < 200; ++cycle)
    stalls.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, 10.0 + 2.0 * 30.0);
  require(std::fabs(stalls.contextCost() - 2.0) < 0.05,
          "k did not grow to a cost of passes far from none: " + std::to_string(stalls.contextCost()));
  for (uint32_t cycle = 0; cycle < 300; ++cycle)
    costs.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, 9.0);
  require(costs.contextCost() == 0.0, "a slope below none was taken");
  StepCosts unmeasured(kClasses, 64, false);
  unmeasured.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, 40.0);
  require(unmeasured.contextCost() == 0.0, "an unmeasured lane count's cycle taught k");
  rejects([&] { costs.observeContext({1, 8, StepCosts::kNarrowCap}, eight, history, -1.0); }, "invalid step sample",
          "an impossible sample was taken");
  rejects([&] { costs.observeContext({2, 8, StepCosts::kNarrowCap}, eight, history, 10.0); }, "invalid step sample",
          "a sample of other lanes than its step's was taken");
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

// 1.6.2's fit of a lane count's samples, from the samples alone: the decayed
// sums of their rows, times and tokens, the least-squares line through them,
// its slope shrunk toward the seed's as much as five cycles spread a row a
// lane would pull it, and the floor, the slope times their tokens per ms.
struct LineSamples final {
  double w = 0.0, r = 0.0, rr = 0.0, t = 0.0, rt = 0.0, k = 0.0;

  void add(uint32_t rows, uint32_t tokens, double milliseconds) {
    w = 0.97 * w + 1.0;
    r = 0.97 * r + rows;
    rr = 0.97 * rr + double(rows) * rows;
    t = 0.97 * t + milliseconds;
    rt = 0.97 * rt + rows * milliseconds;
    k = 0.97 * k + tokens;
  }
  [[nodiscard]] double mean() const { return t / w; }
  // The line's intercept and slope.
  [[nodiscard]] std::pair<double, double> line(uint32_t lanes) const {
    const double meanRows = r / w, meanTime = t / w;
    const double spread = std::max(rr / w - meanRows * meanRows, 0.0);
    const double covariance = rt / w - meanRows * meanTime;
    const uint32_t last = lanes * CostLine::kBlockRows;
    const double seedSlope = (CostLine::seed(lanes, last) - CostLine::seed(lanes, lanes)) / double(last - lanes);
    const double prior = seedSlope * meanTime / CostLine::seed(lanes, static_cast<uint32_t>(std::lround(meanRows)));
    const double slope =
        std::max((w * covariance + 5.0 * lanes * lanes * prior) / (w * spread + 5.0 * lanes * lanes), 0.0);
    return {meanTime - slope * meanRows, slope};
  }
  [[nodiscard]] double floor(uint32_t lanes) const { return std::min(line(lanes).second, t / r) * k / t; }
  // Whether a cost line's fit is this one.
  [[nodiscard]] bool fits(uint32_t lanes, const std::optional<std::pair<double, double>> &fit) const {
    const auto [intercept, slope] = line(lanes);
    return fit && std::fabs(fit->second - slope) <= 1e-9 * std::max(1.0, slope) &&
           std::fabs(fit->first - intercept) <= 1e-9 * mean();
  }
};

// The classes of Qwen3.8-Flash-Next's steps of up to its store's round of 32
// rows (ExecutionPlans::decodeRowClasses on an Apple10 GPU): steps of 1, 2,
// 3, 4 to 8, 9 to 15, 16, 17 to 24 and 25 to 32 rows.
constexpr uint32_t kStreamedClasses[] = {1, 2, 3, 4, 9, 16, 17, 25};

// 1.6.2's cost line, as a streamed-expert target's cycles fit it: a lane
// count's decayed sums of the rows its cycles' lanes verified and of their
// times, the least-squares line through them with its slope shrunk toward
// the seed's (kPriorCycles cycles spread a row a lane), recomputed here from
// the samples alone; the seed's shape (1.6.2's table). Its floor is the
// line's slope times the tokens per ms of the same cycles, their decayed
// tokens over their decayed times, which follows the latest cycles (the
// first here retain every row, the later ones their anchors alone); before
// a sample it is the seed's, the seed's slope times the tokens per cost of
// a cycle of the anchors alone by the seed; and none where the next cycle
// is a lane count's every 32nd, which explores. The slope it takes is at
// most the cycles' time a row, so the floor at most their tokens a row.
// Constrained cycles keep a line of their own.
void costLineFitsTheSamples() {
  require(CostLine::seed(1, 8) == 1.0 && CostLine::seed(2, 2) == 0.5 &&
              std::fabs(CostLine::seed(2, 6) - 0.66) < 1e-12 &&
              std::fabs(CostLine::seed(4, 12) - 0.57) < 1e-12,
          "the cost line's seed is not 1.6.2's");
  for (uint32_t lanes = 1; lanes <= CostLine::kLanes; ++lanes) {
    CostLine costs(kStreamedClasses);
    // The seed's floor: 0.4 / 4.2, 0.5 / 3.5, 0.55 / 3.15 and 0.6 / 2.8 at
    // one to four lanes, and four lanes' 0.6 / 2.8 past them (their seed).
    const double seedSlope = (1.0 - CostLine::seed(lanes, lanes)) / (7.0 * lanes);
    const double seedFloor = seedSlope * lanes / CostLine::seed(lanes, lanes);
    require(!costs.line(lanes, false) && std::fabs(costs.floor(lanes, false) - seedFloor) <= 1e-12 &&
                std::fabs(seedFloor - std::array{0.4 / 4.2, 0.5 / 3.5, 0.55 / 3.15, 0.6 / 2.8}[std::min(lanes, 4U) - 1]) <=
                    1e-12,
            std::to_string(lanes) + " lanes: before a sample the cost line had a fit, or its floor is not the seed's");
    std::mt19937_64 random(0xc057 + lanes);
    std::uniform_int_distribution<uint32_t> rowsOf(lanes, lanes * CostLine::kBlockRows);
    std::uniform_real_distribution<double> noise(-1.0, 1.0);
    LineSamples samples;
    for (uint32_t cycle = 0; cycle < 90; ++cycle) {
      const uint32_t rows = rowsOf(random), tokens = cycle < 45 ? rows : lanes;
      const double milliseconds = 12.0 + 0.6 * rows + 0.5 * noise(random);
      costs.observe(lanes, false, rows, tokens, milliseconds);
      samples.add(rows, tokens, milliseconds);
      require(samples.fits(lanes, costs.line(lanes, false)),
              std::to_string(lanes) + " lanes: the cost line is not 1.6.2's fit of its samples after cycle " +
                  std::to_string(cycle));
      const double floor = costs.floor(lanes, false);
      if ((cycle + 1) % CostLine::kExploreCycles == CostLine::kExploreCycles - 1)
        require(floor == 0.0, std::to_string(lanes) + " lanes: the 32nd cycle of a lane count had a floor");
      else
        require(floor > 0.0 && std::fabs(floor - samples.floor(lanes)) <= 1e-9 * samples.floor(lanes),
                std::to_string(lanes) + " lanes: the floor after cycle " + std::to_string(cycle) +
                    " is not the line's slope times its latest cycles' tokens per ms");
    }
    require(!costs.line(lanes, true) && std::fabs(costs.floor(lanes, true) - seedFloor) <= 1e-12,
            "an unconstrained cycle taught the constrained line");
    // A cold start: a cycle of every row at twenty times the time of the
    // cycles of the anchors alone after it, each retaining every row. The
    // line's slope puts its intercept below nothing, and the floor is the
    // cycles' tokens a row, 1, which a row the target is certain to accept
    // passes.
    CostLine cold(kStreamedClasses);
    LineSamples start;
    for (uint32_t cycle = 0; cycle < 6; ++cycle) {
      const uint32_t rows = cycle ? lanes : lanes * CostLine::kBlockRows;
      const double milliseconds = cycle ? 20.0 : 400.0;
      cold.observe(lanes, false, rows, rows, milliseconds);
      start.add(rows, rows, milliseconds);
    }
    require(start.line(lanes).first < 0.0 && start.line(lanes).second * start.k / start.t > 1.0,
            std::to_string(lanes) + " lanes: the cold start's line does not price a row over the cycles' time a row");
    require(std::fabs(cold.floor(lanes, false) - 1.0) <= 1e-12,
            std::to_string(lanes) + " lanes: a slope over the cycles' time a row put the floor at " +
                std::to_string(cold.floor(lanes, false)) + ", not at their tokens a row");
  }
}

// A streamed-expert target's step and floor. On every 32nd cycle of a lane
// count, constrained or not apart, the floor is none and the step a draft's
// block a lane, no more than each lane holds, at the narrow cap, whatever
// the lanes' chances. Else the floor is its line's, the seed's before a
// sample, and the step's rows the last of the class that holds the lanes'
// anchors and their chances at least the floor (a chance at the floor
// passes, one under it does not), no more of a lane's than a draft's block
// nor than it holds, or a draft's block of rows in all where they are fewer
// (one lane's step is a draft's block whatever its chances), and the step no
// more than a draft's block a lane. The
// choice weighs no expected tokens and no cost. A cycle counts at most twice
// the larger of its line's price and the mean: a stall moves the line as a
// cycle of that time does.
void costLineChoosesItsSteps() {
  constexpr uint32_t kNarrow = CostLine::kBlockRows;
  // The last rows of the class of a step of `rows` rows.
  const auto classEnd = [](uint32_t rows) {
    const auto next = std::upper_bound(std::begin(kStreamedClasses), std::end(kStreamedClasses), rows);
    return next == std::end(kStreamedClasses) ? CostLine::kLanes * kNarrow : *next - 1;
  };
  // The step of `chances` at `floor`, for lanes holding `holds`.
  const auto stepOf = [&](const std::vector<std::vector<float>> &chances, const std::vector<uint32_t> &holds,
                          float floor) {
    uint32_t passed = 0;
    for (size_t lane = 0; lane < chances.size(); ++lane) {
      uint32_t rows = 1;
      for (const float chance : chances[lane])
        rows += chance >= floor;
      passed += std::min({rows, holds[lane], kNarrow});
    }
    return std::min(classEnd(std::max(passed, kNarrow)), StepCosts::block(holds).rows);
  };
  for (uint32_t lanes = 1; lanes <= CostLine::kLanes; ++lanes) {
    const std::string label = std::to_string(lanes) + " lanes: ";
    std::vector<uint32_t> holds(lanes, StepCosts::kWideCap);
    holds.back() = lanes > 1 ? 3 : StepCosts::kWideCap;
    // Each lane's chances no more than it holds past its anchor.
    const auto curves = [&](float chance) {
      std::vector<std::vector<float>> made;
      for (const uint32_t held : holds)
        made.emplace_back(held - 1, chance);
      return made;
    };
    const splash::ops::StepRows block = StepCosts::block(holds);
    const std::vector<std::vector<float>> none(lanes);
    CostLine costs(kStreamedClasses);
    // Before a sample, at the seed's floor: certain chances take a draft's
    // block a lane, and chances of 0.05 or none a draft's block of rows.
    const auto seedFloor = static_cast<float>(costs.floor(lanes, false));
    for (const auto &[chances, rows] :
         {std::pair{curves(1.0F), block.rows}, std::pair{curves(0.05F), kNarrow}, std::pair{none, kNarrow}}) {
      const StepCosts::Choice first = costs.choose(spansOf(chances), holds, false);
      require(first.step.lanes == lanes && first.step.rows == rows && first.step.laneCap == kNarrow &&
                  first.floor == seedFloor && seedFloor > 0.05F && first.expected == 0.0 && first.milliseconds == 0.0,
              label + "before a sample the step is " + std::to_string(first.step.rows) + " rows at a floor of " +
                  std::to_string(first.floor) + ", not the " + std::to_string(rows) + " its seed's floor passes");
    }
    // A line whose rows cost next to nothing, over cycles that retain their
    // anchors alone. `samples` keeps what the line saw.
    LineSamples samples;
    const auto cycleOf = [&](uint32_t rows) {
      costs.observe(lanes, false, rows, lanes, 20.0);
      samples.add(rows, lanes, 20.0);
    };
    for (uint32_t cycle = 0; cycle < 30; ++cycle)
      cycleOf(lanes + cycle % (lanes * (kNarrow - 1) + 1));
    const float floor = costs.choose(spansOf(none), holds, false).floor;
    require(floor > 0.0F && floor < 0.05F && std::fabs(floor - samples.floor(lanes)) <= 1e-6 * samples.floor(lanes),
            label + "the step's floor is not its line's");
    // A draft's block of rows where no chance reaches the floor. Past it,
    // among lanes that hold a verify block's rows, a chance at the floor
    // passes into the next class and one under it does not: a lane's certain
    // chances that with the anchors make a draft's block of rows, and one
    // more in another lane.
    const float under = std::nextafter(floor, 0.0F);
    require(costs.choose(spansOf(none), holds, false).step.rows == kNarrow &&
                costs.choose(spansOf(curves(under)), holds, false).step.rows == kNarrow,
            label + "chances under the floor did not leave the step a draft's block of rows in all");
    if (lanes > 1) {
      const std::vector<uint32_t> full(lanes, StepCosts::kWideCap);
      std::vector<std::vector<float>> edge(lanes);
      edge.front().assign(kNarrow - lanes, 1.0F);
      edge.back() = {under};
      const StepCosts::Choice below = costs.choose(spansOf(edge), full, false);
      edge.back() = {floor, under};
      const StepCosts::Choice atFloor = costs.choose(spansOf(edge), full, false);
      require(below.step.rows == kNarrow && atFloor.step.rows == classEnd(kNarrow + 1) &&
                  atFloor.step.laneCap == kNarrow && atFloor.floor == floor && atFloor.expected == 0.0 &&
                  atFloor.milliseconds == 0.0,
              label + "a chance under the floor and one at it gave " + std::to_string(below.step.rows) + " and " +
                  std::to_string(atFloor.step.rows) + " rows, not a draft's block and the last of the class past it");
    }
    // Certain chances past a verify block's rows: a draft's block a lane, no
    // more than each lane holds.
    const std::vector<std::vector<float>> wide(lanes, std::vector<float>(StepCosts::kWideCap - 1, 1.0F));
    require(costs.choose(spansOf(wide), holds, false).step.rows == block.rows,
            label + "certain chances did not take a draft's block a lane, no more than each lane holds");
    // Every step the chances fill, drawn at random among lanes that hold 1
    // to a verify block's rows.
    std::mt19937_64 random(0xf100 + lanes);
    for (uint32_t trial = 0; trial < 300; ++trial) {
      std::vector<uint32_t> held(lanes);
      std::vector<std::vector<float>> chances(lanes);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        held[lane] = 1 + random() % StepCosts::kWideCap;
        chances[lane].resize(random() % held[lane]);
        for (float &chance : chances[lane]) {
          const uint64_t kind = random() % 4;
          chance = kind == 0 ? floor : kind == 1 ? under : kind == 2 ? 1.0F : 0.0F;
        }
        std::ranges::sort(chances[lane], std::greater<>());
      }
      const StepCosts::Choice got = costs.choose(spansOf(chances), held, false);
      require(got.step.lanes == lanes && got.step.rows == stepOf(chances, held, floor) &&
                  got.step.laneCap == kNarrow && got.floor == floor,
              label + "trial " + std::to_string(trial) + ": the step is " + std::to_string(got.step.rows) +
                  " rows, not the " + std::to_string(stepOf(chances, held, floor)) +
                  " that end the class of the rows its floor passes, within a draft's block a lane");
    }
    // Every 32nd cycle explores: the 31st observed cycle's next step is the
    // block at no floor, whatever the chances; the constrained cycles count
    // their own.
    cycleOf(lanes);
    const StepCosts::Choice explored = costs.choose(spansOf(none), holds, false);
    require(explored.step.rows == block.rows && explored.floor == 0.0F,
            label + "the 32nd cycle did not explore a draft's block a lane at no floor");
    for (uint32_t cycle = 0; cycle < 5; ++cycle)
      costs.observe(lanes, true, lanes + cycle, lanes, 30.0);
    require(costs.choose(spansOf(none), holds, true).floor > 0.0F &&
                costs.choose(spansOf(none), holds, false).floor == 0.0F,
            label + "the constrained cycles explored at the unconstrained cycles' count, or moved it");
    cycleOf(block.rows);
    require(costs.choose(spansOf(none), holds, false).floor > 0.0F,
            label + "the cycle after an exploration explored again");
    // A stall counts as a cycle of twice the larger of its line's price and
    // the mean, by 1.6.2's fit of the samples with that cycle among them.
    // Every cycle so far took 20 ms, and the prior's slope leaves the line's
    // price under the mean at the fewest rows and over it at the most: the
    // mean bounds a stall of the one, the price a stall of the other.
    const auto [intercept, slope] = samples.line(lanes);
    for (const uint32_t rows : {lanes, lanes * kNarrow}) {
      const double price = intercept + slope * rows;
      require(rows == lanes ? price < samples.mean() : price > samples.mean(),
              label + "the line's price at " + std::to_string(rows) + " rows is not on its side of the mean");
      CostLine stalled = costs;
      stalled.observe(lanes, false, rows, lanes, 5.0 * samples.mean());
      LineSamples bounded = samples;
      bounded.add(rows, lanes, 2.0 * std::max(price, samples.mean()));
      require(stalled.line(lanes, false) != costs.line(lanes, false), "a stall of five times the mean did not count");
      require(bounded.fits(lanes, stalled.line(lanes, false)),
              "a stall of five times the mean did not count at twice the larger of its line's price and the mean");
    }
  }
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
// rows; a table whose measurement runs past the startup runway; replays of
// no point, of a point short, of a time that is not positive; and lane
// counts outside a batch.
void invalidSteps() {
  for (const std::vector<uint32_t> &classes :
       {std::vector<uint32_t>{}, std::vector<uint32_t>{3, 9}, std::vector<uint32_t>{1, 9, 3}})
    rejects([&] { StepCosts costs(classes, 64, false); }, "invalid step classes", "impossible classes made a table");
  rejects([&] { StepCosts costs(kClasses, 0, false); }, "invalid step classes", "a table of no rows was made");
  // Four lanes of four rows hold their anchors alone.
  rejects([&] { StepCosts costs(kClasses, StepCosts::kLanes, false); }, "hold nothing past its lanes' anchors",
          "a table of the anchors alone was made");
  // The startup runway of a runtime's lanes holds each lane count's
  // measurement from page 5 (ExecutionLimits::runwayPages): four lanes' 64
  // pages hold four lanes of 14 points, a MoE target's most today, eight
  // lanes' 128 eight lanes of 14 at 128 rows a step; a table of a class every
  // row makes runs past eight lanes' and is refused.
  using splash::model::ExecutionLimits;
  static_assert(ExecutionLimits::runwayPages(32, 1, 1) == 64 && ExecutionLimits::runwayPages(32, 4, 1) == 64 &&
                ExecutionLimits::runwayPages(32, 8, 1) == 128 && ExecutionLimits::runwayPages(128, 5, 1) == 128);
  std::vector<uint32_t> everyRow;
  for (uint32_t row = 1; row <= 128; ++row) everyRow.push_back(row);
  const StepCosts every(everyRow, 128, false), moe(kClasses, 128, true), four(kClasses, 64, true);
  rejects([&] { every.requireRunway(5, ExecutionLimits::runwayPages(128, 8, 1), 8); }, "startup runway",
          "a table whose measurement runs past the startup runway was taken");
  moe.requireRunway(5, ExecutionLimits::runwayPages(32, 8, 1), 8);
  four.requireRunway(5, ExecutionLimits::runwayPages(32, 4, 1), 4);
  StepCosts costs(kClasses, 64, false);
  const size_t points = costs.points(2).size();
  for (const std::vector<std::vector<double>> &replays :
       {std::vector<std::vector<double>>{}, {std::vector<double>(points - 1, 1.0)},
        {std::vector<double>(points, 1.0), std::vector<double>(points, 0.0)},
        {std::vector<double>(points, NAN)}})
    rejects([&] { costs.measure(2, replays); }, "invalid step replays", "impossible replays were measured");
  require(!costs.measured(2), "refused replays measured a lane count");
  for (const uint32_t lanes : {0U, StepCosts::kLanes + 1}) {
    rejects([&] { static_cast<void>(costs.points(lanes)); }, "invalid step lane count",
            "points of an impossible lane count were given");
    rejects([&] { costs.measure(lanes, std::vector<std::vector<double>>{std::vector<double>(points, 1.0)}); },
            "invalid step lane count", "an impossible lane count was measured");
    rejects([&] { costs.recall(lanes, ""); }, "invalid step lane count",
            "an impossible lane count's line was recalled");
    rejects([&] { static_cast<void>(costs.choose(spansOf(std::vector<std::vector<float>>(lanes)))); },
            "invalid step lane count", "a step of an impossible lane count was chosen");
  }
  // A cost line over such classes; a cycle of fewer rows than its lanes or
  // more than a draft's block a lane, of fewer tokens than its lanes or more
  // than its rows, or of no time; lanes that hold no rows of their own; and
  // lane counts outside a batch.
  for (const std::vector<uint32_t> &classes :
       {std::vector<uint32_t>{}, std::vector<uint32_t>{3, 9}, std::vector<uint32_t>{1, 9, 3}})
    rejects([&] { CostLine line(classes); }, "invalid step classes", "impossible classes made a cost line");
  CostLine line(kStreamedClasses);
  for (const auto &[rows, tokens, milliseconds] :
       {std::tuple{1U, 2U, 10.0}, std::tuple{17U, 2U, 10.0}, std::tuple{4U, 1U, 10.0}, std::tuple{4U, 5U, 10.0},
        std::tuple{4U, 2U, 0.0}, std::tuple{4U, 2U, double(NAN)}})
    rejects([&] { line.observe(2, false, rows, tokens, milliseconds); }, "invalid cost line cycle",
            "an impossible cycle was a sample");
  require(!line.line(2, false), "a refused cycle was a sample");
  const std::vector<std::vector<float>> two(2);
  rejects([&] { static_cast<void>(line.choose(spansOf(two), std::vector<uint32_t>{8}, false)); },
          "a step's lanes hold no rows of their own", "lanes that hold no rows of their own took a step");
  for (const uint32_t lanes : {0U, StepCosts::kLanes + 1}) {
    rejects([&] { static_cast<void>(line.floor(lanes, false)); }, "invalid cost line lane count",
            "an impossible lane count had a floor");
    rejects([&] { line.observe(lanes, false, 4, 4, 10.0); }, "invalid cost line lane count",
            "an impossible lane count's cycle was a sample");
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
    contextPerPass();
    costLineFitsTheSamples();
    costLineChoosesItsSteps();
    stepsOfNoCost();
    invalidSteps();
  } catch (const std::exception &error) {
    std::cerr << "verify_budget_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "verify_budget_test: PASS\n";
  return 0;
}
