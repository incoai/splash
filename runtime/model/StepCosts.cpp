#include "model/StepCosts.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace splash::model {
namespace {

// A step of each rows T over `chances.size()` lanes of up to `cap` rows a
// lane, each lane's chances its first cap - 1 (which its tree took in
// order): T's expected accepted tokens, its anchors and its T - lanes
// largest chances, expected[T - lanes]; and the most rows a lane of it
// holds, its anchor and its chances among them, widest[T - lanes].
struct Pool final {
  std::vector<double> expected;
  std::vector<uint32_t> widest;
  // Its passes over history, the sum over lanes of a lane's history in
  // thousands of tokens times the passes its rows take (`contexts` given).
  std::vector<double> passes;
};

// The passes a lane of `rows` rows takes, a pass a draft's block of rows.
uint32_t passesOf(uint32_t rows) { return (rows + StepCosts::kNarrowCap - 1) / StepCosts::kNarrowCap; }

Pool poolOf(std::span<const std::span<const float>> chances, uint32_t cap, std::span<const uint64_t> contexts = {}) {
  const auto history = [&](size_t lane) { return lane < contexts.size() ? double(contexts[lane]) / 1000.0 : 0.0; };
  double anchors = 0.0;
  for (size_t lane = 0; lane < chances.size(); ++lane)
    anchors += history(lane);
  Pool pool{{double(chances.size())}, {1}, {anchors}};
  std::vector<uint32_t> held(chances.size(), 1);
  const auto next = [&](size_t lane) { return chances[lane][held[lane] - 1]; };
  for (;;) {
    // The lane whose next chance is the largest; among equal ones, the lane
    // holding the fewest rows, then the earlier: a forced chain's certain
    // rows beside a first cycle's take the narrow cap where they fit it.
    std::optional<size_t> taker;
    for (size_t lane = 0; lane < chances.size(); ++lane)
      if (held[lane] < std::min<size_t>(chances[lane].size() + 1, cap) &&
          (!taker || next(lane) > next(*taker) || (next(lane) == next(*taker) && held[lane] < held[*taker])))
        taker = lane;
    if (!taker)
      return pool;
    pool.expected.push_back(pool.expected.back() + std::clamp(double(next(*taker)), 0.0, 1.0));
    ++held[*taker];
    pool.widest.push_back(std::max(pool.widest.back(), held[*taker]));
    pool.passes.push_back(pool.passes.back() +
                          (passesOf(held[*taker]) - passesOf(held[*taker] - 1)) * history(*taker));
  }
}

} // namespace

StepCosts::StepCosts(std::span<const uint32_t> classes, uint32_t mostRows, bool rising) {
  if (classes.empty() || classes.front() != 1 || !std::ranges::is_sorted(classes) || !mostRows)
    throw std::invalid_argument("invalid step classes");
  for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
    std::vector<Point> &points = tables_[lanes - 1];
    const uint32_t most = std::min(lanes * kWideCap, mostRows), narrow = lanes * kNarrowCap;
    if (most <= lanes)
      throw std::invalid_argument("a step's most rows hold nothing past its lanes' anchors");
    // A class starts where the kernels change and where the cap does.
    std::vector<uint32_t> starts(classes.begin(), classes.end());
    starts.push_back(narrow + 1);
    std::ranges::sort(starts);
    const auto [end, last] = std::ranges::unique(starts);
    starts.erase(end, last);
    for (size_t i = 0; i < starts.size(); ++i) {
      const uint32_t first = std::max(starts[i], lanes);
      const uint32_t lastRow = std::min(i + 1 < starts.size() ? starts[i + 1] - 1 : most, most);
      // A class of the anchors alone yields a token a lane: the next class
      // beats it wherever a lane's best row has a chance of a few percent.
      if (first > lastRow || lastRow == lanes)
        continue;
      const uint32_t cap = lastRow <= narrow ? kNarrowCap : kWideCap;
      if (rising && first < lastRow) {
        points.push_back({.first = first, .last = first, .cap = cap});
        points.push_back({.first = first + 1, .last = lastRow, .cap = cap, .rises = true});
      } else {
        points.push_back({.first = first, .last = lastRow, .cap = cap});
      }
    }
    // What the wide cap adds to a step of a draft's block a lane, where a
    // lane takes more than a draft's block: from two lanes, after the narrow
    // cap's point there.
    const auto atNarrow = std::ranges::find(points, narrow, &Point::last);
    if (lanes > 1 && atNarrow != points.end())
      points.insert(atNarrow + 1, {.first = narrow, .last = narrow, .cap = kWideCap});
  }
}

std::vector<StepCosts::Point> &StepCosts::table(uint32_t lanes) {
  if (!lanes || lanes > kLanes)
    throw std::invalid_argument("invalid step lane count");
  return tables_[lanes - 1];
}

const std::vector<StepCosts::Point> &StepCosts::table(uint32_t lanes) const {
  if (!lanes || lanes > kLanes)
    throw std::invalid_argument("invalid step lane count");
  return tables_[lanes - 1];
}

std::span<const StepCosts::Point> StepCosts::points(uint32_t lanes) const { return table(lanes); }

std::vector<ops::StepRows> StepCosts::steps(uint32_t lanes) const {
  std::vector<ops::StepRows> steps;
  for (const Point &point : table(lanes))
    steps.push_back({lanes, point.last, point.cap});
  return steps;
}

void StepCosts::measure(uint32_t lanes, std::span<const std::vector<double>> replays) {
  std::vector<Point> &points = table(lanes);
  if (replays.empty() || std::ranges::any_of(replays, [&](const std::vector<double> &replay) {
        return replay.size() != points.size() ||
               std::ranges::any_of(replay, [](double ms) { return !std::isfinite(ms) || ms <= 0.0; });
      }))
    throw std::invalid_argument("invalid step replays");
  for (size_t point = 0; point < points.size(); ++point) {
    std::vector<double> times;
    for (const std::vector<double> &replay : replays)
      times.push_back(replay[point]);
    std::ranges::sort(times);
    const size_t half = times.size() / 2;
    points[point].milliseconds = times.size() % 2 ? times[half] : 0.5 * (times[half - 1] + times[half]);
  }
}

std::string StepCosts::text(uint32_t lanes) const {
  if (!measured(lanes))
    throw std::logic_error("an unmeasured lane count has no costs to keep");
  std::ostringstream line;
  line << std::setprecision(9);
  const char *separator = "";
  for (const Point &point : table(lanes)) {
    line << separator << point.last << ':' << point.cap << ':' << point.milliseconds;
    separator = " ";
  }
  return line.str();
}

void StepCosts::recall(uint32_t lanes, std::string_view line) {
  std::vector<Point> &points = table(lanes);
  std::istringstream text{std::string(line)};
  std::vector<double> costs;
  for (const Point &point : points) {
    uint32_t last = 0, cap = 0;
    double milliseconds = 0.0;
    char colon = 0, second = 0;
    if (!(text >> last >> colon >> cap >> second >> milliseconds) || colon != ':' || second != ':' ||
        last != point.last || cap != point.cap || !std::isfinite(milliseconds) || milliseconds <= 0.0)
      return;
    costs.push_back(milliseconds);
  }
  if (!(text >> std::ws).eof())
    return;
  for (size_t point = 0; point < points.size(); ++point)
    points[point].milliseconds = costs[point];
}

ops::StepRows StepCosts::block(std::span<const uint32_t> holds) {
  ops::StepRows step{.lanes = uint32_t(holds.size()), .rows = 0, .laneCap = kNarrowCap};
  for (const uint32_t held : holds)
    step.rows += std::min(held, kNarrowCap);
  ops::requireStepRows(step);
  return step;
}

ops::StepRows StepCosts::forced(uint32_t lanes, uint32_t rows, std::optional<uint32_t> laneCap) {
  const uint32_t clamped = std::clamp(rows, lanes, lanes * laneCap.value_or(kWideCap));
  return {lanes, clamped, laneCap.value_or(ops::verifyLaneCap(clamped, lanes))};
}

bool StepCosts::measured(uint32_t lanes) const {
  const std::vector<Point> &points = table(lanes);
  return !points.empty() && std::ranges::all_of(points, [](const Point &point) { return point.milliseconds > 0.0; });
}

double StepCosts::price(uint32_t lanes, uint32_t rows, uint32_t cap) const {
  const std::vector<Point> &points = table(lanes);
  const uint32_t narrow = lanes * kNarrowCap;
  // What the wide cap adds at a draft's block a lane, which a narrow class's
  // step at the wide cap adds to its point (none at one lane, whose narrow
  // steps hold no lane past the narrow cap). The wide cap runs the narrow
  // cap's kernels over more rows a lane, so it adds no less than none: a
  // measurement below that is noise (Qwen3.6-35B-A3B's three lanes, -0.4%).
  const auto atNarrow = [&](uint32_t c) {
    return std::ranges::find_if(points, [&](const Point &p) { return p.last == narrow && p.cap == c; });
  };
  const double wider = atNarrow(kWideCap) == points.end()
                           ? 0.0
                           : std::max(atNarrow(kWideCap)->milliseconds - atNarrow(kNarrowCap)->milliseconds, 0.0);
  // The anchors alone, a class no table measures, cost at most the lowest
  // point.
  if (rows <= lanes)
    return points.front().milliseconds;
  auto point = points.begin();
  while (point != points.end() && point->last < rows)
    ++point;
  if (point == points.end())
    throw std::invalid_argument("a step of more rows than its lane count's table");
  // A rising class's step costs the line from its first rows' point.
  const double atRows = point->rises && point != points.begin()
                            ? std::prev(point)->milliseconds +
                                  (point->milliseconds - std::prev(point)->milliseconds) *
                                      (double(rows) - std::prev(point)->last) / (point->last - std::prev(point)->last)
                            : point->milliseconds;
  return atRows + (cap > point->cap ? wider : 0.0);
}

void StepCosts::observeContext(const ops::StepRows &step, std::span<const uint32_t> laneRows,
                               std::span<const uint64_t> contexts, double milliseconds) {
  if (!std::isfinite(milliseconds) || milliseconds <= 0.0 || laneRows.size() != step.lanes ||
      contexts.size() != step.lanes)
    throw std::invalid_argument("invalid step sample");
  if (!measured(step.lanes))
    return;
  double passes = 0.0;
  for (size_t lane = 0; lane < step.lanes; ++lane)
    passes += passesOf(std::max(laneRows[lane], 1U)) * double(contexts[lane]) / 1000.0;
  // A stall counts at twice the cycle's predicted cost, as 1.6.2's line
  // counted one: dropped, every cycle at a history whose passes the table
  // lacks would read as a stall from a k of none, which would never learn.
  const double table = price(step.lanes, step.rows, step.laneCap);
  const double wall = std::min(milliseconds, 2.0 * (table + perPass_ * passes));
  passesTimesExcess_ = kContextDecay * passesTimesExcess_ + passes * (wall - table);
  passesSquared_ = kContextDecay * passesSquared_ + passes * passes;
  perPass_ = passesSquared_ > 0.0 ? std::max(passesTimesExcess_ / passesSquared_, 0.0) : 0.0;
}

StepCosts::Choice StepCosts::choose(std::span<const std::span<const float>> chances,
                                    std::span<const uint64_t> contexts) const {
  const auto lanes = static_cast<uint32_t>(chances.size());
  const std::vector<Point> &points = table(lanes);
  if (!measured(lanes)) {
    // A draft's block a lane at the narrow cap, --verify-rows all's step, as
    // far as the lanes' chances fill it.
    const Pool block = poolOf(chances, kNarrowCap);
    const uint32_t rows = std::min(lanes - 1 + uint32_t(block.expected.size()), points.back().last);
    return {{lanes, rows, kNarrowCap}, block.expected[rows - lanes], 0.0};
  }
  const Pool pool = poolOf(chances, kWideCap, contexts);
  // The most rows the lanes' chances fill.
  const auto bound = static_cast<uint32_t>(lanes - 1 + pool.expected.size());
  // A step of `rows` rows takes the wide cap where a lane of it holds more
  // than a draft's block.
  const auto capOf = [&](uint32_t rows) { return pool.widest[rows - lanes] > kNarrowCap ? kWideCap : kNarrowCap; };
  // Every step the chances fill, at its price and its passes over history.
  Choice best{{lanes, lanes, kNarrowCap}, double(lanes), price(lanes, lanes, kNarrowCap) + perPass_ * pool.passes[0]};
  for (uint32_t rows = lanes + 1; rows <= std::min(bound, points.back().last); ++rows) {
    const uint32_t cap = capOf(rows);
    const double milliseconds = price(lanes, rows, cap) + perPass_ * pool.passes[rows - lanes];
    if (pool.expected[rows - lanes] / milliseconds > best.expected / best.milliseconds)
      best = {{lanes, rows, cap}, pool.expected[rows - lanes], milliseconds};
  }
  return best;
}

} // namespace splash::model
