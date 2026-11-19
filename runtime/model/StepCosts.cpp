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

uint32_t StepCosts::lanePages(size_t points) {
  return static_cast<uint32_t>((1 + (points + 1) * kWideCap + SPLASH_TARGET_KV_BLOCK_TOKENS - 1) /
                               SPLASH_TARGET_KV_BLOCK_TOKENS);
}

void StepCosts::requireRunway(uint32_t first, uint32_t runway) const {
  for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
    if (first + lanes * lanePages(table(lanes).size()) > runway)
      throw std::logic_error("a lane count's step costs take more pages than the startup runway");
}

namespace {

double median(std::vector<double> times) {
  std::ranges::sort(times);
  const size_t half = times.size() / 2;
  return times.size() % 2 ? times[half] : 0.5 * (times[half - 1] + times[half]);
}

bool timed(double ms) { return std::isfinite(ms) && ms > 0.0; }

} // namespace

void StepCosts::measure(uint32_t lanes, std::span<const std::vector<double>> replays) {
  std::vector<Point> &points = table(lanes);
  if (replays.empty() || std::ranges::any_of(replays, [&](const std::vector<double> &replay) {
        return replay.size() != points.size() ||
               !std::ranges::all_of(replay, timed);
      }))
    throw std::invalid_argument("invalid step replays");
  for (size_t point = 0; point < points.size(); ++point) {
    std::vector<double> times;
    for (const std::vector<double> &replay : replays)
      times.push_back(replay[point]);
    points[point].milliseconds = median(std::move(times));
  }
}

double StepCosts::level(uint32_t lanes, std::span<const double> replays) const {
  if (!measured(lanes) || replays.empty() || !std::ranges::all_of(replays, timed))
    throw std::invalid_argument("invalid level check");
  return table(lanes).back().milliseconds / median({replays.begin(), replays.end()});
}

void StepCosts::forget() {
  for (std::vector<Point> &points : tables_)
    for (Point &point : points) point.milliseconds = 0.0;
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
    return {.step = {lanes, rows, kNarrowCap}, .expected = block.expected[rows - lanes]};
  }
  const Pool pool = poolOf(chances, kWideCap, contexts);
  // The most rows the lanes' chances fill.
  const auto bound = static_cast<uint32_t>(lanes - 1 + pool.expected.size());
  // A step of `rows` rows takes the wide cap where a lane of it holds more
  // than a draft's block.
  const auto capOf = [&](uint32_t rows) { return pool.widest[rows - lanes] > kNarrowCap ? kWideCap : kNarrowCap; };
  // Every step the chances fill, at its price and its passes over history.
  Choice best{.step = {lanes, lanes, kNarrowCap},
              .expected = double(lanes),
              .milliseconds = price(lanes, lanes, kNarrowCap) + perPass_ * pool.passes[0]};
  for (uint32_t rows = lanes + 1; rows <= std::min(bound, points.back().last); ++rows) {
    const uint32_t cap = capOf(rows);
    const double milliseconds = price(lanes, rows, cap) + perPass_ * pool.passes[rows - lanes];
    if (pool.expected[rows - lanes] / milliseconds > best.expected / best.milliseconds)
      best = {.step = {lanes, rows, cap}, .expected = pool.expected[rows - lanes], .milliseconds = milliseconds};
  }
  return best;
}

namespace {

// A cycle's cost at 1, 2, 4 and 8 rows a lane, for one to four lanes (every
// row costs 1): the mean shape of Qwen3.8-Flash-Next's cycles (resident and
// with 12 GB of experts) and Qwen3.6-35B-A3B's (resident and with 4 GB),
// 1.6.2's seed.
constexpr std::array<std::array<double, 4>, CostLine::kLanes> kSeed{{
    {0.60, 0.69, 0.85, 1.0},
    {0.50, 0.58, 0.74, 1.0},
    {0.45, 0.52, 0.70, 1.0},
    {0.40, 0.47, 0.67, 1.0},
}};

} // namespace

CostLine::CostLine(std::span<const uint32_t> classes) : classes_(classes.begin(), classes.end()) {
  if (classes_.empty() || classes_.front() != 1 || !std::ranges::is_sorted(classes_))
    throw std::invalid_argument("invalid step classes");
}

double CostLine::seed(uint32_t lanes, uint32_t rows) {
  if (!lanes || lanes > kLanes || rows < lanes || rows > lanes * kBlockRows)
    throw std::invalid_argument("invalid cost line rows");
  const double perLane = double(rows) / lanes;
  const std::array<double, 4> knots{1.0, 2.0, 4.0, 8.0};
  const std::array<double, 4> &shape = kSeed[lanes - 1];
  uint32_t upper = 1;
  while (upper < 3 && perLane > knots[upper])
    ++upper;
  const double t = (perLane - knots[upper - 1]) / (knots[upper] - knots[upper - 1]);
  return shape[upper - 1] + t * (shape[upper] - shape[upper - 1]);
}

double CostLine::seedSlope(uint32_t lanes) {
  const uint32_t first = lanes, last = lanes * kBlockRows;
  return (seed(lanes, last) - seed(lanes, first)) / double(last - first);
}

const CostLine::Cost &CostLine::table(uint32_t lanes, bool constrained) const {
  if (!lanes || lanes > kLanes)
    throw std::invalid_argument("invalid cost line lane count");
  return tables_[constrained ? 1 : 0][lanes - 1];
}

std::optional<std::pair<double, double>> CostLine::line(uint32_t lanes, bool constrained) const {
  const Cost &cost = table(lanes, constrained);
  if (cost.weight <= 0.0)
    return std::nullopt;
  const double meanRows = cost.rows / cost.weight, meanTime = cost.time / cost.weight;
  const double spread = std::max(cost.rowsSquared / cost.weight - meanRows * meanRows, 0.0);
  const double covariance = cost.rowsTime / cost.weight - meanRows * meanTime;
  // The seed's slope at the cycles' level, worth kPriorCycles cycles whose
  // rows spread by a row a lane.
  const double priorSlope =
      seedSlope(lanes) * meanTime / seed(lanes, static_cast<uint32_t>(std::lround(meanRows)));
  const double priorSpread = double(lanes) * lanes;
  const double slope = std::max(
      (cost.weight * covariance + kPriorCycles * priorSpread * priorSlope) /
          (cost.weight * spread + kPriorCycles * priorSpread),
      0.0);
  return std::pair{meanTime - slope * meanRows, slope};
}

double CostLine::floor(uint32_t lanes, bool constrained) const {
  const Cost &cost = table(lanes, constrained);
  if (cycles_[constrained ? 1 : 0][lanes - 1] % kExploreCycles == kExploreCycles - 1)
    return 0.0;
  // Before a sample, the seed's slope at the rate of the one cycle whose
  // tokens need none: the anchors alone, a token a lane at the seed's cost of
  // a row a lane.
  const auto fit = line(lanes, constrained);
  if (!fit)
    return seedSlope(lanes) * lanes / seed(lanes, lanes);
  // A cycle of no rows costs no less than nothing, so a row costs at most the
  // cycles' time a row. A slope past that (after a stall, or a cold start at
  // many rows) would put the floor over the cycles' tokens a row, and over 1
  // where they retained every row, which leaves out a forced chain's rows,
  // estimated 1, that the target is certain to accept.
  return std::min(fit->second, cost.time / cost.rows) * cost.tokens / cost.time;
}

StepCosts::Choice CostLine::choose(std::span<const std::span<const float>> chances, std::span<const uint32_t> holds,
                                   bool constrained) const {
  const auto lanes = static_cast<uint32_t>(chances.size());
  const auto floor = static_cast<float>(this->floor(lanes, constrained));
  if (holds.size() != chances.size())
    throw std::invalid_argument("a step's lanes hold no rows of their own");
  // Without a floor, a draft's block a lane.
  StepCosts::Choice choice{.step = StepCosts::block(holds), .floor = floor};
  if (!(floor > 0.0F))
    return choice;
  // The lanes' anchors and their chances the floor passes, no more of a lane
  // than the block gives it, or a draft's block of rows in all where they
  // are fewer, and the last rows of the class that holds them.
  uint32_t passed = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const auto above = std::ranges::count_if(chances[lane], [&](float chance) { return chance >= floor; });
    passed += std::min({1 + static_cast<uint32_t>(above), holds[lane], kBlockRows});
  }
  passed = std::max(passed, kBlockRows);
  if (const auto next = std::ranges::upper_bound(classes_, passed); next != classes_.end())
    choice.step.rows = std::min(choice.step.rows, *next - 1);
  return choice;
}

void CostLine::observe(uint32_t lanes, bool constrained, uint32_t rows, uint32_t tokens, double milliseconds) {
  static_cast<void>(table(lanes, constrained));
  if (rows < lanes || rows > lanes * kBlockRows || tokens < lanes || tokens > rows || !std::isfinite(milliseconds) ||
      milliseconds <= 0.0)
    throw std::invalid_argument("invalid cost line cycle");
  const size_t policy = constrained ? 1 : 0;
  ++cycles_[policy][lanes - 1];
  Cost &cost = tables_[policy][lanes - 1];
  // A cycle counts at most twice the larger of its line's price and the
  // mean: the mean keeps the bound up where a stall tilted the line to next
  // to nothing, since a bound at the line alone would hold it there, and a
  // cycle the bound cuts still counts, so that costs that step up and stay
  // up move the line.
  double counted = milliseconds;
  if (const auto fit = line(lanes, constrained))
    counted = std::min(counted, 2.0 * std::max(fit->first + fit->second * rows, cost.time / cost.weight));
  cost.weight = kDecay * cost.weight + 1.0;
  cost.rows = kDecay * cost.rows + rows;
  cost.rowsSquared = kDecay * cost.rowsSquared + double(rows) * rows;
  cost.time = kDecay * cost.time + counted;
  cost.rowsTime = kDecay * cost.rowsTime + rows * counted;
  cost.tokens = kDecay * cost.tokens + tokens;
}

} // namespace splash::model
