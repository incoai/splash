#pragma once

// The costs of a decode cycle's steps, and each cycle's step: the rows T
// with the most expected accepted tokens per cost (Runtime's decode cycle,
// --verify-rows auto).
//
// A lane count's steps fall into classes (ops::ExecutionPlans::
// decodeRowClasses, split at the lane cap's eight rows a lane): inside one
// the kernels a step runs do not change. A dense target's class is flat
// (Qwen3.8-27B on a 48 GB M5 Pro, four lanes: 24 rows to 32 cost 3.5% more,
// while 16 to 24, crossing 17, cost 26%), so it costs what a step of its last
// rows does, its point, and a step fills the class it takes. A class whose
// cost rises with its rows, a MoE target's, since its routed experts grow
// with them (Qwen3.8-Flash-Next's one lane routes 17 experts a layer at two
// rows and 43 at eight; Qwen3.6-35B-A3B's row costs 0.86 ms at 4 rows and
// 0.41 at 32), is measured at its first and last rows, and a step inside it
// costs the line between them: at its last rows' cost alone, five certain
// rows of Flash-Next's thinking end lost to two (measured while the table
// priced its streamed store too; such a store's cycles now take CostLine).
// The
// points are measured before the runtime serves, by forced steps of lanes of
// its own (Runtime::warmupDecodeBatch), each the median of its replays, and
// stay as measured.
//
// A step's lane cap follows its rows: the wide cap where the lanes' chances
// give a lane more than a draft's block of them, priced at what the wide cap
// adds at a draft's block a lane. On a 48 GB M5 Pro that is within 1.1% on
// Qwen3.8-27B and Qwen3.6-35B-A3B, but the wide cap adds 1.7% to 2.6% to the
// 35B's steps of a row a lane, so a step no lane needs it for keeps the
// narrow one.
//
// A cycle's rows come from the lanes' chances of the last cycle, before this
// cycle's draws, and from the table: no row is admitted by its own draws, so
// a sampled lane stays exact.

#include "metal/abi/ExecutionGeometry.h"
#include "ops/LaneRows.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::model {

class StepCosts final {
public:
  static constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  // The narrow lane cap, a draft's block, and the wide one, a verify block.
  static constexpr uint32_t kNarrowCap = ops::kNarrowLaneRows;
  static constexpr uint32_t kWideCap = SPLASH_TARGET_VERIFY_ROWS;
  // A point's cost is the median of three replays of its step, each from the
  // lanes' prompts anew: one replay that another app's rendering slowed
  // cannot move it. Qwen3.6-35B-A3B's consecutive cycles at one step differ
  // by a median of 2-6% and a p90 of 5-10%, as routing varies; Qwen3.8-27B's
  // by 0.5% and 1.1%.
  static constexpr uint32_t kReplays = 3;
  // Two replays that other work slowed can: then a point's median is past its
  // fastest replay by what slowed them. A lane count's costs whose every
  // median is at most kSteadySpread past its point's fastest replay are kept
  // for later starts (Runtime::warmupDecodeBatch); others are measured again
  // at the next start. A fifth is twice the 35B's p90 above: routing alone
  // puts a median past its fastest replay by no more than its cycles differ.
  // On a 36 GB M5 Max with nothing else on its GPU, the runtime oracle's
  // tables were within 6.7-10.4% on Qwen3.6-35B-A3B and 1.2-6.3% on
  // Qwen3.8-27B.
  static constexpr double kSteadySpread = 0.2;
  // A step costs its price plus k times its passes over history: the sum
  // over its lanes of each lane's history, in thousands of tokens, times the
  // attention passes its rows take, a pass a draft's block of rows. The start
  // measures its points at a few tokens of history, and each pass reads a
  // lane's whole history: Qwen3.6-35B-A3B on a 48 GB M5 Pro, a forced step's
  // wall at 32K less its wall at 2K is 2.2-3.6 ms a pass at every lane count
  // and step (two lanes: +5.7 ms at 2 passes, +10.9 at 4), where one level a
  // lane count would price 4 passes as 2. k, one for the model, from none at
  // the start and kept in the process alone, is the least-squares slope
  // through the origin of eligible cycles' wall less their price on their
  // passes over history, each cycle's weight decaying by a 64th a cycle, the
  // acceptance fit's refit period (VerifyBudget::kRefitCycles), a half-life of
  // 44 cycles, a second or two of decode; a cycle of more than twice its
  // predicted cost, a stall, counts at twice it, so that k grows to a cost
  // of passes the table lacks however far it is from none. At short
  // histories k is no pass's cost but what serving's cycles cost over the
  // start's table, over a small regressor (Qwen3.6-35B-A3B at 512-token
  // prompts: 0.49-1.43 on a 24 GB M6 and 0.56-5.16 on a 16 GB M4, against the
  // 0.07-0.12 of a pass over 30K above), which does little harm: there a
  // step of one pass a lane is priced within 0.3 ms of its wall and a lane's
  // second pass 3-6% over, and one sample at a long history brings k back.
  // The 64th is a choice, not a measured optimum: replayed through fits of a
  // 32nd, a 64th and a 128th, Qwen3.6-35B-A3B's logged cycles are priced
  // 1.21, 1.22 and 1.24 ms off their walls at a steady 31K on an M5 Max (4%
  // of a cycle), 1.8-2.6, 2.1-3.1 and 2.4-3.7 ms off over the 16 cycles after
  // a jump from 512 tokens to 16K on an M6 and 0.7-0.8, 0.7-1.0 and 0.8-1.3
  // over the 400 back at 512, where k is the table's level error again, and
  // the cycle after one slowed five times is overpriced 1.2, 0.6 and 0.4 ms:
  // a faster fit follows a history that jumps sooner, a slower one averages
  // a stall away, and neither differs from the 64th by a millisecond a cycle.
  static constexpr double kContextDecay = 1.0 - 1.0 / 64.0;

  // A lane count's steps over `classes`, the fewest rows of each class of a
  // decode step's kernels (ops::ExecutionPlans::decodeRowClasses), of up to
  // `mostRows` rows, the rows a step runs in one pass, a verify block a lane
  // (a streamed-expert store's cycles take no table: CostLine); `rising`
  // where a class's cost rises with its rows (a MoE target's).
  StepCosts(std::span<const uint32_t> classes, uint32_t mostRows, bool rising);

  // A class of a lane count's steps, or of a rising class the rows past its
  // first: its rows from `first` to `last`, at the lane cap `cap` its last
  // rows need, and its point's cost, the wall milliseconds of a step of its
  // last rows (none until measured). Where it `rises`, a step inside it costs
  // the line from the point before it, its class's first rows, to its own.
  // From two lanes, one more point is the wide cap's at a draft's block a
  // lane.
  struct Point final {
    uint32_t first = 0;
    uint32_t last = 0;
    uint32_t cap = 0;
    double milliseconds = 0.0;
    bool rises = false;
  };
  // A lane count's points, in rising rows and, at equal rows, rising caps.
  [[nodiscard]] std::span<const Point> points(uint32_t lanes) const;
  // The steps that measure a lane count's points, in the points' order.
  [[nodiscard]] std::vector<ops::StepRows> steps(uint32_t lanes) const;
  // The KV pages of a lane of a measurement of `points` points
  // (Runtime::measureDecodeSteps): its prompt's token, a verify block a
  // point and one more.
  static uint32_t lanePages(size_t points);
  // Throws where a lane count's measurement, its lanes laid from page
  // `first`, runs past the `runway` pages it is measured on.
  void requireRunway(uint32_t first, uint32_t runway) const;
  // A lane count's points' costs from its steps' replays: `replays[r][p]`
  // the wall milliseconds of replay r of point p's step. Returns its spread:
  // the most a point's median is past its fastest replay, a fraction of it.
  double measure(uint32_t lanes, std::span<const std::vector<double>> replays);
  // Whether a lane count's points are measured.
  [[nodiscard]] bool measured(uint32_t lanes) const;
  // A measured lane count's points as a line of text, each point's last
  // rows, lane cap and cost, which recall() takes back.
  [[nodiscard]] std::string text(uint32_t lanes) const;
  // A lane count's costs from a line text() gave, if it names the lane
  // count's points, each at a cost above none; a line of another build's
  // classes leaves them unmeasured.
  void recall(uint32_t lanes, std::string_view line);

  // A cycle's step and what the choice weighed: the step's expected accepted
  // tokens and its cost (none before the lane count is measured, and none
  // where a floor sizes the cycle); and its floor, the estimate below which
  // its lanes take no row of the step past their anchors (VerifyRowsParams;
  // none but a streamed-expert target's, CostLine).
  struct Choice final {
    ops::StepRows step;
    double expected = 0.0;
    double milliseconds = 0.0;
    float floor = 0.0F;
  };
  // The step of a cycle of `chances.size()` lanes, each lane's chances best
  // first (VerifyBudget::laneChances): of the steps that fill a class as far
  // as the lanes' chances go, the one with the most expected accepted tokens
  // per cost, and among equals the fewest rows; before its lane count is
  // measured, a draft's block a lane at the narrow cap, --verify-rows all's
  // step, as far as the lanes' chances fill it. `contexts`, each lane's
  // history in tokens, where given, adds each step's passes over history at
  // k (contextCost).
  [[nodiscard]] Choice choose(std::span<const std::span<const float>> chances,
                              std::span<const uint64_t> contexts = {}) const;
  // A cycle of `step` that could give a sample, in `milliseconds` from its
  // plan to its results, its lanes' rows `laneRows` and histories `contexts`:
  // where its lane count is measured, its wall, at most twice its predicted
  // cost, less the table's price on its passes over history teaches k.
  void observeContext(const ops::StepRows &step, std::span<const uint32_t> laneRows,
                      std::span<const uint64_t> contexts, double milliseconds);
  // k: milliseconds a pass over a thousand tokens of history (none before a
  // cycle observed one).
  [[nodiscard]] double contextCost() const noexcept { return perPass_; }
  // A cycle's step under --verify-rows all: a draft's block a lane, no more
  // than each lane holds (`holds`, its anchor and the rows past it it may
  // verify), at the narrow cap, so that its outputs repeat.
  [[nodiscard]] static ops::StepRows block(std::span<const uint32_t> holds, uint32_t draftRows = kNarrowCap);
  // A forced step of `rows` rows over `lanes` lanes, a row a lane to a
  // verify block's, at `laneCap` where given (the start's measured points,
  // RuntimeInstrumentation::forceStepRows), else at the cap its rows need
  // (ops::verifyLaneCap).
  [[nodiscard]] static ops::StepRows forced(uint32_t lanes, uint32_t rows, std::optional<uint32_t> laneCap);

private:
  [[nodiscard]] std::vector<Point> &table(uint32_t lanes);
  [[nodiscard]] const std::vector<Point> &table(uint32_t lanes) const;

  std::array<std::vector<Point>, kLanes> tables_;
  // The table's price of a step of `rows` rows at `cap` over `lanes` lanes,
  // as choose weighs it: its class's point, or a rising class's line, plus
  // what the wide cap adds at a draft's block a lane where a narrow class's
  // step takes it.
  [[nodiscard]] double price(uint32_t lanes, uint32_t rows, uint32_t cap) const;
  // The decayed sums of passes over history times wall less price, and of
  // its square, and their slope k.
  double passesTimesExcess_ = 0.0;
  double passesSquared_ = 0.0;
  double perPass_ = 0.0;
};

// The cycles of a target whose experts stream from a store (ExpertStore),
// which no table measured at the start prices: its points, on the start's
// lanes over a cold store, read Qwen3.8-Flash-Next's 1-row step at 25.4 ms,
// below serving's, its 16-row step 41% over serving's, and did not rise with
// their rows, while what a step costs follows the store's state as serving
// moves it. Its cycles take 1.6.2's online cost line (its VerifyBudget): a
// cycle whose lanes verify R rows in all over L lanes costs a + b R ms, a
// line a lane count fits to its latest cycles, constrained cycles (a lane
// under a token mask, whose mask round trips its time holds) apart.
//
// Tokens over time across such cycles are the most where every cycle buys
// its rows at one price: a row is verified exactly when its chance of adding
// a token is worth its b ms at the going rate, its estimate at least the
// floor, b times the tokens per ms of the lane count's latest cycles. 1.6.2
// chose on the device the ranked rows' prefix of the most expected tokens
// per cost, which for a cost convex in the rows, as its line is, ends at the
// first row estimated under b times that cycle's own tokens per ms so far:
// the floor with the cycle's own rate in the latest cycles' place, as exact
// (VerifyRowsParams), which buys too many rows on a poor cycle and too few
// on a rich one. Replayed over Qwen3.6-35B-A3B's and Qwen3.8-27B's logged
// trees (code at 31K, one and two lanes) under lines of Qwen3.8-Flash-Next's
// shape, 38 + 6 R and 32 + 1.5 R ms, the floor took 0.6% to 2.8% less time a
// token than that prefix: an estimate, not a run.
//
// A cycle applies the one floor twice. After its draft the device applies
// it to the trees' current estimates, which pick the rows: its lanes take
// the step's rows estimated at least the floor (VerifyRowsParams), and the
// others are rows of no lane, which route no expert. Chosen on the host
// before the cycle, from the lanes' chances of the last one, the rows fell
// on the wrong cycles: Qwen3.8-Flash-Next's one lane on a 24 GB Mac mini M6
// (Q2_0, a 6656M store) verified 3.04 rows a cycle for 2.17 tokens at 38.68
// to 39.04 ms a token, every run behind 1.6.2's 2.75 rows for 2.15 tokens at
// 37.83 to 38.08, and with the floor on the device 2.61 to 2.77 rows for
// 2.11 to 2.18 tokens at 37.73 to 38.72; Qwen3.6-35B-A3B's on a 16 GB Mac
// mini M4 (UD-Q4_K_M, a 4640M store) ran at 53.89 and 54.86, 1.6.2's at
// 48.77 to 51.02, and with the floor at 50.59 to 51.24. Before its draft the
// host applies the floor to those last chances, which so pick only the
// kernels that run: the step's rows are the last of the kernel class
// (ops::ExecutionPlans::decodeRowClasses) that holds the rows the floor
// passes, at most a draft's block a lane (1.6.2's rows) at the narrow cap,
// as a step's rows size its projections, its head and its embedding
// whatever its lanes take of them (Flash-Next's start table on a 48 GB M5
// Pro, four lanes over a 16 GB store: a step's 17th row adds 3.65 ms and
// its 25th 4.15 where a row inside a class adds about 1.5, on cycles of
// about 57 ms), and no fewer than a draft's block of rows in all. A class
// of one row count leaves the current estimates nothing to add, and on
// Apple9, Apple10 and Apple11 the classes under four rows hold one row
// count each (the next, of four to eight, is the smallest that spans more):
// the M6's lane, whose last chances pass fewer than three rows in most
// cycles, verified 2.50 to 2.61 rows a cycle for 2.05 to 2.09 tokens at
// 38.41 to 40.75 ms a token without the bound, where a row costs about 15
// ms and a step under four rows within 2 to 4 ms of what its rows do.
//
// Every kExploreCycles-th cycle of a lane count, as 1.6.2's, the floor is
// none and the step a draft's block a lane (StepCosts::block). Before its
// line has a sample the floor is the seed's, as 1.6.2's first cycles took
// their rows by the seed's shape, and not none: a block a lane over a store
// its prompt just filled is the dearest cycle there is (the M6's lane: 344
// ms for 8 rows, where a later cycle of 8 rows takes 97 to 108). The cost of
// passes over history (StepCosts' k) does not apply: the line's intercept
// holds the lanes' history as it moves.
//
// What the line does not yet tell: a lane's first two cycles after its
// prompt read into a starved store the experts its answer starts on (the
// M6's lane: 312 and 185 ms for cycles of 6 and 5 rows, each retained, where
// such cycles later take about 130 and 113), and as the line's samples of
// the most rows at the most time they hold its slope at 33 to 39 ms a row
// against the 17 it settles at, and its floor at 0.70 to 0.83 against 0.40,
// until its next exploration: its first 31 cycles there run at 46 ms a token
// against 37 to 38 after them, and the M4's at 64 to 69 against 50. Leaving
// those two cycles out of the samples did not help a line that starts with
// none (the M4: 51.35 and 52.71 ms a token against 50.59 to 51.24): it then
// starts from the seed's slope, a third of such a store's, and verifies
// whole blocks.
class CostLine final {
public:
  static constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  static constexpr uint32_t kBlockRows = StepCosts::kNarrowCap;
  // Every kExploreCycles-th cycle of a lane count verifies every row, which
  // keeps the rows of the cost fit spread.
  static constexpr uint32_t kExploreCycles = 32;
  // Each cycle weighs kDecay times the next (about the last 33 cycles), as
  // what a row costs follows the requests served.
  static constexpr double kDecay = 0.97;
  // The slope shrinks toward the seed's as much as kPriorCycles spread
  // cycles would pull it.
  static constexpr double kPriorCycles = 5.0;

  // Over `classes`, the fewest rows of each class of a decode step's kernels
  // (ops::ExecutionPlans::decodeRowClasses).
  explicit CostLine(std::span<const uint32_t> classes);

  // The shape of a cycle's cost by its rows (a cycle of every row costs 1),
  // whose slope the line's shrinks toward.
  [[nodiscard]] static double seed(uint32_t lanes, uint32_t rows);
  // The line of a lane count's cycles, constrained or not: its intercept and
  // slope, none before a sample.
  [[nodiscard]] std::optional<std::pair<double, double>> line(uint32_t lanes, bool constrained) const;
  // The floor of the next cycle of `lanes` lanes, constrained or not: its
  // line's slope, at most its latest cycles' time a row, times their tokens
  // per ms, the tokens they retained over their times as the line counted
  // them, so at most their tokens a row; before the line has a sample, the
  // seed's slope times a cycle of the anchors alone's tokens per cost by the
  // seed; and none where the cycle explores.
  [[nodiscard]] double floor(uint32_t lanes, bool constrained) const;
  // The step of a cycle of `chances.size()` lanes, constrained or not, each
  // lane's chances best first (VerifyBudget::laneChances), each holding
  // `holds` rows (its anchor and the rows past it it may verify), and its
  // floor: the last rows of the class that holds the lanes' anchors and
  // their chances the floor passes, or a draft's block of rows where they
  // are fewer, at most a draft's block a lane; without a floor, a draft's
  // block a lane. It weighs no expected tokens and no cost.
  [[nodiscard]] StepCosts::Choice choose(std::span<const std::span<const float>> chances,
                                         std::span<const uint32_t> holds, bool constrained) const;
  // A cycle of `lanes` lanes, constrained or not, that committed, whose lanes
  // verified `rows` rows in all and retained `tokens` tokens, in
  // `milliseconds` from its plan to its results less its learning: it counts
  // toward its lane count's explorations and is a sample of its line, at
  // most twice the larger of its line's price and the mean (a stall's
  // bound).
  void observe(uint32_t lanes, bool constrained, uint32_t rows, uint32_t tokens, double milliseconds);

private:
  // The weighted sums of the latest cycles' rows R, times t and tokens.
  struct Cost final {
    double weight = 0.0;
    double rows = 0.0;
    double rowsSquared = 0.0;
    double time = 0.0;
    double rowsTime = 0.0;
    double tokens = 0.0;
  };

  // The seed's slope: what a row adds to a cycle's cost, from a row a lane
  // to a draft's block a lane.
  [[nodiscard]] static double seedSlope(uint32_t lanes);
  [[nodiscard]] const Cost &table(uint32_t lanes, bool constrained) const;

  std::vector<uint32_t> classes_;
  std::array<std::array<Cost, kLanes>, 2> tables_{};
  std::array<std::array<uint64_t, kLanes>, 2> cycles_{};
};

} // namespace splash::model
