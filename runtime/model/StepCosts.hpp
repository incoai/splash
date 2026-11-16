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
// rows and 43 at eight, its experts streamed; Qwen3.6-35B-A3B's row costs
// 0.86 ms at 4 rows and 0.41 at 32), is measured at its first and last rows,
// and a step inside it costs the line between them: at its last rows' cost
// alone, five certain rows of Flash-Next's thinking end lost to two. The
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
  // `mostRows` rows: the rows a step runs in one pass (a streamed-expert
  // store's round, else a verify block a lane); `rising` where a class's
  // cost rises with its rows (a MoE target's).
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
  // A lane count's points' costs from its steps' replays: `replays[r][p]`
  // the wall milliseconds of replay r of point p's step.
  void measure(uint32_t lanes, std::span<const std::vector<double>> replays);
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
  // tokens and its cost (none before the lane count is measured).
  struct Choice final {
    ops::StepRows step;
    double expected = 0.0;
    double milliseconds = 0.0;
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
  [[nodiscard]] static ops::StepRows block(std::span<const uint32_t> holds);
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

// The steps of a target whose experts stream from a store (ExpertStore),
// which no table measured at the start prices: its points, on the start's
// lanes over a cold store, read Qwen3.8-Flash-Next's 1-row step at 25.4 ms,
// below serving's, its 16-row step 41% over serving's, and did not rise with
// their rows, while what a step costs follows the store's state as serving
// moves it. Its cycles take 1.6.2's online cost line (rc5's VerifyBudget):
// a cycle of R rows over L lanes costs a + b R ms, a line a lane count fits
// to its latest cycles, constrained cycles (a lane under a token mask, whose
// mask round trips its time holds) apart. A step takes up to a draft's block
// a lane (1.6.2's rows; Qwen3.8-Flash-Next's best fixed steps all were
// within it), at the narrow cap: of the steps the lanes' chances fill, the
// one with the most expected accepted tokens per cost, the fewest rows among
// equals, by the seed's shape before its line has a sample, as 1.6.2's.
// Every kExploreCycles-th cycle of a lane count, as 1.6.2's, takes a draft's
// block a lane (StepCosts::block). The cost of passes over history
// (StepCosts' k) does not apply: the line's intercept holds the lanes'
// history as it moves.
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

  // The shape of a cycle's cost by its rows (a cycle of every row costs 1),
  // whose slope the line's shrinks toward.
  [[nodiscard]] static double seed(uint32_t lanes, uint32_t rows);
  // The line of a lane count's cycles, constrained or not: its intercept and
  // slope, none before a sample.
  [[nodiscard]] std::optional<std::pair<double, double>> line(uint32_t lanes, bool constrained) const;
  // A cycle's cost of `rows` rows by its line, nondecreasing and positive: a
  // negative intercept leaves the fewest rows a twentieth of the mean.
  [[nodiscard]] double cost(uint32_t lanes, bool constrained, uint32_t rows) const;
  // Whether the next cycle of `lanes` lanes, constrained or not, verifies
  // every row.
  [[nodiscard]] bool explores(uint32_t lanes, bool constrained) const;
  // The step of a cycle of `chances.size()` lanes, each lane's chances best
  // first, each holding `holds` rows (its anchor and the rows past it it may
  // verify), constrained or not, at its expected accepted tokens and its
  // cost (none before its line has a sample, when the seed's shape chooses).
  [[nodiscard]] StepCosts::Choice choose(std::span<const std::span<const float>> chances,
                                         std::span<const uint32_t> holds, bool constrained) const;
  // A cycle of `step`, constrained or not, that committed: it counts toward
  // its lane count's explorations, and, with `milliseconds` (its time from
  // its plan to its results, less its learning, where its lanes' rows filled
  // the step), it is a sample of its line, at most twice the larger of its
  // line's price and the mean (a stall's bound).
  void observe(const ops::StepRows &step, bool constrained, std::optional<double> milliseconds);

private:
  // The weighted sums of the latest cycles' rows R and times t.
  struct Cost final {
    double weight = 0.0;
    double rows = 0.0;
    double rowsSquared = 0.0;
    double time = 0.0;
    double rowsTime = 0.0;
  };

  [[nodiscard]] const Cost &table(uint32_t lanes, bool constrained) const;

  std::array<std::array<Cost, kLanes>, 2> tables_{};
  std::array<std::array<uint64_t, kLanes>, 2> cycles_{};
};

} // namespace splash::model
