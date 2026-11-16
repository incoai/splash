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
  // step, as far as the lanes' chances fill it.
  [[nodiscard]] Choice choose(std::span<const std::span<const float>> chances) const;
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
};

} // namespace splash::model
