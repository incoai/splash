#pragma once

// The fakes' time: one Mac and target (TimingProfile), fitted to recorded
// commands (TimingProfile.cpp), the work the fake model reports for a
// prefill command or pass (TimedWork) and the times its commands take
// (prefillTimes, decodeTimes).

#include "TestImmediateTicket.hpp"
#include "model/CommandCost.hpp"
#include "model/Model.hpp"
#include "model/WorkFormulas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <utility>

namespace splash::test {

// Two measured points of a quantity against the experts resident (slots),
// linear through them and kept to its physical range by the caller.
using SlotCurve = std::array<std::pair<uint32_t, double>, 2>;

struct TimingProfile final {
  std::string_view name;
  // Layer l (from 0) attends when (l + 1) % attentionEvery == 0, at `reach`
  // (a window, or block-sparse); with fullAttentionEvery, an attending layer
  // l with l % fullAttentionEvery == 0 attends every key whatever `reach`
  // says (a windowed target's full-attention layers).
  uint32_t layers = 48, attentionEvery = 4, fullAttentionEvery = 0;
  model::AttentionReach reach{};
  // A command's or pass's turnaround u; a measured u replaces the default.
  double turnaroundMilliseconds = 0;
  // GPU work of a whole command (every layer): per chunk the rows fill, per
  // row, per 1e6 query-key pairs and block-index pairs over the layers run,
  // and per distinct expert a layer routes.
  double chunkMilliseconds = 0, rowMilliseconds = 0, megapairMilliseconds = 0, indexMegapairMilliseconds = 0;
  double expertMilliseconds = 0;
  // SSD reads of a prefill: weights read on demand and spilled K/V.
  double weightBytesPerMillisecond = 0, spilledKvBytesPerMillisecond = 0;
  // A command's host steps take at least hostFloorMilliseconds (every
  // layer's); compute hides hiddenShare of its reads.
  double hostFloorMilliseconds = 0, hiddenShare = 0;
  // A decode command's wall by width at measuredResidentShare; decode reads
  // more or fewer experts at another residency at this rate, and each lane
  // whose history holds Index pages adds spillDecodeMilliseconds of hot-block
  // reads.
  std::array<double, model::ExecutionLimits::maximumBatchWidth> decodeMilliseconds{};
  double decodeWeightBytesPerMillisecond = 0;
  double spillDecodeMilliseconds = 0;
  // Routed experts: zero for a dense target. `routing` is the measured
  // distinct experts one layer of a command of `rows` rows routes (rows,
  // experts), log-linear between its points.
  struct Experts final {
    uint32_t experts = 0, topK = 0;
    uint64_t expertBytes = 0;
    std::array<std::pair<uint32_t, double>, 4> routing{};
  } experts{};
  // The distinct experts a layer routes for a prompt of one repeated token
  // (the warm-up's).
  uint32_t warmupDistinctExperts = 0;
  // A decode's expert reads per output token and a prefill's share of routed
  // experts found resident, against the slots resident.
  struct ExpertCurve final {
    SlotCurve decodeBytesPerToken, prefillHit;
  } curve{};
  // A Full page of every attention layer's K/V, as a streaming lane reads its
  // Index pages' history.
  uint64_t spilledPageBytes = 0;
  double measuredResidentShare = 1.0, noise = 0.0;
  uint64_t seed = 1;
};

// A recorded prefill command, and a recorded prompt of commands of
// ExecutionLimits::prefillTokenBudget rows from `position` (TTFT, for the
// targets whose commands were not recorded one by one).
struct MeasuredCommand final {
  uint32_t rows = 0;
  double wallMilliseconds = 0, hostWaitMilliseconds = 0;
  bool warmup = false;  // a prompt of one repeated token
};
struct MeasuredPrompt final {
  uint32_t position = 0, rows = 0;
  double wallMilliseconds = 0;
};

[[nodiscard]] inline double curveAt(const SlotCurve &curve, double slots) noexcept {
  const auto [x0, y0] = curve[0];
  const auto [x1, y1] = curve[1];
  return x0 == x1 ? y0 : y0 + (y1 - y0) * (slots - x0) / (double(x1) - x0);
}
[[nodiscard]] inline double residentSlots(const TimingProfile &profile, double residentShare) noexcept {
  return residentShare * profile.experts.experts * profile.layers;
}
[[nodiscard]] inline double prefillHit(const TimingProfile &profile, double residentShare) noexcept {
  return std::clamp(curveAt(profile.curve.prefillHit, residentSlots(profile, residentShare)), 0.0, 1.0);
}
// Whether the target streams experts: it routes them and a prefill finds
// some of them read on demand at its measured residency.
[[nodiscard]] inline bool streamsExperts(const TimingProfile &profile) noexcept {
  return profile.experts.experts && profile.weightBytesPerMillisecond &&
         prefillHit(profile, profile.measuredResidentShare) < 1.0;
}
[[nodiscard]] inline double decodeBytesPerToken(const TimingProfile &profile, double residentShare) noexcept {
  return std::max(0.0, curveAt(profile.curve.decodeBytesPerToken, residentSlots(profile, residentShare)));
}
// The distinct experts one layer of a command of `rows` rows routes: the
// measured routing, never more than the rows route or the layer has.
[[nodiscard]] inline double distinctExperts(const TimingProfile &profile, uint32_t rows) noexcept {
  const auto &routing = profile.experts.routing;
  if (!profile.experts.experts || !rows)
    return 0.0;
  double experts = routing.back().second;
  if (rows <= routing.front().first) {
    experts = routing.front().second;
  } else {
    for (size_t point = 1; point < routing.size(); ++point) {
      const auto [r0, e0] = routing[point - 1];
      const auto [r1, e1] = routing[point];
      if (rows <= r1) {
        experts = e0 + (e1 - e0) * std::log(double(rows) / r0) / std::log(double(r1) / r0);
        break;
      }
    }
  }
  return std::min({experts, double(rows) * profile.experts.topK, double(profile.experts.experts)});
}

// What the profile's time of a prefill command or pass counts: the work the
// model reports (Model::prefillWork), and the rows, the chunks they fill in
// order, the layers run and the block-index pairs over them, which the
// engine does not read.
struct TimedCommand final {
  model::CommandWork work;
  uint32_t rows = 0, chunks = 0, layers = 0;
  uint64_t indexPairs = 0;
};

// What a prefill command or pass takes on the profile, its layers routing
// `distinctPerLayer` experts each: compute (its layers' share of the whole
// command's, the pairs it counts and its experts), the reads compute does
// not hide, at least its layers' share of the host floor, and its
// turnaround.
[[nodiscard]] inline CommandTimes prefillTimes(const TimingProfile &profile, const TimedCommand &command,
                                               double distinctPerLayer) noexcept {
  const model::CommandWork &work = command.work;
  const double layers = double(command.layers) / profile.layers;
  const double compute =
      layers * (command.chunks * profile.chunkMilliseconds + command.rows * profile.rowMilliseconds) +
      double(work.attentionPairs) * profile.megapairMilliseconds / 1e6 +
      double(command.indexPairs) * profile.indexMegapairMilliseconds / 1e6 +
      command.layers * distinctPerLayer * profile.expertMilliseconds;
  const double weights =
      profile.weightBytesPerMillisecond ? double(work.weightReadBytes) / profile.weightBytesPerMillisecond : 0.0;
  const double stream = profile.spilledKvBytesPerMillisecond
                            ? double(work.spilledKvReadBytes) / profile.spilledKvBytesPerMillisecond
                            : 0.0;
  const double reads = weights + stream;
  const double host =
      std::max(layers * profile.hostFloorMilliseconds, reads - profile.hiddenShare * std::min(reads, compute));
  return {.wallMilliseconds = profile.turnaroundMilliseconds + compute + host,
          .hostWaitMilliseconds = host,
          .historyWaitMilliseconds = stream,
          .sample = {.work = work}};
}
// A command of the profile's measured routing.
[[nodiscard]] inline CommandTimes prefillTimes(const TimingProfile &profile, const TimedCommand &command) noexcept {
  return prefillTimes(profile, command, distinctExperts(profile, command.rows));
}

// A decode command of `width` lanes verifying `rows` rows at the experts'
// residency, `spillLanes` of its lanes reading spilled K/V.
[[nodiscard]] inline CommandTimes decodeTimes(const TimingProfile &profile, uint32_t width, uint32_t rows,
                                              double residentShare, uint32_t spillLanes) noexcept {
  const double reads =
      profile.decodeWeightBytesPerMillisecond
          ? (decodeBytesPerToken(profile, residentShare) - decodeBytesPerToken(profile, profile.measuredResidentShare)) *
                rows / profile.decodeWeightBytesPerMillisecond
          : 0.0;
  return {.wallMilliseconds = profile.decodeMilliseconds.at(width - 1) + reads +
                              spillLanes * profile.spillDecodeMilliseconds};
}

// The fake model's work (Model::prefillWork) on its profile: the pass's
// layers, its attending layers' pairs at their reach and block-index pairs,
// the pairs of the target's heaviest attention layer, the experts its rows
// route that are not resident (once per command or pass, the measured
// routing), and a streaming lane's Index pages' history over its attending
// layers. The residency is the experts pool's now, or the profile's measured
// one without a pool.
class TimedWork final : public model::PrefillWorkSource {
public:
  explicit TimedWork(const TimingProfile &profile, std::function<double()> residentShare = {})
      : profile_(profile), residentShare_(std::move(residentShare)) {}

  [[nodiscard]] model::CommandWork prefillWork(std::span<const model::PrefillLane> lanes, uint32_t first,
                                               uint32_t passesLeft) const override {
    return command(lanes, first, passesLeft).work;
  }
  // What the profile times of the pass (TimedCommand).
  [[nodiscard]] TimedCommand command(std::span<const model::PrefillLane> lanes, uint32_t first,
                                     uint32_t passesLeft) const {
    uint32_t rows = 0;
    for (const model::PrefillLane &lane : lanes)
      rows += lane.rows;
    return timed(lanes, first, passesLeft, distinctExperts(profile_, rows));
  }
  // The time the pass takes (prefillTimes).
  [[nodiscard]] CommandTimes times(std::span<const model::PrefillLane> lanes, uint32_t first,
                                   uint32_t passesLeft) const {
    return prefillTimes(profile_, command(lanes, first, passesLeft));
  }
  // A prompt of one repeated token from position 0: it routes the
  // warm-up's experts.
  [[nodiscard]] TimedCommand warmupCommand(uint32_t rows) const {
    const model::PrefillLane lane{.rows = rows};
    return timed({&lane, 1}, 0, 1, profile_.warmupDistinctExperts);
  }
  [[nodiscard]] CommandTimes warmupTimes(uint32_t rows) const {
    return prefillTimes(profile_, warmupCommand(rows), profile_.warmupDistinctExperts);
  }
  [[nodiscard]] double residentShare() const {
    return residentShare_ ? residentShare_() : profile_.measuredResidentShare;
  }
  [[nodiscard]] const TimingProfile &profile() const noexcept { return profile_; }

private:
  [[nodiscard]] TimedCommand timed(std::span<const model::PrefillLane> lanes, uint32_t first, uint32_t passesLeft,
                                   double distinctPerLayer) const {
    const model::LayerRange range = model::passLayers(first, passesLeft, profile_.layers);
    // The attending and full-attention layers the pass runs, and the
    // target's.
    uint32_t attending = 0, full = 0, allFull = 0;
    for (uint32_t layer = 0; layer < profile_.layers; ++layer) {
      if ((layer + 1) % profile_.attentionEvery)
        continue;
      const bool dense = profile_.fullAttentionEvery && layer % profile_.fullAttentionEvery == 0;
      allFull += dense;
      if (layer >= range.first && layer < range.end) {
        ++attending;
        full += dense;
      }
    }
    const uint32_t allAttending = profile_.layers / profile_.attentionEvery;
    TimedCommand result{.layers = range.end - range.first};
    model::CommandWork &work = result.work;
    uint64_t reachPairs = 0, densePairs = 0, indexPairs = 0, indexPages = 0;
    const uint64_t block = profile_.reach.blockTokens;
    // Complete blocks before each row of [0, end), summed.
    const auto blocks = [block](uint64_t end) {
      const uint64_t whole = end / block;
      return block * whole * (whole - 1) / 2 + whole * (end % block);
    };
    for (const model::PrefillLane &lane : lanes) {
      result.rows += lane.rows;
      reachPairs += model::attentionPairs(profile_.reach, lane.position, lane.rows);
      densePairs += model::denseAttentionPairs(lane.position, lane.rows);
      if (block && lane.rows)
        indexPairs += blocks(uint64_t{lane.position} + lane.rows) - blocks(lane.position);
      indexPages += lane.indexPages;
    }
    result.chunks = (result.rows + model::ExecutionLimits::prefillTokenBudget - 1) /
                    model::ExecutionLimits::prefillTokenBudget;
    work.attentionPairs = (attending - full) * reachPairs + full * densePairs;
    // The target's heaviest attention layer whatever layers the pass runs: a
    // pass does not split a layer (CommandWork::layerAttentionPairs).
    work.layerAttentionPairs = allFull ? densePairs : allAttending ? reachPairs : 0;
    result.indexPairs = (attending - full) * indexPairs;
    work.weightReadBytes = static_cast<uint64_t>(result.layers * distinctPerLayer *
                                                 (1.0 - prefillHit(profile_, residentShare())) *
                                                 double(profile_.experts.expertBytes));
    work.spilledKvReadBytes = allAttending ? indexPages * profile_.spilledPageBytes * attending / allAttending : 0;
    return result;
  }

  const TimingProfile &profile_;
  std::function<double()> residentShare_;
};

// Fitted profiles (TimingProfile.cpp records the fits and their sources).
extern const TimingProfile kM6FlashNextQ2, kStudio18GFlashNextQ2, kStudio36FlashNextQ2, kStoreLessSpillFlashNext,
    kDense27B, kFlint, k35BStudio18G;
extern const std::array<MeasuredCommand, 11> kM6FlashNextQ2Commands;
extern const std::array<MeasuredCommand, 4> kStudio18GFlashNextQ2Commands;
extern const std::array<MeasuredPrompt, 3> kDense27BPrompts;
extern const std::array<MeasuredPrompt, 5> kFlintPrompts;

} // namespace splash::test
