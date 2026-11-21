// How the prefill FFN's Neural Engine split is calibrated (ops/AneFfnCalibration.cpp), from timings it is given: the
// model's fit, the share it chooses, the timings it takes, the least chunk it splits, the calibration later starts
// take again and when a split that serves stops for losing to the GPU alone.

#include "TestChecks.hpp"
#include "ops/AneFfnCalibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace splash::ops::ane_ffn;
using splash::test::rejects;
using splash::test::require;

bool near(double got, double expected) { return std::abs(got - expected) <= 1e-9 * std::max(1.0, std::abs(expected)); }

// The timings of Qwen3.8-27B's 34 units, at 14 and 27 of them, of a Mac whose GPU alone takes `gpuAlone`.
struct Measured {
  const char *mac;
  Timing low, high;
  double gpuAlone;
  uint32_t chosen;
};
// As each Mac timed them (MLX on the M5 Max and the M5 Pro, GGUF IQ3_XXS on the M6), and the units each chooses:
// within a unit of the fastest full chunk each timed over every unit count (AneFfnCalibration.cpp).
const Measured kMeasured[] = {
    {"M5 Max", {14.0 / 34, 11.699, 23.484}, {27.0 / 34, 5.033, 44.683}, 18.990, 8},
    {"M5 Pro", {14.0 / 34, 24.244, 26.277}, {27.0 / 34, 10.147, 45.679}, 38.829, 15},
    {"M6", {14.0 / 34, 40.527, 45.481}, {27.0 / 34, 17.917, 31.079}, 64.007, 26},
};

// G is the line through the GPU part's timings and A proportional to the ANE's channels through the high share's
// split layer, which the ANE's part takes; where the low share's split layer takes no longer than the GPU's part, the
// bandwidth binds nowhere and T is the longer part.
void testFit() {
  const Model model = fit({0.25, 15, 15}, {0.75, 5, 30});
  require(near(model.gpu(0.0), 20) && near(model.gpu.slope, -20) && near(model.ane(0.0), 0) &&
              near(model.ane.slope, 40) && model.u == 0.0,
          "the lines through the timings");
  require(near(model(0.25), 15) && near(model(0.5), 20) && near(model(0.3), 14),
          "T is not the longer part where the bandwidth does not bind");
  for (const auto &[low, high] : std::vector<std::pair<Timing, Timing>>{{{0.5, 10, 12}, {0.5, 8, 14}},
                                                                          {{0.75, 5, 30}, {0.25, 15, 15}}})
    rejects([&] { static_cast<void>(fit(low, high)); }, "a low and a high share", "a fit of shares out of order");
}

// Where the GPU's part takes the longer at the low share, the split layer's excess over it is the bandwidth the ANE's
// part takes from it: here 2 of A(0.25) = 10, so u = 0.2, and T(s) = G(s) + 0.2 A(s) until A takes the longer. Where
// the ANE's part takes the longer there, the excess is the ANE's own.
void testBandwidth() {
  const Model model = fit({0.25, 15, 17}, {0.75, 5, 30});
  require(near(model.u, 0.2) && near(model(0.25), 17) && near(model(0.3), 16.4) && near(model(0.5), 20),
          "the bandwidth's term was not fitted");
  require(fit({0.5, 10, 24}, {0.75, 5, 30}).u == 0.0, "the ANE's part bound the bandwidth where it took the longer");
  require(fit({0.25, 15, 14}, {0.75, 5, 30}).u == 0.0 && fit({0.25, 15, 60}, {0.75, 5, 30}).u == 1.0,
          "the ANE's fraction was not clamped to none or all of the bandwidth");
}

// A split layer timed at another share fits u again where the GPU's part takes the longer there, and T passes through
// it: here 17 at 0.25 against G 15 and A 10, u = 0.2. Where the ANE's part takes the longer there, it says nothing of u.
// On the M5 Max, whose ANE's part takes the longer at the low share by default (int8 activations), the low share
// leaves u at none, and the fit chooses 6 of Qwen3.8-27B's 34 units; the split layer of 6 units, 13.84 ms against G's
// 13.18, gives u = 0.065, which chooses the 7 a sweep of given shares found fastest (6 prefills 1.4-1.5% slower).
void testRefit() {
  const Model lines = fit({0.25, 15, 15}, {0.75, 5, 30});
  const Model refitted = refit(lines, 0.25, 17);
  require(near(refitted.u, 0.2) && near(refitted(0.25), 17) && near(refitted(0.5), 20),
          "the bandwidth's term was not fitted at the share timed");
  require(refit(refitted, 0.5, 30).u == refitted.u && refit(lines, 0.5, 30).u == 0.0,
          "a split layer where the ANE's part takes the longer fitted the bandwidth's term");
  const Timing low{14.0 / 34, 9.9673, 23.929}, high{27.0 / 34, 4.7526, 46.034};
  const Model m5Max = fit(low, high);
  require(m5Max.u == 0.0 && choose(m5Max, 15.834, 34, 33) == 6u, "the M5 Max's low share fitted u or chose not 6");
  const Model atSix = refit(m5Max, 6.0 / 34, 13.84);
  require(atSix.u > 0.06 && atSix.u < 0.07 && choose(atSix, 15.834, 34, 33) == 7u,
          "the M5 Max's split layer at 6 units did not choose 7");
}

// T is least at the units where the GPU's part, slowed by the bandwidth, meets the ANE's, and the units chosen are
// those whose T, or T a unit either way, is least at its longest: G(s) = 20 - 20 s and A(s) = 40 s meet at s = 1/3,
// where of 10 units 3 (G 14, A 12; 16 a unit either way) beat 2 (G 16; 18 below) and 4 (A 16; 20 above), and of 100
// units 33 (13.4; 13.6 either way) beat 32 (13.8 below) and 34 (14.0 above). Each Mac's timings choose their units,
// the best its sweeps measured.
void testChoice() {
  const Model model = fit({0.25, 15, 15}, {0.75, 5, 30});
  require(choose(model, 20, 10, 9) == 3u, "the crossing's units were not chosen");
  require(choose(model, 20, 100, 99) == 33u, "the units nearest the crossing were not chosen");
  for (const uint32_t most : {0u, 10u})
    rejects([&] { static_cast<void>(choose(model, 20, 10, most)); }, "no feasible units",
            "a choice of " + std::to_string(most) + " units of 10");
  for (const Measured &measured : kMeasured)
    require(choose(fit(measured.low, measured.high), measured.gpuAlone, 34, 33) == measured.chosen,
            std::string("the ") + measured.mac + "'s timings did not choose " + std::to_string(measured.chosen) +
                " units");
  // Where a unit past the crossing costs more than a unit short of it, the choice keeps below T's least: G(s) =
  // 16 - 4 s and A(s) = 48 s put it at 10 of 34 units (14.82, but 15.53 a unit above), and 9 (14.94; 15.06 a unit
  // below) is chosen.
  const Model steep = fit({0.25, 15, 15}, {0.75, 13, 36});
  require(choose(steep, 20, 34, 33) == 9u, "the choice did not keep below T's least where a unit past it costs more");
  // Its bandwidth's term moves the M6's choice: without it the M6 would choose 23 units, not 26.
  Model m6 = fit(kMeasured[2].low, kMeasured[2].high);
  require(m6.u > 0.25 && m6.u < 0.35, "the M6's bandwidth term was not fitted");
  m6.u = 0.0;
  require(choose(m6, kMeasured[2].gpuAlone, 34, 33) == 23u,
          "the M6 without its bandwidth term did not choose 23 units");
}

// Timings that put T at zero or below, or that are not numbers, choose nothing.
void testUnusable() {
  require(!choose(fit({0.5, 1, 1}, {0.6, -0.1, 0.0}), 20, 34, 33), "T of zero chose units");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  require(!choose(fit({0.3, nan, 12}, {0.8, 4, 32}), 20, 10, 9), "lines of NaN chose units");
}

// The choice of units the plan holds: the most it holds where T falls beyond them, 30 of 100 (14.0, 14.2 a unit
// below).
void testMostUnits() {
  const Model model = fit({0.25, 15, 15}, {0.75, 5, 30});
  require(choose(model, 20, 100, 30) == 30u, "units the plan does not hold were chosen");
}

// Of two steps, the ANE's halves H(s) = 44 s run beside the mixer, 10 ms, and T(s) = max(H(s), 10 + G(s)): they meet
// at s = 0.47, and of 10 units 4 (22; 24 a unit below) beat 5 (22; 26.4 above) and 3 (24), where one step chose 3. A
// mixer below none or not a number is refused.
void testTwoSteps() {
  Timings timings{{0.25, 15, 15}, {0.75, 5, 30}, 20, 16.5, 10};
  const Model one = fit(timings.low, timings.high), two = twoSteps(one, timings);
  require(near(two.halves.slope, 44) && two.mixer == 10 && near(two(0.4), 22) && near(two(0.5), 22) &&
              near(two(0.3), 24),
          "two steps' T is not the longer of the ANE's halves and the mixer with the GPU's part");
  require(choose(two, 30, 10, 9) == 4u && choose(one, 20, 10, 9) == 3u,
          "the mixer the ANE runs beside did not move the choice from 3 units to 4");
  // A mixer of none, as a prefill no longer than the GPU alone's FFN layers gives, still prices two steps by their
  // halves: 22 at 0.5, where one step takes 20.
  timings.mixer = 0;
  const Model none = twoSteps(one, timings);
  require(near(none(0.5), 22) && near(one(0.5), 20), "two steps beside a mixer of none were priced as one step");
  for (const double mixer : {-1.0, std::numeric_limits<double>::quiet_NaN()}) {
    timings.mixer = mixer;
    rejects([&] { static_cast<void>(twoSteps(one, timings)); }, "mixer is not usable",
            "a mixer of " + std::to_string(mixer) + " ms was taken");
  }
}

// The split runs only where T is predicted at least 5% below the GPU alone: 14 against 14.5 is not, and the GPU's
// timing not a number proves nothing.
void testNoGain() {
  const Model model = fit({0.25, 15, 15}, {0.75, 5, 30});
  require(choose(model, 14.8, 10, 9) == 3u, "a 5.4% gain was refused");
  require(!choose(model, 14.5, 10, 9), "a 3.4% gain was taken");
  require(!choose(model, std::numeric_limits<double>::quiet_NaN(), 10, 9),
          "a gain over a timing not a number was taken");
}

// A measured duration is usable only finite and positive.
void testUsable() {
  require(usable(1.5) == 1.5, "a duration was changed");
  for (const double milliseconds : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::infinity()})
    rejects([&] { static_cast<void>(usable(milliseconds)); }, "calibration timings are not usable",
            std::to_string(milliseconds) + " ms was usable");
}

// The units nearest a share, from one to all but one, and no share outside (0, 1) or of fewer than two units.
void testNearestUnits() {
  require(nearestUnits(0.4, 34) == 14 && nearestUnits(0.8, 34) == 27 && nearestUnits(0.01, 34) == 1 &&
              nearestUnits(0.99, 34) == 33,
          "the units nearest a share");
  for (const auto &[share, units] : std::vector<std::pair<double, uint32_t>>{
           {0.0, 34}, {1.0, 34}, {std::numeric_limits<double>::quiet_NaN(), 34}, {0.5, 1}})
    rejects([&] { static_cast<void>(nearestUnits(share, units)); }, "must lie in (0, 1)",
            "a share of " + std::to_string(share) + " of " + std::to_string(units) + " units");
}

// The GPU alone's layer on the line through 5 ms at 512 rows and 20 at 2048.
double gpuAt(uint32_t rows) { return 5.0 + (static_cast<double>(rows) - 512) * 15.0 / 1536; }

// The timings of functions every 128 rows from 512 to 2048, from the most rows down, each running the chunks above
// the one below it (the least, those of 512 rows): each function's split layer at `ratio` of the GPU's at its own
// rows, but where `split` says otherwise.
ChunkTimings functions(double ratio, const std::vector<std::pair<uint32_t, double>> &split = {}) {
  ChunkTimings timings;
  timings.gpu = {{{512, gpuAt(512)}, {2048, gpuAt(2048)}}};
  for (uint32_t rows = 2048; rows >= 512; rows -= 128) {
    double milliseconds = ratio * gpuAt(rows);
    for (const auto &[function, given] : split)
      if (function == rows) milliseconds = given;
    timings.functions.push_back({rows == 512 ? 512 : rows - 127, rows, milliseconds});
  }
  return timings;
}

// A chunk splits while its function's split layer takes at most 2% longer than the GPU alone at the chunk's rows,
// each row count checked; the least chunk is the one from which every larger one splits.
void testMinimumRows() {
  require(minimumRows(functions(0.5)) == 512u, "a split of every function twice as fast held back a chunk");
  require(!minimumRows(functions(0.8, {{2048, 21.0}})), "a split slower at the most rows split a chunk");
  // Function 640 at 5.5 ms: a batch of a 512-row chunk and another lane's row, 513 rows, runs it slower than the GPU
  // alone (5.01 ms); its chunks split from 553 rows (1.02 x 5.40 ms), and not from 552 (1.02 x 5.39).
  require(minimumRows(functions(0.8, {{640, 5.5}})) == 553u, "the least chunk of a function was not found by its rows");
  // A function slower than the GPU at each of its chunks holds back every chunk below it, though they split.
  for (const uint32_t slow : {768u, 896u, 1024u})
    require(minimumRows(functions(0.8, {{slow, 1.2 * gpuAt(slow)}})) == slow + 1,
            "a function of " + std::to_string(slow) +
                " rows that does not split did not hold back the chunks below it");
  // Within 2%: function 2048 as slow as the GPU at 2048 rows splits chunks of 2008 rows, whose GPU layer takes 1.95%
  // less, and not of 2007; 2.5% slower than the GPU it splits none.
  require(minimumRows(functions(0.8, {{2048, 20.0}})) == 2008u && !minimumRows(functions(0.8, {{2048, 20.5}})),
          "the margin is not 2%");
  ChunkTimings ascending = functions(0.5), gapped = functions(0.5), empty = functions(0.5), reversed = functions(0.5);
  std::ranges::reverse(ascending.functions);
  gapped.functions.erase(gapped.functions.begin() + 3);
  empty.functions.clear();
  std::ranges::reverse(reversed.gpu);
  for (const auto &[timings, what] : std::vector<std::pair<ChunkTimings, std::string>>{
           {ascending, "ascending functions"}, {gapped, "functions with a gap"}, {empty, "no functions"},
           {reversed, "GPU timings out of order"}})
    rejects([&] { static_cast<void>(minimumRows(timings)); }, "not those of its functions", what + " were taken");
}

// A split takes no chunk longer than another where it splits every chunk the other splits, each in at most 2% longer:
// one faster at the most rows and alike elsewhere, one 1.25% slower at every function, and one whose 640-row function
// takes 5.3 ms against 5.5, splitting from 533 rows against 553, do; one 3% slower at 1024 rows does not, nor one whose
// 640-row function takes 6.0 ms, which splits from 603 rows and leaves chunks of 553 to 602 rows to the GPU alone, nor
// one that splits none.
void testSlowerNowhere() {
  const ChunkTimings incumbent = functions(0.8, {{640, 5.5}});
  require(slowerNowhere(incumbent, incumbent) && slowerNowhere(functions(0.8, {{640, 5.5}, {2048, 15.0}}), incumbent) &&
              slowerNowhere(functions(0.81, {{640, 5.5}}), incumbent) &&
              slowerNowhere(functions(0.8, {{640, 5.3}}), incumbent),
          "a split no chunk of which took longer was refused");
  require(!slowerNowhere(functions(0.8, {{640, 5.5}, {1024, 0.824 * gpuAt(1024)}}), incumbent) &&
              !slowerNowhere(functions(0.8, {{640, 6.0}, {2048, 15.0}}), incumbent) &&
              !slowerNowhere(functions(0.8, {{2048, 21.0}}), incumbent),
          "a split that took chunks longer was taken");
}

// A calibration as a line of text comes back as it was, its refusals too, and one of no split as none with its
// refusals, one or kRefusals; one of one step alone too, and of two steps alone; a line of another tag or version,
// with more or fewer fields, units without least rows or two steps, least rows or two steps without units of one
// step, the GPU alone out of order or unusable, a mixer below none, no split and no refusals, more refusals than
// kRefusals or a split beside kRefusals is not one.
void testCalibrationText() {
  const Calibration made{8, 655, {{{512, 5.25}, {2048, 19.375}}}, 1, 15, 13.5};
  const std::optional<Calibration> back = Calibration::parse(made.text());
  require(back && back->aneUnits == 8 && back->minimumRows == 655 && back->gpu[0].rows == 512 &&
              back->gpu[0].milliseconds == 5.25 && back->gpu[1].rows == 2048 &&
              back->gpu[1].milliseconds == 19.375 && back->refusals == 1 && back->twoStepUnits == 15 &&
              back->mixer == 13.5,
          "a calibration did not come back as it was: " + made.text());
  const std::optional<Calibration> oneOnly =
      Calibration::parse(Calibration{8, 655, {{{512, 5.25}, {2048, 19.375}}}, 0, 0, 13.5}.text());
  require(oneOnly && oneOnly->aneUnits == 8 && !oneOnly->twoStepUnits, "a calibration of one step alone did not come back");
  const std::optional<Calibration> twoOnly =
      Calibration::parse(Calibration{8, 0, {{{512, 5.25}, {2048, 19.375}}}, 0, 15, 13.5}.text());
  require(twoOnly && twoOnly->aneUnits == 8 && !twoOnly->minimumRows && twoOnly->twoStepUnits == 15 &&
              twoOnly->mixer == 13.5,
          "a calibration of two steps alone did not come back");
  // A record of no split holds only the Neural Engine's refusals: no gain is not remembered.
  const std::optional<Calibration> once = Calibration::parse(Calibration{.refusals = 1}.text()),
                                   refused = Calibration::parse(Calibration{.refusals = 2}.text());
  require(once && !once->aneUnits && !once->minimumRows && once->refusals == 1 && refused && !refused->aneUnits &&
              refused->refusals == 2 && !Calibration::parse(Calibration{}.text()),
          "a record of no split came back other than of its refusals");
  for (const std::string_view line : {"", "ane-ffn-calibration-3 8 655 512 5.25 2048 19.375 0 15 806 13.5",
                                      "ane-ffn-calibration-4 8 655 512 5.25 2048 19.375 0 0",
                                      "ane-ffn-calibration-4 8 655 512 5.25 2048 19.375 0 0 0 1",
                                      "ane-ffn-calibration-4 8 0 512 5.25 2048 19.375 0 0 0",
                                      "ane-ffn-calibration-4 0 0 512 5.25 2048 19.375 0 15 0",
                                      "ane-ffn-calibration-4 0 655 0 0 0 0 0 0 0",
                                      "ane-ffn-calibration-4 8 655 2048 19.375 512 5.25 0 0 0",
                                      "ane-ffn-calibration-4 8 655 512 0 2048 19.375 0 0 0",
                                      "ane-ffn-calibration-4 8 655 512 nan 2048 19.375 0 0 0",
                                      "ane-ffn-calibration-4 8 655 512 5.25 2048 19.375 0 0 -1",
                                      "ane-ffn-calibration-4 0 0 0 0 0 0 3 0 0",
                                      "ane-ffn-calibration-4 8 655 512 5.25 2048 19.375 2 0 0"})
    require(!Calibration::parse(line), "\"" + std::string(line) + "\" was taken for a calibration");
}

// The breaker against the GPU alone on the line through 8 ms at 512 rows and 20 at 2048, 4 ms + rows / 128: a command
// whose evaluations take as long as the GPU alone's layers loses on its own, one a little faster never does, nor one
// a breaker never given the GPU alone judges. An evaluation of fewer rows counts against the GPU alone's layer of its
// rows, not against its rows' part of the most rows', and each command is judged on its own.
void testBreaker() {
  ChunkTimings timings;
  timings.gpu = {{{512, 8.0}, {2048, 20.0}}};
  // `count` commands of 64 evaluations of the function of `rows` rows, each `ratio` of the GPU alone's layer of
  // those rows: the first reason the breaker gives, and at which command.
  const auto run = [](const Breaker &breaker, uint32_t count, uint32_t rows, double ratio, bool pipelined = false) {
    for (uint32_t command = 1; command <= count; ++command) {
      std::string reason = breaker.judge(rows, 64, pipelined, 64 * ratio * (4.0 + rows / 128.0));
      if (!reason.empty()) return std::pair{command, reason};
    }
    return std::pair{0u, std::string()};
  };
  const Breaker breaker(timings.gpu, 10.0);
  require(run(breaker, 8, 2048, 1.0) ==
              std::pair{1u, std::string("losing to the GPU alone (20.0 ms on the Neural Engine per 2048-row layer "
                                        "against 20.0 ms on the GPU alone)")},
          "a command whose evaluations took as long as the GPU alone's layers did not lose on its own");
  const Breaker faster(timings.gpu, 10.0);
  require(run(faster, 80, 2048, 0.99).first == 0, "evaluations faster than the GPU alone's layers tripped it");
  const Breaker off;
  require(run(off, 80, 2048, 100.0).first == 0, "a breaker given no GPU alone tripped");
  // 1024 rows: the GPU alone takes 12 ms, more than half its 20 ms at 2048 rows.
  const Breaker fewer(timings.gpu, 10.0);
  require(run(fewer, 80, 1024, 0.99).first == 0,
          "evaluations of 1024 rows faster than the GPU alone's layer of 1024 rows tripped it");
  require(run(fewer, 8, 1024, 1.25) ==
              std::pair{1u, std::string("losing to the GPU alone (25.0 ms on the Neural Engine per 2048-row layer "
                                        "against 20.0 ms on the GPU alone)")},
          "evaluations of 1024 rows slower than the GPU alone's layer of 1024 rows did not lose on their own");
  const Breaker commands(timings.gpu, 10.0);
  require(run(commands, 8, 2048, 0.5).first == 0 && run(commands, 8, 2048, 1.5).first == 1,
          "a command was not judged on its own");
  // Two steps run beside the mixer, 10 ms of a 2048-row layer: their evaluations count against the GPU alone's whole
  // layer, 30 ms; 1.2 times its FFN does not trip it, 1.6 times does.
  const Breaker pipelined(timings.gpu, 10.0);
  require(run(pipelined, 80, 2048, 1.2, true).first == 0 && run(pipelined, 8, 2048, 1.6, true).first == 1,
          "two steps' evaluations were not judged against the GPU alone's whole layers");
  rejects([&] { static_cast<void>(Breaker(timings.gpu, -1.0)); }, "mixer is not usable",
          "a breaker took a mixer below none");
  ChunkTimings equal = timings, reversed = timings, unusable = timings;
  equal.gpu[0].rows = 2048;
  std::ranges::reverse(reversed.gpu);
  unusable.gpu[1].milliseconds = 0.0;
  rejects([&] { static_cast<void>(Breaker(equal.gpu, 0.0)); }, "two row counts",
          "a breaker took the GPU alone at one row count");
  rejects([&] { static_cast<void>(Breaker(reversed.gpu, 0.0)); }, "two row counts", "a breaker took descending rows");
  rejects([&] { static_cast<void>(Breaker(unusable.gpu, 0.0)); }, "not usable", "a breaker took a GPU alone of 0 ms");
}

} // namespace

// A GPU whose FFN layer alone of a full chunk reads `first` ms until a program has loaded and 470 after, as an M4
// MacBook Air's settled while calibration compiled its programs (352.9 ms first, 470.2 in the chunks timed next), with
// that Mac's GPU parts and split layers of Qwen3.8-27B MLX 4-bit at 14 and 27 of 34 units (its second calibration's,
// whose split layers, 445.3 and 297.3 less 352.9, were 328.2 and 180.2 less 470), and a split of half the rows of 95.
Readers settling(double first) {
  struct State {
    double pace;
    uint32_t units = 0;
  };
  const auto state = std::make_shared<State>(State{first});
  return {[state](uint32_t units, bool) {
            state->pace = 470.0;
            state->units = units;
          },
          [state] { return state->units == 14 ? 318.9 : 108.6; },
          [state](uint32_t rows) { return state->pace * rows / 2048; },
          [state](uint32_t rows, double standIn) {
            const double split = rows == 2048 ? (state->units == 14 ? 328.2 : 180.2) : 95.0;
            return state->pace * rows / 2048 + split - standIn;
          }};
}

// Each share's split layer is read beside the GPU alone read right before it, so a GPU that settles while the
// programs compile times the split as one that ran at its settled pace throughout: an M4 MacBook Air's 352.9 ms read
// first had made its split layers 117 ms longer and its gate 117 ms shorter, which chose 30 units 1.2% short of no
// gain; read beside them, 328.2 and 180.2 ms against 470, whose fit puts the units it chooses under half the GPU
// alone's layer.
void testSharesAreTimedBesideTheirGpuAlone() {
  const Timings settled = splash::ops::ane_ffn::time(34, 2048, settling(352.9)),
                steady = splash::ops::ane_ffn::time(34, 2048, settling(470.0));
  const auto same = [](const Timing &a, const Timing &b) {
    return near(a.share, b.share) && near(a.gpu, b.gpu) && near(a.split, b.split) && near(a.alone, b.alone);
  };
  require(same(settled.low, steady.low) && same(settled.high, steady.high) && near(settled.gpuAlone, 470.0) &&
              near(settled.low.split, 328.2) && near(settled.high.split, 180.2) && near(settled.highHalf, 95.0) &&
              near(settled.low.alone, 470.0),
          "a GPU that settled while the programs compiled moved the split's timings");
  const Model model = fit(settled.low, settled.high);
  const std::optional<uint32_t> units = choose(model, settled.gpuAlone, 34, 33);
  require(units && model(*units / 34.0) < 0.5 * settled.gpuAlone,
          "the settled timings chose " + (units ? std::to_string(*units) : "no") + " units");
  const Timing alone = splash::ops::ane_ffn::time(20, 34, 2048, settling(352.9));
  require(near(alone.split, 180.2) && near(alone.alone, 470.0), "a split of given units was timed apart from its GPU alone");
}

int main() {
  try {
    testFit();
    testBandwidth();
    testRefit();
    testChoice();
    testUnusable();
    testMostUnits();
    testTwoSteps();
    testSharesAreTimedBesideTheirGpuAlone();
    testNoGain();
    testUsable();
    testNearestUnits();
    testMinimumRows();
    testSlowerNowhere();
    testCalibrationText();
    testBreaker();
    std::cout << "ane ffn calibration tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ane ffn calibration tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
