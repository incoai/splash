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
double gpuAt(uint32_t rows) { return 5.0 + (rows - 512) * 15.0 / 1536; }

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

// A calibration as a line of text comes back as it was, and one of no split as none; a line of another tag or
// version, with more or fewer fields, units without least rows or the GPU alone out of order or unusable is not one.
void testCalibrationText() {
  const Calibration made{8, 655, {{{512, 5.25}, {2048, 19.375}}}};
  const std::optional<Calibration> back = Calibration::parse(made.text());
  require(back && back->aneUnits == 8 && back->minimumRows == 655 && back->gpu[0].rows == 512 &&
              back->gpu[0].milliseconds == 5.25 && back->gpu[1].rows == 2048 && back->gpu[1].milliseconds == 19.375,
          "a calibration did not come back as it was: " + made.text());
  const std::optional<Calibration> none = Calibration::parse(Calibration{}.text());
  require(none && !none->aneUnits && !none->minimumRows, "a calibration of no split did not come back");
  for (const std::string_view line : {"", "ane-ffn-calibration-0 8 655 512 5.25 2048 19.375",
                                      "ane-ffn-calibration-1 8 655 512 5.25 2048",
                                      "ane-ffn-calibration-1 8 655 512 5.25 2048 19.375 1",
                                      "ane-ffn-calibration-1 8 0 512 5.25 2048 19.375",
                                      "ane-ffn-calibration-1 0 655 0 0 0 0",
                                      "ane-ffn-calibration-1 8 655 2048 19.375 512 5.25",
                                      "ane-ffn-calibration-1 8 655 512 0 2048 19.375",
                                      "ane-ffn-calibration-1 8 655 512 nan 2048 19.375"})
    require(!Calibration::parse(line), "\"" + std::string(line) + "\" was taken for a calibration");
}

// The breaker against the GPU alone on the line through 8 ms at 512 rows and 20 at 2048, 4 ms + rows / 128: a window
// of 8 commands whose evaluations take as long as the GPU alone's layers trips it on its 8th command, one a little
// faster never does, nor one that never was given the GPU alone. An evaluation of fewer rows counts against the GPU
// alone's layer of its rows, not against its rows' part of the most rows', and each window is judged on its own.
void testBreaker() {
  ChunkTimings timings;
  timings.gpu = {{{512, 8.0}, {2048, 20.0}}};
  // `count` commands of 64 evaluations of the function of `rows` rows, each `ratio` of the GPU alone's layer of
  // those rows: the first reason the breaker gives, and after how many commands.
  const auto run = [](Breaker &breaker, uint32_t count, uint32_t rows, double ratio) {
    for (uint32_t command = 1; command <= count; ++command) {
      std::string reason = breaker.add(rows, 64, 64 * ratio * (4.0 + rows / 128.0));
      if (!reason.empty()) return std::pair{command, reason};
    }
    return std::pair{0u, std::string()};
  };
  Breaker breaker(timings.gpu);
  require(run(breaker, 8, 2048, 1.0) ==
              std::pair{8u, std::string("losing to the GPU alone (20.0 ms on the Neural Engine per 2048-row layer "
                                        "against 20.0 ms on the GPU alone)")},
          "evaluations as long as the GPU alone's layers did not trip the breaker on the 8th command");
  Breaker faster(timings.gpu);
  require(run(faster, 80, 2048, 0.99).first == 0, "evaluations faster than the GPU alone's layers tripped it");
  Breaker off;
  require(run(off, 80, 2048, 100.0).first == 0, "a breaker given no GPU alone tripped");
  // 1024 rows: the GPU alone takes 12 ms, more than half its 20 ms at 2048 rows.
  Breaker fewer(timings.gpu);
  require(run(fewer, 80, 1024, 0.99).first == 0,
          "evaluations of 1024 rows faster than the GPU alone's layer of 1024 rows tripped it");
  require(run(fewer, 8, 1024, 1.25) ==
              std::pair{8u, std::string("losing to the GPU alone (25.0 ms on the Neural Engine per 2048-row layer "
                                        "against 20.0 ms on the GPU alone)")},
          "evaluations of 1024 rows slower than the GPU alone's layer of 1024 rows did not trip it");
  Breaker windows(timings.gpu);
  require(run(windows, 8, 2048, 0.5).first == 0 && run(windows, 8, 2048, 1.5).first == 8,
          "a window was not judged on its own commands");
  ChunkTimings equal = timings, reversed = timings, unusable = timings;
  equal.gpu[0].rows = 2048;
  std::ranges::reverse(reversed.gpu);
  unusable.gpu[1].milliseconds = 0.0;
  rejects([&] { static_cast<void>(Breaker(equal.gpu)); }, "two row counts",
          "a breaker took the GPU alone at one row count");
  rejects([&] { static_cast<void>(Breaker(reversed.gpu)); }, "two row counts", "a breaker took descending rows");
  rejects([&] { static_cast<void>(Breaker(unusable.gpu)); }, "not usable", "a breaker took a GPU alone of 0 ms");
}

} // namespace

int main() {
  try {
    testFit();
    testBandwidth();
    testChoice();
    testUnusable();
    testMostUnits();
    testNoGain();
    testUsable();
    testNearestUnits();
    testMinimumRows();
    testCalibrationText();
    testBreaker();
    std::cout << "ane ffn calibration tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ane ffn calibration tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
