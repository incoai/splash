// When the prefill FFN's Neural Engine split pauses and stops for losing to the GPU alone (ane_ffn::Breaker and
// ane_ffn::Pause, ops/AneFfnCalibration.cpp), on times Macs measured: a command whose evaluations
// took as long as the GPU alone takes its layers pauses the split on its own, for five minutes each time; the commands
// the GPU runs alone meanwhile neither judge nor end it early; the next command split after it probes, and a probe
// that wins resumes the split while one that loses too stops it; and a healthy split, warm from minutes of prefill,
// never trips.

#include "TestChecks.hpp"
#include "ops/AneFfnCalibration.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using namespace splash::ops::ane_ffn;
using splash::AwakeClock;
using splash::test::require;
using std::chrono::minutes;
using std::chrono::seconds;
using Verdict = Pause::Verdict;

// An M5 Pro's calibration of Qwen3.8-27B (MLX 4-bit): the GPU alone 6.06 ms a 512-row FFN layer and 23.57 a
// 2,048-row one, and the rest of its 2,048-row layer 13.40 ms, which two steps run beside the Neural Engine.
const Breaker kM5Pro({{{512, 6.06275}, {2048, 23.5735625}}}, 13.4039438);
// An M6's (GGUF IQ3_XXS): 16.31 ms a 512-row FFN layer, 63.48 a 2,048-row one, 32.71 the rest of its layer.
const Breaker kM6({{{512, 16.3128333}, {2048, 63.4825522}}}, 32.7093027);
// The 27B's split layers a command: its last layer runs on the GPU alone.
constexpr uint32_t kLayers = 63;

// A cold 8K prompt's commands of 2,048 rows in two steps on the M5 Pro, the Neural Engine's milliseconds over them:
// as the GPU alone's 36.98 ms a layer and more, each loses on its own; less, none does. Beside another client of the
// Neural Engine at its priority, a command took it 3,341 ms, then 4,133 (1.77 of the GPU alone's 2,330); behind one
// at a higher priority, 20,060 then 29,472 (12.7). Alone on it, every command of seven minutes took 1,545-1,566 ms
// (0.66-0.67).
void testCommands() {
  for (const double milliseconds : {2330.0, 3341.1, 4133.1, 20060.3, 29472.2})
    require(!kM5Pro.judge(2048, kLayers, true, milliseconds).empty(),
            std::to_string(milliseconds) + " ms of the Neural Engine over a command did not lose on its own");
  require(kM5Pro.judge(2048, kLayers, true, 4133.1) ==
              "losing to the GPU alone (41.8 ms on the Neural Engine per 2048-row layer against 23.6 ms on the GPU "
              "alone)",
          "a losing command's reason is not the Neural Engine's time against the GPU alone's");
  for (const double milliseconds : {1545.0, 1565.8, 2329.0})
    require(kM5Pro.judge(2048, kLayers, true, milliseconds).empty(),
            std::to_string(milliseconds) + " ms of the Neural Engine over a command lost to the GPU alone");
}

// A command that loses pauses the split for five minutes: commands the GPU runs alone meanwhile change nothing, and
// the first that finishes past the pause ends it. The next command split probes: one that wins resumes the split,
// which a later loss pauses for five minutes again; one that loses stops it. Commands the GPU runs alone between the
// pause's end and the probe leave the probe to the next command split.
void testPause() {
  const AwakeClock::time_point start = AwakeClock::now();
  const auto at = [&](AwakeClock::duration since) { return start + since; };
  Pause pause;
  require(pause.finished(at(seconds(0)), true, false) == Verdict::Kept && !pause.paused(),
          "a command that won paused the split");
  require(pause.finished(at(seconds(10)), true, true) == Verdict::Paused && pause.paused() &&
              pause.until() == at(seconds(10) + minutes(5)),
          "a command that lost did not pause the split for five minutes");
  for (const AwakeClock::duration since : {minutes(1), minutes(5)})
    require(pause.finished(at(since), false, false) == Verdict::Kept && pause.paused(),
            "a command on the GPU alone ended the pause before its end");
  require(pause.finished(at(seconds(10) + minutes(5)), false, false) == Verdict::Kept && !pause.paused(),
          "the first command to finish at the pause's end did not end it");
  require(pause.finished(at(minutes(6)), true, false) == Verdict::Kept && !pause.paused(),
          "a probe that won did not resume the split");
  require(pause.finished(at(minutes(30)), true, true) == Verdict::Paused && pause.until() == at(minutes(35)),
          "a loss after a probe that won did not pause the split for five minutes again");
  require(pause.finished(at(minutes(36)), false, false) == Verdict::Kept && !pause.paused() &&
              pause.finished(at(minutes(37)), false, false) == Verdict::Kept,
          "commands on the GPU alone after the pause's end changed the split");
  require(pause.finished(at(minutes(38)), true, true) == Verdict::Stopped, "a probe that lost did not stop the split");

  // A server that runs for days and meets a busy Neural Engine now and then: every pause after a probe that won is
  // five minutes, the hundredth as the first.
  Pause days;
  AwakeClock::duration since = seconds(0);
  for (int loss = 0; loss < 100; ++loss, since += std::chrono::hours(1))
    require(days.finished(at(since), true, true) == Verdict::Paused && days.until() == at(since + minutes(5)) &&
                days.finished(at(since + minutes(5)), false, false) == Verdict::Kept &&
                days.finished(at(since + minutes(6)), true, false) == Verdict::Kept,
            "a pause after " + std::to_string(loss) + " others was not five minutes");
}

// A healthy split warm from minutes of back-to-back prefill never trips: the slowest command each Mac's split ran
// through them, and each also 5.3% slower, the most an M6's split commands slowed in sustained prefill. The M5 Pro (a
// laptop, nominal to fair), two steps: 1,566 ms of 162 commands over seven minutes of cold 8K prompts (0.67 of the GPU
// alone's layers), 1,841 in a half minute of another run that took 0.67-0.69 before and after (0.79); one step, hot
// after three minutes of 8K prompts, at its least rows (chunks of 552-640 rows on the 640-row function): 388 ms
// (0.82); one step's Neural Engine part fitted to a sustained run's hottest minute, 21.2 ms a 2,048-row layer (0.90).
// The M6, two steps: 2,499 ms of 162 commands over ten minutes (0.41).
void testWarm() {
  struct Command {
    const char *mac;
    const Breaker &breaker;
    uint32_t rows;
    bool pipelined;
    double milliseconds;
  };
  const Command slowest[] = {{"the M5 Pro's two steps", kM5Pro, 2048, true, 1565.8},
                             {"the M5 Pro's two steps, a slow half minute", kM5Pro, 2048, true, 1841.1},
                             {"the M5 Pro's one step at its least rows", kM5Pro, 640, false, 387.6},
                             {"the M5 Pro's hottest one step", kM5Pro, 2048, false, kLayers * 21.2},
                             {"the M6's two steps", kM6, 2048, true, 2499.0}};
  Pause pause;
  AwakeClock::time_point now = AwakeClock::now();
  for (const Command &command : slowest)
    for (const double slower : {1.0, 1.053}) {
      const std::string losing =
          command.breaker.judge(command.rows, kLayers, command.pipelined, slower * command.milliseconds);
      now += seconds(3);
      require(losing.empty() && pause.finished(now, true, false) == Verdict::Kept,
              std::string(command.mac) + ", warm, tripped the breaker: " + losing);
    }
}

} // namespace

int main() {
  try {
    testCommands();
    testPause();
    testWarm();
    std::cout << "ane ffn breaker tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ane ffn breaker tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
