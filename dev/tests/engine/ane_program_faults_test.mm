// The faults runtime/ane/ProgramInstrumentation.hpp injects into ane::Program, on the Neural Engine: an Objective-C
// exception the framework raises becomes a std::runtime_error, at construction and at an evaluation; a method whose
// type encoding changed fails construction; and a throwing enqueue and a failed, a stalled and a delayed evaluation
// each do what they say.
#include "AneProgramFixture.hpp"
#include "AwakeClock.hpp"
#include "TestFiles.hpp"
#include "ane/ProgramInstrumentation.hpp"
#include "metal/MetalBackend.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

using namespace splash;
using ane::Program;
using ane::ProgramInstrumentation;
using test::AneSum;
using test::thrown;

namespace {

using Faults = ProgramInstrumentation::Faults;
const Program::Limits kLimits{std::chrono::seconds(120), {}};
constexpr auto kTimeout = std::chrono::milliseconds(10000);
// How long an evaluation that should not report is given to report anyway.
constexpr auto kQuiet = std::chrono::milliseconds(500);

std::mt19937 rng(20261005);
int failures = 0;

void check(bool value, const std::string &what) {
  if (value) return;
  std::cout << "  FAIL " << what << '\n';
  ++failures;
}
void contains(const std::string &message, std::string_view expected, const std::string &what) {
  check(message.find(expected) != std::string::npos, what + ": " + message);
}

// Exceptions and changed encodings at construction, which Programs constructed
// after it no longer see.
void construction(AneSum &sum, const std::filesystem::path &cache) {
  ProgramInstrumentation::arm({.raiseAt = "model attributes"});
  contains(thrown<std::runtime_error>([&] { Program program(sum.mil("raised"), {}, kLimits, cache); }),
           "ANE model attributes raised SplashInjectedFault", "an exception raised reading the model's attributes");
  ProgramInstrumentation::arm({.changedMethod = "-[_ANEClient loadModel:options:qos:error:]"});
  contains(thrown<std::runtime_error>([&] { Program program(sum.mil("changed"), {}, kLimits, cache); }),
           "AppleNeuralEngine changed -[_ANEClient loadModel:options:qos:error:]", "a changed type encoding");
  ProgramInstrumentation::arm({.construction = true});
  contains(thrown<std::runtime_error>([&] { Program program(sum.mil("failed"), {}, kLimits, cache); }),
           "an injected fault", "a failed construction");
  Program program(sum.mil("clean"), {}, kLimits, cache);
  sum.fill(rng);
  check(sum.evaluate(program, sum.bind(program, "clean"), kTimeout) == true && sum.summed(),
        "a program constructed after the faults evaluates");
  std::cout << "construction: an exception raised and a changed encoding throw std::runtime_error, once\n";
}

// Faults of enqueue() and of the evaluations it queues.
void evaluations(AneSum &sum, const std::filesystem::path &cache) {
  ProgramInstrumentation::arm({.raiseAt = "evaluation"});
  Program raising(sum.mil("raising"), {}, kLimits, cache);
  const auto called = std::make_shared<std::atomic<bool>>(false);
  contains(thrown<std::runtime_error>(
               [&] { sum.enqueue(raising, sum.bind(raising, "raising"), [called](bool) { *called = true; }); }),
           "ANE evaluation raised SplashInjectedFault", "an exception raised queuing an evaluation");
  std::this_thread::sleep_for(kQuiet);
  check(!*called, "done was called for an enqueue() that threw");

  ProgramInstrumentation::arm(
      {.throwingEnqueue = 2, .failingEvaluation = 3, .stalledEvaluation = 4, .delayedEvaluation = 5,
       .delay = std::chrono::milliseconds(300)});
  Program program(sum.mil("faulty"), {}, kLimits, cache);
  const Program::Binding binding = sum.bind(program, "faulty");
  sum.fill(rng);
  check(sum.evaluate(program, binding, kTimeout) == true && sum.summed(), "the first evaluation");
  *called = false;
  contains(thrown<std::runtime_error>([&] { sum.enqueue(program, binding, [called](bool) { *called = true; }); }),
           "an injected fault", "the second enqueue()");
  check(sum.evaluate(program, binding, kTimeout) == false, "the third evaluation reports failure");
  const uint64_t value = sum.enqueue(program, binding, [called](bool) { *called = true; });
  std::this_thread::sleep_for(kQuiet);
  check(!*called && sum.reached() < value, "the fourth evaluation neither reports nor raises its event");
  sum.fill(rng);
  const auto started = AwakeClock::now();
  check(sum.evaluate(program, binding, kTimeout) == true && sum.summed(), "the fifth evaluation succeeds");
  check(millisecondsSince(started) >= 300.0, "the fifth evaluation reports 300 ms late");
  sum.fill(rng);
  check(sum.evaluate(program, binding, kTimeout) == true && sum.summed(), "the sixth evaluation");
  check(!*called, "done was called for the second enqueue(), which threw, or the fourth");
  std::cout << "evaluations: an exception raised, a throwing enqueue(), a failed, a stalled and a late evaluation\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ane-program-faults METALLIB\n";
    return 2;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    AneSum sum(backend);
    const test::TemporaryDirectory cache("splash-ane-faults");
    construction(sum, cache.path());
    evaluations(sum, cache.path());
  } catch (const std::exception &error) {
    std::cout << "FAIL " << error.what() << '\n';
    return 1;
  }
  if (failures) {
    std::cout << failures << " failures\n";
    return 1;
  }
  std::cout << "ane-program-faults: all checks passed\n";
  return 0;
}
