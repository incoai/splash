#include "TestMetalMemory.hpp"
#include "TestRegistry.hpp"

#include <span>

using namespace splash;

std::span<const test::TestCase> requestsTests();
std::span<const test::TestCase> checkpointsTests();
std::span<const test::TestCase> commandsTests();
std::span<const test::TestCase> memoryTests();
std::span<const test::TestCase> spillTests();
std::span<const test::TestCase> diskTests();
std::span<const test::TestCase> rulesTests();
std::span<const test::TestCase> releaseTests();

int main(int argc, char **argv) {
  const test::TestGroup groups[] = {
      {"requests", requestsTests()}, {"checkpoints", checkpointsTests()}, {"commands", commandsTests()},
      {"memory", memoryTests()},     {"spill", spillTests()},             {"disk", diskTests()},
      {"rules", rulesTests()},       {"releases", releaseTests()},
  };
  return test::runTests("engine", groups, argc, argv, [] { test::metalStatistics() = {}; });
}
