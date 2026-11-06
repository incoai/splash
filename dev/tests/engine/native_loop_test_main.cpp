#include "TestMetalMemory.hpp"
#include "TestRegistry.hpp"

#include <span>

using namespace splash;

std::span<const test::TestCase> wireTests();
std::span<const test::TestCase> weightsTests();
std::span<const test::TestCase> memoryTests();

int main(int argc, char **argv) {
  const test::TestGroup groups[] = {{"wire", wireTests()}, {"weights", weightsTests()}, {"memory", memoryTests()}};
  return test::runTests("native-engine-loop", groups, argc, argv, [] { test::metalStatistics() = {}; });
}
