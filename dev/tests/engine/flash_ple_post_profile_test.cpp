// CPU-only route-contract test. No Metal backend is created or submitted.
// Fork before the first profile read to prove each literal and frozen route.
#include "flash/FlashPLE.hpp"
#include "flash/FlashPLEPostFused.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace splash::flash;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

void caseBody(const char *value, bool valid, bool expected) {
  if (value)
    require(setenv("SPLASH_FLASH_PLE_POST_FUSED", value, 1) == 0, "setenv failed");
  else
    require(unsetenv("SPLASH_FLASH_PLE_POST_FUSED") == 0, "unsetenv failed");
  if (valid) {
    require(flashPLEPostFusedEnabled() == expected, "literal chose wrong route");
    const std::string_view semantics = flashPLEPostRouteSemantics();
    require(semantics == (expected ? kFlashPLEPostFusedSemantics :
        "source-seven-dispatch-ple-post-inject-bf16-v1"), "route tag differs");
    // Environment changes after selection must not silently change the graph
    // route or its status identity in the same worker process.
    require(setenv("SPLASH_FLASH_PLE_POST_FUSED", expected ? "0" : "1", 1) == 0,
            "frozen-route setenv failed");
    require(flashPLEPostFusedEnabled() == expected, "route was not frozen");
    require(flashPLEPostRouteSemantics() == semantics, "frozen route tag changed");
  } else {
    bool caught = false;
    try { static_cast<void>(flashPLEPostRouteSemantics()); }
    catch (const std::invalid_argument &error) {
      caught = std::string_view(error.what()) ==
          "SPLASH_FLASH_PLE_POST_FUSED must be 0 or 1";
    }
    require(caught, "malformed literal did not reject");
  }

  // Rejections must occur before appending a partial command graph, including
  // malformed profile values and invalid argument packets on either route.
  splash::metal::CommandGraph graph;
  bool rejected = false;
  try {
    addPLEPostProjectAndInject(graph, FlashPLEWeights{}, {}, {}, {},
                              FlashPLEPostScratch{}, {}, {}, {},
                              FlashPLEGeometry{});
  } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected && graph.empty(), "invalid wrapper partially built graph");
}

void runCase(const char *value, bool valid, bool expected = false) {
  const pid_t child = fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    try { caseBody(value, valid, expected); _exit(0); }
    catch (const std::exception &error) {
      std::fprintf(stderr, "PLE post profile case %s: %s\n",
                   value ? value : "unset", error.what());
      _exit(1);
    }
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0, "profile child failed");
}
} // namespace

int main() {
  try {
    runCase(nullptr, true);
    runCase("0", true);
    runCase("1", true, true);
    for (const char *value : {"", "true", "false", "01", "1 ", " 1", "-1", "2"})
      runCase(value, false);
    std::puts("PASS PLE post profile: 11 literal/frozen/empty-graph cases; zero GPU commands");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PLE post profile test failed: %s\n", error.what());
    return 1;
  }
}
