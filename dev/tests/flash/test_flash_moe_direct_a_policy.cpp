#include "flash/FlashMoEBlocked.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
uint64_t round(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}
void require(bool condition, const char *reason) {
  if (!condition) throw std::runtime_error(reason);
}
}

int main() {
  try {
    using namespace splash::flash;
    const bool direct = flashMoEDirectAEnabled();
    unsigned checks = 0;
    for (uint32_t rows : {1u, 129u, 512u, 2048u, 8192u}) {
      for (uint32_t selections : {1u, 10u}) {
        const uint64_t routes = uint64_t(rows) * selections;
        for (uint64_t alignment : {4096u, 16384u}) {
          // Independent integer formula, not production helper reuse.
          const uint64_t jobs = (routes + 7) / 8 + 511;
          const uint64_t padded = routes + (direct ? 63 : 0);
          uint64_t expected = 0;
          for (uint64_t size : std::array<uint64_t, 10>{2048,2052,routes*4,routes*4,
              padded*5120,2052,4,jobs*8,padded*1280,routes*5120})
            expected += round(size, alignment);
          require(flashMoEBlockedWorkspacePlannedBytes(rows, selections, alignment) == expected,
                  "blocked workspace planned allocation total differs");
          const uint64_t extra = direct
              ? round((routes + 63) * 5120, alignment) - round(routes * 5120, alignment) +
                round((routes + 63) * 1280, alignment) - round(routes * 1280, alignment)
              : 0;
          require(flashMoEDirectAWorkspaceExtraBytes(uint32_t(routes), alignment) == extra,
                  "direct A rounded padding delta differs");
          checks += 2;
        }
      }
    }
    for (const std::array<uint32_t, 2> geometry :
         {std::array<uint32_t,2>{0,10}, {8193,10}, {512,0}, {512,11}}) {
      bool rejected = false;
      try { (void)flashMoEBlockedWorkspacePlannedBytes(geometry[0],geometry[1]); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "invalid planned blocked geometry accepted"); ++checks;
    }
    for (uint64_t alignment : {0u,3u,1000u}) {
      bool rejected = false;
      try { (void)flashMoEBlockedWorkspacePlannedBytes(512,10,alignment); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "invalid planner alignment accepted"); ++checks;
    }
    require(std::string(flashMoEBlockedRouteSemantics()).find("device-a") != std::string::npos || !direct,
            "direct A route is missing from semantic identity"); ++checks;
    require(setenv("SPLASH_FLASH_MOE_DIRECT_A", direct ? "0" : "1", 1) == 0,
            "cannot mutate CPU test environment");
    require(flashMoEDirectAEnabled() == direct, "policy changed after first use"); ++checks;
    std::cout << "{\"pass\":true,\"gpu_work\":false,\"direct_a\":"
              << (direct ? "true" : "false") << ",\"checks\":" << checks << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
