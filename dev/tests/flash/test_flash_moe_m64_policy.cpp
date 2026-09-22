#include "flash/FlashMoEBlocked.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
void require(bool value, const char *reason) {
  if (!value) throw std::runtime_error(reason);
}
}

int main(int argc, char **argv) {
  using namespace splash::flash;
  try {
    require(argc == 2, "usage: POLICY_TEST off|q4|m64|reject");
    const std::string_view mode = argv[1];
    require(mode == "off" || mode == "q4" || mode == "m64" || mode == "reject",
            "unsupported policy test mode");
    if (mode == "reject") {
      bool rejected = false;
      try { (void)flashMoEBlockedRouteSemantics(); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "invalid flag/dependency was accepted");
      std::cout << "PASS strict M64 flag/dependency rejection; GPU work=false\n";
      return 0;
    }
    const bool m64 = mode == "m64";
    const std::string semantics = flashMoEBlockedRouteSemantics();
    require(semantics == (m64 ? kFlashMoEBlockedQ4x8M64Semantics :
        mode == "q4" ? kFlashMoEBlockedQ4x8Semantics : kFlashMoEBlockedSemantics),
        "producer route semantics differs");
    for (uint32_t rows : {1u,255u,256u,512u,1023u,1024u,2048u,4095u,4096u,8192u}) {
      const auto fallback = rows >= 1024 ? FlashMoEBlockedTile::M32N64 : FlashMoEBlockedTile::M16N64;
      require(flashMoEBlockedTile(rows,true) == fallback,
              "hot cache received unsupported M64 geometry");
      require(flashMoEBlockedTile(rows,false) == (m64 && rows >= 4096 ?
          FlashMoEBlockedTile::M64N64 : fallback), "physical-row eligibility differs");
    }
    for (uint32_t rows : {0u,8193u,0xffffffffu}) {
      bool rejected = false;
      try { (void)flashMoEBlockedTile(rows,false); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected,"unsupported physical extent was accepted");
    }
    require(setenv("SPLASH_FLASH_MOE_M64",m64 ? "0" : "1",1) == 0,
            "cannot mutate test flag");
    require(flashMoEBlockedRouteSemantics() == semantics &&
        flashMoEBlockedTile(4096,false) == (m64 ? FlashMoEBlockedTile::M64N64 :
            FlashMoEBlockedTile::M32N64), "producer policy changed after first use");
    std::cout << "PASS M64 row/hot-cache eligibility, identity and freeze; GPU work=false\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
