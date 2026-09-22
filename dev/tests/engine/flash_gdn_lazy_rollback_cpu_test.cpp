// CPU only: policy and admission metadata; no backend/device/model creation.
#include "flash/FlashGDNLazyRollback.hpp"
#include "flash/FlashForward.hpp"
#include "flash/FlashMTPWindow.hpp"
#include "metal/abi/FlashGDNLazyRollback.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace splash::flash;
uint64_t checks = 0;
constexpr uint64_t alignment = 16384, guard = 64;
constexpr const char *flag = "SPLASH_FLASH_GDN_LAZY_ROLLBACK";
constexpr const char *fused = "SPLASH_FLASH_FUSE_GDN";

static_assert(sizeof(FlashGDNLazyVerifyParams) == 56);
static_assert(sizeof(FlashGDNLazyReplayParams) == 72);
static_assert(sizeof(FlashGDNLazyCopyParams) == 8);
static_assert(sizeof(splash::metal::CommandTiming) == 200);
static_assert(sizeof(float) == 4);

void require(bool value, const char *message) {
  ++checks; if (!value) throw std::runtime_error(message);
}
template <class Operation> void rejects(Operation operation, const char *message) {
  bool failed = false;
  try { operation(); } catch (const std::invalid_argument &) { failed = true; }
  require(failed, message);
}
uint64_t rounded(uint64_t value) { return (value + alignment - 1) / alignment * alignment; }
void environment(const char *name, std::string_view value) {
  const std::string text(value);
  require(value == "@unset" ? ::unsetenv(name) == 0 : ::setenv(name, text.c_str(), 1) == 0,
          "could not set test process environment");
}
uint64_t sixPlanes(uint32_t rows, uint32_t lanes) {
  if (rows == 1) return 0;
  const uint64_t initialFP32 = uint64_t(lanes) * 48 * 128 * 128 * sizeof(float);
  const uint64_t initialBF16History = uint64_t(lanes) * 3 * 10240 * sizeof(uint16_t);
  const uint64_t rawBF16QKV = uint64_t(lanes) * rows * 10240 * sizeof(uint16_t);
  const uint64_t preparedBF16Mixed = rawBF16QKV;
  const uint64_t decayFP32 = uint64_t(lanes) * rows * 48 * sizeof(float);
  const uint64_t betaBF16 = uint64_t(lanes) * rows * 48 * sizeof(uint16_t);
  uint64_t result = 0;
  for (auto bytes : {initialFP32, initialBF16History, rawBF16QKV,
                     preparedBF16Mixed, decayFP32, betaBF16})
    result += rounded(bytes + guard);
  return result;
}
void footprint() {
  // Planning is independent of optional policy and may run before flags freeze.
  environment(flag, "planning-must-not-read-this");
  environment(fused, "planning-must-not-read-this");
  for (const auto &[rows, lanes] : std::array<std::pair<uint32_t, uint32_t>, 6>{{
           {0, 1}, {17, 1}, {1, 0}, {1, 5}, {UINT32_MAX, 1}, {1, UINT32_MAX}}})
    rejects([&] { (void)FlashGDNLazyRollback::plannedBytes(rows, lanes); },
            "invalid lazy arena geometry accepted");
  for (uint32_t rows = 1; rows <= 16; ++rows)
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto actual = FlashGDNLazyRollback::plannedBytes(rows, lanes);
      require(actual == sixPlanes(rows, lanes), "six-plane payload/guard accounting differs");
      require(actual % alignment == 0, "planned arena isn't 16K aligned");
      if (rows == 1) require(actual == 0, "R1 created snapshot/prework footprint");
      for (uint32_t kept = 1; kept <= rows; ++kept) {
        const auto prefix = *flashMTPConvolutionPrefix(kept);
        std::array<uint32_t, 3> got{}, expected{};
        for (uint32_t i = 0; i < prefix.oldRows; ++i) got[i] = prefix.oldBegin + i;
        for (uint32_t i = 0; i < prefix.inputRows; ++i)
          got[prefix.destinationInputBegin + i] = 3 + prefix.inputBegin + i;
        for (uint32_t i = 0; i < 3; ++i) expected[i] = kept + i;
        require(got == expected, "history prefix differs from independent concatenation");
      }
    }
  for (uint32_t rows : {4u, 8u, 16u})
    for (uint32_t lanes : {1u, 4u}) {
      const uint64_t currentEagerStateTape = uint64_t(lanes) * (rows - 1) *
          rounded(flashGDNRecurrentLaneBytes());
      const auto lazy = FlashGDNLazyRollback::plannedBytes(rows, lanes);
      require(lazy < currentEagerStateTape, "lazy didn't reduce R4/R8/R16 state tape");
      require(uint64_t(36) * lazy < uint64_t(36) * currentEagerStateTape,
              "36 GDN layers didn't retain footprint savings");
    }
}
void sourcePlanner(bool lazy) {
  const uint64_t pleAllowance = rounded(2 * sizeof(int64_t)) +
      rounded(uint64_t(9) * 10240 * sizeof(uint16_t)) + rounded(sizeof(uint32_t));
  require(pleAllowance == 229376, "PLE rollback allowance changed unexpectedly");
  for (uint32_t rows = 0; rows <= 16; ++rows) {
    uint64_t expected = 0;
    if (rows > 1) {
      const uint64_t gdn = lazy ? uint64_t(36) * sixPlanes(rows, 1)
          : uint64_t(36) * (rows - 1) * (rounded(flashGDNRecurrentLaneBytes()) +
                                             rounded(flashGDNConvolutionLaneBytes()));
      expected = gdn + pleAllowance;
    }
    require(FlashForward::verificationWorkspaceBytes(rows) == expected,
            "actual Forward source admission omitted GDN or PLE rollback allowance");
  }
  for (uint32_t rows : {17u, UINT32_MAX})
    rejects([&] { (void)FlashForward::verificationWorkspaceBytes(rows); },
            "Forward source planner accepted oversized verify cap");
}
void policy(std::string_view initial, std::string_view prerequisite,
            std::string_view expected) {
  environment(flag, initial); environment(fused, prerequisite);
  // This flag is deliberately unrelated to the required FUSE_GDN policy.
  environment("SPLASH_FLASH_GDN_STAGED", "not-a-lazy-prerequisite");
  if (expected == "reject") {
    rejects([] { (void)flashGDNLazyRollbackEnabled(); },
            "malformed flag or missing fused-GDN prerequisite accepted");
    return;
  }
  require(expected == "0" || expected == "1", "invalid expected policy value");
  const bool wanted = expected == "1";
  require(flashGDNLazyRollbackEnabled() == wanted, "wrong lazy flag decision");
  sourcePlanner(wanted);
  for (const char *changed : std::array<const char *, 4>{
           wanted ? "0" : "1", "malformed-after-freeze", "@unset", "1"}) {
    environment(flag, changed); environment(fused, changed);
    require(flashGDNLazyRollbackEnabled() == wanted, "lazy policy reread after freeze");
    sourcePlanner(wanted);
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 1 || (argc == 5 && std::string_view(argv[1]) == "--policy-value"),
            "usage: flash-gdn-lazy-rollback-cpu [--policy-value FLAG FUSE_GDN EXPECTED]");
    footprint();
    if (argc == 5) policy(argv[2], argv[3], argv[4]);
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
        << ",\"source_forward_planner_checked\":" << (argc == 5 && std::string_view(argv[4]) != "reject" ? "true" : "false")
        << ",\"command_timing_size_bytes\":200,\"metal_backend_constructions\":0,\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_gdn_lazy_rollback_cpu_test: " << error.what() << '\n'; return 1;
  }
}
