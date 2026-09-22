// CPU only. Geometry, register ownership and frozen policy are inspected
// without constructing a MetalBackend/device, loading weights or submitting.
#include "flash/FlashGDNBatchILP.hpp"
#include "metal/abi/FlashGDNBatchILP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace splash::flash;
uint64_t checks = 0;
constexpr const char *kFlag = "SPLASH_FLASH_GDN_BATCH_ILP";
constexpr const char *kStaged = "SPLASH_FLASH_GDN_STAGED";
constexpr const char *kPrefill = "SPLASH_FLASH_BATCH_PREFILL";

static_assert(sizeof(splash::metal::CommandTiming) == 200);
static_assert(sizeof(float) == 4);
static_assert(noexcept(flashGDNBatchILPEligible(0, 0)));
static_assert(sizeof(FlashGDNBatchILPParams) == 64);

void require(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

template <class Operation> void rejects(Operation operation, const char *message) {
  bool caught = false;
  try { operation(); } catch (const std::invalid_argument &) { caught = true; }
  require(caught, message);
}

void environment(const char *name, std::string_view value) {
  const std::string owned(value);
  require(value == "@unset" ? ::unsetenv(name) == 0
                            : ::setenv(name, owned.c_str(), 1) == 0,
          "could not configure the test process environment");
}

void geometry() {
  // Pure metadata inspection must neither read nor freeze policy.
  environment(kFlag, "geometry-must-not-read-this");
  environment(kStaged, "geometry-must-not-read-this");
  environment(kPrefill, "geometry-must-not-read-this");
  const std::array<FlashGDNBatchILPLane, 4> masked{};
  for (uint32_t slots = 1; slots <= 4; ++slots)
    for (uint32_t stride : {1u, 2u, 128u, 2048u, 2051u, 4096u})
      require(validateGDNBatchILPGeometry(std::span(masked.data(), slots), stride) == 0,
              "all-masked metadata required synthetic state");
  rejects([&] { (void)validateGDNBatchILPGeometry({}, 7); }, "empty span accepted");
  rejects([&] {
    const std::array<FlashGDNBatchILPLane, 5> tooMany{};
    (void)validateGDNBatchILPGeometry(tooMany, 7);
  }, "five slots accepted");
  for (uint32_t stride : {0u, 4097u, std::numeric_limits<uint32_t>::max()})
    rejects([&] { (void)validateGDNBatchILPGeometry(masked, stride); },
            "invalid physical stride accepted");
  for (float epsilon : {0.f, -1.f, float(NAN), float(INFINITY), -float(INFINITY)})
    rejects([&] { (void)validateGDNBatchILPGeometry(masked, 7, {}, epsilon); },
            "invalid RMS epsilon accepted");
  for (uint32_t slot = 0; slot < 4; ++slot)
    for (uint32_t rows : {1u, 2u, 7u, 8u, 2048u, 2049u,
                         std::numeric_limits<uint32_t>::max()})
      rejects([&] {
        auto missing = masked; missing[slot].rows = rows;
        (void)validateGDNBatchILPGeometry(missing, 7);
      }, "missing live state or row overflow accepted");
  for (FlashGDNBatchILPTile tile : std::array<FlashGDNBatchILPTile, 8>{{
           {0, 32, 8}, {16, 32, 8}, {64, 32, 8}, {32, 0, 8},
           {32, 8, 8}, {32, 64, 8}, {32, 32, 4}, {32, 32, 16}}}) {
    rejects([&] { validateGDNBatchILPTile(tile); }, "unsupported ILP tile accepted");
    rejects([&] { (void)validateGDNBatchILPGeometry(masked, 7, tile); },
            "unsupported ILP tile bypassed geometry");
  }
  for (FlashGDNBatchILPTile tile : std::array<FlashGDNBatchILPTile, 2>{{
           {32, 16, 8}, {32, 32, 8}}}) {
    validateGDNBatchILPTile(tile);
    require(validateGDNBatchILPGeometry(masked, 4096, tile) == 0,
            "qualified tile rejected masked geometry");
  }
  for (uint32_t lanes : {0u, 1u, 2u, 3u, 4u, 5u,
                          std::numeric_limits<uint32_t>::max()})
    for (uint32_t rows : {0u, 1u, 16u, 129u, 511u, 512u, 513u, 1024u,
                         2047u, 2048u, 2049u,
                         std::numeric_limits<uint32_t>::max()})
      require(flashGDNBatchILPEligible(lanes, rows) ==
                  (lanes >= 2 && lanes <= 4 && rows >= 512 && rows <= 2048),
              "wrong batched GDN eligibility boundary");
}

void ownershipAndTails() {
  // Every independent F32 [value,key] cell is owned once by the flattened
  // four value blocks, eight SIMD groups, SIMD32 lanes and four adjacent keys.
  for (FlashGDNBatchILPTile tile : std::array<FlashGDNBatchILPTile, 2>{{
           {32, 16, 8}, {32, 32, 8}}}) {
    std::array<uint8_t, 128 * 128> visits{};
    for (uint32_t block = 0; block < 128 / tile.values; ++block)
      for (uint32_t sg = 0; sg < tile.simds; ++sg)
        for (uint32_t value = 0; value < tile.values / tile.simds; ++value)
          for (uint32_t lane = 0; lane < 32; ++lane)
            for (uint32_t key = 0; key < 4; ++key)
              ++visits[(block * tile.values + value * tile.simds + sg) * 128 +
                       lane * 4 + key];
    for (auto count : visits)
      require(count == 1, "register ownership duplicates or omits an F32 cell");
    for (uint32_t rows : {0u, 1u, 2u, 3u, 7u, 8u, 9u, 15u, 16u, 17u,
                         31u, 32u, 33u, 129u, 511u, 512u, 513u, 2048u}) {
      uint32_t visited = 0;
      for (uint32_t begin = 0; begin < rows; begin += tile.time)
        visited += std::min(tile.time, rows - begin);
      require(visited == rows, "temporal blocks omit or invent actual rows");
    }
  }
}

void policy(std::string_view flag, std::string_view staged,
            std::string_view prefill, std::string_view expected) {
  environment(kFlag, flag);
  environment(kStaged, staged);
  environment(kPrefill, prefill);
  if (expected == "reject") {
    rejects([] { (void)flashGDNBatchILPEnabled(); },
            "malformed policy or missing prerequisite accepted");
    return;
  }
  require(expected == "0" || expected == "1", "invalid expected policy result");
  const bool wanted = expected == "1";
  require(flashGDNBatchILPEnabled() == wanted, "wrong initial policy selection");
  // Frozen selection survives mutation/removal of every prerequisite and
  // malformed later values. Explicit Root0 remains disabled.
  for (const char *changed : std::array<const char *, 4>{
           wanted ? "0" : "1", "malformed-after-freeze", "@unset", "1"}) {
    environment(kFlag, changed);
    environment(kStaged, changed);
    environment(kPrefill, changed);
    require(flashGDNBatchILPEnabled() == wanted, "policy re-read after freezing");
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 1 || (argc == 6 && std::string_view(argv[1]) == "--policy-value"),
        "usage: flash-gdn-batch-ilp-cpu [--policy-value FLAG STAGED PREFILL EXPECTED]");
    geometry();
    if (argc == 1) ownershipAndTails();
    else policy(argv[2], argv[3], argv[4], argv[5]);
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
        << ",\"command_timing_size_bytes\":" << sizeof(splash::metal::CommandTiming)
        << ",\"metal_backend_constructions\":0,\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_gdn_batch_ilp_cpu_test: " << error.what() << '\n';
    return 1;
  }
}
