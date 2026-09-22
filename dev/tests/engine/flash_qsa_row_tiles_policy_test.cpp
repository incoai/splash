// CPU only: no Metal device, model weights, command graph, or runtime service.
// Each flag mode runs in its own process because flag selection freezes lazily.
#include "flash/FlashQSAMPP.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using splash::flash::kFlashQSAOnlineMPPRoute;
using splash::flash::kFlashQSARowTilesRoute;
using splash::flash::qsaOnlineMPPRouteSemantics;
using splash::flash::qsaOnlineMPPRowTilesEnabled;
using splash::flash::qsaOnlineMPPRowTilesGeometry;

constexpr const char *flag = "SPLASH_FLASH_QSA_ROW_TILES";
uint64_t checks = 0;

static_assert(noexcept(qsaOnlineMPPRowTilesGeometry(0, 0, 0)));

void require(bool condition, const std::string &message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

void setFlag(const char *value) {
  require(value ? ::setenv(flag, value, 1) == 0 : ::unsetenv(flag) == 0,
          "could not set the test process environment");
}

void expectGeometry(uint32_t begin, uint32_t rows, uint32_t partitions,
                    bool expected) {
  require(qsaOnlineMPPRowTilesGeometry(begin, rows, partitions) == expected,
          "wrong geometry decision: begin=" + std::to_string(begin) +
          " rows=" + std::to_string(rows) + " partitions=" + std::to_string(partitions));
}

void geometryTests() {
  struct Case {
    uint32_t begin;
    uint32_t rows;
  };
  // Odd positions and row counts are eligible; exact end2048 tails are legal.
  constexpr std::array accepted{
      Case{512, 32}, Case{512, 33}, Case{512, 63}, Case{512, 64},
      Case{512, 65}, Case{512, 127}, Case{512, 128}, Case{513, 33},
      Case{514, 127}, Case{1023, 63}, Case{1024, 64}, Case{1919, 128},
      Case{1920, 128}, Case{1983, 65}, Case{1984, 64}, Case{2000, 32},
      Case{2015, 33}, Case{2016, 32},
  };
  for (const auto &entry : accepted) expectGeometry(entry.begin, entry.rows, 4, true);

  constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
  constexpr std::array rejected{
      Case{0, 32}, Case{511, 32}, Case{511, 128},
      Case{512, 0}, Case{512, 1}, Case{512, 16}, Case{512, 31},
      Case{512, 129}, Case{512, maximum},
      Case{1921, 128}, Case{1984, 65}, Case{2016, 33}, Case{2017, 32},
      Case{2048, 32}, Case{2049, 32}, Case{2051, 64}, Case{4096, 128},
      Case{maximum, 32}, Case{maximum - 31, 32}, Case{maximum, maximum},
  };
  for (const auto &entry : rejected) expectGeometry(entry.begin, entry.rows, 4, false);
  for (uint32_t partitions : {0u, 1u, 2u, 3u, 5u, 8u, 16u, 32u, maximum}) {
    expectGeometry(512, 64, partitions, false);
    expectGeometry(2016, 32, partitions, false);
  }
}

std::string expectedSemantics(bool enabled) {
  return std::string(kFlashQSAOnlineMPPRoute) + (enabled ? kFlashQSARowTilesRoute : "");
}

void checkSemantics(bool enabled) {
  const char *identifier = qsaOnlineMPPRouteSemantics();
  require(identifier != nullptr, "route semantics returned a null pointer");
  const std::string_view actual(identifier);
  require(actual.starts_with(kFlashQSAOnlineMPPRoute),
          "route semantics lost the full qualified MPP prefix");
  require(actual == expectedSemantics(enabled),
          "route semantics did not append the optional marker exactly once");
}

void validFlagTests(const char *initial, bool expected) {
  setFlag(initial);
  require(qsaOnlineMPPRowTilesEnabled() == expected, "wrong first flag decision");
  checkSemantics(expected);
  const std::string initialSemantics(qsaOnlineMPPRouteSemantics());

  // Changing or removing the environment after first use cannot change the
  // bool or identifier, and a later malformed value cannot cause a new error.
  for (const char *changed : std::array<const char *, 4>{
           expected ? "0" : "1", "invalid-after-freeze", nullptr, initial}) {
    setFlag(changed);
    require(qsaOnlineMPPRowTilesEnabled() == expected,
            "flag was re-read after its first successful use");
    checkSemantics(expected);
    require(std::string(qsaOnlineMPPRouteSemantics()) == initialSemantics,
            "route semantics changed after flag selection froze");
  }
}

void invalidFlagTests(const char *value) {
  setFlag(value);
  bool rejected = false;
  try {
    (void)qsaOnlineMPPRowTilesEnabled();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "malformed flag was not rejected by the flag accessor");

  rejected = false;
  try {
    (void)qsaOnlineMPPRouteSemantics();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "malformed flag was not rejected by route semantics");
}
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view mode = argc == 1 ? "--flag-absent" : argv[1];
    require((mode == "--flag-absent" || mode == "--flag0" || mode == "--flag1")
                ? argc <= 2 : mode == "--flag-invalid" && argc == 3,
            "usage: flash-qsa-row-tiles-policy [--flag-absent|--flag0|--flag1|--flag-invalid VALUE]");

    // Geometry inspection must neither parse nor freeze the optional flag.
    setFlag("geometry-must-not-read-this");
    geometryTests();
    if (mode == "--flag-invalid") invalidFlagTests(argv[2]);
    else if (mode == "--flag1") validFlagTests("1", true);
    else if (mode == "--flag0") validFlagTests("0", false);
    else validFlagTests(nullptr, false);

    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
              << ",\"gpu_commands\":0,\"mode\":\"" << mode << "\"}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_qsa_row_tiles_policy_test: " << error.what() << '\n';
    return 1;
  }
}
