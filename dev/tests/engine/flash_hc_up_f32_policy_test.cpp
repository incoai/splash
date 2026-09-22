// CPU only: no Metal device, cache allocation, model load, or runtime service.
// Run flag modes in separate processes because the opt-in freezes on first use.
#include "flash/FlashHCFused.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using splash::flash::flashHCUpF32MPPEnabled;
using splash::flash::flashHCUpF32MPPGeometry;
using splash::flash::kFlashHCFusedExplicitSemantics;
using splash::flash::kFlashHCFusedGroupedSemantics;
using splash::flash::kFlashHCUpF32MPPSemantics;

constexpr const char *flag = "SPLASH_FLASH_HC_UP_F32_MPP";
constexpr std::string_view attentionUp =
    "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up";
constexpr std::string_view mlpUp =
    "language_model.model.layers.47.mlp_hyper_connection.input_mix_weight_up";
constexpr std::string_view finalUp =
    "language_model.model.hyper_connection_mixer.input_mix_weight_up";
uint64_t checks = 0;

static_assert(noexcept(flashHCUpF32MPPGeometry(std::string_view{}, 0, 0, 0)));
static_assert(std::string_view(kFlashHCFusedExplicitSemantics) ==
              "native-hc-literal-f32coeff-lane32-bf16-stages-fused-v1");

void require(bool condition, const std::string &message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

void setFlag(const char *value) {
  require(value ? ::setenv(flag, value, 1) == 0 : ::unsetenv(flag) == 0,
          "could not set the test process environment");
}

void expectGeometry(std::string_view prefix, uint32_t rows, uint32_t n,
                    uint32_t k, bool expected) {
  require(flashHCUpF32MPPGeometry(prefix, rows, n, k) == expected,
          "wrong HC up geometry decision: " + std::string(prefix) + " R" +
          std::to_string(rows) + " N" + std::to_string(n) + " K" + std::to_string(k));
}

void geometryTests() {
  for (uint32_t layer = 0; layer < 48; ++layer)
    for (std::string_view suffix : {
        ".attn_hyper_connection.input_mix_weight_up",
        ".mlp_hyper_connection.input_mix_weight_up"}) {
      const std::string prefix = "language_model.model.layers." + std::to_string(layer) +
                                 std::string(suffix);
      for (uint32_t rows = 4; rows <= 16; ++rows)
        expectGeometry(prefix, rows, 10240, 320, true);
    }
  for (uint32_t rows = 4; rows <= 16; ++rows)
    expectGeometry(finalUp, rows, 10240, 320, true);

  constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
  for (std::string_view prefix : {attentionUp, mlpUp, finalUp}) {
    for (uint32_t rows : {0u, 1u, 2u, 3u, 17u, 32u, 8192u, maximum})
      expectGeometry(prefix, rows, 10240, 320, false);
    for (uint32_t n : {0u, 320u, 10176u, 10304u, maximum})
      expectGeometry(prefix, 16, n, 320, false);
    for (uint32_t k : {0u, 288u, 352u, 10240u, maximum})
      expectGeometry(prefix, 16, 10240, k, false);
  }

  // Correct matrix dimensions cannot admit trained MTP or an unrelated role.
  constexpr std::array rejected{
      std::string_view{""},
      std::string_view{"unknown.input_mix_weight_up"},
      std::string_view{"mtp.layers.0.attn_hyper_connection.input_mix_weight_up"},
      std::string_view{"mtp.layers.0.mlp_hyper_connection.input_mix_weight_up"},
      std::string_view{"mtp.hyper_connection_mixer.input_mix_weight_up"},
      std::string_view{"language_model.lm_head"},
      std::string_view{"language_model.model.layers.0.linear_attn.in_proj_qkv"},
      std::string_view{"language_model.model.layers.3.self_attn.q_proj"},
      std::string_view{"language_model.model.layers.0.mlp.shared_expert.up_proj"},
      std::string_view{"language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down"},
      std::string_view{"language_model.model.layers.0.mlp_hyper_connection.input_mix_weight_down"},
      std::string_view{"language_model.model.hyper_connection_mixer.input_mix_weight_down"},
      std::string_view{"language_model.model.layers.0.attn_hyper_connection.block_inject_weight"},
      std::string_view{"language_model.model.layers.0.other.input_mix_weight_up"},
      std::string_view{"language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up.weight"},
      std::string_view{"language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up.extra"},
      std::string_view{"language_model.model.hyper_connection_mixer.input_mix_weight_up.extra"},
      std::string_view{"unknown.language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up"},
  };
  for (std::string_view prefix : rejected)
    for (uint32_t rows : {4u, 8u, 16u}) expectGeometry(prefix, rows, 10240, 320, false);

  for (std::string_view layer : {"", "-1", "+1", "x", "48", "99", "100",
                                "4294967295", "4294967296", "42949672960"})
    for (std::string_view suffix : {
        ".attn_hyper_connection.input_mix_weight_up",
        ".mlp_hyper_connection.input_mix_weight_up"}) {
      const std::string prefix = "language_model.model.layers." + std::string(layer) +
                                 std::string(suffix);
      expectGeometry(prefix, 8, 10240, 320, false);
    }
}

void markerTests() {
  const std::string_view marker(kFlashHCUpF32MPPSemantics);
  require(!marker.empty(), "F32 MPP reduction marker is empty");
  require(marker != kFlashHCFusedExplicitSemantics && marker != kFlashHCFusedGroupedSemantics,
          "F32 MPP reduction reused another HC arithmetic identity");
}

void validFlagTests(const char *initial, bool expected) {
  setFlag(initial);
  require(flashHCUpF32MPPEnabled() == expected, "wrong first HC up F32 flag decision");
  const std::string marker(kFlashHCUpF32MPPSemantics);
  for (const char *changed : std::array<const char *, 4>{
           expected ? "0" : "1", "invalid-after-freeze", nullptr, initial}) {
    setFlag(changed);
    require(flashHCUpF32MPPEnabled() == expected, "HC up F32 flag was re-read after freezing");
    require(std::string(kFlashHCUpF32MPPSemantics) == marker,
            "explicit F32 MPP reduction marker changed after flag selection");
  }
}

void invalidFlagTests(const char *value) {
  setFlag(value);
  bool rejected = false;
  try { (void)flashHCUpF32MPPEnabled(); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "malformed HC up F32 flag did not throw invalid_argument");
}
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view mode = argc == 1 ? "--flag-absent" : argv[1];
    require((mode == "--flag-absent" || mode == "--flag0" || mode == "--flag1")
                ? argc <= 2 : mode == "--flag-invalid" && argc == 3,
            "usage: flash-hc-up-f32-policy [--flag-absent|--flag0|--flag1|--flag-invalid VALUE]");
    // Pure geometry and static marker inspection must not parse/freeze the flag.
    setFlag("geometry-must-not-read-this");
    geometryTests(); markerTests();
    if (mode == "--flag-invalid") invalidFlagTests(argv[2]);
    else if (mode == "--flag1") validFlagTests("1", true);
    else if (mode == "--flag0") validFlagTests("0", false);
    else validFlagTests(nullptr, false);
    std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
              << ",\"gpu_commands\":0,\"mode\":\"" << mode << "\"}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_hc_up_f32_policy_test: " << error.what() << '\n';
    return 1;
  }
}
