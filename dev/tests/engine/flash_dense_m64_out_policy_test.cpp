// CPU only: no Metal device, model load, cache construction, or runtime service.
// Use separate processes for flag modes because the flag freezes on first use.
#include "flash/FlashDenseCache.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using splash::flash::flashDenseCacheExecutionSemantics;
using splash::flash::flashDenseM64OutEnabled;
using splash::flash::flashDenseM64OutGeometry;
using splash::flash::kFlashDenseCacheExecutionSemantics;
using splash::flash::kFlashDenseM64OutSemantics;

constexpr const char *flag = "SPLASH_FLASH_DENSE_M64_OUT";
constexpr std::string_view linearOut = "language_model.model.layers.0.linear_attn.out_proj";
constexpr std::string_view attentionOut = "language_model.model.layers.47.self_attn.o_proj";
uint64_t checks = 0;

static_assert(noexcept(flashDenseM64OutGeometry(std::string_view{}, 0, 0, 0)));
static_assert(std::string_view(kFlashDenseCacheExecutionSemantics) ==
              "cached-bf16-weights-whole-k-mpp-f32accum-bf16-v1");

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
  require(flashDenseM64OutGeometry(prefix, rows, n, k) == expected,
          "wrong geometry decision: " + std::string(prefix) + " R" +
          std::to_string(rows) + " N" + std::to_string(n) + " K" + std::to_string(k));
}

void geometryTests() {
  // Both output roles admit every main layer; row tails remain eligible.
  for (uint32_t layer = 0; layer < 48; ++layer)
    for (std::string_view suffix : {".linear_attn.out_proj", ".self_attn.o_proj"}) {
      const std::string prefix = "language_model.model.layers." + std::to_string(layer) +
                                 std::string(suffix);
      for (uint32_t rows : {512u, 513u, 2048u, 8192u})
        expectGeometry(prefix, rows, 2560, 6144, true);
    }

  constexpr uint32_t maximum = std::numeric_limits<uint32_t>::max();
  for (std::string_view prefix : {linearOut, attentionOut}) {
    for (uint32_t rows : {519u, 1023u, 4097u})
      expectGeometry(prefix, rows, 2560, 6144, true);
    for (uint32_t rows : {0u, 1u, 16u, 511u, 8193u, maximum})
      expectGeometry(prefix, rows, 2560, 6144, false);
    for (uint32_t n : {0u, 2496u, 2624u, 12288u, maximum})
      expectGeometry(prefix, 2048, n, 6144, false);
    for (uint32_t k : {0u, 6112u, 6176u, 2560u, maximum})
      expectGeometry(prefix, 2048, 2560, k, false);
  }

  // Matching matrix dimensions must not admit a different namespace or role.
  constexpr std::array rejected{
      std::string_view{""},
      std::string_view{"unknown.linear_attn.out_proj"},
      std::string_view{"mtp.layers.0.linear_attn.out_proj"},
      std::string_view{"mtp.layers.0.self_attn.o_proj"},
      std::string_view{"language_model.lm_head"},
      std::string_view{"language_model.model.layers.0.linear_attn.in_proj_qkv"},
      std::string_view{"language_model.model.layers.3.self_attn.q_proj"},
      std::string_view{"language_model.model.layers.0.mlp.shared_expert.down_proj"},
      std::string_view{"language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down"},
      std::string_view{"language_model.model.hyper_connection_mixer.input_mix_weight_down"},
      std::string_view{"language_model.model.layers.0.linear_attn.out_proj.weight"},
      std::string_view{"language_model.model.layers.0.linear_attn.out_proj.extra"},
      std::string_view{"language_model.model.layers.47.self_attn.o_proj.extra"},
      std::string_view{"unknown.language_model.model.layers.0.linear_attn.out_proj"},
      std::string_view{"language_model.model.layers.0.other.out_proj"},
      std::string_view{"language_model.model.layers.0.self_attn.out_proj"},
      std::string_view{"language_model.model.layers.0.linear_attn.o_proj"},
  };
  for (std::string_view prefix : rejected)
    for (uint32_t rows : {512u, 513u, 2048u, 8192u})
      expectGeometry(prefix, rows, 2560, 6144, false);

  for (std::string_view layer : {"", "-1", "+1", "x", "48", "99", "100",
                                "4294967295", "4294967296", "42949672960"})
    for (std::string_view suffix : {".linear_attn.out_proj", ".self_attn.o_proj"}) {
      const std::string prefix = "language_model.model.layers." + std::string(layer) +
                                 std::string(suffix);
      expectGeometry(prefix, 2048, 2560, 6144, false);
    }
}

void checkSemantics(bool enabled) {
  const char *identifier = flashDenseCacheExecutionSemantics();
  require(identifier != nullptr, "execution semantics returned a null pointer");
  const std::string_view actual(identifier);
  require(actual.starts_with(kFlashDenseCacheExecutionSemantics),
          "execution semantics lost the complete qualified BF16 prefix");
  const std::string expected = std::string(kFlashDenseCacheExecutionSemantics) +
                              (enabled ? kFlashDenseM64OutSemantics : "");
  require(actual == expected, "optional output marker was missing, altered, or duplicated");
}

void validFlagTests(const char *initial, bool expected) {
  setFlag(initial);
  if (expected) checkSemantics(true); // Also prove semantics can freeze the flag first.
  else require(!flashDenseM64OutEnabled(), "absent/0 flag unexpectedly enabled M64");

  // Flip immediately after the first public accessor, then check both accessors.
  setFlag(expected ? "0" : "1");
  require(flashDenseM64OutEnabled() == expected, "flag did not freeze on first use");
  checkSemantics(expected);
  const std::string frozen(flashDenseCacheExecutionSemantics());
  for (const char *changed : std::array<const char *, 4>{
           "invalid-after-freeze", nullptr, "0", "1"}) {
    setFlag(changed);
    require(flashDenseM64OutEnabled() == expected, "flag was re-read after freezing");
    checkSemantics(expected);
    require(std::string(flashDenseCacheExecutionSemantics()) == frozen,
            "execution identity changed after flag selection froze");
  }
}

void invalidFlagTests(const char *value) {
  setFlag(value);
  bool rejected = false;
  try { (void)flashDenseM64OutEnabled(); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "malformed flag was not rejected by the flag accessor");
  rejected = false;
  try { (void)flashDenseCacheExecutionSemantics(); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "malformed flag was not rejected by execution semantics");
}
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view mode = argc == 1 ? "--flag-absent" : argv[1];
    require((mode == "--flag-absent" || mode == "--flag0" || mode == "--flag1")
                ? argc <= 2 : mode == "--flag-invalid" && argc == 3,
            "usage: flash-dense-m64-out-policy [--flag-absent|--flag0|--flag1|--flag-invalid VALUE]");
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
    std::cerr << "flash_dense_m64_out_policy_test: " << error.what() << '\n';
    return 1;
  }
}
