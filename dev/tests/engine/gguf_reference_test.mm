// The GGUF CPU reference's fp32 values against golden hashes of upstream
// GGML's dequantization, per format.
//   gguf-reference GOLDENS
// GOLDENS is dev/tests/fixtures/weight-goldens/goldens.json; its README says
// how to update it. With SPLASH_GGML_ORACLE=<libggml-base.dylib> the reference
// is also compared with GGML directly and GGML's hashes are printed.
#include "GgufFixtures.hpp"

#include <dlfcn.h>

using namespace gguf_fixtures;
using namespace gguf_reference;

namespace {

// The goldens hash fixture(f, kRows, kColumns, kSeed + f).
constexpr uint32_t kRows = 256, kColumns = 1024, kSeed = 7;

void checkFormat(void *ggml, const Goldens &hashes, Fmt f) {
  const std::vector<uint8_t> native = fixture(f, kRows, kColumns, kSeed + f);
  std::vector<float> values;
  repack(f, native, kRows, kColumns, &values);
  const std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t *>(values.data()), values.size() * sizeof(float));
  checkGolden(hashes, fmtName(f), bytes, std::string("CPU reference matches the GGML golden hash: ") + fmtName(f));
  if (!ggml) return;
  std::vector<float> official;
  std::string error;
  const bool loaded = ggmlDequantize(ggml, f, native, official, error);
  check(loaded && official.size() == values.size() && !memcmp(official.data(), values.data(), bytes.size()),
        std::string("CPU reference matches GGML: ") + fmtName(f) + (loaded ? "" : " (" + error + ")"));
  if (loaded)
    std::printf("GGML %s %s\n", fmtName(f),
                model::weightDigest({reinterpret_cast<const uint8_t *>(official.data()),
                                     official.size() * sizeof(float)}).c_str());
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::fprintf(stderr, "usage: gguf-reference GOLDENS\n");
      return 2;
    }
    void *ggml = nullptr;
    if (const char *oracle = std::getenv("SPLASH_GGML_ORACLE")) {
      ggml = dlopen(oracle, RTLD_NOW | RTLD_LOCAL);
      check(ggml, std::string("load ") + oracle + (ggml ? "" : std::string(": ") + dlerror()));
    }
    const Goldens hashes = goldens(argv[1], @"gguf_dequantization");
    for (int f = 0; f < FMT_COUNT; ++f) checkFormat(ggml, hashes, Fmt(f));
    std::printf("%s (%d failures)\n", failures ? "GGUF reference tests FAILED" : "GGUF reference tests passed",
                failures);
    return failures ? 1 : 0;
  }
}
