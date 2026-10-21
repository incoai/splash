// The GGUF CPU reference's fp32 values against golden hashes of upstream
// GGML's dequantization per GGUF format and of llama.cpp's NVFP4 tensor type,
// its Q8_0 blocks against one of that library's Q8_0 quantization, and its values
// against MLX's own reading of MLX's quantization per MLX affine format and
// for mxfp4 and nvfp4.
//   gguf-reference GOLDENS MLX_FIXTURE
// GOLDENS is dev/tests/fixtures/weight-goldens/goldens.json; its README says
// how to update it. With SPLASH_GGML_ORACLE=<libggml-base.dylib> the reference
// is also compared with GGML directly and GGML's hashes are printed.
// MLX_FIXTURE is dev/tests/fixtures/mlx-quantization/fixture.json, which
// dev/tools/mlx_quantization_fixture.py writes with MLX.
#include "GgufFixtures.hpp"

#include <dlfcn.h>

using namespace gguf_fixtures;
using namespace gguf_reference;

namespace {

// The goldens hash fixture(f, kRows, kColumns, kSeed + f), and llama.cpp's
// NVFP4 blockNvfp4Fixture(kRows, kColumns, kSeed + GGUF_FMT_NVFP4).
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

// llama.cpp's NVFP4 (ggml type 40), which the loader converts into NVFP4's
// native rows: the reference's reading of block_nvfp4 rows whose scales take
// every byte (blockNvfp4Values) against GGML's golden hash, and GGML's own.
void checkBlockNvfp4(void *ggml, const Goldens &hashes) {
  const std::vector<uint8_t> rows = blockNvfp4Fixture(kRows, kColumns, kSeed + GGUF_FMT_NVFP4);
  const std::vector<float> values = blockNvfp4Values(rows);
  const std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t *>(values.data()), values.size() * sizeof(float));
  checkGolden(hashes, "ggml-nvfp4", bytes, "CPU reference matches the GGML golden hash: llama.cpp's NVFP4");
  if (!ggml) return;
  std::vector<float> official;
  std::string error;
  const bool loaded = ggmlDequantizeRows(ggml, "dequantize_row_nvfp4", rows, values.size(), official, error);
  check(loaded && official.size() == values.size() && !memcmp(official.data(), values.data(), bytes.size()),
        std::string("CPU reference matches GGML: llama.cpp's NVFP4") + (loaded ? "" : " (" + error + ")"));
  if (loaded)
    std::printf("GGML ggml-nvfp4 %s\n",
                model::weightDigest({reinterpret_cast<const uint8_t *>(official.data()),
                                     official.size() * sizeof(float)}).c_str());
}

// The GGUF Q8_0 quantization, which the loader applies to Qwen3.8-Flash-Next's BF16 and F16 hyper-connection
// projections (model/GgufPreparation.cpp): the reference's block_q8_0 rows of q8_0Values against their golden hash,
// and against the oracle library's own (quantize_row_q8_0_ref).
void checkQ8_0Quantization(void *ggml, const Goldens &hashes) {
  const std::vector<float> values = q8_0Values(uint64_t{kRows} * kColumns, kSeed + GGUF_FMT_Q80);
  const std::vector<uint8_t> blocks = quantizeQ8_0(values);
  checkGolden(hashes, "q8_0", blocks, "CPU reference matches the golden hash: Q8_0 quantization");
  if (!ggml) return;
  using Quantize = void (*)(const float *, void *, int64_t);
  const auto quantize = reinterpret_cast<Quantize>(dlsym(ggml, "quantize_row_q8_0_ref"));
  std::vector<uint8_t> official(blocks.size());
  if (quantize) quantize(values.data(), official.data(), int64_t(values.size()));
  check(quantize && official == blocks, std::string("CPU reference matches the oracle: Q8_0 quantization") +
                                            (quantize ? "" : std::string(" (") + dlerror() + ")"));
  if (quantize) std::printf("oracle q8_0 quantization %s\n", model::weightDigest(official).c_str());
}

// Each fixture tensor as the loader's native rows of its format (mlxNative)
// against MLX's own reading of it: an affine tensor's codes and the fp32
// values, bit for bit but for mxfp4's and nvfp4's zero code 8, which MLX reads as -0 and
// GGML's table, as +0.
void checkMlx(const char *path) {
  const std::vector<MlxTensor> tensors = mlxFixture(path);
  check(tensors.size() == GGUF_FMT_AF8G128 - GGUF_FMT_AF2G32 + 3,
        "MLX fixture holds every MLX affine format, mxfp4 and nvfp4");
  for (const MlxTensor &tensor : tensors) {
    const Fmt f = tensor.format();
    const uint32_t rows = tensor.rows, K = tensor.columns;
    const std::vector<uint8_t> native = tensor.native();
    if (tensor.affine()) {
      bool same = true;
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t l = 0; l < K; ++l)
          same &= affineCode(tensor.weight.data() + size_t(r) * K * tensor.bits / 8, tensor.bits, l) ==
                  tensor.codes[size_t(r) * K + l];
      check(same, std::string("CPU reference reads MLX's codes: ") + fmtName(f));
    }
    std::vector<float> reference(size_t(rows) * K);
    for (uint32_t r = 0; r < rows; ++r)
      rowValues(f, native.data() + size_t(r) * rowBytes(f, K), K, reference.data() + size_t(r) * K);
    bool same = native.size() == size_t(rows) * rowBytes(f, K) && reference.size() * 4 == tensor.values.size();
    for (size_t i = 0; same && i < reference.size(); ++i) {
      float value;
      memcpy(&value, tensor.values.data() + 4 * i, 4);
      same = tensor.affine() ? !memcmp(&value, &reference[i], 4) : value == reference[i];
    }
    check(same, std::string("CPU reference matches MLX's values: ") + fmtName(f) +
                    (tensor.affine() ? "" : " (MLX " + tensor.mode + ")"));
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 3) {
      std::fprintf(stderr, "usage: gguf-reference GOLDENS MLX_FIXTURE\n");
      return 2;
    }
    void *ggml = nullptr;
    if (const char *oracle = std::getenv("SPLASH_GGML_ORACLE")) {
      ggml = dlopen(oracle, RTLD_NOW | RTLD_LOCAL);
      check(ggml, std::string("load ") + oracle + (ggml ? "" : std::string(": ") + dlerror()));
    }
    const Goldens hashes = goldens(argv[1], @"gguf_dequantization");
    for (uint32_t f = 0; f < GGUF_FMT_COUNT; ++f)
      if (!quant_loader_format(f)) checkFormat(ggml, hashes, Fmt(f));
    checkBlockNvfp4(ggml, hashes);
    checkQ8_0Quantization(ggml, goldens(argv[1], @"gguf_quantization"));
    checkMlx(argv[2]);
    std::printf("%s (%d failures)\n", failures ? "GGUF reference tests FAILED" : "GGUF reference tests passed",
                failures);
    return failures ? 1 : 0;
  }
}
