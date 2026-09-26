// Prism ML's input rotation (kernels/shared/gguf_rotation.metal): gguf_rotate's H (D x) and the rotated PQ2_0
// token gather's D (H r) are bitwise the bf16 rounding of the fp32 butterflies in source order, which lie within
// one bf16 step of the fp64 transform.
//   gguf-rotation <splash.metallib>
#include "GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace gguf_reference;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

namespace {

constexpr uint32_t kBlock = GGUF_ROTATION_BLOCK;

float bf16ToFloat(uint16_t bits) {
  const uint32_t word = uint32_t{bits} << 16;
  float value;
  std::memcpy(&value, &word, 4);
  return value;
}
// Round to nearest even, as the kernels' bfloat conversion.
uint16_t floatToBf16(float value) {
  uint32_t word;
  std::memcpy(&word, &value, 4);
  return uint16_t((word + 0x7FFF + ((word >> 16) & 1)) >> 16);
}

// The unnormalized transform of one block in fp32, stage by stage as the kernel computes it.
void butterflies(float *v) {
  for (uint32_t stride = 1; stride < kBlock; stride *= 2)
    for (uint32_t a = 0; a < kBlock; ++a)
      if (!(a & stride)) {
        const float x = v[a], y = v[a + stride];
        v[a] = x + y;
        v[a + stride] = x - y;
      }
}
void butterflies(double *v) {
  for (uint32_t stride = 1; stride < kBlock; stride *= 2)
    for (uint32_t a = 0; a < kBlock; ++a)
      if (!(a & stride)) {
        const double x = v[a], y = v[a + stride];
        v[a] = x + y;
        v[a + stride] = x - y;
      }
}

template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = backend.allocateBuffer(values.size() * sizeof(T));
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}

// Compares got with the fp32 reference bitwise and with the fp64 values within one bf16 step.
int check(const char *what, const uint16_t *got, const std::vector<float> &fp32, const std::vector<double> &fp64) {
  size_t differ = 0, far = 0;
  for (size_t i = 0; i < fp32.size(); ++i) {
    differ += got[i] != floatToBf16(fp32[i]);
    const double step = std::ldexp(1.0, std::ilogb(std::max(std::fabs(fp64[i]), 1e-30)) - 7);
    far += std::fabs(double(bf16ToFloat(got[i])) - fp64[i]) > step;
  }
  const bool ok = !differ && !far;
  std::cout << what << ": " << differ << " of " << fp32.size() << " values differ from bf16(fp32 butterflies), "
            << far << " lie beyond a bf16 step of fp64 " << (ok ? "ok" : "FAIL") << '\n';
  return ok ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf-rotation <splash.metallib>\n";
      return 2;
    }
    std::mt19937 rng(7);
    int failures = 0;
    try {
      MetalBackend backend(argv[1]);
      std::vector<int8_t> signs(3 * kBlock);
      for (int8_t &s : signs) s = rng() & 1 ? 1 : -1;
      const MetalBuffer signBuffer = upload(backend, signs);

      {  // gguf_rotate: rows of three blocks.
        constexpr uint32_t rows = 5, width = 3 * kBlock;
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<uint16_t> input(rows * width);
        for (uint16_t &x : input) x = floatToBf16(normal(rng));
        std::vector<float> fp32(input.size());
        std::vector<double> fp64(input.size());
        for (uint32_t r = 0; r < rows; ++r)
          for (uint32_t b = 0; b < width; b += kBlock) {
            float *f = fp32.data() + r * width + b;
            double *d = fp64.data() + r * width + b;
            for (uint32_t i = 0; i < kBlock; ++i) {
              f[i] = bf16ToFloat(input[r * width + b + i]) * float(signs[b + i]);
              d[i] = f[i];
            }
            butterflies(f);
            butterflies(d);
            for (uint32_t i = 0; i < kBlock; ++i) f[i] *= 1.0f / 32.0f, d[i] /= 32.0;
          }
        const MetalBuffer in = upload(backend, input), out = backend.allocateBuffer(input.size() * 2);
        CommandGraph graph;
        graph.add("gguf_rotate", {in, signBuffer, out}, GgufRotationParams{width}, {width / kBlock, rows, 1},
                  {GGUF_ROTATION_THREADS, 1, 1});
        static_cast<void>(backend.submitCommand(graph.dispatches()));
        failures += check("gguf_rotate H (D x)", static_cast<const uint16_t *>(out.contents()), fp32, fp64);
      }

      {  // gguf_embed_rotated_pq20: PQ2_0 rows of two blocks, gathered as D (H r).
        constexpr uint32_t vocabulary = 7, hidden = 2 * kBlock;
        std::vector<uint8_t> table = makeNative(PQ20, vocabulary, hidden, rng);
        const std::vector<uint32_t> tokens{3, 0, 6, 3};
        std::vector<float> fp32(tokens.size() * hidden);
        std::vector<double> fp64(fp32.size());
        std::vector<float> row(hidden);
        for (size_t t = 0; t < tokens.size(); ++t) {
          rowValues(PQ20, table.data() + size_t{tokens[t]} * rowBytes(PQ20, hidden), hidden, row.data());
          for (uint32_t b = 0; b < hidden; b += kBlock) {
            float *f = fp32.data() + t * hidden + b;
            double *d = fp64.data() + t * hidden + b;
            for (uint32_t i = 0; i < kBlock; ++i) f[i] = row[b + i], d[i] = row[b + i];
            butterflies(f);
            butterflies(d);
            for (uint32_t i = 0; i < kBlock; ++i)
              f[i] = f[i] * (1.0f / 32.0f) * float(signs[b + i]), d[i] = d[i] / 32.0 * signs[b + i];
          }
        }
        const MetalBuffer tokenBuffer = upload(backend, tokens), rows = upload(backend, table);
        const MetalBuffer out = backend.allocateBuffer(fp32.size() * 2);
        CommandGraph graph;
        graph.add("gguf_embed_rotated_pq20", {tokenBuffer, rows, signBuffer, out},
                  GgufEmbedParams{uint32_t(tokens.size()), vocabulary, hidden}, {hidden / kBlock, uint32_t(tokens.size()), 1},
                  {GGUF_ROTATION_THREADS, 1, 1});
        static_cast<void>(backend.submitCommand(graph.dispatches()));
        failures += check("gguf_embed_rotated_pq20 D (H r)", static_cast<const uint16_t *>(out.contents()), fp32, fp64);
      }
    } catch (const std::exception &e) {
      std::cerr << "gguf-rotation: FAIL: " << e.what() << '\n';
      return 1;
    }
    std::cout << (failures ? "GGUF rotation tests FAILED" : "GGUF rotation tests passed") << '\n';
    return failures ? 1 : 0;
  }
}
