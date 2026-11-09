// Prism ML's input rotation (kernels/shared/gguf_rotation.metal): gguf_rotate's H (D x) and the rotated PQ2_0
// token gather's D (H r) are bitwise the bf16 rounding of the fp32 butterflies in source order, which lie within
// one bf16 step of the fp64 transform. gguf_rotate_int8 writes those bf16 values' codes by the int8-activation rule
// (kernels/common/int8_activations.h; LinearNumerics.hpp's int8Block), each at its stage position
// (gguf_decode_int8_position), and each block's scale, bit for bit, the NaN bits 0x7fc00000 where a block holds a
// NaN or an infinity, within gguf_rotation_int8_bytes of its storage rows and nothing past its rows: over blocks of
// maxima from 2^-3 to 2^3, and a block whose halves round to even.
//   gguf-rotation <splash.metallib>
#include "GgufFormatReference.hpp"
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"

#include <algorithm>
#include <bit>
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
using splash::test::Int8Block;
using splash::test::int8Block;
using splash::test::int8Code;
using splash::test::sharedBuffer;

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

template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = sharedBuffer(backend, values.size() * sizeof(T));
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
            rotationButterflies(f);
            rotationButterflies(d);
            for (uint32_t i = 0; i < kBlock; ++i) f[i] *= kRotationScale, d[i] *= kRotationScale;
          }
        const MetalBuffer in = upload(backend, input), out = sharedBuffer(backend, input.size() * 2);
        CommandGraph graph;
        graph.add("gguf_rotate", {in, signBuffer, out}, GgufRotationParams{width}, {width / kBlock, rows, 1},
                  {GGUF_ROTATION_THREADS, 1, 1});
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
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
            rotationButterflies(f);
            rotationButterflies(d);
            for (uint32_t i = 0; i < kBlock; ++i)
              f[i] = f[i] * kRotationScale * float(signs[b + i]), d[i] = d[i] * kRotationScale * signs[b + i];
          }
        }
        const MetalBuffer tokenBuffer = upload(backend, tokens), rows = upload(backend, table);
        const MetalBuffer out = sharedBuffer(backend, fp32.size() * 2);
        CommandGraph graph;
        graph.add("gguf_embed_rotated_pq20", {tokenBuffer, rows, signBuffer, out},
                  GgufEmbedParams{uint32_t(tokens.size()), vocabulary, hidden}, {hidden / kBlock, uint32_t(tokens.size()), 1},
                  {GGUF_ROTATION_THREADS, 1, 1});
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        failures += check("gguf_embed_rotated_pq20 D (H r)", static_cast<const uint16_t *>(out.contents()), fp32, fp64);
      }

      {  // gguf_rotate_int8: 5 rows of three blocks in the storage of 8 (a tile's height), block b of row r drawn at
         // 2^((r + 3b) mod 7 - 3), so that a scale or code taken from another block shows; a block that rotates to
         // 63.5, 0.25 and -0.25 (the inputs D (H y) / 32 of that y are bf16 values, its butterflies exact), whose
         // halves code to 0 by rounding to even, not to 1 and -1; a NaN in one block and an infinity in another row's.
        constexpr uint32_t rows = 5, storage = 8, width = 3 * kBlock, blocks = width / kBlock;
        constexpr uint32_t kNaNRow = 1, kNaNBlock = 1, kInfiniteRow = 3, kInfiniteBlock = 2, kTieRow = 2, kTieBlock = 1;
        constexpr uint8_t kFill = 0xA5;
        constexpr uint64_t kGuard = 256;
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<uint16_t> input(rows * width);
        for (uint32_t r = 0; r < rows; ++r)
          for (uint32_t k = 0; k < width; ++k)
            input[r * width + k] = floatToBf16(std::ldexp(normal(rng), int((r + 3 * (k / kBlock)) % 7) - 3));
        {
          std::vector<double> y(kBlock, 0.0);
          y[9] = 63.5, y[300] = 0.25, y[777] = -0.25;
          rotationButterflies(y.data());
          for (uint32_t i = 0; i < kBlock; ++i) {
            const double x = y[i] / 32 * signs[kTieBlock * kBlock + i];
            input[kTieRow * width + kTieBlock * kBlock + i] = floatToBf16(float(x));
            if (double(bf16ToFloat(floatToBf16(float(x)))) != x) throw std::logic_error("a tie input is no bf16 value");
          }
        }
        input[kNaNRow * width + kNaNBlock * kBlock + 5] = 0x7FC0;
        input[kInfiniteRow * width + kInfiniteBlock * kBlock + 700] = 0x7F80;
        const MetalBuffer in = upload(backend, input);
        const uint64_t bytes = gguf_rotation_int8_bytes(width, storage);
        const MetalBuffer out = sharedBuffer(backend, bytes + kGuard);
        std::memset(out.contents(), kFill, bytes + kGuard);
        CommandGraph graph;
        graph.add("gguf_rotate_int8", {in, signBuffer, backend.view(out, 0, bytes)},
                  GgufRotationInt8Params{width, storage}, {blocks, rows, 1}, {GGUF_ROTATION_THREADS, 1, 1});
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        const uint8_t *written = static_cast<const uint8_t *>(out.contents());
        // Input k's code of row r, at its stage position, and block b's scale bits of row r.
        const auto codeOf = [&](uint32_t r, uint32_t k) {
          return int8_t(written[uint64_t{r} * width + (k & ~31u) + gguf_decode_int8_position(k & 31u)]);
        };
        const auto scaleOf = [&](uint32_t r, uint32_t b) {
          uint32_t bits;
          std::memcpy(&bits, written + uint64_t{storage} * width + (r * blocks + b) * sizeof(float), sizeof bits);
          return bits;
        };
        std::vector<float> rotated(kBlock);
        size_t wrongCodes = 0, wrongScales = 0;
        for (uint32_t r = 0; r < rows; ++r)
          for (uint32_t b = 0; b < blocks; ++b) {
            if ((r == kNaNRow && b == kNaNBlock) || (r == kInfiniteRow && b == kInfiniteBlock)) {
              wrongScales += scaleOf(r, b) != 0x7fc00000u;
              continue;
            }
            for (uint32_t i = 0; i < kBlock; ++i)
              rotated[i] = bf16ToFloat(input[r * width + b * kBlock + i]) * float(signs[b * kBlock + i]);
            rotationButterflies(rotated.data());
            float largest = 0.0f;
            for (float &v : rotated) {
              v = bf16ToFloat(floatToBf16(v * kRotationScale));
              largest = std::max(largest, std::fabs(v));
            }
            const Int8Block block = int8Block(largest);
            wrongScales += scaleOf(r, b) != std::bit_cast<uint32_t>(block.scale);
            for (uint32_t i = 0; i < kBlock; ++i)
              wrongCodes += codeOf(r, b * kBlock + i) != int8Code(rotated[i], block);
          }
        // The tie block: scale 63.5 / 127 = 0.5, codes 127 and, for 0.25 / 0.5 and -0.25 / 0.5, 0.
        const uint32_t tie = kTieBlock * kBlock;
        const bool tied = scaleOf(kTieRow, kTieBlock) == std::bit_cast<uint32_t>(0.5f) &&
                          codeOf(kTieRow, tie + 9) == 127 && codeOf(kTieRow, tie + 300) == 0 &&
                          codeOf(kTieRow, tie + 777) == 0;
        // The storage rows past the five, and the guard past the scales, as they were.
        const auto unwritten = [&](uint64_t from, uint64_t to) {
          return std::all_of(written + from, written + to, [&](uint8_t v) { return v == kFill; });
        };
        const bool kept = unwritten(uint64_t{rows} * width, uint64_t{storage} * width) &&
                          unwritten(uint64_t{storage} * width + uint64_t{rows} * blocks * 4, bytes + kGuard);
        const bool ok = !wrongCodes && !wrongScales && tied && kept;
        std::cout << "gguf_rotate_int8: " << wrongCodes << " codes and " << wrongScales << " of " << rows * blocks
                  << " scales differ from the int8-activation rule's (NaN bits where a NaN or an infinity is), "
                  << "halves " << (tied ? "rounded to even" : "not rounded to even") << ", "
                  << (kept ? "no write" : "a write") << " past its rows " << (ok ? "ok" : "FAIL") << '\n';
        failures += ok ? 0 : 1;
      }
    } catch (const std::exception &e) {
      std::cerr << "gguf-rotation: FAIL: " << e.what() << '\n';
      return 1;
    }
    std::cout << (failures ? "GGUF rotation tests FAILED" : "GGUF rotation tests passed") << '\n';
    return failures ? 1 : 0;
  }
}
