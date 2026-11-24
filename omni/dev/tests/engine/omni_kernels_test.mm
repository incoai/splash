// The Omni kernels (runtime/metal/kernels/omni) against fp64 references:
// dense bf16 projections with every epilogue, the row norms and
// activations, the token gathers, code sampling, and the paged decoder
// attention (prepare, prefill, decode) and dense encoder attention. Packed
// bf16 weights unpack to their bits, and project bit for bit as the bf16
// weights do. argv[1] names the production metallib.

#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Omni.h"
#include "ops/Omni.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace splash::ops::omni;
using namespace splash::ops::tuning;
using namespace splash::test;

namespace {

std::mt19937 rng(7);
float uniform(float low, float high) { return std::uniform_real_distribution<float>(low, high)(rng); }

metal::MetalBuffer bf16Buffer(metal::MetalBackend &backend, const std::vector<float> &values, size_t capacity = 0) {
  const size_t count = std::max(capacity, values.size());
  metal::MetalBuffer buffer = sharedBuffer(backend, count * 2);
  auto *bits = static_cast<uint16_t *>(buffer.contents());
  std::memset(bits, 0, count * 2);
  for (size_t i = 0; i < values.size(); ++i) bits[i] = floatToBf16(values[i]);
  return buffer;
}
metal::MetalBuffer f32Buffer(metal::MetalBackend &backend, const std::vector<float> &values, size_t capacity = 0) {
  const size_t count = std::max(capacity, values.size());
  metal::MetalBuffer buffer = sharedBuffer(backend, count * 4);
  std::memset(buffer.contents(), 0, count * 4);
  std::memcpy(buffer.contents(), values.data(), values.size() * 4);
  return buffer;
}
metal::MetalBuffer u32Buffer(metal::MetalBackend &backend, const std::vector<uint32_t> &values, size_t capacity = 0) {
  const size_t count = std::max(capacity, values.size());
  metal::MetalBuffer buffer = sharedBuffer(backend, count * 4);
  std::memset(buffer.contents(), 0, count * 4);
  std::memcpy(buffer.contents(), values.data(), values.size() * 4);
  return buffer;
}
template <class T> metal::MetalBuffer structBuffer(metal::MetalBackend &backend, const std::vector<T> &values) {
  metal::MetalBuffer buffer = sharedBuffer(backend, std::max<size_t>(1, values.size()) * sizeof(T));
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}
std::vector<float> randomValues(size_t count, float scale) {
  std::vector<float> values(count);
  for (float &v : values) v = uniform(-scale, scale);
  return values;
}
// The values as the bf16 buffer holds them.
std::vector<float> roundedBf16(const std::vector<float> &values) {
  std::vector<float> rounded(values.size());
  for (size_t i = 0; i < values.size(); ++i) rounded[i] = bf16ToFloat(floatToBf16(values[i]));
  return rounded;
}
float bf16At(const metal::MetalBuffer &buffer, size_t index) {
  return bf16ToFloat(static_cast<const uint16_t *>(buffer.contents())[index]);
}
float f32At(const metal::MetalBuffer &buffer, size_t index) { return static_cast<const float *>(buffer.contents())[index]; }

void run(metal::MetalBackend &backend, metal::CommandGraph &graph) {
  metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
  (void)ticket.wait();
}

void close(double got, double expected, double relative, double absolute, const std::string &what) {
  const double tolerance = absolute + relative * std::fabs(expected);
  if (std::isnan(got) || std::fabs(got - expected) > tolerance)
    throw std::runtime_error(what + ": got " + std::to_string(got) + ", expected " + std::to_string(expected));
}

double gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }
double silu(double x) { return x / (1.0 + std::exp(-x)); }

// ---------------------------------------------------------------- dense GEMM

// The split scratch of a projection of up to `rows` rows (none when the
// policy splits none of them).
std::pair<metal::MetalBuffer, metal::MetalBuffer> splitScratch(metal::MetalBackend &backend, uint32_t n, uint32_t k,
                                                               uint32_t rows) {
  const DenseSplitScratch size = denseSplitScratch(n, k, rows);
  if (!size.partialsBytes) return {};
  metal::MetalBuffer partials = sharedBuffer(backend, size.partialsBytes);
  metal::MetalBuffer counters = sharedBuffer(backend, size.countersBytes);
  std::memset(partials.contents(), 0xff, size.partialsBytes);
  std::memset(counters.contents(), 0, size.countersBytes);
  return {partials, counters};
}

// Every split counter back at zero after the dispatches.
void requireCountersReset(const metal::MetalBuffer &counters, const std::string &what) {
  if (!counters) return;
  const auto *count = static_cast<const uint32_t *>(counters.contents());
  for (uint64_t i = 0; i < counters.sizeBytes() / 4; ++i)
    if (count[i]) throw std::runtime_error(what + ": split counter " + std::to_string(i) + " left at " + std::to_string(count[i]));
}

// The 32x64 MPP tile, which ran every step of five rows or more before the
// MMA tiles, on rows padded to 32: the MMA tiles' outputs must agree with
// it to the outputs' rounding.
void compareWithMppTile(metal::MetalBackend &backend, const GemmBuffers &buffers, const DenseWeights &up,
                        const DenseWeights *gate, uint32_t rows, const GemmOptions &options, const std::string &what) {
  const uint32_t n = up.outputSize, k = up.inputSize;
  const uint64_t element = elementBytes(options.destination);
  metal::MetalBuffer input = sharedBuffer(backend, uint64_t{32} * k * 2);
  std::memset(input.contents(), 0, input.sizeBytes());
  std::memcpy(input.contents(), buffers.input.contents(), uint64_t{rows} * k * 2);
  metal::MetalBuffer output = sharedBuffer(backend, uint64_t{rows} * n * element);
  OmniGemmParams p{};
  p.rows = rows;
  p.output_size = n;
  p.input_size = k;
  p.flags = (up.bias ? OMNI_GEMM_FLAG_BIAS : 0u) | (options.residual ? OMNI_GEMM_FLAG_RESIDUAL : 0u) |
            (gate ? OMNI_GEMM_FLAG_GATE_UP : 0u);
  p.activation = static_cast<uint32_t>(options.activation);
  p.input_stride = k;
  p.output_stride = n;
  const metal::MetalBuffer bias = up.bias ? up.bias : up.weight;
  metal::CommandGraph graph;
  graph.add(std::string("omni_gemm_bf16_m32n64s8") + (gate ? "_gu" : "") +
                (options.destination == FloatOutput::Float32 ? "_f32" : ""),
            {input, up.weight, gate ? gate->weight : up.weight, bias, gate && gate->bias ? gate->bias : bias,
             options.residual ? buffers.residual : output, output},
            p, {1, n / 64, 1});
  run(backend, graph);
  for (uint64_t i = 0; i < uint64_t{rows} * n; ++i) {
    const double mma = options.destination == FloatOutput::Float32 ? f32At(buffers.output, i) : bf16At(buffers.output, i);
    const double mpp = options.destination == FloatOutput::Float32 ? f32At(output, i) : bf16At(output, i);
    // A bf16 step, or fp32 to 1e-5, beyond the fp32 sums' reassociation
    // (1e-4 at these sizes, which a cancelling residual can expose).
    const double tolerance = options.destination == FloatOutput::Float32 ? 1e-5 * std::fabs(mpp) + 1e-4
                                                                         : std::ldexp(std::fabs(mpp), -7) + 1e-4;
    if (!(std::fabs(mma - mpp) <= tolerance))
      throw std::runtime_error(what + ": [" + std::to_string(i / n) + "," + std::to_string(i % n) + "] " +
                               std::to_string(mma) + " against the MPP tile's " + std::to_string(mpp));
  }
}

void testDense(metal::MetalBackend &backend, uint32_t rows, uint32_t n, uint32_t k, bool bias, Activation act,
               bool residual, bool gateUp, FloatOutput destination, bool split = false) {
  const std::string what = "dense rows=" + std::to_string(rows) + " n=" + std::to_string(n) + " k=" +
                           std::to_string(k) + (bias ? " bias" : "") + (residual ? " residual" : "") +
                           (gateUp ? " gateup" : "") + " act=" + std::to_string(int(act)) +
                           (destination == FloatOutput::Float32 ? " f32" : " bf16") + (split ? " split" : "");
  const uint32_t padded = densePaddedRows(rows);
  const std::vector<float> x = roundedBf16(randomValues(size_t(padded) * k, 1.0F));
  const std::vector<float> w = roundedBf16(randomValues(size_t(n) * k, 0.1F));
  const std::vector<float> g = roundedBf16(randomValues(size_t(n) * k, 0.1F));
  const std::vector<float> b = randomValues(n, 0.5F);
  const std::vector<float> gb = randomValues(n, 0.5F);
  std::vector<float> r = randomValues(size_t(rows) * n, 1.0F);
  if (destination == FloatOutput::BFloat16) r = roundedBf16(r);
  DenseWeights up{bf16Buffer(backend, w), bias ? f32Buffer(backend, b) : metal::MetalBuffer{}, n, k};
  DenseWeights gate{bf16Buffer(backend, g), bias ? f32Buffer(backend, gb) : metal::MetalBuffer{}, n, k};
  const uint64_t element = elementBytes(destination);
  GemmBuffers buffers{bf16Buffer(backend, x), sharedBuffer(backend, uint64_t{rows} * n * element),
                      destination == FloatOutput::Float32 ? f32Buffer(backend, r) : bf16Buffer(backend, r)};
  if (split) std::tie(buffers.partials, buffers.counters) = splitScratch(backend, n, k, rows);
  metal::CommandGraph graph;
  GemmOptions options{act, residual, destination};
  // Twice, so the second dispatch finds the counters the first returned.
  for (int repeat = 0; repeat < 2; ++repeat) {
    if (gateUp) DenseLinear::addGateUp(graph, gate, up, buffers, rows, options);
    else DenseLinear::add(graph, up, buffers, rows, options);
  }
  run(backend, graph);
  requireCountersReset(buffers.counters, what);
  if (rows >= 2 && rows <= 16) compareWithMppTile(backend, buffers, up, gateUp ? &gate : nullptr, rows, options, what);
  for (uint32_t m = 0; m < rows; ++m)
    for (uint32_t col = 0; col < n; ++col) {
      double sum = 0.0, gated = 0.0;
      for (uint32_t i = 0; i < k; ++i) {
        sum += double(x[size_t(m) * k + i]) * w[size_t(col) * k + i];
        gated += double(x[size_t(m) * k + i]) * g[size_t(col) * k + i];
      }
      if (bias) {
        sum += b[col];
        gated += gb[col];
      }
      double value = gateUp ? silu(gated) * sum : act == Activation::Gelu ? gelu(sum) : act == Activation::Silu ? silu(sum) : sum;
      if (residual) value += r[size_t(m) * n + col];
      const double got = destination == FloatOutput::Float32 ? f32At(buffers.output, size_t(m) * n + col)
                                                             : bf16At(buffers.output, size_t(m) * n + col);
      close(got, value, destination == FloatOutput::Float32 ? 2e-3 : 1.2e-2, 2e-2, what + " [" + std::to_string(m) + "," + std::to_string(col) + "]");
    }
}

// Each row of a step projects to the same bits at every row count the
// policy calls the same arithmetic (DenseLinear::sameRowArithmetic): the MMA
// tiles at one split count, whatever the rows, row fragments, column
// fragments and weight copy; the GEMV kernels at one to four rows.
void testDenseRowArithmetic(metal::MetalBackend &backend, uint32_t n, uint32_t k, bool packedCopy) {
  const std::string what = "dense row arithmetic n=" + std::to_string(n) + " k=" + std::to_string(k) +
                           (packedCopy ? " packed" : "");
  const std::vector<float> x = randomValues(size_t(16) * k, 1.0F);
  std::vector<float> wv = randomValues(size_t(n) * k, 0.1F);
  DenseWeights w{bf16Buffer(backend, wv), {}, n, k};
  if (packedCopy) {
    std::vector<uint16_t> bits(wv.size());
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = floatToBf16(wv[i]);
    const uint32_t escapes = packedEscapeBlocks(bits, k);
    std::vector<uint8_t> bytes(packedDenseBytes(n, k, escapes));
    packDenseRows(bits, n, k, 0, n, 0, bytes);
    w.packed = sharedBuffer(backend, bytes.size());
    std::memcpy(w.packed.contents(), bytes.data(), bytes.size());
  }
  const metal::MetalBuffer input = bf16Buffer(backend, x);
  const auto [partials, counters] = splitScratch(backend, n, k, 16);
  std::vector<metal::MetalBuffer> outputs(17);
  metal::CommandGraph graph;
  for (uint32_t rows = 1; rows <= 16; ++rows) {
    outputs[rows] = sharedBuffer(backend, uint64_t{rows} * n * 2);
    DenseLinear::add(graph, w, {input, outputs[rows], {}, 0, 0, partials, counters}, rows);
  }
  run(backend, graph);
  requireCountersReset(counters, what);
  uint32_t matched = 0;
  for (uint32_t rows = 1; rows <= 16; ++rows)
    for (uint32_t other = rows + 1; other <= 16; ++other) {
      if (!DenseLinear::sameRowArithmetic(w, rows, other)) continue;
      ++matched;
      if (std::memcmp(outputs[rows].contents(), outputs[other].contents(), uint64_t{rows} * n * 2))
        throw std::runtime_error(what + ": rows of a " + std::to_string(rows) + "-row step differ at " +
                                 std::to_string(other) + " rows");
    }
  if (!matched) throw std::runtime_error(what + ": no row counts share an arithmetic");
}

// Every MPP tile (gemm_bf16.metal omni_gemm_bf16_*) of a projection writes
// the same bytes, which the policy relies on: it picks the tile by the rows
// (and the audio tower's packed clips take other tiles than a clip alone),
// and the rows match the fp64 reference.
void testMppTilesAgree(metal::MetalBackend &backend, uint32_t rows, uint32_t n, uint32_t k, bool bias, Activation act,
                       bool residual, bool gateUp, FloatOutput destination) {
  const std::string what = "mpp tiles rows=" + std::to_string(rows) + " n=" + std::to_string(n) + " k=" +
                           std::to_string(k) + (bias ? " bias" : "") + (residual ? " residual" : "") +
                           (gateUp ? " gateup" : "") + " act=" + std::to_string(int(act)) +
                           (destination == FloatOutput::Float32 ? " f32" : " bf16");
  const uint32_t padded = (rows + 63) / 64 * 64;
  const std::vector<float> x = roundedBf16(randomValues(size_t(padded) * k, 1.0F));
  const std::vector<float> w = roundedBf16(randomValues(size_t(n) * k, 0.1F));
  const std::vector<float> g = roundedBf16(randomValues(size_t(n) * k, 0.1F));
  const std::vector<float> b = randomValues(n, 0.5F);
  const std::vector<float> gb = randomValues(n, 0.5F);
  std::vector<float> r = randomValues(size_t(rows) * n, 1.0F);
  if (destination == FloatOutput::BFloat16) r = roundedBf16(r);
  const bool f32 = destination == FloatOutput::Float32;
  const uint64_t bytes = uint64_t{rows} * n * (f32 ? 4 : 2);
  const metal::MetalBuffer input = bf16Buffer(backend, x), weight = bf16Buffer(backend, w),
                           gate = bf16Buffer(backend, g), biases = f32Buffer(backend, b),
                           gateBiases = f32Buffer(backend, gb),
                           residualRows = f32 ? f32Buffer(backend, r) : bf16Buffer(backend, r);
  OmniGemmParams p{};
  p.rows = rows;
  p.output_size = n;
  p.input_size = k;
  p.flags = (bias ? OMNI_GEMM_FLAG_BIAS : 0u) | (residual ? OMNI_GEMM_FLAG_RESIDUAL : 0u) |
            (gateUp ? OMNI_GEMM_FLAG_GATE_UP : 0u);
  p.activation = static_cast<uint32_t>(act);
  p.input_stride = k;
  p.output_stride = n;
  const uint32_t tiles[][3] = {{32, 64, 8}, {32, 128, 4}, {64, 64, 4}, {64, 64, 8}, {64, 128, 4}, {64, 128, 8}};
  std::vector<uint8_t> first;
  for (const auto &t : tiles) {
    if (n % t[1]) continue;
    const std::string name = "omni_gemm_bf16_m" + std::to_string(t[0]) + "n" + std::to_string(t[1]) + "s" +
                             std::to_string(t[2]) + (gateUp ? "_gu" : "") + (f32 ? "_f32" : "");
    metal::MetalBuffer output = sharedBuffer(backend, bytes);
    std::memset(output.contents(), 0xff, bytes);
    metal::CommandGraph graph;
    graph.add(name, {input, weight, gate, biases, gateBiases, residual ? residualRows : output, output}, p,
              {(rows + t[0] - 1) / t[0], n / t[1], 1}, {t[2] * 32, 1, 1});
    run(backend, graph);
    const auto *out = static_cast<const uint8_t *>(output.contents());
    if (first.empty()) {
      first.assign(out, out + bytes);
      for (uint32_t m = 0; m < rows; ++m)
        for (uint32_t col = 0; col < n; ++col) {
          double sum = bias ? b[col] : 0.0, gated = bias ? gb[col] : 0.0;
          for (uint32_t i = 0; i < k; ++i) {
            sum += double(x[size_t(m) * k + i]) * w[size_t(col) * k + i];
            gated += double(x[size_t(m) * k + i]) * g[size_t(col) * k + i];
          }
          double value = gateUp ? silu(gated) * sum : act == Activation::Gelu ? gelu(sum) : sum;
          if (residual) value += r[size_t(m) * n + col];
          const double got = f32 ? f32At(output, size_t(m) * n + col) : bf16At(output, size_t(m) * n + col);
          close(got, value, f32 ? 2e-3 : 1.2e-2, 2e-2, what + " " + name + " [" + std::to_string(m) + "," +
                                                         std::to_string(col) + "]");
        }
    } else if (std::memcmp(first.data(), out, bytes)) {
      throw std::runtime_error(what + ": " + name + " differs from the first tile");
    }
  }
}

void testDenseAll(metal::MetalBackend &backend) {
  for (uint32_t rows : {17u, 40u, 100u, 130u}) {
    testMppTilesAgree(backend, rows, 256, 2048, true, Activation::Gelu, false, false, FloatOutput::BFloat16);
    testMppTilesAgree(backend, rows, 384, 512, true, Activation::None, true, false, FloatOutput::Float32);
    testMppTilesAgree(backend, rows, 256, 1024, false, Activation::None, true, false, FloatOutput::BFloat16);
    testMppTilesAgree(backend, rows, 256, 512, true, Activation::None, false, true, FloatOutput::BFloat16);
    testMppTilesAgree(backend, rows, 128, 256, false, Activation::None, false, true, FloatOutput::Float32);
  }
  // Wide enough for the policy's 64x64 and 64x128 tiles.
  for (uint32_t rows : {33u, 100u, 200u}) {
    testDense(backend, rows, 3072, 1024, true, Activation::Gelu, false, false, FloatOutput::BFloat16);
    testDense(backend, rows, 2048, 2048, false, Activation::None, true, true, FloatOutput::BFloat16);
  }
  for (uint32_t rows : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 12u, 15u, 16u, 17u, 20u, 64u, 100u})
    for (bool split : {false, true}) {
      testDense(backend, rows, 128, 256, false, Activation::None, false, false, FloatOutput::BFloat16, split);
      testDense(backend, rows, 256, 192, true, Activation::Gelu, false, false, FloatOutput::BFloat16, split);
      testDense(backend, rows, 256, 128, true, Activation::Silu, true, false, FloatOutput::Float32, split);
      testDense(backend, rows, 128, 256, false, Activation::None, true, true, FloatOutput::BFloat16, split);
      testDense(backend, rows, 384, 128, true, Activation::None, false, true, FloatOutput::BFloat16, split);
      testDense(backend, rows, 128, 512, false, Activation::None, false, true, FloatOutput::Float32, split);
      // K of 4096 and more splits however many threadgroups.
      if (rows <= 16) testDense(backend, rows, 4096, 4096, false, Activation::None, true, false, FloatOutput::BFloat16, split);
    }
  for (bool packed : {false, true}) {
    testDenseRowArithmetic(backend, 256, 512, packed);
    testDenseRowArithmetic(backend, 4096, 256, packed);
    testDenseRowArithmetic(backend, 2048, 4096, packed);
  }
  // The policy's split counts: unsplit without scratch, and a split for
  // few threadgroups or K of 4096.
  const DenseWeights shape{{}, {}, 2048, 6144};
  if (denseSplitScratch(2048, 6144, 8).partialsBytes != uint64_t{2} * 2 * 8 * 2048 * 4 ||
      denseSplitScratch(4096, 2048, 8).partialsBytes || denseSplitScratch(2048, 6144, 1).partialsBytes)
    throw std::runtime_error("dense split scratch sizes");
  (void)shape;
  std::cout << "dense projections ok\n";
}

// ---------------------------------------------------------------- packed bf16

uint16_t bf16Bits(uint32_t sign, uint32_t exponent, uint32_t mantissa) {
  return static_cast<uint16_t>(sign << 15 | exponent << 7 | (mantissa & 0x7fu));
}

// Weights of [n][k] whose 64-wide blocks cycle through the cases the packing
// meets: checkpoint-like values, zeros of both signs, subnormals alone and
// beside small exponents (all with base 0), zeros beside normal values and
// spans of more than 16 exponents (escapes), spans of exactly 16 and 17,
// exponents up to 254, and with `special` infinities and NaN payloads.
std::vector<uint16_t> adversarialWeights(uint32_t n, uint32_t k, bool special, uint32_t *escapes) {
  std::vector<uint16_t> bits(uint64_t{n} * k);
  std::normal_distribution<float> normal(0.0F, 0.02F);
  std::uniform_int_distribution<uint32_t> any(0, 0xffff);
  *escapes = 0;
  for (uint64_t b = 0; b < bits.size() / 64; ++b) {
    uint16_t *block = bits.data() + b * 64;
    const uint32_t kind = b % (special ? 10 : 9);
    for (uint32_t i = 0; i < 64; ++i) {
      const uint32_t sign = any(rng) & 1, mantissa = any(rng);
      switch (kind) {
      case 0: {
        const float value = normal(rng);
        block[i] = floatToBf16(std::fabs(value) < 1e-4F ? 1e-4F : value);
        break;
      }
      case 1: block[i] = bf16Bits(sign, 0, 0); break;
      case 2: block[i] = bf16Bits(sign, 0, i % 3 ? mantissa : 0); break;
      case 3: block[i] = bf16Bits(sign, i % 2 ? 0 : 1 + any(rng) % 15, mantissa); break;
      case 4: block[i] = bf16Bits(sign, i % 5 ? 120 + any(rng) % 4 : 0, mantissa); break;
      case 5: block[i] = bf16Bits(sign, i == 0 ? 100 : i == 1 ? 139 : 100 + any(rng) % 40, mantissa); break;
      case 6: block[i] = bf16Bits(sign, i == 3 ? 110 : i == 7 ? 125 : 110 + any(rng) % 16, mantissa); break;
      case 7: block[i] = bf16Bits(sign, i == 3 ? 110 : i == 9 ? 126 : 110 + any(rng) % 16, mantissa); break;
      case 8: block[i] = bf16Bits(sign, 239 + any(rng) % 16, mantissa); break;
      default: block[i] = bf16Bits(sign, i % 4 ? 125 : 255, i % 8 ? mantissa : 0); break;
      }
    }
    *escapes += kind == 4 || kind == 5 || kind == 7 || kind == 9;
  }
  return bits;
}

std::vector<uint8_t> packAll(const std::vector<uint16_t> &weights, uint32_t n, uint32_t k) {
  const uint32_t escapes = packedEscapeBlocks(weights, k);
  std::vector<uint8_t> packed(packedDenseBytes(n, k, escapes));
  packDenseRows(weights, n, k, 0, n, 0, packed);
  return packed;
}

template <class Call> void expectThrow(Call call, const std::string &what) {
  try {
    call();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(what + " was accepted");
}

void testPackedRoundTrip() {
  for (const auto [n, k] : {std::pair{64u, 64u}, std::pair{128u, 256u}, std::pair{192u, 640u}})
    for (bool special : {false, true}) {
      const std::string what = "packed " + std::to_string(n) + "x" + std::to_string(k);
      uint32_t expected = 0;
      const std::vector<uint16_t> weights = adversarialWeights(n, k, special, &expected);
      const std::vector<uint8_t> packed = packAll(weights, n, k);
      if (packed.size() != packedDenseBytes(n, k, expected)) throw std::runtime_error(what + ": escape blocks");
      if (packed.size() != uint64_t{n} * k * 25 / 16 + uint64_t{expected} * OMNI_PACKED_ESCAPE_BYTES)
        throw std::runtime_error(what + ": packed size");
      std::vector<uint16_t> back(weights.size());
      unpackDense(packed, n, k, back);
      if (back != weights) throw std::runtime_error(what + " does not unpack to its bits");
      // Row ranges with their own escape numbering pack the same bytes.
      std::vector<uint8_t> ranged(packed.size(), 0xa5);
      uint32_t escape = 0;
      for (uint32_t first = 0; first < n; first += 64) {
        packDenseRows(weights, n, k, first, 64, escape, ranged);
        escape += packedEscapeBlocks(std::span(weights).subspan(uint64_t{first} * k, uint64_t{64} * k), k);
      }
      if (ranged != packed) throw std::runtime_error(what + ": row ranges pack other bytes");
    }
  uint32_t escapes = 0;
  const std::vector<uint16_t> weights = adversarialWeights(64, 128, true, &escapes);
  std::vector<uint8_t> small(packedDenseBytes(64, 128, escapes - 1));
  expectThrow([&] { packDenseRows(weights, 64, 128, 0, 64, 0, small); }, "a buffer short of an escape record");
  expectThrow([&] { (void)packedDenseBytes(64, 128, uint64_t{1} << 24); }, "escape records past 24 bits");
  expectThrow([&] { (void)packedEscapeBlocks(weights, 96); }, "rows of 96");
  expectThrow([&] { packDenseRows(weights, 64, 128, 32, 64, 0, small); }, "rows past the weight");
  std::cout << "packed bf16 round trips ok\n";
}

// DenseLinear with the packed copy against the bf16 weights alone, bit for
// bit, over the policy's packed kernels (rows 1-2, gate/up 3-4) and the rows
// it leaves to the bf16 kernels.
void testPackedDense(metal::MetalBackend &backend, uint32_t rows, uint32_t n, uint32_t k, bool bias, Activation act,
                     bool residual, bool gateUp, FloatOutput destination, bool special, bool split) {
  const std::string what = "packed dense rows=" + std::to_string(rows) + " n=" + std::to_string(n) + " k=" +
                           std::to_string(k) + (bias ? " bias" : "") + (residual ? " residual" : "") +
                           (gateUp ? " gateup" : "") + " act=" + std::to_string(int(act)) +
                           (destination == FloatOutput::Float32 ? " f32" : " bf16") + (special ? " inf/nan" : "") +
                           (split ? " split" : "");
  const uint32_t padded = densePaddedRows(rows);
  uint32_t escapes = 0;
  const std::vector<uint16_t> w = adversarialWeights(n, k, special, &escapes);
  const std::vector<uint16_t> g = adversarialWeights(n, k, special, &escapes);
  const auto weightBuffer = [&](const std::vector<uint16_t> &bits) {
    metal::MetalBuffer buffer = sharedBuffer(backend, bits.size() * 2);
    std::memcpy(buffer.contents(), bits.data(), bits.size() * 2);
    return buffer;
  };
  const auto packedBuffer = [&](const std::vector<uint16_t> &bits) {
    const std::vector<uint8_t> packed = packAll(bits, n, k);
    metal::MetalBuffer buffer = sharedBuffer(backend, packed.size());
    std::memcpy(buffer.contents(), packed.data(), packed.size());
    return buffer;
  };
  DenseWeights up{weightBuffer(w), bias ? f32Buffer(backend, randomValues(n, 0.5F)) : metal::MetalBuffer{}, n, k};
  DenseWeights gate{weightBuffer(g), bias ? f32Buffer(backend, randomValues(n, 0.5F)) : metal::MetalBuffer{}, n, k};
  DenseWeights packedUp = up, packedGate = gate;
  packedUp.packed = packedBuffer(w);
  packedGate.packed = packedBuffer(g);
  const metal::MetalBuffer x = bf16Buffer(backend, randomValues(size_t(padded) * k, 1.0F));
  const std::vector<float> r = randomValues(size_t(rows) * n, 1.0F);
  const metal::MetalBuffer residualRows = destination == FloatOutput::Float32 ? f32Buffer(backend, r) : bf16Buffer(backend, r);
  const uint64_t bytes = uint64_t{rows} * n * elementBytes(destination);
  GemmBuffers plainBuffers{x, sharedBuffer(backend, bytes), residualRows};
  GemmBuffers packedBuffers{x, sharedBuffer(backend, bytes), residualRows};
  if (split) {
    std::tie(plainBuffers.partials, plainBuffers.counters) = splitScratch(backend, n, k, rows);
    std::tie(packedBuffers.partials, packedBuffers.counters) = splitScratch(backend, n, k, rows);
  }
  metal::CommandGraph graph;
  const GemmOptions options{act, residual, destination};
  if (gateUp) {
    DenseLinear::addGateUp(graph, gate, up, plainBuffers, rows, options);
    DenseLinear::addGateUp(graph, packedGate, packedUp, packedBuffers, rows, options);
  } else {
    DenseLinear::add(graph, up, plainBuffers, rows, options);
    DenseLinear::add(graph, packedUp, packedBuffers, rows, options);
  }
  run(backend, graph);
  requireCountersReset(plainBuffers.counters, what);
  requireCountersReset(packedBuffers.counters, what);
  if (std::memcmp(plainBuffers.output.contents(), packedBuffers.output.contents(), bytes))
    throw std::runtime_error(what + ": the packed weights project other bits");
}

// One-hot rows read single weights back through the packed kernels (fp32
// out): each equals its bf16 value (zeros of either sign as zero, and
// subnormals as themselves or, flushed, as zero).
void testPackedWeightsRead(metal::MetalBackend &backend) {
  const uint32_t n = 128, k = 256;
  uint32_t escapes = 0;
  const std::vector<uint16_t> w = adversarialWeights(n, k, false, &escapes);
  const std::vector<uint8_t> bytes = packAll(w, n, k);
  metal::MetalBuffer packed = sharedBuffer(backend, bytes.size());
  std::memcpy(packed.contents(), bytes.data(), bytes.size());
  metal::MetalBuffer plain = sharedBuffer(backend, w.size() * 2);
  std::memcpy(plain.contents(), w.data(), w.size() * 2);
  const DenseWeights weights{plain, {}, n, k, packed};
  std::vector<GemmBuffers> reads;
  metal::CommandGraph graph;
  for (uint32_t at = 0; at < k; at += 2) {
    std::vector<float> x(size_t(2) * k, 0.0F);
    x[at] = 1.0F;
    x[k + at + 1] = 1.0F;
    reads.push_back({bf16Buffer(backend, x), sharedBuffer(backend, uint64_t{2} * n * 4), {}});
    DenseLinear::add(graph, weights, reads.back(), 2, {Activation::None, false, FloatOutput::Float32});
  }
  run(backend, graph);
  for (uint32_t at = 0; at < k; at += 2)
    for (uint32_t row = 0; row < 2; ++row)
      for (uint32_t col = 0; col < n; ++col) {
        const float got = f32At(reads[at / 2].output, size_t(row) * n + col);
        const float expected = bf16ToFloat(w[size_t(col) * k + at + row]);
        if (got != expected && !(got == 0.0F && std::fpclassify(expected) == FP_SUBNORMAL))
          throw std::runtime_error("packed weight [" + std::to_string(col) + "][" + std::to_string(at + row) +
                                   "] reads back as " + std::to_string(got));
      }
}

void testPackedAll(metal::MetalBackend &backend) {
  testPackedRoundTrip();
  testPackedWeightsRead(backend);
  for (bool special : {false, true})
    for (uint32_t rows : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 16u})
      for (bool split : {false, true}) {
      testPackedDense(backend, rows, 128, 256, false, Activation::None, false, false, FloatOutput::BFloat16, special, split);
      testPackedDense(backend, rows, 256, 192, true, Activation::Gelu, false, false, FloatOutput::BFloat16, special, split);
      testPackedDense(backend, rows, 256, 128, true, Activation::Silu, true, false, FloatOutput::Float32, special, split);
      testPackedDense(backend, rows, 192, 512, false, Activation::None, false, false, FloatOutput::Float32, special, split);
      testPackedDense(backend, rows, 128, 256, false, Activation::None, true, true, FloatOutput::BFloat16, special, split);
      testPackedDense(backend, rows, 384, 128, true, Activation::None, false, true, FloatOutput::BFloat16, special, split);
      testPackedDense(backend, rows, 256, 4096, false, Activation::None, true, false, FloatOutput::BFloat16, special, split);
    }
  std::cout << "packed bf16 projections ok\n";
}

// ---------------------------------------------------------------- norms

void testNorms(metal::MetalBackend &backend) {
  const uint32_t rows = 5, width = 300;
  const std::vector<float> x = randomValues(size_t(rows) * width, 2.0F);
  const std::vector<float> w = randomValues(width, 1.0F);
  const std::vector<float> b = randomValues(width, 1.0F);
  {
    metal::MetalBuffer output = sharedBuffer(backend, size_t(rows) * width * 4);
    metal::CommandGraph graph;
    Norms::addRms(graph, f32Buffer(backend, x), f32Buffer(backend, w), output, rows, width, 1e-5F, true);
    run(backend, graph);
    for (uint32_t r = 0; r < rows; ++r) {
      double sum = 0.0;
      for (uint32_t c = 0; c < width; ++c) sum += double(x[r * width + c]) * x[r * width + c];
      const double inverse = 1.0 / std::sqrt(sum / width + 1e-5);
      for (uint32_t c = 0; c < width; ++c)
        close(f32At(output, r * width + c), x[r * width + c] * inverse * w[c], 1e-4, 1e-5, "rms f32");
    }
  }
  {
    const std::vector<float> xb = roundedBf16(x), wb = roundedBf16(w), bb = roundedBf16(b);
    metal::MetalBuffer output = sharedBuffer(backend, size_t(rows) * width * 2);
    metal::CommandGraph graph;
    Norms::addRms(graph, bf16Buffer(backend, xb), bf16Buffer(backend, wb), output, rows, width, 1e-6F, false);
    Norms::addLayer(graph, f32Buffer(backend, x), bf16Buffer(backend, wb), bf16Buffer(backend, bb), output, rows,
                    width, 1e-5F, false);
    metal::MetalBuffer rmsOut = sharedBuffer(backend, size_t(rows) * width * 2);
    metal::CommandGraph rmsGraph;
    Norms::addRms(rmsGraph, bf16Buffer(backend, xb), bf16Buffer(backend, wb), rmsOut, rows, width, 1e-6F, false);
    run(backend, rmsGraph);
    run(backend, graph);
    for (uint32_t r = 0; r < rows; ++r) {
      double sum = 0.0, mean = 0.0;
      for (uint32_t c = 0; c < width; ++c) {
        sum += double(xb[r * width + c]) * xb[r * width + c];
        mean += x[r * width + c];
      }
      mean /= width;
      double variance = 0.0;
      for (uint32_t c = 0; c < width; ++c) variance += (x[r * width + c] - mean) * (x[r * width + c] - mean);
      variance /= width;
      const double inverse = 1.0 / std::sqrt(sum / width + 1e-6);
      for (uint32_t c = 0; c < width; ++c) {
        close(bf16At(rmsOut, r * width + c), xb[r * width + c] * inverse * wb[c], 1e-2, 1e-3, "rms bf16");
        close(bf16At(output, r * width + c), (x[r * width + c] - mean) / std::sqrt(variance + 1e-5) * wb[c] + bb[c],
              1e-2, 1e-2, "layer norm");
      }
    }
  }
  {
    const std::vector<float> xb = roundedBf16(x);
    metal::MetalBuffer output = sharedBuffer(backend, x.size() * 2), converted = sharedBuffer(backend, x.size() * 4);
    metal::CommandGraph graph;
    Norms::addActivate(graph, bf16Buffer(backend, xb), output, x.size(), Activation::Gelu, false);
    Norms::addConvert(graph, output, converted, x.size(), true);
    run(backend, graph);
    for (size_t i = 0; i < x.size(); ++i) {
      close(bf16At(output, i), gelu(xb[i]), 1e-2, 2e-3, "gelu bf16");
      close(f32At(converted, i), bf16At(output, i), 0.0, 0.0, "convert");
    }
  }
  std::cout << "norms ok\n";
}

// ---------------------------------------------------------------- gathers

void testEmbedding(metal::MetalBackend &backend) {
  const uint32_t width = 64, codecVocabulary = 40, predictorVocabulary = 24, rows = 3;
  const std::vector<float> codec = roundedBf16(randomValues(size_t(codecVocabulary) * width, 1.0F));
  const std::vector<float> predictor = roundedBf16(randomValues(size_t(15) * predictorVocabulary * width, 1.0F));
  const std::vector<float> trailing = roundedBf16(randomValues(size_t(6) * width, 1.0F));
  std::vector<uint32_t> codes(rows * 16);
  for (uint32_t i = 0; i < rows * 16; ++i) codes[i] = (i % 16 == 0 ? i * 7 % codecVocabulary : i * 5 % predictorVocabulary);
  const std::vector<uint32_t> trailingRows{5, 0, 3};
  Embedding::TtsTables tables{bf16Buffer(backend, codec), bf16Buffer(backend, predictor), codecVocabulary,
                              predictorVocabulary, width};
  metal::MetalBuffer output = sharedBuffer(backend, size_t(rows) * width * 2);
  metal::CommandGraph graph;
  Embedding::addTtsStepInput(graph, u32Buffer(backend, codes), tables, u32Buffer(backend, trailingRows),
                             bf16Buffer(backend, trailing), 6, output, rows);
  run(backend, graph);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < width; ++c) {
      double sum = codec[size_t(codes[r * 16]) * width + c];
      for (uint32_t g = 1; g < 16; ++g)
        sum += predictor[(size_t(g - 1) * predictorVocabulary + codes[r * 16 + g]) * width + c];
      sum += trailing[size_t(trailingRows[r]) * width + c];
      close(bf16At(output, r * width + c), sum, 1e-2, 1e-2, "tts step input");
    }
  // Prompt assembly: text row + codec + frame, and rows with parts missing.
  const std::vector<float> text = roundedBf16(randomValues(size_t(4) * width, 1.0F));
  const std::vector<float> speaker = roundedBf16(randomValues(width, 1.0F));
  std::vector<OmniPromptPosition> positions{{2, 5, -1, 0}, {-1, -1, 1, 1}, {0, -1, -1, 0}, {3, 7, 0, 1}};
  metal::MetalBuffer prompt = sharedBuffer(backend, positions.size() * width * 2);
  metal::CommandGraph promptGraph;
  Embedding::addPromptAssemble(promptGraph, structBuffer(backend, positions), bf16Buffer(backend, text), 4, tables,
                               bf16Buffer(backend, speaker), u32Buffer(backend, codes), rows, prompt,
                               static_cast<uint32_t>(positions.size()));
  run(backend, promptGraph);
  for (size_t p = 0; p < positions.size(); ++p)
    for (uint32_t c = 0; c < width; ++c) {
      double sum = 0.0;
      if (positions[p].text_row >= 0) sum += text[size_t(positions[p].text_row) * width + c];
      if (positions[p].codec >= 0) sum += codec[size_t(positions[p].codec) * width + c];
      if (positions[p].speaker) sum += speaker[c];
      if (positions[p].frame >= 0) {
        const uint32_t *frame = &codes[size_t(positions[p].frame) * 16];
        sum += codec[size_t(frame[0]) * width + c];
        for (uint32_t g = 1; g < 16; ++g) sum += predictor[(size_t(g - 1) * predictorVocabulary + frame[g]) * width + c];
      }
      close(bf16At(prompt, p * width + c), sum, 1e-2, 1e-2, "prompt assembly");
    }
  // Plain gathers.
  const std::vector<uint32_t> tokens{3, 0, 39, 1000};
  metal::MetalBuffer gathered = sharedBuffer(backend, tokens.size() * width * 2), rows2 = sharedBuffer(backend, 2 * width * 2);
  metal::CommandGraph gatherGraph;
  Embedding::addGather(gatherGraph, u32Buffer(backend, tokens), tables.codec, gathered, 4, codecVocabulary, width);
  Embedding::addGatherRows(gatherGraph, u32Buffer(backend, {3, 1}), gathered, rows2, 2, 4, width);
  run(backend, gatherGraph);
  for (size_t t = 0; t < tokens.size(); ++t)
    for (uint32_t c = 0; c < width; ++c)
      close(bf16At(gathered, t * width + c), codec[size_t(tokens[t] < codecVocabulary ? tokens[t] : 0) * width + c], 0.0, 0.0, "gather");
  for (uint32_t c = 0; c < width; ++c) {
    close(bf16At(rows2, c), bf16At(gathered, 3 * width + c), 0.0, 0.0, "gather rows");
    close(bf16At(rows2, width + c), bf16At(gathered, 1 * width + c), 0.0, 0.0, "gather rows");
  }
  std::cout << "gathers ok\n";
}

// ---------------------------------------------------------------- sampling

void testSampling(metal::MetalBackend &backend) {
  const uint32_t V = 3072, rows = 2;
  std::vector<float> logits(size_t(rows) * V);
  for (float &v : logits) v = uniform(-5.0F, 5.0F);
  // Row 0: the best id is suppressed (>= 2048), the second best penalized
  // below the third; greedy must pick the third.
  logits[2500] = 20.0F;
  logits[100] = 10.0F;
  logits[200] = 9.0F;
  // Row 0's EOS beats the rest too, masked by bit 0 of the mask.
  logits[2150] = 25.0F;
  // Row 1: EOS beats everything but is masked; id 7 is next.
  logits[V + 2150] = 30.0F;
  logits[V + 7] = 12.0F;
  const std::vector<uint32_t> history{100, 100, 3, 0, 0, 0, 0, 0}; // row 0 has 3 entries, row 1 none
  const std::vector<uint32_t> counts{3, 0};
  OmniSampleParams params{};
  params.rows = rows;
  params.vocabulary = V;
  params.suppress_from = 2048;
  params.eos = 2150;
  params.top_k = 50;
  params.top_p = 1.0F;
  params.temperature = 0.0F;
  params.repetition_penalty = 1.5F;
  params.history_stride = 4;
  params.mask_eos = 0b11;
  metal::MetalBuffer output = sharedBuffer(backend, rows * 4);
  metal::MetalBuffer logitsBuffer = f32Buffer(backend, logits);
  {
    metal::CommandGraph graph;
    Sampling::addSampleCodes(graph, logitsBuffer, u32Buffer(backend, history), u32Buffer(backend, counts),
                             f32Buffer(backend, {0.5F, 0.5F}), output, params);
    run(backend, graph);
    const auto *tokens = static_cast<const uint32_t *>(output.contents());
    require(tokens[0] == 200, "greedy row 0 picked " + std::to_string(tokens[0]));
    require(tokens[1] == 7, "greedy row 1 picked " + std::to_string(tokens[1]));
  }
  // The mask is per row: row 0's EOS, unmasked, wins; row 1's stays masked.
  params.mask_eos = 0b10;
  {
    metal::CommandGraph graph;
    Sampling::addSampleCodes(graph, logitsBuffer, u32Buffer(backend, history), u32Buffer(backend, counts),
                             f32Buffer(backend, {0.5F, 0.5F}), output, params);
    run(backend, graph);
    const auto *tokens = static_cast<const uint32_t *>(output.contents());
    require(tokens[0] == 2150, "unmasked greedy row 0 picked " + std::to_string(tokens[0]));
    require(tokens[1] == 7, "masked greedy row 1 picked " + std::to_string(tokens[1]));
  }
  logits[2150] = 0.0F;
  // Sampled: temperature 1, top-k 2 over row 1 {7: 12, next: ?}. With
  // uniform 0.0 the draw is the most likely id; with 0.999 the second.
  params.temperature = 1.0F;
  params.top_k = 2;
  params.mask_eos = 0;
  params.repetition_penalty = 1.0F;
  logits[V + 7] = 12.0F;
  logits[V + 2150] = 11.5F; // EOS allowed now: second most likely
  logitsBuffer = f32Buffer(backend, logits);
  {
    metal::CommandGraph graph;
    Sampling::addSampleCodes(graph, logitsBuffer, {}, {}, f32Buffer(backend, {0.0F, 0.999F}), output, params);
    run(backend, graph);
    const auto *tokens = static_cast<const uint32_t *>(output.contents());
    require(tokens[1] == 2150, "sampled row 1 with u=0.999 picked " + std::to_string(tokens[1]));
  }
  {
    metal::CommandGraph graph;
    Sampling::addSampleCodes(graph, logitsBuffer, {}, {}, f32Buffer(backend, {0.0F, 0.0F}), output, params);
    run(backend, graph);
    const auto *tokens = static_cast<const uint32_t *>(output.contents());
    require(tokens[1] == 7, "sampled row 1 with u=0 picked " + std::to_string(tokens[1]));
  }
  // Top-p 0.5 keeps only the first id of row 1 (p(7) ~ 0.62).
  params.top_p = 0.5F;
  {
    metal::CommandGraph graph;
    Sampling::addSampleCodes(graph, logitsBuffer, {}, {}, f32Buffer(backend, {0.999F, 0.999F}), output, params);
    run(backend, graph);
    const auto *tokens = static_cast<const uint32_t *>(output.contents());
    require(tokens[1] == 7, "top-p row 1 picked " + std::to_string(tokens[1]));
  }
  // Random rows with ties against a reference of the selection as repeated
  // argmax defines it (ties to the lowest id): the kept ids in order, the
  // softmax at the temperature, the nucleus and the draw. The draw's
  // boundaries differ by the GPU's exp, so the reference accepts a token
  // whose interval of u comes within 1e-4 of the uniform.
  {
    const uint32_t R = 16;
    std::vector<float> random(size_t(R) * V);
    for (size_t i = 0; i < random.size(); ++i) random[i] = std::round(uniform(-8.0F, 8.0F) * 4.0F) / 4.0F; // many ties
    std::vector<float> us(R);
    for (uint32_t r = 0; r < R; ++r) us[r] = (r + 0.5F) / R;
    metal::MetalBuffer randomLogits = f32Buffer(backend, random);
    metal::MetalBuffer randomOutput = sharedBuffer(backend, R * 4);
    for (const auto [k, topP] : {std::pair{50U, 1.0F}, std::pair{256U, 0.8F}, std::pair{1U, 1.0F}, std::pair{7U, 0.3F}}) {
      OmniSampleParams q{};
      q.rows = R;
      q.vocabulary = V;
      q.suppress_from = 2048;
      q.eos = 2150;
      q.top_k = k;
      q.top_p = topP;
      q.temperature = 0.9F;
      q.repetition_penalty = 1.0F;
      q.history_stride = 4;
      q.mask_eos = 0;
      metal::CommandGraph graph;
      Sampling::addSampleCodes(graph, randomLogits, {}, {}, f32Buffer(backend, us), randomOutput, q);
      run(backend, graph);
      const auto *tokens = static_cast<const uint32_t *>(randomOutput.contents());
      for (uint32_t r = 0; r < R; ++r) {
        std::vector<float> row(random.begin() + size_t(r) * V, random.begin() + size_t(r + 1) * V);
        for (uint32_t i = 2048; i < V; ++i)
          if (i != 2150) row[i] = -INFINITY;
        std::vector<uint32_t> kept;
        std::vector<float> keptValues;
        for (uint32_t step = 0; step < k; ++step) {
          uint32_t best = 0;
          for (uint32_t i = 1; i < V; ++i)
            if (row[i] > row[best]) best = i;
          kept.push_back(best);
          keptValues.push_back(row[best]);
          row[best] = -INFINITY;
        }
        std::vector<double> probabilities(k);
        double mass = 0.0;
        for (uint32_t i = 0; i < k; ++i) {
          probabilities[i] = std::exp(double(keptValues[i] / 0.9F) - double(keptValues[0] / 0.9F));
          mass += probabilities[i];
        }
        uint32_t nucleus = k;
        if (topP < 1.0F) {
          double cumulative = 0.0;
          for (uint32_t i = 0; i < k; ++i) {
            cumulative += probabilities[i] / mass;
            if (cumulative >= topP) {
              nucleus = i + 1;
              break;
            }
          }
        }
        double nucleusMass = 0.0;
        for (uint32_t i = 0; i < nucleus; ++i) nucleusMass += probabilities[i];
        // The interval of u that selects each kept id.
        const double draw = double(us[r]) * nucleusMass;
        bool accepted = false;
        double cumulative = 0.0;
        for (uint32_t i = 0; i < nucleus && !accepted; ++i) {
          const double low = cumulative;
          cumulative += probabilities[i];
          const bool last = i + 1 == nucleus;
          if (tokens[r] == kept[i] && draw >= low - 1e-4 * nucleusMass && (last || draw < cumulative + 1e-4 * nucleusMass))
            accepted = true;
        }
        require(accepted, "random sampling row " + std::to_string(r) + " k " + std::to_string(k) + " picked " +
                              std::to_string(tokens[r]) + " (u " + std::to_string(us[r]) + ")");
      }
    }
  }
  // Wide argmax.
  const uint32_t wide = 151936;
  std::vector<float> wideLogits(size_t(2) * wide, -1.0F);
  wideLogits[151935] = 3.0F;
  wideLogits[wide + 77091] = 2.0F;
  wideLogits[wide + 77090] = 2.0F;
  metal::MetalBuffer wideOut = sharedBuffer(backend, 8);
  metal::CommandGraph graph;
  Sampling::addArgmax(graph, f32Buffer(backend, wideLogits), wideOut, 2, wide, wide);
  run(backend, graph);
  const auto *tokens = static_cast<const uint32_t *>(wideOut.contents());
  require(tokens[0] == 151935 && tokens[1] == 77090, "argmax picked " + std::to_string(tokens[0]) + "," + std::to_string(tokens[1]));
  // Into every third slot (a speculative round's proposals), the others kept.
  const std::vector<uint32_t> sentinel(6, 7);
  metal::MetalBuffer strided = u32Buffer(backend, sentinel);
  metal::CommandGraph stridedGraph;
  Sampling::addArgmax(stridedGraph, f32Buffer(backend, wideLogits), backend.view(strided, 4, 20), 2, wide, wide, 3);
  run(backend, stridedGraph);
  const auto *slots = static_cast<const uint32_t *>(strided.contents());
  require(slots[1] == 151935 && slots[4] == 77090 && slots[0] == 7 && slots[2] == 7 && slots[3] == 7 && slots[5] == 7,
          "strided argmax wrote the wrong slots");
  // Through a token map (a DFlash draft's d2t): each row's argmax, mapped.
  const uint32_t reduced = 48000;
  std::vector<float> reducedLogits(size_t(2) * reduced, -1.0F);
  reducedLogits[47999] = 1.0F;
  reducedLogits[reduced + 5] = 1.0F;
  std::vector<uint32_t> map(reduced);
  for (uint32_t i = 0; i < reduced; ++i) map[i] = i * 3 + 1;
  metal::MetalBuffer mapped = u32Buffer(backend, sentinel);
  metal::CommandGraph mappedGraph;
  Sampling::addArgmaxMapped(mappedGraph, f32Buffer(backend, reducedLogits), backend.view(mapped, 4, 20),
                            u32Buffer(backend, map), 2, reduced, reduced, 3);
  run(backend, mappedGraph);
  const auto *mappedSlots = static_cast<const uint32_t *>(mapped.contents());
  require(mappedSlots[1] == 47999 * 3 + 1 && mappedSlots[4] == 16 && mappedSlots[0] == 7 && mappedSlots[5] == 7,
          "mapped argmax wrote " + std::to_string(mappedSlots[1]) + "," + std::to_string(mappedSlots[4]));
  std::cout << "sampling ok\n";
}

// ---------------------------------------------------------------- attention

struct Sequence {
  uint32_t length = 0; // tokens prefilled
  std::vector<float> qkv; // rows of OMNI_DECODER_QKV_WIDTH, as bf16
};

constexpr uint32_t kQH = OMNI_DECODER_QUERY_HEADS, kKH = OMNI_DECODER_KV_HEADS, kD = OMNI_DECODER_HEAD_DIM;
constexpr uint32_t kQkv = OMNI_DECODER_QKV_WIDTH, kWidth = OMNI_DECODER_ATTENTION_WIDTH;

// The normalized, rotated head as the prepare kernel computes it, in fp64
// from bf16 inputs.
std::vector<double> prepareHead(const float *head, const std::vector<float> &norm, uint32_t position,
                                const std::vector<float> &inverseFrequencies, float epsilon) {
  double sum = 0.0;
  for (uint32_t d = 0; d < kD; ++d) sum += double(head[d]) * head[d];
  const double inverse = 1.0 / std::sqrt(sum / kD + epsilon);
  std::vector<double> normalized(kD), rotated(kD);
  for (uint32_t d = 0; d < kD; ++d) normalized[d] = bf16ToFloat(floatToBf16(float(head[d] * inverse * norm[d])));
  for (uint32_t d = 0; d < kD / 2; ++d) {
    const double angle = double(position) * inverseFrequencies[d];
    rotated[d] = normalized[d] * std::cos(angle) - normalized[d + kD / 2] * std::sin(angle);
    rotated[d + kD / 2] = normalized[d + kD / 2] * std::cos(angle) + normalized[d] * std::sin(angle);
  }
  return rotated;
}

struct AttentionFixture {
  metal::MetalBackend &backend;
  std::vector<Sequence> sequences;
  std::vector<float> qNorm, kNorm, inverseFrequencies;
  uint32_t pageTableStride = 8; // blocks per sequence per layer
  uint32_t layers = 2;
  uint32_t pageCount = 0;
  metal::MetalBuffer pageTable, pages, qNormBuffer, kNormBuffer, frequencies;
  std::vector<uint32_t> table;
  explicit AttentionFixture(metal::MetalBackend &b, size_t count, uint32_t stride = 8) : backend(b), pageTableStride(stride) {
    qNorm = roundedBf16(randomValues(kD, 1.0F));
    kNorm = roundedBf16(randomValues(kD, 1.0F));
    for (uint32_t d = 0; d < kD / 2; ++d) inverseFrequencies.push_back(float(std::pow(1e6, -2.0 * d / kD)));
    qNormBuffer = bf16Buffer(backend, qNorm);
    kNormBuffer = bf16Buffer(backend, kNorm);
    frequencies = f32Buffer(backend, inverseFrequencies);
    sequences.resize(count);
    // Every sequence owns pageTableStride pages per layer, numbered in a
    // scrambled order so page addressing is exercised.
    pageCount = static_cast<uint32_t>(count) * layers * pageTableStride;
    table.resize(pageCount);
    std::vector<uint32_t> order(pageCount);
    for (uint32_t i = 0; i < pageCount; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    table = order;
    pageTable = u32Buffer(backend, table);
    pages = sharedBuffer(backend, uint64_t{pageCount} * OMNI_PAGE_BYTES);
    std::memset(pages.contents(), 0, uint64_t{pageCount} * OMNI_PAGE_BYTES);
  }
  uint32_t tableOffset(size_t sequence) const { return static_cast<uint32_t>(sequence) * layers * pageTableStride; }

  // Runs prepare + attention over `rows` new rows per sequence (all
  // sequences get the same count) and checks the outputs against fp64
  // attention over the bf16 keys/values the kernel stored.
  void step(uint32_t rowsPerSequence, uint32_t layer, bool decode) {
    step(std::vector<uint32_t>(sequences.size(), rowsPerSequence), layer, decode);
  }
  // Rows per sequence; a prefill step whose sequences fit the fused verify
  // kernel also runs it (on its own, into a second output, after the
  // prefill path stored the same keys and values) and checks its outputs.
  // Returns whether it did.
  bool step(const std::vector<uint32_t> &rowCounts, uint32_t layer, bool decode) {
    std::vector<AttentionSequence> descriptors;
    std::vector<float> qkv;
    std::vector<uint32_t> rowSequence;
    uint32_t rows = 0;
    const uint32_t rowsPerSequence = rowCounts.front();
    for (size_t s = 0; s < sequences.size(); ++s) {
      const std::vector<float> fresh = roundedBf16(randomValues(size_t(rowCounts[s]) * kQkv, 1.0F));
      sequences[s].qkv.insert(sequences[s].qkv.end(), fresh.begin(), fresh.end());
      qkv.insert(qkv.end(), fresh.begin(), fresh.end());
      descriptors.push_back({tableOffset(s), pageTableStride, sequences[s].length, rowCounts[s], rows,
                             sequences[s].length});
      for (uint32_t r = 0; r < rowCounts[s]; ++r) rowSequence.push_back(static_cast<uint32_t>(s));
      rows += rowCounts[s];
    }
    const AttentionTables tables = attentionTables(descriptors, decode);
    const AttentionWorkspace workspace =
        decode ? Attention::decodeWorkspace(rows) : Attention::prefillWorkspace(tables.paddedRows, static_cast<uint32_t>(tables.tiles.size()));
    AttentionBuffers b;
    b.qkv = bf16Buffer(backend, qkv);
    b.queryNorm = qNormBuffer;
    b.keyNorm = kNormBuffer;
    b.inverseFrequencies = frequencies;
    b.rowSequence = u32Buffer(backend, rowSequence);
    b.sequences = structBuffer(backend, tables.sequences);
    b.tiles = structBuffer(backend, tables.tiles);
    b.pageTable = pageTable;
    b.pages = pages;
    b.queries = sharedBuffer(backend, workspace.queriesBytes);
    b.partials = sharedBuffer(backend, workspace.partialsBytes);
    b.statistics = sharedBuffer(backend, workspace.statisticsBytes);
    b.output = sharedBuffer(backend, uint64_t{rows} * kWidth * 2);
    // The prepare + split/prefill path writes the queries the checks read;
    // a fusable decode step also runs the fused kernel (on its own, into a
    // second output, from the same stored keys and values; a two-row step
    // runs it alone).
    const bool fused = decode && Attention::fusable(tables);
    const bool splitPath = !decode || rowsPerSequence == 1;
    const bool verify = !decode && Attention::verifyFusable(tables);
    metal::MetalBuffer fusedOutput = sharedBuffer(backend, uint64_t{rows} * kWidth * 2);
    // The fused kernel again, storing its rotations in a table, and once
    // more loading them from it (a forward's later layers): the same bits.
    metal::MetalBuffer storedOutput = sharedBuffer(backend, uint64_t{rows} * kWidth * 2);
    metal::MetalBuffer loadedOutput = sharedBuffer(backend, uint64_t{rows} * kWidth * 2);
    // A step that fits one page runs the one-page kernel, and one of a few
    // rows a threadgroup per query head; the kernel of a threadgroup per KV
    // head must give them bit for bit.
    metal::MetalBuffer kvHeadOutput = sharedBuffer(backend, uint64_t{rows} * kWidth * 2);
    const bool onePage = fused && Attention::fitsOnePage(tables);
    metal::CommandGraph graph;
    if (splitPath) {
      Attention::addPrepare(graph, b, tables, layer, 1e-6F, pageCount, pageCount);
      if (decode) Attention::addDecode(graph, b, tables, layer, pageCount, pageCount);
      else Attention::addPrefill(graph, b, tables, layer, pageCount, pageCount);
    }
    const auto addFused = [&](const AttentionBuffers &f, Rotations rotations) {
      if (fused) Attention::addDecodeFused(graph, f, tables, layer, 1e-6F, pageCount, pageCount, rotations);
      else Attention::addVerifyFused(graph, f, tables, layer, 1e-6F, pageCount, pageCount, rotations);
    };
    if (fused || verify) {
      AttentionBuffers f = b;
      f.output = fusedOutput;
      addFused(f, Rotations::Compute);
      f.rotations = sharedBuffer(backend, Attention::rotationBytes(rows));
      std::memset(f.rotations.contents(), 0, Attention::rotationBytes(rows));
      f.output = storedOutput;
      addFused(f, Rotations::Store);
      f.output = loadedOutput;
      addFused(f, Rotations::Load);
      f.output = kvHeadOutput;
      if (fused) Attention::addDecodeFused(graph, f, tables, layer, 1e-6F, pageCount, pageCount, Rotations::Load, true);
      else Attention::addVerifyFused(graph, f, tables, layer, 1e-6F, pageCount, pageCount, Rotations::Load, true);
    }
    run(backend, graph);
    if ((fused || verify) && std::memcmp(fusedOutput.contents(), kvHeadOutput.contents(), uint64_t{rows} * kWidth * 2))
      throw std::runtime_error(std::string(onePage ? "one-page" : "query-head") + (fused ? " decode" : " verify") +
                               " attention differs from the kernel of a threadgroup per KV head");
    if (fused || verify)
      for (const metal::MetalBuffer &other : {storedOutput, loadedOutput})
        if (std::memcmp(fusedOutput.contents(), other.contents(), uint64_t{rows} * kWidth * 2))
          throw std::runtime_error("fused attention: stored or loaded rotations changed the output");
    if (decode && Attention::longFusable(tables)) {
      // A lone row over a long history runs the long kernel: prepare, the
      // split kernel and the reduce in one dispatch, the same bits as the
      // three over the same splits. Computing, storing and loading the
      // rotations, through one set of counters (the last split of each
      // dispatch returns them to zero).
      b.counters = sharedBuffer(backend, workspace.countersBytes);
      std::memset(b.counters.contents(), 0, workspace.countersBytes);
      AttentionBuffers f = b;
      f.rotations = sharedBuffer(backend, Attention::rotationBytes(rows));
      for (Rotations rotations : {Rotations::Compute, Rotations::Store, Rotations::Load}) {
        f.output = sharedBuffer(backend, uint64_t{rows} * kWidth * 2);
        metal::CommandGraph other;
        Attention::addDecodeLong(other, f, tables, layer, 1e-6F, pageCount, pageCount, rotations);
        run(backend, other);
        if (std::memcmp(b.output.contents(), f.output.contents(), uint64_t{rows} * kWidth * 2))
          throw std::runtime_error("long decode attention differs from prepare, split and reduce over " +
                                   std::to_string(tables.sequences.front().splits) + " splits");
      }
      for (size_t i = 0; i < workspace.countersBytes / 4; ++i)
        if (static_cast<const uint32_t *>(b.counters.contents())[i])
          throw std::runtime_error("long decode attention left a counter set");
    }
    if (!splitPath) {
      // The two-row step: the fused kernel's outputs stand in, and its
      // queries are checked through the attention result only.
      b.output = fusedOutput;
    }
    // Check the prepared queries and keys of this step, then the outputs.
    for (size_t s = 0; s < sequences.size(); ++s) {
      const Sequence &sequence = sequences[s];
      const uint32_t first = descriptors[s].rowOffset;
      for (uint32_t r = 0; r < rowCounts[s]; ++r) {
        const uint32_t position = sequence.length + r;
        const float *row = &qkv[(size_t(first) + r) * kQkv];
        for (uint32_t h = 0; h < kQH && splitPath; ++h) {
          const std::vector<double> expected = prepareHead(row + h * kD, qNorm, position, inverseFrequencies, 1e-6F);
          for (uint32_t d = 0; d < kD; ++d) {
            const size_t index = ((size_t(h / 2) * tables.paddedRows + first + r) * 2 + h % 2) * kD + d;
            close(bf16At(b.queries, index), expected[d], 1e-2, 2e-2, "prepared query");
          }
        }
        for (uint32_t h = 0; h < kKH; ++h) {
          const std::vector<double> expected = prepareHead(row + (kQH + h) * kD, kNorm, position, inverseFrequencies, 1e-6F);
          const uint32_t page = table[tableOffset(s) + layer * pageTableStride + position / 32];
          const auto *keys = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(pages.contents()) + uint64_t{page} * OMNI_PAGE_BYTES);
          const auto *values = keys + OMNI_PAGE_KEY_BYTES / 2;
          for (uint32_t d = 0; d < kD; ++d) {
            close(bf16ToFloat(keys[(h * 32 + position % 32) * kD + d]), expected[d], 1e-2, 2e-2, "stored key");
            close(bf16ToFloat(values[(h * 32 + position % 32) * kD + d]), row[(kQH + kKH + h) * kD + d], 0.0, 0.0, "stored value");
          }
        }
      }
      // fp64 attention of each new row over the stored bf16 keys/values.
      for (uint32_t r = 0; r < rowCounts[s]; ++r) {
        const uint32_t position = sequence.length + r;
        for (uint32_t h = 0; h < kQH; ++h) {
          std::vector<double> q(kD);
          if (splitPath) {
            for (uint32_t d = 0; d < kD; ++d)
              q[d] = bf16At(b.queries, ((size_t(h / 2) * tables.paddedRows + first + r) * 2 + h % 2) * kD + d);
          } else {
            const float *qkvRow = &qkv[(size_t(first) + r) * kQkv];
            const std::vector<double> prepared = prepareHead(qkvRow + h * kD, qNorm, position, inverseFrequencies, 1e-6F);
            for (uint32_t d = 0; d < kD; ++d) q[d] = bf16ToFloat(floatToBf16(float(prepared[d])));
          }
          const uint32_t kvHead = h / 2;
          std::vector<double> scores(position + 1);
          double maximum = -1e300;
          for (uint32_t t = 0; t <= position; ++t) {
            const uint32_t page = table[tableOffset(s) + layer * pageTableStride + t / 32];
            const auto *keys = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(pages.contents()) + uint64_t{page} * OMNI_PAGE_BYTES);
            double dot = 0.0;
            for (uint32_t d = 0; d < kD; ++d) dot += q[d] * bf16ToFloat(keys[(kvHead * 32 + t % 32) * kD + d]);
            scores[t] = dot / std::sqrt(double(kD));
            maximum = std::max(maximum, scores[t]);
          }
          double denominator = 0.0;
          std::vector<double> output(kD, 0.0);
          for (uint32_t t = 0; t <= position; ++t) {
            const double p = std::exp(scores[t] - maximum);
            denominator += p;
            const uint32_t page = table[tableOffset(s) + layer * pageTableStride + t / 32];
            const auto *values = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(pages.contents()) + uint64_t{page} * OMNI_PAGE_BYTES + OMNI_PAGE_KEY_BYTES);
            for (uint32_t d = 0; d < kD; ++d) output[d] += p * bf16ToFloat(values[(kvHead * 32 + t % 32) * kD + d]);
          }
          for (uint32_t d = 0; d < kD; ++d) {
            close(bf16At(b.output, (size_t(first) + r) * kWidth + h * kD + d), output[d] / denominator, 1.5e-2, 2e-2,
                  (decode ? "decode" : "prefill") + std::string(" attention s=") + std::to_string(s) + " r=" +
                      std::to_string(r) + " h=" + std::to_string(h) + " d=" + std::to_string(d));
            if ((fused && splitPath) || verify)
              close(bf16At(fusedOutput, (size_t(first) + r) * kWidth + h * kD + d), output[d] / denominator, 1.5e-2, 2e-2,
                    std::string(verify ? "fused verify" : "fused decode") + " attention s=" + std::to_string(s) +
                        " r=" + std::to_string(r) + " h=" + std::to_string(h) + " d=" + std::to_string(d));
          }
        }
      }
    }
    for (size_t s = 0; s < sequences.size(); ++s) sequences[s].length += rowCounts[s];
    return verify;
  }
};

void testAttention(metal::MetalBackend &backend) {
  AttentionFixture fixture(backend, 3);
  fixture.step(37, 0, false); // ragged prefill across pages
  fixture.step(37, 1, false);
  fixture.step(1, 0, true); // decode after prefill
  fixture.step(1, 1, true);
  fixture.step(20, 0, false); // a second prefill chunk continuing the context
  fixture.step(1, 0, true);
  // Long histories: within the fused kernel's pages (one and two rows per
  // sequence), and beyond them across many splits.
  AttentionFixture longer(backend, 1, 8);
  longer.step(200, 0, false);
  longer.step(1, 0, true);
  longer.step(2, 0, true); // the two-row fused decode
  AttentionFixture fusedWide(backend, 1, 16);
  fusedWide.step(400, 0, false);
  fusedWide.step(1, 0, true); // 13 pages, fused
  fusedWide.step(2, 0, true);
  // A lone row past 16 pages: the long kernel over three splits of at most
  // eight pages, and past 24 pages over two.
  AttentionFixture split(backend, 1, 40);
  split.step(600, 0, false);
  split.step(1, 0, true); // 19 pages
  require(Attention::longFusable(attentionTables(std::vector<AttentionSequence>{{0, 40, 601, 1, 0, 601}}, true)),
          "a lone row over 19 pages does not take the long kernel");
  split.step(400, 0, false);
  split.step(1, 0, true); // 32 pages
  // Several sequences past 16 pages: the fused kernels (by query head for
  // up to four rows, by KV head beyond; one and two rows per sequence).
  AttentionFixture wide(backend, 2, 24);
  wide.step(560, 0, false);
  require(Attention::fusable(attentionTables(
              std::vector<AttentionSequence>{{0, 24, 560, 1, 0, 560}, {48, 24, 560, 1, 1, 560}}, true)),
          "two rows over 18 pages are not fusable");
  wide.step(1, 0, true);
  wide.step(2, 0, true);
  // The code predictor's pattern: a two-row step, then one row at a time
  // until the page is full, all on the one-page kernel; the page starts
  // with stale values, which the masked tokens must not reach.
  AttentionFixture onePage(backend, 3, 1);
  const std::vector<float> stale = roundedBf16(randomValues(size_t(onePage.pageCount) * OMNI_PAGE_BYTES / 2, 1.0F));
  auto *pageBits = static_cast<uint16_t *>(onePage.pages.contents());
  for (size_t i = 0; i < stale.size(); ++i) pageBits[i] = floatToBf16(stale[i]);
  onePage.step(2, 0, true);
  for (uint32_t i = 0; i < OMNI_PAGE_TOKENS - 2; ++i) onePage.step(1, 0, true);
  // Verify forwards: ragged rows (a lane's pending token and proposals, a
  // rider's one row) across page boundaries, through the fused verify
  // kernel, up to the most rows it takes, and a long history.
  // Up to four rows run a threadgroup per query head.
  AttentionFixture verify(backend, 3, 8);
  verify.step({30, 61, 5}, 0, false);
  require(verify.step({6, 1, 5}, 1, false) && verify.step({2, 3, 7}, 0, false) && verify.step({1, 10, 1}, 1, false) &&
              verify.step({1, 2, 1}, 0, false),
          "verify steps did not run the fused verify kernel");
  AttentionFixture verifyLong(backend, 2, 16);
  verifyLong.step(450, 0, false);
  require(verifyLong.step({5, 1}, 0, false) && verifyLong.step({2, 1}, 0, false),
          "a 15-page verify step did not run the fused verify kernel");
  std::cout << "verify attention ok\n";
  std::cout << "paged attention ok\n";
}

// The 1.7B tower's 16 heads and the 0.6B tower's 14.
void testDenseAttention(metal::MetalBackend &backend, uint32_t heads) {
  constexpr uint32_t hd = OMNI_ENCODER_HEAD_DIM;
  const uint32_t width = heads * hd;
  const uint32_t rows = 45, keyRows = 64, stride = width + 64; // a stride wider than the rows
  const std::vector<float> q = roundedBf16(randomValues(size_t(64) * stride, 1.0F));
  const std::vector<float> k = roundedBf16(randomValues(size_t(keyRows) * stride, 1.0F));
  const std::vector<float> v = roundedBf16(randomValues(size_t(keyRows) * stride, 1.0F));
  for (int mask = 0; mask < 2; ++mask) {
    // Two windows: rows 0..19 attend keys 0..19, rows 20..44 keys 20..44.
    std::vector<OmniDenseRow> table;
    for (uint32_t r = 0; r < rows; ++r) table.push_back(r < 20 ? OmniDenseRow{0, 20, r, 0} : OmniDenseRow{20, 45, r, 0});
    DenseAttentionBuffers b{bf16Buffer(backend, q), bf16Buffer(backend, k), bf16Buffer(backend, v),
                            sharedBuffer(backend, size_t(rows) * width * 2), structBuffer(backend, table)};
    const uint32_t window = 7;
    const float scale = 0.125F;
    metal::CommandGraph graph;
    DenseAttention::add(graph, b, rows, keyRows, stride, mask ? DenseMask::CausalWindow : DenseMask::None, window, scale,
                        heads);
    run(backend, graph);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t h = 0; h < heads; ++h) {
        std::vector<double> scores;
        std::vector<uint32_t> keys;
        double maximum = -1e300;
        for (uint32_t t = table[r].key_begin; t < table[r].key_end; ++t) {
          if (mask && (t > r || t + window <= r)) continue;
          double dot = 0.0;
          for (uint32_t d = 0; d < hd; ++d) dot += double(q[size_t(r) * stride + h * hd + d]) * k[size_t(t) * stride + h * hd + d];
          scores.push_back(dot * scale);
          keys.push_back(t);
          maximum = std::max(maximum, dot * scale);
        }
        double denominator = 0.0;
        std::vector<double> output(hd, 0.0);
        for (size_t i = 0; i < keys.size(); ++i) {
          const double p = std::exp(scores[i] - maximum);
          denominator += p;
          for (uint32_t d = 0; d < hd; ++d) output[d] += p * v[size_t(keys[i]) * stride + h * hd + d];
        }
        for (uint32_t d = 0; d < hd; ++d)
          close(bf16At(b.output, size_t(r) * width + h * hd + d), output[d] / denominator, 1.5e-2, 2e-2,
                "dense attention heads=" + std::to_string(heads) + " mask=" + std::to_string(mask) +
                    " r=" + std::to_string(r) + " h=" + std::to_string(h));
      }
  }
  std::cout << "dense attention (" << heads << " heads) ok\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: omni-kernels METALLIB\n";
    return 2;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    testDenseAll(backend);
    testPackedAll(backend);
    testNorms(backend);
    testEmbedding(backend);
    testSampling(backend);
    testAttention(backend);
    testDenseAttention(backend, OMNI_ENCODER_HEADS);
    testDenseAttention(backend, 14);
  } catch (const std::exception &error) {
    std::cerr << "omni kernels: " << error.what() << "\n";
    return 1;
  }
  std::cout << "omni kernels ok\n";
  return 0;
}
