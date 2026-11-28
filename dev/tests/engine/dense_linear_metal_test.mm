// ops::DenseLinear (kernels/shared/dense_linear.metal) and the LayerNorm
// before it (ops::Normalization::addLayer, kernels/shared/normalization.metal)
// against fp64 references, as the vision tower runs them:
//
// - every epilogue (none, GELU tanh and erf into bf16, an fp32 residual
//   updated in place, as it is or scaled per output, a LayerScale's) at row
//   counts below, on and past the 64-row tile; the rows past a projection's
//   rows, followed by NaN, are left as they were;
// - every tile the rule picks, reached through devices of each GPU class
//   and core count: the same bits on each, at every epilogue;
// - a row's bits at any offset in its buffer and in a projection of any
//   rows;
// - each buffer at the extent the operators require, refused one element
//   short;
// - the LayerNorm into bf16 rows and into fp32 rows, its input's own among
//   them;
// - Q8_0 weights (an Inkling 2 mmproj's), as the planes a GGUF image holds,
//   at every epilogue and those rows and more, on the bf16 prefill tile,
//   against fp64 of the Q8_0 values within the tile's error, its fp16 stage
//   of each weight (2^-11 of it).
//
//   dense-linear METALLIB
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/QuantFormat.h"
#include "ops/DenseLinear.hpp"
#include "ops/Normalization.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::DeviceCapabilities;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::DenseEpilogue;
using splash::ops::DenseLinear;
using splash::ops::DenseWeights;
using splash::ops::elementBytes;
using splash::ops::FloatOutput;
using splash::ops::QuantizedSegment;
using splash::ops::KernelCalibration;
using splash::ops::LayerNormWeights;
using splash::ops::Normalization;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::require;
using splash::test::sharedBuffer;

constexpr uint16_t kNan = 0x7FC0;

std::mt19937 rng(17);

constexpr DenseEpilogue kEpilogues[] = {DenseEpilogue::None, DenseEpilogue::GeluTanh, DenseEpilogue::GeluErf,
                                        DenseEpilogue::Residual, DenseEpilogue::ScaledResidual};

std::vector<float> randomBf16(size_t count, float scale) {
  std::uniform_real_distribution<float> uniform(-scale, scale);
  std::vector<float> values(count);
  for (float &value : values) value = bf16ToFloat(floatToBf16(uniform(rng)));
  return values;
}
MetalBuffer bf16Buffer(MetalBackend &backend, const std::vector<float> &values, size_t tail = 0) {
  MetalBuffer buffer = sharedBuffer(backend, (values.size() + tail) * 2);
  auto *bits = static_cast<uint16_t *>(buffer.contents());
  for (size_t i = 0; i < values.size(); ++i) bits[i] = floatToBf16(values[i]);
  std::fill_n(bits + values.size(), tail, kNan);
  return buffer;
}
MetalBuffer f32Buffer(MetalBackend &backend, const std::vector<float> &values, size_t tail = 0) {
  MetalBuffer buffer = sharedBuffer(backend, (values.size() + tail) * 4);
  std::memcpy(buffer.contents(), values.data(), values.size() * 4);
  std::fill_n(static_cast<float *>(buffer.contents()) + values.size(), tail, std::nanf(""));
  return buffer;
}
void run(MetalBackend &backend, CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}
double gelu(DenseEpilogue epilogue, double x) {
  if (epilogue == DenseEpilogue::GeluTanh)
    return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}
const char *kernelSuffix(DenseEpilogue epilogue) {
  switch (epilogue) {
  case DenseEpilogue::None:
    return "";
  case DenseEpilogue::GeluTanh:
    return "_gelu_tanh";
  case DenseEpilogue::GeluErf:
    return "_gelu_erf";
  case DenseEpilogue::Residual:
    return "_residual";
  case DenseEpilogue::ScaledResidual:
    return "_scaled_residual";
  }
  return "";
}
const char *epilogueName(DenseEpilogue epilogue) {
  switch (epilogue) {
  case DenseEpilogue::None:
    return "none";
  case DenseEpilogue::GeluTanh:
    return "gelu tanh";
  case DenseEpilogue::GeluErf:
    return "gelu erf";
  case DenseEpilogue::Residual:
    return "residual";
  case DenseEpilogue::ScaledResidual:
    return "scaled residual";
  }
  return "";
}

// Random Q8_0 weights of `outputs` x `inputs` (an F16 scale of 2^-10 to
// 2^-8 and 32 int8 codes a block) as the planes of a GGUF image
// (metal/abi/QuantFormat.h), and their values, each exact in fp32.
struct Q8Weights final {
  QuantizedSegment segment;
  std::vector<float> values;
};
Q8Weights q8Weights(MetalBackend &backend, uint32_t outputs, uint32_t inputs) {
  const uint32_t groups = inputs / 32;
  MetalBuffer plane0 = sharedBuffer(backend, size_t(outputs) * inputs);
  MetalBuffer meta = sharedBuffer(backend, size_t(outputs) * groups * 2);
  std::uniform_int_distribution<int> code(-127, 127);
  std::uniform_real_distribution<float> exponent(-10.0F, -8.0F);
  std::vector<float> values(size_t(outputs) * inputs);
  for (uint32_t n = 0; n < outputs; ++n)
    for (uint32_t g = 0; g < groups; ++g) {
      const _Float16 d = static_cast<_Float16>(std::exp2(exponent(rng)));
      const uint64_t record = quant_tile_index(n, g, groups);
      std::memcpy(static_cast<uint8_t *>(meta.contents()) + record * 2, &d, 2);
      for (uint32_t e = 0; e < 32; ++e) {
        const int q = code(rng);
        static_cast<int8_t *>(plane0.contents())[record * 32 + quant_slot(e)] = static_cast<int8_t>(q);
        values[size_t(n) * inputs + g * 32 + e] = static_cast<float>(d) * static_cast<float>(q);
      }
    }
  return {QuantizedSegment::planes(GGUF_FMT_Q80, outputs, inputs, plane0, {}, meta), std::move(values)};
}

// What a projection stored and the kernel it ran.
struct Projected final {
  std::vector<uint8_t> bits;
  std::string kernel;
};

// One projection of `rows` rows of `inputs` into `outputs`, its values drawn
// from `seed`, against fp64 unless `exact` is false: an output may differ by
// bf16 rounding of its magnitude (fp32 for the residuals, of the scaled sum
// and the residual each) plus fp32 accumulation over the inputs. The input
// holds whole row tiles, the rows past `rows` large finite values; NaN follows
// the output's rows, which must survive.
Projected testProjection(MetalBackend &backend, const DenseLinear &dense, uint32_t rows, uint32_t outputs,
                         uint32_t inputs, DenseEpilogue epilogue, bool exact = true, uint32_t seed = 17,
                         bool q8 = false) {
  const std::string label = std::string(q8 ? "q8_0 " : "") + epilogueName(epilogue) + " rows " +
                            std::to_string(rows) + " " + std::to_string(outputs) + "x" + std::to_string(inputs);
  rng.seed(seed);
  const uint32_t padded = DenseLinear::paddedRows(rows);
  std::vector<float> x = randomBf16(size_t(padded) * inputs, 1.0F);
  std::fill(x.begin() + size_t(rows) * inputs, x.end(), 64.0F);
  const Q8Weights quantized = q8 ? q8Weights(backend, outputs, inputs) : Q8Weights{};
  const std::vector<float> w = q8 ? quantized.values : randomBf16(size_t(outputs) * inputs, 0.1F),
                           b = randomBf16(outputs, 0.5F);
  const std::vector<float> r = randomBf16(size_t(rows) * outputs, 1.0F);
  const bool scaled = epilogue == DenseEpilogue::ScaledResidual;
  const bool residual = scaled || epilogue == DenseEpilogue::Residual;
  const std::vector<float> scale = randomBf16(outputs, 0.3F);
  const size_t tail = size_t(64) * outputs;
  const MetalBuffer output = residual ? f32Buffer(backend, r, tail)
                                      : bf16Buffer(backend, std::vector<float>(size_t(rows) * outputs, 0.0F), tail);
  const DenseWeights weights{q8 ? MetalBuffer{} : bf16Buffer(backend, w), bf16Buffer(backend, b),
                             bf16Buffer(backend, scale), quantized.segment};
  DenseLinear::requireWeights(weights, outputs, inputs, label);
  CommandGraph graph;
  dense.add(graph, weights, outputs, inputs, bf16Buffer(backend, x),
            backend.view(output, 0, size_t(rows) * outputs * (residual ? 4 : 2)), rows, epilogue);
  // A Q8_0 projection's rows past its whole 32 run the bounded instance.
  const std::string kernel = graph.dispatches().front().pipelineName;
  const std::string selected = dense.kernel(rows, outputs, inputs, epilogue, q8 ? GGUF_FMT_Q80 : QuantizedSegment::kFloat32);
  require(kernel == selected || (q8 && rows < 32 && kernel == selected + "_bounded"),
          label + ": ran " + kernel + ", not its selection");
  run(backend, graph);
  for (uint32_t m = 0; m < (exact ? rows : 0); ++m)
    for (uint32_t n = 0; n < outputs; ++n) {
      double sum = b[n], magnitude = std::fabs(b[n]);
      for (uint32_t k = 0; k < inputs; ++k) {
        sum += double(x[size_t(m) * inputs + k]) * w[size_t(n) * inputs + k];
        magnitude += std::fabs(double(x[size_t(m) * inputs + k]) * w[size_t(n) * inputs + k]);
      }
      // The tile stages each Q8_0 weight in fp16.
      double rounding = q8 ? 0x1p-11 * magnitude : 0.0;
      double expected = sum;
      if (epilogue == DenseEpilogue::GeluTanh || epilogue == DenseEpilogue::GeluErf) {
        expected = gelu(epilogue, sum);
        rounding *= 1.13; // GELU's steepest slope
      }
      if (scaled) expected *= scale[n], magnitude *= std::fabs(scale[n]), rounding *= std::fabs(scale[n]);
      if (residual) expected += r[size_t(m) * outputs + n];
      const size_t index = size_t(m) * outputs + n;
      const double got = residual ? static_cast<const float *>(output.contents())[index]
                                  : bf16ToFloat(static_cast<const uint16_t *>(output.contents())[index]);
      const double bound = (residual ? 0x1p-22 : 0x1p-8) * std::fabs(expected) +
                           (scaled ? 0x1p-22 * std::fabs(r[size_t(m) * outputs + n]) : 0.0) +
                           inputs * 0x1p-24 * magnitude + rounding + 1e-6;
      if (!(std::fabs(got - expected) <= bound))
        throw std::runtime_error(label + " [" + std::to_string(m) + "," + std::to_string(n) + "]: " +
                                 std::to_string(got) + " vs " + std::to_string(expected));
    }
  const auto *bytes = static_cast<const uint8_t *>(output.contents());
  const size_t element = residual ? 4 : 2, end = size_t(rows) * outputs * element;
  for (size_t i = 0; i < tail; ++i) {
    const bool intact = residual ? std::isnan(reinterpret_cast<const float *>(bytes + end)[i])
                                 : reinterpret_cast<const uint16_t *>(bytes + end)[i] == kNan;
    require(intact, label + ": a write reached past the projection's rows");
  }
  return {std::vector<uint8_t>(bytes, bytes + end), kernel};
}

DeviceCapabilities deviceOf(uint32_t appleGpuFamily, uint32_t cores) {
  DeviceCapabilities device;
  device.appleGpuFamily = appleGpuFamily;
  device.gpuCoreCount = cores;
  return device;
}

// One projection through the rule on Apple9, Apple10 and Apple11 devices of
// 1 to 64 cores, which reach each of its tiles at this grid (three row tiles
// by four 64-column tiles, of the inputs the 64 x 128 tile takes): each tile
// runs, the first against fp64, to the same bits at every epilogue.
void testTiles(MetalBackend &backend) {
  constexpr uint32_t rows = 130, outputs = 256;
  const uint32_t inputs = KernelCalibration::of(deviceOf(11, 1)).denseTile.wideInputs;
  for (const DenseEpilogue epilogue : kEpilogues) {
    std::set<std::string> ran;
    std::vector<uint8_t> first;
    for (const uint32_t family : {9U, 10U, 11U})
      for (const uint32_t cores : {1U, 2U, 3U, 4U, 6U, 8U, 12U, 16U, 24U, 32U, 48U, 64U}) {
        const DenseLinear dense(deviceOf(family, cores));
        const Projected projected = testProjection(backend, dense, rows, outputs, inputs, epilogue, first.empty());
        if (first.empty()) first = projected.bits;
        require(projected.bits == first, projected.kernel + ": other bits than the first tile's");
        ran.insert(projected.kernel);
      }
    for (const char *tile : {"dense_linear_m64n64s8", "dense_linear_m64n64s4", "dense_linear_m64n128s4"})
      require(ran.count(tile + std::string(kernelSuffix(epilogue))) == 1,
              std::string(tile) + " " + epilogueName(epilogue) + ": no device ran it");
  }
}

// Rows 0..rows-1 of one input, projected from offset 0 and from offset 7 of
// a buffer whose other rows differ: the same bits, at every epilogue.
void testOffsets(MetalBackend &backend, const DenseLinear &dense) {
  constexpr uint32_t rows = 150, outputs = 256, inputs = 384, shift = 7;
  const std::vector<float> w = randomBf16(size_t(outputs) * inputs, 0.1F), b = randomBf16(outputs, 0.5F);
  const DenseWeights weights{bf16Buffer(backend, w), bf16Buffer(backend, b), bf16Buffer(backend, randomBf16(outputs, 0.3F))};
  const std::vector<float> x = randomBf16(size_t(DenseLinear::paddedRows(rows)) * inputs, 1.0F);
  std::vector<float> shifted = randomBf16(size_t(DenseLinear::paddedRows(rows + shift)) * inputs, 1.0F);
  std::copy_n(x.begin(), size_t(rows) * inputs, shifted.begin() + size_t(shift) * inputs);
  for (const DenseEpilogue epilogue : kEpilogues) {
    const size_t element = epilogue == DenseEpilogue::Residual || epilogue == DenseEpilogue::ScaledResidual ? 4 : 2;
    const MetalBuffer alone = sharedBuffer(backend, size_t(rows) * outputs * element);
    const MetalBuffer packed = sharedBuffer(backend, size_t(rows + shift) * outputs * element);
    std::memset(alone.contents(), 0, alone.sizeBytes());
    std::memset(packed.contents(), 0, packed.sizeBytes());
    CommandGraph graph;
    dense.add(graph, weights, outputs, inputs, bf16Buffer(backend, x), alone, rows, epilogue);
    dense.add(graph, weights, outputs, inputs, bf16Buffer(backend, shifted), packed, rows + shift, epilogue);
    run(backend, graph);
    require(!std::memcmp(alone.contents(), static_cast<const uint8_t *>(packed.contents()) + shift * outputs * element,
                         alone.sizeBytes()),
            std::string(epilogueName(epilogue)) + ": rows at another offset changed their bits");
  }
}

void testProjectionExtents(MetalBackend &backend, const DenseLinear &dense) {
  constexpr uint32_t rows = 70, outputs = 256, inputs = 384;
  const MetalBuffer weight = sharedBuffer(backend, size_t(outputs) * inputs * 2);
  const MetalBuffer bias = sharedBuffer(backend, size_t(outputs) * 2);
  const MetalBuffer input = sharedBuffer(backend, size_t(DenseLinear::paddedRows(rows)) * inputs * 2);
  const MetalBuffer output = sharedBuffer(backend, size_t(rows) * outputs * 4);
  using splash::test::requireExtent;
  requireExtent(backend, input, input.sizeBytes(), 2, "dense input", [&](CommandGraph &g, const MetalBuffer &v) {
    dense.add(g, {weight, bias}, outputs, inputs, v, output, rows);
  });
  requireExtent(backend, output, size_t(rows) * outputs * 2, 2, "dense output",
                [&](CommandGraph &g, const MetalBuffer &v) {
                  dense.add(g, {weight, bias}, outputs, inputs, input, v, rows);
                });
  requireExtent(backend, output, output.sizeBytes(), 4, "dense output", [&](CommandGraph &g, const MetalBuffer &v) {
    dense.add(g, {weight, bias}, outputs, inputs, input, v, rows, DenseEpilogue::Residual);
  });
  // The weights' contract (DenseLinear::requireWeights): their bytes, and
  // sizes of whole tiles, which a tile of 128 columns or a sum of 128-input
  // chunks would otherwise cut short without an error.
  requireExtent(backend, weight, weight.sizeBytes(), 2, "dense weight", [&](CommandGraph &g, const MetalBuffer &v) {
    DenseLinear::requireWeights({v, bias}, outputs, inputs, "dense");
    dense.add(g, {v, bias}, outputs, inputs, input, output, rows);
  });
  requireExtent(backend, bias, bias.sizeBytes(), 2, "dense bias", [&](CommandGraph &g, const MetalBuffer &v) {
    DenseLinear::requireWeights({weight, v}, outputs, inputs, "dense");
    dense.add(g, {weight, v}, outputs, inputs, input, output, rows);
  });
  // A LayerScale's scales, which requireWeights takes where a projection has
  // them: the scaled residual reads them as its owner checked them, as it
  // reads the weights and the bias (their owner states which projections
  // need them, ops/InklingVision.cpp).
  const MetalBuffer scale = sharedBuffer(backend, size_t(outputs) * 2);
  requireExtent(backend, scale, scale.sizeBytes(), 2, "dense scale", [&](CommandGraph &g, const MetalBuffer &v) {
    DenseLinear::requireWeights({weight, bias, v}, outputs, inputs, "dense");
    dense.add(g, {weight, bias, v}, outputs, inputs, input, output, rows, DenseEpilogue::ScaledResidual);
  });
  for (const auto [o, i] : {std::pair{192U, 384U}, std::pair{256U, 192U}, std::pair{0U, 384U}})
    splash::test::rejects([&] { DenseLinear::requireWeights({weight, bias}, o, i, "dense"); }, "not whole multiples",
                          "dense weights of " + std::to_string(o) + " x " + std::to_string(i) + " were accepted");
}

// LayerNorm with bias of fp32 rows into bf16 rows, and into fp32 rows in
// place (Inkling 2's tower's first, epsilon 1e-5) and into others, at a
// width that is not a multiple of the 256 threads and at epsilons of 1e-5 and
// Qwen's tower's 1e-6, against fp64 within bf16 rounding, or fp32's; the row
// after the rows, NaN, stays.
void testLayerNorm(MetalBackend &backend) {
  for (const float epsilon : {1e-5F, 1e-6F})
    for (const FloatOutput element : {FloatOutput::BFloat16, FloatOutput::Float32})
      for (const bool inPlace : {false, true}) {
        if (inPlace && element != FloatOutput::Float32) continue;
        constexpr uint32_t rows = 5, width = 300;
        std::uniform_real_distribution<float> uniform(-2.0F, 2.0F);
        std::vector<float> x(size_t(rows) * width);
        for (float &value : x) value = uniform(rng) + 0.5F;
        const std::vector<float> weight = randomBf16(width, 1.0F), bias = randomBf16(width, 1.0F);
        const bool wide = element == FloatOutput::Float32;
        const MetalBuffer input = f32Buffer(backend, x, width);
        const MetalBuffer output = inPlace ? input
                                   : wide  ? f32Buffer(backend, std::vector<float>(size_t(rows) * width), width)
                                           : bf16Buffer(backend, std::vector<float>(size_t(rows) * width), width);
        CommandGraph graph;
        Normalization::addLayer(graph, backend.view(input, 0, size_t(rows) * width * 4),
                                {bf16Buffer(backend, weight), bf16Buffer(backend, bias)},
                                backend.view(output, 0, size_t(rows) * width * elementBytes(element)), width, rows,
                                epsilon, element);
        run(backend, graph);
        const std::string label = std::string(wide ? inPlace ? "fp32 in place" : "fp32" : "bf16") + " layer norm";
        for (uint32_t r = 0; r < rows; ++r) {
          double mean = 0.0, variance = 0.0;
          for (uint32_t c = 0; c < width; ++c) mean += x[r * width + c];
          mean /= width;
          for (uint32_t c = 0; c < width; ++c) variance += (x[r * width + c] - mean) * (x[r * width + c] - mean);
          variance /= width;
          for (uint32_t c = 0; c < width; ++c) {
            const double expected = (x[r * width + c] - mean) / std::sqrt(variance + epsilon) * weight[c] + bias[c];
            const double got = wide ? static_cast<const float *>(output.contents())[r * width + c]
                                    : bf16ToFloat(static_cast<const uint16_t *>(output.contents())[r * width + c]);
            require(std::fabs(got - expected) <= (wide ? 0x1p-20 : 0x1p-8) * std::fabs(expected) + 1e-4,
                    label + " row " + std::to_string(r) + " column " + std::to_string(c) + ": " +
                        std::to_string(got) + " vs " + std::to_string(expected));
          }
        }
        const auto *tail = static_cast<const uint8_t *>(output.contents()) + size_t(rows) * width * elementBytes(element);
        require(wide ? std::isnan(*reinterpret_cast<const float *>(tail))
                     : *reinterpret_cast<const uint16_t *>(tail) == kNan,
                label + ": a write reached past its rows");
      }
  constexpr uint32_t rows = 3, width = 260;
  const MetalBuffer input = sharedBuffer(backend, size_t(rows) * width * 4);
  const MetalBuffer weight = sharedBuffer(backend, size_t(width) * 2), bias = sharedBuffer(backend, size_t(width) * 2);
  const MetalBuffer output = sharedBuffer(backend, size_t(rows) * width * 2);
  using splash::test::requireExtent;
  requireExtent(backend, input, input.sizeBytes(), 4, "norm input", [&](CommandGraph &g, const MetalBuffer &v) {
    Normalization::addLayer(g, v, {weight, bias}, output, width, rows, 1e-5F);
  });
  requireExtent(backend, output, output.sizeBytes(), 2, "norm output", [&](CommandGraph &g, const MetalBuffer &v) {
    Normalization::addLayer(g, input, {weight, bias}, v, width, rows, 1e-5F);
  });
  requireExtent(backend, input, input.sizeBytes(), 4, "norm output", [&](CommandGraph &g, const MetalBuffer &v) {
    Normalization::addLayer(g, input, {weight, bias}, v, width, rows, 1e-5F, FloatOutput::Float32);
  });
  // The weights' contract (Normalization::requireLayer).
  requireExtent(backend, weight, weight.sizeBytes(), 2, "norm weight", [&](CommandGraph &g, const MetalBuffer &v) {
    Normalization::requireLayer({v, bias}, width, "norm");
    Normalization::addLayer(g, input, {v, bias}, output, width, rows, 1e-5F);
  });
  requireExtent(backend, bias, bias.sizeBytes(), 2, "norm bias", [&](CommandGraph &g, const MetalBuffer &v) {
    Normalization::requireLayer({weight, v}, width, "norm");
    Normalization::addLayer(g, input, {weight, v}, output, width, rows, 1e-5F);
  });
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: dense-linear METALLIB");
      MetalBackend backend(argv[1]);
      const DenseLinear dense(backend.capabilities());
      for (const DenseEpilogue epilogue : kEpilogues)
        for (const uint32_t rows : {1U, 5U, 63U, 64U, 65U, 130U}) {
          (void)testProjection(backend, dense, rows, 128, 128, epilogue);
          (void)testProjection(backend, dense, rows, 384, 512, epilogue);
        }
      for (const DenseEpilogue epilogue : kEpilogues)
        for (const uint32_t rows : {1U, 5U, 63U, 64U, 65U, 130U, 257U}) {
          (void)testProjection(backend, dense, rows, 128, 128, epilogue, true, 17, true);
          (void)testProjection(backend, dense, rows, 384, 1024, epilogue, true, 17, true);
        }
      testTiles(backend);
      testOffsets(backend, dense);
      testProjectionExtents(backend, dense);
      testLayerNorm(backend);
      std::cout << "dense_linear_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "dense_linear_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
