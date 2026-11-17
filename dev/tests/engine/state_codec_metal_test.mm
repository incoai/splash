// The cached form of GDN recurrent layers (ops::StateCodec) against a CPU
// reference, byte for byte: the codes and bf16 scales of every block, and the
// values decoded from them, over fp32 and bf16 running states of every
// family's GDN rows, layers apart by their own strides, and of rows whose
// threads leave most of their threadgroup idle, which write nothing past
// their layer. A block is two threads' halves, so blocks whose largest
// magnitude lies in either half, in both, or nowhere (zeros) check the
// halves' shared scale. Every decoded value is within half its block's step
// of the value coded, and each case's state within the bound measured on the
// models: a relative error of 1%. A block that holds NaN or an infinity in
// either half restores as NaN, every value of it, and the blocks around it
// as the reference codes them.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/Bfloat16.hpp"
#include "ops/StateCodec.hpp"

#include "StateCodecReference.hpp"
#include "TestBuffers.hpp"
#include "TestFamilies.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::GdnStateFormat;
using splash::ops::StateCodec;
using splash::ops::StateCodecShape;
using splash::test::require;

constexpr uint32_t kBlock = SPLASH_STATE_CODEC_BLOCK;
using splash::test::codecBf16;
using splash::test::codecWiden;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

template <class T> T *data(const MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

void submit(MetalBackend &backend, const CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// A running state of `shape`: rows of values whose magnitudes vary by row,
// as value rows of a GDN state do, with blocks whose largest magnitude lies
// in the first half, the second, or both, and blocks of zeros.
std::vector<float> stateValues(const StateCodecShape &shape, uint64_t seed) {
  Random random(seed);
  std::vector<float> values(uint64_t{shape.layers} * shape.rows * shape.columns);
  for (uint64_t row = 0; row < uint64_t{shape.layers} * shape.rows; ++row) {
    const float magnitude = std::ldexp(1.0F, static_cast<int>(row % 23) - 14);
    for (uint32_t block = 0; block < shape.columns / kBlock; ++block) {
      float *values16 = values.data() + row * shape.columns + block * kBlock;
      const uint32_t kind = static_cast<uint32_t>(row * 7 + block) % 5;
      for (uint32_t i = 0; i < kBlock; ++i)
        values16[i] = kind == 4 ? 0.0F : magnitude * random.unit();
      // The largest magnitude in the first half, the second, or in both.
      if (kind == 0) values16[3] = 1.5F * magnitude;
      if (kind == 1) values16[kBlock - 2] = -1.5F * magnitude;
      if (kind == 2) values16[5] = values16[kBlock - 5] = 1.25F * magnitude;
    }
  }
  return values;
}

void runCase(MetalBackend &backend, uint32_t rows, uint32_t layers, GdnStateFormat format, uint64_t padding) {
  const std::string label = std::to_string(rows) + " rows x " + std::to_string(layers) + " layers " +
                            std::string(splash::ops::formatName(format)) + " state: ";
  const uint32_t columns = 128;
  const uint64_t element = splash::ops::elementBytes(format);
  StateCodecShape shape{layers, rows, columns, 0, 0, format};
  shape.stateLayerBytes = shape.stateBytes() + padding;
  shape.codesLayerBytes = shape.codesBytes() + padding;
  require(shape.valid(), label + "the shape is not valid");
  std::vector<float> values = stateValues(shape, rows * 31 + layers);
  if (format == GdnStateFormat::BFloat16)
    for (float &value : values) value = codecWiden(codecBf16(value));

  MetalBuffer state = splash::test::sharedBuffer(backend, uint64_t{layers} * shape.stateLayerBytes);
  MetalBuffer codes = splash::test::sharedBuffer(backend, uint64_t{layers} * shape.codesLayerBytes);
  MetalBuffer decoded = splash::test::sharedBuffer(backend, uint64_t{layers} * shape.stateLayerBytes);
  // What the codec leaves alone: the bytes between layers.
  constexpr uint8_t kUntouched = 0x5A;
  std::memset(codes.contents(), kUntouched, codes.sizeBytes());
  std::memset(decoded.contents(), kUntouched, decoded.sizeBytes());
  const uint64_t layerValues = uint64_t{rows} * columns;
  for (uint32_t layer = 0; layer < layers; ++layer)
    for (uint64_t i = 0; i < layerValues; ++i) {
      std::byte *at = data<std::byte>(state) + layer * shape.stateLayerBytes + i * element;
      const float value = values[layer * layerValues + i];
      if (format == GdnStateFormat::Float32)
        std::memcpy(at, &value, 4);
      else {
        const uint16_t narrow = codecBf16(value);
        std::memcpy(at, &narrow, 2);
      }
    }
  CommandGraph graph;
  StateCodec::addQuantize(graph, state, codes, shape);
  StateCodec::addDequantize(graph, codes, decoded, shape);
  require(graph.dispatches().size() == 2, label + "a codec is not one dispatch each way");
  submit(backend, graph);

  double squaredError = 0.0, squaredNorm = 0.0;
  for (uint32_t layer = 0; layer < layers; ++layer) {
    const auto *layerCodes = data<int8_t>(codes) + layer * shape.codesLayerBytes;
    const std::vector<uint8_t> reference =
        splash::test::stateCodesReference(values.data() + layer * layerValues, rows, columns);
    require(std::memcmp(layerCodes, reference.data(), reference.size()) == 0,
            label + "the codes and scales of layer " + std::to_string(layer) + " are not the reference's");
    const auto *layerScales = reinterpret_cast<const uint16_t *>(layerCodes + state_codec_scales_offset(rows, columns));
    const std::byte *layerDecoded = data<std::byte>(decoded) + layer * shape.stateLayerBytes;
    for (uint64_t index = 0; index < layerValues; ++index) {
      const float value = values[layer * layerValues + index];
      const float step = codecWiden(layerScales[index / kBlock]);
      const float exact = static_cast<float>(layerCodes[index]) * step;
      float got = 0.0F;
      if (format == GdnStateFormat::Float32) {
        std::memcpy(&got, layerDecoded + index * 4, 4);
        require(got == exact, label + "a decoded value is not its code times its scale");
      } else {
        uint16_t narrow = 0;
        std::memcpy(&narrow, layerDecoded + index * 2, 2);
        require(narrow == codecBf16(exact), label + "a decoded bf16 value is not its code times its scale, rounded");
        got = codecWiden(narrow);
      }
      // Half a step, and the bf16 rounding of a bf16 state's value.
      const double bound = 0.5 * step * (1.0 + 1e-6) +
                           (format == GdnStateFormat::BFloat16 ? std::ldexp(std::fabs(exact), -8) : 0.0);
      require(std::fabs(double(got) - value) <= bound,
              label + "a decoded value is further than half a step from the value coded");
      squaredError += (double(got) - value) * (double(got) - value);
      squaredNorm += double(value) * value;
    }
  }
  const auto untouched = [&](const MetalBuffer &buffer, uint64_t layerStride, uint64_t layerBytes) {
    for (uint32_t layer = 0; layer < layers; ++layer)
      for (uint64_t at = layer * layerStride + layerBytes; at < (layer + 1) * layerStride; ++at)
        if (data<uint8_t>(buffer)[at] != kUntouched)
          return false;
    return true;
  };
  require(untouched(codes, shape.codesLayerBytes, shape.codesBytes()) &&
              untouched(decoded, shape.stateLayerBytes, shape.stateBytes()),
          label + "the codec wrote past a layer");
  const double relative = std::sqrt(squaredError / squaredNorm);
  require(relative < 0.01, label + "the state's relative error exceeds 1%");
  std::cout << label << "codes, scales and decoded values are the reference's, relative error " << relative << '\n';
}

// A state that is not finite stays so: a block holding NaN, +inf or -inf in
// either thread's half restores as NaN, every value of it, where its largest
// finite magnitude, or zero, would make one a finite value the state never
// held; the blocks around it code as the reference codes them.
void nonFiniteBlocks(MetalBackend &backend, GdnStateFormat format) {
  const std::string label = std::string(splash::ops::formatName(format)) + " state with values not finite: ";
  constexpr uint32_t rows = 8, columns = 128, blocksPerRow = columns / kBlock;
  StateCodecShape shape{1, rows, columns, 0, 0, format};
  shape.stateLayerBytes = shape.stateBytes();
  shape.codesLayerBytes = shape.codesBytes();
  std::vector<float> values = stateValues(shape, 5);
  // Block by block: NaN, +inf and -inf, each in the first half and in the
  // second, alone, beside finite values or in a block otherwise of zeros.
  const float kinds[] = {NAN, INFINITY, -INFINITY};
  std::vector<bool> finite(rows * blocksPerRow, true);
  for (uint32_t kind = 0; kind < 6; ++kind) {
    const uint32_t block = 3 * kind + 1;
    float *at = values.data() + block * kBlock;
    if (kind >= 3) std::fill(at, at + kBlock, 0.0F);
    at[kind % 2 ? kBlock - 3 : 2] = kinds[kind % 3];
    finite[block] = false;
  }
  if (format == GdnStateFormat::BFloat16)
    for (float &value : values) value = codecWiden(codecBf16(value));
  const uint64_t element = splash::ops::elementBytes(format);
  MetalBuffer state = splash::test::sharedBuffer(backend, shape.stateBytes());
  MetalBuffer codes = splash::test::sharedBuffer(backend, shape.codesBytes());
  MetalBuffer decoded = splash::test::sharedBuffer(backend, shape.stateBytes());
  for (uint64_t i = 0; i < values.size(); ++i) {
    if (format == GdnStateFormat::Float32)
      std::memcpy(data<std::byte>(state) + i * element, &values[i], 4);
    else {
      const uint16_t narrow = codecBf16(values[i]);
      std::memcpy(data<std::byte>(state) + i * element, &narrow, 2);
    }
  }
  CommandGraph graph;
  StateCodec::addQuantize(graph, state, codes, shape);
  StateCodec::addDequantize(graph, codes, decoded, shape);
  submit(backend, graph);
  // The reference of the finite blocks, the others zeros.
  std::vector<float> finiteValues = values;
  for (uint32_t block = 0; block < finite.size(); ++block)
    if (!finite[block]) std::fill_n(finiteValues.begin() + block * kBlock, kBlock, 0.0F);
  const std::vector<uint8_t> reference = splash::test::stateCodesReference(finiteValues.data(), rows, columns);
  const auto *scales = reinterpret_cast<const uint16_t *>(data<uint8_t>(codes) + state_codec_scales_offset(rows, columns));
  const auto *referenceScales =
      reinterpret_cast<const uint16_t *>(reference.data() + state_codec_scales_offset(rows, columns));
  for (uint32_t block = 0; block < finite.size(); ++block) {
    const std::string where = label + "block " + std::to_string(block) + " ";
    for (uint32_t i = 0; i < kBlock; ++i) {
      const uint64_t index = uint64_t{block} * kBlock + i;
      float got = 0.0F;
      if (format == GdnStateFormat::Float32)
        std::memcpy(&got, data<std::byte>(decoded) + index * 4, 4);
      else {
        uint16_t narrow = 0;
        std::memcpy(&narrow, data<std::byte>(decoded) + index * 2, 2);
        got = codecWiden(narrow);
      }
      if (!finite[block])
        require(std::isnan(got), where + "restored a value as " + std::to_string(got) + ", not NaN");
      else
        require(data<uint8_t>(codes)[index] == reference[index] && std::isfinite(got),
                where + "of finite values did not code as the reference beside one that is not finite");
    }
    if (finite[block])
      require(scales[block] == referenceScales[block], where + "has another scale than the reference's");
  }
  std::cout << label << "its blocks that are not restore as NaN\n";
}

// Each buffer at the extent the codec reaches and one element short: the
// last layer's values and its codes and scales.
void bufferExtents(MetalBackend &backend, GdnStateFormat format) {
  StateCodecShape shape{2, 256, 128, 0, 0, format};
  shape.stateLayerBytes = shape.stateBytes() + 4096;
  shape.codesLayerBytes = shape.codesBytes() + 4096;
  const uint64_t stateExtent = shape.stateLayerBytes + shape.stateBytes();
  const uint64_t codesExtent = shape.codesLayerBytes + shape.codesBytes();
  const MetalBuffer state = splash::test::sharedBuffer(backend, stateExtent);
  const MetalBuffer codes = splash::test::sharedBuffer(backend, codesExtent);
  const uint64_t element = splash::ops::elementBytes(format);
  splash::test::requireExtent(backend, state, stateExtent, element, "state codec state",
                              [&](CommandGraph &graph, const MetalBuffer &buffer) {
                                StateCodec::addQuantize(graph, buffer, codes, shape);
                              });
  splash::test::requireExtent(backend, codes, codesExtent, 2, "state codec codes",
                              [&](CommandGraph &graph, const MetalBuffer &buffer) {
                                StateCodec::addDequantize(graph, buffer, state, shape);
                              });
}

void run(const std::string &metallib) {
  MetalBackend backend(metallib);
  for (GdnStateFormat format : {GdnStateFormat::Float32, GdnStateFormat::BFloat16}) {
    bufferExtents(backend, format);
    for (const splash::ops::GdnShape &gdn : splash::test::familyGdnShapes()) {
      const uint32_t rows = gdn.valueHeads * gdn.headDimension;
      runCase(backend, rows, 1, format, 0);
      runCase(backend, rows, 3, format, 16384);
    }
    // Rows whose threads leave a threadgroup of 256 mostly idle: 24 threads
    // in one, and 296 over two.
    runCase(backend, 3, 3, format, 4096);
    runCase(backend, 37, 3, format, 4096);
    nonFiniteBlocks(backend, format);
  }
  std::cout << "state_codec_metal_test passed\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: state_codec_metal_test <metallib>\n";
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
