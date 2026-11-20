// The prefill FFN's Neural Engine split (ops/AneFfn.cpp, kernels/prefill/ane_ffn.metal):
// - inputs: ane_ffn_rotate and ane_ffn_pack turn bf16 rows into the ANE's channel-major int8 segments at their stride
//   and the rows' scales, as a CPU rotation of the rows does;
// - weights: ane_ffn_row_scale_gguf and ane_ffn_weights_gguf turn the rows of every image format (the GGUF formats
//   and MLX's affine ones) into int8 rows at their stride and their scales at theirs, rotated in either block, as a
//   CPU rotation of their dequantized values does; rows of zeros take a normal scale and codes of 0;
// - join: ane_ffn_join adds the ANE's channel-major partial rows, scaled in fp32, to the chunk's output rows and to no
//   others, and flags a value of the chunk's rows that is not finite;
// - the split: which layers, units and chunks it takes; for three layers of MLX af4g64 and of mixed GGUF formats,
//   Metal allocates what plannedBytes plans, and chunks of the fewest rows, of a count between two functions and of the
//   most rows compute what the GPU computes alone within int8's error, as do quiet rows, whose intermediate values
//   fp16 barely holds, and rows of a hidden channel of about 1e5; commands in two pipelined steps compute bit for bit
//   what one step computes, and a step joins before the next and the commit; verify() passes every function of either
//   model and fails a function bound to another's procedure; layers it is given out of order are refused, and it keeps
//   what it needs of the layers it was built from; an infinity in the ANE's output stops it; its program unloaded for
//   the idle release and loaded again computes what it computed before, and a command encoded or in flight refuses the
//   release;
// - faults: each fault of the ANE's evaluations (ane/ProgramInstrumentation.hpp) at each layer of chunks of three
//   sizes, and a program that does not load again after the idle release, stop the split without failing the Metal
//   command or throwing, and the GPU alone then computes what it computes on its own; so does a Neural Engine slower
//   than the GPU alone, through the breaker, after a window of commands whose results stay usable; an evaluation that
//   signals the shared event below its value leaves the value as it is;
// - the program (runtime/ane/Program.mm): an invalid MIL, a function it lacks and a binding short of an input fail as
//   std::exceptions; a limit that runs out and an interrupted wait end the wait, after which a program still comes up;
//   unload() and load() round-trip, and load() does not compile; the cache writes a file of other bytes again and
//   leaves no partial file when it cannot write one.
// The kernels round where the CPU may not (fast-math rsqrt and division, fused multiply-adds), so an int8 value may
// differ by one and a scale by its last bit. The split runs on the Neural Engine, which every Apple Silicon Mac has.
// Metal's validation layer wraps the shared event that orders the split's GPU and ANE work in one the ANE cannot
// share, so `kernels` runs the first three checks, which test-engine-metal validates, and `split`, `faults` and
// `program` the others. The binary links the instrumented Program, whose faults only `faults` and two checks of
// `split` arm.
#include "AneProgramFixture.hpp"
#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "GgufFormatReference.hpp"
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestFiles.hpp"
#include "ane/ProgramInstrumentation.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/QuantFormat.h"
#include "ops/AneFfn.hpp"
#include "ops/Linear.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// What the program checks send the private AppleNeuralEngine client beside
// ane::Program.
@protocol AneTestModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
@end
@protocol AneTestClient
+ (id)sharedConnection;
- (void)purgeCompiledModel:(id)model;
@end

using namespace splash;
using namespace splash::ops;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

namespace {

// A hidden size of two segments of the ANE programs (2560 channels each) in whole groups of ane_ffn_rotate (1024),
// and an intermediate size whose ANE share, kAneUnits of its 12 channel units, spans two segments of down's inputs.
constexpr uint32_t kHidden = 5120, kSegment = 2560, kIntermediate = 6144, kLayers = 3, kAneUnits = 6;
// The most the split's output may differ from the GPU's alone, as the RMS of their difference relative to the GPU's:
// int8 leaves about 1%.
constexpr double kMaximumError = 0.05;
// What a kernel must leave as it was.
constexpr uint8_t kUntouched = 0x5A;
constexpr uint16_t kUntouchedHalf = 0x5A5A;

std::mt19937 rng(20261003);
int failures = 0;
int sectionFailures = 0;

// Names one failure; the first few of a section are printed.
void fail(const std::string &what) {
  constexpr int kShown = 8;
  if (sectionFailures++ < kShown) std::cout << "  FAIL " << what << '\n';
  ++failures;
}
void section(const std::string &what) {
  std::cout << what << (sectionFailures ? " FAIL (" + std::to_string(sectionFailures) + " failures)" : " ok") << '\n';
  sectionFailures = 0;
}

MetalBuffer filled(MetalBackend &backend, uint64_t bytes, uint8_t value) {
  MetalBuffer buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), value, bytes);
  return buffer;
}
template <class T>
MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = test::sharedBuffer(backend, values.size() * sizeof(T));
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}
template <class T>
const T *contents(const MetalBuffer &buffer) {
  return static_cast<const T *>(buffer.contents());
}
void run(MetalBackend &backend, const CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
}

float toHalf(float value) { return float(_Float16(value)); }
float fromHalf(uint16_t bits) { return float(std::bit_cast<_Float16>(bits)); }
int8_t toInt8(float value) {
  return int8_t(std::clamp(std::rint(value), -ANE_FFN_INT8_PEAK, ANE_FFN_INT8_PEAK));
}
bool near(int8_t got, int8_t expected) { return std::abs(int(got) - int(expected)) <= 1; }
bool near(float got, float expected) { return std::fabs(got - expected) <= 0x1p-9f * std::fabs(expected); }

// v -> sign (H v) / sqrt(n) on each block of n values, H the Sylvester Hadamard matrix: the kernels' fp32 butterflies
// in their order.
void rotate(std::span<float> values, uint32_t n, const std::vector<float> &sign) {
  const float norm = 1.0f / std::sqrt(float(n));
  for (size_t begin = 0; begin < values.size(); begin += n) {
    float *v = values.data() + begin;
    for (uint32_t span = 1; span < n; span <<= 1)
      for (uint32_t i = 0; i < n; ++i)
        if (!(i & span)) {
          const float a = v[i], b = v[i + span];
          v[i] = a + b;
          v[i + span] = a - b;
        }
    for (uint32_t i = 0; i < n; ++i) v[i] *= sign[i] * norm;
  }
}
float peak(std::span<const float> values) {
  float result = 0.0f;
  for (const float value : values) result = std::max(result, std::fabs(value));
  return result;
}

// ---------------------------------------------------------------- inputs
// Rows of normal values, one in 997 forty times larger, and a row of zeros, which takes the least scale and codes of
// 0, as the ANE's two int8 input segments, each channel's rows at a stride past them.
void inputs(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign) {
  constexpr uint32_t kRows = 64, kStride = 96, kZeroRow = 7;
  std::normal_distribution<float> normal;
  std::vector<uint16_t> x(uint64_t{kRows} * kHidden);
  for (uint64_t i = 0; i < x.size(); ++i)
    x[i] = i / kHidden == kZeroRow ? 0 : floatToBf16(normal(rng) * (i % 997 ? 1.0f : 40.0f));
  const MetalBuffer rotated = test::sharedBuffer(backend, x.size() * sizeof(uint16_t));
  const MetalBuffer scales = filled(backend, kRows * sizeof(uint16_t), kUntouched);
  CommandGraph graph;
  graph.add("ane_ffn_rotate", {upload(backend, x), signs, rotated, scales}, AneFfnRotateParams{kHidden},
            {kRows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  std::vector<MetalBuffer> packed;
  for (uint32_t segment = 0; segment < kHidden / kSegment; ++segment) {
    packed.push_back(filled(backend, uint64_t{kSegment} * kStride, kUntouched));
    graph.add("ane_ffn_pack", {rotated, packed.back()}, AneFfnPackParams{kHidden, segment * kSegment, kStride},
              {kRows / ANE_FFN_TILE, kSegment / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  }
  run(backend, graph);

  for (uint32_t row = 0; row < kRows; ++row) {
    std::vector<float> v(kHidden);
    for (uint32_t channel = 0; channel < kHidden; ++channel)
      v[channel] = bf16ToFloat(x[uint64_t{row} * kHidden + channel]);
    rotate(v, ANE_FFN_INPUT_BLOCK, sign);
    const float scale = std::max(peak(v), ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK, inverse = 1.0f / scale;
    if (!near(fromHalf(contents<uint16_t>(scales)[row]), toHalf(scale * ANE_FFN_INT8_UNIT)))
      fail("row " + std::to_string(row) + ": scale " + std::to_string(fromHalf(contents<uint16_t>(scales)[row])) +
           ", not " + std::to_string(toHalf(scale * ANE_FFN_INT8_UNIT)));
    for (uint32_t channel = 0; channel < kHidden; ++channel) {
      const int8_t got = contents<int8_t>(packed[channel / kSegment])[uint64_t{channel % kSegment} * kStride + row];
      const int8_t expected = toInt8(toHalf(v[channel] * inverse));
      if (!near(got, expected))
        fail("row " + std::to_string(row) + " channel " + std::to_string(channel) + ": " + std::to_string(got) +
             ", not " + std::to_string(expected));
    }
  }
  for (const MetalBuffer &segment : packed)
    for (uint32_t channel = 0; channel < kSegment; ++channel)
      for (uint32_t byte = kRows; byte < kStride; ++byte)
        if (contents<uint8_t>(segment)[uint64_t{channel} * kStride + byte] != kUntouched)
          fail("channel " + std::to_string(channel) + ": packed past the rows");
  if (!(fromHalf(contents<uint16_t>(scales)[kZeroRow]) >= 0x1p-14f))
    fail("a row of zeros took a scale that is not a normal half");
  section("inputs: 64 rows of 5120 bf16 values, one of zeros, as two int8 segments and the rows' scales");
}

// ---------------------------------------------------------------- weights
constexpr uint32_t kWeightRows = 1536, kWeightInputs = 2048;
// A [kWeightRows, kWeightInputs] projection's planes as the weight kernels bind them, its parameters and its values.
struct Source {
  std::string name;
  std::array<MetalBuffer, 3> planes;
  uint32_t groups = 0, format = 0;
  std::vector<float> values;
};

// A row of zero weights, which takes the least scale and codes of 0: native blocks of zero bytes, whose scales are 0.
constexpr uint32_t kZeroWeightRow = 700;

Source ggufSource(MetalBackend &backend, gguf_reference::Fmt format) {
  using namespace gguf_reference;
  std::vector<uint8_t> native = makeNative(format, kWeightRows, kWeightInputs, rng);
  const uint32_t nativeRow = rowBytes(format, kWeightInputs);
  std::fill_n(native.begin() + uint64_t{kZeroWeightRow} * nativeRow, nativeRow, uint8_t{0});
  const Packed packed = repack(format, native, kWeightRows, kWeightInputs, nullptr);
  const MetalBuffer meta = upload(backend, packed.meta);
  // A format without plane1 binds meta in its place (QuantizedSegment::plane1Slot).
  const MetalBuffer plane1 = kQuantFormats[format].plane1_bytes ? upload(backend, packed.w1) : meta;
  Source source{fmtName(format), {upload(backend, packed.w0), plane1, meta}, kWeightInputs / 32, uint32_t(format), {}};
  source.values.resize(uint64_t{kWeightRows} * kWeightInputs);
  for (uint32_t row = 0; row < kWeightRows; ++row)
    rowValues(format, native.data() + uint64_t{row} * rowBytes(format, kWeightInputs), kWeightInputs,
              source.values.data() + uint64_t{row} * kWeightInputs);
  return source;
}

// Rows [row, kWeightRows) of `source` under their scales over inputs [scaled, kWeightInputs), and their int8 values
// over inputs [input, kWeightInputs), rotated in blocks of `block` from `scaled`: as the split stages the rows of
// gate and up the ANE takes over a segment of the hidden inputs (block 128), or the rows of down over a segment of
// the ANE's inputs (block 512). Each row's values lie a stride apart, and its scales' copies another. The ANE's
// channels lie in descending units of 512, the last first: gate's and up's rows, and down's inputs.
uint32_t descending(uint32_t index, uint32_t span) {
  constexpr uint32_t block = ANE_FFN_INTERMEDIATE_BLOCK;
  return span - (index / block + 1) * block + index % block;
}
void weights(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign, const Source &source,
             uint32_t block, uint32_t row, uint32_t scaled, uint32_t input) {
  const uint32_t rows = kWeightRows - row, width = kWeightInputs - input, stride = width + 64, scaleStride = 3;
  const std::string variant = block == ANE_FFN_INPUT_BLOCK ? "_inputs" : "_intermediate";
  const MetalBuffer rowScales = test::sharedBuffer(backend, rows * sizeof(uint16_t));
  const MetalBuffer output = filled(backend, uint64_t{rows} * stride, kUntouched);
  const MetalBuffer scales = filled(backend, uint64_t{rows} * scaleStride * sizeof(uint16_t), kUntouched);
  const auto &[a, b, c] = source.planes;
  CommandGraph graph;
  graph.add("ane_ffn_row_scale_gguf" + variant, {a, b, c, rowScales, signs},
            AneFfnWeightParams{source.groups, row, scaled, kWeightInputs - scaled, 0, 0, source.format, rows},
            {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  graph.add("ane_ffn_weights_gguf" + variant, {a, b, c, rowScales, output, scales, signs},
            AneFfnWeightParams{source.groups, row, input, width, stride, scaleStride, source.format, rows},
            {rows / ANE_FFN_WEIGHT_ROWS, width / block, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  run(backend, graph);

  const std::string label = source.name + variant;
  const bool rowsDescend = block == ANE_FFN_INPUT_BLOCK;
  for (uint32_t r = 0; r < rows; ++r) {
    const auto first = source.values.begin() + uint64_t{row + (rowsDescend ? descending(r, rows) : r)} * kWeightInputs;
    std::vector<float> v(first + scaled, first + kWeightInputs);
    rotate(v, block, sign);
    const uint16_t bits = contents<uint16_t>(rowScales)[r];
    const float rowScale = fromHalf(bits);
    const float expected = toHalf(std::max(peak(v), ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK * ANE_FFN_INT8_UNIT);
    if (peak(v) == 0.0f && (!(rowScale >= 0x1p-14f) ||
                            std::ranges::any_of(std::span(contents<int8_t>(output) + uint64_t{r} * stride, width),
                                                [](int8_t value) { return value != 0; })))
      fail(label + " row " + std::to_string(r) + ": a row of zeros took a scale that is not a normal half or codes "
                   "other than 0");
    if (!near(rowScale, expected))
      fail(label + " row " + std::to_string(r) + ": scale " + std::to_string(rowScale) + ", not " +
           std::to_string(expected));
    const uint16_t *copy = contents<uint16_t>(scales) + uint64_t{r} * scaleStride;
    if (copy[0] != bits || copy[1] != kUntouchedHalf || copy[2] != kUntouchedHalf)
      fail(label + " row " + std::to_string(r) + ": scale copy");
    const float inverse = ANE_FFN_INT8_UNIT / rowScale;
    const int8_t *values = contents<int8_t>(output) + uint64_t{r} * stride;
    for (uint32_t i = 0; i < width; ++i) {
      const int8_t value = toInt8(v[input - scaled + (rowsDescend ? i : descending(i, width))] * inverse);
      if (!near(values[i], value))
        fail(label + " row " + std::to_string(r) + " input " + std::to_string(input + i) + ": " +
             std::to_string(values[i]) + ", not " + std::to_string(value));
    }
    for (uint32_t byte = width; byte < stride; ++byte)
      if (uint8_t(values[byte]) != kUntouched) fail(label + " row " + std::to_string(r) + ": written past its values");
  }
}

void weights(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign) {
  for (int format = 0; format < gguf_reference::FMT_COUNT; ++format) {
    const Source source = ggufSource(backend, gguf_reference::Fmt(format));
    weights(backend, signs, sign, source, ANE_FFN_INPUT_BLOCK, 512, 0, 512);
    weights(backend, signs, sign, source, ANE_FFN_INTERMEDIATE_BLOCK, 0, 512, 1024);
  }
  section("weights: " + std::to_string(gguf_reference::FMT_COUNT) +
          " image formats as int8 rows and scales, rotated in blocks of 128 and 512, the ANE's channels in descending "
          "units; a row of zeros as codes of 0");
}

// ---------------------------------------------------------------- join
constexpr uint16_t kHalfInfinity = 0x7c00, kHalfNaN = 0x7e00;

// ane_ffn_join of `partial` ([kHidden + 1][stride] fp16, its last row the tokens' intermediate scales) and the
// tokens' input scales into `output`, over a chunk of `rows` rows: the status word it leaves.
uint32_t joined(MetalBackend &backend, const MetalBuffer &output, const std::vector<uint16_t> &partial,
                const std::vector<uint16_t> &tokenScale, uint32_t stride, uint32_t rows) {
  const MetalBuffer status = filled(backend, sizeof(uint32_t), 0);
  CommandGraph graph;
  graph.add("ane_ffn_join", {output, upload(backend, partial), upload(backend, tokenScale), status},
            AneFfnJoinParams{kHidden, stride, rows},
            {(rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE, kHidden / ANE_FFN_TILE, 1},
            {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  run(backend, graph);
  return *contents<uint32_t>(status);
}

// The partial rows of a chunk, times each token's intermediate and input scales in fp32, added to its output rows and
// to no others; a value of the chunk's rows that is not finite sets the status word, and one past them does not.
void join(MetalBackend &backend) {
  constexpr uint32_t kRows = 70, kTiles = (kRows + ANE_FFN_TILE - 1) / ANE_FFN_TILE, kStride = 128;
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> scale(0.5f, 2.0f);
  std::vector<uint16_t> output(uint64_t{kTiles} * ANE_FFN_TILE * kHidden), partial(uint64_t{kHidden + 1} * kStride);
  std::vector<uint16_t> tokenScale(kStride);
  for (uint16_t &value : output) value = floatToBf16(normal(rng) * 64.0f);
  for (uint16_t &value : partial) value = std::bit_cast<uint16_t>(_Float16(normal(rng)));
  for (uint32_t row = 0; row < kStride; ++row) {
    partial[uint64_t{kHidden} * kStride + row] = std::bit_cast<uint16_t>(_Float16(scale(rng) * 1024.0f));
    tokenScale[row] = std::bit_cast<uint16_t>(_Float16(scale(rng) * 64.0f));
  }
  const MetalBuffer result = upload(backend, output);
  if (joined(backend, result, partial, tokenScale, kStride, kRows)) fail("finite values set the status word");
  for (uint32_t row = 0; row < kTiles * ANE_FFN_TILE; ++row)
    for (uint32_t channel = 0; channel < kHidden; ++channel) {
      const uint64_t index = uint64_t{row} * kHidden + channel;
      const float before = bf16ToFloat(output[index]), value = fromHalf(partial[uint64_t{channel} * kStride + row]),
                  scaled = fromHalf(partial[uint64_t{kHidden} * kStride + row]) * fromHalf(tokenScale[row]);
      // The kernel may fuse the multiply and the add.
      const uint16_t got = contents<uint16_t>(result)[index];
      const bool sum =
          got == floatToBf16(before + value * scaled) || got == floatToBf16(std::fma(value, scaled, before));
      if (row >= kRows ? got != output[index] : !sum)
        fail("row " + std::to_string(row) + " channel " + std::to_string(channel) +
             (row < kRows ? ": not the scaled sum" : ": past the chunk, changed"));
    }

  // A value that is not finite: of the chunk's rows, a partial value, an intermediate scale and an input scale each
  // set the status word; past them, none does.
  const auto poisoned = [&](uint64_t index, uint16_t bits, bool token) {
    std::vector<uint16_t> values = token ? tokenScale : partial;
    values[index] = bits;
    return joined(backend, upload(backend, output), token ? partial : values, token ? values : tokenScale, kStride,
                  kRows);
  };
  if (poisoned(uint64_t{100} * kStride + kRows - 1, kHalfInfinity, false) != 1)
    fail("an infinite partial value did not set the status word");
  if (poisoned(uint64_t{kHidden} * kStride + 3, kHalfNaN, false) != 1)
    fail("a NaN intermediate scale did not set the status word");
  if (poisoned(5, kHalfInfinity | 0x8000, true) != 1) fail("an infinite input scale did not set the status word");
  if (poisoned(uint64_t{100} * kStride + kRows, kHalfInfinity, false) ||
      poisoned(uint64_t{kHidden} * kStride + kRows, kHalfNaN, false) || poisoned(kRows, kHalfInfinity, true))
    fail("a value past the chunk's rows set the status word");
  section("join: the scaled partial rows of a 70-row chunk added to its output rows and to no others; a value that "
          "is not finite sets the status word");
}

// ---------------------------------------------------------------- the split
// Three FFN layers: gate and up [kIntermediate, kHidden], down [kHidden, kIntermediate].
struct Model {
  std::string name;
  std::vector<Projection> projections;
  std::vector<SwiGluProjections> layers;
};
// The layers over the model's projections, which stay where they are.
Model withLayers(Model model) {
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    model.layers.push_back(
        {&model.projections[3 * layer], &model.projections[3 * layer + 1], &model.projections[3 * layer + 2]});
  return model;
}
constexpr std::array<LinearMatrix, 3> kMatrices{
    {{kIntermediate, kHidden}, {kIntermediate, kHidden}, {kHidden, kIntermediate}}};

// Layers of `formats`, gate's, up's and down's of each, of weights of a model's magnitude, whose FFN values the
// ANE's fp16 holds as it holds a model's: an MLX format's scales from its range (scaleRange), a GGUF format's a 32nd
// of it. The fixture's random sub-block scales spread a row's values over a range no model's weights span, more so in
// other formats (Q6_K's split error is 4.8% on them, against 0.6% for Q5_K); `weights` checks every format's int8
// rows.
Model blockModel(MetalBackend &backend, std::string name,
                 const std::array<std::array<gguf_reference::Fmt, 3>, kLayers> &formats) {
  using namespace gguf_reference;
  Model model{std::move(name), {}, {}};
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    for (uint32_t index = 0; index < 3; ++index) {
      const Fmt format = formats[layer][index];
      const auto [outputs, inputs] = kMatrices[index];
      std::uniform_real_distribution<float> range = scaleRange(format);
      const std::vector<uint8_t> native =
          affine(format) ? makeAffineNative(format, outputs, inputs, rng, [&] { return range(rng); })
                         : makeNative(format, outputs, inputs, rng, [&] { return f2h(range(rng) / 32); });
      const Packed packed = repack(format, native, outputs, inputs, nullptr);
      model.projections.emplace_back(
          outputs, inputs,
          BlockWeights{{QuantizedSegment::planes(
              format, outputs, inputs, upload(backend, packed.w0),
              kQuantFormats[format].plane1_bytes ? upload(backend, packed.w1) : MetalBuffer{},
              upload(backend, packed.meta))}});
    }
  return withLayers(std::move(model));
}
// An MLX 4-bit target's layers.
Model mlxModel(MetalBackend &backend) {
  using gguf_reference::Fmt;
  constexpr auto kAf4 = Fmt(GGUF_FMT_AF4G64);
  return blockModel(backend, "MLX af4g64", {{{kAf4, kAf4, kAf4}, {kAf4, kAf4, kAf4}, {kAf4, kAf4, kAf4}}});
}
// Layers of mixed GGUF formats.
Model ggufModel(MetalBackend &backend) {
  using namespace gguf_reference;
  return blockModel(backend, "GGUF", {{{Q4K, Q4K, Q5K}, {Q5K, Q5K, Q4K}, {Q4K, Q5K, Q80}}});
}

// Which layers and shares the split takes.
void takes(const Model &mlx, const Model &gguf) {
  if (AneFfn::unsupported(mlx.layers) || AneFfn::unsupported(gguf.layers))
    fail("the split does not take the models");
  if (!AneFfn::unsupported({})) fail("the split takes no layers");
  // Hidden sizes in whole groups of ane_ffn_rotate, a last segment of the rest among them, but not others, and
  // layers of two shapes.
  const auto projection = [](uint32_t outputs, uint32_t inputs) {
    return Projection(outputs, inputs,
                      BlockWeights{{QuantizedSegment::planes(GGUF_FMT_AF4G64, outputs, inputs, {}, {}, {})}});
  };
  const auto layer = [&](uint32_t hidden) {
    return std::pair{projection(kIntermediate, hidden), projection(hidden, kIntermediate)};
  };
  const auto [gate, down] = layer(4096);
  const auto [unevenGate, unevenDown] = layer(4608);
  const std::vector<SwiGluProjections> narrow{{&gate, &gate, &down}}, uneven{{&unevenGate, &unevenGate, &unevenDown}},
      mixed{mlx.layers[0], narrow[0]};
  if (const char *reason = AneFfn::unsupported(narrow)) fail(std::string("the split refuses a hidden size of 4096: ") + reason);
  if (!AneFfn::unsupported(uneven)) fail("the split takes a hidden size of 4608, which ane_ffn_rotate does not");
  if (!AneFfn::unsupported(mixed)) fail("the split takes layers of two shapes");
  if (AneFfn::units(mlx.layers) != kIntermediate / 512) fail("the split moves other units than 512 channels");
  // Units that leave the GPU or the ANE no channels.
  for (const uint32_t aneUnits : {0u, AneFfn::units(mlx.layers)})
    test::rejects([&] { static_cast<void>(AneFfn::plannedBytes(mlx.layers, {aneUnits, aneUnits})); }, "no channels",
                  std::to_string(aneUnits) + " units leave the GPU or the ANE no channels, yet plan");
  section("takes: models of one splittable shape, and units that leave each part channels");
}

// A calibration is remembered under the layers' shapes and formats and the int8 activations the GPU's part plans with
// (AneFfn::calibrationKey): each setting and each model's layers have a key of their own, and a key repeats.
void keys(const DeviceCapabilities &device, const Model &mlx, const Model &gguf) {
  const Linear on(device, Int8Activations::On, StepKind::Prefill), off(device, Int8Activations::Off, StepKind::Prefill);
  const std::string mlxOn = AneFfn::calibrationKey(mlx.layers, on);
  if (mlxOn != AneFfn::calibrationKey(mlx.layers, Linear(device, Int8Activations::On, StepKind::Prefill)))
    fail("a calibration key does not repeat");
  if (mlxOn == AneFfn::calibrationKey(mlx.layers, off) ||
      AneFfn::calibrationKey(gguf.layers, on) == AneFfn::calibrationKey(gguf.layers, off))
    fail("int8 activations on and off share a calibration key");
  if (mlxOn == AneFfn::calibrationKey(gguf.layers, on)) fail("two models' layers share a calibration key");
  test::rejects([&] { static_cast<void>(AneFfn::calibrationKey({}, on)); }, "ANE FFN split has no layers",
                "layers the split does not take have a calibration key");
  section("keys: a calibration's key holds its layers and the int8-activation setting");
}

// The chunk's buffers of `rows` rows at most: the normalized rows and the FFN's scratch, and the hidden rows the
// layers' residuals and outputs alternate between.
struct Chunk {
  PrefillFfnBuffers ffn;
  std::array<MetalBuffer, 2> hidden;
};
Chunk chunk(MetalBackend &backend, uint32_t rows) {
  const auto bf16Rows = [&](uint32_t width) { return test::sharedBuffer(backend, uint64_t{rows} * width * 2); };
  // Chunks of at least kMinimumRows rows take no split scratch, but where the family runs the int8 prefill tile its
  // codes and scales (Linear::scratchSize).
  const Linear linear(backend.capabilities(), Int8Activations::On, StepKind::Prefill);
  LinearScratchSize size = linear.scratchSize({kIntermediate, kHidden}, SPLASH_PREFILL_TOKEN_BUDGET);
  size.include(linear.scratchSize({kHidden, kIntermediate}, SPLASH_PREFILL_TOKEN_BUDGET));
  const auto bytes = [&](uint64_t count) { return count ? test::sharedBuffer(backend, count) : MetalBuffer{}; };
  return {{bf16Rows(kHidden), bf16Rows(kIntermediate), bf16Rows(kIntermediate),
           {.input = bytes(size.input), .sums = bytes(size.sums)}},
          {bf16Rows(kHidden), bf16Rows(kHidden)}};
}
// Normal values scaled by row(r) into the leading rows of the chunk's normalized rows.
template <class Scale>
void fillRows(const Chunk &chunk, uint32_t rows, Scale row) {
  std::normal_distribution<float> normal;
  auto *values = static_cast<uint16_t *>(chunk.ffn.normalized.contents());
  for (uint64_t i = 0; i < uint64_t{rows} * kHidden; ++i) values[i] = floatToBf16(normal(rng) * row(i / kHidden));
}

// A forward of every layer's FFN of a chunk's `rows` normalized rows from a zero residual, split where `split` takes
// the chunk and on the GPU alone otherwise, as the target's prefill encodes it: in the two steps the split's first
// step's rows cut it into (AneFfn::firstStepRows), each step's buffers its rows of the chunk's, a step's join after
// the other step's FFN layer and before its own next (QwenTarget::addPrefillLayers): its bf16 output, whether the
// split's results were usable, and the command's milliseconds.
struct Forward {
  std::vector<uint16_t> bits;
  bool usable = true;
  double milliseconds = 0.0;
  [[nodiscard]] float operator[](uint64_t index) const { return bf16ToFloat(bits[index]); }
};
Forward forward(MetalBackend &backend, const Linear &linear, const Model &model, AneFfn *split, const Chunk &chunk,
                uint32_t rows) {
  const auto &[ffn, hidden] = chunk;
  std::memset(hidden[0].contents(), 0, hidden[0].sizeBytes());
  CommandGraph graph;
  const bool splits = split && split->splits(rows);
  if (split) static_cast<void>(split->begin(split->steps(rows)));
  // Each step's rows from `first` of the chunk's: its normalized and hidden rows.
  struct Step {
    uint32_t first, rows;
    PrefillFfnBuffers ffn;
    std::array<MetalBuffer, 2> hidden;
  };
  const uint32_t firstRows = splits ? split->firstStepRows(rows) : rows;
  std::vector<Step> steps;
  for (const auto &[first, count] : {std::pair{0u, firstRows}, std::pair{firstRows, rows - firstRows}}) {
    if (!count) continue;
    const auto view = [&](const MetalBuffer &buffer) {
      const uint64_t offset = uint64_t{first} * kHidden * 2;
      return backend.view(buffer, offset, buffer.sizeBytes() - offset);
    };
    PrefillFfnBuffers rowsFfn = ffn;
    rowsFfn.normalized = view(ffn.normalized);
    steps.push_back({first, count, rowsFfn, {view(hidden[0]), view(hidden[1])}});
  }
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    for (const Step &step : steps) {
      const MetalBuffer &residual = step.hidden[layer & 1], &output = step.hidden[(layer & 1) ^ 1];
      if (splits) {
        split->join(graph);
        split->add(graph, layer, step.ffn, residual, output, step.rows);
      } else {
        linear.addSwiGlu(graph, model.layers[layer], step.ffn, residual, output, step.rows);
      }
    }
    if (splits && (steps.size() == 1 || layer + 1 == kLayers)) split->join(graph);
  }
  Forward result;
  const auto start = AwakeClock::now();
  if (split)
    static_cast<void>(split->commit(graph, {}).wait());
  else
    run(backend, graph);
  result.milliseconds = millisecondsSince(start);
  result.usable = !split || split->finish();
  const uint16_t *bits = contents<uint16_t>(hidden[kLayers & 1]);
  result.bits.assign(bits, bits + uint64_t{rows} * kHidden);
  return result;
}

// The RMS of the split's output's difference from the GPU's alone, relative to the GPU's, over the elements `counts`
// takes, and whether the split's are finite.
template <class Counts>
std::pair<double, bool> difference(const Forward &expected, const Forward &got, Counts counts) {
  double difference = 0.0, magnitude = 0.0;
  bool finite = true;
  for (uint64_t i = 0; i < got.bits.size(); ++i) {
    if (!counts(i / kHidden, i % kHidden)) continue;
    finite &= std::isfinite(got[i]);
    const double delta = double(got[i]) - expected[i];
    difference += delta * delta;
    magnitude += double(expected[i]) * expected[i];
  }
  return {std::sqrt(difference / magnitude), finite};
}
std::pair<double, bool> difference(const Forward &expected, const Forward &got) {
  return difference(expected, got, [](uint64_t, uint64_t) { return true; });
}

// The split's layers, built from copies of the model's projections that are gone before it runs: it keeps what it
// needs of them. By default the ANE takes kAneUnits in both kinds of command, one step from the least rows and two
// from twice them.
constexpr AneFfn::Shares kShares{kAneUnits, kAneUnits};
constexpr AneFfn::Rows kBothRows{AneFfn::kMinimumRows, 2 * AneFfn::kMinimumRows};
// One step alone, from the least rows: a function of every rows.
constexpr AneFfn::Rows kOneStep{AneFfn::kMinimumRows, 0};
std::unique_ptr<AneFfn> splitOf(MetalBackend &backend, const Model &model, AneFfn::Shares shares = kShares,
                                AneFfn::Rows rows = kBothRows, std::optional<bool> blocks = std::nullopt) {
  const std::vector<Projection> copies = model.projections;
  std::vector<SwiGluProjections> layers;
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    layers.push_back({&copies[3 * layer], &copies[3 * layer + 1], &copies[3 * layer + 2]});
  return std::make_unique<AneFfn>(backend, Linear(backend.capabilities(), Int8Activations::On, StepKind::Prefill),
                                  layers, shares, rows, nullptr, blocks);
}

void split(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  const std::unique_ptr<AneFfn> split = splitOf(backend, model);
  if (split->share(AneFfn::Steps::One) != 0.5 || split->share(AneFfn::Steps::Two) != 0.5)
    fail(model.name + ": runs share " + std::to_string(split->share(AneFfn::Steps::One)));
  if (split->splits(AneFfn::kMinimumRows - 1) || !split->splits(AneFfn::kMinimumRows) ||
      !split->splits(AneFfn::kMaximumRows) || split->splits(AneFfn::kMaximumRows + 1))
    fail(model.name + ": the split takes other commands than those of 512 to 2048 rows");
  // The plan takes whole pages of each of its four buffers, of which Metal may report less.
  const uint64_t planned = AneFfn::plannedBytes(model.layers, kShares), allocated = split->allocatedBytes();
  if (allocated > planned || planned - allocated >= 4 * kHostPageBytes)
    fail(model.name + ": allocated " + std::to_string(allocated) + " bytes, planned " + std::to_string(planned));
  fillRows(chunk, AneFfn::kMaximumRows, [](uint64_t) { return 1.0f; });
  // The most rows first: later commands run on what earlier ones left in the split's surfaces.
  for (const uint32_t rows : {AneFfn::kMaximumRows, AneFfn::kMinimumRows, 700u}) {
    const Forward expected = forward(backend, linear, model, nullptr, chunk, rows);
    const Forward got = forward(backend, linear, model, split.get(), chunk, rows);
    const auto [error, finite] = difference(expected, got);
    std::cout << "  " << model.name << ", " << rows << " rows: " << 100.0 * error << "% RMS from the GPU alone, "
              << got.milliseconds << " ms\n";
    if (!got.usable || !finite || !(error <= kMaximumError))
      fail(model.name + ", " + std::to_string(rows) + " rows: " + std::to_string(100.0 * error) +
           "% RMS from the GPU alone" + (got.usable ? "" : ", unusable: " + split->reason()));
  }
  section("split: " + model.name + " layers at share 0.5 from layers gone before it runs, its planned memory, "
          "chunks of 2048, 512 and 700 rows");

  // Least rows of more leave the smaller commands to the GPU, as two steps alone leave those below 1,024 rows; rows no
  // function holds are refused: of one step below the least function or past the most, or of a program of both kinds
  // one step past 1,023 rows or two steps from other rows than 1,024, where only the halves have functions.
  {
    const std::unique_ptr<AneFfn> oneStep = splitOf(backend, model, {kAneUnits, 0}, kOneStep);
    oneStep->setRows({640, 0});
    if (oneStep->minimumRows() != 640 || oneStep->splits(639) || !oneStep->splits(640) ||
        oneStep->firstStepRows(2048) != 2048)
      fail(model.name + ": least rows of 640 took other commands than those of 640 rows or more in one step");
    using Rows = AneFfn::Rows;
    for (const Rows rows :
         {Rows{AneFfn::kMinimumRows - 1, 0}, Rows{AneFfn::kMaximumRows + 1, 0}, Rows{0, 0}, Rows{256, 1024}})
      test::rejects([&] { oneStep->setRows(rows); }, "no function of",
                    model.name + ": rows " + std::to_string(rows.oneStep) + "/" + std::to_string(rows.twoSteps) +
                        " were taken");
    if (split->steps(1023) != AneFfn::Steps::One || split->steps(1024) != AneFfn::Steps::Two ||
        split->firstStepRows(1023) != 1023)
      fail(model.name + ": commands from 1,024 rows did not run in two steps, or those below in one");
    for (const AneFfn::Rows rows : {AneFfn::Rows{AneFfn::kMinimumRows, 1152}, AneFfn::Rows{AneFfn::kMinimumRows, 896},
                                    AneFfn::Rows{AneFfn::kMinimumRows, 511}, AneFfn::Rows{1024, 1024}})
      test::rejects([&] { split->setRows(rows); }, "no function of",
                    model.name + ": rows no function of its program holds were taken");
    split->setRows({640, AneFfn::kTwoStepRows});
    if (split->minimumRows() != 640 || split->splits(639) || split->steps(1023) != AneFfn::Steps::One)
      fail(model.name + ": least rows of 640 beside two steps took other commands than those of 640 rows or more");
    // Two steps alone take the commands from 1,024 rows, in two steps, and leave those below to the GPU alone.
    split->setRows({0, AneFfn::kTwoStepRows});
    if (split->minimumRows() != AneFfn::kTwoStepRows || split->splits(AneFfn::kTwoStepRows - 1) ||
        !split->splits(AneFfn::kTwoStepRows) || !split->splits(AneFfn::kMaximumRows) ||
        split->firstStepRows(AneFfn::kTwoStepRows) != AneFfn::kMinimumRows ||
        split->firstStepRows(AneFfn::kMaximumRows) != AneFfn::kMaximumRows / 2)
      fail(model.name + ": two steps alone took other commands than those of 1,024 rows or more in two steps");
    for (const uint32_t rows : {AneFfn::kTwoStepRows - 1, AneFfn::kTwoStepRows, AneFfn::kMaximumRows}) {
      const Forward expected = forward(backend, linear, model, nullptr, chunk, rows);
      const Forward got = forward(backend, linear, model, split.get(), chunk, rows);
      const auto [error, finite] = difference(expected, got);
      if (!got.usable || !finite || !(error <= kMaximumError) ||
          (rows < AneFfn::kTwoStepRows && got.bits != expected.bits))
        fail(model.name + ", " + std::to_string(rows) + " rows of two steps alone: " +
             std::to_string(100.0 * error) + "% RMS from the GPU alone" + (got.usable ? "" : ", unusable"));
    }
    split->setRows(kBothRows);
  }
  // verify() checks every function, and both weight sets, against the leading rows of a full chunk on the GPU
  // alone, which computes every row alike whatever the chunk's rows.
  const Forward full = forward(backend, linear, model, nullptr, chunk, AneFfn::kMaximumRows);
  for (const uint32_t rows : {AneFfn::kMinimumRows, 700u})
    if (const Forward fewer = forward(backend, linear, model, nullptr, chunk, rows);
        !std::equal(fewer.bits.begin(), fewer.bits.end(), full.bits.begin()))
      fail(model.name + ": the GPU computed the rows of a chunk of " + std::to_string(rows) +
           " rows other than a full chunk's");
  const auto started = AwakeClock::now();
  const double error = split->verify(model.layers, chunk.ffn, chunk.hidden);
  std::cout << "  " << model.name << ": verified in " << millisecondsSince(started) << " ms, " << 100.0 * error
            << "% from the GPU alone on the Neural Engine's part\n";
  if (split->retired()) fail(model.name + ": verify stopped the split: " + split->reason());
  // Gate and up whole and in blocks, the form this chip does not run too (AneFfn.cpp): every function of each
  // within verify()'s bound of the GPU alone.
  for (const bool blocks : {false, true}) {
    const std::unique_ptr<AneFfn> form = splitOf(backend, model, kShares, kBothRows, blocks);
    const double formError = form->verify(model.layers, chunk.ffn, chunk.hidden);
    std::cout << "  " << model.name << ", gate and up " << (blocks ? "in blocks" : "whole") << ": " << 100.0 * formError
              << "% from the GPU alone on the Neural Engine's part\n";
  }
  section("split: " + model.name + " least chunk rows, a chunk's rows on the GPU alone as a full chunk's, and "
          "verify() of every function, gate and up whole and in blocks");
}

// A command the split takes from 1,024 rows runs as two steps, its halves, each on the function of its rows, whose layers
// interleave; every row computes what one step of the command's rows computes, whose function's rows differ: the ANE
// computes each row alike whatever its function's rows. A share of one step beside a larger share of two, which reads
// the leading rows of the same surfaces, computes what a split of that share alone computes, the larger share's two
// steps what that share's do alone. A step's join must come before the next add and before the commit.
void pipelined(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  const std::unique_ptr<AneFfn> two = splitOf(backend, model), one = splitOf(backend, model, {kAneUnits, 0}, kOneStep);
  if (two->firstStepRows(AneFfn::kMaximumRows) != AneFfn::kMaximumRows / 2 || two->firstStepRows(1100) != 550 ||
      two->firstStepRows(1023) != 1023 || two->firstStepRows(1024) != 512)
    fail(model.name + ": the split's first steps are not the halves of commands of 1,024 rows or more");
  fillRows(chunk, AneFfn::kMaximumRows, [](uint64_t) { return 1.0f; });
  for (const uint32_t rows : {AneFfn::kMaximumRows, 1100u, 2 * AneFfn::kMinimumRows}) {
    const Forward oneStep = forward(backend, linear, model, one.get(), chunk, rows);
    const Forward twoSteps = forward(backend, linear, model, two.get(), chunk, rows);
    const auto [error, finite] = difference(oneStep, twoSteps);
    std::cout << "  " << model.name << ", " << rows << " rows in two steps: " << 100.0 * error
              << "% RMS from one step, " << twoSteps.milliseconds << " ms against " << oneStep.milliseconds << " ms\n";
    if (!oneStep.usable || !twoSteps.usable || !finite || oneStep.bits != twoSteps.bits)
      fail(model.name + ", " + std::to_string(rows) + " rows: two steps computed other rows than one, " +
           std::to_string(100.0 * error) + "% RMS");
  }
  // Shares of 4 units in one step to 1,023 rows and of 6 in two from 1,024, against each share alone.
  constexpr uint32_t kSmall = 4;
  const std::unique_ptr<AneFfn> both = splitOf(backend, model, {kSmall, kAneUnits}, {AneFfn::kMinimumRows, 1024}),
                                small = splitOf(backend, model, {kSmall, 0}, kOneStep);
  for (const auto &[rows, alone] : {std::pair{700u, small.get()}, std::pair{1023u, small.get()},
                                    std::pair{AneFfn::kMaximumRows, two.get()}}) {
    const Forward expected = forward(backend, linear, model, alone, chunk, rows);
    const Forward got = forward(backend, linear, model, both.get(), chunk, rows);
    if (!expected.usable || !got.usable || expected.bits != got.bits)
      fail(model.name + ", " + std::to_string(rows) + " rows: a split of two shares computed other rows than its "
                                                     "share alone, " +
           std::to_string(100.0 * difference(expected, got).first) + "% RMS");
  }
  const auto &[ffn, hidden] = chunk;
  CommandGraph graph;
  static_cast<void>(two->begin(AneFfn::Steps::Two));
  two->add(graph, 0, ffn, hidden[0], hidden[1], AneFfn::kMinimumRows);
  test::rejects([&] { two->add(graph, 0, ffn, hidden[0], hidden[1], AneFfn::kMinimumRows); }, "before it joined",
                model.name + ": a step was added before the one before it joined");
  test::rejects([&] { static_cast<void>(two->commit(graph, {})); }, "did not join",
                model.name + ": a command whose step did not join was committed");
  static_cast<void>(one->begin(AneFfn::Steps::Two));
  test::rejects([&] { one->add(graph, 0, ffn, hidden[0], hidden[1], AneFfn::kMinimumRows); }, "no command of this kind",
                model.name + ": a split of one step took a command of two");
  static_cast<void>(two->begin(AneFfn::Steps::Two));
  section("pipelined: " + model.name + " commands of 2048, 1100 and 1024 rows in two steps compute what one step does, "
          "two shares in one program what each computes alone, and a step joins before the next and the commit");
}

// A program whose function of 640 rows is bound to the procedure of 512 rows leaves rows of the ANE's output
// unwritten, which verify() fails, and the split stops.
void misbound(MetalBackend &backend, const Model &model, const Chunk &chunk) {
  ane::ProgramInstrumentation::arm({.swappedBinding = {"ffn3072_640", "ffn3072_512"}});
  AneFfn split(backend, Linear(backend.capabilities(), Int8Activations::On, StepKind::Prefill), model.layers,
               {kAneUnits, 0}, kOneStep, nullptr);
  test::rejects([&] { static_cast<void>(split.verify(model.layers, chunk.ffn, chunk.hidden)); },
                "split of 640 rows failed", "verify() passed a function bound to another's procedure");
  if (!split.retired() || split.splits(AneFfn::kMaximumRows)) fail("a split that failed verify() did not stop");
  section("verify: a function bound to the procedure of fewer rows fails it");
}

// Layers added out of order and buffers short of a chunk's rows are refused before anything is encoded.
void refusals(MetalBackend &backend, const Model &model, const Chunk &chunk) {
  AneFfn split(backend, Linear(backend.capabilities(), Int8Activations::On, StepKind::Prefill), model.layers,
               {kAneUnits, 0}, kOneStep, nullptr);
  constexpr uint32_t kRows = AneFfn::kMinimumRows;
  const auto &[ffn, hidden] = chunk;
  CommandGraph graph;
  if (!split.begin(AneFfn::Steps::One)) fail("a new split does not begin a command");
  test::rejects([&] { split.add(graph, 1, ffn, hidden[0], hidden[1], kRows); }, "adds layer 1 after 0 layers",
                "the split added layer 1 first");
  split.add(graph, 0, ffn, hidden[0], hidden[1], kRows);
  split.join(graph);
  test::rejects([&] { split.add(graph, 2, ffn, hidden[1], hidden[0], kRows); }, "adds layer 2 after 1 layers",
                "the split skipped layer 1");
  const size_t encoded = graph.dispatches().size();
  const MetalBuffer shorter = backend.view(hidden[0], 0, uint64_t{kRows} * kHidden * 2 - 2);
  test::rejects([&] { split.add(graph, 1, ffn, hidden[1], shorter, kRows); }, "ANE FFN output buffer holds",
                "an output one element short was accepted");
  PrefillFfnBuffers narrow = ffn;
  narrow.normalized = backend.view(ffn.normalized, 0, uint64_t{kRows} * kHidden * 2 - 2);
  test::rejects([&] { split.add(graph, 1, narrow, hidden[1], hidden[0], kRows); }, "ANE FFN input buffer holds",
                "an input one element short was accepted");
  if (graph.dispatches().size() != encoded) fail("a refused layer encoded a dispatch");
  if (!split.begin(AneFfn::Steps::One)) fail("a split does not begin a command again");
  split.add(graph, 0, ffn, hidden[0], hidden[1], kRows);
  section("refusals: layers out of order and buffers short of the chunk's rows");
}

// Rows whose FFN intermediate peaks at about 2^-12 of their input's peak, and louder ones: rows of normal values
// scaled by 2^-j, j = row % 16. The ANE quantizes the intermediate rows against their peak or a floor
// (kIntermediateFloor in AneFfn.cpp) whose inverse fp16 holds. Every output is finite; the rows of each scale whose
// intermediate peaks at kQuietRatio of their input's or more are within the bound, which small gate pre-activations
// broke while the ANE took sigmoid's value (6.5% at 2^-2.6). Quieter rows' intermediate values approach fp16's least
// normal value, below which the ANE's share of them fades (to 70% of their output from the GPU alone at 2^-11), so
// their difference from the GPU alone is bounded against their input, as the residual stream they join holds it:
// within kQuietError.
constexpr double kQuietRatio = 0x1p-5, kQuietError = 0x1p-8;
void quietRows(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  constexpr uint32_t kRows = AneFfn::kMaximumRows, kScales = 16;
  const std::unique_ptr<AneFfn> split = splitOf(backend, model);
  fillRows(chunk, kRows, [](uint64_t row) { return std::ldexp(1.0f, -int(row % kScales)); });
  const Forward expected = forward(backend, linear, model, nullptr, chunk, kRows);
  // The last layer's intermediate rows, which the GPU alone leaves.
  const auto *intermediate = contents<uint16_t>(chunk.ffn.intermediate);
  const auto *normalized = contents<uint16_t>(chunk.ffn.normalized);
  std::array<std::vector<double>, kScales> ratios;
  for (uint32_t row = 0; row < kRows; ++row) {
    float h = 0.0f, x = 0.0f;
    for (uint32_t c = 0; c < kIntermediate; ++c)
      h = std::max(h, std::fabs(bf16ToFloat(intermediate[uint64_t{row} * kIntermediate + c])));
    for (uint32_t c = 0; c < kHidden; ++c)
      x = std::max(x, std::fabs(bf16ToFloat(normalized[uint64_t{row} * kHidden + c])));
    ratios[row % kScales].push_back(h / x);
  }
  const Forward got = forward(backend, linear, model, split.get(), chunk, kRows);
  if (!got.usable) fail("quiet rows: unusable: " + split->reason());
  bool reached = false;
  for (uint32_t scale = 0; scale < kScales; ++scale) {
    std::ranges::nth_element(ratios[scale], ratios[scale].begin() + ratios[scale].size() / 2);
    const double ratio = ratios[scale][ratios[scale].size() / 2];
    const auto rows = [&](uint64_t row, uint64_t) { return row % kScales == scale; };
    const auto [error, finite] = difference(expected, got, rows);
    // The difference against the rows' inputs, which the residual stream they join holds at their scale.
    double difference = 0.0, input = 0.0;
    for (uint64_t i = 0; i < got.bits.size(); ++i) {
      if (!rows(i / kHidden, 0)) continue;
      difference += (double(got[i]) - expected[i]) * (double(got[i]) - expected[i]);
      input += double(bf16ToFloat(normalized[i])) * bf16ToFloat(normalized[i]);
    }
    const double inputError = std::sqrt(difference / input);
    reached |= ratio >= 0x1p-13 && ratio <= 0x1p-11;
    std::cout << "  rows scaled by 2^-" << scale << ", intermediate peak 2^" << std::log2(ratio)
              << " of the input's: " << 100.0 * error << "% RMS from the GPU alone, 2^" << std::log2(inputError)
              << " of the input's RMS\n";
    if (!finite) fail("quiet rows: the split's output of rows scaled by 2^-" + std::to_string(scale) + " not finite");
    if (ratio >= kQuietRatio ? !(error <= kMaximumError) : !(inputError <= kQuietError))
      fail("quiet rows scaled by 2^-" + std::to_string(scale) + ": " + std::to_string(100.0 * error) +
           "% RMS from the GPU alone");
  }
  if (!reached) fail("quiet rows: no scale's intermediate peaks within 2^-13 to 2^-11 of its input's");
  section("quiet rows: all finite; intermediate rows down to 2^-5 of their input's peak within the bound, quieter "
          "ones within 2^-8 of their input");
}

// A hidden channel of about 1e5 on every row, from four of the ANE's intermediate channels of the last layer whose up
// rows are 1000 times larger, and the group of down's row of that channel over them larger to match: the ANE leaves
// its tokens' scales to the join's fp32, where its fp16 output at full scale overflowed.
void massiveChannel(MetalBackend &backend, const Linear &linear, const Chunk &chunk) {
  constexpr uint32_t kRows = AneFfn::kMaximumRows, kChannel = 1234, kFirst = 4096, kCount = 4;
  static_assert(kFirst >= kIntermediate / 2 && kFirst % 64 + kCount <= 64,
                "the channels are the ANE's at share 0.5 and lie in one group of down's inputs");
  const Model model = mlxModel(backend);
  // Scales the bf16 scales and biases of `row`'s 64-input groups [first, last) of the af4g64 `projection` by
  // `factor`: its meta holds a scale and a bias per row and group (metal/abi/QuantFormat.h).
  const auto scale = [](const Projection &projection, uint32_t row, uint32_t first, uint32_t last, float factor) {
    auto *meta = static_cast<uint16_t *>(projection.blocks().segments.front().meta.contents());
    for (uint32_t group = first; group < last; ++group) {
      uint16_t *parameters = meta + 2 * quant_tile_index(row, group, projection.inputSize / 64);
      for (uint32_t i = 0; i < 2; ++i) parameters[i] = floatToBf16(bf16ToFloat(parameters[i]) * factor);
    }
  };
  const Projection &up = *model.layers.back().up, &down = *model.layers.back().down;
  for (uint32_t channel = kFirst; channel < kFirst + kCount; ++channel) scale(up, channel, 0, kHidden / 64, 1000.0f);
  fillRows(chunk, kRows, [](uint64_t) { return 1.0f; });
  const auto peak = [&](const Forward &forward) {
    float result = 0.0f;
    for (uint32_t row = 0; row < kRows; ++row)
      result = std::max(result, std::fabs(forward[uint64_t{row} * kHidden + kChannel]));
    return result;
  };
  // Twice, as the rest of the channel's sum takes a little of it.
  for (int pass = 0; pass < 2; ++pass)
    scale(down, kChannel, kFirst / 64, kFirst / 64 + 1,
          1e5f / peak(forward(backend, linear, model, nullptr, chunk, kRows)));
  const Forward expected = forward(backend, linear, model, nullptr, chunk, kRows);
  const std::unique_ptr<AneFfn> split = splitOf(backend, model);
  const Forward got = forward(backend, linear, model, split.get(), chunk, kRows);
  const auto [error, finite] = difference(expected, got);
  const auto [channelError, channelFinite] =
      difference(expected, got, [](uint64_t, uint64_t channel) { return channel == kChannel; });
  std::cout << "  a channel of " << peak(expected) << " at most: " << 100.0 * channelError << "% RMS from the GPU "
            << "alone, " << 100.0 * error << "% over every channel\n";
  if (!(peak(expected) >= 5e4f && peak(expected) <= 2e5f)) fail("massive channel: the channel does not reach 1e5");
  if (!got.usable || !finite || !channelFinite || !(error <= kMaximumError) || !(channelError <= kMaximumError))
    fail("massive channel: " + std::to_string(100.0 * channelError) + "% RMS from the GPU alone" +
         (got.usable ? "" : ", unusable: " + split->reason()));
  section("massive channel: a hidden channel of about 1e5 from the ANE's channels finite and within the bound");
}

// The idle release: the program unloaded (release()) and loaded again (restore()) computes what it computed before,
// bit for bit; a release while a command is encoded, or committed and not finished, is refused.
void idle(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  constexpr uint32_t kRows = AneFfn::kMaximumRows;
  const std::unique_ptr<AneFfn> split = splitOf(backend, model, {kAneUnits, 0}, kOneStep);
  fillRows(chunk, kRows, [](uint64_t) { return 1.0f; });
  const Forward before = forward(backend, linear, model, split.get(), chunk, kRows);
  if (!split->release()) fail("idle: the program was not unloaded");
  const auto start = AwakeClock::now();
  split->restore();
  const double reload = millisecondsSince(start);
  const Forward after = forward(backend, linear, model, split.get(), chunk, kRows);
  std::cout << "  loaded again in " << reload << " ms\n";
  if (!before.usable || !after.usable || split->retired())
    fail("idle: unusable: " + split->reason());
  else if (after.bits != before.bits)
    fail("idle: the program loaded again computes other values");

  const auto &[ffn, hidden] = chunk;
  CommandGraph graph;
  if (!split->begin(AneFfn::Steps::One)) fail("idle: the split does not begin a command once loaded again");
  split->add(graph, 0, ffn, hidden[0], hidden[1], kRows);
  test::rejects([&] { static_cast<void>(split->release()); }, "unfinished", "a command encoded allowed a release");
  for (uint32_t layer = 1; layer < kLayers; ++layer) {
    split->join(graph);
    split->add(graph, layer, ffn, hidden[layer & 1], hidden[(layer & 1) ^ 1], kRows);
  }
  split->join(graph);
  metal::CommandTicket command = split->commit(graph, {});
  test::rejects([&] { static_cast<void>(split->release()); }, "unfinished", "a command in flight allowed a release");
  static_cast<void>(command.wait());
  test::rejects([&] { static_cast<void>(split->release()); }, "unfinished", "a command not finished allowed a release");
  if (!split->finish()) fail("idle: unusable: " + split->reason());
  if (!split->release()) fail("idle: the program was not unloaded once its command finished");
  split->restore();
  if (split->retired()) fail("idle: stopped: " + split->reason());
  section("idle: the program unloaded and loaded again computes the same values; a command encoded, in flight or "
          "not finished refuses a release");
}
// A fault armed for the next split's program at the evaluation `faults` names in its first forward of `rows` rows: the
// forward's results are unusable and the split stops, without failing its command, within the handoff's bound of
// the evaluation's start, and the GPU alone then computes, bit for bit, what it computes on its own.
void stops(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk,
           ane::ProgramInstrumentation::Faults faults, const std::string &what, uint32_t rows,
           std::string_view reason) {
  ane::ProgramInstrumentation::arm(faults);
  const auto split = std::make_unique<AneFfn>(backend, linear, model.layers, kShares, kBothRows, nullptr);
  const Forward faulted = forward(backend, linear, model, split.get(), chunk, rows);
  const std::string label = what + ", " + std::to_string(rows) + " rows";
  if (faulted.usable) fail(label + ": the results were usable");
  if (!split->retired() || split->splits(rows)) fail(label + ": the split did not stop");
  if (split->reason().find(reason) == std::string::npos) fail(label + ": stopped for " + split->reason());
  if (!backend.healthy()) fail(label + ": the backend is unhealthy: " + backend.unhealthyReason());
  if (!(faulted.milliseconds < 4000.0))
    fail(label + ": the command took " + std::to_string(faulted.milliseconds) + " ms");
  const Forward alone = forward(backend, linear, model, nullptr, chunk, rows);
  const Forward after = forward(backend, linear, model, split.get(), chunk, rows);
  if (!after.usable || after.bits != alone.bits)
    fail(label + ": the GPU's forward after the split stopped differs from the GPU's alone");
}

// A Neural Engine slower than the GPU alone: every evaluation starts a lag after its event reaches its wait, twice the
// longer of the GPU alone's layer of its rows and the Neural Engine's own evaluation of them as timed here, so that on
// a fast GPU as on a slow one the lag is most of the time the evaluations take and alone makes them take twice what
// trips the breaker, the GPU alone's layers. Each command's results are usable, the breaker given those timings stops
// the split after ane_ffn::Breaker::kWindow commands and not before, for losing to the GPU alone, with the Neural
// Engine's time over them counted, lag and all, telling its owner once, and the GPU alone then computes, bit for bit,
// what it computes on its own; idle, the stopped split unloads its program for good.
void loses(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  constexpr uint32_t kRows = AneFfn::kMaximumRows, kWindow = ane_ffn::Breaker::kWindow;
  ane_ffn::ChunkTimings timings;
  for (const uint32_t index : {0u, 1u}) {
    const uint32_t rows = index ? kRows : AneFfn::kMinimumRows;
    timings.gpu[index] = {rows, forward(backend, linear, model, nullptr, chunk, rows).milliseconds / kLayers};
  }
  double evaluation = 0.0;
  {
    AneFfn unlagged(backend, linear, model.layers, kShares, kBothRows, nullptr);
    if (!forward(backend, linear, model, &unlagged, chunk, kRows).usable) {
      fail("a slow Neural Engine: a split without the lag was unusable: " + unlagged.reason());
      return;
    }
    evaluation = unlagged.served().milliseconds / unlagged.served().evaluations;
  }
  const auto lag = std::chrono::ceil<std::chrono::milliseconds>(
      2.0 * std::chrono::duration<double, std::milli>(std::max(timings.gpu[1].milliseconds, evaluation)));
  ane::ProgramInstrumentation::arm({.lag = lag});
  const auto split = std::make_unique<AneFfn>(backend, linear, model.layers, kShares, kBothRows, nullptr);
  uint32_t lost = 0;
  split->setBreaker(ane_ffn::Breaker(timings.gpu, 0.0), [&] { ++lost; });
  for (uint32_t command = 1; command <= kWindow; ++command) {
    const Forward got = forward(backend, linear, model, split.get(), chunk, kRows);
    const std::string label = "a slow Neural Engine, command " + std::to_string(command);
    if (!got.usable) fail(label + ": unusable: " + split->reason());
    if (split->retired() != (command == kWindow) || lost != (command == kWindow))
      fail(label + (split->retired() ? ": the split stopped for " + split->reason() : ": the split did not stop") +
           ", its owner told " + std::to_string(lost) + " times");
  }
  std::cout << "  a lag of " << lag.count() << " ms, the Neural Engine's own evaluation " << evaluation
            << " ms: stopped for " << split->reason() << '\n';
  if (!split->reason().starts_with("losing to the GPU alone ("))
    fail("a slow Neural Engine: stopped for " + split->reason());
  const AneFfn::Served served = split->served();
  // Each layer of a full chunk's command runs its two steps' evaluations.
  if (served.commands != kWindow || served.evaluations != 2 * kWindow * kLayers ||
      !(served.milliseconds >= 0.9 * 2 * kWindow * kLayers * lag.count()))
    fail("a slow Neural Engine: served " + std::to_string(served.commands) + " commands, " +
         std::to_string(served.evaluations) + " evaluations in " + std::to_string(served.milliseconds) + " ms");
  const Forward alone = forward(backend, linear, model, nullptr, chunk, kRows);
  const Forward after = forward(backend, linear, model, split.get(), chunk, kRows);
  if (!after.usable || after.bits != alone.bits)
    fail("a slow Neural Engine: the GPU's forward after the split stopped differs from the GPU's alone");
  if (split->served().commands != kWindow || lost != 1)
    fail("a slow Neural Engine: the stopped split counted a command, or told its owner again");
  // Its evaluations all reported, the stopped split unloads its program while idle, once, and keeps it unloaded.
  if (!split->release()) fail("a slow Neural Engine: the stopped split did not unload its program while idle");
  split->restore();
  if (split->release()) fail("a slow Neural Engine: the stopped split loaded its program again");
}

// A split's program holds the functions of its shares alone (AneFfn::programRows): of one, one step's 13 from 512 to
// 2048 rows; of both, one step's 5 to 1,024, whose last holds the chunks below two steps', and the halves' 5, which
// Apple9's compiler takes. A command queues every evaluation before the GPU starts, which the ANE's client holds at
// most 127 of (AneFfn::fits): a model's last layer runs on the GPU alone, so a split of 64 layers takes commands of two
// steps, 126 evaluations, and one of 65 does not; one of 128 takes commands of one step, 127, and one of 129 none
// (AneFfn::unsupported). A command of more is refused before any evaluation starts, not left to wait on the client.
void programAndInFlight(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  using Steps = AneFfn::Steps;
  const auto every = [](uint32_t least, uint32_t most) {
    std::vector<uint32_t> rows;
    for (uint32_t row = least; row <= most; row += AneFfn::kProgramStep) rows.push_back(row);
    return rows;
  };
  const std::array<std::vector<uint32_t>, 2> one = AneFfn::programRows({kAneUnits, 0}),
                                             both = AneFfn::programRows({4, kAneUnits});
  if (one[0] != every(512, 2048) || !one[1].empty() || both[0] != every(512, 1024) || both[1] != every(512, 1024) ||
      AneFfn::programRows({0, kAneUnits})[0].size() || AneFfn::programRows({kAneUnits, kAneUnits}) != both)
    fail("a split's program holds other functions than those of its shares");
  if (!AneFfn::fits(64, Steps::Two) || AneFfn::fits(65, Steps::Two) || !AneFfn::fits(128, Steps::One) ||
      AneFfn::fits(129, Steps::One) || AneFfn::fits(0, Steps::One))
    fail("the evaluations in flight bound commands other than those of (layers - 1) * steps <= 127");
  // The model's three layers, again and again.
  const auto repeated = [&](uint32_t count) {
    std::vector<SwiGluProjections> layers;
    for (uint32_t layer = 0; layer < count; ++layer) layers.push_back(model.layers[layer % kLayers]);
    return layers;
  };
  if (!AneFfn::unsupported(repeated(129)) || AneFfn::unsupported(repeated(128)))
    fail("a split of 129 layers was taken, or one of 128 refused");
  test::rejects([&] { static_cast<void>(AneFfn(backend, linear, repeated(65), kShares, kBothRows, nullptr)); },
                "in two steps", "a split of 65 layers took commands of two steps");
  AneFfn split(backend, linear, repeated(128), {kAneUnits, 0}, kOneStep, nullptr);
  const auto &[ffn, hidden] = chunk;
  const auto command = [&](uint32_t layers) {
    CommandGraph graph;
    if (!split.begin(Steps::One)) fail("in flight: the split stopped: " + split.reason());
    for (uint32_t layer = 0; layer < layers; ++layer) {
      split.add(graph, layer, ffn, hidden[layer & 1], hidden[(layer & 1) ^ 1], AneFfn::kMinimumRows);
      split.join(graph);
    }
    static_cast<void>(split.commit(graph, {}).wait());
    return split.finish();
  };
  fillRows(chunk, AneFfn::kMinimumRows, [](uint64_t) { return 1.0f; });
  if (!command(127)) fail("in flight: a command of 127 evaluations did not complete: " + split.reason());
  test::rejects([&] { static_cast<void>(command(128)); }, "more than the Neural Engine holds in flight",
                "a command of 128 evaluations was queued");
  section("program and in flight: a split's functions are its shares' alone, two steps of 64 layers and one of 128 "
          "are the most the ANE holds in flight, and a command of 127 evaluations runs, one of 128 is refused");
}

// The split of the MLX and the GGUF model over normalized rows of normal values, and the numerics of rows the ANE's
// fp16 barely holds.
void split(MetalBackend &backend) {
  const Linear linear(backend.capabilities(), Int8Activations::On, StepKind::Prefill);
  const Model mlx = mlxModel(backend), gguf = ggufModel(backend);
  takes(mlx, gguf);
  keys(backend.capabilities(), mlx, gguf);
  const Chunk buffers = chunk(backend, AneFfn::kMaximumRows);
  split(backend, linear, mlx, buffers);
  split(backend, linear, gguf, buffers);
  pipelined(backend, linear, mlx, buffers);
  pipelined(backend, linear, gguf, buffers);
  programAndInFlight(backend, linear, mlx, buffers);
  misbound(backend, mlx, buffers);
  refusals(backend, mlx, buffers);
  quietRows(backend, linear, mlx, buffers);
  massiveChannel(backend, linear, buffers);
  idle(backend, linear, mlx, buffers);
  fillRows(buffers, AneFfn::kMaximumRows, [](uint64_t) { return 1.0f; });
  stops(backend, linear, mlx, buffers, {.poisonedEvaluation = 3}, "an infinity in layer 1's output",
        AneFfn::kMaximumRows, "not finite");
  section("stops: an infinity in the ANE's output stops the split");
}

// ---------------------------------------------------------------- faults
// Each fault of an evaluation at layer 0, 1 and 2 of chunks of 512 and 700 rows, one step, and of 2048 rows, two
// steps, the first step's evaluation of the layer, stops the split (stops()); an evaluation that signals the event
// below its value leaves it there.
void faults(MetalBackend &backend) {
  using Faults = ane::ProgramInstrumentation::Faults;
  const Linear linear(backend.capabilities(), Int8Activations::On, StepKind::Prefill);
  const Model model = mlxModel(backend);
  const Chunk buffers = chunk(backend, AneFfn::kMaximumRows);
  fillRows(buffers, AneFfn::kMaximumRows, [](uint64_t) { return 1.0f; });
  struct Fault {
    const char *name;
    Faults (*arm)(uint64_t evaluation);
    const char *reason;
  };
  const Fault kinds[] = {
      {"a throwing enqueue", [](uint64_t at) { return Faults{.throwingEnqueue = at}; }, "could not start"},
      {"a failed evaluation", [](uint64_t at) { return Faults{.failingEvaluation = at}; }, "an evaluation failed"},
      {"a stalled evaluation", [](uint64_t at) { return Faults{.stalledEvaluation = at}; }, "did not complete"},
      {"a late evaluation",
       [](uint64_t at) { return Faults{.delayedEvaluation = at, .delay = std::chrono::milliseconds(3000)}; },
       "did not complete"},
      {"an infinite output", [](uint64_t at) { return Faults{.poisonedEvaluation = at}; }, "not finite"},
  };
  for (const Fault &kind : kinds) {
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (const uint32_t count : {AneFfn::kMinimumRows, 700u, AneFfn::kMaximumRows}) {
        const uint32_t steps = count >= 2 * AneFfn::kMinimumRows ? 2 : 1;
        stops(backend, linear, model, buffers, kind.arm(steps * layer + 1),
              std::string(kind.name) + " at layer " + std::to_string(layer), count, kind.reason);
      }
    section(std::string("faults: ") + kind.name + " at each layer of chunks of 512, 700 and 2048 rows");
  }

  // A program that does not load again after the idle release.
  ane::ProgramInstrumentation::arm({.failingLoad = true});
  const auto split =
      std::make_unique<AneFfn>(backend, Linear(backend.capabilities(), Int8Activations::On, StepKind::Prefill),
                               model.layers, kShares, kBothRows, nullptr);
  if (!split->release()) fail("a failed reload: the program was not unloaded");
  split->restore();
  if (!split->retired() || split->splits(AneFfn::kMaximumRows)) fail("a failed reload: the split did not stop");
  if (split->reason().find("did not load again: ANE load failed") == std::string::npos)
    fail("a failed reload: stopped for " + split->reason());
  const Forward alone = forward(backend, linear, model, nullptr, buffers, AneFfn::kMaximumRows);
  const Forward after = forward(backend, linear, model, split.get(), buffers, AneFfn::kMaximumRows);
  if (!after.usable || after.bits != alone.bits)
    fail("a failed reload: the GPU's forward after the split stopped differs from the GPU's alone");
  // Idle, the stopped split unloads what a load that ran late may have left loaded, once, and keeps it unloaded.
  if (!split->release()) fail("a failed reload: the stopped split did not unload its program while idle");
  split->restore();
  if (split->release()) fail("a failed reload: the stopped split loaded its program again");
  section("faults: a program that does not load again stops the split, and the GPU alone then computes what it "
          "computes on its own");

  loses(backend, linear, model, buffers);
  section("faults: a Neural Engine slower than the GPU alone stops the split after 8 usable commands, and the GPU "
          "alone then computes what it computes on its own");

  // The CPU raises the event past an evaluation's signal before it runs.
  test::AneSum sum(backend);
  const test::TemporaryDirectory cache("splash-ane-faults");
  ane::Program program(sum.mil("below"), {}, {std::chrono::seconds(120), {}}, cache.path());
  const metal::SharedEvent event = backend.newSharedEvent();
  event.signal(100);
  struct Report final {
    std::mutex mutex;
    std::condition_variable changed;
    std::optional<bool> result;
  };
  const auto report = std::make_shared<Report>();
  sum.fill(rng);
  program.enqueue(sum.bind(program, "below"), event, 50, 60, [report](bool success) {
    std::lock_guard lock(report->mutex);
    report->result = success;
    report->changed.notify_all();
  });
  std::unique_lock lock(report->mutex);
  report->changed.wait_until(lock, AwakeClock::now() + std::chrono::seconds(10),
                             [&] { return report->result.has_value(); });
  const uint64_t value = [(__bridge id<MTLSharedEvent>)event.nativeHandle() signaledValue];
  if (report->result != true || !sum.summed()) fail("an evaluation waiting on a value passed did not run");
  if (value != 100) fail("an evaluation's signal of 60 moved the event from 100 to " + std::to_string(value));
  section("faults: an evaluation's signal below the event's value leaves it there");
}

// ---------------------------------------------------------------- the program
using ane::Program;
using test::AneSum;
using test::thrown;

const Program::Limits kLimits{std::chrono::seconds(120), {}};
constexpr auto kEvaluationTimeout = std::chrono::seconds(10);

// Purges the service's compilation of the program whose files are in
// `directory`, as the service may on its own: its name is the service's key.
void purge(const std::filesystem::path &directory) {
  @autoreleasepool {
    id model = [(Class<AneTestModel>)NSClassFromString(@"_ANEModel")
        modelAtURL:[NSURL fileURLWithPath:@(directory.c_str()) isDirectory:YES]
               key:@(directory.filename().c_str())];
    [[(Class<AneTestClient>)NSClassFromString(@"_ANEClient") sharedConnection] purgeCompiledModel:model];
  }
}

// One evaluation of `binding` that reports success and sums the inputs.
void evaluates(AneSum &sum, Program &program, const Program::Binding &binding, const std::string &what) {
  sum.fill(rng);
  const std::optional<bool> result = sum.evaluate(program, binding, kEvaluationTimeout);
  if (result != true)
    fail(what + (result ? ": the evaluation failed" : ": the evaluation did not complete"));
  else if (!sum.summed())
    fail(what + ": the output is not the inputs' sum");
}

// What fails as a std::exception, an evaluation, and unload() and load().
void programs(AneSum &sum, const std::filesystem::path &cache) {
  std::string message = thrown<std::runtime_error>([&] { Program broken("not a MIL program", {}, kLimits, cache); });
  if (message.find("ANE compilation failed") == std::string::npos) fail("an invalid MIL: " + message);
  Program program(sum.mil("sum"), {}, kLimits, cache);
  message = thrown<std::invalid_argument>([&] { static_cast<void>(program.procedure("none")); });
  if (message.find("no function none") == std::string::npos) fail("a function the program lacks: " + message);
  message = thrown<std::invalid_argument>(
      [&] { static_cast<void>(program.bind(program.procedure("sum"), sum.oneInput(), sum.output())); });
  if (message.find("input count") == std::string::npos) fail("a binding of one input short: " + message);
  const Program::Binding binding = sum.bind(program, "sum");
  evaluates(sum, program, binding, "an evaluation");
  section("program: an invalid MIL, a function it lacks and a binding of one input short throw; it evaluates");

  const std::vector<_Float16> loaded = sum.values();
  program.unload(kLimits);
  program.unload(kLimits);
  message = thrown<std::runtime_error>([&] { sum.enqueue(program, binding, [](bool) {}); });
  if (message.find("not loaded") == std::string::npos) fail("an evaluation of an unloaded program: " + message);
  program.load(kLimits);
  program.load(kLimits);
  const std::optional<bool> result = sum.evaluate(program, binding, kEvaluationTimeout);
  if (result != true)
    fail("an evaluation once loaded again did not succeed");
  else if (sum.values() != loaded)
    fail("an evaluation once loaded again computed other values");

  // A program whose compilation the service no longer holds.
  const test::TemporaryDirectory own("splash-ane-program");
  Program lost(sum.mil("sum_lost"), {}, kLimits, own.path());
  lost.unload(kLimits);
  purge(test::programDirectory(own.path()));
  message = thrown<std::runtime_error>([&] { lost.load(kLimits); });
  if (message.find("not compiled") == std::string::npos) fail("load() of a program not compiled: " + message);
  section("unload() and load(): twice each, the same values once loaded again; load() does not compile");
}

// Limits that end the wait for the service.
void limits(AneSum &sum, const std::filesystem::path &cache) {
  std::string message = thrown<std::runtime_error>(
      [&] { Program late(sum.mil("sum_late"), {}, {std::chrono::nanoseconds(1), {}}, cache); });
  if (message.find("the Neural Engine did not answer within 1e-09 s") == std::string::npos)
    fail("a limit of 1 ns: " + message);
  // Queued behind the work the first stopped waiting for.
  Program next(sum.mil("sum_next"), {}, kLimits, cache);
  evaluates(sum, next, sum.bind(next, "sum_next"), "the program after a limit ran out");
  message = thrown<ane::Interrupted>([&] {
    Program interrupted(sum.mil("sum_interrupted"), {}, {std::chrono::seconds(120), [] { return true; }}, cache);
  });
  if (message.find("interrupted") == std::string::npos) fail("an interrupted wait: " + message);
  section("limits: 1 ns runs out, and a program comes up after it; an interrupted wait throws Interrupted");
}

// The cache's files: one of other bytes is written again, and one that cannot
// be written leaves no partial file.
void cacheFiles(AneSum &sum) {
  const test::TemporaryDirectory cache("splash-ane-program");
  const std::string mil = sum.mil("sum_cached");
  { Program first(mil, {}, kLimits, cache.path()); }
  const std::filesystem::path directory = test::programDirectory(cache.path()), source = directory / "model.mil";
  std::string other = mil;
  other[other.find("sum_cached")] = 'S';
  test::writeFile(source, other);
  {
    Program again(mil, {}, kLimits, cache.path());
    evaluates(sum, again, sum.bind(again, "sum_cached"), "a program whose cached source was written again");
  }
  const std::vector<uint8_t> written = test::readFile(source);
  if (std::string(written.begin(), written.end()) != mil) fail("a cached source of the same size kept other bytes");

  const auto partials = [&] {
    return std::ranges::count_if(std::filesystem::directory_iterator(directory),
                                 [](const auto &entry) { return entry.path().extension() == ".partial"; });
  };
  test::writeFile(source, other);
  std::filesystem::permissions(directory, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
  std::string message = thrown<std::runtime_error>([&] { Program unwritable(mil, {}, kLimits, cache.path()); });
  std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
  if (message.find("unable to write") == std::string::npos) fail("an unwritable cache directory: " + message);
  if (partials()) fail("an unwritable cache directory kept a partial file");
  // A source that cannot be replaced: a directory in its place.
  std::filesystem::remove(source);
  std::filesystem::create_directories(source / "entry");
  message = thrown<std::filesystem::filesystem_error>([&] { Program unreplaced(mil, {}, kLimits, cache.path()); });
  if (message.find("rename") == std::string::npos) fail("a source that cannot be replaced: " + message);
  if (partials()) fail("a source that cannot be replaced left a partial file");
  section("cache: a source of the same size and other bytes is written again; a failed write leaves no partial file");
}

void program(MetalBackend &backend) {
  AneSum sum(backend);
  const test::TemporaryDirectory cache("splash-ane-program");
  programs(sum, cache.path());
  limits(sum, cache.path());
  cacheFiles(sum);
}

} // namespace

int main(int argc, char **argv) {
  const std::string_view mode = argc == 3 ? argv[2] : "";
  if (mode != "kernels" && mode != "split" && mode != "faults" && mode != "program") {
    std::cerr << "usage: ane-ffn METALLIB kernels|split|faults|program\n";
    return 2;
  }
  try {
    MetalBackend backend(argv[1]);
    if (mode == "kernels") {
      std::vector<float> sign(ANE_FFN_INTERMEDIATE_BLOCK);
      for (float &value : sign) value = rng() & 1 ? -1.0f : 1.0f;
      const MetalBuffer signs = upload(backend, sign);
      inputs(backend, signs, sign);
      weights(backend, signs, sign);
      join(backend);
    } else if (mode == "split") {
      split(backend);
    } else if (mode == "faults") {
      faults(backend);
    } else {
      program(backend);
    }
  } catch (const std::exception &error) {
    std::cout << "FAIL " << error.what() << '\n';
    return 1;
  }
  if (failures) {
    std::cout << failures << " failures\n";
    return 1;
  }
  std::cout << "ane-ffn " << mode << ": all checks passed\n";
  return 0;
}
