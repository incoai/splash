// The prefill FFN's Neural Engine split (ops/AneFfn.cpp, kernels/prefill/ane_ffn.metal) at the Omni decoders' shape,
// hidden 2048 and intermediate 6144 of affine Q4 (ported from Splash's ane_ffn_test):
// - kernels: ane_ffn_rotate and ane_ffn_pack turn bf16 rows into the ANE's channel-major int8 inputs at their stride
//   and the rows' scales, as a CPU rotation of the rows does; ane_ffn_row_scale and ane_ffn_weights turn affine Q4
//   rows into int8 rows at their stride and their scales at theirs, rotated in either block, as a CPU rotation of
//   their dequantized values does, and rows of zeros take a normal scale and codes of 0; ane_ffn_join adds the ANE's
//   channel-major partial rows, scaled in fp32, to the chunk's output rows and to no others, and flags a value of the
//   chunk's rows that is not finite;
// - split: for three layers at the engine's share (3 of 12 units), chunks of the fewest rows, of a count between two
//   functions and of the most rows compute what the GPU computes alone within int8's error, as do rows of mixed
//   magnitudes and rows with a hidden channel 300 times the others; verify() passes; a chunk no function holds and
//   layers out of order are refused, and the split keeps what it needs of the layers it was built from;
// - faults: an evaluation that fails, one that never reports, one that writes an infinity and an enqueue that throws
//   (ane/ProgramInstrumentation.hpp) stop the split without failing the Metal command or throwing; the command's
//   results are reported unusable, and the GPU alone then computes, bit for bit, what it computes on its own.
// Metal's validation layer wraps the shared event that orders the split's GPU and ANE work in one the ANE cannot
// share, so `kernels` runs the first checks, which test-engine-metal validates, and `split` and `faults` the others.
// The binary links the instrumented Program, whose faults only `faults` arms.
#include "AffineQ4Fixture.hpp"
#include "AwakeClock.hpp"
#include "TestBuffers.hpp"
#include "ane/ProgramInstrumentation.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/Linear.h"
#include "ops/AneFfn.hpp"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;
using namespace splash::ops;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::tuning::bf16ToFloat;
using splash::ops::tuning::floatToBf16;

namespace {

constexpr uint32_t kHidden = 2048, kIntermediate = 6144, kLayers = 3, kAneUnits = 3;
// The split's functions, and chunks of the fewest rows, of a count between two functions and of the most.
const std::vector<uint32_t> kFunctionRows{160, 256, 512, 2048};
constexpr std::array<uint32_t, 3> kChunks{2048, 160, 300};
// The most the split's output may differ from the GPU's alone, as the RMS of their difference relative to the GPU's:
// int8 leaves about 1% at this share.
constexpr double kMaximumError = 0.05;
// What a kernel must leave as it was.
constexpr uint8_t kUntouched = 0x5A;
constexpr uint16_t kUntouchedHalf = 0x5A5A;
constexpr uint16_t kHalfInfinity = 0x7c00, kHalfNaN = 0x7e00;

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

// v -> sign (H v) / sqrt(n) on each block of n values, H the Sylvester Hadamard matrix.
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
// 0, as the ANE's channel-major int8 inputs, each channel's rows at a stride past them.
void inputs(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign) {
  constexpr uint32_t kRows = 64, kStride = 96, kZeroRow = 7;
  std::normal_distribution<float> normal;
  std::vector<uint16_t> x(uint64_t{kRows} * kHidden);
  for (uint64_t i = 0; i < x.size(); ++i)
    x[i] = i / kHidden == kZeroRow ? 0 : floatToBf16(normal(rng) * (i % 997 ? 1.0f : 40.0f));
  const MetalBuffer rotated = test::sharedBuffer(backend, x.size() * sizeof(uint16_t));
  const MetalBuffer scales = filled(backend, kRows * sizeof(uint16_t), kUntouched);
  const MetalBuffer packed = filled(backend, uint64_t{kHidden} * kStride, kUntouched);
  CommandGraph graph;
  graph.add("ane_ffn_rotate", {upload(backend, x), signs, rotated, scales}, AneFfnRotateParams{kHidden},
            {kRows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  graph.add("ane_ffn_pack", {rotated, packed}, AneFfnPackParams{kHidden, kStride},
            {kRows / ANE_FFN_TILE, kHidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
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
      const int8_t got = contents<int8_t>(packed)[uint64_t{channel} * kStride + row];
      const int8_t expected = toInt8(toHalf(v[channel] * inverse));
      if (!near(got, expected))
        fail("row " + std::to_string(row) + " channel " + std::to_string(channel) + ": " + std::to_string(got) +
             ", not " + std::to_string(expected));
    }
  }
  for (uint32_t channel = 0; channel < kHidden; ++channel)
    for (uint32_t byte = kRows; byte < kStride; ++byte)
      if (contents<uint8_t>(packed)[uint64_t{channel} * kStride + byte] != kUntouched)
        fail("channel " + std::to_string(channel) + ": packed past the rows");
  if (!(fromHalf(contents<uint16_t>(scales)[kZeroRow]) >= 0x1p-14f))
    fail("a row of zeros took a scale that is not a normal half");
  section("inputs: 64 rows of 2048 bf16 values, one of zeros, as channel-major int8 and the rows' scales");
}

// ---------------------------------------------------------------- weights
constexpr uint32_t kWeightRows = 512, kWeightInputs = 2048;
// A row of zero weights, which takes the least scale and codes of 0.
constexpr uint32_t kZeroWeightRow = 300;

// The affine Q4 index of a row's quant group: [rows / 256][groups][256].
uint64_t unitOf(uint32_t row, uint32_t group, uint32_t groups) {
  return (uint64_t{row / SPLASH_AFFINE_TILE_ROWS} * groups + group) * SPLASH_AFFINE_TILE_ROWS +
         row % SPLASH_AFFINE_TILE_ROWS;
}

// Rows [row, kWeightRows) of a [kWeightRows, kWeightInputs] projection under their scales over inputs [scaled,
// kWeightInputs), and their int8 values over inputs [input, kWeightInputs), rotated in blocks of `block` from
// `scaled`: as the split stages the rows of gate and up the ANE takes over the hidden inputs (block 128), or the rows
// of down over the ANE's inputs (block 512). Each row's values lie a stride apart, and its scales' copies another.
void weights(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign, const Projection &source,
             const std::vector<float> &values, uint32_t block, uint32_t row, uint32_t scaled, uint32_t input) {
  const uint32_t rows = kWeightRows - row, width = kWeightInputs - input, stride = width + 64, scaleStride = 3;
  const std::string variant = block == ANE_FFN_INPUT_BLOCK ? "_inputs" : "_intermediate";
  const MetalBuffer rowScales = test::sharedBuffer(backend, rows * sizeof(uint16_t));
  const MetalBuffer output = filled(backend, uint64_t{rows} * stride, kUntouched);
  const MetalBuffer scales = filled(backend, uint64_t{rows} * scaleStride * sizeof(uint16_t), kUntouched);
  const AffineWeights &planes = source.affine();
  CommandGraph graph;
  graph.add("ane_ffn_row_scale" + variant, {planes.weights, planes.scales, planes.biases, rowScales, signs},
            AneFfnWeightParams{kWeightInputs / 64, row, scaled, kWeightInputs - scaled, 0, 0},
            {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  graph.add("ane_ffn_weights" + variant, {planes.weights, planes.scales, planes.biases, rowScales, output, scales, signs},
            AneFfnWeightParams{kWeightInputs / 64, row, input, width, stride, scaleStride},
            {rows / ANE_FFN_WEIGHT_ROWS, width / block, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
  run(backend, graph);

  const std::string label = "affine Q4" + variant;
  for (uint32_t r = 0; r < rows; ++r) {
    const auto first = values.begin() + uint64_t{row + r} * kWeightInputs;
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
    const int8_t *got = contents<int8_t>(output) + uint64_t{r} * stride;
    for (uint32_t i = 0; i < width; ++i) {
      const int8_t value = toInt8(v[input - scaled + i] * inverse);
      if (!near(got[i], value))
        fail(label + " row " + std::to_string(r) + " input " + std::to_string(input + i) + ": " +
             std::to_string(got[i]) + ", not " + std::to_string(value));
    }
    for (uint32_t byte = width; byte < stride; ++byte)
      if (uint8_t(got[byte]) != kUntouched) fail(label + " row " + std::to_string(r) + ": written past its values");
  }
}

void weights(MetalBackend &backend, const MetalBuffer &signs, const std::vector<float> &sign) {
  const Projection projection = test::deterministicQ4Projection(backend, {kWeightRows, kWeightInputs}, 11);
  const AffineWeights &planes = projection.affine();
  const auto *nibbles = contents<uint8_t>(planes.weights);
  auto *scales = static_cast<uint16_t *>(planes.scales.contents());
  auto *biases = static_cast<uint16_t *>(planes.biases.contents());
  constexpr uint32_t groups = kWeightInputs / 64;
  for (uint32_t group = 0; group < groups; ++group) {
    const uint64_t unit = unitOf(kZeroWeightRow, group, groups);
    scales[unit] = biases[unit] = 0;
  }
  std::vector<float> values(uint64_t{kWeightRows} * kWeightInputs);
  for (uint32_t row = 0; row < kWeightRows; ++row)
    for (uint32_t input = 0; input < kWeightInputs; ++input) {
      const uint64_t unit = unitOf(row, input / 64, groups);
      const uint32_t code = (nibbles[unit * 32 + (input & 63) / 2] >> (input & 1 ? 4 : 0)) & 15;
      values[uint64_t{row} * kWeightInputs + input] = float(code) * bf16ToFloat(scales[unit]) + bf16ToFloat(biases[unit]);
    }
  weights(backend, signs, sign, projection, values, ANE_FFN_INPUT_BLOCK, 256, 0, 0);
  weights(backend, signs, sign, projection, values, ANE_FFN_INTERMEDIATE_BLOCK, 0, 512, 512);
  section("weights: affine Q4 rows as int8 rows and scales, rotated in blocks of 128 and 512; a row of zeros as "
          "codes of 0");
}

// ---------------------------------------------------------------- join
// ane_ffn_join of `partial` ([kHidden + 1][stride] fp16, its last row the tokens' intermediate scales) and the
// tokens' input scales into `output`, over a chunk of `rows` rows: the status word it leaves.
uint32_t joined(MetalBackend &backend, const MetalBuffer &output, const std::vector<uint16_t> &partial,
                const std::vector<uint16_t> &tokenScale, uint32_t stride, uint32_t rows) {
  const MetalBuffer status = filled(backend, sizeof(uint32_t), 0);
  CommandGraph graph;
  graph.add("ane_ffn_join", {output, upload(backend, partial), upload(backend, tokenScale), status},
            AneFfnJoinParams{kHidden, stride, rows},
            {(rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE, kHidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
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
  std::vector<Projection> projections;
  std::vector<SwiGluProjections> layers;
};
Model model(MetalBackend &backend) {
  Model result;
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    result.projections.push_back(test::deterministicQ4Projection(backend, {kIntermediate, kHidden}, 100 + 3 * layer));
    result.projections.push_back(test::deterministicQ4Projection(backend, {kIntermediate, kHidden}, 101 + 3 * layer));
    result.projections.push_back(test::deterministicQ4Projection(backend, {kHidden, kIntermediate}, 102 + 3 * layer));
  }
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    result.layers.push_back({&result.projections[3 * layer], &result.projections[3 * layer + 1],
                             &result.projections[3 * layer + 2]});
  return result;
}

// A chunk's buffers of up to the most rows, as the decoder's prefill arena holds them.
struct Chunk {
  PrefillFfnBuffers ffn;
  std::array<MetalBuffer, 2> hidden;
};
Chunk chunk(MetalBackend &backend, const Linear &linear, const Model &model) {
  constexpr uint32_t rows = AneFfn::kMaximumRows;
  const auto bf16Rows = [&](uint32_t width) { return test::sharedBuffer(backend, uint64_t{rows} * width * 2); };
  const auto sumRows = [&](uint32_t width) { return test::sharedBuffer(backend, uint64_t{rows} * (width / 64) * 4); };
  LinearScratchSize scratch;
  for (const Projection *p : {model.layers[0].gate, model.layers[0].down})
    scratch.include(linear.prefillScratchSize(p->shape()));
  const auto bytes = [&](uint64_t size) { return test::sharedBuffer(backend, std::max<uint64_t>(size, 16)); };
  return {{bf16Rows(kHidden), sumRows(kHidden), bf16Rows(kIntermediate), bf16Rows(kIntermediate),
           sumRows(kIntermediate),
           {bytes(scratch.input), bytes(scratch.sums), bytes(scratch.partials), bytes(scratch.counters)}},
          {bf16Rows(kHidden), bf16Rows(kHidden)}};
}
// Normal values scaled by row(r, c) into the leading rows of the chunk's normalized rows.
template <class Scale>
void fillRows(const Chunk &chunk, uint32_t rows, Scale scale) {
  std::normal_distribution<float> normal;
  auto *values = static_cast<uint16_t *>(chunk.ffn.normalized.contents());
  for (uint64_t i = 0; i < uint64_t{rows} * kHidden; ++i)
    values[i] = floatToBf16(normal(rng) * scale(i / kHidden, i % kHidden));
}

// A forward of every layer's FFN of a chunk's `rows` normalized rows from a zero residual, split where `split` takes
// the chunk and on the GPU alone otherwise, as the decoder's prefill encodes it: its bf16 output, whether the split's
// results were usable, and the command's milliseconds.
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
  linear.addPrefillSums(graph, ffn.normalized, ffn.sums, *model.layers.front().gate, rows);
  const bool splits = split && split->splits(rows) && split->begin();
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    const MetalBuffer &residual = hidden[layer & 1], &output = hidden[(layer & 1) ^ 1];
    if (splits)
      split->add(graph, layer, ffn, residual, output, rows);
    else
      linear.addPrefillSwiGlu(graph, model.layers[layer], ffn, residual, output, rows);
  }
  Forward result;
  const auto start = AwakeClock::now();
  if (splits)
    static_cast<void>(split->commit(graph).wait());
  else
    run(backend, graph);
  result.milliseconds = millisecondsSince(start);
  result.usable = !splits || split->finish();
  const uint16_t *bits = contents<uint16_t>(hidden[kLayers & 1]);
  result.bits.assign(bits, bits + uint64_t{rows} * kHidden);
  return result;
}

// The RMS of the split's output's difference from the GPU's alone, relative to the GPU's, over the rows `counts`
// takes, and whether the split's are finite.
template <class Counts>
std::pair<double, bool> difference(const Forward &expected, const Forward &got, Counts counts) {
  double difference = 0.0, magnitude = 0.0;
  bool finite = true;
  for (uint64_t i = 0; i < got.bits.size(); ++i) {
    if (!counts(i / kHidden)) continue;
    finite &= std::isfinite(got[i]);
    const double delta = double(got[i]) - expected[i];
    difference += delta * delta;
    magnitude += double(expected[i]) * expected[i];
  }
  return {std::sqrt(difference / magnitude), finite};
}
std::pair<double, bool> difference(const Forward &expected, const Forward &got) {
  return difference(expected, got, [](uint64_t) { return true; });
}

// The split's layers, built from copies of the model's projections that are gone before it runs: it keeps what it
// needs of them.
std::unique_ptr<AneFfn> splitOf(MetalBackend &backend, const Model &model) {
  const std::vector<Projection> copies = model.projections;
  std::vector<SwiGluProjections> layers;
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    layers.push_back({&copies[3 * layer], &copies[3 * layer + 1], &copies[3 * layer + 2]});
  return std::make_unique<AneFfn>(backend, layers, kAneUnits, kFunctionRows);
}

void split(MetalBackend &backend) {
  const Linear linear(backend.capabilities());
  const Model layers = model(backend);
  const Chunk buffers = chunk(backend, linear, layers);
  if (const char *reason = AneFfn::unsupported(layers.layers)) fail(std::string("the layers are refused: ") + reason);
  if (AneFfn::units(layers.layers) != 12) fail("6144 intermediate channels are not 12 units");
  const std::unique_ptr<AneFfn> split = splitOf(backend, layers);
  if (split->share() != 0.25 || split->aneChannels() != 1536)
    fail("runs share " + std::to_string(split->share()) + ", " + std::to_string(split->aneChannels()) + " channels");
  if (split->splits(159) || !split->splits(160) || !split->splits(2048) || split->splits(2049))
    fail("the split takes other chunks than those of 160 to 2048 rows");
  std::cout << "  " << split->allocatedBytes() / 1048576.0 << " MiB for " << kLayers << " layers\n";
  fillRows(buffers, AneFfn::kMaximumRows, [](uint64_t, uint64_t) { return 1.0f; });
  // The most rows first: later commands run on what earlier ones left in the split's surfaces.
  for (const uint32_t rows : kChunks) {
    const Forward expected = forward(backend, linear, layers, nullptr, buffers, rows);
    const Forward got = forward(backend, linear, layers, split.get(), buffers, rows);
    const auto [error, finite] = difference(expected, got);
    std::cout << "  " << rows << " rows: " << 100.0 * error << "% RMS from the GPU alone, " << got.milliseconds
              << " ms\n";
    if (!got.usable || !finite || !(error <= kMaximumError))
      fail(std::to_string(rows) + " rows: " + std::to_string(100.0 * error) + "% RMS from the GPU alone" +
           (got.usable ? "" : ", unusable: " + split->reason()));
  }
  section("split: three layers at share 0.25 from layers gone before it runs, chunks of 2048, 160 and 300 rows");

  // Rows of three magnitudes, and rows whose hidden channel 5 is 300 times the others, as a norm's weight may make
  // it: each row within int8's error. The FFN's input rows are RMS-normalized, so their magnitudes stay near the norm
  // weights'; far quieter rows lose the ANE's share to the floor of its intermediate scales (kIntermediateFloor).
  const auto mixed = [&](const std::string &what, auto scale) {
    constexpr uint32_t rows = 300;
    fillRows(buffers, rows, scale);
    const Forward expected = forward(backend, linear, layers, nullptr, buffers, rows);
    const Forward got = forward(backend, linear, layers, split.get(), buffers, rows);
    double worst = 0.0;
    for (uint32_t row = 0; row < rows; ++row) {
      const auto [error, finite] = difference(expected, got, [&](uint64_t r) { return r == row; });
      if (!finite) fail(what + ": row " + std::to_string(row) + " is not finite");
      worst = std::max(worst, error);
    }
    std::cout << "  " << what << ": worst row " << 100.0 * worst << "% RMS from the GPU alone\n";
    if (!got.usable || !(worst <= kMaximumError)) fail(what + ": worst row " + std::to_string(100.0 * worst) + "%");
  };
  mixed("rows of magnitudes 0.25, 1 and 4",
        [](uint64_t row, uint64_t) { return row % 3 == 0 ? 0.25f : row % 3 == 1 ? 1.0f : 4.0f; });
  mixed("rows with a channel 300 times the others",
        [](uint64_t, uint64_t channel) { return channel == 5 ? 300.0f : 1.0f; });
  section("split: rows of mixed magnitudes and rows with an outlying channel, each within int8's error");

  // verify() checks every function against the leading rows of a full chunk on the GPU alone, which computes every
  // row alike whatever the chunk's rows.
  fillRows(buffers, AneFfn::kMaximumRows, [](uint64_t, uint64_t) { return 1.0f; });
  const Forward full = forward(backend, linear, layers, nullptr, buffers, AneFfn::kMaximumRows);
  for (const uint32_t rows : {160u, 300u})
    if (const Forward fewer = forward(backend, linear, layers, nullptr, buffers, rows);
        !std::equal(fewer.bits.begin(), fewer.bits.end(), full.bits.begin()))
      fail("the GPU computed the rows of a chunk of " + std::to_string(rows) + " rows other than a full chunk's");
  const auto started = AwakeClock::now();
  const double error = split->verify(layers.layers, buffers.ffn, buffers.hidden);
  std::cout << "  verified in " << millisecondsSince(started) << " ms, " << 100.0 * error
            << "% from the GPU alone on the Neural Engine's part\n";
  if (split->retired()) fail("verify stopped the split: " + split->reason());
  section("split: a chunk's rows on the GPU alone as a full chunk's, and verify() of every function");

  // A least chunk of fewer rows than the smallest function runs it padded; layers out of order and a chunk no
  // function holds are refused.
  split->setMinimumRows(40);
  if (!split->splits(40) || split->splits(39)) fail("a least chunk of 40 rows took other chunks");
  {
    fillRows(buffers, 40, [](uint64_t, uint64_t) { return 1.0f; });
    const Forward expected = forward(backend, linear, layers, nullptr, buffers, 40);
    const Forward got = forward(backend, linear, layers, split.get(), buffers, 40);
    const auto [padded, finite] = difference(expected, got);
    if (!got.usable || !finite || !(padded <= kMaximumError))
      fail("a 40-row chunk on the 160-row function: " + std::to_string(100.0 * padded) + "%");
  }
  test::rejects([&] { split->setMinimumRows(2049); }, "no function of", "a least chunk of 2049 rows was taken");
  {
    CommandGraph graph;
    if (!split->begin()) fail("the split stopped");
    test::rejects([&] { split->add(graph, 1, buffers.ffn, buffers.hidden[0], buffers.hidden[1], 300); },
                  "not its next layer", "layer 1 was added first");
    test::rejects([&] { split->add(graph, 0, buffers.ffn, buffers.hidden[0], buffers.hidden[1], 2049); },
                  "no function of the chunk's rows", "a chunk of 2049 rows was added");
  }
  if (split->retired()) fail("a refusal stopped the split: " + split->reason());
  section("split: a least chunk below the smallest function, and refusals of layers out of order and of a chunk no "
          "function holds");
}

// ---------------------------------------------------------------- faults
// A split whose `faults` fire: its command's results are unusable, without failing the Metal command or taking
// longer than the handoff's bound, the split stops for `reason`, and a forward through it afterwards runs the GPU
// alone, bit for bit.
void stops(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk,
           ane::ProgramInstrumentation::Faults faults, const std::string &what, uint32_t rows,
           std::string_view reason) {
  ane::ProgramInstrumentation::arm(faults);
  const auto split = std::make_unique<AneFfn>(backend, model.layers, kAneUnits, kFunctionRows);
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

void faults(MetalBackend &backend) {
  using Faults = ane::ProgramInstrumentation::Faults;
  const Linear linear(backend.capabilities());
  const Model layers = model(backend);
  const Chunk buffers = chunk(backend, linear, layers);
  fillRows(buffers, AneFfn::kMaximumRows, [](uint64_t, uint64_t) { return 1.0f; });
  struct Fault {
    const char *name;
    Faults (*arm)(uint64_t evaluation);
    const char *reason;
  };
  const Fault kinds[] = {
      {"a throwing enqueue", [](uint64_t at) { return Faults{.throwingEnqueue = at}; }, "could not start"},
      {"a failed evaluation", [](uint64_t at) { return Faults{.failingEvaluation = at}; }, "an evaluation failed"},
      {"a stalled evaluation", [](uint64_t at) { return Faults{.stalledEvaluation = at}; }, "did not complete"},
      {"an infinite output", [](uint64_t at) { return Faults{.poisonedEvaluation = at}; }, "not finite"},
  };
  for (const Fault &kind : kinds) {
    for (const uint32_t layer : {0u, kLayers - 1})
      for (const uint32_t rows : {160u, 2048u})
        stops(backend, linear, layers, buffers, kind.arm(layer + 1),
              std::string(kind.name) + " at layer " + std::to_string(layer), rows, kind.reason);
    section(std::string("faults: ") + kind.name + " at the first and last layer of chunks of 160 and 2048 rows");
  }
}

} // namespace

int main(int argc, char **argv) {
  const std::string_view mode = argc == 3 ? argv[2] : "";
  if (mode != "kernels" && mode != "split" && mode != "faults") {
    std::cerr << "usage: ane-ffn METALLIB kernels|split|faults\n";
    return 2;
  }
  try {
    MetalBackend backend(argv[1]);
    if (mode == "kernels") {
      std::vector<float> sign(ANE_FFN_SIGNS);
      for (float &value : sign) value = rng() & 1 ? -1.0f : 1.0f;
      const MetalBuffer signs = upload(backend, sign);
      inputs(backend, signs, sign);
      weights(backend, signs, sign);
      join(backend);
    } else if (const auto reason = ane::unavailable()) {
      std::cout << "ane-ffn " << mode << ": skipped, the Neural Engine is unavailable (" << *reason << ")\n";
      return 0;
    } else if (mode == "split") {
      split(backend);
    } else {
      faults(backend);
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
