// The GPU's int8 prefill FFN (ops/Int8Ffn.cpp, kernels/prefill/int8_ffn.metal):
// - takes: which layers and devices it takes (an M5 or later for a prefill), and which chunks;
// - forward: three layers of MLX af4g64 and of mixed GGUF formats, over chunks of the fewest rows, of a count that is
//   no whole number of its 128-row tiles and of the most rows, compute what the bf16 kernels compute within int8's
//   error, finite, and leave the output's rows past the chunk as they were; Metal allocates what plannedBytes plans;
// - order: a layer added out of order converts its own weights, as a command's first does, and a second command
//   computes what the first did, bit for bit;
// - massive channel: rows of a hidden channel of about 1e5 within the bound.
// The FFN runs on any GPU splash does, so the checks run on every Mac; a prefill takes it on an M5 or later only.
#include "GgufFormatReference.hpp"
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/QuantFormat.h"
#include "ops/Int8Ffn.hpp"
#include "ops/Linear.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::ops;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

namespace {

constexpr uint32_t kHidden = 2048, kIntermediate = 6144, kLayers = 3;
// The most the FFN's output may differ from the bf16 kernels', as the RMS of their difference relative to theirs:
// int8 leaves about 1% on these layers, 2-3% on Qwen3.8-27B's (as the ANE FFN split's int8 does).
constexpr double kMaximumError = 0.05;
// What a kernel must leave as it was.
constexpr uint16_t kUntouched = 0x5A5A;

std::mt19937 rng(20261012);
int failures = 0;

void fail(const std::string &what) {
  std::cout << "FAIL " << what << '\n';
  ++failures;
}
void section(const std::string &what) { std::cout << "ok   " << what << '\n'; }

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = test::sharedBuffer(backend, bytes.size());
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}

// Three FFN layers: gate and up [kIntermediate, kHidden], down [kHidden, kIntermediate].
struct Model {
  std::string name;
  std::vector<Projection> projections;
  std::vector<SwiGluProjections> layers;
};
constexpr std::array<std::pair<uint32_t, uint32_t>, 3> kMatrices{
    {{kIntermediate, kHidden}, {kIntermediate, kHidden}, {kHidden, kIntermediate}}};
// Layers of `formats`, gate's, up's and down's of each, of weights of a model's magnitude: an MLX format's scales from
// its range (scaleRange), a GGUF format's a 32nd of it.
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
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    model.layers.push_back(
        {&model.projections[3 * layer], &model.projections[3 * layer + 1], &model.projections[3 * layer + 2]});
  return model;
}
Model mlxModel(MetalBackend &backend) {
  using gguf_reference::Fmt;
  constexpr auto kAf4 = Fmt(GGUF_FMT_AF4G64);
  return blockModel(backend, "MLX af4g64", {{{kAf4, kAf4, kAf4}, {kAf4, kAf4, kAf4}, {kAf4, kAf4, kAf4}}});
}
// Layers of mixed GGUF formats. The fixture's random sub-block scales spread a row's values over a range no model's
// weights span, more so in Q6_K and IQ4_XS (6% from the bf16 kernels with them in, against 1% without), so these
// layers leave them out; the formats' values are ane_ffn_gguf_values', which ane-ffn checks for every format.
Model ggufModel(MetalBackend &backend) {
  using namespace gguf_reference;
  return blockModel(backend, "GGUF", {{{Q4K, Q4K, Q5K}, {Q5K, Q5K, Q4K}, {Q4K, Q5K, Q80}}});
}

// The chunk's buffers of the most rows: the normalized rows and the FFN's scratch, and the hidden rows the layers'
// residuals and outputs alternate between.
struct Chunk {
  PrefillFfnBuffers ffn;
  std::array<MetalBuffer, 2> hidden;
};
Chunk chunk(MetalBackend &backend) {
  const auto bf16Rows = [&](uint32_t width) {
    return test::sharedBuffer(backend, uint64_t{Int8Ffn::kMaximumRows} * width * 2);
  };
  return {{bf16Rows(kHidden), bf16Rows(kIntermediate), bf16Rows(kIntermediate), {}},
          {bf16Rows(kHidden), bf16Rows(kHidden)}};
}
// Normal values scaled by row(r) into the chunk's normalized rows.
template <class Scale>
void fillRows(const Chunk &chunk, Scale row) {
  std::normal_distribution<float> normal;
  auto *values = static_cast<uint16_t *>(chunk.ffn.normalized.contents());
  for (uint64_t i = 0; i < uint64_t{Int8Ffn::kMaximumRows} * kHidden; ++i)
    values[i] = floatToBf16(normal(rng) * row(i / kHidden));
}

// The FFN of layers [first, last) of a chunk's `rows` normalized rows from a zero residual, in int8 on `int8` or on
// the bf16 kernels without it, in one command: the output's rows, the chunk's and those past it.
struct Forward {
  std::vector<uint16_t> bits;
  [[nodiscard]] float operator[](uint64_t index) const { return bf16ToFloat(bits[index]); }
};
Forward forward(MetalBackend &backend, const Linear &linear, const Model &model, Int8Ffn *int8, const Chunk &chunk,
                uint32_t rows, uint32_t first = 0, uint32_t last = kLayers) {
  const auto &[ffn, hidden] = chunk;
  std::memset(hidden[first & 1].contents(), 0, hidden[0].sizeBytes());
  std::fill_n(static_cast<uint16_t *>(hidden[(first & 1) ^ 1].contents()), hidden[0].sizeBytes() / 2, kUntouched);
  CommandGraph graph;
  for (uint32_t layer = first; layer < last; ++layer) {
    const MetalBuffer &residual = hidden[layer & 1], &output = hidden[(layer & 1) ^ 1];
    if (layer > first)
      std::fill_n(static_cast<uint16_t *>(output.contents()) + uint64_t{rows} * kHidden,
                  output.sizeBytes() / 2 - uint64_t{rows} * kHidden, kUntouched);
    if (int8)
      int8->add(graph, layer, ffn, residual, output, rows);
    else
      linear.addPrefillSwiGlu(graph, model.layers[layer], ffn, residual, output, rows);
  }
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  const auto *bits = static_cast<const uint16_t *>(hidden[last & 1].contents());
  return {std::vector<uint16_t>(bits, bits + hidden[0].sizeBytes() / 2)};
}

// The RMS of the int8 output's difference from the bf16 kernels', relative to theirs, over the chunk's `rows` rows and
// the channels `counts` takes, and whether the int8 output's are finite.
template <class Counts>
std::pair<double, bool> difference(const Forward &expected, const Forward &got, uint32_t rows, Counts counts) {
  double difference = 0.0, magnitude = 0.0;
  bool finite = true;
  for (uint64_t i = 0; i < uint64_t{rows} * kHidden; ++i) {
    if (!counts(i % kHidden)) continue;
    finite &= std::isfinite(got[i]);
    const double delta = double(got[i]) - expected[i];
    difference += delta * delta;
    magnitude += double(expected[i]) * expected[i];
  }
  return {std::sqrt(difference / magnitude), finite};
}
std::pair<double, bool> difference(const Forward &expected, const Forward &got, uint32_t rows) {
  return difference(expected, got, rows, [](uint64_t) { return true; });
}
// Whether the output's rows past the chunk's are as they were.
bool untouched(const Forward &got, uint32_t rows) {
  return std::all_of(got.bits.begin() + uint64_t{rows} * kHidden, got.bits.end(),
                     [](uint16_t bits) { return bits == kUntouched; });
}

// The FFN of `model`'s layers, built from copies of its projections that are gone before it runs: it keeps what it
// needs of them.
std::unique_ptr<Int8Ffn> int8Of(MetalBackend &backend, const Model &model) {
  const std::vector<Projection> copies = model.projections;
  std::vector<SwiGluProjections> layers;
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    layers.push_back({&copies[3 * layer], &copies[3 * layer + 1], &copies[3 * layer + 2]});
  return std::make_unique<Int8Ffn>(backend, layers);
}

void takes(const Model &mlx, const Model &gguf) {
  if (Int8Ffn::unsupported(mlx.layers) || Int8Ffn::unsupported(gguf.layers)) fail("the FFN does not take the models");
  if (!Int8Ffn::unsupported({})) fail("the FFN takes no layers");
  const auto projection = [](uint32_t outputs, uint32_t inputs) {
    return Projection(outputs, inputs,
                      BlockWeights{{QuantizedSegment::planes(GGUF_FMT_AF4G64, outputs, inputs, {}, {}, {})}});
  };
  // A hidden size of no whole rotation blocks, an intermediate size of none, and layers of two shapes.
  const Projection oddGate = projection(kIntermediate, 2112), oddDown = projection(2112, kIntermediate),
                   narrowGate = projection(5888, kHidden), narrowDown = projection(kHidden, 5888);
  const std::vector<SwiGluProjections> odd{{&oddGate, &oddGate, &oddDown}},
      narrow{{&narrowGate, &narrowGate, &narrowDown}}, mixed{mlx.layers[0], narrow[0]};
  if (!Int8Ffn::unsupported(odd)) fail("the FFN takes a hidden size of 2112");
  if (!Int8Ffn::unsupported(narrow)) fail("the FFN takes an intermediate size of 5888");
  if (!Int8Ffn::unsupported(mixed)) fail("the FFN takes layers of two shapes");
  DeviceCapabilities m4, m5;
  m4.appleGpuFamily = 9;
  m5.appleGpuFamily = 10;
  if (!Int8Ffn::unsupported(m4, mlx.layers) || Int8Ffn::unsupported(m5, mlx.layers))
    fail("a prefill takes the FFN on other GPUs than an M5 or later");
  if (Int8Ffn::takes(Int8Ffn::kMinimumRows - 1) || !Int8Ffn::takes(Int8Ffn::kMinimumRows) ||
      !Int8Ffn::takes(Int8Ffn::kMaximumRows) || Int8Ffn::takes(Int8Ffn::kMaximumRows + 1))
    fail("the FFN takes chunks outside its rows");
  section("takes: layers of one shape in whole rotation blocks, on an M5 or later, chunks of 256 to 2048 rows");
}

void forwards(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  const std::unique_ptr<Int8Ffn> int8 = int8Of(backend, model);
  if (int8->allocatedBytes() > Int8Ffn::plannedBytes(model.layers))
    fail(model.name + ": Metal allocated more than the plan");
  fillRows(chunk, [](uint64_t) { return 1.0f; });
  for (const uint32_t rows : {Int8Ffn::kMinimumRows, 300u, Int8Ffn::kMaximumRows}) {
    const Forward expected = forward(backend, linear, model, nullptr, chunk, rows);
    const Forward got = forward(backend, linear, model, int8.get(), chunk, rows);
    const auto [error, finite] = difference(expected, got, rows);
    std::cout << "  " << model.name << ", " << rows << " rows: " << 100.0 * error << "% RMS from the bf16 kernels\n";
    if (!finite || !(error <= kMaximumError))
      fail(model.name + ", " + std::to_string(rows) + " rows: " + std::to_string(100.0 * error) + "% RMS");
    if (!untouched(got, rows)) fail(model.name + ", " + std::to_string(rows) + " rows: rows past the chunk written");
  }
  section("forward: " + model.name + " within the bound at 256, 300 and 2048 rows; rows past the chunk untouched");
}

void order(MetalBackend &backend, const Linear &linear, const Model &model, const Chunk &chunk) {
  const std::unique_ptr<Int8Ffn> int8 = int8Of(backend, model);
  fillRows(chunk, [](uint64_t) { return 1.0f; });
  constexpr uint32_t kRows = 512;
  // The last layer alone, after a command of every layer, whose last down converted no next layer's.
  const Forward first = forward(backend, linear, model, int8.get(), chunk, kRows);
  const Forward expected = forward(backend, linear, model, nullptr, chunk, kRows, kLayers - 1, kLayers);
  const Forward alone = forward(backend, linear, model, int8.get(), chunk, kRows, kLayers - 1, kLayers);
  const auto [error, finite] = difference(expected, alone, kRows);
  if (!finite || !(error <= kMaximumError))
    fail("a layer added out of order: " + std::to_string(100.0 * error) + "% RMS from the bf16 kernels");
  const Forward again = forward(backend, linear, model, int8.get(), chunk, kRows);
  if (!std::equal(first.bits.begin(), first.bits.begin() + uint64_t{kRows} * kHidden, again.bits.begin()))
    fail("a second command computed other bits than the first");
  section("order: a layer added out of order within the bound; a second command bit for bit the first's");
}

// A hidden channel of about 1e5 on every row, from four intermediate channels of the last layer whose up rows are 1000
// times larger, and the group of down's row of that channel over them larger to match.
void massiveChannel(MetalBackend &backend, const Linear &linear, const Chunk &chunk) {
  constexpr uint32_t kRows = Int8Ffn::kMaximumRows, kChannel = 1234, kFirst = 4096, kCount = 4;
  static_assert(kFirst % 64 + kCount <= 64, "the channels lie in one group of down's inputs");
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
  fillRows(chunk, [](uint64_t) { return 1.0f; });
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
  const std::unique_ptr<Int8Ffn> int8 = int8Of(backend, model);
  const Forward got = forward(backend, linear, model, int8.get(), chunk, kRows);
  const auto [error, finite] = difference(expected, got, kRows);
  const auto [channelError, channelFinite] =
      difference(expected, got, kRows, [](uint64_t channel) { return channel == kChannel; });
  std::cout << "  a channel of " << peak(expected) << " at most: " << 100.0 * channelError << "% RMS from the bf16 "
            << "kernels, " << 100.0 * error << "% over every channel\n";
  if (!(peak(expected) >= 5e4f && peak(expected) <= 2e5f)) fail("massive channel: the channel does not reach 1e5");
  if (!finite || !channelFinite || !(error <= kMaximumError) || !(channelError <= kMaximumError))
    fail("massive channel: " + std::to_string(100.0 * channelError) + "% RMS from the bf16 kernels");
  section("massive channel: a hidden channel of about 1e5 finite and within the bound");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: int8-ffn METALLIB\n";
    return 2;
  }
  try {
    MetalBackend backend(argv[1]);
    const Linear linear(backend.capabilities());
    const Model mlx = mlxModel(backend), gguf = ggufModel(backend);
    const Chunk buffers = chunk(backend);
    takes(mlx, gguf);
    forwards(backend, linear, mlx, buffers);
    forwards(backend, linear, gguf, buffers);
    order(backend, linear, mlx, buffers);
    massiveChannel(backend, linear, buffers);
  } catch (const std::exception &error) {
    std::cout << "FAIL " << error.what() << '\n';
    return 1;
  }
  if (failures) {
    std::cout << failures << " failures\n";
    return 1;
  }
  std::cout << "int8-ffn: all checks passed\n";
  return 0;
}
