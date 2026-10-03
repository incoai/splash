// GDN verify decode and commit kernels against a direct CPU reference: the
// four-tap convolution with SiLU, the q/k RMS norms, the gates, the eight-row
// delta-rule recurrence over the fp32 state, the gated RMSNorm of the
// recurrent rows and the convolution carry, for both compiled geometries,
// every lane count, a two-layer state cell (so the layer offsets are
// exercised) and every retained count of the commit. The recurrent state is
// checked at fp32 accuracy from the kernel's own k/v and gates, after those
// were checked against the reference; the hidden rows from the recurrent rows
// read from that state with the reference q and rounded to bf16.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/StateLayout.hpp"
#include "ops/GDN.hpp"
#include "tuning/LinearNumerics.hpp"

#include "LinearInputReference.hpp"
#include "NormReference.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;
using namespace splash::test;
using splash::ops::tuning::bf16ToFloat;
using splash::ops::tuning::floatToBf16;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kMaxLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kLayers = 2;
// The scales of the normalized q and k rows.
constexpr double kQueryScale = 0.0078125, kKeyScale = 0.08838834765;
constexpr std::array kShapes{GdnShape{16, 48, 128, 10240, 16640},
                             GdnShape{16, 32, 128, 8192, 12544}};

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid GDN request was accepted");
}

double roundBfloat(double value) {
  return bf16ToFloat(floatToBf16(static_cast<float>(value)));
}

// One bf16 unit in the last place at the reference's magnitude.
double bfloatUlp(double reference) {
  int exponent = 0;
  std::frexp(std::fabs(reference), &exponent);
  return std::ldexp(1.0, exponent - 8);
}

bool closeBfloat(uint16_t got, double reference, double ulps, double floor) {
  return std::fabs(double(bf16ToFloat(got)) - reference) <=
         ulps * bfloatUlp(reference) + floor;
}

bool closeFloat(float got, double reference, double tolerance) {
  return std::fabs(double(got) - reference) <=
         tolerance * (1.0 + std::fabs(reference));
}

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

double sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }

// The production state cell layout: every layer's conv rows, then every
// layer's recurrent state, both padded to 16 KiB.
struct Cell final {
  splash::model::GdnStateLayout layout;
  uint64_t convLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convBytes = 0;
  uint64_t bytes = 0;

  explicit Cell(const GdnShape &shape)
      : layout{kLayers, splash::model::kGdnConvolutionTaps - 1, shape.convolutionDimension,
               shape.valueHeads, shape.headDimension, shape.headDimension},
        convLayerBytes(layout.convolutionLayerBytes()),
        recurrentLayerBytes(layout.recurrentLayerBytes()),
        convBytes(layout.convolutionBytes()), bytes(layout.cellBytes()) {}

  GdnStateStrides strides() const {
    return {convLayerBytes, recurrentLayerBytes, convBytes};
  }
  const uint16_t *conv(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const uint16_t *>(cell + layer * convLayerBytes);
  }
  const float *recurrent(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const float *>(cell + convBytes +
                                           layer * recurrentLayerBytes);
  }
};

// The mixer norm is bf16, or F32 as a GGUF stores it.
struct Fixture final {
  const GdnShape &shape;
  Cell cell;
  uint32_t lanes;
  MetalBuffer packed, convWeights, mixed, decayWeights, timeBias, decay, beta,
      hidden, retained;
  NormWeights mixerNorm;
  std::array<MetalBuffer, kMaxLanes> current, next;
  std::vector<MetalBuffer> packedLayer, mixedLayer, decayLayer, betaLayer;
  uint64_t packedStride, mixedStride, gateStride;

  Fixture(MetalBackend &backend, const GdnShape &geometry, uint32_t laneCount,
          bool float32 = false)
      : shape(geometry), cell(geometry), lanes(laneCount),
        packedStride(uint64_t{kRows} * shape.packedWidth),
        mixedStride(uint64_t{kRows} * shape.convolutionDimension),
        gateStride(uint64_t{kRows} * shape.valueHeads) {
    Random random(0x6D4E1234ULL + laneCount);
    auto alloc = [&](uint64_t bytes, const char *label) {
      return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
    };
    auto fill = [&](MetalBuffer &buffer, float scale) {
      auto *values = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t index = 0; index < buffer.sizeBytes() / 2; ++index)
        values[index] = floatToBf16(random.unit() * scale);
    };
    packed = alloc(kLayers * kMaxLanes * packedStride * 2, "gdn packed");
    fill(packed, 1.0F);
    convWeights = alloc(uint64_t{shape.convolutionDimension} * 4 * 2,
                        "gdn conv weights");
    fill(convWeights, 0.5F);
    mixed = alloc(kLayers * kMaxLanes * mixedStride * 2, "gdn mixed");
    decayWeights = alloc(uint64_t{shape.valueHeads} * 4, "gdn decay weights");
    for (uint32_t head = 0; head < shape.valueHeads; ++head)
      static_cast<float *>(decayWeights.contents())[head] =
          -std::exp(random.unit() * 1.5F);
    timeBias = alloc(uint64_t{shape.valueHeads} * 2, "gdn time bias");
    fill(timeBias, 0.5F);
    decay = alloc(kLayers * kMaxLanes * gateStride * 4, "gdn decay");
    beta = alloc(kLayers * kMaxLanes * gateStride * 2, "gdn beta");
    hidden = alloc(uint64_t{kMaxLanes} * kRows * shape.valueHeads * kHeadDim * 2,
                   "gdn hidden");
    mixerNorm = makeNormWeights(backend, kHeadDim, float32,
                                [&](uint32_t) { return random.unit(); });
    retained = alloc(kMaxLanes * 4, "gdn retained");
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      current[lane] = alloc(cell.bytes, "gdn current");
      next[lane] = alloc(cell.bytes, "gdn next");
      auto *bytes = static_cast<uint8_t *>(current[lane].contents());
      auto *conv = reinterpret_cast<uint16_t *>(bytes);
      for (uint64_t index = 0; index < cell.convBytes / 2; ++index)
        conv[index] = floatToBf16(random.unit());
      auto *state = reinterpret_cast<float *>(bytes + cell.convBytes);
      for (uint64_t index = 0; index < (cell.bytes - cell.convBytes) / 4;
           ++index)
        state[index] = random.unit() * 0.5F;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      packedLayer.push_back(backend.view(packed,
                                         layer * kMaxLanes * packedStride * 2,
                                         kMaxLanes * packedStride * 2));
      mixedLayer.push_back(backend.view(mixed,
                                        layer * kMaxLanes * mixedStride * 2,
                                        kMaxLanes * mixedStride * 2));
      decayLayer.push_back(backend.view(decay,
                                        layer * kMaxLanes * gateStride * 4,
                                        kMaxLanes * gateStride * 4));
      betaLayer.push_back(backend.view(beta, layer * kMaxLanes * gateStride * 2,
                                       kMaxLanes * gateStride * 2));
    }
    clear();
  }

  void clear() {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      std::memset(next[lane].contents(), 0, cell.bytes);
    for (MetalBuffer *buffer :
         {&mixed, &decay, &beta, &hidden})
      std::memset(buffer->contents(), 0, buffer->sizeBytes());
  }

  GdnDecodeBuffers decodeBuffers(uint32_t layer) const {
    return {packedLayer[layer], convWeights,     current,
            next,               mixedLayer[layer], decayWeights,
            timeBias,           decayLayer[layer], betaLayer[layer],
            mixerNorm,          hidden};
  }
  GdnCommitBuffers commitBuffers() const {
    return {packed, mixed, decay, beta, current, next, retained};
  }

  // Row `token` of lane `lane` in layer `layer` of the packed projection.
  const uint16_t *packedRow(uint32_t layer, uint32_t lane,
                            uint32_t token) const {
    return static_cast<const uint16_t *>(packed.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * packedStride +
           uint64_t{token} * shape.packedWidth;
  }
  const uint16_t *mixedRow(uint32_t layer, uint32_t lane,
                           uint32_t token) const {
    return static_cast<const uint16_t *>(mixed.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * mixedStride +
           uint64_t{token} * shape.convolutionDimension;
  }
  const float *decayRow(uint32_t layer, uint32_t lane, uint32_t token) const {
    return static_cast<const float *>(decay.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * gateStride +
           uint64_t{token} * shape.valueHeads;
  }
  const uint16_t *betaRow(uint32_t layer, uint32_t lane,
                          uint32_t token) const {
    return static_cast<const uint16_t *>(beta.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * gateStride +
           uint64_t{token} * shape.valueHeads;
  }
  const uint16_t *hiddenRow(uint32_t lane, uint32_t token,
                            uint32_t head) const {
    return static_cast<const uint16_t *>(hidden.contents()) +
           ((uint64_t{lane} * kRows + token) * shape.valueHeads + head) *
               kHeadDim;
  }
  const uint8_t *cellBytes(const MetalBuffer &buffer) const {
    return static_cast<const uint8_t *>(buffer.contents());
  }
};

// Four-tap causal convolution of one channel at one token, three carried
// rows then the command's rows, rounded to bf16 and gated by SiLU.
double convolutionSilu(const Fixture &fixture, uint32_t layer, uint32_t lane,
                       uint32_t token, uint32_t channel) {
  const uint16_t *weights =
      static_cast<const uint16_t *>(fixture.convWeights.contents()) +
      uint64_t{channel} * 4;
  const uint16_t *carried =
      fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  double value = 0.0;
  for (uint32_t tap = 0; tap < 4; ++tap) {
    const uint32_t position = token + tap;
    const uint16_t input =
        position < 3
            ? carried[position * fixture.shape.convolutionDimension + channel]
            : fixture.packedRow(layer, lane, position - 3)[channel];
    value += double(bf16ToFloat(input)) * bf16ToFloat(weights[tap]);
  }
  value = roundBfloat(value);
  return roundBfloat(value * sigmoid(value));
}

uint16_t convolutionCarry(const Fixture &fixture, uint32_t layer,
                          uint32_t lane, uint32_t consumed, uint32_t row,
                          uint32_t channel) {
  const uint32_t source = consumed + row;
  const uint16_t *carried =
      fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  return source < 3
             ? carried[source * fixture.shape.convolutionDimension + channel]
             : fixture.packedRow(layer, lane, source - 3)[channel];
}

// The q or k row of one key head at one token, from the head's first
// channel: the convolution RMS-normalised over the head, rounded, then scaled
// (the kernel rounds again).
std::vector<double> normalizedHead(const Fixture &fixture, uint32_t layer,
                                   uint32_t lane, uint32_t token,
                                   uint32_t first, double scale) {
  std::vector<double> conv(kHeadDim);
  double squares = 0.0;
  for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
    conv[dim] = convolutionSilu(fixture, layer, lane, token, first + dim);
    squares += conv[dim] * conv[dim];
  }
  const double inverse = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
  for (double &value : conv)
    value = roundBfloat(value * inverse) * scale;
  return conv;
}

// The k and v rows the commit reads; the kernel writes no q rows.
void checkConvolution(const Fixture &fixture, uint32_t layer, uint32_t lane,
                      const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *mixedRow = fixture.mixedRow(layer, lane, token);
    for (uint32_t head = 0; head < shape.keyHeads; ++head) {
      const uint32_t first = keyWidth + head * kHeadDim;
      const std::vector<double> key =
          normalizedHead(fixture, layer, lane, token, first, kKeyScale);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        require(closeBfloat(mixedRow[first + dim], key[dim], 3.0, 1e-6),
                where + ": mixed k row mismatch");
    }
    for (uint32_t channel = 2 * keyWidth; channel < shape.convolutionDimension;
         ++channel)
      require(closeBfloat(mixedRow[channel],
                          convolutionSilu(fixture, layer, lane, token, channel),
                          3.0, 1e-6),
              where + ": mixed v row mismatch");
  }
}

void checkGates(const Fixture &fixture, uint32_t layer, uint32_t lane,
                const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t bOffset = shape.convolutionDimension +
                           shape.valueHeads * kHeadDim;
  const uint32_t aOffset = bOffset + shape.valueHeads;
  const auto *decayWeights =
      static_cast<const float *>(fixture.decayWeights.contents());
  const auto *timeBias =
      static_cast<const uint16_t *>(fixture.timeBias.contents());
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *packed = fixture.packedRow(layer, lane, token);
    const float *decay = fixture.decayRow(layer, lane, token);
    const uint16_t *beta = fixture.betaRow(layer, lane, token);
    for (uint32_t head = 0; head < shape.valueHeads; ++head) {
      const double b = bf16ToFloat(packed[bOffset + head]);
      require(closeBfloat(beta[head], sigmoid(b), 2.0, 1e-6),
              where + ": beta mismatch");
      const double x = roundBfloat(double(bf16ToFloat(packed[aOffset + head])) +
                                   bf16ToFloat(timeBias[head]));
      const double softplus =
          std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
      // The kernel rounds softplus to bf16 with fast transcendentals, so a
      // value near a rounding boundary may land one bf16 step away.
      bool matched = false;
      const uint16_t rounded = floatToBf16(static_cast<float>(softplus));
      for (int step = -1; step <= 1 && !matched; ++step) {
        const double candidate =
            bf16ToFloat(static_cast<uint16_t>(rounded + step));
        matched = closeFloat(decay[head],
                             std::exp(double(decayWeights[head]) * candidate),
                             1e-4);
      }
      require(matched, where + ": decay mismatch");
    }
  }
}

// The delta rule over `tokens` rows of one value head from the lane's
// incoming state, driven by the kernel's own k/v rows and gates. Returns the
// final state; the recurrent output rows, read with the reference q, go to
// `rows` when requested.
std::vector<double> recurrence(const Fixture &fixture, uint32_t layer,
                               uint32_t lane, uint32_t head, uint32_t tokens,
                               std::vector<double> *rows) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  const uint32_t keyHead = head / (shape.valueHeads / shape.keyHeads);
  const float *stateIn =
      fixture.cell.recurrent(fixture.cellBytes(fixture.current[lane]), layer) +
      uint64_t{head} * kHeadDim * kHeadDim;
  std::vector<double> state(stateIn, stateIn + kHeadDim * kHeadDim);
  if (rows)
    rows->assign(uint64_t{tokens} * kHeadDim, 0.0);
  for (uint32_t token = 0; token < tokens; ++token) {
    const uint16_t *mixed = fixture.mixedRow(layer, lane, token);
    const std::vector<double> query =
        rows ? normalizedHead(fixture, layer, lane, token, keyHead * kHeadDim,
                              kQueryScale)
             : std::vector<double>{};
    const uint16_t *key = mixed + keyWidth + keyHead * kHeadDim;
    const uint16_t *value = mixed + 2 * keyWidth + head * kHeadDim;
    const double decay = fixture.decayRow(layer, lane, token)[head];
    const double beta = bf16ToFloat(fixture.betaRow(layer, lane, token)[head]);
    for (uint32_t valueDim = 0; valueDim < kHeadDim; ++valueDim) {
      double *row = state.data() + uint64_t{valueDim} * kHeadDim;
      double memory = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim) {
        row[keyDim] *= decay;
        memory += row[keyDim] * bf16ToFloat(key[keyDim]);
      }
      const double delta = (bf16ToFloat(value[valueDim]) - memory) * beta;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim)
        row[keyDim] += bf16ToFloat(key[keyDim]) * delta;
      if (!rows)
        continue;
      double output = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim)
        output += row[keyDim] * query[keyDim];
      (*rows)[uint64_t{token} * kHeadDim + valueDim] = output;
    }
  }
  return state;
}

void checkState(const Fixture &fixture, uint32_t layer, uint32_t lane,
                uint32_t head, const std::vector<double> &expected,
                const std::string &where) {
  const float *state =
      fixture.cell.recurrent(fixture.cellBytes(fixture.next[lane]), layer) +
      uint64_t{head} * kHeadDim * kHeadDim;
  for (uint32_t index = 0; index < kHeadDim * kHeadDim; ++index)
    require(closeFloat(state[index], expected[index], 1e-4),
            where + ": recurrent state mismatch");
}

void checkCarry(const Fixture &fixture, uint32_t layer, uint32_t lane,
                uint32_t consumed, const std::string &where) {
  const uint16_t *carry =
      fixture.cell.conv(fixture.cellBytes(fixture.next[lane]), layer);
  for (uint32_t row = 0; row < 3; ++row)
    for (uint32_t channel = 0; channel < fixture.shape.convolutionDimension;
         ++channel)
      require(carry[row * fixture.shape.convolutionDimension + channel] ==
                  convolutionCarry(fixture, layer, lane, consumed, row,
                                   channel),
              where + ": convolution carry mismatch");
}

void checkDecode(const Fixture &fixture, uint32_t layer, uint32_t lane) {
  const GdnShape &shape = fixture.shape;
  const std::string where = "decode layer " + std::to_string(layer) +
                            " lane " + std::to_string(lane);
  checkConvolution(fixture, layer, lane, where);
  checkGates(fixture, layer, lane, where);
  checkCarry(fixture, layer, lane, kRows, where);
  // Every layer writes the hidden rows into the same scratch, as the model
  // graph does, so those hold the last layer's values: the gated RMSNorm of
  // the recurrent rows, here the reference's rounded to bf16 as the kernel
  // rounds its own. Where a row cancels, the kernel's fp32 row can round to
  // the neighbouring bf16 value, which moves its normalized value by up to a
  // bf16 step on top of the two units of the output's own rounding, hence
  // three units. Almost every output is the reference's own double rounding;
  // norm weights rounded to bf16 would move a large fraction of them.
  const bool lastLayer = layer + 1 == kLayers;
  const uint32_t zOffset = shape.convolutionDimension;
  uint64_t inexact = 0;
  std::vector<double> rows;
  for (uint32_t head = 0; head < shape.valueHeads; ++head) {
    const std::vector<double> state = recurrence(
        fixture, layer, lane, head, kRows, lastLayer ? &rows : nullptr);
    checkState(fixture, layer, lane, head, state, where);
    if (!lastLayer)
      continue;
    for (uint32_t token = 0; token < kRows; ++token) {
      const uint16_t *packed = fixture.packedRow(layer, lane, token);
      std::array<uint16_t, kHeadDim> recurrent;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        recurrent[dim] =
            floatToBf16(static_cast<float>(rows[token * kHeadDim + dim]));
      const uint16_t *hidden = fixture.hiddenRow(lane, token, head);
      const std::vector<double> exact =
          rmsNorm(recurrent.data(), fixture.mixerNorm, kHeadDim);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double normalized = roundBfloat(exact[dim]);
        const double gate = bf16ToFloat(packed[zOffset + head * kHeadDim + dim]);
        const double reference = normalized * gate * sigmoid(gate);
        require(closeBfloat(hidden[dim], reference, 3.0, 1e-6),
                where + ": hidden mismatch");
        inexact += bf16ToFloat(hidden[dim]) != roundBfloat(reference);
      }
    }
  }
  if (!lastLayer)
    return;
  require(inexact <= uint64_t{kRows} * shape.valueHeads * kHeadDim / 100,
          where + ": hidden differs from the reference in more than 1% of values");
}

void requireUntouched(const Fixture &fixture, uint32_t lane,
                      const std::string &where) {
  const uint8_t *bytes = fixture.cellBytes(fixture.next[lane]);
  for (uint64_t index = 0; index < fixture.cell.bytes; ++index)
    require(bytes[index] == 0, where + ": idle lane cell was written");
}

void runDecode(MetalBackend &backend, const GdnShape &shape, uint32_t lanes,
               bool float32) {
  Fixture fixture(backend, shape, lanes, float32);
  CommandGraph graph;
  const std::string where =
      "lanes " + std::to_string(lanes) + (float32 ? " f32 norm" : "");
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    require(GDN::addDecode(graph, fixture.decodeBuffers(layer), shape, lanes,
                           layer, fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain)
                    .layout == LinearInput::Plain,
            where + ": plain GDN claimed a table");
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    if (lane >= lanes) {
      requireUntouched(fixture, lane, where);
      continue;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      checkDecode(fixture, layer, lane);
  }
  // The commit reads no norm weights; the bf16 pass covers it.
  if (float32)
    return;

  // The commit replays the retained rows from the incoming cell over the
  // decoded q/k/v and gates; eight retained rows leave the decoded cell.
  std::vector<std::vector<uint8_t>> decoded;
  for (uint32_t lane = 0; lane < lanes; ++lane)
    decoded.emplace_back(fixture.cellBytes(fixture.next[lane]),
                         fixture.cellBytes(fixture.next[lane]) +
                             fixture.cell.bytes);
  CommandGraph commit;
  GDN::addCommit(commit, fixture.commitBuffers(), shape, kLayers, lanes,
                 fixture.cell.strides());
  for (uint32_t base = 1; base <= kRows; ++base) {
    auto *retained = static_cast<uint32_t *>(fixture.retained.contents());
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      retained[lane] = 1 + (base + lane - 1) % kRows;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      std::memcpy(fixture.next[lane].contents(), decoded[lane].data(),
                  fixture.cell.bytes);
    static_cast<void>(backend.submitCommand(commit.dispatches()));
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t count = retained[lane];
      const std::string commitWhere =
          where + " commit retained " + std::to_string(count) + " lane " +
          std::to_string(lane);
      if (count == kRows) {
        require(std::memcmp(fixture.next[lane].contents(),
                            decoded[lane].data(), fixture.cell.bytes) == 0,
                commitWhere + ": full acceptance rewrote the cell");
        continue;
      }
      // Every head at the shortest, a middle and the longest partial replay;
      // a spread of heads for the other counts keeps the double-precision
      // replay short under shader validation.
      std::vector<uint32_t> heads{0, shape.valueHeads / 2,
                                  shape.valueHeads - 1};
      if (count == 1 || count == 4 || count == 7) {
        heads.clear();
        for (uint32_t head = 0; head < shape.valueHeads; ++head)
          heads.push_back(head);
      }
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        checkCarry(fixture, layer, lane, count, commitWhere);
        for (uint32_t head : heads)
          checkState(fixture, layer, lane, head,
                     recurrence(fixture, layer, lane, head, count, nullptr),
                     commitWhere);
      }
    }
  }
}

// The out-projection table a fused decode writes into its scratch and the one
// the projection's own preparation writes from the decoded hidden rows.
struct PreparedTables final {
  LinearInput layout;
  uint32_t width, lanes;
  uint64_t tableSize, sumsSize;
  MetalBuffer table, sums, referenceTable, referenceSums;

  PreparedTables(MetalBackend &backend, LinearInput tableLayout, uint32_t hiddenWidth,
                 uint32_t laneCount)
      : layout(tableLayout), width(hiddenWidth), lanes(laneCount),
        tableSize(tableBytes(width, lanes * kRows)),
        sumsSize(tableSumsBytes(layout, width, lanes * kRows)),
        table(sharedBuffer(backend, tableSize)), sums(sharedBuffer(backend, sumsSize)),
        referenceTable(sharedBuffer(backend, tableSize)),
        referenceSums(sharedBuffer(backend, sumsSize)) {}

  LinearScratch scratch() const { return {table, sums, {}, {}}; }
  void addReference(CommandGraph &graph, const MetalBuffer &hidden) const {
    addReferencePreparation(graph, layout, hidden, referenceTable, referenceSums, width, lanes);
  }
  // The decode reported the table it wrote, and wrote the reference's bytes.
  void requireWritten(const PreparedInput &reported, const MetalBuffer &hidden,
                      const std::string &what) const {
    require(reported.layout == layout && reported.source.sameView(hidden),
            what + " did not report the table it wrote");
    require(!std::memcmp(table.contents(), referenceTable.contents(), tableSize),
            what + " table mismatch");
    require(!std::memcmp(sums.contents(), referenceSums.contents(), sumsSize),
            what + " sums mismatch");
  }
};

std::string caseName(const char *test, const GdnShape &shape, uint32_t lanes, LinearInput layout) {
  return std::string(test) + " vh" + std::to_string(shape.valueHeads) + " lanes " +
         std::to_string(lanes) +
         (layout == LinearInput::Plain     ? " plain"
          : layout == LinearInput::Table64 ? " Table64"
                                           : " Table16");
}

void fusedPreparation(MetalBackend &backend, const GdnShape &shape, uint32_t lanes, LinearInput layout,
                      bool float32) {
  const std::string what = caseName("fused GDN", shape, lanes, layout);
  Fixture fixture(backend, shape, lanes, float32);
  const PreparedTables tables(backend, layout, shape.valueHeads * shape.headDimension, lanes);
  CommandGraph reference;
  // Without scratch the table cannot be written: GDN refuses it before it
  // encodes anything.
  rejects([&] {
    GDN::addDecode(reference, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides(),
                   GdnHeadOrder::Grouped, layout);
  });
  require(GDN::addDecode(reference, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides(),
                         GdnHeadOrder::Grouped, LinearInput::Plain)
                  .layout == LinearInput::Plain,
          what + " plain kernel claimed a table");
  tables.addReference(reference, fixture.hidden);
  (void)backend.submitCommand(reference.dispatches());
  // The active lanes' bf16 hidden rows.
  std::vector<uint8_t> expected(uint64_t{lanes} * kRows * tables.width * 2);
  std::memcpy(expected.data(), fixture.hidden.contents(), expected.size());
  fixture.clear();
  auto buffers = fixture.decodeBuffers(0);
  buffers.linearScratch = tables.scratch();
  // A plain consumer gets no table from the scratch it shares.
  CommandGraph unprepared;
  require(GDN::addDecode(unprepared, buffers, shape, lanes, 0, fixture.cell.strides(),
                         GdnHeadOrder::Grouped, LinearInput::Plain).layout == LinearInput::Plain,
          what + " claimed a table for a plain consumer");
  CommandGraph fused;
  const PreparedInput prepared =
      GDN::addDecode(fused, buffers, shape, lanes, 0, fixture.cell.strides(), GdnHeadOrder::Grouped, layout);
  (void)backend.submitCommand(fused.dispatches());
  require(!std::memcmp(expected.data(), fixture.hidden.contents(), expected.size()),
          what + " changed output");
  tables.requireWritten(prepared, fixture.hidden, what);
  for (uint32_t lane=0;lane<lanes;++lane) checkDecode(fixture, 0, lane);
}

// The tiled head order only moves each value head's output block: the tiled
// output, and the table prepared from it in the same dispatch, are the
// grouped output with head h at (h % heads per key) * key heads + h / heads
// per key, byte for byte.
void tiledHeadOrder(MetalBackend &backend, const GdnShape &shape, uint32_t lanes, LinearInput layout,
                    bool float32) {
  Fixture fixture(backend, shape, lanes, float32);
  const uint32_t width = shape.valueHeads * shape.headDimension;
  const uint32_t headsPerKey = shape.valueHeads / shape.keyHeads;
  const uint64_t headBytes = uint64_t{shape.headDimension} * 2;
  const std::string what = caseName("tiled GDN", shape, lanes, layout);
  CommandGraph grouped;
  require(GDN::addDecode(grouped, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides(),
                         GdnHeadOrder::Grouped, LinearInput::Plain)
                  .layout == LinearInput::Plain,
          what + " grouped reference claimed a table");
  (void)backend.submitCommand(grouped.dispatches());
  const auto *hidden = static_cast<const uint8_t *>(fixture.hidden.contents());
  std::vector<uint8_t> expected(fixture.hidden.sizeBytes());
  for (uint64_t row = 0; row < uint64_t{kMaxLanes} * kRows; ++row)
    for (uint32_t head = 0; head < shape.valueHeads; ++head) {
      const uint32_t tiled = (head % headsPerKey) * shape.keyHeads + head / headsPerKey;
      std::memcpy(expected.data() + (row * shape.valueHeads + tiled) * headBytes,
                  hidden + (row * shape.valueHeads + head) * headBytes, headBytes);
    }
  fixture.clear();
  CommandGraph tiled;
  if (layout == LinearInput::Plain) {
    require(GDN::addDecode(tiled, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides(),
                           GdnHeadOrder::Tiled, LinearInput::Plain).layout == LinearInput::Plain,
            what + " claimed a table");
    (void)backend.submitCommand(tiled.dispatches());
  } else {
    const PreparedTables tables(backend, layout, width, lanes);
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch = tables.scratch();
    const PreparedInput prepared =
        GDN::addDecode(tiled, buffers, shape, lanes, 0, fixture.cell.strides(), GdnHeadOrder::Tiled, layout);
    tables.addReference(tiled, fixture.hidden);
    (void)backend.submitCommand(tiled.dispatches());
    tables.requireWritten(prepared, fixture.hidden, what);
  }
  require(!std::memcmp(expected.data(), hidden, expected.size()),
          what + " output is not the grouped output in tiled head order");
}

void rejectsInvalid(MetalBackend &backend) {
  const GdnShape &shape = kShapes[1];
  Fixture fixture(backend, shape, 1);
  CommandGraph graph;
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0), shape, 0, 0,
                   fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain);
  });
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0), shape, kMaxLanes + 1, 0,
                   fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain);
  });
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0),
                   GdnShape{16, 40, 128, 9216, 14400}, 1, 0,
                   fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain);
  });
  rejects([&] {
    GDN::addCommit(graph, fixture.commitBuffers(), shape, 0, 1,
                   fixture.cell.strides());
  });
  rejects([&] {
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch.input = sharedBuffer(backend, 16);
    buffers.linearScratch.sums = sharedBuffer(backend, 4);
    GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                   splash::ops::LinearInput::Table64);
  });
  rejects([&] {
    // Sums sized for the affine table are below the GGUF table's.
    const uint32_t width = shape.valueHeads * shape.headDimension;
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch.input = sharedBuffer(backend, tableBytes(width, kRows));
    buffers.linearScratch.sums = sharedBuffer(backend, tableSumsBytes(LinearInput::Table64, width, kRows));
    GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                   LinearInput::Table16);
  });
  rejects([&] {
    // F32 weights need twice the bytes of bf16 ones.
    auto buffers = fixture.decodeBuffers(0);
    buffers.mixerNorm.float32 = true;
    GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                   LinearInput::Plain);
  });
  require(graph.empty(), "invalid GDN request partially encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: gdn-decode METALLIB");
    MetalBackend backend(argv[1]);
    rejectsInvalid(backend);
    // Table64 feeds the affine models, whose norms are bf16; Table16 a GGUF's, whose norms are F32.
    for (LinearInput layout : {LinearInput::Table64, LinearInput::Table16})
      for (const GdnShape &shape : kShapes)
        for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes) {
          fusedPreparation(backend, shape, lanes, layout, layout == LinearInput::Table16);
          tiledHeadOrder(backend, shape, lanes, layout, layout == LinearInput::Table16);
        }
    // A GGUF's out-projection on the staged tile reads the tiled rows plain.
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        tiledHeadOrder(backend, shape, lanes, LinearInput::Plain, true);
    for (bool float32 : {false, true})
      for (const GdnShape &shape : kShapes)
        for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
          runDecode(backend, shape, lanes, float32);
    std::cout << "gdn_decode_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gdn_decode_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
