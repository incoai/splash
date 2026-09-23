// GDN verify decode and commit kernels against a direct CPU reference: the
// four-tap convolution with SiLU, the q/k RMS norms, the gates, the eight-row
// delta-rule recurrence over the fp32 state, the gated RMSNorm of the
// recurrent rows and the convolution carry, for both compiled geometries,
// every lane count, a two-layer state cell (so the layer offsets are
// exercised) and every retained count of the commit. The recurrence and the
// gate are checked from the kernel's own q/k/v and gates after those were
// checked against the reference, so their tolerances stay at fp32 accuracy.
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/GDN.hpp"

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

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kMaxLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kLayers = 2;
constexpr std::array kShapes{GdnShape{16, 48, 128, 10240, 16640},
                             GdnShape{16, 32, 128, 8192, 12544}};

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid GDN request was accepted");
}

uint16_t toBfloat(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

float fromBfloat(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

double roundBfloat(double value) {
  return fromBfloat(toBfloat(static_cast<float>(value)));
}

// One bf16 unit in the last place at the reference's magnitude.
double bfloatUlp(double reference) {
  int exponent = 0;
  std::frexp(std::fabs(reference), &exponent);
  return std::ldexp(1.0, exponent - 8);
}

bool closeBfloat(uint16_t got, double reference, double ulps, double floor) {
  return std::fabs(double(fromBfloat(got)) - reference) <=
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

uint64_t align16k(uint64_t bytes) { return (bytes + 16383) & ~uint64_t{16383}; }

double sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }

// The production state cell layout: every layer's conv rows, then every
// layer's recurrent state, both padded to 16 KiB.
struct Cell final {
  uint64_t convLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convBytes = 0;
  uint64_t bytes = 0;

  explicit Cell(const GdnShape &shape)
      : convLayerBytes(align16k(uint64_t{3} * shape.convolutionDimension * 2)),
        recurrentLayerBytes(align16k(uint64_t{shape.valueHeads} * kHeadDim *
                                     kHeadDim * 4)),
        convBytes(kLayers * convLayerBytes),
        bytes(convBytes + kLayers * recurrentLayerBytes) {}

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

struct Fixture final {
  const GdnShape &shape;
  Cell cell;
  uint32_t lanes;
  MetalBuffer packed, convWeights, mixed, decayWeights, timeBias, decay, beta,
      recurrent, mixerNorm, hidden, arrived, generation, retained;
  std::array<MetalBuffer, kMaxLanes> current, next;
  std::vector<MetalBuffer> packedLayer, mixedLayer, decayLayer, betaLayer;
  uint64_t packedStride, mixedStride, gateStride;

  Fixture(MetalBackend &backend, const GdnShape &geometry, uint32_t laneCount)
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
        values[index] = toBfloat(random.unit() * scale);
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
    const uint64_t rowBytes =
        uint64_t{kMaxLanes} * kRows * shape.valueHeads * kHeadDim * 2;
    recurrent = alloc(rowBytes, "gdn recurrent rows");
    hidden = alloc(rowBytes, "gdn hidden");
    mixerNorm = alloc(uint64_t{kHeadDim} * 2, "gdn mixer norm");
    fill(mixerNorm, 1.0F);
    arrived = alloc(kMaxLanes * 4, "gdn arrived");
    generation = alloc(kMaxLanes * 4, "gdn generation");
    retained = alloc(kMaxLanes * 4, "gdn retained");
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      current[lane] = alloc(cell.bytes, "gdn current");
      next[lane] = alloc(cell.bytes, "gdn next");
      auto *bytes = static_cast<uint8_t *>(current[lane].contents());
      auto *conv = reinterpret_cast<uint16_t *>(bytes);
      for (uint64_t index = 0; index < cell.convBytes / 2; ++index)
        conv[index] = toBfloat(random.unit());
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
         {&mixed, &decay, &beta, &recurrent, &hidden, &arrived, &generation})
      std::memset(buffer->contents(), 0, buffer->sizeBytes());
  }

  GdnDecodeBuffers decodeBuffers(uint32_t layer) const {
    return {packedLayer[layer], convWeights,     current,
            next,               mixedLayer[layer], decayWeights,
            timeBias,           decayLayer[layer], betaLayer[layer],
            recurrent,          mixerNorm,       hidden,
            arrived,            generation};
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
  const uint16_t *rowsOf(const MetalBuffer &buffer, uint32_t lane,
                         uint32_t token, uint32_t head) const {
    return static_cast<const uint16_t *>(buffer.contents()) +
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
    value += double(fromBfloat(input)) * fromBfloat(weights[tap]);
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

void checkConvolution(const Fixture &fixture, uint32_t layer, uint32_t lane,
                      const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *mixedRow = fixture.mixedRow(layer, lane, token);
    std::vector<double> conv(shape.convolutionDimension);
    for (uint32_t channel = 0; channel < shape.convolutionDimension; ++channel)
      conv[channel] = convolutionSilu(fixture, layer, lane, token, channel);
    // q and k: RMS-normalised per key head, rounded, then scaled and
    // rounded again.
    for (uint32_t part = 0; part < 2; ++part) {
      const double scale = part == 0 ? 0.0078125 : 0.08838834765;
      for (uint32_t head = 0; head < shape.keyHeads; ++head) {
        const uint32_t base = part * keyWidth + head * kHeadDim;
        double squares = 0.0;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim)
          squares += conv[base + dim] * conv[base + dim];
        const double inverse = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
          const double normalized = roundBfloat(conv[base + dim] * inverse);
          require(closeBfloat(mixedRow[base + dim], normalized * scale, 3.0,
                              1e-6),
                  where + ": mixed q/k row mismatch");
        }
      }
    }
    for (uint32_t channel = 2 * keyWidth; channel < shape.convolutionDimension;
         ++channel)
      require(closeBfloat(mixedRow[channel], conv[channel], 3.0, 1e-6),
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
      const double b = fromBfloat(packed[bOffset + head]);
      require(closeBfloat(beta[head], sigmoid(b), 2.0, 1e-6),
              where + ": beta mismatch");
      const double x = roundBfloat(double(fromBfloat(packed[aOffset + head])) +
                                   fromBfloat(timeBias[head]));
      const double softplus =
          std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
      // The kernel rounds softplus to bf16 with fast transcendentals, so a
      // value near a rounding boundary may land one bf16 step away.
      bool matched = false;
      const uint16_t rounded = toBfloat(static_cast<float>(softplus));
      for (int step = -1; step <= 1 && !matched; ++step) {
        const double candidate =
            fromBfloat(static_cast<uint16_t>(rounded + step));
        matched = closeFloat(decay[head],
                             std::exp(double(decayWeights[head]) * candidate),
                             1e-4);
      }
      require(matched, where + ": decay mismatch");
    }
  }
}

// The delta rule over `tokens` rows of one value head from the lane's
// incoming state, driven by the kernel's own q/k/v rows and gates. Returns
// the final state; the recurrent output rows go to `rows` when requested.
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
    const uint16_t *query = mixed + keyHead * kHeadDim;
    const uint16_t *key = mixed + keyWidth + keyHead * kHeadDim;
    const uint16_t *value = mixed + 2 * keyWidth + head * kHeadDim;
    const double decay = fixture.decayRow(layer, lane, token)[head];
    const double beta = fromBfloat(fixture.betaRow(layer, lane, token)[head]);
    for (uint32_t valueDim = 0; valueDim < kHeadDim; ++valueDim) {
      double *row = state.data() + uint64_t{valueDim} * kHeadDim;
      double memory = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim) {
        row[keyDim] *= decay;
        memory += row[keyDim] * fromBfloat(key[keyDim]);
      }
      const double delta = (fromBfloat(value[valueDim]) - memory) * beta;
      double output = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim) {
        row[keyDim] += fromBfloat(key[keyDim]) * delta;
        output += row[keyDim] * fromBfloat(query[keyDim]);
      }
      if (rows)
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
  // Every layer writes the recurrent rows and the hidden rows into the same
  // scratch, as the model graph does, so those hold the last layer's values.
  const bool lastLayer = layer + 1 == kLayers;
  std::vector<double> rows;
  for (uint32_t head = 0; head < shape.valueHeads; ++head) {
    const std::vector<double> state =
        recurrence(fixture, layer, lane, head, kRows, &rows);
    checkState(fixture, layer, lane, head, state, where);
    if (!lastLayer)
      continue;
    for (uint32_t token = 0; token < kRows; ++token) {
      const uint16_t *recurrent =
          fixture.rowsOf(fixture.recurrent, lane, token, head);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        require(closeBfloat(recurrent[dim], rows[token * kHeadDim + dim], 2.0,
                            1e-6),
                where + ": recurrent row mismatch");
    }
  }
  if (!lastLayer)
    return;
  // The gated RMSNorm, from the kernel's own recurrent rows.
  const auto *norm =
      static_cast<const uint16_t *>(fixture.mixerNorm.contents());
  const uint32_t zOffset = shape.convolutionDimension;
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *packed = fixture.packedRow(layer, lane, token);
    for (uint32_t head = 0; head < shape.valueHeads; ++head) {
      const uint16_t *recurrent =
          fixture.rowsOf(fixture.recurrent, lane, token, head);
      const uint16_t *hidden =
          fixture.rowsOf(fixture.hidden, lane, token, head);
      double squares = 0.0;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double row = fromBfloat(recurrent[dim]);
        squares += row * row;
      }
      const double inverse = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double normalized = roundBfloat(
            fromBfloat(recurrent[dim]) * inverse * fromBfloat(norm[dim]));
        const double gate = fromBfloat(packed[zOffset + head * kHeadDim + dim]);
        require(closeBfloat(hidden[dim], normalized * gate * sigmoid(gate), 2.0,
                            1e-6),
                where + ": hidden mismatch");
      }
    }
  }
}

void requireUntouched(const Fixture &fixture, uint32_t lane,
                      const std::string &where) {
  const uint8_t *bytes = fixture.cellBytes(fixture.next[lane]);
  for (uint64_t index = 0; index < fixture.cell.bytes; ++index)
    require(bytes[index] == 0, where + ": idle lane cell was written");
}

void runDecode(MetalBackend &backend, const GdnShape &shape, uint32_t lanes) {
  Fixture fixture(backend, shape, lanes);
  CommandGraph graph;
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    GDN::addDecode(graph, fixture.decodeBuffers(layer), shape, lanes, layer,
                   fixture.cell.strides());
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const std::string where = "lanes " + std::to_string(lanes);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    const auto *generation =
        static_cast<const uint32_t *>(fixture.generation.contents());
    const auto *arrived =
        static_cast<const uint32_t *>(fixture.arrived.contents());
    require(arrived[lane] == 0, where + ": completion counter not reset");
    if (lane >= lanes) {
      require(generation[lane] == 0, where + ": idle lane completed");
      requireUntouched(fixture, lane, where);
      continue;
    }
    require(generation[lane] == kLayers, where + ": layer completion count");
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      checkDecode(fixture, layer, lane);
  }

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

// The kernel that prepares a plain bf16 input in `layout`, the reference for
// the GDN kernels that write the table themselves.
const char *prepareKernel(LinearInput layout) {
  return layout == LinearInput::Table16 ? "decode_linear_gguf_prepare" : "decode_linear_q4_prepare";
}

void fusedPreparation(MetalBackend &backend, const GdnShape &shape, uint32_t lanes, LinearInput layout) {
  Fixture fixture(backend, shape, lanes);
  const uint32_t width = shape.valueHeads * shape.headDimension, rows = lanes * kRows;
  const uint64_t tableSize = tableBytes(width, rows), sumsSize = tableSumsBytes(layout, width, rows);
  auto table = backend.allocateBuffer(tableSize);
  auto sums = backend.allocateBuffer(sumsSize);
  auto referenceTable = backend.allocateBuffer(tableSize);
  auto referenceSums = backend.allocateBuffer(sumsSize);
  CommandGraph reference;
  GDN::addDecode(reference, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides());
  reference.add(prepareKernel(layout), {fixture.hidden, referenceTable, referenceSums},
                width, {width / 32, lanes, 1}, {128, 1, 1});
  (void)backend.submitCommand(reference.dispatches());
  std::vector<uint8_t> expected(width * 16 * lanes);
  std::memcpy(expected.data(), fixture.hidden.contents(), expected.size());
  fixture.clear();
  auto buffers = fixture.decodeBuffers(0);
  buffers.linearScratch = {table, sums, {}, {}};
  CommandGraph fused;
  GDN::addDecode(fused, buffers, shape, lanes, 0, fixture.cell.strides(), GdnHeadOrder::Grouped, layout);
  (void)backend.submitCommand(fused.dispatches());
  require(!std::memcmp(expected.data(), fixture.hidden.contents(), expected.size()),
          "fused GDN changed output");
  require(!std::memcmp(table.contents(), referenceTable.contents(), tableSize),
          "fused GDN table mismatch");
  require(!std::memcmp(sums.contents(), referenceSums.contents(), sumsSize),
          "fused GDN sums mismatch");
  for (uint32_t lane=0;lane<lanes;++lane) checkDecode(fixture, 0, lane);
}

// The tiled head order only moves each value head's output block: the tiled
// output, and the Q4 table prepared from it in the same dispatch, are the
// grouped output with head h at (h % heads per key) * key heads + h / heads
// per key, byte for byte.
void tiledHeadOrder(MetalBackend &backend, const GdnShape &shape, uint32_t lanes, LinearInput layout) {
  Fixture fixture(backend, shape, lanes);
  const uint32_t width = shape.valueHeads * shape.headDimension;
  const uint32_t headsPerKey = shape.valueHeads / shape.keyHeads;
  const uint64_t headBytes = uint64_t{shape.headDimension} * 2;
  CommandGraph grouped;
  GDN::addDecode(grouped, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides());
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
  const uint32_t rows = lanes * kRows;
  const uint64_t tableSize = tableBytes(width, rows), sumsSize = tableSumsBytes(layout, width, rows);
  auto table = backend.allocateBuffer(tableSize);
  auto sums = backend.allocateBuffer(sumsSize);
  auto referenceTable = backend.allocateBuffer(tableSize);
  auto referenceSums = backend.allocateBuffer(sumsSize);
  auto buffers = fixture.decodeBuffers(0);
  buffers.linearScratch = {table, sums, {}, {}};
  CommandGraph tiled;
  GDN::addDecode(tiled, buffers, shape, lanes, 0, fixture.cell.strides(), GdnHeadOrder::Tiled, layout);
  tiled.add(prepareKernel(layout), {fixture.hidden, referenceTable, referenceSums},
            width, {width / 32, lanes, 1}, {128, 1, 1});
  (void)backend.submitCommand(tiled.dispatches());
  require(!std::memcmp(expected.data(), hidden, expected.size()),
          "tiled GDN output is not the grouped output in tiled head order");
  require(!std::memcmp(table.contents(), referenceTable.contents(), tableSize),
          "tiled GDN table mismatch");
  require(!std::memcmp(sums.contents(), referenceSums.contents(), sumsSize),
          "tiled GDN sums mismatch");
}

void rejectsInvalid(MetalBackend &backend) {
  const GdnShape &shape = kShapes[1];
  Fixture fixture(backend, shape, 1);
  CommandGraph graph;
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0), shape, 0, 0,
                   fixture.cell.strides());
  });
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0), shape, kMaxLanes + 1, 0,
                   fixture.cell.strides());
  });
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0),
                   GdnShape{16, 40, 128, 9216, 14400}, 1, 0,
                   fixture.cell.strides());
  });
  rejects([&] {
    GDN::addCommit(graph, fixture.commitBuffers(), shape, 0, 1,
                   fixture.cell.strides());
  });
  rejects([&] {
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch.input = backend.allocateBuffer(16);
    buffers.linearScratch.sums = backend.allocateBuffer(4);
    GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                   splash::ops::LinearInput::Table64);
  });
  rejects([&] {
    // Sums sized for the affine table are below the GGUF table's.
    const uint32_t width = shape.valueHeads * shape.headDimension;
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch.input = backend.allocateBuffer(tableBytes(width, kRows));
    buffers.linearScratch.sums = backend.allocateBuffer(tableSumsBytes(LinearInput::Table64, width, kRows));
    GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                   LinearInput::Table16);
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
    for (LinearInput layout : {LinearInput::Table64, LinearInput::Table16})
      for (const GdnShape &shape : kShapes)
        for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes) {
          fusedPreparation(backend, shape, lanes, layout);
          tiledHeadOrder(backend, shape, lanes, layout);
        }
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        runDecode(backend, shape, lanes);
    std::cout << "gdn_decode_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gdn_decode_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
