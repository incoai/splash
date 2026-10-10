// GDN verify decode kernels against a direct CPU reference: the four-tap
// convolution with SiLU, the q/k RMS norms, the gates, the eight-row
// delta-rule recurrence over the fp32 state, the gated RMSNorm of the
// recurrent rows and the convolution carry, for both compiled geometries,
// every lane count and a two-layer state cell (so the layer offsets are
// exercised). A step folds the rows of the lane's previous step it retained
// into the state before its own: a first step retained none, and a second one
// each count of the first's rows. The step's tape is checked against the
// reference, and the recurrent state at fp32 accuracy from the kernel's own
// k/v and gates on the tape; the hidden rows from the recurrent rows read
// from that state with the reference q and rounded to bf16. The decode
// gate's hidden rows are also checked bit for bit against the prefill gate's
// from the same recurrent rows.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "model/StateLayout.hpp"
#include "ops/GDN.hpp"

#include "LinearInputReference.hpp"
#include "LinearNumerics.hpp"
#include "NormReference.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;
using namespace splash::test;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kMaxLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kHeadDim = 128;
// The convolution's taps; a state carries the inputs of all but the current
// token's.
constexpr uint32_t kTaps = SPLASH_GDN_CONVOLUTION_TAPS, kCarried = kTaps - 1;
constexpr uint32_t kLayers = 2;
// The scales of the normalized q and k rows.
constexpr double kQueryScale = 0.0078125, kKeyScale = 0.08838834765;
constexpr std::array kShapes{GdnShape{16, 48, 128, 10240, 16640},
                             GdnShape{16, 32, 128, 8192, 12544}};

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

// The mixer norm is bf16, or F32 as a GGUF stores it. Lane l's tape slots
// are slot(l, 0) and slot(l, 1): a step reads the rows the lane retained from
// one and leaves its own in the other, and advance() swaps them, with the
// cells, for the lane's next step.
struct Fixture final {
  const GdnShape &shape;
  Cell cell;
  uint32_t lanes;
  uint64_t packedStride, tapeBytes, kvWidth;
  Random random;
  MetalBuffer packed, convWeights, decayWeights, timeBias, hidden, tape;
  NormWeights mixerNorm;
  std::array<MetalBuffer, kMaxLanes> current, next;
  std::array<GdnTapeLane, kMaxLanes> tapeLanes{};
  std::vector<MetalBuffer> packedLayer;

  Fixture(MetalBackend &backend, const GdnShape &geometry, uint32_t laneCount,
          bool float32 = false)
      : shape(geometry), cell(geometry), lanes(laneCount),
        packedStride(uint64_t{kRows} * shape.packedWidth),
        tapeBytes(gdnTapeLayerBytes(shape)),
        kvWidth((shape.keyHeads + shape.valueHeads) * kHeadDim),
        random(0x6D4E1234ULL + laneCount) {
    auto alloc = [&](uint64_t bytes, const char *label) {
      return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
    };
    packed = alloc(kLayers * kMaxLanes * packedStride * 2, "gdn packed");
    fillPacked();
    convWeights = alloc(uint64_t{shape.convolutionDimension} * kTaps * 2,
                        "gdn conv weights");
    fill(convWeights, 0.5F);
    decayWeights = alloc(uint64_t{shape.valueHeads} * 4, "gdn decay weights");
    for (uint32_t head = 0; head < shape.valueHeads; ++head)
      static_cast<float *>(decayWeights.contents())[head] =
          -std::exp(random.unit() * 1.5F);
    timeBias = alloc(uint64_t{shape.valueHeads} * 2, "gdn time bias");
    fill(timeBias, 0.5F);
    hidden = alloc(uint64_t{kMaxLanes} * kRows * shape.valueHeads * kHeadDim * 2,
                   "gdn hidden");
    tape = alloc(2 * kMaxLanes * kLayers * tapeBytes, "gdn tape");
    std::memset(tape.contents(), 0, tape.sizeBytes());
    mixerNorm = makeNormWeights(backend, kHeadDim, float32,
                                [&](uint32_t) { return random.unit(); });
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      tapeLanes[lane] = {slot(lane, 0), slot(lane, 1), 0};
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
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      packedLayer.push_back(backend.view(packed,
                                         layer * kMaxLanes * packedStride * 2,
                                         kMaxLanes * packedStride * 2));
    clear();
  }

  void fill(MetalBuffer &buffer, float scale) {
    auto *values = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t index = 0; index < buffer.sizeBytes() / 2; ++index)
      values[index] = floatToBf16(random.unit() * scale);
  }
  // New packed rows for the next step.
  void fillPacked() { fill(packed, 1.0F); }

  void clear() {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      std::memset(next[lane].contents(), 0, cell.bytes);
    std::memset(hidden.contents(), 0, hidden.sizeBytes());
  }

  uint64_t slot(uint32_t lane, uint32_t parity) const {
    return (uint64_t{lane} * 2 + parity) * kLayers * tapeBytes;
  }
  // Every lane's next step: the cells and tape slots trade places, and the
  // step folds in `pending[lane]` rows of this one.
  void advance(std::span<const uint32_t> pending) {
    std::swap(current, next);
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      tapeLanes[lane] = {tapeLanes[lane].stepSlot, tapeLanes[lane].pendingSlot, pending[lane]};
  }

  GdnDecodeBuffers decodeBuffers(uint32_t layer) const {
    return {packedLayer[layer], convWeights, current, next, tape, tapeLanes,
            decayWeights, timeBias, mixerNorm, hidden};
  }

  // Row `token` of lane `lane` in layer `layer` of the packed projection.
  const uint16_t *packedRow(uint32_t layer, uint32_t lane,
                            uint32_t token) const {
    return static_cast<const uint16_t *>(packed.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * packedStride +
           uint64_t{token} * shape.packedWidth;
  }
  // Layer `layer` of the tape at offset `slot`.
  const uint8_t *tapeLayer(uint64_t slot, uint32_t layer) const {
    return static_cast<const uint8_t *>(tape.contents()) + slot + layer * tapeBytes;
  }
  const uint16_t *tapeInputs(uint64_t slot, uint32_t layer, uint32_t token) const {
    return reinterpret_cast<const uint16_t *>(tapeLayer(slot, layer)) +
           uint64_t{token} * shape.convolutionDimension;
  }
  const uint16_t *tapeKv(uint64_t slot, uint32_t layer, uint32_t token) const {
    return reinterpret_cast<const uint16_t *>(
               tapeLayer(slot, layer) + gdn_tape_kv_offset(shape.convolutionDimension)) +
           uint64_t{token} * kvWidth;
  }
  const float *tapeDecay(uint64_t slot, uint32_t layer, uint32_t token) const {
    return reinterpret_cast<const float *>(
               tapeLayer(slot, layer) +
               gdn_tape_decay_offset(shape.convolutionDimension, kvWidth)) +
           uint64_t{token} * shape.valueHeads;
  }
  const uint16_t *tapeBeta(uint64_t slot, uint32_t layer, uint32_t token) const {
    return reinterpret_cast<const uint16_t *>(
               tapeLayer(slot, layer) +
               gdn_tape_beta_offset(shape.convolutionDimension, kvWidth, shape.valueHeads)) +
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

// What one lane's step reads in one layer: the convolution inputs before its
// rows, the carried rows of the state its retained rows leave, and its packed
// rows.
struct StepInputs final {
  std::vector<uint16_t> history; // [kCarried][conv]
  std::vector<uint16_t> packed;  // [kRows][packed width]
};

// A step that retained no rows of the one before: the current cell's carried
// rows and the packed rows.
StepInputs firstStep(const Fixture &fixture, uint32_t layer, uint32_t lane) {
  const uint64_t carried = uint64_t{kCarried} * fixture.shape.convolutionDimension;
  const uint16_t *history = fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  const uint16_t *packed = fixture.packedRow(layer, lane, 0);
  return {{history, history + carried}, {packed, packed + fixture.packedStride}};
}

// The step after `previous`, of which the lane retained `retained` rows: the
// last carried rows of those inputs, and the packed rows.
StepInputs laterStep(const Fixture &fixture, const StepInputs &previous, uint32_t retained, uint32_t layer,
                     uint32_t lane) {
  const uint32_t conv = fixture.shape.convolutionDimension;
  StepInputs step;
  step.history.resize(uint64_t{kCarried} * conv);
  for (uint32_t row = 0; row < kCarried; ++row) {
    const uint32_t source = retained + row;
    for (uint32_t channel = 0; channel < conv; ++channel)
      step.history[row * conv + channel] =
          source < kCarried ? previous.history[source * conv + channel]
                            : previous.packed[(source - kCarried) * fixture.shape.packedWidth + channel];
  }
  const uint16_t *packed = fixture.packedRow(layer, lane, 0);
  step.packed.assign(packed, packed + fixture.packedStride);
  return step;
}

// The causal convolution of one channel at one token over its taps, the
// inputs before the step then its rows, rounded to bf16 and gated by SiLU.
double convolutionSilu(const Fixture &fixture, const StepInputs &step,
                       uint32_t token, uint32_t channel) {
  const uint16_t *weights =
      static_cast<const uint16_t *>(fixture.convWeights.contents()) +
      uint64_t{channel} * kTaps;
  double value = 0.0;
  for (uint32_t tap = 0; tap < kTaps; ++tap) {
    const uint32_t position = token + tap;
    const uint16_t input =
        position < kCarried
            ? step.history[position * fixture.shape.convolutionDimension + channel]
            : step.packed[(position - kCarried) * fixture.shape.packedWidth + channel];
    value += double(bf16ToFloat(input)) * bf16ToFloat(weights[tap]);
  }
  value = roundBfloat(value);
  return roundBfloat(value * sigmoid(value));
}

// The q or k row of one key head at one token, from the head's first
// channel: the convolution RMS-normalised over the head, rounded, then scaled
// (the kernel rounds again).
std::vector<double> normalizedHead(const Fixture &fixture, const StepInputs &step,
                                   uint32_t token, uint32_t first, double scale) {
  std::vector<double> conv(kHeadDim);
  double squares = 0.0;
  for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
    conv[dim] = convolutionSilu(fixture, step, token, first + dim);
    squares += conv[dim] * conv[dim];
  }
  const double inverse =
      1.0 / std::sqrt(squares / kHeadDim + SPLASH_RMS_EPSILON);
  for (double &value : conv)
    value = roundBfloat(value * inverse) * scale;
  return conv;
}

// The step's tape: its rows' convolution inputs as packed, each key head's k
// and each value head's v rows.
void checkTapeRows(const Fixture &fixture, const StepInputs &step, uint32_t layer, uint32_t lane,
                   const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  const uint64_t slot = fixture.tapeLanes[lane].stepSlot;
  for (uint32_t token = 0; token < kRows; ++token) {
    require(!std::memcmp(fixture.tapeInputs(slot, layer, token), step.packed.data() + token * shape.packedWidth,
                         uint64_t{shape.convolutionDimension} * 2),
            where + ": tape inputs are not the packed rows");
    const uint16_t *kv = fixture.tapeKv(slot, layer, token);
    for (uint32_t head = 0; head < shape.keyHeads; ++head) {
      const std::vector<double> key =
          normalizedHead(fixture, step, token, keyWidth + head * kHeadDim, kKeyScale);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        require(closeBfloat(kv[head * kHeadDim + dim], key[dim], 3.0, 1e-6),
                where + ": tape k row mismatch");
    }
    for (uint32_t channel = 2 * keyWidth; channel < shape.convolutionDimension; ++channel)
      require(closeBfloat(kv[channel - keyWidth], convolutionSilu(fixture, step, token, channel), 3.0, 1e-6),
              where + ": tape v row mismatch");
  }
}

void checkGates(const Fixture &fixture, const StepInputs &step, uint32_t layer, uint32_t lane,
                const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t bOffset = shape.convolutionDimension +
                           shape.valueHeads * kHeadDim;
  const uint32_t aOffset = bOffset + shape.valueHeads;
  const uint64_t slot = fixture.tapeLanes[lane].stepSlot;
  const auto *decayWeights =
      static_cast<const float *>(fixture.decayWeights.contents());
  const auto *timeBias =
      static_cast<const uint16_t *>(fixture.timeBias.contents());
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *packed = step.packed.data() + token * shape.packedWidth;
    const float *decay = fixture.tapeDecay(slot, layer, token);
    const uint16_t *beta = fixture.tapeBeta(slot, layer, token);
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

// The delta rule over `tokens` rows of one value head from `state`, driven by
// the k/v rows and gates the kernel left on the tape at `slot`. Returns the
// final state; the recurrent output rows, read with the reference q of
// `step`, go to `rows` when requested, and the sums of their terms'
// magnitudes, which bound their fp32 error, to `magnitudes`.
std::vector<double> recurrence(const Fixture &fixture, const float *state, uint64_t slot, uint32_t layer,
                               uint32_t head, uint32_t tokens, const StepInputs *step,
                               std::vector<double> *rows, std::vector<double> *magnitudes = nullptr) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  const uint32_t keyHead = head / (shape.valueHeads / shape.keyHeads);
  const float *stateIn = state + uint64_t{head} * kHeadDim * kHeadDim;
  std::vector<double> result(stateIn, stateIn + kHeadDim * kHeadDim);
  if (rows)
    rows->assign(uint64_t{tokens} * kHeadDim, 0.0);
  if (magnitudes)
    magnitudes->assign(uint64_t{tokens} * kHeadDim, 0.0);
  for (uint32_t token = 0; token < tokens; ++token) {
    const uint16_t *kv = fixture.tapeKv(slot, layer, token);
    const std::vector<double> query =
        rows ? normalizedHead(fixture, *step, token, keyHead * kHeadDim, kQueryScale)
             : std::vector<double>{};
    const uint16_t *key = kv + keyHead * kHeadDim;
    const uint16_t *value = kv + keyWidth + head * kHeadDim;
    const double decay = fixture.tapeDecay(slot, layer, token)[head];
    const double beta = bf16ToFloat(fixture.tapeBeta(slot, layer, token)[head]);
    for (uint32_t valueDim = 0; valueDim < kHeadDim; ++valueDim) {
      double *row = result.data() + uint64_t{valueDim} * kHeadDim;
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
      double output = 0.0, magnitude = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim) {
        output += row[keyDim] * query[keyDim];
        magnitude += std::fabs(row[keyDim] * query[keyDim]);
      }
      (*rows)[uint64_t{token} * kHeadDim + valueDim] = output;
      if (magnitudes)
        (*magnitudes)[uint64_t{token} * kHeadDim + valueDim] = magnitude;
    }
  }
  return result;
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

// The next cell carries the inputs before the step's rows.
void checkCarry(const Fixture &fixture, const StepInputs &step, uint32_t layer, uint32_t lane,
                const std::string &where) {
  require(!std::memcmp(fixture.cell.conv(fixture.cellBytes(fixture.next[lane]), layer), step.history.data(),
                       step.history.size() * 2),
          where + ": convolution carry mismatch");
}

// One lane's step in one layer, from its inputs: its tape, the carry of the
// state its retained rows leave and, in the last layer (every layer writes the
// same hidden scratch, as the model graph does), its hidden rows from that
// state as the kernel wrote it.
void checkStep(const Fixture &fixture, const StepInputs &step, uint32_t layer, uint32_t lane,
               const std::string &where) {
  const GdnShape &shape = fixture.shape;
  checkTapeRows(fixture, step, layer, lane, where);
  checkGates(fixture, step, layer, lane, where);
  checkCarry(fixture, step, layer, lane, where);
  if (layer + 1 != kLayers)
    return;
  // The gated RMSNorm of the recurrent rows, here the reference's rounded to
  // bf16 as the kernel rounds its own. Where a row cancels, the kernel's fp32
  // row can round to the neighbouring bf16 value, which moves its normalized
  // value by up to a bf16 step on top of the two units of the output's own
  // rounding, hence three units. The reference's q is within two bf16 units
  // of the kernel's, which moves a row by up to 2^-7 of its terms'
  // magnitudes: that, times the output's derivative, is allowed on top, and
  // dominates where a row cancels. Almost every output is the reference's own
  // double rounding; norm weights rounded to bf16 would move a large fraction
  // of them.
  const uint32_t zOffset = shape.convolutionDimension;
  const float *committed = fixture.cell.recurrent(fixture.cellBytes(fixture.next[lane]), layer);
  uint64_t inexact = 0;
  std::vector<double> rows, magnitudes;
  for (uint32_t head = 0; head < shape.valueHeads; ++head) {
    static_cast<void>(recurrence(fixture, committed, fixture.tapeLanes[lane].stepSlot, layer, head, kRows, &step,
                                 &rows, &magnitudes));
    for (uint32_t token = 0; token < kRows; ++token) {
      const uint16_t *packed = step.packed.data() + token * shape.packedWidth;
      std::array<uint16_t, kHeadDim> recurrent;
      double squares = 0.0;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        recurrent[dim] =
            floatToBf16(static_cast<float>(rows[token * kHeadDim + dim]));
        squares += double(bf16ToFloat(recurrent[dim])) * bf16ToFloat(recurrent[dim]);
      }
      const double inverse = 1.0 / std::sqrt(squares / kHeadDim + SPLASH_RMS_EPSILON);
      const uint16_t *hidden = fixture.hiddenRow(lane, token, head);
      const std::vector<double> exact =
          rmsNorm(recurrent.data(), fixture.mixerNorm, kHeadDim);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double normalized = roundBfloat(exact[dim]);
        const double gate = bf16ToFloat(packed[zOffset + head * kHeadDim + dim]);
        const double reference = normalized * gate * sigmoid(gate);
        const double derivative = inverse * std::fabs(normWeight(fixture.mixerNorm, dim) * gate * sigmoid(gate));
        require(closeBfloat(hidden[dim], reference, 3.0,
                            1e-6 + derivative * std::ldexp(magnitudes[token * kHeadDim + dim], -7)),
                where + ": hidden mismatch");
        inexact += bf16ToFloat(hidden[dim]) != roundBfloat(reference);
      }
    }
  }
  require(inexact <= uint64_t{kRows} * shape.valueHeads * kHeadDim / 100,
          where + ": hidden differs from the reference in more than 1% of values");
}

// A first step of each layer of a lane, with no pending rows: the next cell
// holds the current cell's state.
void checkFirstStep(const Fixture &fixture, uint32_t layer, uint32_t lane) {
  const std::string where = "decode layer " + std::to_string(layer) + " lane " + std::to_string(lane);
  require(!std::memcmp(fixture.cell.recurrent(fixture.cellBytes(fixture.next[lane]), layer),
                       fixture.cell.recurrent(fixture.cellBytes(fixture.current[lane]), layer),
                       uint64_t{fixture.shape.valueHeads} * kHeadDim * kHeadDim * 4),
          where + ": a step without pending rows changed the state");
  checkStep(fixture, firstStep(fixture, layer, lane), layer, lane, where);
}

void requireUntouched(const Fixture &fixture, uint32_t lane,
                      const std::string &where) {
  const uint8_t *bytes = fixture.cellBytes(fixture.next[lane]);
  for (uint64_t index = 0; index < fixture.cell.bytes; ++index)
    require(bytes[index] == 0, where + ": idle lane cell was written");
  const uint8_t *tape = fixture.tapeLayer(fixture.slot(lane, 0), 0);
  for (uint64_t index = 0; index < 2 * kLayers * fixture.tapeBytes; ++index)
    require(tape[index] == 0, where + ": idle lane tape was written");
}

void encodeDecode(const Fixture &fixture, CommandGraph &graph, const std::string &where) {
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    require(GDN::addDecode(graph, fixture.decodeBuffers(layer), fixture.shape, fixture.lanes, layer,
                           fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain)
                    .layout == LinearInput::Plain,
            where + ": plain GDN claimed a table");
}

void runDecode(MetalBackend &backend, const GdnShape &shape, uint32_t lanes,
               bool float32) {
  Fixture fixture(backend, shape, lanes, float32);
  const std::string where =
      "lanes " + std::to_string(lanes) + (float32 ? " f32 norm" : "");
  CommandGraph first;
  encodeDecode(fixture, first, where);
  static_cast<void>(backend.submitCommandAsync(first.dispatches()).wait());
  std::vector<std::vector<StepInputs>> firstSteps(kLayers);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    if (lane >= lanes) {
      requireUntouched(fixture, lane, where);
      continue;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      checkFirstStep(fixture, layer, lane);
      firstSteps[layer].push_back(firstStep(fixture, layer, lane));
    }
  }
  // The norm weights only reach the hidden rows; the bf16 pass covers the
  // second step.
  if (float32)
    return;

  // The second step folds the rows of the first its lane retained into the
  // state before its own, from the first step's tape: lane l retained
  // 1 + (base + l - 1) % 8 rows, so every lane retains each count once.
  std::array<uint32_t, kMaxLanes> retained{};
  fixture.advance(retained);
  fixture.fillPacked();
  for (uint32_t base = 1; base <= kRows; ++base) {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      retained[lane] = 1 + (base + lane - 1) % kRows;
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      fixture.tapeLanes[lane].pendingRows = retained[lane];
    fixture.clear();
    CommandGraph second;
    encodeDecode(fixture, second, where);
    static_cast<void>(backend.submitCommandAsync(second.dispatches()).wait());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const std::string stepWhere =
          where + " second step after " + std::to_string(retained[lane]) + " retained, lane " + std::to_string(lane);
      // Every head at the shortest, a middle and the longest replay; a
      // spread of heads for the other counts keeps the double-precision
      // replay short under shader validation.
      std::vector<uint32_t> heads{0, shape.valueHeads / 2, shape.valueHeads - 1};
      if (retained[lane] == 1 || retained[lane] == 4 || retained[lane] == kRows) {
        heads.clear();
        for (uint32_t head = 0; head < shape.valueHeads; ++head)
          heads.push_back(head);
      }
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        const float *state = fixture.cell.recurrent(fixture.cellBytes(fixture.current[lane]), layer);
        for (uint32_t head : heads)
          checkState(fixture, layer, lane, head,
                     recurrence(fixture, state, fixture.tapeLanes[lane].pendingSlot, layer, head, retained[lane],
                                nullptr, nullptr),
                     stepWhere);
        checkStep(fixture, laterStep(fixture, firstSteps[layer][lane], retained[lane], layer, lane), layer, lane,
                  stepWhere);
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
         (layout == LinearInput::Plain ? " plain" : " Table16");
}

void fusedPreparation(MetalBackend &backend, const GdnShape &shape, uint32_t lanes, LinearInput layout,
                      bool float32) {
  const std::string what = caseName(float32 ? "fused GDN, F32 norms," : "fused GDN, bf16 norms,", shape, lanes, layout);
  Fixture fixture(backend, shape, lanes, float32);
  const PreparedTables tables(backend, layout, shape.valueHeads * shape.headDimension, lanes);
  CommandGraph reference;
  // Without scratch the table cannot be written: GDN refuses it before it
  // encodes anything.
  rejects(
      [&] {
        GDN::addDecode(reference, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides(),
                       GdnHeadOrder::Grouped, layout);
      },
      "linear table buffer holds", what + " wrote a table without scratch");
  require(GDN::addDecode(reference, fixture.decodeBuffers(0), shape, lanes, 0, fixture.cell.strides(),
                         GdnHeadOrder::Grouped, LinearInput::Plain)
                  .layout == LinearInput::Plain,
          what + " plain kernel claimed a table");
  tables.addReference(reference, fixture.hidden);
  (void)backend.submitCommandAsync(reference.dispatches()).wait();
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
  (void)backend.submitCommandAsync(fused.dispatches()).wait();
  require(!std::memcmp(expected.data(), fixture.hidden.contents(), expected.size()),
          what + " changed output");
  tables.requireWritten(prepared, fixture.hidden, what);
  for (uint32_t lane = 0; lane < lanes; ++lane) checkFirstStep(fixture, 0, lane);
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
  (void)backend.submitCommandAsync(grouped.dispatches()).wait();
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
    (void)backend.submitCommandAsync(tiled.dispatches()).wait();
  } else {
    const PreparedTables tables(backend, layout, width, lanes);
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch = tables.scratch();
    const PreparedInput prepared =
        GDN::addDecode(tiled, buffers, shape, lanes, 0, fixture.cell.strides(), GdnHeadOrder::Tiled, layout);
    tables.addReference(tiled, fixture.hidden);
    (void)backend.submitCommandAsync(tiled.dispatches()).wait();
    tables.requireWritten(prepared, fixture.hidden, what);
  }
  require(!std::memcmp(expected.data(), hidden, expected.size()),
          what + " output is not the grouped output in tiled head order");
}

// Each buffer the decode of `lanes` lanes reaches, at its extent and one
// element short. The decode in the last layer reads each lane's rows of the
// packed projection up to its last row's alpha, writes its hidden rows and
// the out-projection's table, reaches the end of that layer's tape in both of
// each lane's slots (the last lane's second slot ends the buffer), and each
// lane's state cells to the end of that layer's state.
void bufferExtents(MetalBackend &backend, const GdnShape &shape, LinearInput layout, bool float32,
                   uint32_t lanes) {
  constexpr uint32_t layer = kLayers - 1;
  Fixture fixture(backend, shape, lanes, float32);
  const uint32_t width = shape.valueHeads * shape.headDimension;
  const PreparedTables tables(backend, layout, width, lanes);
  GdnDecodeBuffers decode = fixture.decodeBuffers(layer);
  decode.linearScratch = tables.scratch();
  const GdnStateStrides production = fixture.cell.strides();
  const auto encodeDecode = [&](CommandGraph &graph, const GdnDecodeBuffers &buffers, GdnStateStrides strides) {
    (void)GDN::addDecode(graph, buffers, shape, lanes, layer, strides, GdnHeadOrder::Grouped, layout);
  };
  const uint64_t rows = uint64_t{lanes} * kRows;
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer GdnDecodeBuffers::*, uint64_t, uint64_t, const char *>>{
           {&GdnDecodeBuffers::packed,
            ((rows - 1) * shape.packedWidth + shape.convolutionDimension + width + 2 * shape.valueHeads) * 2, 2,
            "GDN packed"},
           {&GdnDecodeBuffers::convolutionWeights, uint64_t{shape.convolutionDimension} * kTaps * 2, 2,
            "GDN convolution weight"},
           {&GdnDecodeBuffers::decayWeights, uint64_t{shape.valueHeads} * 4, 4, "GDN decay weight"},
           {&GdnDecodeBuffers::timeBias, uint64_t{shape.valueHeads} * 2, 2, "GDN time bias"},
           {&GdnDecodeBuffers::tape, fixture.slot(lanes - 1, 1) + kLayers * fixture.tapeBytes, 2, "GDN tape"},
           {&GdnDecodeBuffers::hidden, rows * width * 2, 2, "GDN hidden"}})
    requireExtent(backend, decode.*member, bytes, element, what, [&](CommandGraph &graph, const MetalBuffer &buffer) {
      GdnDecodeBuffers changed = decode;
      changed.*member = buffer;
      encodeDecode(graph, changed, production);
    });
  const uint64_t normElement = float32 ? 4 : 2;
  requireExtent(backend, decode.mixerNorm.buffer, shape.headDimension * normElement, normElement, "norm weight",
                [&](CommandGraph &graph, const MetalBuffer &buffer) {
                  GdnDecodeBuffers changed = decode;
                  changed.mixerNorm.buffer = buffer;
                  encodeDecode(graph, changed, production);
                });
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer LinearScratch::*, uint64_t, uint64_t, const char *>>{
           {&LinearScratch::input, tables.tableSize, 2, "linear table"},
           {&LinearScratch::sums, tables.sumsSize, 4, "linear table sums"}})
    requireExtent(backend, decode.linearScratch.*member, bytes, element, what,
                  [&](CommandGraph &graph, const MetalBuffer &buffer) {
                    GdnDecodeBuffers changed = decode;
                    changed.linearScratch.*member = buffer;
                    encodeDecode(graph, changed, production);
                  });
  // Each running lane's current and next state cells, to the end of the last
  // layer's state: its recurrent state in the production layout (Cell), and
  // its convolution rows in a layout that puts them after every recurrent
  // state (layer 0's rows, the recurrent states, layer 1's rows). The
  // fixture's cells hold either layout.
  const uint64_t carried = uint64_t{fixture.cell.layout.convolutionHistory} * shape.convolutionDimension * 2,
                 recurrent = uint64_t{width} * shape.headDimension * 4;
  const GdnStateStrides convolutionLast{carried + kLayers * recurrent, recurrent, carried};
  const auto requireStates = [&](const auto &encode) {
    for (const auto &[strides, bytes, element] : std::initializer_list<std::tuple<GdnStateStrides, uint64_t, uint64_t>>{
             {production, production.convolutionStateBytes + layer * production.recurrentLayerBytes + recurrent, 4},
             {convolutionLast, layer * convolutionLast.convolutionLayerBytes + carried, 2}})
      for (const bool next : {false, true})
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const MetalBuffer &cell = (next ? fixture.next : fixture.current)[lane];
          requireExtent(backend, cell, bytes, element, next ? "GDN next state" : "GDN current state",
                        [&](CommandGraph &graph, const MetalBuffer &buffer) {
                          std::array<MetalBuffer, kMaxLanes> current = fixture.current, nextStates = fixture.next;
                          (next ? nextStates : current)[lane] = buffer;
                          encode(graph, strides, current, nextStates);
                        });
        }
  };
  requireStates([&](CommandGraph &graph, GdnStateStrides strides, std::span<const MetalBuffer> current,
                    std::span<const MetalBuffer> next) {
    GdnDecodeBuffers changed = decode;
    changed.currentStates = current;
    changed.nextStates = next;
    encodeDecode(graph, changed, strides);
  });

}

// The decode gate (gdn_decode_gate, decode/gdn.metal) reproduces the prefill
// gate (gdn_gate_phase, common/gdn_primitives.h) bit for bit from the same
// recurrent rows. Both recurrences produce those rows exactly here: a_scale 0
// makes every decay one, convolution weights only on the query channels'
// current input make the keys and values zero, and at token t every key
// head's query input is one on columns 17 t and 127, which the RMS norm and
// its scale turn into 1/16 on both. Every recurrent row is then a state
// column over 16, which is exact for a state of bf16 values that is zero in
// column 127. The prefill's rows, both decays and the decode's unchanged
// state confirm that before the hidden rows of every lane are compared.
void gateMatchesPrefill(MetalBackend &backend, const GdnShape &shape, bool float32, GdnHeadOrder order) {
  const std::string what = std::string("gate parity vh") + std::to_string(shape.valueHeads) +
                           (float32 ? " f32 norm" : "") + (order == GdnHeadOrder::Tiled ? " tiled" : "");
  constexpr uint32_t kSharedColumn = kHeadDim - 1;
  const auto queryColumn = [](uint32_t token) { return 17 * token; };
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  const uint32_t valueWidth = shape.valueHeads * kHeadDim;
  const uint64_t stateFloats = uint64_t{valueWidth} * kHeadDim, stateBytes = stateFloats * 4;
  constexpr uint16_t kOne = 0x3F80;
  Fixture fixture(backend, shape, kMaxLanes, float32);
  Random random(0x6a7e + shape.valueHeads + float32);

  auto *convWeights = static_cast<uint16_t *>(fixture.convWeights.contents());
  std::fill_n(convWeights, uint64_t{shape.convolutionDimension} * kTaps, uint16_t{0});
  for (uint32_t channel = 0; channel < keyWidth; ++channel)
    convWeights[channel * kTaps + kCarried] = kOne;
  std::fill_n(static_cast<float *>(fixture.decayWeights.contents()), shape.valueHeads, 0.0F);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    for (uint32_t token = 0; token < kRows; ++token) {
      uint16_t *row = static_cast<uint16_t *>(fixture.packed.contents()) + lane * fixture.packedStride +
                      uint64_t{token} * shape.packedWidth;
      std::fill_n(row, keyWidth, uint16_t{0});
      for (uint32_t head = 0; head < shape.keyHeads; ++head)
        row[head * kHeadDim + queryColumn(token)] = row[head * kHeadDim + kSharedColumn] = kOne;
    }
    auto *state =
        reinterpret_cast<float *>(static_cast<uint8_t *>(fixture.current[lane].contents()) + fixture.cell.convBytes);
    for (uint64_t index = 0; index < stateFloats; ++index)
      state[index] = index % kHeadDim == kSharedColumn ? 0.0F : roundBfloat(random.unit());
  }

  CommandGraph decode;
  require(GDN::addDecode(decode, fixture.decodeBuffers(0), shape, kMaxLanes, 0, fixture.cell.strides(), order,
                         LinearInput::Plain)
                  .layout == LinearInput::Plain,
          what + ": plain GDN claimed a table");
  (void)backend.submitCommandAsync(decode.dispatches()).wait();

  const auto buffer = [&](uint64_t bytes) { return sharedBuffer(backend, bytes); };
  const uint64_t valueRows = uint64_t{kRows} * valueWidth * 2;
  GdnPrefillBuffers prefill{{},
                            fixture.convWeights,
                            {},
                            buffer(uint64_t{kCarried} * shape.convolutionDimension * 2),
                            buffer(uint64_t{kRows} * keyWidth * 2),
                            buffer(uint64_t{kRows} * keyWidth * 2),
                            buffer(valueRows),
                            fixture.decayWeights,
                            fixture.timeBias,
                            buffer(uint64_t{kRows} * shape.valueHeads * 4),
                            buffer(uint64_t{kRows} * shape.valueHeads * 2),
                            {},
                            buffer(stateBytes),
                            buffer(valueRows),
                            fixture.mixerNorm,
                            buffer(valueRows)};
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    const std::string where = what + " lane " + std::to_string(lane);
    prefill.packed = backend.view(fixture.packed, uint64_t{lane} * fixture.packedStride * 2,
                                  fixture.packedStride * 2);
    prefill.convolutionIn =
        backend.view(fixture.current[lane], 0, uint64_t{kCarried} * shape.convolutionDimension * 2);
    prefill.recurrentIn = backend.view(fixture.current[lane], fixture.cell.convBytes, stateBytes);
    CommandGraph graph;
    GDN::addPrefill(graph, prefill, shape, kRows, order);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();

    const float *state = fixture.cell.recurrent(fixture.cellBytes(fixture.current[lane]), 0);
    const auto *rows = static_cast<const uint16_t *>(prefill.recurrentRows.contents());
    for (uint32_t token = 0; token < kRows; ++token)
      for (uint32_t head = 0; head < shape.valueHeads; ++head)
        for (uint32_t dim = 0; dim < kHeadDim; ++dim)
          require(rows[(uint64_t{token} * shape.valueHeads + head) * kHeadDim + dim] ==
                      floatToBf16(state[(uint64_t{head} * kHeadDim + dim) * kHeadDim + queryColumn(token)] / 16),
                  where + ": prefill recurrent rows are not the state's columns over 16");
    const auto *prefillDecay = static_cast<const float *>(prefill.decay.contents());
    for (uint32_t token = 0; token < kRows; ++token)
      for (uint32_t head = 0; head < shape.valueHeads; ++head)
        require(fixture.tapeDecay(fixture.tapeLanes[lane].stepSlot, 0, token)[head] == 1.0F &&
                    prefillDecay[token * shape.valueHeads + head] == 1.0F,
                where + ": a decay is not one");
    require(!std::memcmp(fixture.cell.recurrent(fixture.cellBytes(fixture.next[lane]), 0), state, stateBytes),
            where + ": the decode changed the state");
    require(!std::memcmp(fixture.hiddenRow(lane, 0, 0), prefill.hidden.contents(), valueRows),
            where + ": decode and prefill gates wrote different hidden rows");
  }
}

void rejectsInvalid(MetalBackend &backend) {
  const GdnShape &shape = kShapes[1];
  Fixture fixture(backend, shape, 1);
  CommandGraph graph;
  for (const uint32_t lanes : {0U, kMaxLanes + 1})
    rejects(
        [&] {
          GDN::addDecode(graph, fixture.decodeBuffers(0), shape, lanes, 0,
                         fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain);
        },
        "invalid GDN decode geometry", "a GDN decode of no lanes or more than a batch was accepted");
  rejects(
      [&] {
        GDN::addDecode(graph, fixture.decodeBuffers(0),
                       GdnShape{16, 40, 128, 9216, 14400}, 1, 0,
                       fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain);
      },
      "invalid GDN shape", "an inconsistent GDN shape was accepted");
  rejects(
      [&] {
        auto buffers = fixture.decodeBuffers(0);
        buffers.tapeLanes = {};
        GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                       LinearInput::Plain);
      },
      "invalid GDN decode geometry", "a GDN decode without a lane's tape was accepted");
  // More rows than a step's pending; a step's tape misaligned, or overlapping
  // the pending tape of its own lane or of another, or another lane's step.
  const uint64_t tape = fixture.tapeBytes;
  for (const std::array<GdnTapeLane, 2> lanes :
       {std::array{GdnTapeLane{0, tape, kRows + 1}, GdnTapeLane{2 * tape, 3 * tape, 0}},
        std::array{GdnTapeLane{0, tape + 8, 0}, GdnTapeLane{2 * tape, 3 * tape, 0}},
        std::array{GdnTapeLane{0, tape - 16, 0}, GdnTapeLane{2 * tape, 3 * tape, 0}},
        std::array{GdnTapeLane{0, tape, 0}, GdnTapeLane{2 * tape, 16, 0}},
        std::array{GdnTapeLane{0, tape, 0}, GdnTapeLane{3 * tape, tape + 16, 0}}})
    rejects(
        [&] {
          auto buffers = fixture.decodeBuffers(0);
          buffers.tapeLanes = lanes;
          GDN::addDecode(graph, buffers, shape, 2, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                         LinearInput::Plain);
        },
        "invalid GDN decode tape", "a GDN decode took more pending rows than a step's or overlapping tapes");
  rejects(
      [&] {
        auto buffers = fixture.decodeBuffers(0);
        buffers.linearScratch.input = sharedBuffer(backend, 16);
        buffers.linearScratch.sums = sharedBuffer(backend, 4);
        GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                       LinearInput::Table16);
      },
      "linear table buffer holds", "a GDN decode wrote a table into short scratch");
  rejects(
      [&] {
        // F32 weights need twice the bytes of bf16 ones.
        auto buffers = fixture.decodeBuffers(0);
        buffers.mixerNorm.float32 = true;
        GDN::addDecode(graph, buffers, shape, 1, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                       LinearInput::Plain);
      },
      "norm weight buffer holds", "bf16 norm weights were read as F32");
  require(graph.empty(), "invalid GDN request partially encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: gdn-decode METALLIB");
    MetalBackend backend(argv[1]);
    rejectsInvalid(backend);
    for (const GdnShape &shape : kShapes)
      for (const uint32_t lanes : {1U, kMaxLanes})
        for (const bool float32 : {false, true}) bufferExtents(backend, shape, LinearInput::Table16, float32, lanes);
    // Table16 feeds the register tile after norms that are F32 (a GGUF's, in
    // tiled head order) or bf16 (an MLX target's, grouped).
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes) {
        fusedPreparation(backend, shape, lanes, LinearInput::Table16, true);
        tiledHeadOrder(backend, shape, lanes, LinearInput::Table16, true);
        fusedPreparation(backend, shape, lanes, LinearInput::Table16, false);
      }
    // A GGUF's out-projection on the staged tile reads the tiled rows plain.
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        tiledHeadOrder(backend, shape, lanes, LinearInput::Plain, true);
    for (bool float32 : {false, true})
      for (const GdnShape &shape : kShapes)
        for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
          runDecode(backend, shape, lanes, float32);
    for (bool float32 : {false, true})
      for (const GdnShape &shape : kShapes)
        for (GdnHeadOrder order : {GdnHeadOrder::Grouped, GdnHeadOrder::Tiled})
          gateMatchesPrefill(backend, shape, float32, order);
    std::cout << "gdn_decode_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gdn_decode_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
