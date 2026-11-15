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
// from the same recurrent rows. Changing multi-step batches are compared
// bit for bit with independent lanes, with rejected tape rows poisoned. The
// decode over a bf16 state is checked against the fp32 decode. A lane's step
// rows past its verify rows change only those rows' outputs, which are zero.
// A fold of a step's first rows alone stores the state the lane's next step
// holds when it retained those or more, bit for bit.
// Lanes whose blocks of a step's rows hold fewer rows than a lane's eight, at
// any first row, leave what each block run alone leaves.
#include "ChainRows.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestFamilies.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "model/StateLayout.hpp"
#include "ops/GDN.hpp"
#include "ops/LaneRows.hpp"

#include "LinearInputReference.hpp"
#include "LinearNumerics.hpp"
#include "NormReference.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
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
using splash::test::chainPath;
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
// Every family's shape (Qwen3.8-Flash-Next's gate is sigmoid(z), the others'
// silu(z)).
const std::vector<GdnShape> kShapes = splash::test::familyGdnShapes();

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
// The output gate of a GDN of `shape`: sigmoid(z), or silu(z) = z sigmoid(z).
double outputGate(const GdnShape &shape, double z) { return shape.sigmoidGate ? sigmoid(z) : z * sigmoid(z); }

// The production state cell layout: every layer's conv rows, then every
// layer's recurrent state, both padded to 16 KiB.
struct Cell final {
  splash::model::GdnStateLayout layout;
  uint64_t convLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convBytes = 0;
  uint64_t bytes = 0;

  explicit Cell(const GdnShape &shape, GdnStateFormat format = GdnStateFormat::Float32)
      : layout{kLayers, splash::model::kGdnConvolutionTaps - 1, shape.convolutionDimension,
               shape.valueHeads, shape.headDimension, shape.headDimension, format},
        convLayerBytes(layout.convolutionLayerBytes()),
        recurrentLayerBytes(layout.recurrentLayerBytes()),
        convBytes(layout.convolutionBytes()), bytes(layout.cellBytes()) {}

  GdnStateStrides strides() const { return layout.strides(); }
  const uint16_t *conv(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const uint16_t *>(cell + layer * convLayerBytes);
  }
  const float *recurrent(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const float *>(cell + convBytes +
                                           layer * recurrentLayerBytes);
  }
};

// A lane's block of a decode step's rows, which the step's proposer writes on
// the device (SplashLaneRows), and its tape.
struct GdnDecodeBlock final {
  uint32_t firstRow = 0;
  uint32_t rows = 0;
  GdnTapeLane tape;
};

// Blocks as a step takes them (GdnDecodeBuffers): their tapes, and their rows
// in a buffer of lane rows on the device, the step's rows ending at the last
// block's and each lane's cap the kernels' eight or sixteen.
struct StepBlocks final {
  std::array<GdnTapeLane, kMaxLanes> tapes{};
  MetalBuffer laneRows;
  uint32_t lanes = 0, rows = 0, cap = 0;

  StepBlocks(MetalBackend &backend, std::span<const GdnDecodeBlock> blocks)
      : laneRows(sharedBuffer(backend, kMaxLanes * sizeof(SplashLaneRows))) {
    write(blocks);
  }
  void write(std::span<const GdnDecodeBlock> blocks) {
    auto *device = static_cast<SplashLaneRows *>(laneRows.contents());
    lanes = static_cast<uint32_t>(blocks.size());
    rows = cap = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      tapes[lane] = blocks[lane].tape;
      device[lane] = {blocks[lane].firstRow, blocks[lane].rows};
      rows = std::max(rows, blocks[lane].firstRow + blocks[lane].rows);
      cap = std::max(cap, blocks[lane].rows);
    }
    cap = cap <= SPLASH_GDN_NARROW_ROWS ? SPLASH_GDN_NARROW_ROWS : kRows;
  }
  GdnDecodeBuffers into(GdnDecodeBuffers buffers) const {
    buffers.tapes = std::span(tapes).first(lanes);
    buffers.laneRows = laneRows;
    buffers.rows = rows;
    buffers.laneCap = cap;
    return buffers;
  }
};

// The mixer norm is bf16, or F32 as a GGUF stores it. Lane l's block is rows
// [8 l, 8 l + 8) of the step, and its tape slots are slot(l, 0) and slot(l,
// 1): a step reads the rows the lane retained from one and leaves its own in
// the other, and advance() swaps them, with the cells, for the lane's next
// step.
struct Fixture final {
  const GdnShape &shape;
  Cell cell;
  uint32_t lanes;
  uint64_t packedStride, tapeBytes, kvWidth;
  Random random;
  MetalBuffer packed, convWeights, decayWeights, timeBias, hidden, tape;
  // Each lane's verify rows: every row.
  MetalBuffer verifyRows;
  // The step's rows' places in their blocks' trees: each block a chain.
  MetalBuffer tree;
  // Its blocks as its steps take them (decodeBuffers).
  mutable std::optional<StepBlocks> steps;
  NormWeights mixerNorm;
  std::array<MetalBuffer, kMaxLanes> current, next;
  std::array<GdnDecodeBlock, kMaxLanes> blocks{};
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
    verifyRows = alloc(kMaxLanes * sizeof(uint32_t), "gdn verify rows");
    std::fill_n(static_cast<uint32_t *>(verifyRows.contents()), kMaxLanes, kRows);
    tree = alloc(uint64_t{kMaxLanes} * kRows * sizeof(TreeRow), "gdn tree rows");
    steps.emplace(backend, std::span(blocks).first(0));
    mixerNorm = makeNormWeights(backend, kHeadDim, float32,
                                [&](uint32_t) { return random.unit(); });
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      blocks[lane] = {lane * kRows, kRows, {slot(lane, 0), slot(lane, 1), 0}};
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
    chains();
    clear();
  }

  // Each block of `blocks` (the fixture's own unless given) a chain.
  void chains() { chains(blocks); }
  void chains(std::span<const GdnDecodeBlock> of) {
    auto *rows = static_cast<TreeRow *>(tree.contents());
    for (const GdnDecodeBlock &block : of)
      for (uint32_t row = 0; row < block.rows; ++row) rows[block.firstRow + row] = chainRow(row);
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
  // step folds in the first `pending[lane]` rows of this one, a chain's.
  void advance(std::span<const uint32_t> pending) {
    std::swap(current, next);
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      blocks[lane].tape = {blocks[lane].tape.stepSlot, blocks[lane].tape.pendingSlot, chainPath(pending[lane])};
  }

  // A step of its lanes' blocks, whose rows this writes to its lane rows on
  // the device.
  GdnDecodeBuffers decodeBuffers(uint32_t layer) const {
    steps->write(std::span(blocks).first(lanes));
    return steps->into({packedLayer[layer], convWeights, current, next, tape, {}, {}, 0, 0, decayWeights, timeBias,
                        mixerNorm, hidden, verifyRows, tree});
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
  const uint64_t slot = fixture.blocks[lane].tape.stepSlot;
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
  const uint64_t slot = fixture.blocks[lane].tape.stepSlot;
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
    static_cast<void>(recurrence(fixture, committed, fixture.blocks[lane].tape.stepSlot, layer, head, kRows, &step,
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
        const double reference = normalized * outputGate(shape, gate);
        const double derivative = inverse * std::fabs(normWeight(fixture.mixerNorm, dim) * outputGate(shape, gate));
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
    require(GDN::addDecode(graph, fixture.decodeBuffers(layer), fixture.shape, layer, fixture.cell.strides(),
                           GdnHeadOrder::Grouped, LinearInput::Plain)
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
  // The norm weights only reach the hidden rows; the pass of the first norm
  // type the shape's families store covers the second step.
  if (float32 != familyNorms(shape).front())
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
      fixture.blocks[lane].tape.pendingPath = chainPath(retained[lane]);
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
                     recurrence(fixture, state, fixture.blocks[lane].tape.pendingSlot, layer, head, retained[lane],
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
  // The preparation takes the lanes' rows in blocks of a table tile's rows.
  void addReference(CommandGraph &graph, const MetalBuffer &hidden) const {
    addReferencePreparation(graph, layout, hidden, referenceTable, referenceSums, width,
                            lanes * kRows / GGUF_TABLE16_ROWS);
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

// A lane's result depends on its retained prefix, not its position in a
// batch or whether other lanes ran while it was idle. Compare ten changing
// batches against one-lane submissions, including repeated partial/full
// acceptance and both tape parities. Each storage lane keeps its own cells
// and slots while the batch shrinks, reorders and later rejoins it.
void replayAcrossBatches(MetalBackend &backend, const GdnShape &shape, bool float32,
                         GdnHeadOrder order, LinearInput layout) {
  Fixture batched(backend, shape, kMaxLanes, float32);
  Fixture individual(backend, shape, kMaxLanes, float32);
  const uint32_t width = shape.valueHeads * shape.headDimension;
  const uint64_t hiddenBytes = uint64_t{kRows} * width * 2;
  const std::string what = caseName("replay across batches", shape, kMaxLanes, layout) +
                           (float32 ? " f32 norm" : " bf16 norm") +
                           (order == GdnHeadOrder::Tiled ? " tiled" : " grouped");
  constexpr std::array widths{4U, 2U, 3U, 1U, 4U};
  // No rows are pending initially. Later, poison only the rejected suffix
  // of each step's tape; reading beyond its retained prefix must fail.
  std::memset(batched.tape.contents(), 0xFF, batched.tape.sizeBytes());
  for (uint32_t cycle = 0; cycle < 10; ++cycle) {
    const uint32_t lanes = widths[cycle % widths.size()];
    const std::string where = what + " cycle " + std::to_string(cycle);
    batched.fillPacked();
    individual.fillPacked();
    std::array<uint32_t, kMaxLanes> storage{};
    std::array<MetalBuffer, kMaxLanes> current, next;
    std::array<GdnDecodeBlock, kMaxLanes> blocks{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t stateLane = storage[lane] = (lane + cycle) % kMaxLanes;
      current[lane] = batched.current[stateLane];
      next[lane] = batched.next[stateLane];
      blocks[lane] = {lane * kRows, kRows, batched.blocks[stateLane].tape};
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        std::memcpy(static_cast<uint16_t *>(batched.packedLayer[layer].contents()) + lane * batched.packedStride,
                    individual.packedRow(layer, stateLane, 0), batched.packedStride * 2);
    }
    for (uint32_t lane = lanes; lane < kMaxLanes; ++lane) {
      current[lane] = current[0];
      next[lane] = next[0];
    }
    std::optional<PreparedTables> tables;
    if (layout != LinearInput::Plain)
      tables.emplace(backend, layout, width, lanes);
    CommandGraph batchGraph, individualGraph;
    PreparedInput prepared;
    const StepBlocks batchBlocks(backend, std::span(blocks).first(lanes));
    std::vector<StepBlocks> singleBlocks;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      singleBlocks.emplace_back(backend,
                                std::array{GdnDecodeBlock{0, kRows, individual.blocks[storage[lane]].tape}});
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      auto buffers = batchBlocks.into(batched.decodeBuffers(layer));
      buffers.currentStates = current;
      buffers.nextStates = next;
      if (tables)
        buffers.linearScratch = tables->scratch();
      prepared = GDN::addDecode(batchGraph, buffers, shape, layer, batched.cell.strides(), order, layout);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t stateLane = storage[lane];
        auto single = singleBlocks[lane].into(individual.decodeBuffers(layer));
        std::array<MetalBuffer, kMaxLanes> singleCurrent, singleNext;
        singleCurrent.fill(individual.current[stateLane]);
        singleNext.fill(individual.next[stateLane]);
        single.packed = backend.view(individual.packedLayer[layer], stateLane * individual.packedStride * 2,
                                     individual.packedStride * 2);
        single.hidden = backend.view(individual.hidden, stateLane * hiddenBytes, hiddenBytes);
        single.currentStates = singleCurrent;
        single.nextStates = singleNext;
        (void)GDN::addDecode(individualGraph, single, shape, layer, individual.cell.strides(), order,
                             LinearInput::Plain);
      }
    }
    if (tables)
      tables->addReference(batchGraph, batched.hidden);
    (void)backend.submitCommandAsync(batchGraph.dispatches()).wait();
    (void)backend.submitCommandAsync(individualGraph.dispatches()).wait();
    if (tables)
      tables->requireWritten(prepared, batched.hidden, where);
    else
      require(prepared.layout == LinearInput::Plain, where + ": plain GDN claimed a table");
    for (uint32_t stateLane = 0; stateLane < kMaxLanes; ++stateLane) {
      // Includes idle lanes: neither parity may change when a lane is omitted.
      require(!std::memcmp(batched.current[stateLane].contents(), individual.current[stateLane].contents(),
                           batched.cell.bytes), where + ": incoming cell mismatch");
      require(!std::memcmp(batched.next[stateLane].contents(), individual.next[stateLane].contents(),
                           batched.cell.bytes), where + ": folded cell mismatch");
    }
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t stateLane = storage[lane];
      require(!std::memcmp(batched.hiddenRow(lane, 0, 0), individual.hiddenRow(stateLane, 0, 0), hiddenBytes),
              where + ": hidden rows changed with batching");
      const uint32_t retained = 1 + (cycle + stateLane) % kRows;
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        const uint64_t slot = batched.blocks[stateLane].tape.stepSlot;
        require(!std::memcmp(batched.tapeLayer(slot, layer), individual.tapeLayer(slot, layer), batched.tapeBytes),
                where + ": step tape changed with batching");
        auto *tape = static_cast<uint8_t *>(batched.tape.contents()) + slot + layer * batched.tapeBytes;
        for (const auto &[offset, rowBytes] :
             {std::pair{uint64_t{0}, uint64_t{shape.convolutionDimension} * 2},
              std::pair{gdn_tape_kv_offset(shape.convolutionDimension), batched.kvWidth * 2},
              std::pair{gdn_tape_decay_offset(shape.convolutionDimension, batched.kvWidth),
                        uint64_t{shape.valueHeads} * 4},
              std::pair{gdn_tape_beta_offset(shape.convolutionDimension, batched.kvWidth, shape.valueHeads),
                        uint64_t{shape.valueHeads} * 2}})
          std::memset(tape + offset + retained * rowBytes, 0xFF, (kRows - retained) * rowBytes);
      }
      for (Fixture *fixture : {&batched, &individual}) {
        std::swap(fixture->current[stateLane], fixture->next[stateLane]);
        auto &tape = fixture->blocks[stateLane].tape;
        std::swap(tape.pendingSlot, tape.stepSlot);
        tape.pendingPath = chainPath(retained);
      }
    }
  }
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
        GDN::addDecode(reference, fixture.decodeBuffers(0), shape, 0, fixture.cell.strides(),
                       GdnHeadOrder::Grouped, layout);
      },
      "linear table buffer holds", what + " wrote a table without scratch");
  require(GDN::addDecode(reference, fixture.decodeBuffers(0), shape, 0, fixture.cell.strides(),
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
  require(GDN::addDecode(unprepared, buffers, shape, 0, fixture.cell.strides(),
                         GdnHeadOrder::Grouped, LinearInput::Plain).layout == LinearInput::Plain,
          what + " claimed a table for a plain consumer");
  CommandGraph fused;
  const PreparedInput prepared =
      GDN::addDecode(fused, buffers, shape, 0, fixture.cell.strides(), GdnHeadOrder::Grouped, layout);
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
  require(GDN::addDecode(grouped, fixture.decodeBuffers(0), shape, 0, fixture.cell.strides(),
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
    require(GDN::addDecode(tiled, fixture.decodeBuffers(0), shape, 0, fixture.cell.strides(),
                           GdnHeadOrder::Tiled, LinearInput::Plain).layout == LinearInput::Plain,
            what + " claimed a table");
    (void)backend.submitCommandAsync(tiled.dispatches()).wait();
  } else {
    const PreparedTables tables(backend, layout, width, lanes);
    auto buffers = fixture.decodeBuffers(0);
    buffers.linearScratch = tables.scratch();
    const PreparedInput prepared =
        GDN::addDecode(tiled, buffers, shape, 0, fixture.cell.strides(), GdnHeadOrder::Tiled, layout);
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
    (void)GDN::addDecode(graph, buffers, shape, layer, strides, GdnHeadOrder::Grouped, layout);
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
           {&GdnDecodeBuffers::hidden, rows * width * 2, 2, "GDN hidden"},
           {&GdnDecodeBuffers::verifyRows, uint64_t{lanes} * 4, 4, "GDN verify rows"},
           {&GdnDecodeBuffers::tree, rows * sizeof(TreeRow), sizeof(TreeRow), "GDN tree rows"},
           {&GdnDecodeBuffers::laneRows, uint64_t{lanes} * sizeof(SplashLaneRows), 4, "GDN lane rows"}})
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
  const GdnStateStrides convolutionLast{carried + kLayers * recurrent, recurrent, carried,
                                        GdnStateFormat::Float32};
  // A bf16 state's cells reach half as far into each layer's state.
  const GdnStateStrides narrow = Cell(shape, GdnStateFormat::BFloat16).strides();
  const auto requireStates = [&](const auto &encode) {
    for (const auto &[strides, bytes, element] : std::initializer_list<std::tuple<GdnStateStrides, uint64_t, uint64_t>>{
             {production, production.convolutionStateBytes + layer * production.recurrentLayerBytes + recurrent, 4},
             {narrow, narrow.convolutionStateBytes + layer * narrow.recurrentLayerBytes + recurrent / 2, 2},
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
// head's query input is one on columns 7 t and 127, which the RMS norm and
// its scale turn into 1/16 on both. Every recurrent row is then a state
// column over 16, which is exact for a state of bf16 values that is zero in
// column 127. The prefill's rows, both decays and the decode's unchanged
// state confirm that before the hidden rows of every lane are compared.
void gateMatchesPrefill(MetalBackend &backend, const GdnShape &shape, bool float32, GdnHeadOrder order) {
  const std::string what = std::string("gate parity vh") + std::to_string(shape.valueHeads) +
                           (float32 ? " f32 norm" : "") + (order == GdnHeadOrder::Tiled ? " tiled" : "");
  constexpr uint32_t kSharedColumn = kHeadDim - 1;
  static_assert(7 * (kRows - 1) < kSharedColumn, "each token's own query column");
  const auto queryColumn = [](uint32_t token) { return 7 * token; };
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
  require(GDN::addDecode(decode, fixture.decodeBuffers(0), shape, 0, fixture.cell.strides(), order,
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
    GDN::addPrefill(graph, backend, prefill, shape, kRows, GdnStateFormat::Float32, order);
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
        require(fixture.tapeDecay(fixture.blocks[lane].tape.stepSlot, 0, token)[head] == 1.0F &&
                    prefillDecay[token * shape.valueHeads + head] == 1.0F,
                where + ": a decay is not one");
    require(!std::memcmp(fixture.cell.recurrent(fixture.cellBytes(fixture.next[lane]), 0), state, stateBytes),
            where + ": the decode changed the state");
    require(!std::memcmp(fixture.hiddenRow(lane, 0, 0), prefill.hidden.contents(), valueRows),
            where + ": decode and prefill gates wrote different hidden rows");
  }
}

// Two steps of every layer over cells of `format`, the second folding
// retained[l] rows of the first, with lane l verifying verify[l] rows of each
// step: what they leave in the hidden rows, the tape and the next cells. Where
// `layout` is a table, each step's out-projection table and sums must be the
// reference preparation of its hidden rows.
struct StepsRun final {
  std::vector<uint8_t> hidden, tape;
  std::vector<std::vector<uint8_t>> cells;
};
StepsRun runSteps(MetalBackend &backend, const GdnShape &shape, uint32_t lanes, LinearInput layout, bool float32,
                  GdnHeadOrder order, GdnStateFormat format, const std::array<uint32_t, kMaxLanes> &verify,
                  const std::array<uint32_t, kMaxLanes> &retained, const std::string &what) {
  const auto bytesOf = [](const MetalBuffer &buffer) {
    const auto *bytes = static_cast<const uint8_t *>(buffer.contents());
    return std::vector<uint8_t>(bytes, bytes + buffer.sizeBytes());
  };
  Fixture fixture(backend, shape, lanes, float32);
  const Cell cell(shape, format);
  const bool narrow = format == GdnStateFormat::BFloat16;
  std::array<MetalBuffer, kMaxLanes> current = fixture.current, next = fixture.next;
  if (narrow) {
    // The fixture's fp32 cells rounded to bf16, as bfloatState takes them.
    const uint64_t elements = uint64_t{shape.valueHeads} * shape.headDimension * shape.headDimension;
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      current[lane] = sharedBuffer(backend, cell.bytes);
      next[lane] = sharedBuffer(backend, cell.bytes);
      std::memset(next[lane].contents(), 0, cell.bytes);
      std::memcpy(current[lane].contents(), fixture.current[lane].contents(), cell.convBytes);
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        const float *values = fixture.cell.recurrent(fixture.cellBytes(fixture.current[lane]), layer);
        auto *rounded = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(current[lane].contents()) +
                                                     cell.convBytes + layer * cell.recurrentLayerBytes);
        for (uint64_t index = 0; index < elements; ++index) rounded[index] = floatToBf16(values[index]);
      }
    }
  }
  std::optional<PreparedTables> tables;
  if (layout != LinearInput::Plain)
    tables.emplace(backend, layout, shape.valueHeads * shape.headDimension, lanes);
  std::memcpy(fixture.verifyRows.contents(), verify.data(), sizeof(verify));
  for (uint32_t step = 0; step < 2; ++step) {
    if (step) {
      fixture.advance(retained);
      if (narrow) {
        std::swap(current, next);
        for (const MetalBuffer &buffer : next) std::memset(buffer.contents(), 0, cell.bytes);
      } else {
        current = fixture.current;
        next = fixture.next;
      }
      fixture.fillPacked();
      fixture.clear();
    }
    CommandGraph graph;
    PreparedInput prepared;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      GdnDecodeBuffers buffers = fixture.decodeBuffers(layer);
      buffers.currentStates = current;
      buffers.nextStates = next;
      if (tables) buffers.linearScratch = tables->scratch();
      prepared = GDN::addDecode(graph, buffers, shape, layer, cell.strides(), order, layout);
    }
    if (tables) tables->addReference(graph, fixture.hidden);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    if (tables)
      tables->requireWritten(prepared, fixture.hidden, what + " step " + std::to_string(step));
    else
      require(prepared.layout == LinearInput::Plain, what + ": plain GDN claimed a table");
  }
  StepsRun run{bytesOf(fixture.hidden), bytesOf(fixture.tape), {}};
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) run.cells.push_back(bytesOf(next[lane]));
  return run;
}

// A lane verifying fewer rows leaves the state, the tape and its verified
// rows' outputs as every row does, and zero in its other rows; its next step
// retains at most the rows it verified.
void partialVerifyRows(MetalBackend &backend, const GdnShape &shape, uint32_t lanes, LinearInput layout,
                       bool float32, GdnHeadOrder order, GdnStateFormat format) {
  const std::string what = caseName("partial verify rows", shape, lanes, layout) +
                           (order == GdnHeadOrder::Tiled ? " tiled" : " grouped") + " " +
                           std::string(formatName(format));
  std::array<uint32_t, kMaxLanes> every{}, verify{}, retained{};
  every.fill(kRows);
  // Lane l verifies 1 + (5 l + 3) % 8 rows (4, 1, 6, 3) and retains
  // 1 + (3 l + 2) % those (3, 1, 3, 3).
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    verify[lane] = 1 + (5 * lane + 3) % kRows;
    retained[lane] = 1 + (3 * lane + 2) % verify[lane];
  }
  const StepsRun reference =
      runSteps(backend, shape, lanes, layout, float32, order, format, every, retained, what + " every row");
  const StepsRun run = runSteps(backend, shape, lanes, layout, float32, order, format, verify, retained, what);
  require(run.tape == reference.tape, what + ": tape differs from every row's");
  require(run.cells == reference.cells, what + ": state cells differ from every row's");
  const uint64_t rowValues = uint64_t{shape.valueHeads} * shape.headDimension;
  const auto *got = reinterpret_cast<const uint16_t *>(run.hidden.data());
  const auto *expected = reinterpret_cast<const uint16_t *>(reference.hidden.data());
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t token = 0; token < kRows; ++token) {
      const uint64_t row = (uint64_t{lane} * kRows + token) * rowValues;
      if (token < verify[lane])
        require(!std::memcmp(got + row, expected + row, rowValues * 2),
                what + ": a verified row differs from every row's");
      else
        require(std::all_of(got + row, got + row + rowValues, [](uint16_t v) { return (v & 0x7FFF) == 0; }),
                what + ": an unverified row is not zero");
    }
}

// Lanes 0, 1 and 2 take blocks of 8, 5 and 3 rows at rows 0, 8 and 13 of a
// step's 16 rows, the last two in one table tile; they verify 7, 5 and 2 of
// them, and their next steps fold 5, 3 and 2. Over two steps of every layer
// from the same inputs, the packed step leaves each lane's state cells, tape
// and hidden rows as its block run alone does (a one-lane step of its rows
// from row 0, in buffers of its own), byte for byte, and the out-projection
// table the reference preparation of those hidden rows; a lane's tape rows
// past its block's rows, poisoned, and the hidden rows past the last block's
// stay as they were.
void raggedBlocks(MetalBackend &backend, const GdnShape &shape, bool float32, GdnHeadOrder order, LinearInput layout,
                  GdnStateFormat format) {
  constexpr uint32_t kBlocks = 3, kStepRows = 16;
  constexpr std::array<uint32_t, kBlocks> kFirst{0, 8, 13}, kBlockRows{8, 5, 3}, kVerify{7, 5, 2};
  constexpr std::array<uint32_t, kMaxLanes> kRetained{5, 3, 2, 0};
  const std::string what = caseName("ragged blocks", shape, kBlocks, layout) + (float32 ? " f32 norm" : " bf16 norm") +
                           (order == GdnHeadOrder::Tiled ? " tiled" : " grouped") + " " +
                           std::string(formatName(format));
  const uint32_t width = shape.valueHeads * shape.headDimension;
  const uint64_t hiddenRowBytes = uint64_t{width} * 2, packedRowBytes = uint64_t{shape.packedWidth} * 2;
  const Cell cell(shape, format);
  Fixture packed(backend, shape, kBlocks, float32), alone(backend, shape, kBlocks, float32);
  for (Fixture *fixture : {&packed, &alone}) {
    if (format == GdnStateFormat::BFloat16) {
      // The fixture's fp32 states rounded to bf16, in the cells' bf16 layout.
      const uint64_t elements = uint64_t{shape.valueHeads} * shape.headDimension * shape.headDimension;
      for (const MetalBuffer &buffer : fixture->current) {
        auto *bytes = static_cast<uint8_t *>(buffer.contents());
        const std::vector<uint8_t> wide(bytes, bytes + fixture->cell.bytes);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
          const float *values = fixture->cell.recurrent(wide.data(), layer);
          auto *rounded = reinterpret_cast<uint16_t *>(bytes + cell.convBytes + layer * cell.recurrentLayerBytes);
          for (uint64_t index = 0; index < elements; ++index) rounded[index] = floatToBf16(values[index]);
        }
      }
    }
    std::memset(fixture->tape.contents(), 0xFF, fixture->tape.sizeBytes());
    std::memcpy(fixture->verifyRows.contents(), kVerify.data(), sizeof(kVerify));
  }
  for (uint32_t lane = 0; lane < kBlocks; ++lane) {
    packed.blocks[lane].firstRow = kFirst[lane];
    packed.blocks[lane].rows = kBlockRows[lane];
  }
  packed.chains(std::span(packed.blocks).first(kBlocks));
  const auto view = [&](const MetalBuffer &buffer, uint64_t offset, uint64_t bytes) {
    return backend.view(buffer, offset, bytes);
  };
  for (uint32_t step = 0; step < 2; ++step) {
    const std::string where = what + " step " + std::to_string(step);
    if (step)
      for (Fixture *fixture : {&packed, &alone}) {
        fixture->advance(kRetained);
        fixture->fillPacked();
        fixture->clear();
      }
    // Each lane's rows of the packed step are its rows of `alone`.
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (uint32_t lane = 0; lane < kBlocks; ++lane)
        std::memcpy(static_cast<uint8_t *>(packed.packedLayer[layer].contents()) + kFirst[lane] * packedRowBytes,
                    alone.packedRow(layer, lane, 0), kBlockRows[lane] * packedRowBytes);
    std::optional<PreparedTables> tables;
    std::vector<PreparedTables> laneTables;
    if (layout != LinearInput::Plain) {
      tables.emplace(backend, layout, width, kStepRows / kRows);
      for (uint32_t lane = 0; lane < kBlocks; ++lane) laneTables.emplace_back(backend, layout, width, 1);
    }
    CommandGraph packedGraph, aloneGraph;
    PreparedInput prepared;
    std::vector<StepBlocks> singleBlocks;
    for (uint32_t lane = 0; lane < kBlocks; ++lane)
      singleBlocks.emplace_back(backend, std::array{GdnDecodeBlock{0, kBlockRows[lane], alone.blocks[lane].tape}});
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      GdnDecodeBuffers buffers = packed.decodeBuffers(layer);
      if (tables) buffers.linearScratch = tables->scratch();
      prepared = GDN::addDecode(packedGraph, buffers, shape, layer, cell.strides(), order, layout);
      for (uint32_t lane = 0; lane < kBlocks; ++lane) {
        GdnDecodeBuffers single = singleBlocks[lane].into(alone.decodeBuffers(layer));
        std::array<MetalBuffer, kMaxLanes> current, next;
        current.fill(alone.current[lane]);
        next.fill(alone.next[lane]);
        single.packed = view(alone.packedLayer[layer], lane * alone.packedStride * 2, alone.packedStride * 2);
        single.currentStates = current;
        single.nextStates = next;
        single.hidden = view(alone.hidden, lane * kRows * hiddenRowBytes, kRows * hiddenRowBytes);
        single.verifyRows = view(alone.verifyRows, lane * sizeof(uint32_t), sizeof(uint32_t));
        if (tables) single.linearScratch = laneTables[lane].scratch();
        (void)GDN::addDecode(aloneGraph, single, shape, layer, cell.strides(), order, layout);
      }
    }
    if (tables) tables->addReference(packedGraph, packed.hidden);
    (void)backend.submitCommandAsync(packedGraph.dispatches()).wait();
    (void)backend.submitCommandAsync(aloneGraph.dispatches()).wait();
    if (tables)
      tables->requireWritten(prepared, packed.hidden, where);
    else
      require(prepared.layout == LinearInput::Plain, where + ": plain GDN claimed a table");
    for (uint32_t lane = 0; lane < kBlocks; ++lane) {
      const std::string label = where + " lane " + std::to_string(lane);
      require(!std::memcmp(packed.next[lane].contents(), alone.next[lane].contents(), cell.bytes),
              label + ": state cell differs from the block's alone");
      require(!std::memcmp(packed.hiddenRow(0, kFirst[lane], 0), alone.hiddenRow(lane, 0, 0),
                           kBlockRows[lane] * hiddenRowBytes),
              label + ": hidden rows differ from the block's alone");
      // Each field of the step's tape past the block's rows.
      const uint64_t slot = packed.blocks[lane].tape.stepSlot;
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        for (const auto &[first, rowBytes] :
             {std::pair{reinterpret_cast<const uint8_t *>(packed.tapeInputs(slot, layer, kBlockRows[lane])),
                        uint64_t{shape.convolutionDimension} * 2},
              std::pair{reinterpret_cast<const uint8_t *>(packed.tapeKv(slot, layer, kBlockRows[lane])),
                        packed.kvWidth * 2},
              std::pair{reinterpret_cast<const uint8_t *>(packed.tapeDecay(slot, layer, kBlockRows[lane])),
                        uint64_t{shape.valueHeads} * 4},
              std::pair{reinterpret_cast<const uint8_t *>(packed.tapeBeta(slot, layer, kBlockRows[lane])),
                        uint64_t{shape.valueHeads} * 2}})
          require(std::all_of(first, first + (kRows - kBlockRows[lane]) * rowBytes,
                              [](uint8_t byte) { return byte == 0xFF; }),
                  label + ": a tape row past the block's rows was written");
    }
    require(!std::memcmp(packed.tape.contents(), alone.tape.contents(), packed.tape.sizeBytes()),
            where + ": tapes differ from the blocks' alone");
    const auto *rest = static_cast<const uint8_t *>(packed.hidden.contents()) + kStepRows * hiddenRowBytes;
    require(std::all_of(rest, rest + (kMaxLanes * kRows - kStepRows) * hiddenRowBytes,
                        [](uint8_t byte) { return byte == 0; }),
            where + ": a hidden row past the last block's was written");
  }
}

// A tree of a block's rows (metal/abi/VerifyTree.h): each row's parent
// before it, the anchor's none.
struct BlockTree final {
  std::vector<uint32_t> parent;

  [[nodiscard]] uint32_t rows() const { return static_cast<uint32_t>(parent.size()); }
  // The rows from the anchor down to `row`.
  [[nodiscard]] std::vector<uint32_t> path(uint32_t row) const {
    std::vector<uint32_t> rows{row};
    while (rows.front()) rows.insert(rows.begin(), parent[rows.front()]);
    return rows;
  }
  [[nodiscard]] uint32_t ancestors(uint32_t row) const {
    uint32_t mask = 0;
    for (const uint32_t on : path(row)) mask |= 1u << on;
    return mask;
  }
  [[nodiscard]] TreeRow node(uint32_t row) const {
    return {.ancestors = ancestors(row),
            .parent = static_cast<uint16_t>(row ? parent[row] : SPLASH_TREE_ROOT),
            .depth = static_cast<uint16_t>(path(row).size() - 1)};
  }
  // The rows below `rows` that are no such row's parent.
  [[nodiscard]] std::vector<uint32_t> leaves(uint32_t below) const {
    std::vector<uint32_t> result;
    for (uint32_t row = 0; row < below; ++row)
      if (std::none_of(parent.begin() + row + 1, parent.begin() + below, [&](uint32_t p) { return p == row; }))
        result.push_back(row);
    return result;
  }
};

// A tree of `rows` rows whose row r hangs off one of the `reach` rows before
// it, drawn from `random`.
BlockTree randomTree(Random &random, uint32_t rows, uint32_t reach) {
  BlockTree tree{{0}};
  for (uint32_t row = 1; row < rows; ++row) {
    const uint32_t span = std::min(row, reach);
    const auto back = std::min(span - 1, static_cast<uint32_t>((random.unit() + 1.0F) * 0.5F * span));
    tree.parent.push_back(row - 1 - back);
  }
  return tree;
}

// A step of every layer of `fixture`'s lanes, of blocks `blocks`, over cells
// of `cell`'s strides.
void decodeStep(MetalBackend &backend, const Fixture &fixture, const Cell &cell,
                std::span<const GdnDecodeBlock> blocks) {
  CommandGraph graph;
  const StepBlocks step(backend, blocks);
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    (void)GDN::addDecode(graph, step.into(fixture.decodeBuffers(layer)), fixture.shape, layer, cell.strides(),
                         GdnHeadOrder::Grouped, LinearInput::Plain);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
}

// Every field of tape row `row` of one layer of a step's tape at `slot`.
std::vector<uint8_t> tapeRow(const Fixture &fixture, uint64_t slot, uint32_t layer, uint32_t row) {
  const GdnShape &shape = fixture.shape;
  std::vector<uint8_t> bytes;
  const auto append = [&](const void *data, uint64_t size) {
    const auto *first = static_cast<const uint8_t *>(data);
    bytes.insert(bytes.end(), first, first + size);
  };
  append(fixture.tapeInputs(slot, layer, row), uint64_t{shape.convolutionDimension} * 2);
  append(fixture.tapeKv(slot, layer, row), fixture.kvWidth * 2);
  append(fixture.tapeDecay(slot, layer, row), uint64_t{shape.valueHeads} * 4);
  append(fixture.tapeBeta(slot, layer, row), uint64_t{shape.valueHeads} * 2);
  return bytes;
}

// The fixture's fp32 cells' recurrent states rounded to bf16, in place, in
// the layout of `narrow`'s cells (raggedBlocks).
void narrowCells(Fixture &fixture, const Cell &narrow) {
  const uint64_t elements =
      uint64_t{fixture.shape.valueHeads} * fixture.shape.headDimension * fixture.shape.headDimension;
  for (const MetalBuffer &buffer : fixture.current) {
    auto *bytes = static_cast<uint8_t *>(buffer.contents());
    const std::vector<uint8_t> wide(bytes, bytes + fixture.cell.bytes);
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      const float *values = fixture.cell.recurrent(wide.data(), layer);
      auto *rounded = reinterpret_cast<uint16_t *>(bytes + narrow.convBytes + layer * narrow.recurrentLayerBytes);
      for (uint64_t index = 0; index < elements; ++index) rounded[index] = floatToBf16(values[index]);
    }
  }
}

// Lanes 0 to 3 take blocks of 16, 5, 9 and 12 rows at rows 0, 16, 21 and 30
// of a step's 42, each block a random tree whose rows 1 and 2 are the
// anchor's children, so that a path past row 2 skips a row; the next step's
// blocks of 12, 9, 5 and 16 rows fold each lane's path to its deepest leaf of
// the first step's tree, which reaches past row 2 (row 4 hangs off row 1 or
// later). Over both steps of every layer,
// each lane's state cells, tape and hidden rows are its blocks' run alone
// (one-lane steps of its trees from row 0, in buffers of their own), byte for
// byte, and the hidden rows past the last block's stay zero. Every tree row
// of no block's row holds 0xFF, whose ancestors read as a chain's, so a lane
// that took its tree rows at another row than its first (lanes 2 and 3 of
// the first step, at lane x 16) runs another lane's tree or a chain.
void raggedTrees(MetalBackend &backend, const GdnShape &shape, bool float32, GdnStateFormat format) {
  constexpr std::array<std::array<uint32_t, kMaxLanes>, 2> kBlockRows{{{16, 5, 9, 12}, {12, 9, 5, 16}}};
  const std::string what = std::string("ragged trees vh") + std::to_string(shape.valueHeads) +
                           (float32 ? " f32 norm " : " bf16 norm ") + std::string(formatName(format));
  const GdnHeadOrder order = float32 ? GdnHeadOrder::Tiled : GdnHeadOrder::Grouped;
  const Cell cell(shape, format);
  const uint64_t hiddenRowBytes = uint64_t{shape.valueHeads} * kHeadDim * 2;
  const uint64_t packedRowBytes = uint64_t{shape.packedWidth} * 2;
  Random random(0x2A66ED + shape.valueHeads + float32 + 2 * (format == GdnStateFormat::BFloat16));
  Fixture packed(backend, shape, kMaxLanes, float32), alone(backend, shape, kMaxLanes, float32);
  for (Fixture *fixture : {&packed, &alone}) {
    if (format == GdnStateFormat::BFloat16) narrowCells(*fixture, cell);
    std::memset(fixture->tape.contents(), 0xFF, fixture->tape.sizeBytes());
  }
  const auto view = [&](const MetalBuffer &buffer, uint64_t offset, uint64_t bytes) {
    return backend.view(buffer, offset, bytes);
  };
  // Each lane's accepted path of its first step's tree.
  std::array<uint32_t, kMaxLanes> accepted{};
  for (uint32_t step = 0; step < 2; ++step) {
    const std::string where = what + " step " + std::to_string(step);
    const std::array<uint32_t, kMaxLanes> &blockRows = kBlockRows[step];
    std::array<uint32_t, kMaxLanes> first{};
    for (uint32_t lane = 1; lane < kMaxLanes; ++lane) first[lane] = first[lane - 1] + blockRows[lane - 1];
    const uint32_t stepRows = first.back() + blockRows.back();
    for (Fixture *fixture : {&packed, &alone}) {
      if (step) {
        std::swap(fixture->current, fixture->next);
        for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
          auto &tape = fixture->blocks[lane].tape;
          tape = {tape.stepSlot, tape.pendingSlot, accepted[lane]};
        }
        fixture->fillPacked();
        fixture->clear();
      }
      std::memset(fixture->tree.contents(), 0xFF, fixture->tree.sizeBytes());
    }
    std::array<BlockTree, kMaxLanes> trees;
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      trees[lane] = randomTree(random, blockRows[lane], 3);
      trees[lane].parent[2] = 0;
      packed.blocks[lane].firstRow = first[lane];
      packed.blocks[lane].rows = blockRows[lane];
      for (uint32_t row = 0; row < blockRows[lane]; ++row) {
        static_cast<TreeRow *>(packed.tree.contents())[first[lane] + row] = trees[lane].node(row);
        static_cast<TreeRow *>(alone.tree.contents())[lane * kRows + row] = trees[lane].node(row);
      }
      // Each lane's rows of the packed step are its rows of `alone`.
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        std::memcpy(static_cast<uint8_t *>(packed.packedLayer[layer].contents()) + first[lane] * packedRowBytes,
                    alone.packedRow(layer, lane, 0), blockRows[lane] * packedRowBytes);
    }
    CommandGraph packedGraph, aloneGraph;
    std::vector<StepBlocks> singleBlocks;
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      singleBlocks.emplace_back(backend, std::array{GdnDecodeBlock{0, blockRows[lane], alone.blocks[lane].tape}});
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      (void)GDN::addDecode(packedGraph, packed.decodeBuffers(layer), shape, layer, cell.strides(), order,
                           LinearInput::Plain);
      for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
        GdnDecodeBuffers single = singleBlocks[lane].into(alone.decodeBuffers(layer));
        std::array<MetalBuffer, kMaxLanes> current, next;
        current.fill(alone.current[lane]);
        next.fill(alone.next[lane]);
        single.packed = view(alone.packedLayer[layer], lane * alone.packedStride * 2, alone.packedStride * 2);
        single.currentStates = current;
        single.nextStates = next;
        single.hidden = view(alone.hidden, lane * kRows * hiddenRowBytes, kRows * hiddenRowBytes);
        single.tree = view(alone.tree, lane * kRows * sizeof(TreeRow), kRows * sizeof(TreeRow));
        (void)GDN::addDecode(aloneGraph, single, shape, layer, cell.strides(), order, LinearInput::Plain);
      }
    }
    (void)backend.submitCommandAsync(packedGraph.dispatches()).wait();
    (void)backend.submitCommandAsync(aloneGraph.dispatches()).wait();
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      const std::string label = where + " lane " + std::to_string(lane);
      require(!std::memcmp(packed.next[lane].contents(), alone.next[lane].contents(), cell.bytes),
              label + ": state cell differs from the lane's alone");
      require(!std::memcmp(packed.hiddenRow(0, first[lane], 0), alone.hiddenRow(lane, 0, 0),
                           blockRows[lane] * hiddenRowBytes),
              label + ": hidden rows differ from the lane's alone");
      const uint64_t slot = packed.blocks[lane].tape.stepSlot;
      require(!std::memcmp(packed.tapeLayer(slot, 0), alone.tapeLayer(slot, 0), kLayers * packed.tapeBytes),
              label + ": tape differs from the lane's alone");
    }
    const auto *rest = static_cast<const uint8_t *>(packed.hidden.contents()) + stepRows * hiddenRowBytes;
    require(std::all_of(rest, rest + (kMaxLanes * kRows - stepRows) * hiddenRowBytes,
                        [](uint8_t byte) { return byte == 0; }),
            where + ": a hidden row past the last block's was written");
    if (step) continue;
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      const std::vector<uint32_t> leaves = trees[lane].leaves(blockRows[lane]);
      const uint32_t deepest = *std::max_element(leaves.begin(), leaves.end(), [&](uint32_t a, uint32_t b) {
        return trees[lane].path(a).size() < trees[lane].path(b).size();
      });
      accepted[lane] = trees[lane].ancestors(deepest);
      require(accepted[lane] != (2u << deepest) - 1u, where + ": a lane's deepest path skips no row");
    }
  }
  std::cout << what << ": lanes of 16, 5, 9 and 12 rows of trees, then of 12, 9, 5 and 16, each as alone\n";
}

// The rows of a tree block each compute what the chain of their path
// computes: four lanes of random trees of `blockRows` rows (the last
// verifying only its first rows) from the same states, each a step of every
// layer, on the kernels of eight rows or of sixteen.
// Each row's tape row is the last tape row of its path run as a chain (a
// lane of its own from the same state, its packed rows the path's), bit for
// bit, and its hidden row that chain's last within their sums' rounding: the
// chain advances the state row by row, the tree reads the state before the
// block through its query carried up the path (gdn_decode_transform). An
// unverified row's hidden row is zero. The next step, folding each lane's
// deepest path, leaves the cells, tape and hidden rows that folding the
// chain of that path leaves, bit for bit.
void treeRows(MetalBackend &backend, const GdnShape &shape, bool float32, GdnStateFormat format,
              uint32_t blockRows) {
  const std::string what = std::string("tree rows vh") + std::to_string(shape.valueHeads) +
                           (float32 ? " f32 norm " : " bf16 norm ") + std::string(formatName(format)) + " blocks of " +
                           std::to_string(blockRows);
  const Cell cell(shape, format);
  const bool narrow = format == GdnStateFormat::BFloat16;
  Random random(0x7EE5 + shape.valueHeads + float32 + 2 * narrow + blockRows);
  Fixture tree(backend, shape, kMaxLanes, float32);
  if (narrow) narrowCells(tree, cell);
  // Narrow and wide trees, and a chain whose last row is its second last's
  // sibling; the last lane verifies only its first rows.
  std::array<BlockTree, kMaxLanes> trees{randomTree(random, blockRows, 3), randomTree(random, blockRows, blockRows),
                                         randomTree(random, blockRows, 1), randomTree(random, blockRows, 4)};
  trees[2].parent.back() = blockRows - 3;
  const std::array<uint32_t, kMaxLanes> verified{blockRows, blockRows, blockRows, blockRows - 3};
  std::memcpy(tree.verifyRows.contents(), verified.data(), sizeof(verified));
  auto *rows = static_cast<TreeRow *>(tree.tree.contents());
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    tree.blocks[lane].rows = blockRows;
    for (uint32_t row = 0; row < blockRows; ++row) rows[lane * kRows + row] = trees[lane].node(row);
  }
  decodeStep(backend, tree, cell, tree.blocks);

  const uint64_t hiddenRowBytes = uint64_t{shape.valueHeads} * kHeadDim * 2;
  const uint64_t packedRowBytes = uint64_t{shape.packedWidth} * 2;
  uint64_t compared = 0, inexact = 0;
  // A chain lane of `chain` runs the path of `leaf` of tree lane `lane`.
  const auto runPath = [&](Fixture &chain, uint32_t lane, uint32_t laneOfChain, uint32_t leaf) {
    const std::vector<uint32_t> path = trees[lane].path(leaf);
    chain.blocks[laneOfChain].rows = static_cast<uint32_t>(path.size());
    std::memcpy(chain.current[laneOfChain].contents(), tree.current[lane].contents(), tree.cell.bytes);
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (uint32_t row = 0; row < path.size(); ++row)
        std::memcpy(const_cast<uint16_t *>(chain.packedRow(layer, laneOfChain, row)),
                    tree.packedRow(layer, lane, path[row]), packedRowBytes);
    static_cast<uint32_t *>(chain.verifyRows.contents())[laneOfChain] = blockRows;
  };
  const auto comparePath = [&](const Fixture &chain, uint32_t lane, uint32_t laneOfChain, uint32_t leaf) {
    const std::vector<uint32_t> path = trees[lane].path(leaf);
    const std::string where = what + " lane " + std::to_string(lane) + " leaf " + std::to_string(leaf);
    for (uint32_t row = 0; row < path.size(); ++row) {
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        require(tapeRow(tree, tree.blocks[lane].tape.stepSlot, layer, path[row]) ==
                    tapeRow(chain, chain.blocks[laneOfChain].tape.stepSlot, layer, row),
                where + ": a tree row's tape row is not its path's");
      // Two units, or where a recurrent value cancels, the fp32 error of its
      // sum relative to its head's largest value: a 512th of it (one value
      // of a row ran 3.4 units off in 10^6, the rest of its head exact).
      for (uint32_t head = 0; head < shape.valueHeads; ++head) {
        const uint16_t *got = tree.hiddenRow(lane, path[row], head);
        const uint16_t *expected = chain.hiddenRow(laneOfChain, row, head);
        double largest = 0.0;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) largest = std::max(largest, std::fabs(double(bf16ToFloat(expected[dim]))));
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
          require(closeBfloat(got[dim], bf16ToFloat(expected[dim]), 2.0, 1e-6 + std::ldexp(largest, -9)),
                  where + ": a tree row's hidden row is not its path's");
          inexact += got[dim] != expected[dim];
        }
      }
      compared += hiddenRowBytes / 2;
    }
  };
  // Each lane's deepest leaf first, one chain lane a tree lane, then the
  // other leaves four at a time.
  std::array<uint32_t, kMaxLanes> deepest{};
  std::vector<std::pair<uint32_t, uint32_t>> others;
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    const std::vector<uint32_t> leaves = trees[lane].leaves(verified[lane]);
    deepest[lane] = *std::max_element(leaves.begin(), leaves.end(), [&](uint32_t a, uint32_t b) {
      return trees[lane].path(a).size() < trees[lane].path(b).size();
    });
    for (const uint32_t leaf : leaves)
      if (leaf != deepest[lane]) others.emplace_back(lane, leaf);
  }
  Fixture chain(backend, shape, kMaxLanes, float32);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) runPath(chain, lane, lane, deepest[lane]);
  chain.chains();
  decodeStep(backend, chain, cell, chain.blocks);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) comparePath(chain, lane, lane, deepest[lane]);
  for (size_t first = 0; first < others.size(); first += kMaxLanes) {
    Fixture leaves(backend, shape, kMaxLanes, float32);
    const size_t count = std::min<size_t>(kMaxLanes, others.size() - first);
    for (uint32_t index = 0; index < count; ++index)
      runPath(leaves, others[first + index].first, index, others[first + index].second);
    leaves.chains();
    decodeStep(backend, leaves, cell, std::span(leaves.blocks).first(count));
    for (uint32_t index = 0; index < count; ++index)
      comparePath(leaves, others[first + index].first, index, others[first + index].second);
  }
  // Rows past a lane's verified rows output zero.
  for (uint32_t row = verified.back(); row < blockRows; ++row) {
    const uint16_t *hidden = tree.hiddenRow(kMaxLanes - 1, row, 0);
    require(std::all_of(hidden, hidden + hiddenRowBytes / 2, [](uint16_t value) { return (value & 0x7FFF) == 0; }),
            what + ": an unverified tree row is not zero");
  }
  require(inexact * 50 <= compared, what + ": more than 2% of tree rows' hidden values differ from their paths'");

  // The next step folds each tree lane's deepest path, the chain lane the
  // same rows of its own step, both over chains of the same new rows.
  std::array<uint32_t, kMaxLanes> treePaths{}, chainPaths{};
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    treePaths[lane] = trees[lane].ancestors(deepest[lane]);
    chainPaths[lane] = chainPath(static_cast<uint32_t>(trees[lane].path(deepest[lane]).size()));
  }
  for (Fixture *fixture : {&tree, &chain}) {
    std::swap(fixture->current, fixture->next);
    const std::array<uint32_t, kMaxLanes> &paths = fixture == &tree ? treePaths : chainPaths;
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      auto &tape = fixture->blocks[lane].tape;
      tape = {tape.stepSlot, tape.pendingSlot, paths[lane]};
      fixture->blocks[lane].rows = blockRows;
    }
    std::fill_n(static_cast<uint32_t *>(fixture->verifyRows.contents()), kMaxLanes, blockRows);
    fixture->chains();
    fixture->clear();
  }
  tree.fillPacked();
  std::memcpy(chain.packed.contents(), tree.packed.contents(), tree.packed.sizeBytes());
  decodeStep(backend, tree, cell, tree.blocks);
  decodeStep(backend, chain, cell, chain.blocks);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    const std::string where = what + " fold lane " + std::to_string(lane);
    require(!std::memcmp(tree.next[lane].contents(), chain.next[lane].contents(), cell.bytes),
            where + ": folding a tree's path left another state than folding its chain");
    require(!std::memcmp(tree.hiddenRow(lane, 0, 0), chain.hiddenRow(lane, 0, 0), blockRows * hiddenRowBytes),
            where + ": the step after a tree's path differs from the step after its chain");
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (uint32_t row = 0; row < blockRows; ++row)
        require(tapeRow(tree, tree.blocks[lane].tape.stepSlot, layer, row) ==
                    tapeRow(chain, chain.blocks[lane].tape.stepSlot, layer, row),
                where + ": the step after a tree's path left another tape");
  }
  std::cout << what << ": " << inexact << " of " << compared << " hidden values off their paths' by rounding\n";
}

// A step of eight rows a lane after one of sixteen: the 8-row kernels fold a
// lane's pending path of at most eight rows by its rows, which may lie past
// row 7 of the block before (GDN::addDecode takes them for it). Each lane's
// 16-row tree accepts the path of its last row past row 7 whose path holds
// at most eight rows, and a lane of the same state runs that path as a
// chain; the step of eight new rows a lane after each, folding the tree's
// path or the chain's rows, on the 8-row kernels, leaves the same cells, tape
// and hidden rows, bit for bit.
void narrowKernelsFoldPathsPastTheirRows(MetalBackend &backend, const GdnShape &shape, GdnStateFormat format) {
  constexpr uint32_t kNarrowRows = SPLASH_GDN_NARROW_ROWS;
  const std::string what = std::string("8-row fold of a 16-row path vh") + std::to_string(shape.valueHeads) + " " +
                           std::string(formatName(format));
  const bool float32 = familyNorms(shape).front();
  const Cell cell(shape, format);
  Random random(0xF0D8 + shape.valueHeads + (format == GdnStateFormat::BFloat16));
  Fixture tree(backend, shape, kMaxLanes, float32);
  if (format == GdnStateFormat::BFloat16) narrowCells(tree, cell);
  std::array<BlockTree, kMaxLanes> trees;
  std::array<uint32_t, kMaxLanes> accepted{};
  auto *rows = static_cast<TreeRow *>(tree.tree.contents());
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    trees[lane] = randomTree(random, kRows, kRows);
    tree.blocks[lane].rows = kRows;
    for (uint32_t row = 0; row < kRows; ++row) rows[lane * kRows + row] = trees[lane].node(row);
    for (uint32_t row = kRows; row-- > kNarrowRows && !accepted[lane];)
      if (trees[lane].path(row).size() <= kNarrowRows) accepted[lane] = row;
    require(accepted[lane], what + ": a lane's tree has no path of at most eight rows past row 7");
  }
  std::fill_n(static_cast<uint32_t *>(tree.verifyRows.contents()), kMaxLanes, kRows);
  decodeStep(backend, tree, cell, tree.blocks);
  // Each accepted path run as a chain from the tree lane's state.
  const uint64_t packedRowBytes = uint64_t{shape.packedWidth} * 2;
  Fixture chain(backend, shape, kMaxLanes, float32);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    const std::vector<uint32_t> path = trees[lane].path(accepted[lane]);
    chain.blocks[lane].rows = static_cast<uint32_t>(path.size());
    std::memcpy(chain.current[lane].contents(), tree.current[lane].contents(), tree.cell.bytes);
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (uint32_t row = 0; row < path.size(); ++row)
        std::memcpy(const_cast<uint16_t *>(chain.packedRow(layer, lane, row)), tree.packedRow(layer, lane, path[row]),
                    packedRowBytes);
  }
  chain.chains();
  decodeStep(backend, chain, cell, chain.blocks);
  // The next step of eight new rows a lane, which folds the tree's paths and
  // the chains' rows, on the 8-row kernels.
  for (Fixture *fixture : {&tree, &chain}) {
    std::swap(fixture->current, fixture->next);
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      auto &tape = fixture->blocks[lane].tape;
      const uint32_t path = fixture == &tree ? trees[lane].ancestors(accepted[lane])
                                             : chainPath(static_cast<uint32_t>(trees[lane].path(accepted[lane]).size()));
      tape = {tape.stepSlot, tape.pendingSlot, path};
      fixture->blocks[lane].rows = kNarrowRows;
    }
    std::fill_n(static_cast<uint32_t *>(fixture->verifyRows.contents()), kMaxLanes, kNarrowRows);
    fixture->chains();
    fixture->clear();
  }
  tree.fillPacked();
  std::memcpy(chain.packed.contents(), tree.packed.contents(), tree.packed.sizeBytes());
  for (const Fixture *fixture : {&tree, &chain}) {
    CommandGraph graph;
    const StepBlocks step(backend, fixture->blocks);
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      (void)GDN::addDecode(graph, step.into(fixture->decodeBuffers(layer)), shape, layer, cell.strides(),
                           GdnHeadOrder::Grouped, LinearInput::Plain);
    for (const auto &dispatch : graph.dispatches())
      require(!dispatch.pipelineName.ends_with("_r16"), what + ": a step of eight rows a lane ran a 16-row kernel");
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
  }
  const uint64_t hiddenRowBytes = uint64_t{shape.valueHeads} * kHeadDim * 2;
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    const std::string where = what + " lane " + std::to_string(lane);
    require(!std::memcmp(tree.next[lane].contents(), chain.next[lane].contents(), cell.bytes),
            where + ": folding a tree's path left another state than folding its chain");
    require(!std::memcmp(tree.hiddenRow(lane, 0, 0), chain.hiddenRow(lane, 0, 0), kNarrowRows * hiddenRowBytes),
            where + ": the step after a tree's path differs from the step after its chain");
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (uint32_t row = 0; row < kNarrowRows; ++row)
        require(tapeRow(tree, tree.blocks[lane].tape.stepSlot, layer, row) ==
                    tapeRow(chain, chain.blocks[lane].tape.stepSlot, layer, row),
                where + ": the step after a tree's path left another tape");
  }
  std::cout << what << ": paths of rows";
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
    for (const uint32_t row : trees[lane].path(accepted[lane])) std::cout << (row ? "," : (lane ? " | " : " ")) << row;
  std::cout << '\n';
}

// A 16-row chain is two 8-row steps of the same rows with an fp32 state,
// bit for bit: the first step's rows are the chain's first eight, the step
// after it, which folds them in and convolves the next eight over their
// tape, its last eight, on the 8-row kernels against the 16-row ones; and a
// step after either, folding what each left pending, leaves the same cells,
// tape and rows.
void chainHalves(MetalBackend &backend, const GdnShape &shape, bool float32) {
  const std::string what = std::string("chain halves vh") + std::to_string(shape.valueHeads) +
                           (float32 ? " f32 norm" : " bf16 norm");
  constexpr uint32_t kHalf = SPLASH_GDN_NARROW_ROWS;
  static_assert(2 * kHalf == kRows, "a verify block is two chains of the 8-row kernels");
  const Cell cell(shape);
  Fixture whole(backend, shape, kMaxLanes, float32), halves(backend, shape, kMaxLanes, float32);
  const auto copy = [](const MetalBuffer &from, const MetalBuffer &to) {
    std::memcpy(to.contents(), from.contents(), from.sizeBytes());
  };
  copy(whole.packed, halves.packed);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) copy(whole.current[lane], halves.current[lane]);
  const uint64_t hiddenRowBytes = uint64_t{shape.valueHeads} * kHeadDim * 2;
  const uint64_t packedRowBytes = uint64_t{shape.packedWidth} * 2;
  decodeStep(backend, whole, cell, whole.blocks);
  // Each lane's first eight rows, then its next eight as the next step's
  // first.
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) halves.blocks[lane].rows = kHalf;
  std::fill_n(static_cast<uint32_t *>(halves.verifyRows.contents()), kMaxLanes, kHalf);
  decodeStep(backend, halves, cell, halves.blocks);
  std::vector<uint8_t> firstHidden(static_cast<const uint8_t *>(halves.hidden.contents()),
                                   static_cast<const uint8_t *>(halves.hidden.contents()) + halves.hidden.sizeBytes());
  const auto requireRows = [&](const Fixture &half, const std::vector<uint8_t> &hidden, uint32_t from,
                               const std::string &which) {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      require(!std::memcmp(hidden.data() + (uint64_t{lane} * kRows) * hiddenRowBytes,
                           whole.hiddenRow(lane, from, 0), kHalf * hiddenRowBytes),
              what + ": the " + which + " step's rows are not the chain's");
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        for (uint32_t row = 0; row < kHalf; ++row)
          require(tapeRow(half, half.blocks[lane].tape.stepSlot, layer, row) ==
                      tapeRow(whole, whole.blocks[lane].tape.stepSlot, layer, from + row),
                  what + ": the " + which + " step's tape rows are not the chain's");
    }
  };
  requireRows(halves, firstHidden, 0, "first");
  std::array<uint32_t, kMaxLanes> pending{};
  pending.fill(kHalf);
  halves.advance(pending);
  halves.clear();
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      std::memmove(const_cast<uint16_t *>(halves.packedRow(layer, lane, 0)), halves.packedRow(layer, lane, kHalf),
                   kHalf * packedRowBytes);
  decodeStep(backend, halves, cell, halves.blocks);
  const std::vector<uint8_t> secondHidden(static_cast<const uint8_t *>(halves.hidden.contents()),
                                          static_cast<const uint8_t *>(halves.hidden.contents()) +
                                              halves.hidden.sizeBytes());
  requireRows(halves, secondHidden, kHalf, "second");
  // A step after each, of the same new rows.
  pending.fill(kRows);
  whole.advance(pending);
  pending.fill(kHalf);
  halves.advance(pending);
  for (Fixture *fixture : {&whole, &halves}) {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) fixture->blocks[lane].rows = kHalf;
    std::fill_n(static_cast<uint32_t *>(fixture->verifyRows.contents()), kMaxLanes, kHalf);
    fixture->clear();
  }
  whole.fillPacked();
  copy(whole.packed, halves.packed);
  decodeStep(backend, whole, cell, whole.blocks);
  decodeStep(backend, halves, cell, halves.blocks);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    require(!std::memcmp(whole.next[lane].contents(), halves.next[lane].contents(), cell.bytes),
            what + ": folding the chain's rows left another state than folding its halves'");
    require(!std::memcmp(whole.hiddenRow(lane, 0, 0), halves.hiddenRow(lane, 0, 0), kHalf * hiddenRowBytes),
            what + ": the step after the chain differs from the step after its halves");
  }
  std::cout << what << ": a 16-row chain is its two 8-row steps bit for bit\n";
}

void rejectsInvalid(MetalBackend &backend) {
  const GdnShape &shape = kShapes.front();
  Fixture fixture(backend, shape, 1);
  CommandGraph graph;
  const auto encode = [&](const GdnDecodeBuffers &buffers, uint32_t layer = 0) {
    GDN::addDecode(graph, buffers, shape, layer, fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain);
  };
  const auto withTapes = [&](std::span<const GdnTapeLane> tapes, uint32_t layer = 0) {
    GdnDecodeBuffers buffers = fixture.decodeBuffers(layer);
    buffers.tapes = tapes;
    buffers.rows = static_cast<uint32_t>(tapes.size()) * kRows;
    encode(buffers, layer);
  };
  const std::array<GdnTapeLane, kMaxLanes + 1> wide{};
  for (const size_t lanes : {size_t{0}, wide.size()})
    rejects([&] { withTapes(std::span(wide).first(lanes)); }, "invalid GDN decode geometry",
            "a GDN decode of no lanes or more than a batch was accepted");
  rejects(
      [&] {
        GDN::addDecode(graph, fixture.decodeBuffers(0), GdnShape{16, 40, 128, 9216, 14400}, 0,
                       fixture.cell.strides(), GdnHeadOrder::Grouped, LinearInput::Plain);
      },
      "invalid GDN shape", "an inconsistent GDN shape was accepted");
  // A lane cap of no rows or past a verify block's, and fewer of the step's
  // rows than its lanes.
  for (const auto &[cap, rows] : {std::pair{0U, kRows}, std::pair{kRows + 1, kRows}, std::pair{kRows, 0U}})
    rejects(
        [&] {
          GdnDecodeBuffers buffers = fixture.decodeBuffers(0);
          buffers.laneCap = cap;
          buffers.rows = rows;
          encode(buffers);
        },
        "invalid GDN decode geometry", "a GDN decode took a lane cap out of range or fewer rows than lanes");
  // A pending path of a row past a step's, or without its anchor's row; a
  // step's tape misaligned, or overlapping the pending tape of its own lane
  // or of another, or another lane's step.
  const uint64_t tape = fixture.tapeBytes;
  for (const std::array<GdnTapeLane, 2> lanes :
       {std::array{GdnTapeLane{0, tape, chainPath(kRows + 1)}, GdnTapeLane{2 * tape, 3 * tape, 0}},
        std::array{GdnTapeLane{0, tape, 0b110}, GdnTapeLane{2 * tape, 3 * tape, 0}},
        std::array{GdnTapeLane{0, tape + 8, 0}, GdnTapeLane{2 * tape, 3 * tape, 0}},
        std::array{GdnTapeLane{0, tape - 16, 0}, GdnTapeLane{2 * tape, 3 * tape, 0}},
        std::array{GdnTapeLane{0, tape, 0}, GdnTapeLane{2 * tape, 16, 0}},
        std::array{GdnTapeLane{0, tape, 0}, GdnTapeLane{3 * tape, tape + 16, 0}}})
    rejects(
        [&] { withTapes(lanes); },
        "invalid GDN decode tape", "a GDN decode took a pending path of no step's rows or overlapping tapes");
  // Offset arithmetic must not wrap either when selecting a layer or when
  // reaching the end of its tape. Otherwise the extent/overlap checks can
  // accept an offset far outside the buffer without submitting it here.
  const uint64_t nearEnd = std::numeric_limits<uint64_t>::max() - 15;
  for (const bool pending : {false, true})
    for (const auto &[slot, layer] :
         {std::pair{nearEnd, 0U}, std::pair{uint64_t{0} - tape, 1U}})
      rejects(
          [&] {
            std::array tapes{fixture.blocks[0].tape};
            (pending ? tapes[0].pendingSlot : tapes[0].stepSlot) = slot;
            withTapes(tapes, layer);
          },
          "invalid GDN decode tape", "a GDN decode accepted an overflowing tape offset");
  rejects(
      [&] {
        auto buffers = fixture.decodeBuffers(0);
        buffers.linearScratch.input = sharedBuffer(backend, 16);
        buffers.linearScratch.sums = sharedBuffer(backend, 4);
        GDN::addDecode(graph, buffers, shape, 0, fixture.cell.strides(), GdnHeadOrder::Grouped,
                       LinearInput::Table16);
      },
      "linear table buffer holds", "a GDN decode wrote a table into short scratch");
  rejects(
      [&] {
        // F32 weights need twice the bytes of bf16 ones.
        auto buffers = fixture.decodeBuffers(0);
        buffers.mixerNorm.float32 = true;
        encode(buffers);
      },
      "norm weight buffer holds", "bf16 norm weights were read as F32");
  // A fold takes one to all of a step's rows of every layer, from an aligned
  // tape slot whose layers the tape holds, into a cell as large as the
  // state's.
  const MetalBuffer retainedRows = sharedBuffer(backend, 4);
  const GdnFoldBuffers fold{fixture.next[0], fixture.tape, fixture.slot(0, 1), retainedRows, fixture.current[0]};
  for (const uint32_t rows : {0U, kRows + 1})
    rejects([&] { GDN::addFold(graph, fold, shape, kLayers, rows, fixture.cell.strides()); },
            "invalid GDN fold geometry", "a fold of no rows or more than a step's was accepted");
  rejects(
      [&] {
        GdnFoldBuffers misaligned = fold;
        misaligned.stepSlot += 8;
        GDN::addFold(graph, misaligned, shape, kLayers, 1, fixture.cell.strides());
      },
      "invalid GDN fold geometry", "a fold of a misaligned tape slot was accepted");
  rejects([&] { GDN::addFold(graph, fold, shape, 2 * kMaxLanes * kLayers, 1, fixture.cell.strides()); },
          "GDN tape buffer holds", "a fold read past the tape");
  rejects(
      [&] {
        GdnFoldBuffers small = fold;
        small.fold = sharedBuffer(backend, fixture.cell.bytes / 2);
        GDN::addFold(graph, small, shape, kLayers, 1, fixture.cell.strides());
      },
      "GDN folded state buffer holds", "a fold wrote past its cell");
  require(graph.empty(), "invalid GDN request partially encoded a graph");
}

// A fold of the first rows of a step's retained path alone (GDN::addFold)
// stores the state the lane's next step holds when the lane retained those
// rows, bit for bit, in every layer: the recurrent state and the convolution
// rows, over either state format, for each count of a chain's rows and of a
// tree's path that skips rows, as a capture on a branch takes it. The step
// folds rows of the one before it, so a fold from the state it started from,
// or from the earlier step's tape slot, stores another state. A lane whose
// path holds fewer rows leaves the fold's cell as it was.
void foldMatchesTheNextStep(MetalBackend &backend, const GdnShape &shape, GdnStateFormat format) {
  const std::string what =
      "fold vh" + std::to_string(shape.valueHeads) + " " + std::string(formatName(format));
  // Over the first norm type the shape's families store: the fold reads the
  // tape the step wrote, whichever norms gave its rows.
  Fixture fixture(backend, shape, 1, familyNorms(shape).front());
  const Cell cell(shape, format);
  const uint64_t carried = uint64_t{kCarried} * shape.convolutionDimension * 2;
  const uint64_t recurrent =
      uint64_t{shape.valueHeads} * kHeadDim * kHeadDim * elementBytes(format);
  // The state before the steps: random convolution rows and recurrent
  // states in the cell's format.
  Random random(0xF01D + shape.valueHeads);
  auto *start = static_cast<uint8_t *>(fixture.current[0].contents());
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    auto *rows = reinterpret_cast<uint16_t *>(start + layer * cell.convLayerBytes);
    for (uint64_t index = 0; index < carried / 2; ++index)
      rows[index] = floatToBf16(random.unit());
    uint8_t *state = start + cell.convBytes + layer * cell.recurrentLayerBytes;
    for (uint64_t index = 0; index < recurrent / elementBytes(format); ++index) {
      const float value = random.unit() * 0.5F;
      if (format == GdnStateFormat::BFloat16)
        reinterpret_cast<uint16_t *>(state)[index] = floatToBf16(value);
      else
        reinterpret_cast<float *>(state)[index] = value;
    }
  }
  const auto addStep = [&](CommandGraph &graph, std::span<const MetalBuffer> current,
                           std::span<const MetalBuffer> next, GdnTapeLane tape) {
    fixture.blocks[0] = {0, kRows, tape};
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      GdnDecodeBuffers buffers = fixture.decodeBuffers(layer);
      buffers.currentStates = current;
      buffers.nextStates = next;
      (void)GDN::addDecode(graph, buffers, shape, layer, cell.strides(), GdnHeadOrder::Grouped,
                           LinearInput::Plain);
    }
  };
  // The step before it leaves its rows in lane 1's slot of parity 1; the
  // step, over new rows, folds three of them and leaves its own in the slot
  // of parity 0.
  fixture.blocks[0].tape = {fixture.slot(1, 0), fixture.slot(1, 1), 0};
  CommandGraph before;
  addStep(before, fixture.current, fixture.next, fixture.blocks[0].tape);
  (void)backend.submitCommandAsync(before.dispatches()).wait();
  const std::array<uint32_t, kMaxLanes> pending{3};
  fixture.advance(pending);
  fixture.fillPacked();
  CommandGraph step;
  addStep(step, fixture.current, fixture.next, fixture.blocks[0].tape);
  (void)backend.submitCommandAsync(step.dispatches()).wait();
  const uint64_t stepSlot = fixture.blocks[0].tape.stepSlot, pendingSlot = fixture.blocks[0].tape.pendingSlot;

  const MetalBuffer retainedRows = sharedBuffer(backend, 4), fold = sharedBuffer(backend, cell.bytes);
  std::array<MetalBuffer, kMaxLanes> current, next;
  current.fill(sharedBuffer(backend, cell.bytes));
  next.fill(sharedBuffer(backend, cell.bytes));
  const auto bytes = [](const MetalBuffer &buffer, uint64_t offset) {
    return static_cast<const uint8_t *>(buffer.contents()) + offset;
  };
  // The fold of the first `rows` rows of the path `retained` against the
  // lane's next step after it retained `first`, those rows. The next step
  // leaves its own rows in a slot of another lane's, so that the earlier
  // step's rows stay for the step to run again.
  const auto check = [&](uint32_t rows, uint32_t retained, uint32_t first, const std::string &where) {
    std::memset(fold.contents(), 0xA5, cell.bytes);
    *static_cast<uint32_t *>(retainedRows.contents()) = retained;
    std::memcpy(current[0].contents(), fixture.next[0].contents(), cell.bytes);
    std::memset(next[0].contents(), 0, cell.bytes);
    CommandGraph graph;
    GDN::addFold(graph, {fixture.next[0], fixture.tape, stepSlot, retainedRows, fold}, shape, kLayers, rows,
                 cell.strides());
    addStep(graph, current, next, {stepSlot, fixture.slot(2, 0), first});
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    if (std::popcount(retained) < static_cast<int>(rows)) {
      require(std::all_of(bytes(fold, 0), bytes(fold, cell.bytes), [](uint8_t byte) { return byte == 0xA5; }),
              where + ": a fold of rows the lane did not retain wrote its cell");
      return;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      const uint64_t rowsAt = layer * cell.convLayerBytes;
      const uint64_t stateAt = cell.convBytes + layer * cell.recurrentLayerBytes;
      require(!std::memcmp(bytes(fold, rowsAt), bytes(next[0], rowsAt), carried),
              where + ": the fold's convolution rows are not the next step's");
      require(!std::memcmp(bytes(fold, stateAt), bytes(next[0], stateAt), recurrent),
              where + ": the fold's state is not the next step's");
    }
  };
  for (uint32_t rows = 1; rows <= kRows; ++rows)
    for (const uint32_t count : {rows - 1, rows, kRows})
      check(rows, chainPath(count), chainPath(rows),
            what + " of " + std::to_string(rows) + " rows, " + std::to_string(count) + " retained");
  // The step again over the same rows as a tree whose deepest path skips a
  // branch at every third row: a fold of its path's first rows, for a path
  // of fewer, as many and more, folds the path's rows of the tape, not the
  // step's first.
  static_assert(kRows == 16, "the tree's rows");
  const BlockTree branched{{0, 0, 0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14}};
  for (uint32_t row = 0; row < kRows; ++row) static_cast<TreeRow *>(fixture.tree.contents())[row] = branched.node(row);
  CommandGraph tree;
  addStep(tree, fixture.current, fixture.next, {pendingSlot, stepSlot, chainPath(3)});
  (void)backend.submitCommandAsync(tree.dispatches()).wait();
  const std::vector<uint32_t> deepest = branched.path(kRows - 1);
  for (uint32_t rows = 1; rows <= deepest.size(); ++rows) {
    const uint32_t first = branched.ancestors(deepest[rows - 1]);
    for (const uint32_t retained : {rows > 1 ? branched.ancestors(deepest[rows - 2]) : 0U, first,
                                    branched.ancestors(deepest.back())})
      check(rows, retained, first,
            what + " of " + std::to_string(rows) + " rows of a branched path, " +
                std::to_string(std::popcount(retained)) + " retained");
  }
  std::cout << what << ": each count of a chain's rows and of a branched path's folds to the next step's state\n";
}

} // namespace

// A bf16 state's decode computes as the fp32 state's does. From recurrent
// states of the same bf16 values, a first step and a second that folds each
// count of the first's rows leave the fp32 decode's tape, hidden rows and
// out-projection table byte for byte and its convolution rows, and store its
// recurrent state rounded to nearest even.
void bfloatState(MetalBackend &backend, const GdnShape &shape, uint32_t lanes, bool float32,
                 LinearInput layout) {
  const std::string where = caseName("bf16 state", shape, lanes, layout) + (float32 ? " f32 norm" : "");
  Fixture fixture(backend, shape, lanes, float32);
  const Cell narrow(shape, GdnStateFormat::BFloat16);
  const uint64_t elements = uint64_t{shape.valueHeads} * shape.headDimension * shape.headDimension;
  std::optional<PreparedTables> tables;
  if (layout != LinearInput::Plain) tables.emplace(backend, layout, shape.valueHeads * shape.headDimension, lanes);
  std::array<MetalBuffer, kMaxLanes> current, next;
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    current[lane] = sharedBuffer(backend, narrow.bytes);
    next[lane] = sharedBuffer(backend, narrow.bytes);
  }
  const auto wideState = [&](const MetalBuffer &cell, uint32_t layer) {
    return reinterpret_cast<float *>(static_cast<uint8_t *>(cell.contents()) + fixture.cell.convBytes +
                                     layer * fixture.cell.recurrentLayerBytes);
  };
  const auto narrowState = [&](const MetalBuffer &cell, uint32_t layer) {
    return reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(cell.contents()) + narrow.convBytes +
                                        layer * narrow.recurrentLayerBytes);
  };
  // The bf16 cells take the fp32 cells' convolution rows and recurrent states
  // rounded, which the fp32 cells then hold as well.
  const auto roundStates = [&] {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      std::memcpy(current[lane].contents(), fixture.current[lane].contents(), narrow.convBytes);
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        float *values = wideState(fixture.current[lane], layer);
        uint16_t *rounded = narrowState(current[lane], layer);
        for (uint64_t index = 0; index < elements; ++index) {
          rounded[index] = floatToBf16(values[index]);
          values[index] = bf16ToFloat(rounded[index]);
        }
      }
    }
  };
  const auto encode = [&](CommandGraph &graph, std::span<const MetalBuffer> currentStates,
                          std::span<const MetalBuffer> nextStates, const Cell &cell) {
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      GdnDecodeBuffers buffers = fixture.decodeBuffers(layer);
      buffers.currentStates = currentStates;
      buffers.nextStates = nextStates;
      if (tables) buffers.linearScratch = tables->scratch();
      (void)GDN::addDecode(graph, buffers, shape, layer, cell.strides(), GdnHeadOrder::Grouped, layout);
    }
  };
  const auto bytesOf = [](const MetalBuffer &buffer) {
    const auto *bytes = static_cast<const uint8_t *>(buffer.contents());
    return std::vector<uint8_t>(bytes, bytes + buffer.sizeBytes());
  };
  const auto step = [&](const std::string &when) {
    fixture.clear();
    for (const MetalBuffer &cell : next) std::memset(cell.contents(), 0, narrow.bytes);
    CommandGraph wide;
    encode(wide, fixture.current, fixture.next, fixture.cell);
    static_cast<void>(backend.submitCommandAsync(wide.dispatches()).wait());
    const std::vector<uint8_t> hidden = bytesOf(fixture.hidden), tape = bytesOf(fixture.tape);
    std::vector<uint8_t> table, sums;
    if (tables) table = bytesOf(tables->table), sums = bytesOf(tables->sums);
    // The bf16 decode writes them again.
    std::memset(fixture.hidden.contents(), 0, fixture.hidden.sizeBytes());
    for (uint32_t lane = 0; lane < lanes; ++lane)
      std::memset(static_cast<uint8_t *>(fixture.tape.contents()) + fixture.blocks[lane].tape.stepSlot, 0,
                  kLayers * fixture.tapeBytes);
    if (tables) {
      std::memset(tables->table.contents(), 0, tables->table.sizeBytes());
      std::memset(tables->sums.contents(), 0, tables->sums.sizeBytes());
    }
    CommandGraph bfloat;
    encode(bfloat, current, next, narrow);
    static_cast<void>(backend.submitCommandAsync(bfloat.dispatches()).wait());
    const std::string label = where + " " + when;
    require(bytesOf(fixture.hidden) == hidden, label + ": hidden rows differ from the fp32 decode's");
    require(bytesOf(fixture.tape) == tape, label + ": tape differs from the fp32 decode's");
    if (tables)
      require(bytesOf(tables->table) == table && bytesOf(tables->sums) == sums,
              label + ": out-projection table differs from the fp32 decode's");
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(!std::memcmp(next[lane].contents(), fixture.next[lane].contents(), narrow.convBytes),
              label + ": convolution rows differ from the fp32 decode's");
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        const float *values = wideState(fixture.next[lane], layer);
        const uint16_t *stored = narrowState(next[lane], layer);
        for (uint64_t index = 0; index < elements; ++index)
          require(stored[index] == floatToBf16(values[index]),
                  label + ": the state is not the fp32 decode's rounded to nearest even");
      }
    }
  };
  roundStates();
  step("first step");
  // As in runDecode, lane l retains 1 + (base + l - 1) % 8 rows of the first.
  std::array<uint32_t, kMaxLanes> retained{};
  fixture.advance(retained);
  std::swap(current, next);
  fixture.fillPacked();
  roundStates();
  for (uint32_t base = 1; base <= kRows; ++base) {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      fixture.blocks[lane].tape.pendingPath = chainPath(1 + (base + lane - 1) % kRows);
    step("second step " + std::to_string(base));
  }
  std::cout << where << ": the fp32 decode's rows and its state rounded\n";
}

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: gdn-decode METALLIB");
    MetalBackend backend(argv[1]);
    rejectsInvalid(backend);
    for (const GdnShape &shape : kShapes)
      for (const uint32_t lanes : {1U, kMaxLanes})
        for (const bool float32 : familyNorms(shape))
          bufferExtents(backend, shape, LinearInput::Table16, float32, lanes);
    // Table16 feeds the register tile after norms that are F32 (a GGUF's, in
    // tiled head order) or bf16 (an MLX target's, grouped), each norm type
    // the shape's families store.
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes) {
        for (const bool float32 : familyNorms(shape))
          fusedPreparation(backend, shape, lanes, LinearInput::Table16, float32);
        tiledHeadOrder(backend, shape, lanes, LinearInput::Table16, true);
      }
    // A GGUF's out-projection on the staged tile reads the tiled rows plain.
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        tiledHeadOrder(backend, shape, lanes, LinearInput::Plain, true);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape))
        for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
          runDecode(backend, shape, lanes, float32);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape))
        for (GdnHeadOrder order : {GdnHeadOrder::Grouped, GdnHeadOrder::Tiled})
          gateMatchesPrefill(backend, shape, float32, order);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape))
        for (GdnHeadOrder order : {GdnHeadOrder::Grouped, GdnHeadOrder::Tiled})
          for (LinearInput layout : {LinearInput::Plain, LinearInput::Table16})
            replayAcrossBatches(backend, shape, float32, order, layout);
    for (const GdnShape &shape : kShapes)
      for (const uint32_t lanes : {1U, kMaxLanes})
        for (bool float32 : familyNorms(shape))
          for (LinearInput layout : {LinearInput::Plain, LinearInput::Table16})
            bfloatState(backend, shape, lanes, float32, layout);
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        for (bool float32 : familyNorms(shape))
          for (LinearInput layout : {LinearInput::Plain, LinearInput::Table16})
            for (GdnStateFormat format : {GdnStateFormat::Float32, GdnStateFormat::BFloat16})
              partialVerifyRows(backend, shape, lanes, layout, float32,
                                float32 ? GdnHeadOrder::Tiled : GdnHeadOrder::Grouped, format);
    for (const GdnShape &shape : kShapes)
      for (const GdnStateFormat format : {GdnStateFormat::Float32, GdnStateFormat::BFloat16})
        foldMatchesTheNextStep(backend, shape, format);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape))
        for (GdnStateFormat format : {GdnStateFormat::Float32, GdnStateFormat::BFloat16})
          raggedTrees(backend, shape, float32, format);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape))
        for (LinearInput layout : {LinearInput::Plain, LinearInput::Table16})
          for (GdnStateFormat format : {GdnStateFormat::Float32, GdnStateFormat::BFloat16})
            raggedBlocks(backend, shape, float32, float32 ? GdnHeadOrder::Tiled : GdnHeadOrder::Grouped, layout,
                         format);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape))
        for (GdnStateFormat format : {GdnStateFormat::Float32, GdnStateFormat::BFloat16})
          for (const uint32_t blockRows : {SPLASH_GDN_NARROW_ROWS, kRows})
            treeRows(backend, shape, float32, format, blockRows);
    for (const GdnShape &shape : kShapes)
      for (const GdnStateFormat format : {GdnStateFormat::Float32, GdnStateFormat::BFloat16})
        narrowKernelsFoldPathsPastTheirRows(backend, shape, format);
    for (const GdnShape &shape : kShapes)
      for (bool float32 : familyNorms(shape)) chainHalves(backend, shape, float32);
    std::cout << "gdn_decode_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gdn_decode_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
