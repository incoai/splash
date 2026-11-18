// The Mamba-2 mixer (ops::Mamba2, kernels/shared/mamba2.metal) against an
// fp64 reference written from llama.cpp's build_mamba2_layer and its CPU
// ssm_scan, on the compiled geometry of 8 groups and 128 heads of 64 rows
// (Nemotron 3.5 Bolt's), with random weights, packed rows, states and
// carries: two lanes on layer 1 of a two-layer cell. A first decode step folds
// no pending rows; a second folds 0, 3 or 8 of the first's rows from its tape.
// Checked: the gated rows y, the state and convolution carry the step
// commits, the tape, and the grouped RMS norm. The reference without the D
// skip, and with the norm before the gate, must disagree with the kernels, so
// the checks can fail. A prefill step of a lane's rows from its own state
// views computes the decode's bits.
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/GDN.h"
#include "metal/abi/Mamba2.h"
#include "ops/Mamba2.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace splash;
using metal::CommandGraph;
using metal::MetalBackend;
using metal::MetalBuffer;
using test::bf16ToFloat;
using test::floatToBf16;

constexpr uint32_t kGroups = 8, kHeads = 128, kP = 64, kN = 128;
constexpr ops::Mamba2Shape kShape{kGroups, kHeads, kP, kN};
constexpr uint32_t kXWidth = kHeads * kP, kConv = kXWidth + 2 * kGroups * kN;
constexpr uint32_t kPacked = 2 * kXWidth + 2 * kGroups * kN + kHeads, kXbc = kXWidth, kDt = kPacked - kHeads;
constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS, kTaps = SPLASH_GDN_CONVOLUTION_TAPS, kCarried = kTaps - 1;
constexpr uint32_t kLanes = 2, kLayers = 2, kLayer = 1, kBxWidth = kGroups * kN + kXWidth;
constexpr double kEpsilon = 1e-5;
const uint64_t kTapeBytes = mamba2_tape_layer_bytes(kConv, kBxWidth, kHeads);
// A cell of two layers: the convolution carries, then the recurrent states.
constexpr uint64_t kConvLayerBytes = uint64_t{kCarried} * kConv * 2;
constexpr uint64_t kRecurrentLayerBytes = uint64_t{kHeads} * kP * kN * 4;
constexpr uint64_t kConvBytes = kLayers * kConvLayerBytes, kCellBytes = kConvBytes + kLayers * kRecurrentLayerBytes;
constexpr ops::GdnStateStrides kStrides{kConvLayerBytes, kRecurrentLayerBytes, kConvBytes, ops::GdnStateFormat::Float32};

int failures = 0;
void check(bool ok, const std::string &what) {
  if (!ok && failures++ < 12) std::fprintf(stderr, "FAIL %s\n", what.c_str());
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

struct Weights {
  std::vector<float> conv, bias, a, dtBias, d, norm;
};

// Weights of the layer's ranges: taps and bias of the convolution, A < 0,
// a dt bias, D and norm weights about 1.
Weights randomWeights(Random &random) {
  Weights w{std::vector<float>(kConv * kTaps), std::vector<float>(kConv), std::vector<float>(kHeads),
            std::vector<float>(kHeads), std::vector<float>(kHeads), std::vector<float>(kXWidth)};
  for (float &v : w.conv) v = random.unit() * 0.6F;
  for (float &v : w.bias) v = random.unit() * 0.2F;
  for (float &v : w.a) v = -std::exp(random.unit() * 2.0F);
  for (float &v : w.dtBias) v = random.unit() * 2.0F;
  for (float &v : w.d) v = random.unit();
  for (float &v : w.norm) v = 1.0F + 0.3F * random.unit();
  return w;
}

double silu(double v) { return v / (1.0 + std::exp(-v)); }
double softplus(double v) { return v > 20.0 ? v : std::log1p(std::exp(v)); }

// One lane's conv input rows in order (the carried rows, then each step's
// x B C), as doubles of their bf16 values, and its state.
struct LaneReference {
  std::vector<std::vector<double>> inputs;
  std::vector<double> state; // [head][row][column]
};

// What a step computes for one lane: its rows' x, B, C, softplus(dt), decay
// and gated outputs y, from the lane's inputs before the step.
struct StepReference {
  std::vector<double> x, b, c, dt, decay, y, yNoSkip;
};

StepReference referenceStep(const Weights &w, const LaneReference &lane, const uint16_t *packed) {
  StepReference step;
  step.x.resize(kRows * kXWidth);
  step.b.resize(kRows * kGroups * kN);
  step.c.resize(kRows * kGroups * kN);
  step.dt.resize(kRows * kHeads);
  step.decay.resize(kRows * kHeads);
  step.y.resize(kRows * kXWidth);
  step.yNoSkip.resize(kRows * kXWidth);
  const size_t before = lane.inputs.size();
  std::vector<double> state = lane.state;
  for (uint32_t t = 0; t < kRows; ++t) {
    const uint16_t *row = packed + t * kPacked;
    std::vector<double> xbc(kConv);
    for (uint32_t ch = 0; ch < kConv; ++ch) {
      double sum = 0.0;
      for (uint32_t tap = 0; tap < kTaps; ++tap) {
        const size_t position = before - kCarried + t + tap; // in lane.inputs then this step's rows
        const double in = position < before ? lane.inputs[position][ch]
                                            : double(bf16ToFloat(packed[(position - before) * kPacked + kXbc + ch]));
        sum += in * double(w.conv[ch * kTaps + tap]);
      }
      xbc[ch] = silu(sum + double(w.bias[ch]));
    }
    for (uint32_t i = 0; i < kXWidth; ++i) step.x[t * kXWidth + i] = xbc[i];
    for (uint32_t i = 0; i < kGroups * kN; ++i) {
      step.b[t * kGroups * kN + i] = xbc[kXWidth + i];
      step.c[t * kGroups * kN + i] = xbc[kXWidth + kGroups * kN + i];
    }
    for (uint32_t h = 0; h < kHeads; ++h) {
      const double dt = softplus(double(bf16ToFloat(row[kDt + h])) + double(w.dtBias[h]));
      const double decay = std::exp(dt * double(w.a[h]));
      step.dt[t * kHeads + h] = dt;
      step.decay[t * kHeads + h] = decay;
      const uint32_t g = h / (kHeads / kGroups);
      for (uint32_t p = 0; p < kP; ++p) {
        const double xdt = step.x[t * kXWidth + h * kP + p] * dt;
        double sum = 0.0;
        for (uint32_t n = 0; n < kN; ++n) {
          double &s = state[(size_t(h) * kP + p) * kN + n];
          s = s * decay + step.b[t * kGroups * kN + g * kN + n] * xdt;
          sum += s * step.c[t * kGroups * kN + g * kN + n];
        }
        const double z = double(bf16ToFloat(row[h * kP + p]));
        step.y[t * kXWidth + h * kP + p] = (sum + step.x[t * kXWidth + h * kP + p] * double(w.d[h])) * silu(z);
        step.yNoSkip[t * kXWidth + h * kP + p] = sum * silu(z);
      }
    }
  }
  return step;
}

// The lane after `retained` of the step's rows.
void advance(const Weights &w, LaneReference &lane, const uint16_t *packed, const StepReference &step,
             uint32_t retained) {
  for (uint32_t t = 0; t < retained; ++t) {
    std::vector<double> row(kConv);
    for (uint32_t ch = 0; ch < kConv; ++ch) row[ch] = bf16ToFloat(packed[t * kPacked + kXbc + ch]);
    lane.inputs.push_back(row);
    for (uint32_t h = 0; h < kHeads; ++h) {
      const uint32_t g = h / (kHeads / kGroups);
      const double dt = step.dt[t * kHeads + h], decay = step.decay[t * kHeads + h];
      for (uint32_t p = 0; p < kP; ++p) {
        const double xdt = step.x[t * kXWidth + h * kP + p] * dt;
        for (uint32_t n = 0; n < kN; ++n) {
          double &s = lane.state[(size_t(h) * kP + p) * kN + n];
          s = s * decay + step.b[t * kGroups * kN + g * kN + n] * xdt;
        }
      }
    }
  }
  (void)w;
}

bool close(double got, double ref, double tolerance) { return std::fabs(got - ref) <= tolerance * (1.0 + std::fabs(ref)); }

struct Fixture {
  MetalBackend &backend;
  Weights weights;
  MetalBuffer convW, convB, a, dtBias, d, norm, packed, tape, y, hidden;
  std::vector<MetalBuffer> current, next;

  Fixture(MetalBackend &b, Weights w) : backend(b), weights(std::move(w)) {
    const auto upload = [&](const std::vector<float> &v) {
      MetalBuffer buffer = test::sharedBuffer(backend, v.size() * 4);
      std::memcpy(buffer.contents(), v.data(), v.size() * 4);
      return buffer;
    };
    convW = upload(weights.conv), convB = upload(weights.bias), a = upload(weights.a);
    dtBias = upload(weights.dtBias), d = upload(weights.d), norm = upload(weights.norm);
    packed = test::sharedBuffer(backend, uint64_t{kLanes} * kRows * kPacked * 2);
    tape = test::sharedBuffer(backend, 5 * kLayers * kTapeBytes);
    y = test::sharedBuffer(backend, uint64_t{kLanes} * kRows * kXWidth * 4);
    hidden = test::sharedBuffer(backend, uint64_t{kLanes} * kRows * kXWidth * 2);
    for (uint32_t l = 0; l < kLanes; ++l) {
      current.push_back(test::sharedBuffer(backend, kCellBytes));
      next.push_back(test::sharedBuffer(backend, kCellBytes));
    }
  }
  uint16_t *conv(const MetalBuffer &cell) const {
    return reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(cell.contents()) + kLayer * kConvLayerBytes);
  }
  float *state(const MetalBuffer &cell) const {
    return reinterpret_cast<float *>(static_cast<uint8_t *>(cell.contents()) + kConvBytes +
                                     kLayer * kRecurrentLayerBytes);
  }
  uint16_t *packedRows(uint32_t lane) const {
    return static_cast<uint16_t *>(packed.contents()) + uint64_t{laneRows[lane].first} * kPacked;
  }
  uint8_t *tapeAt(uint64_t offset) const { return static_cast<uint8_t *>(tape.contents()) + offset; }
  // Slot s of the tape, which holds a step of both layers.
  static uint64_t slot(uint32_t s) { return uint64_t{s} * kLayers * kTapeBytes; }
  static uint64_t layerTape(uint32_t s) { return slot(s) + kLayer * kTapeBytes; }
  ops::Mamba2Weights mamba2Weights() const { return {convW, convB, a, dtBias, d, norm}; }

  // Lane l's rows of the step, packed from row first (default: a verify
  // block's rows at l x kRows).
  std::array<SplashLaneRows, kLanes> laneRows{{{0, kRows}, {kRows, kRows}}};

  void run(const uint32_t pendingSlots[], const uint32_t stepSlots[], const uint32_t pendingRows[]) {
    std::array<ops::GdnTapeLane, kLanes> tapes{};
    // A chain's path: the previous step's first pendingRows rows.
    for (uint32_t l = 0; l < kLanes; ++l)
      tapes[l] = {slot(pendingSlots[l]), slot(stepSlots[l]), (1u << pendingRows[l]) - 1u};
    const MetalBuffer laneRows = test::sharedBuffer(backend, kLanes * sizeof(SplashLaneRows));
    std::memcpy(laneRows.contents(), this->laneRows.data(), sizeof(this->laneRows));
    const auto lanes = [](const std::vector<MetalBuffer> &v) {
      return std::vector<MetalBuffer>{v[0], v[1], v[0], v[0]};
    };
    const std::vector<MetalBuffer> currentLanes = lanes(current), nextLanes = lanes(next);
    CommandGraph graph;
    uint32_t rows = 0;
    for (const SplashLaneRows &lane : this->laneRows) rows = std::max(rows, lane.first + lane.rows);
    ops::Mamba2::addDecode(graph, packed, mamba2Weights(), currentLanes, nextLanes, tape, tapes, laneRows,
                           std::max(rows, kLanes), y, kShape, kLayer, kStrides);
    ops::Mamba2::addGroupNorm(graph, y, norm, hidden, kEpsilon, kShape, kLanes * kRows);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
  }
};

void fillPacked(Fixture &f, Random &random) {
  for (uint32_t l = 0; l < kLanes; ++l) {
    uint16_t *rows = f.packedRows(l);
    for (uint32_t t = 0; t < kRows; ++t)
      for (uint32_t c = 0; c < kPacked; ++c) {
        const float scale = c < kXWidth ? 2.0F : c < kDt ? 1.5F : 3.0F; // z, x B C, dt
        rows[t * kPacked + c] = floatToBf16(random.unit() * scale);
      }
  }
}

// Checks one lane's step against the reference: y and the grouped norm, the
// committed state and carry (`committed`, the lane after the pending rows),
// and the step's tape.
void checkStep(Fixture &f, uint32_t lane, const LaneReference &committed, const StepReference &step,
               uint32_t stepSlot, const std::string &name) {
  const uint64_t stepTape = Fixture::layerTape(stepSlot);
  const uint32_t first = f.laneRows[lane].first, rows = f.laneRows[lane].rows;
  const float *y = static_cast<const float *>(f.y.contents()) + uint64_t{first} * kXWidth;
  const uint16_t *hidden = static_cast<const uint16_t *>(f.hidden.contents()) + uint64_t{first} * kXWidth;
  double worst = 0.0;
  uint32_t skipAgrees = 0, normFirstAgrees = 0;
  for (uint32_t i = 0; i < rows * kXWidth; ++i) {
    worst = std::max(worst, std::fabs(y[i] - step.y[i]) / (1.0 + std::fabs(step.y[i])));
    check(close(y[i], step.y[i], 2e-4), name + " y[" + std::to_string(i) + "] " + std::to_string(y[i]) + " vs " +
                                            std::to_string(step.y[i]));
    skipAgrees += close(y[i], step.yNoSkip[i], 2e-4) && !close(step.y[i], step.yNoSkip[i], 2e-4);
  }
  // The grouped norm of the kernel's own y, and of y with the norm before the gate.
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t g = 0; g < kGroups; ++g) {
      const uint32_t base = t * kXWidth + g * 1024;
      double squares = 0.0, ungatedSquares = 0.0;
      for (uint32_t i = 0; i < 1024; ++i) squares += double(y[base + i]) * y[base + i];
      const double inverse = 1.0 / std::sqrt(squares / 1024 + kEpsilon);
      std::vector<double> ungated(1024);
      for (uint32_t i = 0; i < 1024; ++i) {
        const uint32_t column = g * 1024 + i;
        const double z = bf16ToFloat(f.packedRows(lane)[t * kPacked + column]);
        ungated[i] = step.y[base + i] / silu(z);
        ungatedSquares += ungated[i] * ungated[i];
      }
      const double ungatedInverse = 1.0 / std::sqrt(ungatedSquares / 1024 + kEpsilon);
      for (uint32_t i = 0; i < 1024; ++i) {
        const double want = double(y[base + i]) * inverse * f.weights.norm[g * 1024 + i];
        const double got = bf16ToFloat(hidden[base + i]);
        check(std::fabs(got - want) <= std::ldexp(std::fabs(want), -8) + 1e-30,
              name + " norm[" + std::to_string(base + i) + "]");
        const double z = bf16ToFloat(f.packedRows(lane)[t * kPacked + g * 1024 + i]);
        const double normFirst = ungated[i] * ungatedInverse * f.weights.norm[g * 1024 + i] * silu(z);
        normFirstAgrees += std::fabs(got - normFirst) <= std::ldexp(std::fabs(normFirst), -8);
      }
    }
  check(skipAgrees < rows * kXWidth / 100, name + " the kernel's y also matches the reference without the D skip");
  check(normFirstAgrees < rows * kXWidth / 2, name + " the norm also matches with the norm before the gate");
  // The committed state and carry.
  const float *state = f.state(f.next[lane]);
  for (size_t i = 0; i < committed.state.size(); ++i)
    check(close(state[i], committed.state[i], 2e-5), name + " state[" + std::to_string(i) + "]");
  const uint16_t *carry = f.conv(f.next[lane]);
  for (uint32_t r = 0; r < kCarried; ++r)
    for (uint32_t ch = 0; ch < kConv; ++ch)
      check(bf16ToFloat(carry[r * kConv + ch]) == committed.inputs[committed.inputs.size() - kCarried + r][ch],
            name + " carry[" + std::to_string(r) + "][" + std::to_string(ch) + "]");
  // The tape: inputs as stored, B and x, decay and dt.
  const uint16_t *inputs = reinterpret_cast<const uint16_t *>(f.tapeAt(stepTape));
  const float *bx = reinterpret_cast<const float *>(f.tapeAt(stepTape + mamba2_tape_bx_offset(kConv)));
  const float *decay = reinterpret_cast<const float *>(f.tapeAt(stepTape + mamba2_tape_decay_offset(kConv, kBxWidth)));
  const float *dt = reinterpret_cast<const float *>(f.tapeAt(stepTape + mamba2_tape_dt_offset(kConv, kBxWidth, kHeads)));
  for (uint32_t t = 0; t < rows; ++t) {
    for (uint32_t ch = 0; ch < kConv; ++ch)
      check(inputs[t * kConv + ch] == f.packedRows(lane)[t * kPacked + kXbc + ch], name + " tape input");
    for (uint32_t i = 0; i < kGroups * kN; ++i)
      check(close(bx[t * kBxWidth + i], step.b[t * kGroups * kN + i], 1e-5), name + " tape B");
    for (uint32_t i = 0; i < kXWidth; ++i)
      check(close(bx[t * kBxWidth + kGroups * kN + i], step.x[t * kXWidth + i], 1e-5), name + " tape x");
    for (uint32_t h = 0; h < kHeads; ++h) {
      check(close(decay[t * kHeads + h], step.decay[t * kHeads + h], 1e-5), name + " tape decay");
      check(close(dt[t * kHeads + h], step.dt[t * kHeads + h], 1e-5), name + " tape dt");
    }
  }
  std::printf("%s lane %u: y within %.2e of the reference\n", name.c_str(), lane, worst);
}

// Lane `lane`'s first step again as a prefill step from views of its cells'
// layer: the same state, carry, tape and rows as the decode's.
void checkPrefillStep(Fixture &f, uint32_t lane) {
  const auto view = [&](const MetalBuffer &cell, uint64_t offset, uint64_t bytes) {
    return f.backend.view(cell, offset, bytes);
  };
  const MetalBuffer convOut = test::sharedBuffer(f.backend, kConvLayerBytes);
  const MetalBuffer stateOut = test::sharedBuffer(f.backend, kRecurrentLayerBytes);
  const MetalBuffer y = test::sharedBuffer(f.backend, uint64_t{kRows} * kXWidth * 4);
  const uint64_t stepTape = Fixture::layerTape(4);
  CommandGraph graph;
  ops::Mamba2::addPrefillStep(
      graph, view(f.packed, uint64_t{lane} * kRows * kPacked * 2, uint64_t{kRows} * kPacked * 2), f.mamba2Weights(),
      {view(f.current[lane], kLayer * kConvLayerBytes, kConvLayerBytes),
       view(f.current[lane], kConvBytes + kLayer * kRecurrentLayerBytes, kRecurrentLayerBytes), convOut, stateOut},
      f.tape, Fixture::layerTape(0), stepTape, 0, y, kShape);
  (void)f.backend.submitCommandAsync(graph.dispatches()).wait();
  check(!std::memcmp(y.contents(), static_cast<const float *>(f.y.contents()) + lane * kRows * kXWidth, y.sizeBytes()),
        "the prefill step's rows differ from the decode's");
  check(!std::memcmp(stateOut.contents(), f.state(f.next[lane]), kRecurrentLayerBytes) &&
            !std::memcmp(convOut.contents(), f.conv(f.next[lane]), kConvLayerBytes),
        "the prefill step's state differs from the decode's");
  check(!std::memcmp(f.tapeAt(stepTape), f.tapeAt(Fixture::layerTape(1 + lane)), kTapeBytes),
        "the prefill step's tape differs from the decode's");
  std::printf("prefill step: lane %u's decode bit for bit\n", lane);
}

// A geometry the kernels do not compile is refused.
void checkRefusals(MetalBackend &backend) {
  const MetalBuffer buffer = test::sharedBuffer(backend, 1 << 20);
  CommandGraph graph;
  test::rejects([&] { ops::Mamba2::addGroupNorm(graph, buffer, buffer, buffer, kEpsilon, {8, 64, 64, 128}, 1); },
                "no Mamba-2 kernel for the geometry", "a geometry without kernels was accepted");
  check(graph.empty(), "a refused Mamba-2 dispatch was encoded");
}

// Two steps of both lanes from random states: the first with no pending
// rows, the second folding `retained` rows (lane 1 one fewer) of the first's;
// `shortLanes` packs the second step's lanes as a decode step of few draft
// rows does: 5 rows from row 0 and 3 from row 5.
void runCase(MetalBackend &backend, const Weights &weights, uint32_t retained, uint64_t seed,
             bool shortLanes = false) {
  Fixture f(backend, weights);
  Random random(seed);
  std::vector<LaneReference> lanes(kLanes);
  for (uint32_t l = 0; l < kLanes; ++l) {
    uint16_t *carry = f.conv(f.current[l]);
    lanes[l].inputs.assign(kCarried, std::vector<double>(kConv));
    for (uint32_t r = 0; r < kCarried; ++r)
      for (uint32_t ch = 0; ch < kConv; ++ch) {
        carry[r * kConv + ch] = floatToBf16(random.unit() * 1.5F);
        lanes[l].inputs[r][ch] = bf16ToFloat(carry[r * kConv + ch]);
      }
    float *state = f.state(f.current[l]);
    lanes[l].state.resize(size_t(kHeads) * kP * kN);
    for (size_t i = 0; i < lanes[l].state.size(); ++i) lanes[l].state[i] = state[i] = random.unit() * 0.5F;
  }
  // First step: no pending rows, tape slots 0 (pending, unread) and 1 + lane.
  fillPacked(f, random);
  const uint32_t none[kLanes] = {0, 0};
  const uint32_t first[kLanes] = {1, 2};
  const uint32_t zero[kLanes] = {0, 0};
  std::vector<StepReference> firstSteps;
  std::vector<std::vector<uint16_t>> firstPacked;
  for (uint32_t l = 0; l < kLanes; ++l) {
    firstSteps.push_back(referenceStep(weights, lanes[l], f.packedRows(l)));
    firstPacked.emplace_back(f.packedRows(l), f.packedRows(l) + kRows * kPacked);
  }
  f.run(none, first, zero);
  for (uint32_t l = 0; l < kLanes; ++l)
    checkStep(f, l, lanes[l], firstSteps[l], first[l], "first step (retained " + std::to_string(retained) + ")");
  checkPrefillStep(f, 1);
  // Second step: the first's next cells are current; fold `retained` rows
  // (lane 1 one fewer, so lanes differ) from the first's tape.
  std::swap(f.current, f.next);
  if (shortLanes) f.laneRows = {{{0, 5}, {5, 3}}};
  const uint32_t pending[kLanes] = {retained, retained ? retained - 1 : 0};
  for (uint32_t l = 0; l < kLanes; ++l) advance(weights, lanes[l], firstPacked[l].data(), firstSteps[l], pending[l]);
  fillPacked(f, random);
  std::vector<StepReference> secondSteps;
  for (uint32_t l = 0; l < kLanes; ++l) secondSteps.push_back(referenceStep(weights, lanes[l], f.packedRows(l)));
  const uint32_t second[kLanes] = {3, 0};
  f.run(first, second, pending);
  for (uint32_t l = 0; l < kLanes; ++l)
    checkStep(f, l, lanes[l], secondSteps[l], second[l],
              std::string(shortLanes ? "short lanes, " : "") + "second step, " + std::to_string(pending[l]) +
                  " pending rows");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: mamba2 METALLIB");
    Random random(7);
    const Weights weights = randomWeights(random);
    MetalBackend backend(argv[1]);
    checkRefusals(backend);
    for (const uint32_t retained : {0U, 3U, 8U, kRows}) runCase(backend, weights, retained, 1000 + retained);
    runCase(backend, weights, 3, 2003, true);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  if (failures) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("Mamba-2: PASS\n");
}
