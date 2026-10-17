// GDN prefill against a CPU reference through the operator's own dispatch
// geometry: the convolution/normalisation prepare pass, the blocked recurrent
// scan (output rows and the fp32 state it hands to decode), and the gated
// output norm, for both compiled head layouts and ragged token counts that
// leave partial scan blocks and single-token chunks. The scan over a bf16
// state is checked against the fp32 scan, and a prefill that captures states
// inside its rows against the same rows split there.
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/GDN.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"

#include "LinearNumerics.hpp"
#include "NormReference.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::GDN;
using splash::ops::GdnHeadOrder;
using splash::ops::GdnPrefillBuffers;
using splash::ops::GdnPrefillCapture;
using splash::ops::GdnShape;
using splash::ops::GdnStateFormat;
using splash::ops::NormWeights;

constexpr uint32_t kHeadDim = 128;
// The convolution's taps; a state carries the inputs of all but the current
// token's.
constexpr uint32_t kTaps = SPLASH_GDN_CONVOLUTION_TAPS, kCarried = kTaps - 1;
constexpr double kQueryScale = 0.0078125;
constexpr double kKeyScale = 0.08838834765;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }
  float unit() { return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F; }
  float gauss() {
    float sum = 0.0F;
    for (int i = 0; i < 4; ++i)
      sum += unit();
    return sum * 0.8660254F;
  }

private:
  uint64_t state_;
};

uint16_t toBf16(double value) {
  return splash::test::floatToBf16(static_cast<float>(value));
}
double fromBf16(uint16_t value) { return splash::test::bf16ToFloat(value); }
double roundBf16(double value) { return fromBf16(toBf16(value)); }
double bf16Ulp(double value) {
  return std::max(std::fabs(value), 1e-30) * 0.0078125;
}
double silu(double value) { return value / (1.0 + std::exp(-value)); }

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}
template <class T> T *data(const MetalBuffer &buffer) {
  return static_cast<T *>(buffer.contents());
}

struct Worst final {
  double error = 0.0;
  double reference = 0.0;
  void add(double got, double ref) {
    // NaN compares false against every tolerance, so reject it here.
    require(std::isfinite(got), "GPU output is not finite");
    const double e = std::fabs(got - ref);
    if (e > error) {
      error = e;
      reference = ref;
    }
  }
};

// Per-token outputs of a prefill over `tokens` rows; the caller supplies the
// packed rows, the convolution carry and the recurrent state it starts from.
GdnPrefillBuffers prefillBuffers(MetalBackend &backend, const GdnShape &shape,
                                 uint32_t tokens, MetalBuffer packed,
                                 MetalBuffer convolutionIn,
                                 MetalBuffer recurrentIn,
                                 MetalBuffer convolutionWeights,
                                 MetalBuffer decayWeights,
                                 MetalBuffer timeBias, NormWeights mixerNorm) {
  const uint32_t convDim = shape.convolutionDimension;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  const uint32_t valueWidth = shape.valueHeads * kHeadDim;
  const uint64_t stateBytes =
      uint64_t{shape.valueHeads} * kHeadDim * kHeadDim * 4;
  return {std::move(packed),
          std::move(convolutionWeights),
          std::move(convolutionIn),
          shared(backend, uint64_t{kCarried} * convDim * 2, "conv out"),
          shared(backend, uint64_t{tokens} * keyWidth * 2, "queries"),
          shared(backend, uint64_t{tokens} * keyWidth * 2, "keys"),
          shared(backend, uint64_t{tokens} * valueWidth * 2, "values"),
          std::move(decayWeights),
          std::move(timeBias),
          shared(backend, uint64_t{tokens} * shape.valueHeads * 4, "decay"),
          shared(backend, uint64_t{tokens} * shape.valueHeads * 2, "beta"),
          std::move(recurrentIn),
          shared(backend, stateBytes, "state out"),
          shared(backend, uint64_t{tokens} * valueWidth * 2, "recurrent rows"),
          std::move(mixerNorm),
          shared(backend, uint64_t{tokens} * valueWidth * 2, "hidden")};
}

// The mixer norm is bf16, or F32 as a GGUF stores it.
GdnPrefillBuffers randomPrefill(MetalBackend &backend, const GdnShape &shape,
                                uint32_t tokens, bool float32 = false) {
  const uint32_t valueHeads = shape.valueHeads;
  const uint32_t convDim = shape.convolutionDimension;
  const uint32_t packedWidth = shape.packedWidth;
  const uint64_t stateElements = uint64_t{valueHeads} * kHeadDim * kHeadDim;
  GdnPrefillBuffers buffers = prefillBuffers(
      backend, shape, tokens,
      shared(backend, uint64_t{tokens} * packedWidth * 2, "packed"),
      shared(backend, uint64_t{kCarried} * convDim * 2, "conv in"),
      shared(backend, stateElements * 4, "state in"),
      shared(backend, uint64_t{convDim} * kTaps * 2, "conv weights"),
      shared(backend, valueHeads * 4, "a scale"),
      shared(backend, valueHeads * 2, "dt bias"), {});

  Random random(0x9E3779B97F4A7C15ULL ^ (uint64_t{valueHeads} << 32) ^ tokens);
  auto *packed = data<uint16_t>(buffers.packed);
  // Qwen3.8-Flash-Next's 16512-column rows hold the first 16512 values of the
  // 16640-column rows, so its _p16512 kernels see the same values, and the
  // same bounds hold for them.
  const uint32_t drawn = packedWidth == 16512 ? 16640 : packedWidth;
  for (uint64_t token = 0; token < tokens; ++token)
    for (uint32_t column = 0; column < drawn; ++column) {
      const uint16_t value = toBf16(random.gauss());
      if (column < packedWidth) packed[token * packedWidth + column] = value;
    }
  auto *convWeights = data<uint16_t>(buffers.convolutionWeights);
  for (uint64_t i = 0; i < uint64_t{convDim} * kTaps; ++i)
    convWeights[i] = toBf16(0.3 * random.gauss());
  auto *convIn = data<uint16_t>(buffers.convolutionIn);
  for (uint64_t i = 0; i < uint64_t{kCarried} * convDim; ++i)
    convIn[i] = toBf16(random.gauss());
  auto *aScale = data<float>(buffers.decayWeights);
  auto *dtBias = data<uint16_t>(buffers.timeBias);
  for (uint32_t head = 0; head < valueHeads; ++head) {
    // Mostly gentle decays with one head at the reference's strongest
    // a_scale, whose decay underflows to zero for most tokens.
    const float unit = 0.5F * (random.unit() + 1.0F);
    aScale[head] = head % 11 == 7 ? -105.0F : -(0.05F + 8.0F * unit * unit);
    dtBias[head] = toBf16(random.gauss());
  }
  auto *stateIn = data<float>(buffers.recurrentIn);
  for (uint64_t i = 0; i < stateElements; ++i)
    stateIn[i] = 0.05F * random.gauss();
  buffers.mixerNorm = splash::test::makeNormWeights(backend, kHeadDim, float32, [&](uint32_t) {
    return static_cast<float>(1.0 + 0.2 * random.gauss());
  });
  return buffers;
}

void submitPrefill(MetalBackend &backend, const GdnPrefillBuffers &buffers,
                   const GdnShape &shape, uint32_t tokens,
                   const std::string &label,
                   GdnHeadOrder order = GdnHeadOrder::Grouped,
                   GdnStateFormat state = GdnStateFormat::Float32) {
  CommandGraph graph;
  GDN::addPrefill(graph, backend, buffers, shape, tokens, state, order);
  require(graph.dispatches().size() == 3,
          label + "prefill is not three dispatches");
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
}

void runCase(MetalBackend &backend, const GdnShape &shape, uint32_t tokens,
             bool float32 = false) {
  const uint32_t keyHeads = shape.keyHeads, valueHeads = shape.valueHeads;
  const uint32_t convDim = shape.convolutionDimension;
  const uint32_t packedWidth = shape.packedWidth;
  const uint32_t keyWidth = keyHeads * kHeadDim;
  const uint32_t valueWidth = valueHeads * kHeadDim;
  const uint32_t headsPerKey = valueHeads / keyHeads;
  const uint32_t bOffset = convDim + valueWidth, aOffset = bOffset + valueHeads;
  const std::string label = "vh" + std::to_string(valueHeads) + " tokens=" +
                            std::to_string(tokens) +
                            (float32 ? " f32 norm: " : ": ");

  GdnPrefillBuffers buffers = randomPrefill(backend, shape, tokens, float32);
  const auto *packed = data<uint16_t>(buffers.packed);
  const auto *convWeights = data<uint16_t>(buffers.convolutionWeights);
  const auto *convIn = data<uint16_t>(buffers.convolutionIn);
  const auto *aScale = data<float>(buffers.decayWeights);
  const auto *dtBias = data<uint16_t>(buffers.timeBias);
  const auto *stateIn = data<float>(buffers.recurrentIn);
  submitPrefill(backend, buffers, shape, tokens, label);

  // Prepare pass: convolution + SiLU per channel, q/k RMS normalisation.
  auto convolved = [&](uint32_t token, uint32_t channel) {
    double value = 0.0;
    for (uint32_t tap = 0; tap < kTaps; ++tap) {
      const uint32_t position = token + tap;
      const double input =
          position < kCarried
              ? fromBf16(convIn[position * convDim + channel])
              : fromBf16(packed[uint64_t{position - kCarried} * packedWidth +
                                channel]);
      value += input * fromBf16(convWeights[channel * kTaps + tap]);
    }
    return roundBf16(silu(roundBf16(value)));
  };
  const auto *queries = data<uint16_t>(buffers.queries);
  const auto *keys = data<uint16_t>(buffers.keys);
  const auto *values = data<uint16_t>(buffers.values);
  Worst worstPrepare;
  double prepareUlps = 0.0;
  auto checkPrepare = [&](double got, double ref) {
    worstPrepare.add(got, ref);
    prepareUlps = std::max(prepareUlps, std::fabs(got - ref) / bf16Ulp(ref));
  };
  std::vector<double> row(kHeadDim);
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t head = 0; head < keyHeads; ++head) {
      for (uint32_t which = 0; which < 2; ++which) {
        double squares = 0.0;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
          row[dim] = convolved(token, which * keyWidth + head * kHeadDim + dim);
          squares += row[dim] * row[dim];
        }
        const double inverse =
            1.0 / std::sqrt(squares / kHeadDim + SPLASH_RMS_EPSILON);
        const double scale = which == 0 ? kQueryScale : kKeyScale;
        const uint16_t *out = (which == 0 ? queries : keys) +
                              uint64_t{token} * keyWidth + head * kHeadDim;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim)
          checkPrepare(fromBf16(out[dim]),
                       roundBf16(roundBf16(row[dim] * inverse) * scale));
      }
    }
    for (uint32_t channel = 0; channel < valueWidth; ++channel)
      checkPrepare(fromBf16(values[uint64_t{token} * valueWidth + channel]),
                   convolved(token, 2 * keyWidth + channel));
  }
  require(prepareUlps <= 2.0, label + "prepare output is off by " +
                                  std::to_string(prepareUlps) + " bf16 ulps");

  // Gates: beta = bf16(sigmoid(b)); decay = exp(a_scale * bf16(softplus(x)))
  // where the GPU's fp32 softplus may round to either bf16 neighbour.
  const auto *decay = data<float>(buffers.decay);
  const auto *beta = data<uint16_t>(buffers.beta);
  Worst worstBeta, worstDecay;
  for (uint32_t token = 0; token < tokens; ++token) {
    const uint16_t *packedRow = packed + uint64_t{token} * packedWidth;
    for (uint32_t head = 0; head < valueHeads; ++head) {
      const uint64_t gate = uint64_t{token} * valueHeads + head;
      const double b = fromBf16(packedRow[bOffset + head]);
      const double refBeta = roundBf16(1.0 / (1.0 + std::exp(-b)));
      worstBeta.add(fromBf16(beta[gate]), refBeta);
      require(std::fabs(fromBf16(beta[gate]) - refBeta) <= bf16Ulp(refBeta),
              label + "beta differs from the reference");
      const double x =
          roundBf16(fromBf16(packedRow[aOffset + head]) + fromBf16(dtBias[head]));
      const double softplus = std::log1p(std::exp(-std::fabs(x))) + std::max(x, 0.0);
      const uint16_t nearest = toBf16(softplus);
      const double low = std::exp(aScale[head] * fromBf16(nearest + 1));
      const double high = std::exp(aScale[head] * fromBf16(nearest - 1));
      const double got = decay[gate];
      worstDecay.add(got, std::exp(aScale[head] * fromBf16(nearest)));
      require(got >= low * (1.0 - 1e-5) - 1e-30 &&
                  got <= high * (1.0 + 1e-5) + 1e-30,
              label + "decay differs from the reference");
    }
  }

  // Carried convolution state: the last inputs seen, one per carried row.
  const auto *convOut = data<uint16_t>(buffers.convolutionOut);
  for (uint32_t carried = 0; carried < kCarried; ++carried) {
    for (uint32_t channel = 0; channel < convDim; ++channel) {
      const uint32_t source = tokens + carried;
      const uint16_t expected =
          source < kCarried
              ? convIn[source * convDim + channel]
              : packed[uint64_t{source - kCarried} * packedWidth + channel];
      require(convOut[carried * convDim + channel] == expected,
              label + "carried convolution state is wrong");
    }
  }

  // Recurrence in fp64 from the GPU's own q/k/v/gates. Every head for short
  // chunks, a spread of heads for the long ones.
  std::vector<uint32_t> heads;
  if (tokens <= 64) {
    for (uint32_t head = 0; head < valueHeads; ++head)
      heads.push_back(head);
  } else {
    heads = {0, valueHeads / 2, valueHeads - 1};
  }
  const auto *recurrent = data<uint16_t>(buffers.recurrentRows);
  const auto *stateOut = data<float>(buffers.recurrentOut);
  Worst worstRows, worstState;
  double rowUlps = 0.0;
  std::vector<double> state(uint64_t{kHeadDim} * kHeadDim);
  for (uint32_t head : heads) {
    const uint32_t keyHead = head / headsPerKey;
    for (uint64_t i = 0; i < state.size(); ++i)
      state[i] = stateIn[uint64_t{head} * kHeadDim * kHeadDim + i];
    for (uint32_t token = 0; token < tokens; ++token) {
      const uint64_t gate = uint64_t{token} * valueHeads + head;
      const double d = decay[gate], b = fromBf16(beta[gate]);
      const uint16_t *k = keys + (uint64_t{token} * keyHeads + keyHead) * kHeadDim;
      const uint16_t *q = queries + (uint64_t{token} * keyHeads + keyHead) * kHeadDim;
      const uint64_t rowBase = (uint64_t{token} * valueHeads + head) * kHeadDim;
      for (uint32_t r = 0; r < kHeadDim; ++r) {
        double *s = &state[uint64_t{r} * kHeadDim];
        double memory = 0.0;
        for (uint32_t c = 0; c < kHeadDim; ++c) {
          s[c] *= d;
          memory += s[c] * fromBf16(k[c]);
        }
        const double delta = (fromBf16(values[rowBase + r]) - memory) * b;
        double output = 0.0;
        for (uint32_t c = 0; c < kHeadDim; ++c) {
          s[c] += fromBf16(k[c]) * delta;
          output += s[c] * fromBf16(q[c]);
        }
        const double got = fromBf16(recurrent[rowBase + r]);
        worstRows.add(got, output);
        // Half a bf16 ulp of rounding plus the fp32 accumulation drift.
        rowUlps = std::max(rowUlps, std::fabs(got - output) /
                                        (bf16Ulp(output) + 1e-5));
      }
    }
    for (uint64_t i = 0; i < state.size(); ++i)
      worstState.add(stateOut[uint64_t{head} * kHeadDim * kHeadDim + i],
                     state[i]);
  }
  require(rowUlps <= 1.0, label + "recurrent rows are off by " +
                              std::to_string(rowUlps) + " bf16 ulps");
  require(worstState.error <= 1e-5,
          label + "recurrent state differs from the fp64 reference by " +
              std::to_string(worstState.error));

  // Gated output norm from the GPU's recurrent rows. Almost every output is
  // the reference's own double rounding; the rest are rounding ties the
  // kernel's fp32 math breaks the other way. Norm weights rounded to bf16
  // would move a large fraction of them.
  const auto *hidden = data<uint16_t>(buffers.hidden);
  double hiddenUlps = 0.0;
  uint64_t hiddenInexact = 0;
  for (uint32_t token = 0; token < tokens; ++token) {
    for (uint32_t head = 0; head < valueHeads; ++head) {
      const uint64_t base = (uint64_t{token} * valueHeads + head) * kHeadDim;
      const std::vector<double> normalized =
          splash::test::rmsNorm(recurrent + base, buffers.mixerNorm, kHeadDim);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double z = fromBf16(packed[uint64_t{token} * packedWidth + convDim +
                                         head * kHeadDim + dim]);
        const double ref = roundBf16(roundBf16(normalized[dim]) * silu(z));
        const double got = fromBf16(hidden[base + dim]);
        require(std::isfinite(got), label + "gated output is not finite");
        hiddenUlps = std::max(hiddenUlps, std::fabs(got - ref) / bf16Ulp(ref));
        hiddenInexact += got != ref;
      }
    }
  }
  require(hiddenUlps <= 2.0, label + "gated output is off by " +
                                 std::to_string(hiddenUlps) + " bf16 ulps");
  const double inexact =
      double(hiddenInexact) / (uint64_t{tokens} * valueWidth);
  require(inexact <= 0.01, label + "gated output differs from the reference in " +
                               std::to_string(inexact * 100) + "% of values");

  std::cout << label << "prepare " << prepareUlps << " ulps (max|err| "
            << worstPrepare.error << "), beta max|err| " << worstBeta.error
            << ", decay max|err| " << worstDecay.error << ", rows " << rowUlps
            << " ulps (max|err| " << worstRows.error << "), state max|err| "
            << worstState.error << ", hidden " << hiddenUlps << " ulps ("
            << inexact * 100 << "% inexact)\n";
}

void requireSameBytes(const MetalBuffer &got, const MetalBuffer &want,
                      uint64_t wantOffset, uint64_t bytes,
                      const std::string &message) {
  require(std::memcmp(got.contents(),
                      static_cast<const uint8_t *>(want.contents()) +
                          wantOffset,
                      bytes) == 0,
          message);
}

// The scan is exactly sequential, so a chunk split at any token, with the
// fp32 state and the convolution carry handed from the first part to the
// second, reproduces the single-pass outputs byte for byte. Decode continues
// a prefill state on that basis; a reassociated or deferred-decay scan would
// break it, so the invariant is checked rather than assumed. The split points
// are not multiples of the sixteen-token scan block.
void runSplitCase(MetalBackend &backend, const GdnShape &shape,
                  uint32_t tokens, uint32_t split) {
  const std::string label = "vh" + std::to_string(shape.valueHeads) +
                            " tokens=" + std::to_string(tokens) + " split=" +
                            std::to_string(split) + ": ";
  const uint64_t rowBytes = uint64_t{shape.packedWidth} * 2;
  const uint64_t keyBytes = uint64_t{shape.keyHeads} * kHeadDim * 2;
  const uint64_t valueBytes = uint64_t{shape.valueHeads} * kHeadDim * 2;
  const uint32_t rest = tokens - split;

  GdnPrefillBuffers whole = randomPrefill(backend, shape, tokens);
  submitPrefill(backend, whole, shape, tokens, label);
  GdnPrefillBuffers first = prefillBuffers(
      backend, shape, split, backend.view(whole.packed, 0, split * rowBytes),
      whole.convolutionIn, whole.recurrentIn, whole.convolutionWeights,
      whole.decayWeights, whole.timeBias, whole.mixerNorm);
  submitPrefill(backend, first, shape, split, label);
  GdnPrefillBuffers second = prefillBuffers(
      backend, shape, rest,
      backend.view(whole.packed, split * rowBytes, rest * rowBytes),
      first.convolutionOut, first.recurrentOut, whole.convolutionWeights,
      whole.decayWeights, whole.timeBias, whole.mixerNorm);
  submitPrefill(backend, second, shape, rest, label);

  requireSameBytes(second.queries, whole.queries, split * keyBytes,
                   rest * keyBytes, label + "queries differ after the carry");
  requireSameBytes(second.keys, whole.keys, split * keyBytes,
                   rest * keyBytes, label + "keys differ after the carry");
  requireSameBytes(second.values, whole.values, split * valueBytes,
                   rest * valueBytes, label + "values differ after the carry");
  requireSameBytes(second.convolutionOut, whole.convolutionOut, 0,
                   whole.convolutionOut.sizeBytes(),
                   label + "convolution carry differs");
  requireSameBytes(first.recurrentRows, whole.recurrentRows, 0,
                   split * valueBytes, label + "first-part rows differ");
  requireSameBytes(second.recurrentRows, whole.recurrentRows,
                   split * valueBytes, rest * valueBytes,
                   label + "second-part rows differ");
  requireSameBytes(second.recurrentOut, whole.recurrentOut, 0,
                   whole.recurrentOut.sizeBytes(),
                   label + "recurrent state differs");
  requireSameBytes(second.hidden, whole.hidden, split * valueBytes,
                   rest * valueBytes, label + "hidden rows differ");
  std::cout << label << "byte-identical to the single pass\n";
}

// A bf16 state's scan computes as the fp32 state's does: from the same state,
// whose values bf16 holds, its recurrent and hidden rows are the fp32 scan's
// byte for byte, and the state it stores is the fp32 scan's rounded to
// nearest even.
void runBfloatStateCase(MetalBackend &backend, const GdnShape &shape,
                        uint32_t tokens) {
  const std::string label = "vh" + std::to_string(shape.valueHeads) +
                            " tokens=" + std::to_string(tokens) +
                            " bf16 state: ";
  GdnPrefillBuffers wide = randomPrefill(backend, shape, tokens);
  const uint64_t elements = wide.recurrentIn.sizeBytes() / 4;
  MetalBuffer narrowIn = shared(backend, elements * 2, "bf16 state in");
  auto *wideState = data<float>(wide.recurrentIn);
  auto *narrowState = data<uint16_t>(narrowIn);
  for (uint64_t i = 0; i < elements; ++i) {
    narrowState[i] = toBf16(wideState[i]);
    wideState[i] = static_cast<float>(fromBf16(narrowState[i]));
  }
  GdnPrefillBuffers narrow = prefillBuffers(
      backend, shape, tokens, wide.packed, wide.convolutionIn, narrowIn,
      wide.convolutionWeights, wide.decayWeights, wide.timeBias,
      wide.mixerNorm);
  submitPrefill(backend, wide, shape, tokens, label);
  submitPrefill(backend, narrow, shape, tokens, label, GdnHeadOrder::Grouped,
                GdnStateFormat::BFloat16);
  requireSameBytes(narrow.recurrentRows, wide.recurrentRows, 0,
                   wide.recurrentRows.sizeBytes(),
                   label + "recurrent rows differ from the fp32 scan's");
  requireSameBytes(narrow.hidden, wide.hidden, 0, wide.hidden.sizeBytes(),
                   label + "hidden rows differ from the fp32 scan's");
  const auto *wideOut = data<float>(wide.recurrentOut);
  const auto *narrowOut = data<uint16_t>(narrow.recurrentOut);
  for (uint64_t i = 0; i < elements; ++i)
    require(narrowOut[i] == toBf16(wideOut[i]),
            label + "the state is not the fp32 scan's rounded to nearest even");
  std::cout << label << "the fp32 scan's rows and its state rounded\n";
}

// A lane of a ragged prefill command: its rows among the command's and the
// rows after which it captures its state.
struct RaggedLane final {
  uint32_t rowBegin = 0;
  uint32_t tokens = 0;
  std::vector<uint32_t> captures;
};

// One run of a ragged command: the command's row buffers and each lane's
// final state and, where the run captures them, the states inside its rows.
struct RaggedRun final {
  GdnPrefillBuffers rows;
  std::vector<MetalBuffer> convolution;
  std::vector<MetalBuffer> recurrent;
  std::vector<std::vector<GdnPrefillCapture>> captures;
};

// A prefill that captures states inside its rows runs the prepare and scan
// dispatches once per run between them, chaining the state through the captured
// ones, and the gate once: ragged lanes in one command, each from the shared
// starting state, capturing one to three states at rows off the sixteen-token
// scan block, with segments of one and two rows. Each captured state is the state
// a prefill split there stores, byte for byte, convolution carry and
// recurrent state, and the whole command equals the split; with an fp32
// state, whose segments are invisible, it also equals the single pass.
void runCapturedStatesCase(MetalBackend &backend, const GdnShape &shape, GdnStateFormat format) {
  const std::string label = "vh" + std::to_string(shape.valueHeads) + " " +
                            std::string(splash::ops::formatName(format)) + " captured states: ";
  const std::vector<RaggedLane> lanes{
      {0, 37, {17}}, {37, 1000, {1, 333, 998}}, {1037, 611, {300, 302}}, {1648, 400, {}}};
  const uint32_t tokens = 2048;
  GdnPrefillBuffers source = randomPrefill(backend, shape, tokens);
  const uint64_t elements = source.recurrentIn.sizeBytes() / 4;
  const uint64_t stateBytes = elements * splash::ops::elementBytes(format);
  const uint64_t carriedBytes = uint64_t{kCarried} * shape.convolutionDimension * 2;
  if (format == GdnStateFormat::BFloat16) {
    MetalBuffer narrow = shared(backend, stateBytes, "bf16 state in");
    for (uint64_t i = 0; i < elements; ++i)
      data<uint16_t>(narrow)[i] = toBf16(data<float>(source.recurrentIn)[i]);
    source.recurrentIn = narrow;
  }
  const uint64_t keyBytes = uint64_t{shape.keyHeads} * kHeadDim * 2;
  const uint64_t valueBytes = uint64_t{shape.valueHeads} * kHeadDim * 2;
  const auto run = [&] {
    RaggedRun result{prefillBuffers(backend, shape, tokens, source.packed, source.convolutionIn,
                                    source.recurrentIn, source.convolutionWeights, source.decayWeights,
                                    source.timeBias, source.mixerNorm),
                     {}, {}, {}};
    for (size_t lane = 0; lane < lanes.size(); ++lane) {
      result.convolution.push_back(shared(backend, carriedBytes, "lane conv out"));
      result.recurrent.push_back(shared(backend, stateBytes, "lane state out"));
    }
    return result;
  };
  // A run's rows [begin, begin + count) from the given states to the given
  // states.
  const auto rows = [&](const RaggedRun &target, uint32_t begin, uint32_t count, MetalBuffer convolutionIn,
                        MetalBuffer recurrentIn, MetalBuffer convolutionOut, MetalBuffer recurrentOut) {
    const auto view = [&](const MetalBuffer &buffer, uint64_t rowBytes) {
      return backend.view(buffer, begin * rowBytes, count * rowBytes);
    };
    const GdnPrefillBuffers &r = target.rows;
    return GdnPrefillBuffers{view(r.packed, uint64_t{shape.packedWidth} * 2), r.convolutionWeights,
                             std::move(convolutionIn), std::move(convolutionOut), view(r.queries, keyBytes),
                             view(r.keys, keyBytes), view(r.values, valueBytes), r.decayWeights, r.timeBias,
                             view(r.decay, uint64_t{shape.valueHeads} * 4),
                             view(r.beta, uint64_t{shape.valueHeads} * 2), std::move(recurrentIn),
                             std::move(recurrentOut), view(r.recurrentRows, valueBytes), r.mixerNorm,
                             view(r.hidden, valueBytes)};
  };
  const auto submit = [&](const CommandGraph &graph) { (void)backend.submitCommandAsync(graph.dispatches()).wait(); };

  RaggedRun captured = run(), single = run(), split = run();
  captured.captures.resize(lanes.size());
  split.captures.resize(lanes.size());
  CommandGraph capturing, whole, parts;
  uint64_t dispatches = 0;
  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    const RaggedLane &ragged = lanes[lane];
    for (RaggedRun *target : {&captured, &split})
      for (uint32_t row : ragged.captures)
        target->captures[lane].push_back(
            {row, shared(backend, carriedBytes, "captured conv"), shared(backend, stateBytes, "captured state")});
    GDN::addPrefill(capturing, backend,
                    rows(captured, ragged.rowBegin, ragged.tokens, source.convolutionIn, source.recurrentIn,
                         captured.convolution[lane], captured.recurrent[lane]),
                    shape, ragged.tokens, format, GdnHeadOrder::Grouped, captured.captures[lane]);
    dispatches += 2 * (ragged.captures.size() + 1) + 1;
    GDN::addPrefill(whole, backend,
                    rows(single, ragged.rowBegin, ragged.tokens, source.convolutionIn, source.recurrentIn,
                         single.convolution[lane], single.recurrent[lane]),
                    shape, ragged.tokens, format, GdnHeadOrder::Grouped);
    MetalBuffer convolution = source.convolutionIn, recurrent = source.recurrentIn;
    uint32_t begin = 0;
    for (size_t part = 0; part <= ragged.captures.size(); ++part) {
      const bool last = part == ragged.captures.size();
      const uint32_t end = last ? ragged.tokens : ragged.captures[part];
      const MetalBuffer convolutionOut = last ? split.convolution[lane] : split.captures[lane][part].convolution;
      const MetalBuffer recurrentOut = last ? split.recurrent[lane] : split.captures[lane][part].recurrent;
      GDN::addPrefill(parts, backend,
                      rows(split, ragged.rowBegin + begin, end - begin, convolution, recurrent, convolutionOut,
                           recurrentOut),
                      shape, end - begin, format, GdnHeadOrder::Grouped);
      convolution = convolutionOut;
      recurrent = recurrentOut;
      begin = end;
    }
  }
  require(capturing.dispatches().size() == dispatches,
          label + "a captured state does not add one prepare and one scan");
  submit(capturing);
  submit(whole);
  submit(parts);

  const auto same = [&](const MetalBuffer &got, const MetalBuffer &want, const std::string &what) {
    requireSameBytes(got, want, 0, want.sizeBytes(), label + what);
  };
  const auto sameRows = [&](const RaggedRun &got, const RaggedRun &want, const std::string &against) {
    for (const auto &[member, what] :
         std::initializer_list<std::pair<MetalBuffer GdnPrefillBuffers::*, const char *>>{
             {&GdnPrefillBuffers::queries, "queries"}, {&GdnPrefillBuffers::keys, "keys"},
             {&GdnPrefillBuffers::values, "values"}, {&GdnPrefillBuffers::decay, "decay"},
             {&GdnPrefillBuffers::beta, "beta"}, {&GdnPrefillBuffers::recurrentRows, "recurrent rows"},
             {&GdnPrefillBuffers::hidden, "hidden rows"}})
      same(got.rows.*member, want.rows.*member, std::string(what) + " differ from " + against);
    for (size_t lane = 0; lane < lanes.size(); ++lane) {
      same(got.convolution[lane], want.convolution[lane], "convolution carry differs from " + against);
      same(got.recurrent[lane], want.recurrent[lane], "recurrent state differs from " + against);
    }
  };
  for (size_t lane = 0; lane < lanes.size(); ++lane)
    for (size_t index = 0; index < lanes[lane].captures.size(); ++index) {
      const std::string where = " of lane " + std::to_string(lane) + " at row " +
                                std::to_string(lanes[lane].captures[index]);
      same(captured.captures[lane][index].convolution, split.captures[lane][index].convolution,
           "captured convolution carry differs from the split's" + where);
      same(captured.captures[lane][index].recurrent, split.captures[lane][index].recurrent,
           "captured recurrent state differs from the split's" + where);
    }
  sameRows(captured, split, "the split");
  if (format == GdnStateFormat::Float32)
    sameRows(captured, single, "the single pass");
  std::cout << label << "each captured state is the split's, and the command "
            << (format == GdnStateFormat::Float32 ? "the single pass\n" : "the split\n");
}

// The tiled head order only moves each value head's output block: over the
// same inputs, the tiled hidden rows are the grouped ones with head h at
// (h % heads per key) * key heads + h / heads per key, byte for byte, and the
// recurrent rows and state are unchanged. Tiled is a GGUF's order, so the
// norm is F32 as a GGUF stores it.
void runTiledCase(MetalBackend &backend, const GdnShape &shape,
                  uint32_t tokens) {
  const std::string label = "vh" + std::to_string(shape.valueHeads) +
                            " tokens=" + std::to_string(tokens) +
                            " tiled f32 norm: ";
  const uint32_t headsPerKey = shape.valueHeads / shape.keyHeads;
  const uint64_t headBytes = uint64_t{kHeadDim} * 2;
  GdnPrefillBuffers grouped = randomPrefill(backend, shape, tokens, true);
  submitPrefill(backend, grouped, shape, tokens, label);
  GdnPrefillBuffers tiled = prefillBuffers(
      backend, shape, tokens, grouped.packed, grouped.convolutionIn,
      grouped.recurrentIn, grouped.convolutionWeights, grouped.decayWeights,
      grouped.timeBias, grouped.mixerNorm);
  submitPrefill(backend, tiled, shape, tokens, label, GdnHeadOrder::Tiled);
  requireSameBytes(tiled.recurrentRows, grouped.recurrentRows, 0,
                   grouped.recurrentRows.sizeBytes(),
                   label + "recurrent rows differ");
  requireSameBytes(tiled.recurrentOut, grouped.recurrentOut, 0,
                   grouped.recurrentOut.sizeBytes(),
                   label + "recurrent state differs");
  const auto *got = data<const uint8_t>(tiled.hidden);
  const auto *want = data<const uint8_t>(grouped.hidden);
  for (uint64_t token = 0; token < tokens; ++token)
    for (uint32_t head = 0; head < shape.valueHeads; ++head) {
      const uint32_t position =
          (head % headsPerKey) * shape.keyHeads + head / headsPerKey;
      require(std::memcmp(got + (token * shape.valueHeads + position) * headBytes,
                          want + (token * shape.valueHeads + head) * headBytes,
                          headBytes) == 0,
              label + "hidden rows are not the grouped rows in tiled head order");
    }
  std::cout << label << "byte-identical to the grouped order\n";
}

// Each buffer the prefill reaches, at its extent and one element short: the
// packed rows up to the last row's alpha, the four taps of every channel's
// convolution weights and its three carried rows, the q/k/v and gate rows of
// every token, a value head's state in its format, and the mixer norm in
// either type. The prefill captures a state inside its rows, so each run of its
// dispatches reaches its views' extents, and the captured state's buffers have
// the state's.
void bufferExtents(MetalBackend &backend, const GdnShape &shape, bool float32,
                   GdnStateFormat format) {
  constexpr uint32_t tokens = 37;
  const GdnPrefillBuffers buffers = randomPrefill(backend, shape, tokens, float32);
  const uint64_t valueWidth = uint64_t{shape.valueHeads} * kHeadDim;
  const uint64_t keyRows = uint64_t{tokens} * shape.keyHeads * kHeadDim * 2;
  const uint64_t valueRows = tokens * valueWidth * 2;
  const uint64_t stateElement = splash::ops::elementBytes(format);
  const uint64_t state = valueWidth * kHeadDim * stateElement;
  const uint64_t carried = uint64_t{kCarried} * shape.convolutionDimension * 2;
  const GdnPrefillCapture capture{17, shared(backend, carried, "captured conv"), shared(backend, state, "captured state")};
  const auto encode = [&](CommandGraph &graph, const GdnPrefillBuffers &changed,
                          const GdnPrefillCapture &capture) {
    GDN::addPrefill(graph, backend, changed, shape, tokens, format, GdnHeadOrder::Grouped, {&capture, 1});
  };
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer GdnPrefillBuffers::*, uint64_t, uint64_t, const char *>>{
           {&GdnPrefillBuffers::packed,
            (uint64_t{tokens - 1} * shape.packedWidth + shape.convolutionDimension + valueWidth +
             2 * shape.valueHeads) * 2,
            2, "GDN packed"},
           {&GdnPrefillBuffers::convolutionWeights, uint64_t{shape.convolutionDimension} * kTaps * 2, 2,
            "GDN convolution weight"},
           {&GdnPrefillBuffers::convolutionIn, uint64_t{kCarried} * shape.convolutionDimension * 2, 2,
            "GDN convolution state"},
           {&GdnPrefillBuffers::convolutionOut, uint64_t{kCarried} * shape.convolutionDimension * 2, 2,
            "GDN next convolution state"},
           {&GdnPrefillBuffers::queries, keyRows, 2, "GDN query"},
           {&GdnPrefillBuffers::keys, keyRows, 2, "GDN key"},
           {&GdnPrefillBuffers::values, valueRows, 2, "GDN value"},
           {&GdnPrefillBuffers::decayWeights, uint64_t{shape.valueHeads} * 4, 4, "GDN decay weight"},
           {&GdnPrefillBuffers::timeBias, uint64_t{shape.valueHeads} * 2, 2, "GDN time bias"},
           {&GdnPrefillBuffers::decay, uint64_t{tokens} * shape.valueHeads * 4, 4, "GDN decay"},
           {&GdnPrefillBuffers::beta, uint64_t{tokens} * shape.valueHeads * 2, 2, "GDN beta"},
           {&GdnPrefillBuffers::recurrentIn, state, stateElement, "GDN recurrent state"},
           {&GdnPrefillBuffers::recurrentOut, state, stateElement, "GDN next recurrent state"},
           {&GdnPrefillBuffers::recurrentRows, valueRows, 2, "GDN recurrent row"},
           {&GdnPrefillBuffers::hidden, valueRows, 2, "GDN hidden"}})
    splash::test::requireExtent(backend, buffers.*member, bytes, element, what,
                                [&](CommandGraph &graph, const MetalBuffer &buffer) {
                                  GdnPrefillBuffers changed = buffers;
                                  changed.*member = buffer;
                                  encode(graph, changed, capture);
                                });
  const uint64_t normElement = float32 ? 4 : 2;
  splash::test::requireExtent(backend, buffers.mixerNorm.buffer, kHeadDim * normElement, normElement,
                              "norm weight", [&](CommandGraph &graph, const MetalBuffer &buffer) {
                                GdnPrefillBuffers changed = buffers;
                                changed.mixerNorm.buffer = buffer;
                                encode(graph, changed, capture);
                              });
  for (const auto &[member, bytes, element, what] :
       std::initializer_list<std::tuple<MetalBuffer GdnPrefillCapture::*, uint64_t, uint64_t, const char *>>{
           {&GdnPrefillCapture::convolution, carried, 2, "GDN captured convolution state"},
           {&GdnPrefillCapture::recurrent, state, stateElement, "GDN captured recurrent state"}})
    splash::test::requireExtent(backend, capture.*member, bytes, element, what,
                                [&](CommandGraph &graph, const MetalBuffer &buffer) {
                                  GdnPrefillCapture changed = capture;
                                  changed.*member = buffer;
                                  encode(graph, buffers, changed);
                                });
}

void run(const std::string &metallib) {
  MetalBackend backend(metallib);
  for (const GdnShape &shape :
       {GdnShape{16, 48, 128, 10240, 16640}, GdnShape{16, 32, 128, 8192, 12544},
        GdnShape{16, 48, 128, 10240, 16512}}) {
    for (bool float32 : {false, true})
      bufferExtents(backend, shape, float32, GdnStateFormat::Float32);
    bufferExtents(backend, shape, false, GdnStateFormat::BFloat16);
    for (uint32_t tokens : {1u, 37u, 1000u, 2048u})
      runCase(backend, shape, tokens);
    for (uint32_t tokens : {1u, 37u})
      runCase(backend, shape, tokens, true);
    runSplitCase(backend, shape, 2048, 1000);
    runSplitCase(backend, shape, 37, 17);
    for (uint32_t tokens : {1u, 37u, 1000u})
      runTiledCase(backend, shape, tokens);
    for (uint32_t tokens : {1u, 37u, 1000u})
      runBfloatStateCase(backend, shape, tokens);
    for (GdnStateFormat format : {GdnStateFormat::Float32, GdnStateFormat::BFloat16})
      runCapturedStatesCase(backend, shape, format);
  }
  std::cout << "gdn_metal_test passed\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gdn_metal_test <metallib>\n";
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
