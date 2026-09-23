// GDN prefill against a CPU reference through the operator's own dispatch
// geometry: the convolution/normalisation prepare pass, the blocked recurrent
// scan (output rows and the fp32 state it hands to decode), and the gated
// output norm, for both compiled head layouts and ragged token counts that
// leave partial scan blocks and single-token chunks.
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/ops/GDN.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::GDN;
using splash::ops::GdnHeadOrder;
using splash::ops::GdnPrefillBuffers;
using splash::ops::GdnShape;
using splash::ops::NormWeights;

constexpr uint32_t kHeadDim = 128;
constexpr double kEpsilon = 1e-6;
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
  uint32_t bits = std::bit_cast<uint32_t>(static_cast<float>(value));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}
double fromBf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}
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
          shared(backend, uint64_t{3} * convDim * 2, "conv out"),
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
      shared(backend, uint64_t{3} * convDim * 2, "conv in"),
      shared(backend, stateElements * 4, "state in"),
      shared(backend, uint64_t{convDim} * 4 * 2, "conv weights"),
      shared(backend, valueHeads * 4, "a scale"),
      shared(backend, valueHeads * 2, "dt bias"),
      {shared(backend, kHeadDim * (float32 ? 4 : 2), "mixer norm"), float32});

  Random random(0x9E3779B97F4A7C15ULL ^ (uint64_t{valueHeads} << 32) ^ tokens);
  auto *packed = data<uint16_t>(buffers.packed);
  for (uint64_t i = 0; i < uint64_t{tokens} * packedWidth; ++i)
    packed[i] = toBf16(random.gauss());
  auto *convWeights = data<uint16_t>(buffers.convolutionWeights);
  for (uint64_t i = 0; i < uint64_t{convDim} * 4; ++i)
    convWeights[i] = toBf16(0.3 * random.gauss());
  auto *convIn = data<uint16_t>(buffers.convolutionIn);
  for (uint64_t i = 0; i < uint64_t{3} * convDim; ++i)
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
  for (uint32_t i = 0; i < kHeadDim; ++i) {
    const double weight = 1.0 + 0.2 * random.gauss();
    if (float32)
      data<float>(buffers.mixerNorm.buffer)[i] = static_cast<float>(weight);
    else
      data<uint16_t>(buffers.mixerNorm.buffer)[i] = toBf16(weight);
  }
  return buffers;
}

double normWeight(const NormWeights &norm, uint32_t index) {
  return norm.float32 ? data<float>(norm.buffer)[index]
                      : fromBf16(data<uint16_t>(norm.buffer)[index]);
}

void submitPrefill(MetalBackend &backend, const GdnPrefillBuffers &buffers,
                   const GdnShape &shape, uint32_t tokens,
                   const std::string &label,
                   GdnHeadOrder order = GdnHeadOrder::Grouped) {
  CommandGraph graph;
  GDN::addPrefill(graph, buffers, shape, tokens, order);
  require(graph.dispatches().size() == 3,
          label + "prefill is not three dispatches");
  (void)backend.submitCommand(graph.dispatches());
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
    for (uint32_t tap = 0; tap < 4; ++tap) {
      const uint32_t position = token + tap;
      const double input =
          position < 3 ? fromBf16(convIn[position * convDim + channel])
                       : fromBf16(packed[uint64_t{position - 3} * packedWidth +
                                         channel]);
      value += input * fromBf16(convWeights[channel * 4 + tap]);
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
        const double inverse = 1.0 / std::sqrt(squares / kHeadDim + kEpsilon);
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

  // Carried convolution state: the last three inputs seen.
  const auto *convOut = data<uint16_t>(buffers.convolutionOut);
  for (uint32_t carried = 0; carried < 3; ++carried) {
    for (uint32_t channel = 0; channel < convDim; ++channel) {
      const uint32_t source = tokens + carried;
      const uint16_t expected =
          source < 3 ? convIn[source * convDim + channel]
                     : packed[uint64_t{source - 3} * packedWidth + channel];
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
      double squares = 0.0;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        row[dim] = fromBf16(recurrent[base + dim]);
        squares += row[dim] * row[dim];
      }
      const double inverse = 1.0 / std::sqrt(squares / kHeadDim + kEpsilon);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double z = fromBf16(packed[uint64_t{token} * packedWidth + convDim +
                                         head * kHeadDim + dim]);
        const double ref = roundBf16(
            roundBf16(row[dim] * inverse * normWeight(buffers.mixerNorm, dim)) *
            silu(z));
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

// The tiled head order only moves each value head's output block: over the
// same inputs, the tiled hidden rows are the grouped ones with head h at
// (h % heads per key) * key heads + h / heads per key, byte for byte, and the
// recurrent rows and state are unchanged.
void runTiledCase(MetalBackend &backend, const GdnShape &shape,
                  uint32_t tokens) {
  const std::string label = "vh" + std::to_string(shape.valueHeads) +
                            " tokens=" + std::to_string(tokens) + " tiled: ";
  const uint32_t headsPerKey = shape.valueHeads / shape.keyHeads;
  const uint64_t headBytes = uint64_t{kHeadDim} * 2;
  GdnPrefillBuffers grouped = randomPrefill(backend, shape, tokens);
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

void run(const std::string &metallib) {
  MetalBackend backend(metallib);
  for (const GdnShape &shape :
       {GdnShape{16, 48, 128, 10240, 16640}, GdnShape{16, 32, 128, 8192, 12544}}) {
    for (uint32_t tokens : {1u, 37u, 1000u, 2048u})
      runCase(backend, shape, tokens);
    for (uint32_t tokens : {1u, 37u})
      runCase(backend, shape, tokens, true);
    runSplitCase(backend, shape, 2048, 1000);
    runSplitCase(backend, shape, 37, 17);
    for (uint32_t tokens : {1u, 37u, 1000u})
      runTiledCase(backend, shape, tokens);
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
