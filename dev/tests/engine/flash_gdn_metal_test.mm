// Independent scalar oracle for Qwen4's BF16 array boundaries, lane-isolated
// recurrent state, short convolution carry, and split versus whole prefill.
// This executable only submits GPU work when given a metallib path. The
// --cpu-only mode checks the reference's zero and carry identities.
#include "flash/FlashGDN.hpp"
#include "flash/FlashGDNFused.hpp"
#include "flash/FlashGDNSeparate.hpp"
#include "flash/FlashMTPWindow.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

using namespace splash::flash;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t C = kFlashGDNConvolutionWidth;
constexpr uint32_t V = kFlashGDNOutputWidth;
constexpr uint32_t H = kFlashGDNValueHeads;
constexpr uint32_t K = kFlashGDNHeadDimension;
constexpr uint32_t KW = kFlashGDNKeyHeads * K;
constexpr uint64_t StateElements = uint64_t{H} * K * K;
uint32_t fusionMode = 0;

void encodeGDN(CommandGraph &graph, const FlashGDNWeights &weights,
                 const FlashGDNBuffers &buffers, const FlashGDNState &state,
                 uint32_t rows, uint32_t lanes = 1, float epsilon = 1e-6f) {
  if (fusionMode)
    addGDNFused(graph, weights, buffers, state, rows, lanes,
                 fusionMode == 1 ? FlashGDNFusion::Prepare
                     : fusionMode == 4 ? FlashGDNFusion::PersistentHead512
                     : fusionMode == 5 ? FlashGDNFusion::PersistentHead1024
                                      : FlashGDNFusion::PersistentHead,
                 epsilon);
  else
    addGDN(graph, weights, buffers, state, rows, lanes, epsilon);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

uint16_t bits(float value) {
  uint32_t raw;
  std::memcpy(&raw, &value, sizeof(raw));
  raw += 0x7fff + ((raw >> 16) & 1);
  return static_cast<uint16_t>(raw >> 16);
}

float number(uint16_t value) {
  const uint32_t raw = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}

float rounded(float value) { return number(bits(value)); }
float sigmoid(float value) {
  return static_cast<float>(1.0 / (1.0 + std::exp(-double(value))));
}
float sigmoidBF16(float value) {
  const float exponent = rounded(std::exp(std::abs(value)));
  const float denominator = rounded(1.0f + exponent);
  const float tail = rounded(1.0f / denominator);
  return value < 0.0f ? tail : rounded(1.0f - tail);
}

float softplusBF16(float value) {
  const float exponent = rounded(std::exp(-std::abs(value)));
  const float xp1 = 1.0f + exponent;
  const float logarithm =
      xp1 == 1.0f ? exponent
                  : rounded(exponent * (std::log(xp1) / (xp1 - 1.0f)));
  return rounded(std::max(value, 0.0f) + logarithm);
}

struct Random {
  uint64_t state = 0x315789635abcdeULL;
  float value(float scale) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (static_cast<float>((state >> 40) & 0xffffff) / 8388608.0f - 1.0f) *
           scale;
  }
};

struct Reference {
  uint32_t rows, lanes;
  std::vector<uint16_t> qkv, z, a, b, convolution, aLog, timeBias, norm;
  std::vector<uint16_t> history, mixed, beta, recurrentRows, output;
  std::vector<float> recurrent, decay;

  Reference(uint32_t rowCount, uint32_t laneCount, bool cold)
      : rows(rowCount), lanes(laneCount), qkv(uint64_t{rows} * lanes * C),
        z(uint64_t{rows} * lanes * V), a(uint64_t{rows} * lanes * H),
        b(a.size()), convolution(uint64_t{C} * 4), aLog(H), timeBias(H),
        norm(K), history(uint64_t{lanes} * 3 * C), mixed(qkv.size()),
        beta(a.size()), recurrentRows(z.size()), output(z.size()),
        recurrent(uint64_t{lanes} * StateElements), decay(a.size()) {
    Random random;
    auto fill = [&](std::vector<uint16_t> &values, float scale) {
      for (auto &value : values)
        value = bits(random.value(scale));
    };
    fill(qkv, 0.8f);
    fill(z, 2.0f);
    fill(a, 1.5f);
    fill(b, 2.0f);
    // Actual MLX unary and compiled BF16 sigmoid differ at this finite input.
    // The beta projection feeds an unfused unary primitive, so this case
    // prevents accidentally using convolution SiLU's fast sigmoid helper.
    b[0] = bits(-6.84375f);
    fill(convolution, 0.25f);
    fill(aLog, 0.8f);
    fill(timeBias, 0.5f);
    for (auto &value : norm)
      value = bits(1.0f + random.value(0.25f));
    if (!cold) {
      fill(history, 0.8f);
      for (auto &value : recurrent)
        value = random.value(0.1f);
    }
  }

  void run() {
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      for (uint32_t token = 0; token < rows; ++token) {
        const uint64_t row = uint64_t{lane} * rows + token;
        for (uint32_t channel = 0; channel < C; ++channel) {
          float total = 0.0f;
          for (uint32_t tap = 0; tap < 4; ++tap) {
            const uint32_t position = token + tap;
            const uint16_t source =
                position < 3
                    ? history[(uint64_t{lane} * 3 + position) * C + channel]
                    : qkv[(uint64_t{lane} * rows + position - 3) * C + channel];
            total += number(source) * number(convolution[channel * 4 + tap]);
          }
          const float value = rounded(total);
          mixed[row * C + channel] = bits(value * sigmoidBF16(value));
        }
        for (uint32_t head = 0; head < kFlashGDNKeyHeads; ++head) {
          float qSquared = 0.0f, kSquared = 0.0f;
          for (uint32_t reductionLane = 0; reductionLane < 32; ++reductionLane) {
            float qPartial = 0.0f, kPartial = 0.0f;
            for (uint32_t element = 0; element < 4; ++element) {
              const uint32_t dimension = 4 * reductionLane + element;
              const float q = number(mixed[row * C + head * K + dimension]);
              const float k = number(mixed[row * C + KW + head * K + dimension]);
              qPartial = rounded(qPartial + rounded(q * q));
              kPartial = rounded(kPartial + rounded(k * k));
            }
            qSquared += qPartial;
            kSquared += kPartial;
          }
          const float qInverse =
              rounded(1.0f / std::sqrt(rounded(rounded(qSquared) + rounded(1e-6f))));
          const float kInverse =
              rounded(1.0f / std::sqrt(rounded(rounded(kSquared) + rounded(1e-6f))));
          for (uint32_t dimension = 0; dimension < K; ++dimension) {
            const uint64_t qIndex = row * C + head * K + dimension;
            const uint64_t kIndex = qIndex + KW;
            mixed[qIndex] = bits(rounded(number(mixed[qIndex]) * qInverse) *
                                 rounded(1.0f / std::sqrt(float(K))));
            mixed[kIndex] = bits(number(mixed[kIndex]) * kInverse);
          }
        }
        for (uint32_t head = 0; head < H; ++head) {
          const uint64_t gateIndex = row * H + head;
          const float x = rounded(number(a[gateIndex]) + number(timeBias[head]));
          decay[gateIndex] =
              std::exp(-std::exp(number(aLog[head])) * softplusBF16(x));
          beta[gateIndex] = bits(sigmoidBF16(number(b[gateIndex])));
        }
      }
      // Each matrix row belongs to one value dimension. A direct scalar sum
      // along the key dimension is independent of the SIMD GPU reduction.
      for (uint32_t token = 0; token < rows; ++token) {
        const uint64_t row = uint64_t{lane} * rows + token;
        for (uint32_t head = 0; head < H; ++head) {
          const uint32_t keyHead = head / (H / kFlashGDNKeyHeads);
          for (uint32_t valueDimension = 0; valueDimension < K;
               ++valueDimension) {
            const uint64_t stateBase = uint64_t{lane} * StateElements +
                                       (head * K + valueDimension) * K;
            float memory = 0.0f;
            for (uint32_t key = 0; key < K; ++key) {
              recurrent[stateBase + key] *= decay[row * H + head];
              memory += recurrent[stateBase + key] *
                        number(mixed[row * C + KW + keyHead * K + key]);
            }
            const float value =
                number(mixed[row * C + 2 * KW + head * K + valueDimension]);
            const float delta =
                (value - memory) * number(beta[row * H + head]);
            float out = 0.0f;
            for (uint32_t key = 0; key < K; ++key) {
              recurrent[stateBase + key] +=
                  number(mixed[row * C + KW + keyHead * K + key]) * delta;
              out += recurrent[stateBase + key] *
                     number(mixed[row * C + keyHead * K + key]);
            }
            recurrentRows[row * V + head * K + valueDimension] = bits(out);
          }
          float sum = 0.0f;
          for (uint32_t dimension = 0; dimension < K; ++dimension) {
            const float value =
                number(recurrentRows[row * V + head * K + dimension]);
            sum += value * value;
          }
          const float inverse = 1.0f / std::sqrt(sum / float(K) + 1e-6f);
          for (uint32_t dimension = 0; dimension < K; ++dimension) {
            const uint64_t index = row * V + head * K + dimension;
            const float normalized = rounded(number(recurrentRows[index]) * inverse);
            const float weighted = rounded(normalized * number(norm[dimension]));
            output[index] = bits(weighted * sigmoid(number(z[index])));
          }
        }
      }
      for (uint32_t previous = 0; previous < 3; ++previous) {
        const uint32_t source = rows + previous;
        for (uint32_t channel = 0; channel < C; ++channel)
          history[(uint64_t{lane} * 3 + previous) * C + channel] =
              source < 3
                  ? history[(uint64_t{lane} * 3 + source) * C + channel]
                  : qkv[(uint64_t{lane} * rows + source - 3) * C + channel];
      }
    }
  }
};

FlashTensor tensor(MetalBackend &backend, const std::vector<uint16_t> &values,
                   std::vector<uint64_t> shape, const char *label) {
  const uint64_t bytes = values.size() * sizeof(uint16_t);
  auto buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), values.data(), bytes);
  return {buffer, FlashDType::BF16, std::move(shape), bytes};
}

MetalBuffer buffer(MetalBackend &backend, uint64_t bytes, const char *label) {
  auto result = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(result.contents(), 0, result.sizeBytes());
  return result;
}

void copy(const MetalBuffer &destination, const std::vector<uint16_t> &source) {
  std::memcpy(destination.contents(), source.data(), source.size() * 2);
}

struct Fixture {
  FlashTensor convolution, aLog, timeBias, norm;
  FlashGDNBuffers buffers;
  FlashGDNState state;
  uint64_t convStride, recurrentStride;

  Fixture(MetalBackend &backend, const Reference &r)
      : convolution(tensor(backend, r.convolution, {C, 4, 1}, "flash conv")),
        aLog(tensor(backend, r.aLog, {H}, "flash A log")),
        timeBias(tensor(backend, r.timeBias, {H}, "flash dt bias")),
        norm(tensor(backend, r.norm, {K}, "flash gated norm")),
        convStride(flashGDNConvolutionLaneBytes() + 128),
        recurrentStride(flashGDNRecurrentLaneBytes() + 128) {
    buffers.qkv = buffer(backend, r.qkv.size() * 2, "flash qkv");
    buffers.z = buffer(backend, r.z.size() * 2, "flash z");
    buffers.a = buffer(backend, r.a.size() * 2, "flash a");
    buffers.b = buffer(backend, r.b.size() * 2, "flash b");
    buffers.mixed = buffer(backend, r.qkv.size() * 2, "flash mixed");
    buffers.decay = buffer(backend, r.a.size() * 4, "flash decay");
    buffers.beta = buffer(backend, r.a.size() * 2, "flash beta");
    buffers.recurrentRows = buffer(backend, r.z.size() * 2, "flash recurrent rows");
    buffers.output = buffer(backend, r.z.size() * 2, "flash output");
    buffers.diagnostics = buffer(backend, 4, "flash diagnostics");
    state.convolution = buffer(backend, convStride * r.lanes, "flash conv state");
    state.recurrent = buffer(backend, recurrentStride * r.lanes, "flash state");
    state.convolutionLaneStrideBytes = convStride;
    state.recurrentLaneStrideBytes = recurrentStride;
    copy(buffers.qkv, r.qkv);
    copy(buffers.z, r.z);
    copy(buffers.a, r.a);
    copy(buffers.b, r.b);
    for (uint32_t lane = 0; lane < r.lanes; ++lane) {
      std::memcpy(static_cast<uint8_t *>(state.convolution.contents()) +
                      lane * convStride,
                  r.history.data() + uint64_t{lane} * 3 * C,
                  flashGDNConvolutionLaneBytes());
      std::memcpy(static_cast<uint8_t *>(state.recurrent.contents()) +
                      lane * recurrentStride,
                  r.recurrent.data() + uint64_t{lane} * StateElements,
                  flashGDNRecurrentLaneBytes());
    }
  }

  FlashGDNWeights weights() const {
    return {&convolution, &aLog, &timeBias, &norm};
  }
};

double compareBF16(const MetalBuffer &got, const std::vector<uint16_t> &expected,
                    double ulps, double floor, const char *label) {
  const auto *values = static_cast<const uint16_t *>(got.contents());
  double maximum = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float wanted = number(expected[i]);
    const float observed = number(values[i]);
    int exponent;
    std::frexp(std::abs(wanted), &exponent);
    const double allowed = ulps * std::ldexp(1.0, exponent - 8) + floor;
    const double difference = std::abs(double(observed) - wanted);
    maximum = std::max(maximum, difference);
    require(std::isfinite(observed) && difference <= allowed,
             std::string(label) + " differs at " + std::to_string(i) +
                 ": observed=" + std::to_string(observed) +
                 " expected=" + std::to_string(wanted));
  }
  return maximum;
}

std::vector<uint16_t> gatedRMSReference(const MetalBuffer &recurrentRows,
                                        const Reference &reference) {
  const auto *values = static_cast<const uint16_t *>(recurrentRows.contents());
  std::vector<uint16_t> expected(reference.output.size());
  for (uint64_t base = 0; base < expected.size(); base += K) {
    float sum = 0.0f;
    for (uint32_t dimension = 0; dimension < K; ++dimension) {
      const float value = number(values[base + dimension]);
      sum += value * value;
    }
    const float inverse = 1.0f / std::sqrt(sum / float(K) + 1e-6f);
    for (uint32_t dimension = 0; dimension < K; ++dimension) {
      const uint64_t index = base + dimension;
      const float normalized = rounded(number(values[index]) * inverse);
      const float weighted = rounded(normalized * number(reference.norm[dimension]));
      expected[index] = bits(weighted * sigmoid(number(reference.z[index])));
    }
  }
  return expected;
}

double endToEndOutputDifference(const MetalBuffer &output,
                                 const Reference &reference) {
  const auto *values = static_cast<const uint16_t *>(output.contents());
  double maximum = 0.0;
  for (size_t i = 0; i < reference.output.size(); ++i)
    maximum = std::max(maximum,
                       std::abs(double(number(values[i])) -
                                number(reference.output[i])));
  return maximum;
}

void diagnoseOutput(const Fixture &fixture, const Reference &reference,
                    const std::vector<uint16_t> &stageExpected) {
  const auto *output = static_cast<const uint16_t *>(fixture.buffers.output.contents());
  const auto *gpuRows = static_cast<const uint16_t *>(fixture.buffers.recurrentRows.contents());
  for (size_t i = 0; i < reference.output.size(); ++i) {
    int exponent;
    const float expected = number(stageExpected[i]);
    std::frexp(std::abs(expected), &exponent);
    if (std::abs(double(number(output[i])) - expected) <=
        2.0 * std::ldexp(1.0, exponent - 8) + 2e-5)
      continue;
    const uint64_t base = (i / K) * K;
    float gpuSum = 0.0f, referenceSum = 0.0f;
    uint32_t differentRows = 0;
    for (uint32_t d = 0; d < K; ++d) {
      const float observed = number(gpuRows[base + d]);
      const float wanted = number(reference.recurrentRows[base + d]);
      gpuSum += observed * observed;
      referenceSum += wanted * wanted;
      differentRows += gpuRows[base + d] != reference.recurrentRows[base + d];
    }
    const float gpuInverse = 1.0f / std::sqrt(gpuSum / K + 1e-6f);
    const float referenceInverse = 1.0f / std::sqrt(referenceSum / K + 1e-6f);
    const float gamma = number(reference.norm[i % K]);
    const float gate = sigmoid(number(reference.z[i]));
    const float fromGpuNormalized = rounded(number(gpuRows[i]) * gpuInverse);
    const float fromReferenceNormalized =
        rounded(number(reference.recurrentRows[i]) * referenceInverse);
    const float fromGpuWeighted = rounded(fromGpuNormalized * gamma);
    const float fromReferenceWeighted = rounded(fromReferenceNormalized * gamma);
    std::cerr << std::setprecision(10)
              << "output_diagnostic index=" << i << " head_base=" << base
              << " recurrent_gpu=" << number(gpuRows[i])
              << " recurrent_reference=" << number(reference.recurrentRows[i])
              << " differing_head_elements=" << differentRows
              << " square_sum_gpu=" << gpuSum
              << " square_sum_reference=" << referenceSum
              << " inverse_gpu=" << gpuInverse
              << " inverse_reference=" << referenceInverse
              << " gamma=" << gamma << " z=" << number(reference.z[i])
              << " sigmoid=" << gate
              << " normalized_from_gpu=" << fromGpuNormalized
              << " normalized_from_reference=" << fromReferenceNormalized
              << " weighted_from_gpu=" << fromGpuWeighted
              << " weighted_from_reference=" << fromReferenceWeighted
              << " expected_from_gpu_recurrent=" << rounded(fromGpuWeighted * gate)
              << " expected_from_reference_recurrent=" << rounded(fromReferenceWeighted * gate)
              << " observed=" << number(output[i]) << '\n';
    for (uint32_t d = 0; d < K; ++d)
      if (gpuRows[base + d] != reference.recurrentRows[base + d])
        std::cerr << "recurrent_difference dimension=" << d
                  << " gpu=" << number(gpuRows[base + d])
                  << " reference=" << number(reference.recurrentRows[base + d]) << '\n';
    return;
  }
}

void run(MetalBackend &backend, uint32_t rows, uint32_t lanes, bool cold) {
  Reference reference(rows, lanes, cold);
  Fixture fixture(backend, reference);
  Fixture control(backend, reference);
  CommandGraph graph;
  encodeGDN(graph, fixture.weights(), fixture.buffers, fixture.state, rows, lanes);
  require(graph.dispatches().size() == (fusionMode == 1 ? 4 : fusionMode ? 2 : 6),
          "GDN graph has unexpected dispatches");
  const auto timing = backend.submitCommand(graph.dispatches());
  if (fusionMode) {
    CommandGraph canonical;
    addGDN(canonical, control.weights(), control.buffers, control.state, rows, lanes);
    static_cast<void>(backend.submitCommand(canonical.dispatches()));
    for (const auto &[candidate, baseline] :
         std::array<std::pair<MetalBuffer, MetalBuffer>, 7>{
             {{fixture.buffers.mixed, control.buffers.mixed},
              {fixture.buffers.decay, control.buffers.decay},
              {fixture.buffers.beta, control.buffers.beta},
              {fixture.buffers.recurrentRows, control.buffers.recurrentRows},
              {fixture.buffers.output, control.buffers.output},
              {fixture.state.convolution, control.state.convolution},
              {fixture.state.recurrent, control.state.recurrent}}})
      require(std::memcmp(candidate.contents(), baseline.contents(),
                          candidate.sizeBytes()) == 0,
              "fused GDN changed canonical intermediate/output/state bytes");
  }
  require(*static_cast<const uint32_t *>(fixture.buffers.diagnostics.contents()) ==
              0,
          "GDN valid inputs set diagnostics");
  reference.run();
  compareBF16(fixture.buffers.mixed, reference.mixed, 0.0, 0.0, "mixed qkv");
  compareBF16(fixture.buffers.beta, reference.beta, 0.0, 0.0, "beta");
  const auto *decay = static_cast<const float *>(fixture.buffers.decay.contents());
  for (size_t i = 0; i < reference.decay.size(); ++i)
    require(std::abs(decay[i] - reference.decay[i]) <= 2e-6f,
            "decay differs from scalar reference");
  const double recurrentError =
      compareBF16(fixture.buffers.recurrentRows, reference.recurrentRows,
                    1.0, 2e-7, "recurrent rows");
  // Validate normalization independently from the already checked recurrence.
  // One BF16 recurrence ULP can become several final-output ULPs when RMS
  // magnifies a nearly zero row (observed inverse~990). Applying a second
  // uniform output ULP bound to a different scalar-reduction input would
  // conflate the two stages. Neither recurrence nor F32-state bounds change.
  const auto stageExpected = gatedRMSReference(fixture.buffers.recurrentRows,
                                              reference);
  diagnoseOutput(fixture, reference, stageExpected);
  const double outputError =
      compareBF16(fixture.buffers.output, stageExpected, 2.0, 2e-5,
                    "gated RMS output");
  const double endToEndError = endToEndOutputDifference(fixture.buffers.output,
                                                       reference);
  double stateError = 0.0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const auto *state = reinterpret_cast<const float *>(
        static_cast<const uint8_t *>(fixture.state.recurrent.contents()) +
        lane * fixture.recurrentStride);
    for (uint64_t i = 0; i < StateElements; ++i) {
      const float expected = reference.recurrent[uint64_t{lane} * StateElements + i];
      const double difference = std::abs(double(state[i]) - expected);
      stateError = std::max(stateError, difference);
      require(std::isfinite(state[i]) && difference <= 2e-6,
              "recurrent F32 state differs from scalar reference");
    }
    const auto *history = reinterpret_cast<const uint16_t *>(
        static_cast<const uint8_t *>(fixture.state.convolution.contents()) +
        lane * fixture.convStride);
    require(std::memcmp(history, reference.history.data() + uint64_t{lane} * 3 * C,
                        flashGDNConvolutionLaneBytes()) == 0,
            "short convolution carry differs");
    const auto *convPadding = reinterpret_cast<const uint8_t *>(history) +
                              flashGDNConvolutionLaneBytes();
    const auto *statePadding = reinterpret_cast<const uint8_t *>(state) +
                               flashGDNRecurrentLaneBytes();
    for (uint32_t i = 0; i < 128; ++i)
      require(convPadding[i] == 0 && statePadding[i] == 0,
              "GDN wrote padded lane stride bytes");
  }
  // Add a second one-token request using each lane's newly carried state.
  // Scratch and projected views shrink, while the padded state strides stay.
  FlashGDNBuffers next = fixture.buffers;
  auto nextInput = [&](const std::vector<uint16_t> &source, uint32_t width) {
    auto result = buffer(backend, uint64_t{lanes} * width * 2, "next projected");
    for (uint32_t lane = 0; lane < lanes; ++lane)
      std::memcpy(static_cast<uint16_t *>(result.contents()) + lane * width,
                  source.data() + (uint64_t{lane} * rows + rows - 1) * width,
                  uint64_t{width} * 2);
    return result;
  };
  next.qkv = nextInput(reference.qkv, C);
  next.z = nextInput(reference.z, V);
  next.a = nextInput(reference.a, H);
  next.b = nextInput(reference.b, H);
  Reference continuation(1, lanes, true);
  continuation.history = reference.history;
  continuation.recurrent = reference.recurrent;
  continuation.convolution = reference.convolution;
  continuation.aLog = reference.aLog;
  continuation.timeBias = reference.timeBias;
  continuation.norm = reference.norm;
  std::memcpy(continuation.qkv.data(), next.qkv.contents(), continuation.qkv.size() * 2);
  std::memcpy(continuation.z.data(), next.z.contents(), continuation.z.size() * 2);
  std::memcpy(continuation.a.data(), next.a.contents(), continuation.a.size() * 2);
  std::memcpy(continuation.b.data(), next.b.contents(), continuation.b.size() * 2);
  CommandGraph second;
  encodeGDN(second, fixture.weights(), next, fixture.state, 1, lanes);
  static_cast<void>(backend.submitCommand(second.dispatches()));
  continuation.run();
  compareBF16(next.recurrentRows, continuation.recurrentRows,
                1.0, 2e-7, "continuation recurrent rows");
  compareBF16(next.output, gatedRMSReference(next.recurrentRows, continuation),
                2.0, 2e-5, "continuation gated RMS output");
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const auto *state = reinterpret_cast<const float *>(
        static_cast<const uint8_t *>(fixture.state.recurrent.contents()) +
        lane * fixture.recurrentStride);
    for (uint64_t i = 0; i < StateElements; ++i) {
      const float expected = continuation.recurrent[uint64_t{lane} * StateElements + i];
      require(std::isfinite(state[i]) && std::abs(double(state[i]) - expected) <= 2e-6,
              "continuation F32 state differs from scalar reference");
    }
  }
  require(*static_cast<const uint32_t *>(next.diagnostics.contents()) == 0,
          "continuation set diagnostics");
  std::cout << "rows=" << rows << " lanes=" << lanes << " cold=" << cold
            << " gpu_ms=" << timing.gpuSeconds * 1000
            << " recurrent_max_abs=" << recurrentError
            << " gated_RMS_stage_max_abs=" << outputError
            << " end_to_end_output_max_abs=" << endToEndError
            << " state_max_abs=" << stateError << '\n';
}

void splitPrefill(MetalBackend &backend) {
  constexpr uint32_t totalRows = 7, lanes = 2;
  Reference seed(totalRows, lanes, false);
  Fixture whole(backend, seed), split(backend, seed);
  CommandGraph graph;
  encodeGDN(graph, whole.weights(), whole.buffers, whole.state, totalRows, lanes);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  std::vector<uint16_t> observed(uint64_t{lanes} * totalRows * V);
  uint32_t offset = 0;
  for (uint32_t count : {2u, 1u, 4u}) {
    FlashGDNBuffers chunk = split.buffers;
    auto project = [&](const std::vector<uint16_t> &source, uint32_t width) {
      auto result = buffer(backend, uint64_t{lanes} * count * width * 2,
                           "split projected input");
      for (uint32_t lane = 0; lane < lanes; ++lane)
        std::memcpy(static_cast<uint16_t *>(result.contents()) +
                        uint64_t{lane} * count * width,
                    source.data() + (uint64_t{lane} * totalRows + offset) * width,
                    uint64_t{count} * width * 2);
      return result;
    };
    chunk.qkv = project(seed.qkv, C);
    chunk.z = project(seed.z, V);
    chunk.a = project(seed.a, H);
    chunk.b = project(seed.b, H);
    CommandGraph part;
    encodeGDN(part, split.weights(), chunk, split.state, count, lanes);
    static_cast<void>(backend.submitCommand(part.dispatches()));
    for (uint32_t lane = 0; lane < lanes; ++lane)
      std::memcpy(observed.data() + (uint64_t{lane} * totalRows + offset) * V,
                  static_cast<const uint16_t *>(chunk.output.contents()) +
                      uint64_t{lane} * count * V,
                  uint64_t{count} * V * 2);
    offset += count;
  }
  require(std::memcmp(observed.data(), whole.buffers.output.contents(),
                      observed.size() * 2) == 0,
          "split prefill changed BF16 output");
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const auto *wholeConv = static_cast<const uint8_t *>(whole.state.convolution.contents()) +
                            lane * whole.convStride;
    const auto *splitConv = static_cast<const uint8_t *>(split.state.convolution.contents()) +
                            lane * split.convStride;
    const auto *wholeState = static_cast<const uint8_t *>(whole.state.recurrent.contents()) +
                             lane * whole.recurrentStride;
    const auto *splitState = static_cast<const uint8_t *>(split.state.recurrent.contents()) +
                             lane * split.recurrentStride;
    require(std::memcmp(wholeConv, splitConv, flashGDNConvolutionLaneBytes()) == 0,
            "split prefill changed convolution carry");
    require(std::memcmp(wholeState, splitState, flashGDNRecurrentLaneBytes()) == 0,
            "split prefill changed F32 recurrent state");
  }
  require(*static_cast<const uint32_t *>(whole.buffers.diagnostics.contents()) == 0 &&
              *static_cast<const uint32_t *>(split.buffers.diagnostics.contents()) == 0,
          "split prefill set diagnostics");
  std::cout << "split_prefill=2+1+4 lanes=2 BF16_output=exact F32_state=exact\n";
}

void rejectsInvalid(MetalBackend &backend) {
  Reference seed(1, 1, true);
  Fixture fixture(backend, seed);
  CommandGraph graph;
  auto rejects = [&](auto &&function) {
    bool caught = false;
    try {
      function();
    } catch (const std::invalid_argument &) {
      caught = true;
    }
    require(caught, "invalid GDN request was accepted");
    require(graph.empty(), "invalid GDN request partially modified its graph");
  };
  rejects([&] { encodeGDN(graph, fixture.weights(), fixture.buffers, fixture.state, 0); });
  rejects([&] { encodeGDN(graph, fixture.weights(), fixture.buffers, fixture.state, 2049); });
  rejects([&] { encodeGDN(graph, fixture.weights(), fixture.buffers, fixture.state, 1, 0); });
  rejects([&] {
    encodeGDN(graph, fixture.weights(), fixture.buffers, fixture.state, 1, 1,
            std::numeric_limits<float>::quiet_NaN());
  });
  rejects([&] {
    auto state = fixture.state;
    state.convolutionLaneStrideBytes = 3;
    encodeGDN(graph, fixture.weights(), fixture.buffers, state, 1);
  });
  rejects([&] {
    auto buffers = fixture.buffers;
    buffers.output = buffers.qkv;
    encodeGDN(graph, fixture.weights(), buffers, fixture.state, 1);
  });
}

void capturePrefixes(MetalBackend &backend) {
  for (uint32_t rows : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 12u, 15u, 16u})
    for (uint32_t lanes : {1u, 2u})
      for (bool cold : {true, false}) {
        Reference seed(rows, lanes, cold);
        Fixture candidate(backend, seed), sequential(backend, seed),
            canonical(backend, seed);
        Reference future(1, lanes, false);
        future.convolution = seed.convolution;
        future.aLog = seed.aLog;
        future.timeBias = seed.timeBias;
        future.norm = seed.norm;
        Fixture restored(backend, future), continued(backend, future);
        const uint64_t rowStride = flashGDNRecurrentLaneBytes() + 128;
        const uint32_t captureRows = rows - 1;
        const uint64_t laneStride = captureRows * rowStride + 128;
        auto tape = captureRows
            ? buffer(backend, lanes * laneStride, "GDN prefix tape")
            : MetalBuffer{};
        CommandGraph graph;
        addGDNFusedCaptured(graph, candidate.weights(), candidate.buffers,
                            candidate.state,
                            FlashGDNCapture{tape, rowStride, laneStride, captureRows,
                              fusionMode == 6 ? 512u : fusionMode == 7 ? 1024u : 256u},
                            rows, lanes);
        require(graph.dispatches().size() == 2,
                "captured GDN graph has unexpected dispatches");
        static_cast<void>(backend.submitCommand(graph.dispatches()));
        CommandGraph whole;
        addGDN(whole, canonical.weights(), canonical.buffers, canonical.state,
                 rows, lanes);
        static_cast<void>(backend.submitCommand(whole.dispatches()));
        for (const auto &[got, expected] :
             std::array<std::pair<MetalBuffer, MetalBuffer>, 7>{
                 {{candidate.buffers.mixed, canonical.buffers.mixed},
                  {candidate.buffers.decay, canonical.buffers.decay},
                  {candidate.buffers.beta, canonical.buffers.beta},
                  {candidate.buffers.recurrentRows, canonical.buffers.recurrentRows},
                  {candidate.buffers.output, canonical.buffers.output},
                  {candidate.state.convolution, canonical.state.convolution},
                  {candidate.state.recurrent, canonical.state.recurrent}}})
          require(std::memcmp(got.contents(), expected.contents(),
                              got.sizeBytes()) == 0,
                  "captured GDN changed canonical intermediate/output/state");
        for (uint32_t prefix = 0; prefix < rows; ++prefix) {
          FlashGDNBuffers single = sequential.buffers;
          auto projected = [&](const std::vector<uint16_t> &source,
                               uint32_t width) {
            auto result = buffer(backend, uint64_t{lanes} * width * 2,
                                  "prefix projected input");
            for (uint32_t lane = 0; lane < lanes; ++lane)
              std::memcpy(static_cast<uint16_t *>(result.contents()) +
                              uint64_t{lane} * width,
                          source.data() + (uint64_t{lane} * rows + prefix) * width,
                          uint64_t{width} * 2);
            return result;
          };
          single.qkv = projected(seed.qkv, C);
          single.z = projected(seed.z, V);
          single.a = projected(seed.a, H);
          single.b = projected(seed.b, H);
          CommandGraph step;
          addGDN(step, sequential.weights(), single, sequential.state, 1, lanes);
          static_cast<void>(backend.submitCommand(step.dispatches()));
          if (prefix >= captureRows) continue;
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            const auto *captured = static_cast<const uint8_t *>(tape.contents()) +
                                   lane * laneStride + prefix * rowStride;
            const auto *expected =
                static_cast<const uint8_t *>(sequential.state.recurrent.contents()) +
                lane * sequential.recurrentStride;
            require(std::memcmp(captured, expected,
                                flashGDNRecurrentLaneBytes()) == 0,
                    "captured GDN F32 prefix differs from sequential canonical");
            for (uint32_t padding = 0; padding < 128; ++padding)
              require(captured[flashGDNRecurrentLaneBytes() + padding] == 0,
                      "captured GDN wrote tape padding");
            std::memcpy(static_cast<uint8_t *>(restored.state.recurrent.contents()) +
                            lane * restored.recurrentStride,
                        captured, flashGDNRecurrentLaneBytes());
            std::memcpy(static_cast<uint8_t *>(continued.state.recurrent.contents()) +
                            lane * continued.recurrentStride,
                        expected, flashGDNRecurrentLaneBytes());
            const uint32_t kept = prefix + 1;
            const auto copyGeometry = *flashMTPConvolutionPrefix(kept);
            const uint64_t rowBytes = uint64_t{C} * 2;
            auto *history = static_cast<uint8_t *>(restored.state.convolution.contents()) +
                            lane * restored.convStride;
            if (copyGeometry.oldRows)
              std::memcpy(history, seed.history.data() +
                              (uint64_t{lane} * 3 + copyGeometry.oldBegin) * C,
                          uint64_t{copyGeometry.oldRows} * rowBytes);
            std::memcpy(history + uint64_t{copyGeometry.destinationInputBegin} * rowBytes,
                        seed.qkv.data() + (uint64_t{lane} * rows + copyGeometry.inputBegin) * C,
                        uint64_t{copyGeometry.inputRows} * rowBytes);
            const auto *expectedHistory =
                static_cast<const uint8_t *>(sequential.state.convolution.contents()) +
                lane * sequential.convStride;
            require(std::memcmp(history, expectedHistory, flashGDNConvolutionLaneBytes()) == 0,
                    "retained GDN convolution prefix differs from sequential canonical");
            std::memcpy(static_cast<uint8_t *>(continued.state.convolution.contents()) +
                            lane * continued.convStride,
                        expectedHistory, flashGDNConvolutionLaneBytes());
          }
          // A restored tape must remain exact after future work, including
          // retained prefixes longer than the three-row convolution history.
          for (uint32_t continuation = 0; continuation < 2; ++continuation) {
            CommandGraph resumed, control;
            const auto fusion = fusionMode == 6 ? FlashGDNFusion::PersistentHead512
                : fusionMode == 7 ? FlashGDNFusion::PersistentHead1024
                                  : FlashGDNFusion::PersistentHead;
            addGDNFused(resumed, restored.weights(), restored.buffers, restored.state,
                        1, lanes, fusion);
            addGDN(control, continued.weights(), continued.buffers, continued.state, 1, lanes);
            static_cast<void>(backend.submitCommand(resumed.dispatches()));
            static_cast<void>(backend.submitCommand(control.dispatches()));
            for (const auto &[got, expected] :
                 std::array<std::pair<MetalBuffer, MetalBuffer>, 7>{
                     {{restored.buffers.mixed, continued.buffers.mixed},
                      {restored.buffers.decay, continued.buffers.decay},
                      {restored.buffers.beta, continued.buffers.beta},
                      {restored.buffers.recurrentRows, continued.buffers.recurrentRows},
                      {restored.buffers.output, continued.buffers.output},
                      {restored.state.convolution, continued.state.convolution},
                      {restored.state.recurrent, continued.state.recurrent}}})
              require(std::memcmp(got.contents(), expected.contents(), got.sizeBytes()) == 0,
                      "restored GDN prefix changed subsequent canonical work");
          }
        }
        require(*static_cast<const uint32_t *>(candidate.buffers.diagnostics.contents()) == 0,
                "captured GDN set diagnostics for valid input");
        std::cout << "capture_rows=" << rows << " lanes=" << lanes
                  << " cold=" << cold << " F32_prefixes=exact canonical=exact\n";
      }
}

void benchmarkFusion(MetalBackend &backend, uint32_t maximumRows) {
  for (uint32_t rows : {1u, 2u, 4u, 16u, 32u, 128u, 512u, 2048u}) {
    if (rows > maximumRows) continue;
    Reference seed(rows, 1, false);
    for (const auto &[mode, name] :
         std::array<std::pair<FlashGDNFusion, const char *>, 4>{
             {{FlashGDNFusion::Prepare, "prepare"},
              {FlashGDNFusion::PersistentHead, "persistent256"},
              {FlashGDNFusion::PersistentHead512, "persistent512"},
              {FlashGDNFusion::PersistentHead1024, "persistent1024"}}}) {
      Fixture baseline(backend, seed), candidate(backend, seed);
      CommandGraph original, fused;
      addGDN(original, baseline.weights(), baseline.buffers, baseline.state,
                 rows, 1);
      addGDNFused(fused, candidate.weights(), candidate.buffers, candidate.state,
                      rows, 1, mode);
      auto reset = [&](Fixture &fixture) {
        std::memcpy(fixture.state.convolution.contents(), seed.history.data(),
                    flashGDNConvolutionLaneBytes());
        std::memcpy(fixture.state.recurrent.contents(), seed.recurrent.data(),
                    flashGDNRecurrentLaneBytes());
        *static_cast<uint32_t *>(fixture.buffers.diagnostics.contents()) = 0;
      };
      auto submit = [&](Fixture &fixture, CommandGraph &graph) {
        reset(fixture);
        const auto timing = backend.submitCommand(graph.dispatches());
        require(*static_cast<const uint32_t *>(fixture.buffers.diagnostics.contents()) == 0,
                "fusion benchmark set numeric diagnostics");
        return timing;
      };
      for (uint32_t warmup = 0; warmup < 3; ++warmup) {
        static_cast<void>(submit(baseline, original));
        static_cast<void>(submit(candidate, fused));
      }
      constexpr uint32_t pairs = 12;
      double baselineGPU = 0.0, candidateGPU = 0.0;
      double baselineWall = 0.0, candidateWall = 0.0;
      for (uint32_t pair = 0; pair < pairs; ++pair) {
        splash::metal::CommandTiming control, experimental;
        if (pair % 2 == 0) {
          control = submit(baseline, original);
          experimental = submit(candidate, fused);
        } else {
          experimental = submit(candidate, fused);
          control = submit(baseline, original);
        }
        baselineGPU += control.gpuSeconds; candidateGPU += experimental.gpuSeconds;
        baselineWall += control.wallSeconds; candidateWall += experimental.wallSeconds;
      }
      std::cout << std::setprecision(8)
                << "bench_rows=" << rows << " route=" << name << " pairs=" << pairs
                << " baseline_gpu_ms=" << baselineGPU / pairs * 1000
                << " candidate_gpu_ms=" << candidateGPU / pairs * 1000
                << " gpu_speedup_percent=" << (baselineGPU / candidateGPU - 1) * 100
                << " baseline_command_wall_ms=" << baselineWall / pairs * 1000
                << " candidate_command_wall_ms=" << candidateWall / pairs * 1000 << '\n';
    }
  }
}

void separateStatesOracle(MetalBackend &backend) {
  for (uint32_t lanes = 1; lanes <= 4; ++lanes)
    for (bool cold : {true, false}) {
      Reference seed(1, lanes, cold);
      Fixture candidate(backend, seed), control(backend, seed);
      std::array<FlashGDNState, 4> states;
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        states[lane].convolution = buffer(backend, flashGDNConvolutionLaneBytes() + 128,
                                           "request convolution state");
        states[lane].recurrent = buffer(backend, flashGDNRecurrentLaneBytes() + 128,
                                         "request recurrent state");
        std::memcpy(states[lane].convolution.contents(),
                    seed.history.data() + uint64_t{lane} * 3 * C,
                    flashGDNConvolutionLaneBytes());
        std::memcpy(states[lane].recurrent.contents(),
                    seed.recurrent.data() + uint64_t{lane} * StateElements,
                    flashGDNRecurrentLaneBytes());
      }
      auto rejects = [&](auto &&function) {
        CommandGraph invalid;
        bool caught = false;
        try { function(invalid); }
        catch (const std::invalid_argument &) { caught = true; }
        require(caught && invalid.empty(),
                "invalid separate-state GDN metadata modified a graph");
      };
      rejects([&](CommandGraph &g) {
        addGDNFusedSeparateStates(g, candidate.weights(), candidate.buffers, states, 0);
      });
      rejects([&](CommandGraph &g) {
        addGDNFusedSeparateStates(g, candidate.weights(), candidate.buffers, states, 5);
      });
      if (lanes > 1)
        rejects([&](CommandGraph &g) {
          auto duplicated = states;
          duplicated[1] = duplicated[0];
          addGDNFusedSeparateStates(g, candidate.weights(), candidate.buffers,
                                      duplicated, lanes);
        });
      for (uint32_t step = 0; step < 3; ++step) {
        CommandGraph fused, original;
        addGDNFusedSeparateStates(fused, candidate.weights(), candidate.buffers,
                                    states, lanes);
        addGDN(original, control.weights(), control.buffers, control.state, 1, lanes);
        require(fused.dispatches().size() == 2,
                "separate-state GDN unexpectedly packs/scatters state");
        static_cast<void>(backend.submitCommand(original.dispatches()));
        static_cast<void>(backend.submitCommand(fused.dispatches()));
        for (const auto &[got, expected] :
             std::array<std::pair<MetalBuffer, MetalBuffer>, 5>{
                 {{candidate.buffers.mixed, control.buffers.mixed},
                  {candidate.buffers.decay, control.buffers.decay},
                  {candidate.buffers.beta, control.buffers.beta},
                  {candidate.buffers.recurrentRows, control.buffers.recurrentRows},
                  {candidate.buffers.output, control.buffers.output}}})
          require(std::memcmp(got.contents(), expected.contents(), got.sizeBytes()) == 0,
                  "separate-state GDN changed canonical intermediate/output");
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const auto *history = static_cast<const uint8_t *>(control.state.convolution.contents()) +
                                lane * control.convStride;
          const auto *recurrent = static_cast<const uint8_t *>(control.state.recurrent.contents()) +
                                  lane * control.recurrentStride;
          require(std::memcmp(states[lane].convolution.contents(), history,
                              flashGDNConvolutionLaneBytes() + 128) == 0,
                  "separate-state GDN changed request convolution state/padding");
          require(std::memcmp(states[lane].recurrent.contents(), recurrent,
                              flashGDNRecurrentLaneBytes() + 128) == 0,
                  "separate-state GDN changed request F32 state/padding");
        }
        require(*static_cast<const uint32_t *>(candidate.buffers.diagnostics.contents()) == 0,
                "separate-state GDN set diagnostics for valid inputs");
      }
      std::cout << "separate_lanes=" << lanes << " cold=" << cold
                << " repeated_steps=3 canonical_intermediates=exact request_states=exact\n";
    }
}

void cpuOnly() {
  for (uint32_t rows : {1u, 2u, 7u}) {
    Reference zero(rows, 1, true);
    std::fill(zero.qkv.begin(), zero.qkv.end(), 0);
    zero.run();
    require(zero.beta[0] == 0x3a8b,
            "beta reference lost the verified unary sigmoid accuracy case");
    require(std::all_of(zero.output.begin(), zero.output.end(),
                         [](uint16_t value) { return value == 0; }),
            "zero recurrence produced a nonzero output");
    require(std::all_of(zero.recurrent.begin(), zero.recurrent.end(),
                         [](float value) { return value == 0.0f; }),
            "zero recurrence produced a nonzero state");
  }
  Reference carry(1, 2, false);
  const auto initial = carry.history;
  carry.run();
  for (uint32_t lane = 0; lane < 2; ++lane) {
    require(std::equal(initial.begin() + (uint64_t{lane} * 3 + 1) * C,
                        initial.begin() + (uint64_t{lane} * 3 + 3) * C,
                        carry.history.begin() + uint64_t{lane} * 3 * C),
            "one-token history did not shift its first two rows");
    require(std::equal(carry.qkv.begin() + uint64_t{lane} * C,
                        carry.qkv.begin() + uint64_t{lane + 1} * C,
                        carry.history.begin() + (uint64_t{lane} * 3 + 2) * C),
            "one-token history omitted the current projected row");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::invalid_argument("usage: flash-gdn METALLIB [MAX_ROWS] [prepare|persistent|capture] | --cpu-only");
    cpuOnly();
    if (std::string(argv[1]) == "--cpu-only") {
      std::cout << "flash_gdn_cpu_reference: PASS\n";
      return 0;
    }
    const uint32_t maximum = argc >= 3 ? static_cast<uint32_t>(std::stoul(argv[2])) : 32;
    if (argc == 4) {
      const std::string mode = argv[3];
      if (mode == "prepare") fusionMode = 1;
      else if (mode == "persistent") fusionMode = 2;
      else if (mode == "capture") fusionMode = 3;
      else if (mode == "persistent512") fusionMode = 4;
      else if (mode == "persistent1024") fusionMode = 5;
      else if (mode == "capture512") fusionMode = 6;
      else if (mode == "capture1024") fusionMode = 7;
      else if (mode == "bench") fusionMode = 8;
      else if (mode == "separate") fusionMode = 9;
      else throw std::invalid_argument("invalid GDN fusion mode");
    }
    MetalBackend backend(argv[1]);
    if (fusionMode == 9) {
      separateStatesOracle(backend);
      std::cout << "flash_gdn_separate_metal_test: PASS\n";
      return 0;
    }
    if (fusionMode == 8) {
      benchmarkFusion(backend, maximum);
      return 0;
    }
    if (fusionMode == 3 || fusionMode == 6 || fusionMode == 7) {
      capturePrefixes(backend);
      std::cout << "flash_gdn_capture_metal_test: PASS\n";
      return 0;
    }
    rejectsInvalid(backend);
    splitPrefill(backend);
    for (uint32_t rows : {1u, 2u, 7u, 32u, 128u, 2048u})
      if (rows <= maximum)
        for (uint32_t lanes : {1u, 2u})
          for (bool cold : {true, false})
            run(backend, rows, lanes, cold);
    std::cout << "flash_gdn_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "flash_gdn_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
