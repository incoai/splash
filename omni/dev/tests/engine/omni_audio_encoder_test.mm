// The ASR audio front end and audio tower (omni/MelSpectrogram,
// ops/OmniAudioEncoder):
//
//   omni-audio-encoder METALLIB FIXTURE_DIR [MODEL_DIR]
//
// 1. The log-mel of the fixture's 2-second chirp against transformers'
//    WhisperFeatureExtractor (dev/tools/omni_mel_reference.py).
// 2. The token count and window tables, and a two-layer tower of random
//    weights against an fp64 reference that rounds to bf16 where the kernels
//    do, over a padded last chunk and over a lone short chunk.
// 3. With the Qwen3-ASR weights (MODEL_DIR, else the Hub cache's
//    mlx-community/Qwen3-ASR-1.7B-4bit snapshot): parity with mlx-audio's
//    audio tower (dev/tools/omni_audio_encoder_reference.py) on the fixture
//    signals, then the encode time of 30 s of audio. Skipped without them.

#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "omni/MelSpectrogram.hpp"
#include "ops/OmniAudioEncoder.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace splash::ops::omni;
using namespace splash::ops::tuning;
using namespace splash::test;

namespace {

constexpr uint32_t kMelBins = OMNI_AUDIO_MEL_BINS;
constexpr uint32_t kChannels = OMNI_AUDIO_CONV_CHANNELS;
constexpr uint32_t kPaddedChannels = OMNI_AUDIO_CONV_PADDED_CHANNELS;
constexpr uint32_t kTaps = OMNI_AUDIO_CONV_TAPS;
constexpr uint32_t kPaddedTaps = OMNI_AUDIO_CONV_PADDED_TAPS;
constexpr uint32_t kHidden = OMNI_AUDIO_HIDDEN;
constexpr uint32_t kConvOutInput = OMNI_AUDIO_CONV_OUT_INPUT;
constexpr uint32_t kFfn = OMNI_AUDIO_FFN;
constexpr uint32_t kOutput = OMNI_AUDIO_OUTPUT;
constexpr uint32_t kHeads = OMNI_ENCODER_HEADS, kHeadDim = OMNI_ENCODER_HEAD_DIM;
constexpr uint32_t kSampleRate = splash::omni::kMelSampleRate;

std::vector<float> readFloats(const std::filesystem::path &path) {
  const std::vector<uint8_t> bytes = readFile(path);
  require(bytes.size() % 4 == 0, path.string() + ": not whole floats");
  std::vector<float> values(bytes.size() / 4);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

metal::MetalBuffer f32Buffer(metal::MetalBackend &backend, const std::vector<float> &values) {
  metal::MetalBuffer buffer = sharedBuffer(backend, std::max<size_t>(1, values.size()) * 4);
  std::memcpy(buffer.contents(), values.data(), values.size() * 4);
  return buffer;
}
metal::MetalBuffer bf16Buffer(metal::MetalBackend &backend, const std::vector<float> &values) {
  metal::MetalBuffer buffer = sharedBuffer(backend, std::max<size_t>(1, values.size()) * 2);
  auto *bits = static_cast<uint16_t *>(buffer.contents());
  for (size_t i = 0; i < values.size(); ++i) bits[i] = floatToBf16(values[i]);
  return buffer;
}
std::vector<float> readBf16(const metal::MetalBuffer &buffer, size_t count) {
  const auto *bits = static_cast<const uint16_t *>(buffer.contents());
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) values[i] = bf16ToFloat(bits[i]);
  return values;
}

void run(metal::MetalBackend &backend, metal::CommandGraph &graph, double *gpuSeconds = nullptr) {
  metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
  const metal::CommandTiming timing = ticket.wait();
  if (gpuSeconds) *gpuSeconds = timing.gpuSeconds;
}

struct ErrorStats final {
  double maxAbs = 0.0, maxRef = 0.0, sumSquares = 0.0, refSquares = 0.0, maxRel = 0.0;
  size_t worst = 0;
  // The row (of `width` elements) with the largest relative rms, and it.
  size_t worstRow = 0;
  double worstRowRms = 0.0;
  [[nodiscard]] double relativeRms() const { return std::sqrt(sumSquares / std::max(refSquares, 1e-300)); }
};
// Element errors of `got` against `expected`, rows of `width`; the relative
// error counts elements above a tenth of the reference's largest magnitude.
ErrorStats compare(const std::vector<float> &got, const std::vector<float> &expected, size_t width) {
  require(got.size() == expected.size() && got.size() % width == 0, "compared vectors differ in size");
  ErrorStats stats;
  for (float value : expected) stats.maxRef = std::max(stats.maxRef, double(std::fabs(value)));
  for (size_t row = 0; row < got.size() / width; ++row) {
    double rowSquares = 0.0, rowReference = 0.0;
    for (size_t i = row * width; i < (row + 1) * width; ++i) {
      const double difference = std::fabs(double(got[i]) - expected[i]);
      if (std::isnan(got[i]) || difference > stats.maxAbs) {
        stats.maxAbs = std::isnan(got[i]) ? INFINITY : difference;
        stats.worst = i;
      }
      rowSquares += difference * difference;
      rowReference += double(expected[i]) * expected[i];
      if (std::fabs(expected[i]) > 0.1 * stats.maxRef)
        stats.maxRel = std::max(stats.maxRel, difference / std::fabs(expected[i]));
    }
    stats.sumSquares += rowSquares;
    stats.refSquares += rowReference;
    const double rowRms = std::sqrt(rowSquares / std::max(rowReference, 1e-300));
    if (rowRms > stats.worstRowRms) {
      stats.worstRowRms = rowRms;
      stats.worstRow = row;
    }
  }
  return stats;
}
void report(const std::string &what, const ErrorStats &stats) {
  std::cout << what << ": max abs " << stats.maxAbs << " (reference |max| " << stats.maxRef << ", at " << stats.worst
            << "), max rel " << stats.maxRel << ", relative rms " << stats.relativeRms() << " (worst row "
            << stats.worstRow << ": " << stats.worstRowRms << ")\n";
}

// ---------------------------------------------------------------- log-mel

void testMel(const std::filesystem::path &fixtures) {
  const std::vector<float> audio = readFloats(fixtures / "audio_2000ms.f32");
  const std::vector<float> golden = readFloats(fixtures / "mel_2000ms.f32");
  require(audio.size() == 32000, "the 2 s fixture holds 32000 samples");
  const uint32_t frames = splash::omni::melFrames(audio.size());
  require(frames == 200 && golden.size() == size_t(frames) * kMelBins, "the mel golden holds 200 frames");
  const std::vector<float> mel = splash::omni::logMelSpectrogram(audio);
  require(mel.size() == golden.size(), "the mel holds one row per frame");
  const ErrorStats stats = compare(mel, golden, kMelBins);
  report("log-mel 2 s", stats);
  require(stats.maxAbs <= 5e-4, "the log-mel differs from the WhisperFeatureExtractor golden");
  // Frame counts: one per hop, the trailing centered frame dropped.
  require(splash::omni::melFrames(11200) == 70 && splash::omni::melFrames(159) == 0 &&
              splash::omni::melFrames(160) == 1 && splash::omni::melFrames(480000) == 3000,
          "mel frame counts");
  require(splash::omni::logMelSpectrogram(std::vector<float>(100, 0.0F)).empty(), "no frames below a hop");
  // A short signal is zero-extended, not rejected, and keeps its frames.
  require(splash::omni::logMelSpectrogram(std::vector<float>(320, 0.1F)).size() == 2 * kMelBins, "short signal");
  std::cout << "log-mel ok\n";
}

// ---------------------------------------------------------------- tables

// The reference's floor-division formula.
int64_t referenceTokens(int64_t frames) {
  const auto floorDiv = [](int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
  const int64_t leave = frames % 100;
  const int64_t feat = floorDiv(leave - 1, 2) + 1;
  return floorDiv(floorDiv(feat - 1, 2) + 1 - 1, 2) + 1 + (frames / 100) * 13;
}

void testTables() {
  for (uint32_t frames = 0; frames <= 1200; ++frames)
    require(int64_t(audioTokens(frames)) == referenceTokens(frames),
            "audio tokens of " + std::to_string(frames) + " frames");
  require(audioTokens(0) == 0 && audioTokens(1) == 1 && audioTokens(100) == 13 && audioTokens(101) == 14 &&
              audioTokens(3000) == 390,
          "audio token samples");
  const std::vector<OmniDenseRow> rows = audioAttentionRows(117);
  require(rows.size() == 117, "117 rows");
  for (uint32_t r = 0; r < 117; ++r) {
    const uint32_t begin = r < 104 ? 0 : 104, end = r < 104 ? 104 : 117;
    require(rows[r].key_begin == begin && rows[r].key_end == end && rows[r].query_index == r,
            "window of row " + std::to_string(r));
  }
  require(audioAttentionRows(9).back().key_end == 9, "a lone window ends at its tokens");
  std::cout << "tables ok\n";
}

// ---------------------------------------------------------------- random tower

std::mt19937_64 rng(11);
float uniform(float low, float high) { return std::uniform_real_distribution<float>(low, high)(rng); }
std::vector<float> randomValues(size_t count, float scale) {
  std::vector<float> values(count);
  for (float &v : values) v = uniform(-scale, scale);
  return values;
}
std::vector<float> roundedBf16(std::vector<float> values) {
  for (float &v : values) v = bf16ToFloat(floatToBf16(v));
  return values;
}
double bf16(double value) { return bf16ToFloat(floatToBf16(float(value))); }
double gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::numbers::sqrt2)); }

struct HostLinear final {
  std::vector<float> weight; // [out][in], bf16 values
  std::vector<float> bias;   // fp32, empty without
  uint32_t out = 0, in = 0;
};
struct HostNorm final {
  std::vector<float> weight, bias; // bf16 values
};
struct HostLayer final {
  HostNorm norm1, norm2;
  HostLinear q, k, v, out, fc1, fc2;
};
struct HostTower final {
  HostLinear conv1, conv2, conv3, convOut; // conv weights [co][(kh*3+kw)*ci + ci]
  std::vector<HostLayer> layers;
  HostNorm post;
  HostLinear proj1, proj2;
};

HostLinear randomLinear(uint32_t out, uint32_t in, bool bias) {
  HostLinear linear;
  linear.out = out;
  linear.in = in;
  linear.weight = roundedBf16(randomValues(size_t(out) * in, std::sqrt(3.0F / float(in))));
  if (bias) linear.bias = randomValues(out, 0.1F);
  return linear;
}
HostNorm randomNorm() {
  HostNorm norm;
  norm.weight.resize(kHidden);
  for (float &w : norm.weight) w = uniform(0.5F, 1.5F);
  norm.weight = roundedBf16(norm.weight);
  norm.bias = roundedBf16(randomValues(kHidden, 0.1F));
  return norm;
}
HostTower randomTower(uint32_t layers) {
  HostTower tower;
  tower.conv1 = randomLinear(kChannels, 9, true);
  tower.conv2 = randomLinear(kChannels, kTaps, true);
  tower.conv3 = randomLinear(kChannels, kTaps, true);
  tower.convOut = randomLinear(kHidden, kConvOutInput, false);
  for (uint32_t i = 0; i < layers; ++i) {
    HostLayer layer;
    layer.norm1 = randomNorm();
    layer.norm2 = randomNorm();
    layer.q = randomLinear(kHidden, kHidden, true);
    layer.k = randomLinear(kHidden, kHidden, true);
    layer.v = randomLinear(kHidden, kHidden, true);
    layer.out = randomLinear(kHidden, kHidden, true);
    layer.fc1 = randomLinear(kFfn, kHidden, true);
    layer.fc2 = randomLinear(kHidden, kFfn, true);
    tower.layers.push_back(std::move(layer));
  }
  tower.post = randomNorm();
  tower.proj1 = randomLinear(kHidden, kHidden, true);
  tower.proj2 = randomLinear(kOutput, kHidden, true);
  return tower;
}

DenseWeights denseWeights(metal::MetalBackend &backend, const HostLinear &linear) {
  return {bf16Buffer(backend, linear.weight), linear.bias.empty() ? metal::MetalBuffer{} : f32Buffer(backend, linear.bias),
          linear.out, linear.in};
}
// A conv weight [480][4320] padded to the GEMM's [512][4352], bias to 512.
DenseWeights convWeights(metal::MetalBackend &backend, const HostLinear &conv) {
  std::vector<float> padded(size_t(kPaddedChannels) * kPaddedTaps, 0.0F);
  for (uint32_t co = 0; co < kChannels; ++co)
    std::copy_n(conv.weight.begin() + size_t(co) * kTaps, kTaps, padded.begin() + size_t(co) * kPaddedTaps);
  std::vector<float> bias(kPaddedChannels, 0.0F);
  std::copy(conv.bias.begin(), conv.bias.end(), bias.begin());
  return {bf16Buffer(backend, padded), f32Buffer(backend, bias), kPaddedChannels, kPaddedTaps};
}
AudioLayerNorm normWeights(metal::MetalBackend &backend, const HostNorm &norm) {
  return {bf16Buffer(backend, norm.weight), bf16Buffer(backend, norm.bias)};
}
AudioEncoderWeights deviceWeights(metal::MetalBackend &backend, const HostTower &tower) {
  AudioEncoderWeights weights;
  weights.conv1Weight = bf16Buffer(backend, tower.conv1.weight);
  weights.conv1Bias = f32Buffer(backend, tower.conv1.bias);
  weights.conv2 = convWeights(backend, tower.conv2);
  weights.conv3 = convWeights(backend, tower.conv3);
  weights.convOut = denseWeights(backend, tower.convOut);
  for (const HostLayer &layer : tower.layers) {
    AudioEncoderLayerWeights device;
    device.attentionNorm = normWeights(backend, layer.norm1);
    std::vector<float> qkv, bias;
    for (const HostLinear *p : {&layer.q, &layer.k, &layer.v}) {
      qkv.insert(qkv.end(), p->weight.begin(), p->weight.end());
      bias.insert(bias.end(), p->bias.begin(), p->bias.end());
    }
    device.qkv = {bf16Buffer(backend, qkv), f32Buffer(backend, bias), 3 * kHidden, kHidden};
    device.out = denseWeights(backend, layer.out);
    device.ffnNorm = normWeights(backend, layer.norm2);
    device.fc1 = denseWeights(backend, layer.fc1);
    device.fc2 = denseWeights(backend, layer.fc2);
    weights.layers.push_back(std::move(device));
  }
  weights.postNorm = normWeights(backend, tower.post);
  weights.proj1 = denseWeights(backend, tower.proj1);
  weights.proj2 = denseWeights(backend, tower.proj2);
  weights.dims.layers = static_cast<uint32_t>(weights.layers.size());
  return weights;
}

// A 3x3 stride-2 pad-1 convolution with GELU over [ci][h][w] into
// [co][ho][wo], rounded to bf16 as the kernels store it.
std::vector<double> convolve(const std::vector<double> &input, uint32_t channels, uint32_t height, uint32_t width,
                             const HostLinear &conv, uint32_t &outHeight, uint32_t &outWidth) {
  outHeight = (height - 1) / 2 + 1;
  outWidth = (width - 1) / 2 + 1;
  std::vector<double> output(size_t(conv.out) * outHeight * outWidth);
  for (uint32_t co = 0; co < conv.out; ++co) {
    const float *w = conv.weight.data() + size_t(co) * conv.in;
    for (uint32_t ho = 0; ho < outHeight; ++ho)
      for (uint32_t wo = 0; wo < outWidth; ++wo) {
        double sum = conv.bias[co];
        for (uint32_t kh = 0; kh < 3; ++kh) {
          const int hi = int(ho) * 2 - 1 + int(kh);
          if (hi < 0 || hi >= int(height)) continue;
          for (uint32_t kw = 0; kw < 3; ++kw) {
            const int wi = int(wo) * 2 - 1 + int(kw);
            if (wi < 0 || wi >= int(width)) continue;
            const double *pixel = input.data() + (size_t(hi) * width + wi);
            const float *taps = w + (kh * 3 + kw) * channels;
            for (uint32_t ci = 0; ci < channels; ++ci) sum += pixel[size_t(ci) * height * width] * taps[ci];
          }
        }
        output[(size_t(co) * outHeight + ho) * outWidth + wo] = bf16(gelu(sum));
      }
  }
  return output;
}

// y = W x + b over a row, in fp64.
void project(const HostLinear &linear, const double *x, double *y) {
  for (uint32_t o = 0; o < linear.out; ++o) {
    double sum = linear.bias.empty() ? 0.0 : linear.bias[o];
    const float *w = linear.weight.data() + size_t(o) * linear.in;
    for (uint32_t i = 0; i < linear.in; ++i) sum += w[i] * x[i];
    y[o] = sum;
  }
}
// LayerNorm of a row, rounded to bf16.
void layerNorm(const HostNorm &norm, const double *x, double *y) {
  double mean = 0.0;
  for (uint32_t i = 0; i < kHidden; ++i) mean += x[i];
  mean /= kHidden;
  double variance = 0.0;
  for (uint32_t i = 0; i < kHidden; ++i) variance += (x[i] - mean) * (x[i] - mean);
  variance /= kHidden;
  const double inverse = 1.0 / std::sqrt(variance + 1e-5);
  for (uint32_t i = 0; i < kHidden; ++i) y[i] = bf16((x[i] - mean) * inverse * norm.weight[i] + norm.bias[i]);
}

// The positions as the encoder forms them (fp32 timescales and angles).
std::vector<float> positionTable() {
  std::vector<float> table(13 * kHidden);
  const double increment = std::log(10000.0) / 511.0;
  for (uint32_t t = 0; t < 13; ++t)
    for (uint32_t i = 0; i < 512; ++i) {
      const float inverse = float(std::exp(-increment * double(i)));
      const float angle = float(t) * inverse;
      table[t * kHidden + i] = std::sin(angle);
      table[t * kHidden + 512 + i] = std::cos(angle);
    }
  return table;
}

// The tower in fp64 over `frames` frames of `mel` ([frames][128]), rounding
// to bf16 where the kernels do: [tokens][2048].
std::vector<float> referenceEncode(const HostTower &tower, const std::vector<float> &mel, uint32_t frames) {
  const uint32_t tokens = audioTokens(frames);
  const uint32_t chunks = (frames + 99) / 100;
  const uint32_t chunkFrames = chunks == 1 ? frames : 100;
  std::vector<double> hidden(size_t(tokens) * kHidden);
  const std::vector<float> positions = positionTable();
  for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
    std::vector<double> input(size_t(kMelBins) * chunkFrames, 0.0);
    for (uint32_t t = 0; t < chunkFrames; ++t) {
      const uint32_t frame = chunk * 100 + t;
      if (frame >= frames) break;
      for (uint32_t f = 0; f < kMelBins; ++f) input[size_t(f) * chunkFrames + t] = mel[size_t(frame) * kMelBins + f];
    }
    uint32_t h1, w1, h2, w2, h3, w3;
    const std::vector<double> y1 = convolve(input, 1, kMelBins, chunkFrames, tower.conv1, h1, w1);
    const std::vector<double> y2 = convolve(y1, kChannels, h1, w1, tower.conv2, h2, w2);
    const std::vector<double> y3 = convolve(y2, kChannels, h2, w2, tower.conv3, h3, w3);
    require(h3 == 16, "conv3 keeps 16 frequency rows");
    const uint32_t valid = chunk + 1 == chunks ? tokens - chunk * 13 : 13;
    require(valid <= w3, "valid frames fit the conv3 width");
    std::vector<double> flat(kConvOutInput);
    for (uint32_t t = 0; t < valid; ++t) {
      for (uint32_t c = 0; c < kChannels; ++c)
        for (uint32_t f = 0; f < 16; ++f) flat[c * 16 + f] = y3[(size_t(c) * 16 + f) * w3 + t];
      double *row = hidden.data() + size_t(chunk * 13 + t) * kHidden;
      project(tower.convOut, flat.data(), row);
      for (uint32_t d = 0; d < kHidden; ++d) row[d] += positions[t * kHidden + d];
    }
  }
  const std::vector<OmniDenseRow> windows = audioAttentionRows(tokens);
  std::vector<double> normed(kHidden), q(size_t(tokens) * kHidden), k(q.size()), v(q.size()), attention(q.size());
  std::vector<double> projected(kHidden), ffn(kFfn);
  for (const HostLayer &layer : tower.layers) {
    for (uint32_t r = 0; r < tokens; ++r) {
      layerNorm(layer.norm1, hidden.data() + size_t(r) * kHidden, normed.data());
      project(layer.q, normed.data(), q.data() + size_t(r) * kHidden);
      project(layer.k, normed.data(), k.data() + size_t(r) * kHidden);
      project(layer.v, normed.data(), v.data() + size_t(r) * kHidden);
    }
    for (double &x : q) x = bf16(x);
    for (double &x : k) x = bf16(x);
    for (double &x : v) x = bf16(x);
    for (uint32_t r = 0; r < tokens; ++r)
      for (uint32_t h = 0; h < kHeads; ++h) {
        const double *qr = q.data() + size_t(r) * kHidden + h * kHeadDim;
        std::vector<double> scores;
        double maximum = -INFINITY;
        for (uint32_t key = windows[r].key_begin; key < windows[r].key_end; ++key) {
          const double *kr = k.data() + size_t(key) * kHidden + h * kHeadDim;
          double dot = 0.0;
          for (uint32_t d = 0; d < kHeadDim; ++d) dot += qr[d] * kr[d];
          scores.push_back(dot * 0.125);
          maximum = std::max(maximum, dot * 0.125);
        }
        double denominator = 0.0;
        std::vector<double> sum(kHeadDim, 0.0);
        for (uint32_t key = windows[r].key_begin; key < windows[r].key_end; ++key) {
          const double p = std::exp(scores[key - windows[r].key_begin] - maximum);
          denominator += p;
          const double *vr = v.data() + size_t(key) * kHidden + h * kHeadDim;
          for (uint32_t d = 0; d < kHeadDim; ++d) sum[d] += p * vr[d];
        }
        for (uint32_t d = 0; d < kHeadDim; ++d)
          attention[size_t(r) * kHidden + h * kHeadDim + d] = bf16(sum[d] / denominator);
      }
    for (uint32_t r = 0; r < tokens; ++r) {
      double *row = hidden.data() + size_t(r) * kHidden;
      project(layer.out, attention.data() + size_t(r) * kHidden, projected.data());
      for (uint32_t d = 0; d < kHidden; ++d) row[d] += projected[d];
      layerNorm(layer.norm2, row, normed.data());
      project(layer.fc1, normed.data(), ffn.data());
      for (double &x : ffn) x = bf16(gelu(x));
      project(layer.fc2, ffn.data(), projected.data());
      for (uint32_t d = 0; d < kHidden; ++d) row[d] += projected[d];
    }
  }
  std::vector<float> output(size_t(tokens) * kOutput);
  std::vector<double> out(kOutput);
  for (uint32_t r = 0; r < tokens; ++r) {
    layerNorm(tower.post, hidden.data() + size_t(r) * kHidden, normed.data());
    project(tower.proj1, normed.data(), projected.data());
    for (double &x : projected) x = bf16(gelu(x));
    project(tower.proj2, projected.data(), out.data());
    for (uint32_t d = 0; d < kOutput; ++d) output[size_t(r) * kOutput + d] = float(bf16(out[d]));
  }
  return output;
}

void testRandomTower(metal::MetalBackend &backend) {
  const HostTower tower = randomTower(2);
  AudioEncoder encoder(backend, deviceWeights(backend, tower), 130, 1);
  // 130 frames: a full chunk and a 30-frame chunk zero-padded to 100 (17
  // tokens); 70 frames: one chunk convolved at its own length (9 tokens).
  for (uint32_t frames : {130u, 70u}) {
    const std::vector<float> mel = randomValues(size_t(frames) * kMelBins, 1.0F);
    const uint32_t tokens = audioTokens(frames);
    metal::MetalBuffer output = sharedBuffer(backend, size_t(tokens) * kOutput * 2);
    metal::CommandGraph graph;
    encoder.addEncode(graph, f32Buffer(backend, mel), frames, output);
    run(backend, graph);
    const std::vector<float> got = readBf16(output, size_t(tokens) * kOutput);
    const std::vector<float> expected = referenceEncode(tower, mel, frames);
    const ErrorStats stats = compare(got, expected, kOutput);
    report("random tower " + std::to_string(frames) + " frames", stats);
    for (size_t i = 0; i < got.size(); ++i) {
      const double tolerance = 1.5e-2 + 1.5e-2 * std::fabs(expected[i]);
      if (std::isnan(got[i]) || std::fabs(double(got[i]) - expected[i]) > tolerance)
        throw std::runtime_error("random tower " + std::to_string(frames) + " frames [" + std::to_string(i) +
                                 "]: got " + std::to_string(got[i]) + ", expected " + std::to_string(expected[i]));
    }
  }
  // Extents: the mel and the output are checked before anything is encoded.
  {
    const uint32_t frames = 70, tokens = audioTokens(frames);
    metal::MetalBuffer mel = sharedBuffer(backend, size_t(frames) * kMelBins * 4);
    metal::MetalBuffer output = sharedBuffer(backend, size_t(tokens) * kOutput * 2);
    requireExtent(backend, mel, size_t(frames) * kMelBins * 4, 4, "audio mel",
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    encoder.addEncode(graph, view, frames, output);
                  });
    requireExtent(backend, output, size_t(tokens) * kOutput * 2, 2, "audio embedding",
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    encoder.addEncode(graph, mel, frames, view);
                  });
    metal::CommandGraph graph;
    rejects([&] { encoder.addEncode(graph, mel, 131, output); }, "mel frames", "frames past the maximum were accepted");
    rejects([&] { encoder.addEncode(graph, mel, 0, output); }, "mel frames", "zero frames were accepted");
  }
  std::cout << "random tower ok\n";
}

// The bf16 bits of `count` elements.
std::vector<uint16_t> readBits(const metal::MetalBuffer &buffer, size_t offset, size_t count) {
  const auto *bits = static_cast<const uint16_t *>(buffer.contents()) + offset;
  return {bits, bits + count};
}

// Clips encoded in one packed pass against each encoded alone: every clip's
// rows must be the same bits. `mels` holds each clip's log-mel frames.
void requirePackedEqual(metal::MetalBackend &backend, AudioEncoder &encoder, const std::string &what,
                        const std::vector<std::vector<float>> &mels, uint32_t outputWidth) {
  std::vector<AudioClip> clips;
  std::vector<std::vector<uint16_t>> alone;
  uint32_t tokens = 0;
  for (const std::vector<float> &mel : mels) {
    const auto frames = static_cast<uint32_t>(mel.size() / kMelBins);
    clips.push_back({f32Buffer(backend, mel), frames});
    const uint32_t clipTokens = audioTokens(frames);
    metal::MetalBuffer output = sharedBuffer(backend, size_t(clipTokens) * outputWidth * 2);
    metal::CommandGraph graph;
    encoder.addEncode(graph, clips.back().mel, frames, output);
    run(backend, graph);
    alone.push_back(readBits(output, 0, size_t(clipTokens) * outputWidth));
    tokens += clipTokens;
  }
  metal::MetalBuffer packed = sharedBuffer(backend, size_t(tokens) * outputWidth * 2);
  metal::CommandGraph graph;
  encoder.addEncode(graph, clips, packed);
  run(backend, graph);
  size_t offset = 0;
  for (size_t i = 0; i < clips.size(); ++i) {
    const std::vector<uint16_t> got = readBits(packed, offset, alone[i].size());
    size_t differing = 0;
    for (size_t j = 0; j < got.size(); ++j) differing += got[j] != alone[i][j];
    require(differing == 0, what + ": packed clip " + std::to_string(i) + " (" + std::to_string(clips[i].melFrames) +
                                " frames) differs from its own encode in " + std::to_string(differing) + " of " +
                                std::to_string(got.size()) + " elements");
    offset += got.size();
  }
  std::cout << what << ": " << clips.size() << " clips of " << tokens << " tokens packed, bit-identical\n";
}

void testPackedTower(metal::MetalBackend &backend) {
  const HostTower tower = randomTower(2);
  AudioEncoder encoder(backend, deviceWeights(backend, tower), 1700, 2);
  // Padded last chunks, a lone chunk and a clip of two windows, 17 + 9 + 34
  // + 137 tokens: the packed projections run 64 x 128 tiles where the short
  // clips alone run 32 x 64 ones, and the clips' windows start off the key
  // blocks of the packed rows.
  std::vector<std::vector<float>> mels;
  for (uint32_t frames : {130u, 70u, 260u, 1050u}) mels.push_back(randomValues(size_t(frames) * kMelBins, 1.0F));
  requirePackedEqual(backend, encoder, "packed random tower", mels, kOutput);
  // The packed tokens are bounded by one encode's.
  {
    metal::MetalBuffer mel = sharedBuffer(backend, size_t(1700) * kMelBins * 4);
    metal::MetalBuffer output = sharedBuffer(backend, size_t(2) * audioTokens(1700) * kOutput * 2);
    const std::vector<AudioClip> clips{{mel, 1700}, {mel, 10}};
    metal::CommandGraph graph;
    rejects([&] { encoder.addEncode(graph, clips, output); }, "packed audio clips",
            "clips past the workspace's tokens were accepted");
    // A clip of four tokens runs the GEMV alone and packs with no other.
    require(!packableClip(30) && packableClip(40), "clips of more than four tokens pack");
    const std::vector<AudioClip> short_{{mel, 130}, {mel, 30}};
    rejects([&] { encoder.addEncode(graph, short_, output); }, "more than four tokens",
            "a clip of four tokens was packed");
  }
  std::cout << "packed random tower ok\n";
}

// ---------------------------------------------------------------- real weights

std::optional<std::filesystem::path> findModel(const char *argument) {
  if (argument) return std::filesystem::path(argument);
  const char *home = std::getenv("HOME");
  if (!home) return std::nullopt;
  const std::filesystem::path snapshots =
      std::filesystem::path(home) / ".cache/huggingface/hub/models--mlx-community--Qwen3-ASR-1.7B-4bit/snapshots";
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(snapshots, error))
    if (std::filesystem::exists(entry.path() / "model.safetensors")) return entry.path();
  return std::nullopt;
}

// dev/tools/omni_mel_reference.py's tone_signal.
std::vector<float> toneSignal(uint32_t milliseconds) {
  const size_t samples = size_t(kSampleRate) * milliseconds / 1000;
  std::vector<float> signal(samples);
  constexpr double tau = 2.0 * std::numbers::pi;
  for (size_t n = 0; n < samples; ++n) {
    const double t = double(n) / kSampleRate;
    signal[n] = float(0.3 * std::sin(tau * 440.0 * t) + 0.2 * std::sin(tau * 1234.5 * t + 1.0) +
                      0.1 * std::sin(tau * 3000.0 * t) + 0.2 * std::sin(tau * 220.0 * t) * std::sin(tau * 0.5 * t));
  }
  return signal;
}

std::vector<float> encode(metal::MetalBackend &backend, AudioEncoder &encoder, const std::vector<float> &audio,
                          double *gpuSeconds = nullptr, double *wallSeconds = nullptr, double *melSeconds = nullptr) {
  const auto melStart = std::chrono::steady_clock::now();
  const std::vector<float> mel = splash::omni::logMelSpectrogram(audio);
  if (melSeconds) *melSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - melStart).count();
  const uint32_t frames = splash::omni::melFrames(audio.size());
  const uint32_t tokens = audioTokens(frames);
  metal::MetalBuffer output = sharedBuffer(backend, size_t(tokens) * kOutput * 2);
  metal::CommandGraph graph;
  encoder.addEncode(graph, f32Buffer(backend, mel), frames, output);
  const auto start = std::chrono::steady_clock::now();
  run(backend, graph, gpuSeconds);
  if (wallSeconds) *wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return readBf16(output, size_t(tokens) * kOutput);
}

void testParity(metal::MetalBackend &backend, const std::filesystem::path &model,
                const std::filesystem::path &fixtures) {
  std::cout << "loading " << model << "\n";
  const model::SafetensorsCheckpoint checkpoint(model);
  AudioEncoderWeights weights = loadAudioEncoderWeights(backend, checkpoint);
  require(weights.layers.size() == 24, "the audio tower has 24 layers");
  AudioEncoder encoder(backend, std::move(weights), 3000);
  double worstRms = 0.0;
  const auto check = [&](const std::string &what, const std::vector<float> &got, const std::vector<float> &expected) {
    const ErrorStats stats = compare(got, expected, kOutput);
    report("mlx-audio parity " + what, stats);
    worstRms = std::max(worstRms, stats.relativeRms());
    require(stats.relativeRms() <= 2.5e-2 && stats.maxAbs <= 0.25 * stats.maxRef,
            "the audio tower disagrees with mlx-audio on " + what);
  };
  for (uint32_t milliseconds : {2000u, 700u, 1300u}) {
    const std::string name = std::to_string(milliseconds) + "ms";
    const std::vector<float> audio = readFloats(fixtures / ("audio_" + name + ".f32"));
    const std::vector<float> expected = readFloats(fixtures / ("tower_" + name + ".f32"));
    const uint32_t tokens = audioTokens(splash::omni::melFrames(audio.size()));
    require(expected.size() == size_t(tokens) * kOutput, "the tower golden of " + name + " has the token count");
    check(name + " (" + std::to_string(tokens) + " tokens)", encode(backend, encoder, audio), expected);
  }
  {
    const std::vector<float> expected = readFloats(fixtures / "tower_9000ms_rows96.f32");
    std::vector<float> got = encode(backend, encoder, toneSignal(9000));
    require(got.size() == size_t(117) * kOutput, "9 s of audio is 117 tokens");
    got.erase(got.begin(), got.begin() + 96 * kOutput);
    check("9000ms rows 96..116 (two windows)", got, expected);
  }
  std::cout << "mlx-audio parity ok (worst relative rms " << worstRms << ")\n";

  // The fixture signals, a tone and noise packed into one encode.
  {
    std::vector<std::vector<float>> mels;
    for (uint32_t milliseconds : {2000u, 700u, 1300u})
      mels.push_back(splash::omni::logMelSpectrogram(
          readFloats(fixtures / ("audio_" + std::to_string(milliseconds) + "ms.f32"))));
    mels.push_back(splash::omni::logMelSpectrogram(toneSignal(9000)));
    std::vector<float> noise(size_t(kSampleRate) * 15);
    std::mt19937 generator(5);
    for (float &sample : noise) sample = std::uniform_real_distribution<float>(-0.5F, 0.5F)(generator);
    mels.push_back(splash::omni::logMelSpectrogram(noise));
    requirePackedEqual(backend, encoder, "packed Qwen3-ASR tower", mels, kOutput);
  }

  // Timing: 30 s of noise, the second run of the same graph shape.
  std::vector<float> noise(size_t(kSampleRate) * 30);
  std::mt19937 generator(3);
  for (float &sample : noise) sample = std::uniform_real_distribution<float>(-0.5F, 0.5F)(generator);
  double gpuSeconds = 0.0, wallSeconds = 0.0, melSeconds = 0.0;
  (void)encode(backend, encoder, noise);
  (void)encode(backend, encoder, noise, &gpuSeconds, &wallSeconds, &melSeconds);
  std::cout << "30 s of audio: log-mel " << melSeconds * 1e3 << " ms on the CPU, encoder " << gpuSeconds * 1e3
            << " ms GPU / " << wallSeconds * 1e3 << " ms wall (" << audioTokens(3000) << " tokens)\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: omni-audio-encoder METALLIB FIXTURE_DIR [MODEL_DIR]\n";
    return 2;
  }
  try {
    const std::filesystem::path fixtures = argv[2];
    testMel(fixtures);
    testTables();
    metal::MetalBackend backend(argv[1]);
    testRandomTower(backend);
    testPackedTower(backend);
    if (const std::optional<std::filesystem::path> model = findModel(argc > 3 ? argv[3] : nullptr))
      testParity(backend, *model, fixtures);
    else
      std::cout << "mlx-audio parity: skipped, no Qwen3-ASR weights found\n";
  } catch (const std::exception &error) {
    std::cerr << "omni audio encoder: " << error.what() << "\n";
    return 1;
  }
  std::cout << "omni audio encoder ok\n";
  return 0;
}
