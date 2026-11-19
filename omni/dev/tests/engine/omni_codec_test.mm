// The speech tokenizer decoder (ops/OmniCodec, kernels/omni/codec.metal):
// the implicit-GEMM convolution kernel's paths (causal conv with dilation,
// transposed conv, linear with GELU, scale and residual) against fp64, the whole decoder at a reduced configuration with
// random weights against an fp64 reference of the PyTorch model written
// here, and, when the TTS model is in the Hugging Face cache, the real
// decoder against mlx-audio's waveform of a fixed code sequence
// (dev/tests/fixtures/omni/codec_40_1.bin, from
// dev/tools/omni_codec_reference.py) with the time of a 100-frame decode.
// Streaming: chunked decodes through CodecStreams (two streams interleaved
// in the same graphs) against the whole-sequence samples, with the time per
// frame at chunk sizes 1, 4 and 25. argv[1] names the production metallib.

#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/OmniCodec.h"
#include "model/SafetensorsCheckpoint.hpp"
#include "ops/OmniCodec.hpp"
#include "ops/OmniCodecAne.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops::omni;
using namespace splash::test;

namespace {

std::mt19937 rng(11);
float uniform(float low, float high) { return std::uniform_real_distribution<float>(low, high)(rng); }
std::vector<float> randomValues(size_t count, float scale, float center = 0.0F) {
  std::vector<float> values(count);
  for (float &v : values) v = center + uniform(-scale, scale);
  return values;
}

metal::MetalBuffer f32Buffer(metal::MetalBackend &backend, const std::vector<float> &values) {
  metal::MetalBuffer buffer = sharedBuffer(backend, std::max<size_t>(1, values.size()) * 4);
  std::memcpy(buffer.contents(), values.data(), values.size() * 4);
  return buffer;
}
std::vector<float> readBack(const metal::MetalBuffer &buffer, size_t count) {
  std::vector<float> values(count);
  std::memcpy(values.data(), buffer.contents(), count * 4);
  return values;
}

metal::CommandTiming run(metal::MetalBackend &backend, metal::CommandGraph &graph) {
  metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
  return ticket.wait();
}

void close(double got, double expected, double relative, double absolute, const std::string &what) {
  const double tolerance = absolute + relative * std::fabs(expected);
  if (std::isnan(got) || std::fabs(got - expected) > tolerance)
    throw std::runtime_error(what + ": got " + std::to_string(got) + ", expected " + std::to_string(expected));
}

double gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }
double silu(double x) { return x / (1.0 + std::exp(-x)); }
double snake(double x, double alpha, double beta) {
  const double s = std::sin(x * std::exp(alpha));
  return x + 1.0 / (std::exp(beta) + 1e-9) * s * s;
}

// ------------------------------------------------ fp64 reference operators
// Tensors are [time][channel] doubles; weights are in the checkpoint's
// (PyTorch) layouts.

using Tensor = std::vector<double>;

struct Conv final { // weight [out][in][taps], bias [out]
  std::vector<float> weight, bias;
  uint32_t out = 0, in = 0, taps = 1, dilation = 1;
};
struct Transposed final { // weight [in][out][kernel], bias [out]
  std::vector<float> weight, bias;
  uint32_t in = 0, out = 0, stride = 1, kernel = 2;
};
struct Linear final { // weight [out][in], bias [out] (optional)
  std::vector<float> weight, bias;
  uint32_t out = 0, in = 0;
};
struct Snake final {
  std::vector<float> alpha, beta;
};

Tensor causalConv(const Conv &c, const Tensor &x, uint32_t rows, const Snake *act) {
  Tensor y(size_t(rows) * c.out);
  const int pad = int((c.taps - 1) * c.dilation);
  Tensor activated = x;
  if (act)
    for (uint32_t t = 0; t < rows; ++t)
      for (uint32_t ci = 0; ci < c.in; ++ci)
        activated[size_t(t) * c.in + ci] = snake(x[size_t(t) * c.in + ci], act->alpha[ci], act->beta[ci]);
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t co = 0; co < c.out; ++co) {
      double sum = c.bias.empty() ? 0.0 : c.bias[co];
      for (uint32_t k = 0; k < c.taps; ++k) {
        const int r = int(t) - pad + int(k * c.dilation);
        if (r < 0) continue;
        for (uint32_t ci = 0; ci < c.in; ++ci)
          sum += double(c.weight[(size_t(co) * c.in + ci) * c.taps + k]) * activated[size_t(r) * c.in + ci];
      }
      y[size_t(t) * c.out + co] = sum;
    }
  return y;
}

// PyTorch ConvTranspose1d then the right trim to stride * rows samples.
Tensor transposedConv(const Transposed &c, const Tensor &x, uint32_t rows, const Snake *act) {
  const uint32_t kernel = c.kernel;
  const uint32_t outRows = rows * c.stride;
  Tensor y(size_t(outRows) * c.out);
  for (uint32_t t = 0; t < outRows; ++t)
    for (uint32_t co = 0; co < c.out; ++co) y[size_t(t) * c.out + co] = c.bias[co];
  for (uint32_t i = 0; i < rows; ++i)
    for (uint32_t ci = 0; ci < c.in; ++ci) {
      double v = x[size_t(i) * c.in + ci];
      if (act) v = snake(v, act->alpha[ci], act->beta[ci]);
      for (uint32_t co = 0; co < c.out; ++co)
        for (uint32_t kk = 0; kk < kernel; ++kk) {
          const uint32_t t = i * c.stride + kk;
          if (t >= outRows) continue;
          y[size_t(t) * c.out + co] += v * c.weight[(size_t(ci) * c.out + co) * kernel + kk];
        }
    }
  return y;
}

Tensor linear(const Linear &l, const Tensor &x, uint32_t rows) {
  Tensor y(size_t(rows) * l.out);
  for (uint32_t t = 0; t < rows; ++t)
    for (uint32_t o = 0; o < l.out; ++o) {
      double sum = l.bias.empty() ? 0.0 : l.bias[o];
      for (uint32_t i = 0; i < l.in; ++i) sum += double(l.weight[size_t(o) * l.in + i]) * x[size_t(t) * l.in + i];
      y[size_t(t) * l.out + o] = sum;
    }
  return y;
}

Tensor rmsNorm(const Tensor &x, uint32_t rows, uint32_t width, const std::vector<float> &weight, double eps) {
  Tensor y(x.size());
  for (uint32_t t = 0; t < rows; ++t) {
    double sum = 0.0;
    for (uint32_t c = 0; c < width; ++c) sum += x[size_t(t) * width + c] * x[size_t(t) * width + c];
    const double inverse = 1.0 / std::sqrt(sum / width + eps);
    for (uint32_t c = 0; c < width; ++c) y[size_t(t) * width + c] = x[size_t(t) * width + c] * inverse * weight[c];
  }
  return y;
}

Tensor layerNorm(const Tensor &x, uint32_t rows, uint32_t width, const std::vector<float> &weight,
                 const std::vector<float> &bias, double eps) {
  Tensor y(x.size());
  for (uint32_t t = 0; t < rows; ++t) {
    double mean = 0.0;
    for (uint32_t c = 0; c < width; ++c) mean += x[size_t(t) * width + c];
    mean /= width;
    double variance = 0.0;
    for (uint32_t c = 0; c < width; ++c) {
      const double d = x[size_t(t) * width + c] - mean;
      variance += d * d;
    }
    variance /= width;
    const double inverse = 1.0 / std::sqrt(variance + eps);
    for (uint32_t c = 0; c < width; ++c)
      y[size_t(t) * width + c] = (x[size_t(t) * width + c] - mean) * inverse * weight[c] + bias[c];
  }
  return y;
}

// ------------------------------------------------------- host-side model

struct HostLayer final {
  std::vector<float> inputNorm, postNorm, attentionScale, mlpScale;
  Linear q, k, v, o, gate, up, down;
};
struct HostConvNeXt final {
  Transposed upsample;
  std::vector<float> dwWeight /* [dim][1][7] */, dwBias, normWeight, normBias, gamma;
  Linear pw1, pw2;
};
struct HostUnit final {
  Snake act1, act2;
  Conv conv1, conv2;
};
struct HostBlock final {
  Snake snake;
  Transposed upsample;
  std::array<HostUnit, 3> units;
};
struct HostModel final {
  CodecDims dims;
  std::vector<float> codebooks; // [quantizers][size][half]
  Linear rvqFirst, rvqRest;     // [codebookDim][half]
  Conv preConv;
  Linear inputProj, outputProj;
  std::vector<HostLayer> layers;
  std::vector<float> finalNorm;
  std::vector<HostConvNeXt> upsample;
  Conv initialConv;
  std::vector<HostBlock> blocks;
  Snake outputSnake;
  Conv outputConv;
};

Conv randomConv(uint32_t out, uint32_t in, uint32_t taps, uint32_t dilation = 1) {
  Conv c;
  c.out = out;
  c.in = in;
  c.taps = taps;
  c.dilation = dilation;
  c.weight = randomValues(size_t(out) * in * taps, 1.0F / std::sqrt(float(in * taps)));
  c.bias = randomValues(out, 0.1F);
  return c;
}
Transposed randomTransposed(uint32_t in, uint32_t out, uint32_t stride, uint32_t kernel) {
  Transposed c;
  c.in = in;
  c.out = out;
  c.stride = stride;
  c.kernel = kernel;
  c.weight = randomValues(size_t(in) * out * kernel, 1.0F / std::sqrt(float(kernel / stride * in)));
  c.bias = randomValues(out, 0.1F);
  return c;
}
Linear randomLinear(uint32_t out, uint32_t in, bool bias) {
  Linear l;
  l.out = out;
  l.in = in;
  l.weight = randomValues(size_t(out) * in, 1.0F / std::sqrt(float(in)));
  if (bias) l.bias = randomValues(out, 0.1F);
  return l;
}
Snake randomSnake(uint32_t channels) { return {randomValues(channels, 0.5F), randomValues(channels, 0.5F)}; }

HostModel randomModel() {
  HostModel m;
  CodecDims &d = m.dims;
  d.hidden = 128;
  d.intermediate = 128;
  d.latent = 96;
  d.codebookDim = 64;
  d.codebookSize = 16;
  d.quantizers = 4;
  d.decoderDim = 512;
  d.layers = 2;
  d.heads = 2;
  d.window = 4;
  d.upsampleRates = {2, 3, 2, 2};
  d.upsamplingRatios = {2, 2};
  const uint32_t half = d.codebookDim / 2;
  m.codebooks = randomValues(size_t(d.quantizers) * d.codebookSize * half, 1.0F);
  m.rvqFirst = randomLinear(d.codebookDim, half, false);
  m.rvqRest = randomLinear(d.codebookDim, half, false);
  m.preConv = randomConv(d.latent, d.codebookDim, 3);
  m.inputProj = randomLinear(d.hidden, d.latent, true);
  const uint32_t attention = d.heads * 64;
  for (uint32_t l = 0; l < d.layers; ++l) {
    HostLayer layer;
    layer.inputNorm = randomValues(d.hidden, 0.2F, 1.0F);
    layer.postNorm = randomValues(d.hidden, 0.2F, 1.0F);
    layer.attentionScale = randomValues(d.hidden, 0.1F, 0.2F);
    layer.mlpScale = randomValues(d.hidden, 0.1F, 0.2F);
    layer.q = randomLinear(attention, d.hidden, false);
    layer.k = randomLinear(attention, d.hidden, false);
    layer.v = randomLinear(attention, d.hidden, false);
    layer.o = randomLinear(d.hidden, attention, false);
    layer.gate = randomLinear(d.intermediate, d.hidden, false);
    layer.up = randomLinear(d.intermediate, d.hidden, false);
    layer.down = randomLinear(d.hidden, d.intermediate, false);
    m.layers.push_back(std::move(layer));
  }
  m.finalNorm = randomValues(d.hidden, 0.2F, 1.0F);
  m.outputProj = randomLinear(d.latent, d.hidden, true);
  for (uint32_t ratio : d.upsamplingRatios) {
    HostConvNeXt s;
    s.upsample = randomTransposed(d.latent, d.latent, ratio, ratio);
    s.dwWeight = randomValues(size_t(d.latent) * 7, 0.3F);
    s.dwBias = randomValues(d.latent, 0.1F);
    s.normWeight = randomValues(d.latent, 0.2F, 1.0F);
    s.normBias = randomValues(d.latent, 0.1F);
    s.gamma = randomValues(d.latent, 0.1F, 0.2F);
    s.pw1 = randomLinear(4 * d.latent, d.latent, true);
    s.pw2 = randomLinear(d.latent, 4 * d.latent, true);
    m.upsample.push_back(std::move(s));
  }
  m.initialConv = randomConv(d.decoderDim, d.latent, 7);
  uint32_t channels = d.decoderDim;
  for (uint32_t rate : d.upsampleRates) {
    HostBlock b;
    b.snake = randomSnake(channels);
    b.upsample = randomTransposed(channels, channels / 2, rate, 2 * rate);
    channels /= 2;
    const uint32_t dilations[3] = {1, 3, 9};
    for (uint32_t u = 0; u < 3; ++u) {
      b.units[u].act1 = randomSnake(channels);
      b.units[u].conv1 = randomConv(channels, channels, 7, dilations[u]);
      b.units[u].act2 = randomSnake(channels);
      b.units[u].conv2 = randomConv(channels, channels, 1);
    }
    m.blocks.push_back(std::move(b));
  }
  m.outputSnake = randomSnake(channels);
  m.outputConv = randomConv(1, channels, 7);
  m.outputConv.weight = randomValues(size_t(channels) * 7, 3.0F / std::sqrt(float(channels * 7)));
  return m;
}

// The reference decoder: the PyTorch model's forward in fp64.
Tensor referenceDecode(const HostModel &m, const std::vector<uint32_t> &codes, uint32_t frames) {
  const CodecDims &d = m.dims;
  const uint32_t half = d.codebookDim / 2;
  // RVQ.
  Tensor first(size_t(frames) * half), rest(size_t(frames) * half);
  for (uint32_t f = 0; f < frames; ++f)
    for (uint32_t q = 0; q < d.quantizers; ++q) {
      const uint32_t code = codes[size_t(f) * d.quantizers + q];
      for (uint32_t i = 0; i < half; ++i) {
        const double v = m.codebooks[(size_t(q) * d.codebookSize + code) * half + i];
        if (q == 0) first[size_t(f) * half + i] = v;
        else rest[size_t(f) * half + i] += v;
      }
    }
  const Tensor projectedFirst = linear(m.rvqFirst, first, frames), projectedRest = linear(m.rvqRest, rest, frames);
  Tensor hidden(projectedFirst.size());
  for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = projectedFirst[i] + projectedRest[i];
  hidden = causalConv(m.preConv, hidden, frames, nullptr);
  // Transformer.
  const uint32_t attention = d.heads * 64;
  Tensor x = linear(m.inputProj, hidden, frames);
  for (const HostLayer &layer : m.layers) {
    Tensor n = rmsNorm(x, frames, d.hidden, layer.inputNorm, d.rmsEpsilon);
    Tensor q = linear(layer.q, n, frames), k = linear(layer.k, n, frames), v = linear(layer.v, n, frames);
    for (uint32_t t = 0; t < frames; ++t)
      for (uint32_t h = 0; h < d.heads; ++h)
        for (uint32_t i = 0; i < 32; ++i) {
          const double inverse = 1.0 / std::pow(double(d.ropeTheta), double(2 * i) / 64.0);
          const double angle = double(t) * inverse, c = std::cos(angle), s = std::sin(angle);
          for (Tensor *rotated : {&q, &k}) {
            double &a = (*rotated)[size_t(t) * attention + h * 64 + i];
            double &b = (*rotated)[size_t(t) * attention + h * 64 + i + 32];
            const double a0 = a, b0 = b;
            a = a0 * c - b0 * s;
            b = b0 * c + a0 * s;
          }
        }
    Tensor att(size_t(frames) * attention);
    for (uint32_t t = 0; t < frames; ++t)
      for (uint32_t h = 0; h < d.heads; ++h) {
        const uint32_t begin = t + 1 > d.window ? t + 1 - d.window : 0;
        std::vector<double> scores;
        double maximum = -1e300;
        for (uint32_t j = begin; j <= t; ++j) {
          double s = 0.0;
          for (uint32_t i = 0; i < 64; ++i)
            s += q[size_t(t) * attention + h * 64 + i] * k[size_t(j) * attention + h * 64 + i];
          s /= 8.0;
          scores.push_back(s);
          maximum = std::max(maximum, s);
        }
        double sum = 0.0;
        for (double &s : scores) {
          s = std::exp(s - maximum);
          sum += s;
        }
        for (uint32_t i = 0; i < 64; ++i) {
          double o = 0.0;
          for (uint32_t j = begin; j <= t; ++j) o += scores[j - begin] * v[size_t(j) * attention + h * 64 + i];
          att[size_t(t) * attention + h * 64 + i] = o / sum;
        }
      }
    const Tensor projected = linear(layer.o, att, frames);
    for (uint32_t t = 0; t < frames; ++t)
      for (uint32_t c = 0; c < d.hidden; ++c)
        x[size_t(t) * d.hidden + c] += layer.attentionScale[c] * projected[size_t(t) * d.hidden + c];
    n = rmsNorm(x, frames, d.hidden, layer.postNorm, d.rmsEpsilon);
    Tensor gate = linear(layer.gate, n, frames);
    const Tensor up = linear(layer.up, n, frames);
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = silu(gate[i]) * up[i];
    const Tensor down = linear(layer.down, gate, frames);
    for (uint32_t t = 0; t < frames; ++t)
      for (uint32_t c = 0; c < d.hidden; ++c)
        x[size_t(t) * d.hidden + c] += layer.mlpScale[c] * down[size_t(t) * d.hidden + c];
  }
  x = rmsNorm(x, frames, d.hidden, m.finalNorm, d.rmsEpsilon);
  hidden = linear(m.outputProj, x, frames);
  uint32_t rows = frames;
  // Upsample.
  for (const HostConvNeXt &s : m.upsample) {
    Tensor y = transposedConv(s.upsample, hidden, rows, nullptr);
    rows *= s.upsample.stride;
    Tensor dw(y.size());
    for (uint32_t t = 0; t < rows; ++t)
      for (uint32_t c = 0; c < d.latent; ++c) {
        double sum = s.dwBias[c];
        for (uint32_t k = 0; k < 7; ++k) {
          const int r = int(t) - 6 + int(k);
          if (r >= 0) sum += double(s.dwWeight[size_t(c) * 7 + k]) * y[size_t(r) * d.latent + c];
        }
        dw[size_t(t) * d.latent + c] = sum;
      }
    Tensor wide = linear(s.pw1, layerNorm(dw, rows, d.latent, s.normWeight, s.normBias, 1e-6), rows);
    for (double &v : wide) v = gelu(v);
    const Tensor narrow = linear(s.pw2, wide, rows);
    for (uint32_t t = 0; t < rows; ++t)
      for (uint32_t c = 0; c < d.latent; ++c) y[size_t(t) * d.latent + c] += s.gamma[c] * narrow[size_t(t) * d.latent + c];
    hidden = std::move(y);
  }
  // Decoder.
  Tensor w = causalConv(m.initialConv, hidden, rows, nullptr);
  for (const HostBlock &b : m.blocks) {
    Tensor y = transposedConv(b.upsample, w, rows, &b.snake);
    rows *= b.upsample.stride;
    for (const HostUnit &u : b.units) {
      const Tensor inner = causalConv(u.conv1, y, rows, &u.act1);
      const Tensor outer = causalConv(u.conv2, inner, rows, &u.act2);
      for (size_t i = 0; i < y.size(); ++i) y[i] += outer[i];
    }
    w = std::move(y);
  }
  Tensor samples = causalConv(m.outputConv, w, rows, &m.outputSnake);
  for (double &s : samples) s = std::clamp(s, -1.0, 1.0);
  return samples;
}

CodecDecoderWeights uploadModel(metal::MetalBackend &backend, const HostModel &m) {
  const CodecDims &d = m.dims;
  const uint32_t half = d.codebookDim / 2;
  CodecDecoderWeights w;
  w.dims = d;
  w.codebooks = makeCodecVector(backend, m.codebooks);
  std::vector<float> joined(size_t(d.codebookDim) * d.codebookDim);
  for (uint32_t co = 0; co < d.codebookDim; ++co)
    for (uint32_t ci = 0; ci < half; ++ci) {
      joined[size_t(co) * d.codebookDim + ci] = m.rvqFirst.weight[size_t(co) * half + ci];
      joined[size_t(co) * d.codebookDim + half + ci] = m.rvqRest.weight[size_t(co) * half + ci];
    }
  w.rvqProjection = makeCodecLinear(backend, joined, nullptr, d.codebookDim, d.codebookDim);
  auto conv = [&](const Conv &c) { return makeCodecConv(backend, c.weight, &c.bias, c.out, c.in, c.taps, c.dilation); };
  auto transposed = [&](const Transposed &c) {
    return makeCodecTransposedConv(backend, c.weight, &c.bias, c.in, c.out, c.stride, c.kernel);
  };
  auto lin = [&](const Linear &l) {
    return makeCodecLinear(backend, l.weight, l.bias.empty() ? nullptr : &l.bias, l.out, l.in);
  };
  auto snakeOf = [&](const Snake &s) { return makeCodecSnake(backend, s.alpha, s.beta); };
  w.preConv = conv(m.preConv);
  w.inputProj = lin(m.inputProj);
  const uint32_t attention = d.heads * 64;
  for (const HostLayer &layer : m.layers) {
    CodecTransformerLayer l;
    l.inputNorm = makeCodecVector(backend, layer.inputNorm);
    l.postNorm = makeCodecVector(backend, layer.postNorm);
    l.attentionScale = makeCodecVector(backend, layer.attentionScale);
    l.mlpScale = makeCodecVector(backend, layer.mlpScale);
    std::vector<float> qkv;
    for (const Linear *part : {&layer.q, &layer.k, &layer.v}) qkv.insert(qkv.end(), part->weight.begin(), part->weight.end());
    l.qkv = makeCodecLinear(backend, qkv, nullptr, 3 * attention, d.hidden);
    l.o = lin(layer.o);
    std::vector<float> gateUp = layer.gate.weight;
    gateUp.insert(gateUp.end(), layer.up.weight.begin(), layer.up.weight.end());
    l.gateUp = makeCodecLinear(backend, gateUp, nullptr, 2 * d.intermediate, d.hidden);
    l.down = lin(layer.down);
    w.layers.push_back(std::move(l));
  }
  w.finalNorm = makeCodecVector(backend, m.finalNorm);
  w.outputProj = lin(m.outputProj);
  for (const HostConvNeXt &s : m.upsample) {
    CodecConvNeXt stage;
    stage.ratio = s.upsample.stride;
    stage.upsample = transposed(s.upsample);
    std::vector<float> packed(size_t(7) * d.latent);
    for (uint32_t c = 0; c < d.latent; ++c)
      for (uint32_t k = 0; k < 7; ++k) packed[size_t(k) * d.latent + c] = s.dwWeight[size_t(c) * 7 + k];
    stage.depthwiseWeight = makeCodecVector(backend, packed);
    stage.depthwiseBias = makeCodecVector(backend, s.dwBias);
    stage.depthwiseTaps = 7;
    stage.normWeight = makeCodecVector(backend, s.normWeight);
    stage.normBias = makeCodecVector(backend, s.normBias);
    stage.pw1 = lin(s.pw1);
    stage.pw2 = lin(s.pw2);
    stage.gamma = makeCodecVector(backend, s.gamma);
    w.upsample.push_back(std::move(stage));
  }
  w.initialConv = conv(m.initialConv);
  for (const HostBlock &b : m.blocks) {
    CodecDecoderBlock block;
    block.rate = b.upsample.stride;
    block.snake = snakeOf(b.snake);
    block.upsample = transposed(b.upsample);
    for (uint32_t u = 0; u < 3; ++u) {
      block.units[u].act1 = snakeOf(b.units[u].act1);
      block.units[u].conv1 = conv(b.units[u].conv1);
      block.units[u].act2 = snakeOf(b.units[u].act2);
      block.units[u].conv2 = conv(b.units[u].conv2);
    }
    w.blocks.push_back(std::move(block));
  }
  w.outputSnake = snakeOf(m.outputSnake);
  const uint32_t channels = m.outputConv.in;
  std::vector<float> packed(size_t(7) * channels);
  for (uint32_t c = 0; c < channels; ++c)
    for (uint32_t k = 0; k < 7; ++k) packed[size_t(k) * channels + c] = m.outputConv.weight[size_t(c) * 7 + k];
  w.outputConv.weight = makeCodecVector(backend, packed);
  w.outputConv.bias = makeCodecVector(backend, m.outputConv.bias);
  w.outputConv.inputChannels = channels;
  w.outputConv.outputChannels = 1;
  w.outputConv.taps = 7;
  return w;
}

// -------------------------------------------------- the GEMM kernel alone

struct KernelCase final {
  const char *name;
  uint32_t rows, in, out, taps, dilation, stride; // stride > 1: transposed of kernel taps * stride
  bool gelu, scale, residual, bias;
};

void testConvKernel(metal::MetalBackend &backend, const KernelCase &c) {
  const Tensor xd = [&] {
    std::vector<float> x = randomValues(size_t(c.rows) * c.in, 1.5F);
    return Tensor(x.begin(), x.end());
  }();
  const std::vector<float> x(xd.begin(), xd.end());
  CodecConv conv;
  Tensor expected;
  uint32_t outRows = c.rows, outWidth = c.out;
  if (c.stride > 1) {
    Transposed t = randomTransposed(c.in, c.out, c.stride, c.taps * c.stride);
    if (!c.bias) t.bias.assign(c.out, 0.0F);
    conv = makeCodecTransposedConv(backend, t.weight, c.bias ? &t.bias : nullptr, c.in, c.out, c.stride, t.kernel);
    expected = transposedConv(t, xd, c.rows, nullptr);
    outRows = c.rows * c.stride;
  } else {
    Conv t = randomConv(c.out, c.in, c.taps, c.dilation);
    if (!c.bias) t.bias.clear();
    conv = makeCodecConv(backend, t.weight, c.bias ? &t.bias : nullptr, c.out, c.in, c.taps, c.dilation);
    expected = causalConv(t, xd, c.rows, nullptr);
  }
  const std::vector<float> scale = randomValues(outWidth, 1.0F);
  const std::vector<float> residual = randomValues(size_t(outRows) * outWidth, 1.0F);
  for (uint32_t t = 0; t < outRows; ++t)
    for (uint32_t n = 0; n < outWidth; ++n) {
      double &v = expected[size_t(t) * outWidth + n];
      if (c.gelu) v = gelu(v);
      if (c.scale) v *= scale[n];
      if (c.residual) v += residual[size_t(t) * outWidth + n];
    }
  OmniCodecConvParams p{};
  p.rows = c.rows;
  p.input_rows = c.rows;
  p.in_channels = c.in;
  p.out_channels = conv.outputChannels;
  p.taps = conv.taps;
  p.dilation = conv.dilation;
  p.pad = c.stride > 1 ? int32_t(c.taps - 1) : int32_t((c.taps - 1) * c.dilation);
  p.input_stride = c.in;
  p.output_stride = conv.outputChannels;
  p.flags = (c.bias ? OMNI_CODEC_CONV_FLAG_BIAS : 0u) | (c.residual ? OMNI_CODEC_CONV_FLAG_RESIDUAL : 0u) |
            (c.gelu ? OMNI_CODEC_CONV_FLAG_GELU : 0u) | (c.scale ? OMNI_CODEC_CONV_FLAG_SCALE : 0u);
  metal::MetalBuffer input = f32Buffer(backend, x);
  metal::MetalBuffer output = sharedBuffer(backend, size_t(outRows) * outWidth * 4);
  metal::MetalBuffer residualBuffer = f32Buffer(backend, residual);
  metal::MetalBuffer scaleBuffer = f32Buffer(backend, scale);
  metal::CommandGraph graph;
  graph.add("omni_codec_conv",
            {input, conv.weight, c.bias ? conv.bias : conv.weight, residualBuffer, scaleBuffer, output, input},
            p, {(c.rows + 63) / 64, (conv.outputChannels + 63) / 64, 1}, {OMNI_CODEC_CONV_THREADS, 1, 1});
  run(backend, graph);
  const std::vector<float> got = readBack(output, expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    close(got[i], expected[i], 2e-5, 2e-5, std::string("conv kernel ") + c.name + " [" + std::to_string(i) + "]");
  if (conv.weightHalf) {
    // The half-precision kernel: each operand rounded to half, so a sum of
    // K products is off by about K^0.5 * 2^-11 of the products' magnitudes.
    metal::MetalBuffer halfOutput = sharedBuffer(backend, size_t(outRows) * outWidth * 4);
    metal::CommandGraph half;
    half.add("omni_codec_conv_half",
             {input, conv.weightHalf, c.bias ? conv.bias : conv.weight, residualBuffer, scaleBuffer, halfOutput, input}, p,
             {(c.rows + 63) / 64, (conv.outputChannels + 63) / 64, 1}, {OMNI_CODEC_CONV_THREADS, 1, 1});
    run(backend, half);
    const std::vector<float> gotHalf = readBack(halfOutput, expected.size());
    double peak = 0.0;
    for (float v : expected) peak = std::max(peak, double(std::fabs(v)));
    for (size_t i = 0; i < expected.size(); ++i)
      close(gotHalf[i], expected[i], 4e-3, 4e-3 * peak, std::string("half conv kernel ") + c.name + " [" + std::to_string(i) + "]");
  }
  std::cout << "conv kernel " << c.name << (conv.weightHalf ? " (fp32 and half) ok\n" : " ok\n");
}

void testConvKernels(metal::MetalBackend &backend) {
  testConvKernel(backend, {"linear bias gelu", 70, 64, 160, 1, 1, 1, true, false, false, true});
  testConvKernel(backend, {"linear scale residual", 9, 128, 96, 1, 1, 1, false, true, true, false});
  testConvKernel(backend, {"causal dilated residual", 100, 96, 160, 7, 3, 1, false, false, true, true});
  testConvKernel(backend, {"causal k3", 5, 32, 4, 3, 1, 1, false, false, false, true});
  testConvKernel(backend, {"causal k3 columns 36", 5, 32, 36, 3, 1, 1, false, false, false, true});
  testConvKernel(backend, {"transposed stride 3", 10, 64, 32, 2, 1, 3, false, false, false, true});
  testConvKernel(backend, {"transposed stride 2", 65, 32, 64, 2, 1, 2, false, false, false, true});
  testConvKernel(backend, {"transposed kernel = stride 2", 65, 32, 64, 1, 1, 2, false, false, false, true});
  testConvKernel(backend, {"transposed kernel 3 x stride 2", 7, 32, 32, 3, 1, 2, false, false, false, true});
}

// --------------------------------------------- the decoder, random weights

void testRandomDecoder(metal::MetalBackend &backend) {
  const HostModel model = randomModel();
  const CodecDims &d = model.dims;
  const uint32_t frames = 9;
  std::vector<uint32_t> codes(size_t(frames) * d.quantizers);
  for (uint32_t &c : codes) c = uint32_t(rng()) % d.codebookSize;
  const Tensor expected = referenceDecode(model, codes, frames);
  double peak = 0.0;
  for (double v : expected) peak = std::max(peak, std::fabs(v));
  require(peak > 0.05, "the random decoder's waveform is silent");

  metal::MetalBuffer codeBuffer = sharedBuffer(backend, codes.size() * 4);
  std::memcpy(codeBuffer.contents(), codes.data(), codes.size() * 4);
  const uint32_t samples = frames * d.samplesPerFrame();
  require(samples == expected.size(), "the reference's sample count differs from samplesPerFrame");
  // The fp32 convolutions against the fp64 reference at fp32 noise; the
  // half-precision ones (the default) within their rounding. The stream
  // checks below run the default decoder.
  {
    CodecDecoderWeights exact = uploadModel(backend, model);
    exact.forEachConv([](CodecConv &conv) { conv.weightHalf = {}; });
    CodecDecoder exactDecoder(backend, std::move(exact), 16);
    metal::MetalBuffer exactOutput = sharedBuffer(backend, size_t(samples) * 4);
    metal::CommandGraph graph;
    exactDecoder.addDecode(graph, codeBuffer, frames, exactOutput);
    run(backend, graph);
    const std::vector<float> exactGot = readBack(exactOutput, samples);
    for (uint32_t i = 0; i < samples; ++i)
      close(exactGot[i], expected[i], 1e-4, 3e-4, "random decoder sample " + std::to_string(i));
  }
  CodecDecoder decoder(backend, uploadModel(backend, model), 16);
  metal::MetalBuffer output = sharedBuffer(backend, size_t(samples) * 4);
  metal::CommandGraph graph;
  decoder.addDecode(graph, codeBuffer, frames, output);
  run(backend, graph);
  const std::vector<float> got = readBack(output, samples);
  double maxError = 0.0;
  for (uint32_t i = 0; i < samples; ++i) {
    maxError = std::max(maxError, std::fabs(double(got[i]) - expected[i]));
    close(got[i], expected[i], 4e-3, 4e-3 * peak, "random decoder (half convolutions) sample " + std::to_string(i));
  }
  std::cout << "random decoder ok (" << samples << " samples, peak " << peak << ", max error " << maxError << ")\n";

  // The same frames in chunks of 2, 1, 3 and 3 through a stream, twice
  // (reset between), against the whole decode: identical sums.
  CodecStream stream(backend, decoder.weights(), 4);
  for (int pass = 0; pass < 2; ++pass) {
    stream.reset();
    uint32_t first = 0;
    for (uint32_t chunk : {2u, 1u, 3u, 3u}) {
      metal::MetalBuffer chunkCodes = sharedBuffer(backend, size_t(chunk) * d.quantizers * 4);
      std::memcpy(chunkCodes.contents(), codes.data() + size_t(first) * d.quantizers, size_t(chunk) * d.quantizers * 4);
      metal::MetalBuffer chunkOutput = sharedBuffer(backend, size_t(chunk) * d.samplesPerFrame() * 4);
      metal::CommandGraph g;
      decoder.addDecodeStream(g, stream, chunkCodes, chunk, chunkOutput);
      run(backend, g);
      const std::vector<float> streamed = readBack(chunkOutput, size_t(chunk) * d.samplesPerFrame());
      for (size_t i = 0; i < streamed.size(); ++i)
        close(streamed[i], got[size_t(first) * d.samplesPerFrame() + i], 0.0, 1e-6,
              "random stream pass " + std::to_string(pass) + " frame " + std::to_string(first) + " sample " + std::to_string(i));
      first += chunk;
    }
    require(stream.position() == frames, "the stream's position does not count its frames");
  }
  rejects([&] { metal::CommandGraph g; decoder.addDecodeStream(g, stream, codeBuffer, 5, output); }, "1 .. 4 frames",
          "a chunk past the stream's maximum was accepted");
  std::cout << "random stream ok (" << stream.bytes() << " bytes of state)\n";

  // Rejections: too many frames, short code and output buffers.
  rejects([&] { metal::CommandGraph g; decoder.addDecode(g, codeBuffer, 17, output); }, "1 .. 16 frames",
          "a decode past the maximum frames was accepted");
  requireExtent(backend, codeBuffer, codes.size() * 4, 4, "codec codes",
                [&](metal::CommandGraph &g, const metal::MetalBuffer &view) { decoder.addDecode(g, view, frames, output); });
  requireExtent(backend, output, size_t(samples) * 4, 4, "codec output",
                [&](metal::CommandGraph &g, const metal::MetalBuffer &view) { decoder.addDecode(g, codeBuffer, frames, view); });
}

// ------------------------------------------------- the decoder, real weights

std::vector<uint32_t> xorshiftCodes(uint32_t frames, uint32_t quantizers, uint32_t seed, uint32_t codebookSize) {
  uint32_t state = seed;
  std::vector<uint32_t> codes(size_t(frames) * quantizers);
  for (uint32_t &c : codes) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    c = state % codebookSize;
  }
  return codes;
}

std::filesystem::path findSpeechTokenizer() {
  if (const char *model = std::getenv("SPLASH_OMNI_TTS_MODEL")) return std::filesystem::path(model) / "speech_tokenizer";
  const char *home = std::getenv("HOME");
  if (!home) return {};
  const std::filesystem::path snapshots =
      std::filesystem::path(home) / ".cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-4bit/snapshots";
  std::error_code error;
  if (!std::filesystem::is_directory(snapshots, error)) return {};
  for (const auto &entry : std::filesystem::directory_iterator(snapshots, error))
    if (std::filesystem::exists(entry.path() / "speech_tokenizer/model.safetensors", error))
      return entry.path() / "speech_tokenizer";
  return {};
}

// The real decoder's tensors as the fp64 reference takes them.
HostModel loadHostModel(const std::filesystem::path &tokenizer) {
  model::SafetensorsCheckpoint checkpoint(tokenizer);
  auto floats = [&](const std::string &name) {
    const model::SourceTensor &tensor = checkpoint.require(name);
    require(tensor.dtype == "F32", name + " is not F32");
    std::vector<float> values(tensor.bytes / 4);
    tensor.read(0, {reinterpret_cast<uint8_t *>(values.data()), tensor.bytes});
    return values;
  };
  HostModel m;
  const CodecDims &d = m.dims;
  const uint32_t half = d.codebookDim / 2;
  m.codebooks.resize(size_t(d.quantizers) * d.codebookSize * half);
  for (uint32_t q = 0; q < d.quantizers; ++q) {
    const std::string prefix = q == 0 ? "decoder.quantizer.rvq_first.vq.layers.0._codebook"
                                      : "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(q - 1) + "._codebook";
    const std::vector<float> sum = floats(prefix + ".embedding_sum"), usage = floats(prefix + ".cluster_usage");
    for (uint32_t e = 0; e < d.codebookSize; ++e)
      for (uint32_t i = 0; i < half; ++i)
        m.codebooks[(size_t(q) * d.codebookSize + e) * half + i] = sum[size_t(e) * half + i] / std::max(usage[e], 1e-5F);
  }
  auto lin = [&](const std::string &prefix, uint32_t out, uint32_t in, bool bias) {
    Linear l;
    l.out = out;
    l.in = in;
    l.weight = floats(prefix + ".weight");
    if (bias) l.bias = floats(prefix + ".bias");
    return l;
  };
  auto conv = [&](const std::string &prefix, uint32_t out, uint32_t in, uint32_t taps, uint32_t dilation = 1) {
    Conv c;
    c.out = out;
    c.in = in;
    c.taps = taps;
    c.dilation = dilation;
    c.weight = floats(prefix + ".weight");
    c.bias = floats(prefix + ".bias");
    return c;
  };
  auto transposed = [&](const std::string &prefix, uint32_t in, uint32_t out, uint32_t stride, uint32_t kernel) {
    Transposed t;
    t.in = in;
    t.out = out;
    t.stride = stride;
    t.kernel = kernel;
    t.weight = floats(prefix + ".weight");
    t.bias = floats(prefix + ".bias");
    return t;
  };
  auto snakeOf = [&](const std::string &prefix) { return Snake{floats(prefix + ".alpha"), floats(prefix + ".beta")}; };
  m.rvqFirst = lin("decoder.quantizer.rvq_first.output_proj", d.codebookDim, half, false);
  m.rvqRest = lin("decoder.quantizer.rvq_rest.output_proj", d.codebookDim, half, false);
  m.preConv = conv("decoder.pre_conv.conv", d.latent, d.codebookDim, 3);
  m.inputProj = lin("decoder.pre_transformer.input_proj", d.hidden, d.latent, true);
  const uint32_t attention = d.heads * 64;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string prefix = "decoder.pre_transformer.layers." + std::to_string(l) + ".";
    HostLayer layer;
    layer.inputNorm = floats(prefix + "input_layernorm.weight");
    layer.postNorm = floats(prefix + "post_attention_layernorm.weight");
    layer.attentionScale = floats(prefix + "self_attn_layer_scale.scale");
    layer.mlpScale = floats(prefix + "mlp_layer_scale.scale");
    layer.q = lin(prefix + "self_attn.q_proj", attention, d.hidden, false);
    layer.k = lin(prefix + "self_attn.k_proj", attention, d.hidden, false);
    layer.v = lin(prefix + "self_attn.v_proj", attention, d.hidden, false);
    layer.o = lin(prefix + "self_attn.o_proj", d.hidden, attention, false);
    layer.gate = lin(prefix + "mlp.gate_proj", d.intermediate, d.hidden, false);
    layer.up = lin(prefix + "mlp.up_proj", d.intermediate, d.hidden, false);
    layer.down = lin(prefix + "mlp.down_proj", d.hidden, d.intermediate, false);
    m.layers.push_back(std::move(layer));
  }
  m.finalNorm = floats("decoder.pre_transformer.norm.weight");
  m.outputProj = lin("decoder.pre_transformer.output_proj", d.latent, d.hidden, true);
  for (uint32_t s = 0; s < d.upsamplingRatios.size(); ++s) {
    const std::string prefix = "decoder.upsample." + std::to_string(s) + ".";
    HostConvNeXt stage;
    stage.upsample = transposed(prefix + "0.conv", d.latent, d.latent, d.upsamplingRatios[s], d.upsamplingRatios[s]);
    stage.dwWeight = floats(prefix + "1.dwconv.conv.weight");
    stage.dwBias = floats(prefix + "1.dwconv.conv.bias");
    stage.normWeight = floats(prefix + "1.norm.weight");
    stage.normBias = floats(prefix + "1.norm.bias");
    stage.gamma = floats(prefix + "1.gamma");
    stage.pw1 = lin(prefix + "1.pwconv1", 4 * d.latent, d.latent, true);
    stage.pw2 = lin(prefix + "1.pwconv2", d.latent, 4 * d.latent, true);
    m.upsample.push_back(std::move(stage));
  }
  m.initialConv = conv("decoder.decoder.0.conv", d.decoderDim, d.latent, 7);
  uint32_t channels = d.decoderDim;
  for (uint32_t b = 0; b < d.upsampleRates.size(); ++b) {
    const std::string prefix = "decoder.decoder." + std::to_string(b + 1) + ".block.";
    HostBlock block;
    block.snake = snakeOf(prefix + "0");
    block.upsample = transposed(prefix + "1.conv", channels, channels / 2, d.upsampleRates[b], 2 * d.upsampleRates[b]);
    channels /= 2;
    const uint32_t dilations[3] = {1, 3, 9};
    for (uint32_t u = 0; u < 3; ++u) {
      const std::string unit = prefix + std::to_string(u + 2) + ".";
      block.units[u].act1 = snakeOf(unit + "act1");
      block.units[u].conv1 = conv(unit + "conv1.conv", channels, channels, 7, dilations[u]);
      block.units[u].act2 = snakeOf(unit + "act2");
      block.units[u].conv2 = conv(unit + "conv2.conv", channels, channels, 1);
    }
    m.blocks.push_back(std::move(block));
  }
  m.outputSnake = snakeOf("decoder.decoder.5");
  m.outputConv = conv("decoder.decoder.6.conv", 1, channels, 7);
  return m;
}

struct Distance final {
  double maxError = 0.0, rms = 0.0;
  size_t worst = 0;
};
template <class A, class B> Distance distance(const A &got, const B &expected, size_t count) {
  Distance d;
  double squares = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double error = std::fabs(double(got[i]) - double(expected[i]));
    if (error > d.maxError) {
      d.maxError = error;
      d.worst = i;
    }
    squares += error * error;
  }
  d.rms = std::sqrt(squares / count);
  return d;
}

// Two streams of the same 60-frame sequence, chunked [1, 4, 8, 25, 22] and
// [25, 25, 10], their chunks interleaved in the same graphs, against the
// whole decode; then the time per frame at chunk sizes 1, 4 and 25.
void testRealStreams(metal::MetalBackend &backend, CodecDecoder &decoder) {
  const CodecDims &d = decoder.weights().dims;
  const uint32_t frames = 60, perFrame = d.samplesPerFrame();
  const std::vector<uint32_t> codes = xorshiftCodes(frames, d.quantizers, 3, d.codebookSize);
  metal::MetalBuffer codeBuffer = sharedBuffer(backend, codes.size() * 4);
  std::memcpy(codeBuffer.contents(), codes.data(), codes.size() * 4);
  metal::MetalBuffer whole = sharedBuffer(backend, size_t(frames) * perFrame * 4);
  {
    metal::CommandGraph g;
    decoder.addDecode(g, codeBuffer, frames, whole);
    run(backend, g);
  }
  const std::vector<float> expected = readBack(whole, size_t(frames) * perFrame);

  struct Lane final {
    std::vector<uint32_t> chunks;
    CodecStream stream;
    uint32_t first = 0;
    size_t next = 0;
    std::vector<float> samples;
  };
  Lane lanes[2] = {{{1, 4, 8, 25, 22}, CodecStream(backend, decoder.weights(), 25), 0, 0, {}},
                   {{25, 25, 10}, CodecStream(backend, decoder.weights(), 25), 0, 0, {}}};
  while (lanes[0].next < lanes[0].chunks.size() || lanes[1].next < lanes[1].chunks.size()) {
    metal::CommandGraph g;
    std::vector<std::pair<Lane *, metal::MetalBuffer>> pending;
    for (Lane &lane : lanes) {
      if (lane.next >= lane.chunks.size()) continue;
      const uint32_t chunk = lane.chunks[lane.next];
      metal::MetalBuffer chunkCodes = sharedBuffer(backend, size_t(chunk) * d.quantizers * 4);
      std::memcpy(chunkCodes.contents(), codes.data() + size_t(lane.first) * d.quantizers, size_t(chunk) * d.quantizers * 4);
      metal::MetalBuffer chunkOutput = sharedBuffer(backend, size_t(chunk) * perFrame * 4);
      decoder.addDecodeStream(g, lane.stream, chunkCodes, chunk, chunkOutput);
      pending.emplace_back(&lane, chunkOutput);
    }
    run(backend, g);
    for (auto &[lane, chunkOutput] : pending) {
      const uint32_t chunk = lane->chunks[lane->next++];
      const std::vector<float> streamed = readBack(chunkOutput, size_t(chunk) * perFrame);
      lane->samples.insert(lane->samples.end(), streamed.begin(), streamed.end());
      lane->first += chunk;
    }
  }
  for (size_t l = 0; l < 2; ++l) {
    require(lanes[l].samples.size() == expected.size(), "a stream produced the wrong number of samples");
    const Distance distanceToWhole = distance(lanes[l].samples, expected, expected.size());
    std::printf("real stream %zu: max abs error %.3e at %zu, rms %.3e vs the whole decode\n", l, distanceToWhole.maxError,
                distanceToWhole.worst, distanceToWhole.rms);
    require(distanceToWhole.maxError <= 1e-5, "a stream differs from the whole decode");
  }

  // Timing: per chunk size, decode four chunks after a reset and time the
  // last (steady state, pipelines warm).
  for (uint32_t chunk : {1u, 4u, 25u}) {
    CodecStream &stream = lanes[0].stream;
    stream.reset();
    metal::MetalBuffer chunkCodes = sharedBuffer(backend, size_t(chunk) * d.quantizers * 4);
    std::memcpy(chunkCodes.contents(), codes.data(), size_t(chunk) * d.quantizers * 4);
    metal::MetalBuffer chunkOutput = sharedBuffer(backend, size_t(chunk) * perFrame * 4);
    metal::CommandTiming timing{};
    for (int pass = 0; pass < 4; ++pass) {
      metal::CommandGraph g;
      decoder.addDecodeStream(g, stream, chunkCodes, chunk, chunkOutput);
      timing = run(backend, g);
    }
    std::printf("real stream: chunk of %2u frames in %.2f ms GPU (%.2f ms per frame); state %.1f MB\n", chunk,
                timing.gpuSeconds * 1e3, timing.gpuSeconds * 1e3 / chunk, double(stream.bytes()) / 1e6);
  }
}

void testRealDecoder(metal::MetalBackend &backend) {
  const std::filesystem::path tokenizer = findSpeechTokenizer();
  const std::filesystem::path fixture = "dev/tests/fixtures/omni/codec_40_1.bin";
  if (tokenizer.empty() || !std::filesystem::exists(fixture)) {
    std::cout << "real decoder skipped (no cached Qwen3-TTS model or fixture)\n";
    return;
  }
  std::ifstream in(fixture, std::ios::binary);
  char magic[4];
  uint32_t frames = 0, seed = 0, samples = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char *>(&frames), 4);
  in.read(reinterpret_cast<char *>(&seed), 4);
  in.read(reinterpret_cast<char *>(&samples), 4);
  require(in && std::memcmp(magic, "OCRF", 4) == 0, "the codec fixture header is not OCRF");
  std::vector<float> expected(samples);
  in.read(reinterpret_cast<char *>(expected.data()), size_t(samples) * 4);
  require(bool(in), "the codec fixture is truncated");

  CodecDecoder decoder(backend, loadCodecDecoderWeights(backend, tokenizer), 128);
  const CodecDims &d = decoder.weights().dims;
  require(samples == frames * d.samplesPerFrame(), "the codec fixture's samples do not match the frames");
  const std::vector<uint32_t> codes = xorshiftCodes(frames, d.quantizers, seed, d.codebookSize);
  metal::MetalBuffer codeBuffer = sharedBuffer(backend, codes.size() * 4);
  std::memcpy(codeBuffer.contents(), codes.data(), codes.size() * 4);
  metal::MetalBuffer output = sharedBuffer(backend, size_t(samples) * 4);
  {
    metal::CommandGraph graph;
    decoder.addDecode(graph, codeBuffer, frames, output);
    run(backend, graph);
  }
  const std::vector<float> got = readBack(output, samples);
  double peak = 0.0;
  for (float v : expected) peak = std::max(peak, std::fabs(double(v)));
  const Distance mlx = distance(got, expected, samples);
  std::printf("real decoder: %u frames, %u samples, reference peak %.4f; vs mlx-audio max abs error %.3e at %zu, "
              "rms error %.3e\n",
              frames, samples, peak, mlx.maxError, mlx.worst, mlx.rms);
  // SPLASH_OMNI_CODEC_FP64=<frames>: the fp64 reference over the first
  // frames (the decoder is causal, so a prefix decodes to the same samples)
  // places both the GPU and mlx-audio against the exact result.
  if (const char *check = std::getenv("SPLASH_OMNI_CODEC_FP64")) {
    const uint32_t prefix = std::min<uint32_t>(frames, uint32_t(std::atoi(check)));
    const HostModel host = loadHostModel(tokenizer);
    const std::vector<uint32_t> prefixCodes(codes.begin(), codes.begin() + size_t(prefix) * d.quantizers);
    const Tensor exact = referenceDecode(host, prefixCodes, prefix);
    const Distance gpu = distance(got, exact, exact.size()), reference = distance(expected, exact, exact.size());
    std::printf("real decoder fp64 over %u frames: GPU max abs error %.3e rms %.3e; mlx-audio max abs error %.3e rms "
                "%.3e\n",
                prefix, gpu.maxError, gpu.rms, reference.maxError, reference.rms);
  }
  require(mlx.maxError < 2e-2 && mlx.rms < 2e-3, "the real decoder differs from mlx-audio by more than fp32 noise");

  // Timing of a 100-frame decode (the second run, after pipeline creation).
  const uint32_t timedFrames = 100;
  const std::vector<uint32_t> timedCodes = xorshiftCodes(timedFrames, d.quantizers, 2, d.codebookSize);
  metal::MetalBuffer timedCodeBuffer = sharedBuffer(backend, timedCodes.size() * 4);
  std::memcpy(timedCodeBuffer.contents(), timedCodes.data(), timedCodes.size() * 4);
  metal::MetalBuffer timedOutput = sharedBuffer(backend, size_t(timedFrames) * d.samplesPerFrame() * 4);
  metal::CommandTiming timing{};
  for (int pass = 0; pass < 3; ++pass) {
    metal::CommandGraph graph;
    decoder.addDecode(graph, timedCodeBuffer, timedFrames, timedOutput);
    timing = run(backend, graph);
  }
  std::printf("real decoder: 100 frames (%u samples) in %.1f ms GPU, %.1f ms wall; workspace %.0f MB for 128 frames\n",
              timedFrames * d.samplesPerFrame(), timing.gpuSeconds * 1e3, timing.wallSeconds * 1e3,
              double(CodecDecoder::workspaceBytes(d, 128)) / 1e6);
  testRealStreams(backend, decoder);
}

// ------------------------------------------ the stack on the Neural Engine

// How a chunk of a stream decodes: the front on the GPU and the stack on the
// Neural Engine; wholly on the GPU; or the front for the Neural Engine, whose
// evaluation is then given up and the GPU decodes the stack from the lane's
// input and histories, as the engine's fallback does.
enum class ChunkMode { Ane, Gpu, Recover };

struct AneReport final {
  std::mutex mutex;
  std::condition_variable changed;
  bool reported = false, success = false;
};

// Waits for an evaluation's report: true if it ran.
bool awaitReport(AneReport &report) {
  std::unique_lock lock(report.mutex);
  if (!report.changed.wait_for(lock, std::chrono::seconds(60), [&] { return report.reported; }))
    throw std::runtime_error("the Neural Engine did not report an evaluation");
  return report.success;
}

struct AneLane final {
  std::vector<std::pair<uint32_t, ChunkMode>> chunks;
  CodecStream stream;
  uint32_t first = 0;
  size_t next = 0;
  bool onAne = true; // where the stream's stack state lives
  std::vector<float> samples;
};

// Decodes the lanes' chunk schedules of `codes`, one round of every lane's
// next chunk per command as the engine does; returns nothing, filling each
// lane's samples.
void streamLanes(metal::MetalBackend &backend, CodecDecoder &decoder, CodecAneStack &ane, std::span<AneLane> lanes,
                 const std::vector<uint32_t> &codes) {
  const CodecDims &d = decoder.weights().dims;
  const uint32_t perFrame = d.samplesPerFrame();
  metal::SharedEvent event = backend.newSharedEvent();
  uint64_t value = 0;
  for (uint32_t l = 0; l < lanes.size(); ++l) {
    lanes[l].stream.reset();
    ane.reset(l);
    lanes[l].onAne = true;
  }
  bool more = true;
  while (more) {
    more = false;
    metal::CommandGraph graph;
    struct Pending final {
      uint32_t lane, frames, function;
      ChunkMode mode;
      metal::MetalBuffer output;
    };
    std::vector<Pending> pending;
    for (uint32_t l = 0; l < lanes.size(); ++l) {
      AneLane &lane = lanes[l];
      if (lane.next >= lane.chunks.size()) continue;
      const auto [frames, mode] = lane.chunks[lane.next];
      metal::MetalBuffer chunkCodes = sharedBuffer(backend, size_t(frames) * d.quantizers * 4);
      std::memcpy(chunkCodes.contents(), codes.data() + size_t(lane.first) * d.quantizers, size_t(frames) * d.quantizers * 4);
      metal::MetalBuffer output = sharedBuffer(backend, size_t(frames) * perFrame * 4);
      const uint32_t function = ane.functionFor(frames);
      if (mode == ChunkMode::Gpu) {
        if (lane.onAne) ane.addHistoriesToStream(graph, l, lane.stream, lane.stream.phase());
        decoder.addDecodeStream(graph, lane.stream, chunkCodes, frames, output);
        lane.onAne = false;
      } else {
        require(function, "no ANE function holds the chunk");
        if (!lane.onAne) ane.addHistoriesFromStream(graph, l, lane.stream, lane.stream.phase());
        decoder.addFrontStream(graph, lane.stream, chunkCodes, frames);
        ane.addInput(graph, l, decoder.stackInput(frames), function);
        lane.onAne = true;
        if (mode == ChunkMode::Recover) {
          const uint32_t phase = lane.stream.phase() ^ 1u;
          ane.addHistoriesToStream(graph, l, lane.stream, phase);
          ane.addInputToRows(graph, l, decoder.stackInput(frames));
          decoder.addStackStream(graph, lane.stream, frames, phase, output);
          lane.onAne = false;
        }
      }
      pending.push_back({l, frames, function, mode, output});
    }
    if (pending.empty()) break;
    more = true;
    graph.signal(event, ++value);
    std::vector<std::shared_ptr<AneReport>> reports(pending.size());
    for (size_t i = 0; i < pending.size(); ++i) {
      if (pending[i].mode != ChunkMode::Ane) continue;
      auto report = reports[i] = std::make_shared<AneReport>();
      ane.enqueue(pending[i].lane, pending[i].function, event, value, value, [report](bool success) {
        std::lock_guard lock(report->mutex);
        report->reported = true;
        report->success = success;
        report->changed.notify_all();
      });
    }
    run(backend, graph);
    for (size_t i = 0; i < pending.size(); ++i) {
      const Pending &p = pending[i];
      AneLane &lane = lanes[p.lane];
      const size_t count = size_t(p.frames) * perFrame;
      if (p.mode == ChunkMode::Ane) {
        require(awaitReport(*reports[i]), "a Neural Engine evaluation failed");
        ane.advance(p.lane);
        std::vector<float> samples(count);
        ane.samples(p.lane, p.function, samples);
        lane.samples.insert(lane.samples.end(), samples.begin(), samples.end());
      } else {
        const std::vector<float> streamed = readBack(p.output, count);
        lane.samples.insert(lane.samples.end(), streamed.begin(), streamed.end());
      }
      lane.first += p.frames;
      ++lane.next;
    }
  }
}

// The median wall time of an evaluation of each function, the event raised
// before it is queued.
void timeFunctions(metal::MetalBackend &backend, CodecAneStack &ane, uint32_t perFrame) {
  metal::SharedEvent event = backend.newSharedEvent();
  uint64_t value = 1000;
  for (const uint32_t function : ane.functionFrames()) {
    std::vector<double> times;
    for (int pass = 0; pass < 7; ++pass) {
      event.signal(++value);
      auto report = std::make_shared<AneReport>();
      const auto start = std::chrono::steady_clock::now();
      ane.enqueue(0, function, event, value, value, [report](bool success) {
        std::lock_guard lock(report->mutex);
        report->reported = true;
        report->success = success;
        report->changed.notify_all();
      });
      require(awaitReport(*report), "a timed Neural Engine evaluation failed");
      times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    std::printf("codec ANE stack: %2u frames in %.2f ms (%.3f ms per frame, %u samples)\n", function, times[3],
                times[3] / function, function * perFrame);
  }
  ane.reset(0);
}

// The stack on the Neural Engine against the whole GPU decode of `codes`
// (and against mlx-audio's waveform when given): lane 0 in the engine's
// schedule (4, 8, 16, then 25 frames, the last chunk padded up to the next
// function), lane 1 through every handoff (a chunk on the GPU, the GPU's
// histories into the lane, an evaluation given up and decoded by the GPU
// from the lane's input and histories, back to the Neural Engine).
void testAneStack(metal::MetalBackend &backend, CodecDecoder &decoder, CodecAneStack &ane,
                  const std::vector<uint32_t> &codes, const char *what, double rmsBound,
                  const std::vector<float> *reference) {
  const CodecDims &d = decoder.weights().dims;
  const uint32_t perFrame = d.samplesPerFrame(), frames = static_cast<uint32_t>(codes.size() / d.quantizers);
  metal::MetalBuffer codeBuffer = sharedBuffer(backend, codes.size() * 4);
  std::memcpy(codeBuffer.contents(), codes.data(), codes.size() * 4);
  metal::MetalBuffer whole = sharedBuffer(backend, size_t(frames) * perFrame * 4);
  {
    metal::CommandGraph g;
    decoder.addDecode(g, codeBuffer, frames, whole);
    run(backend, g);
  }
  const std::vector<float> expected = readBack(whole, size_t(frames) * perFrame);
  double peak = 0.0;
  for (float v : expected) peak = std::max(peak, std::fabs(double(v)));

  const auto schedule = [&](std::vector<std::pair<uint32_t, ChunkMode>> head) {
    std::vector<std::pair<uint32_t, ChunkMode>> chunks;
    uint32_t used = 0;
    for (const auto &[chunk, mode] : head) {
      if (used >= frames) break;
      chunks.emplace_back(std::min(chunk, frames - used), mode);
      used += chunks.back().first;
    }
    for (; used < frames; used += chunks.back().first) chunks.emplace_back(std::min(25u, frames - used), ChunkMode::Ane);
    return chunks;
  };
  std::array<AneLane, 2> lanes{
      AneLane{schedule({{4, ChunkMode::Ane}, {8, ChunkMode::Ane}, {16, ChunkMode::Ane}}),
              CodecStream(backend, decoder.weights(), 25), 0, 0, true, {}},
      AneLane{schedule({{4, ChunkMode::Gpu}, {8, ChunkMode::Ane}, {16, ChunkMode::Recover}}),
              CodecStream(backend, decoder.weights(), 25), 0, 0, true, {}}};
  for (int pass = 0; pass < 2; ++pass) {
    for (AneLane &lane : lanes) {
      lane.first = 0;
      lane.next = 0;
      lane.samples.clear();
    }
    streamLanes(backend, decoder, ane, lanes, codes);
    for (size_t l = 0; l < lanes.size(); ++l) {
      require(lanes[l].samples.size() == expected.size(), "an ANE stream produced the wrong number of samples");
      const Distance e = distance(lanes[l].samples, expected, expected.size());
      std::printf("%s: ANE stream %zu (pass %d, %u frames) vs the whole GPU decode: max abs error %.3e at %zu, rms "
                  "%.3e (peak %.3f)\n",
                  what, l, pass, frames, e.maxError, e.worst, e.rms, peak);
      require(e.rms <= rmsBound, "an ANE stream differs from the GPU decode by more than fp16 noise");
      if (reference && pass == 0) {
        const Distance r = distance(lanes[l].samples, *reference, reference->size());
        const Distance g = distance(expected, *reference, reference->size());
        std::printf("%s: vs mlx-audio over %zu samples: ANE stream %zu max abs error %.3e rms %.3e; GPU max abs "
                    "error %.3e rms %.3e\n",
                    what, reference->size(), l, r.maxError, r.rms, g.maxError, g.rms);
        require(r.rms < 2e-3, "an ANE stream differs from mlx-audio by more than the GPU's noise");
      }
    }
  }
}

void testAne(metal::MetalBackend &backend) {
  if (const auto reason = ane::unavailable()) {
    std::cout << "codec ANE stack skipped: " << *reason << "\n";
    return;
  }
  const ane::Program::Limits limits{std::chrono::seconds(600), [] { return false; }};
  {
    const HostModel model = randomModel();
    CodecDecoder decoder(backend, uploadModel(backend, model), 128);
    CodecAneStack ane(backend, decoder.weights(), {4, 8, 16, 25}, 2, limits);
    const CodecDims &d = decoder.weights().dims;
    testAneStack(backend, decoder, ane, xorshiftCodes(85, d.quantizers, 7, d.codebookSize), "random decoder", 2e-3,
                 nullptr);
    rejects([&] { ane.enqueue(0, 5, backend.newSharedEvent(), 1, 1, [](bool) {}); }, "no function of 5 frames",
            "an evaluation of no function was queued");
    require(ane.functionFor(3) == 4 && ane.functionFor(17) == 25 && ane.functionFor(26) == 0,
            "the ANE stack's function for a chunk is not the smallest that holds it");
  }
  const std::filesystem::path tokenizer = findSpeechTokenizer();
  if (tokenizer.empty()) {
    std::cout << "real codec ANE stack skipped (no cached Qwen3-TTS model)\n";
    return;
  }
  CodecDecoder decoder(backend, loadCodecDecoderWeights(backend, tokenizer), 128);
  const CodecDims &d = decoder.weights().dims;
  const auto start = std::chrono::steady_clock::now();
  CodecAneStack ane(backend, decoder.weights(), {4, 8, 16, 25}, 2, limits);
  std::printf("real decoder: codec ANE stack compiled and loaded in %.1f s, %.1f MB of surfaces\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), double(ane.bytes()) / 1e6);
  testAneStack(backend, decoder, ane, xorshiftCodes(85, d.quantizers, 7, d.codebookSize), "real decoder", 2e-3, nullptr);
  // mlx-audio's waveform of the fixture's codes.
  std::ifstream in("dev/tests/fixtures/omni/codec_40_1.bin", std::ios::binary);
  char magic[4];
  uint32_t frames = 0, seed = 0, samples = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char *>(&frames), 4);
  in.read(reinterpret_cast<char *>(&seed), 4);
  in.read(reinterpret_cast<char *>(&samples), 4);
  if (in && std::memcmp(magic, "OCRF", 4) == 0) {
    std::vector<float> reference(samples);
    in.read(reinterpret_cast<char *>(reference.data()), size_t(samples) * 4);
    require(bool(in), "the codec fixture is truncated");
    testAneStack(backend, decoder, ane, xorshiftCodes(frames, d.quantizers, seed, d.codebookSize),
                 "real decoder (mlx-audio fixture)", 2e-3, &reference);
  }
  timeFunctions(backend, ane, d.samplesPerFrame());
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: omni-codec METALLIB [--ane]\n";
    return 2;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    // The Neural Engine cannot share an event Metal's validation layer
    // wraps: --ane runs the stack's tests alone, without the validation.
    if (argc > 2 && std::string_view(argv[2]) == "--ane") {
      testAne(backend);
      std::cout << "omni codec ANE ok\n";
      return 0;
    }
    testConvKernels(backend);
    testRandomDecoder(backend);
    testRealDecoder(backend);
  } catch (const std::exception &error) {
    std::cerr << "omni codec: " << error.what() << "\n";
    return 1;
  }
  std::cout << "omni codec ok\n";
  return 0;
}
