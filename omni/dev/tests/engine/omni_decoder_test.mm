// The Qwen3 decoder forward (runtime/omni/Qwen3Decoder) at a small shape
// with random weights, in both projection formats, against an fp64
// reference of the whole model: a ragged prefill of two sequences, then
// decode steps over both lanes, then a verify forward of several rows per
// lane over their histories. argv[1] names the production metallib.

#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"
#include "model/WeightStore.hpp"
#include "omni/KvPages.hpp"
#include "omni/Qwen3Decoder.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops::tuning;
using namespace splash::omni;
using namespace splash::test;

namespace {

std::mt19937 rng(11);
float uniform(float low, float high) { return std::uniform_real_distribution<float>(low, high)(rng); }

constexpr uint32_t kHidden = 256, kIntermediate = 512, kLayers = 2, kHeadRows = 512, kVocabulary = 500;
constexpr uint32_t kQkv = OMNI_DECODER_QKV_WIDTH, kAttn = OMNI_DECODER_ATTENTION_WIDTH, kD = OMNI_DECODER_HEAD_DIM;

// A matrix [n][k] of floats, and its forms: exact (dense bf16) or affine Q4.
struct Matrix {
  uint32_t n = 0, k = 0;
  std::vector<float> values;          // as generated
  std::vector<double> effective;      // what the kernels multiply (bf16 or dequantized Q4)
};

Matrix randomMatrix(uint32_t n, uint32_t k) {
  Matrix m{n, k, std::vector<float>(size_t(n) * k), {}};
  const float bound = 1.0F / std::sqrt(float(k));
  for (float &v : m.values) v = uniform(-bound, bound);
  return m;
}

ops::omni::DenseWeights denseOf(metal::MetalBackend &backend, Matrix &m) {
  metal::MetalBuffer weight = sharedBuffer(backend, m.values.size() * 2);
  auto *bits = static_cast<uint16_t *>(weight.contents());
  m.effective.resize(m.values.size());
  for (size_t i = 0; i < m.values.size(); ++i) {
    bits[i] = floatToBf16(m.values[i]);
    m.effective[i] = bf16ToFloat(bits[i]);
  }
  return {weight, {}, m.n, m.k};
}

// MLX affine Q4 (group 64) in the kernels' 256-row storage tiles.
ops::Projection affineOf(metal::MetalBackend &backend, Matrix &m) {
  const uint32_t groups = m.k / 64;
  const uint64_t parameters = uint64_t{m.n} * groups;
  ops::Projection p(m.n, m.k, ops::AffineWeights{sharedBuffer(backend, parameters * 32), sharedBuffer(backend, parameters * 2),
                                                 sharedBuffer(backend, parameters * 2)});
  auto *codes = static_cast<uint8_t *>(p.affine().weights.contents());
  auto *scales = static_cast<uint16_t *>(p.affine().scales.contents());
  auto *biases = static_cast<uint16_t *>(p.affine().biases.contents());
  std::memset(codes, 0, parameters * 32);
  m.effective.assign(m.values.size(), 0.0);
  for (uint32_t n = 0; n < m.n; ++n)
    for (uint32_t g = 0; g < groups; ++g) {
      const float *group = &m.values[size_t(n) * m.k + g * 64];
      float low = group[0], high = group[0];
      for (uint32_t i = 1; i < 64; ++i) {
        low = std::min(low, group[i]);
        high = std::max(high, group[i]);
      }
      const float scale = std::max((high - low) / 15.0F, 1e-7F);
      const uint16_t scaleBits = floatToBf16(scale), biasBits = floatToBf16(low);
      const double scaleValue = bf16ToFloat(scaleBits), biasValue = bf16ToFloat(biasBits);
      const uint64_t at = (uint64_t(n / 256) * groups + g) * 256 + n % 256;
      scales[at] = scaleBits;
      biases[at] = biasBits;
      for (uint32_t i = 0; i < 64; ++i) {
        const int q = std::clamp(int(std::lround((group[i] - low) / scale)), 0, 15);
        codes[at * 32 + i / 2] |= uint8_t(q << (4 * (i % 2)));
        m.effective[size_t(n) * m.k + g * 64 + i] = scaleValue * q + biasValue;
      }
    }
  return p;
}

struct LayerRef {
  Matrix qkv, o, gate, up, down;
  std::vector<float> norm1, norm2, qNorm, kNorm;
};
struct ModelRef {
  std::vector<LayerRef> layers;
  Matrix head;
  std::vector<float> finalNorm;
};

ops::NormWeights normOf(metal::MetalBackend &backend, std::vector<float> &values) {
  ops::NormWeights norm{sharedBuffer(backend, values.size() * 2)};
  auto *bits = static_cast<uint16_t *>(norm.buffer.contents());
  for (size_t i = 0; i < values.size(); ++i) {
    bits[i] = floatToBf16(values[i]);
    values[i] = bf16ToFloat(bits[i]);
  }
  return norm;
}
metal::MetalBuffer bf16Of(metal::MetalBackend &backend, std::vector<float> &values) { return normOf(backend, values).buffer; }

std::vector<float> randomNorm(uint32_t width) {
  std::vector<float> v(width);
  for (float &x : v) x = uniform(0.5F, 1.5F);
  return v;
}

Qwen3Weights buildWeights(metal::MetalBackend &backend, ModelRef &ref, bool affine) {
  Qwen3Weights weights;
  weights.dims = {kLayers, kHidden, kIntermediate, kHeadRows, kVocabulary, 1e-6F, 1e6};
  const auto project = [&](Matrix &m) -> Qwen3Projection {
    if (affine) return affineOf(backend, m);
    return denseOf(backend, m);
  };
  for (uint32_t l = 0; l < kLayers; ++l) {
    LayerRef layer{randomMatrix(kQkv, kHidden), randomMatrix(kHidden, kAttn), randomMatrix(kIntermediate, kHidden),
                   randomMatrix(kIntermediate, kHidden), randomMatrix(kHidden, kIntermediate),
                   randomNorm(kHidden), randomNorm(kHidden), randomNorm(kD), randomNorm(kD)};
    Qwen3LayerWeights w;
    w.inputNorm = normOf(backend, layer.norm1);
    w.postAttentionNorm = normOf(backend, layer.norm2);
    w.queryNorm = bf16Of(backend, layer.qNorm);
    w.keyNorm = bf16Of(backend, layer.kNorm);
    w.qkv = project(layer.qkv);
    w.output = project(layer.o);
    w.gate = project(layer.gate);
    w.up = project(layer.up);
    w.down = project(layer.down);
    weights.layers.push_back(std::move(w));
    ref.layers.push_back(std::move(layer));
  }
  ref.head = randomMatrix(kHeadRows, kHidden);
  ref.finalNorm = randomNorm(kHidden);
  weights.finalNorm = normOf(backend, ref.finalNorm);
  weights.head = project(ref.head);
  return weights;
}

// ---------------------------------------------------------------- reference

using Row = std::vector<double>;
Row bf16Round(const Row &row) {
  Row out(row.size());
  for (size_t i = 0; i < row.size(); ++i) out[i] = bf16ToFloat(floatToBf16(float(row[i])));
  return out;
}
Row rmsNorm(const Row &x, const std::vector<float> &w, double epsilon) {
  double sum = 0.0;
  for (double v : x) sum += v * v;
  const double inverse = 1.0 / std::sqrt(sum / double(x.size()) + epsilon);
  Row out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * inverse * w[i];
  return bf16Round(out);
}
Row matmul(const Matrix &m, const Row &x) {
  Row out(m.n);
  for (uint32_t n = 0; n < m.n; ++n) {
    double sum = 0.0;
    for (uint32_t k = 0; k < m.k; ++k) sum += m.effective[size_t(n) * m.k + k] * x[k];
    out[n] = sum;
  }
  return out;
}
double silu(double x) { return x / (1.0 + std::exp(-x)); }

struct Sequence {
  std::vector<Row> embeddings;                       // all rows so far (bf16 values)
  std::vector<std::vector<Row>> keys, values;        // [layer][position] prepared keys / stored values
};

// Runs rows [first, embeddings.size()) of a sequence through the reference,
// extending its KV, and returns the final-normed hidden of the last row and
// its logits.
std::pair<Row, Row> referenceForward(const ModelRef &ref, Sequence &s, size_t first) {
  s.keys.resize(kLayers);
  s.values.resize(kLayers);
  std::vector<Row> hidden(s.embeddings.begin() + first, s.embeddings.end());
  std::vector<double> inverseFrequencies(kD / 2);
  for (uint32_t i = 0; i < kD / 2; ++i) inverseFrequencies[i] = float(std::pow(1e6, -2.0 * i / kD));
  for (uint32_t l = 0; l < kLayers; ++l) {
    const LayerRef &layer = ref.layers[l];
    std::vector<Row> attentionOut(hidden.size());
    std::vector<Row> queries(hidden.size());
    for (size_t r = 0; r < hidden.size(); ++r) {
      const size_t position = first + r;
      const Row normalized = rmsNorm(hidden[r], layer.norm1, 1e-6);
      const Row qkv = bf16Round(matmul(layer.qkv, normalized));
      Row q(kAttn), k(8 * kD), v(8 * kD);
      const auto prepare = [&](const double *head, const std::vector<float> &w) {
        double sum = 0.0;
        for (uint32_t d = 0; d < kD; ++d) sum += head[d] * head[d];
        const double inverse = 1.0 / std::sqrt(sum / kD + 1e-6);
        Row n(kD), out(kD);
        for (uint32_t d = 0; d < kD; ++d) n[d] = bf16ToFloat(floatToBf16(float(head[d] * inverse * w[d])));
        for (uint32_t d = 0; d < kD / 2; ++d) {
          const double angle = double(position) * inverseFrequencies[d];
          out[d] = n[d] * std::cos(angle) - n[d + kD / 2] * std::sin(angle);
          out[d + kD / 2] = n[d + kD / 2] * std::cos(angle) + n[d] * std::sin(angle);
        }
        return bf16Round(out);
      };
      for (uint32_t h = 0; h < 16; ++h) {
        const Row head = prepare(&qkv[h * kD], layer.qNorm);
        std::copy(head.begin(), head.end(), q.begin() + h * kD);
      }
      for (uint32_t h = 0; h < 8; ++h) {
        const Row head = prepare(&qkv[(16 + h) * kD], layer.kNorm);
        std::copy(head.begin(), head.end(), k.begin() + h * kD);
        std::copy(qkv.begin() + (24 + h) * kD, qkv.begin() + (25 + h) * kD, v.begin() + h * kD);
      }
      s.keys[l].push_back(k);
      s.values[l].push_back(v);
      queries[r] = q;
    }
    for (size_t r = 0; r < hidden.size(); ++r) {
      const size_t position = first + r;
      Row out(kAttn);
      for (uint32_t h = 0; h < 16; ++h) {
        const uint32_t kvHead = h / 2;
        std::vector<double> scores(position + 1);
        double maximum = -1e300;
        for (size_t t = 0; t <= position; ++t) {
          double dot = 0.0;
          for (uint32_t d = 0; d < kD; ++d) dot += queries[r][h * kD + d] * s.keys[l][t][kvHead * kD + d];
          scores[t] = dot / std::sqrt(double(kD));
          maximum = std::max(maximum, scores[t]);
        }
        double denominator = 0.0;
        Row acc(kD, 0.0);
        for (size_t t = 0; t <= position; ++t) {
          const double p = std::exp(scores[t] - maximum);
          denominator += p;
          for (uint32_t d = 0; d < kD; ++d) acc[d] += p * s.values[l][t][kvHead * kD + d];
        }
        for (uint32_t d = 0; d < kD; ++d) out[h * kD + d] = acc[d] / denominator;
      }
      attentionOut[r] = bf16Round(out);
    }
    for (size_t r = 0; r < hidden.size(); ++r) {
      Row projected = matmul(layer.o, attentionOut[r]);
      for (size_t i = 0; i < kHidden; ++i) projected[i] += hidden[r][i];
      const Row residual = bf16Round(projected);
      const Row normalized = rmsNorm(residual, layer.norm2, 1e-6);
      const Row gate = bf16Round(matmul(layer.gate, normalized)), up = bf16Round(matmul(layer.up, normalized));
      Row inter(kIntermediate);
      for (size_t i = 0; i < kIntermediate; ++i) inter[i] = silu(gate[i]) * up[i];
      inter = bf16Round(inter);
      Row down = matmul(layer.down, inter);
      for (size_t i = 0; i < kHidden; ++i) down[i] += residual[i];
      hidden[r] = bf16Round(down);
    }
  }
  const Row final = rmsNorm(hidden.back(), ref.finalNorm, 1e-6);
  return {final, matmul(ref.head, final)};
}

// ---------------------------------------------------------------- the test

void compareLogits(const metal::MetalBuffer &logits, uint32_t row, const Row &expected, const std::string &what) {
  const auto *values = static_cast<const float *>(logits.contents()) + size_t(row) * kHeadRows;
  double worst = 0.0, scale = 0.0;
  for (uint32_t v = 0; v < kVocabulary; ++v) {
    worst = std::max(worst, std::fabs(double(values[v]) - expected[v]));
    scale = std::max(scale, std::fabs(expected[v]));
  }
  // Two layers of bf16 activations: a few percent of the logit range.
  if (!(worst <= 0.04 * scale + 0.02))
    throw std::runtime_error(what + ": logits differ by " + std::to_string(worst) + " (scale " + std::to_string(scale) + ")");
  std::cout << "  " << what << ": max logit error " << worst << " of scale " << scale << "\n";
}

void testFormat(metal::MetalBackend &backend, bool affine) {
  const std::string format = affine ? "affine Q4" : "dense bf16";
  ModelRef ref;
  Qwen3Decoder decoder(backend, buildWeights(backend, ref, affine));
  KvPagePool pool(backend, 64);
  LanePageTables tables(backend, 2, kLayers, 4);
  LaneKv lane0(pool, tables, 0), lane1(pool, tables, 1);
  const KvBinding kv{tables.buffer(), tables.entries(), pool.pages(), pool.pageCount()};
  Qwen3Arena prefillArena(backend, decoder.linear(), decoder.weights(), 128, 2, 2, false);
  Qwen3Arena decodeArena(backend, decoder.linear(), decoder.weights(), 8, 8, 8, true);
  std::vector<Sequence> sequences(2);
  const uint32_t lengths[2] = {37, 20};
  // Prefill both sequences in one ragged dispatch.
  {
    std::vector<ops::omni::AttentionSequence> descriptors;
    uint32_t rows = 0;
    auto *hidden = static_cast<uint16_t *>(prefillArena.hidden.contents());
    for (uint32_t s = 0; s < 2; ++s) {
      LaneKv &lane = s ? lane1 : lane0;
      require(lane.reserve(lengths[s]), "reserve pages");
      for (uint32_t r = 0; r < lengths[s]; ++r) {
        Row embedding(kHidden);
        for (uint32_t i = 0; i < kHidden; ++i) {
          const uint16_t bits = floatToBf16(uniform(-1.0F, 1.0F));
          hidden[(size_t(rows) + r) * kHidden + i] = bits;
          embedding[i] = bf16ToFloat(bits);
        }
        sequences[s].embeddings.push_back(embedding);
      }
      descriptors.push_back({tables.offset(s), tables.stride(), 0, lengths[s], rows, 0});
      rows += lengths[s];
    }
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
    Qwen3Decoder::uploadTables(prefillArena, attention);
    auto *logitRows = static_cast<uint32_t *>(prefillArena.logitRows.contents());
    logitRows[0] = lengths[0] - 1;
    logitRows[1] = lengths[0] + lengths[1] - 1;
    metal::CommandGraph graph;
    decoder.addForward(graph, prefillArena, kv, {&attention, 2});
    metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
    (void)ticket.wait();
    for (uint32_t s = 0; s < 2; ++s) {
      const auto [final, logits] = referenceForward(ref, sequences[s], 0);
      compareLogits(prefillArena.logits, s, logits, format + " prefill sequence " + std::to_string(s));
    }
  }
  // One decode step over both lanes.
  for (uint32_t step = 0; step < 2; ++step) {
    std::vector<ops::omni::AttentionSequence> descriptors;
    auto *hidden = static_cast<uint16_t *>(decodeArena.hidden.contents());
    for (uint32_t s = 0; s < 2; ++s) {
      LaneKv &lane = s ? lane1 : lane0;
      const uint32_t committed = static_cast<uint32_t>(sequences[s].embeddings.size());
      require(lane.reserve(committed + 1), "reserve decode page");
      Row embedding(kHidden);
      for (uint32_t i = 0; i < kHidden; ++i) {
        const uint16_t bits = floatToBf16(uniform(-1.0F, 1.0F));
        hidden[size_t(s) * kHidden + i] = bits;
        embedding[i] = bf16ToFloat(bits);
      }
      sequences[s].embeddings.push_back(embedding);
      descriptors.push_back({tables.offset(s), tables.stride(), committed, 1, s, committed});
    }
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, true);
    Qwen3Decoder::uploadTables(decodeArena, attention);
    auto *logitRows = static_cast<uint32_t *>(decodeArena.logitRows.contents());
    logitRows[0] = 0;
    logitRows[1] = 1;
    metal::CommandGraph graph;
    decoder.addForward(graph, decodeArena, kv, {&attention, 2});
    metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
    (void)ticket.wait();
    for (uint32_t s = 0; s < 2; ++s) {
      const auto [final, logits] = referenceForward(ref, sequences[s], sequences[s].embeddings.size() - 1);
      compareLogits(decodeArena.logits, s, logits, format + " decode step " + std::to_string(step) + " lane " + std::to_string(s));
    }
  }
  // A verify forward (decode projections, ragged prefill attention over the
  // histories): three rows of lane 0 and five of lane 1, every row's logits.
  {
    Qwen3Arena verifyArena(backend, decoder.linear(), decoder.weights(), 32, 2, 32, true, true);
    const uint32_t counts[2] = {3, 5};
    std::vector<ops::omni::AttentionSequence> descriptors;
    std::vector<Row> expected;
    auto *hidden = static_cast<uint16_t *>(verifyArena.hidden.contents());
    uint32_t rows = 0;
    for (uint32_t s = 0; s < 2; ++s) {
      LaneKv &lane = s ? lane1 : lane0;
      const uint32_t committed = static_cast<uint32_t>(sequences[s].embeddings.size());
      require(lane.reserve(committed + counts[s]), "reserve verify pages");
      for (uint32_t r = 0; r < counts[s]; ++r) {
        Row embedding(kHidden);
        for (uint32_t i = 0; i < kHidden; ++i) {
          const uint16_t bits = floatToBf16(uniform(-1.0F, 1.0F));
          hidden[size_t(rows + r) * kHidden + i] = bits;
          embedding[i] = bf16ToFloat(bits);
        }
        sequences[s].embeddings.push_back(embedding);
        expected.push_back(referenceForward(ref, sequences[s], sequences[s].embeddings.size() - 1).second);
      }
      descriptors.push_back({tables.offset(s), tables.stride(), committed, counts[s], rows, committed});
      rows += counts[s];
    }
    const ops::omni::AttentionTables attention = ops::omni::attentionTables(descriptors, false);
    Qwen3Decoder::uploadTables(verifyArena, attention);
    auto *logitRows = static_cast<uint32_t *>(verifyArena.logitRows.contents());
    for (uint32_t r = 0; r < rows; ++r) logitRows[r] = r;
    metal::CommandGraph graph;
    decoder.addForward(graph, verifyArena, kv, {&attention, rows});
    metal::CommandTicket ticket = backend.submitCommandAsync(graph.command());
    (void)ticket.wait();
    for (uint32_t r = 0; r < rows; ++r)
      compareLogits(verifyArena.logits, r, expected[r], format + " verify row " + std::to_string(r));
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: omni-decoder METALLIB\n";
    return 2;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    testFormat(backend, false);
    testFormat(backend, true);
  } catch (const std::exception &error) {
    std::cerr << "omni decoder: " << error.what() << "\n";
    return 1;
  }
  std::cout << "omni decoder ok\n";
  return 0;
}
