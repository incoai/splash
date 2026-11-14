// The parity gate: the tiny Ripple model's prefill through the engine's real
// machinery — the loaded weights, the prefill arena's buffers, the paged K/V,
// the target's dispatches and the head — against a CPU reference that reads
// the fixture's GGUF itself and runs the pinned block graph in double
// precision. The reference is independent of the loader: it dequantizes the
// GGUF's own bytes, so a loader bug cannot hide in it.
#include "LinearNumerics.hpp"
#include "TestChecks.hpp"
#include "ClaimScope.hpp"
#include "engine/MemoryGovernor.hpp"
#include "ops/RowCopy.hpp"
#include "model/ModelFactory.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/Runtime.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/PagedKv.hpp"
#include "ops/PageStorage.hpp"
#include "ops/RoPE.hpp"

#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::model::LoadedModel;
using splash::model::QwenTarget;
using splash::model::QwenTargetPrefillBuffers;
using splash::model::QwenTargetPrefillChunk;
using splash::model::QwenTargetPrefillSequence;
using splash::model::QwenTargetRequest;
using splash::ops::ExecutionPlans;
using splash::model::RuntimeGeometry;
using splash::model::SpilledKvReads;
using splash::test::require;
using splash::test::bf16ToFloat;

using namespace splash;

constexpr uint32_t kRows = 4;
constexpr uint32_t kLayers = 2;
constexpr uint32_t kHidden = 256;
constexpr uint32_t kVocab = 256;
constexpr uint32_t kQueryHeads = 80;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kPairs = 64;
constexpr uint32_t kIntermediate = 512;
constexpr uint32_t kAttentionWidth = kQueryHeads * kHeadDim;
constexpr uint32_t kKvWidth = kKvHeads * kHeadDim;
constexpr float kEmbedScale = 1.0F;
constexpr float kClamp = 6.0F;
constexpr float kTheta = 10000.0F;
constexpr uint32_t kWindow = 1024; // the family's: keys total, the layout's 1023 before

// ---------------------------------------------------------------- GGUF read

constexpr uint32_t kTypeF32 = 0, kTypeString = 8, kTypeArray = 9, kTypeQ8_0 = 8;

float f16Bits(uint16_t bits) {
  const uint32_t sign = bits >> 15 & 1, exponent = bits >> 10 & 0x1F, fraction = bits & 0x3FF;
  if (!exponent) return std::ldexp(fraction, -24) * (sign ? -1.0F : 1.0F);
  if (exponent == 31) return sign ? -INFINITY : INFINITY;
  return std::ldexp((fraction | 0x400) * (sign ? -1.0F : 1.0F), static_cast<int>(exponent) - 25);
}

class Gguf final {
public:
  explicit Gguf(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("the fixture's GGUF is missing");
    std::vector<char> bytes(std::filesystem::file_size(path));
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    data_ = std::move(bytes);
    uint32_t magic = 0, version = 0;
    read(magic), read(version);
    if (magic != 0x46554747 || version != 3) throw std::runtime_error("not a GGUF v3");
    uint64_t tensorCount = 0, metadataCount = 0;
    read(tensorCount), read(metadataCount);
    for (uint64_t index = 0; index < metadataCount; ++index) { readString(); readValue(); }
    for (uint64_t index = 0; index < tensorCount; ++index) {
      const std::string name = readString();
      uint32_t dimensions = 0;
      read(dimensions);
      std::vector<uint64_t> shape(dimensions);
      for (uint32_t axis = 0; axis < dimensions; ++axis) read(shape[axis]);
      uint32_t kind = 0;
      uint64_t offset = 0;
      read(kind), read(offset);
      tensors_.push_back({name, shape, kind, offset});
    }
    const uint64_t aligned = cursor_;
    dataStart_ = aligned + (aligned % 32 ? 32 - aligned % 32 : 0);
  }

  [[nodiscard]] const auto &tensors() const noexcept { return tensors_; }
  // The tensor's values as doubles, dequantized: Q8_0 blocks of a float16
  // scale and 32 int8 elements, or float32 rows.
  [[nodiscard]] std::vector<double> values(const std::string &name) const {
    for (const auto &tensor : tensors_)
      if (tensor.name == name) {
        uint64_t count = 1;
        for (const uint64_t dimension : tensor.shape) count *= dimension;
        const uint8_t *base = reinterpret_cast<const uint8_t *>(data_.data()) + dataStart_ + tensor.offset;
        std::vector<double> out(count);
        if (tensor.kind == kTypeQ8_0) {
          for (uint64_t block = 0; block * 32 < count; ++block) {
            const float scale = f16Bits(static_cast<uint16_t>(
                base[block * 34] | base[block * 34 + 1] << 8));
            const uint8_t *quantized = base + block * 34 + 2;
            for (uint32_t element = 0; element < 32 && block * 32 + element < count; ++element)
              out[block * 32 + element] = static_cast<int8_t>(quantized[element]) * scale;
          }
        } else if (tensor.kind == kTypeF32) {
          for (uint64_t element = 0; element < count; ++element) {
            float value = 0.0F;
            std::memcpy(&value, base + 4 * element, 4);
            out[element] = value;
          }
        } else throw std::runtime_error("unexpected GGUF type");
        return out;
      }
    throw std::runtime_error("the GGUF holds no tensor " + name);
  }

private:
  template <class T> void read(T &value) {
    std::memcpy(&value, data_.data() + cursor_, sizeof(T));
    cursor_ += sizeof(T);
  }
  std::string readString() {
    uint64_t length = 0;
    read(length);
    std::string out(data_.data() + cursor_, length);
    cursor_ += length;
    return out;
  }
  void readValue() {
    uint32_t kind = 0;
    read(kind);
    if (kind == kTypeString) {
      readString();
    } else if (kind == kTypeArray) {
      uint32_t elementKind = 0;
      uint64_t count = 0;
      read(elementKind), read(count);
      for (uint64_t index = 0; index < count; ++index)
        if (elementKind == kTypeString) readString();
        else if (elementKind == 4) { uint32_t v; read(v); }
        else if (elementKind == 5) { int32_t v; read(v); }
        else if (elementKind == 6) { float v; read(v); }
        else if (elementKind == 7) { uint8_t v; read(v); }
        else if (elementKind == 10) { uint64_t v; read(v); }
        else throw std::runtime_error("unexpected GGUF array element type");
    } else if (kind == 4) { uint32_t v; read(v); }
    else if (kind == 5) { int32_t v; read(v); }
    else if (kind == 6) { float v; read(v); }
    else if (kind == 7) { uint8_t v; read(v); }
    else if (kind == 10) { uint64_t v; read(v); }
    else throw std::runtime_error("unexpected GGUF value type");
  }
  std::vector<char> data_;
  uint64_t cursor_ = 0, dataStart_ = 0;
  struct Tensor final {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t kind;
    uint64_t offset;
  };
  std::vector<Tensor> tensors_;
};

// ------------------------------------------------------------- the reference

// The pinned block graph (dev/benchmarks/ripple/STEP1-NOTES.md), in double:
// post-norm blocks, the value residual's mix, per-head query and key norms,
// the elementwise output gate, the clamped SwiGLU, and NoPE on the full
// layers. Layer 0 slides (its history here is shorter than the window, so
// the mask does not bind), layer 1 is full.
struct Reference final {
  const Gguf &gguf;

  std::vector<double> matmul(const std::vector<double> &weight, uint32_t rows, uint32_t cols,
                             const std::vector<double> &input) const {
    std::vector<double> out(rows, 0.0);
    for (uint32_t row = 0; row < rows; ++row) {
      double sum = 0.0;
      for (uint32_t column = 0; column < cols; ++column)
        sum += weight[uint64_t{row} * cols + column] * input[column];
      out[row] = sum;
    }
    return out;
  }

  std::vector<double> rmsNorm(const std::vector<double> &x, const std::vector<double> &weight) const {
    double sum = 0.0;
    for (double value : x) sum += value * value;
    const double inverse = 1.0 / std::sqrt(sum / x.size() + 1e-6);
    std::vector<double> out(x.size());
    for (size_t index = 0; index < x.size(); ++index) out[index] = x[index] * inverse * weight[index];
    return out;
  }

  std::vector<double> perHeadNorm(const std::vector<double> &x, uint32_t heads,
                                  const std::vector<double> &weight) const {
    std::vector<double> out(x.size());
    for (uint32_t head = 0; head < heads; ++head) {
      std::vector<double> head_(x.begin() + uint64_t{head} * kHeadDim,
                                x.begin() + uint64_t{head + 1} * kHeadDim);
      const auto normed = rmsNorm(head_, weight);
      std::copy(normed.begin(), normed.end(), out.begin() + uint64_t{head} * kHeadDim);
    }
    return out;
  }

  // NeoX half-split pairs, the tables' convention: pair i of a head rotates
  // dimensions i and i + pairs by position * theta^(-i / pairs).
  std::vector<double> rope(std::vector<double> x, uint32_t position) const {
    for (uint32_t pair = 0; pair < kPairs; ++pair) {
      const double angle = double(position) * std::pow(double{kTheta}, -double(pair) / double(kPairs));
      const double cosine = std::cos(angle), sine = std::sin(angle);
      const double first = x[pair], second = x[pair + kPairs];
      x[pair] = first * cosine - second * sine;
      x[pair + kPairs] = second * cosine + first * sine;
    }
    return x;
  }

  double sigmoid(double value) const { return 1.0 / (1.0 + std::exp(-value)); }

  double silu(double value) const { return value * sigmoid(value); }

  double clamp(double value) const { return std::fmin(std::fmax(value, -double{kClamp}), double{kClamp}); }

  // The logits of `tokens` from position 0 on, as double rows.
  [[nodiscard]] std::vector<std::vector<double>> logits(std::span<const uint32_t> tokens) const {
    const auto embedding = gguf.values("token_embd.weight"); // [vocab][hidden]
    const auto finalNorm = gguf.values("output_norm.weight");
    const auto output = gguf.values("output.weight"); // [vocab][hidden]
    std::vector<std::vector<double>> hidden(tokens.size(), std::vector<double>(kHidden, 0.0));
    for (uint32_t row = 0; row < tokens.size(); ++row)
      for (uint32_t column = 0; column < kHidden; ++column)
        hidden[row][column] = embedding[uint64_t{tokens[row]} * kHidden + column] * kEmbedScale;

    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      const std::string p = "blk." + std::to_string(layer) + ".";
      const auto attnNorm = gguf.values(p + "attn_norm.weight");
      const auto query = gguf.values(p + "attn_q.weight");
      const auto gate = gguf.values(p + "attn_gate.weight");
      const auto keys = gguf.values(p + "attn_k.weight");
      const auto values = gguf.values(p + "attn_v.weight");
      const auto queryNorm = gguf.values(p + "attn_q_norm.weight");
      const auto keyNorm = gguf.values(p + "attn_k_norm.weight");
      const auto outProjection = gguf.values(p + "attn_output.weight");
      const double lambda = gguf.values(p + "attn_vres_lambda.weight")[0];
      const auto postAttention = gguf.values(p + "post_attention_norm.weight");
      const auto ffnGate = gguf.values(p + "ffn_gate.weight");
      const auto ffnUp = gguf.values(p + "ffn_up.weight");
      const auto ffnDown = gguf.values(p + "ffn_down.weight");
      const auto ffnNorm = gguf.values(p + "ffn_norm.weight");
      const auto postFfw = gguf.values(p + "post_ffw_norm.weight");
      const bool sliding = layer == 0;

      // The mixer's inputs, from the attention norm's rows.
      std::vector<std::vector<double>> mixed(tokens.size());
      std::vector<std::vector<double>> vIn(tokens.size());
      for (uint32_t row = 0; row < tokens.size(); ++row) {
        const auto a = rmsNorm(hidden[row], attnNorm);
        mixed[row] = a;
        // The value residual's mix: sigma(lambda) of the normed rows over
        // the scaled embedding.
        const double weight = sigmoid(lambda);
        vIn[row].resize(kHidden);
        for (uint32_t column = 0; column < kHidden; ++column)
          vIn[row][column] = weight * a[column] +
                             (1.0 - weight) * embedding[uint64_t{tokens[row]} * kHidden + column] * kEmbedScale;
      }

      // Queries, keys, values: the projections, the per-head norms, the
      // rotation where the layer rotates.
      std::vector<std::vector<std::vector<double>>> q(tokens.size(),
                                                      std::vector<std::vector<double>>(kQueryHeads));
      std::vector<std::vector<std::vector<double>>> k(tokens.size(),
                                                      std::vector<std::vector<double>>(kKvHeads));
      std::vector<std::vector<std::vector<double>>> v(tokens.size(),
                                                      std::vector<std::vector<double>>(kKvHeads));
      for (uint32_t row = 0; row < tokens.size(); ++row) {
        const auto queries = matmul(query, kAttentionWidth, kHidden, mixed[row]);
        const auto normedQ = perHeadNorm(queries, kQueryHeads, queryNorm);
        const auto keyRows = matmul(keys, kKvWidth, kHidden, mixed[row]);
        const auto normedK = perHeadNorm(keyRows, kKvHeads, keyNorm);
        const auto valueRows = matmul(values, kKvWidth, kHidden, vIn[row]);
        for (uint32_t head = 0; head < kQueryHeads; ++head) {
          std::vector<double> head_(normedQ.begin() + uint64_t{head} * kHeadDim,
                                    normedQ.begin() + uint64_t{head + 1} * kHeadDim);
          q[row][head] = sliding ? rope(std::move(head_), row) : std::move(head_);
        }
        for (uint32_t head = 0; head < kKvHeads; ++head) {
          std::vector<double> head_(normedK.begin() + uint64_t{head} * kHeadDim,
                                    normedK.begin() + uint64_t{head + 1} * kHeadDim);
          k[row][head] = sliding ? rope(std::move(head_), row) : std::move(head_);
          std::vector<double> valueHead(valueRows.begin() + uint64_t{head} * kHeadDim,
                                        valueRows.begin() + uint64_t{head + 1} * kHeadDim);
          v[row][head] = std::move(valueHead);
        }
      }

      // Attention: causal over the history, the group's heads sharing a kv
      // head, scores of 1/sqrt(head dimension), the elementwise gate after.
      std::vector<std::vector<double>> attended(tokens.size(), std::vector<double>(kAttentionWidth, 0.0));
      for (uint32_t row = 0; row < tokens.size(); ++row) {
        const uint32_t start = sliding && row >= kWindow - 1 ? row - (kWindow - 1) : 0;
        for (uint32_t head = 0; head < kQueryHeads; ++head) {
          const uint32_t kvHead = head / (kQueryHeads / kKvHeads);
          std::vector<double> scores;
          for (uint32_t past = start; past <= row; ++past) {
            double score = 0.0;
            for (uint32_t dimension = 0; dimension < kHeadDim; ++dimension)
              score += q[row][head][dimension] * k[past][kvHead][dimension];
            scores.push_back(score / std::sqrt(double{kHeadDim}));
          }
          const double maximum = *std::max_element(scores.begin(), scores.end());
          double denominator = 0.0;
          for (double &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          for (uint32_t dimension = 0; dimension < kHeadDim; ++dimension) {
            double sum = 0.0;
            for (uint32_t past = start; past <= row; ++past)
              sum += scores[past - start] / denominator * v[past][kvHead][dimension];
            attended[row][uint64_t{head} * kHeadDim + dimension] = sum;
          }
        }
        // The output gate, the attention's rows scaled elementwise.
        const auto gateRows = matmul(gate, kAttentionWidth, kHidden, mixed[row]);
        for (uint32_t element = 0; element < kAttentionWidth; ++element)
          attended[row][element] *= sigmoid(gateRows[element]);
      }

      for (uint32_t row = 0; row < tokens.size(); ++row) {
        const auto attnOut = matmul(outProjection, kHidden, kAttentionWidth, attended[row]);
        const auto attnPost = rmsNorm(attnOut, postAttention);
        std::vector<double> ffnIn(kHidden);
        for (uint32_t column = 0; column < kHidden; ++column) ffnIn[column] = attnPost[column] + hidden[row][column];
        const auto normed = rmsNorm(ffnIn, ffnNorm);
        const auto gateRows = matmul(ffnGate, kIntermediate, kHidden, normed);
        const auto upRows = matmul(ffnUp, kIntermediate, kHidden, normed);
        std::vector<double> gated(kIntermediate);
        for (uint32_t element = 0; element < kIntermediate; ++element)
          gated[element] = silu(clamp(gateRows[element])) * clamp(upRows[element]);
        const auto ffnOut = matmul(ffnDown, kHidden, kIntermediate, gated);
        const auto ffnPost = rmsNorm(ffnOut, postFfw);
        for (uint32_t column = 0; column < kHidden; ++column)
          hidden[row][column] = ffnPost[column] + ffnIn[column];
      }
    }

    for (uint32_t row = 0; row < tokens.size(); ++row)
      for (double value : hidden[row])
        if (!std::isfinite(value))
          throw std::runtime_error("the reference's hidden stream diverged");
    std::vector<std::vector<double>> logits_(tokens.size(), std::vector<double>(kVocab, 0.0));
    for (uint32_t row = 0; row < tokens.size(); ++row) {
      const auto normed = rmsNorm(hidden[row], finalNorm);
      logits_[row] = matmul(output, kVocab, kHidden, normed);
    }
    return logits_;
  }
};

// -------------------------------------------------------------------- misc

MetalBuffer upload(MetalBackend &backend, const std::vector<uint32_t> &values, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(values.size() * 4, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), values.data(), values.size() * 4);
  return buffer;
}

MetalBuffer uploadFloats(MetalBackend &backend, const std::vector<float> &values, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(values.size() * 4, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), values.data(), values.size() * 4);
  return buffer;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: ripple-parity-test METALLIB");
    MetalBackend backend(argv[1]);

    const splash::model::RippleLayout layout = [] {
      splash::model::RippleLayout tiny;
      tiny.nativeContextTokens = 4096;
      tiny.layers = kLayers;
      tiny.hiddenSize = kHidden;
      tiny.vocabularySize = kVocab;
      tiny.packedFullWidth = 2 * kAttentionWidth + 2 * kKvWidth;
      tiny.attentionWidth = kAttentionWidth;
      tiny.attentionQueryHeads = kQueryHeads;
      tiny.attentionKvHeads = kKvHeads;
      tiny.attentionHeadDimension = kHeadDim;
      tiny.rotaryPairs = kPairs;
      tiny.rotaryTheta = kTheta;
      tiny.slidingWindow = kWindow;
      tiny.slidingLayers = {1, 0};
      tiny.unrotatedLayers = {2, 0};
      tiny.fullAttentionPeriod = 1;
      tiny.ffnKind = splash::model::QwenFfnKind::Dense;
      tiny.intermediateSize = kIntermediate;
      tiny.embeddingScale = kEmbedScale;
      tiny.swigluClamp = kClamp;
      tiny.valueResidual = true;
      tiny.postFfnNorm = true;
      tiny.maskToken = 200;
      tiny.stopTokens = {1, 0};
      return tiny;
    }();
    const std::vector<uint32_t> targetLayers{0};
    const splash::model::DFlashDraftLayout draft = splash::model::draftLayout(
        kHidden, kVocab, 1, kIntermediate, kTheta, targetLayers, SPLASH_DRAFT_VARIANTS[0]);
    const splash::model::ModelDescriptor descriptor = splash::model::makeModelDescriptor(
        "test/ripple-parity", layout, draft, {.outputHiddenSize = kHidden},
        splash::model::TargetSource::Gguf, splash::model::VisionSource::None);
    const std::filesystem::path root("dev/tests/fixtures/ripple-tiny");
    const LoadedModel model = splash::model::loadModel(backend, root, descriptor);

    const ExecutionPlans operators(backend.capabilities(), ops::Int8Activations::Off);
    const RuntimeGeometry geometry =
        RuntimeGeometry::from(model, kv::Format::BFloat16, splash::ops::GdnStateFormat::Float32);
    auto target = std::visit(
        [&](const auto &weights) {
          return makeQwenTarget(weights, geometry.target, backend, operators, geometry.prefillChunks);
        },
        model.target);

    // The decode arena's linear scratch, whose sizes the vocabulary head's
    // projection reads (the head is a decode kernel).
    const model::DecodeArena decodeArena(backend, geometry, operators);

    engine::MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1,
                                    engine::queryHostMemory, 0);
    const test::ClaimScope claim(governor, engine::ClaimClass::Floor);
    const kv::Layout kvLayout = model.targetKvLayout(kv::Format::BFloat16);
    const uint32_t extent = kvLayout.minimumExtentPages();
    kv::PageStorage pages(backend, governor.allocationAdmission(), kvLayout, 2 * extent, extent);
    require(static_cast<bool>(pages.allocateExtent(0)), "the parity test's page did not allocate");
    const std::vector<uint32_t> pageIds{0};

    const std::vector<uint32_t> tokens{0, 5, 17, 200};
    const Reference reference{Gguf(root / "target" / "ripple.gguf")};
    const auto expected = reference.logits(tokens);

    MetalBuffer logits;
    const model::PrefillArena prefillArena(backend, geometry, operators);
    const model::QwenTargetPrefillBuffers buffers = model::prefillBuffers(prefillArena, 0);
    {
      CommandGraph graph;
      // The RoPE tables of the chunk's rows, then the scaled embeddings.
      // A row's position is triple {row, row, row} (an image's grid position
      // for a text row), as RoPE::addTables reads.
      const MetalBuffer positions =
          upload(backend, {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3}, "rope positions");
      // A draft table of no rows: the kernel still takes its buffers. The
      // inverse frequencies the arena's tensors held, on the host.
      const MetalBuffer none = backend.allocateBuffer(4, BufferStorage::Shared, "unbound");
      const MetalBuffer inverse = uploadFloats(
          backend, [] {
            std::vector<float> values(kPairs);
            for (uint32_t pair = 0; pair < kPairs; ++pair)
              values[pair] = std::pow(kTheta, -static_cast<float>(pair) / kPairs);
            return values;
          }(), "target inverse frequencies");
      const MetalBuffer ropeCos =
          backend.allocateBuffer(uint64_t{kRows} * kPairs * sizeof(float), BufferStorage::Shared, "rope cosine");
      const MetalBuffer ropeSin =
          backend.allocateBuffer(uint64_t{kRows} * kPairs * sizeof(float), BufferStorage::Shared, "rope sine");
      ops::RoPE::addTables(graph, positions, none, inverse, none, ropeCos,
                           ropeSin, none, none,
                           {.target_rows = kRows, .draft_rows = 0, .target_scale = 1.0F}, kRows, kPairs);
      const MetalBuffer tokenBuffer = upload(backend, tokens, "prefill tokens");
      target->addEmbedding(graph, tokenBuffer, buffers.hidden[0], kRows);

      // One sequence of the chunk's rows: its own page, no spilled history,
      // no GDN states (the family's layers are all attention).
      QwenTargetPrefillSequence sequence;
      sequence.lane = 0;
      sequence.rowBegin = 0;
      sequence.rows = kRows;
      sequence.tokens = tokens;
      sequence.request = QwenTargetRequest{
          tokens, [](uint64_t logical) { return std::array<uint32_t, 3>{uint32_t(logical), uint32_t(logical), uint32_t(logical)}; }};
      sequence.attentionStride = (kRows + kv::kPageTokens - 1) / kv::kPageTokens * kv::kPageTokens;
      sequence.queryOffset = 0;
      sequence.kvOffset = 0;
      sequence.chunk = ops::PagedAttention::prefillParams(0, kRows, sequence.attentionStride, 1);
      MetalBuffer table = backend.allocateBuffer(sizeof(SplashKvPage), BufferStorage::Shared, "page table");
      pages.writeEntries(pageIds, 0, table);
      sequence.pageTable = table;

      QwenTargetPrefillChunk chunk;
      chunk.buffers = buffers;
      chunk.sequences = {&sequence, 1};
      chunk.rows = kRows;

      SpilledKvReads spilled;
      const std::vector<MetalBuffer> finalHidden =
          target->addPrefill(graph, {&chunk, 1}, pages, spilled, {0, kLayers});

      // The head over the last row, as a verify lane's: the final norm and
      // the vocabulary projection of kVerifyRows rows, the last row copied
      // into the first.
      constexpr uint32_t kVerifyRows = SPLASH_TARGET_VERIFY_ROWS;
      MetalBuffer headInput = backend.allocateBuffer(uint64_t{kVerifyRows} * kHidden * 2, BufferStorage::Shared,
                                                     "head input");
      std::memset(headInput.contents(), 0, headInput.sizeBytes());
      MetalBuffer finalHidden0 = backend.allocateBuffer(uint64_t{kVerifyRows} * kHidden * 2, BufferStorage::Shared,
                                                        "final hidden");
      // The head's logits are fp32 (the vocabulary projection's plain
      // epilogue), as the sampling reads them.
      MetalBuffer logitsBuffer = backend.allocateBuffer(uint64_t{kVerifyRows} * kVocab * 4, BufferStorage::Shared,
                                                        "logits");
      // Every row's head, so the comparison holds all four positions.
      for (uint32_t row = 0; row < kRows; ++row)
        ops::RowCopy::add(graph, finalHidden[0], {row, kHidden, 0}, headInput, {row, kHidden, 0}, 1, kHidden);
      target->addHeadBatch(graph, headInput, finalHidden0, logitsBuffer, 1, kVerifyRows,
                           decodeArena.linearScratch());
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    if (const char *ropeDebug = std::getenv("SPLASH_DEBUG_ROPE")) {
      static_cast<void>(ropeDebug);
      const auto *cos = static_cast<const float *>(ropeCos.contents());
      const auto *sin = static_cast<const float *>(ropeSin.contents());
      for (uint32_t row = 0; row < 2; ++row)
        std::fprintf(stderr, "rope row %u: cos[0]=%.6f sin[0]=%.6f cos[1]=%.6f sin[1]=%.6f\n",
                     row, cos[row * kPairs], sin[row * kPairs], cos[row * kPairs + 1], sin[row * kPairs + 1]);
    }
    logits = logitsBuffer;
  }

  // The logits of the last row against the reference's. The engine's
    // path stages bf16 between ops, whose rounding the fp64 reference does
    // not take; the gate holds the pipeline's accumulated rounding, which
    // the report prints.
    const auto *values = static_cast<const float *>(logits.contents());
    struct Delta final {
      double delta, got, want;
      uint32_t row, token;
      bool operator<(const Delta &other) const { return delta > other.delta; }
    };
    std::vector<Delta> deltas;
    for (uint32_t row = 0; row < kRows; ++row) {
      double scale = 0.0;
      for (uint32_t token = 0; token < kVocab; ++token)
        scale = std::max(scale, std::fabs(expected[row][token]));
      // The gate is the pipeline's accumulated bf16 rounding, measured at
      // about half a percent of the row's largest logit.
      const double bound = 0.008 * scale + 0.35;
      for (uint32_t token = 0; token < kVocab; ++token) {
        const double got = values[uint64_t{row} * kVocab + token];
        const double want = expected[row][token];
        if (!std::isfinite(got) || std::fabs(got - want) > bound) {
          std::cerr << "logits[" << row << "][" << token << "]: " << got << " vs " << want << " (scale " << scale
                    << ", bound " << bound << ")\n";
          throw std::runtime_error("the engine's prefill diverged from the reference");
        }
        deltas.push_back({std::fabs(got - want), got, want, row, token});
      }
    }
    std::sort(deltas.begin(), deltas.end());
    const Delta &worst = deltas.front();
    std::cout << "ripple-parity-test: PASS (worst delta " << worst.delta << " at token " << worst.token
              << " of row " << worst.row << ": " << worst.got << " vs " << worst.want << ")\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple-parity-test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
