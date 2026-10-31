// A prototype of DeepSeek-V4.1-Flash's forward pass on its kernels
// (model/DeepSeek41.hpp, ops/LatentAttention.hpp and the others) and its real
// weights, before its target joins the runtime: one sequence's prompt, all its
// rows at once, then a decode step of one row per generated token, each
// layer's latents kept for every position the sequence holds. A decode step's
// projections and MoE run the decode plans of one lane (its row the lane's
// one verify row) unless DEEPSEEK41_DECODE_KERNELS=0, which runs them as
// prefill chunks of one row. DEEPSEEK41_KV_ROUNDING=none keeps the latents,
// index queries and keys and the attention output unrounded, as the
// reference's --kv-rounding none does. DEEPSEEK41_PROFILE=1 times the tenth
// decode step's dispatches one by one (metal::BackendInstrumentation) and
// prints their GPU time by kernel.
//
//   deepseek41-forward METALLIB GGUF_DIRECTORY REFERENCE_DIRECTORY EXPERT_GIB [LAYERS [GENERATE [ENGRAM]]]
//
// GGUF_DIRECTORY holds the target's parts alone, the Engram's tables its last
// (llama.cpp's layout); its routed experts stream through a store of
// EXPERT_GIB GiB (at least 4.9 for IQ1_M: a layer's experts in slots of the
// widest expert's size, and the staging). With the streams of
// dev/tools/deepseek41_reference.py --dump-streams in REFERENCE_DIRECTORY it
// compares the streams after each sublayer and the prompt's logits with the
// reference's, an Engram layer reading the reference's gathered rows
// (engram_values_<layer>.npy), which the rows staged from the target's
// tables, with ENGRAM 1, are checked against. Without them it takes the
// prompt's token ids from REFERENCE_DIRECTORY/../prompt.txt, skips the Engram
// unless ENGRAM is 1, and prints each sublayer's mean stream norm (which the
// reference logs after each layer). Then it prints the next token's top 5
// and, with GENERATE, adds that many tokens greedily, a decode step of one
// row each, with their Engram rows from the tables with ENGRAM 1. Those are
// staged by a model::SparseRowStore per Engram layer and gathered on the GPU
// (ops::Embedding over the staging), as the target reads them.
//
// With DEEPSEEK41_DSPARK=<dspark GGUF> (model/DSpark.hpp, its experts loaded
// whole beside the target's store) it generates speculatively instead: after
// the prompt, each cycle DSpark drafts a block from the last token, and one
// verify step of the target's decode plans takes it and the drafts (a lane's
// rows), keeps the drafts up to the first its argmax rejects and the token it
// predicts there; DSpark's rings take the latents of the kept rows' hidden
// views (the target layers' input streams averaged). It prints the drafts
// kept per cycle and the time of the drafts and of the verify steps, and the
// tokens, which equal the greedy decode's up to the numerics of a step of
// more rows. With DEEPSEEK41_DSPARK_THRESHOLD=<score> a cycle verifies only
// the drafts up to the first whose confidence is below it, and the anchor
// alone where the first is. It prints every cycle's drafts' confidences, the
// drafts it verified and kept, and the verify steps' median time by rows.
#include "metal/BackendInstrumentation.hpp"
#include "metal/MetalBackend.hpp"
#include "model/DSpark.hpp"
#include "model/DeepSeek41.hpp"
#include "model/EngramHash.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufTarget.hpp"
#include "model/SparseRowStore.hpp"
#include "model/SsdReaders.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightSource.hpp"
#include "ops/Embedding.hpp"
#include "ops/Engram.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/LatentAttention.hpp"
#include "ops/LatentIndex.hpp"
#include "ops/LatentRotation.hpp"
#include "ops/Linear.hpp"
#include "ops/ManifoldHyperConnection.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace splash;
using metal::MetalBuffer;

namespace {

// A .npy array of little-endian float32 or uint32 values, mapped (a long
// prompt's streams take gigabytes).
struct Npy {
  std::vector<uint64_t> shape;
  std::string descr;
  std::shared_ptr<const char> mapping;
  const char *data = nullptr;
  [[nodiscard]] uint64_t count() const {
    uint64_t n = 1;
    for (uint64_t d : shape) n *= d;
    return n;
  }
  template <class T> [[nodiscard]] const T *values() const { return reinterpret_cast<const T *>(data); }
};

Npy readNpy(const std::string &path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat info {};
  ::fstat(fd, &info);
  const size_t size = size_t(info.st_size);
  void *mapped = size >= 10 ? ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
  ::close(fd);
  if (mapped == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  Npy result;
  result.mapping = std::shared_ptr<const char>(static_cast<const char *>(mapped),
                                               [size](const char *p) { ::munmap(const_cast<char *>(p), size); });
  const char *bytes = result.mapping.get();
  if (std::memcmp(bytes, "\x93NUMPY", 6) != 0) throw std::runtime_error("not npy: " + path);
  const uint8_t major = uint8_t(bytes[6]);
  uint32_t headerLength = 0;
  size_t start = 0;
  if (major == 1) {
    headerLength = uint8_t(bytes[8]) | uint32_t(uint8_t(bytes[9])) << 8;
    start = 10;
  } else {
    std::memcpy(&headerLength, bytes + 8, 4);
    start = 12;
  }
  const std::string header(bytes + start, headerLength);
  const auto descr = header.find("'descr': '");
  result.descr = header.substr(descr + 10, header.find('\'', descr + 10) - descr - 10);
  if (header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error("fortran order: " + path);
  const auto open = header.find('(', header.find("'shape'")), close = header.find(')', open);
  std::stringstream dims(header.substr(open + 1, close - open - 1));
  for (std::string dim; std::getline(dims, dim, ',');)
    if (dim.find_first_not_of(' ') != std::string::npos) result.shape.push_back(std::stoull(dim));
  result.data = bytes + start + headerLength;
  return result;
}

uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return uint16_t((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
float f32(uint16_t bits) {
  const uint32_t widened = uint32_t(bits) << 16;
  float out;
  std::memcpy(&out, &widened, 4);
  return out;
}

// The rows a buffer of `rows` rows holds: the whole tiles a prefill
// projection reads, the staged tile's 32 rows up to 32
// (ops::kMaximumDecodeTileRows) and 128-row tiles past them
// (Linear::ggufBaseline).
uint32_t pad(uint32_t rows) { return rows <= 32 ? 32 : (rows + 127) / 128 * 128; }

// RoPE's per-pair frequencies: a compressed layer's, base 160000 stretched by
// YaRN, else base 10000 (the reference's rope_angles).
std::vector<float> ropeFrequencies(const model::DeepSeek41Layout &layout, bool compressed) {
  const model::DeepSeek41Dimensions &d = layout.deepseek;
  const uint32_t dim = 2 * layout.rotaryPairs;
  const float base = compressed ? d.compressRopeTheta : layout.rotaryTheta;
  std::vector<float> freqs(dim / 2);
  for (uint32_t i = 0; i < dim / 2; ++i) freqs[i] = 1.0f / std::pow(base, float(2 * i) / float(dim));
  if (!compressed) return freqs;
  const double original = d.yarnOriginalContextTokens, factor = d.yarnFactor;
  const auto corrected = [&](double rotations) {
    return dim * std::log(original / (rotations * 2 * M_PI)) / (2 * std::log(double(base)));
  };
  const double low = std::max(std::floor(corrected(d.yarnBetaFast)), 0.0);
  const double high = std::min(std::ceil(corrected(d.yarnBetaSlow)), double(dim - 1));
  for (uint32_t i = 0; i < dim / 2; ++i) {
    const float ramp = float(std::clamp((double(i) - low) / std::max(high - low, 1e-3), 0.0, 1.0));
    const float smooth = 1.0f - ramp;
    freqs[i] = freqs[i] / float(factor) * (1.0f - smooth) + freqs[i] * smooth;
  }
  return freqs;
}

// One sequence's forward pass, a step of new rows at a time (its prompt,
// then each generated token), its state that of the rows before: every
// layer's latent of each position and every compressor layer's projections
// of each row, of which a step computes the complete groups again. A
// sequence holds at most `capacity` tokens; a window reads its pool's
// positions directly.
class Forward {
public:
  Forward(metal::MetalBackend &backend, const model::DeepSeek41Weights &weights, uint32_t capacity)
      : backend_(backend), weights_(weights), layout_(weights.layout), d_(layout_.deepseek),
        operators_(backend.capabilities()), capacity_(capacity) {
    const uint32_t hidden = layout_.hiddenSize, latent = layout_.attentionHeadDimension;
    mhc_ = {hidden, d_.sinkhornIterations, d_.hcEpsilon, d_.normEpsilon};
    freqs_[0] = upload(ropeFrequencies(layout_, false));
    freqs_[1] = upload(ropeFrequencies(layout_, true));
    // Every row's window latent and compressor projections, and room for a
    // step's padded rows past them.
    const uint64_t rows = pool();
    for (const model::DeepSeek41LayerWeights &layer : weights.layers) {
      windows_.push_back(alloc(rows * latent * 2));
      compressorKv_.push_back(layer.compressor ? alloc(rows * latent * 4) : MetalBuffer{});
      compressorGate_.push_back(layer.compressor && layer.compressor->gate ? alloc(rows * latent * 4)
                                                                           : MetalBuffer{});
    }
    // Every projection's scratch, of its prefill chunks and of its decode
    // plans.
    ops::LinearScratchSize size{}, decodeSize{};
    const auto include = [&](const ops::Projection &p) {
      size.include(operators_.linear().prefillScratchSize(p.shape()));
      decodeSize.include(operators_.linear().decodeScratchSize(p.shape()));
    };
    for (const model::DeepSeek41LayerWeights &layer : weights.layers) {
      for (const ops::Projection *p : {&layer.queryA, &layer.queryB, &layer.kv, &layer.outputB}) include(*p);
      for (const ops::Projection &p : layer.outputA) include(p);
      if (layer.engram) include(layer.engram->kv);
    }
    include(weights.logitsProjection);
    linearScratch_ = {zeroed(size.input), zeroed(size.sums), zeroed(size.partials), zeroed(size.counters),
                      zeroed(size.rotated)};
    decodeScratch_ = {zeroed(decodeSize.input), zeroed(decodeSize.sums), zeroed(decodeSize.partials),
                      zeroed(decodeSize.counters), zeroed(decodeSize.rotated)};
    verifyRows_ = upload(std::vector<uint32_t>{1});
    const char *kernels = std::getenv("DEEPSEEK41_DECODE_KERNELS");
    decodeKernels_ = !kernels || std::string_view(kernels) != "0";
    const char *rounding = std::getenv("DEEPSEEK41_KV_ROUNDING");
    rounds_ = !rounding || std::string_view(rounding) != "none";
    // The MoE's shape, its formats those of most of its blocks' experts.
    moeShape_ = layout_.moeBlockShape();
    moeShape_.routeScale = d_.routeScale;
    moeShape_.swigluLimit = d_.swigluLimit;
    std::map<uint32_t, uint64_t> formats;
    for (const model::DeepSeek41LayerWeights &layer : weights.layers)
      for (const ops::BlockExpertProjection *p : {&layer.ffn.gate, &layer.ffn.up, &layer.ffn.down})
        if (!p->routed.isFloat()) formats[p->routed.formatId] += uint64_t{p->routed.outputSize} * p->routed.inputSize;
    moeShape_.expertFormat =
        std::max_element(formats.begin(), formats.end(), [](auto &a, auto &b) { return a.second < b.second; })->first;
    moeShape_.registerFormats = false;
    moeShape_.routerFormat = weights.layers.front().ffn.router.formatId;
    if (!moeShape_.valid()) throw std::runtime_error("invalid MoE shape");
  }

  // Starts a step of `tokens` at the sequence's next positions.
  void begin(const std::vector<uint32_t> &tokens) {
    rows_ = uint32_t(tokens.size());
    // A step of a lane's rows at most runs the decode plans of one lane, as a
    // verify step does.
    const bool lane = rows_ <= model::ExecutionLimits::targetVerifyRows;
    decode_ = decodeKernels_ && lane;
    // The last step's commands ran: its buffers serve this one.
    for (auto &[bytes, buffer] : taken_) free_.emplace(bytes, std::move(buffer));
    taken_.clear();
    pooled_ = lane;
    verifyRows_ = upload(std::vector<uint32_t>{rows_});
    if (!rows_ || past_ + rows_ > capacity_)
      throw std::runtime_error("the sequence holds " + std::to_string(capacity_) + " tokens");
    std::vector<uint32_t> positions(rows_);
    for (uint32_t r = 0; r < rows_; ++r) positions[r] = past_ + r;
    positions_ = upload(positions);
    tokenBuffer_ = upload(tokens);
    streams_ = alloc(uint64_t{pad(rows_)} * MHC_STREAMS * layout_.hiddenSize * 4);
    // The first attention input reads stream 0 alone.
    std::vector<float> first(size_t(rows_) * MHC_MIXES, 0.0f);
    for (uint32_t r = 0; r < rows_; ++r) first[size_t(r) * MHC_MIXES] = 1.0f;
    previousMixes_ = upload(first);
    chunks_ = alloc(ops::ManifoldHyperConnection::mixScratchBytes(mhc_, rows_));
    routes_.clear();
    if (captureLayers_) mainHidden_ = alloc(uint64_t{pad(rows_)} * captureLayers_ * layout_.hiddenSize * 2);
  }
  // Ends the step: its rows are the sequence's, or the first `kept` of them.
  void end() { past_ += rows_; }
  void end(uint32_t kept) {
    if (kept > rows_) throw std::runtime_error("a step keeps more rows than it ran");
    past_ += kept;
  }
  // Each step's hidden views of `layers` layers (capture), for DSpark.
  void captures(uint32_t layers) { captureLayers_ = layers; }
  // The streams at the input of the layer about to run, averaged into
  // capture slot `slot` of the step's hidden views.
  void capture(uint32_t slot) {
    const uint32_t hidden = layout_.hiddenSize;
    ops::ManifoldHyperConnection::addStreamMean(*pending_, streams_, mainHidden_, captureLayers_ * hidden,
                                                slot * hidden, mhc_, rows_);
  }
  // The step's hidden views, bf16 [rows][capture layers * hidden], after the
  // work added so far runs.
  [[nodiscard]] const MetalBuffer &mainHidden() {
    flush();
    return mainHidden_;
  }
  [[nodiscard]] uint32_t past() const noexcept { return past_; }

  // Embeds the step's tokens into the streams, four copies of each row.
  void embed() {
    const uint32_t hidden = layout_.hiddenSize;
    MetalBuffer embedding = alloc(uint64_t{pad(rows_)} * hidden * 2);
    metal::CommandGraph &graph = *pending_;
    ops::Embedding::add(graph, tokenBuffer_, weights_.tokenEmbedding, embedding, rows_);
    ops::ManifoldHyperConnection::addInit(graph, embedding, streams_, mhc_, rows_);
  }

  // The step's Engram rows of one layer, bf16 [rows][engram width]: the
  // reference's gathered values.
  MetalBuffer engramRows(const std::vector<float> &values) {
    const uint32_t width = d_.engramWidth();
    if (values.size() != uint64_t{rows_} * width) throw std::runtime_error("Engram rows of another shape");
    std::vector<uint16_t> rows(size_t(pad(rows_)) * width, 0);
    for (size_t i = 0; i < values.size(); ++i) rows[i] = bf16(values[i]);
    return upload(rows);
  }
  // Or the rows of `ids` (each row's ids of the layer, in its gathered
  // order) staged by `store` and gathered on the GPU, the staging read as a
  // table of as many rows as there are ids.
  MetalBuffer engramRows(model::SparseRowStore &store, const std::vector<uint32_t> &ids) {
    const uint32_t count = uint32_t(ids.size()), rowBytes = store.table().rowBytes;
    if (count != uint64_t{rows_} * (d_.engramNgram - 1) * d_.engramHeads)
      throw std::runtime_error("Engram ids of another shape");
    MetalBuffer staging = alloc(uint64_t{count} * rowBytes), slots = alloc(uint64_t{count} * 4);
    store.stage(ids, {static_cast<uint8_t *>(staging.contents()), uint64_t{count} * rowBytes},
                {static_cast<uint32_t *>(slots.contents()), count},
                rows_ > 1 ? model::SparseRowUse::Prefill : model::SparseRowUse::Decode);
    MetalBuffer rows = zeroed(uint64_t{pad(rows_)} * d_.engramWidth() * 2);
    ops::Embedding::add(*pending_, slots,
                        ops::EmbeddingWeights{count, d_.engramHeadDimension, ops::NativeRows(staging, GGUF_FMT_IQ4NL)},
                        rows, count);
    return rows;
  }
  // The first `count` bf16 values of `buffer`, after the work added so far.
  std::vector<uint16_t> bf16Values(const MetalBuffer &buffer, uint64_t count) {
    flush();
    const auto *values = static_cast<const uint16_t *>(buffer.contents());
    return {values, values + count};
  }

  // Layer `index`'s Engram on its gathered rows (engramRows).
  void engram(uint32_t index, const MetalBuffer &input) {
    const model::DeepSeek41LayerWeights &layer = weights_.layers[index];
    const uint32_t hidden = layout_.hiddenSize;
    MetalBuffer kv = alloc(uint64_t{pad(rows_)} * (MHC_STREAMS + 1) * hidden * 2);
    metal::CommandGraph &graph = *pending_;
    project(graph, input, layer.engram->kv, kv, rows_);
    ops::Engram::addGate(graph, streams_, kv, layer.engram->gate, hidden, d_.normEpsilon, rows_);

  }

  void attention(uint32_t index) {
    const model::DeepSeek41LayerWeights &layer = weights_.layers[index];
    const auto has = [&](uint64_t layers) { return model::DeepSeek41Dimensions::has(layers, index); };
    const bool compressed = has(d_.compressedLayers);
    const uint32_t ratio = !compressed ? 0 : has(d_.pairedLayers) ? 2 : 1;
    const uint32_t hidden = layout_.hiddenSize, latent = layout_.attentionHeadDimension;
    const uint32_t heads = layout_.attentionQueryHeads, pairs = layout_.rotaryPairs, R = rows_;
    const MetalBuffer &freqs = freqs_[compressed];
    MetalBuffer mixes = alloc(uint64_t{pad(R)} * MHC_MIXES * 4), x = alloc(uint64_t{pad(R)} * hidden * 2);
    MetalBuffer qa = alloc(uint64_t{pad(R)} * d_.queryRank * 2), qr = alloc(uint64_t{pad(R)} * d_.queryRank * 2);
    MetalBuffer q = alloc(uint64_t{pad(R)} * layout_.attentionWidth * 2), qRotated = alloc(uint64_t{pad(R)} * layout_.attentionWidth * 2);
    MetalBuffer kv = alloc(uint64_t{pad(R)} * latent * 2), window = windows_[index];
    const MetalBuffer windowRows = backend_.view(window, uint64_t{past_} * latent * 2, uint64_t{R} * latent * 2);
    metal::CommandGraph &graph = *pending_;
    ops::ManifoldHyperConnection::addMix(graph, streams_, layer.attentionMix.fn, layer.attentionMix.scale,
                                         layer.attentionMix.base, chunks_, mixes, mhc_, R);
    ops::ManifoldHyperConnection::addPreNorm(graph, streams_, previousMixes_, layer.attentionNorm.buffer, x, mhc_, R);
    project(graph, x, layer.queryA, qa, R);
    ops::LatentRotation::add(graph, qa, false, layer.queryNorm.buffer, freqs, positions_, qr,
                             {1, d_.queryRank, 0, false, ops::LatentRounding::None, true, d_.normEpsilon}, R);
    project(graph, qr, layer.queryB, q, R);
    ops::LatentRotation::add(graph, q, false, layer.queryNorm.buffer, freqs, positions_, qRotated,
                             {heads, latent, pairs, false, ops::LatentRounding::None, false, 0.0f}, R);
    project(graph, x, layer.kv, kv, R);
    ops::LatentRotation::add(graph, kv, false, layer.kvNorm.buffer, freqs, positions_, windowRows,
                             {1, latent, pairs, false, round(ops::LatentRounding::Fp8), true, d_.normEpsilon}, R);
    // A compressor layer's groups, every complete one of the sequence: their
    // latents, rotated at their first row's position and rounded to FP4, and
    // their index keys, from the projections of every row.
    if (layer.compressor) {
      const uint32_t groups = (past_ + R) / ratio;
      MetalBuffer ckv = compressorKv_[index], cgate = compressorGate_[index];
      floatProjection(graph, x, layer.compressor->kv,
                      backend_.view(ckv, uint64_t{past_} * latent * 4, uint64_t{R} * latent * 4), R);
      if (ratio > 1)
        floatProjection(graph, x, *layer.compressor->gate,
                        backend_.view(cgate, uint64_t{past_} * latent * 4, uint64_t{R} * latent * 4), R);
      MetalBuffer pooled = alloc(uint64_t{pad(groups)} * latent * 4);
      MetalBuffer normalized = alloc(uint64_t{pad(groups)} * latent * 2);
      MetalBuffer keys = alloc(uint64_t{pad(groups)} * d_.indexerHeadDimension * 4);
      compressed_ = alloc(uint64_t{pad(groups)} * latent * 2);
      indexKeys_ = alloc(uint64_t{pad(groups)} * d_.indexerHeadDimension * 2);
      std::vector<uint32_t> groupPositions(std::max(groups, 1u));
      for (uint32_t g = 0; g < groups; ++g) groupPositions[g] = g * ratio;
      MetalBuffer groupPositionBuffer = upload(groupPositions);
      groups_ = groups;
      if (groups) {
        if (ratio > 1)
          ops::LatentIndex::addCompress(graph, ckv, cgate, pooled, groups, latent, ratio, true);
        else
          pooled = ckv;
        // The normalized latent, unrotated, which the index key projects.
        ops::LatentRotation::add(graph, pooled, true, layer.compressor->norm.buffer, freqs, groupPositionBuffer,
                                 normalized, {1, latent, 0, false, ops::LatentRounding::None, true, d_.normEpsilon},
                                 groups);
        ops::LatentRotation::add(graph, pooled, true, layer.compressor->norm.buffer, freqs, groupPositionBuffer,
                                 compressed_,
                                 {1, latent, pairs, false, round(ops::LatentRounding::Fp4E4m3Scale), true,
                                  d_.normEpsilon},
                                 groups);
        floatProjection(graph, normalized, layer.compressor->indexKey, keys, groups);
        ops::LatentRotation::add(graph, keys, true, layer.compressor->indexKeyNorm.buffer, freqs, groupPositionBuffer,
                                 indexKeys_,
                                 {1, d_.indexerHeadDimension, pairs, false, round(ops::LatentRounding::Fp4Pow2Scale),
                                  true, d_.normEpsilon},
                                 groups);
      }
    }
    // An indexer layer's selection of each row's groups.
    const uint32_t topK = d_.indexerTopGroups;
    // An indexer layer selects each row's groups for it and the layers up to
    // the next; before the first complete group, none.
    if (layer.indexer && !groups_) selectedStride_ = 0;
    if (layer.indexer && groups_) {
      const uint32_t indexHeads = d_.indexerHeads, indexDim = d_.indexerHeadDimension, stride = groups_;
      MetalBuffer iq = alloc(uint64_t{pad(R)} * indexHeads * indexDim * 4), iqRotated = alloc(uint64_t{pad(R)} * indexHeads * indexDim * 2);
      MetalBuffer iw = alloc(uint64_t{pad(R)} * indexHeads * 4), scores = alloc(uint64_t{pad(R)} * stride * 4);
      selected_ = alloc(uint64_t{pad(R)} * topK * 4);
      selectedStride_ = topK;
      floatProjection(graph, qr, layer.indexer->query, iq, R);
      ops::LatentRotation::add(graph, iq, true, layer.queryNorm.buffer, freqs, positions_, iqRotated,
                               {indexHeads, indexDim, pairs, false, round(ops::LatentRounding::Fp4Pow2Scale), false,
                                0.0f},
                               R);
      // The heads' weights, 32 columns, are no whole 64-row tile: the float
      // projection alone, in 8-column tiles, as a router's.
      ops::addGgufFloat(graph, x, layer.indexer->weights.blocks().segments.front(), iw, R, indexHeads, 0,
                        ops::FloatOutput::Float32, ops::FloatTile::Simdgroup);
      ops::LatentIndex::addScores(graph, iqRotated, iw, indexKeys_, positions_, scores, indexHeads, groups_, stride,
                                  ratio, 1.0f / std::sqrt(float(indexDim)) / std::sqrt(float(indexHeads)), R);
      const uint32_t block = d_.candidateBlockGroups, blocks = (groups_ + block - 1) / block;
      if (index == d_.candidateLayer) {
        MetalBuffer blockScores = alloc(uint64_t{pad(R)} * blocks * 4);
        keptStride_ = d_.candidateTopBlocks;
        kept_ = alloc(uint64_t{pad(R)} * keptStride_ * 4);
        ops::LatentIndex::addBlocks(graph, scores, positions_, blockScores, groups_, stride, block, ratio, R);
        ops::LatentIndex::addSelect(graph, blockScores, kept_, blocks, d_.candidateTopBlocks, blocks, keptStride_, R);
      } else if (index > d_.candidateLayer && kept_) {
        ops::LatentIndex::addMask(graph, scores, kept_, groups_, stride, block, keptStride_, R);
      }
      ops::LatentIndex::addSelect(graph, scores, selected_, groups_, topK, stride, topK, R);
    }
    // Each row's keys: its window (the pool holds position p at row p), then
    // its selected groups, listed on the GPU.
    const uint32_t keyStride = layout_.slidingWindow + topK;
    MetalBuffer indexBuffer = alloc(uint64_t{pad(R)} * keyStride * 4), countBuffer = alloc(uint64_t{pad(R)} * 4);
    ops::LatentAttention::addKeys(graph, positions_, selected_, indexBuffer, countBuffer, layout_.slidingWindow,
                                  pool(), compressed ? selectedStride_ : 0, keyStride, R);
    MetalBuffer scores = alloc(uint64_t{pad(R)} * heads * keyStride * 4), out = alloc(uint64_t{pad(R)} * heads * latent * 4);
    const uint64_t partBytes = ops::LatentAttention::partBytes(heads, R);
    MetalBuffer parts = partBytes ? alloc(partBytes) : MetalBuffer{};
    MetalBuffer rotatedBack = alloc(uint64_t{pad(R)} * layout_.attentionWidth * 2);
    const uint32_t groupWidth = d_.groupWidth(layout_.attentionWidth);
    MetalBuffer groupInput = alloc(uint64_t{pad(R)} * groupWidth * 2), groupOutput = alloc(uint64_t{pad(R)} * d_.outputRank * 2);
    MetalBuffer joined = alloc(uint64_t{pad(R)} * d_.outputGroups * d_.outputRank * 2), branch = alloc(uint64_t{pad(R)} * hidden * 2);
    MetalBuffer compressedPool = compressed_ ? compressed_ : window;
    metal::CommandGraph &second = *pending_;
    ops::LatentAttention::add(second, qRotated, window, compressedPool, indexBuffer, countBuffer, layer.sinks, scores,
                              parts, out, heads, keyStride, 1.0f / std::sqrt(float(latent)), R);
    ops::LatentRotation::add(second, out, true, layer.queryNorm.buffer, freqs, positions_, rotatedBack,
                             {heads, latent, pairs, true, round(ops::LatentRounding::Fp8), false, 0.0f}, R);
    for (uint32_t g = 0; g < d_.outputGroups; ++g) {
      ops::RowCopy::add(second, rotatedBack, {0, layout_.attentionWidth, g * groupWidth}, groupInput,
                        {0, groupWidth, 0}, R, groupWidth);
      project(second, groupInput, layer.outputA[g], groupOutput, R);
      ops::RowCopy::add(second, groupOutput, {0, d_.outputRank, 0}, joined,
                        {0, d_.outputGroups * d_.outputRank, g * d_.outputRank}, R, d_.outputRank);
    }
    project(second, joined, layer.outputB, branch, R);
    ops::ManifoldHyperConnection::addPost(second, streams_, branch, mixes, mhc_, R);

    attentionMixes_ = mixes;
  }

  void ffn(uint32_t index) {
    const model::DeepSeek41LayerWeights &layer = weights_.layers[index];
    const uint32_t hidden = layout_.hiddenSize, R = rows_;
    MetalBuffer mixes = alloc(uint64_t{pad(R)} * MHC_MIXES * 4), x = alloc(uint64_t{pad(R)} * hidden * 2);
    MetalBuffer residual = zeroed(uint64_t{pad(R)} * hidden * 2), out = alloc(uint64_t{pad(R)} * hidden * 2);
    const ops::MoePlan plan = decode_ ? operators_.moeDecode(moeShape_, 1) : operators_.moePrefill(moeShape_, R);
    ops::MoeScratch scratch;
    for (const ops::MoeScratchField &field : ops::kMoeScratchFields)
      scratch.*field.buffer = zeroed(plan.workspace().*field.bytes);
    metal::CommandGraph &graph = *pending_;
    ops::ManifoldHyperConnection::addMix(graph, streams_, layer.ffnMix.fn, layer.ffnMix.scale, layer.ffnMix.base,
                                         chunks_, mixes, mhc_, R);
    ops::ManifoldHyperConnection::addPreNorm(graph, streams_, attentionMixes_, layer.ffnNorm.buffer, x, mhc_, R);
    ops::MoE::add(graph, {x, residual, out, scratch, decode_ ? verifyRows_ : MetalBuffer{}}, layer.ffn, plan);
    routes_.push_back(scratch.selectedExperts);
    ops::ManifoldHyperConnection::addPost(graph, streams_, out, mixes, mhc_, R);

    previousMixes_ = mixes;
  }

  // The prompt's logits [rows][vocabulary], from bf16 (a prefill projection
  // writes bf16 alone).
  std::vector<float> logits() {
    const uint32_t hidden = layout_.hiddenSize, R = rows_;
    MetalBuffer x = alloc(uint64_t{pad(R)} * hidden * 2), out = alloc(uint64_t{pad(R)} * layout_.vocabularySize * 2);
    ops::Projection head = weights_.logitsProjection;
    head.destination = ops::FloatOutput::BFloat16;
    metal::CommandGraph &graph = *pending_;
    ops::ManifoldHyperConnection::addPreNorm(graph, streams_, previousMixes_, weights_.finalNorm.buffer, x, mhc_, R);
    project(graph, x, head, out, R);
    flush();
    const auto *values = static_cast<const uint16_t *>(out.contents());
    std::vector<float> result(size_t(R) * layout_.vocabularySize);
    for (size_t i = 0; i < result.size(); ++i) result[i] = f32(values[i]);
    return result;
  }

  // The streams after the work added so far, which runs first.
  // The routed experts of row `row` in each layer of the step so far, in
  // their routing order (after the work added so far runs).
  [[nodiscard]] std::vector<std::vector<uint32_t>> routes(uint32_t row) {
    flush();
    std::vector<std::vector<uint32_t>> result;
    const uint32_t width = layout_.expertsPerToken + layout_.sharedExpertSlices;
    for (const MetalBuffer &buffer : routes_) {
      const auto *ids = static_cast<const uint32_t *>(buffer.contents()) + size_t(row) * width;
      result.emplace_back(ids, ids + layout_.expertsPerToken);
    }
    return result;
  }

  [[nodiscard]] const float *streams() {
    flush();
    return static_cast<const float *>(streams_.contents());
  }

  // The commands' times since the last call (metal::CommandTiming, summed):
  // the GPU's, of it the waits on host steps (the expert loads), and the
  // wall time of their submissions.
  metal::CommandTiming takeTiming() { return std::exchange(timing_, {}); }

private:
  // The rows of a layer's latent pool: the sequence's positions and a step's
  // padded rows past them.
  [[nodiscard]] uint32_t pool() const noexcept { return capacity_ + 128; }

  // A rounding of the model's, or none without them.
  [[nodiscard]] ops::LatentRounding round(ops::LatentRounding rounding) const noexcept {
    return rounds_ ? rounding : ops::LatentRounding::None;
  }

  // A projection of the step's rows: a decode plan of one lane in a decode
  // step, else prefill chunks.
  void project(metal::CommandGraph &graph, const MetalBuffer &input, const ops::Projection &projection,
               const MetalBuffer &output, uint32_t rows) {
    const ops::Linear &linear = operators_.linear();
    if (decode_)
      static_cast<void>(linear.add(graph, {.input = input, .output = output, .scratch = decodeScratch_}, projection,
                                   linear.decodePlan(projection, 1)));
    else
      linear.addPrefill(graph, input, projection, output, rows, linearScratch_);
  }

  // A float (BF16) projection into fp32 rows: a prefill projection writes
  // bf16 alone, so the float projection runs as a router's.
  void floatProjection(metal::CommandGraph &graph, const MetalBuffer &input, const ops::Projection &projection,
                       const MetalBuffer &output, uint32_t rows) {
    ops::addGgufFloat(graph, input, projection.blocks().segments.front(), output, rows, projection.outputSize, 0,
                      ops::FloatOutput::Float32, ops::FloatTile::Simdgroup);
  }

  template <class T> MetalBuffer upload(const std::vector<T> &values) {
    MetalBuffer buffer = alloc(values.size() * sizeof(T));
    std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
    return buffer;
  }
  // A buffer of the forward pass, or of a decode step: those of a decode step
  // are reused by the next ones (the sizes repeat step by step), as a
  // target's scratch would be, so a step allocates nothing once warm. A
  // prompt's step allocates as it goes, and its buffers go with its commands.
  MetalBuffer alloc(uint64_t bytes) {
    bytes = std::max<uint64_t>(bytes, 16);
    if (!pooled_) return backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, "deepseek41-forward");
    MetalBuffer buffer;
    if (const auto found = free_.find(bytes); found != free_.end()) {
      buffer = std::move(found->second);
      free_.erase(found);
    } else {
      buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, "deepseek41-forward");
    }
    taken_.emplace_back(bytes, buffer);
    return buffer;
  }
  MetalBuffer zeroed(uint64_t bytes) {
    MetalBuffer buffer = alloc(bytes);
    std::memset(buffer.contents(), 0, std::max<uint64_t>(bytes, 16));
    return buffer;
  }
  void run(metal::CommandGraph &graph) {
    const metal::CommandTiming timing = backend_.submitCommandAsync(graph.command()).wait();
    timing_.gpuSeconds += timing.gpuSeconds;
    timing_.wallSeconds += timing.wallSeconds;
    timing_.hostSeconds += timing.hostSeconds;
  }
  // Runs the work added since the last flush as one command: a step's
  // sublayers run together up to where the host reads something back.
  void flush() {
    if (!pending_->empty()) run(*pending_);
    pending_ = std::make_unique<metal::CommandGraph>();
  }

  metal::MetalBackend &backend_;
  const model::DeepSeek41Weights &weights_;
  const model::DeepSeek41Layout &layout_;
  const model::DeepSeek41Dimensions &d_;
  ops::ExecutionPlans operators_;
  uint32_t capacity_ = 0, rows_ = 0, past_ = 0, captureLayers_ = 0;
  MetalBuffer mainHidden_;
  metal::CommandTiming timing_{};
  // Decode steps' buffers: those the current step took, and the free ones by
  // size (alloc).
  bool pooled_ = false;
  std::vector<std::pair<uint64_t, MetalBuffer>> taken_;
  std::multimap<uint64_t, MetalBuffer> free_;
  std::unique_ptr<metal::CommandGraph> pending_ = std::make_unique<metal::CommandGraph>();
  // Each layer's window latents and each compressor layer's projections, a
  // row per position.
  std::vector<MetalBuffer> windows_, compressorKv_, compressorGate_;
  // The MoE's selected experts of each layer of the step.
  std::vector<MetalBuffer> routes_;
  ops::ManifoldHyperShape mhc_{};
  ops::LinearScratch linearScratch_{}, decodeScratch_{};
  // Whether decode steps run the decode plans, and whether this step is one;
  // whether the latents round as the model's do.
  bool decodeKernels_ = true, decode_ = false, rounds_ = true;
  MetalBuffer verifyRows_;
  ops::MoeShape moeShape_{};
  MetalBuffer positions_, tokenBuffer_, freqs_[2], streams_, previousMixes_, attentionMixes_, chunks_;
  // The last compressor's pool and index keys, its groups, the last
  // indexer's selection and the candidate layer's kept blocks.
  MetalBuffer compressed_, indexKeys_, kept_;
  uint32_t groups_ = 0, keptStride_ = 0;
  // The last indexer layer's selection, selectedStride_ groups per row (0:
  // none yet).
  MetalBuffer selected_;
  uint32_t selectedStride_ = 0;
};

// DSpark (model/DSpark.hpp) beside the forward pass: each of its layers' ring
// of the latents of the target rows' hidden views at the last 128 positions,
// and a block's draft from the target's last token, on the decode plans of
// one lane; the Markov head's chain and the confidence on the host.
class Drafter {
public:
  struct Draft final {
    // The drafted tokens after the anchor, and each one's confidence.
    std::vector<uint32_t> tokens;
    std::vector<float> confidence;
  };

  Drafter(metal::MetalBackend &backend, const model::DeepSeek41Weights &target, const model::DSparkWeights &weights)
      : backend_(backend), target_(target), weights_(weights), layout_(weights.layout.layers),
        d_(layout_.deepseek), operators_(backend.capabilities()) {
    const uint32_t hidden = layout_.hiddenSize, latent = layout_.attentionHeadDimension;
    mhc_ = {hidden, d_.sinkhornIterations, d_.hcEpsilon, d_.normEpsilon};
    freqs_ = upload(ropeFrequencies(layout_, false));
    for (size_t layer = 0; layer < weights.layers.size(); ++layer)
      rings_.push_back(zeroed(uint64_t{layout_.slidingWindow} * latent * 2));
    // What lives across the calls stays out of their pool.
    persistent_ = std::exchange(buffers_, {});
    next_ = 0;
    ops::LinearScratchSize decode{}, prefill{};
    const auto include = [&](const ops::Projection &p) {
      decode.include(operators_.linear().decodeScratchSize(p.shape()));
      prefill.include(operators_.linear().prefillScratchSize(p.shape()));
    };
    for (const model::DeepSeek41LayerWeights &layer : weights.layers) {
      for (const ops::Projection *p : {&layer.queryA, &layer.queryB, &layer.kv, &layer.outputB}) include(*p);
      for (const ops::Projection &p : layer.outputA) include(p);
    }
    include(weights.mainProjection);
    include(target.logitsProjection);
    decodeScratch_ = {zeroed(decode.input), zeroed(decode.sums), zeroed(decode.partials), zeroed(decode.counters),
                      zeroed(decode.rotated)};
    prefillScratch_ = {zeroed(prefill.input), zeroed(prefill.sums), zeroed(prefill.partials),
                       zeroed(prefill.counters), zeroed(prefill.rotated)};
    moeShape_ = layout_.moeBlockShape();
    std::map<uint32_t, uint64_t> formats;
    for (const model::DeepSeek41LayerWeights &layer : weights.layers)
      for (const ops::BlockExpertProjection *p : {&layer.ffn.gate, &layer.ffn.up, &layer.ffn.down})
        if (!p->routed.isFloat()) formats[p->routed.formatId] += uint64_t{p->routed.outputSize} * p->routed.inputSize;
    moeShape_.expertFormat =
        std::max_element(formats.begin(), formats.end(), [](auto &a, auto &b) { return a.second < b.second; })->first;
    moeShape_.registerFormats = false;
    moeShape_.routerFormat = weights.layers.front().ffn.router.formatId;
    if (!moeShape_.valid()) throw std::runtime_error("invalid DSpark MoE shape");
    for (MetalBuffer &buffer : buffers_) persistent_.push_back(std::move(buffer));
    buffers_.clear();
    next_ = 0;
  }

  // The latents of the target rows `first` to first + rows - 1 from their
  // hidden views (bf16 [rows][targets * hidden]) into each layer's ring.
  void addContext(const MetalBuffer &mainHidden, uint32_t first, uint32_t rows) {
    next_ = 0;
    const uint32_t hidden = layout_.hiddenSize, latent = layout_.attentionHeadDimension;
    std::vector<uint32_t> positions(rows);
    for (uint32_t r = 0; r < rows; ++r) positions[r] = first + r;
    const MetalBuffer positionBuffer = upload(positions);
    MetalBuffer projected = alloc(uint64_t{pad(rows)} * hidden * 2), main = alloc(uint64_t{pad(rows)} * hidden * 2);
    MetalBuffer raw = alloc(uint64_t{pad(rows)} * latent * 2), latents = alloc(uint64_t{pad(rows)} * latent * 2);
    metal::CommandGraph graph;
    operators_.linear().addPrefill(graph, mainHidden, weights_.mainProjection, projected, rows, prefillScratch_);
    ops::Normalization::addRms(graph, projected, weights_.mainNorm, main, hidden, rows);
    for (size_t index = 0; index < weights_.layers.size(); ++index) {
      const model::DeepSeek41LayerWeights &layer = weights_.layers[index];
      operators_.linear().addPrefill(graph, main, layer.kv, raw, rows, prefillScratch_);
      ops::LatentRotation::add(graph, raw, false, layer.kvNorm.buffer, freqs_, positionBuffer, latents,
                               {1, latent, layout_.rotaryPairs, false, ops::LatentRounding::Fp8, true, d_.normEpsilon},
                               rows);
      ops::LatentAttention::addRingUpdate(graph, rings_[index], latents, rings_[index], {}, first, rows,
                                          layout_.slidingWindow);
    }
    run(graph);
    seen_ = std::max(seen_, first + rows);
  }

  // A block's draft from `anchor`, the token at position `position`, past
  // every position the rings took.
  Draft draft(uint32_t anchor, uint32_t position) {
    next_ = 0;
    const model::DSparkLayout &dspark = weights_.layout;
    const uint32_t B = dspark.blockSize, hidden = layout_.hiddenSize, latent = layout_.attentionHeadDimension;
    const uint32_t heads = layout_.attentionQueryHeads, pairs = layout_.rotaryPairs, ring = layout_.slidingWindow;
    const uint32_t vocabulary = layout_.vocabularySize, rows = pad(B);
    if (position != seen_) throw std::runtime_error("a draft past positions its rings did not take");
    std::vector<uint32_t> tokens(B, dspark.noiseToken), positions(B);
    tokens[0] = anchor;
    for (uint32_t r = 0; r < B; ++r) positions[r] = position + r;
    const MetalBuffer tokenBuffer = upload(tokens), positionBuffer = upload(positions);
    // Every row's keys: the rings' filled slots, then every row of the block.
    const uint32_t filled = std::min(ring, position), keyStride = ring + B;
    std::vector<uint32_t> indices(size_t(B) * keyStride, 0), counts(B, filled + B);
    for (uint32_t r = 0; r < B; ++r) {
      for (uint32_t s = 0; s < filled; ++s) indices[size_t(r) * keyStride + s] = s | LATENT_ATTENTION_WINDOW_KEY;
      for (uint32_t j = 0; j < B; ++j) indices[size_t(r) * keyStride + filled + j] = (ring + j) | LATENT_ATTENTION_WINDOW_KEY;
    }
    const MetalBuffer indexBuffer = upload(indices), countBuffer = upload(counts);
    const MetalBuffer verifyRows = upload(std::vector<uint32_t>{B});
    MetalBuffer streams = zeroed(uint64_t{rows} * MHC_STREAMS * hidden * 4);
    MetalBuffer embedding = alloc(uint64_t{rows} * hidden * 2), chunks = alloc(ops::ManifoldHyperConnection::mixScratchBytes(mhc_, B));
    std::vector<float> first(size_t(B) * MHC_MIXES, 0.0f);
    for (uint32_t r = 0; r < B; ++r) first[size_t(r) * MHC_MIXES] = 1.0f;
    MetalBuffer previous = upload(first);
    metal::CommandGraph graph;
    ops::Embedding::add(graph, tokenBuffer, target_.tokenEmbedding, embedding, B);
    ops::ManifoldHyperConnection::addInit(graph, embedding, streams, mhc_, B);
    const ops::MoePlan plan = operators_.moeDecode(moeShape_, 1);
    for (const model::DeepSeek41LayerWeights &layer : weights_.layers) {
      const size_t index = &layer - weights_.layers.data();
      MetalBuffer mixes = alloc(uint64_t{rows} * MHC_MIXES * 4), x = alloc(uint64_t{rows} * hidden * 2);
      MetalBuffer qa = alloc(uint64_t{rows} * d_.queryRank * 2), qr = alloc(uint64_t{rows} * d_.queryRank * 2);
      MetalBuffer q = alloc(uint64_t{rows} * layout_.attentionWidth * 2),
                  qRotated = alloc(uint64_t{rows} * layout_.attentionWidth * 2);
      MetalBuffer raw = alloc(uint64_t{rows} * latent * 2), block = alloc(uint64_t{rows} * latent * 2);
      MetalBuffer scores = alloc(uint64_t{rows} * heads * keyStride * 4),
                  out = alloc(uint64_t{rows} * heads * latent * 4);
      const uint64_t partBytes = ops::LatentAttention::partBytes(heads, B);
      MetalBuffer parts = partBytes ? alloc(partBytes) : MetalBuffer{};
      MetalBuffer rotatedBack = alloc(uint64_t{rows} * layout_.attentionWidth * 2);
      const uint32_t groupWidth = d_.groupWidth(layout_.attentionWidth);
      MetalBuffer groupInput = alloc(uint64_t{rows} * groupWidth * 2), groupOutput = alloc(uint64_t{rows} * d_.outputRank * 2);
      MetalBuffer joined = alloc(uint64_t{rows} * d_.outputGroups * d_.outputRank * 2), branch = alloc(uint64_t{rows} * hidden * 2);
      ops::ManifoldHyperConnection::addMix(graph, streams, layer.attentionMix.fn, layer.attentionMix.scale,
                                           layer.attentionMix.base, chunks, mixes, mhc_, B);
      ops::ManifoldHyperConnection::addPreNorm(graph, streams, previous, layer.attentionNorm.buffer, x, mhc_, B);
      project(graph, x, layer.queryA, qa);
      ops::LatentRotation::add(graph, qa, false, layer.queryNorm.buffer, freqs_, positionBuffer, qr,
                               {1, d_.queryRank, 0, false, ops::LatentRounding::None, true, d_.normEpsilon}, B);
      project(graph, qr, layer.queryB, q);
      ops::LatentRotation::add(graph, q, false, layer.queryNorm.buffer, freqs_, positionBuffer, qRotated,
                               {heads, latent, pairs, false, ops::LatentRounding::None, false, 0.0f}, B);
      project(graph, x, layer.kv, raw);
      ops::LatentRotation::add(graph, raw, false, layer.kvNorm.buffer, freqs_, positionBuffer, block,
                               {1, latent, pairs, false, ops::LatentRounding::Fp8, true, d_.normEpsilon}, B);
      ops::LatentAttention::add(graph, qRotated, {block, rings_[index], ring}, {block, {}, {}}, indexBuffer,
                                countBuffer, layer.sinks, scores, parts, out, heads, keyStride,
                                1.0f / std::sqrt(float(latent)), B);
      ops::LatentRotation::add(graph, out, true, layer.queryNorm.buffer, freqs_, positionBuffer, rotatedBack,
                               {heads, latent, pairs, true, ops::LatentRounding::Fp8, false, 0.0f}, B);
      for (uint32_t g = 0; g < d_.outputGroups; ++g) {
        ops::RowCopy::add(graph, rotatedBack, {0, layout_.attentionWidth, g * groupWidth}, groupInput,
                          {0, groupWidth, 0}, B, groupWidth);
        project(graph, groupInput, layer.outputA[g], groupOutput);
        ops::RowCopy::add(graph, groupOutput, {0, d_.outputRank, 0}, joined,
                          {0, d_.outputGroups * d_.outputRank, g * d_.outputRank}, B, d_.outputRank);
      }
      project(graph, joined, layer.outputB, branch);
      ops::ManifoldHyperConnection::addPost(graph, streams, branch, mixes, mhc_, B);
      // The FFN.
      MetalBuffer ffnMixes = alloc(uint64_t{rows} * MHC_MIXES * 4), y = alloc(uint64_t{rows} * hidden * 2);
      MetalBuffer residual = zeroed(uint64_t{rows} * hidden * 2), ffnOut = alloc(uint64_t{rows} * hidden * 2);
      ops::MoeScratch scratch;
      for (const ops::MoeScratchField &field : ops::kMoeScratchFields)
        scratch.*field.buffer = zeroed(plan.workspace().*field.bytes);
      ops::ManifoldHyperConnection::addMix(graph, streams, layer.ffnMix.fn, layer.ffnMix.scale, layer.ffnMix.base,
                                           chunks, ffnMixes, mhc_, B);
      ops::ManifoldHyperConnection::addPreNorm(graph, streams, mixes, layer.ffnNorm.buffer, y, mhc_, B);
      ops::MoE::add(graph, {y, residual, ffnOut, scratch, verifyRows}, layer.ffn, plan);
      ops::ManifoldHyperConnection::addPost(graph, streams, ffnOut, ffnMixes, mhc_, B);
      previous = ffnMixes;
    }
    // The head: the last mix's pre-sum through the output norm, the target's
    // logits of it.
    MetalBuffer x = alloc(uint64_t{rows} * hidden * 2), logits = alloc(uint64_t{rows} * vocabulary * 4);
    ops::ManifoldHyperConnection::addPreNorm(graph, streams, previous, weights_.outputNorm.buffer, x, mhc_, B);
    project(graph, x, target_.logitsProjection, logits);
    run(graph);
    // On the host: each row's pre-sum (the confidence head's input), and the
    // Markov chain from the anchor, each step's bias from its token's
    // embedding on the GPU.
    const auto *s = static_cast<const float *>(streams.contents());
    const auto *mix = static_cast<const float *>(previous.contents());
    std::vector<float> pre(size_t(B) * hidden, 0.0f);
    for (uint32_t r = 0; r < B; ++r)
      for (uint32_t k = 0; k < MHC_STREAMS; ++k)
        for (uint32_t c = 0; c < hidden; ++c)
          pre[size_t(r) * hidden + c] += mix[size_t(r) * MHC_MIXES + k] * s[(size_t(r) * MHC_STREAMS + k) * hidden + c];
    const auto *weights = static_cast<const float *>(weights_.confidence.contents());
    const auto *rowLogits = static_cast<const float *>(logits.contents());
    const uint32_t rank = dspark.markovRank;
    MetalBuffer token = alloc(16), markov = alloc(uint64_t{32} * rank * 2), bias = alloc(uint64_t{32} * vocabulary * 4);
    Draft result;
    uint32_t previousToken = anchor;
    for (uint32_t r = 0; r < B; ++r) {
      *static_cast<uint32_t *>(token.contents()) = previousToken;
      metal::CommandGraph step;
      ops::Embedding::add(step, token, weights_.markovEmbedding, markov, 1);
      ops::addGgufFloat(step, markov, weights_.markovHead.blocks().segments.front(), bias, 1, vocabulary, 0,
                        ops::FloatOutput::Float32, ops::FloatTile::Simdgroup);
      run(step);
      const auto *b = static_cast<const float *>(bias.contents());
      const float *l = rowLogits + size_t(r) * vocabulary;
      uint32_t best = 0;
      for (uint32_t v = 1; v < vocabulary; ++v)
        if (l[v] + b[v] > l[best] + b[best]) best = v;
      const auto *e = static_cast<const uint16_t *>(markov.contents());
      double score = 0;
      for (uint32_t c = 0; c < hidden; ++c) score += double(weights[c]) * f32(bf16(pre[size_t(r) * hidden + c]));
      for (uint32_t c = 0; c < rank; ++c) score += double(weights[hidden + c]) * f32(e[c]);
      result.tokens.push_back(best);
      result.confidence.push_back(float(score));
      previousToken = best;
    }
    return result;
  }

private:
  // A projection of a block's rows on the decode plans of one lane.
  void project(metal::CommandGraph &graph, const MetalBuffer &input, const ops::Projection &projection,
               const MetalBuffer &output) {
    const ops::Linear &linear = operators_.linear();
    static_cast<void>(linear.add(graph, {.input = input, .output = output, .scratch = decodeScratch_}, projection,
                                 linear.decodePlan(projection, 1)));
  }
  template <class T> MetalBuffer upload(const std::vector<T> &values) {
    MetalBuffer buffer = alloc(values.size() * sizeof(T));
    std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
    return buffer;
  }
  // The buffers of a draft or a context, the next of those the previous ones
  // took in the same order (each call's run waits for its commands), larger
  // ones allocated where a call needs more.
  MetalBuffer alloc(uint64_t bytes) {
    bytes = std::max<uint64_t>(bytes, 16);
    if (next_ < buffers_.size() && buffers_[next_].sizeBytes() >= bytes) return buffers_[next_++];
    MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, "dspark");
    if (next_ < buffers_.size()) buffers_[next_] = buffer;
    else buffers_.push_back(buffer);
    ++next_;
    return buffer;
  }
  MetalBuffer zeroed(uint64_t bytes) {
    MetalBuffer buffer = alloc(bytes);
    std::memset(buffer.contents(), 0, std::max<uint64_t>(bytes, 16));
    return buffer;
  }
  void run(metal::CommandGraph &graph) {
    const metal::CommandTiming timing = backend_.submitCommandAsync(graph.command()).wait();
    gpuSeconds_ += timing.gpuSeconds;
  }

  metal::MetalBackend &backend_;
  const model::DeepSeek41Weights &target_;
  const model::DSparkWeights &weights_;
  const model::DeepSeek41Layout &layout_;
  const model::DeepSeek41Dimensions &d_;
  ops::ExecutionPlans operators_;
  ops::ManifoldHyperShape mhc_{};
  ops::MoeShape moeShape_{};
  ops::LinearScratch decodeScratch_{}, prefillScratch_{};
  MetalBuffer freqs_;
  std::vector<MetalBuffer> rings_;
  // The positions the rings took, from the start.
  uint32_t seen_ = 0;
  double gpuSeconds_ = 0;
  std::vector<MetalBuffer> buffers_, persistent_;
  size_t next_ = 0;
};

// The Engram's tables in the target's split GGUF (per_layer_token_embd, its
// last part): each Engram layer's rows of IQ4_NL blocks from its first row
// (DeepSeek41Weights::engramFirstRows), which a model::SparseRowStore per
// table reads from the file on the SSD readers and stages for a step.
class EngramTables {
public:
  EngramTables(const model::GgufFile &target, const model::DeepSeek41Weights &weights,
               std::shared_ptr<model::SsdReaders> readers) {
    const model::DeepSeek41EngramTables tables = model::deepSeek41EngramTables(target, weights.layout);
    source_ = std::make_shared<model::WeightSource>(target.source(tables.file).path());
    const std::vector<uint32_t> &first = weights.engramFirstRows;
    for (size_t table = 0; table < first.size(); ++table) {
      const uint64_t end = table + 1 < first.size() ? first[table + 1] : tables.table.rows;
      stores_.push_back(std::make_unique<model::SparseRowStore>(
          model::SparseRowTable{tables.table.name, end - first[table], tables.table.rowBytes,
                                tables.table.offset + uint64_t{first[table]} * tables.table.rowBytes},
          model::openRowReader(source_),
          model::SparseRowStoreOptions{.cacheBytes = 64ull << 20, .readers = readers, .stageRows = 1u << 15}));
    }
  }
  // Engram layer `table`'s store.
  [[nodiscard]] model::SparseRowStore &store(uint32_t table) const { return *stores_.at(table); }

private:
  std::shared_ptr<model::WeightSource> source_;
  std::vector<std::unique_ptr<model::SparseRowStore>> stores_;
};

// The mean norm of each row's streams.
double meanNorm(const float *streams, uint32_t rows, uint32_t hidden) {
  double total = 0;
  for (uint32_t v = 0; v < rows * MHC_STREAMS; ++v) {
    double squares = 0;
    for (uint32_t i = 0; i < hidden; ++i) squares += double(streams[size_t(v) * hidden + i]) * streams[size_t(v) * hidden + i];
    total += std::sqrt(squares);
  }
  return total / (rows * MHC_STREAMS);
}

// The largest difference of the streams from the reference's, over the
// reference's largest magnitude, and their relative L2 distance.
std::pair<double, double> compare(const float *got, const float *want, size_t count) {
  double worst = 0, peak = 0, diff = 0, norm = 0;
  for (size_t i = 0; i < count; ++i) {
    const double e = double(got[i]) - want[i];
    worst = std::max(worst, std::fabs(e));
    peak = std::max(peak, std::fabs(double(want[i])));
    diff += e * e;
    norm += double(want[i]) * want[i];
  }
  return {worst / (peak + 1e-30), std::sqrt(diff / (norm + 1e-30))};
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc < 5)
        throw std::runtime_error(
            "usage: deepseek41-forward METALLIB GGUF_DIRECTORY REFERENCE_DIRECTORY EXPERT_GIB [LAYERS [GENERATE "
            "[ENGRAM]]]");
      metal::MetalBackend backend(argv[1]);
      const std::string reference = argv[3];
      const model::DeepSeek41Layout layout;
      const auto began = std::chrono::steady_clock::now();
      const auto seconds = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
      };
      model::WeightImages images(backend, "deepseek41-forward");
      const auto readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);
      const model::ExpertStoreOptions experts{.memoryBytes = uint64_t(std::stod(argv[4]) * (1ull << 30)),
                                              .readers = readers};
      model::GgufTargetLoader files(backend, images, model::findTargetGgufs(argv[2]), layout.ggufArchitecture, layout,
                                    {}, experts);
      const model::DeepSeek41Weights weights = model::loadDeepSeek41Weights(backend, layout, files);
      std::printf("loaded in %.1f s\n", seconds());
      const bool compared = std::filesystem::exists(reference + "/streams.npy");
      std::optional<Npy> streams;
      std::vector<uint32_t> prompt;
      if (compared) {
        streams = readNpy(reference + "/streams.npy");
        const Npy tokens = readNpy(reference + "/tokens.npy");
        prompt.assign(tokens.values<uint32_t>(), tokens.values<uint32_t>() + streams->shape.at(2));
      } else {
        std::ifstream in(std::filesystem::path(reference).parent_path() / "prompt.txt");
        for (uint32_t token; in >> token;) prompt.push_back(token);
        std::printf("no reference streams: the prompt's %zu tokens, stream norms alone\n", prompt.size());
      }
      const uint32_t rows = uint32_t(prompt.size());
      const uint32_t layers = argc > 5 ? uint32_t(std::stoul(argv[5])) : layout.layers;
      const uint32_t generate = argc > 6 ? uint32_t(std::stoul(argv[6])) : 0;
      // With ENGRAM 1, each row's Engram rows from the target's tables (the
      // reference's gathered rows for a compared prompt, which those from the
      // file are checked against).
      std::optional<EngramTables> engram;
      if (argc > 7 && std::string_view(argv[7]) == "1") engram.emplace(files.file(), weights, readers);
      const model::EngramHashTables hash = weights.engramHash();
      std::vector<uint32_t> sequence = prompt;
      // Engram layer `layer`'s table, and the row ids of the sequence's rows
      // from `first` on, `count` of them, in each row's gathered order.
      const auto engramTable = [&](uint32_t layer) {
        uint32_t table = 0;
        for (uint32_t l = 0; l < layer; ++l) table += model::DeepSeek41Dimensions::has(layout.deepseek.engramLayers, l);
        return table;
      };
      const auto engramIds = [&](uint32_t layer, uint32_t first, uint32_t count) {
        const uint32_t perLayer = hash.rowsPerLayer(), table = engramTable(layer);
        std::vector<uint32_t> all(size_t(hash.layers) * perLayer), ids;
        ids.reserve(size_t(count) * perLayer);
        for (uint32_t r = 0; r < count; ++r) {
          model::engramRows(hash, sequence, first + r, all.data());
          ids.insert(ids.end(), all.begin() + long(table) * perLayer, all.begin() + long(table + 1) * perLayer);
        }
        return ids;
      };
      if (engram) std::printf("Engram rows from %s\n", argv[7]);
      // With DEEPSEEK41_DSPARK, DSpark beside the target, its experts loaded
      // whole.
      model::WeightImages draftImages(backend, "dspark");
      std::optional<model::DSparkWeights> dspark;
      if (const char *path = std::getenv("DEEPSEEK41_DSPARK")) {
        const model::OpenGguf file(std::vector<std::filesystem::path>{path});
        dspark.emplace(model::loadDSparkWeights(backend, draftImages, path, model::readDSparkLayout(file.file(), layout)));
        const model::DSparkLayout &draft = dspark->layout;
        std::printf("DSpark: %u layers of %u experts (top %u), block %u, noise token %u, Markov rank %u, target layers",
                    draft.layers.layers, draft.layers.experts, draft.layers.expertsPerToken, draft.blockSize,
                    draft.noiseToken, draft.markovRank);
        for (const uint32_t layer : draft.targetLayers()) std::printf(" %u", layer);
        std::printf(" (%.1f s)\n", seconds());
      }
      // The capture slot of a layer DSpark reads, if it is one.
      const auto captureSlot = [&](uint32_t layer) -> std::optional<uint32_t> {
        if (!dspark) return std::nullopt;
        const std::span<const uint32_t> targets = dspark->layout.targetLayers();
        const auto found = std::find(targets.begin(), targets.end(), layer);
        if (found == targets.end()) return std::nullopt;
        return uint32_t(found - targets.begin());
      };
      // The prompt and the generated tokens, in whole 64-token pieces, and
      // a verify step's rows past them.
      Forward forward(backend, weights, (rows + generate + 63) / 64 * 64 + 64);
      std::optional<Drafter> drafter;
      if (dspark) {
        forward.captures(dspark->layout.targetLayerCount);
        drafter.emplace(backend, weights, *dspark);
      }
      forward.begin(prompt);
      forward.embed();
      const size_t width = size_t(rows) * MHC_STREAMS * layout.hiddenSize;
      for (uint32_t layer = 0; layer < layers; ++layer) {
        const bool engramLayer = model::DeepSeek41Dimensions::has(layout.deepseek.engramLayers, layer);
        if (compared && engramLayer) {
          const Npy values = readNpy(reference + "/engram_values_" + std::to_string(layer) + ".npy");
          const MetalBuffer want =
              forward.engramRows(std::vector<float>(values.values<float>(), values.values<float>() + values.count()));
          // The rows staged from the file, which the reference's must equal
          // as bf16.
          if (engram) {
            const MetalBuffer got = forward.engramRows(engram->store(engramTable(layer)), engramIds(layer, 0, rows));
            const uint64_t count = uint64_t{rows} * layout.deepseek.engramWidth();
            const std::vector<uint16_t> a = forward.bf16Values(got, count), b = forward.bf16Values(want, count);
            uint64_t differ = 0;
            for (uint64_t i = 0; i < count; ++i) differ += a[i] != b[i];
            std::printf("layer %2u: Engram rows staged from the file: %llu of %llu bf16 values differ from the "
                        "reference's\n",
                        layer, (unsigned long long)differ, (unsigned long long)count);
          }
          forward.engram(layer, want);
        } else if (engram && engramLayer) {
          forward.engram(layer, forward.engramRows(engram->store(engramTable(layer)), engramIds(layer, 0, rows)));
        }
        if (const std::optional<uint32_t> slot = captureSlot(layer)) forward.capture(*slot);
        forward.attention(layer);
        const double attnNorm = meanNorm(forward.streams(), rows, layout.hiddenSize);
        std::pair<double, double> attn{}, ffn{};
        if (compared) attn = compare(forward.streams(), streams->values<float>() + (2 * layer) * width, width);
        forward.ffn(layer);
        const double ffnNorm = meanNorm(forward.streams(), rows, layout.hiddenSize);
        if (compared) ffn = compare(forward.streams(), streams->values<float>() + (2 * layer + 1) * width, width);
        std::printf("layer %2u: attention max %.2e L2 %.2e norm %.4g, ffn max %.2e L2 %.2e norm %.4g (%.1f s)\n", layer,
                    attn.first, attn.second, attnNorm, ffn.first, ffn.second, ffnNorm, seconds());
        // A long prompt's FFN streams by rows: those whose window holds the
        // whole prompt, those past it, and those whose indexer selects
        // among more groups than it keeps.
        if (compared && rows > layout.slidingWindow) {
          const size_t row = size_t(MHC_STREAMS) * layout.hiddenSize;
          const uint32_t bounds[] = {0, layout.slidingWindow, layout.deepseek.indexerTopGroups, rows};
          std::printf("          ffn L2 by rows:");
          for (size_t b = 0; b + 1 < std::size(bounds); ++b) {
            const uint32_t from = std::min(bounds[b], rows), to = std::min(bounds[b + 1], rows);
            if (from >= to) continue;
            const auto [peak, l2] = compare(forward.streams() + from * row,
                                            streams->values<float>() + (2 * layer + 1) * width + from * row,
                                            (to - from) * row);
            std::printf(" [%u, %u) %.2e", from, to, l2);
          }
          std::printf("\n");
        }
        std::fflush(stdout);
      }
      if (layers != layout.layers) return 0;
      const uint32_t vocabulary = layout.vocabularySize;
      const std::vector<float> logits = forward.logits();
      forward.end();
      if (drafter) drafter->addContext(forward.mainHidden(), 0, rows);
      const float *last = logits.data() + size_t(rows - 1) * vocabulary;
      if (compared) {
        const Npy want = readNpy(reference + "/logits.npy");
        uint32_t agree = 0;
        for (uint32_t r = 0; r < rows; ++r) {
          const float *got = logits.data() + size_t(r) * vocabulary, *ref = want.values<float>() + size_t(r) * vocabulary;
          agree += std::max_element(got, got + vocabulary) - got == std::max_element(ref, ref + vocabulary) - ref;
        }
        const auto [peak, l2] = compare(logits.data(), want.values<float>(), size_t(rows) * vocabulary);
        std::printf("logits: max %.2e L2 %.2e, argmax agrees on %u of %u rows\n", peak, l2, agree, rows);
      }
      // The last row's most likely next tokens.
      std::vector<uint32_t> order(vocabulary);
      for (uint32_t i = 0; i < vocabulary; ++i) order[i] = i;
      std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                        [&](uint32_t a, uint32_t b) { return last[a] > last[b]; });
      std::printf("next token, top 5:");
      for (uint32_t i = 0; i < 5; ++i) std::printf(" %u (%.3g)", order[i], last[order[i]]);
      std::printf("\n");
      // The expert store's work so far, of prefill and decode loads: its
      // blocks' loads, the distinct experts they routed and those it read
      // from the SSD.
      const auto expertUse = [&] {
        model::ExpertStoreStats::Use sum{};
        if (!weights.experts) return sum;
        for (const model::ExpertStoreStats::Use &use : weights.experts->stats().uses) {
          sum.loads += use.loads;
          sum.routed += use.routed;
          sum.read += use.read;
          sum.readBytes += use.readBytes;
        }
        return sum;
      };
      const auto printUse = [](const char *what, const model::ExpertStoreStats::Use &now,
                               const model::ExpertStoreStats::Use &before) {
        std::printf("%s experts: %llu block loads, %llu routed, %llu read (%.2f GiB)\n", what,
                    (unsigned long long)(now.loads - before.loads), (unsigned long long)(now.routed - before.routed),
                    (unsigned long long)(now.read - before.read), (now.readBytes - before.readBytes) / double(1ull << 30));
      };
      const model::ExpertStoreStats::Use afterPrompt = expertUse();
      printUse("prompt", afterPrompt, {});
      if (drafter && generate) {
        // Draft, verify the anchor and the drafts in one step, keep the
        // drafts up to the first the target's argmax rejects and the token it
        // predicts there.
        const uint32_t block = dspark->layout.blockSize;
        const char *threshold = std::getenv("DEEPSEEK41_DSPARK_THRESHOLD");
        const std::optional<float> cut = threshold ? std::optional<float>(std::stof(threshold)) : std::nullopt;
        struct Cycle {
          std::vector<float> confidence;
          uint32_t verified = 0, accepted = 0;
          double verify = 0;
        };
        std::vector<Cycle> cycles;
        uint32_t anchor = order[0], produced = 1;
        std::printf("generated: %u", anchor);
        std::vector<uint32_t> kept(block + 1, 0);
        std::vector<double> drafts, verifies, confidence[2];
        static_cast<void>(forward.takeTiming());
        const double began = seconds();
        while (produced < generate) {
          const uint32_t position = forward.past();
          const double started = seconds();
          const Drafter::Draft draft = drafter->draft(anchor, position);
          const double drafted = seconds();
          uint32_t verified = block;
          if (cut) {
            verified = 0;
            while (verified < block && draft.confidence[verified] >= *cut) ++verified;
          }
          std::vector<uint32_t> step{anchor};
          step.insert(step.end(), draft.tokens.begin(), draft.tokens.begin() + verified);
          sequence.resize(position);
          sequence.insert(sequence.end(), step.begin(), step.end());
          forward.begin(step);
          forward.embed();
          for (uint32_t layer = 0; layer < layout.layers; ++layer) {
            if (engram && model::DeepSeek41Dimensions::has(layout.deepseek.engramLayers, layer))
              forward.engram(layer, forward.engramRows(engram->store(engramTable(layer)),
                                                       engramIds(layer, position, uint32_t(step.size()))));
            if (const std::optional<uint32_t> slot = captureSlot(layer)) forward.capture(*slot);
            forward.attention(layer);
            forward.ffn(layer);
          }
          const std::vector<float> next = forward.logits();
          const auto argmax = [&](uint32_t row) {
            const float *l = next.data() + size_t(row) * vocabulary;
            return uint32_t(std::max_element(l, l + vocabulary) - l);
          };
          uint32_t accepted = 0;
          while (accepted < verified && argmax(accepted) == draft.tokens[accepted]) ++accepted;
          const uint32_t predicted = argmax(accepted);
          forward.end(accepted + 1);
          drafter->addContext(forward.mainHidden(), position, accepted + 1);
          for (uint32_t i = 0; i < verified; ++i) confidence[i < accepted].push_back(draft.confidence[i]);
          for (uint32_t i = 0; i < accepted; ++i) std::printf(" %u", draft.tokens[i]);
          std::printf(" %u", predicted);
          std::fflush(stdout);
          ++kept[accepted];
          produced += accepted + 1;
          anchor = predicted;
          sequence.resize(position + accepted + 1);
          drafts.push_back(drafted - started);
          verifies.push_back(seconds() - drafted);
          cycles.push_back({draft.confidence, verified, accepted, verifies.back()});
        }
        const double elapsed = seconds() - began;
        const auto median = [](std::vector<double> values) {
          std::sort(values.begin(), values.end());
          return values.empty() ? 0.0 : values[values.size() / 2];
        };
        const auto mean = [](const std::vector<double> &values) {
          double sum = 0;
          for (const double value : values) sum += value;
          return values.empty() ? 0.0 : sum / double(values.size());
        };
        std::printf("\nspeculative: %u tokens in %zu cycles, %.2f a cycle, %.3f s (%.2f tokens/s)\n", produced - 1,
                    drafts.size(), double(produced - 1) / double(drafts.size()), elapsed,
                    double(produced - 1) / elapsed);
        std::printf("drafts kept per cycle:");
        for (uint32_t i = 0; i <= block; ++i) std::printf(" %u:%u", i, kept[i]);
        std::printf("\ncycle medians: draft %.3f s, verify %.3f s; confidence of kept drafts %.3f, of rejected %.3f\n",
                    median(drafts), median(verifies), mean(confidence[1]), mean(confidence[0]));
        printUse("speculative", expertUse(), afterPrompt);
        // Each cycle, and the verify steps' median time by the rows they ran.
        for (const Cycle &cycle : cycles) {
          std::printf("cycle: verified %u kept %u verify %.3f s confidence", cycle.verified, cycle.accepted,
                      cycle.verify);
          for (const float score : cycle.confidence) std::printf(" %.2f", score);
          std::printf("\n");
        }
        for (uint32_t rows = 1; rows <= block + 1; ++rows) {
          std::vector<double> times;
          for (const Cycle &cycle : cycles)
            if (cycle.verified + 1 == rows) times.push_back(cycle.verify);
          if (!times.empty()) std::printf("verify of %u rows: %zu steps, median %.3f s\n", rows, times.size(), median(times));
        }
        return 0;
      }
      // Greedy decode steps of one row each.
      uint32_t token = order[0];
      if (generate) std::printf("generated: %u", token);
      std::vector<double> steps, gpu, waits, submitted;
      static_cast<void>(forward.takeTiming());
      const char *profiling = std::getenv("DEEPSEEK41_PROFILE");
      const uint32_t profiledStep = profiling && std::string_view(profiling) == "1" ? 10 : 0;
      // Each decode step's routed experts per layer.
      std::vector<std::vector<std::vector<uint32_t>>> decodeRoutes;
      for (uint32_t step = 1; step < generate; ++step) {
        if (step == profiledStep) metal::BackendInstrumentation::setDispatchProfiling(backend, true);
        const double started = seconds();
        sequence.push_back(token);
        forward.begin({token});
        forward.embed();
        for (uint32_t layer = 0; layer < layers; ++layer) {
          if (engram && model::DeepSeek41Dimensions::has(layout.deepseek.engramLayers, layer))
            forward.engram(layer, forward.engramRows(engram->store(engramTable(layer)),
                                                     engramIds(layer, uint32_t(sequence.size() - 1), 1)));
          forward.attention(layer);
          forward.ffn(layer);
        }
        const std::vector<float> next = forward.logits();
        decodeRoutes.push_back(forward.routes(0));
        forward.end();
        token = uint32_t(std::max_element(next.begin(), next.end()) - next.begin());
        const metal::CommandTiming timing = forward.takeTiming();
        if (step == profiledStep) {
          // The step's dispatches, each its own command: their GPU time by
          // kernel, the largest first. The step itself counts in no median.
          metal::BackendInstrumentation::setDispatchProfiling(backend, false);
          std::map<std::string, std::pair<double, uint32_t>> kernels;
          double total = 0;
          for (const metal::DispatchTiming &dispatch : metal::BackendInstrumentation::takeDispatchProfile(backend)) {
            auto &[time, count] = kernels[dispatch.pipelineName];
            time += dispatch.gpuSeconds;
            ++count;
            total += dispatch.gpuSeconds;
          }
          std::vector<std::pair<std::string, std::pair<double, uint32_t>>> ranked(kernels.begin(), kernels.end());
          std::sort(ranked.begin(), ranked.end(),
                    [](const auto &a, const auto &b) { return a.second.first > b.second.first; });
          std::printf("\nprofiled decode step %u: %.1f ms GPU in its dispatches, by kernel:\n", step, total * 1e3);
          for (size_t i = 0; i < ranked.size() && i < 30; ++i)
            std::printf("  %7.2f ms %5.1f%% %5u x %s\n", ranked[i].second.first * 1e3,
                        100.0 * ranked[i].second.first / total, ranked[i].second.second, ranked[i].first.c_str());
          std::printf("generated (continued):");
        } else {
          steps.push_back(seconds() - started);
          gpu.push_back(timing.gpuSeconds);
          waits.push_back(timing.hostSeconds);
          submitted.push_back(timing.wallSeconds);
        }
        std::printf(" %u", token);
        std::fflush(stdout);
      }
      if (generate) {
        std::printf("\n");
        printUse("decode", expertUse(), afterPrompt);
        // The distinct experts k consecutive decode tokens route per layer,
        // averaged over every window of k tokens and every layer: what a
        // verify step of k rows reads, against k * experts per token.
        for (uint32_t k = 1; k <= 8 && k <= decodeRoutes.size(); ++k) {
          double distinct = 0;
          uint64_t windows = 0;
          for (size_t first = 0; first + k <= decodeRoutes.size(); ++first)
            for (uint32_t layer = 0; layer < layout.layers; ++layer) {
              std::vector<uint32_t> union_;
              for (uint32_t i = 0; i < k; ++i) {
                const auto &ids = decodeRoutes[first + i][layer];
                union_.insert(union_.end(), ids.begin(), ids.end());
              }
              std::sort(union_.begin(), union_.end());
              distinct += double(std::unique(union_.begin(), union_.end()) - union_.begin());
              ++windows;
            }
          const double mean = distinct / double(windows);
          std::printf("experts of %u consecutive tokens per layer: %.2f distinct (%.0f%% of %u)\n", k, mean,
                      100.0 * mean / (k * layout.expertsPerToken), k * layout.expertsPerToken);
        }
        std::sort(steps.begin(), steps.end());
        std::printf("decode steps: %zu, median %.3f s, fastest %.3f s, slowest %.3f s\n", steps.size(),
                    steps.empty() ? 0.0 : steps[steps.size() / 2], steps.empty() ? 0.0 : steps.front(),
                    steps.empty() ? 0.0 : steps.back());
        // Where a step's time goes: its commands' GPU time, of it the waits
        // on the expert loads' host steps, and the commands' wall time from
        // submission to completion (the rest of a step is the host's work
        // before and between them).
        const auto median = [](std::vector<double> values) {
          std::sort(values.begin(), values.end());
          return values.empty() ? 0.0 : values[values.size() / 2];
        };
        std::printf("decode step medians: commands %.3f s wall, GPU %.3f s, of it %.3f s waiting on expert loads\n",
                    median(submitted), median(gpu), median(waits));
      }
      return 0;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "error: %s\n", error.what());
      return 1;
    }
  }
}
