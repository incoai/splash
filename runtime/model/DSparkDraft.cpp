#include "model/DSparkDraft.hpp"

#include "Checked.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/QuantFormat.h"
#include "model/Bfloat16.hpp"
#include "model/GgufPreparation.hpp"
#include "model/GgufTarget.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

constexpr uint32_t kRows = ExecutionLimits::draftQueryRows;
constexpr uint32_t kRank = SPLASH_DSPARK_RANK;
constexpr uint32_t kTaps = SPLASH_DSPARK_TAPS;
constexpr uint32_t kLanes = ExecutionLimits::maximumBatchWidth;
// The architecture of a DSpark GGUF: llama.cpp's DFlash drafts', whose
// DSpark keys (dflash.dspark.*) tell it apart.
constexpr std::string_view kArchitecture = "dflash";
// The most rows of a fold's block, a decode step's of every lane: a block is
// the runtime's step (DraftModel::addSetup), at most these.
constexpr uint32_t kFoldBlock = ExecutionLimits::maximumBatchWidth * ExecutionLimits::targetVerifyRows;

std::string key(std::string_view name) { return std::string(kArchitecture) + "." + std::string(name); }

uint32_t requireUnsigned(const GgufMetadata &metadata, std::string_view name) {
  const std::optional<uint64_t> value = metadata.unsignedValue(key(name));
  if (!value || *value > UINT32_MAX) throw GgufError("the DSpark GGUF states no " + key(name));
  return static_cast<uint32_t>(*value);
}

float requireFloat(const GgufMetadata &metadata, std::string_view name) {
  const std::optional<double> value = metadata.floatValue(key(name));
  if (!value) throw GgufError("the DSpark GGUF states no " + key(name));
  return static_cast<float>(*value);
}

void expect(const GgufMetadata &metadata, std::string_view name, uint64_t value) {
  if (requireUnsigned(metadata, name) != value)
    throw GgufError("the DSpark kernels take " + key(name) + " " + std::to_string(value));
}

// Plans one image of a DSpark GGUF's tensors, each of the type and shape the
// draft reads (shape as GGUF states it: the row length first).
class Builder final {
public:
  Builder(const GgufFile &file, std::string name, uint32_t layer, uint32_t type)
      : file_(file), image_(std::move(name), layer, type) {}

  // F32 values as stored: a norm, a mix's gamma, a bias, the mix's weights.
  void floats(const std::string &name, std::initializer_list<uint64_t> shape) {
    const GgufTensor &t = require(name, shape, ggml::kF32);
    image_.copy(rows(t, 1, t.bytes));
  }
  // F32 values as the bf16 values they equal, which the kernels read.
  void narrowed(const std::string &name, uint64_t elements) {
    const GgufTensor &t = require(name, {elements}, ggml::kF32);
    image_.copy(rows(t, 1, t.bytes), gguf::Conversion::NarrowToBfloat16);
  }
  // BF16 rows [rows][columns] as stored.
  void bfloat16(const std::string &name, uint64_t rowCount, uint64_t columns) {
    const GgufTensor &t = require(name, {columns, rowCount}, ggml::kBF16);
    image_.copy(rows(t, 1, t.bytes));
  }
  // Tensors [N_i, columns] stacked into one projection, a segment of each run
  // of tensors of one type, repacked in its format.
  void projection(std::initializer_list<std::pair<std::string, uint64_t>> parts, uint64_t columns) {
    std::vector<std::pair<const GgufTensor *, uint64_t>> tensors;
    for (const auto &[name, count] : parts) tensors.emplace_back(&require(name, {columns, count}), count);
    for (size_t run = 0; run < tensors.size();) {
      size_t end = run;
      uint64_t runRows = 0;
      while (end < tensors.size() && tensors[end].first->type == tensors[run].first->type)
        runRows += tensors[end++].second;
      const uint32_t format = gguf_format_of(tensors[run].first->type);
      if (format == GGUF_FMT_COUNT || runRows % QUANT_TILE_ROWS)
        throw GgufError("a DSpark projection is not of a quantized format in whole tiles: " +
                        tensors[run].first->name);
      gguf::Repack repack = image_.planes(format, runRows, columns, tensors[run].first->name);
      for (size_t part = run; part < end; ++part)
        repack.sources.push_back(
            rows(*tensors[part].first, tensors[part].second, ggufRowBytes(kQuantFormats[format], columns)));
      image_.repack(std::move(repack));
      run = end;
    }
  }
  [[nodiscard]] gguf::Image finish() { return image_.finish(); }

private:
  const GgufTensor &require(const std::string &name, std::initializer_list<uint64_t> shape,
                            std::optional<uint32_t> type = std::nullopt) const {
    const GgufTensor &t = file_.require(name);
    if (!std::ranges::equal(t.dims, shape) || (type && t.type != *type))
      throw GgufError("unexpected type or shape for DSpark tensor " + name);
    return t;
  }
  gguf::TensorRows rows(const GgufTensor &t, uint64_t count, uint64_t rowBytes) const {
    if (!count || count * rowBytes != t.bytes) throw GgufError("unexpected size for DSpark tensor " + t.name);
    return {t.name, t.type, t.offset, count, rowBytes, {}, &file_.source(t.file)};
  }

  const GgufFile &file_;
  gguf::ImageBuilder image_;
};

gguf::Image layerImage(const GgufFile &file, const DFlashDraftLayout &layout, uint32_t layer) {
  Builder b(file, "layer-" + std::to_string(layer) + ".bin", layer, 0);
  const std::string p = "blk." + std::to_string(layer) + ".";
  const uint64_t hidden = layout.hiddenSize, kv = uint64_t{layout.kvHeads} * layout.attentionHeadDimension;
  b.floats(p + "attn_norm.weight", {hidden});
  b.projection({{p + "attn_q.weight", layout.attentionSize}, {p + "attn_k.weight", kv}, {p + "attn_v.weight", kv}},
               hidden);
  b.narrowed(p + "attn_q_norm.weight", layout.attentionHeadDimension);
  b.narrowed(p + "attn_k_norm.weight", layout.attentionHeadDimension);
  b.projection({{p + "attn_output.weight", hidden}}, layout.attentionSize);
  b.floats(p + "ffn_norm.weight", {hidden});
  b.projection({{p + "ffn_gate.weight", layout.intermediateSize}}, hidden);
  b.projection({{p + "ffn_up.weight", layout.intermediateSize}}, hidden);
  b.projection({{p + "ffn_down.weight", hidden}}, layout.intermediateSize);
  b.floats(p + "dflare_norm.weight", {hidden});
  b.projection({{p + "attn_k_ctx.weight", kv}, {p + "attn_v_ctx.weight", kv}}, hidden);
  b.narrowed(p + "attn_k_ctx_norm.weight", layout.attentionHeadDimension);
  return b.finish();
}

gguf::Image modelImage(const GgufFile &file, const DFlashDraftLayout &layout) {
  Builder b(file, "model.bin", layout.layers, 1);
  const uint64_t hidden = layout.hiddenSize;
  b.floats("output_norm.weight", {hidden});
  b.floats("dspark_norm.weight", {hidden});
  b.floats("dspark_fuse.weight", {kTaps, layout.layers + 1});
  b.narrowed("mask_embd.weight", hidden);
  b.bfloat16("markov_w1.weight", layout.vocabularySize, kRank);
  b.bfloat16("joint_proj.weight", SPLASH_DSPARK_JOINT, 2 * kRank + 2 * hidden);
  b.floats("joint_proj.bias", {SPLASH_DSPARK_JOINT});
  b.bfloat16("dspark_hidden_out.weight", hidden, kRank);
  b.floats("dspark_hidden_out.bias", {hidden});
  return b.finish();
}

// A projection of the segments its planner wrote for its runs of tensors of
// one type, in row order, each at its first row.
ops::Projection readProjection(WeightFile &file, uint32_t rows, uint32_t inputs, const std::string &label) {
  ops::BlockWeights weights;
  uint32_t offset = 0;
  while (offset < rows) {
    ops::QuantizedSegment segment = readQuantizedSegment(file, label + "-" + std::to_string(weights.segments.size()));
    if (segment.inputSize != inputs || segment.outputSize > rows - offset)
      throw WeightStoreError("DSpark tensor does not match the layout: " + label);
    segment.columnOffset = offset;
    offset += segment.outputSize;
    weights.segments.push_back(std::move(segment));
  }
  return {rows, inputs, std::move(weights)};
}

// Softmax of each consumer's raw mix weights (dspark_fuse, a row a consumer:
// the layers', then the Markov head's), as the reference computes them.
std::vector<std::array<float, kTaps>> mixWeights(std::span<const uint8_t> raw, uint32_t consumers) {
  std::vector<std::array<float, kTaps>> result(consumers);
  for (uint32_t consumer = 0; consumer < consumers; ++consumer) {
    std::array<double, kTaps> values{};
    double top = -INFINITY;
    for (uint32_t tap = 0; tap < kTaps; ++tap) {
      float value;
      std::memcpy(&value, raw.data() + (consumer * kTaps + tap) * sizeof(float), sizeof(float));
      values[tap] = value;
      top = std::max(top, values[tap]);
    }
    double sum = 0.0;
    for (double &value : values) sum += value = std::exp(value - top);
    for (uint32_t tap = 0; tap < kTaps; ++tap) result[consumer][tap] = static_cast<float>(values[tap] / sum);
  }
  return result;
}

// The fold's rows: the output projection's columns (W_out [hidden][rank],
// bf16) as rows of `hidden`, then its F32 bias as two bf16 rows, its value
// rounded and the rest, which sum to it within 2^-16 of it.
std::vector<uint16_t> foldRows(std::span<const uint8_t> weights, std::span<const uint8_t> bias, uint32_t hidden) {
  std::vector<uint16_t> rows(uint64_t{kRank + 2} * hidden);
  for (uint32_t output = 0; output < hidden; ++output)
    for (uint32_t column = 0; column < kRank; ++column)
      std::memcpy(&rows[uint64_t{column} * hidden + output],
                  weights.data() + (uint64_t{output} * kRank + column) * sizeof(uint16_t), sizeof(uint16_t));
  const auto bfloat16 = [](float value) {
    // Round to nearest even.
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  };
  for (uint32_t output = 0; output < hidden; ++output) {
    float value;
    std::memcpy(&value, bias.data() + uint64_t{output} * sizeof(float), sizeof(float));
    const uint16_t high = bfloat16(value);
    rows[uint64_t{kRank} * hidden + output] = high;
    rows[uint64_t{kRank + 1} * hidden + output] = bfloat16(value - widenBfloat16(high));
  }
  return rows;
}

// Layer `layer`'s rows of a buffer of each layer's context rows, its rows'
// share of the buffer ([layer][row], each layer's as many rows as the buffer
// holds of all), from row `first` on, `count` rows or, for 0, all its rows:
// a projection of a layer's rows binds the rows its plan stores.
metal::MetalBuffer layerRows(const metal::MetalBackend &backend, const metal::MetalBuffer &rows,
                             const DFlashDraftLayout &layout, uint32_t layer, uint32_t first = 0, uint32_t count = 0) {
  const uint64_t rowBytes = uint64_t{layout.hiddenSize} * kBFloat16Bytes;
  const uint64_t share = rows.sizeBytes() / layout.layers / rowBytes * rowBytes;
  if (uint64_t{first + count} * rowBytes > share) throw std::invalid_argument("DSpark context rows past their buffer");
  return backend.view(rows, layer * share + first * rowBytes, count ? count * rowBytes : share - first * rowBytes);
}

// Throws unless `layout` is a DSpark draft's.
void requireDSpark(const DFlashDraftLayout &layout) {
  if (layout.kind != DraftKind::DSpark || !layout.present() || layout.targetLayerCount != kTaps ||
      layout.selectorRank != kRank || !layout.proposedVocabulary)
    throw WeightStoreError("not a DSpark draft's layout");
}

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes, metal::BufferStorage storage,
                            std::string_view label) {
  return backend.allocateBuffer(bytes, storage, std::string("dspark-") + std::string(label));
}

// The fold's rows, its output projection's columns and its bias's two
// parts, padded for its blocks: a block of up to kFoldBlock rows binds the
// rows its plan stores, up to kFoldBlock past its first
// (DSparkDraft::addSetup).
uint64_t foldInputBytes(const DFlashDraftLayout &layout) {
  return uint64_t{(kRank + 2 + kFoldBlock - 1) / kFoldBlock * kFoldBlock} * layout.hiddenSize * kBFloat16Bytes;
}

// The bytes the draft allocates beside its images: W2 and b2, the Markov
// workspace of every lane and the lanes' f_rnn rows, and the fold's rows.
uint64_t workspaceBytes(const DFlashDraftLayout &layout) {
  const ops::DSparkMarkovWorkspace markov = ops::DSpark::markovWorkspace(kLanes, layout.vocabularySize);
  return uint64_t{layout.vocabularySize} * kRank * kBFloat16Bytes + uint64_t{layout.vocabularySize} * sizeof(float) +
         markov.jointBytes + markov.stateBytes + markov.previousBytes + markov.logitsBytes +
         markov.partialValuesBytes + markov.partialIdsBytes + uint64_t{kLanes} * layout.hiddenSize * kBFloat16Bytes +
         foldInputBytes(layout);
}

} // namespace

DSparkHeader readDSparkHeader(const GgufFile &file) {
  const GgufMetadata &metadata = file.metadata();
  if (metadata.architecture() != kArchitecture || !metadata.unsignedValue(key("dspark.markov_rank")))
    throw GgufError("the draft GGUF is not a DSpark draft (" + std::string(kArchitecture) + " with dspark keys)");
  DSparkHeader header;
  header.hiddenSize = requireUnsigned(metadata, "embedding_length");
  header.vocabularySize = requireUnsigned(metadata, "vocab_size");
  header.layers = requireUnsigned(metadata, "block_count");
  header.intermediateSize = requireUnsigned(metadata, "feed_forward_length");
  header.rotaryTheta = requireFloat(metadata, "rope.freq_base");
  header.markovRank = requireUnsigned(metadata, "dspark.markov_rank");
  header.proposedVocabulary = requireUnsigned(metadata, "vocab_size_unpadded");
  header.logitCap = requireFloat(metadata, "final_logit_softcapping");
  // The attention the draft kernels take: 24 query heads over 8 KV heads of
  // 128 dimensions, RoPE over all of them, a block of the kernels' rows that
  // attends non-causally, proposing from row 1 on.
  expect(metadata, "attention.head_count", 3 * SPLASH_DRAFT_KV_HEADS);
  expect(metadata, "attention.head_count_kv", SPLASH_DRAFT_KV_HEADS);
  expect(metadata, "attention.key_length", SPLASH_DRAFT_HEAD_DIMENSION);
  expect(metadata, "attention.value_length", SPLASH_DRAFT_HEAD_DIMENSION);
  expect(metadata, "rope.dimension_count", SPLASH_DRAFT_HEAD_DIMENSION);
  expect(metadata, "block_size", kRows);
  expect(metadata, "attention.causal", 0);
  expect(metadata, "sample_from_anchor", 0);
  expect(metadata, "has_confidence_head", 0);
  const float epsilon = requireFloat(metadata, "attention.layer_norm_rms_epsilon");
  if (std::fabs(epsilon - static_cast<float>(SPLASH_RMS_EPSILON)) > 1e-12F)
    throw GgufError("the DSpark kernels take an RMS epsilon of " + std::to_string(SPLASH_RMS_EPSILON));
  const auto ids = file.numericArray(key("target_layers"));
  if (!ids || ids->empty()) throw GgufError("the DSpark GGUF states no " + key("target_layers"));
  for (const double id : *ids) {
    if (!(id >= 0 && id < UINT32_MAX && id == std::floor(id)))
      throw GgufError("the DSpark GGUF's target layers are layer indices");
    header.targetFeatures.push_back(static_cast<uint32_t>(id));
  }
  const auto pattern = file.numericArray(key("attention.sliding_window_pattern"));
  if (!pattern || pattern->size() != header.layers)
    throw GgufError("the DSpark GGUF states no window pattern of its layers");
  for (uint32_t layer = 0; layer < header.layers && layer < 32; ++layer)
    if ((*pattern)[layer] != 0) header.slidingLayers |= 1u << layer;
  if (header.slidingLayers) header.slidingWindow = requireUnsigned(metadata, "attention.sliding_window");
  // Its projections' format, every one's, by which its decode plans store
  // rows (draftStepLanes).
  for (uint32_t layer = 0; layer < header.layers; ++layer)
    for (const char *name :
         {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "attn_k_ctx", "attn_v_ctx"}) {
      const uint32_t format =
          gguf_format_of(file.require("blk." + std::to_string(layer) + "." + name + ".weight").type);
      if (format == GGUF_FMT_COUNT || (header.projectionFormat != GGUF_FMT_COUNT && format != header.projectionFormat))
        throw GgufError("a DSpark draft's projections are of one quantized format");
      header.projectionFormat = format;
    }
  return header;
}

DSparkHeader readDSparkHeader(const std::filesystem::path &directory) {
  const OpenGguf source(findTargetGgufs(directory), GgufContents::Headers);
  return readDSparkHeader(source.file());
}

std::vector<gguf::Image> planDSparkImages(const GgufFile &file, const DFlashDraftLayout &layout) {
  requireDSpark(layout);
  std::vector<gguf::Image> images;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) images.push_back(layerImage(file, layout, layer));
  images.push_back(modelImage(file, layout));
  return images;
}

uint64_t dsparkDraftBytes(const std::filesystem::path &directory, const DFlashDraftLayout &layout) {
  const OpenGguf source(findTargetGgufs(directory), GgufContents::Headers);
  uint64_t bytes = workspaceBytes(layout);
  for (const gguf::Image &image : planDSparkImages(source.file(), layout)) bytes += image.bytes;
  return bytes;
}

DFlashDraftWeights loadDSparkDraftWeights(metal::MetalBackend &backend, WeightImages &images,
                                          const std::filesystem::path &directory, const DFlashDraftLayout &layout) {
  requireDSpark(layout);
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : source(findTargetGgufs(directory)) {}
    OpenGguf source;
    std::vector<gguf::Image> images;
  };
  const auto planned = std::make_shared<Planned>(directory);
  planned->images = planDSparkImages(planned->source.file(), layout);
  const auto open = [&](size_t index) {
    return loadImage(backend, images, "draft/", {planned, &planned->images[index]},
                     [planned] { planned->source.checkUnchanged(); });
  };

  const uint64_t baseline = backend.memoryStats().allocatedBytes;
  DFlashDraftWeights result;
  result.layout = layout;
  auto weights = std::make_shared<DSparkDraftWeights>();
  const uint32_t hidden = layout.hiddenSize;
  const uint64_t headNorm = uint64_t{layout.attentionHeadDimension} * kBFloat16Bytes;
  const uint32_t kv = 2 * layout.kvHeads * layout.attentionHeadDimension;
  std::vector<std::array<float, kTaps>> mixes;
  for (uint32_t index = 0; index < layout.layers; ++index) {
    WeightFile file = open(index);
    DSparkLayerWeights &layer = weights->layers.emplace_back();
    layer.inputNorm = readNorm(file, hidden, true, "input-norm");
    layer.qkv = readProjection(file, layout.qkvSize, hidden, "qkv");
    layer.queryNorm = file.section(headNorm, "query-norm");
    layer.keyNorm = file.section(headNorm, "key-norm");
    layer.output = readProjection(file, hidden, layout.attentionSize, "attention-output");
    layer.ffnNorm = readNorm(file, hidden, true, "ffn-norm");
    layer.gate = readProjection(file, layout.intermediateSize, hidden, "ffn-gate");
    layer.up = readProjection(file, layout.intermediateSize, hidden, "ffn-up");
    layer.down = readProjection(file, hidden, layout.intermediateSize, "ffn-down");
    layer.context.gamma = file.section(uint64_t{hidden} * sizeof(float), "context-norm");
    layer.contextKv = readProjection(file, kv, hidden, "context-kv");
    layer.contextKeyNorm = file.section(headNorm, "context-key-norm");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = open(layout.layers);
    weights->finalNorm = readNorm(file, hidden, true, "final-norm");
    weights->headContext.gamma = file.section(uint64_t{hidden} * sizeof(float), "head-context-norm");
    mixes = mixWeights(file.read(uint64_t{kTaps} * (layout.layers + 1) * sizeof(float), "mix"), layout.layers + 1);
    weights->maskEmbedding = file.section(uint64_t{hidden} * kBFloat16Bytes, "mask-embedding");
    weights->markov.embedding = file.section(uint64_t{layout.vocabularySize} * kRank * kBFloat16Bytes, "markov-embedding");
    weights->markov.joint =
        file.section(uint64_t{SPLASH_DSPARK_JOINT} * (2 * kRank + 2 * hidden) * kBFloat16Bytes, "joint");
    weights->markov.jointBias = file.section(uint64_t{SPLASH_DSPARK_JOINT} * sizeof(float), "joint-bias");
    const std::span<const uint8_t> output = file.read(uint64_t{hidden} * kRank * kBFloat16Bytes, "hidden-out");
    const std::vector<uint8_t> outputs(output.begin(), output.end());
    const std::span<const uint8_t> bias = file.read(uint64_t{hidden} * sizeof(float), "hidden-out-bias");
    weights->foldRows = foldRows(outputs, bias, hidden);
    file.finish();
    result.files.push_back(file.record());
  }
  for (uint32_t layer = 0; layer < layout.layers; ++layer) weights->layers[layer].context.alpha = mixes[layer];
  weights->headContext.alpha = mixes[layout.layers];

  // W2 and b2, filled by the fold, the Markov workspace, and the fold's rows.
  using metal::BufferStorage;
  const uint32_t vocabulary = layout.vocabularySize;
  weights->markov.foldWeights =
      allocate(backend, uint64_t{vocabulary} * kRank * kBFloat16Bytes, BufferStorage::Private, "fold-weights");
  weights->markov.foldBias = allocate(backend, uint64_t{vocabulary} * sizeof(float), BufferStorage::Private, "fold-bias");
  const ops::DSparkMarkovWorkspace markov = ops::DSpark::markovWorkspace(kLanes, vocabulary);
  weights->markovJoint = allocate(backend, markov.jointBytes, BufferStorage::Private, "joint-rows");
  weights->markovState = allocate(backend, markov.stateBytes, BufferStorage::Private, "state");
  weights->markovPrevious = allocate(backend, markov.previousBytes, BufferStorage::Private, "previous");
  weights->markovLogits = allocate(backend, markov.logitsBytes, BufferStorage::Shared, "step-logits");
  weights->markovPartialValues = allocate(backend, markov.partialValuesBytes, BufferStorage::Private, "partial-values");
  weights->markovPartialIds = allocate(backend, markov.partialIdsBytes, BufferStorage::Private, "partial-ids");
  weights->frnnRows = allocate(backend, uint64_t{kLanes} * hidden * kBFloat16Bytes, BufferStorage::Private, "frnn");
  // The fold's rows, in whole blocks: a block's plan binds the rows it
  // stores.
  weights->foldInput = allocate(backend, foldInputBytes(layout), BufferStorage::Shared, "fold-rows");
  std::memset(weights->foldInput.contents(), 0, foldInputBytes(layout));
  std::memcpy(weights->foldInput.contents(), weights->foldRows.data(), weights->foldRows.size() * sizeof(uint16_t));
  result.actualAllocatedBytes = metal::allocationDelta(baseline, backend.memoryStats().allocatedBytes);
  result.dspark = std::move(weights);
  return result;
}

DSparkDraft::DSparkDraft(const DFlashDraftWeights &weights, metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators, const ops::Projection &head,
                         const ops::NormWeights &anchorNorm)
    : layout_(weights.layout), weights_(*weights.dspark), backend_(backend), operators_(operators),
      selector_(weights.layout.vocabularySize), head_(head), anchorNorm_(anchorNorm) {
  requireDSpark(layout_);
  if (weights_.layers.size() != layout_.layers || head.outputSize != layout_.vocabularySize ||
      head.inputSize != layout_.hiddenSize)
    throw std::invalid_argument("a DSpark draft's weights and head are its layout's");
}

DFlashDraftStep DSparkDraft::step(std::span<const uint32_t> cacheLengths, std::span<const std::span<const uint32_t>>,
                                  std::span<const ops::DraftTreeLane> lanes) const {
  if (cacheLengths.size() != lanes.size() || lanes.empty() || lanes.size() > kLanes)
    throw std::invalid_argument("a draft step's lanes, a batch's at most, each have a cache length and a tree");
  return {cacheLengths, lanes, {}};
}

metal::MetalBuffer DSparkDraft::layerWindow(const metal::MetalBuffer &window, uint32_t layer) const {
  const uint64_t bytes = ops::DraftAttention::contextWindowBytes(layout_.hiddenSize);
  return backend_.view(window, layer * bytes, bytes);
}

ops::DraftAttentionWindow DSparkDraft::attentionWindow(uint32_t layer) const {
  // A windowed layer sees the window before the anchor, as trained; the
  // others the ring's every key but the slot the anchor's position takes.
  if (layout_.slidingLayers >> layer & 1) return {layout_.slidingWindow, true};
  return {SPLASH_DRAFT_SLIDING_WINDOW - 1, true};
}

void DSparkDraft::addContextPrefill(metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
                                    std::span<const DFlashPrefillSpan> spans,
                                    std::span<const DFlashWindowCapture> windowCaptures) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  for (const DFlashPrefillSpan &span : spans)
    if (span.ring.size() != layout_.layers || !span.ring.front().auxiliary)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  const uint32_t hidden = layout_.hiddenSize;
  // Layer l's context rows, [l][row] in the context buffer.
  const auto contextRows = [&](uint32_t layer, uint32_t first, uint32_t count) {
    return layerRows(backend_, buffers.hidden, layout_, layer, first, count);
  };
  for (uint32_t layer = 0; layer < layout_.layers; ++layer)
    ops::DSpark::addMix(graph, buffers.capturedTargetHidden, weights_.layers[layer].context,
                        contextRows(layer, 0, rows), rows, hidden);
  // Each span's last row's f_rnn into its lane's; a lane's last span, the
  // last written, holds its prompt's last row.
  ops::DSpark::addMix(graph, buffers.capturedTargetHidden, weights_.headContext, buffers.projected, rows, hidden);
  for (const DFlashPrefillSpan &span : spans)
    ops::RowCopy::add(graph, buffers.projected, {span.compactRow + span.rows - 1, hidden, 0},
                      span.ring.front().auxiliary, {0, hidden, 0}, 1, hidden);
  const auto store = [&](uint32_t compactRow, uint32_t count, uint32_t startPosition,
                         const metal::MetalBuffer &window) {
    for (uint32_t layer = 0; layer < layout_.layers; ++layer)
      ops::DraftAttention::addWindowStore(graph, contextRows(layer, compactRow, count), layerWindow(window, layer),
                                          count, startPosition, hidden);
  };
  for (const DFlashPrefillSpan &span : spans) store(span.compactRow, span.rows, span.startPosition, span.window);
  // Only the window: a cached state keeps no rings.
  for (const DFlashWindowCapture &capture : windowCaptures)
    store(capture.compactRow, capture.rows, capture.startPosition, capture.window);
  addContextKv(graph, buffers, rows, spans);
}

void DSparkDraft::addContextRebuild(metal::CommandGraph &graph, DFlashPrefillBuffers buffers,
                                    const metal::MetalBuffer &window, std::span<const DFlashDraftRingLayer> ring,
                                    uint32_t rows, uint32_t startPosition) const {
  if (!rows || rows > ExecutionLimits::draftContextTokens || ring.size() != layout_.layers)
    throw std::invalid_argument("invalid draft context rebuild");
  for (uint32_t layer = 0; layer < layout_.layers; ++layer)
    ops::DraftAttention::addWindowLoad(graph, layerWindow(window, layer),
                                       layerRows(backend_, buffers.hidden, layout_, layer, 0, rows), rows, startPosition,
                                       layout_.hiddenSize);
  const DFlashPrefillSpan span{0, rows, startPosition, ring, window};
  addContextKv(graph, buffers, rows, {&span, 1});
}

void DSparkDraft::addContextKv(metal::CommandGraph &graph, const DFlashPrefillBuffers &buffers, uint32_t rows,
                               std::span<const DFlashPrefillSpan> spans) const {
  const uint64_t kvBytes = uint64_t{layout_.contextKvSize()} * kBFloat16Bytes;
  const uint64_t ropeBytes = uint64_t{layout_.attentionHeadDimension / 2} * sizeof(float);
  for (uint32_t layer = 0; layer < layout_.layers; ++layer) {
    const DSparkLayerWeights &w = weights_.layers[layer];
    operators_.draftLinear().addProjection(graph, layerRows(backend_, buffers.hidden, layout_, layer), w.contextKv,
                                           buffers.contextKv, rows, buffers.linearScratch);
    for (const DFlashPrefillSpan &span : spans)
      ops::DraftAttention::addContextPrefill(
          graph, backend_.view(buffers.contextKv, span.compactRow * kvBytes, span.rows * kvBytes), w.contextKeyNorm,
          backend_.view(buffers.ropeCos, span.compactRow * ropeBytes, span.rows * ropeBytes),
          backend_.view(buffers.ropeSin, span.compactRow * ropeBytes, span.rows * ropeBytes), span.ring[layer].keys,
          span.ring[layer].values, span.rows, span.startPosition, layout_.attentionShape());
  }
}

void DSparkDraft::addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers, const DFlashDraftStep &step) const {
  const std::span<const uint32_t> cacheLengths = step.cacheLengths;
  const auto lanes = static_cast<uint32_t>(cacheLengths.size());
  if (!lanes || lanes > kLanes || buffers.persistentKeys.size() != layout_.layers ||
      buffers.persistentValues.size() != layout_.layers || step.lanes.size() != lanes)
    throw std::invalid_argument("invalid draft decode batch");
  const uint32_t hidden = layout_.hiddenSize, rows = lanes * kRows;
  const uint64_t rowBytes = uint64_t{hidden} * kBFloat16Bytes;
  const ops::Linear &linear = operators_.draftLinear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  const auto attentionPlan = operators_.draftAttention(layout_.attentionShape(), lanes);

  // The block's rows: each lane's anchor's embedding row normed as its
  // target's embedding is, then the mask rows.
  for (uint32_t lane = 0; lane < lanes; ++lane)
    ops::Normalization::addRms(graph, backend_.view(buffers.hidden[0], lane * kRows * rowBytes, rowBytes), anchorNorm_,
                               backend_.view(buffers.hidden[1], lane * kRows * rowBytes, rowBytes), hidden, 1);
  ops::Embedding::addDraftMaskRows(graph, weights_.maskEmbedding, buffers.hidden[1], hidden, lanes);

  uint32_t current = 1;
  for (uint32_t layer = 0; layer < layout_.layers; ++layer) {
    const DSparkLayerWeights &w = weights_.layers[layer];
    const ops::LinearPlan qkvPlan = linear.plan(w.qkv, rows);
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        graph, buffers.hidden[current], w.inputNorm, buffers.normalized, hidden, rows, scratch, qkvPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.proposalQkv, .scratch = scratch, .prepared = normalized},
               w.qkv, qkvPlan);
    ops::DraftAttention::addPrepare(graph,
                                    {buffers.proposalQkv, buffers.attention, w.queryNorm, w.keyNorm, buffers.ropeCos,
                                     buffers.ropeSin, buffers.queryKeys, buffers.queryValues},
                                    attentionPlan);
    ops::DraftAttention::addDecode(graph,
                                   {buffers.attention, buffers.persistentKeys[layer], buffers.persistentValues[layer],
                                    buffers.queryKeys, buffers.queryValues},
                                   cacheLengths, attentionPlan, attentionWindow(layer));
    linear.addResidualProjection(graph, buffers.attention, w.output, buffers.hidden[current], buffers.residual, rows,
                                 scratch);
    const ops::LinearPlan gateUpPlan = linear.plan(w.up, rows, ops::LinearEpilogue::GateUp, &w.gate);
    const ops::PreparedInput ffnInput = ops::Normalization::addRms(graph, buffers.residual, w.ffnNorm,
                                                                   buffers.normalized, hidden, rows, scratch,
                                                                   gateUpPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.intermediate, .gateScratch = buffers.gateScratch,
                .scratch = scratch, .prepared = ffnInput},
               w.up, gateUpPlan, &w.gate);
    current ^= 1;
    linear.addResidualProjection(graph, buffers.intermediate, w.down, buffers.residual, buffers.hidden[current], rows,
                                 scratch);
  }
  // The final norm and the target's head, on the target's plans of a
  // decode step as its storage and scratch are sized (draftStepLanes,
  // DecodeArena::linearScratchSize): the base logits, uncapped.
  const ops::Linear &target = operators_.decodeLinear();
  const ops::LinearPlan headPlan = target.plan(head_, rows);
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[current], weights_.finalNorm, buffers.finalHidden, hidden, rows, scratch, headPlan.input());
  target.add(graph, {.input = buffers.finalHidden, .output = buffers.logits, .scratch = scratch, .prepared = finalHidden},
             head_, headPlan);
  // Each lane's Markov chain, from its f_rnn, into the selector's lattice,
  // whose candidates a sampled verify row tests the chain's draws as: the
  // target's probabilities of those tokens against the chain's.
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!buffers.laneAuxiliary[lane]) throw std::invalid_argument("a DSpark lane has no f_rnn row");
    ops::RowCopy::add(graph, buffers.laneAuxiliary[lane], {0, hidden, 0}, weights_.frnnRows, {lane, hidden, 0}, 1,
                      hidden);
  }
  ops::DSpark::addMarkov(graph, weights_.markov,
                         {.hidden = buffers.finalHidden,
                          .base = buffers.logits,
                          .frnn = weights_.frnnRows,
                          .joint = weights_.markovJoint,
                          .state = weights_.markovState,
                          .previous = weights_.markovPrevious,
                          .logits = weights_.markovLogits,
                          .partialValues = weights_.markovPartialValues,
                          .partialIds = weights_.markovPartialIds,
                          .candidates = buffers.candidates,
                          .unary = buffers.unary},
                         {hidden, layout_.vocabularySize, layout_.proposedVocabulary, layout_.logitCap}, step.lanes);
}

void DSparkDraft::addSelection(metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
                               const DFlashDraftStep &step) const {
  selector_.addChain(graph, buffers, step.lanes);
}

void DSparkDraft::addContextCommit(metal::CommandGraph &graph, DFlashContextBuffers buffers,
                                   std::span<const uint32_t> startPositions, const ops::StepRows &step) const {
  const auto lanes = static_cast<uint32_t>(startPositions.size());
  if (!lanes || lanes > kLanes || step.lanes != lanes || buffers.persistentKeys.size() != layout_.layers ||
      buffers.persistentValues.size() != layout_.layers)
    throw std::invalid_argument("invalid draft context batch");
  ops::requireStepRows(step);
  const uint32_t hidden = layout_.hiddenSize, rows = step.rows;
  const auto contextRows = [&](uint32_t layer) { return layerRows(backend_, buffers.hidden, layout_, layer); };
  // The context projections run over the step's rows, on the draft's plans
  // of a decode step.
  const ops::Linear &linear = operators_.draftDecodeLinear();
  for (uint32_t layer = 0; layer < layout_.layers; ++layer) {
    const DSparkLayerWeights &w = weights_.layers[layer];
    ops::DSpark::addMix(graph, buffers.capturedTargetHidden, w.context, contextRows(layer), rows, hidden);
    // A lane holds at most the cap's rows and the step's.
    for (uint32_t lane = 0; lane < lanes; ++lane)
      if (buffers.windows[lane])
        ops::DraftAttention::addWindowCommit(
            graph, contextRows(layer), layerWindow(buffers.windows[lane], layer),
            backend_.view(buffers.retainedRows, lane * sizeof(uint32_t), sizeof(uint32_t)),
            backend_.view(buffers.laneRows, lane * sizeof(ops::LaneRows), sizeof(ops::LaneRows)), rows,
            std::min(step.laneCap, rows), startPositions[lane], hidden);
    linear.add(graph, {.input = contextRows(layer), .output = buffers.contextKv, .scratch = buffers.linearScratch},
               w.contextKv, linear.plan(w.contextKv, rows));
    ops::DraftAttention::addContextCommit(graph, buffers.contextKv, w.contextKeyNorm, buffers.ropeCos, buffers.ropeSin,
                                          buffers.persistentKeys[layer], buffers.persistentValues[layer],
                                          buffers.retainedRows, buffers.laneRows, startPositions, step,
                                          layout_.attentionShape());
  }
  // Each lane's last retained row's f_rnn, its next draft's context.
  ops::DSpark::addMix(graph, buffers.capturedTargetHidden, weights_.headContext, buffers.projected, rows, hidden);
  ops::DSpark::addFrnnCommit(graph, buffers.projected, buffers.laneAuxiliary, buffers.retainedRows, buffers.laneRows,
                             lanes, hidden);
}

void DSparkDraft::addSetup(metal::CommandGraph &graph, ops::LinearScratch scratch, metal::MetalBuffer rows,
                           uint32_t stepRows) const {
  if (!stepRows || stepRows > kFoldBlock) throw std::invalid_argument("invalid DSpark fold block");
  const ops::Linear &target = operators_.decodeLinear();
  const uint32_t total = kRank + 2;
  const uint64_t rowBytes = uint64_t{layout_.hiddenSize} * kBFloat16Bytes;
  for (uint32_t first = 0; first < total; first += stepRows) {
    const uint32_t count = std::min(stepRows, total - first);
    target.add(graph,
               {.input = backend_.view(weights_.foldInput, first * rowBytes, kFoldBlock * rowBytes),
                .output = rows,
                .scratch = scratch},
               head_, target.plan(head_, count));
    ops::DSpark::addFold(graph, rows, weights_.markov, first, count, layout_.vocabularySize);
  }
}

std::unique_ptr<DraftModel> makeDraftModel(const DFlashDraftWeights &weights, metal::MetalBackend &backend,
                                           const ops::ExecutionPlans &operators, const ops::Projection &head,
                                           ops::DraftVocabulary subset, const ops::NormWeights *embeddingNorm) {
  if (weights.layout.kind != DraftKind::DSpark)
    return std::make_unique<DFlashDraft>(weights, backend, operators, head, subset);
  if (!weights.dspark || !embeddingNorm)
    throw std::invalid_argument("a DSpark draft takes its weights and its target's embedding norm");
  return std::make_unique<DSparkDraft>(weights, backend, operators, head, *embeddingNorm);
}

} // namespace splash::model
