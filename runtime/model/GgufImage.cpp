#include "model/GgufImage.hpp"

#include "metal/abi/Gguf.h"
#include "model/GgufImageLayout.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>

namespace splash::model::gguf {
namespace {

static_assert([] {
  for (const QuantFormat &format : kQuantFormats) {
    const GgmlTypeTraits *type = ggmlTypeTraits(format.ggml_type);
    if (!type || type->blockElements != format.block_elements || type->blockBytes != format.block_bytes)
      return false;
  }
  return true;
}(), "format table types are the GGUF types, block for block");
// The repack and the reference find meta unit u in native block u.
static_assert([] {
  for (const QuantFormat &format : kQuantFormats)
    if (format.meta_groups * 32 != format.block_elements) return false;
  return true;
}(), "a meta unit is one native block");
static_assert(GGUF_TYPE_F32 == ggml::kF32, "float segments carry the GGUF type id");

bool quantizedType(uint32_t type) { return gguf_format_of(type) != GGUF_FMT_COUNT; }
bool floatType(uint32_t type) { return type == ggml::kF32; }
// The token rows the embedding kernel gathers.
bool embeddingType(uint32_t type) { return gguf_embedding_format(gguf_format_of(type)); }
// alpha/beta run as one segment of their shared type: any format of the
// projections (one repacked tensor) or F32 (one float tensor), which BF16
// becomes exactly.
bool alphaBetaType(uint32_t type) { return quantizedType(type) || type == ggml::kF32 || type == ggml::kBF16; }

// Plans one image. A missing tensor or one of a type this build cannot load
// is added to `problems` and left out of the image, so the planner can name
// every such tensor at once; a tensor of the wrong shape throws.
class Builder {
public:
  Builder(const GgufFile &file, const TargetGeometry &geometry, std::vector<std::string> &problems,
          std::string name, uint32_t layer, uint32_t type)
      : file_(file), geometry_(geometry), problems_(problems) {
    image_.name = std::move(name);
    image_.layer = layer;
    image_.type = type;
    const auto header = weightFileHeader(kGgufImageMagic, layer, type);
    image_.fills.push_back({0, {header.begin(), header.end()}});
    cursor_ = header.size();
  }

  // A norm as stored: F32, which the norm kernels read unrounded, as
  // llama.cpp does (ops::NormWeights).
  void floatNorm(const std::string &name, uint64_t elements) {
    if (const GgufTensor *tensor = floatVector(name, elements)) copy(tensorRows(*tensor, 1, tensor->bytes));
  }

  // Quantized rows [N, K] repacked into planes, rows in `order`.
  void quantized(const std::string &name, uint64_t rows, uint64_t columns, RowOrder order = {}) {
    const GgufTensor *tensor = find(name, quantizedType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns) throw GgufError("unexpected shape for " + name);
    const uint32_t format = gguf_format_of(tensor->type);
    Repack repack = planes(format, rows, columns, name);
    repack.sources.push_back(tensorRows(*tensor, rows, ggufRowBytes(kQuantFormats[format], columns), order));
    image_.repacks.push_back(std::move(repack));
  }

  // beta (value heads rows) | alpha (value heads rows), rows in grouped head
  // order: one 256-row tensor of their format padded with zero rows, or F32 as
  // one float tensor, which BF16 is widened to exactly.
  void alphaBeta(const std::string &betaName, const std::string &alphaName) {
    const GgufTensor *beta = file_.find(betaName), *alpha = file_.find(alphaName);
    if (!beta || !alpha || beta->type != alpha->type || !alphaBetaType(beta->type)) {
      reject(betaName, beta);
      reject(alphaName, alpha);
      return;
    }
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    for (const GgufTensor *t : {beta, alpha})
      if (t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be [" + std::to_string(heads) + ", hidden]: " + t->name);
    if (beta->type == ggml::kF32 || beta->type == ggml::kBF16) {
      const uint64_t widening = beta->type == ggml::kBF16 ? 2 : 1;
      const uint64_t bytes = (beta->bytes + alpha->bytes) * widening;
      descriptor(ggml::kF32, 2ull * heads, hidden, {}, {bytes, 0, 0}, betaName);
      uint64_t destination = section(bytes);
      for (const GgufTensor *t : {beta, alpha}) {
        image_.copies.push_back(
            {destination, tensorRows(*t, heads, t->bytes / heads, grouped(0, 1)), false, widening == 2, false});
        destination += t->bytes * widening;
      }
      return;
    }
    const uint32_t format = gguf_format_of(beta->type);
    const uint64_t rowBytes = ggufRowBytes(kQuantFormats[format], hidden);
    // The 2 heads rows padded with zero rows to whole plane tiles.
    const uint64_t tiled = (2ull * heads + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
    Repack repack = planes(format, tiled, hidden, alphaName);
    for (const GgufTensor *t : {beta, alpha}) repack.sources.push_back(tensorRows(*t, heads, rowBytes, grouped(0, 1)));
    image_.repacks.push_back(std::move(repack));
  }

  // The convolution taps of every channel, q and k channels as stored, then
  // value channels in grouped head order, as the exact bf16 values the GDN
  // kernels read.
  void convolution(const std::string &name, uint64_t keyRows) {
    const uint32_t channels = geometry_.convolutionDimension;
    if (const GgufTensor *tensor = floatVector(name, uint64_t{channels} * kGdnConvolutionTaps))
      copy(tensorRows(*tensor, channels, tensor->bytes / channels, grouped(keyRows, geometry_.gdnHeadDimension)), true);
  }

  // A per value head F32 vector in grouped head order: as stored, or as the
  // exact bf16 values the kernels read.
  void headVector(const std::string &name, bool bfloat16) {
    const uint32_t heads = geometry_.gdnValueHeads;
    if (const GgufTensor *tensor = floatVector(name, heads))
      copy(tensorRows(*tensor, heads, tensor->bytes / heads, grouped(0, 1)), bfloat16);
  }

  // Native token rows, gathered by the embedding kernel.
  void embeddingRows(const std::string &name) {
    const GgufTensor *tensor = find(name, embeddingType);
    if (!tensor) return;
    if (tensor->rows() != geometry_.vocabularySize || tensor->columns() != geometry_.hiddenSize)
      throw GgufError("unexpected shape for " + name);
    copiedRows(*tensor);
  }

  // An F32 tensor [rows, columns] as stored (the MoE router and the
  // shared-expert scalar gate, which llama.cpp keeps unquantized).
  void floatTensor(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor *tensor = find(name, floatType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected an F32 [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + name);
    copiedRows(*tensor);
  }

  // An F32 or BF16 tensor [rows, columns] as F32 rows: as stored, or the F32
  // values BF16 widens to exactly (the QSA indexer's projections and the
  // hyper-connections' injection weights). With `descriptor` it is a float
  // tensor a projection runs, else its rows alone.
  // A Q8_0 tensor (the MTP head's injection weights) becomes the F32
  // values it equals.
  void floatRows(const std::string &name, uint64_t rows, uint64_t columns, bool withDescriptor) {
    const GgufTensor *tensor = find(name, [](uint32_t type) {
      return type == ggml::kF32 || type == ggml::kBF16 || type == ggml::kQ8_0;
    });
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected an F32, BF16 or Q8_0 [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + name);
    const bool widen = tensor->type == ggml::kBF16, dequantize = tensor->type == ggml::kQ8_0;
    const uint64_t bytes = rows * columns * sizeof(float);
    if (withDescriptor) descriptor(ggml::kF32, rows, columns, {}, {bytes, 0, 0}, name);
    image_.copies.push_back({section(bytes), tensorRows(*tensor, rows, tensor->bytes / rows), false, widen, dequantize});
  }

  // A table's native rows [rows, columns] in an embedding format, after their
  // descriptor (the PLE n-gram table, which no projection reads).
  void nativeTable(const std::string &name, uint64_t columns) {
    const GgufTensor *tensor = find(name, embeddingType);
    if (!tensor) return;
    if (tensor->columns() != columns) throw GgufError("unexpected shape for " + name);
    copiedRows(*tensor);
  }

  // Value heads of headRows rows from row `from` on, in grouped order.
  RowOrder grouped(uint64_t from, uint32_t headRows) const {
    return {from, headRows, geometry_.gdnKeyHeads, geometry_.gdnValueHeads / geometry_.gdnKeyHeads};
  }

  Image finish() {
    image_.bytes = alignWeightOffset(cursor_);
    return std::move(image_);
  }

private:
  void reject(const std::string &name, const GgufTensor *tensor) {
    problems_.push_back(name + (tensor ? " (" + ggmlTypeName(tensor->type) + ")" : " (missing)"));
  }

  // The tensor `name` when it is present and of an accepted type.
  const GgufTensor *find(const std::string &name, bool (*accepted)(uint32_t type)) {
    const GgufTensor *tensor = file_.find(name);
    if (tensor && accepted(tensor->type)) return tensor;
    reject(name, tensor);
    return nullptr;
  }

  const GgufTensor *floatVector(const std::string &name, uint64_t elements) {
    const GgufTensor *tensor = find(name, floatType);
    if (tensor && tensor->elements() != elements) throw GgufError("unexpected shape for " + name);
    return tensor;
  }

  uint64_t section(uint64_t bytes) {
    if (!bytes) throw GgufError("empty image section in " + image_.name);
    const uint64_t start = alignWeightOffset(cursor_);
    cursor_ = start + bytes;
    return start;
  }

  static TensorRows tensorRows(const GgufTensor &tensor, uint64_t count, uint64_t rowBytes, RowOrder order = {}) {
    if (!count || count * rowBytes != tensor.bytes) throw GgufError("unexpected size for " + tensor.name);
    return {tensor.name, tensor.type, tensor.offset, count, rowBytes, order, tensor.file};
  }

  // Rows written as stored, or converted to bf16, into their own section.
  void copy(TensorRows source, bool bfloat16 = false) {
    const uint64_t bytes = source.rows * source.rowBytes / (bfloat16 ? 2 : 1);
    image_.copies.push_back({section(bytes), std::move(source), bfloat16, false, false});
  }

  // A tensor's rows as stored, after their descriptor.
  void copiedRows(const GgufTensor &tensor) {
    descriptor(tensor.type, tensor.rows(), tensor.columns(), {}, {tensor.bytes, 0, 0}, tensor.name);
    copy(tensorRows(tensor, tensor.rows(), tensor.bytes / tensor.rows()));
  }

  // The descriptor and planes of a [rows, columns] quantized tensor.
  Repack planes(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name) {
    const QuantFormat &layout = kQuantFormats[format];
    if (rows % QUANT_TILE_ROWS || columns % ggufColumnUnit(layout)) throw GgufError("tensor is not tile aligned: " + name);
    const GgufPlaneBytes bytes = ggufPlaneBytes(layout, rows, columns);
    descriptor(layout.ggml_type, rows, columns, layout, bytes, name);
    Repack repack;
    repack.format = format;
    repack.rows = rows;
    repack.columns = columns;
    repack.plane0 = section(bytes.plane0);
    repack.plane1 = bytes.plane1 ? section(bytes.plane1) : 0;
    repack.meta = section(bytes.meta);
    return repack;
  }

  // The descriptor of a tensor of `type`, quantized in `format` (a float
  // tensor has neither per-group nor meta bytes).
  void descriptor(uint32_t type, uint64_t rows, uint64_t columns, const QuantFormat &format,
                  const GgufPlaneBytes &bytes, const std::string &name) {
    if (rows > std::numeric_limits<uint32_t>::max() || columns > std::numeric_limits<uint32_t>::max())
      throw GgufError("tensor is too large for its descriptor: " + name);
    GgufTensorDescriptor d{};
    d.type = type;
    d.outputSize = static_cast<uint32_t>(rows);
    d.inputSize = static_cast<uint32_t>(columns);
    d.p0 = format.plane0_bytes;
    d.p1 = format.plane1_bytes;
    d.metaBytes = format.meta_bytes;
    d.metaGroups = format.meta_groups;
    d.plane0Bytes = bytes.plane0;
    d.plane1Bytes = bytes.plane1;
    d.metaTotalBytes = bytes.meta;
    std::vector<uint8_t> encoded(sizeof d);
    std::memcpy(encoded.data(), &d, sizeof d);
    image_.fills.push_back({section(encoded.size()), std::move(encoded)});
  }

  const GgufFile &file_;
  const TargetGeometry &geometry_;
  std::vector<std::string> &problems_;
  Image image_;
  uint64_t cursor_ = 0;
};

std::string prefix(uint32_t layer) { return "blk." + std::to_string(layer) + "."; }

// The target geometry the metadata declares, with the rotary embedding and
// norms the kernels compute: the RoPE base and rotated dimensions, the RMS
// epsilon and no RoPE scaling. One error names every mismatch.
void requireMetadata(const GgufFile &file, const TargetGeometry &geometry) {
  const std::string arch = geometry.architecture();
  if (file.architecture() != arch)
    throw GgufError("GGUF architecture is " + file.architecture() + ", but the package's target is " + arch);
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  // A float equal to value up to its F32 rounding.
  const auto expectFloat = [&](const char *key, double value) {
    const std::optional<double> found = file.floatValue(arch + "." + key);
    if (found && std::abs(*found - value) <= value * 1e-6) return;
    std::ostringstream text;
    text << (mismatched.empty() ? "" : ", ") << key << ' ';
    if (found) text << *found;
    else text << "missing";
    text << " (expected " << value << ')';
    mismatched += text.str();
  };
  expect("block_count", geometry.layers + file.unsignedValue(arch + ".nextn_predict_layers").value_or(0));
  expect("embedding_length", geometry.hiddenSize);
  expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  expect("attention.head_count_kv", geometry.attentionKvHeads);
  expect("attention.key_length", geometry.attentionHeadDimension);
  expect("attention.value_length", geometry.attentionHeadDimension);
  expect("rope.dimension_count", 2ull * geometry.rotaryPairs);
  expectFloat("rope.freq_base", geometry.rotaryTheta);
  expectFloat("attention.layer_norm_rms_epsilon", 1e-6);
  if (const auto scaling = file.stringValue(arch + ".rope.scaling.type"); scaling && *scaling != "none")
    mismatched += (mismatched.empty() ? "" : ", ") + std::string("rope.scaling.type ") + *scaling + " (expected none)";
  expect("full_attention_interval", geometry.fullAttentionPeriod);
  expect("ssm.conv_kernel", kGdnConvolutionTaps);
  expect("ssm.group_count", geometry.gdnKeyHeads);
  expect("ssm.time_step_rank", geometry.gdnValueHeads);
  expect("ssm.state_size", geometry.gdnHeadDimension);
  expect("ssm.inner_size", uint64_t{geometry.gdnValueHeads} * geometry.gdnHeadDimension);
  if (geometry.qwen4()) {
    expect("hyper_connection.count", geometry.hyperConnections);
    expect("hyper_connection.low_rank", geometry.hyperRank);
    expect("attention.indexer.head_count", geometry.indexerHeads);
    expect("attention.indexer.key_length", geometry.indexerHeadDimension);
    expect("attention.indexer.top_k", geometry.indexerTokens);
    const auto ratios = file.numericArray(arch + ".attention.compress_ratios");
    bool ratiosMatch = ratios && ratios->size() >= geometry.layers;
    for (uint32_t layer = 0; ratiosMatch && layer < geometry.layers; ++layer)
      ratiosMatch = (*ratios)[layer] == (geometry.isFullAttentionLayer(layer) ? geometry.indexerBlockTokens : 0);
    if (!ratiosMatch) mismatched += (mismatched.empty() ? "" : ", ") + std::string("attention.compress_ratios");
    if (geometry.pleLayer < geometry.layers) {
      const auto layers = file.numericArray(arch + ".ple.layers");
      if (!layers || layers->size() != 1 || (*layers)[0] != geometry.pleLayer)
        mismatched += (mismatched.empty() ? "" : ", ") + std::string("ple.layers");
      expect("ple.ngram_size", geometry.pleNgram);
      expect("ple.heads_per_ngram", geometry.pleHeadsPerNgram);
      expect("ple.conv_kernel", geometry.pleConvolutionTaps);
      expect("embedding_length_per_layer_input", geometry.pleHeadDimension);
    }
  }
  if (geometry.sparseMoe()) {
    expect("expert_count", geometry.experts);
    expect("expert_used_count", geometry.expertsPerToken);
    expect("expert_feed_forward_length", geometry.expertIntermediateSize);
    expect("expert_shared_feed_forward_length", geometry.expertIntermediateSize);
  } else {
    expect("feed_forward_length", geometry.intermediateSize);
  }
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
}

// One hyper-connection mix (model/Qwen4Exp.hpp Qwen4HyperWeights): its norm,
// down and up projections and, when it injects, its injection weights.
void hyperImage(Builder &b, const TargetGeometry &g, const std::string &stem, bool inject) {
  const uint64_t streams = uint64_t{g.hyperConnections} * g.hiddenSize;
  b.floatNorm(stem + "_norm.weight", streams);
  b.quantized(stem + "_down.weight", g.hyperRank, streams);
  b.quantized(stem + "_up.weight", streams, g.hyperRank);
  if (inject) b.floatRows(stem + "_inject.weight", g.hyperConnections, streams, false);
}

void mixerImage(Builder &b, const TargetGeometry &g, const std::string &p, bool full);
void ffnImage(Builder &b, const TargetGeometry &g, const std::string &p);

Image qwen4LayerImage(const GgufFile &file, const TargetGeometry &g, std::vector<std::string> &problems,
                      uint32_t index) {
  const std::string p = prefix(index);
  const bool full = g.isFullAttentionLayer(index);
  Builder b(file, g, problems, "layer-" + std::to_string(index) + ".bin", index, full ? 1u : 0u);
  hyperImage(b, g, p + "hc_attn", true);
  mixerImage(b, g, p, full);
  if (full) {
    b.floatRows(p + "indexer.q_proj.weight", uint64_t{g.indexerHeads} * g.indexerHeadDimension, g.hiddenSize, true);
    b.floatRows(p + "indexer.k_proj.weight", g.indexerHeadDimension, g.hiddenSize, true);
    b.floatNorm(p + "indexer.q_norm.weight", g.indexerHeadDimension);
    b.floatNorm(p + "indexer.k_norm.weight", g.indexerHeadDimension);
  }
  hyperImage(b, g, p + "hc_ffn", true);
  ffnImage(b, g, p);
  if (index == g.pleLayer) {
    const uint64_t streams = uint64_t{g.hyperConnections} * g.hiddenSize;
    const uint64_t width = uint64_t{g.pleNgram - 1} * g.pleHeadsPerNgram * g.pleHeadDimension;
    b.quantized(p + "ple_key.weight", streams, width);
    b.quantized(p + "ple_value.weight", g.hiddenSize, width);
    b.floatNorm(p + "ple_norm_key.weight", streams);
    b.floatNorm(p + "ple_norm_query.weight", streams);
    b.floatNorm(p + "ple_norm_conv.weight", streams);
    b.floatNorm(p + "ple_conv1d.weight", streams * g.pleConvolutionTaps);
  }
  return b.finish();
}

void mixerImage(Builder &b, const TargetGeometry &g, const std::string &p, bool full) {
  if (full) {
    b.quantized(p + "attn_q.weight", 2ull * g.attentionHeadDimension * (g.attentionWidth / g.attentionHeadDimension),
                g.hiddenSize);
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.quantized(p + "attn_k.weight", kvRows, g.hiddenSize);
    b.quantized(p + "attn_v.weight", kvRows, g.hiddenSize);
    b.floatNorm(p + "attn_q_norm.weight", g.attentionHeadDimension);
    b.floatNorm(p + "attn_k_norm.weight", g.attentionHeadDimension);
    b.quantized(p + "attn_output.weight", g.hiddenSize, g.attentionWidth);
  } else {
    const uint32_t valueRows = g.gdnValueHeads * g.gdnHeadDimension;
    const uint32_t keyRows = g.convolutionDimension - valueRows; // q and k
    b.quantized(p + "attn_qkv.weight", g.convolutionDimension, g.hiddenSize, b.grouped(keyRows, g.gdnHeadDimension));
    b.quantized(p + "attn_gate.weight", valueRows, g.hiddenSize, b.grouped(0, g.gdnHeadDimension));
    b.alphaBeta(p + "ssm_beta.weight", p + "ssm_alpha.weight");
    b.convolution(p + "ssm_conv1d.weight", keyRows);
    b.headVector(p + "ssm_a", false);
    b.headVector(p + "ssm_dt.bias", true);
    b.floatNorm(p + "ssm_norm.weight", g.gdnHeadDimension);
    b.quantized(p + "ssm_out.weight", g.hiddenSize, valueRows);
  }
}

void ffnImage(Builder &b, const TargetGeometry &g, const std::string &p) {
  if (g.sparseMoe()) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(p + "ffn_gate_inp.weight", routed, g.hiddenSize);
    b.quantized(p + "ffn_gate_exps.weight", routed * width, g.hiddenSize);
    b.quantized(p + "ffn_up_exps.weight", routed * width, g.hiddenSize);
    b.quantized(p + "ffn_down_exps.weight", routed * g.hiddenSize, width);
    b.quantized(p + "ffn_gate_shexp.weight", width, g.hiddenSize);
    b.quantized(p + "ffn_up_shexp.weight", width, g.hiddenSize);
    b.quantized(p + "ffn_down_shexp.weight", g.hiddenSize, width);
    b.floatTensor(p + "ffn_gate_inp_shexp.weight", 1, g.hiddenSize);
  } else {
    b.quantized(p + "ffn_gate.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_up.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_down.weight", g.hiddenSize, g.intermediateSize);
  }
}

Image layerImage(const GgufFile &file, const TargetGeometry &g, std::vector<std::string> &problems,
                 uint32_t index) {
  if (g.qwen4()) return qwen4LayerImage(file, g, problems, index);
  const std::string p = prefix(index);
  const bool full = g.isFullAttentionLayer(index);
  Builder b(file, g, problems, "layer-" + std::to_string(index) + ".bin", index, full ? 1u : 0u);
  b.floatNorm(p + "attn_norm.weight", g.hiddenSize);
  if (full) {
    b.quantized(p + "attn_q.weight", 2ull * g.attentionHeadDimension * (g.attentionWidth / g.attentionHeadDimension),
                g.hiddenSize);
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.quantized(p + "attn_k.weight", kvRows, g.hiddenSize);
    b.quantized(p + "attn_v.weight", kvRows, g.hiddenSize);
    b.floatNorm(p + "attn_q_norm.weight", g.attentionHeadDimension);
    b.floatNorm(p + "attn_k_norm.weight", g.attentionHeadDimension);
    b.quantized(p + "attn_output.weight", g.hiddenSize, g.attentionWidth);
  } else {
    const uint32_t valueRows = g.gdnValueHeads * g.gdnHeadDimension;
    const uint32_t keyRows = g.convolutionDimension - valueRows; // q and k
    b.quantized(p + "attn_qkv.weight", g.convolutionDimension, g.hiddenSize, b.grouped(keyRows, g.gdnHeadDimension));
    b.quantized(p + "attn_gate.weight", valueRows, g.hiddenSize, b.grouped(0, g.gdnHeadDimension));
    b.alphaBeta(p + "ssm_beta.weight", p + "ssm_alpha.weight");
    b.convolution(p + "ssm_conv1d.weight", keyRows);
    b.headVector(p + "ssm_a", false);
    b.headVector(p + "ssm_dt.bias", true);
    b.floatNorm(p + "ssm_norm.weight", g.gdnHeadDimension);
    b.quantized(p + "ssm_out.weight", g.hiddenSize, valueRows);
  }
  b.floatNorm(p + "post_attention_norm.weight", g.hiddenSize);
  if (g.sparseMoe()) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(p + "ffn_gate_inp.weight", routed, g.hiddenSize);
    b.quantized(p + "ffn_gate_exps.weight", routed * width, g.hiddenSize);
    b.quantized(p + "ffn_up_exps.weight", routed * width, g.hiddenSize);
    b.quantized(p + "ffn_down_exps.weight", routed * g.hiddenSize, width);
    b.quantized(p + "ffn_gate_shexp.weight", width, g.hiddenSize);
    b.quantized(p + "ffn_up_shexp.weight", width, g.hiddenSize);
    b.quantized(p + "ffn_down_shexp.weight", g.hiddenSize, width);
    b.floatTensor(p + "ffn_gate_inp_shexp.weight", 1, g.hiddenSize);
  } else {
    b.quantized(p + "ffn_gate.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_up.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_down.weight", g.hiddenSize, g.intermediateSize);
  }
  return b.finish();
}

// A rotated GGUF (GgufRotation) must name exactly what the loader rotates:
// every tensor the images of a dense target repack, whose segments read
// H (D x) (ops::InputRotation) while float segments (F32 or BF16 alpha/beta)
// read x as it is, and the token table, which the rotated gather decodes from
// PQ2_0 rows, with the GDN value heads of the rotated inputs grouped.
void requireRotation(const GgufFile &file, const TargetGeometry &g, const std::vector<Image> &images) {
  const GgufRotation &rotation = *file.rotation();
  if (g.sparseMoe()) throw GgufError("rotated weights are supported for dense targets only");
  if (!rotation.valueHeadsGrouped)
    throw GgufError("rotated GDN inputs must keep their value heads grouped (prism.hadamard.gdn_v_grouped)");
  std::set<std::string, std::less<>> repacked;
  for (const Image &image : images)
    for (const Repack &repack : image.repacks)
      for (const TensorRows &source : repack.sources) repacked.insert(source.name);
  if (rotation.weights != repacked)
    throw GgufError("the rotation must name every quantized tensor of the target and nothing else");
  if (rotation.tables != std::set<std::string, std::less<>>{"token_embd.weight"} ||
      file.require("token_embd.weight").type != ggml::kPQ2_0)
    throw GgufError("the rotation's one token table must be token_embd.weight in PQ2_0");
}

} // namespace

std::vector<Image> planImages(const GgufFile &file, const TargetGeometry &geometry) {
  requireMetadata(file, geometry);
  std::vector<std::string> problems;
  std::vector<Image> images;
  for (uint32_t layer = 0; layer < geometry.layers; ++layer)
    images.push_back(layerImage(file, geometry, problems, layer));
  Builder head(file, geometry, problems, "head.bin", geometry.layers, 2);
  if (geometry.qwen4()) hyperImage(head, geometry, "output_hc", false);
  else head.floatNorm("output_norm.weight", geometry.hiddenSize);
  head.quantized("output.weight", geometry.vocabularySize, geometry.hiddenSize);
  images.push_back(head.finish());
  Builder embedding(file, geometry, problems, "embedding.bin", geometry.vocabularySize, geometry.hiddenSize);
  embedding.embeddingRows("token_embd.weight");
  images.push_back(embedding.finish());
  if (geometry.qwen4() && geometry.pleLayer < geometry.layers) {
    Builder ple(file, geometry, problems, "ple.bin", geometry.pleLayer, 3);
    ple.nativeTable("per_layer_token_embd.weight", geometry.pleHeadDimension);
    images.push_back(ple.finish());
  }
  if (!problems.empty()) {
    std::string names;
    for (const std::string &problem : problems) names += (names.empty() ? "" : ", ") + problem;
    throw GgufError("GGUF tensors this build cannot load: " + names);
  }
  if (file.rotation()) requireRotation(file, geometry, images);
  return images;
}

} // namespace splash::model::gguf

namespace splash::model::gguf {

Image planMtpImage(const GgufFile &file, const TargetGeometry &target) {
  TargetGeometry g = target;
  g.pleLayer = g.layers;  // the MTP file carries no PLE metadata
  requireMetadata(file, g);
  if (file.unsignedValue(g.architecture() + std::string(".nextn_predict_layers")) != 1)
    throw GgufError("the MTP GGUF must hold one nextn layer");
  std::vector<std::string> problems;
  const std::string p = prefix(g.layers);
  Builder b(file, g, problems, "mtp.bin", g.layers, 4);
  hyperImage(b, g, p + "hc_attn", true);
  mixerImage(b, g, p, true);
  b.floatRows(p + "indexer.q_proj.weight", uint64_t{g.indexerHeads} * g.indexerHeadDimension, g.hiddenSize, true);
  b.floatRows(p + "indexer.k_proj.weight", g.indexerHeadDimension, g.hiddenSize, true);
  b.floatNorm(p + "indexer.q_norm.weight", g.indexerHeadDimension);
  b.floatNorm(p + "indexer.k_norm.weight", g.indexerHeadDimension);
  hyperImage(b, g, p + "hc_ffn", true);
  ffnImage(b, g, p);
  b.quantized(p + "nextn.eh_proj.weight", g.hiddenSize, 2ull * g.hiddenSize);
  b.floatNorm(p + "nextn.enorm.weight", g.hiddenSize);
  b.floatNorm(p + "nextn.hnorm.weight", uint64_t{g.hyperConnections} * g.hiddenSize);
  hyperImage(b, g, p + "nextn.hc_head", false);
  if (!problems.empty()) {
    std::string names;
    for (const std::string &problem : problems) names += (names.empty() ? "" : ", ") + problem;
    throw GgufError("MTP GGUF tensors this build cannot load: " + names);
  }
  return b.finish();
}

PleHash readPleHash(const GgufFile &file, const TargetGeometry &g) {
  PleHash hash;
  if (!g.qwen4() || g.pleLayer >= g.layers) return hash;
  const std::string arch = g.architecture();
  const uint32_t heads = (g.pleNgram - 1) * g.pleHeadsPerNgram;
  const auto array = [&](const char *key, size_t count) {
    const auto values = file.numericArray(arch + ".ple." + key);
    if (!values || values->size() < count) throw GgufError("GGUF PLE metadata is missing: ple." + std::string(key));
    return *values;
  };
  // Every value is an integer below 2^53, which the parser's doubles hold exactly.
  const auto exact = [](double value, double limit) {
    if (!(value >= 0 && value <= limit) || value != std::floor(value))
      throw GgufError("GGUF PLE metadata holds an invalid value");
    return value;
  };
  for (double m : array("layer_multipliers", g.pleNgram))
    hash.multipliers.push_back(static_cast<uint64_t>(exact(m, 9007199254740992.0)));
  hash.multipliers.resize(g.pleNgram);
  const auto offsets = array("head_offsets", heads), sizes = array("head_vocab_sizes", heads);
  const GgufTensor &table = file.require("per_layer_token_embd.weight");
  hash.tableRows = table.rows();
  for (uint32_t h = 0; h < heads; ++h) {
    hash.headOffsets.push_back(static_cast<uint32_t>(exact(offsets[h], 2147483647.0)));
    hash.headVocabularies.push_back(static_cast<uint32_t>(exact(sizes[h], 2147483647.0)));
    if (!hash.headVocabularies.back() ||
        uint64_t{hash.headOffsets.back()} + hash.headVocabularies.back() > hash.tableRows)
      throw GgufError("a PLE head's rows lie outside per_layer_token_embd");
  }
  hash.eosToken = static_cast<uint32_t>(file.unsignedValue(arch + ".ple.eos_token_id").value_or(0));
  if (!hash.eosToken) throw GgufError("GGUF PLE metadata is missing: ple.eos_token_id");
  return hash;
}

} // namespace splash::model::gguf
