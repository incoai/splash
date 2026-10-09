#include "model/MlxImage.hpp"

#include "metal/abi/Gguf.h"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <string_view>

namespace splash::model::mlx {
namespace {

using gguf::Conversion;
using gguf::Repack;
using gguf::TensorRows;

// Plans one image from the checkpoint's tensors, which must have the dtypes
// and shapes its target reads.
class Builder {
public:
  Builder(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry, std::string name,
          uint32_t layer, uint32_t type)
      : checkpoint_(checkpoint), geometry_(geometry), image_(std::move(name), layer, type) {}

  // A norm as stored: bf16, which the norm kernels read (ops::NormWeights).
  void norm(const std::string &name, uint64_t elements) { stored(name + ".weight", {"BF16"}, {elements}); }

  // The convolution taps of every channel, as stored: q and k channels, then
  // the value channels grouped by key head.
  void convolution(const std::string &name) {
    stored(name + ".weight", {"BF16"}, {geometry_.convolutionDimension, kGdnConvolutionTaps, 1});
  }

  // The bf16 GDN time-step bias of every value head, as stored.
  void timeBias(const std::string &name) { stored(name, {"BF16"}, {geometry_.gdnValueHeads}); }

  // The F32 GDN decay -exp(A_log) of every value head.
  void decay(const std::string &name) {
    const SourceTensor &tensor = require(name, {"BF16", "F32"}, {geometry_.gdnValueHeads});
    image_.copy(rows(name, tensor, tensor.dtype == "BF16" ? ggml::kBF16 : ggml::kF32, 1, tensor.bytes),
                Conversion::Decay);
  }

  // A quantized [rows, columns] tensor, its planes (or experts of them,
  // stacked as one tensor of experts * rows rows).
  void projection(const std::string &module, uint64_t rows, uint64_t columns, uint64_t experts = 1) {
    const std::optional<TensorRows> source = quantized(module, rows, columns, experts);
    if (!source) throw WeightStoreError(module + " is unquantized; Splash loads quantized MLX projections");
    Repack repack = image_.planes(gguf_format_of(source->type), experts * rows, columns, module);
    repack.sources.push_back(*source);
    image_.repack(std::move(repack));
  }

  // beta (value heads rows) | alpha (value heads rows): one 256-row tensor
  // of their shared format padded with zero rows, or else one F32 tensor.
  void alphaBeta(const std::string &beta, const std::string &alpha) {
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    const std::optional<TensorRows> b = quantized(beta, heads, hidden), a = quantized(alpha, heads, hidden);
    if (b && a && b->type == a->type) {
      if (2 * heads > QUANT_TILE_ROWS) throw WeightStoreError("alpha/beta rows exceed one 256-row tile");
      Repack repack = image_.planes(gguf_format_of(b->type), QUANT_TILE_ROWS, hidden, alpha);
      repack.sources = {*b, *a};
      image_.repack(std::move(repack));
      return;
    }
    const uint64_t rowBytes = uint64_t{hidden} * sizeof(float);
    image_.descriptor(ggml::kF32, 2ull * heads, hidden, {}, {2 * heads * rowBytes, 0, 0}, beta);
    const uint64_t destination = image_.section(2 * heads * rowBytes);
    floatRows(beta, heads, hidden, destination);
    floatRows(alpha, heads, hidden, destination + heads * rowBytes);
  }

  // An F32 [rows, columns] tensor (the MoE router and the shared-expert
  // gate, which the block kernels read unquantized).
  void floatTensor(const std::string &module, uint64_t rows, uint64_t columns) {
    const uint64_t bytes = rows * columns * sizeof(float);
    image_.descriptor(ggml::kF32, rows, columns, {}, {bytes, 0, 0}, module);
    floatRows(module, rows, columns, image_.section(bytes));
  }

  // The token table's native rows, which the embedding kernel gathers.
  void embedding(const std::string &module) {
    const uint64_t vocabulary = geometry_.vocabularySize, hidden = geometry_.hiddenSize;
    const std::optional<TensorRows> source = quantized(module, vocabulary, hidden);
    if (!source) throw WeightStoreError(module + " is unquantized; Splash loads a quantized MLX token table");
    if (!gguf_embedding_format(gguf_format_of(source->type)))
      throw WeightStoreError(module + " is in a format the token gather does not read");
    image_.descriptor(source->type, vocabulary, hidden, {}, {vocabulary * source->rowBytes, 0, 0}, module);
    image_.copy(*source);
  }

  [[nodiscard]] gguf::Image finish() { return image_.finish(); }

private:
  const SourceTensor &require(const std::string &name, std::initializer_list<std::string_view> dtypes,
                              const std::vector<uint64_t> &shape) const {
    const SourceTensor &tensor = checkpoint_.require(name);
    if (std::find(dtypes.begin(), dtypes.end(), tensor.dtype) == dtypes.end() || tensor.shape != shape)
      throw WeightStoreError("source tensor type or shape does not match: " + name);
    return tensor;
  }

  static TensorRows rows(const std::string &name, const SourceTensor &tensor, uint32_t type, uint64_t count,
                         uint64_t rowBytes) {
    return {name, type, tensor.offset, count, rowBytes, {}, tensor.file};
  }

  // A tensor copied as stored into a section of its own.
  void stored(const std::string &name, std::initializer_list<std::string_view> dtypes, std::vector<uint64_t> shape) {
    const SourceTensor &tensor = require(name, dtypes, shape);
    image_.copy(rows(name, tensor, ggml::kBF16, 1, tensor.bytes));
  }

  // The module's native rows when the checkpoint holds it quantized, in the
  // format MLX infers from its tensors: codes [experts, rows, columns * bits /
  // 32] (U32) and scales [experts, rows, columns / group size], bf16 with bf16
  // biases (affine) or uint8 E8M0 exponents (mxfp4, 4 bits in groups of 32);
  // none when the checkpoint holds the module's weight unquantized.
  std::optional<TensorRows> quantized(const std::string &module, uint64_t rows, uint64_t columns,
                                      uint64_t experts = 1) const {
    const SourceTensor *scales = checkpoint_.find(module + ".scales");
    if (!scales) return std::nullopt;
    const SourceTensor &codes = checkpoint_.require(module + ".weight");
    std::vector<uint64_t> leading{rows};
    if (experts > 1) leading.insert(leading.begin(), experts);
    const auto shaped = [&](const SourceTensor &tensor) {
      return tensor.shape.size() == leading.size() + 1 &&
             std::equal(leading.begin(), leading.end(), tensor.shape.begin()) && tensor.shape.back();
    };
    if (codes.dtype != "U32" || !shaped(codes) || !shaped(*scales) || columns % scales->shape.back() ||
        codes.shape.back() * 32 % columns)
      throw WeightStoreError("source tensor type or shape does not match: " + module);
    const uint64_t group = columns / scales->shape.back(), bits = codes.shape.back() * 32 / columns;
    uint32_t format = GGUF_FMT_COUNT;
    const SourceTensor *biases = nullptr;
    if (scales->dtype == "BF16") {
      format = quant_affine_format_of(uint32_t(bits), uint32_t(group));
      biases = &require(module + ".biases", {"BF16"}, scales->shape);
    } else if (scales->dtype == "U8" && bits == 4 && group == 32) {
      format = GGUF_FMT_MXFP4;
    }
    if (format == GGUF_FMT_COUNT)
      throw WeightStoreError(module + " is " + std::to_string(bits) + "-bit in groups of " + std::to_string(group) +
                             " with " + scales->dtype + " scales; MLX weights load as affine 2, 3, 4, 5, 6 or 8 "
                             "bits in groups of 32, 64 or 128 with bf16 scales, or as mxfp4");
    return TensorRows{module, kQuantFormats[format].ggml_type, 0, experts * rows,
                      ggufRowBytes(kQuantFormats[format], columns), {}, codes.file, {&codes, scales, biases}};
  }

  // The F32 values of a [rows, columns] module at `destination`: its MLX
  // dequantization, or its bf16 or F32 weight.
  void floatRows(const std::string &module, uint64_t rows, uint64_t columns, uint64_t destination) {
    if (const std::optional<TensorRows> source = quantized(module, rows, columns)) {
      image_.copyAt(destination, *source, Conversion::DequantizeToFloat32);
      return;
    }
    const SourceTensor &weight = require(module + ".weight", {"BF16", "F32"}, {rows, columns});
    const bool bfloat16 = weight.dtype == "BF16";
    image_.copyAt(destination, this->rows(module, weight, bfloat16 ? ggml::kBF16 : ggml::kF32, rows, weight.bytes / rows),
                  bfloat16 ? Conversion::WidenToFloat32 : Conversion::None);
  }

  const SafetensorsCheckpoint &checkpoint_;
  const QwenTargetDimensions &geometry_;
  gguf::ImageBuilder image_;
};

// Plans nothing, and names the modules a walk of the images reads only
// quantized (quantizedModules): the projections and the token table, which
// Builder refuses unquantized.
struct QuantizedModules {
  void norm(const std::string &, uint64_t) {}
  void convolution(const std::string &) {}
  void timeBias(const std::string &) {}
  void decay(const std::string &) {}
  void projection(const std::string &module, uint64_t, uint64_t, uint64_t = 1) { modules.push_back(module); }
  void alphaBeta(const std::string &, const std::string &) {}
  void floatTensor(const std::string &, uint64_t, uint64_t) {}
  void embedding(const std::string &module) { modules.push_back(module); }

  std::vector<std::string> modules;
};

// The sections of each image, in their order, which a Builder plans and
// QuantizedModules names: a layer's, the head's and the embedding's.
template <class Walk> void walkLayer(Walk &b, const QwenTargetDimensions &g, uint32_t index) {
  const std::string prefix = "language_model.model.layers." + std::to_string(index) + ".";
  b.norm(prefix + "input_layernorm", g.hiddenSize);
  if (g.isFullAttentionLayer(index)) {
    const std::string attention = prefix + "self_attn.";
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.projection(attention + "q_proj", 2ull * g.attentionWidth, g.hiddenSize);
    b.projection(attention + "k_proj", kvRows, g.hiddenSize);
    b.projection(attention + "v_proj", kvRows, g.hiddenSize);
    b.norm(attention + "q_norm", g.attentionHeadDimension);
    b.norm(attention + "k_norm", g.attentionHeadDimension);
    b.projection(attention + "o_proj", g.hiddenSize, g.attentionWidth);
  } else {
    const std::string gdn = prefix + "linear_attn.";
    b.projection(gdn + "in_proj_qkv", g.convolutionDimension, g.hiddenSize);
    b.projection(gdn + "in_proj_z", g.attentionWidth, g.hiddenSize);
    b.alphaBeta(gdn + "in_proj_b", gdn + "in_proj_a");
    b.convolution(gdn + "conv1d");
    b.decay(gdn + "A_log");
    b.timeBias(gdn + "dt_bias");
    b.norm(gdn + "norm", g.gdnHeadDimension);
    b.projection(gdn + "out_proj", g.hiddenSize, g.attentionWidth);
  }
  b.norm(prefix + "post_attention_layernorm", g.hiddenSize);
  const std::string mlp = prefix + "mlp.";
  if (g.ffnKind == QwenFfnKind::SparseMoe) {
    const uint64_t experts = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(mlp + "gate", experts, g.hiddenSize);
    b.projection(mlp + "switch_mlp.gate_proj", width, g.hiddenSize, experts);
    b.projection(mlp + "switch_mlp.up_proj", width, g.hiddenSize, experts);
    b.projection(mlp + "switch_mlp.down_proj", g.hiddenSize, width, experts);
    b.projection(mlp + "shared_expert.gate_proj", width, g.hiddenSize);
    b.projection(mlp + "shared_expert.up_proj", width, g.hiddenSize);
    b.projection(mlp + "shared_expert.down_proj", g.hiddenSize, width);
    b.floatTensor(mlp + "shared_expert_gate", 1, g.hiddenSize);
  } else {
    b.projection(mlp + "gate_proj", g.intermediateSize, g.hiddenSize);
    b.projection(mlp + "up_proj", g.intermediateSize, g.hiddenSize);
    b.projection(mlp + "down_proj", g.hiddenSize, g.intermediateSize);
  }
}

template <class Walk> void walkHead(Walk &b, const QwenTargetDimensions &g) {
  b.norm("language_model.model.norm", g.hiddenSize);
  b.projection("language_model.lm_head", g.vocabularySize, g.hiddenSize);
}

template <class Walk> void walkEmbedding(Walk &b) { b.embedding("language_model.model.embed_tokens"); }

} // namespace

std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry) {
  std::vector<gguf::Image> images;
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
    Builder b(checkpoint, geometry, "layer-" + std::to_string(layer) + ".bin", layer,
              geometry.isFullAttentionLayer(layer) ? 1u : 0u);
    walkLayer(b, geometry, layer);
    images.push_back(b.finish());
  }
  Builder head(checkpoint, geometry, "head.bin", geometry.layers, 2);
  walkHead(head, geometry);
  images.push_back(head.finish());
  Builder embedding(checkpoint, geometry, "embedding.bin", geometry.vocabularySize, geometry.hiddenSize);
  walkEmbedding(embedding);
  images.push_back(embedding.finish());
  return images;
}

std::vector<std::string> quantizedModules(const QwenTargetDimensions &geometry) {
  QuantizedModules walk;
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) walkLayer(walk, geometry, layer);
  walkHead(walk, geometry);
  walkEmbedding(walk);
  return std::move(walk.modules);
}

} // namespace splash::model::mlx
