#include "model/SafetensorsImage.hpp"

#include "metal/abi/QuantFormat.h"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <optional>
#include <string_view>

namespace splash::model::safetensors {
namespace {

using gguf::Conversion;
using gguf::SafetensorsSource;
using gguf::Repack;
using gguf::TensorRows;

// The prefix of a checkpoint's language-model modules and its head's name,
// by the names it takes (ModuleNames).
std::string modelPrefix(ModuleNames names) {
  switch (names) {
  case ModuleNames::Mlx: return "language_model.model.";
  case ModuleNames::MlxLm: return "model.";
  case ModuleNames::Transformers: return "model.language_model.";
  }
  return {};
}
std::string headName(ModuleNames names) { return names == ModuleNames::Mlx ? "language_model.lm_head" : "lm_head"; }

// Plans one image from the checkpoint's tensors, which must have the dtypes
// and shapes its target reads.
class Builder {
public:
  // `norms` is how the image holds the norms (normConversion): None as
  // stored (bf16), or the F32 values WidenToFloat32 or CenteredNorm makes of
  // them.
  Builder(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry, ModuleNames names,
          Conversion norms, std::string name, uint32_t layer, uint32_t type, gguf::PlanOptions options = {})
      : checkpoint_(checkpoint), geometry_(geometry), names_(names), norms_(norms),
        image_(std::move(name), layer, type, options.streamExperts) {}

  // An RMSNorm's weight, which the norm kernels read (ops::NormWeights), as
  // the image holds norms.
  void norm(const std::string &name, uint64_t elements) { normWeight(name, elements, norms_); }

  // The GDN's gated norm, whose weight every checkpoint stores as the norm
  // multiplies by: as stored (bf16), or widened to F32 in images of F32
  // norms.
  void gatedNorm(const std::string &name, uint64_t elements) {
    normWeight(name, elements, norms_ == Conversion::None ? Conversion::None : Conversion::WidenToFloat32);
  }

  // The convolution taps of every channel, as stored: q and k channels, then
  // the value channels grouped by key head. MLX stores the depthwise weight
  // as [channels, taps, 1] and transformers as [channels, 1, taps], which
  // hold the same bytes.
  void convolution(const std::string &name) {
    const uint64_t channels = geometry_.convolutionDimension, taps = kGdnConvolutionTaps;
    stored(name + ".weight", {"BF16"},
           names_ != ModuleNames::Transformers ? std::vector<uint64_t>{channels, taps, 1}
                                      : std::vector<uint64_t>{channels, 1, taps});
  }

  // The bf16 GDN time-step bias of every value head, as stored.
  void timeBias(const std::string &name) { stored(name, {"BF16"}, {geometry_.gdnValueHeads}); }

  // The F32 GDN decay -exp(A_log) of every value head.
  void decay(const std::string &name) {
    const SourceTensor &tensor = require(name, {"BF16", "F32"}, {geometry_.gdnValueHeads});
    image_.copy(rows(name, tensor, tensor.dtype == "BF16" ? ggml::kBF16 : ggml::kF32, 1, tensor.bytes),
                Conversion::Decay);
  }

  // A quantized [rows, windows * columns] tensor, its planes: of windows
  // column windows of `columns` inputs, window w's rows after window w - 1's
  // (gguf::TensorRows::windows; a shared expert's down projection as slices
  // of the routed experts' width, ops::MoeShape::sharedSlices).
  void projection(const std::string &module, uint64_t rows, uint64_t columns, uint32_t windows = 1) {
    const TensorRows source = requireQuantized(module, rows, windows * columns);
    Repack repack = image_.planes(gguf_format_of(source.type), windows * rows, columns, module);
    for (uint32_t window = 0; window < windows; ++window) {
      TensorRows rows = source;
      rows.windows = windows;
      rows.window = window;
      repack.sources.push_back(std::move(rows));
    }
    image_.repack(std::move(repack));
  }

  // The routed experts' [rows, columns] tensors of one projection (gate_proj,
  // up_proj or down_proj) under the MLP prefix `mlp`, as one tensor of
  // experts * rows rows (gguf::ImageBuilder::expertPlanes): MLX's one stacked
  // tensor, or the experts' modules under their per-expert names, which
  // must share a format.
  void experts(const std::string &mlp, const std::string &projection, uint64_t rows, uint64_t columns) {
    const uint64_t count = geometry_.experts;
    if (names_ != ModuleNames::Transformers) {
      const std::string module = mlp + "switch_mlp." + projection;
      const std::optional<TensorRows> source = quantized(module, rows, columns, count);
      if (!source) throw WeightStoreError(module + " is unquantized; Splash loads quantized projections");
      Repack repack = image_.expertPlanes(gguf_format_of(source->type), count * rows, columns, module);
      repack.sources.push_back(*source);
      image_.expertRepack(std::move(repack));
      return;
    }
    std::vector<TensorRows> sources;
    for (uint64_t e = 0; e < count; ++e) {
      sources.push_back(requireQuantized(mlp + "experts." + std::to_string(e) + "." + projection, rows, columns));
      if (sources.back().type != sources.front().type)
        throw WeightStoreError(sources.back().name + " is in another format than its layer's first expert");
    }
    Repack repack = image_.expertPlanes(gguf_format_of(sources.front().type), count * rows, columns, mlp + "experts");
    repack.sources = std::move(sources);
    image_.expertRepack(std::move(repack));
  }

  // beta (value heads rows) | alpha (value heads rows): one tensor of their
  // shared format padded with zero rows to whole plane tiles, or else one F32
  // tensor.
  void alphaBeta(const std::string &beta, const std::string &alpha) {
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    const std::optional<TensorRows> b = quantized(beta, heads, hidden), a = quantized(alpha, heads, hidden);
    if (b && a && b->type == a->type) {
      Repack repack = image_.planes(gguf_format_of(b->type), gguf::tiledRows(2ull * heads), hidden, alpha);
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

  // The MoE router [rows, columns], which the float projection runs in the
  // format the checkpoint holds it (ops::QuantizedSegment): a quantized
  // tensor's native rows, or its bf16 or F32 values, as stored.
  void router(const std::string &module, uint64_t rows, uint64_t columns) {
    if (const std::optional<TensorRows> source = quantized(module, rows, columns)) {
      image_.descriptor(source->type, rows, columns, {}, {rows * source->rowBytes, 0, 0}, module);
      image_.copy(*source);
      return;
    }
    const SourceTensor &weight = require(module + ".weight", {"BF16", "F32"}, {rows, columns});
    const uint32_t type = weight.dtype == "BF16" ? ggml::kBF16 : ggml::kF32;
    image_.descriptor(type, rows, columns, {}, {weight.bytes, 0, 0}, module);
    image_.copy(this->rows(module, weight, type, rows, weight.bytes / rows));
  }

  // An F32 [rows, columns] tensor (the shared-expert scalar gate, which the
  // MoE selection reads in F32).
  void floatTensor(const std::string &module, uint64_t rows, uint64_t columns) {
    const uint64_t bytes = rows * columns * sizeof(float);
    image_.descriptor(ggml::kF32, rows, columns, {}, {bytes, 0, 0}, module);
    floatRows(module, rows, columns, image_.section(bytes));
  }

  // The F32 values of a [rows, columns] module in a section of their own,
  // without a descriptor (Qwen3.8-Flash-Next's hyper-connection injection
  // weights, which its mix reads as rows).
  void floatValues(const std::string &module, uint64_t rows, uint64_t columns) {
    floatRows(module, rows, columns, image_.section(rows * columns * sizeof(float)));
  }

  // Rows [first, first + rows) of a quantized [allRows, columns] module as
  // their native rows after their descriptor, which the float projection
  // decodes in fp32 (Qwen3.8-Flash-Next's QSA indexer, which runs float
  // segments: its queries' and key's rows, which one module index_qk_proj
  // holds).
  void nativeRows(const std::string &module, uint64_t allRows, uint64_t first, uint64_t rows, uint64_t columns) {
    TensorRows source = requireQuantized(module, allRows, columns);
    source.first = first;
    source.rows = rows;
    image_.descriptor(source.type, rows, columns, {}, {rows * source.rowBytes, 0, 0}, module);
    image_.copy(std::move(source));
  }

  // A table of `columns` values a row stored as the quantized modules
  // `stem` 0, 1, ... (each its rows' share, in order, as
  // Qwen3.8-Flash-Next's PLE table split into split_ngram_parts shards), as
  // their format's native rows, back to back after their descriptor; with
  // `stream`, rows that stream from the checkpoint instead
  // (gguf::Image::streamedRows), the copies naming where each shard is.
  void shardedTable(const std::string &stem, uint64_t columns, bool stream) {
    std::vector<TensorRows> shards;
    for (uint32_t shard = 0;; ++shard) {
      const std::string module = stem + std::to_string(shard);
      const SourceTensor *codes = checkpoint_.find(module + ".weight");
      if (!codes) break;
      if (codes->shape.empty()) throw WeightStoreError("source tensor type or shape does not match: " + module);
      shards.push_back(requireQuantized(module, codes->shape.front(), columns));
      if (shards.back().type != shards.front().type)
        throw WeightStoreError(module + " is in another format than the table's first shard");
    }
    if (shards.empty()) throw WeightStoreError("missing source tensor: " + stem + "0.weight");
    uint64_t rows = 0;
    for (const TensorRows &shard : shards) rows += shard.rows;
    const uint64_t rowBytes = shards.front().rowBytes;
    image_.descriptor(shards.front().type, rows, columns, {}, {rows * rowBytes, 0, 0}, stem);
    uint64_t destination = image_.section(rows * rowBytes);
    for (const TensorRows &shard : shards) {
      image_.copyAt(destination, shard, Conversion::None);
      destination += shard.rows * rowBytes;
    }
    if (stream)
      streamedRows_ = gguf::StreamedRows{stem, 0, 0, rows, static_cast<uint32_t>(rowBytes),
                                         gguf_format_of(shards.front().type)};
  }

  // F32 values the kernels read, of a tensor's bf16 values widened (Flint's
  // attention sinks, Qwen3.8-Flash-Next's PLE convolution [channels][taps], as
  // MLX's [channels, taps, 1] holds it; bf16 in every checkpoint of them known:
  // mlx-community/flint-mixed-3-4bit, Qwen3.8-Flash-Next-4bit and -OptiQ-2bit).
  void floatVector(const std::string &name, std::vector<uint64_t> shape) {
    const SourceTensor &tensor = require(name, {"BF16"}, shape);
    image_.copy(rows(name, tensor, ggml::kBF16, 1, tensor.bytes), Conversion::WidenToFloat32);
  }

  // The token table's rows, which the embedding kernel gathers: native rows
  // of its quantized format, or its bf16 rows as stored.
  void embedding(const std::string &module) {
    const uint64_t vocabulary = geometry_.vocabularySize, hidden = geometry_.hiddenSize;
    if (const std::optional<TensorRows> source = quantized(module, vocabulary, hidden)) {
      image_.descriptor(source->type, vocabulary, hidden, {}, {vocabulary * source->rowBytes, 0, 0}, module);
      image_.copy(*source);
      return;
    }
    const SourceTensor &table = require(module + ".weight", {"BF16"}, {vocabulary, hidden});
    image_.descriptor(ggml::kBF16, vocabulary, hidden, {}, {table.bytes, 0, 0}, module);
    image_.copy(rows(module + ".weight", table, ggml::kBF16, vocabulary, table.bytes / vocabulary));
  }

  [[nodiscard]] gguf::Image finish() {
    gguf::Image image = image_.finish();
    image.streamedRows = streamedRows_;
    return image;
  }

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

  // A norm's bf16 weight, as stored or converted as `conversion` says.
  void normWeight(const std::string &name, uint64_t elements, Conversion conversion) {
    if (conversion == Conversion::None) return stored(name + ".weight", {"BF16"}, {elements});
    const SourceTensor &tensor = require(name + ".weight", {"BF16"}, {elements});
    image_.copy(rows(name + ".weight", tensor, ggml::kBF16, 1, tensor.bytes), conversion);
  }

  // The module's native rows, which must be quantized.
  TensorRows requireQuantized(const std::string &module, uint64_t rows, uint64_t columns) const {
    const std::optional<TensorRows> source = quantized(module, rows, columns);
    if (!source) throw WeightStoreError(module + " is unquantized; Splash loads quantized projections");
    return *source;
  }

  // The module's native rows when the checkpoint holds it quantized, in the
  // format its tensors hold (metal/abi/QuantFormat.h); none when the
  // checkpoint holds the module's weight unquantized; a module without a
  // weight is a missing source tensor. MLX's format is the one
  // MLX infers from codes [experts, rows, columns * bits / 32] (U32) and
  // scales [experts, rows, columns / group size]: bf16 with bf16 biases
  // (affine), uint8 E8M0 exponents (mxfp4, 4 bits in groups of 32) or uint8
  // E4M3 scales (nvfp4, 4 bits in groups of 16). Model Optimizer's and
  // compressed-tensors' (floatQuantized) is the one their tensor types state.
  std::optional<TensorRows> quantized(const std::string &module, uint64_t rows, uint64_t columns,
                                      uint64_t experts = 1) const {
    if (experts == 1 && checkpoint_.find(module + ".weight_scale")) return floatQuantized(module, rows, columns);
    const SourceTensor &codes = checkpoint_.require(module + ".weight");
    const SourceTensor *scales = checkpoint_.find(module + ".scales");
    if (!scales) return std::nullopt;
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
    // The mode MLX infers from the scales: bf16 affine's, uint8 nvfp4's in
    // groups of 16 and mxfp4's otherwise.
    const bool affine = scales->dtype == "BF16";
    const std::string_view mode = affine ? "affine" : scales->dtype != "U8" ? "" : group == 16 ? "nvfp4" : "mxfp4";
    const uint32_t format = mlxFormat(mode, uint32_t(bits), uint32_t(group));
    const SourceTensor *biases = affine ? &require(module + ".biases", {"BF16"}, scales->shape) : nullptr;
    if (format == GGUF_FMT_COUNT)
      throw WeightStoreError(module + " is " + std::to_string(bits) + "-bit in groups of " + std::to_string(group) +
                             " with " + scales->dtype + " scales; MLX weights load as " +
                             std::string(kMlxAffineFormats) + " with bf16 scales, or as " +
                             std::string(kMlxFloatFormats));
    requireWholeBlocks(module, format, columns);
    return TensorRows{module, kQuantFormats[format].ggml_type, 0, experts * rows,
                      ggufRowBytes(kQuantFormats[format], columns), {}, codes.file, {&codes, scales, biases}};
  }

  // A module's native rows in NVFP4 or FP8 as Model Optimizer and
  // compressed-tensors store them. NVFP4: U8 codes [rows, columns / 2]
  // (element e in bits 4 (e % 2) of byte e / 2) and F8_E4M3 scales [rows,
  // columns / 16] beside the tensor scale g, Model Optimizer's .weight with g
  // its F32 weight_scale_2, compressed-tensors' .weight_packed with g the
  // reciprocal of its F32 weight_global_scale. FP8: F8_E4M3 values [rows,
  // columns] with an F32 or BF16 weight_scale g, the tensor's or (the
  // channel strategy of compressed-tensors) each row's. Input scales, which
  // scale activations, play no part in weights-only kernels.
  TensorRows floatQuantized(const std::string &module, uint64_t rows, uint64_t columns) const {
    const bool packed = checkpoint_.find(module + ".weight_packed") != nullptr;
    const std::string codes = module + (packed ? ".weight_packed" : ".weight");
    const SourceTensor &weight = checkpoint_.require(codes);
    SafetensorsSource source{&weight};
    const SourceTensor *scale = nullptr;
    bool perRow = false;
    uint32_t format;
    if (weight.dtype == "U8") {
      format = GGUF_FMT_NVFP4;
      require(codes, {"U8"}, {rows, columns / 2});
      source.scales = &require(module + ".weight_scale", {"F8_E4M3"}, {rows, columns / 16});
      scale = &scalar(module + (packed ? ".weight_global_scale" : ".weight_scale_2"), {"F32"});
    } else if (weight.dtype == "F8_E4M3" && !packed) {
      format = GGUF_FMT_FP8;
      require(codes, {"F8_E4M3"}, {rows, columns});
      perRow = checkpoint_.require(module + ".weight_scale").shape == std::vector<uint64_t>{rows, 1};
      scale = perRow ? &require(module + ".weight_scale", {"F32", "BF16"}, {rows, 1})
                     : &scalar(module + ".weight_scale", {"F32", "BF16"});
    } else {
      throw WeightStoreError(codes + " is " + weight.dtype + "; Splash loads NVFP4 (U8 codes) and FP8 (F8_E4M3 "
                             "values) as Model Optimizer and compressed-tensors store them");
    }
    requireWholeBlocks(module, format, columns);
    TensorRows result{module, kQuantFormats[format].ggml_type, 0, rows, ggufRowBytes(kQuantFormats[format], columns),
                      {}, weight.file, source};
    result.scale = {scale->file, scale->offset, scale->bytes, perRow ? 1 : rows, scale->dtype == "BF16", packed};
    return result;
  }

  // A tensor of one value, of shape [] or [1].
  const SourceTensor &scalar(const std::string &name, std::initializer_list<std::string_view> dtypes) const {
    return require(name, dtypes,
                   checkpoint_.require(name).shape.empty() ? std::vector<uint64_t>{} : std::vector<uint64_t>{1});
  }

  // Native rows hold whole native blocks of the module's format; only nvfp4's
  // and fp8's, of 256 elements, are wider than its groups.
  static void requireWholeBlocks(const std::string &module, uint32_t format, uint64_t columns) {
    if (columns % kQuantFormats[format].block_elements)
      throw WeightStoreError(module + "'s rows are not whole " + kQuantFormats[format].name + " blocks of " +
                             std::to_string(kQuantFormats[format].block_elements) + " elements");
  }

  // The F32 values of a [rows, columns] module at `destination`: its
  // quantization's values, or its bf16 or F32 weight.
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
  const ModuleNames names_;
  const Conversion norms_;
  gguf::ImageBuilder image_;
  std::optional<gguf::StreamedRows> streamedRows_;
};

// Plans nothing, and names the modules a walk of the images reads only
// quantized (quantizedModules): the projections, a layer's routed experts
// (MLX's stacked tensor of each projection, or once transformers' module
// mlp.experts, which holds them), and the head, which Builder refuses
// unquantized.
struct QuantizedModules {
  explicit QuantizedModules(ModuleNames names) : names(names) {}
  void norm(const std::string &, uint64_t) {}
  void gatedNorm(const std::string &, uint64_t) {}
  void convolution(const std::string &) {}
  void timeBias(const std::string &) {}
  void decay(const std::string &) {}
  void projection(const std::string &module, uint64_t, uint64_t, uint32_t = 1) { modules.push_back(module); }
  void experts(const std::string &mlp, const std::string &projection, uint64_t, uint64_t) {
    const std::string module = names != ModuleNames::Transformers ? mlp + "switch_mlp." + projection : mlp + "experts";
    if (modules.empty() || modules.back() != module) modules.push_back(module);
  }
  void alphaBeta(const std::string &, const std::string &) {}
  void router(const std::string &, uint64_t, uint64_t) {}
  void floatTensor(const std::string &, uint64_t, uint64_t) {}
  void floatVector(const std::string &, std::vector<uint64_t>) {}
  void floatValues(const std::string &, uint64_t, uint64_t) {}
  void nativeRows(const std::string &, uint64_t, uint64_t, uint64_t, uint64_t) {}
  void shardedTable(const std::string &, uint64_t, bool) {}
  void embedding(const std::string &) {}

  ModuleNames names;
  std::vector<std::string> modules;
};

// The sections of each image, in their order, which a Builder plans and
// QuantizedModules names, as each family's GGUF walk orders them
// (GgufImage.cpp): the mixers and FFNs every family's layers hold, then each
// family's layer and head around them, and the embedding.
//
// A layer's GDN or attention mixer. The queries' projection holds every
// query head's queries and, gated (attentionGate), its gate.
template <class Walk> void mixer(Walk &b, const QwenTargetDimensions &g, const std::string &prefix, bool full) {
  if (full) {
    const std::string attention = prefix + "self_attn.";
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.projection(attention + "q_proj", (g.attentionGate ? 2ull : 1ull) * g.attentionWidth, g.hiddenSize);
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
    b.gatedNorm(gdn + "norm", g.gdnHeadDimension);
    b.projection(gdn + "out_proj", g.hiddenSize, g.attentionWidth);
  }
}

// Layer `index`'s dense FFN or sparse MoE: a MoE layer's router, routed
// experts and shared expert of its slices (QwenTargetDimensions::
// sharedExpertSlices: gate and up of slices * width rows, down as slices
// column windows of width inputs) and, where its blocks have one, the shared
// expert's scalar gate; a dense layer's SwiGLU.
template <class Walk> void ffn(Walk &b, const QwenTargetDimensions &g, const std::string &prefix, uint32_t index) {
  const std::string mlp = prefix + "mlp.";
  if (g.isMoeLayer(index)) {
    const uint64_t width = g.expertIntermediateSize, shared = uint64_t{g.sharedExpertSlices} * width;
    b.router(mlp + "gate", g.experts, g.hiddenSize);
    b.experts(mlp, "gate_proj", width, g.hiddenSize);
    b.experts(mlp, "up_proj", width, g.hiddenSize);
    b.experts(mlp, "down_proj", g.hiddenSize, width);
    b.projection(mlp + "shared_expert.gate_proj", shared, g.hiddenSize);
    b.projection(mlp + "shared_expert.up_proj", shared, g.hiddenSize);
    b.projection(mlp + "shared_expert.down_proj", g.hiddenSize, width, g.sharedExpertSlices);
    if (g.sharedExpertGate) b.floatTensor(mlp + "shared_expert_gate", 1, g.hiddenSize);
  } else {
    b.projection(mlp + "gate_proj", g.intermediateSize, g.hiddenSize);
    b.projection(mlp + "up_proj", g.intermediateSize, g.hiddenSize);
    b.projection(mlp + "down_proj", g.hiddenSize, g.intermediateSize);
  }
}

// One hyper-connection mix (model/Qwen4Exp.hpp Qwen4HyperWeights): its norm,
// down and up projections and, when it injects, its injection weights as F32
// rows.
template <class Walk> void hyper(Walk &b, const QwenTargetDimensions &g, const std::string &stem, bool inject) {
  const uint64_t streams = g.streamWidth();
  b.norm(stem + "hc_norm", streams);
  b.projection(stem + "input_mix_weight_down", g.hyperRank, streams);
  b.projection(stem + "input_mix_weight_up", streams, g.hyperRank);
  if (inject) b.floatValues(stem + "block_inject_weight", g.hyperConnections, streams);
}

// A layer, by what its geometry holds: before its mixer and its FFN a norm,
// or a hyper-connection mix where the residual is hyper-connected
// (Qwen3.8-Flash-Next's); after the mixer its attention's learned sinks
// (attentionSinks: Flint's self_attn.softmax_offset, one per query head,
// widened to the F32 the GGUF's attn_sinks hold) and a full-attention layer's
// QSA indexer (hasIndexer: its queries and key as the native rows of
// index_qk_proj, which its float projection runs, and their norms); after the
// FFN, the PLE layer's projections, norms and convolution.
template <class Walk> void layer(Walk &b, const QwenTargetDimensions &g, const std::string &p, uint32_t index) {
  const bool full = g.isFullAttentionLayer(index);
  if (g.hyperConnected()) hyper(b, g, p + "attn_hyper_connection.", true);
  else b.norm(p + "input_layernorm", g.hiddenSize);
  mixer(b, g, p, full);
  if (g.attentionSinks) b.floatVector(p + "self_attn.softmax_offset", {g.attentionQueryHeads});
  if (full && g.hasIndexer()) {
    const std::string indexer = p + "self_attn.indexer.";
    const uint64_t queries = uint64_t{g.indexerHeads} * g.indexerHeadDimension;
    b.nativeRows(indexer + "index_qk_proj", queries + g.indexerHeadDimension, 0, queries, g.hiddenSize);
    b.nativeRows(indexer + "index_qk_proj", queries + g.indexerHeadDimension, queries, g.indexerHeadDimension,
                 g.hiddenSize);
    b.norm(indexer + "q_layernorm", g.indexerHeadDimension);
    b.norm(indexer + "k_layernorm", g.indexerHeadDimension);
  }
  if (g.hyperConnected()) hyper(b, g, p + "mlp_hyper_connection.", true);
  else b.norm(p + "post_attention_layernorm", g.hiddenSize);
  ffn(b, g, p, index);
  if (!g.pleWidth() || index != g.pleLayer) return;
  const std::string ple = p + "ple.";
  const uint64_t streams = g.streamWidth();
  b.projection(ple + "key_proj", streams, g.pleWidth());
  b.projection(ple + "value_proj", g.hiddenSize, g.pleWidth());
  b.norm(ple + "norm_key", streams);
  b.norm(ple + "norm_query", streams);
  b.norm(ple + "norm_conv", streams);
  b.floatVector(ple + "conv1d.weight", {streams, g.pleConvolutionTaps, 1});
}

// How the norms load. MLX stores them as the multipliers they are (mlx-lm's
// conversion adds the 1), held bf16; transformers 1 below them, held as the
// F32 1 + w. Flint's plain RMS norms (those of mlx-community/flint-mixed-3-4bit
// equal unsloth's GGUF's, its sinks the GGUF's sinks, on the 64 GB M5 Pro)
// widen to the F32 its GGUF's images hold, as Flint's kernel shapes are
// compiled for F32 norms alone (dev/tests/engine/TestFamilies.hpp
// normsOfFamilies). mlx-vlm keeps Qwen3.8-Flash-Next's RMSNorm weights as
// transformers stores them, 1 below the multiplier, but the GDN's gated norm's:
// the hyper, PLE, q and indexer norms of mlx-community/Qwen3.8-Flash-Next-4bit
// are those of ISTA-DASLab's Q2_0 GGUF less 1 exactly, its gated norms and
// convolution equal (the 48 GB M5 Pro); its images hold them as the GGUF's do, F32.
Conversion normConversion(ModuleNames names, std::string_view architecture) {
  if (architecture == "flint") return Conversion::WidenToFloat32;
  if (architecture == "qwen4exp") return Conversion::CenteredNorm;
  return names == ModuleNames::Transformers ? Conversion::CenteredNorm : Conversion::None;
}

// Walks every image, each through the walker `make(name, layer, type)` makes,
// which `done` takes once walked: the layers', the head's (before its
// projection a norm, or the final hyper-connection mix), the embedding's and,
// where the geometry has a PLE layer, its table's (ple.bin: the table split
// into shards, shardedTable), whose rows stream with streamTable.
template <class Make, class Done>
void walkImages(const QwenTargetDimensions &g, ModuleNames names, bool streamTable, Make make, Done done) {
  for (uint32_t index = 0; index < g.layers; ++index) {
    auto b = make("layer-" + std::to_string(index) + ".bin", index, g.isFullAttentionLayer(index) ? 1u : 0u);
    layer(b, g, modelPrefix(names) + "layers." + std::to_string(index) + ".", index);
    done(b);
  }
  auto head = make("head.bin", g.layers, 2u);
  if (g.hyperConnected()) hyper(head, g, modelPrefix(names) + "hyper_connection_mixer.", false);
  else head.norm(modelPrefix(names) + "norm", g.hiddenSize);
  head.projection(headName(names), g.vocabularySize, g.hiddenSize);
  done(head);
  auto embedding = make("embedding.bin", g.vocabularySize, g.hiddenSize);
  embedding.embedding(modelPrefix(names) + "embed_tokens");
  done(embedding);
  if (!g.pleWidth()) return;
  auto table = make("ple.bin", g.pleLayer, gguf::kTableImageType);
  table.shardedTable(layerPrefix(names, g.pleLayer) + "ple.ple_embedding.ngram_embedding.shards.", g.pleHeadDimension,
                     streamTable);
  done(table);
}

} // namespace

ModuleNames moduleNames(const SafetensorsCheckpoint &checkpoint) {
  if (checkpoint.find("language_model.model.embed_tokens.weight")) return ModuleNames::Mlx;
  if (checkpoint.find("model.language_model.embed_tokens.weight")) return ModuleNames::Transformers;
  if (checkpoint.find("model.embed_tokens.weight")) return ModuleNames::MlxLm;
  throw WeightStoreError("the checkpoint holds no token table: language_model.model.embed_tokens, "
                         "model.language_model.embed_tokens or model.embed_tokens");
}

std::string layerPrefix(ModuleNames names, uint32_t layer) {
  return modelPrefix(names) + "layers." + std::to_string(layer) + ".";
}

bool float32Norms(ModuleNames names, std::string_view architecture) {
  return normConversion(names, architecture) != Conversion::None;
}

std::vector<gguf::Image> planImages(const SafetensorsCheckpoint &checkpoint, const QwenTargetDimensions &geometry,
                                   std::string_view architecture, gguf::PlanOptions options) {
  const ModuleNames names = moduleNames(checkpoint);
  const Conversion norms = normConversion(names, architecture);
  std::vector<gguf::Image> images;
  walkImages(
      geometry, names, options.streamTable,
      [&](std::string name, uint32_t layer, uint32_t type) {
        return Builder(checkpoint, geometry, names, norms, std::move(name), layer, type, options);
      },
      [&](Builder &b) { images.push_back(b.finish()); });
  return images;
}

uint32_t mlxFormat(std::string_view mode, uint32_t bits, uint32_t group) noexcept {
  if (mode == "affine") return quant_affine_format_of(bits, group);
  if (mode == "mxfp4" && bits == 4 && group == 32) return GGUF_FMT_MXFP4;
  if (mode == "nvfp4" && bits == 4 && group == 16) return GGUF_FMT_NVFP4;
  return GGUF_FMT_COUNT;
}

std::vector<std::string> quantizedModules(const QwenTargetDimensions &geometry, ModuleNames names) {
  std::vector<std::string> modules;
  walkImages(
      geometry, names, false, [&](std::string, uint32_t, uint32_t) { return QuantizedModules(names); },
      [&](QuantizedModules &b) { modules.insert(modules.end(), b.modules.begin(), b.modules.end()); });
  return modules;
}

} // namespace splash::model::safetensors
