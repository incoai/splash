#include "model/GgufImage.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/QuantFormat.h"
#include "model/GgufImageLayout.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <sstream>

namespace splash::model::gguf {
namespace {

static_assert([] {
  for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format) {
    if (quant_loader_format(format)) continue;
    const QuantFormat &f = kQuantFormats[format];
    const GgmlTypeTraits *type = ggmlTypeTraits(f.ggml_type);
    if (!type || f.block_elements % type->blockElements ||
        f.block_bytes != f.block_elements / type->blockElements * type->blockBytes)
      return false;
  }
  return true;
}(), "a GGUF format's native block is a whole number of its GGUF type's blocks");
// The repack and the reference find meta unit u in native block u.
static_assert([] {
  for (const QuantFormat &format : kQuantFormats)
    if (format.meta_groups * 32 != format.block_elements) return false;
  return true;
}(), "a meta unit is one native block");

// The format whose native rows hold a GGUF tensor of `type`: the table's, or
// NVFP4's for llama.cpp's NVFP4, whose block_nvfp4 rows the writer builds
// NVFP4's native rows from (TensorRows::ggufNvfp4RowBytes).
uint32_t imageFormat(uint32_t type) { return type == ggml::kNVFP4 ? GGUF_FMT_NVFP4 : gguf_format_of(type); }

// A type of a format, BF16 and F16 among them: repacked into the planes
// projections multiply, or rows as stored (storedRowBytes), which the
// embedding kernels gather.
bool quantizedType(uint32_t type) { return imageFormat(type) != GGUF_FMT_COUNT; }
bool floatType(uint32_t type) { return type == ggml::kF32; }
// The 16-bit float types: a norm of one widens to F32 (floatNorm), a
// hyper-connection projection of one is prepared as Q8_0 (quantizedOrQ8_0),
// and an alpha/beta pair of them is a float tensor (alphaBeta).
bool float16Type(uint32_t type) { return type == ggml::kBF16 || type == ggml::kF16; }
// alpha/beta run as one segment: of their format when both are of one
// quantized type but BF16 and F16 (one repacked tensor), else one float
// tensor, of a BF16 pair's rows as stored or of F32.
bool alphaBetaType(uint32_t type) { return quantizedType(type) || type == ggml::kF32; }
// A vector the kernels read as F32 (floatNorm): F32, or a 16-bit float type
// that widens to it.
bool normType(uint32_t type) { return floatType(type) || float16Type(type); }
// A tensor the float projection runs in fp32 (floatTensor): F32, or BF16 as
// stored.
bool floatProjectionType(uint32_t type) { return floatType(type) || type == ggml::kBF16; }

// The bytes of a row of a tensor as the image copies it: F32's values, or
// its format's rows as stored (quant_stored_row_bytes), NVFP4's native rows
// for llama.cpp's NVFP4 (Builder::tensorRows); 0 for a width that holds none.
uint64_t storedRowBytes(const GgufTensor &tensor) {
  return floatType(tensor.type) ? tensor.columns() * sizeof(float)
                                : quant_stored_row_bytes(imageFormat(tensor.type), tensor.columns());
}

// A MoE router, which the float projection runs in fp32 as stored (router):
// a float projection's type, or a quantized one but F16, whose native rows it
// decodes.
bool routerType(uint32_t type) { return floatProjectionType(type) || (quantizedType(type) && type != ggml::kF16); }
// A tensor of F32 rows (floatRows): F32, or BF16 or Q8_0, which widen or
// decode to it.
bool floatRowType(uint32_t type) { return floatProjectionType(type) || type == ggml::kQ8_0; }

// The header type of the image of a table a family adds (planImages), beside
// the layers' 0 and 1 (GDN and full attention) and the head's 2.
constexpr uint32_t kTableImageType = 3;

// A repack of a [rows, columns] tensor in format, of no image section yet;
// throws unless the shape is whole plane tiles.
Repack tiledRepack(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name) {
  if (rows % QUANT_TILE_ROWS || columns % ggufColumnUnit(kQuantFormats[format]))
    throw GgufError("tensor is not tile aligned: " + name);
  Repack result;
  result.format = format;
  result.rows = rows;
  result.columns = columns;
  return result;
}

// The tensors a plan cannot load, each named past its layer's blk.N. with
// its fault (its type, missing, or unused) and counted in the layers it has
// it in, in the order first found: one refusal names every fault once, as
// "attn_q.weight (Q8_1) in 40 layers", however many layers. It also records
// every tensor the plan reads, so that a tensor no block reads is refused
// rather than silently dropped (a router or attention bias would change what
// the model computes).
class Problems final {
public:
  // Records `tensor` as one a block reads, whether or not its type loads, so
  // a tensor carries one fault.
  void read(const GgufTensor &tensor) { read_.insert(tensor.name); }
  // Adds every tensor of `file` the plan did not read, but those of the
  // `extraBlocks` multi-token prediction blocks after the target's `layers`
  // (nextn_predict_layers), which the target does not run.
  void addUnread(const GgufFile &file, uint32_t layers, uint64_t extraBlocks) {
    for (const GgufTensor &tensor : file.tensors())
      if (!read_.contains(tensor.name) && !inBlocks(tensor.name, layers, layers + extraBlocks))
        add(tensor.name, "unused");
  }
  void add(std::string_view tensor, std::string fault) {
    const size_t dot = tensor.starts_with("blk.") ? tensor.find('.', 4) : std::string_view::npos;
    const bool layered = dot != std::string_view::npos;
    if (layered) tensor.remove_prefix(dot + 1);
    const auto found = std::find_if(faults_.begin(), faults_.end(),
                                    [&](const Fault &f) { return f.tensor == tensor && f.fault == fault; });
    Fault &entry = found != faults_.end() ? *found : faults_.emplace_back(Fault{std::string(tensor), std::move(fault)});
    entry.layers += layered;
  }
  [[nodiscard]] bool empty() const noexcept { return faults_.empty(); }
  [[nodiscard]] std::string text() const {
    std::string result;
    for (const Fault &f : faults_) {
      result += (result.empty() ? "" : ", ") + f.tensor + " (" + f.fault + ")";
      if (f.layers) result += " in " + std::to_string(f.layers) + (f.layers == 1 ? " layer" : " layers");
    }
    return result;
  }

private:
  struct Fault {
    std::string tensor, fault;
    uint32_t layers = 0; // 0 for a tensor of no layer
  };
  // Whether `name` is a tensor of a block blk.N. with first <= N < end.
  static bool inBlocks(std::string_view name, uint64_t first, uint64_t end) {
    if (!name.starts_with("blk.")) return false;
    uint64_t block = 0;
    const char *last = name.data() + name.size();
    const auto [at, error] = std::from_chars(name.data() + 4, last, block);
    return error == std::errc{} && at != last && *at == '.' && block >= first && block < end;
  }
  std::vector<Fault> faults_;
  std::set<std::string, std::less<>> read_;
};

// Plans one image. A missing tensor or one of a type this build cannot load
// is added to `problems` and left out of the image, so the planner can name
// every such tensor at once; a tensor of the wrong shape throws.
class Builder {
public:
  Builder(const GgufFile &file, const QwenTargetDimensions &geometry, Problems &problems, std::string name,
          uint32_t layer, uint32_t type, PlanOptions options = {})
      : file_(file), geometry_(geometry), problems_(problems),
        image_(std::move(name), layer, type, options.streamExperts) {}

  // A vector the kernels read as F32 unrounded, as the reference GGUF
  // implementation does: a norm (ops::NormWeights) or Qwen3.8-Flash-Next's PLE
  // convolution. As stored, or the F32 values a BF16 or F16 one widens to
  // exactly.
  void floatNorm(const std::string &name, uint64_t elements) {
    const GgufTensor *tensor = find(name, normType);
    if (!tensor) return;
    if (tensor->elements() != elements) throw GgufError("unexpected shape for " + name);
    image_.copy(tensorRows(*tensor, 1, tensor->bytes),
                floatType(tensor->type) ? Conversion::None : Conversion::WidenToFloat32);
  }

  // Quantized rows [N, windows * K] repacked into planes, rows in `order`, as
  // `windows` column windows of K inputs, window w's N rows after window
  // w - 1's (TensorRows::windows).
  void quantized(const std::string &name, uint64_t rows, uint64_t columns, RowOrder order = {},
                 uint32_t windows = 1) {
    const GgufTensor *tensor = find(name, quantizedType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != windows * columns)
      throw GgufError("unexpected shape for " + name);
    const uint32_t format = imageFormat(tensor->type);
    Repack repack = image_.planes(format, windows * rows, columns, name);
    for (uint32_t window = 0; window < windows; ++window) {
      TensorRows source = tensorRows(*tensor, rows, ggufRowBytes(kQuantFormats[format], windows * columns), order);
      source.windows = windows;
      source.window = window;
      repack.sources.push_back(std::move(source));
    }
    image_.repack(std::move(repack));
  }

  // Rows [N, K] as quantized() repacks them, but of a BF16 or F16 tensor
  // as the Q8_0 rows the reference quantize_row_q8_0_ref writes from its
  // values, which the writer quantizes (TensorRows::ggufFloatRowBytes).
  void quantizedOrQ8_0(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor *tensor = file_.find(name);
    if (!tensor || !float16Type(tensor->type)) return quantized(name, rows, columns);
    problems_.read(*tensor);
    if (tensor->rows() != rows || tensor->columns() != columns) throw GgufError("unexpected shape for " + name);
    Repack repack = image_.planes(GGUF_FMT_Q80, rows, columns, name);
    TensorRows source = tensorRows(*tensor, rows, tensor->bytes / rows);
    source.ggufFloatType = source.type;
    source.ggufFloatRowBytes = source.rowBytes;
    source.type = ggml::kQ8_0;
    source.rowBytes = ggufRowBytes(kQuantFormats[GGUF_FMT_Q80], columns);
    repack.sources.push_back(std::move(source));
    image_.repack(std::move(repack));
  }

  // One routed experts' tensor [experts * N, K] (routedExperts).
  void expertTensor(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor *tensor = find(name, quantizedType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns) throw GgufError("unexpected shape for " + name);
    const uint32_t format = imageFormat(tensor->type);
    Repack repack = image_.expertPlanes(format, rows, columns, name);
    repack.sources.push_back(tensorRows(*tensor, rows, ggufRowBytes(kQuantFormats[format], columns)));
    image_.expertRepack(std::move(repack));
  }

  // A layer's routed experts (ImageBuilder::expertPlanes): gate and up, of
  // `experts` * width rows each, from ffn_gate_exps and ffn_up_exps or
  // the fused ffn_gate_up_exps some GGUFs store instead, each
  // expert's gate rows then its up rows, which the two take expert by expert;
  // then down, of `experts` * hidden rows, from ffn_down_exps.
  void routedExperts(const std::string &prefix, uint64_t experts, uint64_t width, uint64_t hidden) {
    const std::string fusedName = prefix + "ffn_gate_up_exps.weight";
    if (!file_.find(fusedName)) {
      expertTensor(prefix + "ffn_gate_exps.weight", experts * width, hidden);
      expertTensor(prefix + "ffn_up_exps.weight", experts * width, hidden);
    } else if (const GgufTensor *fused = find(fusedName, quantizedType)) {
      if (fused->rows() != 2 * experts * width || fused->columns() != hidden)
        throw GgufError("unexpected shape for " + fusedName);
      const uint32_t format = imageFormat(fused->type);
      const TensorRows rows = tensorRows(*fused, 2 * experts * width, ggufRowBytes(kQuantFormats[format], hidden));
      // Expert e's gate rows are rows 2e width.. of the fused tensor, its up rows the next width.
      for (const uint64_t up : {0, 1}) {
        Repack repack = image_.expertPlanes(format, experts * width, hidden, fusedName);
        for (uint64_t expert = 0; expert < experts; ++expert) {
          TensorRows expertRows = rows;
          expertRows.first = (2 * expert + up) * width;
          expertRows.rows = width;
          repack.sources.push_back(std::move(expertRows));
        }
        image_.expertRepack(std::move(repack));
      }
    }
    expertTensor(prefix + "ffn_down_exps.weight", experts * hidden, width);
  }

  // A layer's shared expert of `slices` slices of `width`, which the grouped
  // expert passes run as experts of their own (ops::MoeShape::sharedSlices,
  // ops::BlockExpertProjection): gate and up of slices * width rows, slice s
  // rows s * width on, and down's column windows of width inputs, window s's
  // rows from image row s * hidden on.
  void sharedExpert(const std::string &prefix, uint32_t slices, uint64_t width, uint64_t hidden) {
    quantized(prefix + "ffn_gate_shexp.weight", slices * width, hidden);
    quantized(prefix + "ffn_up_shexp.weight", slices * width, hidden);
    quantized(prefix + "ffn_down_shexp.weight", hidden, width, {}, slices);
  }

  // beta (value heads rows) | alpha (value heads rows), rows in grouped head
  // order: one tensor of their format padded with zero rows to whole plane
  // tiles when both are of one quantized type, else one float tensor of the
  // values each holds, all exactly: a BF16 pair as stored (the float
  // projection reads BF16 weights as it reads their F32 values), any other
  // pair as F32, F32 as stored and any other type decoded. A pair of BF16 or
  // F16 is a float tensor too: a float segment reads the input unrotated
  // (requireRotation), as Prism's BF16 alpha/beta need, and leaves the GDN
  // input projection on Apple9's register tile, which takes no F16.
  void alphaBeta(const std::string &betaName, const std::string &alphaName) {
    const GgufTensor *beta = find(betaName, alphaBetaType), *alpha = find(alphaName, alphaBetaType);
    if (!beta || !alpha) return;
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    for (const GgufTensor *t : {beta, alpha})
      if (t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be [" + std::to_string(heads) + ", hidden]: " + t->name);
    if (beta->type != alpha->type || !quantizedType(beta->type) || float16Type(beta->type)) {
      const bool bfloat16 = beta->type == ggml::kBF16 && alpha->type == ggml::kBF16;
      const uint64_t rowBytes = uint64_t{hidden} * (bfloat16 ? sizeof(uint16_t) : sizeof(float));
      image_.descriptor(bfloat16 ? ggml::kBF16 : ggml::kF32, 2ull * heads, hidden, {}, {2 * heads * rowBytes, 0, 0},
                        betaName);
      uint64_t destination = image_.section(2 * heads * rowBytes);
      for (const GgufTensor *t : {beta, alpha}) {
        if (quantizedType(t->type) && !bfloat16) {
          image_.copyAt(destination, tensorRows(*t, heads, storedRowBytes(*t), grouped(0, 1)),
                        Conversion::DequantizeToFloat32);
        } else {
          image_.copyAt(destination, tensorRows(*t, heads, t->bytes / heads, grouped(0, 1)), Conversion::None);
        }
        destination += heads * rowBytes;
      }
      return;
    }
    const uint32_t format = imageFormat(beta->type);
    const uint64_t rowBytes = ggufRowBytes(kQuantFormats[format], hidden);
    Repack repack = image_.planes(format, tiledRows(2ull * heads), hidden, alphaName);
    for (const GgufTensor *t : {beta, alpha}) repack.sources.push_back(tensorRows(*t, heads, rowBytes, grouped(0, 1)));
    image_.repack(std::move(repack));
  }

  // The convolution taps of every channel, q and k channels as stored, then
  // value channels in grouped head order, as the exact bf16 values the GDN
  // kernels read.
  void convolution(const std::string &name, uint64_t keyRows) {
    const uint32_t channels = geometry_.convolutionDimension;
    if (const GgufTensor *tensor = floatVector(name, uint64_t{channels} * kGdnConvolutionTaps))
      image_.copy(tensorRows(*tensor, channels, tensor->bytes / channels, grouped(keyRows, geometry_.gdnHeadDimension)),
                  Conversion::NarrowToBfloat16);
  }

  // A per value head F32 vector in grouped head order: as stored, or as the
  // exact bf16 values the kernels read.
  void headVector(const std::string &name, Conversion conversion) {
    const uint32_t heads = geometry_.gdnValueHeads;
    if (const GgufTensor *tensor = floatVector(name, heads))
      image_.copy(tensorRows(*tensor, heads, tensor->bytes / heads, grouped(0, 1)), conversion);
  }

  // An F32 vector the kernels read as stored (Flint's attention sinks).
  void floatValues(const std::string &name, uint64_t elements) {
    if (const GgufTensor *tensor = floatVector(name, elements)) image_.copy(tensorRows(*tensor, 1, tensor->bytes));
  }

  // Token rows as stored, gathered by the embedding kernel.
  void embeddingRows(const std::string &name) {
    const GgufTensor *tensor = find(name, quantizedType);
    if (!tensor) return;
    if (tensor->rows() != geometry_.vocabularySize || tensor->columns() != geometry_.hiddenSize)
      throw GgufError("unexpected shape for " + name);
    copiedRows(*tensor, storedRowBytes(*tensor));
  }

  // An F32 or BF16 tensor [rows, columns] as stored, which the float
  // projection runs in fp32 (the QSA indexer's projections).
  void floatTensor(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor *tensor = find(name, floatProjectionType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected an F32 or BF16 [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + name);
    copiedRows(*tensor, storedRowBytes(*tensor));
  }

  // The MoE router [rows, columns] as stored, which the float projection runs
  // in fp32 (ops::QuantizedSegment::rows): F32 or BF16, as quantized GGUFs
  // keep it, or a quantized type's native rows, as an MLX checkpoint's
  // quantized router loads (model/SafetensorsImage.hpp).
  void router(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor *tensor = find(name, routerType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected a [" + std::to_string(rows) + ", " + std::to_string(columns) + "] router: " + name);
    copiedRows(*tensor, storedRowBytes(*tensor));
  }

  // An F32, BF16 or Q8_0 tensor [rows, columns] as F32 rows: as stored, or
  // the F32 values BF16 widens to and Q8_0 decodes to exactly (the
  // hyper-connections' injection weights and the MoE shared-expert scalar
  // gate, which the select kernel reads). withDescriptor makes it a float
  // tensor a projection or the MoE block reads, else it is its rows alone.
  void floatRows(const std::string &name, uint64_t rows, uint64_t columns, bool withDescriptor) {
    const GgufTensor *tensor = find(name, floatRowType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected an F32, BF16 or Q8_0 [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + name);
    const bool widen = tensor->type == ggml::kBF16, dequantize = tensor->type == ggml::kQ8_0;
    if (withDescriptor)
      image_.descriptor(ggml::kF32, rows, columns, {}, {rows * columns * sizeof(float), 0, 0}, name);
    image_.copy(tensorRows(*tensor, rows, tensor->bytes / rows),
                widen        ? Conversion::WidenToFloat32
                : dequantize ? Conversion::DequantizeToFloat32
                             : Conversion::None);
  }

  // A table's rows [rows, columns] in a GGUF format, after their descriptor
  // (a family's table, which no projection reads), as stored but for GGUF's
  // NVFP4 (tableRows). An F16 table's values reach the model rounded to
  // bf16, as the gather writes them.
  void storedTable(const std::string &name, uint64_t columns) {
    const GgufTensor *tensor = find(name, quantizedType);
    if (!tensor) return;
    if (tensor->columns() != columns) throw GgufError("unexpected shape for " + name);
    copiedRows(*tensor, storedRowBytes(*tensor));
  }

  // Value heads of headRows rows from row `from` on, in grouped order.
  RowOrder grouped(uint64_t from, uint32_t headRows) const {
    return {from, headRows, geometry_.gdnKeyHeads, geometry_.gdnValueHeads / geometry_.gdnKeyHeads};
  }

  Image finish() { return image_.finish(); }

private:
  void reject(const std::string &name, const GgufTensor *tensor) {
    problems_.add(name, tensor ? ggmlTypeName(tensor->type) : "missing");
  }

  // The tensor `name` when it is present and of an accepted type.
  const GgufTensor *find(const std::string &name, bool (*accepted)(uint32_t type)) {
    const GgufTensor *tensor = file_.find(name);
    if (tensor) problems_.read(*tensor);
    if (tensor && accepted(tensor->type)) return tensor;
    reject(name, tensor);
    return nullptr;
  }

  const GgufTensor *floatVector(const std::string &name, uint64_t elements) {
    const GgufTensor *tensor = find(name, floatType);
    if (tensor && tensor->elements() != elements) throw GgufError("unexpected shape for " + name);
    return tensor;
  }

  // Rows [0, count) of `tensor` as the image holds them, rowBytes each: as
  // stored, or, of llama.cpp's NVFP4, NVFP4's native rows, which the writer
  // builds from its block_nvfp4 rows and the g of its .scale tensor. llama.cpp
  // multiplies a projection's products by its .scale, of one value, or an
  // expert's by its value of an experts tensor's (_exps), one per expert;
  // only NVFP4's native rows hold such a scale, so only theirs is read and a
  // .scale beside any other tensor is refused as unused. An NVFP4 weight may
  // also carry an .input_scale of as many values, the activation scale of
  // NVIDIA's W4A4: a weights-only decode (ours, as llama.cpp's) never reads
  // it, so it is held to that count and recorded as read, unused; beside any
  // other tensor, or alone, it is refused as unused too.
  TensorRows tensorRows(const GgufTensor &tensor, uint64_t count, uint64_t rowBytes, RowOrder order = {}) const {
    if (tensor.type != ggml::kNVFP4) {
      if (!count || count * rowBytes != tensor.bytes) throw GgufError("unexpected size for " + tensor.name);
      return {tensor.name, tensor.type, tensor.offset, count, rowBytes, order, &file_.source(tensor.file)};
    }
    const QuantFormat &format = kQuantFormats[GGUF_FMT_NVFP4];
    const GgmlTypeTraits &blocks = *ggmlTypeTraits(ggml::kNVFP4);
    const uint64_t sourceRowBytes =
        rowBytes / format.block_bytes * (format.block_elements / blocks.blockElements) * blocks.blockBytes;
    if (!count || rowBytes % format.block_bytes || count * sourceRowBytes != tensor.bytes)
      throw GgufError(tensor.name + "'s rows are not whole nvfp4 blocks of " + std::to_string(format.block_elements) +
                      " elements");
    TensorRows rows{tensor.name, QUANT_NVFP4_TYPE, tensor.offset, count, rowBytes, order, &file_.source(tensor.file)};
    rows.ggufNvfp4RowBytes = sourceRowBytes;
    const std::string stem = tensor.name.ends_with(".weight") ? tensor.name.substr(0, tensor.name.size() - 7) : "";
    const uint64_t values = stem.ends_with("_exps") ? geometry_.experts : 1;
    // The weight's F32 tensor stem + suffix of `values` values, read, or
    // nullptr without one.
    const auto sidecar = [&](const char *suffix) -> const GgufTensor * {
      const GgufTensor *found = stem.empty() ? nullptr : file_.find(stem + suffix);
      if (!found) return nullptr;
      if (found->type != ggml::kF32 || found->elements() != values || count % values)
        throw GgufError(found->name + " must hold " + (values == 1 ? "one F32 value" : "one F32 value per expert") +
                        " for " + tensor.name);
      problems_.read(*found);
      return found;
    };
    if (const GgufTensor *scale = sidecar(".scale"))
      rows.scale = {&file_.source(scale->file), scale->offset, scale->bytes, count / values};
    sidecar(".input_scale");
    return rows;
  }

  // A tensor's rows as the image holds them (tensorRows), rowBytes each,
  // after their descriptor.
  void copiedRows(const GgufTensor &tensor, uint64_t rowBytes) {
    const TensorRows rows = tensorRows(tensor, tensor.rows(), rowBytes);
    image_.descriptor(rows.type, tensor.rows(), tensor.columns(), {}, {rows.rows * rows.rowBytes, 0, 0}, tensor.name);
    image_.copy(rows);
  }

  const GgufFile &file_;
  const QwenTargetDimensions &geometry_;
  Problems &problems_;
  ImageBuilder image_;
};

std::string prefix(uint32_t layer) { return "blk." + std::to_string(layer) + "."; }

// A layer's GDN or attention mixer, which every family's layers hold. The
// queries' projection holds every query head's queries and, gated
// (attentionGate), its gate.
void mixerImage(Builder &b, const QwenTargetDimensions &g, const std::string &p, bool full) {
  if (full) {
    b.quantized(p + "attn_q.weight", (g.attentionGate ? 2ull : 1ull) * g.attentionWidth, g.hiddenSize);
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
    b.headVector(p + "ssm_a", Conversion::None);
    b.headVector(p + "ssm_dt.bias", Conversion::NarrowToBfloat16);
    b.floatNorm(p + "ssm_norm.weight", g.gdnHeadDimension);
    b.quantized(p + "ssm_out.weight", g.hiddenSize, valueRows);
  }
}

// Layer `index`'s dense FFN or sparse MoE, which every family's layers hold:
// a MoE layer's router, routed experts and shared expert of its slices
// (QwenTargetDimensions::sharedExpertSlices) and, where its blocks have one,
// the shared expert's scalar gate; a dense layer's SwiGLU.
void ffnImage(Builder &b, const QwenTargetDimensions &g, const std::string &p, uint32_t index) {
  if (g.isMoeLayer(index)) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.router(p + "ffn_gate_inp.weight", routed, g.hiddenSize);
    b.routedExperts(p, routed, width, g.hiddenSize);
    b.sharedExpert(p, g.sharedExpertSlices, width, g.hiddenSize);
    if (g.sharedExpertGate) b.floatRows(p + "ffn_gate_inp_shexp.weight", 1, g.hiddenSize, true);
  } else {
    b.quantized(p + "ffn_gate.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_up.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_down.weight", g.hiddenSize, g.intermediateSize);
  }
}

// One hyper-connection mix (model/Qwen4Exp.hpp Qwen4HyperWeights): its norm,
// down and up projections and, when it injects, its injection weights. The
// projections of BF16 or F16 (some GGUFs keep them BF16) are prepared as Q8_0,
// the type other GGUFs store them in, in the rows the reference GGUF
// quantization of Q8_0 makes from their values (Builder::quantizedOrQ8_0):
// every decode cycle reads all of them, bandwidth-bound, and Q8_0 holds a value
// in little more than half the bytes of a 16-bit float.
void hyperImage(Builder &b, const QwenTargetDimensions &g, const std::string &stem, bool inject) {
  const uint64_t streams = g.streamWidth();
  b.floatNorm(stem + "_norm.weight", streams);
  b.quantizedOrQ8_0(stem + "_down.weight", g.hyperRank, streams);
  b.quantizedOrQ8_0(stem + "_up.weight", streams, g.hyperRank);
  if (inject) b.floatRows(stem + "_inject.weight", g.hyperConnections, streams, false);
}

// What a family's GGUFs hold beside the mixers and FFNs every family's layers
// hold, which requireMetadata and planImages take once from a GGUF's
// architecture (withFamily): the metadata keys and arrays they add, each
// layer's sections around its mixer and FFN, the head's before the logits,
// and the images of the tables they add after the embedding's.
//
// The Qwen3.5 hybrid's GGUFs (qwen35, qwen35moe): a norm before each layer's
// mixer and FFN and before the logits.
struct HybridGguf final {
  static constexpr std::array<std::string_view, 2> architectures{"qwen35", "qwen35moe"};
  template <class Expect, class ExpectFloat, class ExpectArray>
  static void metadata(const Expect &, const ExpectFloat &, const ExpectArray &, const QwenTargetDimensions &) {}
  static void layer(Builder &b, const QwenTargetDimensions &g, uint32_t index) {
    const std::string p = prefix(index);
    b.floatNorm(p + "attn_norm.weight", g.hiddenSize);
    mixerImage(b, g, p, g.isFullAttentionLayer(index));
    b.floatNorm(p + "post_attention_norm.weight", g.hiddenSize);
    ffnImage(b, g, p, index);
  }
  static void head(Builder &b, const QwenTargetDimensions &g) { b.floatNorm("output_norm.weight", g.hiddenSize); }
  static void tables(const GgufFile &, const QwenTargetDimensions &, PlanOptions, Problems &,
                     std::vector<Image> &) {}
};

// Qwen3.8-Flash-Next's GGUFs (qwen4exp): hyper-connection mixes in place of
// the hybrid's norms, a QSA indexer after each full-attention mixer, and the
// PLE n-gram embedding: its projections, norms and convolution after its
// layer's FFN, and its table in an image of its own.
struct Qwen4ExpGguf final {
  static constexpr std::string_view architecture = "qwen4exp";

  // Its arrays: each layer's QSA compress ratio and the PLE layer.
  template <class Expect, class ExpectFloat, class ExpectArray>
  static void metadata(const Expect &expect, const ExpectFloat &, const ExpectArray &expectArray,
                       const QwenTargetDimensions &g) {
    expect("hyper_connection.count", g.hyperConnections);
    expect("hyper_connection.low_rank", g.hyperRank);
    expect("attention.indexer.head_count", g.indexerHeads);
    expect("attention.indexer.key_length", g.indexerHeadDimension);
    expect("attention.indexer.top_k", g.indexerTokens());
    expect("ple.ngram_size", g.pleNgram);
    expect("ple.heads_per_ngram", g.pleHeadsPerNgram);
    expect("ple.conv_kernel", g.pleConvolutionTaps);
    expect("embedding_length_per_layer_input", g.pleHeadDimension);
    expectArray("attention.compress_ratios", [&](std::span<const double> ratios) {
      bool matches = ratios.size() >= g.layers;
      for (uint32_t layer = 0; matches && layer < g.layers; ++layer)
        matches = ratios[layer] == (g.isFullAttentionLayer(layer) ? g.indexerBlockTokens : 0);
      return matches;
    });
    expectArray("ple.layers",
                [&](std::span<const double> layers) { return layers.size() == 1 && layers[0] == g.pleLayer; });
  }

  static void layer(Builder &b, const QwenTargetDimensions &g, uint32_t index) {
    const std::string p = prefix(index);
    const bool full = g.isFullAttentionLayer(index);
    hyperImage(b, g, p + "hc_attn", true);
    mixerImage(b, g, p, full);
    if (full) {
      b.floatTensor(p + "indexer.q_proj.weight", uint64_t{g.indexerHeads} * g.indexerHeadDimension, g.hiddenSize);
      b.floatTensor(p + "indexer.k_proj.weight", g.indexerHeadDimension, g.hiddenSize);
      b.floatNorm(p + "indexer.q_norm.weight", g.indexerHeadDimension);
      b.floatNorm(p + "indexer.k_norm.weight", g.indexerHeadDimension);
    }
    hyperImage(b, g, p + "hc_ffn", true);
    ffnImage(b, g, p, index);
    if (index != g.pleLayer) return;
    const uint64_t streams = g.streamWidth();
    b.quantized(p + "ple_key.weight", streams, g.pleWidth());
    b.quantized(p + "ple_value.weight", g.hiddenSize, g.pleWidth());
    b.floatNorm(p + "ple_norm_key.weight", streams);
    b.floatNorm(p + "ple_norm_query.weight", streams);
    b.floatNorm(p + "ple_norm_conv.weight", streams);
    b.floatNorm(p + "ple_conv1d.weight", streams * g.pleConvolutionTaps);
  }

  static void head(Builder &b, const QwenTargetDimensions &g) { hyperImage(b, g, "output_hc", false); }

  // The PLE table's rows as stored in an image of their own (ple.bin), whose
  // rows stream from the file instead with streamTable once every tensor
  // the plan reads was found (planImages names those it misses first).
  static void tables(const GgufFile &file, const QwenTargetDimensions &g, PlanOptions options, Problems &problems,
                     std::vector<Image> &images) {
    const std::string table = "per_layer_token_embd.weight";
    Builder b(file, g, problems, "ple.bin", g.pleLayer, kTableImageType);
    b.storedTable(table, g.pleHeadDimension);
    Image image = b.finish();
    if (options.streamTable && problems.empty()) image.streamedRows = tableRows(file, table, g.pleHeadDimension);
    images.push_back(std::move(image));
  }
};

// Flint's GGUFs (flint): every layer attends, those sliding_window_pattern
// names over a sliding window and those rope_pattern names with RoPE, each
// with its learned sinks after its mixer (attn_sinks, an F32 per query head,
// as llama.cpp writes them); a norm before each layer's mixer and FFN
// (ffn_norm) and before the logits; and the FFN of the
// leading_dense_block_count first layers dense, of the others a MoE with a
// sigmoid router and an ungated shared expert of expert_shared_count routed
// experts' width. It has no GDN, and its scalars hold no rope.dimension_count:
// its RoPE rotates whole heads.
struct FlintGguf final {
  static constexpr std::string_view architecture = "flint";
  // llama.cpp's expert_gating_func of each router
  // (LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID).
  static constexpr uint64_t kSoftmaxGating = 1, kSigmoidGating = 2;

  // Its routing weights are the selected experts' normalized scores, which
  // nothing scales (expert_weights_scale 1). Its arrays: each layer's
  // sliding window and RoPE.
  template <class Expect, class ExpectFloat, class ExpectArray>
  static void metadata(const Expect &expect, const ExpectFloat &expectFloat, const ExpectArray &expectArray,
                       const QwenTargetDimensions &g) {
    expect("attention.sliding_window", g.slidingWindow);
    expect("leading_dense_block_count", g.leadingDenseLayers);
    expect("feed_forward_length", g.intermediateSize);
    expect("expert_shared_count", g.sharedExpertSlices);
    expect("expert_gating_func", g.moeRouter == ops::MoeRouter::SigmoidNormalized ? kSigmoidGating : kSoftmaxGating);
    expect("expert_weights_norm", 1);
    expectFloat("expert_weights_scale", 1.0);
    // Every layer attends: a layer's bit is its attention layer's.
    const auto pattern = [&g](uint64_t mask) {
      return [&g, mask](std::span<const double> values) {
        bool matches = values.size() == g.layers;
        for (uint32_t layer = 0; matches && layer < g.layers; ++layer)
          matches = values[layer] == double(layer < 64 && (mask >> layer & 1));
        return matches;
      };
    };
    expectArray("attention.sliding_window_pattern", pattern(g.slidingLayers));
    expectArray("attention.rope_pattern", pattern(~g.unrotatedLayers));
  }

  static void layer(Builder &b, const QwenTargetDimensions &g, uint32_t index) {
    const std::string p = prefix(index);
    b.floatNorm(p + "attn_norm.weight", g.hiddenSize);
    mixerImage(b, g, p, true);
    b.floatValues(p + "attn_sinks.weight", g.attentionQueryHeads);
    b.floatNorm(p + "ffn_norm.weight", g.hiddenSize);
    ffnImage(b, g, p, index);
  }
  static void head(Builder &b, const QwenTargetDimensions &g) { b.floatNorm("output_norm.weight", g.hiddenSize); }
  static void tables(const GgufFile &, const QwenTargetDimensions &, PlanOptions, Problems &,
                     std::vector<Image> &) {}
};

// Calls `use` with what the GGUFs of `architecture` hold: Qwen3.8-Flash-Next's
// (Qwen4ExpGguf), Flint's (FlintGguf) or the Qwen3.5 hybrid's (HybridGguf).
// Another architecture is refused, so a layout without a family here fails
// by name rather than planning as the hybrid.
template <class Use> decltype(auto) withFamily(std::string_view architecture, Use &&use) {
  if (architecture == Qwen4ExpGguf::architecture) return use(Qwen4ExpGguf{});
  if (architecture == FlintGguf::architecture) return use(FlintGguf{});
  if (std::ranges::find(HybridGguf::architectures, architecture) == HybridGguf::architectures.end())
    throw GgufError("no GGUF family plans architecture " + std::string(architecture));
  return use(HybridGguf{});
}

// The images of a target from a GGUF of Family (planImages).
template <class Family>
std::vector<Image> familyImages(const GgufFile &file, const QwenTargetDimensions &g, PlanOptions options) {
  Problems problems;
  std::vector<Image> images;
  for (uint32_t index = 0; index < g.layers; ++index) {
    Builder layer(file, g, problems, "layer-" + std::to_string(index) + ".bin", index,
                  g.isFullAttentionLayer(index) ? 1u : 0u, options);
    Family::layer(layer, g, index);
    images.push_back(layer.finish());
  }
  Builder head(file, g, problems, "head.bin", g.layers, 2);
  Family::head(head, g);
  head.quantized("output.weight", g.vocabularySize, g.hiddenSize);
  images.push_back(head.finish());
  Builder embedding(file, g, problems, "embedding.bin", g.vocabularySize, g.hiddenSize);
  embedding.embeddingRows("token_embd.weight");
  images.push_back(embedding.finish());
  Family::tables(file, g, options, problems, images);
  const GgufMetadata &metadata = file.metadata();
  problems.addUnread(file, g.layers,
                     metadata.unsignedValue(metadata.architecture() + ".nextn_predict_layers").value_or(0));
  if (!problems.empty()) throw GgufError("GGUF tensors this build cannot load: " + problems.text());
  return images;
}

// A rotated GGUF (GgufRotation) must name exactly what the loader rotates:
// every tensor the images of a dense target repack, whose segments read
// H (D x) (ops::InputRotation) while float segments (alpha/beta as one float
// tensor) read x as it is, and the token table, which the rotated gather
// decodes from PQ2_0 rows, with the GDN value heads of the rotated inputs
// grouped.
void requireRotation(const GgufFile &file, const QwenTargetDimensions &g, const std::vector<Image> &images) {
  const GgufRotation &rotation = *file.rotation();
  if (g.ffnKind == QwenFfnKind::SparseMoe) throw GgufError("rotated weights are supported for dense targets only");
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

std::vector<StreamedExperts> streamedExperts(std::span<const Image> layers) {
  std::vector<StreamedExperts> result;
  for (const Image &layer : layers) {
    if (!layer.experts) throw GgufError("a target layer's experts do not stream");
    result.push_back(*layer.experts);
  }
  return result;
}

uint64_t loadedBytes(const Image &image) { return image.streamedRows ? 0 : image.bytes; }

uint64_t convertedBytes(const TensorRows &source, Conversion conversion, uint64_t sourceBytes) {
  switch (conversion) {
  case Conversion::NarrowToBfloat16: return sourceBytes / 2;
  case Conversion::WidenToFloat32: case Conversion::CenteredNorm: return sourceBytes * 2;
  case Conversion::Decay: return source.type == ggml::kBF16 ? sourceBytes * 2 : sourceBytes;
  case Conversion::DequantizeToFloat32: {
    const uint64_t columns = decodedColumns(source);
    if (sourceBytes % source.rowBytes) throw GgufError("decoded bytes are not whole rows: " + source.name);
    return sourceBytes / source.rowBytes * columns * sizeof(float);
  }
  case Conversion::None: break;
  }
  return sourceBytes;
}

uint64_t decodedColumns(const TensorRows &rows) {
  const uint32_t format = gguf_format_of(rows.type);
  if (format == GGUF_FMT_COUNT) throw GgufError("not a quantized tensor: " + rows.name);
  // Rows as stored take a native block's bytes as stored for each of its
  // elements, at any width they hold.
  const uint64_t block = kQuantFormats[format].block_elements;
  const uint64_t columns = rows.rowBytes * block / quant_stored_row_bytes(format, block);
  if (!columns || quant_stored_row_bytes(format, columns) != rows.rowBytes)
    throw GgufError("decoded rows are not rows as stored: " + rows.name);
  return columns;
}

ImageBuilder::ImageBuilder(std::string name, uint32_t layer, uint32_t type, bool streamExperts)
    : streamExperts_(streamExperts) {
  image_.name = std::move(name);
  image_.magic = kGgufImageMagic;
  image_.layer = layer;
  image_.type = type;
  const auto header = weightFileHeader(image_.magic, layer, type);
  image_.fills.push_back({0, {header.begin(), header.end()}});
  cursor_ = residentCursor_ = header.size();
}

uint64_t ImageBuilder::section(uint64_t bytes) {
  if (!bytes) throw GgufError("empty image section in " + image_.name);
  const uint64_t start = alignWeightOffset(cursor_);
  cursor_ = start + bytes;
  residentCursor_ = alignWeightOffset(residentCursor_) + bytes;
  return start;
}

void ImageBuilder::copy(TensorRows source, Conversion conversion) {
  const uint64_t bytes = convertedBytes(source, conversion, source.rows * source.rowBytes);
  copyAt(section(bytes), std::move(source), conversion);
}

void ImageBuilder::copyAt(uint64_t destination, TensorRows source, Conversion conversion) {
  image_.copies.push_back({destination, std::move(source), conversion});
}

void ImageBuilder::descriptor(uint32_t type, uint64_t rows, uint64_t columns, const QuantFormat &format,
                              const GgufPlaneBytes &bytes, const std::string &name) {
  if (rows > std::numeric_limits<uint32_t>::max() || columns > std::numeric_limits<uint32_t>::max())
    throw GgufError("tensor is too large for its descriptor: " + name);
  const auto encoded = GgufTensorDescriptor{.type = type,
                                            .outputSize = static_cast<uint32_t>(rows),
                                            .inputSize = static_cast<uint32_t>(columns),
                                            .p0 = format.plane0_bytes,
                                            .p1 = format.plane1_bytes,
                                            .metaBytes = format.meta_bytes,
                                            .metaGroups = format.meta_groups,
                                            .plane0Bytes = bytes.plane0,
                                            .plane1Bytes = bytes.plane1,
                                            .metaTotalBytes = bytes.meta}
                           .encode();
  image_.fills.push_back({section(encoded.size()), {encoded.begin(), encoded.end()}});
}

Repack ImageBuilder::planes(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name) {
  Repack result = tiledRepack(format, rows, columns, name);
  const QuantFormat &layout = kQuantFormats[format];
  const GgufPlaneBytes bytes = ggufPlaneBytes(layout, rows, columns);
  descriptor(layout.ggml_type, rows, columns, layout, bytes, name);
  result.plane0 = section(bytes.plane0);
  result.plane1 = bytes.plane1 ? section(bytes.plane1) : 0;
  result.meta = section(bytes.meta);
  return result;
}

void ImageBuilder::repack(Repack repack) { image_.repacks.push_back(std::move(repack)); }

Repack ImageBuilder::expertPlanes(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name) {
  if (!streamExperts_) return planes(format, rows, columns, name);
  // The sections planes() would lay out, the descriptor's and the planes',
  // count in the image's resident bytes.
  const GgufPlaneBytes bytes = ggufPlaneBytes(kQuantFormats[format], rows, columns);
  const uint64_t descriptorBytes = GgufTensorDescriptor{}.encode().size();
  for (const uint64_t section : {descriptorBytes, bytes.plane0, bytes.plane1, bytes.meta})
    if (section) residentCursor_ = alignWeightOffset(residentCursor_) + section;
  return tiledRepack(format, rows, columns, name);
}

void ImageBuilder::expertRepack(Repack repack) {
  if (!streamExperts_) return this->repack(std::move(repack));
  if (!image_.experts) image_.experts.emplace();
  auto &projections = image_.experts->projections;
  if (expertProjections_ == projections.size())
    throw GgufError("a layer has more than three routed expert tensors: " + image_.name);
  projections[expertProjections_++] = std::move(repack);
}

Image ImageBuilder::finish() {
  image_.bytes = alignWeightOffset(cursor_);
  image_.residentBytes = alignWeightOffset(residentCursor_);
  return std::move(image_);
}

namespace {

// The target geometry a GGUF of `architecture` declares in its metadata, the
// architecture included, with the rotary embedding and norms the kernels
// compute: the RoPE base and rotated dimensions, the RMS epsilon and no RoPE
// scaling; and the keys and arrays of what the family adds. Throws GgufError
// naming every mismatch.
void requireMetadata(const GgufFile &file, std::string_view architecture, const QwenTargetDimensions &geometry) {
  const GgufMetadata &metadata = file.metadata();
  const std::string arch(architecture);
  if (metadata.architecture() != arch)
    throw GgufError("GGUF architecture is " + metadata.architecture() + ", but the model's target is " + arch);
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = metadata.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  // A float equal to value up to its F32 rounding.
  const auto expectFloat = [&](const char *key, double value) {
    const std::optional<double> found = metadata.floatValue(arch + "." + key);
    if (found && std::abs(*found - value) <= value * 1e-6) return;
    std::ostringstream text;
    text << (mismatched.empty() ? "" : ", ") << key << ' ';
    if (found) text << *found;
    else text << "missing";
    text << " (expected " << value << ')';
    mismatched += text.str();
  };
  // An array, named alone, that `matches` takes.
  const auto expectArray = [&](const char *key, const auto &matches) {
    const std::optional<std::span<const double>> values = file.numericArray(arch + "." + key);
    if (!values || !matches(*values)) mismatched += (mismatched.empty() ? "" : ", ") + std::string(key);
  };
  expect("block_count", geometry.layers + metadata.unsignedValue(arch + ".nextn_predict_layers").value_or(0));
  expect("embedding_length", geometry.hiddenSize);
  expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  expect("attention.head_count_kv", geometry.attentionKvHeads);
  expect("attention.key_length", geometry.attentionHeadDimension);
  expect("attention.value_length", geometry.attentionHeadDimension);
  // A GGUF whose RoPE rotates whole heads (Flint's) need not state the
  // rotated dimensions, but one that does states the target's.
  if (2 * geometry.rotaryPairs != geometry.attentionHeadDimension ||
      metadata.unsignedValue(arch + ".rope.dimension_count"))
    expect("rope.dimension_count", 2ull * geometry.rotaryPairs);
  expectFloat("rope.freq_base", geometry.rotaryTheta);
  expectFloat("attention.layer_norm_rms_epsilon", SPLASH_RMS_EPSILON);
  if (const auto scaling = metadata.stringValue(arch + ".rope.scaling.type"); scaling && *scaling != "none")
    mismatched += (mismatched.empty() ? "" : ", ") + std::string("rope.scaling.type ") + *scaling + " (expected none)";
  if (geometry.gdnLayerCount()) {
    expect("full_attention_interval", geometry.fullAttentionPeriod);
    expect("ssm.conv_kernel", kGdnConvolutionTaps);
    expect("ssm.group_count", geometry.gdnKeyHeads);
    expect("ssm.time_step_rank", geometry.gdnValueHeads);
    expect("ssm.state_size", geometry.gdnHeadDimension);
    expect("ssm.inner_size", uint64_t{geometry.gdnValueHeads} * geometry.gdnHeadDimension);
  }
  withFamily(architecture, [&](auto family) { family.metadata(expect, expectFloat, expectArray, geometry); });
  if (geometry.ffnKind == QwenFfnKind::SparseMoe) {
    expect("expert_count", geometry.experts);
    expect("expert_used_count", geometry.expertsPerToken);
    expect("expert_feed_forward_length", geometry.expertIntermediateSize);
    expect("expert_shared_feed_forward_length",
           uint64_t{geometry.sharedExpertSlices} * geometry.expertIntermediateSize);
  } else {
    expect("feed_forward_length", geometry.intermediateSize);
  }
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
}

} // namespace

std::vector<Image> planImages(const GgufFile &file, std::string_view architecture,
                              const QwenTargetDimensions &geometry, PlanOptions options) {
  requireMetadata(file, architecture, geometry);
  std::vector<Image> images = withFamily(
      architecture, [&](auto family) { return familyImages<decltype(family)>(file, geometry, options); });
  if (file.rotation()) requireRotation(file, geometry, images);
  return images;
}

StreamedRows tableRows(const GgufFile &file, const std::string &name, uint64_t columns) {
  const GgufTensor &tensor = file.require(name);
  if (file.rotation()) throw GgufError("the tables of a rotated GGUF do not stream: " + name);
  // A table's image holds its rows as stored (storedRowBytes), but for
  // NVFP4's, which the writer rebuilds.
  if (!quantizedType(tensor.type) || tensor.type == ggml::kNVFP4)
    throw GgufError("a table of type " + ggmlTypeName(tensor.type) + " does not stream: " + name);
  if (tensor.columns() != columns || !tensor.rows()) throw GgufError("unexpected shape for " + name);
  const uint32_t format = imageFormat(tensor.type);
  const uint64_t rowBytes = storedRowBytes(tensor);
  if (!rowBytes || rowBytes > UINT32_MAX || tensor.rows() * rowBytes != tensor.bytes)
    throw GgufError("unexpected size for " + name);
  const WeightSource &source = file.source(tensor.file);
  if (tensor.offset > source.bytes() - source.dataOffset() ||
      tensor.bytes > source.bytes() - source.dataOffset() - tensor.offset)
    throw GgufError(name + " lies outside its file");
  StreamedRows rows;
  rows.tensor = name;
  rows.file = tensor.file;
  rows.offset = source.dataOffset() + tensor.offset;
  rows.rows = tensor.rows();
  rows.rowBytes = static_cast<uint32_t>(rowBytes);
  rows.format = format;
  return rows;
}

} // namespace splash::model::gguf
