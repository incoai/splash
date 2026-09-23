#include "model/GgufImage.hpp"

#include "model/StateLayout.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <optional>

namespace splash::model::gguf {
namespace {

static_assert(kQuantFormats[GGUF_FMT_Q4K].ggml_type == ggml::kQ4_K &&
                  kQuantFormats[GGUF_FMT_IQ4XS].ggml_type == ggml::kIQ4_XS &&
                  kQuantFormats[GGUF_FMT_IQ4NL].ggml_type == ggml::kIQ4_NL &&
                  kQuantFormats[GGUF_FMT_Q5K].ggml_type == ggml::kQ5_K &&
                  kQuantFormats[GGUF_FMT_Q6K].ggml_type == ggml::kQ6_K &&
                  kQuantFormats[GGUF_FMT_Q3K].ggml_type == ggml::kQ3_K &&
                  kQuantFormats[GGUF_FMT_Q80].ggml_type == ggml::kQ8_0 &&
                  kQuantFormats[GGUF_FMT_IQ3S].ggml_type == ggml::kIQ3_S,
              "format table types are the GGUF type ids");
static_assert(GGUF_TYPE_F32 == ggml::kF32, "float segments carry the GGUF type id");

constexpr uint32_t kNoPermute = 0xFFFFFFFFu;

uint64_t alignUp(uint64_t value) {
  return (value + kSectionAlignment - 1) / kSectionAlignment * kSectionAlignment;
}

void appendLittle32(std::vector<uint8_t> &out, uint32_t value) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
void appendLittle64(std::vector<uint8_t> &out, uint64_t value) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

// llama.cpp stores value-head-major tensors in tiled order (group * keyHeads +
// head); splash uses the grouped order. Destination head h maps to source head
// (h % groups) * groupHeads + h / groups.
uint32_t sourceHead(uint32_t destinationHead, uint32_t groupHeads, uint32_t groups) {
  return (destinationHead % groups) * groupHeads + destinationHead / groups;
}

std::vector<uint8_t> readBytes(const GgufFile &file, const GgufTensor &tensor) {
  std::ifstream stream(file.path(), std::ios::binary);
  if (!stream) throw GgufError("cannot open GGUF file: " + file.path().string());
  std::vector<uint8_t> bytes(tensor.bytes);
  stream.seekg(static_cast<std::streamoff>(file.absoluteOffset(tensor)));
  stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw GgufError("cannot read tensor " + tensor.name);
  return bytes;
}

std::vector<float> readFloats(const GgufFile &file, const GgufTensor &tensor) {
  if (tensor.type != ggml::kF32) throw GgufError("expected an F32 tensor: " + tensor.name);
  std::vector<uint8_t> bytes = readBytes(file, tensor);
  std::vector<float> values(bytes.size() / 4);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

// Reorders rows [from, end) in blocks of headRows from tiled to grouped order.
template <class T>
std::vector<T> unreorderRows(std::vector<T> values, uint32_t rowWidth, uint32_t from,
                             uint32_t headRows, uint32_t groupHeads, uint32_t groups) {
  std::vector<T> out = values;
  const uint32_t rows = static_cast<uint32_t>(values.size() / rowWidth);
  for (uint32_t n = from; n < rows; ++n) {
    const uint32_t head = (n - from) / headRows, element = (n - from) % headRows;
    const uint32_t source = from + sourceHead(head, groupHeads, groups) * headRows + element;
    std::copy_n(values.begin() + size_t(source) * rowWidth, rowWidth, out.begin() + size_t(n) * rowWidth);
  }
  return out;
}

// The GDN kernels read the convolution weights and the time bias as bf16,
// which holds them exactly when they come from a bf16 checkpoint. Any other
// value would be rounded silently, so the tensor is refused by name.
std::vector<uint8_t> exactBfloat16(const std::vector<float> &values, const std::string &name) {
  std::vector<uint8_t> out;
  out.reserve(values.size() * 2);
  for (float value : values) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    if (bits & 0xFFFFu) throw GgufError(name + " is not bf16-exact; it needs an F32 path");
    out.push_back(static_cast<uint8_t>(bits >> 16));
    out.push_back(static_cast<uint8_t>(bits >> 24));
  }
  return out;
}

class Builder {
public:
  Builder(const GgufFile &file, const TargetGeometry &geometry, std::string name,
          uint32_t layer, uint32_t type)
      : file_(file), geometry_(geometry) {
    image_.name = std::move(name);
    image_.layer = layer;
    image_.type = type;
    std::vector<uint8_t> header(kImageMagic, kImageMagic + 8);
    appendLittle32(header, layer);
    appendLittle32(header, type);
    image_.fills.push_back({0, std::move(header)});
    cursor_ = 16;
  }

  uint64_t section(uint64_t bytes) {
    if (!bytes) throw GgufError("empty image section in " + image_.name);
    const uint64_t start = alignUp(cursor_);
    cursor_ = start + bytes;
    return start;
  }

  void fill(std::vector<uint8_t> bytes) {
    const uint64_t offset = section(bytes.size());
    image_.fills.push_back({offset, std::move(bytes)});
  }

  // A norm as stored: F32, which the norm kernels read unrounded, as
  // llama.cpp does (ops::NormWeights).
  void floatNorm(const char *name, uint64_t elements) {
    const GgufTensor &tensor = file_.require(name);
    if (tensor.elements() != elements) throw GgufError("unexpected shape for " + tensor.name);
    if (tensor.type != ggml::kF32) throw GgufError("expected an F32 tensor: " + tensor.name);
    fill(readBytes(file_, tensor));
  }

  // Quantized rows [N, K] repacked into planes; rows >= permuteFrom come from
  // llama.cpp's tiled value-head order.
  void quantized(const GgufTensor &tensor, uint64_t rows, uint64_t columns,
                 uint32_t permuteFrom = kNoPermute, uint32_t headRows = 0) {
    const uint32_t format = gguf_format_of(tensor.type);
    if (format == GGUF_FMT_COUNT)
      throw GgufError("unsupported tensor type " + ggmlTypeName(tensor.type) + " for " + tensor.name);
    const QuantFormat &layout = kQuantFormats[format];
    if (tensor.rows() != rows || tensor.columns() != columns)
      throw GgufError("unexpected shape for " + tensor.name);
    if (rows % 256 || columns % 256) throw GgufError("tensor is not tile aligned: " + tensor.name);
    const uint64_t rowBytes = columns / layout.block_elements * layout.block_bytes;
    if (tensor.bytes != rows * rowBytes) throw GgufError("unexpected size for " + tensor.name);
    const uint64_t groups = columns / 32;
    const uint64_t plane0 = rows * groups * layout.plane0_bytes;
    const uint64_t plane1 = rows * groups * layout.plane1_bytes;
    const uint64_t meta = rows * (groups / layout.meta_groups) * layout.meta_bytes;
    descriptor(layout.ggml_type, rows, columns, layout.plane0_bytes, layout.plane1_bytes,
               layout.meta_bytes, layout.meta_groups, plane0, plane1, meta);
    Repack repack;
    repack.params.rows = static_cast<uint32_t>(rows);
    repack.params.input_size = static_cast<uint32_t>(columns);
    repack.params.fmt = format;
    repack.params.src_row_bytes = static_cast<uint32_t>(rowBytes);
    repack.params.dst_plane0 = offset32(section(plane0));
    repack.params.dst_plane1 = plane1 ? offset32(section(plane1)) : 0;
    repack.params.dst_meta = offset32(section(meta));
    repack.params.permute_from_row = permuteFrom;
    repack.params.permute_head_rows = headRows;
    repack.params.permute_group_heads = geometry_.gdnKeyHeads;
    repack.params.permute_groups = geometry_.gdnValueHeads / geometry_.gdnKeyHeads;
    repack.sourceOffset = file_.absoluteOffset(tensor);
    repack.sourceBytes = tensor.bytes;
    image_.repacks.push_back(repack);
  }

  // beta (48 rows) | alpha (48 rows) | zeros as one 256-row Q8_0 tensor, rows in
  // grouped head order; built on the CPU (half a megabyte).
  void alphaBeta(const GgufTensor &beta, const GgufTensor &alpha) {
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    for (const GgufTensor *t : {&beta, &alpha})
      if (t->type != ggml::kQ8_0 || t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be Q8_0 [" + std::to_string(heads) + ", hidden]: " + t->name);
    const QuantFormat &q8 = kQuantFormats[GGUF_FMT_Q80];
    const uint32_t groups = hidden / 32, rows = 256;
    const uint64_t plane0 = uint64_t{rows} * groups * q8.plane0_bytes;
    const uint64_t meta = uint64_t{rows} * groups * q8.meta_bytes;
    descriptor(q8.ggml_type, rows, hidden, q8.plane0_bytes, q8.plane1_bytes, q8.meta_bytes,
               q8.meta_groups, plane0, 0, meta);
    std::vector<uint8_t> betaBytes = readBytes(file_, beta), alphaBytes = readBytes(file_, alpha);
    std::vector<uint8_t> plane(plane0, 0), metaBytes(meta, 0);
    const uint32_t groupHeads = geometry_.gdnKeyHeads, valueGroups = heads / groupHeads;
    for (uint32_t n = 0; n < 2 * heads; ++n) {
      const std::vector<uint8_t> &source = n < heads ? betaBytes : alphaBytes;
      const uint32_t row = sourceHead(n % heads, groupHeads, valueGroups);
      for (uint32_t g = 0; g < groups; ++g) {
        const uint8_t *block = source.data() + (size_t(row) * groups + g) * q8.block_bytes;
        const uint64_t tile = quant_tile_index(n, g, groups);
        for (uint32_t e = 0; e < 32; ++e) plane[tile * q8.plane0_bytes + quant_slot(e)] = block[2 + e];
        std::memcpy(metaBytes.data() + tile * q8.meta_bytes, block, 2);
      }
    }
    fill(std::move(plane));
    fill(std::move(metaBytes));
  }

  void embeddingRows(const GgufTensor &tensor) {
    if (tensor.type != ggml::kQ4_K && tensor.type != ggml::kQ6_K && tensor.type != ggml::kQ8_0)
      throw GgufError("unsupported token embedding type " + ggmlTypeName(tensor.type));
    if (tensor.rows() != geometry_.vocabularySize || tensor.columns() != geometry_.hiddenSize)
      throw GgufError("unexpected shape for " + tensor.name);
    copiedRows(tensor);
  }

  // An F32 tensor [rows, columns] as stored (the MoE router and the
  // shared-expert gate, which llama.cpp keeps unquantized).
  void floatTensor(const GgufTensor &tensor, uint64_t rows, uint64_t columns) {
    if (tensor.type != ggml::kF32 || tensor.rows() != rows || tensor.columns() != columns)
      throw GgufError("expected an F32 [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + tensor.name);
    copiedRows(tensor);
  }

  // F32 beta (value heads rows) | alpha (value heads rows), rows in grouped
  // head order, as one float tensor; built on the CPU (512 KB for the 35B).
  void floatAlphaBeta(const GgufTensor &beta, const GgufTensor &alpha) {
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    const uint32_t groupHeads = geometry_.gdnKeyHeads, valueGroups = heads / groupHeads;
    std::vector<float> rows;
    for (const GgufTensor *t : {&beta, &alpha}) {
      if (t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be [" + std::to_string(heads) + ", hidden]: " + t->name);
      const std::vector<float> grouped = unreorderRows(readFloats(file_, *t), hidden, 0, 1, groupHeads, valueGroups);
      rows.insert(rows.end(), grouped.begin(), grouped.end());
    }
    std::vector<uint8_t> bytes(rows.size() * sizeof(float));
    std::memcpy(bytes.data(), rows.data(), bytes.size());
    descriptor(ggml::kF32, 2ull * heads, hidden, 0, 0, 0, 0, bytes.size(), 0, 0);
    fill(std::move(bytes));
  }

  Image finish() {
    image_.bytes = alignUp(cursor_);
    return std::move(image_);
  }

private:
  // Word 7 is reserved (0).
  void descriptor(uint32_t type, uint64_t rows, uint64_t columns, uint32_t p0, uint32_t p1,
                  uint32_t metaBytes, uint32_t metaGroups,
                  uint64_t plane0Bytes, uint64_t plane1Bytes, uint64_t metaTotal) {
    std::vector<uint8_t> bytes;
    for (uint32_t word : {type, static_cast<uint32_t>(rows), static_cast<uint32_t>(columns), p0, p1,
                          metaBytes, metaGroups, 0u})
      appendLittle32(bytes, word);
    appendLittle64(bytes, plane0Bytes);
    appendLittle64(bytes, plane1Bytes);
    appendLittle64(bytes, metaTotal);
    bytes.resize(64, 0);
    fill(std::move(bytes));
  }

  // A tensor's rows as stored, copied by the GPU from the mapped file.
  void copiedRows(const GgufTensor &tensor) {
    descriptor(tensor.type, tensor.rows(), tensor.columns(), 0, 0, 0, 0, tensor.bytes, 0, 0);
    Copy copy;
    copy.params.dst_offset = offset32(section(tensor.bytes));
    copy.params.bytes = static_cast<uint32_t>(tensor.bytes);
    copy.sourceOffset = file_.absoluteOffset(tensor);
    copy.sourceBytes = tensor.bytes;
    image_.copies.push_back(copy);
  }

  static uint32_t offset32(uint64_t offset) {
    if (offset > std::numeric_limits<uint32_t>::max()) throw GgufError("image exceeds 4 GiB");
    return static_cast<uint32_t>(offset);
  }

  const GgufFile &file_;
  const TargetGeometry &geometry_;
  Image image_;
  uint64_t cursor_ = 0;
};

std::string prefix(uint32_t layer) { return "blk." + std::to_string(layer) + "."; }

} // namespace

ImagePlanner::ImagePlanner(const GgufFile &file, TargetGeometry geometry)
    : file_(file), geometry_(geometry) {
  const std::string arch = geometry.architecture();
  if (file.architecture() != arch)
    throw GgufError("GGUF architecture is " + file.architecture() + ", but the package's target is " + arch);
  // The target geometry the metadata declares; one error names every mismatch.
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  expect("block_count", geometry.layers + file.unsignedValue(arch + ".nextn_predict_layers").value_or(0));
  expect("embedding_length", geometry.hiddenSize);
  expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  expect("attention.head_count_kv", geometry.attentionKvHeads);
  expect("attention.key_length", geometry.attentionHeadDimension);
  expect("attention.value_length", geometry.attentionHeadDimension);
  expect("full_attention_interval", geometry.fullAttentionPeriod);
  expect("ssm.conv_kernel", kGdnConvolutionTaps);
  expect("ssm.group_count", geometry.gdnKeyHeads);
  expect("ssm.time_step_rank", geometry.gdnValueHeads);
  expect("ssm.state_size", geometry.gdnHeadDimension);
  expect("ssm.inner_size", uint64_t{geometry.gdnValueHeads} * geometry.gdnHeadDimension);
  if (geometry.sparseMoe()) {
    expect("expert_count", geometry.experts);
    expect("expert_used_count", geometry.expertsPerToken);
    expect("expert_feed_forward_length", geometry.expertIntermediateSize);
    expect("expert_shared_feed_forward_length", geometry.expertIntermediateSize);
  } else {
    expect("feed_forward_length", geometry.intermediateSize);
  }
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
  // Whole-file type check first so one error names every unsupported tensor.
  std::string unsupported;
  const auto reject = [&](const std::string &name, const GgufTensor *tensor) {
    unsupported += (unsupported.empty() ? "" : ", ") + name +
                   (tensor ? " (" + ggmlTypeName(tensor->type) + ")" : " (missing)");
  };
  auto check = [&](const std::string &name, bool embedding = false) {
    const GgufTensor *tensor = file.find(name);
    const bool ok = tensor && (embedding ? tensor->type == ggml::kQ4_K || tensor->type == ggml::kQ6_K ||
                                               tensor->type == ggml::kQ8_0
                                         : gguf_format_of(tensor->type) != GGUF_FMT_COUNT);
    if (!ok) reject(name, tensor);
  };
  // Tensors llama.cpp keeps in F32, which run unrounded.
  auto checkFloat = [&](const std::string &name) {
    const GgufTensor *tensor = file.find(name);
    if (!tensor || tensor->type != ggml::kF32) reject(name, tensor);
  };
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
    const std::string p = prefix(layer);
    if (geometry.isFullAttentionLayer(layer)) {
      for (const char *name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"})
        check(p + name);
    } else {
      for (const char *name : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"}) check(p + name);
      // alpha/beta run in their stored format: both Q8_0 (one repacked
      // segment) or both F32 (one float segment).
      const GgufTensor *alpha = file.find(p + "ssm_alpha.weight"), *beta = file.find(p + "ssm_beta.weight");
      if (!alpha || !beta || alpha->type != beta->type ||
          (alpha->type != ggml::kQ8_0 && alpha->type != ggml::kF32)) {
        reject(p + "ssm_alpha.weight", alpha);
        reject(p + "ssm_beta.weight", beta);
      }
    }
    if (geometry.sparseMoe()) {
      for (const char *name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                               "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
        check(p + name);
      checkFloat(p + "ffn_gate_inp.weight");
      checkFloat(p + "ffn_gate_inp_shexp.weight");
    } else {
      for (const char *name : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) check(p + name);
    }
  }
  check("output.weight");
  check("token_embd.weight", true);
  if (!unsupported.empty()) throw GgufError("GGUF tensors this build cannot load: " + unsupported);
}

Image ImagePlanner::layer(uint32_t index) const {
  const TargetGeometry &g = geometry_;
  const std::string p = prefix(index);
  const bool full = g.isFullAttentionLayer(index);
  Builder b(file_, g, "layer-" + std::to_string(index) + ".bin", index, full ? 1u : 0u);
  b.floatNorm((p + "attn_norm.weight").c_str(), g.hiddenSize);
  if (full) {
    b.quantized(file_.require(p + "attn_q.weight"), 2ull * g.attentionHeadDimension * (g.attentionWidth / g.attentionHeadDimension), g.hiddenSize);
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.quantized(file_.require(p + "attn_k.weight"), kvRows, g.hiddenSize);
    b.quantized(file_.require(p + "attn_v.weight"), kvRows, g.hiddenSize);
    b.floatNorm((p + "attn_q_norm.weight").c_str(), g.attentionHeadDimension);
    b.floatNorm((p + "attn_k_norm.weight").c_str(), g.attentionHeadDimension);
    b.quantized(file_.require(p + "attn_output.weight"), g.hiddenSize, g.attentionWidth);
  } else {
    const uint32_t valueRows = g.gdnValueHeads * g.gdnHeadDimension;
    const uint32_t keyRows = g.convolutionDimension - valueRows;             // q and k
    const uint32_t groups = g.gdnValueHeads / g.gdnKeyHeads;
    b.quantized(file_.require(p + "attn_qkv.weight"), g.convolutionDimension, g.hiddenSize, keyRows, g.gdnHeadDimension);
    b.quantized(file_.require(p + "attn_gate.weight"), valueRows, g.hiddenSize, 0, g.gdnHeadDimension);
    const GgufTensor &beta = file_.require(p + "ssm_beta.weight"), &alpha = file_.require(p + "ssm_alpha.weight");
    if (beta.type == ggml::kF32) b.floatAlphaBeta(beta, alpha);
    else b.alphaBeta(beta, alpha);
    const GgufTensor &conv = file_.require(p + "ssm_conv1d.weight");
    if (conv.elements() != uint64_t{g.convolutionDimension} * 4) throw GgufError("unexpected shape for " + conv.name);
    b.fill(exactBfloat16(unreorderRows(readFloats(file_, conv), 4, keyRows, g.gdnHeadDimension, g.gdnKeyHeads, groups),
                         conv.name));
    const GgufTensor &decay = file_.require(p + "ssm_a");
    if (decay.elements() != g.gdnValueHeads) throw GgufError("unexpected shape for " + decay.name);
    std::vector<float> decayValues = unreorderRows(readFloats(file_, decay), 1, 0, 1, g.gdnKeyHeads, groups);
    std::vector<uint8_t> decayBytes(decayValues.size() * 4);
    std::memcpy(decayBytes.data(), decayValues.data(), decayBytes.size());
    b.fill(std::move(decayBytes));
    const GgufTensor &timeBias = file_.require(p + "ssm_dt.bias");
    if (timeBias.elements() != g.gdnValueHeads) throw GgufError("unexpected shape for " + timeBias.name);
    b.fill(exactBfloat16(unreorderRows(readFloats(file_, timeBias), 1, 0, 1, g.gdnKeyHeads, groups), timeBias.name));
    b.floatNorm((p + "ssm_norm.weight").c_str(), g.gdnHeadDimension);
    b.quantized(file_.require(p + "ssm_out.weight"), g.hiddenSize, valueRows);
  }
  b.floatNorm((p + "post_attention_norm.weight").c_str(), g.hiddenSize);
  if (g.sparseMoe()) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(file_.require(p + "ffn_gate_inp.weight"), routed, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_gate_exps.weight"), routed * width, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_up_exps.weight"), routed * width, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_down_exps.weight"), routed * g.hiddenSize, width);
    b.quantized(file_.require(p + "ffn_gate_shexp.weight"), width, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_up_shexp.weight"), width, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_down_shexp.weight"), g.hiddenSize, width);
    b.floatTensor(file_.require(p + "ffn_gate_inp_shexp.weight"), 1, g.hiddenSize);
  } else {
    b.quantized(file_.require(p + "ffn_gate.weight"), g.intermediateSize, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_up.weight"), g.intermediateSize, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_down.weight"), g.hiddenSize, g.intermediateSize);
  }
  return b.finish();
}

Image ImagePlanner::head() const {
  Builder b(file_, geometry_, "head.bin", geometry_.layers, 2);
  b.floatNorm("output_norm.weight", geometry_.hiddenSize);
  b.quantized(file_.require("output.weight"), geometry_.vocabularySize, geometry_.hiddenSize);
  return b.finish();
}

Image ImagePlanner::embedding() const {
  Builder b(file_, geometry_, "embedding.bin", geometry_.vocabularySize, geometry_.hiddenSize);
  b.embeddingRows(file_.require("token_embd.weight"));
  return b.finish();
}

uint64_t ImagePlanner::totalBytes() const {
  uint64_t total = head().bytes + embedding().bytes;
  for (uint32_t i = 0; i < geometry_.layers; ++i) total += layer(i).bytes;
  return total;
}

} // namespace splash::model::gguf
