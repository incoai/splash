#include "model/GgufImage.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>

namespace splash::model::gguf {
namespace {

// Per-32-weight plane layout of every supported type (convert_gguf_to_splash.py FMT).
constexpr std::array<FormatLayout, 8> kFormats{{
    {KQ_FMT_Q4K, ggml::kQ4_K, 256, 144, 16, 0, 16, 8, 1},
    {KQ_FMT_IQ4XS, ggml::kIQ4_XS, 256, 136, 16, 0, 8, 8, 0},
    {KQ_FMT_IQ4NL, ggml::kIQ4_NL, 32, 18, 16, 0, 2, 1, 1},
    {KQ_FMT_Q5K, ggml::kQ5_K, 256, 176, 16, 4, 16, 8, 1},
    {KQ_FMT_Q6K, ggml::kQ6_K, 256, 210, 16, 8, 20, 8, 1},
    {KQ_FMT_Q3K, ggml::kQ3_K, 256, 110, 8, 4, 16, 8, 1},
    {KQ_FMT_Q80, ggml::kQ8_0, 32, 34, 32, 0, 2, 1, 1},
    {KQ_FMT_IQ3S, ggml::kIQ3_S, 256, 110, 16, 0, 2, 8, 1},
}};

constexpr uint32_t kNoPermute = 0xFFFFFFFFu;

uint64_t alignUp(uint64_t value) {
  return (value + kSectionAlignment - 1) / kSectionAlignment * kSectionAlignment;
}

uint16_t bfloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
  return static_cast<uint16_t>(bits);
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

std::vector<uint8_t> toBfloat16(const std::vector<float> &values) {
  std::vector<uint8_t> out;
  out.reserve(values.size() * 2);
  for (float value : values) {
    const uint16_t bits = bfloat16(value);
    out.push_back(static_cast<uint8_t>(bits));
    out.push_back(static_cast<uint8_t>(bits >> 8));
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

  void bfloatNorm(const char *name, uint64_t elements) {
    const GgufTensor &tensor = file_.require(name);
    if (tensor.elements() != elements) throw GgufError("unexpected shape for " + tensor.name);
    fill(toBfloat16(readFloats(file_, tensor)));
  }

  // Quantized rows [N, K] repacked into planes; rows >= permuteFrom come from
  // llama.cpp's tiled value-head order.
  void quantized(const GgufTensor &tensor, uint64_t rows, uint64_t columns,
                 uint32_t permuteFrom = kNoPermute, uint32_t headRows = 0) {
    const FormatLayout *layout = formatLayout(tensor.type);
    if (!layout) throw GgufError("unsupported tensor type " + ggmlTypeName(tensor.type) + " for " + tensor.name);
    if (tensor.rows() != rows || tensor.columns() != columns)
      throw GgufError("unexpected shape for " + tensor.name);
    if (rows % 256 || columns % 256) throw GgufError("tensor is not tile aligned: " + tensor.name);
    const uint64_t groups = columns / 32;
    const uint64_t plane0 = rows * groups * layout->p0;
    const uint64_t plane1 = rows * groups * layout->p1;
    const uint64_t meta = rows * (groups / layout->metaGroups) * layout->metaBytes;
    descriptor(layout->ggmlType, rows, columns, layout->p0, layout->p1, layout->metaBytes,
               layout->metaGroups, layout->interleave, plane0, plane1, meta);
    Repack repack;
    repack.params.rows = static_cast<uint32_t>(rows);
    repack.params.input_size = static_cast<uint32_t>(columns);
    repack.params.fmt = layout->fmt;
    repack.params.src_row_bytes = static_cast<uint32_t>(columns / layout->blockElements * layout->blockBytes);
    repack.params.dst_plane0 = offset32(section(plane0));
    repack.params.dst_plane1 = plane1 ? offset32(section(plane1)) : 0;
    repack.params.dst_meta = offset32(section(meta));
    repack.params.permute_from_row = permuteFrom;
    repack.params.permute_head_rows = headRows;
    repack.params.permute_group_heads = geometry_.gdnKeyHeads;
    repack.params.permute_groups = geometry_.gdnValueHeads / geometry_.gdnKeyHeads;
    repack.sourceOffset = file_.absoluteOffset(tensor);
    repack.sourceBytes = tensor.bytes;
    image_.sourceBegin = std::min(image_.sourceBegin, repack.sourceOffset);
    image_.sourceEnd = std::max(image_.sourceEnd, repack.sourceOffset + repack.sourceBytes);
    image_.repacks.push_back(repack);
  }

  // beta (48 rows) | alpha (48 rows) | zeros as one 256-row Q8_0 tensor, rows in
  // grouped head order; built on the CPU (half a megabyte).
  void alphaBeta(const GgufTensor &beta, const GgufTensor &alpha) {
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    for (const GgufTensor *t : {&beta, &alpha})
      if (t->type != ggml::kQ8_0 || t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be Q8_0 [" + std::to_string(heads) + ", hidden]: " + t->name);
    const uint32_t groups = hidden / 32, rows = 256;
    const uint64_t plane0 = uint64_t{rows} * groups * 32, meta = uint64_t{rows} * groups * 2;
    descriptor(ggml::kQ8_0, rows, hidden, 32, 0, 2, 1, 1, plane0, 0, meta);
    std::vector<uint8_t> betaBytes = readBytes(file_, beta), alphaBytes = readBytes(file_, alpha);
    std::vector<uint8_t> plane(plane0, 0), metaBytes(meta, 0);
    const uint32_t groupHeads = geometry_.gdnKeyHeads, valueGroups = heads / groupHeads;
    for (uint32_t n = 0; n < 2 * heads; ++n) {
      const std::vector<uint8_t> &source = n < heads ? betaBytes : alphaBytes;
      const uint32_t row = sourceHead(n % heads, groupHeads, valueGroups);
      for (uint32_t g = 0; g < groups; ++g) {
        const uint8_t *block = source.data() + (size_t(row) * groups + g) * 34;
        const size_t tile = (size_t(g) * 256 + n);
        std::memcpy(plane.data() + tile * 32, block + 2, 32);
        std::memcpy(metaBytes.data() + tile * 2, block, 2);
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
    descriptor(tensor.type, tensor.rows(), tensor.columns(), 0, 0, 0, 0, 0, tensor.bytes, 0, 0);
    Copy copy;
    copy.params.dst_offset = offset32(section(tensor.bytes));
    copy.params.bytes = static_cast<uint32_t>(tensor.bytes);
    copy.sourceOffset = file_.absoluteOffset(tensor);
    copy.sourceBytes = tensor.bytes;
    image_.sourceBegin = std::min(image_.sourceBegin, copy.sourceOffset);
    image_.sourceEnd = std::max(image_.sourceEnd, copy.sourceOffset + copy.sourceBytes);
    image_.copies.push_back(copy);
  }

  Image finish() {
    image_.bytes = alignUp(cursor_);
    if (image_.repacks.empty() && image_.copies.empty()) image_.sourceBegin = image_.sourceEnd = 0;
    return std::move(image_);
  }

private:
  void descriptor(uint32_t type, uint64_t rows, uint64_t columns, uint32_t p0, uint32_t p1,
                  uint32_t metaBytes, uint32_t metaGroups, uint32_t interleave,
                  uint64_t plane0Bytes, uint64_t plane1Bytes, uint64_t metaTotal) {
    std::vector<uint8_t> bytes;
    for (uint32_t word : {type, static_cast<uint32_t>(rows), static_cast<uint32_t>(columns), p0, p1,
                          metaBytes, metaGroups, interleave})
      appendLittle32(bytes, word);
    appendLittle64(bytes, plane0Bytes);
    appendLittle64(bytes, plane1Bytes);
    appendLittle64(bytes, metaTotal);
    bytes.resize(64, 0);
    fill(std::move(bytes));
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

const FormatLayout *formatLayout(uint32_t ggmlType) noexcept {
  for (const FormatLayout &layout : kFormats)
    if (layout.ggmlType == ggmlType) return &layout;
  return nullptr;
}

ImagePlanner::ImagePlanner(const GgufFile &file, TargetGeometry geometry)
    : file_(file), geometry_(geometry) {
  if (file.architecture() != "qwen35")
    throw GgufError("GGUF architecture is " + file.architecture() + ", expected qwen35");
  const uint64_t blocks = file.unsignedValue("qwen35.block_count").value_or(0);
  const uint64_t nextn = file.unsignedValue("qwen35.nextn_predict_layers").value_or(0);
  if (blocks != geometry.layers + nextn)
    throw GgufError("GGUF has " + std::to_string(blocks) + " blocks, expected " +
                    std::to_string(geometry.layers) + " layers plus " + std::to_string(nextn) + " MTP");
  if (file.unsignedValue("qwen35.embedding_length").value_or(0) != geometry.hiddenSize)
    throw GgufError("GGUF embedding length does not match the target");
  // Whole-file type check first so one error names every unsupported tensor.
  std::string unsupported;
  auto check = [&](const std::string &name, bool embedding = false) {
    const GgufTensor *tensor = file.find(name);
    if (!tensor) {
      unsupported += (unsupported.empty() ? "" : ", ") + name + " (missing)";
      return;
    }
    const bool ok = embedding ? tensor->type == ggml::kQ4_K || tensor->type == ggml::kQ6_K ||
                                    tensor->type == ggml::kQ8_0
                              : formatLayout(tensor->type) != nullptr;
    if (!ok) unsupported += (unsupported.empty() ? "" : ", ") + name + " (" + ggmlTypeName(tensor->type) + ")";
  };
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
    const std::string p = prefix(layer);
    if (geometry.isFullAttentionLayer(layer)) {
      for (const char *name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"})
        check(p + name);
    } else {
      for (const char *name : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"}) check(p + name);
      for (const char *name : {"ssm_alpha.weight", "ssm_beta.weight"}) {
        const GgufTensor *t = file.find(p + name);
        if (!t || t->type != ggml::kQ8_0)
          unsupported += (unsupported.empty() ? "" : ", ") + p + name + (t ? " (" + ggmlTypeName(t->type) + ")" : " (missing)");
      }
    }
    for (const char *name : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) check(p + name);
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
  b.bfloatNorm((p + "attn_norm.weight").c_str(), g.hiddenSize);
  if (full) {
    b.quantized(file_.require(p + "attn_q.weight"), 2ull * g.attentionHeadDimension * (g.attentionWidth / g.attentionHeadDimension), g.hiddenSize);
    b.quantized(file_.require(p + "attn_k.weight"), file_.require(p + "attn_k.weight").rows(), g.hiddenSize);
    b.quantized(file_.require(p + "attn_v.weight"), file_.require(p + "attn_v.weight").rows(), g.hiddenSize);
    b.bfloatNorm((p + "attn_q_norm.weight").c_str(), g.attentionHeadDimension);
    b.bfloatNorm((p + "attn_k_norm.weight").c_str(), g.attentionHeadDimension);
    b.quantized(file_.require(p + "attn_output.weight"), g.hiddenSize, g.attentionWidth);
  } else {
    const uint32_t valueRows = g.gdnValueHeads * g.gdnHeadDimension;       // 6144
    const uint32_t keyRows = g.convolutionDimension - valueRows;             // 4096 (q and k)
    const uint32_t groups = g.gdnValueHeads / g.gdnKeyHeads;
    b.quantized(file_.require(p + "attn_qkv.weight"), g.convolutionDimension, g.hiddenSize, keyRows, g.gdnHeadDimension);
    b.quantized(file_.require(p + "attn_gate.weight"), valueRows, g.hiddenSize, 0, g.gdnHeadDimension);
    b.alphaBeta(file_.require(p + "ssm_beta.weight"), file_.require(p + "ssm_alpha.weight"));
    const GgufTensor &conv = file_.require(p + "ssm_conv1d.weight");
    if (conv.elements() != uint64_t{g.convolutionDimension} * 4) throw GgufError("unexpected shape for " + conv.name);
    b.fill(toBfloat16(unreorderRows(readFloats(file_, conv), 4, keyRows, g.gdnHeadDimension, g.gdnKeyHeads, groups)));
    const GgufTensor &decay = file_.require(p + "ssm_a");
    if (decay.elements() != g.gdnValueHeads) throw GgufError("unexpected shape for " + decay.name);
    std::vector<float> decayValues = unreorderRows(readFloats(file_, decay), 1, 0, 1, g.gdnKeyHeads, groups);
    std::vector<uint8_t> decayBytes(decayValues.size() * 4);
    std::memcpy(decayBytes.data(), decayValues.data(), decayBytes.size());
    b.fill(std::move(decayBytes));
    const GgufTensor &timeBias = file_.require(p + "ssm_dt.bias");
    if (timeBias.elements() != g.gdnValueHeads) throw GgufError("unexpected shape for " + timeBias.name);
    b.fill(toBfloat16(unreorderRows(readFloats(file_, timeBias), 1, 0, 1, g.gdnKeyHeads, groups)));
    b.bfloatNorm((p + "ssm_norm.weight").c_str(), g.gdnHeadDimension);
    b.quantized(file_.require(p + "ssm_out.weight"), g.hiddenSize, valueRows);
  }
  b.bfloatNorm((p + "post_attention_norm.weight").c_str(), g.hiddenSize);
  b.quantized(file_.require(p + "ffn_gate.weight"), g.intermediateSize, g.hiddenSize);
  b.quantized(file_.require(p + "ffn_up.weight"), g.intermediateSize, g.hiddenSize);
  b.quantized(file_.require(p + "ffn_down.weight"), g.hiddenSize, g.intermediateSize);
  return b.finish();
}

Image ImagePlanner::head() const {
  Builder b(file_, geometry_, "head.bin", geometry_.layers, 2);
  b.bfloatNorm("output_norm.weight", geometry_.hiddenSize);
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
