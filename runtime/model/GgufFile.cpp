#include "model/GgufFile.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>

namespace splash::model {
namespace {

uint64_t checkedMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw GgufError("GGUF size overflows uint64");
  return a * b;
}

// (id, name, block elements, block bytes) for every ggml type that can appear
// in a Qwen3.8 GGUF; sizes follow ggml-common.h.
constexpr std::array<std::pair<uint32_t, GgmlTypeTraits>, 30> kTypes{{
    {0, {"F32", 1, 4}},         {1, {"F16", 1, 2}},         {2, {"Q4_0", 32, 18}},
    {3, {"Q4_1", 32, 20}},      {6, {"Q5_0", 32, 22}},      {7, {"Q5_1", 32, 24}},
    {8, {"Q8_0", 32, 34}},      {9, {"Q8_1", 32, 36}},      {10, {"Q2_K", 256, 84}},
    {11, {"Q3_K", 256, 110}},   {12, {"Q4_K", 256, 144}},   {13, {"Q5_K", 256, 176}},
    {14, {"Q6_K", 256, 210}},   {15, {"Q8_K", 256, 292}},   {16, {"IQ2_XXS", 256, 66}},
    {17, {"IQ2_XS", 256, 74}},  {18, {"IQ3_XXS", 256, 98}}, {19, {"IQ1_S", 256, 50}},
    {20, {"IQ4_NL", 32, 18}},   {21, {"IQ3_S", 256, 110}},  {22, {"IQ2_S", 256, 82}},
    {23, {"IQ4_XS", 256, 136}}, {24, {"I8", 1, 1}},         {25, {"I16", 1, 2}},
    {26, {"I32", 1, 4}},        {27, {"I64", 1, 8}},        {28, {"F64", 1, 8}},
    {29, {"IQ1_M", 256, 56}},   {30, {"BF16", 1, 2}},       {39, {"MXFP4", 32, 17}},
}};

enum ValueType : uint32_t {
  kUint8 = 0, kInt8 = 1, kUint16 = 2, kInt16 = 3, kUint32 = 4, kInt32 = 5,
  kFloat32 = 6, kBool = 7, kString = 8, kArray = 9, kUint64 = 10, kInt64 = 11,
  kFloat64 = 12,
};

class Reader {
public:
  Reader(const std::filesystem::path &path, uint64_t size) : stream_(path, std::ios::binary), size_(size) {
    if (!stream_) throw GgufError("cannot open GGUF file: " + path.string());
  }
  void bytes(void *destination, uint64_t count) {
    requireRemaining(count);
    if (count > std::numeric_limits<std::streamsize>::max())
      throw GgufError("GGUF field is too large");
    stream_.read(static_cast<char *>(destination), static_cast<std::streamsize>(count));
    if (!stream_) throw GgufError("GGUF header is truncated");
    position_ += count;
  }
  void skip(uint64_t count) {
    requireRemaining(count);
    stream_.seekg(static_cast<std::streamoff>(count), std::ios::cur);
    if (!stream_) throw GgufError("GGUF header is truncated");
    position_ += count;
  }
  template <class T> T scalar() {
    T value{};
    bytes(&value, sizeof value);
    return value;
  }
  std::string string() {
    const uint64_t length = scalar<uint64_t>();
    if (length > (64u << 20)) throw GgufError("GGUF string is too long");
    std::string value(length, '\0');
    if (length) bytes(value.data(), length);
    return value;
  }
  [[nodiscard]] uint64_t position() const noexcept { return position_; }

private:
  void requireRemaining(uint64_t count) const {
    if (count > size_ - position_ || count > std::numeric_limits<std::streamoff>::max())
      throw GgufError("GGUF header is truncated");
  }
  std::ifstream stream_;
  uint64_t size_;
  uint64_t position_ = 0;
};

uint64_t scalarBytes(uint32_t type) {
  switch (type) {
  case kUint8: case kInt8: case kBool: return 1;
  case kUint16: case kInt16: return 2;
  case kUint32: case kInt32: case kFloat32: return 4;
  case kUint64: case kInt64: case kFloat64: return 8;
  default: throw GgufError("unknown GGUF value type " + std::to_string(type));
  }
}

// Skips a value whose contents are not kept (arrays, mostly the tokenizer).
void skipValue(Reader &reader, uint32_t type, unsigned depth = 0) {
  if (depth > 16) throw GgufError("GGUF metadata nesting is too deep");
  if (type == kString) {
    (void)reader.string();
  } else if (type == kArray) {
    const uint32_t element = reader.scalar<uint32_t>();
    const uint64_t count = reader.scalar<uint64_t>();
    if (element == kString || element == kArray) {
      for (uint64_t i = 0; i < count; ++i) skipValue(reader, element, depth + 1);
    } else {
      reader.skip(checkedMultiply(count, scalarBytes(element)));
    }
  } else {
    reader.skip(scalarBytes(type));
  }
}

} // namespace

const GgmlTypeTraits *ggmlTypeTraits(uint32_t type) noexcept {
  for (const auto &[id, traits] : kTypes)
    if (id == type) return &traits;
  return nullptr;
}

std::string ggmlTypeName(uint32_t type) {
  const GgmlTypeTraits *traits = ggmlTypeTraits(type);
  return traits ? traits->name : "type-" + std::to_string(type);
}

uint64_t GgufTensor::rows() const {
  uint64_t rows = 1;
  for (size_t i = 1; i < dims.size(); ++i) rows = checkedMultiply(rows, dims[i]);
  return rows;
}

uint64_t GgufTensor::elements() const {
  uint64_t elements = 1;
  for (uint64_t dim : dims) elements = checkedMultiply(elements, dim);
  return elements;
}

GgufFile::GgufFile(std::filesystem::path path) : path_(std::move(path)) {
  std::error_code error;
  fileBytes_ = std::filesystem::file_size(path_, error);
  if (error) throw GgufError("cannot stat GGUF file: " + path_.string());
  Reader reader(path_, fileBytes_);
  char magic[4];
  reader.bytes(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0) throw GgufError("not a GGUF file: " + path_.string());
  const uint32_t version = reader.scalar<uint32_t>();
  if (version != 3) throw GgufError("unsupported GGUF version " + std::to_string(version));
  const uint64_t tensorCount = reader.scalar<uint64_t>();
  const uint64_t keyCount = reader.scalar<uint64_t>();
  if (tensorCount > 1u << 20 || keyCount > 1u << 20) throw GgufError("implausible GGUF header counts");
  for (uint64_t i = 0; i < keyCount; ++i) {
    const std::string key = reader.string();
    const uint32_t type = reader.scalar<uint32_t>();
    switch (type) {
    case kUint8: unsigned_[key] = reader.scalar<uint8_t>(); break;
    case kUint16: unsigned_[key] = reader.scalar<uint16_t>(); break;
    case kUint32: unsigned_[key] = reader.scalar<uint32_t>(); break;
    case kUint64: unsigned_[key] = reader.scalar<uint64_t>(); break;
    case kInt8: unsigned_[key] = static_cast<uint64_t>(reader.scalar<int8_t>()); break;
    case kInt16: unsigned_[key] = static_cast<uint64_t>(reader.scalar<int16_t>()); break;
    case kInt32: unsigned_[key] = static_cast<uint64_t>(reader.scalar<int32_t>()); break;
    case kInt64: unsigned_[key] = static_cast<uint64_t>(reader.scalar<int64_t>()); break;
    case kBool: unsigned_[key] = reader.scalar<uint8_t>() != 0; break;
    case kString: strings_[key] = reader.string(); break;
    default: skipValue(reader, type); break;
    }
  }
  if (auto alignment = unsignedValue("general.alignment")) {
    if (*alignment == 0 || *alignment > 65536 || (*alignment & (*alignment - 1)))
      throw GgufError("invalid GGUF alignment");
    alignment_ = static_cast<uint32_t>(*alignment);
  }
  architecture_ = stringValue("general.architecture").value_or("");
  tensors_.reserve(tensorCount);
  for (uint64_t i = 0; i < tensorCount; ++i) {
    GgufTensor tensor;
    tensor.name = reader.string();
    const uint32_t dimensions = reader.scalar<uint32_t>();
    if (dimensions == 0 || dimensions > 4) throw GgufError("invalid tensor rank for " + tensor.name);
    for (uint32_t d = 0; d < dimensions; ++d) {
      const uint64_t dim = reader.scalar<uint64_t>();
      if (!dim) throw GgufError("zero tensor dimension for " + tensor.name);
      tensor.dims.push_back(dim);
    }
    (void)tensor.elements(); // Reject overflowing shapes even when quantized bytes would fit.
    tensor.type = reader.scalar<uint32_t>();
    tensor.offset = reader.scalar<uint64_t>();
    const GgmlTypeTraits *traits = ggmlTypeTraits(tensor.type);
    if (!traits) throw GgufError("unknown ggml type " + std::to_string(tensor.type) + " for " + tensor.name);
    if (tensor.columns() % traits->blockElements)
      throw GgufError("tensor row is not block aligned: " + tensor.name);
    tensor.bytes = checkedMultiply(checkedMultiply(tensor.rows(), tensor.columns() / traits->blockElements),
                                   traits->blockBytes);
    if (!index_.emplace(tensor.name, tensors_.size()).second)
      throw GgufError("duplicate GGUF tensor: " + tensor.name);
    tensors_.push_back(std::move(tensor));
  }
  const uint64_t headerEnd = reader.position();
  const uint64_t padding = (alignment_ - headerEnd % alignment_) % alignment_;
  if (padding > fileBytes_ - headerEnd) throw GgufError("GGUF data section is truncated");
  dataOffset_ = headerEnd + padding;
  for (const GgufTensor &tensor : tensors_) {
    if (tensor.offset % alignment_) throw GgufError("tensor data is misaligned: " + tensor.name);
    if (tensor.offset > fileBytes_ - dataOffset_ ||
        tensor.bytes > fileBytes_ - dataOffset_ - tensor.offset)
      throw GgufError("tensor data runs past the end of the file: " + tensor.name);
  }
}

std::optional<uint64_t> GgufFile::unsignedValue(std::string_view key) const {
  auto it = unsigned_.find(key);
  if (it == unsigned_.end()) return std::nullopt;
  return it->second;
}

std::optional<std::string> GgufFile::stringValue(std::string_view key) const {
  auto it = strings_.find(key);
  if (it == strings_.end()) return std::nullopt;
  return it->second;
}

const GgufTensor *GgufFile::find(std::string_view name) const noexcept {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

const GgufTensor &GgufFile::require(std::string_view name) const {
  const GgufTensor *tensor = find(name);
  if (!tensor) throw GgufError("GGUF is missing tensor " + std::string(name));
  return *tensor;
}

} // namespace splash::model
