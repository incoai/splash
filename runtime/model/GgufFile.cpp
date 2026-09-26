#include "model/GgufFile.hpp"

#include "metal/abi/Gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace splash::model {
namespace {

uint64_t checkedMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw GgufError("GGUF size overflows uint64");
  return a * b;
}

// (id, name, block elements, block bytes) of the ggml types this parser can
// size, as ggml-common.h defines them; a tensor of another type is rejected.
constexpr std::array<std::pair<uint32_t, GgmlTypeTraits>, 31> kTypes{{
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
    {142, {"PQ2_0", 128, 34}},
}};

enum ValueType : uint32_t {
  kUint8 = 0, kInt8 = 1, kUint16 = 2, kInt16 = 3, kUint32 = 4, kInt32 = 5,
  kFloat32 = 6, kBool = 7, kString = 8, kArray = 9, kUint64 = 10, kInt64 = 11,
  kFloat64 = 12,
};

// Reads the header sequentially through a buffer.
class Reader {
public:
  explicit Reader(const WeightSource &source) : source_(source) {}
  void bytes(void *destination, uint64_t count) {
    requireRemaining(count);
    auto *to = static_cast<uint8_t *>(destination);
    while (count) {
      if (position_ < bufferStart_ || position_ >= bufferStart_ + buffered_) fill();
      const uint64_t at = position_ - bufferStart_;
      const uint64_t part = std::min(count, buffered_ - at);
      std::memcpy(to, buffer_.data() + at, part);
      to += part;
      count -= part;
      position_ += part;
    }
  }
  void skip(uint64_t count) {
    requireRemaining(count);
    position_ += count;
  }
  template <class T> T scalar() {
    T value{};
    bytes(&value, sizeof value);
    return value;
  }
  std::string string() {
    const uint64_t length = scalar<uint64_t>();
    if (length > (8u << 20) || retainedStrings_ > (8u << 20) - length)
      throw GgufError("GGUF string metadata exceeds bounds");
    retainedStrings_ += length;
    std::string value(length, '\0');
    if (length) bytes(value.data(), length);
    return value;
  }
  void skipString() { skip(scalar<uint64_t>()); }
  [[nodiscard]] uint64_t position() const noexcept { return position_; }

private:
  void requireRemaining(uint64_t count) const {
    if (count > source_.bytes() - position_) throw GgufError("GGUF header is truncated");
  }
  void fill() {
    bufferStart_ = position_;
    buffered_ = std::min<uint64_t>(buffer_.size(), source_.bytes() - position_);
    readWeightBytes(source_.descriptor(), bufferStart_, std::span(buffer_).first(buffered_));
  }
  const WeightSource &source_;
  std::vector<uint8_t> buffer_ = std::vector<uint8_t>(1024 * 1024);
  uint64_t bufferStart_ = 0, buffered_ = 0;
  uint64_t position_ = 0;
  uint64_t retainedStrings_ = 0;
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

// The longest metadata array kept: the vision metadata read from it has one
// entry per block or channel, and tokenizer arrays are far longer.
constexpr uint64_t kMaximumKeptArray = 1024;
// Prism ML's rotation keys (GgufRotation), whose arrays are kept whole: a
// tensor name per rotated tensor and a sign per input of each width.
constexpr std::string_view kRotationPrefix = "prism.hadamard.";
constexpr uint64_t kMaximumRotationArray = uint64_t{1} << 20;

double numericValue(Reader &reader, uint32_t type) {
  switch (type) {
  case kUint8: return reader.scalar<uint8_t>();
  case kInt8: return reader.scalar<int8_t>();
  case kBool: return reader.scalar<uint8_t>() != 0;
  case kUint16: return reader.scalar<uint16_t>();
  case kInt16: return reader.scalar<int16_t>();
  case kUint32: return reader.scalar<uint32_t>();
  case kInt32: return reader.scalar<int32_t>();
  case kFloat32: return reader.scalar<float>();
  case kUint64: return double(reader.scalar<uint64_t>());
  case kInt64: return double(reader.scalar<int64_t>());
  case kFloat64: return reader.scalar<double>();
  default: throw GgufError("unknown GGUF value type " + std::to_string(type));
  }
}

void skipValue(Reader &reader, uint32_t type, unsigned depth);

// Skips the count elements of an array whose contents are not kept (mostly
// the tokenizer's).
void skipArray(Reader &reader, uint32_t element, uint64_t count, unsigned depth) {
  if (element == kString || element == kArray) {
    for (uint64_t i = 0; i < count; ++i) skipValue(reader, element, depth + 1);
  } else {
    reader.skip(checkedMultiply(count, scalarBytes(element)));
  }
}

void skipValue(Reader &reader, uint32_t type, unsigned depth) {
  if (depth > 16) throw GgufError("GGUF metadata nesting is too deep");
  if (type == kString) {
    reader.skipString();
  } else if (type == kArray) {
    const uint32_t element = reader.scalar<uint32_t>();
    skipArray(reader, element, reader.scalar<uint64_t>(), depth);
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

GgufFile::GgufFile(WeightSource &source) : source_(source) {
  Reader reader(source);
  char magic[4];
  reader.bytes(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0) throw GgufError("not a GGUF file: " + source.path().string());
  const uint32_t version = reader.scalar<uint32_t>();
  if (version != 3) throw GgufError("unsupported GGUF version " + std::to_string(version));
  const uint64_t tensorCount = reader.scalar<uint64_t>();
  const uint64_t keyCount = reader.scalar<uint64_t>();
  if (tensorCount > 16384 || keyCount > 16384) throw GgufError("implausible GGUF header counts");
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
    case kFloat32: floats_[key] = reader.scalar<float>(); break;
    case kFloat64: floats_[key] = reader.scalar<double>(); break;
    case kArray: {
      // Small numeric arrays of any key are kept; strings and long arrays
      // (vocabularies, merges, token types) are skipped without reading them,
      // but for the rotation's.
      const uint32_t element = reader.scalar<uint32_t>();
      const uint64_t count = reader.scalar<uint64_t>();
      if (key.starts_with(kRotationPrefix)) {
        if (count > kMaximumRotationArray || element == kArray)
          throw GgufError("implausible rotation metadata: " + key);
        if (element == kString) {
          auto &names = names_[key];
          for (uint64_t j = 0; j < count; ++j) names.push_back(reader.string());
        } else {
          auto &values = arrays_[key];
          values.reserve(count);
          for (uint64_t j = 0; j < count; ++j) values.push_back(numericValue(reader, element));
        }
        break;
      }
      if (count > kMaximumKeptArray || element == kString || element == kArray) {
        skipArray(reader, element, count, 0);
        break;
      }
      auto &values = arrays_[key];
      values.reserve(count);
      for (uint64_t j = 0; j < count; ++j) values.push_back(numericValue(reader, element));
      break;
    }
    default: skipValue(reader, type, 0); break;
    }
  }
  const uint64_t alignment = unsignedValue("general.alignment").value_or(32);
  if (alignment == 0 || alignment > 65536 || (alignment & (alignment - 1)))
    throw GgufError("invalid GGUF alignment");
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
  const uint64_t padding = (alignment - headerEnd % alignment) % alignment;
  if (padding > source.bytes() - headerEnd) throw GgufError("GGUF data section is truncated");
  source.setDataOffset(headerEnd + padding);
  const uint64_t dataBytes = source.bytes() - source.dataOffset();
  for (const GgufTensor &tensor : tensors_) {
    if (tensor.offset % alignment) throw GgufError("tensor data is misaligned: " + tensor.name);
    if (tensor.offset > dataBytes || tensor.bytes > dataBytes - tensor.offset)
      throw GgufError("tensor data runs past the end of the file: " + tensor.name);
  }
  rotation_ = readRotation();
}

std::optional<GgufRotation> GgufFile::readRotation() const {
  // Every rotation key, which must be one this parser knows: another one
  // describes a rotation it cannot run.
  std::set<std::string, std::less<>> keys;
  const auto collect = [&](const auto &values) {
    for (const auto &entry : values)
      if (std::string_view(entry.first).starts_with(kRotationPrefix)) keys.insert(entry.first);
  };
  collect(unsigned_);
  collect(strings_);
  collect(floats_);
  collect(arrays_);
  collect(names_);
  if (keys.empty()) return std::nullopt;
  static constexpr std::string_view kKnown[] = {"version",       "block_size",   "transform",
                                                "axis",          "sign_mode",    "gdn_v_grouped",
                                                "weight_names",  "inverse_weight_names", "sign_widths",
                                                "sign_values"};
  for (const std::string &key : keys)
    if (std::find(std::begin(kKnown), std::end(kKnown), std::string_view(key).substr(kRotationPrefix.size())) ==
        std::end(kKnown))
      throw GgufError("unsupported rotation metadata: " + key);
  const auto key = [](std::string_view name) { return std::string(kRotationPrefix) + std::string(name); };
  if (unsignedValue(key("version")) != 1) throw GgufError("unsupported prism.hadamard.version");
  if (unsignedValue(key("block_size")) != GGUF_ROTATION_BLOCK ||
      stringValue(key("transform")) != "normalized-sylvester-walsh-hadamard" ||
      stringValue(key("axis")) != "input-last-dimension" || stringValue(key("sign_mode")) != "explicit")
    throw GgufError("unsupported rotation: the kernels run explicit signs and normalized Sylvester "
                    "Walsh-Hadamard blocks of " + std::to_string(GGUF_ROTATION_BLOCK) + " inputs");
  GgufRotation rotation;
  rotation.valueHeadsGrouped = unsignedValue(key("gdn_v_grouped")).value_or(0) != 0;
  const auto widths = numericArray(key("sign_widths")), values = numericArray(key("sign_values"));
  if (!widths || !values || widths->empty()) throw GgufError("rotation signs are missing");
  size_t at = 0;
  for (const double width : *widths) {
    if (!(width > 0) || width != std::floor(width) || width > std::numeric_limits<uint32_t>::max() ||
        uint64_t(width) % GGUF_ROTATION_BLOCK || width > double(values->size() - at))
      throw GgufError("invalid rotation sign width");
    auto [signs, inserted] = rotation.signs.emplace(uint32_t(width), std::vector<int8_t>{});
    if (!inserted) throw GgufError("duplicate rotation sign width");
    signs->second.reserve(uint32_t(width));
    for (uint32_t i = 0; i < uint32_t(width); ++i, ++at) {
      const double sign = (*values)[at];
      if (sign != 1 && sign != -1) throw GgufError("rotation signs are +1 or -1");
      signs->second.push_back(int8_t(sign));
    }
  }
  if (at != values->size()) throw GgufError("rotation signs do not match their widths");
  const auto named = [&](std::string_view name, std::set<std::string, std::less<>> &into) {
    const auto found = names_.find(key(name));
    if (found == names_.end()) return;
    for (const std::string &tensorName : found->second) {
      const GgufTensor &tensor = require(tensorName);
      if (tensor.dims.size() != 2 || tensor.columns() > std::numeric_limits<uint32_t>::max() ||
          !rotation.signs.contains(uint32_t(tensor.columns())))
        throw GgufError("no rotation signs of the width of " + tensorName);
      if (!into.insert(tensorName).second) throw GgufError("rotation names a tensor twice: " + tensorName);
    }
  };
  named("weight_names", rotation.weights);
  named("inverse_weight_names", rotation.tables);
  if (rotation.weights.empty()) throw GgufError("the rotation names no weights");
  for (const std::string &name : rotation.tables)
    if (rotation.weights.contains(name)) throw GgufError("rotation names a tensor twice: " + name);
  return rotation;
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

std::optional<double> GgufFile::floatValue(std::string_view key) const {
  const auto it = floats_.find(key);
  return it == floats_.end() ? std::nullopt : std::optional<double>(it->second);
}
std::optional<std::span<const double>> GgufFile::numericArray(std::string_view key) const {
  const auto it = arrays_.find(key);
  if (it == arrays_.end()) return std::nullopt;
  return std::span<const double>(it->second);
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
