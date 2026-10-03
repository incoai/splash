#pragma once

// GGUF version 3 files for tests, laid out as llama.cpp writes them, and the
// value encoders that malformed metadata is built from.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::test::gguf {

using Bytes = std::vector<uint8_t>;

// Metadata value types.
inline constexpr uint32_t kUint32 = 4, kInt32 = 5, kFloat32 = 6, kBool = 7, kString = 8, kArray = 9, kUint64 = 10;
// The alignment of the tensor data and of each tensor in it when the file
// has no general.alignment key.
inline constexpr uint64_t kAlignment = 32;

template <class T> void append(Bytes &out, T value) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
  out.insert(out.end(), bytes, bytes + sizeof value);
}

inline void appendString(Bytes &out, std::string_view value) {
  append<uint64_t>(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

// A metadata key, its value type and its encoded value.
struct Key {
  std::string name;
  uint32_t type;
  Bytes value;
};

inline Key uint32Key(std::string name, uint32_t value) {
  Key key{std::move(name), kUint32, {}};
  append(key.value, value);
  return key;
}

inline Key float32Key(std::string name, float value) {
  Key key{std::move(name), kFloat32, {}};
  append(key.value, value);
  return key;
}

inline Key boolKey(std::string name, bool value) {
  Key key{std::move(name), kBool, {}};
  append<uint8_t>(key.value, value);
  return key;
}

inline Key stringKey(std::string name, std::string_view value) {
  Key key{std::move(name), kString, {}};
  appendString(key.value, value);
  return key;
}

inline Key int32ArrayKey(std::string name, const std::vector<int32_t> &values) {
  Key key{std::move(name), kArray, {}};
  append(key.value, kInt32);
  append<uint64_t>(key.value, values.size());
  for (int32_t value : values) append(key.value, value);
  return key;
}

inline Key stringArrayKey(std::string name, const std::vector<std::string> &values) {
  Key key{std::move(name), kArray, {}};
  append(key.value, kString);
  append<uint64_t>(key.value, values.size());
  for (const std::string &value : values) appendString(key.value, value);
  return key;
}

struct Tensor {
  std::string name;
  std::vector<uint64_t> dims; // dims[0] is the row length
  uint32_t type;
  Bytes data;
  // Added to the data offset the tensor table declares: a nonzero
  // displacement points the tensor at bytes that are not its data.
  uint64_t displacement = 0;
};

// The header, the keys, the tensor table, then each tensor's data at the
// next aligned offset; the file ends with the last tensor's data.
inline Bytes file(const std::vector<Key> &keys, const std::vector<Tensor> &tensors) {
  const auto aligned = [](uint64_t size) { return (size + kAlignment - 1) / kAlignment * kAlignment; };
  Bytes out{'G', 'G', 'U', 'F'};
  append<uint32_t>(out, 3);
  append<uint64_t>(out, tensors.size());
  append<uint64_t>(out, keys.size());
  for (const Key &key : keys) {
    appendString(out, key.name);
    append(out, key.type);
    out.insert(out.end(), key.value.begin(), key.value.end());
  }
  uint64_t offset = 0;
  for (const Tensor &tensor : tensors) {
    appendString(out, tensor.name);
    append<uint32_t>(out, tensor.dims.size());
    for (uint64_t dim : tensor.dims) append(out, dim);
    append(out, tensor.type);
    append(out, offset + tensor.displacement);
    offset += aligned(tensor.data.size());
  }
  for (const Tensor &tensor : tensors) {
    out.resize(aligned(out.size()), 0);
    out.insert(out.end(), tensor.data.begin(), tensor.data.end());
  }
  return out;
}

} // namespace splash::test::gguf
