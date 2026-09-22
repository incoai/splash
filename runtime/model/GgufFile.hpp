#pragma once

// Header parser for llama.cpp GGUF files (version 3): metadata scalars, tensor
// table and data section offset. Tensor data is never read here.

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

class GgufError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// ggml type ids as stored in GGUF tensor infos.
namespace ggml {
inline constexpr uint32_t kF32 = 0, kF16 = 1, kQ8_0 = 8, kQ3_K = 11, kQ4_K = 12,
                          kQ5_K = 13, kQ6_K = 14, kIQ4_NL = 20, kIQ3_S = 21,
                          kIQ4_XS = 23, kBF16 = 30;
}

struct GgmlTypeTraits {
  const char *name;
  uint32_t blockElements;
  uint32_t blockBytes;
};
// nullptr for type ids this parser does not know.
[[nodiscard]] const GgmlTypeTraits *ggmlTypeTraits(uint32_t type) noexcept;
[[nodiscard]] std::string ggmlTypeName(uint32_t type);

struct GgufTensor {
  std::string name;
  uint32_t type = 0;
  std::vector<uint64_t> dims; // dims[0] is the fastest (row length)
  uint64_t offset = 0;        // relative to the data section
  uint64_t bytes = 0;
  [[nodiscard]] uint64_t columns() const noexcept { return dims.empty() ? 0 : dims[0]; }
  [[nodiscard]] uint64_t rows() const;
  [[nodiscard]] uint64_t elements() const;
};

class GgufFile final {
public:
  explicit GgufFile(std::filesystem::path path);

  [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }
  [[nodiscard]] uint64_t fileBytes() const noexcept { return fileBytes_; }
  [[nodiscard]] uint64_t dataOffset() const noexcept { return dataOffset_; }
  [[nodiscard]] uint32_t alignment() const noexcept { return alignment_; }
  [[nodiscard]] const std::string &architecture() const noexcept { return architecture_; }

  [[nodiscard]] std::optional<uint64_t> unsignedValue(std::string_view key) const;
  [[nodiscard]] std::optional<std::string> stringValue(std::string_view key) const;

  [[nodiscard]] const std::vector<GgufTensor> &tensors() const noexcept { return tensors_; }
  [[nodiscard]] const GgufTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const GgufTensor &require(std::string_view name) const;
  // Absolute byte offset of a tensor's data in the file.
  [[nodiscard]] uint64_t absoluteOffset(const GgufTensor &tensor) const noexcept {
    return dataOffset_ + tensor.offset;
  }

private:
  std::filesystem::path path_;
  uint64_t fileBytes_ = 0;
  uint64_t dataOffset_ = 0;
  uint32_t alignment_ = 32;
  std::string architecture_;
  std::map<std::string, uint64_t, std::less<>> unsigned_;
  std::map<std::string, std::string, std::less<>> strings_;
  std::vector<GgufTensor> tensors_;
  std::map<std::string, size_t, std::less<>> index_;
};

} // namespace splash::model
