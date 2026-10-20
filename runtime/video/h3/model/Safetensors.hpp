#pragma once

// Read-only safetensors file: header index and pread of tensor bytes straight into caller memory (Metal buffers).
#include "video/h3/engine/Status.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace h3::model {

struct TensorInfo {
  std::string dtype;  // "I8", "F16", "BF16", "F32", "I32"
  std::vector<int64_t> shape;
  uint64_t offset = 0;  // absolute file offset of the data
  uint64_t bytes = 0;
};

class SafetensorsFile {
public:
  SafetensorsFile() = default;
  SafetensorsFile(SafetensorsFile &&other) noexcept;
  SafetensorsFile &operator=(SafetensorsFile &&other) noexcept;
  SafetensorsFile(const SafetensorsFile &) = delete;
  SafetensorsFile &operator=(const SafetensorsFile &) = delete;
  ~SafetensorsFile();

  static Status open(const std::string &path, SafetensorsFile &out);

  [[nodiscard]] const std::string &path() const { return path_; }
  [[nodiscard]] bool has(const std::string &name) const { return tensors_.count(name) != 0; }
  // Tensor entry with a dtype and shape check (shape empty = any).
  Status find(const std::string &name, const char *dtype, const std::vector<int64_t> &shape, TensorInfo &out) const;
  // Reads the whole tensor into dst (exactly info.bytes).
  Status read(const TensorInfo &info, void *dst) const;
  Status read(const std::string &name, const char *dtype, const std::vector<int64_t> &shape, void *dst,
              size_t capacity) const;
  // Reads [offset, offset + bytes) of the file.
  Status readRange(uint64_t offset, size_t bytes, void *dst) const;

private:
  std::string path_;
  int fd_ = -1;
  std::map<std::string, TensorInfo> tensors_;
};

// pread loop of [offset, offset + bytes) into dst.
Status readFully(int fd, uint64_t offset, size_t bytes, void *dst, const std::string &path);

} // namespace h3::model
