#pragma once

// Files of the weight preparation tests: a private temporary directory and
// whole-file writes and reads that fail with the path they could not use.

#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace splash::test {

// A new directory under the system temporary directory, removed with its
// contents when this is destroyed. A forked child must leave with _exit so it
// does not remove the parent's directory.
class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(std::string_view name) {
    std::string pattern = (std::filesystem::temp_directory_path() / (std::string(name) + "-XXXXXX")).string();
    if (!mkdtemp(pattern.data()))
      throw std::system_error(errno, std::generic_category(), "cannot create a temporary directory");
    path_ = pattern;
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
  [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }

private:
  std::filesystem::path path_;
};

// Replaces path with bytes.
inline void writeFile(const std::filesystem::path &path, std::span<const uint8_t> bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file.flush()) throw std::runtime_error("cannot write " + path.string());
}

inline void writeFile(const std::filesystem::path &path, std::string_view text) {
  writeFile(path, {reinterpret_cast<const uint8_t *>(text.data()), text.size()});
}

inline std::vector<uint8_t> readFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(file), {}};
}

} // namespace splash::test
