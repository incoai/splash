#pragma once

// Files of the weight preparation tests: a private temporary directory and
// whole-file writes and reads that fail with the path they could not use;
// and of the SSD cache's tests, what a file holds on its volume.

#include <fcntl.h>
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
// contents when this is destroyed.
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

// What a file holds on its volume: the bytes of its data, the ranges up to
// its end that are not holes. Not its blocks (st_blocks), which APFS lets
// run past the data of a file written to for over a second until the file
// closes: 16 MiB past it after 2.5 s of writes past a file's end, each
// punching the hole before it, on the Studio (M5 Max).
inline uint64_t dataBytes(const std::filesystem::path &path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) throw std::system_error(errno, std::generic_category(), "cannot open " + path.string());
  const off_t end = ::lseek(descriptor, 0, SEEK_END);
  uint64_t bytes = 0;
  for (off_t at = 0; at < end;) {
    // ENXIO: no data from `at` to the end.
    const off_t data = ::lseek(descriptor, at, SEEK_DATA);
    if (data < 0) break;
    at = ::lseek(descriptor, data, SEEK_HOLE);
    if (at <= data) {
      ::close(descriptor);
      throw std::runtime_error("cannot measure the data of " + path.string());
    }
    bytes += static_cast<uint64_t>(at - data);
  }
  ::close(descriptor);
  return bytes;
}

} // namespace splash::test
