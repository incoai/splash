#pragma once

#include "StderrLine.hpp"

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <system_error>

namespace splash {

// Splash 1.1 kept prepared copies of every model it served in
// ~/Library/Caches/Splash/weights, which no later version reads, writes or
// removes. While the directory at `path` holds any, says how much and how to
// free it; the files are the user's to delete. A link at the path, links
// inside it and a directory that cannot be read go unmentioned.
inline void reportLegacyWeightCache(const std::filesystem::path &path) noexcept {
  try {
    std::error_code ignored;
    if (!std::filesystem::is_directory(std::filesystem::symlink_status(path, ignored)))
      return;
    std::uintmax_t bytes = 0;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(path))
      if (!entry.is_symlink() && entry.is_regular_file())
        bytes += entry.file_size();
    // 1.1 wrote them read-only, which a plain rm -r asks about file by file.
    if (bytes)
      logLine("Splash 1.1's prepared weights in ", path.string(), " hold ", std::fixed,
              std::setprecision(1), static_cast<double>(bytes) / (1ULL << 30),
              " GiB, which nothing reads now: rm -rf \"", path.string(), "\" frees them");
  } catch (...) {
    // Diagnostics must not affect startup.
  }
}

} // namespace splash
