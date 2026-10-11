#pragma once

#include <unistd.h>

#include <climits>
#include <cstddef>
#include <filesystem>
#include <stdexcept>

namespace splash {

// The cache directory macOS keeps for this user (_CS_DARWIN_USER_CACHE_DIR).
// Throws if macOS names none.
[[nodiscard]] inline std::filesystem::path userCacheDirectory() {
  char path[PATH_MAX];
  const size_t size = confstr(_CS_DARWIN_USER_CACHE_DIR, path, sizeof path);
  if (!size || size > sizeof path)
    throw std::runtime_error("macOS names no per-user cache directory");
  return path;
}

} // namespace splash
