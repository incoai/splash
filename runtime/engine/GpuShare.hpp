#pragma once

#include "engine/Types.hpp"

#include <filesystem>
#include <optional>

namespace splash::engine {

// Splash servers that share one Mac's GPU (--share-gpu) tell each other the
// highest priority they decode at: each holds a read lock on that priority's
// byte of one file they all open. A POSIX record lock belongs to its process:
// a server's own lock never answers its query, and the kernel drops a
// server's lock when the process ends. Closing any descriptor of the file
// drops all of the process's locks on it, so nothing else in the process may
// open it.
class GpuShare final {
public:
  // splash-gpu/decoding in the per-user cache directory: the servers of one
  // macOS user share the GPU.
  [[nodiscard]] static std::filesystem::path defaultPath();

  // Opens the file, creating it and its directory. Throws an exception that
  // names the path.
  explicit GpuShare(std::filesystem::path path);
  ~GpuShare();
  GpuShare(const GpuShare &) = delete;
  GpuShare &operator=(const GpuShare &) = delete;

  // Holds the lock of the highest priority this server decodes at, none
  // without one, and returns the highest another server decodes at. A file
  // removed or replaced since is opened again. A failure, logged once, ends
  // this server's sharing as leave() does.
  [[nodiscard]] std::optional<RequestPriority>
  exchange(std::optional<RequestPriority> decoding) noexcept;
  // Drops this server's lock; from then on exchange() returns none.
  void leave() noexcept;

private:
  // Opens the path, creating it, in place of the descriptor held before.
  void open();

  std::filesystem::path path_;
  int descriptor_ = -1;
  std::optional<RequestPriority> held_;
};

} // namespace splash::engine
