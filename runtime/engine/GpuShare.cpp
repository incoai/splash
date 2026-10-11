#include "engine/GpuShare.hpp"

#include "StderrLine.hpp"
#include "UserCacheDirectory.hpp"

#include <cerrno>
#include <exception>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace splash::engine {
namespace {

constexpr RequestPriority kPriorities[] = {RequestPriority::Foreground, RequestPriority::Normal,
                                           RequestPriority::Background};

struct flock byteLock(int type, RequestPriority priority) {
  struct flock lock{};
  lock.l_type = static_cast<short>(type);
  lock.l_whence = SEEK_SET;
  lock.l_start = static_cast<off_t>(priority);
  lock.l_len = 1;
  return lock;
}

} // namespace

std::filesystem::path GpuShare::defaultPath() {
  return userCacheDirectory() / "splash-gpu" / "decoding";
}

GpuShare::GpuShare(std::filesystem::path path) : path_(std::move(path)) { open(); }

GpuShare::~GpuShare() { leave(); }

void GpuShare::open() {
  std::filesystem::create_directories(path_.parent_path());
  const int opened = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (opened < 0)
    throw std::system_error(errno, std::generic_category(), "cannot open " + path_.string());
  // The descriptor held before names a file no longer at the path, whose
  // locks no other server queries.
  if (descriptor_ >= 0)
    close(descriptor_);
  descriptor_ = opened;
  held_.reset();
}

void GpuShare::leave() noexcept {
  if (descriptor_ < 0)
    return;
  // Closing the file drops this process's lock.
  close(descriptor_);
  descriptor_ = -1;
  held_.reset();
}

std::optional<RequestPriority>
GpuShare::exchange(std::optional<RequestPriority> decoding) noexcept {
  if (descriptor_ < 0)
    return std::nullopt;
  try {
    // A server that starts after the file was removed or replaced opens the
    // one at the path, and so does this one.
    struct stat named{}, opened{};
    if (stat(path_.c_str(), &named) != 0 || fstat(descriptor_, &opened) != 0 ||
        named.st_dev != opened.st_dev || named.st_ino != opened.st_ino)
      open();
    if (decoding != held_) {
      if (held_) {
        struct flock lock = byteLock(F_UNLCK, *held_);
        if (fcntl(descriptor_, F_SETLK, &lock) != 0)
          throw std::system_error(errno, std::generic_category(), "cannot unlock " + path_.string());
        held_.reset();
      }
      if (decoding) {
        struct flock lock = byteLock(F_RDLCK, *decoding);
        if (fcntl(descriptor_, F_SETLK, &lock) != 0)
          throw std::system_error(errno, std::generic_category(), "cannot lock " + path_.string());
        held_ = decoding;
      }
    }
    // A write lock would conflict with any other process's read lock on the
    // byte, and F_GETLK reports one such lock; this process's own never do.
    for (const RequestPriority priority : kPriorities) {
      struct flock lock = byteLock(F_WRLCK, priority);
      if (fcntl(descriptor_, F_GETLK, &lock) != 0)
        throw std::system_error(errno, std::generic_category(), "cannot query " + path_.string());
      if (lock.l_type != F_UNLCK)
        return priority;
    }
    return std::nullopt;
  } catch (const std::exception &error) {
    logWarning("--share-gpu: ", error.what(),
               "; this server no longer shares the GPU with other Splash servers.");
  }
  leave();
  return std::nullopt;
}

} // namespace splash::engine
