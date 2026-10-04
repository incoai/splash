#include "engine/CacheDirectory.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace splash::engine {
namespace {

constexpr char kLock[] = "lock";
constexpr char kServing[] = "serving";
constexpr char kProbation[] = "probation";
// What a cache holds, and every file a cache directory may hold: nothing
// else in one is ever touched.
constexpr const char *kCacheFiles[] = {"kv.slots", "kv.records", "state.slots", "state.records"};
constexpr const char *kMarks[] = {kServing, kProbation, kLock};

[[noreturn]] void fail(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

class Descriptor final {
public:
  explicit Descriptor(int descriptor) noexcept : descriptor_(descriptor) {}
  Descriptor(Descriptor &&other) noexcept : descriptor_(other.release()) {}
  ~Descriptor() {
    if (descriptor_ >= 0)
      ::close(descriptor_);
  }
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  Descriptor &operator=(Descriptor &&) = delete;
  [[nodiscard]] int get() const noexcept { return descriptor_; }
  [[nodiscard]] int release() noexcept { return std::exchange(descriptor_, -1); }

private:
  int descriptor_;
};

// A namespace is one directory name of lowercase hex digits: nothing a path
// could take apart.
bool plainName(std::string_view name) noexcept {
  return !name.empty() && name.size() <= 64 &&
         std::all_of(name.begin(), name.end(), [](char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

// Its files are trusted as this process's own, so nobody else may write
// there.
bool owned(int directory) noexcept {
  struct stat status {};
  return ::fstat(directory, &status) == 0 && status.st_uid == ::geteuid() &&
         !(status.st_mode & (S_IWGRP | S_IWOTH));
}

Descriptor openDirectory(const std::filesystem::path &path, const char *what) {
  if (::mkdir(path.c_str(), 0700) < 0 && errno != EEXIST)
    fail(what);
  Descriptor directory(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (directory.get() < 0)
    fail(what);
  if (!owned(directory.get()))
    throw std::system_error(std::make_error_code(std::errc::permission_denied),
                            std::string(what) +
                                ": it belongs to another user or others may write to it");
  return directory;
}

bool present(int directory, const char *name) noexcept {
  struct stat status {};
  return ::fstatat(directory, name, &status, AT_SYMLINK_NOFOLLOW) == 0;
}

// A held lock gives its directory only while the directory's path still
// names it. Stale cleanup removes the lock it took, so a process that held,
// or waited for, the same lock may hold one that names nothing.
bool named(const std::filesystem::path &path, int lock) noexcept {
  struct stat held {}, current {};
  return ::fstat(lock, &held) == 0 && ::lstat((path / kLock).c_str(), &current) == 0 &&
         held.st_dev == current.st_dev && held.st_ino == current.st_ino;
}

// Removes another namespace's cache that nobody has used for staleAfter,
// once it holds its lock. A cache is used when it opens (its lock's time) and
// while it serves (its files' times). Anything in the way leaves it as it is.
void removeStale(const std::filesystem::path &path, std::chrono::hours staleAfter) noexcept {
  Descriptor directory(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (directory.get() < 0 || !owned(directory.get()))
    return;
  Descriptor lock(::openat(directory.get(), kLock, O_RDWR | O_CLOEXEC | O_NOFOLLOW));
  struct stat status {};
  if (lock.get() < 0 || ::flock(lock.get(), LOCK_EX | LOCK_NB) < 0 ||
      !named(path, lock.get()) || ::fstat(lock.get(), &status) < 0)
    return;
  auto lastUsed = status.st_mtimespec.tv_sec;
  for (const char *name : kCacheFiles) {
    struct stat file {};
    if (::fstatat(directory.get(), name, &file, AT_SYMLINK_NOFOLLOW) == 0)
      lastUsed = std::max(lastUsed, file.st_mtimespec.tv_sec);
  }
  if (std::chrono::system_clock::now() - std::chrono::system_clock::from_time_t(lastUsed) <
      staleAfter)
    return;
  for (const char *name : kCacheFiles)
    ::unlinkat(directory.get(), name, 0);
  for (const char *name : kMarks)
    ::unlinkat(directory.get(), name, 0);
  // Fails, leaving it, while anything else is in it.
  ::rmdir(path.c_str());
}

} // namespace

std::unique_ptr<CacheDirectory>
CacheDirectory::open(const std::filesystem::path &root, std::string_view cacheNamespace,
                     std::chrono::milliseconds lockWait, std::chrono::hours staleAfter,
                     const std::function<bool()> &cancelled) {
  if (!plainName(cacheNamespace))
    throw std::invalid_argument("a cache namespace is a name of lowercase hex digits");
  if (root.has_parent_path())
    std::filesystem::create_directories(root.parent_path());
  static_cast<void>(openDirectory(root, "open cache root"));
  const std::filesystem::path path = root / cacheNamespace;
  // A restarting server's new engine can race the end of the old one.
  const auto deadline = std::chrono::steady_clock::now() + lockWait;
  const auto expired = [&] {
    return std::chrono::steady_clock::now() >= deadline || (cancelled && cancelled());
  };
  std::unique_ptr<CacheDirectory> result;
  for (;;) {
    Descriptor directory = openDirectory(path, "open cache directory");
    Descriptor lock(
        ::openat(directory.get(), kLock, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (lock.get() < 0)
      fail("open cache directory lock");
    result.reset(new CacheDirectory(path, directory.release(), lock.release()));
    while (::flock(result->lock_, LOCK_EX | LOCK_NB) < 0) {
      if (errno == EINTR)
        continue;
      if (errno != EWOULDBLOCK)
        fail("lock cache directory");
      if (expired())
        return nullptr;
      std::this_thread::sleep_for(std::min<std::chrono::steady_clock::duration>(
          std::chrono::milliseconds(200), deadline - std::chrono::steady_clock::now()));
    }
    // Stale cleanup may have removed the lock while this process waited for
    // it; the namespace belongs to whoever holds the lock its path names now.
    if (named(path, result->lock_))
      break;
    if (expired())
      return nullptr;
  }
  // The lock's time says when the namespace was last opened.
  if (::futimens(result->lock_, nullptr) < 0)
    fail("stamp cache directory lock");
  result->uncleanExit_ = present(result->directory_, kServing);
  if (present(result->directory_, kProbation)) {
    for (const char *name : kCacheFiles) {
      if (::unlinkat(result->directory_, name, 0) < 0 && errno != ENOENT)
        fail("empty cache directory");
    }
    result->unmark(kProbation);
    result->coldReason_ = "the last start failed within its probation";
  }
  // Removing stale caches is housekeeping: nothing it meets stops the open.
  std::error_code ignored;
  for (const auto &entry : std::filesystem::directory_iterator(root, ignored)) {
    const std::string name = entry.path().filename().string();
    if (name != cacheNamespace && plainName(name) && entry.is_directory(ignored))
      removeStale(entry.path(), staleAfter);
  }
  return result;
}

CacheDirectory::CacheDirectory(std::filesystem::path path, int directory, int lock)
    : path_(std::move(path)), directory_(directory), lock_(lock) {}

CacheDirectory::~CacheDirectory() {
  // Closing the lock releases it; the marks stay as they are.
  ::close(lock_);
  ::close(directory_);
}

void CacheDirectory::beginServing(bool probation) {
  mark(kServing);
  if (probation)
    mark(kProbation);
}

void CacheDirectory::endProbation() { unmark(kProbation); }

void CacheDirectory::close() {
  unmark(kProbation);
  unmark(kServing);
}

void CacheDirectory::mark(const char *name) {
  Descriptor file(::openat(directory_, name, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW,
                           0600));
  if (file.get() < 0 || ::fsync(file.get()) < 0 || ::fsync(directory_) < 0)
    fail("mark cache directory");
}

void CacheDirectory::unmark(const char *name) {
  if ((::unlinkat(directory_, name, 0) < 0 && errno != ENOENT) || ::fsync(directory_) < 0)
    fail("unmark cache directory");
}

} // namespace splash::engine
