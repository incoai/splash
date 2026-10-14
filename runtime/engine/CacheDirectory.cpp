#include "engine/CacheDirectory.hpp"
#include "AwakeClock.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <optional>
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

// A namespace's name (CacheDirectory::kNamespaceDigits).
bool namespaceName(std::string_view name) noexcept {
  return name.size() == CacheDirectory::kNamespaceDigits &&
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

// Makes the directory at path (0700) when missing and opens it; empty when
// it was removed between the two.
std::optional<Descriptor> openDirectory(const std::filesystem::path &path, const char *what) {
  if (::mkdir(path.c_str(), 0700) < 0 && errno != EEXIST)
    fail(what);
  Descriptor directory(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (directory.get() < 0) {
    if (errno == ENOENT)
      return std::nullopt;
    fail(what);
  }
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

// A namespace's directory with its lock taken: nobody else holds it while
// this is open.
struct Taken final {
  Descriptor directory;
  Descriptor lock;
};

// Takes the lock of the namespace at path, when nobody holds it.
std::optional<Taken> take(const std::filesystem::path &path) noexcept {
  Descriptor directory(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (directory.get() < 0 || !owned(directory.get()))
    return std::nullopt;
  Descriptor lock(::openat(directory.get(), kLock, O_RDWR | O_CLOEXEC | O_NOFOLLOW));
  if (lock.get() < 0 || ::flock(lock.get(), LOCK_EX | LOCK_NB) < 0 || !named(path, lock.get()))
    return std::nullopt;
  return Taken{std::move(directory), std::move(lock)};
}

// When a taken namespace was last used: when it opened (its lock's time) or
// served (its files' times). Its files' blocks go to `bytes`.
std::chrono::system_clock::time_point lastUse(const Taken &taken, uint64_t &bytes) noexcept {
  struct stat status {};
  auto lastUsed = ::fstat(taken.lock.get(), &status) == 0 ? status.st_mtimespec.tv_sec : 0;
  bytes = 0;
  for (const char *name : kCacheFiles) {
    struct stat file {};
    if (::fstatat(taken.directory.get(), name, &file, AT_SYMLINK_NOFOLLOW) == 0) {
      lastUsed = std::max(lastUsed, file.st_mtimespec.tv_sec);
      bytes += static_cast<uint64_t>(file.st_blocks) * 512;
    }
  }
  return std::chrono::system_clock::from_time_t(lastUsed);
}

} // namespace

std::unique_ptr<CacheDirectory>
CacheDirectory::open(const std::filesystem::path &root, std::string_view cacheNamespace,
                     std::chrono::milliseconds lockWait, const std::function<bool()> &cancelled) {
  if (!namespaceName(cacheNamespace))
    throw std::invalid_argument("a cache namespace is 32 lowercase hex digits");
  if (root.has_parent_path())
    std::filesystem::create_directories(root.parent_path());
  if (!openDirectory(root, "open cache root"))
    throw std::system_error(std::make_error_code(std::errc::no_such_file_or_directory),
                            "open cache root");
  const std::filesystem::path path = root / cacheNamespace;
  // A restarting server's new engine can race the end of the old one.
  const auto deadline = AwakeClock::now() + lockWait;
  const auto expired = [&] {
    return AwakeClock::now() >= deadline || (cancelled && cancelled());
  };
  std::unique_ptr<CacheDirectory> result;
  for (;;) {
    // Another process that removes the namespace as a cache nobody holds
    // can take its directory away before this one opens it or makes its
    // lock there: the next turn makes it again, empty.
    std::optional<Descriptor> directory = openDirectory(path, "open cache directory");
    Descriptor lock(directory ? ::openat(directory->get(), kLock,
                                         O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600)
                              : -1);
    if (lock.get() < 0) {
      if (directory && errno != ENOENT)
        fail("open cache directory lock");
      if (expired())
        return nullptr;
      continue;
    }
    result.reset(new CacheDirectory(path, directory->release(), lock.release()));
    while (::flock(result->lock_, LOCK_EX | LOCK_NB) < 0) {
      if (errno == EINTR)
        continue;
      if (errno != EWOULDBLOCK)
        fail("lock cache directory");
      if (expired())
        return nullptr;
      std::this_thread::sleep_for(std::min<AwakeClock::duration>(
          std::chrono::milliseconds(200), deadline - AwakeClock::now()));
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
  return result;
}

std::vector<CacheDirectory::Idle> CacheDirectory::idle(const std::filesystem::path &root) noexcept {
  std::vector<Idle> result;
  try {
    std::error_code ignored;
    for (const auto &entry : std::filesystem::directory_iterator(root, ignored)) {
      if (!namespaceName(entry.path().filename().string()) || !entry.is_directory(ignored))
        continue;
      if (const std::optional<Taken> taken = take(entry.path())) {
        Idle cache{entry.path(), {}, 0};
        cache.lastUsed = lastUse(*taken, cache.bytes);
        result.push_back(std::move(cache));
      }
    }
    std::sort(result.begin(), result.end(),
              [](const Idle &left, const Idle &right) { return left.lastUsed < right.lastUsed; });
  } catch (...) {
    // Housekeeping: what it found so far, in any order, is as good.
  }
  return result;
}

std::vector<CacheDirectory::Idle> CacheDirectory::removeStale(std::vector<Idle> caches,
                                                              std::chrono::hours staleAfter) noexcept {
  const auto now = std::chrono::system_clock::now();
  std::erase_if(caches, [&](const Idle &cache) {
    if (now - cache.lastUsed < staleAfter)
      return false;
    static_cast<void>(remove(cache));
    return true;
  });
  return caches;
}

bool CacheDirectory::remove(const Idle &cache) noexcept {
  const std::optional<Taken> taken = take(cache.path);
  uint64_t bytes = 0;
  if (!taken || lastUse(*taken, bytes) > cache.lastUsed)
    return false;
  for (const char *name : kCacheFiles)
    ::unlinkat(taken->directory.get(), name, 0);
  for (const char *name : kMarks)
    ::unlinkat(taken->directory.get(), name, 0);
  // Fails, leaving it, while anything else is in it.
  ::rmdir(cache.path.c_str());
  return true;
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
