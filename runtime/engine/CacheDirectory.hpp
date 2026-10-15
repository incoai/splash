#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace splash::engine {

// Where a persistent cache tier keeps its files: one directory per cache
// namespace under a root. The namespace names what the cache holds (the
// models, the KV and state layouts, and the cache's own format), so caches
// of several models stay side by side and none is reused for anything else.
// A lock in each directory gives it to one process at a time. A namespace
// nobody has opened or written for a while is removed (removeStale()), by
// its known files only, and so is one the root's limit no longer holds
// (engine::DiskSpace). Removal decides on a namespace nobody holds at that
// moment, so one a starting process is about to open can go first: the open
// then makes it again, and that start begins cold.
//
// Two marks say how the last process ended. A serving process is marked
// until it closes cleanly, so a start that finds the mark follows an
// unclean end. A start that takes back a cache after an unclean end serves
// on probation, marked until it has served long enough; a start that finds
// that mark empties the cache, so a cached point that breaks the engine
// cannot break every start after it.
class CacheDirectory final {
public:
  // A namespace is a name of this many lowercase hex digits, 128 bits of a
  // digest (engine::persistentCacheNamespace): nothing a path could take
  // apart, and nothing a directory of the user's is likely to be named, so
  // that no other directory under a root is ever taken for a cache.
  static constexpr size_t kNamespaceDigits = 32;

  // A namespace nobody held when it was looked at: where it is, when it was
  // last used (its lock's time, and its files') and the bytes its files'
  // blocks take.
  struct Idle final {
    std::filesystem::path path;
    std::chrono::system_clock::time_point lastUsed;
    uint64_t bytes = 0;
  };

  // Opens root/cacheNamespace (made 0700 when missing) and takes its lock,
  // trying every 200 ms while another process holds it: null when one still
  // does after lockWait, or once cancelled().
  // Throws std::invalid_argument for a namespace that is not such a name,
  // and std::system_error when the directory cannot be used or belongs to
  // another user or others may write to it.
  [[nodiscard]] static std::unique_ptr<CacheDirectory>
  open(const std::filesystem::path &root, std::string_view cacheNamespace,
       std::chrono::milliseconds lockWait, const std::function<bool()> &cancelled = {});
  // The namespaces under root whose lock nobody holds, this process's own
  // included, oldest first; a directory of any other name is none.
  // Housekeeping: one that cannot be looked at is left out.
  [[nodiscard]] static std::vector<Idle> idle(const std::filesystem::path &root) noexcept;
  // Removes the idle namespaces nobody opened or wrote for staleAfter, by
  // their known files only, and returns the others. One someone holds by now
  // stays, this process's own included. Housekeeping: nothing it meets is
  // reported.
  [[nodiscard]] static std::vector<Idle> removeStale(std::vector<Idle> caches,
                                                     std::chrono::hours staleAfter) noexcept;
  // Removes an idle namespace by its known files, and its directory once
  // nothing else is in it, unless someone holds it or used it since it was
  // looked at; false then.
  static bool remove(const Idle &cache) noexcept;
  ~CacheDirectory();
  CacheDirectory(const CacheDirectory &) = delete;
  CacheDirectory &operator=(const CacheDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }
  [[nodiscard]] std::filesystem::path kvSlots() const { return path_ / "kv.slots"; }
  [[nodiscard]] std::filesystem::path kvRecords() const { return path_ / "kv.records"; }
  [[nodiscard]] std::filesystem::path stateSlots() const { return path_ / "state.slots"; }
  [[nodiscard]] std::filesystem::path stateRecords() const { return path_ / "state.records"; }
  // Why this start found the cache emptied: the last start failed within
  // its probation. Empty otherwise.
  [[nodiscard]] const std::string &coldReason() const noexcept { return coldReason_; }
  // The last process served and did not close cleanly.
  [[nodiscard]] bool uncleanExit() const noexcept { return uncleanExit_; }

  // Serving begins, on probation when this start took back a cache after an
  // unclean end. Until close(), the process counts as ending uncleanly.
  void beginServing(bool probation);
  // The process has served long enough: a failure no longer empties the
  // cache.
  void endProbation();
  // A clean end; the tier's files are already on the drive.
  void close();

private:
  CacheDirectory(std::filesystem::path path, int directory, int lock);
  // Creates or removes a mark, durably.
  void mark(const char *name);
  void unmark(const char *name);

  std::filesystem::path path_;
  int directory_ = -1;
  int lock_ = -1;
  std::string coldReason_;
  bool uncleanExit_ = false;
};

} // namespace splash::engine
