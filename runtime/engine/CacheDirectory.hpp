#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace splash::engine {

// Where a persistent cache tier keeps its files: one directory per cache
// namespace under a root. The namespace names what the cache holds (the
// models, the KV and state layouts, and the cache's own format), so caches
// of several models stay side by side and none is reused for anything else.
// A lock in each directory gives it to one process at a time. A namespace
// nobody has opened or written for a while is removed when another one
// opens, by its known files only.
//
// Two marks say how the last process ended. A serving process is marked
// until it closes cleanly, so a start that finds the mark follows an
// unclean end. A start that takes back a cache after an unclean end serves
// on probation, marked until it has served long enough; a start that finds
// that mark empties the cache, so a cached point that breaks the engine
// cannot break every start after it.
class CacheDirectory final {
public:
  // Opens root/cacheNamespace (made 0700 when missing) and takes its lock,
  // trying every 200 ms while another process holds it: null when one still
  // does after lockWait, or once cancelled(). Other namespaces' directories
  // nobody opened or wrote for staleAfter, and whose lock is free, are
  // removed.
  // Throws std::invalid_argument for a namespace that is not a plain name,
  // and std::system_error when the directory cannot be used or belongs to
  // another user or others may write to it.
  [[nodiscard]] static std::unique_ptr<CacheDirectory>
  open(const std::filesystem::path &root, std::string_view cacheNamespace,
       std::chrono::milliseconds lockWait, std::chrono::hours staleAfter,
       const std::function<bool()> &cancelled = {});
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
