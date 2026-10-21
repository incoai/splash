#include "TestChecks.hpp"
#include "engine/CacheDirectory.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using splash::engine::CacheDirectory;
using splash::test::rejects;
using splash::test::require;
using namespace std::chrono_literals;

namespace {

// A namespace's name, 32 lowercase hex digits, and one of 32 `digit`s.
const std::string kNamespace = "0123456789abcdef0123456789abcdef";
std::string cacheNamespace(char digit) {
  return std::string(CacheDirectory::kNamespaceDigits, digit);
}

// A root in the temporary directory, removed with everything in it.
class Root final {
public:
  Root() {
    std::string name = (std::filesystem::temp_directory_path() / "splash-cache-root-XXXXXX").string();
    require(::mkdtemp(name.data()) != nullptr, "temporary directory could not be made");
    path_ = std::filesystem::path(name) / "prefix-cache";
  }
  ~Root() {
    std::error_code ignored;
    std::filesystem::remove_all(path_.parent_path(), ignored);
  }
  Root(const Root &) = delete;
  Root &operator=(const Root &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }
  [[nodiscard]] std::unique_ptr<CacheDirectory> open(const std::string &cacheNamespace = kNamespace,
                                                     std::chrono::milliseconds wait = 0ms) const {
    return CacheDirectory::open(path_, cacheNamespace, wait);
  }

private:
  std::filesystem::path path_;
};

void touch(const std::filesystem::path &path) { std::ofstream(path) << "cache"; }

// A namespace belongs to one process at a time; a second opener gives up
// after its wait, and takes it once it is free. A name of anything but 32
// lowercase hex digits is refused.
void testOneProcessAtATime() {
  Root root;
  auto first = root.open();
  require(first && std::filesystem::is_directory(first->path()) &&
              first->path() == root.path() / kNamespace && !first->uncleanExit() &&
              first->coldReason().empty(),
          "a new cache directory did not open clean");
  struct stat status {};
  require(::stat(root.path().c_str(), &status) == 0 && (status.st_mode & 0777) == 0700 &&
              ::stat(first->path().c_str(), &status) == 0 && (status.st_mode & 0777) == 0700,
          "the cache directories were not private");
  const auto started = std::chrono::steady_clock::now();
  require(!root.open(kNamespace, 300ms), "a held cache directory was opened twice");
  require(std::chrono::steady_clock::now() - started >= 300ms,
          "a second opener gave up before its wait");
  bool asked = false;
  require(!CacheDirectory::open(root.path(), kNamespace, 10s, [&] { return asked = true; }) &&
              asked,
          "a cancelled wait for the lock did not stop");
  first.reset();
  require(root.open() != nullptr, "a released cache directory could not be opened");
  for (const std::string &name : {std::string("../escape"), std::string("cafe"),
                                  kNamespace.substr(1), kNamespace + "0", cacheNamespace('A')})
    rejects([&] { static_cast<void>(root.open(name)); },
            "a cache namespace is 32 lowercase hex digits",
            "a namespace that is not 32 lowercase hex digits was accepted");
}

// The marks say how the last processes ended: serving without closing is an
// unclean end, a clean close is not. Every start serves on probation. The
// start after one that failed within it keeps the cache, and the start after
// a second such failure in a row, or after an engine that reported its own
// failure within probation, empties it. A start that served its probation
// out ends a run of failures, and neither a failure nor an engine failure
// after probation counts.
void testMarksFollowTheLastEnd() {
  Root root;
  const auto fill = [](const CacheDirectory &directory) {
    for (const auto &file : {directory.kvSlots(), directory.kvRecords(), directory.stateSlots(),
                             directory.stateRecords()})
      touch(file);
  };
  const auto kept = [](const CacheDirectory &directory) {
    return directory.coldReason().empty() && std::filesystem::exists(directory.kvSlots()) &&
           std::filesystem::exists(directory.stateRecords());
  };
  const auto emptied = [](const CacheDirectory &directory) {
    return !directory.coldReason().empty() && !std::filesystem::exists(directory.kvSlots()) &&
           !std::filesystem::exists(directory.stateRecords());
  };
  {
    auto directory = root.open();
    fill(*directory);
    directory->beginServing();
  }
  {
    auto directory = root.open();
    require(directory->uncleanExit() && kept(*directory),
            "one failure within probation emptied the cache");
    directory->beginServing();
    directory->endProbation();
  }
  {
    auto directory = root.open();
    require(directory->uncleanExit() && kept(*directory),
            "a failure after probation emptied the cache");
    directory->beginServing();
  }
  {
    auto directory = root.open();
    require(kept(*directory),
            "a failure within probation after a probation served out emptied the cache");
    directory->beginServing();
  }
  {
    auto directory = root.open();
    require(directory->uncleanExit() && emptied(*directory),
            "a second failure in a row within probation left the cache to the next start");
    fill(*directory);
    directory->beginServing();
    directory->close();
  }
  {
    auto directory = root.open();
    require(!directory->uncleanExit() && kept(*directory), "a clean close counted as unclean");
    directory->beginServing();
    directory->engineFailed();
  }
  {
    auto directory = root.open();
    require(emptied(*directory),
            "an engine failure within probation left the cache to the next start");
    fill(*directory);
    directory->beginServing();
    directory->endProbation();
    directory->engineFailed();
  }
  auto directory = root.open();
  require(kept(*directory), "an engine failure after probation emptied the cache");
}

// Cleanup removes a namespace nobody opened or wrote for the stale period,
// by its known files, unless its lock is held, by this process too, or it
// holds something else, and returns the idle ones it kept. A missing root
// has nothing to remove.
void testStaleNamespacesGo() {
  Root root;
  const std::string a = cacheNamespace('a'), b = cacheNamespace('b'), c = cacheNamespace('c'),
                    d = cacheNamespace('d'), e = cacheNamespace('e');
  const auto age = [&](const std::string &cacheNamespace,
                       std::initializer_list<const char *> files) {
    const timeval old[2] = {{0, 0}, {1000000000, 0}};
    for (const char *file : files)
      require(::utimes((root.path() / cacheNamespace / file).c_str(), old) == 0,
              "a cache file could not be aged");
  };
  for (const std::string &cacheNamespace : {a, b, c, d, e}) {
    auto directory = root.open(cacheNamespace);
    touch(directory->kvSlots());
  }
  age(a, {"lock", "kv.slots"});
  age(c, {"lock", "kv.slots"});
  age(d, {"lock", "kv.slots"});
  touch(root.path() / d / "notes");
  // Opened long ago, written since: in use.
  age(e, {"lock"});
  auto held = root.open(c);
  age(c, {"lock"});
  const std::vector<CacheDirectory::Idle> kept =
      CacheDirectory::removeStale(CacheDirectory::idle(root.path()), std::chrono::hours(24 * 14));
  const auto keptNamed = [&](const std::string &cacheNamespace) {
    return std::any_of(kept.begin(), kept.end(), [&](const CacheDirectory::Idle &cache) {
      return cache.path == root.path() / cacheNamespace;
    });
  };
  require(kept.size() == 2 && keptNamed(b) && keptNamed(e),
          "cleanup did not return the idle namespaces it kept");
  require(!std::filesystem::exists(root.path() / a) &&
              std::filesystem::exists(root.path() / b / "kv.slots") &&
              std::filesystem::exists(root.path() / c / "kv.slots") &&
              std::filesystem::exists(root.path() / d / "notes") &&
              !std::filesystem::exists(root.path() / d / "kv.slots") &&
              std::filesystem::exists(root.path() / e / "kv.slots"),
          "stale namespaces were not removed exactly by their known files");
  require(CacheDirectory::removeStale(CacheDirectory::idle(root.path() / "missing"),
                                      std::chrono::hours(24 * 14))
              .empty(),
          "a missing root had caches");
}

// The namespaces nobody holds are listed oldest first, with the blocks of
// their files. A listed namespace goes only while nobody holds it and nobody
// used it since it was listed.
void testIdleNamespacesGoOldestFirst() {
  Root root;
  const std::string a = cacheNamespace('a'), b = cacheNamespace('b'), c = cacheNamespace('c'),
                    d = cacheNamespace('d');
  const std::string block(64 * 1024, 'x');
  long when = 1000000000;
  for (const std::string &cacheNamespace : {b, a, c, d}) {
    auto directory = root.open(cacheNamespace);
    std::ofstream(directory->kvSlots()) << block;
    const timeval used[2] = {{when, 0}, {when, 0}};
    for (const char *file : {"lock", "kv.slots"})
      require(::utimes((directory->path() / file).c_str(), used) == 0,
              "a cache file could not be aged");
    when += 1000;
  }
  auto held = root.open(d);
  const std::vector<CacheDirectory::Idle> idle = CacheDirectory::idle(root.path());
  require(idle.size() == 3 && idle[0].path == root.path() / b && idle[1].path == root.path() / a &&
              idle[2].path == root.path() / c && idle[0].lastUsed < idle[1].lastUsed &&
              idle[0].bytes >= block.size() && idle[1].bytes >= block.size(),
          "the namespaces nobody holds were not listed oldest first with their blocks");
  touch(root.path() / a / "kv.slots");
  require(!CacheDirectory::remove(idle[1]) && std::filesystem::exists(root.path() / a / "kv.slots"),
          "a namespace used since it was listed was removed");
  require(CacheDirectory::remove(idle[0]) && !std::filesystem::exists(root.path() / b),
          "an idle namespace was not removed");
  require(!CacheDirectory::remove({root.path() / d, std::chrono::system_clock::now(), 0}) &&
              std::filesystem::exists(root.path() / d / "kv.slots"),
          "a held namespace was removed");
}

// Only a directory named as a namespace is, 32 lowercase hex digits, is a
// cache. A root that is a directory of the user's (--cache-dir) may hold
// others of hex names, with a lock and the cache's files in them: cleanup
// lists and removes none of them, however old they are.
void testOnlyNamespacesAreCaches() {
  Root root;
  const std::string stale = cacheNamespace('a');
  static_cast<void>(root.open(stale));
  // Uppercase F: a volume that ignores case would take an uppercase A for
  // the stale namespace's directory.
  const std::vector<std::string> others{"cafe", "0123abcd", kNamespace.substr(1), kNamespace + "0",
                                        cacheNamespace('F')};
  const timeval old[2] = {{0, 0}, {1000000000, 0}};
  for (const std::string &name : others) {
    std::filesystem::create_directory(root.path() / name);
    for (const char *file : {"lock", "kv.slots", "state.slots"}) {
      touch(root.path() / name / file);
      require(::utimes((root.path() / name / file).c_str(), old) == 0, "a file could not be aged");
    }
  }
  require(::utimes((root.path() / stale / "lock").c_str(), old) == 0,
          "a cache file could not be aged");
  const std::vector<CacheDirectory::Idle> idle = CacheDirectory::idle(root.path());
  require(idle.size() == 1 && idle[0].path == root.path() / stale,
          "a directory not named as a namespace was listed as a cache");
  require(CacheDirectory::removeStale(idle, std::chrono::hours(24 * 14)).empty() &&
              !std::filesystem::exists(root.path() / stale),
          "the stale namespace beside the others was not removed");
  for (const std::string &name : others)
    for (const char *file : {"lock", "kv.slots", "state.slots"})
      require(std::filesystem::exists(root.path() / name / file),
              "cleanup touched " + name + ", which is not named as a namespace");
}

// Stale cleanup can remove a namespace's lock while another process waits
// for it. The waiter then takes a lock its path no longer names: it must
// contend for the one the path names now, not own the directory by the
// removed one beside that lock's holder.
void testRemovedLockGivesNoOwnership() {
  Root root;
  auto first = root.open();
  std::promise<void> waiting;
  auto waited = waiting.get_future();
  bool signalled = false;
  auto second = std::async(std::launch::async, [&] {
    return CacheDirectory::open(root.path(), kNamespace, 10s, [&] {
      if (!std::exchange(signalled, true))
        waiting.set_value();
      return false;
    });
  });
  waited.wait();
  // Cleanup removed the lock the waiter has open, and another engine made
  // a new one and holds it.
  const std::filesystem::path lock = root.path() / kNamespace / "lock";
  std::filesystem::remove(lock);
  const int replacement = ::open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  require(replacement >= 0 && ::flock(replacement, LOCK_EX) == 0,
          "a replacement lock could not be held");
  first.reset();
  require(second.wait_for(600ms) == std::future_status::timeout,
          "a removed lock gave its directory while the lock its path names was held");
  ::close(replacement);
  const auto owner = second.get();
  const int probe = ::open(lock.c_str(), O_RDWR | O_CLOEXEC);
  const bool held = probe >= 0 && ::flock(probe, LOCK_EX | LOCK_NB) < 0 && errno == EWOULDBLOCK;
  if (probe >= 0)
    ::close(probe);
  require(owner && held, "the directory's owner does not hold the lock its path names");
}

// Nobody else may write where the cache trusts its files.
void testForeignDirectoriesAreRefused() {
  Root root;
  static_cast<void>(root.open());
  require(::chmod((root.path() / kNamespace).c_str(), 0777) == 0,
          "cache directory permissions could not be widened");
  rejects([&] { static_cast<void>(root.open()); },
          "it belongs to another user or others may write to it",
          "a cache directory others may write to was opened");
}

} // namespace

int main() {
  try {
    testOneProcessAtATime();
    testMarksFollowTheLastEnd();
    testStaleNamespacesGo();
    testIdleNamespacesGoOldestFirst();
    testOnlyNamespacesAreCaches();
    testRemovedLockGivesNoOwnership();
    testForeignDirectoriesAreRefused();
    std::cout << "Cache directory tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
