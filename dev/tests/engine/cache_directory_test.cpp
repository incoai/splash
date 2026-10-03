#include "TestChecks.hpp"
#include "engine/CacheDirectory.hpp"

#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

using splash::engine::CacheDirectory;
using splash::test::require;
using namespace std::chrono_literals;

namespace {

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
  [[nodiscard]] std::unique_ptr<CacheDirectory> open(const char *cacheNamespace = "0123abcd",
                                                     std::chrono::milliseconds wait = 0ms) const {
    return CacheDirectory::open(path_, cacheNamespace, wait, std::chrono::hours(24 * 14));
  }

private:
  std::filesystem::path path_;
};

void touch(const std::filesystem::path &path) { std::ofstream(path) << "cache"; }

template <typename Exception, typename Call>
bool throws(Call call) {
  try {
    call();
  } catch (const Exception &) {
    return true;
  }
  return false;
}

// A namespace belongs to one process at a time; a second opener gives up
// after its wait, and takes it once it is free.
void testOneProcessAtATime() {
  Root root;
  auto first = root.open();
  require(first && std::filesystem::is_directory(first->path()) &&
              first->path() == root.path() / "0123abcd" && !first->uncleanExit() &&
              first->coldReason().empty(),
          "a new cache directory did not open clean");
  struct stat status {};
  require(::stat(root.path().c_str(), &status) == 0 && (status.st_mode & 0777) == 0700 &&
              ::stat(first->path().c_str(), &status) == 0 && (status.st_mode & 0777) == 0700,
          "the cache directories were not private");
  const auto started = std::chrono::steady_clock::now();
  require(!root.open("0123abcd", 300ms), "a held cache directory was opened twice");
  require(std::chrono::steady_clock::now() - started >= 300ms,
          "a second opener gave up before its wait");
  bool asked = false;
  require(!CacheDirectory::open(root.path(), "0123abcd", 10s, std::chrono::hours(1),
                                [&] { return asked = true; }) && asked,
          "a cancelled wait for the lock did not stop");
  first.reset();
  require(root.open() != nullptr, "a released cache directory could not be opened");
  require(throws<std::invalid_argument>([&] { static_cast<void>(root.open("../escape")); }) &&
              throws<std::invalid_argument>([&] { static_cast<void>(root.open("ABCD")); }),
          "a namespace that is not a plain name was accepted");
}

// The marks say how the last process ended: serving without closing is an
// unclean end, a clean close is not; an end within probation empties the
// cache, an end after it does not.
void testMarksFollowTheLastEnd() {
  Root root;
  {
    auto directory = root.open();
    directory->beginServing(false);
  }
  {
    auto directory = root.open();
    require(directory->uncleanExit() && directory->coldReason().empty(),
            "an end without closing did not count as unclean");
    directory->beginServing(false);
    directory->close();
  }
  {
    auto directory = root.open();
    require(!directory->uncleanExit(), "a clean close counted as unclean");
    for (const auto &file : {directory->kvSlots(), directory->kvRecords(), directory->stateSlots(),
                             directory->stateRecords()})
      touch(file);
    directory->beginServing(true);
  }
  {
    auto directory = root.open();
    require(directory->uncleanExit() && !directory->coldReason().empty() &&
                !std::filesystem::exists(directory->kvSlots()) &&
                !std::filesystem::exists(directory->stateRecords()),
            "a failure within probation left the cache to the next start");
    touch(directory->kvSlots());
    directory->beginServing(true);
    directory->endProbation();
  }
  auto directory = root.open();
  require(directory->uncleanExit() && directory->coldReason().empty() &&
              std::filesystem::exists(directory->kvSlots()),
          "a failure after probation emptied the cache");
}

// Opening one namespace removes another nobody opened or wrote for the
// stale period, by its known files, unless its lock is held or it holds
// something else.
void testStaleNamespacesGo() {
  Root root;
  const auto age = [&](const char *cacheNamespace, std::initializer_list<const char *> files) {
    const timeval old[2] = {{0, 0}, {1000000000, 0}};
    for (const char *file : files)
      require(::utimes((root.path() / cacheNamespace / file).c_str(), old) == 0,
              "a cache file could not be aged");
  };
  for (const char *cacheNamespace : {"aaaa", "bbbb", "cccc", "dddd", "eeee"}) {
    auto directory = root.open(cacheNamespace);
    touch(directory->kvSlots());
  }
  age("aaaa", {"lock", "kv.slots"});
  age("cccc", {"lock", "kv.slots"});
  age("dddd", {"lock", "kv.slots"});
  touch(root.path() / "dddd" / "notes");
  // Opened long ago, written since: in use.
  age("eeee", {"lock"});
  auto held = root.open("cccc");
  age("cccc", {"lock"});
  auto opened = root.open("ffff");
  require(!std::filesystem::exists(root.path() / "aaaa") &&
              std::filesystem::exists(root.path() / "bbbb" / "kv.slots") &&
              std::filesystem::exists(root.path() / "cccc" / "kv.slots") &&
              std::filesystem::exists(root.path() / "dddd" / "notes") &&
              !std::filesystem::exists(root.path() / "dddd" / "kv.slots") &&
              std::filesystem::exists(root.path() / "eeee" / "kv.slots"),
          "stale namespaces were not removed exactly by their known files");
}

// Nobody else may write where the cache trusts its files.
void testForeignDirectoriesAreRefused() {
  Root root;
  static_cast<void>(root.open());
  require(::chmod((root.path() / "0123abcd").c_str(), 0777) == 0,
          "cache directory permissions could not be widened");
  require(throws<std::system_error>([&] { static_cast<void>(root.open()); }),
          "a cache directory others may write to was opened");
}

} // namespace

int main() {
  try {
    testOneProcessAtATime();
    testMarksFollowTheLastEnd();
    testStaleNamespacesGo();
    testForeignDirectoriesAreRefused();
    std::cout << "Cache directory tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
