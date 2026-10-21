// While the prepared weights Splash 1.1 kept remain, the runtime says how much
// they hold and how to free them, counting no file outside the directory; a
// missing or empty directory and a link at the path go unmentioned.
#include "LegacyWeightCache.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestStderr.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using splash::reportLegacyWeightCache;
using splash::test::require;

namespace {

std::string report(const std::filesystem::path &path) {
  return splash::test::capturedStderr([&] { reportLegacyWeightCache(path); });
}

// A sparse file of `bytes` size that takes no disk.
void sparseFile(const std::filesystem::path &path, std::uintmax_t bytes) {
  splash::test::writeFile(path, "");
  std::filesystem::resize_file(path, bytes);
}

void testTheCacheIsReportedWithItsSize(const std::filesystem::path &root) {
  const std::filesystem::path cache = root / "weights";
  std::filesystem::create_directories(cache / "inputs-1/target");
  sparseFile(cache / "inputs-1/target/layer-14.bin", 3ULL << 30);
  // Read-only, as 1.1 wrote its files.
  splash::test::writeFile(cache / "inputs-1/sha256", "digest");
  std::filesystem::permissions(cache / "inputs-1/sha256", std::filesystem::perms::owner_read);
  // Links inside point at what the directory does not hold.
  const std::filesystem::path outside = root / "outside";
  std::filesystem::create_directories(outside);
  sparseFile(outside / "model.bin", 5ULL << 30);
  std::filesystem::create_symlink(outside / "model.bin", cache / "inputs-1/source");
  std::filesystem::create_directory_symlink(outside, cache / "inputs-1/linked");
  const std::string log = report(cache);
  require(log.find("Splash 1.1's prepared weights in " + cache.string() + " hold 3.0 GiB") !=
                  std::string::npos &&
              log.find("rm -rf \"" + cache.string() + "\" frees them") != std::string::npos,
          "the cache was not reported with its size and how to free it: " + log);
}

void testAnEmptyOrMissingCacheOrALinkGoesUnmentioned(const std::filesystem::path &root) {
  const std::filesystem::path cache = root / "linked-weights";
  require(report(cache).empty(), "a missing cache was reported");
  std::filesystem::create_directories(root / "empty/inputs-1");
  require(report(root / "empty").empty(), "an empty cache was reported");
  const std::filesystem::path elsewhere = root / "elsewhere";
  std::filesystem::create_directories(elsewhere);
  sparseFile(elsewhere / "weights", 1ULL << 30);
  std::filesystem::create_directory_symlink(elsewhere, cache);
  require(report(cache).empty(), "a link at the path was reported");
}

} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory directory("legacy-weight-cache");
    testTheCacheIsReportedWithItsSize(directory.path());
    testAnEmptyOrMissingCacheOrALinkGoesUnmentioned(directory.path());
    std::cout << "legacy weight cache tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "legacy weight cache tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
