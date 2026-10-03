#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/PreparedWeights.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace splash::model;
using splash::test::rejects;
using splash::test::require;

namespace {

// Runs body in a child process. The child leaves through _exit with body's
// status, 1 when it throws, so it never returns into the parent's scenarios
// or removes their directory.
template <class F> pid_t spawn(F body) {
  const pid_t child = fork();
  require(child >= 0, "fork a child process");
  if (child == 0) {
    int status = 1;
    try {
      status = body();
    } catch (...) {
    }
    _exit(status);
  }
  return child;
}

// The exit status of child, -1 when a signal ended it.
int exitStatus(pid_t child) {
  int status = 0;
  require(waitpid(child, &status, 0) == child, "wait for a child process");
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Writes one byte other than the fixture's at offset of a prepared or source
// file, which preparation left read-only.
void corrupt(const std::filesystem::path &path, uint64_t offset) {
  require(chmod(path.c_str(), 0600) == 0, "chmod " + path.string());
  const int fd = open(path.c_str(), O_WRONLY);
  require(fd >= 0, "open " + path.string());
  const uint8_t bad = 9;
  writeWeightBytes(fd, offset, std::span(&bad, 1));
  close(fd);
}

std::string key(uint8_t value) { return weightDigest(std::span(&value, 1)); }

constexpr uint64_t kGiB = uint64_t{1} << 30;

// The store under SPLASH_WEIGHT_CACHE and the content every entry is
// written with; builds counts the writes.
struct Cache {
  std::filesystem::path root;
  PreparedWeights store;
  std::vector<uint8_t> bytes;
  int builds = 0;

  explicit Cache(const std::filesystem::path &directory) : root(directory), bytes(128 * 1024) {
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 37);
  }
  // An entry of its own component, so that publishing it supersedes nothing.
  [[nodiscard]] PreparedWeight entry(uint8_t value, uint64_t size) const {
    return {key(value), size, "test/" + std::to_string(value), key(value), "/test"};
  }
  [[nodiscard]] PreparedWeight entry(uint8_t value) const { return entry(value, bytes.size()); }
  [[nodiscard]] WeightWriter writer() {
    return [this](int fd, const PreparationCheck &) {
      ++builds;
      writeWeightBytes(fd, 0, bytes);
    };
  }
  std::filesystem::path prepare(const PreparedWeight &weight, const PreparationGuards &guards = {}) {
    return store.prepare(weight, writer(), guards);
  }
  std::filesystem::path prepare(uint8_t value, const PreparationGuards &guards = {}) {
    return prepare(entry(value), guards);
  }
  // The writes run by preparing value.
  int buildsOf(uint8_t value, const PreparationGuards &guards = {}) {
    const int before = builds;
    static_cast<void>(prepare(value, guards));
    return builds - before;
  }
};

const PreparationCheck noWorkspace = [] { throw std::runtime_error("no conversion workspace"); };

// A staging entry an interrupted writer left.
void abandonStaging(const Cache &cache, uint8_t value) {
  const auto staging = cache.root / (key(value) + ".partial");
  std::filesystem::create_directory(staging);
  splash::test::writeFile(staging / "weights", "x");
}

bool staged(const Cache &cache, uint8_t value) {
  return std::filesystem::exists(cache.root / (key(value) + ".partial"));
}

long proofCount(const Cache &cache) {
  const std::filesystem::directory_iterator proofs(cache.root / "verified-v3");
  return std::distance(begin(proofs), end(proofs));
}

void coldPreparationWritesOnceAndWarmReuses(Cache &cache) {
  require(cache.buildsOf(1) == 1, "cold preparation did not write once");
  const auto path = cache.prepare(1);
  require(std::filesystem::file_size(path) == cache.bytes.size() &&
              WeightSource(path).digest() == weightDigest(cache.bytes),
          "prepared content differs");
  struct stat info{};
  require(stat(path.c_str(), &info) == 0 && !(info.st_mode & 0222), "prepared file is writable");
  require(cache.buildsOf(1) == 0, "warm preparation rebuilt weights");
}

void onlyAMissAdmitsWorkspace(Cache &cache) {
  static_cast<void>(cache.prepare(1));
  require(cache.buildsOf(1, {{}, noWorkspace}) == 0, "warm hit required workspace admission");
  rejects([&] { static_cast<void>(cache.prepare(10, {{}, noWorkspace})); }, "no conversion workspace",
          "cold preparation ignored workspace admission");
}

void warmHitsAreCancellable(Cache &cache) {
  static_cast<void>(cache.prepare(1));
  rejects([&] { static_cast<void>(cache.prepare(1, {[] { throw std::runtime_error("cancelled"); }})); },
          "cancelled", "warm hit ignored cancellation");
}

// A warm load must finish while an unrelated converter holds the lock, and
// leaves the staging that converter may own. Pipes order the two processes;
// the holder's alarm turns a deadlock into a failure.
void warmLoadDoesNotWaitForTheConverterLock(Cache &cache) {
  static_cast<void>(cache.prepare(1));
  abandonStaging(cache, 62);
  int ready[2], release[2];
  require(pipe(ready) == 0 && pipe(release) == 0, "lock fixture pipes");
  const pid_t holder = spawn([&] {
    alarm(5);
    const int lock = open((cache.root / "prepare.lock").c_str(), O_RDWR);
    if (lock < 0 || flock(lock, LOCK_EX)) return 1;
    char byte = 'x';
    if (write(ready[1], &byte, 1) != 1 || read(release[0], &byte, 1) != 1) return 2;
    return 0;
  });
  // With the holder the only writer of ready, the read ends if it exits
  // before taking the lock. The parent keeps its reading end of release, so
  // releasing a holder the alarm ended fails the check below, not the write.
  close(ready[1]);
  char signal = 'x';
  require(read(ready[0], &signal, 1) == 1, "the lock holder exited before taking the converter lock");
  const std::array<PreparedWeight, 1> warm{{cache.entry(1)}};
  cache.store.requireSpace(warm, {});
  static_cast<void>(cache.prepare(1, {{}, noWorkspace}));
  require(write(release[1], &signal, 1) == 1, "release the lock holder");
  for (int end : {ready[0], release[0], release[1]}) close(end);
  require(exitStatus(holder) == 0, "warm load waited for the converter lock");
  require(staged(cache, 62), "a warm load removed staging while the converter lock was held");
}

void diskChecksKeepTheReserveAndCoverTheModel(Cache &cache) {
  // 2 GiB stay free, to the byte, and nothing to write needs no reserve.
  constexpr uint64_t kReserve = uint64_t{2} << 30;
  requireWeightDiskSpace(0, 0);
  requireWeightDiskSpace(kReserve + 128, 128);
  rejects([] { requireWeightDiskSpace(kReserve + 127, 128); }, "not enough disk space", "disk reserve boundary");
  // The store checks its volume: a file 1 GiB smaller than the free space
  // does not fit.
  const uint64_t available = std::filesystem::space(cache.root).available;
  const std::array<PreparedWeight, 1> reserve{{cache.entry(22, available > kGiB ? available - kGiB : 1)}};
  rejects([&] { cache.store.requireSpace(reserve, {}); }, "not enough disk space", "disk reserve ignored");
  const std::array<PreparedWeight, 2> tooLarge{{cache.entry(20, UINT64_MAX / 2), cache.entry(21, UINT64_MAX / 2)}};
  rejects([&] { cache.store.requireSpace(tooLarge, {}); }, "not enough disk space", "model-wide disk budget ignored");
  require(!std::filesystem::exists(cache.root / key(20)), "disk preflight wrote a partial model");
}

// Publishing a file evicts the entries it supersedes, so preparing a model
// again under a new identity needs room for its largest file, not for all of
// it; a model that supersedes nothing still needs room for every file.
void supersededEntriesAreCredited(Cache &cache) {
  const uint64_t available = std::filesystem::space(cache.root).available;
  const uint64_t size = available / 4;
  std::vector<PreparedWeight> model, earlier;
  for (uint8_t i = 0; i < 8; ++i) {
    const std::string component = "model/layer-" + std::to_string(i) + ".bin";
    model.push_back({key(90 + i), size, component, key(110 + i), "/model"});
    earlier.push_back({key(100 + i), cache.bytes.size(), component, key(110 + i), "/model"});
  }
  rejects([&] { cache.store.requireSpace(model, {}); }, "not enough disk space", "a new model's budget was credited");
  // The earlier generation of each file, as large as its replacement, sparse.
  for (const auto &weight : earlier) {
    const auto path = cache.prepare(weight);
    std::filesystem::permissions(path, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
    std::filesystem::resize_file(path, size);
  }
  cache.store.requireSpace(model, {});
  // A file is written while the entry it replaces remains.
  const std::array<PreparedWeight, 1> large{{{key(98), available - kGiB, model[0].component, model[0].inputs, "/model"}}};
  rejects([&] { cache.store.requireSpace(large, {}); }, "not enough disk space", "a replaced entry was counted free");
  for (const auto &weight : earlier) std::filesystem::remove_all(cache.root / weight.key);
}

// Same-size corruption must not pass a metadata-only check.
void corruptionIsRepaired(Cache &cache) {
  const auto path = cache.prepare(1);
  corrupt(path, 0);
  require(cache.buildsOf(1) == 1 && WeightSource(path).digest() == weightDigest(cache.bytes),
          "corruption not repaired");
}

// Interrupted and ENOSPC writes do not publish anything and can be retried.
void failedWritesPublishNothing(Cache &cache) {
  rejects([&] {
    static_cast<void>(cache.store.prepare(cache.entry(2), [&](int output, const PreparationCheck &) {
      writeWeightBytes(output, 0, std::span(cache.bytes).first(64));
      throw std::system_error(ENOSPC, std::generic_category(), "fixture disk full");
    }, {}));
  }, "fixture disk full", "failed write accepted");
  require(!std::filesystem::exists(cache.root / key(2)), "partial file published");
  require(cache.buildsOf(2) == 1, "retry after a failed write did not write");
  rejects([&] { static_cast<void>(cache.prepare(3, {[] { throw std::runtime_error("memory pressure"); }})); },
          "memory pressure", "pressure ignored");
  require(!std::filesystem::exists(cache.root / key(3)), "pressure rejection published weights");
  rejects([&] { static_cast<void>(cache.prepare({"../outside", cache.bytes.size(), "test/outside", key(6), "/test"})); },
          "invalid prepared weight identity", "unsafe cache key accepted");
}

void crashedWriteIsReclaimed(Cache &cache) {
  constexpr int kCrashed = 7;
  const pid_t crash = spawn([&] {
    static_cast<void>(cache.store.prepare(cache.entry(4), [&](int output, const PreparationCheck &) {
      writeWeightBytes(output, 0, std::span(cache.bytes).first(64));
      _exit(kCrashed);
    }, {}));
    return 0;
  });
  require(exitStatus(crash) == kCrashed, "the writer did not crash");
  require(!std::filesystem::exists(cache.root / key(4)), "crash published partial weights");
  const std::array<PreparedWeight, 1> retry{{cache.entry(4)}};
  cache.store.requireSpace(retry, {});
  require(!std::filesystem::exists(cache.root / (key(4) + ".partial")), "abandoned staging not cleaned");
  require(cache.buildsOf(4) == 1, "retry after a crash did not write");
}

// Every converter-lock acquisition removes abandoned staging, as does a warm
// start that finds the lock free (warmLoadDoesNotWaitForTheConverterLock
// holds it).
void everyLockReclaimsAbandonedStaging(Cache &cache) {
  abandonStaging(cache, 60);
  static_cast<void>(cache.prepare(61));
  require(!staged(cache, 60), "a cold preparation left abandoned staging");
  abandonStaging(cache, 60);
  const std::array<PreparedWeight, 1> warm{{cache.entry(61)}};
  cache.store.requireSpace(warm, {});
  require(!staged(cache, 60), "a warm start with the lock free left abandoned staging");
}

// A warm start's reclaim never fails the load: a lock it cannot open and
// staging it cannot remove stay for a converter.
void warmLoadSkipsAReclaimItCannotRun(Cache &cache) {
  const std::array<PreparedWeight, 1> warm{{cache.entry(61)}};
  const auto lock = cache.root / "prepare.lock";
  require(chmod(lock.c_str(), 0400) == 0, "chmod the converter lock");
  cache.store.requireSpace(warm, {});
  require(chmod(lock.c_str(), 0600) == 0, "chmod the converter lock");
  abandonStaging(cache, 60);
  const auto staging = cache.root / (key(60) + ".partial");
  require(chmod(staging.c_str(), 0500) == 0, "chmod abandoned staging");
  cache.store.requireSpace(warm, {});
  require(chmod(staging.c_str(), 0700) == 0, "chmod abandoned staging");
}

// The device-keyed proofs Splash 1.1 wrote go at a lock acquisition.
void legacyProofsAreRemovedUnderTheLock(Cache &cache) {
  std::filesystem::create_directory(cache.root / "verified");
  splash::test::writeFile(cache.root / "verified" / "proof", "x");
  static_cast<void>(cache.prepare(63));
  require(!std::filesystem::exists(cache.root / "verified"), "legacy proofs were kept");
}

// An invalid entry is replaced together with its file's proof.
void replacingAnInvalidEntryDropsItsProof(Cache &cache) {
  static_cast<void>(cache.prepare(64));
  const long proofs = proofCount(cache);
  const auto record = cache.root / key(64) / "sha256";
  std::filesystem::permissions(record, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
  splash::test::writeFile(record, key(65));
  require(cache.buildsOf(64) == 1, "an entry recording another digest was reused");
  require(proofCount(cache) == proofs, "replacing an invalid entry kept its proof");
}

// Two processes requesting the same identity must run its writer only once.
void concurrentMissesWriteOnce(Cache &cache) {
  const auto counter = cache.root / "builds";
  const WeightWriter competing = [&](int output, const PreparationCheck &) {
    const int count = open(counter.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0600);
    require(count >= 0 && write(count, "x", 1) == 1, "write build counter");
    close(count);
    writeWeightBytes(output, 0, cache.bytes);
  };
  const pid_t child = spawn([&] {
    static_cast<void>(cache.store.prepare(cache.entry(5), competing, {}));
    return 0;
  });
  static_cast<void>(cache.store.prepare(cache.entry(5), competing, {}));
  require(exitStatus(child) == 0 && std::filesystem::file_size(counter) == 1,
          "concurrent cache miss rebuilt or corrupted weights");
}

void changedSourcesAreRejected(Cache &cache) {
  const auto path = cache.prepare(8);
  const WeightSource source(path);
  corrupt(path, 5);
  rejects([&] { source.checkUnchanged(); }, "source weights changed", "changed source was accepted");
  // An open descriptor surviving a rename must not hide path replacement.
  const WeightSource beforeReplacement(path);
  std::filesystem::rename(path, cache.root / "old-source");
  std::filesystem::copy_file(cache.root / "old-source", path);
  rejects([&] { beforeReplacement.checkUnchanged(); }, "source weights changed", "replaced source was accepted");
  // A source that cannot be opened is named, not taken for the cache.
  const auto missing = cache.root / "missing.gguf";
  rejects([&] { static_cast<void>(WeightSource(missing)); }, "open weight source " + missing.string(),
          "a missing source was reported without its path");
}

// A prepared file is mapped only as the cache verified it (WeightFile): not
// once replaced, even by the same bytes, until prepare verifies it again, and
// not once modified.
void onlyVerifiedFilesAreMapped(Cache &cache) {
  const auto path = cache.prepare(11);
  const auto map = [&] {
    const int file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(file >= 0, "open " + path.string());
    try {
      requireVerifiedFile(file, path);
    } catch (...) {
      close(file);
      throw;
    }
    close(file);
  };
  map();
  std::filesystem::copy_file(path, cache.root / "copy");
  std::filesystem::rename(cache.root / "copy", path);
  rejects(map, "changed after verification", "a replaced prepared file was mapped");
  require(cache.buildsOf(11) == 0, "an identical replacement was prepared again");
  map();
  corrupt(path, 0);
  rejects(map, "changed after verification", "a modified prepared file was mapped");
}

// Damaged memoization proofs must be recomputed, not trusted.
void damagedProofsAreRecomputed(Cache &cache) {
  static_cast<void>(cache.prepare(9));
  int damaged = 0;
  for (const auto &entry : std::filesystem::directory_iterator(cache.root / "verified-v3")) {
    const int proof = open(entry.path().c_str(), O_WRONLY | O_TRUNC);
    require(proof >= 0, "open proof fixture");
    const uint8_t bad = 9;
    writeWeightBytes(proof, 0, std::span(&bad, 1));
    close(proof);
    ++damaged;
  }
  require(damaged > 0, "preparation left no proof to damage");
  require(cache.buildsOf(9) == 0, "damaged proof forced an unnecessary rebuild");
}

// Publishing an entry removes the earlier preparations of its component from
// the same source data; others stay.
void publishingSupersedesEarlierPreparations(Cache &cache) {
  static_cast<void>(cache.prepare(36));
  const PreparedWeight older{key(30), cache.bytes.size(), "target/layer-0.bin", key(40), "/models/a"};
  const PreparedWeight otherData{key(31), cache.bytes.size(), "target/layer-0.bin", key(41), "/models/b"};
  const PreparedWeight otherComponent{key(32), cache.bytes.size(), "target/head.bin", key(40), "/models/a"};
  for (const auto &weight : {older, otherData, otherComponent}) static_cast<void>(cache.prepare(weight));
  static_cast<void>(cache.prepare({key(35), cache.bytes.size(), "target/layer-0.bin", key(40), "/models/a"}));
  const auto kept = [&](uint8_t value) { return std::filesystem::exists(cache.root / key(value)); };
  require(!kept(30) && kept(31) && kept(32) && kept(35) && kept(36), "superseded entries were kept or others removed");
}

} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory directory("splash-prepared-weights");
    setenv("SPLASH_WEIGHT_CACHE", directory.path().c_str(), 1);
    Cache cache(directory.path());
    coldPreparationWritesOnceAndWarmReuses(cache);
    onlyAMissAdmitsWorkspace(cache);
    warmHitsAreCancellable(cache);
    warmLoadDoesNotWaitForTheConverterLock(cache);
    diskChecksKeepTheReserveAndCoverTheModel(cache);
    supersededEntriesAreCredited(cache);
    corruptionIsRepaired(cache);
    failedWritesPublishNothing(cache);
    crashedWriteIsReclaimed(cache);
    everyLockReclaimsAbandonedStaging(cache);
    warmLoadSkipsAReclaimItCannotRun(cache);
    legacyProofsAreRemovedUnderTheLock(cache);
    replacingAnInvalidEntryDropsItsProof(cache);
    concurrentMissesWriteOnce(cache);
    changedSourcesAreRejected(cache);
    onlyVerifiedFilesAreMapped(cache);
    damagedProofsAreRecomputed(cache);
    publishingSupersedesEarlierPreparations(cache);
    std::cout << "prepared weights: content, reuse, corruption, interruption, reclaimed staging and proofs, "
                 "pressure, concurrency and superseded entries PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
