#include "model/PreparedWeights.hpp"
#include "StderrLine.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <thread>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

[[noreturn]] void fail(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

void run(const PreparationCheck &check) {
  if (check) check();
}

// A source is a model's own file, not the cache's, so its failures name it.
[[noreturn]] void failSource(const char *operation, const std::filesystem::path &path) {
  throw std::system_error(errno, std::generic_category(),
                          std::string(operation) + " " + path.string());
}

int openSource(const std::filesystem::path &path) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) failSource("open weight source", path);
  return fd;
}

// Whether name is a cache key: 64 lowercase hex digits.
bool isKey(std::string_view name) {
  return name.size() == 64 && name.find_first_not_of("0123456789abcdef") == name.npos;
}

// Where an entry is written before it is published. Every staging entry a
// converter finds under the lock is abandoned.
std::filesystem::path stagingPath(const std::filesystem::path &root, std::string_view key) {
  return root / (std::string(key) + ".partial");
}

// Leave room for the OS and other applications; this is a disk reserve, not
// a promise that concurrent system activity can never exhaust the volume.
constexpr uint64_t kDiskReserveBytes = uint64_t{2} << 30;

class Descriptor final {
public:
  explicit Descriptor(int fd) : fd_(fd) { if (fd < 0) fail("open prepared weights"); }
  Descriptor(Descriptor &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  ~Descriptor() { if (fd_ >= 0) close(fd_); }
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  operator int() const noexcept { return fd_; }
private:
  int fd_;
};

// The converter lock of a cache, which every writer holds. The OS releases
// it when its holder exits, crashes included.
class PreparationLock final {
public:
  // Waits while another process holds the lock; check runs meanwhile.
  PreparationLock(const std::filesystem::path &root, const PreparationCheck &check) : file_(lockFile(root)) {
    while (!take(file_)) {
      run(check);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  // The lock, or nothing when another process holds it; never waits.
  [[nodiscard]] static std::optional<PreparationLock> tryAcquire(const std::filesystem::path &root) {
    Descriptor file(lockFile(root));
    if (!take(file)) return std::nullopt;
    return PreparationLock(std::move(file));
  }
private:
  explicit PreparationLock(Descriptor file) : file_(std::move(file)) {}
  static Descriptor lockFile(const std::filesystem::path &root) {
    return Descriptor(open((root / "prepare.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600));
  }
  // False when another process holds the lock.
  static bool take(int file) {
    while (flock(file, LOCK_EX | LOCK_NB) < 0) {
      if (errno == EWOULDBLOCK) return false;
      if (errno != EINTR) fail("lock weight preparation");
    }
    return true;
  }
  Descriptor file_;
};

// Whether two stats describe the same, unmodified file.
bool sameFile(const struct stat &a, const struct stat &b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
         a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec &&
         a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
         a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec &&
         a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec;
}

std::filesystem::path cacheRoot() {
  if (const char *path = std::getenv("SPLASH_WEIGHT_CACHE"); path && *path) return path;
  const char *home = std::getenv("HOME");
  if (!home || !*home) throw std::runtime_error("cannot locate prepared weight cache");
  return std::filesystem::path(home) / "Library/Caches/Splash/weights";
}

// SHA-256 of bytes [from, end) of fd, which must not change while it is read.
std::string fileDigest(int fd, uint64_t from, const PreparationCheck &check) {
  struct stat before{}, after{};
  if (fstat(fd, &before)) fail("stat source weights");
  if (!S_ISREG(before.st_mode) || before.st_size < 0 || uint64_t(before.st_size) < from)
    throw std::runtime_error("weights must be a regular file");
  run(check);
  std::vector<uint8_t> buffer(1024 * 1024);
  CC_SHA256_CTX context;
  CC_SHA256_Init(&context);
  for (uint64_t at = from; at < uint64_t(before.st_size); at += buffer.size()) {
    run(check);
    const auto part = std::span(buffer).first(std::min<uint64_t>(buffer.size(), before.st_size - at));
    readWeightBytes(fd, at, part);
    CC_SHA256_Update(&context, part.data(), static_cast<CC_LONG>(part.size()));
  }
  if (fstat(fd, &after)) fail("stat source weights after read");
  if (!sameFile(before, after)) throw std::runtime_error("weight file changed while reading");
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &context);
  return digestHex(digest);
}

// Where the proofs of hashed files are.
constexpr std::string_view kProofDirectory = "verified-v3";

// The proof of a hashed file names the digest of its bytes [from, end) and
// holds for this inode, length, birth time, mtime and ctime only. The device
// is left out: an external volume's follows its BSD disk number, which can
// change at every mount, and the rest already identify the file (a
// block-level clone that reproduces them reproduces its bytes).
std::string verificationKey(const struct stat &state, uint64_t from) {
  std::ostringstream identity;
  identity << "splash-verified-file-v3 " << state.st_ino << ' ' << state.st_size << ' '
           << state.st_mtimespec.tv_sec << ' ' << state.st_mtimespec.tv_nsec << ' '
           << state.st_ctimespec.tv_sec << ' ' << state.st_ctimespec.tv_nsec << ' '
           << state.st_birthtimespec.tv_sec << ' ' << state.st_birthtimespec.tv_nsec << ' ' << from;
  return weightDigest(identity.str());
}

// Writes contents to the empty file and flushes it.
void writeSmallFile(int file, std::string_view contents, const char *what) {
  writeWeightBytes(file, 0, {reinterpret_cast<const uint8_t *>(contents.data()), contents.size()});
  if (fsync(file)) fail(what);
}

std::string verificationRecord(std::string_view key, std::string_view digest) {
  return std::string(digest) + weightDigest(std::string(key) + std::string(digest));
}

void rememberDigest(const std::filesystem::path &root, const struct stat &state, uint64_t from,
                    const std::string &digest) {
  const auto directory = root / kProofDirectory;
  std::filesystem::create_directories(directory);
  const auto key = verificationKey(state, from);
  std::string temporary = (directory / ".pending-XXXXXX").string();
  Descriptor file(mkstemp(temporary.data()));
  try {
    writeSmallFile(file, verificationRecord(key, digest), "flush weight verification");
    std::filesystem::rename(temporary, directory / key);
  } catch (...) {
    unlink(temporary.c_str());
    throw;
  }
}

// The digest a valid proof remembers for bytes [from, end) of the file of
// state; empty when none does.
std::string provenDigest(const struct stat &state, uint64_t from, const std::filesystem::path &root) {
  const auto key = verificationKey(state, from);
  const int existing = open((root / kProofDirectory / key).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (existing < 0) {
    if (errno != ENOENT) fail("open weight verification");
    return {};
  }
  Descriptor proof(existing);
  struct stat info{};
  if (fstat(proof, &info) || !S_ISREG(info.st_mode) || info.st_size != 128) return {};
  std::array<uint8_t, 128> bytes;
  readWeightBytes(proof, 0, bytes);
  const std::string record(bytes.begin(), bytes.end());
  const auto value = record.substr(0, 64);
  if (value.find_first_not_of("0123456789abcdef") != value.npos || verificationRecord(key, value) != record) return {};
  return value;
}

// The content hash is computed on first use. A proof is reusable only for the
// same inode, length, birth time, mtime and ctime. Replacing or writing even a
// same-size file invalidates it. No upstream file or extended attribute is
// modified. This avoids rereading two entire models at every warm startup.
std::string verifiedDigest(int fd, uint64_t from, const std::filesystem::path &path,
                           const std::filesystem::path &root, const PreparationCheck &check) {
  run(check);
  struct stat before{}, after{};
  if (fstat(fd, &before)) fail("stat verified weights");
  if (!S_ISREG(before.st_mode) || before.st_size < 0) throw std::runtime_error("weights must be a regular file");
  std::string digest = provenDigest(before, from, root);
  const bool missing = digest.empty();
  if (missing) {
    writeStderrLine("Hashing " + path.string() + " (" +
                    std::to_string((uint64_t(before.st_size) - from) / (1024 * 1024)) +
                    " MiB) once; later starts reuse the result");
    digest = fileDigest(fd, from, check);
  }
  if (fstat(fd, &after)) fail("stat verified weights after read");
  if (!sameFile(before, after)) throw std::runtime_error("weight file changed during verification");
  if (missing) rememberDigest(root, before, from, digest);
  return digest;
}

// What an entry records in its source file: the component, the digest of the
// source data it was written from and the source path.
constexpr std::string_view kProvenance = "splash-prepared-weight-v1";

std::string provenance(const PreparedWeight &weight) {
  std::ostringstream text;
  text << kProvenance << "\ncomponent " << weight.component << "\ninputs " << weight.inputs << "\nsource "
       << weight.source << '\n';
  return text.str();
}

// The first lines of the source file of the entry at `directory`.
std::vector<std::string> sourceLines(const std::filesystem::path &directory) {
  std::ifstream stream(directory / "source");
  std::vector<std::string> lines;
  for (std::string line; lines.size() < 5 && std::getline(stream, line);) lines.push_back(line);
  return lines;
}

// Whether an entry whose source file starts with `lines` is an earlier
// preparation of what weight holds: the same component from the same source
// data under another key (a new preparation identity or plan).
bool supersedes(const PreparedWeight &weight, std::span<const std::string> lines) {
  return lines.size() == 4 && lines[0] == kProvenance && lines[1] == "component " + weight.component &&
         lines[2] == "inputs " + weight.inputs;
}

// The proof of the file of the entry at `directory`; none when it has no
// file.
std::optional<std::filesystem::path> proofOf(const std::filesystem::path &root,
                                             const std::filesystem::path &directory) {
  struct stat state{};
  if (stat((directory / "weights").c_str(), &state)) return std::nullopt;
  return root / kProofDirectory / verificationKey(state, 0);
}

// Removes a complete entry and its file's proof. Its directory is first
// renamed to staging, which any converter reclaims if the removal is
// interrupted; a process mapping its file keeps the file until it unmaps it.
void removeEntry(const std::filesystem::path &root, const std::filesystem::path &directory) {
  const auto proof = proofOf(root, directory);
  const auto staging = stagingPath(root, directory.filename().string());
  std::error_code error;
  std::filesystem::remove_all(staging, error);
  std::filesystem::rename(directory, staging, error);
  if (error) return;
  std::filesystem::remove_all(staging, error);
  if (proof) unlink(proof->c_str());
}

// Removes what interrupted writers left, under the converter lock, which
// every live writer holds: each staging entry, whichever model or version
// wrote it, and the proofs of an earlier key format.
void reclaimAbandoned(const std::filesystem::path &root) {
  for (const auto &entry : std::filesystem::directory_iterator(root))
    if (entry.path().extension() == ".partial") std::filesystem::remove_all(entry.path());
  // v2 proofs keyed by device, written by Splash 1.1; delete this line in the
  // release after next.
  std::filesystem::remove_all(root / "verified");
}

// The digest the entry at `directory` records for its file; empty when it
// records none.
std::string recordedDigest(const std::filesystem::path &directory) {
  const int manifest = open((directory / "sha256").c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (manifest < 0) {
    if (errno == ENOENT) return {};
    fail("open cached weight digest");
  }
  Descriptor hashFile(manifest);
  struct stat state{};
  if (fstat(hashFile, &state)) fail("stat cached weight digest");
  if (!S_ISREG(state.st_mode) || state.st_size != 64) return {};
  std::array<uint8_t, 64> digest{};
  readWeightBytes(hashFile, 0, digest);
  return std::string(digest.begin(), digest.end());
}

bool complete(const std::filesystem::path &directory, uint64_t bytes,
              const PreparationCheck &check) {
  const int input = open((directory / "weights").c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (input < 0) {
    if (errno == ENOENT) return false;
    fail("open cached weights");
  }
  Descriptor file(input);
  struct stat state{};
  if (fstat(file, &state)) fail("stat cached weights");
  if (!S_ISREG(state.st_mode) || state.st_size < 0 || uint64_t(state.st_size) != bytes)
    return false;
  const std::string recorded = recordedDigest(directory);
  return !recorded.empty() &&
         verifiedDigest(file, 0, directory / "weights", directory.parent_path(), check) == recorded;
}

// Writes the file of weight into staging and seals it: read-only, flushed to
// the drive, its digest remembered and recorded beside it with its provenance.
void writeStaged(const std::filesystem::path &root, const std::filesystem::path &staging, const PreparedWeight &weight,
                 const WeightWriter &write, const PreparationGuards &guards) {
  Descriptor file(open((staging / "weights").c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600));
  // Do not let dirty filesystem pages grow into a hidden model-sized buffer.
  if (fcntl(file, F_NOCACHE, 1)) fail("set preparation uncached I/O");
  fstore_t allocation{};
  allocation.fst_flags = F_ALLOCATEALL;
  allocation.fst_posmode = F_PEOFPOSMODE;
  allocation.fst_length = static_cast<off_t>(weight.bytes);
  if (fcntl(file, F_PREALLOCATE, &allocation)) fail("reserve prepared weight disk space");
  if (ftruncate(file, static_cast<off_t>(weight.bytes))) fail("size prepared weights");
  write(file, [&] {
    run(guards.check);
    run(guards.admitConversion);
  });
  run(guards.unchanged);
  struct stat state{};
  if (fstat(file, &state)) fail("stat prepared weights");
  if (state.st_size < 0 || uint64_t(state.st_size) != weight.bytes)
    throw std::runtime_error("prepared weight size changed");
  const std::string digest = fileDigest(file, 0, guards.check);
  // fsync leaves the data in the drive's cache on macOS; the proof below must
  // not outlive a power loss that the data does not survive. F_FULLFSYNC
  // fails on file systems that lack it, where fsync is the strongest flush.
  if (fchmod(file, 0400) || (fcntl(file, F_FULLFSYNC) && fsync(file)))
    fail("flush prepared weights");
  if (fstat(file, &state)) fail("stat completed weights");
  rememberDigest(root, state, 0, digest);
  writeSmallFile(Descriptor(open((staging / "sha256").c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0400)),
                 digest, "flush prepared weight digest");
  writeSmallFile(Descriptor(open((staging / "source").c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0400)),
                 provenance(weight), "flush prepared weight source");
}

// Renames the sealed staging entry to its key. Invalid cached generations may
// be replaced, with their proofs; existing read-only mappings retain their
// inode. No valid generation is rewritten in place.
void publish(const std::filesystem::path &root, const std::filesystem::path &staging,
             const std::filesystem::path &destination) {
  const auto replaced = proofOf(root, destination);
  std::filesystem::remove_all(destination);
  if (replaced) unlink(replaced->c_str());
  std::filesystem::rename(staging, destination);
  Descriptor directory(open(root.c_str(), O_RDONLY | O_CLOEXEC));
  if (fsync(directory)) fail("flush prepared weight directory");
}

// Removes the earlier preparations weight supersedes, still under the
// converter lock, which every writer holds.
void evictSuperseded(const std::filesystem::path &root, const PreparedWeight &weight) {
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(root, error)) {
    const auto name = entry.path().filename().string();
    if (name != weight.key && isKey(name) && supersedes(weight, sourceLines(entry.path())))
      removeEntry(root, entry.path());
  }
}

// The free space preparing the missing weights needs. Publishing a file
// evicts the entries it supersedes (evictSuperseded), each credited here to
// one file, so the model needs what its files add beyond those entries, plus
// the largest file written while the entries it replaces remain. Other
// generations stay and are not credited.
uint64_t requiredBytes(const std::filesystem::path &root, std::span<const PreparedWeight *const> missing,
                       const PreparationCheck &check) {
  std::vector<uint64_t> evicted(missing.size());
  for (const auto &entry : std::filesystem::directory_iterator(root)) {
    run(check);
    const auto name = entry.path().filename().string();
    struct stat state{};
    if (!isKey(name) || lstat((entry.path() / "weights").c_str(), &state) || !S_ISREG(state.st_mode)) continue;
    const auto lines = sourceLines(entry.path());
    for (size_t i = 0; i < missing.size(); ++i)
      if (name != missing[i]->key && supersedes(*missing[i], lines)) {
        evicted[i] += uint64_t(state.st_size);
        break;
      }
  }
  uint64_t added = 0, largest = 0;
  for (size_t i = 0; i < missing.size(); ++i) {
    const uint64_t replacing = std::min(missing[i]->bytes, evicted[i]);
    if (missing[i]->bytes - replacing > std::numeric_limits<uint64_t>::max() - added)
      throw std::overflow_error("prepared model size overflow");
    added += missing[i]->bytes - replacing;
    largest = std::max(largest, replacing);
  }
  if (largest > std::numeric_limits<uint64_t>::max() - added)
    throw std::overflow_error("prepared model size overflow");
  return added + largest;
}

} // namespace

void readWeightBytes(int fd, uint64_t offset, std::span<uint8_t> bytes) {
  if (bytes.size() > uint64_t(std::numeric_limits<off_t>::max()) ||
      offset > uint64_t(std::numeric_limits<off_t>::max()) - bytes.size())
    throw std::overflow_error("weight read offset overflow");
  while (!bytes.empty()) {
    const ssize_t count = pread(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) fail("read weight bytes");
    if (!count) throw std::runtime_error("truncated weight file");
    offset += count;
    bytes = bytes.subspan(count);
  }
}

void writeWeightBytes(int fd, uint64_t offset, std::span<const uint8_t> bytes) {
  if (bytes.size() > uint64_t(std::numeric_limits<off_t>::max()) ||
      offset > uint64_t(std::numeric_limits<off_t>::max()) - bytes.size())
    throw std::overflow_error("weight write offset overflow");
  while (!bytes.empty()) {
    const ssize_t count = pwrite(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) fail("write prepared weights");
    if (!count) throw std::runtime_error("zero-length prepared weight write");
    offset += count;
    bytes = bytes.subspan(count);
  }
}

std::string weightDigest(std::span<const uint8_t> bytes) {
  if (bytes.size() > std::numeric_limits<CC_LONG>::max())
    throw std::overflow_error("weight identity is too large");
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest);
  return digestHex(digest);
}

std::string weightDigest(std::string_view text) {
  return weightDigest({reinterpret_cast<const uint8_t *>(text.data()), text.size()});
}

struct WeightSource::Impl {
  std::filesystem::path path;
  Descriptor file;
  struct stat state{};
  PreparationCheck check;
  uint64_t dataOffset = 0;
  // The tensor data's digest, once hashed.
  std::optional<std::string> digest;
  Impl(const std::filesystem::path &path, PreparationCheck check)
      : path(path), file(openSource(path)), check(std::move(check)) {
    if (fstat(file, &state)) failSource("stat weight source", path);
  }
};

WeightSource::WeightSource(const std::filesystem::path &path, PreparationCheck check)
    : impl_(std::make_unique<Impl>(path, std::move(check))) { checkUnchanged(); }
WeightSource::~WeightSource() = default;
const std::filesystem::path &WeightSource::path() const noexcept { return impl_->path; }
int WeightSource::descriptor() const noexcept { return impl_->file; }
uint64_t WeightSource::bytes() const noexcept { return uint64_t(impl_->state.st_size); }
void WeightSource::setDataOffset(uint64_t offset) {
  if (offset > bytes()) throw std::runtime_error("weight source data starts past its end: " + impl_->path.string());
  impl_->dataOffset = offset;
}
uint64_t WeightSource::dataOffset() const noexcept { return impl_->dataOffset; }
void WeightSource::readData(uint64_t offset, std::span<uint8_t> bytes) const {
  if (offset > std::numeric_limits<uint64_t>::max() - impl_->dataOffset)
    throw std::overflow_error("weight read offset overflow");
  readWeightBytes(impl_->file, impl_->dataOffset + offset, bytes);
}
const std::string &WeightSource::digest() const {
  if (!impl_->digest)
    impl_->digest = verifiedDigest(impl_->file, impl_->dataOffset, impl_->path, cacheRoot(), impl_->check);
  return *impl_->digest;
}
void WeightSource::checkUnchanged() const {
  struct stat current{};
  if (fstat(impl_->file, &current)) fail("stat weight source");
  struct stat named{};
  if (stat(impl_->path.c_str(), &named)) fail("stat weight source path");
  if (!sameFile(impl_->state, current) || !sameFile(impl_->state, named))
    throw std::runtime_error("source weights changed during preparation; retry with an immutable source");
}

void SourceTensor::read(uint64_t at, std::span<uint8_t> destination) const {
  if (at > bytes || destination.size() > bytes - at) throw std::out_of_range("source tensor read is out of bounds");
  file->readData(offset + at, destination);
}

void SourceTensor::copy(int destination, uint64_t to, std::span<uint8_t> staging,
                        const PreparationCheck &check) const {
  if (bytes && staging.empty()) throw std::invalid_argument("weight copy staging is empty");
  for (uint64_t at = 0; at < bytes; at += staging.size()) {
    run(check);
    const auto piece = staging.first(std::min<uint64_t>(staging.size(), bytes - at));
    read(at, piece);
    writeWeightBytes(destination, to + at, piece);
  }
}

void SourceTensor::identify(WeightIdentity &identity) const { identity.input(*file, offset, bytes, dtype, shape); }

WeightIdentity &WeightIdentity::input(const WeightSource &source, uint64_t offset, uint64_t bytes,
                                      std::string_view type, std::span<const uint64_t> shape) {
  const std::string &digest = source.digest();
  digests_.insert(digest);
  text_ << "input " << digest << ' ' << offset << ' ' << bytes << ' ' << type;
  for (uint64_t dimension : shape) text_ << ' ' << dimension;
  text_ << '\n';
  return *this;
}

PreparedWeight WeightIdentity::weight(uint64_t bytes, std::string component, std::string source) const {
  std::string inputs;
  for (const auto &digest : digests_) inputs += digest;
  return {weightDigest(text_.str()), bytes, std::move(component), weightDigest(inputs), std::move(source)};
}

void requireWeightDiskSpace(uint64_t available, uint64_t required) {
  if (required && (available < kDiskReserveBytes || required > available - kDiskReserveBytes))
    throw std::runtime_error("not enough disk space to prepare weights: need " +
        std::to_string(required) + " bytes plus a 2 GiB free-space reserve");
}

void requireVerifiedFile(int descriptor, const std::filesystem::path &path) {
  struct stat state{};
  if (fstat(descriptor, &state)) fail("stat prepared weights");
  const auto directory = path.parent_path();
  const std::string proven = provenDigest(state, 0, directory.parent_path());
  if (proven.empty() || proven != recordedDigest(directory))
    throw std::runtime_error("prepared weights changed after verification: " + path.string());
}

PreparedWeights::PreparedWeights() : root_(cacheRoot()) {}

void PreparedWeights::requireSpace(std::span<const PreparedWeight> weights,
                                   const PreparationCheck &check) const {
  const auto missing = [&] {
    std::vector<const PreparedWeight *> result;
    for (const auto &weight : weights) {
      if (!isKey(weight.key) || !weight.bytes)
        throw std::invalid_argument("invalid prepared weight identity or size");
      run(check);
      if (!complete(root_ / weight.key, weight.bytes, check)) result.push_back(&weight);
    }
    return result;
  };
  if (missing().empty()) {
    // A warm start reclaims abandoned writes only when no converter runs: it
    // never waits for the lock. A cache it cannot lock or clean (read-only,
    // or without flock) still serves its complete entries, so a failed
    // reclaim leaves the writes to the next converter.
    try {
      if (const auto lock = PreparationLock::tryAcquire(root_)) reclaimAbandoned(root_);
    } catch (const std::system_error &) {
    }
    return;
  }
  std::filesystem::create_directories(root_);
  PreparationLock lock(root_, check);
  // Abandoned writes go before a retry is budgeted.
  reclaimAbandoned(root_);
  requireWeightDiskSpace(std::filesystem::space(root_).available, requiredBytes(root_, missing(), check));
}

std::filesystem::path PreparedWeights::prepare(const PreparedWeight &weight, const WeightWriter &write,
                                               const PreparationGuards &guards) const {
  if (!isKey(weight.key) || !weight.bytes || weight.bytes > uint64_t(std::numeric_limits<off_t>::max()))
    throw std::invalid_argument("invalid prepared weight identity or size");
  run(guards.check);
  run(guards.unchanged);
  const auto destination = root_ / weight.key;
  // Immutable hits need neither conversion admission nor the converter lock.
  if (complete(destination, weight.bytes, guards.check)) {
    run(guards.unchanged);
    return destination / "weights";
  }
  std::filesystem::create_directories(root_);
  // One converter per user cache: concurrent cold loads cannot multiply the
  // bounded conversion workspace. OS locks are released on crashes.
  PreparationLock lock(root_, guards.check);
  reclaimAbandoned(root_);
  if (complete(destination, weight.bytes, guards.check)) {
    run(guards.unchanged);
    return destination / "weights";
  }
  run(guards.check);
  run(guards.admitConversion);
  const auto started = std::chrono::steady_clock::now();
  writeStderrLine("Preparing weights: " + weight.component);
  // This name belongs only to this key under the converter lock, and
  // reclaimAbandoned removed any earlier writer's.
  const auto staging = stagingPath(root_, weight.key);
  requireWeightDiskSpace(std::filesystem::space(root_).available, weight.bytes);
  std::filesystem::create_directory(staging);
  try {
    writeStaged(root_, staging, weight, write, guards);
    publish(root_, staging, destination);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove_all(staging, ignored);
    throw;
  }
  evictSuperseded(root_, weight);
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::ostringstream prepared;
  prepared << "Prepared " << weight.component << " in " << seconds << " s";
  writeStderrLine(prepared.str());
  run(guards.unchanged);
  return destination / "weights";
}

} // namespace splash::model
