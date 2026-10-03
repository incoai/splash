#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace splash::model {

// Shared by source adapters. Preparation owns bounded buffers; inference only
// opens completed, immutable artifacts through WeightFile. The admission reserve
// includes source metadata and small staging buffers; it is not a model copy.
inline constexpr uint64_t kWeightPreparationWorkspaceBytes = 64 * 1024 * 1024;
// The staging of one conversion step, its input and output together. Every
// adapter sizes its chunks to it, whatever the tensor, layer or expert count.
inline constexpr uint64_t kWeightPreparationStagingBytes = kWeightPreparationWorkspaceBytes / 2;

using PreparationCheck = std::function<void()>;

// A prepared file: its cache key and size, the component it is (such as
// target/layer-0.bin), the digest of the source data it is written from and
// where that source is. An entry of the same component and inputs under
// another key is an earlier preparation, which publishing this one removes.
struct PreparedWeight {
  std::string key;
  uint64_t bytes;
  std::string component;
  std::string inputs;
  std::string source;
};

void readWeightBytes(int descriptor, uint64_t offset, std::span<uint8_t> bytes);
void writeWeightBytes(int descriptor, uint64_t offset, std::span<const uint8_t> bytes);
// The lowercase hex SHA-256 of bytes or text.
[[nodiscard]] std::string weightDigest(std::span<const uint8_t> bytes);
[[nodiscard]] std::string weightDigest(std::string_view text);
// A SHA-256 digest in lowercase hex.
[[nodiscard]] inline std::string digestHex(std::span<const uint8_t, 32> digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(2 * digest.size());
  for (uint8_t byte : digest) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

// A source file, opened once; checkUnchanged throws when it was modified or
// replaced since. Its parser reads the metadata through descriptor() and
// sets where the tensor data starts; tensor offsets are relative to it. The
// digest of the tensor data is computed on first use, after the parser has
// validated the metadata, and remembered for this file identity, so a warm
// start does not read the file again.
class WeightSource final {
public:
  explicit WeightSource(const std::filesystem::path &path, PreparationCheck check = {});
  ~WeightSource();
  WeightSource(const WeightSource &) = delete;
  WeightSource &operator=(const WeightSource &) = delete;
  [[nodiscard]] const std::filesystem::path &path() const noexcept;
  [[nodiscard]] int descriptor() const noexcept;
  // The file's size when it was opened.
  [[nodiscard]] uint64_t bytes() const noexcept;
  void setDataOffset(uint64_t offset);
  [[nodiscard]] uint64_t dataOffset() const noexcept;
  // Bytes [offset, offset + size) of the tensor data.
  void readData(uint64_t offset, std::span<uint8_t> bytes) const;
  // SHA-256 of the tensor data: editing only the metadata keeps the identity
  // of every tensor.
  [[nodiscard]] const std::string &digest() const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class WeightIdentity;

// A tensor of a source file: dtype and shape as the file names them, and
// where its bytes are in the file's tensor data.
struct SourceTensor final {
  const WeightSource *file = nullptr;
  std::string dtype;
  std::vector<uint64_t> shape;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  void read(uint64_t at, std::span<uint8_t> destination) const;
  // Copies the tensor to destination at `to` through staging; check runs
  // before each piece.
  void copy(int destination, uint64_t to, std::span<uint8_t> staging, const PreparationCheck &check) const;
  // Records the tensor as an input of a prepared file.
  void identify(WeightIdentity &identity) const;
};

// A field of a plan record: an integer, which prints as its value (a
// character type would print as a character), or a string without
// whitespace (the separator).
template <class T>
concept IdentityField =
    (std::is_integral_v<T> && !std::is_same_v<T, char> && !std::is_same_v<T, signed char> &&
     !std::is_same_v<T, unsigned char> && !std::is_same_v<T, char8_t>) ||
    std::is_convertible_v<const T &, std::string_view>;

// What a prepared file's key is the SHA-256 of: the adapter's preparation
// identity, its plan as one record per line, and every source tensor it
// reads, by the digest of its file's tensor data, its offset there, size,
// type and shape.
class WeightIdentity final {
public:
  explicit WeightIdentity(std::string_view preparation) { text_ << preparation << '\n'; }
  template <IdentityField... Fields> WeightIdentity &record(const Fields &...fields) {
    (field(fields), ...);
    text_ << '\n';
    return *this;
  }
  // Bytes [offset, offset + bytes) of source's tensor data.
  WeightIdentity &input(const WeightSource &source, uint64_t offset, uint64_t bytes, std::string_view type,
                        std::span<const uint64_t> shape);
  [[nodiscard]] PreparedWeight weight(uint64_t bytes, std::string component, std::string source) const;
private:
  template <class T> void field(const T &value) {
    if constexpr (std::is_integral_v<T>) {
      text_ << +value << ' ';
    } else {
      const std::string_view text(value);
      if (text.empty() || text.find_first_of(" \t\n") != text.npos)
        throw std::invalid_argument("prepared weight identity field is empty or has whitespace");
      text_ << text << ' ';
    }
  }
  std::ostringstream text_;
  std::set<std::string> digests_;
};

// The callbacks of one load. check runs throughout (cancellation, memory
// pressure), also while waiting for the converter lock or hashing;
// admitConversion admits the conversion workspace on a cache miss, before
// anything is allocated and again before each chunk; unchanged throws when a
// source changed after its tensor data was hashed.
struct PreparationGuards {
  PreparationCheck check{};
  PreparationCheck admitConversion{};
  PreparationCheck unchanged{};
};

// Writes a prepared file into its empty, preallocated destination; admit
// runs before each chunk of conversion work.
using WeightWriter = std::function<void(int destination, const PreparationCheck &admit)>;

// Throws unless `required` bytes fit in `available` free bytes and leave a
// 2 GiB free-space reserve; a model with nothing to write needs no reserve.
void requireWeightDiskSpace(uint64_t available, uint64_t required);

// Throws unless descriptor, open on the prepared file at path, is a file
// whose digest the cache verified and remembers for this unmodified file,
// equal to the one its entry records. WeightFile maps a prepared file only
// then, so a file that replaced or changed it after prepare returned its
// path is refused.
void requireVerifiedFile(int descriptor, const std::filesystem::path &path);

// The cache is SPLASH_WEIGHT_CACHE, or ~/Library/Caches/Splash/weights.
class PreparedWeights final {
public:
  PreparedWeights();
  // Check the missing model before writing its first artifact: the space its
  // files add beyond the entries they supersede, which publishing them
  // evicts, and its largest file, written while the entries it replaces
  // remain. Completed layers remain reusable after an interruption; they are
  // not partial files, which this removes under the converter lock first. A
  // complete model takes the lock only when it is free, never waiting for it,
  // and a cache it cannot lock or clean does not fail it.
  void requireSpace(std::span<const PreparedWeight> weights,
                    const PreparationCheck &check) const;
  // The complete file of weight: reused, or written now under the converter
  // lock and published atomically; writer failures never publish partial
  // data. The sources are checked unchanged before, after writing and before
  // the path is returned, so no file of a modified source is published or
  // used.
  [[nodiscard]] std::filesystem::path prepare(const PreparedWeight &weight, const WeightWriter &write,
                                              const PreparationGuards &guards) const;

private:
  std::filesystem::path root_;
};

} // namespace splash::model
