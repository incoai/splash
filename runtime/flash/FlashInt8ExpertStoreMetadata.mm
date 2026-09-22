#include "flash/FlashInt8ExpertStoreMetadata.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <charconv>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;
constexpr uint64_t kMaximumManifestBytes = 8ULL << 20;
constexpr std::string_view kSchema = "splash-flash-int8-expert-store-v1";
constexpr std::string_view kCoefficientPolicy =
    "source_q4_g64_f32_separate_multiply_add_then_bf16_rne_v1";
constexpr std::string_view kQuantizationFormat = "signed-symmetric-int8-rowwise-f32-scale";
constexpr std::string_view kRounding =
    "F32 absmax/127; F32 division; nearest-even integer; clamp [-127,127]; zero row scale=1";

[[noreturn]] void fail(std::string_view message) {
  throw std::invalid_argument("Flash INT8 expert store: " + std::string(message));
}

std::string text(id value) {
  if (![value isKindOfClass:[NSString class]]) fail("expected string metadata");
  NSString *string = value;
  const char *bytes = string.UTF8String;
  if (!bytes) fail("invalid UTF-8 string metadata");
  std::string result(bytes, [string lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
  if (result.find('\0') != std::string::npos) fail("embedded NUL in metadata");
  return result;
}
NSDictionary *object(id value) {
  if (![value isKindOfClass:[NSDictionary class]]) fail("expected object metadata");
  return value;
}
NSArray *array(id value) {
  if (![value isKindOfClass:[NSArray class]]) fail("expected array metadata");
  return value;
}
void keys(NSDictionary *value, std::initializer_list<const char *> expected) {
  if (value.count != expected.size()) fail("unexpected or missing metadata fields");
  for (const char *key : expected)
    if (!value[[NSString stringWithUTF8String:key]]) fail("missing metadata field");
}
uint64_t integer(id value) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID() ||
      CFNumberIsFloatType((__bridge CFNumberRef)value))
    fail("expected integer metadata; booleans/floats are invalid");
  NSNumber *number = value;
  const char type = number.objCType[0];
  if (std::string_view("cCsSiIlLqQ").find(type) == std::string_view::npos ||
      (std::string_view("csilq").find(type) != std::string_view::npos && number.longLongValue < 0))
    fail("integer metadata is negative or fractional");
  return number.unsignedLongLongValue;
}
void equal(uint64_t actual, uint64_t expected) {
  if (actual != expected) fail("metadata count, geometry or extent mismatch");
}
uint64_t add(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left) fail("count overflow");
  return left + right;
}
uint64_t multiply(uint64_t left, uint64_t right) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left) fail("count overflow");
  return left * right;
}
uint64_t aligned(uint64_t value) {
  return add(value, kAlignment - 1) & ~(kAlignment - 1);
}
void digest(std::string_view value) {
  if (value.size() != 64 || value.find_first_not_of("0123456789abcdef") != std::string_view::npos)
    fail("invalid lowercase SHA256 metadata");
}
std::string sha256(std::span<const uint8_t> bytes) {
  if (bytes.size() > std::numeric_limits<CC_LONG>::max()) fail("hash input exceeds CPU metadata limit");
  std::array<uint8_t, CC_SHA256_DIGEST_LENGTH> output{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), output.data());
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (uint8_t byte : output) { result += alphabet[byte >> 4]; result += alphabet[byte & 15]; }
  return result;
}
std::string sha256(std::string_view bytes) {
  return sha256({reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()});
}

// Foundation accepts duplicate keys and may canonicalize integral floating
// numbers. Validate the original JSON grammar before it builds the object tree.
class StrictJson final {
 public:
  explicit StrictJson(std::span<const uint8_t> bytes)
      : bytes_(reinterpret_cast<const char *>(bytes.data()), bytes.size()) {}
  void validate() {
    value(0);
    space();
    if (cursor_ != bytes_.size()) fail("trailing JSON metadata");
  }
 private:
  std::string_view bytes_;
  size_t cursor_ = 0;
  void space() {
    while (cursor_ < bytes_.size() && std::string_view(" \r\n\t").find(bytes_[cursor_]) != std::string_view::npos)
      ++cursor_;
  }
  bool take(char token) {
    space();
    if (cursor_ < bytes_.size() && bytes_[cursor_] == token) { ++cursor_; return true; }
    return false;
  }
  void require(char token) { if (!take(token)) fail("invalid JSON metadata syntax"); }
  std::string_view string() {
    space();
    const size_t began = cursor_;
    if (cursor_ == bytes_.size() || bytes_[cursor_++] != '"') fail("expected JSON string");
    while (cursor_ < bytes_.size()) {
      const unsigned char token = static_cast<unsigned char>(bytes_[cursor_++]);
      if (token == '"') return bytes_.substr(began, cursor_ - began);
      if (token < 0x20) fail("unescaped JSON control byte");
      if (token != '\\') continue;
      if (cursor_ == bytes_.size()) fail("unterminated JSON string escape");
      const char escape = bytes_[cursor_++];
      if (escape == 'u') {
        for (unsigned i = 0; i < 4; ++i)
          if (cursor_ == bytes_.size() || std::string_view("0123456789abcdefABCDEF").find(bytes_[cursor_++]) == std::string_view::npos)
            fail("invalid JSON Unicode escape");
      } else if (std::string_view("\"\\/bfnrt").find(escape) == std::string_view::npos)
        fail("invalid JSON string escape");
    }
    fail("unterminated JSON string");
  }
  std::string key() {
    const auto token = string();
    NSData *data = [NSData dataWithBytes:token.data() length:token.size()];
    NSError *error = nil;
    id decoded = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:&error];
    if (error) fail("invalid JSON object key");
    return text(decoded);
  }
  void value(unsigned depth) {
    if (depth > 32) fail("JSON metadata nesting exceeds limit");
    space();
    if (cursor_ == bytes_.size()) fail("missing JSON metadata value");
    switch (bytes_[cursor_]) {
      case '{': {
        ++cursor_;
        std::set<std::string> names;
        if (take('}')) return;
        do {
          if (!names.insert(key()).second) fail("duplicate JSON metadata key");
          require(':'); value(depth + 1);
          if (take('}')) return;
          require(',');
        } while (true);
      }
      case '[':
        ++cursor_;
        if (take(']')) return;
        do {
          value(depth + 1);
          if (take(']')) return;
          require(',');
        } while (true);
      case '"': (void)string(); return;
      case 't': case 'f': case 'n': {
        const std::string_view token = bytes_[cursor_] == 't' ? "true" : bytes_[cursor_] == 'f' ? "false" : "null";
        if (bytes_.substr(cursor_, token.size()) != token) fail("invalid JSON literal");
        cursor_ += token.size(); return;
      }
      default: {
        if (bytes_[cursor_] < '0' || bytes_[cursor_] > '9') fail("expected unsigned integer JSON token");
        const size_t began = cursor_++;
        while (cursor_ < bytes_.size() && bytes_[cursor_] >= '0' && bytes_[cursor_] <= '9') ++cursor_;
        if ((bytes_[began] == '0' && cursor_ != began + 1) ||
            (cursor_ < bytes_.size() && std::string_view(".eE+-").find(bytes_[cursor_]) != std::string_view::npos))
          fail("floating, signed or noncanonical integer JSON token");
        uint64_t number = 0;
        const auto result = std::from_chars(bytes_.data() + began, bytes_.data() + cursor_, number);
        if (result.ec != std::errc{}) fail("integer JSON token exceeds uint64");
        return;
      }
    }
  }
};

std::filesystem::path checkedDirectory(const std::filesystem::path &directory) {
  if (directory.empty() || directory.native().find('\0') != std::string::npos)
    fail("empty store directory or embedded NUL in path");
  for (const auto &part : directory)
    if (part == "..") fail("parent traversal in store directory");
  std::error_code error;
  const auto absolute = std::filesystem::absolute(directory, error).lexically_normal();
  if (error) fail("could not resolve store directory");
  std::filesystem::path cursor;
  for (const auto &part : absolute) {
    cursor /= part;
    struct stat status{};
    if (::lstat(cursor.c_str(), &status) || !S_ISDIR(status.st_mode))
      fail("store directory has missing, non-directory or symlinked component");
  }
  return absolute;
}

std::vector<uint8_t> readManifest(const std::filesystem::path &path) {
  // O_NONBLOCK prevents a concurrent file substitution with a FIFO/device from
  // blocking before fstat can reject it. It has no effect on regular files.
  const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) fail("manifest could not be opened without following symlinks");
  struct File final { int fd; ~File() { ::close(fd); } } file{fd};
  struct stat status{};
  if (::fstat(fd, &status) || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
      uint64_t(status.st_size) > kMaximumManifestBytes)
    fail("manifest is not a bounded regular JSON file");
  std::vector<uint8_t> data(static_cast<size_t>(status.st_size));
  size_t offset = 0;
  while (offset < data.size()) {
    const ssize_t count = ::read(fd, data.data() + offset, data.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) fail("manifest changed or could not be read completely");
    offset += static_cast<size_t>(count);
  }
  uint8_t extra = 0;
  ssize_t count;
  do { count = ::read(fd, &extra, 1); } while (count < 0 && errno == EINTR);
  if (count != 0) fail("manifest changed size while being read");
  return data;
}

void shape(id value, std::initializer_list<uint64_t> expected) {
  NSArray *actual = array(value);
  equal(actual.count, expected.size());
  size_t i = 0;
  for (uint64_t dimension : expected) equal(integer(actual[i++]), dimension);
}

FlashInt8ExpertStorePlane plane(id value, bool codes, uint64_t hot, uint64_t n,
                                uint64_t k, uint64_t &offset) {
  NSDictionary *entry = object(value);
  keys(entry, {"dtype", "shape", "offset", "length", "sha256"});
  if (text(entry[@"dtype"]) != (codes ? "I8" : "F32")) fail("unsupported operand dtype");
  if (codes) shape(entry[@"shape"], {hot, n, k});
  else shape(entry[@"shape"], {hot, n});
  FlashInt8ExpertStorePlane result;
  result.offset = integer(entry[@"offset"]);
  result.length = integer(entry[@"length"]);
  result.sha256 = text(entry[@"sha256"]);
  digest(result.sha256);
  offset = aligned(offset);
  equal(result.offset, offset);
  equal(result.length, multiply(multiply(hot, n), codes ? k : sizeof(float)));
  offset = add(offset, result.length);
  return result;
}
} // namespace

FlashInt8ExpertStoreMetadata loadFlashInt8ExpertStoreMetadata(
    const std::filesystem::path &directory, std::string_view expectedSourceIdentity,
    std::string_view expectedManifestFingerprint, NormConvention convention) {
  @autoreleasepool {
    digest(expectedSourceIdentity);
    digest(expectedManifestFingerprint);
    if (convention != NormConvention::OnePlusWeight && convention != NormConvention::DirectGamma)
      fail("unsupported norm convention");
    FlashInt8ExpertStoreMetadata result;
    result.directory = checkedDirectory(directory);
    const auto bytes = readManifest(result.directory / "manifest.json");
    StrictJson(bytes).validate();
    NSData *data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
    NSError *error = nil;
    id decoded = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (error) fail("invalid manifest JSON");
    NSDictionary *root = object(decoded);
    keys(root, {"schema", "source_identity_sha256", "source_manifest_sha256", "plan_sha256",
                "alignment", "target_layers", "selected_experts", "coefficient_policy",
                "quantization_format", "integer_rounding", "layers", "total_bytes",
                "planned_allocation_bytes"});
    if (text(root[@"schema"]) != kSchema || text(root[@"coefficient_policy"]) != kCoefficientPolicy ||
        text(root[@"quantization_format"]) != kQuantizationFormat || text(root[@"integer_rounding"]) != kRounding)
      fail("unsupported schema or coefficient/quantization/rounding policy");
    equal(integer(root[@"alignment"]), kAlignment);
    equal(integer(root[@"target_layers"]), result.layers.size());
    result.sourceIdentity = text(root[@"source_identity_sha256"]);
    result.sourceManifestSha256 = text(root[@"source_manifest_sha256"]);
    result.planSha256 = text(root[@"plan_sha256"]);
    digest(result.sourceIdentity); digest(result.sourceManifestSha256); digest(result.planSha256);
    if (result.sourceIdentity != expectedSourceIdentity) fail("source checkpoint identity mismatch");
    const std::string weightsIdentity = "splash.native-flash-weights-v1\nsource=" + result.sourceIdentity +
        "\nmanifest=" + result.sourceManifestSha256 + "\nnorm=" +
        (convention == NormConvention::OnePlusWeight ? "one-plus-weight\n" : "direct-gamma\n");
    if (sha256(weightsIdentity) != expectedManifestFingerprint) fail("effective original weights fingerprint mismatch");
    result.identitySha256 = sha256(bytes);

    NSArray *selected = array(root[@"selected_experts"]), *layers = array(root[@"layers"]);
    equal(selected.count, result.layers.size()); equal(layers.count, result.layers.size());
    const uint64_t inventoryCount = array(selected[0]).count;
    if (inventoryCount != 32 && inventoryCount != 64 && inventoryCount != 128)
      fail("selected expert inventory must contain exactly 32, 64 or 128 IDs per layer");
    for (id selectedLayer in selected)
      equal(array(selectedLayer).count, inventoryCount);
    for (uint32_t index = 0; index < result.layers.size(); ++index) {
      auto &layer = result.layers[index];
      NSArray *ids = array(selected[index]);
      for (id value in ids) {
        const uint64_t expert = integer(value);
        if (expert >= 512 || (!layer.selectedIDs.empty() && expert <= layer.selectedIDs.back()))
          fail("selected expert IDs must be sorted unique integers in [0,511]");
        layer.selectedIDs.push_back(static_cast<uint32_t>(expert));
      }
      NSDictionary *entry = object(layers[index]);
      keys(entry, {"layer_index", "path", "bytes", "sha256", "projections"});
      equal(integer(entry[@"layer_index"]), index);
      const std::string basename = "layer-" + std::string(index < 10 ? "0" : "") + std::to_string(index) + ".bin";
      if (text(entry[@"path"]) != basename) fail("layer path must be its canonical relative basename");
      layer.path = result.directory / basename;
      layer.bytes = integer(entry[@"bytes"]);
      layer.sha256 = text(entry[@"sha256"]); digest(layer.sha256);
      uint64_t offset = 0;
      NSDictionary *projections = object(entry[@"projections"]);
      keys(projections, {"gate_proj", "up_proj", "down_proj"});
      for (uint32_t projection = 0; projection < 3; ++projection) {
        const std::string name = projection == 0 ? "gate_proj" : projection == 1 ? "up_proj" : "down_proj";
        NSDictionary *matrix = object(projections[[NSString stringWithUTF8String:name.c_str()]]);
        keys(matrix, {"source_prefix", "dimensions", "codes", "scales"});
        const std::string prefix = "language_model.model.layers." + std::to_string(index) + ".mlp.switch_mlp." + name;
        if (text(matrix[@"source_prefix"]) != prefix) fail("source projection binding mismatch");
        const uint64_t n = projection == 2 ? 2560 : 640, k = projection == 2 ? 640 : 2560;
        shape(matrix[@"dimensions"], {ids.count, n, k});
        layer.codes[projection] = plane(matrix[@"codes"], true, ids.count, n, k, offset);
        layer.scales[projection] = plane(matrix[@"scales"], false, ids.count, n, k, offset);
      }
      equal(layer.bytes, aligned(offset));
      struct stat status{};
      if (::lstat(layer.path.c_str(), &status) || !S_ISREG(status.st_mode) ||
          (status.st_mode & 0222) || status.st_size < 0 || uint64_t(status.st_size) != layer.bytes)
        fail("layer payload must be a read-only regular file of its exact aligned size");
      result.totalBytes = add(result.totalBytes, layer.bytes);
    }
    result.plannedBytes = add(result.totalBytes, multiply(result.layers.size(), kAlignment));
    equal(integer(root[@"total_bytes"]), result.totalBytes);
    equal(integer(root[@"planned_allocation_bytes"]), result.plannedBytes);
    return result;
  }
}
} // namespace splash::flash
