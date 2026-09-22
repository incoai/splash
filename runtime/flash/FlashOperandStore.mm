#include "FlashOperandStore.hpp"
#include "FlashDenseCache.hpp"
#include "FlashFloatDenseCache.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::flash {
namespace {
[[noreturn]] void fail(std::string_view value) {
  throw std::runtime_error("Flash operand store: " + std::string(value));
}
NSString *ns(std::string_view value) {
  NSString *result = [[NSString alloc] initWithBytes:value.data() length:value.size()
                                             encoding:NSUTF8StringEncoding];
  if (!result) fail("invalid UTF-8 metadata");
  return result;
}
std::string str(id value) {
  if (![value isKindOfClass:[NSString class]]) fail("expected string");
  NSString *text = value;
  if (!text.UTF8String) fail("invalid string encoding");
  std::string result(text.UTF8String, [text lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
  if (result.find('\0') != std::string::npos) fail("embedded NUL string");
  return result;
}
uint64_t integer(id value) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) fail("expected integer");
  NSNumber *number = value;
  const char type = number.objCType[0];
  if (std::string_view("cCsSiIlLqQ").find(type) == std::string_view::npos ||
      (std::string_view("csilq").find(type) != std::string_view::npos && number.longLongValue < 0))
    fail("fractional or negative integer");
  return number.unsignedLongLongValue;
}
uint32_t narrow(id value) {
  const uint64_t result = integer(value);
  if (result > UINT32_MAX) fail("source dimension exceeds uint32");
  return static_cast<uint32_t>(result);
}
NSDictionary *object(id value) {
  if (![value isKindOfClass:[NSDictionary class]]) fail("expected object");
  return value;
}
NSArray *array(id value) {
  if (![value isKindOfClass:[NSArray class]]) fail("expected array");
  return value;
}
void keys(NSDictionary *value, std::initializer_list<const char *> expected) {
  if (value.count != expected.size()) fail("unexpected object fields");
  for (const char *key : expected) if (!value[ns(key)]) fail("missing object field");
}
void digest(std::string_view value) {
  if (value.size() != 64 || value.find_first_not_of("0123456789abcdef") != std::string_view::npos)
    fail("invalid lowercase SHA256");
}
std::string hash(std::span<const std::byte> bytes) {
  CC_SHA256_CTX context{};
  if (!CC_SHA256_Init(&context)) fail("SHA256 initialization failed");
  while (!bytes.empty()) {
    const CC_LONG count = static_cast<CC_LONG>(std::min<size_t>(bytes.size(), 1ULL << 30));
    if (!CC_SHA256_Update(&context, bytes.data(), count)) fail("SHA256 update failed");
    bytes = bytes.subspan(count);
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> output{};
  if (!CC_SHA256_Final(output.data(), &context)) fail("SHA256 finalization failed");
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char value : output) { result += digits[value >> 4]; result += digits[value & 15]; }
  return result;
}
std::string hash(std::string_view value) {
  return hash({reinterpret_cast<const std::byte *>(value.data()), value.size()});
}
std::string mathDigest() {
  return hash(std::string(kFlashDenseCacheOperandFormat) + "\n" +
      kFlashFloatDenseCacheOperandFormat + "\n" + std::string(kFlashOperandStoreMathVersion) + "\n");
}
// Foundation collapses duplicate object keys and may canonicalize 1.0 to an
// integer NSNumber. Inspect the raw token structure first: every numeric field
// in this schema must use integer JSON syntax, and escaped keys must be unique.
class StrictJSON final {
public:
  explicit StrictJSON(std::span<const std::byte> data)
      : text_(reinterpret_cast<const char *>(data.data()), data.size()) {}
  void check() { value(0); space(); if (cursor_ != text_.size()) fail("trailing JSON bytes"); }
private:
  void space() { while (cursor_ < text_.size() && std::string_view(" \r\n\t").find(text_[cursor_]) != std::string_view::npos) ++cursor_; }
  char next() { space(); if (cursor_ >= text_.size()) fail("truncated JSON"); return text_[cursor_]; }
  void expect(char c) { if (next() != c) fail("invalid JSON structure"); ++cursor_; }
  std::string string() {
    expect('"'); const size_t begin = cursor_ - 1;
    while (cursor_ < text_.size()) {
      const char c = text_[cursor_++];
      if (c == '\\') { if (cursor_ == text_.size()) fail("truncated JSON escape"); ++cursor_; }
      else if (c == '"') {
        const std::string token = "[" + std::string(text_.substr(begin, cursor_ - begin)) + "]";
        NSData *data = [NSData dataWithBytes:token.data() length:token.size()];
        NSError *error = nil;
        NSArray *decoded = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (!decoded || error || decoded.count != 1) fail("invalid JSON string");
        return str(decoded[0]);
      }
    }
    fail("unterminated JSON string");
  }
  void value(uint32_t depth) {
    if (depth > 64) fail("JSON nesting limit exceeded");
    const char c = next();
    if (c == '{') {
      ++cursor_; std::set<std::string> seen;
      if (next() == '}') { ++cursor_; return; }
      while (true) {
        if (!seen.emplace(string()).second) fail("duplicate JSON object key");
        expect(':'); value(depth + 1);
        if (next() == '}') { ++cursor_; return; }
        expect(',');
      }
    } else if (c == '[') {
      ++cursor_;
      if (next() == ']') { ++cursor_; return; }
      while (true) {
        value(depth + 1);
        if (next() == ']') { ++cursor_; return; }
        expect(',');
      }
    } else if (c == '"') { (void)string(); }
    else if ((c >= '0' && c <= '9') || c == '-') {
      if (c == '-') fail("negative JSON integers rejected");
      const size_t begin = cursor_;
      while (cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9') ++cursor_;
      if (cursor_ - begin > 1 && text_[begin] == '0') fail("noncanonical JSON integer");
      if (cursor_ < text_.size() && std::string_view(".eE").find(text_[cursor_]) != std::string_view::npos)
        fail("floating JSON number rejected");
    } else {
      bool matched = false;
      for (std::string_view token : {"true", "false", "null"})
        if (text_.substr(cursor_).starts_with(token)) { cursor_ += token.size(); matched = true; break; }
      if (!matched) fail("invalid JSON token");
    }
  }
  std::string_view text_; size_t cursor_ = 0;
};
uint64_t rounded(uint64_t value) {
  if (value > UINT64_MAX - (kFlashOperandStoreAlignment - 1)) fail("byte alignment overflows");
  return (value + kFlashOperandStoreAlignment - 1) & ~(kFlashOperandStoreAlignment - 1);
}
std::string_view formatName(FlashOperandFormat format) {
  switch (format) { case FlashOperandFormat::BF16: return "BF16";
  case FlashOperandFormat::F32: return "F32"; }
  fail("unsupported format");
}
std::string_view operandMath(FlashOperandFormat format) {
  switch (format) { case FlashOperandFormat::BF16: return kFlashDenseCacheOperandFormat;
  case FlashOperandFormat::F32: return kFlashFloatDenseCacheOperandFormat; }
  fail("unsupported format");
}
uint64_t logicalBytes(const FlashOperandSpec &p) {
  if (p.projection.empty() || p.projection.size() > 1024 || p.projection.find('\0') != std::string::npos ||
      p.projection.find("embed_tokens") != std::string::npos ||
      p.projection.find("ngram_embedding") != std::string::npos ||
      p.experts != 1 || !p.outputSize || p.outputSize % 64 || !p.inputSize || p.inputSize > 32768 ||
      (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) ||
      (p.groupSize != 32 && p.groupSize != 64 && p.groupSize != 128) ||
      p.inputSize % p.groupSize || p.parameterRowStrideBytes % 2 || p.weightRowStrideBytes % 4 ||
      p.weightRowStrideBytes < (uint64_t(p.inputSize) * p.bits + 7) / 8 ||
      p.parameterRowStrideBytes < uint64_t(p.inputSize / p.groupSize) * 2 ||
      p.weightExpertStrideBytes / p.outputSize < p.weightRowStrideBytes ||
      p.parameterExpertStrideBytes / p.outputSize < p.parameterRowStrideBytes)
    fail("invalid dense projection contract");
  const uint64_t bytes = uint64_t(p.outputSize) * p.inputSize *
      (p.format == FlashOperandFormat::BF16 ? 2 : 4);
  (void)formatName(p.format);
  return bytes;
}
std::string entryKey(std::string_view prefix, FlashOperandFormat format) {
  return std::string(formatName(format)) + ":" + std::string(prefix);
}
void equalSpec(const FlashOperandSpec &actual, const FlashOperandSpec &expected) {
  if (actual.projection != expected.projection || actual.format != expected.format ||
      actual.experts != expected.experts || actual.outputSize != expected.outputSize ||
      actual.inputSize != expected.inputSize || actual.bits != expected.bits ||
      actual.groupSize != expected.groupSize || actual.weightRowStrideBytes != expected.weightRowStrideBytes ||
      actual.weightExpertStrideBytes != expected.weightExpertStrideBytes ||
      actual.parameterRowStrideBytes != expected.parameterRowStrideBytes ||
      actual.parameterExpertStrideBytes != expected.parameterExpertStrideBytes)
    fail("saved projection/source geometry mismatch: " + expected.projection);
}
std::filesystem::path checkedFile(const std::filesystem::path &root, std::string_view name) {
  if (name.empty() || name.size() > 128 || !name.ends_with(".bin") ||
      std::string_view("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").find(name.front()) == std::string_view::npos ||
      name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") != std::string_view::npos ||
      name.find("..") != std::string_view::npos) fail("noncanonical payload filename");
  const auto path = root / std::string(name);
  if (std::filesystem::is_symlink(std::filesystem::symlink_status(path))) fail("symlink payload rejected");
  if (std::filesystem::canonical(path).parent_path() != root) fail("payload escaped store");
  return path;
}
std::vector<std::byte> readSmall(const std::filesystem::path &path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) fail("cannot open metadata");
  struct stat status{};
  if (::fstat(fd, &status) || !S_ISREG(status.st_mode) || status.st_size <= 0 || status.st_size > (16LL << 20)) {
    ::close(fd); fail("metadata extent invalid");
  }
  std::vector<std::byte> result(static_cast<size_t>(status.st_size));
  size_t position = 0;
  while (position < result.size()) {
    const ssize_t count = ::read(fd, result.data() + position, result.size() - position);
    if (count <= 0) { ::close(fd); fail("cannot read metadata"); }
    position += static_cast<size_t>(count);
  }
  ::close(fd);
  return result;
}
class Mapping final {
public:
  static std::shared_ptr<Mapping> open(const std::filesystem::path &path, uint64_t bytes) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) fail("cannot open readonly payload");
    struct stat status{};
    if (::fstat(fd, &status) || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
        uint64_t(status.st_size) != bytes || bytes % kFlashOperandStoreAlignment || bytes > SIZE_MAX) {
      ::close(fd); fail("payload extent/alignment mismatch");
    }
    void *base = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) fail("readonly payload mmap failed");
    return std::shared_ptr<Mapping>(new Mapping(base, bytes));
  }
  ~Mapping() { ::munmap(base_, bytes_); }
  void *base() const noexcept { return base_; }
  std::span<const std::byte> bytes() const noexcept {
    return {static_cast<const std::byte *>(base_), static_cast<size_t>(bytes_)};
  }
private:
  Mapping(void *base, uint64_t bytes) : base_(base), bytes_(bytes) {}
  void *base_; uint64_t bytes_;
};
void verifyMapping(const Mapping &mapping, uint64_t logical, std::string_view expected) {
  const auto bytes = mapping.bytes();
  if (logical > bytes.size()) fail("logical payload extent exceeds mapping");
  for (const auto value : bytes.subspan(static_cast<size_t>(logical)))
    if (value != std::byte{0}) fail("nonzero operand alignment padding");
  if (hash(bytes) != expected) fail("payload SHA256 mismatch");
}
void writeFile(const std::filesystem::path &path, std::span<const std::byte> bytes,
               uint64_t allocated = 0) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) fail("cannot create fresh artifact file");
  if (!allocated) allocated = bytes.size();
  if (allocated < bytes.size() || allocated > INT64_MAX || ::ftruncate(fd, static_cast<off_t>(allocated))) {
    ::close(fd); fail("artifact file extent rejected");
  }
  size_t position = 0;
  while (position < bytes.size()) {
    const size_t requested = std::min<size_t>(bytes.size() - position, 16ULL << 20);
    const ssize_t count = ::write(fd, bytes.data() + position, requested);
    if (count <= 0) { ::close(fd); fail("artifact write failed"); }
    position += static_cast<size_t>(count);
  }
  if (::fsync(fd) || ::fchmod(fd, 0444)) { ::close(fd); fail("artifact synchronization failed"); }
  if (::close(fd)) fail("artifact close failed");
}
void syncDirectory(const std::filesystem::path &path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
  if (fd < 0) fail("cannot open artifact directory for synchronization");
  const int result = ::fsync(fd); ::close(fd);
  if (result) fail("artifact directory synchronization failed");
}
NSDictionary *sourceObject(const FlashOperandSpec &p) {
  return @{ @"experts": @(p.experts), @"output_size": @(p.outputSize), @"input_size": @(p.inputSize),
    @"bits": @(p.bits), @"group_size": @(p.groupSize), @"weight_row_stride_bytes": @(p.weightRowStrideBytes),
    @"weight_expert_stride_bytes": @(p.weightExpertStrideBytes),
    @"parameter_row_stride_bytes": @(p.parameterRowStrideBytes),
    @"parameter_expert_stride_bytes": @(p.parameterExpertStrideBytes) };
}
} // namespace

FlashOperandSpec flashOperandSpec(std::string_view prefix, FlashOperandFormat format,
                                 const FlashAffineProjection &p) {
  FlashOperandSpec result{std::string(prefix), format, p.experts, p.outputSize, p.inputSize,
      p.bits, p.groupSize, p.weightRowStrideBytes, p.weightExpertStrideBytes,
      p.parameterRowStrideBytes, p.parameterExpertStrideBytes};
  (void)logicalBytes(result);
  return result;
}

struct FlashOperandStore::Impl final {
  struct Entry final { FlashOperandSpec spec; std::filesystem::path file;
    uint64_t logical = 0, allocated = 0; std::string digest; };
  std::string identity, source, manifest;
  std::map<std::string, Entry, std::less<>> entries;
  std::vector<FlashOperandSpec> specs;
};
FlashOperandStore::FlashOperandStore() : impl_(std::make_unique<Impl>()) {}
FlashOperandStore::~FlashOperandStore() = default;
FlashOperandStore::FlashOperandStore(FlashOperandStore &&) noexcept = default;
FlashOperandStore &FlashOperandStore::operator=(FlashOperandStore &&) noexcept = default;

FlashOperandStore FlashOperandStore::load(const std::filesystem::path &directory,
    std::string_view sourceIdentity, std::string_view manifestFingerprint) {
  @autoreleasepool {
    digest(sourceIdentity); digest(manifestFingerprint);
    if (std::filesystem::is_symlink(std::filesystem::symlink_status(directory))) fail("symlink store directory rejected");
    const auto root = std::filesystem::canonical(directory);
    if (!std::filesystem::is_directory(root)) fail("store is not a directory");
    const auto manifestBytes = readSmall(root / "manifest.json");
    const auto digestBytes = readSmall(root / "manifest.sha256");
    const std::string manifestDigest(reinterpret_cast<const char *>(digestBytes.data()), digestBytes.size());
    if (manifestDigest.size() != 65 || manifestDigest.back() != '\n') fail("manifest digest file invalid");
    digest(std::string_view(manifestDigest).substr(0, 64));
    const std::string actualDigest = hash(manifestBytes);
    if (actualDigest != manifestDigest.substr(0, 64)) fail("manifest SHA256 mismatch");
    StrictJSON(manifestBytes).check();
    NSError *error = nil;
    NSData *data = [NSData dataWithBytes:manifestBytes.data() length:manifestBytes.size()];
    NSDictionary *manifest = object([NSJSONSerialization JSONObjectWithData:data options:0 error:&error]);
    if (error) fail("invalid manifest JSON");
    keys(manifest, {"schema", "alignment_bytes", "source_identity_sha256", "weights_manifest_fingerprint",
                    "math_version_sha256", "entries"});
    if (str(manifest[@"schema"]) != kFlashOperandStoreSchema ||
        integer(manifest[@"alignment_bytes"]) != kFlashOperandStoreAlignment) fail("schema/alignment mismatch");
    FlashOperandStore result;
    auto &impl = *result.impl_;
    impl.identity = actualDigest; impl.source = str(manifest[@"source_identity_sha256"]);
    impl.manifest = str(manifest[@"weights_manifest_fingerprint"]);
    digest(impl.source); digest(impl.manifest);
    const std::string math = str(manifest[@"math_version_sha256"]); digest(math);
    if (impl.source != sourceIdentity || impl.manifest != manifestFingerprint) fail("model identity mismatch");
    if (math != mathDigest()) fail("operand math version mismatch");
    NSArray *entries = array(manifest[@"entries"]);
    if (!entries.count || entries.count > 8192) fail("operand selection size invalid");
    std::map<std::string, bool> files;
    for (id value in entries) {
      NSDictionary *item = object(value);
      keys(item, {"projection", "format", "operand_math", "shape", "logical_bytes", "allocated_bytes",
                  "file", "offset_bytes", "payload_sha256", "source"});
      Impl::Entry entry;
      entry.spec.projection = str(item[@"projection"]);
      const std::string format = str(item[@"format"]);
      if (format == "BF16") entry.spec.format = FlashOperandFormat::BF16;
      else if (format == "F32") entry.spec.format = FlashOperandFormat::F32;
      else fail("unsupported operand format");
      if (str(item[@"operand_math"]) != operandMath(entry.spec.format)) fail("operand semantics mismatch");
      NSDictionary *p = object(item[@"source"]);
      keys(p, {"experts", "output_size", "input_size", "bits", "group_size", "weight_row_stride_bytes",
                "weight_expert_stride_bytes", "parameter_row_stride_bytes", "parameter_expert_stride_bytes"});
      entry.spec.experts = narrow(p[@"experts"]); entry.spec.outputSize = narrow(p[@"output_size"]);
      entry.spec.inputSize = narrow(p[@"input_size"]); entry.spec.bits = narrow(p[@"bits"]);
      entry.spec.groupSize = narrow(p[@"group_size"]);
      entry.spec.weightRowStrideBytes = integer(p[@"weight_row_stride_bytes"]);
      entry.spec.weightExpertStrideBytes = integer(p[@"weight_expert_stride_bytes"]);
      entry.spec.parameterRowStrideBytes = integer(p[@"parameter_row_stride_bytes"]);
      entry.spec.parameterExpertStrideBytes = integer(p[@"parameter_expert_stride_bytes"]);
      entry.logical = logicalBytes(entry.spec); entry.allocated = rounded(entry.logical);
      NSArray *shape = array(item[@"shape"]);
      if (shape.count != 2 || integer(shape[0]) != entry.spec.outputSize || integer(shape[1]) != entry.spec.inputSize ||
          integer(item[@"logical_bytes"]) != entry.logical || integer(item[@"allocated_bytes"]) != entry.allocated ||
          integer(item[@"offset_bytes"]) != 0) fail("operand shape/extent/offset mismatch");
      const std::string filename = str(item[@"file"]);
      if (!files.emplace(filename, true).second) fail("duplicate payload file");
      entry.file = checkedFile(root, filename); entry.digest = str(item[@"payload_sha256"]); digest(entry.digest);
      struct stat status{};
      if (::lstat(entry.file.c_str(), &status) || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
          uint64_t(status.st_size) != entry.allocated) fail("payload size mismatch");
      const auto key = entryKey(entry.spec.projection, entry.spec.format);
      impl.specs.push_back(entry.spec);
      if (!impl.entries.emplace(key, std::move(entry)).second) fail("duplicate projection/format");
    }
    return result;
  }
}
std::unique_ptr<FlashOperandStore> FlashOperandStore::fromEnvironment(const FlashWeights &weights) {
  const char *value = std::getenv("SPLASH_FLASH_OPERAND_STORE");
  if (!value || std::string_view(value) == "0") return {};
  if (!*value) fail("SPLASH_FLASH_OPERAND_STORE cannot be empty");
  auto result = std::make_unique<FlashOperandStore>(load(value, weights.sourceIdentity(), weights.manifestFingerprint()));
  result->validateSource(weights);
  return result;
}
void FlashOperandStore::validateSource(const FlashWeights &weights) const {
  if (!impl_ || impl_->source != weights.sourceIdentity() || impl_->manifest != weights.manifestFingerprint())
    fail("source identity mismatch");
  for (const auto &spec : impl_->specs)
    equalSpec(spec, flashOperandSpec(spec.projection, spec.format, weights.projection(spec.projection)));
}
void FlashOperandStore::verifyPayloads() const {
  if (!impl_) fail("store not initialized");
  for (const auto &[key, entry] : impl_->entries) {
    (void)key;
    auto mapping = Mapping::open(entry.file, entry.allocated);
    verifyMapping(*mapping, entry.logical, entry.digest);
  }
}
bool FlashOperandStore::contains(std::string_view prefix, FlashOperandFormat format) const noexcept {
  if (!impl_) return false;
  // Avoid allocation in this noexcept query, also for malformed enum values.
  for (const auto &spec : impl_->specs) if (spec.projection == prefix && spec.format == format) return true;
  return false;
}
FlashTensor FlashOperandStore::mapTensor(metal::MetalBackend &backend, const FlashOperandSpec &expected) const {
  (void)logicalBytes(expected);
  if (!impl_) fail("store not initialized");
  const auto found = impl_->entries.find(entryKey(expected.projection, expected.format));
  if (found == impl_->entries.end()) fail("requested operand absent");
  const auto &entry = found->second; equalSpec(entry.spec, expected);
  auto mapping = Mapping::open(entry.file, entry.allocated);
  verifyMapping(*mapping, entry.logical, entry.digest);
  FlashTensor result;
  result.buffer = backend.wrapSharedMemory(mapping->base(), entry.allocated, mapping,
      "flash-saved-operand:" + entryKey(expected.projection, expected.format));
  result.dtype = expected.format == FlashOperandFormat::BF16 ? FlashDType::BF16 : FlashDType::F32;
  result.shape = {expected.outputSize, expected.inputSize}; result.logicalBytes = entry.logical;
  return result;
}
const std::string &FlashOperandStore::identitySha256() const {
  if (!impl_) fail("store not initialized");
  return impl_->identity;
}
const std::vector<FlashOperandSpec> &FlashOperandStore::specs() const {
  if (!impl_) fail("store not initialized"); return impl_->specs;
}
uint64_t FlashOperandStore::plannedBytes(std::span<const FlashOperandSpec> selection) const {
  uint64_t result = 0;
  std::map<std::string, bool> seen;
  for (const auto &spec : selection) {
    if (!seen.emplace(entryKey(spec.projection, spec.format), true).second) fail("duplicate planned operand");
    const uint64_t bytes = rounded(logicalBytes(spec));
    if (result > UINT64_MAX - bytes) fail("planned operand bytes overflow");
    result += bytes;
  }
  return result;
}

struct FlashOperandStoreWriter::Impl final {
  std::filesystem::path destination, staging;
  std::string source, manifest;
  NSMutableArray *entries = [NSMutableArray array];
  std::map<std::string, bool> seen;
  bool published = false;
  ~Impl() { if (!published && !staging.empty()) { std::error_code ignored;
    std::filesystem::remove_all(staging, ignored); } }
};
FlashOperandStoreWriter::FlashOperandStoreWriter(const std::filesystem::path &destination,
    std::string_view sourceIdentity, std::string_view manifestFingerprint) : impl_(std::make_unique<Impl>()) {
  digest(sourceIdentity); digest(manifestFingerprint);
  if (destination.filename().empty() || destination.filename() == "." || destination.filename() == "..")
    fail("invalid destination directory");
  const auto parent = std::filesystem::canonical(destination.has_parent_path() ? destination.parent_path() : ".");
  impl_->destination = parent / destination.filename();
  if (std::filesystem::exists(std::filesystem::symlink_status(impl_->destination))) fail("destination already exists");
  impl_->source = sourceIdentity; impl_->manifest = manifestFingerprint;
  std::string pattern = (parent / ("." + destination.filename().string() + ".staging-XXXXXX")).string();
  std::vector<char> buffer(pattern.begin(), pattern.end()); buffer.push_back('\0');
  char *created = ::mkdtemp(buffer.data());
  if (!created) fail("cannot create private staging directory");
  impl_->staging = created;
}
FlashOperandStoreWriter::~FlashOperandStoreWriter() = default;
void FlashOperandStoreWriter::append(const FlashOperandSpec &spec, std::span<const std::byte> operand) {
  @autoreleasepool {
    if (!impl_ || impl_->published) fail("writer already published");
    const uint64_t logical = logicalBytes(spec), allocated = rounded(logical);
    if (operand.size() != logical || !operand.data()) fail("operand exact byte extent mismatch");
    if (!impl_->seen.emplace(entryKey(spec.projection, spec.format), true).second) fail("duplicate written operand");
    if (impl_->entries.count >= 8192) fail("too many written operands");
    const std::string filename = "operand-" + std::to_string(impl_->entries.count) + ".bin";
    const auto path = impl_->staging / filename;
    writeFile(path, operand, allocated);
    const auto mapping = Mapping::open(path, allocated);
    const std::string payloadDigest = hash(mapping->bytes());
    [impl_->entries addObject:@{ @"projection": ns(spec.projection), @"format": ns(formatName(spec.format)),
      @"operand_math": ns(operandMath(spec.format)), @"shape": @[@(spec.outputSize), @(spec.inputSize)],
      @"logical_bytes": @(logical), @"allocated_bytes": @(allocated), @"file": ns(filename),
      @"offset_bytes": @0, @"payload_sha256": ns(payloadDigest), @"source": sourceObject(spec) }];
  }
}
void FlashOperandStoreWriter::append(const FlashOperandSpec &spec, const FlashTensor &operand) {
  const uint64_t bytes = logicalBytes(spec);
  const FlashDType type = spec.format == FlashOperandFormat::BF16 ? FlashDType::BF16 : FlashDType::F32;
  if (operand.dtype != type || operand.shape != std::vector<uint64_t>{spec.outputSize, spec.inputSize} ||
      operand.logicalBytes != bytes || operand.buffer.sizeBytes() < bytes || !operand.buffer.contents())
    fail("cached operand dtype/shape/Shared extent mismatch");
  append(spec, {static_cast<const std::byte *>(operand.buffer.contents()), static_cast<size_t>(bytes)});
}
std::string FlashOperandStoreWriter::publish() {
  @autoreleasepool {
    if (!impl_ || impl_->published || !impl_->entries.count) fail("writer publication state invalid");
    NSDictionary *manifest = @{ @"schema": ns(kFlashOperandStoreSchema),
      @"alignment_bytes": @(kFlashOperandStoreAlignment), @"source_identity_sha256": ns(impl_->source),
      @"weights_manifest_fingerprint": ns(impl_->manifest), @"math_version_sha256": ns(mathDigest()),
      @"entries": impl_->entries };
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingSortedKeys error:&error];
    if (!data || error) fail("manifest serialization failed");
    const std::span<const std::byte> bytes(static_cast<const std::byte *>(data.bytes), data.length);
    const std::string identity = hash(bytes), digestText = identity + "\n";
    writeFile(impl_->staging / "manifest.json", bytes);
    writeFile(impl_->staging / "manifest.sha256",
        {reinterpret_cast<const std::byte *>(digestText.data()), digestText.size()});
    syncDirectory(impl_->staging);
    // macOS RENAME_EXCL closes the check/rename race and never replaces a
    // destination created by another process, including an empty directory.
    if (::renamex_np(impl_->staging.c_str(), impl_->destination.c_str(), RENAME_EXCL))
      fail("atomic fresh-destination publication failed");
    impl_->published = true;
    syncDirectory(impl_->destination.parent_path());
    return identity;
  }
}
} // namespace splash::flash
