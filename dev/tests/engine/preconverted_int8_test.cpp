#include "model/PreconvertedINT8.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

namespace artifact = splash::model::preconverted_int8;

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Function>
void requireArtifactError(Function &&function, std::string_view message) {
  try {
    function();
  } catch (const artifact::ArtifactError &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "splash-int8-artifact-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char *created = mkdtemp(writable.data());
    if (!created)
      throw std::runtime_error("could not create artifact test directory");
    path_ = created;
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
  const std::filesystem::path &path() const noexcept { return path_; }

private:
  std::filesystem::path path_;
};

class ReadOnlyMapping final {
public:
  ReadOnlyMapping(const std::filesystem::path &path, size_t bytes) : bytes_(bytes) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0)
      throw std::runtime_error("could not open synthetic mapped payload");
    struct stat status{};
    if (::fstat(descriptor, &status) || status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) != bytes_) {
      ::close(descriptor);
      throw std::runtime_error("synthetic mapped payload size mismatch");
    }
    address_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, descriptor, 0);
    ::close(descriptor);
    if (address_ == MAP_FAILED)
      throw std::runtime_error("could not map synthetic payload");
  }
  ~ReadOnlyMapping() { ::munmap(address_, bytes_); }
  ReadOnlyMapping(const ReadOnlyMapping &) = delete;
  ReadOnlyMapping &operator=(const ReadOnlyMapping &) = delete;
  std::span<const uint8_t> bytes() const {
    return {static_cast<const uint8_t *>(address_), bytes_};
  }

private:
  size_t bytes_;
  void *address_ = nullptr;
};

std::vector<uint8_t> readBytes(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("could not read synthetic artifact");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void writeBytes(const std::filesystem::path &path,
                std::span<const uint8_t> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output)
    throw std::runtime_error("could not write synthetic artifact");
}

// This oracle deliberately does not call the artifact digest implementation.
std::string sha256Oracle(std::span<const uint8_t> bytes) {
  require(bytes.size() <= std::numeric_limits<CC_LONG>::max(),
          "test SHA input exceeds CommonCrypto's one-shot limit");
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  require(CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()),
                    digest.data()) != nullptr,
          "test SHA-256 failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned char byte : digest) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 15]);
  }
  return result;
}

std::string sha256Oracle(std::string_view value) {
  return sha256Oracle(std::span(
      reinterpret_cast<const uint8_t *>(value.data()), value.size()));
}

std::string fingerprintOracle(const artifact::Identity &identity,
                              std::string_view dataSha256,
                              std::string_view scaleSha256) {
  constexpr std::array<std::string_view, 4> policies{
      "all", "target", "hybrid", "q4"};
  require(identity.policy < policies.size(), "invalid test policy");
  // Existing WeightStore.cpp v2 identity, independent of artifact schema/key.
  std::ostringstream canonical;
  canonical << "splash.affine-q4-to-whole-row-int8-w8a8-profile-v2\n"
            << "policy=" << policies[identity.policy]
            << "\nrole=" << identity.role << "\nkind=" << identity.kind << '\n'
            << "n=" << identity.n << "\nk=" << identity.k
            << "\nweight_scale=maxabs-whole-output-row-div127\n"
            << "activation_scale=maxabs-whole-input-row-div127\n"
            << "rounding=nearest-even\nclamp=-127,127\nzero_scale=1\n"
            << "layout=output-major-n-k\nphases=prefill-decode-replay\n"
            << "diagnostic_abi=sticky-nonfinite-shape-16-v1\n"
            << "data=" << dataSha256 << '\n'
            << "scales=" << scaleSha256 << '\n';
  return sha256Oracle(canonical.str());
}

struct Fixture final {
  artifact::Identity identity{256, 256, 2, 1, 1, sha256Oracle("source-q4")};
  std::vector<int8_t> data;
  std::vector<float> scales;

  Fixture() : data(size_t{identity.n} * identity.k), scales(identity.n) {
    for (size_t index = 0; index < data.size(); ++index)
      data[index] = static_cast<int8_t>(static_cast<int>(index % 255) - 127);
    for (size_t index = 0; index < scales.size(); ++index)
      scales[index] = std::bit_cast<float>(
          uint32_t{0x3b800001} + static_cast<uint32_t>(index) * 257);
  }

  std::string fingerprint() const {
    const auto dataBytes = std::span(
        reinterpret_cast<const uint8_t *>(data.data()), data.size());
    const auto scaleBytes = std::span(
        reinterpret_cast<const uint8_t *>(scales.data()),
        scales.size() * sizeof(float));
    return fingerprintOracle(identity, sha256Oracle(dataBytes),
                             sha256Oracle(scaleBytes));
  }
};

size_t entryCount(const std::filesystem::path &directory) {
  return static_cast<size_t>(std::distance(
      std::filesystem::directory_iterator(directory),
      std::filesystem::directory_iterator()));
}

void identityTests() {
  constexpr std::array<uint8_t, 3> abc{'a', 'b', 'c'};
  require(artifact::digest(abc.data(), abc.size()) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "artifact digest does not match the standard SHA-256 vector");
  std::array<uint8_t, 32> packed{};
  for (size_t index = 0; index < packed.size(); ++index)
    packed[index] = static_cast<uint8_t>(index);
  constexpr std::array<uint8_t, 2> scales{0xa1, 0xb2};
  constexpr std::array<uint8_t, 2> biases{0xc3, 0xd4};
  std::vector<uint8_t> sourceBytes(packed.begin(), packed.end());
  sourceBytes.insert(sourceBytes.end(), scales.begin(), scales.end());
  sourceBytes.insert(sourceBytes.end(), biases.begin(), biases.end());
  require(artifact::sourceDigest(packed, scales, biases) ==
              sha256Oracle(sourceBytes),
          "source digest did not stream packed/scales/biases in order");
  require(artifact::sourceDigest(packed, biases, scales) !=
              artifact::sourceDigest(packed, scales, biases),
          "source plane ordering was lost");
  requireArtifactError(
      [&] { static_cast<void>(artifact::sourceDigest(
          std::span<const uint8_t>(packed).first(31), scales, biases)); },
      "source digest accepted incomplete affine Q4 planes");

  Fixture fixture;
  const std::string key = artifact::cacheKey(fixture.identity);
  require(key.size() == 64 && key.find_first_not_of("0123456789abcdef") ==
                                  std::string::npos,
          "artifact key is not canonical lowercase SHA-256");
  auto changed = fixture.identity;
  changed.n = 512;
  require(artifact::cacheKey(changed) != key, "N is absent from cache key");
  changed = fixture.identity;
  changed.k = 512;
  require(artifact::cacheKey(changed) != key, "K is absent from cache key");
  changed = fixture.identity;
  changed.policy = 1;
  require(artifact::cacheKey(changed) != key, "policy is absent from cache key");
  changed = fixture.identity;
  changed.role = 2;
  require(artifact::cacheKey(changed) != key, "role is absent from cache key");
  changed = fixture.identity;
  changed.kind = 0;
  require(artifact::cacheKey(changed) != key, "kind is absent from cache key");
  changed = fixture.identity;
  changed.sourceSha256 = sha256Oracle("different-source-q4");
  require(artifact::cacheKey(changed) != key, "source hash is absent from cache key");

  const std::string dataHash = sha256Oracle("converted codes");
  const std::string scaleHash = sha256Oracle("converted FP32 scales");
  require(artifact::projectionFingerprint(fixture.identity, dataHash, scaleHash) ==
              fingerprintOracle(fixture.identity, dataHash, scaleHash),
          "projection fingerprint changed the existing v2 canonical text");
  require(artifact::projectionFingerprint(changed, dataHash, scaleHash) ==
              artifact::projectionFingerprint(fixture.identity, dataHash, scaleHash),
          "disk source identity changed the existing projection fingerprint");

  for (uint32_t policy = 0; policy < 4; ++policy) {
    changed = fixture.identity;
    changed.policy = policy;
    require(artifact::projectionFingerprint(changed, dataHash, scaleHash) ==
                fingerprintOracle(changed, dataHash, scaleHash),
            "precision policy spelling changed the projection fingerprint");
  }

  for (const auto &[n, k] : std::array<std::array<uint32_t, 2>, 5>{
           {{0, 256}, {255, 256}, {256, 0}, {256, 64}, {256, 25856}}}) {
    changed = fixture.identity;
    changed.n = n;
    changed.k = k;
    requireArtifactError([&] { static_cast<void>(artifact::cacheKey(changed)); },
                         "artifact key accepted incompatible tensor dimensions");
  }
  changed = fixture.identity;
  changed.sourceSha256 = "../not-a-sha256";
  requireArtifactError([&] { static_cast<void>(artifact::paths("unused", changed)); },
                       "artifact path accepted a malformed source SHA-256");
  changed.sourceSha256 = std::string(64, 'g');
  requireArtifactError([&] { static_cast<void>(artifact::cacheKey(changed)); },
                       "artifact key accepted a correctly sized nonhex source hash");
}

void roundTripAndPublicationTests() {
  TemporaryDirectory directory;
  Fixture fixture;
  const auto artifactPaths = artifact::paths(directory.path(), fixture.identity);
  requireArtifactError(
      [&] { static_cast<void>(artifact::loadMetadata(directory.path(), fixture.identity)); },
      "missing artifact was accepted");
  const auto written = artifact::writeAtomically(
      directory.path(), fixture.identity, fixture.data, fixture.scales,
      fixture.fingerprint());
  const auto loaded = artifact::loadMetadata(directory.path(), fixture.identity);
  auto payload = readBytes(artifactPaths.payload);
  artifact::validatePayload(loaded, payload);
  constexpr uint64_t dataBytes = 256 * 256;
  constexpr uint64_t scaleBytes = 256 * sizeof(float);
  constexpr uint64_t expectedBytes = dataBytes + 16 * 1024;
  require(loaded.dataOffset == 0 && loaded.scaleOffset == dataBytes &&
              loaded.dataBytes == dataBytes && loaded.scaleBytes == scaleBytes &&
              loaded.payloadBytes == expectedBytes && payload.size() == expectedBytes,
          "artifact payload did not preserve separately page-aligned plane footprint");
  require(readBytes(artifactPaths.metadata).size() == 512,
          "metadata is not the fixed 512-byte sidecar");
  require(loaded.projectionFingerprint == fixture.fingerprint() &&
              written.projectionFingerprint == loaded.projectionFingerprint,
          "cache hit changed the converted projection fingerprint");
  require(std::memcmp(payload.data() + loaded.dataOffset, fixture.data.data(),
                      fixture.data.size()) == 0,
          "signed code bytes did not round-trip");
  require(std::memcmp(payload.data() + loaded.scaleOffset, fixture.scales.data(),
                      fixture.scales.size() * sizeof(float)) == 0,
          "FP32 scale bits did not round-trip");

  const auto originalMetadata = readBytes(artifactPaths.metadata);
  const auto originalPayload = payload;
  static_cast<void>(artifact::writeAtomically(directory.path(), fixture.identity,
                                             fixture.data, fixture.scales,
                                             fixture.fingerprint()));
  require(readBytes(artifactPaths.metadata) == originalMetadata &&
              readBytes(artifactPaths.payload) == originalPayload,
          "idempotent publication rewrote an existing artifact");
  fixture.data[0] = 42;
  try {
    static_cast<void>(artifact::writeAtomically(directory.path(), fixture.identity,
                                               fixture.data, fixture.scales));
  } catch (const artifact::ArtifactError &) {
    // Rejecting different bytes is also valid; an existing key is immutable.
  }
  require(readBytes(artifactPaths.metadata) == originalMetadata &&
              readBytes(artifactPaths.payload) == originalPayload,
          "publication overwrote a previously valid key");
  require(entryCount(directory.path()) == 1 &&
              entryCount(artifactPaths.directory) == 2,
          "atomic publication left temporary files or directories");
}

// Recompute the valid hashes/fingerprint after a mutation so rejection proves
// range/scaling validation rather than merely detection by the payload hash.
void refreshHashes(artifact::Metadata &metadata,
                   std::span<const uint8_t> payload) {
  metadata.dataSha256 = sha256Oracle(payload.subspan(
      static_cast<size_t>(metadata.dataOffset),
      static_cast<size_t>(metadata.dataBytes)));
  metadata.scaleSha256 = sha256Oracle(payload.subspan(
      static_cast<size_t>(metadata.scaleOffset),
      static_cast<size_t>(metadata.scaleBytes)));
  metadata.projectionFingerprint = fingerprintOracle(
      metadata.identity, metadata.dataSha256, metadata.scaleSha256);
}

void invalidPayloadTests() {
  TemporaryDirectory directory;
  Fixture fixture;
  const auto metadata = artifact::writeAtomically(
      directory.path(), fixture.identity, fixture.data, fixture.scales);
  const auto original = readBytes(metadata.payloadPath);

  auto corrupted = original;
  corrupted.pop_back();
  requireArtifactError([&] { artifact::validatePayload(metadata, corrupted); },
                       "truncated payload was accepted");
  corrupted = original;
  corrupted.push_back(0);
  requireArtifactError([&] { artifact::validatePayload(metadata, corrupted); },
                       "oversized payload was accepted");
  corrupted = original;
  corrupted[0] ^= 1;
  requireArtifactError([&] { artifact::validatePayload(metadata, corrupted); },
                       "tampered code bytes were accepted");
  corrupted = original;
  corrupted[static_cast<size_t>(metadata.scaleOffset)] ^= 1;
  requireArtifactError([&] { artifact::validatePayload(metadata, corrupted); },
                       "tampered scale bytes were accepted");
  corrupted = original;
  corrupted.back() = 1;
  requireArtifactError([&] { artifact::validatePayload(metadata, corrupted); },
                       "nonzero canonical padding was accepted");

  corrupted = original;
  corrupted[0] = 0x80;
  auto changedMetadata = metadata;
  refreshHashes(changedMetadata, corrupted);
  requireArtifactError([&] { artifact::validatePayload(changedMetadata, corrupted); },
                       "INT8 -128 code was accepted despite valid hashes");

  constexpr std::array<uint32_t, 5> invalidScaleBits{
      0x00000000, 0x80000000, 0xbf800000, 0x7fc00001, 0x7f800000};
  for (uint32_t bits : invalidScaleBits) {
    corrupted = original;
    std::memcpy(corrupted.data() + metadata.scaleOffset, &bits, sizeof(bits));
    changedMetadata = metadata;
    refreshHashes(changedMetadata, corrupted);
    requireArtifactError([&] { artifact::validatePayload(changedMetadata, corrupted); },
                         "invalid row scale was accepted despite valid hashes");
  }

  const auto artifactPaths = artifact::paths(directory.path(), fixture.identity);
  auto metadataBytes = readBytes(artifactPaths.metadata);
  metadataBytes.resize(metadataBytes.size() - 1);
  writeBytes(artifactPaths.metadata, metadataBytes);
  requireArtifactError(
      [&] { static_cast<void>(artifact::loadMetadata(directory.path(), fixture.identity)); },
      "truncated metadata was accepted");
  requireArtifactError(
      [&] { static_cast<void>(artifact::writeAtomically(directory.path(), fixture.identity,
                                                       fixture.data, fixture.scales)); },
      "publication silently replaced an existing corrupt artifact");
  require(readBytes(artifactPaths.metadata) == metadataBytes &&
              readBytes(artifactPaths.payload) == original &&
              entryCount(directory.path()) == 1 &&
              entryCount(artifactPaths.directory) == 2,
          "failed publication altered an existing corrupt artifact or left debris");
}

void metadataIdentityTests() {
  TemporaryDirectory directory;
  Fixture fixture;
  static_cast<void>(artifact::writeAtomically(directory.path(), fixture.identity,
                                             fixture.data, fixture.scales));
  const auto originalPaths = artifact::paths(directory.path(), fixture.identity);
  const auto originalMetadata = readBytes(originalPaths.metadata);
  const auto originalPayload = readBytes(originalPaths.payload);
  auto changed = fixture.identity;
  changed.policy = 1;
  auto changedPaths = artifact::paths(directory.path(), changed);
  std::filesystem::create_directory(changedPaths.directory);
  writeBytes(changedPaths.metadata, originalMetadata);
  writeBytes(changedPaths.payload, originalPayload);
  requireArtifactError(
      [&] { static_cast<void>(artifact::loadMetadata(directory.path(), changed)); },
      "metadata from another precision policy was accepted under a new key");
  changed = fixture.identity;
  changed.sourceSha256 = sha256Oracle("other-source");
  changedPaths = artifact::paths(directory.path(), changed);
  std::filesystem::create_directory(changedPaths.directory);
  writeBytes(changedPaths.metadata, originalMetadata);
  writeBytes(changedPaths.payload, originalPayload);
  requireArtifactError(
      [&] { static_cast<void>(artifact::loadMetadata(directory.path(), changed)); },
      "metadata from another source was accepted under a new key");
}

void putLittleEndian(std::vector<uint8_t> &bytes, size_t offset,
                     uint64_t value, size_t width) {
  for (size_t index = 0; index < width; ++index)
    bytes.at(offset + index) = static_cast<uint8_t>(value >> (index * 8));
}

void refreshMetadataChecksum(std::vector<uint8_t> &bytes) {
  // Published 512-byte file-format contract; no production encoder is used.
  require(bytes.size() == 512, "wrong test metadata size");
  std::fill_n(bytes.begin() + 336, 32, 0);
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> checksum{};
  require(CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()),
                    checksum.data()) != nullptr,
          "test metadata checksum failed");
  std::copy(checksum.begin(), checksum.end(), bytes.begin() + 336);
}

void metadataFormatTests() {
  TemporaryDirectory directory;
  Fixture fixture;
  static_cast<void>(artifact::writeAtomically(directory.path(), fixture.identity,
                                             fixture.data, fixture.scales));
  const auto artifactPaths = artifact::paths(directory.path(), fixture.identity);
  const auto original = readBytes(artifactPaths.metadata);
  struct Mutation {
    size_t offset;
    size_t width;
    uint64_t value;
    std::string_view failure;
  };
  constexpr std::array<Mutation, 7> mutations{{
      {8, 4, 2, "unknown metadata schema was accepted with a valid checksum"},
      {12, 4, 511, "wrong header byte count was accepted with a valid checksum"},
      {16, 4, 3, "unknown converter version was accepted with a valid checksum"},
      {40, 8, 16384, "payload header overhead was accepted as a canonical layout"},
      {48, 8, 65535, "incorrect code-plane byte count was accepted"},
      {56, 8, 65537, "misaligned scale-plane layout was accepted"},
      {72, 8, 81921, "incorrect aligned footprint was accepted"},
  }};
  for (const auto &mutation : mutations) {
    auto bytes = original;
    putLittleEndian(bytes, mutation.offset, mutation.value, mutation.width);
    refreshMetadataChecksum(bytes);
    writeBytes(artifactPaths.metadata, bytes);
    requireArtifactError(
        [&] { static_cast<void>(artifact::loadMetadata(directory.path(), fixture.identity)); },
        mutation.failure);
  }
  for (size_t offset : {size_t{0}, size_t{80}, size_t{112}, size_t{144},
                        size_t{176}, size_t{208}, size_t{368}}) {
    auto bytes = original;
    bytes[offset] ^= 1;
    refreshMetadataChecksum(bytes);
    writeBytes(artifactPaths.metadata, bytes);
    requireArtifactError(
        [&] { static_cast<void>(artifact::loadMetadata(directory.path(), fixture.identity)); },
        "invalid magic/source/hash/fingerprint/algorithm/reserved metadata was accepted");
  }
  auto bytes = original;
  bytes[336] ^= 1;
  writeBytes(artifactPaths.metadata, bytes);
  requireArtifactError(
      [&] { static_cast<void>(artifact::loadMetadata(directory.path(), fixture.identity)); },
      "corrupt metadata checksum was accepted");
  writeBytes(artifactPaths.metadata, original);
  artifact::validatePayload(artifact::loadMetadata(directory.path(), fixture.identity),
                            readBytes(artifactPaths.payload));
}

void invalidPublicationTests() {
  TemporaryDirectory directory;
  Fixture fixture;
  fixture.data[0] = std::numeric_limits<int8_t>::min();
  requireArtifactError(
      [&] { static_cast<void>(artifact::writeAtomically(directory.path(), fixture.identity,
                                                       fixture.data, fixture.scales)); },
      "publication accepted a forbidden -128 code");
  fixture.data[0] = 0;
  fixture.scales[0] = 0.0f;
  requireArtifactError(
      [&] { static_cast<void>(artifact::writeAtomically(directory.path(), fixture.identity,
                                                       fixture.data, fixture.scales)); },
      "publication accepted a zero row scale");
  fixture.scales[0] = 1.0f;
  requireArtifactError(
      [&] { static_cast<void>(artifact::writeAtomically(directory.path(), fixture.identity,
                                                       fixture.data, fixture.scales,
                                                       std::string(64, '0'))); },
      "publication accepted a mismatched runtime projection fingerprint");
  require(entryCount(directory.path()) == 0,
          "failed publication left a partial artifact or temporary directory");
}

void largePayloadLifetimeTests() {
  TemporaryDirectory directory;
  Fixture fixture;
  fixture.identity.n = 512;
  fixture.identity.k = 16384;
  fixture.data.resize(size_t{fixture.identity.n} * fixture.identity.k);
  fixture.scales.resize(fixture.identity.n);
  for (size_t index = 0; index < fixture.data.size(); ++index)
    fixture.data[index] = static_cast<int8_t>(static_cast<int>(index % 255) - 127);
  for (size_t index = 0; index < fixture.scales.size(); ++index)
    fixture.scales[index] = std::bit_cast<float>(
        uint32_t{0x3b800001} + static_cast<uint32_t>(index) * 257);
  const std::string expectedFingerprint = fixture.fingerprint();
  static_cast<void>(artifact::writeAtomically(
      directory.path(), fixture.identity, fixture.data, fixture.scales,
      expectedFingerprint));
  const auto metadata = artifact::loadMetadata(directory.path(), fixture.identity);
  require(metadata.dataBytes == 8 * 1024 * 1024 &&
              metadata.payloadBytes == metadata.dataBytes + 16 * 1024 &&
              metadata.projectionFingerprint == expectedFingerprint,
          "large artifact did not preserve the aligned footprint or legacy fingerprint");
  {
    ReadOnlyMapping mapping(metadata.payloadPath,
                            static_cast<size_t>(metadata.payloadBytes));
    artifact::validatePayload(metadata, mapping.bytes());
    require(std::memcmp(mapping.bytes().data(), fixture.data.data(),
                        fixture.data.size()) == 0 &&
                std::memcmp(mapping.bytes().data() + metadata.scaleOffset,
                            fixture.scales.data(), fixture.scales.size() * sizeof(float)) == 0,
            "large mapped code/scale bytes did not round-trip");
  }

  // Each lambda owns its input. An exception unwinds and destroys that input
  // before requireArtifactError catches it, exercising the worker join contract.
  requireArtifactError(
      [&] {
        auto payload = readBytes(metadata.payloadPath);
        payload.at(static_cast<size_t>(metadata.dataBytes - 1)) = 0x80;
        auto changedMetadata = metadata;
        refreshHashes(changedMetadata, payload);
        artifact::validatePayload(changedMetadata, payload);
      },
      "large late -128 code was accepted despite valid hashes");

  requireArtifactError(
      [&] {
        auto payload = readBytes(metadata.payloadPath);
        constexpr uint32_t nanBits = 0x7fc00001;
        std::memcpy(payload.data() + metadata.scaleOffset, &nanBits, sizeof(nanBits));
        auto changedMetadata = metadata;
        refreshHashes(changedMetadata, payload);
        const auto invalidPath = directory.path() / "invalid-scale.bin";
        writeBytes(invalidPath, payload);
        ReadOnlyMapping mapping(invalidPath,
                                static_cast<size_t>(changedMetadata.payloadBytes));
        artifact::validatePayload(changedMetadata, mapping.bytes());
      },
      "large nonfinite row scale was accepted despite valid hashes");
}

void concurrentPublicationTests() {
  TemporaryDirectory directory;
  Fixture fixture;
  const auto artifactPaths = artifact::paths(directory.path(), fixture.identity);
  std::barrier start(4);
  std::atomic<unsigned> finishedWriters{0};
  std::array<std::exception_ptr, 3> failures{};
  std::array<std::thread, 2> writers;
  for (size_t index = 0; index < writers.size(); ++index) {
    writers[index] = std::thread([&, index] {
      start.arrive_and_wait();
      try {
        static_cast<void>(artifact::writeAtomically(
            directory.path(), fixture.identity, fixture.data, fixture.scales,
            fixture.fingerprint()));
      } catch (...) {
        failures[index] = std::current_exception();
      }
      finishedWriters.fetch_add(1, std::memory_order_release);
    });
  }
  std::thread reader([&] {
    start.arrive_and_wait();
    try {
      do {
        // A final directory is the publication boundary. If it is visible,
        // both checked files must already be readable, even during the race.
        if (std::filesystem::exists(artifactPaths.directory)) {
          const auto metadata =
              artifact::loadMetadata(directory.path(), fixture.identity);
          artifact::validatePayload(metadata, readBytes(metadata.payloadPath));
        }
        std::this_thread::yield();
      } while (finishedWriters.load(std::memory_order_acquire) < writers.size());
    } catch (...) {
      failures[2] = std::current_exception();
    }
  });
  start.arrive_and_wait();
  for (auto &writer : writers)
    writer.join();
  reader.join();
  for (const auto &failure : failures)
    if (failure)
      std::rethrow_exception(failure);
  const auto metadata = artifact::loadMetadata(directory.path(), fixture.identity);
  artifact::validatePayload(metadata, readBytes(metadata.payloadPath));
  require(metadata.projectionFingerprint == fixture.fingerprint(),
          "concurrent publication produced the wrong artifact identity");
  require(entryCount(directory.path()) == 1 &&
              entryCount(artifactPaths.directory) == 2,
          "concurrent writers left more than one complete metadata/payload pair");
}

} // namespace

int main() {
  try {
    identityTests();
    roundTripAndPublicationTests();
    invalidPayloadTests();
    metadataIdentityTests();
    metadataFormatTests();
    invalidPublicationTests();
    largePayloadLifetimeTests();
    concurrentPublicationTests();
    std::cout << "PASS: preconverted INT8 CPU artifacts\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
