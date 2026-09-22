#pragma once

#include "flash/FlashDescriptor.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash {

struct FlashInt8ExpertStorePlane final {
  uint64_t offset = 0;
  uint64_t length = 0;
  std::string sha256;
};

struct FlashInt8ExpertStoreLayer final {
  std::filesystem::path path;
  uint64_t bytes = 0;
  std::string sha256;
  std::vector<uint32_t> selectedIDs;
  // Fixed projection order: gate, up, down.
  std::array<FlashInt8ExpertStorePlane, 3> codes;
  std::array<FlashInt8ExpertStorePlane, 3> scales;
};

// CPU metadata and filesystem checks only. The consuming constructor must
// verify whole-file/plane hashes, padding, code ranges, and scale values before
// it exposes any operand to GPU execution. The selected expert inventory is
// uniform across all 48 layers, with exactly 32, 64 or 128 IDs in each layer.
struct FlashInt8ExpertStoreMetadata final {
  std::filesystem::path directory;
  std::string sourceIdentity;
  std::string sourceManifestSha256;
  std::string planSha256;
  std::string identitySha256;
  uint64_t plannedBytes = 0;
  uint64_t totalBytes = 0;
  std::array<FlashInt8ExpertStoreLayer, 48> layers;
};

[[nodiscard]] FlashInt8ExpertStoreMetadata
loadFlashInt8ExpertStoreMetadata(const std::filesystem::path &directory,
                                std::string_view expectedSourceIdentity,
                                std::string_view expectedManifestFingerprint,
                                NormConvention convention);

} // namespace splash::flash
