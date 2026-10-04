#pragma once

#include "engine/KvCache.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace splash::engine {

// What a persistent tier's record keeps with a slot, so that the next
// process can put the slot back where it was: a KV block with its key and
// parent, or a state with its block. Both carry the recency of their copy.
// Labels are stored as this host lays them out; Splash runs on
// little-endian arm64 only.
struct KvBlockLabel final {
  uint64_t id = 0;
  uint64_t parent = 0;
  uint64_t lastUsed = 0;
  uint64_t imagesLo = 0;
  uint64_t imagesHi = 0;
  std::array<uint32_t, KvCache::pageTokens> tokens{};
};

struct StateLabel final {
  uint64_t block = 0;
  uint64_t lastUsed = 0;
  // The state's boundary: its block's chain, in tokens.
  uint32_t tokens = 0;
  uint32_t checkpoint = 0;
};

static_assert(std::endian::native == std::endian::little);
static_assert(std::is_trivially_copyable_v<KvBlockLabel> && sizeof(KvBlockLabel) == 168);
static_assert(std::is_trivially_copyable_v<StateLabel> && sizeof(StateLabel) == 24);

template <typename Label>
[[nodiscard]] std::vector<std::byte> encodeLabel(const Label &label) {
  std::vector<std::byte> bytes(sizeof(Label));
  std::memcpy(bytes.data(), &label, sizeof(Label));
  return bytes;
}

template <typename Label>
[[nodiscard]] std::optional<Label> decodeLabel(std::span<const std::byte> bytes) noexcept {
  if (bytes.size() != sizeof(Label))
    return std::nullopt;
  Label label;
  std::memcpy(&label, bytes.data(), sizeof(Label));
  return label;
}

} // namespace splash::engine
