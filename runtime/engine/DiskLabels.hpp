#pragma once

#include "engine/KvCache.hpp"
#include "ops/PagedKv.hpp"

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
  uint64_t identityLo = 0;
  uint64_t identityHi = 0;
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

// A KV slot's label: its block's, then for a windowed layout (kv::Layout::
// windowedLayers) the form of the page the slot holds, one byte: a build
// whose windowed pages take another form keeps the namespace, and a build
// that cannot restore that form drops those copies. Other layouts' slots
// hold Full pages alone and keep the label they always had.
struct KvSlotLabel final {
  KvBlockLabel block;
  kv::PageForm form = kv::PageForm::Full;
};

[[nodiscard]] constexpr uint32_t kvSlotLabelBytes(bool windowed) noexcept {
  return sizeof(KvBlockLabel) + (windowed ? sizeof(kv::PageForm) : 0);
}

[[nodiscard]] inline std::vector<std::byte> encodeKvSlotLabel(const KvSlotLabel &label,
                                                             bool windowed) {
  std::vector<std::byte> bytes = encodeLabel(label.block);
  if (windowed)
    bytes.push_back(static_cast<std::byte>(label.form));
  return bytes;
}

[[nodiscard]] inline std::optional<KvSlotLabel>
decodeKvSlotLabel(std::span<const std::byte> bytes, bool windowed) noexcept {
  if (bytes.size() != kvSlotLabelBytes(windowed))
    return std::nullopt;
  KvSlotLabel label{*decodeLabel<KvBlockLabel>(bytes.first(sizeof(KvBlockLabel)))};
  if (windowed)
    label.form = static_cast<kv::PageForm>(bytes.back());
  return label;
}

} // namespace splash::engine
