#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::flash {

inline constexpr uint64_t kFlashPLESSDAlignment = 16384;

inline bool flashPLESSDStreamingValue(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_PLE_SSD_STREAMING must be 0 or 1");
}

inline uint64_t flashPLESSDCacheBytes(const char *value, bool streamingEnabled) {
  if (!value) return 64ULL << 20;
  if (!streamingEnabled)
    throw std::invalid_argument("SPLASH_FLASH_PLE_SSD_CACHE_MB requires SPLASH_FLASH_PLE_SSD_STREAMING=1");
  const std::string_view text(value);
  if (text.empty()) throw std::invalid_argument("SPLASH_FLASH_PLE_SSD_CACHE_MB must be decimal 0..1024");
  uint64_t mib = 0;
  for (char digit : text) {
    if (digit < '0' || digit > '9' || mib > 1024)
      throw std::invalid_argument("SPLASH_FLASH_PLE_SSD_CACHE_MB must be decimal 0..1024");
    mib = mib * 10 + static_cast<uint64_t>(digit - '0');
  }
  if (mib > 1024)
    throw std::invalid_argument("SPLASH_FLASH_PLE_SSD_CACHE_MB must be decimal 0..1024");
  return mib << 20;
}

inline bool flashPLESSDTableTensor(std::string_view name) {
  constexpr std::string_view prefix =
      "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.";
  if (!name.starts_with(prefix)) return false;
  name.remove_prefix(prefix.size());
  const auto dot = name.find('.');
  if (dot == std::string_view::npos || dot == 0) return false;
  const auto part = name.substr(0, dot);
  if (part.size() > 1 && part.front() == '0') return false;
  uint32_t index = 0;
  for (char digit : part) {
    if (digit < '0' || digit > '9' || index > 127) return false;
    index = index * 10 + static_cast<uint32_t>(digit - '0');
  }
  const auto suffix = name.substr(dot);
  return index < 128 && (suffix == ".weight" || suffix == ".scales" || suffix == ".biases");
}

struct FlashPLESSDLayoutRange final {
  uint64_t begin = 0;
  uint64_t end = 0;
  bool diskOnly = false;
};

struct FlashPLESSDLayoutPlan final {
  std::vector<FlashPLESSDLayoutRange> windows;
  uint64_t mappedBytes = 0;
  uint64_t diskOnlyBytes = 0;
  uint64_t diskTensorCount = 0;
};

inline uint64_t flashPLESSDAligned(uint64_t value) {
  if (value > UINT64_MAX - (kFlashPLESSDAlignment - 1))
    throw std::overflow_error("PLE SSD layout alignment overflows");
  return (value + kFlashPLESSDAlignment - 1) & ~(kFlashPLESSDAlignment - 1);
}

// Canonical existing tensor offsets partition each payload into unique native
// map windows; no disk-only PLE page belongs to a GPU-backed allocation.
inline FlashPLESSDLayoutPlan flashPLESSDPlan(
    uint64_t fileBytes, std::vector<FlashPLESSDLayoutRange> ranges) {
  if (!fileBytes || fileBytes % kFlashPLESSDAlignment || ranges.empty())
    throw std::invalid_argument("PLE SSD layout has an empty or unaligned payload");
  std::sort(ranges.begin(), ranges.end(), [](const auto &a, const auto &b) {
    return a.begin < b.begin;
  });
  FlashPLESSDLayoutPlan result;
  uint64_t cursor = 0;
  for (const auto &range : ranges) {
    if (range.begin != flashPLESSDAligned(cursor) || range.end <= range.begin ||
        range.end > fileBytes)
      throw std::invalid_argument("PLE SSD layout is not canonical and nonoverlapping");
    const uint64_t paddedEnd = flashPLESSDAligned(range.end);
    if (paddedEnd > fileBytes)
      throw std::invalid_argument("PLE SSD layout extends beyond its payload");
    if (range.diskOnly) {
      result.diskOnlyBytes += paddedEnd - range.begin;
      ++result.diskTensorCount;
    } else {
      result.mappedBytes += paddedEnd - range.begin;
      if (!result.windows.empty() && result.windows.back().end == range.begin)
        result.windows.back().end = paddedEnd;
      else
        result.windows.push_back({range.begin, paddedEnd, false});
    }
    cursor = range.end;
  }
  if (flashPLESSDAligned(cursor) != fileBytes ||
      result.mappedBytes > fileBytes || result.diskOnlyBytes != fileBytes - result.mappedBytes)
    throw std::invalid_argument("PLE SSD layout does not partition its original payload");
  return result;
}

} // namespace splash::flash
