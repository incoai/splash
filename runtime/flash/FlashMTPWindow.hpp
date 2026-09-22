#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace splash::flash {

inline constexpr uint32_t kFlashSingletonMaximumMTPDepth = 15;
inline constexpr uint32_t kFlashSingletonMaximumVerifyRows =
    kFlashSingletonMaximumMTPDepth + 1;
inline constexpr uint32_t kFlashMTPConvolutionHistoryRows = 3;
inline constexpr uint32_t kFlashMTPMaximumCommittedFoldRows = 8;

// An explicit singleton override is canonical ASCII decimal, 1 through 15.
// The caller owns the default when the environment variable is absent.
[[nodiscard]] constexpr std::optional<uint32_t>
flashParseSingletonMTPDepth(std::string_view value) noexcept {
  if (value.size() == 1 && value[0] >= '1' && value[0] <= '9')
    return static_cast<uint32_t>(value[0] - '0');
  if (value.size() == 2 && value[0] == '1' && value[1] >= '0' && value[1] <= '5')
    return 10 + static_cast<uint32_t>(value[1] - '0');
  return std::nullopt;
}

// Keep each committed head fold in the raw-coefficient route. Splitting a
// larger retained prefix into ordered chunks avoids its BF16 cache boundary.
[[nodiscard]] constexpr std::optional<uint32_t>
flashMTPCommittedFoldChunkRows(uint32_t remaining) noexcept {
  if (!remaining || remaining > kFlashSingletonMaximumVerifyRows)
    return std::nullopt;
  return remaining < kFlashMTPMaximumCommittedFoldRows
      ? remaining : kFlashMTPMaximumCommittedFoldRows;
}

// Copy old history first, then the final incoming rows of the retained prefix.
// All offsets and lengths are in rows, so callers can apply their own stride.
struct FlashMTPConvolutionPrefix final {
  uint32_t oldRows = 0;
  uint32_t oldBegin = 0;
  uint32_t inputRows = 0;
  uint32_t inputBegin = 0;
  uint32_t destinationInputBegin = 0;
};

[[nodiscard]] constexpr std::optional<FlashMTPConvolutionPrefix>
flashMTPConvolutionPrefix(uint32_t kept) noexcept {
  if (!kept || kept > kFlashSingletonMaximumVerifyRows) return std::nullopt;
  const uint32_t oldRows = kept < kFlashMTPConvolutionHistoryRows
      ? kFlashMTPConvolutionHistoryRows - kept : 0;
  const uint32_t inputRows = kept < kFlashMTPConvolutionHistoryRows
      ? kept : kFlashMTPConvolutionHistoryRows;
  return FlashMTPConvolutionPrefix{
      oldRows, oldRows ? kept : 0, inputRows,
      kept - inputRows, oldRows};
}

enum class FlashMTPPrefixFinish : uint8_t { None, Stop, Length };

struct FlashMTPAcceptedPrefix final {
  std::array<uint32_t, kFlashSingletonMaximumVerifyRows> output{};
  // Count matches before EOS/quota truncation, for unbiased acceptance data.
  uint32_t matchedDrafts = 0;
  // This is also the number of verifier input rows that must be committed.
  uint32_t retainedRows = 0;
  FlashMTPPrefixFinish finish = FlashMTPPrefixFinish::None;
};

// inputs[0] is the known anchor. prediction[i] predicts the token after
// inputs[i]. Emit matched drafts and the first correction (or final bonus).
// EOS wins if it occurs at the quota boundary, matching ordinary greedy decode.
[[nodiscard]] constexpr std::optional<FlashMTPAcceptedPrefix>
flashMTPAcceptGreedyPrefix(std::span<const uint32_t> inputs,
                          std::span<const uint32_t> predictions,
                          uint32_t remaining,
                          std::span<const uint32_t> stopTokens = {}) noexcept {
  if (inputs.empty() || inputs.size() > kFlashSingletonMaximumVerifyRows ||
      predictions.size() != inputs.size() || !remaining)
    return std::nullopt;
  FlashMTPAcceptedPrefix result;
  const uint32_t drafts = static_cast<uint32_t>(inputs.size() - 1);
  while (result.matchedDrafts < drafts &&
         predictions[result.matchedDrafts] == inputs[result.matchedDrafts + 1])
    ++result.matchedDrafts;
  for (uint32_t index = 0; index <= result.matchedDrafts &&
       result.retainedRows < remaining; ++index) {
    const uint32_t token = index < result.matchedDrafts
        ? inputs[index + 1] : predictions[index];
    result.output[result.retainedRows++] = token;
    bool stop = false;
    for (const uint32_t stopToken : stopTokens) stop |= token == stopToken;
    if (stop) { result.finish = FlashMTPPrefixFinish::Stop; break; }
    if (result.retainedRows == remaining) {
      result.finish = FlashMTPPrefixFinish::Length; break;
    }
  }
  return result;
}

} // namespace splash::flash
