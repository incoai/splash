#include "model/CopySpec.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::model {
namespace {

// FNV-1a over 64-bit words, mod 2^64: the bucket key of a window. The ids
// are masked to 64 bits before the XOR, as the reference hashes them.
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t hashTokens(const uint32_t *tokens, uint32_t count) {
  uint64_t value = kFnvOffsetBasis;
  for (uint32_t i = 0; i < count; ++i) {
    value ^= static_cast<uint64_t>(tokens[i]);
    value *= kFnvPrime;
  }
  return value;
}

} // namespace

CopySpecMode copySpecMode(std::string_view value) {
  if (value == "off")
    return CopySpecMode::Off;
  if (value == "conservative")
    return CopySpecMode::Conservative;
  throw std::invalid_argument("copyspec mode must be off or conservative");
}

CopySpecIndex::CopySpecIndex(std::span<const uint32_t> promptTokens,
                             uint32_t windowSize)
    : tokens_(promptTokens.begin(), promptTokens.end()), window_(windowSize) {
  if (!window_)
    throw std::invalid_argument("copyspec window must be positive");
  enabled_ = tokens_.size() > window_;
  if (!enabled_ || tokens_.size() < window_)
    return;
  for (uint32_t start = 0; start + window_ <= tokens_.size(); ++start)
    indexWindow(start);
}

bool CopySpecIndex::enabled() const { return enabled_; }

void CopySpecIndex::appendCommitted(std::span<const uint32_t> ids) {
  for (const uint32_t id : ids) {
    tokens_.push_back(id);
    if (!enabled_)
      continue;
    if (tokens_.size() >= window_)
      indexWindow(static_cast<uint32_t>(tokens_.size()) - window_);
  }
}

void CopySpecIndex::rebuild(std::span<const uint32_t> history) {
  tokens_.assign(history.begin(), history.end());
  index_.clear();
  if (!enabled_ || tokens_.size() < window_)
    return;
  for (uint32_t start = 0; start + window_ <= tokens_.size(); ++start)
    indexWindow(start);
}

void CopySpecIndex::indexWindow(uint32_t start) {
  const uint32_t end = start + window_;
  const uint64_t hash = hashTokens(tokens_.data() + start, window_);
  const auto found = index_.find(hash);
  if (found == index_.end()) {
    index_.emplace(hash, Ends{end});
  } else if (auto *one = std::get_if<uint32_t>(&found->second)) {
    found->second = Ends{std::vector<uint32_t>{*one, end}};
  } else {
    std::get<std::vector<uint32_t>>(found->second).push_back(end);
  }
}

uint32_t CopySpecIndex::draftAfter(uint32_t stagedFirst, uint32_t maxTokens,
                                   std::span<const uint32_t> forbidden,
                                   std::span<uint32_t> out) const {
  if (!enabled_ || !maxTokens)
    return 0;
  if (out.size() < maxTokens)
    throw std::invalid_argument("copyspec draft output is too small");
  // The query is the window ending in the uncommitted token.
  if (tokens_.size() + 1 < window_)
    return 0;
  std::vector<uint32_t> window;
  window.reserve(window_);
  if (window_ > 1) {
    window.insert(window.end(), tokens_.end() - (window_ - 1), tokens_.end());
  }
  window.push_back(stagedFirst);

  // The match with the longest continuation wins; the first of the
  // chronologically listed ends wins a tie.
  uint32_t bestEnd = 0;
  uint32_t bestAvailable = 0;
  bool found = false;
  const auto at = index_.find(hashTokens(window.data(), window_));
  if (at != index_.end()) {
    const auto ends = [&](auto &&visit) {
      if (const auto *one = std::get_if<uint32_t>(&at->second))
        visit(*one);
      else
        for (const uint32_t end : std::get<std::vector<uint32_t>>(at->second))
          visit(end);
    };
    ends([&](uint32_t end) {
      const uint32_t start = end - window_;
      if (start + window_ > tokens_.size() ||
          !std::equal(window.begin(), window.end(), tokens_.begin() + start))
        return;
      const uint32_t available =
          static_cast<uint32_t>(tokens_.size()) - end;
      if (!found || available > bestAvailable) {
        bestEnd = end;
        bestAvailable = available;
        found = true;
      }
    });
  }
  if (!found || !bestAvailable)
    return 0;

  const uint32_t count = std::min(maxTokens, bestAvailable);
  if (count < maxTokens)
    return 0;
  std::copy_n(tokens_.begin() + bestEnd, count, out.begin());
  for (uint32_t i = 0; i < count; ++i)
    for (const uint32_t token : forbidden)
      if (out[i] == token)
        return 0;
  return count;
}

void CopySpecAcceptance::observe(uint32_t verified, uint32_t accepted) {
  accepted = std::min(accepted, verified);
  accepted_ += accepted;
  reached_ += accepted < verified ? accepted + 1 : accepted;
}

double CopySpecAcceptance::rate() const {
  return (static_cast<double>(accepted_) + kPriorAccepted) /
         (static_cast<double>(reached_) + kPriorAccepted + kPriorRejected);
}

} // namespace splash::model
