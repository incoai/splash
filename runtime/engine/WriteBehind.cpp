#include "engine/WriteBehind.hpp"

#include <algorithm>

namespace splash::engine {
namespace {

constexpr double kHourMilliseconds = 3'600'000.0;
constexpr double kSampleMilliseconds = 60'000.0;

} // namespace

WriteBehind::WriteBehind(Cache &cache, uint64_t hourlyBytes)
    : cache_(cache), hourlyBytes_(hourlyBytes) {}

void WriteBehind::published(uint64_t block, uint32_t tokens, double now) {
  if (!cache_.persistent() || tokens < kMinimumTokens || !queued_.insert(block).second)
    return;
  waiting_.push_back({block, now + kDelayMilliseconds});
}

bool WriteBehind::settle(PersistStatus status) {
  switch (status) {
  case PersistStatus::Durable:
    ++counters_.durable;
    return true;
  case PersistStatus::Unneeded:
    ++counters_.unneeded;
    return true;
  case PersistStatus::Refused:
    ++counters_.refused;
    return true;
  case PersistStatus::Started:
  case PersistStatus::Busy:
    return false;
  }
  return false;
}

bool WriteBehind::run(double now) {
  stalled_ = false;
  if (!cache_.persistent())
    return false;
  sampleWrites(now);
  bool settled = false;
  while (!waiting_.empty() && waiting_.front().due <= now && !limited_) {
    const uint64_t block = waiting_.front().block;
    if (!settle(cache_.persist(block))) {
      stalled_ = true;
      break;
    }
    queued_.erase(block);
    waiting_.pop_front();
    settled = true;
  }
  return settled;
}

bool WriteBehind::flush() {
  while (!waiting_.empty()) {
    const uint64_t block = waiting_.back().block;
    if (!settle(cache_.persist(block)))
      return false;
    queued_.erase(block);
    waiting_.pop_back();
  }
  return true;
}

std::optional<double> WriteBehind::nextWakeup() const {
  if (waiting_.empty() || stalled_)
    return std::nullopt;
  const double due = waiting_.front().due;
  // A limit reached is looked at again with the next sample.
  return limited_ ? std::max(due, written_.back().first + kSampleMilliseconds) : due;
}

WriteBehindSnapshot WriteBehind::snapshot() const noexcept {
  WriteBehindSnapshot result = counters_;
  result.waiting = static_cast<uint32_t>(waiting_.size());
  return result;
}

void WriteBehind::sampleWrites(double now) {
  const uint64_t written = cache_.diskWrittenBytes();
  if (written_.empty() || now - written_.back().first >= kSampleMilliseconds)
    written_.emplace_back(now, written);
  while (written_.size() > 1 && now - written_[1].first >= kHourMilliseconds)
    written_.pop_front();
  limited_ = written - written_.front().second >= hourlyBytes_;
}

} // namespace splash::engine
