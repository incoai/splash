#include "engine/WriteBehind.hpp"

namespace splash::engine {

WriteBehind::WriteBehind(Cache &cache) : cache_(cache) {}

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
  bool settled = false;
  while (!waiting_.empty() && waiting_.front().due <= now) {
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
  cache_.liftDiskWriteBudget();
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
  if (waiting_.empty())
    return std::nullopt;
  // A stalled step goes on when its writes land, which wakes the engine, or
  // once the write budget has room again, which does not.
  return stalled_ ? cache_.diskWritesAdmittedAt() : waiting_.front().due;
}

WriteBehindSnapshot WriteBehind::snapshot() const noexcept {
  WriteBehindSnapshot result = counters_;
  result.waiting = static_cast<uint32_t>(waiting_.size());
  return result;
}

} // namespace splash::engine
