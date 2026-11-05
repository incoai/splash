#include "engine/WriteBehind.hpp"

namespace splash::engine {

WriteBehind::WriteBehind(Cache &cache) : cache_(cache), copying_(cache.persistent()) {}

void WriteBehind::published(uint64_t block, uint32_t tokens, double now) {
  if (!cache_.kvTierWritable() || tokens < kMinimumTokens || queued_.contains(block))
    return;
  // While copies wait for demand, the queue keeps only the points their
  // conversations still resume from.
  copying_ = copying();
  if (!copying_)
    std::erase_if(waiting_, [&](const Point &point) {
      if (cache_.newestPoint(point.block))
        return false;
      queued_.erase(point.block);
      ++counters_.unneeded;
      return true;
    });
  queued_.insert(block);
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

bool WriteBehind::copying() const noexcept { return cache_.persistent() || demandBytes_ != 0; }

bool WriteBehind::demanded(uint64_t demandBytes) const {
  // What the cache gives a claim that cannot wait now, which waits for no
  // write to land, and what the copies ahead in flight let go once they
  // land. An upper bound, as peek counts (VictimSource), the count the
  // ledger judges a claim by (MemoryLedger::bound): copies ahead are
  // demanded where that count does not cover the claims ahead. A lower
  // bound would copy ahead where the ledger takes what frees at once and
  // reaches no uncopied cache.
  uint64_t covered = cache_.copyingAheadBytes();
  if (covered >= demandBytes)
    return false;
  const VictimQuery query{.need = {.bytes = demandBytes}};
  for (std::optional<Victim> unit = cache_.peek(query, nullptr); unit; unit = cache_.peek(query, &*unit)) {
    if (!unit->later)
      covered += unit->bytes;
    if (covered >= demandBytes)
      return false;
  }
  return true;
}

bool WriteBehind::inUse(uint64_t block) const noexcept {
  return !cache_.persistent() && cache_.stateInUse(block);
}

bool WriteBehind::run(double now, uint64_t demandBytes, bool claimWaiting) {
  demandBytes_ = demandBytes;
  copying_ = copying();
  return step(now, claimWaiting, [&] { return cache_.persistent() || demanded(demandBytes); });
}

bool WriteBehind::step(double now, bool claimWaiting, const std::function<bool()> &copies) {
  stopped_ = Stop::None;
  if (!copying_)
    return false;
  if (claimWaiting) {
    stopped_ = Stop::ClaimWaiting;
    return false;
  }
  bool settled = false;
  for (auto point = waiting_.begin(); point != waiting_.end() && point->due <= now;) {
    if (inUse(point->block)) {
      ++point;
      continue;
    }
    if (!copies()) {
      stopped_ = Stop::Covered;
      break;
    }
    if (!settle(cache_.persist(point->block))) {
      stopped_ = Stop::Stalled;
      break;
    }
    queued_.erase(point->block);
    point = waiting_.erase(point);
    settled = true;
  }
  return settled;
}

bool WriteBehind::flush() {
  if (!cache_.persistent())
    return true;
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
  // A stalled step goes on when the writes or restores it waits for land,
  // which wakes the engine, or once the write budget has room again, which
  // does not. A point in use waits for its request's end, a step of the
  // engine, which runs this again. A claim that waits wakes the engine as
  // what it waits for lands, and only a step changes the demand.
  if (!copying_ || stopped_ == Stop::ClaimWaiting || stopped_ == Stop::Covered)
    return std::nullopt;
  for (const Point &point : waiting_)
    if (!inUse(point.block))
      return stopped_ == Stop::Stalled ? cache_.diskWritesAdmittedAt() : point.due;
  return std::nullopt;
}

WriteBehindSnapshot WriteBehind::snapshot() const noexcept {
  WriteBehindSnapshot result = counters_;
  result.waiting = static_cast<uint32_t>(waiting_.size());
  return result;
}

} // namespace splash::engine
