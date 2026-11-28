#include "engine/WriteBehind.hpp"

#include <algorithm>

namespace splash::engine {

WriteBehind::WriteBehind(Cache &cache) : cache_(cache), copying_(cache.persistent()) {}

void WriteBehind::published(uint64_t block, uint32_t tokens, double now, bool head) {
  head = head && cache_.persistent();
  if (!cache_.kvTierWritable() || (tokens < kMinimumTokens && !head) || queued_.contains(block))
    return;
  Point point{.block = block, .due = now + kDelayMilliseconds, .head = head};
  if (cache_.persistent())
    point.due = head || fallsDue(point) ? now : kNever;
  // While copies wait for demand, and of the points not due, the queue keeps
  // only those their conversations still resume from.
  copying_ = copying();
  std::erase_if(waiting_, [&](const Point &point) {
    if ((copying_ && point.due != kNever) || cache_.newestPoint(point.block))
      return false;
    queued_.erase(point.block);
    ++counters_.unneeded;
    return true;
  });
  queued_.insert(block);
  waiting_.push_back(point);
}

bool WriteBehind::fallsDue(Point &point) const {
  // A gone point settles at once and one being copied goes on. Its base is
  // found by a walk of its chain only when the last one is no longer taken
  // back, a point landed (settle) or a copy its chain holds went (step), and
  // checked in O(1) a step.
  if (!cache_.chainLength(point.block) || cache_.stillTakenBack(point.block))
    return true;
  if (!point.base || !cache_.stillTakenBack(point.base)) {
    if (point.chain.empty())
      point.chain = cache_.chainBlocks(point.block);
    point.base = cache_.deepestTakenBack(point.block);
  }
  const uint32_t pages = cache_.chainLength(point.block) - (point.base ? cache_.chainLength(point.base) : 0);
  return pages * KvCache::pageTokens >= kMinimumTokens;
}

bool WriteBehind::settle(uint64_t block, PersistStatus status) {
  if (status == PersistStatus::Started || status == PersistStatus::Busy)
    return false;
  queued_.erase(block);
  // A point that landed may be a deeper base for those waiting.
  if (status == PersistStatus::Durable)
    for (Point &point : waiting_)
      point.base = 0;
  ++(status == PersistStatus::Durable    ? counters_.durable
     : status == PersistStatus::Unneeded ? counters_.unneeded
                                         : counters_.refused);
  return true;
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
  // A copy given up or lost under a state breaks the bases whose chains
  // hold it, and those alone are found again (fallsDue): at a full quota
  // nearly every copy given up lies under a state.
  for (const uint64_t block : cache_.takeBrokenCopies())
    if (const uint32_t depth = cache_.chainLength(block))
      for (Point &point : waiting_)
        if (point.base && depth <= cache_.chainLength(point.base) && point.chain[depth - 1] == block)
          point.base = 0;
  bool settled = false;
  for (auto point = waiting_.begin(); point != waiting_.end();) {
    // Measured again: its base may have gone, or a point before it started.
    if (cache_.persistent() && !point->head)
      point->due = fallsDue(*point) ? std::min(point->due, now) : kNever;
    if (point->due > now || inUse(point->block)) {
      ++point;
      continue;
    }
    if (!copies()) {
      stopped_ = Stop::Covered;
      break;
    }
    if (!settle(point->block, cache_.persist(point->block))) {
      stopped_ = Stop::Stalled;
      break;
    }
    point = waiting_.erase(point);
    settled = true;
  }
  return settled;
}

bool WriteBehind::flush() {
  if (!cache_.persistent())
    return true;
  while (!waiting_.empty()) {
    if (!cache_.takesCacheWrites()) {
      counters_.refused += waiting_.size();
      waiting_.clear();
      queued_.clear();
      break;
    }
    if (!settle(waiting_.back().block, cache_.persist(waiting_.back().block)))
      return false;
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
    if (point.due != kNever && !inUse(point.block))
      return stopped_ == Stop::Stalled ? cache_.diskWritesAdmittedAt() : point.due;
  return std::nullopt;
}

WriteBehindSnapshot WriteBehind::snapshot() const noexcept {
  WriteBehindSnapshot result = counters_;
  result.waiting = static_cast<uint32_t>(waiting_.size());
  return result;
}

} // namespace splash::engine
