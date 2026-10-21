#include "model/SlotRecycler.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kNone = UINT32_MAX;

} // namespace

void SlotRecycler::KeyIndex::reserve(uint32_t keys) {
  const size_t buckets = std::bit_ceil(std::max<size_t>(2, 2 * size_t{keys}));
  std::vector<uint64_t> heldKeys = std::move(keys_);
  std::vector<uint32_t> heldValues = std::move(values_);
  keys_.assign(buckets, 0);
  values_.assign(buckets, kAbsent);
  shift_ = static_cast<uint32_t>(64 - std::countr_zero(buckets));
  for (size_t bucket = 0; bucket < heldValues.size(); ++bucket)
    if (heldValues[bucket] != kAbsent) insert(heldKeys[bucket], heldValues[bucket]);
}

uint32_t SlotRecycler::KeyIndex::find(uint64_t key) const noexcept {
  if (values_.empty()) return kAbsent;
  size_t bucket = home(key);
  while (values_[bucket] != kAbsent && keys_[bucket] != key) bucket = next(bucket);
  return values_[bucket];
}

void SlotRecycler::KeyIndex::insert(uint64_t key, uint32_t value) noexcept {
  size_t bucket = home(key);
  while (values_[bucket] != kAbsent) bucket = next(bucket);
  keys_[bucket] = key;
  values_[bucket] = value;
}

void SlotRecycler::KeyIndex::erase(uint64_t key) noexcept {
  if (values_.empty()) return;
  size_t hole = home(key);
  while (values_[hole] != kAbsent && keys_[hole] != key) hole = next(hole);
  if (values_[hole] == kAbsent) return;
  // Each key after the hole in its run moves into it unless its home lies
  // after the hole, up to the key: probing from its home would not reach it.
  for (size_t bucket = next(hole); values_[bucket] != kAbsent; bucket = next(bucket)) {
    const size_t from = home(keys_[bucket]);
    const bool homeAfterHole = hole <= bucket ? hole < from && from <= bucket : hole < from || from <= bucket;
    if (homeAfterHole) continue;
    keys_[hole] = keys_[bucket];
    values_[hole] = values_[bucket];
    hole = bucket;
  }
  values_[hole] = kAbsent;
}

void SlotRecycler::KeyIndex::clear() noexcept { std::ranges::fill(values_, kAbsent); }

void SlotRecycler::Links::unlink(Order &order, uint32_t at) noexcept {
  (previous[at] == kNone ? order.head : next[previous[at]]) = next[at];
  (next[at] == kNone ? order.tail : previous[next[at]]) = previous[at];
  --order.count;
}

void SlotRecycler::Links::append(Order &order, uint32_t at) noexcept {
  previous[at] = order.tail;
  next[at] = kNone;
  (order.tail == kNone ? order.head : next[order.tail]) = at;
  order.tail = at;
  ++order.count;
}

void SlotRecycler::Links::prepend(Order &order, uint32_t at) noexcept {
  next[at] = order.head;
  previous[at] = kNone;
  (order.head == kNone ? order.tail : previous[order.head]) = at;
  order.head = at;
  ++order.count;
}

void SlotRecycler::Links::replace(Order &order, uint32_t at, uint32_t by) noexcept {
  previous[by] = previous[at];
  next[by] = next[at];
  (previous[at] == kNone ? order.head : next[previous[at]]) = by;
  (next[at] == kNone ? order.tail : previous[next[at]]) = by;
}

SlotRecycler::SlotRecycler(uint32_t slots) { resize(slots); }

void SlotRecycler::unlink(uint32_t slot) {
  links_.unlink(slots(List(list_[slot])), slot);
  list_[slot] = Free;
}

void SlotRecycler::append(uint32_t slot, List list) {
  list_[slot] = list;
  links_.append(slots(list), slot);
}

void SlotRecycler::prepend(uint32_t slot, List list) {
  list_[slot] = list;
  links_.prepend(slots(list), slot);
}

uint32_t SlotRecycler::oldest(List list, uint64_t load) const {
  uint32_t slot = list == Recent ? recent_.head : frequent_.head;
  while (slot != kNone && lastLoad_[slot] == load) slot = links_.next[slot];
  return slot;
}

void SlotRecycler::addGhost(uint64_t key, List list) {
  // ARC keeps |B1| + |B2| at most the slots: a ghost's claim drops it before
  // a slot is recycled, and a recycling past the target drops one first.
  if (freeGhosts_.empty()) throw std::logic_error("a slot recycler holds more ghosts than slots");
  const uint32_t ghost = freeGhosts_.back();
  freeGhosts_.pop_back();
  ghostKey_[ghost] = key;
  ghostList_[ghost] = list;
  ghostLinks_.append(ghosts(list), ghost);
  ghostOf_.insert(key, ghost);
}

void SlotRecycler::dropGhost(uint32_t ghost) {
  ghostLinks_.unlink(ghosts(List(ghostList_[ghost])), ghost);
  ghostOf_.erase(ghostKey_[ghost]);
  freeGhosts_.push_back(ghost);
}

void SlotRecycler::keepGhosts(uint32_t slots) {
  while (recentGhosts_.count && recent_.count + recentGhosts_.count > slots) dropGhost(recentGhosts_.head);
  while (recentGhosts_.count + frequentGhosts_.count > slots)
    dropGhost(frequentGhosts_.count ? frequentGhosts_.head : recentGhosts_.head);
  // As many places below `slots` are free as ghosts lie past it.
  std::erase_if(freeGhosts_, [slots](uint32_t ghost) { return ghost >= slots; });
  for (Order *order : {&recentGhosts_, &frequentGhosts_}) {
    for (uint32_t ghost = order->head; ghost != kNone;) {
      const uint32_t next = ghostLinks_.next[ghost];
      if (ghost >= slots) {
        const uint32_t place = freeGhosts_.back();
        freeGhosts_.pop_back();
        ghostKey_[place] = ghostKey_[ghost];
        ghostList_[place] = ghostList_[ghost];
        ghostLinks_.replace(*order, ghost, place);
        ghostOf_.erase(ghostKey_[ghost]);
        ghostOf_.insert(ghostKey_[place], place);
      }
      ghost = next;
    }
  }
}

void SlotRecycler::forgetGhosts() {
  recentGhosts_ = frequentGhosts_ = {};
  ghostOf_.clear();
  freeGhosts_.clear();
  for (uint32_t ghost = static_cast<uint32_t>(ghostKey_.size()); ghost-- > 0;) freeGhosts_.push_back(ghost);
}

void SlotRecycler::evict(uint32_t slot, bool ghost) {
  const List list = List(list_[slot]);
  unlink(slot);
  slotOf_.erase(key_[slot]);
  if (ghost && !scan_[slot]) addGhost(key_[slot], list);
  scan_[slot] = 0;
}

uint32_t SlotRecycler::replace(uint64_t load, bool frequentGhost) {
  const bool fromRecent =
      recent_.count && (recent_.count > target_ || (frequentGhost && recent_.count == target_));
  uint32_t slot = oldest(fromRecent ? Recent : Frequent, load);
  if (slot == kNone) slot = oldest(fromRecent ? Frequent : Recent, load);
  if (slot == kNone) throw std::runtime_error("a slot recycler holds fewer slots than one load's keys");
  evict(slot, true);
  return slot;
}

std::optional<uint32_t> SlotRecycler::find(uint64_t key) const noexcept {
  const uint32_t slot = slotOf_.find(key);
  if (slot == KeyIndex::kAbsent) return std::nullopt;
  return slot;
}

void SlotRecycler::touch(uint32_t slot, uint64_t load, bool chunk) {
  lastLoad_[slot] = load;
  // A scan moves nothing; another use makes a key of T1 frequent.
  if (chunk) return;
  unlink(slot);
  scan_[slot] = 0;
  append(slot, Frequent);
}

uint32_t SlotRecycler::claim(uint64_t key, uint64_t load, bool chunk) {
  const uint64_t slots = size();
  const uint32_t ghost = ghostOf_.find(key);
  const List gaveUp = ghost == KeyIndex::kAbsent ? Free : List(ghostList_[ghost]);
  const auto take = [&](bool frequentGhost) {
    if (free_.empty()) return replace(load, frequentGhost);
    const uint32_t slot = free_.back();
    free_.pop_back();
    return slot;
  };
  uint32_t slot = kNone;
  bool frequent = false;
  if (!chunk && gaveUp != Free) {
    // The list that gave the key up lately grows its share; then the key
    // stops being a ghost before a slot is recycled, which may add one.
    const Order &mine = ghosts(gaveUp), &other = ghosts(gaveUp == Recent ? Frequent : Recent);
    const uint64_t step = std::max<uint64_t>(1, other.count / mine.count);
    target_ = gaveUp == Recent ? std::min<uint64_t>(slots, target_ + step) : target_ > step ? target_ - step : 0;
    ++ghostHits_;
    dropGhost(ghost);
    slot = take(gaveUp == Frequent);
    frequent = true;
  } else {
    if (gaveUp != Free) dropGhost(ghost);
    if (free_.empty()) {
      const uint64_t recent = recent_.count + recentGhosts_.count;
      const uint64_t all = recent + frequent_.count + frequentGhosts_.count;
      if (recent >= slots) {
        if (recent_.count < slots && recentGhosts_.count) {
          dropGhost(recentGhosts_.head);
          slot = replace(load, false);
        } else {
          // T1 holds every slot: its oldest goes without a ghost.
          slot = oldest(Recent, load);
          if (slot == kNone) throw std::runtime_error("a slot recycler holds fewer slots than one load's keys");
          evict(slot, false);
        }
      } else {
        if (all >= 2 * slots && frequentGhosts_.count) dropGhost(frequentGhosts_.head);
        slot = replace(load, false);
      }
    } else {
      slot = take(false);
    }
  }
  key_[slot] = key;
  lastLoad_[slot] = load;
  scan_[slot] = chunk;
  slotOf_.insert(key, slot);
  if (frequent) append(slot, Frequent);
  else if (chunk) prepend(slot, Recent);
  else append(slot, Recent);
  return slot;
}

void SlotRecycler::free(uint32_t slot) {
  if (list_[slot] != Free) {
    unlink(slot);
    slotOf_.erase(key_[slot]);
    free_.push_back(slot);
  }
  scan_[slot] = 0;
  lastLoad_[slot] = 0;
}

void SlotRecycler::reset() {
  recent_ = frequent_ = {};
  slotOf_.clear();
  forgetGhosts();
  target_ = 0;
  free_.clear();
  for (uint32_t slot = size(); slot-- > 0;) free_.push_back(slot);
  std::ranges::fill(list_, uint8_t{Free});
  std::ranges::fill(scan_, uint8_t{0});
  std::ranges::fill(lastLoad_, 0);
}

void SlotRecycler::resize(uint32_t slots) {
  const uint32_t held = size();
  const bool fewer = slots < held;
  if (fewer) {
    for (uint32_t slot = slots; slot < held; ++slot)
      if (list_[slot] != Free) evict(slot, false);
    std::erase_if(free_, [slots](uint32_t slot) { return slot >= slots; });
    target_ = std::min<uint64_t>(target_, slots);
    keepGhosts(slots);
  }
  list_.resize(slots, Free);
  scan_.resize(slots, 0);
  links_.previous.resize(slots);
  links_.next.resize(slots);
  key_.resize(slots);
  lastLoad_.resize(slots, 0);
  for (uint32_t slot = slots; slot-- > held;) free_.push_back(slot);
  slotOf_.reserve(slots);
  // As many ghosts as slots: more add free ones.
  ghostKey_.resize(slots);
  ghostList_.resize(slots);
  ghostLinks_.previous.resize(slots);
  ghostLinks_.next.resize(slots);
  if (!fewer)
    for (uint32_t ghost = slots; ghost-- > held;) freeGhosts_.push_back(ghost);
  ghostOf_.reserve(slots);
}

} // namespace splash::model
