#include "model/ExpertSlots.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kNone = UINT32_MAX;

} // namespace

void ExpertSlots::Ghosts::add(uint64_t expert) {
  order.push_back(expert);
  where[expert] = std::prev(order.end());
}

void ExpertSlots::Ghosts::remove(uint64_t expert) {
  if (const auto found = where.find(expert); found != where.end()) {
    order.erase(found->second);
    where.erase(found);
  }
}

void ExpertSlots::Ghosts::dropOldest() {
  where.erase(order.front());
  order.pop_front();
}

ExpertSlots::ExpertSlots(uint32_t slots)
    : list_(slots), scan_(slots), previous_(slots), next_(slots), expert_(slots), lastLoad_(slots) {
  reset();
}

void ExpertSlots::unlink(uint32_t slot) {
  Order &order = list_[slot] == Recent ? recent_ : frequent_;
  (previous_[slot] == kNone ? order.head : next_[previous_[slot]]) = next_[slot];
  (next_[slot] == kNone ? order.tail : previous_[next_[slot]]) = previous_[slot];
  --order.count;
  list_[slot] = Free;
}

void ExpertSlots::append(uint32_t slot, List list) {
  Order &order = list == Recent ? recent_ : frequent_;
  list_[slot] = list;
  previous_[slot] = order.tail;
  next_[slot] = kNone;
  (order.tail == kNone ? order.head : next_[order.tail]) = slot;
  order.tail = slot;
  ++order.count;
}

void ExpertSlots::prepend(uint32_t slot, List list) {
  Order &order = list == Recent ? recent_ : frequent_;
  list_[slot] = list;
  next_[slot] = order.head;
  previous_[slot] = kNone;
  (order.head == kNone ? order.tail : previous_[order.head]) = slot;
  order.head = slot;
  ++order.count;
}

uint32_t ExpertSlots::oldest(List list, uint64_t load) const {
  uint32_t slot = list == Recent ? recent_.head : frequent_.head;
  while (slot != kNone && lastLoad_[slot] == load) slot = next_[slot];
  return slot;
}

void ExpertSlots::evict(uint32_t slot) {
  const List list = List(list_[slot]);
  unlink(slot);
  if (!scan_[slot]) (list == Recent ? recentGhosts_ : frequentGhosts_).add(expert_[slot]);
  scan_[slot] = 0;
}

uint32_t ExpertSlots::replace(uint64_t expert, uint64_t load) {
  const bool fromRecent = recent_.count && (recent_.count > target_ ||
                                            (frequentGhosts_.holds(expert) && recent_.count == target_));
  uint32_t slot = oldest(fromRecent ? Recent : Frequent, load);
  if (slot == kNone) slot = oldest(fromRecent ? Frequent : Recent, load);
  if (slot == kNone) throw std::runtime_error("the expert arena holds fewer slots than one load's experts");
  evict(slot);
  return slot;
}

void ExpertSlots::touch(uint32_t slot, uint64_t load, bool chunk) {
  lastLoad_[slot] = load;
  // A scan moves nothing; another route makes an expert of T1 frequent.
  if (chunk) return;
  unlink(slot);
  scan_[slot] = 0;
  append(slot, Frequent);
}

uint32_t ExpertSlots::claim(uint64_t expert, uint64_t load, bool chunk) {
  const uint64_t slots = size();
  const auto take = [&](uint32_t replaced) {
    if (free_.empty()) return replaced == kNone ? replace(expert, load) : replaced;
    const uint32_t slot = free_.back();
    free_.pop_back();
    return slot;
  };
  uint32_t slot = kNone;
  bool frequent = false;
  if (!chunk && recentGhosts_.holds(expert)) {
    const uint64_t step = std::max<uint64_t>(1, frequentGhosts_.size() / recentGhosts_.size());
    target_ = std::min<uint64_t>(slots, target_ + step);
    slot = take(kNone);
    recentGhosts_.remove(expert);
    frequent = true;
  } else if (!chunk && frequentGhosts_.holds(expert)) {
    const uint64_t step = std::max<uint64_t>(1, recentGhosts_.size() / frequentGhosts_.size());
    target_ = target_ > step ? target_ - step : 0;
    slot = take(kNone);
    frequentGhosts_.remove(expert);
    frequent = true;
  } else {
    recentGhosts_.remove(expert);
    frequentGhosts_.remove(expert);
    if (free_.empty()) {
      const uint64_t recent = recent_.count + recentGhosts_.size();
      const uint64_t all = recent + frequent_.count + frequentGhosts_.size();
      if (recent >= slots) {
        if (recent_.count < slots && recentGhosts_.size()) {
          recentGhosts_.dropOldest();
          slot = replace(expert, load);
        } else {
          // T1 holds every slot: its oldest goes without a ghost.
          slot = oldest(Recent, load);
          if (slot == kNone) throw std::runtime_error("the expert arena holds fewer slots than one load's experts");
          unlink(slot);
          scan_[slot] = 0;
        }
      } else {
        if (all >= 2 * slots && frequentGhosts_.size()) frequentGhosts_.dropOldest();
        slot = replace(expert, load);
      }
    }
    slot = take(slot);
  }
  expert_[slot] = expert;
  lastLoad_[slot] = load;
  scan_[slot] = chunk;
  if (frequent) append(slot, Frequent);
  else if (chunk) prepend(slot, Recent);
  else append(slot, Recent);
  return slot;
}

void ExpertSlots::free(uint32_t slot) {
  if (list_[slot] != Free) {
    unlink(slot);
    free_.push_back(slot);
  }
  scan_[slot] = 0;
  lastLoad_[slot] = 0;
}

void ExpertSlots::reset() {
  recent_ = frequent_ = {};
  recentGhosts_ = {};
  frequentGhosts_ = {};
  target_ = 0;
  free_.clear();
  for (uint32_t slot = size(); slot-- > 0;) free_.push_back(slot);
  std::fill(list_.begin(), list_.end(), uint8_t{Free});
  std::fill(scan_.begin(), scan_.end(), uint8_t{0});
  std::fill(lastLoad_.begin(), lastLoad_.end(), 0);
}

} // namespace splash::model
