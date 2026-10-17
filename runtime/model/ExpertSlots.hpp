#pragma once

// The order the slots of an expert arena are recycled in (model::
// ExpertStore): which slot a load's missing expert takes, never one of an
// expert the same load routes. A load is a number that grows from load to
// load; a long prefill chunk's (`chunk`) routes nearly every expert of its
// block, a scan of them.
//
// Adaptive replacement (ARC, Megiddo and Modha, FAST 2003): the slots of
// experts routed once lately (T1) and of those routed again since (T2), each
// recycled least recently used first, and the experts each recently gave up
// (ghosts, B1 and B2, as many as the slots). A miss of an expert T1 gave up
// lately grows T1's target share, one T2 gave up shrinks it, and a slot is
// recycled from T1 while T1 holds more than its target, else from T2. So a
// set of experts routed again and again stays while new experts pass
// through T1: decode cycles whose experts outnumber the slots keep hitting
// the ones they route most, where recycling the least recently used slot
// over the loop of layers would keep none (each cycle's experts of a layer
// gone before the next cycle routes them again). A long chunk's routes are a
// scan: its misses enter T1 at its least recently used end and leave no
// ghost, and its hits move nothing, so a prompt does not flush what decode
// steps route.
//
// The store's own thread uses it alone (no locking).

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace splash::model {

class ExpertSlots final {
public:
  // `slots` slots, every one free.
  explicit ExpertSlots(uint32_t slots);

  // Load `load` routes the expert in `slot`.
  void touch(uint32_t slot, uint64_t load, bool chunk);
  // The slot load `load`'s missing expert `expert` (an identity unique in the
  // store) takes: free, or the one whose expert to evict; then it is the
  // load's. Throws std::runtime_error when every slot holds an expert the
  // load routes.
  [[nodiscard]] uint32_t claim(uint64_t expert, uint64_t load, bool chunk);
  // `slot` holds no expert (a failed load claimed it): the first claimed.
  void free(uint32_t slot);
  // Every slot free, and no ghost.
  void reset();

  [[nodiscard]] uint32_t size() const noexcept { return static_cast<uint32_t>(list_.size()); }

private:
  enum List : uint8_t { Free, Recent, Frequent };
  // The experts a list gave up, least recently first.
  struct Ghosts final {
    std::list<uint64_t> order;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> where;
    [[nodiscard]] bool holds(uint64_t expert) const { return where.contains(expert); }
    void add(uint64_t expert);
    void remove(uint64_t expert);
    void dropOldest();
    [[nodiscard]] size_t size() const noexcept { return where.size(); }
  };
  struct Order final {
    uint32_t head = UINT32_MAX, tail = UINT32_MAX, count = 0;
  };
  void unlink(uint32_t slot);
  void append(uint32_t slot, List list);
  void prepend(uint32_t slot, List list);
  // The least recently used slot of `list` the load does not route, or none.
  [[nodiscard]] uint32_t oldest(List list, uint64_t load) const;
  // Recycles a slot of T1 or T2 for `expert` (ARC's REPLACE): its expert
  // becomes a ghost, unless a scan brought it.
  [[nodiscard]] uint32_t replace(uint64_t expert, uint64_t load);
  void evict(uint32_t slot);

  std::vector<uint8_t> list_, scan_;
  std::vector<uint32_t> previous_, next_, free_;
  std::vector<uint64_t> expert_, lastLoad_;
  Order recent_, frequent_;
  Ghosts recentGhosts_, frequentGhosts_;
  // T1's target size.
  uint64_t target_ = 0;
};

} // namespace splash::model
