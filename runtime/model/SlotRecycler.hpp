#pragma once

// The order the slots of a store are recycled in, and which key each holds:
// which slot a load's missing key takes, never one of a key the same load
// uses. The stores are an expert arena (model::ExpertStore), whose keys are
// experts, and a layer of the hot block cache of spilled K/V
// (model::KvBlockCache), whose keys are blocks' tier copies. A load is a
// number that grows from load to load; a long prefill chunk's (`chunk`)
// routes nearly every expert of its block, a scan of them.
//
// Adaptive replacement (ARC, Megiddo and Modha, FAST 2003): the slots of
// keys used once lately (T1) and of those used again since (T2), each
// recycled least recently used first, and the keys each recently gave up
// (ghosts, B1 and B2, as many as the slots). A miss of a key T1 gave up
// lately grows T1's target share, one T2 gave up shrinks it, and a slot is
// recycled from T1 while T1 holds more than its target, else from T2. So a
// set of keys used again and again stays while new keys pass through T1:
// decode cycles whose experts outnumber the slots keep hitting the ones they
// route most, where recycling the least recently used slot over the loop of
// layers would keep none (each cycle's experts of a layer gone before the
// next cycle routes them again). A long chunk's routes are a scan: its
// misses enter T1 at its least recently used end and leave no ghost, and its
// hits move nothing, so a prompt does not flush what decode steps route.
//
// The keys of the slots and of the ghosts are in flat tables, and the ghosts
// in lists over arrays as many as the slots: a claim allocates nothing, and
// only resize() does.
//
// The store's own thread uses it alone (no locking).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace splash::model {

class SlotRecycler final {
public:
  // `slots` slots, every one free.
  explicit SlotRecycler(uint32_t slots);

  // The slot that holds `key`, if any.
  [[nodiscard]] std::optional<uint32_t> find(uint64_t key) const noexcept;
  // Load `load` uses the key in `slot`.
  void touch(uint32_t slot, uint64_t load, bool chunk);
  // Whether load `load` touched or claimed `slot`, which it then never
  // recycles.
  [[nodiscard]] bool pinned(uint32_t slot, uint64_t load) const noexcept { return lastLoad_[slot] == load; }
  // The slot load `load`'s missing key `key` (unique in the store) takes:
  // free, or the one whose key to evict; then it holds `key` and is the
  // load's. Throws std::runtime_error when every slot holds a key the load
  // uses.
  [[nodiscard]] uint32_t claim(uint64_t key, uint64_t load, bool chunk);
  // `slot` holds no key (a failed load claimed it): the first claimed.
  void free(uint32_t slot);
  // Every slot free, and no ghost.
  void reset();
  // `slots` slots: those added are free, and those past it are gone with
  // their keys. The ghosts stay, as many as the slots at most: fewer keep the
  // newest, those of T1 within what T1 leaves of the slots.
  void resize(uint32_t slots);

  [[nodiscard]] uint32_t size() const noexcept { return static_cast<uint32_t>(list_.size()); }
  // The claims, not a scan's, of a key T1 or T2 gave up lately.
  [[nodiscard]] uint64_t ghostHits() const noexcept { return ghostHits_; }

private:
  enum List : uint8_t { Free, Recent, Frequent };
  // A flat map from keys to values below UINT32_MAX: open addressing with
  // linear probing over at least twice as many buckets as it holds keys at
  // most (reserve), deletion by backward shift.
  class KeyIndex final {
  public:
    static constexpr uint32_t kAbsent = UINT32_MAX;
    // Holds up to `keys` keys from now on, with those it holds.
    void reserve(uint32_t keys);
    [[nodiscard]] uint32_t find(uint64_t key) const noexcept;
    // `key` is not held.
    void insert(uint64_t key, uint32_t value) noexcept;
    void erase(uint64_t key) noexcept;
    void clear() noexcept;

  private:
    [[nodiscard]] size_t home(uint64_t key) const noexcept {
      return static_cast<size_t>((key * 0x9e3779b97f4a7c15ull) >> shift_);
    }
    [[nodiscard]] size_t next(size_t bucket) const noexcept { return (bucket + 1) & (keys_.size() - 1); }

    std::vector<uint64_t> keys_;
    std::vector<uint32_t> values_;
    uint32_t shift_ = 63;
  };
  // A list of indices of the arrays `previous` and `next` link, least
  // recently used at its head.
  struct Order final {
    uint32_t head = UINT32_MAX, tail = UINT32_MAX, count = 0;
  };
  struct Links final {
    std::vector<uint32_t> previous, next;
    void unlink(Order &order, uint32_t at) noexcept;
    void append(Order &order, uint32_t at) noexcept;
    void prepend(Order &order, uint32_t at) noexcept;
    // `by` takes `at`'s place in the list.
    void replace(Order &order, uint32_t at, uint32_t by) noexcept;
  };

  [[nodiscard]] Order &slots(List list) noexcept { return list == Recent ? recent_ : frequent_; }
  [[nodiscard]] Order &ghosts(List list) noexcept { return list == Recent ? recentGhosts_ : frequentGhosts_; }
  void unlink(uint32_t slot);
  void append(uint32_t slot, List list);
  void prepend(uint32_t slot, List list);
  // The least recently used slot of `list` the load does not use, or none.
  [[nodiscard]] uint32_t oldest(List list, uint64_t load) const;
  // Recycles a slot of T1 or T2 (ARC's REPLACE, `frequentGhost` when the
  // key T2 gave up claims it): its key becomes a ghost, unless a scan brought
  // it.
  [[nodiscard]] uint32_t replace(uint64_t load, bool frequentGhost);
  // `slot` leaves its list and gives its key up, as a ghost or not.
  void evict(uint32_t slot, bool ghost);
  void addGhost(uint64_t key, List list);
  void dropGhost(uint32_t ghost);
  void forgetGhosts();
  // Before the arrays shrink to `slots`: drops the oldest ghosts past ARC's
  // bounds there (|T1| + |B1| and |B1| + |B2| at most the slots), and moves
  // those past `slots` to free places below it.
  void keepGhosts(uint32_t slots);

  std::vector<uint8_t> list_, scan_;
  Links links_;
  std::vector<uint32_t> free_;
  std::vector<uint64_t> key_, lastLoad_;
  Order recent_, frequent_;
  KeyIndex slotOf_;
  // The ghosts: each a key of B1 or B2 (ghostList_) in a list of its own, at
  // most as many as the slots; the unused ones are freeGhosts_.
  std::vector<uint64_t> ghostKey_;
  std::vector<uint8_t> ghostList_;
  Links ghostLinks_;
  std::vector<uint32_t> freeGhosts_;
  Order recentGhosts_, frequentGhosts_;
  KeyIndex ghostOf_;
  // T1's target size.
  uint64_t target_ = 0;
  uint64_t ghostHits_ = 0;
};

} // namespace splash::model
