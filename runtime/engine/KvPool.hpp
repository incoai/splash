#pragma once

#include "ops/PagedKv.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace splash::engine {

using KvBacking = kv::Backing;

struct KvPoolSnapshot {
  uint32_t pagesTotal = 0;
  uint32_t pagesFree = 0;
  uint32_t pagesActive = 0;
  uint32_t pagesPrefix = 0;
  uint32_t pagesResident = 0;
  uint32_t pagesFreeResident = 0;
  uint32_t reclaimableExtents = 0;
  uint64_t residentBackingBytes = 0;
  uint64_t reclaimableBackingBytes = 0;
};

enum class KvPageAcquireFailure : uint8_t {
  None,
  LogicalCapacity,
  PhysicalCapacity,
  // A transfer in flight (a KV copy, a restore or the one state write) holds
  // what the request needs; retry when it lands.
  Pending,
};

struct KvPageAcquisition {
  std::vector<uint32_t> pages;
  KvPageAcquireFailure failure = KvPageAcquireFailure::None;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept {
    return failure == KvPageAcquireFailure::None;
  }
};

// Sole owner of logical KV page references and backing residency. Resource
// policy may ask for pages or release references, but cannot directly map or
// unmap Metal memory. Free resident pages are handed out from the extent with
// the most live pages first, so partially used extents fill up, empty extents
// are touched last, and cold extents drain to empty, the only state in which
// their backing can be released.
class KvPool final {
public:
  explicit KvPool(KvBacking &backing);

  [[nodiscard]] KvPageAcquisition acquirePages(uint32_t count,
                                               bool prefixOwner);
  void retainPage(uint32_t page, bool prefixOwner);
  void releasePage(uint32_t page, bool prefixOwner);

  [[nodiscard]] uint32_t pageCount() const noexcept;
  [[nodiscard]] uint64_t bytesPerPage() const noexcept;
  [[nodiscard]] uint32_t freePageCount() const noexcept;
  // Free pages whose backing is mapped; acquisition hands these out first.
  [[nodiscard]] uint32_t freeResidentPageCount() const noexcept;
  [[nodiscard]] uint32_t activeReferences(uint32_t page) const;
  [[nodiscard]] bool pageFree(uint32_t page) const;
  [[nodiscard]] uint64_t residentBackingBytes() const noexcept;

  // Reclaims only completely unreferenced extents. keepRunway retains one
  // resident extent to avoid adding mapping latency to the next request.
  // Releases stop at maxExtents and whenever the backing is still tearing
  // down a previous release; the remaining empty extents stay reclaimable.
  [[nodiscard]] uint32_t reclaimEmptyExtents(
      bool keepRunway,
      uint32_t maxExtents = std::numeric_limits<uint32_t>::max());
  [[nodiscard]] uint32_t reclaimableExtentCount() const noexcept;
  [[nodiscard]] bool releaseReady() const noexcept;
  // Changes on release submission and observed completion; does not poll.
  [[nodiscard]] uint64_t releaseGeneration() const noexcept;
  // Startup only: the serving path never waits on a release, and shutdown
  // pacing belongs to the backing's destructor.
  void awaitRelease();
  [[nodiscard]] KvPoolSnapshot snapshot() const;

private:
  static constexpr uint32_t noIndex = std::numeric_limits<uint32_t>::max();
  enum class FreeClass : uint8_t { None, Resident, Unbacked };

  struct PageRecord {
    uint32_t activeReferences = 0;
    uint32_t prefixReferences = 0;
    uint32_t extent = noIndex;
    uint32_t previousFree = noIndex;
    uint32_t nextFree = noIndex;
    FreeClass freeClass = FreeClass::None;
  };

  struct IndexList {
    uint32_t head = noIndex;
    uint32_t count = 0;
  };

  struct ExtentRecord {
    uint32_t firstPage = 0;
    uint32_t pageCount = 0;
    uint32_t usedPages = 0;
    uint32_t previousReclaimable = noIndex;
    uint32_t nextReclaimable = noIndex;
    IndexList freeResident;
    bool resident = false;
    bool reclaimable = false;
  };

  [[nodiscard]] IndexList &freeList(FreeClass kind, uint32_t page) noexcept;
  void insertFree(uint32_t page, FreeClass kind) noexcept;
  void removeFree(uint32_t page) noexcept;
  [[nodiscard]] uint32_t popFreeResident() noexcept;
  [[nodiscard]] uint32_t packingExtent() noexcept;
  void markUsed(uint32_t page) noexcept;
  void markFree(uint32_t page) noexcept;
  void setExtentResident(uint32_t extent, bool resident) noexcept;
  void setExtentReclaimable(uint32_t extent, bool reclaimable) noexcept;

  bool releaseBacking(uint32_t page);

  KvBacking &backing_;
  mutable uint64_t releaseGeneration_ = 0;
  mutable bool releaseOutstanding_ = false;
  std::vector<PageRecord> pages_;
  std::vector<ExtentRecord> extents_;
  uint32_t freeResidentPages_ = 0;
  // The extent currently being filled. Stays valid while only this extent
  // changes, so a burst of allocations rescans the extents once per extent
  // it moves into.
  uint32_t packingExtent_ = noIndex;
  IndexList freeUnbacked_;
  IndexList reclaimableExtents_;
  uint32_t activePages_ = 0;
  uint32_t prefixPages_ = 0;
  uint32_t residentPages_ = 0;
  uint64_t reclaimableBackingBytes_ = 0;
};

} // namespace splash::engine
