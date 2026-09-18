#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/Model.hpp"
#include "model/SlotFile.hpp"
#include "ops/KvCopy.hpp"
#include "ops/Q8PageStorage.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

// Moves KV pages between the Q8 page pool and a slot file. Pages live in
// private sparse buffers, so every transfer is a copy through host-visible
// staging that rides a command: a demotion copies the page out and the worker
// then writes the staging slot; a restore reads the disk slot into staging
// and the next command copies it into the page. Every transfer in flight
// holds one staging slot; demotions hold at most half of the ring and
// restores at most three quarters, so a burst of either kind leaves the
// other its share.
class KvPageTier final : public KvTier {
public:
  struct DiskSlot final : KvDiskSlot {
    explicit DiskSlot(std::shared_ptr<SlotFile::Slot> held) : slot(std::move(held)) {}
    std::shared_ptr<SlotFile::Slot> slot;
  };
  static constexpr uint32_t kDefaultStagingSlots = 128;

  // Disk bytes per page: the page rounded up for uncached IO.
  [[nodiscard]] static uint64_t slotBytesFor(const kv::Q8PageStorage &pages) noexcept;

  KvPageTier(metal::MetalBackend &backend, kv::Q8PageStorage &pages,
             std::shared_ptr<SlotFile> file,
             uint32_t stagingSlots = kDefaultStagingSlots);
  ~KvPageTier() override;
  KvPageTier(const KvPageTier &) = delete;
  KvPageTier &operator=(const KvPageTier &) = delete;

  [[nodiscard]] uint64_t slotBytes() const noexcept override;
  [[nodiscard]] uint64_t capacityBytes() const noexcept override;
  [[nodiscard]] uint64_t usedBytes() const noexcept override;
  [[nodiscard]] bool writable() const noexcept override;
  [[nodiscard]] std::shared_ptr<KvDiskSlot> acquireSlot() override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot,
         std::function<void()> completion) override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
          std::function<void()> completion) override;
  [[nodiscard]] bool copiesQueued() const noexcept override;
  void poll() override;

  // Runtime side. encode() appends every queued copy to the command being
  // built and returns its batch number, zero when nothing was added; the
  // command's completion reports that number back, from any thread. Copies
  // are keyed by staging slot, so a command only ever sees entries owned by
  // transfers it carries.
  [[nodiscard]] uint64_t encode(metal::CommandGraph &graph);
  void commandCompleted(uint64_t batch) noexcept;

private:
  struct Transfer;
  class Ticket;
  struct Batch {
    uint64_t number = 0;
    std::vector<std::shared_ptr<Transfer>> copies;
  };

  [[nodiscard]] std::span<std::byte> staging(uint32_t slot) noexcept;
  void setTable(uint32_t slot, uint32_t page, ops::KvCopy::Direction direction) noexcept;
  void finish(Transfer &transfer, bool success) noexcept;

  metal::MetalBackend &backend_;
  kv::Q8PageStorage &pages_;
  std::shared_ptr<SlotFile> file_;
  uint64_t slotBytes_;
  uint32_t stagingSlots_;
  std::shared_ptr<std::byte> memory_;
  metal::MetalBuffer staging_;
  // One entry per staging slot, non-idle exactly while its copy is encoded.
  metal::MetalBuffer table_;
  std::vector<uint32_t> freeStaging_;
  uint32_t demotionSlots_;
  uint32_t restoreSlots_;
  uint32_t demotionsInFlight_ = 0;
  uint32_t restoresInFlight_ = 0;
  std::vector<std::shared_ptr<Transfer>> queued_;
  std::deque<Batch> inFlight_;
  std::vector<std::shared_ptr<Transfer>> io_;
  uint64_t encodedBatches_ = 0;
  std::atomic<uint64_t> completedBatch_{0};
};

} // namespace splash::model
