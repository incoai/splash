#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/Model.hpp"
#include "model/SlotFile.hpp"
#include "ops/KvCopy.hpp"
#include "ops/PageStorage.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

// Moves KV pages between the KV page pool and a slot file. Pages live in
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
  [[nodiscard]] static uint64_t slotBytesFor(const kv::PageStorage &pages) noexcept;
  // Metal bytes a tier allocates for pages of this layout: its staging ring
  // and copy table. The memory plan sets them aside before the pool exists.
  [[nodiscard]] static uint64_t
  stagingBytesFor(kv::Layout layout, uint32_t stagingSlots = kDefaultStagingSlots) noexcept;

  KvPageTier(metal::MetalBackend &backend, kv::PageStorage &pages,
             std::shared_ptr<SlotFile> file,
             uint32_t stagingSlots = kDefaultStagingSlots);
  ~KvPageTier() override;
  KvPageTier(const KvPageTier &) = delete;
  KvPageTier &operator=(const KvPageTier &) = delete;

  // Metal bytes the staging ring and copy table actually hold.
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept { return actualAllocatedBytes_; }
  [[nodiscard]] uint64_t slotBytes() const noexcept override;
  [[nodiscard]] uint64_t capacityBytes() const noexcept override;
  [[nodiscard]] uint64_t usedBytes() const noexcept override;
  [[nodiscard]] uint64_t readBytes() const noexcept override { return file_->readBytes(); }
  [[nodiscard]] uint64_t writtenBytes() const noexcept override { return file_->writtenBytes(); }
  [[nodiscard]] bool writable() const noexcept override;
  [[nodiscard]] bool canDemote() const noexcept override;
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
  // built and returns the report the command's completion runs, from any
  // thread, or nothing when no copy was added. The report shares the batch
  // counter rather than pointing at the tier, which a late completion may
  // outlive. The runtime calls it for each batch command and each copy-only
  // command (see KvTier). Copies are keyed by staging slot, and encode()
  // first clears the entries of batches poll() has not retired, whose
  // command has finished since the backend runs one at a time, so a command
  // only ever sees entries owned by transfers it carries.
  [[nodiscard]] std::function<void()> encode(metal::CommandGraph &graph);

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
  kv::PageStorage &pages_;
  std::shared_ptr<SlotFile> file_;
  uint64_t slotBytes_;
  uint32_t stagingSlots_;
  std::shared_ptr<std::byte> memory_;
  metal::MetalBuffer staging_;
  // One entry per staging slot, non-idle exactly while its copy is encoded.
  metal::MetalBuffer table_;
  uint64_t actualAllocatedBytes_ = 0;
  std::vector<uint32_t> freeStaging_;
  uint32_t demotionSlots_;
  uint32_t restoreSlots_;
  uint32_t demotionsInFlight_ = 0;
  uint32_t restoresInFlight_ = 0;
  std::vector<std::shared_ptr<Transfer>> queued_;
  std::deque<Batch> inFlight_;
  std::vector<std::shared_ptr<Transfer>> io_;
  uint64_t encodedBatches_ = 0;
  // Shared with the reports encode() hands out.
  std::shared_ptr<std::atomic<uint64_t>> completedBatch_ =
      std::make_shared<std::atomic<uint64_t>>(0);
};

} // namespace splash::model
