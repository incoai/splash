#pragma once

#include "engine/KvTier.hpp"
#include "model/SlotFile.hpp"
#include "ops/PageStorage.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace splash::engine {

// Moves KV pages between their extents and a slot file. The file's IO worker
// writes a demoted page straight from its extent and reads a restored page
// straight back into it (PageStorage::spans), so the tier holds no memory of
// its own. That one worker serves both kinds in submission order, and a
// bounded number of transfers is in flight: demotions take at most half of
// it and restores at most three quarters, so a burst of either kind leaves
// the other its share.
class KvPageTier final : public KvTier {
public:
  struct DiskSlot final : KvDiskSlot {
    explicit DiskSlot(std::shared_ptr<model::SlotFile::Slot> held) : slot(std::move(held)) {}
    std::shared_ptr<model::SlotFile::Slot> slot;
  };
  // Transfers in flight at once. Tests set another bound through TestConfig.
  static constexpr uint32_t kTransfers = 128;

  KvPageTier(kv::PageStorage &pages, std::shared_ptr<model::SlotFile> file);
  ~KvPageTier() override;
  KvPageTier(const KvPageTier &) = delete;
  KvPageTier &operator=(const KvPageTier &) = delete;

  [[nodiscard]] uint64_t slotBytes() const noexcept override;
  [[nodiscard]] bool writable() const noexcept override;
  [[nodiscard]] bool canDemote() const noexcept override;
  [[nodiscard]] bool canRestore() const noexcept override;
  [[nodiscard]] std::shared_ptr<KvDiskSlot> acquireSlot() override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot,
         std::function<void()> completion) override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
          std::function<void()> completion) override;
  void poll() override;

private:
  struct Transfer;
  class Ticket;

  kv::PageStorage &pages_;
  std::shared_ptr<model::SlotFile> file_;
  uint32_t transferLimit_;
  uint32_t demotionLimit_;
  uint32_t restoreLimit_;
  uint32_t demotions_ = 0;
  uint32_t restores_ = 0;
  // Every transfer whose IO the worker may still run.
  std::vector<std::shared_ptr<Transfer>> inFlight_;
};

} // namespace splash::engine
