#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace splash::engine {

// One slot of the disk tier holding a KV page; releasing the last handle
// frees the slot.
class KvDiskSlot {
public:
  virtual ~KvDiskSlot() = default;
};

// One KV page moving between its pool page and the disk tier, read from the
// page or written to it in place, on the host, beside whatever command runs.
// ready() means the write has finished (demotion) or the page holds the data
// (restore), and finish() reports success. The page stays valid throughout,
// so a failed demotion loses nothing.
class KvTransfer {
public:
  virtual ~KvTransfer() = default;
  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual bool finish() = 0;
};

// The disk tier for KV pages as the engine drives it. Until a transfer is
// ready its page is the caller's to keep still: no command writes a page
// being demoted, no command reads or writes a page being restored, and
// neither page's extent is released.
class KvTier {
public:
  virtual ~KvTier() = default;
  [[nodiscard]] virtual uint64_t slotBytes() const noexcept = 0;
  // False once a write has failed; existing copies stay readable.
  [[nodiscard]] virtual bool writable() const noexcept = 0;
  // The tier's copies outlive the process (model::SlotFile::Persistence).
  [[nodiscard]] virtual bool persistent() const noexcept = 0;
  // In a persistent tier, keeps the label with the slot once the demotion
  // started before it has landed (model::SlotFile::label).
  virtual void label(const std::shared_ptr<KvDiskSlot> &slot, std::vector<std::byte> label) = 0;
  // Engine-thread admission probe, before replacing any disk copies.
  [[nodiscard]] virtual bool canDemote() const noexcept = 0;
  // Engine-thread admission probe, before building a restore's arguments.
  [[nodiscard]] virtual bool canRestore() const noexcept = 0;
  // Null when the disk quota is full.
  [[nodiscard]] virtual std::shared_ptr<KvDiskSlot> acquireSlot() = 0;
  // Starts writing the page. Null when the tier takes no more demotions for
  // now; the caller waits while transfers are in flight.
  [[nodiscard]] virtual std::unique_ptr<KvTransfer>
  demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot,
         std::function<void()> completion) = 0;
  // Starts reading the page. Null when the tier takes no more restores for
  // now (canRestore); the caller retries later.
  [[nodiscard]] virtual std::unique_ptr<KvTransfer>
  restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
          std::function<void()> completion) = 0;
  // Engine-thread bookkeeping after IO completes.
  virtual void poll() = 0;
};

} // namespace splash::engine
