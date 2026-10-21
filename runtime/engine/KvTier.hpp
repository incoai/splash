#pragma once

#include "model/SlotFile.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace splash::model {
class KvCopy;
} // namespace splash::model

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

// What a write to the tier is for, each with its share of the writes in
// flight (KvPageTier), so that one kind never starves another: a page whose
// RAM a reclaim frees once its copy lands, a copy a persistent tier keeps of
// a page that stays (Cache::persist), and a Spill request's page written
// through (Cache::writeThrough).
enum class WriteClass : uint8_t { Reclaim, Persist, Spill };

// The disk tier for KV pages as the engine drives it. Until a transfer is
// ready its page is the caller's to keep still: no command writes a page
// being demoted, no command reads or writes a page being restored, and
// neither page's extent is released.
class KvTier {
public:
  virtual ~KvTier() = default;
  [[nodiscard]] virtual uint64_t slotBytes() const noexcept = 0;
  // False once its file stopped taking writes (model::SlotFile::writable);
  // existing copies stay readable.
  [[nodiscard]] virtual bool writable() const noexcept = 0;
  // The tier's copies outlive the process (model::SlotFile::Persistence).
  [[nodiscard]] virtual bool persistent() const noexcept = 0;
  // In a persistent tier, keeps the label with the slot once the demotion
  // started before it has landed (model::SlotFile::label).
  virtual void label(const std::shared_ptr<KvDiskSlot> &slot, std::vector<std::byte> label) = 0;
  // Engine-thread admission probe, before replacing any disk copies: room
  // for another write of the class in flight, within its share, in a
  // writable tier. The write budget holds no KV write back
  // (model::DiskBudget).
  [[nodiscard]] virtual bool canDemote(WriteClass write) const noexcept = 0;
  // Engine-thread admission probe, before building a restore's arguments.
  [[nodiscard]] virtual bool canRestore() const noexcept = 0;
  // Null when the tier is at its limit for the draw (model::DiskBudget::
  // reserve).
  [[nodiscard]] virtual std::shared_ptr<KvDiskSlot> acquireSlot(model::DiskBudget::Draw draw) = 0;
  // Starts writing the page, which holds its K/V (kv::PageForm::Full). Null
  // when the tier takes no more writes of the class for now; the caller
  // waits while transfers are in flight.
  [[nodiscard]] virtual std::unique_ptr<KvTransfer>
  demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot, WriteClass write,
         std::function<void()> completion) = 0;
  // Reads the page back from its slot, what the page's form holds: its
  // index alone, or all of it. Null when the tier takes no more restores for
  // now (canRestore); the caller retries later.
  [[nodiscard]] virtual std::unique_ptr<KvTransfer>
  restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
          std::function<void()> completion) = 0;
  // Starts the restores queued since the last call, in batches that each
  // run their first restore's completion once all of theirs have landed; a
  // tier that starts each as it comes has none to start.
  virtual void submitRestores() {}
  // The slot's copy as reads of spilled K/V reach it (model::KvPageSource),
  // on any thread, for as long as they hold it.
  [[nodiscard]] virtual std::shared_ptr<const model::KvCopy>
  copy(const std::shared_ptr<KvDiskSlot> &slot) const = 0;
  // Engine-thread bookkeeping after IO completes.
  virtual void poll() = 0;
};

} // namespace splash::engine
