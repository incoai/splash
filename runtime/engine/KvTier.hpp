#pragma once

#include "model/Model.hpp"
#include "model/SlotFile.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace splash::engine {

// A block's copy on the disk tier: a part for each component of its K/V
// (kv::Component), each a slot of its component's file with its own bytes
// of the quota and of the write budget, and its own presence. Releasing the
// last handle frees every slot it holds.
class KvDiskSlot {
public:
  // A part: none, no slot; Writing, a slot whose write has not landed (not
  // started, in flight, or failed); Landed, a slot a restore reads.
  enum class Presence : uint8_t { None, Writing, Landed };
  virtual ~KvDiskSlot() = default;
  // What its parts' slots take of the tier's quota while they are held
  // (model::SlotFile::Slot::bytes), in all and by part.
  [[nodiscard]] virtual uint64_t bytes() const noexcept = 0;
  [[nodiscard]] virtual uint64_t bytes(kv::Component part) const noexcept = 0;
  [[nodiscard]] virtual Presence presence(kv::Component part) const noexcept = 0;
  // The parts that hold a slot.
  [[nodiscard]] kv::Components parts() const noexcept {
    kv::Components held = 0;
    for (const kv::Component part : kv::kAllComponents)
      if (presence(part) != Presence::None) held |= kv::componentBit(part);
    return held;
  }
  // Every one of the parts has landed.
  [[nodiscard]] bool landed(kv::Components parts) const noexcept {
    for (const kv::Component part : kv::kAllComponents)
      if (parts & kv::componentBit(part) && presence(part) != Presence::Landed) return false;
    return true;
  }
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
// RAM a reclaim frees once its copy lands, a copy of a page that stays,
// which a restore point's chain gets ahead of need (Cache::persist), and a
// Spill request's page written through (Cache::writeThrough).
enum class WriteClass : uint8_t { Reclaim, Persist, Spill };

// The disk tier for KV blocks as the engine drives it, a copy's parts each
// in the file of its component. Every call names the parts it acts on, of
// the tier's components(): a part a call names that the copy lacks, or
// holds when it should not, is a std::logic_error. Until a transfer is
// ready its pages are the caller's to keep still: no command writes a page
// being demoted, no command reads or writes a page being restored, and
// neither page's extent is released.
class KvTier {
public:
  virtual ~KvTier() = default;
  // The components of the layout's blocks (kv::Layout::components).
  [[nodiscard]] virtual kv::Components components() const noexcept = 0;
  // What the parts' slots acquired now take of the quota (model::SlotFile::
  // quotaBytesFor of what each stores of a block).
  [[nodiscard]] virtual uint64_t slotBytes(kv::Components parts) const = 0;
  // False once a file stopped taking writes (model::SlotFile::writable);
  // existing copies stay readable.
  [[nodiscard]] virtual bool writable() const noexcept = 0;
  // The tier's copies outlive the process (model::SlotFile::Persistence).
  [[nodiscard]] virtual bool persistent() const noexcept = 0;
  // In a persistent tier, keeps the label with each of the parts' slots
  // once its demotion, started before, has landed (model::SlotFile::label).
  virtual void label(const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts,
                     std::vector<std::byte> label) = 0;
  // Engine-thread admission probe, before replacing any disk copies: room
  // for another write of the class in flight, within its share, in a
  // writable tier. A block's write of any of its parts is one write. The
  // cache asks the write budget before it writes (Cache::takesCacheWrites),
  // the tier does not.
  [[nodiscard]] virtual bool canDemote(WriteClass write) const noexcept = 0;
  // Engine-thread admission probe, before building a restore's arguments.
  [[nodiscard]] virtual bool canRestore() const noexcept = 0;
  // Slots for the parts, all or none, in a new copy or in `into`, which
  // holds none of them: `into` or the new copy, or null when the tier's
  // limit for the draw has no room for them all (model::DiskBudget::reserve).
  [[nodiscard]] virtual std::shared_ptr<KvDiskSlot> acquireSlot(kv::Components parts, model::DiskBudget::Draw draw,
                                                                std::shared_ptr<KvDiskSlot> into) = 0;
  // Starts writing the parts of a block's K/V from its pages, base page
  // first (kv::Layout::blockPages, of kv::PageForm::Full), into their slots,
  // which have not been written. Ready once every part's write ended, and a
  // success once every one landed. Null when the tier takes no more writes
  // of the class for now; the caller waits while transfers are in flight.
  [[nodiscard]] virtual std::unique_ptr<KvTransfer>
  demote(std::span<const uint32_t> pages, const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts,
         WriteClass write, std::function<void()> completion) = 0;
  // Reads the landed parts back into the block's pages, what the base
  // page's form holds of the base: its index alone, or its K/V and index.
  // Ready once those parts are read. Null when the tier takes no more
  // restores for now (canRestore); the caller retries later.
  [[nodiscard]] virtual std::unique_ptr<KvTransfer>
  restore(const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts, std::span<const uint32_t> pages,
          std::function<void()> completion) = 0;
  // Gives the parts' slots back: their bytes return to the quota once no
  // reader holds them (copy()).
  virtual void giveUp(const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts) = 0;
  // Starts the restores queued since the last call, in batches that each
  // run their first restore's completion once all of theirs have landed.
  virtual void submitRestores() = 0;
  // The copy's base part as reads of spilled K/V reach it
  // (model::KvPageSource), on any thread, for as long as they hold it.
  [[nodiscard]] virtual std::shared_ptr<const model::KvCopy>
  copy(const std::shared_ptr<KvDiskSlot> &slot) const = 0;
  // Engine-thread bookkeeping after IO completes.
  virtual void poll() = 0;
};

} // namespace splash::engine
