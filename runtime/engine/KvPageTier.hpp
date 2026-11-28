#pragma once

#include "engine/KvTier.hpp"
#include "model/KvCopyRead.hpp"
#include "model/Model.hpp"
#include "model/SlotFile.hpp"
#include "model/SsdReaders.hpp"
#include "ops/KvSlotFormat.hpp"
#include "ops/PageStorage.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::engine {

// Moves KV blocks between their pages' extents and slot files, each part of
// a block (kv::Component) in a file of its own, laid out in its slot as
// kv::SlotFormat says, so the tier holds no memory of its own beyond what
// reads land in. Each file's IO worker writes a part straight from its
// extent (SlotFormat::transferSpans). The tier bounds the writes in flight,
// counted by block: a block's write of its parts is one write, whatever
// parts it writes, its parts' IO on their files' workers at once. Reclaim
// may use all of them, copies ahead of need a quarter and write-through
// half, so that one kind never starves another and reclaim always keeps a
// quarter. A part's slot holds its K/V and, for the base, its index
// (kv::SlotFormat::payloadBytes), and takes that much of the quota
// (model::SlotFile::quotaBytesFor). Reads go to the SSD readers beside the
// workers (model::readKvCopy): a restore is a task of a background batch
// that reads what its parts hold, of the base what its page's form holds,
// the index alone or all of it, into memory of its own and scatters it into
// the pages. The restores asked for since the last submitRestores() go out
// together, the index restores as one batch on every reader and the whole
// restores as another on at most kWholeRestoreReaders, whose reads are a
// part's slot each. A base part's slot is read the same way by whoever holds
// it (copy()).
//
// The tier reaches its pages as SlotFormat's page functions do, with their
// layout() and each page's form(page): Pages is kv::PageStorage for the
// pool's extents (KvPageTier), and host memory where tests run the tier
// under the sanitizers.
template <class Pages> class BasicKvPageTier final : public KvTier {
public:
  // A part's slot, which is also the copy reads of spilled K/V reach
  // (model::KvCopy).
  struct PartSlot final : model::KvCopy {
    PartSlot(std::shared_ptr<model::SlotFile::Slot> held, std::shared_ptr<const model::SlotFile> owner)
        : slot(std::move(held)), file(std::move(owner)) {}
    [[nodiscard]] uint64_t id() const noexcept override { return slot->id(); }
    [[nodiscard]] std::optional<std::span<const std::byte>>
    readAt(uint64_t offset, uint64_t bytes, std::span<std::byte> buffer) const noexcept override {
      return file->readAt(*slot, offset, bytes, buffer);
    }
    std::shared_ptr<model::SlotFile::Slot> slot;
    std::shared_ptr<const model::SlotFile> file;
  };
  // A block's copy: a part's slot by component, and its presence.
  struct DiskSlot final : KvDiskSlot {
    [[nodiscard]] uint64_t bytes() const noexcept override {
      uint64_t total = 0;
      for (const kv::Component part : kv::kAllComponents) total += bytes(part);
      return total;
    }
    [[nodiscard]] uint64_t bytes(kv::Component part) const noexcept override {
      const std::shared_ptr<PartSlot> &held = slots[static_cast<size_t>(part)];
      return held ? held->slot->bytes() : 0;
    }
    [[nodiscard]] Presence presence(kv::Component part) const noexcept override {
      return slots[static_cast<size_t>(part)] ? presences[static_cast<size_t>(part)] : Presence::None;
    }
    std::array<std::shared_ptr<PartSlot>, kv::kComponents> slots;
    std::array<Presence, kv::kComponents> presences{};
  };
  // Writes in flight at once, by default, and restores at most; the readers
  // whole restores take at once (four threads read 460 KiB pieces at
  // 8.1 GB/s, model::SsdReaders).
  static constexpr uint32_t kWrites = 64;
  static constexpr uint32_t kRestores = 4096;
  static constexpr unsigned kWholeRestoreReaders = 8;

  // `files` holds a file for each of the layout's components
  // (kv::Layout::components), base first.
  BasicKvPageTier(Pages &pages, kv::SlotFormat format, std::vector<std::shared_ptr<model::SlotFile>> files,
                  std::shared_ptr<model::SsdReaders> readers, uint32_t writes = kWrites);
  // How the tier lays a block out in its slots, as a persistent cache's
  // namespace names it (persistentCacheNamespace's kv line): empty for a
  // layout with one component whose slots hold the order of its spans
  // (kv::SlotFormat version 0), so that it keeps its namespace; the block
  // records' version for a spillable layout; and the parts of a windowed
  // layout, each in its own file. Whatever changes what the tier writes
  // changes this with it, so that no process adopts a slot of another layout.
  [[nodiscard]] static std::string_view slotFormatToken(const kv::Layout &layout) noexcept;
  ~BasicKvPageTier() override;
  BasicKvPageTier(const BasicKvPageTier &) = delete;
  BasicKvPageTier &operator=(const BasicKvPageTier &) = delete;

  [[nodiscard]] kv::Components components() const noexcept override { return format_.layout().components(); }
  [[nodiscard]] uint64_t slotBytes(kv::Components parts) const override;
  [[nodiscard]] bool writable() const noexcept override;
  [[nodiscard]] bool persistent() const noexcept override { return files_.front()->persistent(); }
  void label(const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts, std::vector<std::byte> label) override;
  // Takes back as the copy's part a slot an earlier process recorded in
  // that part's file (model::SlotFile::adopt), into `into` or a new copy.
  [[nodiscard]] std::shared_ptr<KvDiskSlot> adopt(const model::SlotRecord &record, kv::Component part,
                                                  std::shared_ptr<KvDiskSlot> into);
  [[nodiscard]] bool canDemote(WriteClass write) const noexcept override;
  [[nodiscard]] bool canRestore() const noexcept override { return restores_ < kRestores; }
  [[nodiscard]] std::shared_ptr<KvDiskSlot> acquireSlot(kv::Components parts, model::DiskBudget::Draw draw,
                                                        std::shared_ptr<KvDiskSlot> into) override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  demote(std::span<const uint32_t> pages, const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts,
         WriteClass write, std::function<void()> completion) override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  restore(const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts, std::span<const uint32_t> pages,
          std::function<void()> completion) override;
  void giveUp(const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts) override;
  void submitRestores() override;
  [[nodiscard]] std::shared_ptr<const model::KvCopy>
  copy(const std::shared_ptr<KvDiskSlot> &slot) const override {
    return diskSlot(slot)->slots[static_cast<size_t>(kv::Component::Base)];
  }
  void poll() override;

private:
  // What a transfer's ticket reports once the tier retired it.
  struct Outcome final {
    bool ready = false;
    bool success = false;
  };
  class Ticket final : public KvTransfer {
  public:
    explicit Ticket(std::shared_ptr<Outcome> outcome) : outcome_(std::move(outcome)) {}
    bool ready() const noexcept override { return outcome_->ready; }
    bool finish() override { return outcome_->success; }

  private:
    std::shared_ptr<Outcome> outcome_;
  };
  // A block's write: its copy, and the IO of each part it writes; a part
  // whose file refused it has none, and fails the write.
  struct Write final {
    WriteClass write;
    std::shared_ptr<DiskSlot> copy;
    kv::Components parts = 0;
    std::array<std::shared_ptr<model::SlotFile::Operation>, kv::kComponents> io;
    std::shared_ptr<Outcome> outcome;
  };
  // A restore: the slots of the parts it reads, its block's pages and the
  // base page's form, which says what it reads of the base: the page's
  // index alone (an Index page) or the block's payload.
  struct Restore final {
    std::array<std::shared_ptr<const PartSlot>, kv::kComponents> parts;
    std::vector<uint32_t> pages;
    kv::PageForm form = kv::PageForm::Full;
    std::function<void()> completion;
    std::shared_ptr<Outcome> outcome;
  };
  // Restores submitted together, until the tier retired each of them.
  struct RestoreBatch final {
    std::shared_ptr<std::vector<Restore>> restores;
    std::shared_ptr<model::SsdReaders::Batch> batch;
    std::vector<bool> retired;
    size_t left = 0;
  };
  [[nodiscard]] static std::shared_ptr<DiskSlot> diskSlot(const std::shared_ptr<KvDiskSlot> &slot);
  // std::logic_error unless the parts are some of the layout's components
  // and the copy's presence of each is `presence`.
  void requireParts(const DiskSlot &copy, kv::Components parts, KvDiskSlot::Presence presence) const;
  [[nodiscard]] model::SlotFile &file(kv::Component part) const { return *files_[static_cast<size_t>(part)]; }
  // A reader's task: reads one restore's parts and scatters them.
  void read(const Restore &restore) const;
  [[nodiscard]] uint32_t share(WriteClass write) const noexcept;

  Pages &pages_;
  const kv::SlotFormat format_;
  std::vector<std::shared_ptr<model::SlotFile>> files_;
  std::shared_ptr<model::SsdReaders> readers_;
  uint32_t writeLimit_;
  std::array<uint32_t, 3> classWrites_{};
  // Every write whose IO a worker may still run.
  std::vector<Write> writes_;
  // Restores asked for and not yet submitted, and those submitted.
  std::vector<Restore> queued_;
  std::vector<RestoreBatch> batches_;
  uint32_t restores_ = 0;
};

// The tier of the pool's extents.
using KvPageTier = BasicKvPageTier<kv::PageStorage>;
extern template class BasicKvPageTier<kv::PageStorage>;

template <class Pages>
BasicKvPageTier<Pages>::BasicKvPageTier(Pages &pages, kv::SlotFormat format,
                                        std::vector<std::shared_ptr<model::SlotFile>> files,
                                        std::shared_ptr<model::SsdReaders> readers, uint32_t writes)
    : pages_(pages), format_(std::move(format)), files_(std::move(files)), readers_(std::move(readers)),
      writeLimit_(writes) {
  if (files_.size() != static_cast<size_t>(std::popcount(components())) ||
      std::ranges::any_of(files_, [](const auto &file) { return !file; }) || !readers_ || !writeLimit_)
    throw std::invalid_argument("KV tier needs a slot file for each component, readers and room for a write");
  if (format_.layout() != pages_.layout())
    throw std::invalid_argument("KV tier's slot format is another layout's");
  // Tracking a submitted IO must not allocate: the tier drains every IO it
  // tracks before it goes, and an IO it lost would outlive the page.
  writes_.reserve(writeLimit_);
}

// A transfer in flight reads or writes its pages in place, and the cache
// that held the pages for it is gone before the tier is: the workers and the
// readers must be done with every page first. Nothing will read the slots a
// cancelled write leaves behind, nor the pages an unread restore leaves,
// because the tier that maps pages to them is going away too.
template <class Pages> BasicKvPageTier<Pages>::~BasicKvPageTier() {
  for (const Write &write : writes_)
    for (const auto &io : write.io)
      if (io) io->drain();
  for (RestoreBatch &batch : batches_)
    readers_->stop(*batch.batch);
}

template <class Pages>
std::string_view BasicKvPageTier<Pages>::slotFormatToken(const kv::Layout &layout) noexcept {
  // kv::SlotFormat's version 1, the block records of a spillable layout; a
  // windowed layout's base and window parts in two files.
  if (layout.spillable())
    return " slots block-records-v1";
  return layout.components() != kv::componentBit(kv::Component::Base) ? " slots base-window-parts-v1" : "";
}

template <class Pages>
std::shared_ptr<typename BasicKvPageTier<Pages>::DiskSlot>
BasicKvPageTier<Pages>::diskSlot(const std::shared_ptr<KvDiskSlot> &slot) {
  auto disk = std::dynamic_pointer_cast<DiskSlot>(slot);
  if (!disk)
    throw std::invalid_argument("KV disk slot belongs to another tier");
  return disk;
}

template <class Pages>
void BasicKvPageTier<Pages>::requireParts(const DiskSlot &copy, kv::Components parts,
                                          KvDiskSlot::Presence presence) const {
  bool held = parts && !(parts & ~components());
  for (const kv::Component part : kv::kAllComponents)
    held = held && (!(parts & kv::componentBit(part)) || copy.presence(part) == presence);
  if (!held)
    throw std::logic_error("a KV copy's parts are not as the tier's call needs them");
}

template <class Pages> uint64_t BasicKvPageTier<Pages>::slotBytes(kv::Components parts) const {
  uint64_t bytes = 0;
  for (const kv::Component part : kv::kAllComponents)
    if (parts & components() & kv::componentBit(part))
      bytes += file(part).quotaBytesFor(format_.payloadBytes(part));
  return bytes;
}

template <class Pages> bool BasicKvPageTier<Pages>::writable() const noexcept {
  return std::ranges::all_of(files_, [](const auto &file) { return file->writable(); });
}

template <class Pages>
void BasicKvPageTier<Pages>::label(const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts,
                                   std::vector<std::byte> label) {
  const std::shared_ptr<DiskSlot> disk = diskSlot(copy);
  requireParts(*disk, parts, KvDiskSlot::Presence::Writing);
  for (const kv::Component part : kv::kAllComponents)
    if (parts & kv::componentBit(part))
      file(part).label(disk->slots[static_cast<size_t>(part)]->slot, label);
}

template <class Pages>
std::shared_ptr<KvDiskSlot> BasicKvPageTier<Pages>::adopt(const model::SlotRecord &record, kv::Component part,
                                                          std::shared_ptr<KvDiskSlot> into) {
  std::shared_ptr<DiskSlot> copy = into ? diskSlot(into) : std::make_shared<DiskSlot>();
  requireParts(*copy, kv::componentBit(part), KvDiskSlot::Presence::None);
  const auto index = static_cast<size_t>(part);
  copy->slots[index] = std::make_shared<PartSlot>(file(part).adopt(record), files_[index]);
  copy->presences[index] = KvDiskSlot::Presence::Landed;
  return copy;
}

template <class Pages> uint32_t BasicKvPageTier<Pages>::share(WriteClass write) const noexcept {
  switch (write) {
  case WriteClass::Reclaim: return writeLimit_;
  case WriteClass::Persist: return std::max(1u, writeLimit_ / 4);
  case WriteClass::Spill: return std::max(1u, writeLimit_ / 2);
  }
  return 0;
}

template <class Pages> bool BasicKvPageTier<Pages>::canDemote(WriteClass write) const noexcept {
  return writes_.size() < writeLimit_ && classWrites_[static_cast<size_t>(write)] < share(write) &&
         writable();
}

template <class Pages>
std::shared_ptr<KvDiskSlot> BasicKvPageTier<Pages>::acquireSlot(kv::Components parts, model::DiskBudget::Draw draw,
                                                                std::shared_ptr<KvDiskSlot> into) {
  const std::shared_ptr<DiskSlot> copy = into ? diskSlot(into) : std::make_shared<DiskSlot>();
  requireParts(*copy, parts, KvDiskSlot::Presence::None);
  // All or none: a part whose file has no room gives back those taken.
  std::array<std::shared_ptr<PartSlot>, kv::kComponents> taken;
  for (const kv::Component part : kv::kAllComponents) {
    if (!(parts & kv::componentBit(part)))
      continue;
    auto slot = file(part).acquire(format_.payloadBytes(part), draw);
    if (!slot)
      return {};
    taken[static_cast<size_t>(part)] = std::make_shared<PartSlot>(std::move(slot), files_[static_cast<size_t>(part)]);
  }
  for (size_t part = 0; part < kv::kComponents; ++part) {
    if (!taken[part])
      continue;
    copy->slots[part] = std::move(taken[part]);
    copy->presences[part] = KvDiskSlot::Presence::Writing;
  }
  return copy;
}

template <class Pages>
std::unique_ptr<KvTransfer> BasicKvPageTier<Pages>::demote(std::span<const uint32_t> pages,
                                                           const std::shared_ptr<KvDiskSlot> &copy,
                                                           kv::Components parts, WriteClass write,
                                                           std::function<void()> completion) {
  if (!canDemote(write))
    return {};
  if (pages_.form(pages.front()) == kv::PageForm::Index)
    throw std::logic_error("only a page that holds its K/V is written to the KV tier");
  const std::shared_ptr<DiskSlot> disk = diskSlot(copy);
  requireParts(*disk, parts, KvDiskSlot::Presence::Writing);
  if (std::ranges::any_of(writes_, [&](const Write &other) { return other.copy == disk && other.parts & parts; }))
    throw std::logic_error("a KV copy's part is written twice at once");
  // Whatever the write allocates comes before its first IO starts, so that
  // an allocation that fails leaves no IO the tier does not track.
  Write entry{write, disk, parts, {}, std::make_shared<Outcome>()};
  auto ticket = std::make_unique<Ticket>(entry.outcome);
  std::array<std::vector<std::span<const std::byte>>, kv::kComponents> spans;
  for (const kv::Component part : kv::kAllComponents)
    if (parts & kv::componentBit(part))
      spans[static_cast<size_t>(part)] = format_.transferSpans(pages_, pages, part);
  // A file that stopped taking writes since canDemote() refuses its part as
  // canDemote() does from now on, and so does one whose submission threw
  // once another part started: the write is refused while nothing of it
  // started, and fails once a part did, which keeps its pages still until
  // that part ends.
  for (const kv::Component part : kv::kAllComponents) {
    if (!(parts & kv::componentBit(part)))
      continue;
    const auto index = static_cast<size_t>(part);
    const bool started = std::ranges::any_of(entry.io, [](const auto &io) { return io != nullptr; });
    try {
      entry.io[index] = file(part).write(disk->slots[index]->slot, std::move(spans[index]), completion);
    } catch (...) {
      if (!started) throw;
    }
    if (!entry.io[index] && !started)
      return {};
  }
  ++classWrites_[static_cast<size_t>(write)];
  writes_.push_back(std::move(entry));
  return ticket;
}

template <class Pages>
std::unique_ptr<KvTransfer> BasicKvPageTier<Pages>::restore(const std::shared_ptr<KvDiskSlot> &copy,
                                                            kv::Components parts, std::span<const uint32_t> pages,
                                                            std::function<void()> completion) {
  if (!canRestore())
    return {};
  const std::shared_ptr<DiskSlot> disk = diskSlot(copy);
  requireParts(*disk, parts, KvDiskSlot::Presence::Landed);
  Restore restore{{}, {pages.begin(), pages.end()}, pages_.form(pages.front()), std::move(completion),
                  std::make_shared<Outcome>()};
  for (const kv::Component part : kv::kAllComponents)
    if (parts & kv::componentBit(part))
      restore.parts[static_cast<size_t>(part)] = disk->slots[static_cast<size_t>(part)];
  auto ticket = std::make_unique<Ticket>(restore.outcome);
  queued_.push_back(std::move(restore));
  ++restores_;
  return ticket;
}

template <class Pages>
void BasicKvPageTier<Pages>::giveUp(const std::shared_ptr<KvDiskSlot> &copy, kv::Components parts) {
  const std::shared_ptr<DiskSlot> disk = diskSlot(copy);
  if (!parts || parts & ~disk->parts() ||
      std::ranges::any_of(writes_, [&](const Write &other) { return other.copy == disk && other.parts & parts; }))
    throw std::logic_error("a KV copy gives up a part it lacks or writes");
  for (const kv::Component part : kv::kAllComponents)
    if (parts & kv::componentBit(part))
      disk->slots[static_cast<size_t>(part)].reset();
}

template <class Pages> void BasicKvPageTier<Pages>::submitRestores() {
  // Index restores read a few KiB each, on every reader; a whole restore
  // reads a slot, so fewer of them run at once and their memory stays
  // within kWholeRestoreReaders slots.
  for (const bool index : {true, false}) {
    const auto first = std::stable_partition(queued_.begin(), queued_.end(), [&](const Restore &restore) {
      return (restore.form == kv::PageForm::Index) != index;
    });
    const auto count = static_cast<size_t>(queued_.end() - first);
    if (!count)
      continue;
    // A batch that does not start leaves its restores queued for the next
    // call; one that starts is tracked without another allocation.
    batches_.reserve(batches_.size() + 1);
    RestoreBatch batch{std::make_shared<std::vector<Restore>>(), nullptr, std::vector<bool>(count), count};
    batch.restores->reserve(count);
    std::move(first, queued_.end(), std::back_inserter(*batch.restores));
    try {
      // A failed read throws, and the readers try it once more. The batch
      // wakes the engine once every restore in it has ended.
      batch.batch = readers_->start(
          count, [this, restores = batch.restores](size_t task) { read((*restores)[task]); },
          index ? readers_->readers() : std::min(kWholeRestoreReaders, readers_->readers()),
          [restores = batch.restores](const model::SsdReaders::Batch &) {
            if (const std::function<void()> &completion = restores->front().completion)
              completion();
          });
    } catch (...) {
      std::move(batch.restores->begin(), batch.restores->end(), first);
      throw;
    }
    queued_.erase(first, queued_.end());
    batches_.push_back(std::move(batch));
  }
}

template <class Pages> void BasicKvPageTier<Pages>::read(const Restore &restore) const {
  // The extents of the pages a restore fills stay while it is in flight,
  // and their memory is found beside the engine thread's changes to other
  // extents. A window part's read also finds the spans of its block's base
  // page, which it leaves, and that page's extent may go meanwhile: the
  // pages find them under the lock their extents' changes take
  // (kv::PageStorage::blockSpans), and the read fails if it went.
  // An index, a few KiB that a restored spilled chain reads by the
  // thousand, goes through the reader's own buffer; a whole part brings its
  // own, so that no reader keeps a slot's worth of memory.
  const bool index = restore.form == kv::PageForm::Index;
  model::KvReadBuffer buffer;
  for (const kv::Component part : kv::kAllComponents) {
    const std::shared_ptr<const PartSlot> &copy = restore.parts[static_cast<size_t>(part)];
    if (!copy)
      continue;
    const uint64_t payload = format_.payloadBytes(part);
    // A copy an earlier process wrote reads the whole segments its range
    // touches (model::SlotFile::readAt): a slot's worth holds them.
    if (!index && copy->file->persistent())
      static_cast<void>(buffer.hold(model::SlotFile::slotBytesFor(payload)));
    const std::optional<std::span<const std::byte>> bytes =
        index ? model::readKvCopy(*copy, format_.indexRange()) : model::readKvCopy(*copy, {0, payload}, buffer);
    if (!bytes)
      throw std::runtime_error("a KV tier slot could not be read");
    if (index)
      format_.scatterIndex(*bytes, pages_.indexSpans(restore.pages.front()));
    else
      format_.scatterPage(*bytes, pages_, restore.pages, part);
  }
}

template <class Pages> void BasicKvPageTier<Pages>::poll() {
  // A block's write ends once each of its parts' IO has: each part that
  // landed is present, and the write succeeds once every one did.
  for (size_t index = 0; index < writes_.size();) {
    Write &write = writes_[index];
    if (std::ranges::any_of(write.io, [](const auto &io) { return io && !io->ready(); })) {
      ++index;
      continue;
    }
    --classWrites_[static_cast<size_t>(write.write)];
    bool success = true;
    for (const kv::Component part : kv::kAllComponents) {
      const auto at = static_cast<size_t>(part);
      if (!(write.parts & kv::componentBit(part)))
        continue;
      const bool landed = write.io[at] && write.io[at]->wait();
      if (landed && write.copy->slots[at])
        write.copy->presences[at] = KvDiskSlot::Presence::Landed;
      success = success && landed;
    }
    write.outcome->success = success;
    write.outcome->ready = true;
    writes_.erase(writes_.begin() + static_cast<std::ptrdiff_t>(index));
  }
  // A restore that ended lets its copy go: the cache may give the copy up
  // once the restore has landed.
  for (auto batch = batches_.begin(); batch != batches_.end();) {
    for (size_t index = 0; index < batch->retired.size(); ++index) {
      if (batch->retired[index] || !batch->batch->ended(index))
        continue;
      Restore &restore = (*batch->restores)[index];
      restore.outcome->success = !batch->batch->failed(index);
      restore.outcome->ready = true;
      restore.parts = {};
      batch->retired[index] = true;
      --batch->left;
      --restores_;
    }
    if (batch->left) {
      ++batch;
      continue;
    }
    // The reader that ended the last restore runs the batch's end once that
    // restore shows ended: the batch goes when its end has returned, so no
    // completion runs past the tier.
    readers_->stop(*batch->batch);
    batch = batches_.erase(batch);
  }
}

} // namespace splash::engine
