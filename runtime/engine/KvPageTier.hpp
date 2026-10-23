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

// Moves KV pages between their extents and a slot file, each page laid out
// in its slot as kv::SlotFormat says, so the tier holds no memory of its own
// beyond what reads land in. The file's IO worker writes a page straight from
// its extent (SlotFormat::transferSpans), a bounded number at once: reclaim
// may use all of them, persistent copies a quarter and write-through half, so
// that one kind never starves another and reclaim always keeps a quarter. A
// page's slot holds what the page holds as it is written: a Full page whole,
// a Slim page its first regions and its index (kv::SlotFormat::payloadBytes).
// Reads go to the SSD readers beside the worker (model::readKvCopy): a
// restore is a task of a background batch that reads what its page's form
// holds, the index alone, a Slim page's payload or all of it, into memory of
// its own and scatters it into the page. The restores asked for since the last
// submitRestores() go out together, the index restores as one batch on
// every reader and the whole restores as another on at most
// kWholeRestoreReaders, whose reads are a slot each. Copies of slots are
// read the same way by whoever holds them (copy()).
//
// The tier reaches its pages as SlotFormat's page functions do, with their
// layout() and each page's form(page): Pages is kv::PageStorage for the
// pool's extents (KvPageTier), and host memory where tests run the tier
// under the sanitizers.
template <class Pages> class BasicKvPageTier final : public KvTier {
public:
  // A slot of the tier, which is also the copy reads of spilled K/V reach
  // (model::KvCopy).
  struct DiskSlot final : KvDiskSlot, model::KvCopy {
    DiskSlot(std::shared_ptr<model::SlotFile::Slot> held,
             std::shared_ptr<const model::SlotFile> owner)
        : slot(std::move(held)), file(std::move(owner)) {}
    [[nodiscard]] uint64_t id() const noexcept override { return slot->id(); }
    [[nodiscard]] std::optional<std::span<const std::byte>>
    readAt(uint64_t offset, uint64_t bytes, std::span<std::byte> buffer) const noexcept override {
      return file->readAt(*slot, offset, bytes, buffer);
    }
    std::shared_ptr<model::SlotFile::Slot> slot;
    std::shared_ptr<const model::SlotFile> file;
  };
  // Writes in flight at once, by default, and restores at most; the readers
  // whole restores take at once (an M5 Pro reads 460 KiB pieces at 8.1 GB/s
  // from four threads, model::SsdReaders).
  static constexpr uint32_t kWrites = 64;
  static constexpr uint32_t kRestores = 4096;
  static constexpr unsigned kWholeRestoreReaders = 8;

  BasicKvPageTier(Pages &pages, kv::SlotFormat format, std::shared_ptr<model::SlotFile> file,
                  std::shared_ptr<model::SsdReaders> readers, uint32_t writes = kWrites);
  // How the tier lays a page of the layout out in its slot, as a persistent
  // cache's namespace names it (persistentCacheNamespace's kv line): empty
  // for the order of the page's spans (kv::SlotFormat version 0), which every
  // cache before block records was written in, and the block records' version
  // for a spillable layout. Whatever changes what the tier writes changes
  // this with it, so that no process adopts a slot of another layout.
  [[nodiscard]] static std::string_view slotFormatToken(const kv::Layout &layout) noexcept;
  ~BasicKvPageTier() override;
  BasicKvPageTier(const BasicKvPageTier &) = delete;
  BasicKvPageTier &operator=(const BasicKvPageTier &) = delete;

  [[nodiscard]] uint64_t slotBytes() const noexcept override { return file_->slotBytes(); }
  [[nodiscard]] bool writable() const noexcept override { return file_->writable(); }
  [[nodiscard]] bool persistent() const noexcept override { return file_->persistent(); }
  void label(const std::shared_ptr<KvDiskSlot> &slot, std::vector<std::byte> label) override;
  // Takes back a slot an earlier process recorded in a persistent tier's
  // file (model::SlotFile::adopt).
  [[nodiscard]] std::shared_ptr<KvDiskSlot> adopt(const model::SlotRecord &record);
  [[nodiscard]] bool canDemote(WriteClass write) const noexcept override;
  [[nodiscard]] bool canRestore() const noexcept override { return restores_ < kRestores; }
  [[nodiscard]] std::shared_ptr<KvDiskSlot> acquireSlot(model::DiskBudget::Draw draw) override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot, WriteClass write,
         std::function<void()> completion) override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
          std::function<void()> completion) override;
  void submitRestores() override;
  [[nodiscard]] std::shared_ptr<const model::KvCopy>
  copy(const std::shared_ptr<KvDiskSlot> &slot) const override {
    return diskSlot(slot);
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
  struct Write final {
    WriteClass write;
    std::shared_ptr<model::SlotFile::Operation> io;
    std::shared_ptr<Outcome> outcome;
  };
  // A restore: its slot, its page and the page's form, which says what it
  // reads: the page's index alone (an Index page) or the payload of what
  // the page holds.
  struct Restore final {
    std::shared_ptr<const DiskSlot> copy;
    uint32_t page = 0;
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
  // A reader's task: reads one restore's range and scatters it.
  void read(const Restore &restore) const;
  [[nodiscard]] uint32_t share(WriteClass write) const noexcept;

  Pages &pages_;
  const kv::SlotFormat format_;
  std::shared_ptr<model::SlotFile> file_;
  std::shared_ptr<model::SsdReaders> readers_;
  uint32_t writeLimit_;
  std::array<uint32_t, 3> classWrites_{};
  // Every write whose IO the worker may still run.
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
                                        std::shared_ptr<model::SlotFile> file,
                                        std::shared_ptr<model::SsdReaders> readers, uint32_t writes)
    : pages_(pages), format_(std::move(format)), file_(std::move(file)), readers_(std::move(readers)),
      writeLimit_(writes) {
  if (!file_ || !readers_ || !writeLimit_)
    throw std::invalid_argument("KV tier needs a slot file, readers and room for a write");
  if (format_.layout() != pages_.layout())
    throw std::invalid_argument("KV tier's slot format is another layout's");
  // Tracking a submitted IO must not allocate: the tier drains every IO it
  // tracks before it goes, and an IO it lost would outlive the page.
  writes_.reserve(writeLimit_);
}

// A transfer in flight reads or writes its page in place, and the cache that
// held the page for it is gone before the tier is: the worker and the
// readers must be done with every page first. Nothing will read the slots a
// cancelled write leaves behind, nor the pages an unread restore leaves,
// because the tier that maps pages to them is going away too.
template <class Pages> BasicKvPageTier<Pages>::~BasicKvPageTier() {
  for (const Write &write : writes_)
    write.io->drain();
  for (RestoreBatch &batch : batches_)
    readers_->stop(*batch.batch);
}

template <class Pages>
std::string_view BasicKvPageTier<Pages>::slotFormatToken(const kv::Layout &layout) noexcept {
  // kv::SlotFormat's version 1, the block records of a spillable layout.
  return layout.spillable() ? " slots block-records-v1" : "";
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
void BasicKvPageTier<Pages>::label(const std::shared_ptr<KvDiskSlot> &slot, std::vector<std::byte> label) {
  file_->label(diskSlot(slot)->slot, std::move(label));
}

template <class Pages>
std::shared_ptr<KvDiskSlot> BasicKvPageTier<Pages>::adopt(const model::SlotRecord &record) {
  return std::make_shared<DiskSlot>(file_->adopt(record), file_);
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
std::shared_ptr<KvDiskSlot> BasicKvPageTier<Pages>::acquireSlot(model::DiskBudget::Draw draw) {
  auto slot = file_->acquire(draw);
  if (!slot)
    return {};
  return std::make_shared<DiskSlot>(std::move(slot), file_);
}

template <class Pages>
std::unique_ptr<KvTransfer> BasicKvPageTier<Pages>::demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot,
                                                           WriteClass write, std::function<void()> completion) {
  if (!canDemote(write))
    return {};
  if (pages_.form(page) == kv::PageForm::Index)
    throw std::logic_error("only a page that holds its K/V is written to the KV tier");
  auto disk = diskSlot(slot);
  auto outcome = std::make_shared<Outcome>();
  auto ticket = std::make_unique<Ticket>(outcome);
  auto io = file_->write(disk->slot, format_.transferSpans(pages_, page), std::move(completion));
  // A file that stopped taking writes since canDemote() refuses this one as
  // canDemote() does from now on.
  if (!io)
    return {};
  ++classWrites_[static_cast<size_t>(write)];
  writes_.push_back({write, std::move(io), std::move(outcome)});
  return ticket;
}

template <class Pages>
std::unique_ptr<KvTransfer> BasicKvPageTier<Pages>::restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
                                                            std::function<void()> completion) {
  if (!canRestore())
    return {};
  Restore restore{diskSlot(slot), page, pages_.form(page), std::move(completion), std::make_shared<Outcome>()};
  auto ticket = std::make_unique<Ticket>(restore.outcome);
  queued_.push_back(std::move(restore));
  ++restores_;
  return ticket;
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
  // The page's extent stays while its restore is in flight, and its memory
  // is found beside the engine thread's changes to other extents.
  const bool index = restore.form == kv::PageForm::Index;
  const kv::SlotRange range = index ? format_.indexRange() : kv::SlotRange{0, format_.payloadBytes(restore.form)};
  model::KvReadBuffer buffer;
  const std::optional<std::span<const std::byte>> bytes = model::readKvCopy(*restore.copy, range, buffer);
  if (!bytes)
    throw std::runtime_error("a KV tier slot could not be read");
  if (index)
    format_.scatterIndex(*bytes, pages_.indexSpans(restore.page));
  else
    format_.scatterPage(*bytes, pages_, restore.page);
}

template <class Pages> void BasicKvPageTier<Pages>::poll() {
  for (size_t index = 0; index < writes_.size();) {
    Write &write = writes_[index];
    if (!write.io->ready()) {
      ++index;
      continue;
    }
    --classWrites_[static_cast<size_t>(write.write)];
    write.outcome->success = write.io->wait();
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
      restore.copy.reset();
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
