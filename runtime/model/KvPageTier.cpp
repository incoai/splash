#include "model/KvPageTier.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace splash::model {

// A demotion copies to staging, then writes; a restore reads, then copies to
// the page. Either way the transfer owns its staging slot until it is ready.
struct KvPageTier::Transfer final {
  bool toStaging = false;
  uint32_t page = 0;
  uint32_t stagingSlot = 0;
  std::shared_ptr<DiskSlot> disk;
  std::function<void()> completion;
  std::shared_ptr<SlotFile::Operation> io;
  bool ready = false;
  bool success = false;
};

class KvPageTier::Ticket final : public KvTransfer {
public:
  explicit Ticket(std::shared_ptr<Transfer> transfer) : transfer_(std::move(transfer)) {}
  bool ready() const noexcept override { return transfer_->ready; }
  bool finish() override { return transfer_->success; }

private:
  std::shared_ptr<Transfer> transfer_;
};

namespace {
std::shared_ptr<std::byte> allocateStaging(uint64_t bytes) {
  void *memory = nullptr;
  if (::posix_memalign(&memory, SlotFile::kAlignmentBytes, bytes) != 0)
    throw std::bad_alloc();
  // Touch the pages now rather than under the first copy.
  std::memset(memory, 0, bytes);
  return std::shared_ptr<std::byte>(static_cast<std::byte *>(memory),
                                    [](std::byte *pointer) { std::free(pointer); });
}

std::shared_ptr<KvPageTier::DiskSlot> diskSlot(const std::shared_ptr<KvDiskSlot> &slot) {
  auto disk = std::dynamic_pointer_cast<KvPageTier::DiskSlot>(slot);
  if (!disk) throw std::invalid_argument("KV disk slot belongs to another tier");
  return disk;
}

uint64_t alignedBytes(uint64_t bytes) noexcept {
  const uint64_t unit = SlotFile::kAlignmentBytes;
  return (bytes + unit - 1) / unit * unit;
}
} // namespace

uint64_t KvPageTier::slotBytesFor(const kv::PageStorage &pages) noexcept {
  return alignedBytes(pages.bytesPerPage());
}

// The ring as allocated, and the small table rounded up to a whole 16 KiB
// page as a bound on its allocation.
uint64_t KvPageTier::stagingBytesFor(kv::Layout layout, uint32_t stagingSlots) noexcept {
  return uint64_t{stagingSlots} * alignedBytes(layout.bytesPerModelPage()) +
         alignedBytes(ops::KvCopy::tableBytes(stagingSlots));
}

KvPageTier::KvPageTier(metal::MetalBackend &backend, kv::PageStorage &pages,
                       std::shared_ptr<SlotFile> file, uint32_t stagingSlots)
    : backend_(backend), pages_(pages), file_(std::move(file)),
      slotBytes_(slotBytesFor(pages)), stagingSlots_(stagingSlots) {
  if (!file_ || file_->slotBytes() != slotBytes_)
    throw std::invalid_argument("KV slot file does not match the page size");
  if (!stagingSlots_)
    throw std::invalid_argument("KV staging needs at least one slot");
  demotionSlots_ = std::max<uint32_t>(1, stagingSlots_ / 2);
  restoreSlots_ = std::max<uint32_t>(1, stagingSlots_ - stagingSlots_ / 4);
  // Registering a submitted IO must not allocate: staging owns its memory
  // until the operation is tracked and drained.
  io_.reserve(stagingSlots_);
  const uint64_t bytes = uint64_t{stagingSlots_} * slotBytes_;
  const uint64_t before = backend_.memoryStats().allocatedBytes;
  memory_ = allocateStaging(bytes);
  staging_ = backend_.wrapSharedMemory(memory_.get(), bytes, memory_, "kv-staging");
  table_ = backend_.allocateBuffer(ops::KvCopy::tableBytes(stagingSlots_),
                                   metal::BufferStorage::Shared, "kv-copy-table");
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend_.memoryStats().allocatedBytes);
  for (uint32_t slot = 0; slot < stagingSlots_; ++slot)
    setTable(slot, 0, ops::KvCopy::Direction::None);
  freeStaging_.reserve(stagingSlots_);
  for (uint32_t slot = stagingSlots_; slot-- > 0;) freeStaging_.push_back(slot);
}

// Staging is the source of every write and the destination of every read,
// and it dies with the tier: the IO worker must be done with it first, the
// way a state write waits for its own staging. Nothing will read the slots a
// cancelled write leaves behind, because the tier that maps pages to them is
// going away too.
KvPageTier::~KvPageTier() {
  for (const std::shared_ptr<Transfer> &transfer : io_) {
    transfer->io->cancel();
    static_cast<void>(transfer->io->wait());
  }
}

uint64_t KvPageTier::slotBytes() const noexcept { return slotBytes_; }

uint64_t KvPageTier::capacityBytes() const noexcept { return file_->capacityBytes(); }

uint64_t KvPageTier::usedBytes() const noexcept { return file_->usedBytes(); }

bool KvPageTier::writable() const noexcept { return file_->writable(); }

bool KvPageTier::canDemote() const noexcept {
  return demotionsInFlight_ < demotionSlots_ && !freeStaging_.empty() && writable();
}

std::shared_ptr<KvDiskSlot> KvPageTier::acquireSlot() {
  auto slot = file_->acquire();
  if (!slot) return {};
  return std::make_shared<DiskSlot>(std::move(slot));
}

std::span<std::byte> KvPageTier::staging(uint32_t slot) noexcept {
  return {memory_.get() + uint64_t{slot} * slotBytes_, slotBytes_};
}

void KvPageTier::setTable(uint32_t slot, uint32_t page,
                          ops::KvCopy::Direction direction) noexcept {
  ops::KvCopy::setEntry(table_, slot, page, direction);
}

std::unique_ptr<KvTransfer> KvPageTier::demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot,
                                               std::function<void()> completion) {
  auto disk = diskSlot(slot);
  if (page >= pages_.pageCount()) throw std::invalid_argument("invalid KV page");
  if (!canDemote())
    return {};
  auto transfer = std::make_shared<Transfer>();
  transfer->toStaging = true;
  transfer->page = page;
  transfer->stagingSlot = freeStaging_.back();
  transfer->disk = std::move(disk);
  transfer->completion = std::move(completion);
  auto ticket = std::make_unique<Ticket>(transfer);
  queued_.push_back(transfer);
  freeStaging_.pop_back();
  ++demotionsInFlight_;
  return ticket;
}

std::unique_ptr<KvTransfer> KvPageTier::restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
                                                std::function<void()> completion) {
  auto disk = diskSlot(slot);
  if (page >= pages_.pageCount()) throw std::invalid_argument("invalid KV page");
  if (restoresInFlight_ >= restoreSlots_ || freeStaging_.empty()) return {};
  auto transfer = std::make_shared<Transfer>();
  transfer->page = page;
  transfer->stagingSlot = freeStaging_.back();
  transfer->disk = std::move(disk);
  transfer->completion = std::move(completion);
  auto ticket = std::make_unique<Ticket>(transfer);
  transfer->io = file_->read(transfer->disk->slot, {staging(transfer->stagingSlot)},
                             transfer->completion);
  freeStaging_.pop_back();
  ++restoresInFlight_;
  io_.push_back(transfer);
  return ticket;
}

bool KvPageTier::copiesQueued() const noexcept { return !queued_.empty(); }

std::function<void()> KvPageTier::encode(metal::CommandGraph &graph) {
  if (queued_.empty()) return {};
  // Earlier batches' commands have finished even when poll() has not seen
  // their report yet; this command must not run their copies again.
  for (const Batch &batch : inFlight_) {
    for (const auto &transfer : batch.copies)
      setTable(transfer->stagingSlot, 0, ops::KvCopy::Direction::None);
  }
  for (auto &transfer : queued_) {
    setTable(transfer->stagingSlot, transfer->page,
             transfer->toStaging ? ops::KvCopy::Direction::ToStaging
                                 : ops::KvCopy::Direction::ToPage);
  }
  ops::KvCopy::addPages(graph, pages_, staging_, table_, stagingSlots_, slotBytes_);
  inFlight_.push_back(Batch{++encodedBatches_, std::move(queued_)});
  queued_.clear();
  return [completed = completedBatch_, batch = encodedBatches_] {
    completed->store(batch, std::memory_order_release);
  };
}

void KvPageTier::finish(Transfer &transfer, bool success) noexcept {
  transfer.success = success;
  freeStaging_.push_back(transfer.stagingSlot);
  --(transfer.toStaging ? demotionsInFlight_ : restoresInFlight_);
  // A finished transfer holds nothing: the slot is the caller's to keep.
  transfer.disk.reset();
  transfer.io.reset();
  transfer.completion = nullptr;
  transfer.ready = true;
}

void KvPageTier::poll() {
  const uint64_t completed = completedBatch_->load(std::memory_order_acquire);
  while (!inFlight_.empty() && inFlight_.front().number <= completed) {
    Batch batch = std::move(inFlight_.front());
    inFlight_.pop_front();
    for (auto &transfer : batch.copies) {
      setTable(transfer->stagingSlot, 0, ops::KvCopy::Direction::None);
      if (!transfer->toStaging) {
        finish(*transfer, true);
        continue;
      }
      try {
        transfer->io = file_->write(transfer->disk->slot,
                                    {std::span<const std::byte>(staging(transfer->stagingSlot))},
                                    transfer->completion);
        io_.push_back(transfer);
      } catch (const std::exception &) {
        finish(*transfer, false);
      }
    }
  }
  for (size_t index = 0; index < io_.size();) {
    if (!io_[index]->io->ready()) {
      ++index;
      continue;
    }
    std::shared_ptr<Transfer> transfer = std::move(io_[index]);
    io_.erase(io_.begin() + static_cast<std::ptrdiff_t>(index));
    const bool success = transfer->io->wait();
    if (!transfer->toStaging && success) {
      // The disk copy is in staging; the next command moves it into the page.
      queued_.push_back(std::move(transfer));
    } else {
      finish(*transfer, success);
    }
  }
}

} // namespace splash::model
