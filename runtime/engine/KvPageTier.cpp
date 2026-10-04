#include "engine/KvPageTier.hpp"
#include "TestConfig.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace splash::engine {

// A demotion or a restore of one page, until the tier retires its IO.
struct KvPageTier::Transfer final {
  bool demotion = false;
  std::shared_ptr<model::SlotFile::Operation> io;
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
std::shared_ptr<KvPageTier::DiskSlot> diskSlot(const std::shared_ptr<KvDiskSlot> &slot) {
  auto disk = std::dynamic_pointer_cast<KvPageTier::DiskSlot>(slot);
  if (!disk) throw std::invalid_argument("KV disk slot belongs to another tier");
  return disk;
}
} // namespace

KvPageTier::KvPageTier(kv::PageStorage &pages, std::shared_ptr<model::SlotFile> file)
    : pages_(pages), file_(std::move(file)),
      transferLimit_(testConfig().kvTierTransfers.value_or(kTransfers)) {
  if (!file_)
    throw std::invalid_argument("KV tier needs a slot file");
  demotionLimit_ = std::max<uint32_t>(1, transferLimit_ / 2);
  restoreLimit_ = std::max<uint32_t>(1, transferLimit_ - transferLimit_ / 4);
  // Tracking a submitted IO must not allocate: the tier drains every IO it
  // tracks before it goes, and an IO it lost would outlive the page.
  inFlight_.reserve(transferLimit_);
}

// A transfer in flight reads or writes its page in place, and the cache that
// held the page for it is gone before the tier is: the IO worker must be done
// with every page first. Nothing will read the slots a cancelled write leaves
// behind, because the tier that maps pages to them is going away too.
KvPageTier::~KvPageTier() {
  for (const std::shared_ptr<Transfer> &transfer : inFlight_)
    transfer->io->drain();
}

uint64_t KvPageTier::slotBytes() const noexcept { return file_->slotBytes(); }

bool KvPageTier::writable() const noexcept { return file_->writable(); }

bool KvPageTier::persistent() const noexcept { return file_->persistent(); }

void KvPageTier::label(const std::shared_ptr<KvDiskSlot> &slot, std::vector<std::byte> label) {
  file_->label(diskSlot(slot)->slot, std::move(label));
}

std::shared_ptr<KvDiskSlot> KvPageTier::adopt(const model::SlotRecord &record) {
  return std::make_shared<DiskSlot>(file_->adopt(record));
}

bool KvPageTier::canDemote() const noexcept {
  return demotions_ < demotionLimit_ && inFlight_.size() < transferLimit_ && writable();
}

bool KvPageTier::canRestore() const noexcept {
  return restores_ < restoreLimit_ && inFlight_.size() < transferLimit_;
}

std::shared_ptr<KvDiskSlot> KvPageTier::acquireSlot() {
  auto slot = file_->acquire();
  if (!slot) return {};
  return std::make_shared<DiskSlot>(std::move(slot));
}

std::unique_ptr<KvTransfer> KvPageTier::demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot,
                                               std::function<void()> completion) {
  if (!canDemote())
    return {};
  auto disk = diskSlot(slot);
  const std::vector<std::span<std::byte>> bytes = pages_.spans(page);
  auto transfer = std::make_shared<Transfer>();
  transfer->demotion = true;
  auto ticket = std::make_unique<Ticket>(transfer);
  transfer->io = file_->write(disk->slot, {bytes.begin(), bytes.end()}, std::move(completion));
  // A write that failed since canDemote() has closed the file, which
  // refuses this one as canDemote() does from now on.
  if (!transfer->io)
    return {};
  ++demotions_;
  inFlight_.push_back(std::move(transfer));
  return ticket;
}

std::unique_ptr<KvTransfer> KvPageTier::restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
                                                std::function<void()> completion) {
  if (!canRestore())
    return {};
  auto disk = diskSlot(slot);
  auto transfer = std::make_shared<Transfer>();
  auto ticket = std::make_unique<Ticket>(transfer);
  transfer->io = file_->read(disk->slot, pages_.spans(page), std::move(completion));
  ++restores_;
  inFlight_.push_back(std::move(transfer));
  return ticket;
}

void KvPageTier::poll() {
  for (size_t index = 0; index < inFlight_.size();) {
    if (!inFlight_[index]->io->ready()) {
      ++index;
      continue;
    }
    const std::shared_ptr<Transfer> transfer = std::move(inFlight_[index]);
    inFlight_.erase(inFlight_.begin() + static_cast<std::ptrdiff_t>(index));
    --(transfer->demotion ? demotions_ : restores_);
    transfer->success = transfer->io->wait();
    transfer->io.reset();
    transfer->ready = true;
  }
}

} // namespace splash::engine
