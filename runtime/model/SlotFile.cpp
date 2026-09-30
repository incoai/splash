#include "model/SlotFile.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace splash::model {

std::optional<DiskReservation>
DiskReservation::acquire(std::shared_ptr<DiskBudget> budget, uint64_t bytes) {
  if (!budget || !budget->reserve(bytes))
    return std::nullopt;
  return DiskReservation(std::move(budget), bytes);
}
DiskReservation::DiskReservation(DiskReservation &&other) noexcept
    : budget_(std::move(other.budget_)),
      bytes_(std::exchange(other.bytes_, 0)) {}
DiskReservation &DiskReservation::operator=(DiskReservation &&other) noexcept {
  if (this != &other) {
    if (budget_)
      budget_->release(bytes_);
    budget_ = std::move(other.budget_);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}
DiskReservation::~DiskReservation() {
  if (budget_)
    budget_->release(bytes_);
}
void DiskReservation::consume(uint64_t bytes) {
  if (bytes > bytes_)
    throw std::logic_error("slot exceeds disk reservation");
  bytes_ -= bytes;
}

struct SlotFile::Backing {
  int descriptor = -1;
  std::filesystem::path directory;
  std::shared_ptr<CacheStore> store;
  std::unordered_map<uint64_t, std::weak_ptr<Slot>> reopened;
  uint64_t slotBytes = 0;
  std::shared_ptr<DiskBudget> budget;
  uint32_t allocated = 0;
  std::atomic<bool> failed{false};
  std::mutex mutex;
  std::vector<uint32_t> free;
  ~Backing() { if (descriptor >= 0) ::close(descriptor); }
};

SlotFile::Slot::Slot(std::shared_ptr<Backing> backing, uint32_t index)
    : backing_(std::move(backing)), index_(index) {}
uint64_t SlotFile::Slot::bytes() const noexcept { return backing_->slotBytes; }

bool SlotFile::Slot::durable() const noexcept {
  return budget_ != backing_->budget;
}
void SlotFile::Slot::transfer(DiskReservation &reservation) {
  if (!backing_->store || !reusable_ ||
      reservation.budget() == backing_->budget)
    throw std::logic_error("invalid durable slot transfer");
  if (budget_ == reservation.budget())
    return;
  reservation.consume(bytes());
  budget_->release(bytes());
  budget_ = reservation.budget();
}
bool SlotFile::Slot::returnTemporary() {
  if (!reusable_)
    return false;
  if (!durable())
    return true;
  if (!backing_->budget->reserve(bytes()))
    return false;
  budget_->release(bytes());
  budget_ = backing_->budget;
  return true;
}

std::shared_ptr<SlotFile> SlotFile::sibling(uint64_t bytes) const {
  return std::make_shared<SlotFile>(bytes, backing_->budget,
                                    backing_->directory, backing_->store);
}

SlotFile::Slot::~Slot() {
  if (!backing_) return;
  if (backing_->store)
    backing_->store->retire(recordId_);
  else {
    std::lock_guard lock(backing_->mutex);
    backing_->free.push_back(index_);
  }
  budget_->release(backing_->slotBytes);
}

bool SlotFile::Operation::ready() const noexcept {
  return done_.load(std::memory_order_acquire);
}
bool SlotFile::Operation::wait() {
  std::unique_lock lock(mutex_);
  wake_.wait(lock, [&] { return ready(); });
  return success_;
}

SlotFile::SlotFile(uint64_t slotBytes, uint64_t capacityBytes,
                   const std::filesystem::path &directory)
    : SlotFile(slotBytes, std::make_shared<DiskBudget>(capacityBytes), directory) {}

SlotFile::SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget,
                   const std::filesystem::path &directory,
                   std::shared_ptr<CacheStore> store)
    : backing_(std::make_shared<Backing>()) {
  if (!budget)
    throw std::invalid_argument("slot file needs a disk budget");
  const uint64_t capacityBytes = budget->capacityBytes();
  if (!slotBytes || capacityBytes > uint64_t{std::numeric_limits<off_t>::max()} ||
      capacityBytes / slotBytes > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("invalid slot file capacity");
  if (!store && capacityBytes < slotBytes)
    throw std::invalid_argument("slot file quota holds no slot");
  if (slotBytes % kAlignmentBytes)
    throw std::invalid_argument("slot size is not aligned for uncached IO");
  backing_->directory = directory;
  backing_->slotBytes = slotBytes;
  backing_->budget = std::move(budget);
  backing_->store = std::move(store);
  if (backing_->store) {
    worker_ = std::thread([this] { run(); });
    return;
  }
  // A write past the file-size limit (ulimit -f, a launchd FileSize) raises
  // SIGXFSZ, whose default action kills the process. Ignored, the write fails
  // with EFBIG and stops the file like any other storage error. The change is
  // process-wide, which is safe because every other engine writer checks its
  // errors; a handler someone installed is left alone.
  struct sigaction fileSize {};
  if (::sigaction(SIGXFSZ, nullptr, &fileSize) == 0 && fileSize.sa_handler == SIG_DFL)
    std::signal(SIGXFSZ, SIG_IGN);
  std::string name = (directory / "splash-cache-XXXXXX").string();
  backing_->descriptor = ::mkstemp(name.data());
  if (backing_->descriptor < 0)
    throw std::system_error(errno, std::generic_category(), "create slot file");
  const int unlinked = ::unlink(name.c_str());
  if (unlinked < 0)
    throw std::system_error(errno, std::generic_category(), "unlink slot file");
  if (::fcntl(backing_->descriptor, F_SETFD, FD_CLOEXEC) < 0)
    throw std::system_error(errno, std::generic_category(), "close-on-exec slot file");
  // Avoid turning the cold tier into another long-lived RAM copy. Host VM
  // pressure accounting still covers any transient kernel IO memory.
  if (::fcntl(backing_->descriptor, F_NOCACHE, 1) < 0)
    throw std::system_error(errno, std::generic_category(), "uncached slot file");
  worker_ = std::thread([this] { run(); });
}

SlotFile::~SlotFile() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
    for (auto &work : work_) work.operation->cancel();
  }
  wake_.notify_one();
  worker_.join();
}

std::shared_ptr<SlotFile::Slot> SlotFile::acquire() {
  auto reservation =
      DiskReservation::acquire(backing_->budget, backing_->slotBytes);
  return reservation ? allocate(*reservation) : nullptr;
}
std::shared_ptr<SlotFile::Slot>
SlotFile::acquire(DiskReservation &reservation) {
  if (!backing_->store)
    throw std::logic_error("durable slots require a cache store");
  return allocate(reservation);
}
std::shared_ptr<SlotFile::Slot>
SlotFile::allocate(DiskReservation &reservation) {
  auto slot = std::shared_ptr<Slot>(new Slot({}, 0));
  std::lock_guard lock(backing_->mutex);
  // Prepare the free list before transferring the charge, so returning a
  // scratch slot never allocates and allocation failure cannot strand quota.
  if (!backing_->store && backing_->free.empty() &&
      backing_->free.capacity() == backing_->allocated)
    backing_->free.reserve(std::max<size_t>(1, 2 * backing_->free.capacity()));
  reservation.consume(backing_->slotBytes);
  slot->backing_ = backing_;
  slot->budget_ = reservation.budget();
  if (backing_->store) {
    slot->recordId_ = backing_->store->allocate();
  } else if (!backing_->free.empty()) {
    slot->index_ = backing_->free.back();
    backing_->free.pop_back();
  } else {
    slot->index_ = backing_->allocated++;
  }
  return slot;
}

std::shared_ptr<SlotFile::Slot> SlotFile::reopen(uint64_t id,
                                                 DiskReservation &reservation) {
  if (!backing_->store || !id)
    throw std::invalid_argument("invalid durable slot");
  if (auto found = backing_->reopened.find(id);
      found != backing_->reopened.end())
    if (auto slot = found->second.lock())
      return slot;
  auto slot = std::shared_ptr<Slot>(new Slot({}, 0));
  reservation.consume(backing_->slotBytes);
  slot->backing_ = backing_;
  slot->budget_ = reservation.budget();
  slot->recordId_ = id;
  slot->written_ = true;
  backing_->reopened[id] = slot;
  return slot;
}

std::shared_ptr<SlotFile::Operation>
SlotFile::metadata(std::function<void()> work,
                   std::function<void()> completion) {
  return submit(
      [work = std::move(work)](const std::atomic<bool> &cancelled) {
        if (cancelled.load())
          return false;
        work();
        return true;
      },
      std::move(completion));
}

uint64_t SlotFile::slotBytes() const noexcept { return backing_->slotBytes; }

uint64_t SlotFile::capacityBytes() const noexcept {
  return backing_->budget->capacityBytes();
}

uint64_t SlotFile::usedBytes() const noexcept { return backing_->budget->usedBytes(); }
uint64_t SlotFile::readBytes() const noexcept { return backing_->budget->readBytes(); }
uint64_t SlotFile::writtenBytes() const noexcept { return backing_->budget->writtenBytes(); }

bool SlotFile::writable() const noexcept {
  return !backing_->failed.load(std::memory_order_relaxed);
}

bool SlotFile::idle() const {
  std::lock_guard lock(mutex_);
  return work_.empty() && !running_;
}

std::shared_ptr<SlotFile::Operation> SlotFile::submit(
    std::function<bool(const std::atomic<bool> &)> run,
    std::function<void()> completion) {
  auto operation = std::make_shared<Operation>();
  {
    std::lock_guard lock(mutex_);
    if (stopping_)
      throw std::logic_error("slot file is shutting down");
    work_.push_back({operation, std::move(run), std::move(completion)});
  }
  wake_.notify_one();
  return operation;
}

void SlotFile::run() {
  for (;;) {
    Work work;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || !work_.empty(); });
      if (work_.empty()) return;
      work = std::move(work_.front());
      work_.pop_front();
      running_ = true;
    }
    bool success = false;
    try { success = work.run(work.operation->cancelled_); }
    catch (...) { success = false; }
    work.run = {};
    // Idle before the operation reports, so whoever wakes on it sees a worker
    // that has let go of the memory it moved.
    {
      std::lock_guard lock(mutex_);
      running_ = false;
    }
    {
      std::lock_guard lock(work.operation->mutex_);
      work.operation->success_ = success;
      work.operation->done_.store(true, std::memory_order_release);
    }
    work.operation->wake_.notify_all();
    // A completion is a wake-up, not part of the transfer: one that throws
    // must not take the worker with it.
    if (work.completion) {
      try { work.completion(); } catch (...) {}
    }
  }
}

namespace {
constexpr size_t kChunkBytes = 1 << 20;

template <typename Span>
uint64_t totalBytes(const std::vector<Span> &spans) {
  uint64_t total = 0;
  for (auto span : spans) {
    if (span.size() > std::numeric_limits<uint64_t>::max() - total)
      throw std::invalid_argument("slot transfer size overflowed");
    total += span.size();
  }
  return total;
}

// Moves the spans in order through io(data, bytes, offset), one chunk at a
// time so that cancellation and short transfers are noticed promptly.
template <typename Span, typename Io>
bool transfer(const std::vector<Span> &spans, off_t offset,
              const std::atomic<bool> &cancelled, Io io) {
  for (auto span : spans) {
    while (!span.empty()) {
      if (cancelled.load(std::memory_order_relaxed)) return false;
      const ssize_t count = io(span.data(), std::min(span.size(), kChunkBytes), offset);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return false;
      span = span.subspan(static_cast<size_t>(count));
      offset += count;
    }
  }
  return true;
}

off_t slotOffset(uint64_t index, uint64_t slotBytes) {
  return static_cast<off_t>(index * slotBytes);
}
} // namespace

std::shared_ptr<SlotFile::Operation> SlotFile::write(
    std::shared_ptr<Slot> slot, std::vector<std::span<const std::byte>> source,
    std::function<void()> completion) {
  if (!slot || slot->backing_ != backing_ || totalBytes(source) != backing_->slotBytes)
    throw std::invalid_argument("slot write does not match this file's slots");
  if (!writable())
    throw std::logic_error("slot file no longer takes writes");
  return submit(
      [slot, budget = slot->budget_,
       source = std::move(source)](const std::atomic<bool> &cancelled) {
        Backing &backing = *slot->backing_;
        slot->written_ = false;
        if (backing.store) {
          try {
            slot->written_ =
                backing.store->write(slot->recordId_, source, cancelled);
          } catch (...) {
            backing.failed.store(true);
            throw;
          }
          if (slot->written_)
            budget->written_.fetch_add(backing.slotBytes);
          else if (!cancelled.load())
            backing.failed.store(true);
          return slot->written_;
        }
        if (transfer(source, slotOffset(slot->index_, backing.slotBytes),
                     cancelled,
                     [&backing, &budget](const std::byte *data, size_t bytes,
                                         off_t offset) {
                       const auto count =
                           ::pwrite(backing.descriptor, data, bytes, offset);
                       if (count > 0)
                         budget->written_.fetch_add(count,
                                                    std::memory_order_relaxed);
                       return count;
                     })) {
          slot->written_ = true;
          return true;
        }
        if (!cancelled.load(std::memory_order_relaxed))
          backing.failed.store(true, std::memory_order_relaxed);
        return false;
      },
      std::move(completion));
}

std::shared_ptr<SlotFile::Operation> SlotFile::read(
    std::shared_ptr<Slot> slot, std::vector<std::span<std::byte>> destination,
    std::function<void()> completion) {
  if (!slot || slot->backing_ != backing_ || totalBytes(destination) != backing_->slotBytes)
    throw std::invalid_argument("slot read does not match this file's slots");
  return submit(
      [slot, budget = slot->budget_, destination = std::move(destination)](
          const std::atomic<bool> &cancelled) {
        if (!slot->written_)
          return false;
        Backing &backing = *slot->backing_;
        if (backing.store) {
          const bool result =
              backing.store->read(slot->recordId_, destination, cancelled);
          if (result)
            budget->read_.fetch_add(backing.slotBytes);
          return result;
        }
        return transfer(
            destination, slotOffset(slot->index_, slot->backing_->slotBytes),
            cancelled,
            [&backing, &budget](std::byte *data, size_t bytes, off_t offset) {
              const auto count =
                  ::pread(backing.descriptor, data, bytes, offset);
              if (count > 0)
                budget->read_.fetch_add(count, std::memory_order_relaxed);
              return count;
            });
      },
      std::move(completion));
}

} // namespace splash::model
