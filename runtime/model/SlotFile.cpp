#include "model/SlotFile.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace splash::model {

struct SlotFile::Backing {
  int descriptor = -1;
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
SlotFile::Slot::~Slot() {
  if (!backing_) return;
  {
    std::lock_guard lock(backing_->mutex);
    backing_->free.push_back(index_);
  }
  backing_->budget->release(backing_->slotBytes);
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
                   const std::filesystem::path &directory)
    : backing_(std::make_shared<Backing>()) {
  if (!budget)
    throw std::invalid_argument("slot file needs a disk budget");
  const uint64_t capacityBytes = budget->capacityBytes();
  if (!slotBytes || capacityBytes > uint64_t{std::numeric_limits<off_t>::max()} ||
      capacityBytes / slotBytes > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("invalid slot file capacity");
  if (capacityBytes < slotBytes)
    throw std::invalid_argument("slot file quota holds no slot");
  if (slotBytes % kAlignmentBytes)
    throw std::invalid_argument("slot size is not aligned for uncached IO");
  backing_->slotBytes = slotBytes;
  backing_->budget = std::move(budget);
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
  ::fcntl(backing_->descriptor, F_NOCACHE, 1);
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
  auto slot = std::shared_ptr<Slot>(new Slot({}, 0));
  if (!backing_->budget->reserve(backing_->slotBytes)) return {};
  std::lock_guard lock(backing_->mutex);
  uint32_t index;
  if (!backing_->free.empty()) {
    index = backing_->free.back();
    backing_->free.pop_back();
  } else {
    backing_->free.reserve(backing_->allocated + 1);
    index = backing_->allocated++;
  }
  slot->backing_ = backing_;
  slot->index_ = index;
  return slot;
}

uint64_t SlotFile::slotBytes() const noexcept { return backing_->slotBytes; }

uint64_t SlotFile::capacityBytes() const noexcept {
  return backing_->budget->capacityBytes();
}

uint64_t SlotFile::usedBytes() const noexcept { return backing_->budget->usedBytes(); }

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
      throw std::invalid_argument("state transfer size overflow");
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
    throw std::invalid_argument("state write size mismatch");
  if (!writable())
    throw std::logic_error("slot file no longer takes writes");
  return submit([slot, source = std::move(source)](const std::atomic<bool> &cancelled) {
    Backing &backing = *slot->backing_;
    slot->written_ = false;
    if (transfer(source, slotOffset(slot->index_, backing.slotBytes), cancelled,
                 [&backing](const std::byte *data, size_t bytes, off_t offset) {
                   return ::pwrite(backing.descriptor, data, bytes, offset);
                 })) {
      slot->written_ = true;
      return true;
    }
    if (!cancelled.load(std::memory_order_relaxed))
      backing.failed.store(true, std::memory_order_relaxed);
    return false;
  }, std::move(completion));
}

std::shared_ptr<SlotFile::Operation> SlotFile::read(
    std::shared_ptr<Slot> slot, std::vector<std::span<std::byte>> destination,
    std::function<void()> completion) {
  if (!slot || slot->backing_ != backing_ || totalBytes(destination) != backing_->slotBytes)
    throw std::invalid_argument("state read size mismatch");
  return submit([slot, destination = std::move(destination)](const std::atomic<bool> &cancelled) {
    if (!slot->written_) return false;
    const int descriptor = slot->backing_->descriptor;
    return transfer(destination, slotOffset(slot->index_, slot->backing_->slotBytes), cancelled,
                    [descriptor](std::byte *data, size_t bytes, off_t offset) {
                      return ::pread(descriptor, data, bytes, offset);
                    });
  }, std::move(completion));
}

} // namespace splash::model
