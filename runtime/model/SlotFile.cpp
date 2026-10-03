#include "model/SlotFile.hpp"

#include "StderrLine.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace splash::model {

struct SlotFile::Backing {
  int descriptor = -1;
  uint64_t slotBytes = 0;
  std::shared_ptr<DiskBudget> budget;
  std::atomic<bool> failed{false};
  // Guards the slots and the worker's queue.
  std::mutex mutex;
  std::condition_variable wake;
  uint32_t allocated = 0;
  std::vector<uint32_t> free;
  // Freed slots whose blocks go back before the next operation runs.
  std::vector<uint32_t> punches;
  std::deque<Work> work;
  bool stopping = false;
  // Worker-only: the slots whose blocks the file holds, from a write's first
  // chunk until a punch returns them.
  std::vector<bool> backed;
  ~Backing() { if (descriptor >= 0) ::close(descriptor); }

  // Marks a slot's blocks held; true when they were not.
  bool back(uint32_t index) {
    if (backed.size() <= index)
      backed.resize(index + 1);
    if (backed[index])
      return false;
    backed[index] = true;
    return true;
  }
};

SlotFile::Slot::Slot(std::shared_ptr<Backing> backing, uint32_t index)
    : backing_(std::move(backing)), index_(index) {}
SlotFile::Slot::~Slot() {
  if (!backing_) return;
  bool punch = false;
  {
    std::lock_guard lock(backing_->mutex);
    // The punch precedes any write of the index's next holder, which is
    // submitted after acquire() finds the index on the free list. Once the
    // file is stopping, its blocks go with it.
    punch = returnsBlocks_ && !backing_->stopping;
    if (punch)
      backing_->punches.push_back(index_);
    backing_->free.push_back(index_);
  }
  // The quota returns now; the blocks only once the worker has finished the
  // transfer it is in and punched them. A write to another file of the
  // budget can take the quota meanwhile, so the files' blocks exceed the
  // budget by this slot until the punch.
  backing_->budget->release(backing_->slotBytes);
  if (punch)
    backing_->wake.notify_one();
}

bool SlotFile::Operation::ready() const noexcept {
  return done_.load(std::memory_order_acquire);
}
bool SlotFile::Operation::wait() {
  std::unique_lock lock(mutex_);
  wake_.wait(lock, [&] { return ready(); });
  return success_;
}

namespace {
// One transfer of the worker: a whole number of host pages.
constexpr size_t kChunkBytes = 1 << 20;
static_assert(kChunkBytes % kHostPageBytes == 0);

off_t slotOffset(uint64_t index, uint64_t slotBytes) {
  return static_cast<off_t>(index * slotBytes);
}

// Once per process: the volume keeps the blocks of freed slots.
void reportPunchFailure(int error) noexcept {
  static std::atomic<bool> reported{false};
  if (!reported.exchange(true, std::memory_order_relaxed))
    writeStderrLine("Freed cache slots cannot return their blocks on this volume (" +
                    std::generic_category().message(error) +
                    "); disk use may exceed --max-cache-disk.");
}

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

// Memory the file can move straight: at least one chunk at an aligned
// address.
template <typename Span>
bool direct(Span span) noexcept {
  return span.size() >= kChunkBytes &&
         reinterpret_cast<std::uintptr_t>(span.data()) % kHostPageBytes == 0;
}

// Walks the spans of an operation in order, one piece at a time.
template <typename Span>
class Pieces final {
public:
  explicit Pieces(const std::vector<Span> &spans) : spans_(spans), left_(totalBytes(spans)) {
    settle();
  }
  [[nodiscard]] bool done() const noexcept { return !left_; }
  // The rest of the current span when it is memory the file can move
  // straight; empty otherwise.
  [[nodiscard]] Span directRun() const noexcept {
    if (span_ == spans_.size()) return {};
    const Span rest = spans_[span_].subspan(offset_);
    return direct(rest) ? rest : Span{};
  }
  // The length of the next chunk through the buffer: `limit`, or the bytes
  // before a later span the file can move straight when they are fewer and
  // a whole number of host pages, so that span starts at an aligned offset
  // of the slot.
  [[nodiscard]] size_t gatherBytes(size_t limit) const noexcept {
    if (span_ == spans_.size()) return limit;
    size_t before = spans_[span_].size() - offset_;
    for (size_t index = span_ + 1; index < spans_.size() && before < limit; ++index) {
      if (direct(spans_[index]) && before % kHostPageBytes == 0) return before;
      before += spans_[index].size();
    }
    return limit;
  }
  // The next piece of at most `bytes` bytes; empty once every span is done.
  Span next(size_t bytes) {
    if (span_ == spans_.size()) return {};
    const Span piece = spans_[span_].subspan(offset_, std::min(bytes, spans_[span_].size() - offset_));
    skip(piece.size());
    return piece;
  }
  // Moves past `bytes` bytes of the current span.
  void skip(size_t bytes) noexcept {
    offset_ += bytes;
    left_ -= bytes;
    settle();
  }

private:
  // Moves past every span that is done, so the current one has bytes left.
  void settle() noexcept {
    while (span_ < spans_.size() && offset_ == spans_[span_].size()) {
      ++span_;
      offset_ = 0;
    }
  }

  const std::vector<Span> &spans_;
  uint64_t left_;
  size_t span_ = 0;
  size_t offset_ = 0;
};

// Moves one chunk through io(data, bytes, offset), continuing a short
// transfer where it stopped.
template <typename Span, typename Io>
bool moveChunk(Span chunk, off_t offset, Io io) {
  while (!chunk.empty()) {
    const ssize_t count = io(chunk.data(), chunk.size(), offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    chunk = chunk.subspan(static_cast<size_t>(count));
    offset += count;
  }
  return true;
}
} // namespace

SlotFile::SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget)
    : backing_(std::make_shared<Backing>()) {
  if (!budget)
    throw std::invalid_argument("slot file needs a disk budget");
  const uint64_t capacityBytes = budget->capacityBytes();
  if (!slotBytes || capacityBytes > uint64_t{std::numeric_limits<off_t>::max()} ||
      capacityBytes / slotBytes > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("invalid slot file capacity");
  if (capacityBytes < slotBytes)
    throw std::invalid_argument("slot file quota holds no slot");
  if (slotBytes % kHostPageBytes)
    throw std::invalid_argument("slot size is not aligned for uncached IO");
  backing_->slotBytes = slotBytes;
  backing_->budget = std::move(budget);
  // A write past the file-size limit (ulimit -f, a launchd FileSize) raises
  // SIGXFSZ, whose default action kills the process. Ignored, the write fails
  // with EFBIG and stops the file like any other storage error. The change is
  // process-wide, which is safe because every other engine writer checks its
  // errors; a handler someone installed is left alone.
  struct sigaction fileSize {};
  if (::sigaction(SIGXFSZ, nullptr, &fileSize) == 0 && fileSize.sa_handler == SIG_DFL)
    std::signal(SIGXFSZ, SIG_IGN);
  std::string name = (std::filesystem::temp_directory_path() / "splash-cache-XXXXXX").string();
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
  void *buffer = nullptr;
  if (::posix_memalign(&buffer, kHostPageBytes, kChunkBytes) != 0)
    throw std::bad_alloc();
  buffer_.reset(static_cast<std::byte *>(buffer));
  worker_ = std::thread([this] { run(); });
}

void SlotFile::Free::operator()(std::byte *memory) const noexcept { std::free(memory); }

SlotFile::~SlotFile() {
  {
    std::lock_guard lock(backing_->mutex);
    backing_->stopping = true;
    for (auto &work : backing_->work) work.operation->cancel();
  }
  backing_->wake.notify_one();
  worker_.join();
}

std::shared_ptr<SlotFile::Slot> SlotFile::acquire() {
  auto slot = std::shared_ptr<Slot>(new Slot({}, 0));
  std::lock_guard lock(backing_->mutex);
  // A new index needs room in the free and punch lists, made before bytes
  // are reserved so allocation failure cannot strand quota, and returning a
  // slot never needs to allocate. An index waits for at most one punch:
  // freeing it again punches only after a write, which runs once the worker
  // has taken the earlier punch. Each list grows on its own, so one a failed
  // reserve left short grows on the next call.
  if (backing_->free.empty()) {
    for (std::vector<uint32_t> *list : {&backing_->free, &backing_->punches}) {
      if (list->capacity() == backing_->allocated)
        list->reserve(std::max<size_t>(1, 2 * list->capacity()));
    }
  }
  if (!backing_->budget->reserve(backing_->slotBytes)) return {};
  uint32_t index;
  if (!backing_->free.empty()) {
    index = backing_->free.back();
    backing_->free.pop_back();
  } else {
    index = backing_->allocated++;
  }
  slot->backing_ = backing_;
  slot->index_ = index;
  return slot;
}

uint64_t SlotFile::slotBytes() const noexcept { return backing_->slotBytes; }

bool SlotFile::writable() const noexcept {
  return !backing_->failed.load(std::memory_order_relaxed);
}

std::shared_ptr<SlotFile::Operation> SlotFile::submit(Run run,
                                                     std::function<void()> completion) {
  auto operation = std::make_shared<Operation>();
  {
    std::lock_guard lock(backing_->mutex);
    if (backing_->stopping)
      throw std::logic_error("slot file is shutting down");
    backing_->work.push_back({operation, std::move(run), std::move(completion)});
  }
  backing_->wake.notify_one();
  return operation;
}

void SlotFile::run() {
  Backing &backing = *backing_;
  for (;;) {
    Work work;
    {
      std::unique_lock lock(backing.mutex);
      backing.wake.wait(lock, [&] {
        return backing.stopping || !backing.work.empty() || !backing.punches.empty();
      });
      if (!backing.stopping && !backing.punches.empty()) {
        const uint32_t index = backing.punches.back();
        backing.punches.pop_back();
        lock.unlock();
        punchHole(index);
        continue;
      }
      if (backing.work.empty()) return;
      work = std::move(backing.work.front());
      backing.work.pop_front();
    }
    bool success = false;
    try { success = work.run({buffer_.get(), kChunkBytes}, work.operation->cancelled_); }
    catch (...) { success = false; }
    // Drop the operation's captures (its slot and the memory it moves)
    // before it reports, so a waiter wakes to a worker that holds nothing of
    // it.
    work.run = {};
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

void SlotFile::punchHole(uint32_t index) noexcept {
  Backing &backing = *backing_;
  // A write cancelled before its first chunk took no blocks.
  if (index >= backing.backed.size() || !backing.backed[index])
    return;
  // Slot offsets and sizes are whole alignment units, so whole blocks.
  fpunchhole_t hole{0, 0, slotOffset(index, backing.slotBytes),
                    static_cast<off_t>(backing.slotBytes)};
  if (::fcntl(backing.descriptor, F_PUNCHHOLE, &hole) < 0) {
    reportPunchFailure(errno);
    return;
  }
  backing.backed[index] = false;
  backing.budget->file_.fetch_sub(backing.slotBytes, std::memory_order_relaxed);
}

// A write gathers its spans into the worker's buffer one chunk at a time and
// zeros the slot past them; a read moves the chunks its spans reach and
// scatters each into them, ignoring the slot past them. A chunk of a span
// the file can move straight skips the buffer. Every pwrite and pread is an
// aligned range of the slot, and cancellation is noticed before each chunk.
std::shared_ptr<SlotFile::Operation> SlotFile::write(
    std::shared_ptr<Slot> slot, std::vector<std::span<const std::byte>> source,
    std::function<void()> completion) {
  if (!slot || slot->backing_ != backing_ || totalBytes(source) > backing_->slotBytes)
    throw std::invalid_argument("slot write does not match this file's slots");
  if (!writable())
    return nullptr;
  slot->returnsBlocks_ = true;
  return submit([slot, source = std::move(source)](std::span<std::byte> buffer,
                                                   const std::atomic<bool> &cancelled) {
    Backing &backing = *slot->backing_;
    const off_t start = slotOffset(slot->index_, backing.slotBytes);
    const auto store = [&backing](const std::byte *data, size_t bytes, off_t offset) {
      const auto count = ::pwrite(backing.descriptor, data, bytes, offset);
      if (count > 0)
        backing.budget->written_.fetch_add(count, std::memory_order_relaxed);
      return count;
    };
    slot->written_ = false;
    Pieces pieces(source);
    for (uint64_t done = 0; done < backing.slotBytes;) {
      if (cancelled.load(std::memory_order_relaxed)) return false;
      if (!done && backing.back(slot->index_))
        backing.budget->file_.fetch_add(backing.slotBytes, std::memory_order_relaxed);
      std::span<const std::byte> chunk = pieces.directRun();
      if (!chunk.empty()) {
        chunk = chunk.first(kChunkBytes);
        pieces.skip(kChunkBytes);
      } else {
        const auto gathered = buffer.first(
            pieces.gatherBytes(std::min<uint64_t>(buffer.size(), backing.slotBytes - done)));
        size_t filled = 0;
        for (auto piece = pieces.next(gathered.size()); !piece.empty();
             piece = pieces.next(gathered.size() - filled)) {
          std::memcpy(gathered.data() + filled, piece.data(), piece.size());
          filled += piece.size();
        }
        std::memset(gathered.data() + filled, 0, gathered.size() - filled);
        chunk = gathered;
      }
      if (!moveChunk(chunk, start + static_cast<off_t>(done), store)) {
        backing.failed.store(true, std::memory_order_relaxed);
        return false;
      }
      done += chunk.size();
    }
    slot->written_ = true;
    return true;
  }, std::move(completion));
}

std::shared_ptr<SlotFile::Operation> SlotFile::read(
    std::shared_ptr<Slot> slot, std::vector<std::span<std::byte>> destination,
    std::function<void()> completion) {
  if (!slot || slot->backing_ != backing_ || totalBytes(destination) > backing_->slotBytes)
    throw std::invalid_argument("slot read does not match this file's slots");
  return submit([slot, destination = std::move(destination)](std::span<std::byte> buffer,
                                                             const std::atomic<bool> &cancelled) {
    if (!slot->written_) return false;
    Backing &backing = *slot->backing_;
    const off_t start = slotOffset(slot->index_, backing.slotBytes);
    const auto load = [&backing](std::byte *data, size_t bytes, off_t offset) {
      const auto count = ::pread(backing.descriptor, data, bytes, offset);
      if (count > 0)
        backing.budget->read_.fetch_add(count, std::memory_order_relaxed);
      return count;
    };
    Pieces pieces(destination);
    for (uint64_t done = 0; !pieces.done();) {
      if (cancelled.load(std::memory_order_relaxed)) return false;
      const off_t offset = start + static_cast<off_t>(done);
      if (const auto run = pieces.directRun(); !run.empty()) {
        if (!moveChunk(run.first(kChunkBytes), offset, load)) return false;
        pieces.skip(kChunkBytes);
        done += kChunkBytes;
        continue;
      }
      const auto chunk = buffer.first(
          pieces.gatherBytes(std::min<uint64_t>(buffer.size(), backing.slotBytes - done)));
      if (!moveChunk(chunk, offset, load)) return false;
      size_t used = 0;
      for (auto piece = pieces.next(chunk.size()); !piece.empty();
           piece = pieces.next(chunk.size() - used)) {
        std::memcpy(piece.data(), chunk.data() + used, piece.size());
        used += piece.size();
      }
      done += chunk.size();
    }
    return true;
  }, std::move(completion));
}

} // namespace splash::model
