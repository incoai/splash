#include "model/SlotFile.hpp"

#include "StderrLine.hpp"
#include "model/Crc32c.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

namespace splash::model {

struct SlotFile::Backing {
  int descriptor = -1;
  uint64_t slotBytes = 0;
  std::shared_ptr<DiskBudget> budget;
  std::atomic<bool> failed{false};
  // A persistent file's record file, the label each record holds, the size
  // of a record and a cleared one.
  bool persistent = false;
  int records = -1;
  uint32_t labelBytes = 0;
  uint64_t recordBytes = 0;
  std::vector<std::byte> clearedRecord;
  // Guards the slots and the worker's queue.
  std::mutex mutex;
  std::condition_variable wake;
  uint32_t allocated = 0;
  std::vector<uint32_t> free;
  // Freed slots whose blocks go back before the next operation runs.
  std::vector<uint32_t> punches;
  std::deque<Work> work;
  bool stopping = false;
  // A persistent file between opening and finishAdoption(): the worker
  // waits, and the slots are adoption's alone.
  bool adopting = false;
  // The process is ending: freed slots keep their blocks and records.
  bool sealed = false;
  // Worker-only: the slots whose blocks the file holds, from a write's first
  // chunk until a punch returns them.
  std::vector<bool> backed;
  ~Backing() {
    if (descriptor >= 0) ::close(descriptor);
    if (records >= 0) ::close(records);
  }

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
    // file is stopping, its blocks go with it; once it is sealed, they stay
    // for the next process.
    punch = returnsBlocks_ && !backing_->stopping && !backing_->sealed;
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
// The slots a persistent file does not take back keep their records until
// they are reused: their payloads are punched, so a later read of one fails
// its check, and the next start looks at them again.
void reportRecordFailure(int error) noexcept {
  static std::atomic<bool> reported{false};
  if (!reported.exchange(true, std::memory_order_relaxed))
    writeStderrLine("A persistent cache file could not clear the records it does not take back (" +
                    std::generic_category().message(error) + "); they stay until reused.");
}

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

bool writeAll(int descriptor, std::span<const std::byte> bytes, off_t offset) {
  return moveChunk(bytes, offset, [descriptor](const std::byte *data, size_t size, off_t at) {
    return ::pwrite(descriptor, data, size, at);
  });
}

bool readAll(int descriptor, std::span<std::byte> bytes, off_t offset) {
  return moveChunk(bytes, offset, [descriptor](std::byte *data, size_t size, off_t at) {
    return ::pread(descriptor, data, size, at);
  });
}

// The record file: a header naming what the slots are for, then one record
// per slot index. A record is the payload's size and CRC-32C, the slot's
// label, and a CRC-32C of all of that; an all-zero record is a free slot.
// Fields are stored as this host lays them out: Splash runs on
// little-endian arm64 only.
constexpr char kRecordsMagic[8] = {'S', 'P', 'L', 'A', 'S', 'H', 'S', 'F'};
constexpr uint32_t kRecordsVersion = 1;
constexpr off_t kRecordsHeaderBytes = 4096;
constexpr uint32_t kRecordMagic = 0x53524543;

struct RecordsHeader final {
  char magic[8];
  uint32_t version;
  uint32_t labelBytes;
  uint64_t slotBytes;
  uint32_t tagBytes;
  uint32_t checksum;
};

struct RecordHead final {
  uint32_t magic;
  uint32_t payloadChecksum;
  uint64_t payloadBytes;
};

static_assert(std::endian::native == std::endian::little);
static_assert(std::is_trivially_copyable_v<RecordsHeader> && sizeof(RecordsHeader) == 32);
static_assert(std::is_trivially_copyable_v<RecordHead> && sizeof(RecordHead) == 16);

// The header and the tag after it; its checksum covers both.
std::vector<std::byte> recordsHeader(uint64_t slotBytes, uint32_t labelBytes,
                                     std::span<const std::byte> tag) {
  if (sizeof(RecordsHeader) + tag.size() > kRecordsHeaderBytes)
    throw std::invalid_argument("slot file tag is too long");
  RecordsHeader header{};
  std::memcpy(header.magic, kRecordsMagic, sizeof(header.magic));
  header.version = kRecordsVersion;
  header.labelBytes = labelBytes;
  header.slotBytes = slotBytes;
  header.tagBytes = static_cast<uint32_t>(tag.size());
  std::vector<std::byte> bytes(sizeof(header) + tag.size());
  std::memcpy(bytes.data(), &header, sizeof(header));
  std::memcpy(bytes.data() + sizeof(header), tag.data(), tag.size());
  header.checksum = crc32c(bytes);
  std::memcpy(bytes.data(), &header, sizeof(header));
  return bytes;
}

uint64_t recordBytesFor(uint32_t labelBytes) noexcept {
  constexpr uint64_t alignment = 64;
  const uint64_t bytes = sizeof(RecordHead) + uint64_t{labelBytes} + sizeof(uint32_t);
  return (bytes + alignment - 1) / alignment * alignment;
}

off_t recordOffset(uint64_t index, uint64_t recordBytes) {
  return kRecordsHeaderBytes + static_cast<off_t>(index * recordBytes);
}

std::vector<std::byte> encodeRecord(uint64_t recordBytes, uint64_t payloadBytes,
                                    uint32_t payloadChecksum,
                                    std::span<const std::byte> label) {
  std::vector<std::byte> record(recordBytes);
  const RecordHead head{kRecordMagic, payloadChecksum, payloadBytes};
  std::memcpy(record.data(), &head, sizeof(head));
  std::memcpy(record.data() + sizeof(head), label.data(), label.size());
  const size_t covered = sizeof(head) + label.size();
  const uint32_t checksum = crc32c(std::span(record).first(covered));
  std::memcpy(record.data() + covered, &checksum, sizeof(checksum));
  return record;
}

// The record's slot, or nothing for a free index or a damaged record.
std::optional<SlotRecord> decodeRecord(std::span<const std::byte> record, uint32_t index,
                                       uint32_t labelBytes, uint64_t slotBytes) {
  RecordHead head;
  std::memcpy(&head, record.data(), sizeof(head));
  if (head.magic != kRecordMagic || !head.payloadBytes || head.payloadBytes > slotBytes)
    return std::nullopt;
  const size_t covered = sizeof(head) + labelBytes;
  uint32_t checksum;
  std::memcpy(&checksum, record.data() + covered, sizeof(checksum));
  if (checksum != crc32c(record.first(covered)))
    return std::nullopt;
  const auto label = record.subspan(sizeof(head), labelBytes);
  return SlotRecord{index, head.payloadBytes, head.payloadChecksum, {label.begin(), label.end()}};
}

int openPersistent(const std::filesystem::path &path, const char *what) {
  const int descriptor = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (descriptor < 0)
    throw std::system_error(errno, std::generic_category(), what);
  return descriptor;
}

uint64_t fileBytes(int descriptor, const char *what) {
  struct stat status {};
  if (::fstat(descriptor, &status) < 0)
    throw std::system_error(errno, std::generic_category(), what);
  return static_cast<uint64_t>(status.st_size);
}
} // namespace

SlotFile::SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget)
    : backing_(std::make_shared<Backing>()) {
  prepare(slotBytes, std::move(budget));
  std::string name = (std::filesystem::temp_directory_path() / "splash-cache-XXXXXX").string();
  backing_->descriptor = ::mkstemp(name.data());
  if (backing_->descriptor < 0)
    throw std::system_error(errno, std::generic_category(), "create slot file");
  const int unlinked = ::unlink(name.c_str());
  if (unlinked < 0)
    throw std::system_error(errno, std::generic_category(), "unlink slot file");
  if (::fcntl(backing_->descriptor, F_SETFD, FD_CLOEXEC) < 0)
    throw std::system_error(errno, std::generic_category(), "close-on-exec slot file");
  uncache();
  worker_ = std::thread([this] { run(); });
}

SlotFile::SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget,
                   const Persistence &persistence)
    : backing_(std::make_shared<Backing>()) {
  prepare(slotBytes, std::move(budget));
  open(persistence);
  worker_ = std::thread([this] { run(); });
}

void SlotFile::prepare(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget) {
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
  void *buffer = nullptr;
  if (::posix_memalign(&buffer, kHostPageBytes, kChunkBytes) != 0)
    throw std::bad_alloc();
  buffer_.reset(static_cast<std::byte *>(buffer));
}

void SlotFile::uncache() {
  // Avoid turning the cold tier into another long-lived RAM copy. Host VM
  // pressure accounting still covers any transient kernel IO memory.
  if (::fcntl(backing_->descriptor, F_NOCACHE, 1) < 0)
    throw std::system_error(errno, std::generic_category(), "uncached slot file");
}

void SlotFile::open(const Persistence &persistence) {
  Backing &backing = *backing_;
  if (!persistence.labelBytes)
    throw std::invalid_argument("a persistent slot file needs a label size");
  backing.persistent = true;
  backing.labelBytes = persistence.labelBytes;
  backing.recordBytes = recordBytesFor(persistence.labelBytes);
  backing.clearedRecord.assign(backing.recordBytes, std::byte{0});
  const std::vector<std::byte> header =
      recordsHeader(backing.slotBytes, backing.labelBytes, persistence.tag);
  backing.descriptor = openPersistent(persistence.slots, "open slot file");
  uncache();
  backing.records = openPersistent(persistence.records, "open slot record file");
  backing.adopting = true;
  const uint64_t recordFileBytes = fileBytes(backing.records, "stat slot record file");
  std::vector<std::byte> found(header.size());
  if (recordFileBytes < uint64_t{kRecordsHeaderBytes} || !readAll(backing.records, found, 0) ||
      found != header) {
    // Written for something else, or new: start empty.
    if (::ftruncate(backing.descriptor, 0) < 0 || ::ftruncate(backing.records, 0) < 0 ||
        !writeAll(backing.records, header, 0) ||
        ::ftruncate(backing.records, kRecordsHeaderBytes) < 0)
      throw std::system_error(errno, std::generic_category(), "reset slot file");
    return;
  }
  // A record counts only while its slot lies wholly inside the slot file.
  const uint64_t slots = fileBytes(backing.descriptor, "stat slot file") / backing.slotBytes;
  const uint64_t recorded = (recordFileBytes - kRecordsHeaderBytes) / backing.recordBytes;
  if (std::max(slots, recorded) > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("slot file holds more slots than it can index");
  adopted_.assign(std::max(slots, recorded), false);
  std::vector<std::byte> records(std::min(slots, recorded) * backing.recordBytes);
  if (!readAll(backing.records, records, kRecordsHeaderBytes))
    throw std::system_error(errno, std::generic_category(), "read slot records");
  for (uint64_t index = 0; index < records.size() / backing.recordBytes; ++index) {
    const auto record = std::span(records).subspan(index * backing.recordBytes, backing.recordBytes);
    if (auto slot = decodeRecord(record, static_cast<uint32_t>(index), backing.labelBytes,
                                 backing.slotBytes))
      records_.push_back(std::move(*slot));
  }
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
  if (backing_->adopting)
    throw std::logic_error("a persistent slot file acquires slots only after adoption");
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

bool SlotFile::persistent() const noexcept { return backing_->persistent; }

// Adoption runs before the worker takes any operation or punch, so it owns
// the slots and their blocks (Backing::backed) meanwhile.
std::shared_ptr<SlotFile::Slot> SlotFile::adopt(const SlotRecord &record) {
  Backing &backing = *backing_;
  auto slot = std::shared_ptr<Slot>(new Slot({}, 0));
  std::lock_guard lock(backing.mutex);
  if (!backing.adopting)
    throw std::logic_error("slots are adopted only while a persistent slot file opens");
  if (record.index >= adopted_.size() || adopted_[record.index] || !record.payloadBytes ||
      record.payloadBytes > backing.slotBytes)
    throw std::invalid_argument("adopted slot is not one of this file's records");
  // Room for every index in the free and punch lists, as acquire() makes
  // it, so that returning a slot never needs to allocate.
  const uint32_t allocated = std::max(backing.allocated, record.index + 1);
  for (std::vector<uint32_t> *list : {&backing.free, &backing.punches})
    list->reserve(allocated);
  adopted_[record.index] = true;
  backing.allocated = allocated;
  backing.budget->used_.fetch_add(backing.slotBytes, std::memory_order_relaxed);
  if (backing.back(record.index))
    backing.budget->file_.fetch_add(backing.slotBytes, std::memory_order_relaxed);
  slot->backing_ = backing_;
  slot->index_ = record.index;
  slot->written_ = true;
  slot->payloadBytes_ = record.payloadBytes;
  slot->checksum_ = record.checksum;
  slot->verify_ = true;
  slot->returnsBlocks_ = true;
  return slot;
}

void SlotFile::finishAdoption() {
  Backing &backing = *backing_;
  {
    std::lock_guard lock(backing.mutex);
    if (!backing.adopting)
      throw std::logic_error("slot file adoption already finished");
    // Past the last slot adopted, the files end; below it, each run of
    // slots not adopted is free, its records cleared and its blocks
    // returned. None of it is needed for serving: a failure leaves records
    // behind, and the file goes on.
    if (::ftruncate(backing.descriptor,
                    static_cast<off_t>(uint64_t{backing.allocated} * backing.slotBytes)) < 0 ||
        ::ftruncate(backing.records, recordOffset(backing.allocated, backing.recordBytes)) < 0)
      reportRecordFailure(errno);
    const std::vector<std::byte> cleared(kChunkBytes);
    for (uint32_t begin = 0; begin < backing.allocated;) {
      if (adopted_[begin]) {
        ++begin;
        continue;
      }
      uint32_t end = begin;
      while (end < backing.allocated && !adopted_[end])
        backing.free.push_back(end++);
      for (off_t offset = recordOffset(begin, backing.recordBytes),
                 stop = recordOffset(end, backing.recordBytes);
           offset < stop;) {
        const auto bytes = std::min<off_t>(stop - offset, static_cast<off_t>(cleared.size()));
        if (!writeAll(backing.records, std::span(cleared).first(static_cast<size_t>(bytes)), offset)) {
          reportRecordFailure(errno);
          break;
        }
        offset += bytes;
      }
      fpunchhole_t hole{0, 0, slotOffset(begin, backing.slotBytes),
                        static_cast<off_t>(uint64_t{end - begin} * backing.slotBytes)};
      if (::fcntl(backing.descriptor, F_PUNCHHOLE, &hole) < 0)
        reportPunchFailure(errno);
      begin = end;
    }
    backing.adopting = false;
  }
  adopted_ = {};
  records_ = {};
  // Slots dropped during adoption wait for the worker to punch them.
  backing.wake.notify_one();
}

void SlotFile::label(std::shared_ptr<Slot> slot, std::vector<std::byte> label) {
  if (!slot || slot->backing_ != backing_)
    throw std::invalid_argument("slot label does not match this file's slots");
  if (!backing_->persistent)
    return;
  if (label.size() != backing_->labelBytes)
    throw std::invalid_argument("slot label does not match the file's label size");
  static_cast<void>(submit(
      [slot, label = std::move(label)](std::span<std::byte>, const std::atomic<bool> &) {
        const Backing &backing = *slot->backing_;
        if (!slot->written_) return false;
        return writeAll(backing.records,
                        encodeRecord(backing.recordBytes, slot->payloadBytes_,
                                     slot->checksum_, label),
                        recordOffset(slot->index_, backing.recordBytes));
      },
      {}));
}

std::shared_ptr<SlotFile::Operation> SlotFile::synchronize(std::function<void()> completion) {
  return submit(
      [backing = backing_](std::span<std::byte>, const std::atomic<bool> &) {
        // Both files' metadata first; F_FULLFSYNC then asks the drive to
        // flush its cache, which holds the slots' uncached writes too.
        return !backing->persistent ||
               (::fsync(backing->descriptor) == 0 && ::fcntl(backing->records, F_FULLFSYNC) == 0);
      },
      std::move(completion));
}

void SlotFile::seal() noexcept {
  std::lock_guard lock(backing_->mutex);
  backing_->sealed = true;
}

std::shared_ptr<SlotFile::Operation> SlotFile::submit(Run run,
                                                     std::function<void()> completion) {
  auto operation = std::make_shared<Operation>();
  {
    std::lock_guard lock(backing_->mutex);
    if (backing_->stopping)
      throw std::logic_error("slot file is shutting down");
    if (backing_->adopting)
      throw std::logic_error("a persistent slot file takes operations only after adoption");
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
      // Adoption owns the slots until it finishes, and no operation is
      // submitted before.
      backing.wake.wait(lock, [&] {
        return backing.stopping || !backing.work.empty() ||
               (!backing.punches.empty() && !backing.adopting);
      });
      if (!backing.stopping && !backing.adopting && !backing.punches.empty()) {
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
  // The record goes first: a crash between the two leaves an unrecorded
  // slot, which the next process frees. A record whose clearing failed names
  // a payload that is gone, and its slot's first read fails.
  if (backing.persistent)
    static_cast<void>(writeAll(backing.records, backing.clearedRecord,
                               recordOffset(index, backing.recordBytes)));
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
// In a persistent file a write checksums its spans as they go, and the first
// read of an adopted slot checks them.
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
    slot->verify_ = false;
    uint32_t checksum = 0;
    Pieces pieces(source);
    for (uint64_t done = 0; done < backing.slotBytes;) {
      if (cancelled.load(std::memory_order_relaxed)) return false;
      if (!done && backing.back(slot->index_))
        backing.budget->file_.fetch_add(backing.slotBytes, std::memory_order_relaxed);
      std::span<const std::byte> chunk = pieces.directRun();
      if (!chunk.empty()) {
        chunk = chunk.first(kChunkBytes);
        pieces.skip(kChunkBytes);
        if (backing.persistent)
          checksum = crc32c(chunk, checksum);
      } else {
        const auto gathered = buffer.first(
            pieces.gatherBytes(std::min<uint64_t>(buffer.size(), backing.slotBytes - done)));
        size_t filled = 0;
        for (auto piece = pieces.next(gathered.size()); !piece.empty();
             piece = pieces.next(gathered.size() - filled)) {
          std::memcpy(gathered.data() + filled, piece.data(), piece.size());
          filled += piece.size();
        }
        if (backing.persistent)
          checksum = crc32c(gathered.first(filled), checksum);
        std::memset(gathered.data() + filled, 0, gathered.size() - filled);
        chunk = gathered;
      }
      if (!moveChunk(chunk, start + static_cast<off_t>(done), store)) {
        backing.failed.store(true, std::memory_order_relaxed);
        return false;
      }
      done += chunk.size();
    }
    slot->payloadBytes_ = totalBytes(source);
    slot->checksum_ = checksum;
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
    // The first read of an adopted slot reads its whole payload and checks
    // it against the record.
    const bool verify = slot->verify_;
    if (verify && totalBytes(destination) != slot->payloadBytes_) return false;
    uint32_t checksum = 0;
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
        const auto chunk = run.first(kChunkBytes);
        if (!moveChunk(chunk, offset, load)) return false;
        if (verify)
          checksum = crc32c(chunk, checksum);
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
      if (verify)
        checksum = crc32c(chunk.first(used), checksum);
      done += chunk.size();
    }
    if (verify) {
      if (checksum != slot->checksum_) return false;
      slot->verify_ = false;
    }
    return true;
  }, std::move(completion));
}

} // namespace splash::model
