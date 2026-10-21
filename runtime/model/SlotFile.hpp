#pragma once

#include "AwakeClock.hpp"
#include "Checked.hpp"

#include <sys/mount.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace splash::model {

// The disk one cache tier may use, shared by every slot file of the tier.
// Reservations and releases come from the engine thread and from whoever
// drops the last handle of a slot. The budget bounds live slots by its
// limit: the quota at most, and less while the volume the files live on
// leaves less (engine::DiskSpace sets it). A limit below what the tier holds
// is met by the cache giving copies up (Cache::keepDiskLimit), so live slots
// can exceed it for a while, as slots a persistent file adopted can. A freed
// slot returns its bytes at once but its blocks only when its file's worker
// punches them, after the transfer that worker is in, so the files' blocks
// can briefly exceed the limit by the slots freed but not yet punched.
//
// Active KV, the copies of the KV requests hold and the quota Spill requests
// committed for the pages they will write, is bounded by the active limit
// instead, which leaves it the volume's whole room where the automatic
// quota's copies keep a quarter of it (engine::DiskSpace): a commitment
// locks quota without taking a slot or giving a copy up, and a committed
// page's slot draws on it (Draw::Commitment). Every draw checks what the
// files hold against its limit, so none passes a full volume.
//
// A write budget that spares the drive is the user's to set
// (setWriteBudget, --max-cache-disk-writes); without one, the default, every
// write passes and nothing is held back. With one, every write a file of the
// tier submits draws on it: a token bucket of this process, counted at
// submission in whole slots, which each start begins full. The budget then
// decides which states are written: while it is spent, an evicted state is
// dropped instead and a persistent restore point waits, unless the state is
// in use (the state a conversation resumes from), which passes. KV is written
// only for a state the budget let through, its chain's pages kept reachable
// or copied after it: those writes pass, counted all the same, so such a
// state never loses its chain for want of budget, and what a chain costs
// comes out of later states' share. Writes are submitted on the engine
// thread, which alone reads the budget (admits), so a write the engine
// admitted is never refused for it.
class DiskBudget final {
public:
  // Milliseconds on the clock the write budget refills by; it does not
  // throw. It is the clock the engine schedules its wake-ups on
  // (Cache::diskWritesAdmittedAt).
  using Clock = std::function<double()>;

  // Without a clock the budget refills by the awake clock
  // (awakeMilliseconds), the native loop's.
  explicit DiskBudget(uint64_t capacityBytes, Clock clock = {})
      : capacity_(capacityBytes), limit_(capacityBytes), activeLimit_(capacityBytes),
        clock_(clock ? std::move(clock) : Clock(awakeMilliseconds)) {}
  // The quota.
  [[nodiscard]] uint64_t capacityBytes() const noexcept { return capacity_; }
  // What the tier may hold now: the quota until setLimitBytes() sets less.
  [[nodiscard]] uint64_t limitBytes() const noexcept {
    return limit_.load(std::memory_order_relaxed);
  }
  // At most the quota.
  void setLimitBytes(uint64_t bytes) noexcept {
    limit_.store(std::min(bytes, capacity_), std::memory_order_relaxed);
  }
  // What the volume had available when it was last measured; zero before.
  [[nodiscard]] uint64_t volumeAvailableBytes() const noexcept {
    return volumeAvailable_.load(std::memory_order_relaxed);
  }
  void setVolumeAvailableBytes(uint64_t bytes) noexcept {
    volumeAvailable_.store(bytes, std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t usedBytes() const noexcept {
    return used_.load(std::memory_order_relaxed);
  }

  // Active KV, the copies of the KV requests hold and what Spill requests
  // committed, may use more of the volume than cached copies may: its limit
  // (engine::DiskSpace sets it), at most the quota, which it is until set.
  [[nodiscard]] uint64_t activeLimitBytes() const noexcept {
    return activeLimit_.load(std::memory_order_relaxed);
  }
  void setActiveLimitBytes(uint64_t bytes) noexcept {
    activeLimit_.store(std::min(bytes, capacity_), std::memory_order_relaxed);
  }
  // Quota a Spill request's pages will take, locked without taking a slot
  // or giving any copy up: granted iff heldBytes (the copies of the KV
  // requests hold, the cache's count) plus what is committed plus `bytes`
  // stays within the active limit. Engine thread.
  [[nodiscard]] bool commit(uint64_t bytes, uint64_t heldBytes) noexcept {
    const uint64_t limit = activeLimitBytes(), committed = committedBytes();
    if (heldBytes > limit || committed > limit - heldBytes || bytes > limit - heldBytes - committed)
      return false;
    committed_.store(committed + bytes, std::memory_order_relaxed);
    return true;
  }
  // Returns what a request committed and will not write. Engine thread.
  void uncommit(uint64_t bytes) noexcept { committed_.fetch_sub(bytes, std::memory_order_relaxed); }
  [[nodiscard]] uint64_t committedBytes() const noexcept {
    return committed_.load(std::memory_order_relaxed);
  }
  // What a slot's reservation draws on: a cached copy's room within the
  // limit, or a committed page's, whose bytes move from what is committed
  // to what is used within the active limit. Either checks what the files
  // hold, so no draw passes a full quota; a draw of more than is committed
  // is refused. A commitment's draw is the engine thread's.
  enum class Draw : uint8_t { Cache, Commitment };
  [[nodiscard]] bool reserve(uint64_t bytes, Draw draw = Draw::Cache) noexcept {
    const bool committed = draw == Draw::Commitment;
    if (committed && bytes > committedBytes())
      return false;
    const uint64_t limit = committed ? activeLimitBytes() : limitBytes();
    uint64_t used = used_.load(std::memory_order_relaxed);
    do {
      // Adopted slots, or a limit that fell, may leave more in use than it.
      if (used > limit || bytes > limit - used)
        return false;
    } while (!used_.compare_exchange_weak(used, used + bytes, std::memory_order_relaxed));
    if (committed)
      uncommit(bytes);
    return true;
  }
  void release(uint64_t bytes) noexcept { used_.fetch_sub(bytes, std::memory_order_relaxed); }

  // The write budget: bytesPerDay on average, and at most a quarter of a
  // day's at once, which a new budget starts with. Without one, as until
  // this is called, every write passes.
  void setWriteBudget(uint64_t bytesPerDay) {
    refillPerMillisecond_ = static_cast<double>(bytesPerDay) / kDayMilliseconds;
    burstBytes_ = static_cast<double>(bytesPerDay) / 4.0;
    balance_ = burstBytes_;
    balanceTime_ = clock_();
  }
  // Whether a state's write may start now: one in use always, another
  // while the budget is not overdrawn. Engine thread.
  [[nodiscard]] bool admits(bool inUse) const noexcept {
    return inUse || lifted_ || !refillPerMillisecond_ || balanceAt(clock_()) >= 0.0;
  }
  // Whether a request may start to spill, or convert to it, when Spill
  // requests already committed `outstandingBytes` of writes: without a
  // write budget, or while its balance covers them. Their writes are in use
  // and pass whatever the balance. Engine thread.
  [[nodiscard]] bool admitsSpill(uint64_t outstandingBytes) const noexcept {
    return lifted_ || !refillPerMillisecond_ || balanceAt(clock_()) >= static_cast<double>(outstandingBytes);
  }
  // When admitsSpill(outstandingBytes) holds again as the balance refills,
  // on the budget's clock; empty while it holds now. The writes of Spill
  // requests lower their commitments as they charge the balance, so the
  // balance's shortfall closes only by its refill, and sooner as they end
  // and return what they did not write.
  [[nodiscard]] std::optional<double> admitsSpillAt(uint64_t outstandingBytes) const noexcept {
    if (admitsSpill(outstandingBytes))
      return std::nullopt;
    const double now = clock_();
    return now + (static_cast<double>(outstandingBytes) - balanceAt(now)) / refillPerMillisecond_;
  }
  // When a state's write not in use may start again, on the budget's clock;
  // empty while one may start now.
  [[nodiscard]] std::optional<double> admitsAgainAt() const noexcept {
    if (lifted_ || !refillPerMillisecond_)
      return std::nullopt;
    const double now = clock_();
    const double balance = balanceAt(now);
    if (balance >= 0.0)
      return std::nullopt;
    return now - balance / refillPerMillisecond_;
  }
  // What the next state's write not in use may take now: negative once
  // overdrawn, by the last write's slot or by the writes that pass; zero
  // without a budget.
  [[nodiscard]] int64_t writeBudgetBytes() const noexcept {
    return refillPerMillisecond_ ? static_cast<int64_t>(balanceAt(clock_())) : 0;
  }
  // At a clean stop the budget holds nothing back: the newest restore points
  // reach the disk (WriteBehind::flush).
  void lift() noexcept { lifted_ = true; }

  // A write of the tier found its volume full (ENOSPC, EDQUOT). The worker
  // that met it only reports it: its file goes on, and what the engine
  // admits does not change until it measures the volume
  // (engine::DiskSpace::measure), which takes the report.
  void reportFull() noexcept { full_.store(true, std::memory_order_relaxed); }
  [[nodiscard]] bool fullReported() const noexcept {
    return full_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] bool takeFullReport() noexcept {
    return full_.exchange(false, std::memory_order_relaxed);
  }
  // A file of the tier takes no more writes (SlotFile::writable).
  [[nodiscard]] bool writesStopped() const noexcept {
    return stopped_.load(std::memory_order_relaxed);
  }
  // Cumulative bytes accepted by file IO, including partial/cancelled work.
  // This is application IO, not physical SSD traffic or filesystem overhead.
  [[nodiscard]] uint64_t readBytes() const noexcept {
    return read_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t writtenBytes() const noexcept {
    return written_.load(std::memory_order_relaxed);
  }
  // Bytes the budget's files occupy on disk: the slots written since their
  // blocks last went back.
  [[nodiscard]] uint64_t fileBytes() const noexcept {
    return file_.load(std::memory_order_relaxed);
  }
  // Cumulative microseconds the files' writes waited for reads a step waits
  // for (SlotFile::Options::deferWrites).
  [[nodiscard]] uint64_t writesDeferredMicroseconds() const noexcept {
    return deferred_.load(std::memory_order_relaxed);
  }

private:
  friend class SlotFile;
  static constexpr double kDayMilliseconds = 86'400'000.0;

  // The balance at `now` on the budget's clock.
  [[nodiscard]] double balanceAt(double now) const noexcept {
    return std::min(burstBytes_, balance_ + refillPerMillisecond_ * (now - balanceTime_));
  }
  // A write's whole slot, as SlotFile::write submits it.
  void charge(uint64_t bytes) noexcept {
    if (!refillPerMillisecond_)
      return;
    const double now = clock_();
    balance_ = balanceAt(now) - static_cast<double>(bytes);
    balanceTime_ = now;
  }

  uint64_t capacity_;
  std::atomic<uint64_t> limit_;
  std::atomic<uint64_t> activeLimit_;
  std::atomic<uint64_t> volumeAvailable_{0};
  std::atomic<uint64_t> used_{0};
  std::atomic<uint64_t> committed_{0};
  std::atomic<uint64_t> read_{0};
  std::atomic<uint64_t> written_{0};
  std::atomic<uint64_t> file_{0};
  std::atomic<uint64_t> deferred_{0};
  std::atomic<bool> full_{false};
  std::atomic<bool> stopped_{false};
  // The write budget, on the engine thread: its refill per millisecond (zero
  // without one), its most, and its balance at balanceTime_.
  Clock clock_;
  double refillPerMillisecond_ = 0.0;
  double burstBytes_ = 0.0;
  double balance_ = 0.0;
  double balanceTime_ = 0.0;
  bool lifted_ = false;
};

// One slot a persistent file keeps from an earlier process, as its record
// says: where it is, the size of the payload its last complete write stored
// and the CRC-32C of each of its segments (SlotFile::Options::segments), and
// the label its writer gave it.
struct SlotRecord final {
  uint32_t index = 0;
  uint64_t payloadBytes = 0;
  std::vector<uint32_t> checksums;
  std::vector<std::byte> label;
};

// Fixed-size slots in one file, served by one IO worker in submission order
// and bounded by a disk budget. Callers own the memory an operation moves
// and keep it alive until the operation is ready. That memory may have any
// size and alignment. Whole 1 MiB chunks of a span that start at an aligned
// offset of the slot, in memory aligned like slot offsets, move straight
// between memory and the file; everything else, such as the rest of a span
// short of a chunk, moves through an aligned buffer of the worker's own, so
// the file sees only transfers aligned to the host page (kHostPageBytes), as
// uncached IO wants; a slot is a whole number of host pages. A slot is
// readable only after one complete write; a failed or cancelled write leaves
// it unreadable. Beside the worker, any thread may read part of a written
// slot straight from the file (readAt). While reads a step waits for run
// (Options::deferWrites), the worker holds each chunk of a write back, at
// most kWriteDeferral, unless a read of the file waits behind the write. A
// write that finds the volume full is reported to the budget
// (DiskBudget::fullReported) and the file goes on; after one that fails for
// any other reason, which the worker logs, the file accepts no further
// writes. A freed slot returns its quota at once; one that was written
// returns its blocks to the volume (F_PUNCHHOLE) on the worker, after the
// operation in flight and before any later one (DiskBudget). So that a
// file-size limit fails a write rather than killing the process, a file
// ignores SIGXFSZ from its construction on.
//
// A temporary file is unlinked at once and gone with the process. A
// persistent file stays, with a record file beside it: one fixed-size record
// per slot, written by the worker after the slot's write has landed, from
// the label its writer gives the slot and the CRC-32C of each segment of the
// payload, which the worker computed while writing it. A freed slot's record
// is cleared before its blocks return. The next process reads the records,
// adopts the slots it keeps, and frees the others. A read of an adopted
// slot fails unless each segment it covers matches its record, and the
// worker's first read covers the whole payload, after which the slot is
// trusted; so a slot that changed after its record was written, or whose
// blocks never reached the drive, is a cache miss and nothing worse. Slots
// this process wrote are trusted as a temporary file's are.
class SlotFile final {
  struct Backing;

public:
  // Where a persistent file keeps its slots and their records, and what the
  // slots are for: a file written for another tag, slot size or label size
  // is emptied when it opens.
  struct Persistence final {
    std::filesystem::path slots;
    std::filesystem::path records;
    std::vector<std::byte> tag;
    uint32_t labelBytes = 0;
  };

  // The slot that holds payloadBytes: writes zero the rest.
  [[nodiscard]] static constexpr uint64_t slotBytesFor(uint64_t payloadBytes) noexcept {
    return alignUp(payloadBytes);
  }

  // How a file's slots are written, beside their size and persistence.
  struct Options final {
    // The byte lengths of a payload's consecutive segments from its start
    // (kv::SlotFormat::segments). A persistent file keeps a CRC-32C of each,
    // which a read of an adopted slot checks for every segment it covers;
    // empty or one, the whole payload, whatever its size, is one segment.
    // Several hold every payload of the file: a write stores their sum.
    std::vector<uint64_t> segments;
    // While it says so, the worker holds a write's next chunk back, at most
    // kWriteDeferral each (model::SsdReaders::criticalBusyProbe: reads a
    // step waits for go first), unless a read of the file waits behind the
    // write and so would wait as long. It runs on the worker; the file's
    // slots keep it, and the last may go on any thread, a reader's too, so
    // it owns nothing that thread may not destroy.
    std::function<bool()> deferWrites;
  };
  // The longest a write's chunk waits for reads (Options::deferWrites).
  static constexpr std::chrono::milliseconds kWriteDeferral{2};

  class Slot final {
  public:
    ~Slot();
    Slot(const Slot &) = delete;
    Slot &operator=(const Slot &) = delete;
    // Unique in the process, from acquire() or adopt(): a slot is written
    // once while held, so the id names the bytes of one write.
    [[nodiscard]] uint64_t id() const noexcept { return id_; }

  private:
    friend class SlotFile;
    Slot(std::shared_ptr<Backing> backing, uint32_t index);
    std::shared_ptr<Backing> backing_;
    uint32_t index_;
    uint64_t id_;
    // Set by the worker, which runs a file's operations in submission
    // order, once a write landed; read beside it too (readAt), which then
    // finds the fields below the write's.
    std::atomic<bool> written_{false};
    // The size of the payload the last complete write stored and, in a
    // persistent file, the CRC-32C of each of its segments. Reads of an
    // adopted slot check the segments they cover against them while verify_
    // holds, which the worker's first read clears once the whole payload
    // matched.
    uint64_t payloadBytes_ = 0;
    std::vector<uint32_t> checksums_;
    std::atomic<bool> verify_{false};
    // Set when a write is submitted: freeing the slot then returns the
    // blocks that write may have taken.
    bool returnsBlocks_ = false;
  };

  class Operation final {
  public:
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] bool wait();
    void cancel() noexcept { cancelled_.store(true, std::memory_order_relaxed); }
    // Stops the operation before its next chunk, or before it starts, and
    // waits until the worker has let go of the memory it moves; the owner may
    // free that memory afterwards.
    void drain() {
      cancel();
      static_cast<void>(wait());
    }

  private:
    friend class SlotFile;
    std::atomic<bool> cancelled_{false};
    std::atomic<bool> done_{false};
    bool success_ = false;
    std::mutex mutex_;
    std::condition_variable wake_;
  };

  // The budget holds at least one slot: a smaller quota is a configuration
  // error. Files sharing a budget compete for its bytes. A temporary file
  // goes to the temporary directory.
  SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget, const Options &options = {});
  // A persistent file: opening it reads the records an earlier process left,
  // and it takes no slot of its own until finishAdoption().
  SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget,
           const Persistence &persistence, const Options &options = {});
  ~SlotFile();
  SlotFile(const SlotFile &) = delete;
  SlotFile &operator=(const SlotFile &) = delete;
  // Null when the live slots are at the budget's limit for the draw
  // (DiskBudget::reserve).
  [[nodiscard]] std::shared_ptr<Slot> acquire(DiskBudget::Draw draw = DiskBudget::Draw::Cache);
  [[nodiscard]] uint64_t slotBytes() const noexcept;
  // False once a write has failed for another reason than a full volume;
  // complete slots stay readable.
  [[nodiscard]] bool writable() const noexcept;
  // The file takes a state's write now: it is writable, the limit holds a
  // slot, so making room can find one, and the write budget admits the write
  // (DiskBudget::admits). Engine thread.
  [[nodiscard]] bool takesWrite(bool inUse) const noexcept;
  [[nodiscard]] bool persistent() const noexcept;
  // What fstatfs says of the volume the file lives on, linked or not; empty
  // when it cannot say.
  [[nodiscard]] std::optional<struct statfs> volume() const noexcept;
  // The bytes the file's blocks take on that volume, its records' included;
  // zero when they cannot be measured.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;

  // The slots earlier processes left in a persistent file, as read when it
  // opened: empty for a temporary or a new one.
  [[nodiscard]] const std::vector<SlotRecord> &records() const noexcept { return records_; }
  // Takes back the slot a record names, before finishAdoption(). It counts
  // against the budget even past its capacity, so that the caller can give
  // up what the quota no longer holds; its reads fail unless what they
  // cover matches the record.
  [[nodiscard]] std::shared_ptr<Slot> adopt(const SlotRecord &record);
  // Ends adoption: every slot not adopted is freed, its record cleared and
  // its blocks returned. A record that cannot be cleared stays, reported,
  // until its slot is reused; its punched payload fails any later check.
  // acquire() needs it in a persistent file.
  void finishAdoption();
  // Keeps the label in the slot's record once the write submitted before it
  // has landed; a slot whose write failed gets no record. A label is the
  // file's label size. Nothing happens in a temporary file.
  void label(std::shared_ptr<Slot> slot, std::vector<std::byte> label);
  // Flushes the slots and records written so far to the drive.
  [[nodiscard]] std::shared_ptr<Operation> synchronize(std::function<void()> completion);
  // The process is ending: a slot dropped from now on keeps its blocks and
  // its record for the next one.
  void seal() noexcept;
  // The spans total at most one slot, a write's exactly the segments in a
  // file of several, and stay valid until the operation is ready. A write
  // stores them in order from the start of the slot and zeros the rest, and
  // takes the whole slot from the write budget as it is submitted; a read
  // fills them from the start of the slot, the whole payload while an
  // adopted slot is not yet trusted. A write is null once the file accepts
  // no further writes, as acquire() is null at the limit; the write budget
  // never makes it null.
  [[nodiscard]] std::shared_ptr<Operation> write(
      std::shared_ptr<Slot> slot, std::vector<std::span<const std::byte>> source,
      std::function<void()> completion);
  [[nodiscard]] std::shared_ptr<Operation> read(
      std::shared_ptr<Slot> slot, std::vector<std::span<std::byte>> destination,
      std::function<void()> completion);
  // Reads `bytes` at `offset` of a slot's payload on the calling thread,
  // beside the worker, with one uncached pread into `buffer`: 4 KiB-aligned
  // memory that holds the 4 KiB-aligned range around the bytes, widened to
  // the whole segments it touches in an adopted slot, whose checksums it
  // checks; a buffer of slotBytes() always holds it. Returns the bytes within
  // `buffer`, or nothing for a slot whose write has not landed, bytes past
  // its payload, a buffer that cannot hold the range, a short or failed read
  // or a segment that does not match. The caller holds the slot while it
  // reads, and no write of it runs meanwhile.
  [[nodiscard]] std::optional<std::span<const std::byte>>
  readAt(const Slot &slot, uint64_t offset, uint64_t bytes, std::span<std::byte> buffer) const noexcept;

private:
  // Runs on the worker, moving the slot straight or through the worker's
  // buffer.
  using Run = std::function<bool(std::span<std::byte>, const std::atomic<bool> &)>;
  struct Work {
    std::shared_ptr<Operation> operation;
    Run run;
    std::function<void()> completion;
    // A read: the writes ahead of it hold back for no other reads.
    bool read = false;
  };
  struct Free {
    void operator()(std::byte *memory) const noexcept;
  };
  [[nodiscard]] std::shared_ptr<Operation> submit(Run run, std::function<void()> completion,
                                                  bool read = false);
  // The checks and the setup both constructors share before the file opens;
  // each starts the worker last.
  void prepare(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget, const Options &options);
  void uncache();
  // Stops the file's writes; true for the call that stopped them.
  static bool stop(Backing &backing) noexcept;
  // Opens a persistent file and reads its records, or empties it when it
  // was written for something else.
  void open(const Persistence &persistence);
  void run();
  // On the worker: clears a freed slot's record and returns its blocks to
  // the volume.
  void punchHole(uint32_t index) noexcept;
  // Holds the worker's queue too, so a slot freed on any thread queues the
  // return of its blocks.
  std::shared_ptr<Backing> backing_;
  // The worker's own, aligned for uncached IO: every chunk that does not
  // move straight between memory and the file goes through it.
  std::unique_ptr<std::byte, Free> buffer_;
  std::thread worker_;
  std::vector<SlotRecord> records_;
  // Indices adopted until finishAdoption().
  std::vector<bool> adopted_;
};

} // namespace splash::model
