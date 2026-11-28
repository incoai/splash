#pragma once

#include "AwakeClock.hpp"
#include "Checked.hpp"

#include <algorithm>
#include <array>
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
// drops the last handle of a slot. A slot counts what it keeps on the volume
// (SlotFile::quotaBytesFor): the host pages its write stores, which the write
// budget charges too. The budget bounds live slots by its limit: the
// quota at most, and less while the volume the files live on leaves less
// (engine::DiskSpace sets it). A limit below what the tier holds is met by
// the cache giving copies up (Cache::keepDiskLimit), so live slots can exceed
// it for a while, as slots a persistent file adopted can. A freed slot
// returns its bytes at once but its blocks only when its file's worker
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
// A write budget that spares the drive is the user's to set (setWriteBudget,
// --max-cache-disk-writes); without one, the default, every write passes and
// nothing is held back. With one, every write a file of the tier submits
// draws on it: a token bucket of this process, counted at submission in the
// host pages each write stores, which each start begins full. It is a cap on
// the cache's writes: while it is spent the cache writes nothing, no state,
// no KV page and no restore point's copy, and keeps to memory as without a
// tier until it has refilled (engine::Cache::takesCacheWrites), so a write
// overdraws it by one write at most; once overdrawn it stays closed until
// its balance is back at an eighth of its burst (kReopenShare). Only a Spill
// request's writes, which its admission paid for (admitsSpill), pass it.
// Writes are submitted on the engine thread, which alone reads the budget
// (admits), so a write the engine admitted is never refused for it.
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
    spent_ = false;
  }
  // Whether a cache write may start now: until a write overdraws the
  // budget, then once it has refilled to an eighth of its burst
  // (kReopenShare). Engine thread.
  [[nodiscard]] bool admits() const noexcept {
    return !refillPerMillisecond_ || writeBudgetAt(clock_()) >= 0.0;
  }
  // Whether a request may start to spill, or convert to it, when Spill
  // requests already committed `outstandingBytes` of writes: without a
  // write budget, or while its balance covers them. Their writes are in use
  // and pass whatever the balance. Engine thread.
  [[nodiscard]] bool admitsSpill(uint64_t outstandingBytes) const noexcept {
    return !refillPerMillisecond_ || balanceAt(clock_()) >= static_cast<double>(outstandingBytes);
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
  // When a cache write may start again, on the budget's clock; empty while
  // one may start now.
  [[nodiscard]] std::optional<double> admitsAgainAt() const noexcept {
    if (!refillPerMillisecond_)
      return std::nullopt;
    const double now = clock_();
    const double budget = writeBudgetAt(now);
    if (budget >= 0.0)
      return std::nullopt;
    return now - budget / refillPerMillisecond_;
  }
  // What the next cache write may take now; negative while none may start
  // (admits), by what the balance lacks of the level it reopens at; zero
  // without a budget.
  [[nodiscard]] int64_t writeBudgetBytes() const noexcept {
    return refillPerMillisecond_ ? static_cast<int64_t>(writeBudgetAt(clock_())) : 0;
  }

  // The write rate: what the tier's writes took over the last hour and the
  // last day, counted as the write budget counts them, in the host pages
  // each write stores and on its clock, with a write budget or without. The
  // day's bytes are what a write budget would have had to cover, so they
  // are the number to set one from and to hold against the drive's rated
  // endurance; the hour's are what the load of the moment writes, 24 times
  // which projects a day before one has passed. The cumulative count alone
  // (writtenBytes) says neither without a second reading and the time
  // between them: a deployment that read its count after 22 h found 1.85 TB,
  // 7 times its quota a day. Each window ends now and is exact to a minute
  // of the clock; a process younger than a window has written all of its
  // count. Engine thread.
  struct WriteRate final {
    uint64_t hourBytes = 0;
    uint64_t dayBytes = 0;
  };
  [[nodiscard]] WriteRate writeRate() const noexcept {
    const uint64_t minute = minuteAt(clock_());
    return {chargedWithin(kHourMinutes, minute), chargedWithin(kDayMinutes, minute)};
  }

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
  // Bytes the budget's files hold on disk: for each slot's range, what the
  // slot written there since its blocks last went back counts
  // (SlotFile::quotaBytesFor), the most of them where a punch failed between
  // two, and the whole range where it lies in a hole a write past the
  // file's end left whose punch failed.
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
  // An overdrawn budget reopens once its balance is back at this share of
  // its burst, so that a read-only spell ends in writes worth reopening
  // for, not in one write the next reclaim discards: 45 minutes of refill
  // at any budget, then 640 MiB at 20 GiB a day.
  static constexpr double kReopenShare = 1.0 / 8.0;

  static constexpr double kMinuteMilliseconds = 60'000.0;
  static constexpr uint64_t kHourMinutes = 60;
  static constexpr uint64_t kDayMinutes = 1440;

  // The balance at `now` on the budget's clock.
  [[nodiscard]] double balanceAt(double now) const noexcept {
    return std::min(burstBytes_, balance_ + refillPerMillisecond_ * (now - balanceTime_));
  }
  // The balance at `now` less the level a spent budget reopens at, while
  // it is below it: negative while no cache write may start.
  [[nodiscard]] double writeBudgetAt(double now) const noexcept {
    const double balance = balanceAt(now), reopen = burstBytes_ * kReopenShare;
    return spent_ && balance < reopen ? balance - reopen : balance;
  }
  [[nodiscard]] static uint64_t minuteAt(double now) noexcept {
    return static_cast<uint64_t>(now / kMinuteMilliseconds);
  }
  // What was charged in the `minutes` minutes of the clock that end with
  // `minute`. The minutes after the last charge took nothing: their places
  // still hold what was charged a day before them.
  [[nodiscard]] uint64_t chargedWithin(uint64_t minutes, uint64_t minute) const noexcept {
    uint64_t bytes = 0;
    for (uint64_t ago = minute - chargedMinute_; ago < minutes && ago <= minute; ++ago)
      bytes += charged_[(minute - ago) % kDayMinutes];
    return bytes;
  }
  // The host pages a write stores, as SlotFile::write submits it.
  void charge(uint64_t bytes) noexcept {
    const double now = clock_();
    const uint64_t minute = minuteAt(now);
    // The minutes since the last charge took nothing.
    for (uint64_t idle = chargedMinute_ + 1; idle <= minute && idle - chargedMinute_ <= kDayMinutes; ++idle)
      charged_[idle % kDayMinutes] = 0;
    chargedMinute_ = minute;
    charged_[minute % kDayMinutes] += bytes;
    if (!refillPerMillisecond_)
      return;
    // A write admitted at the level a spent budget reopens at ends its
    // spell; one that overdraws the budget, a Spill request's too, starts
    // the next.
    spent_ = writeBudgetAt(now) < 0.0;
    balance_ = balanceAt(now) - static_cast<double>(bytes);
    balanceTime_ = now;
    spent_ = spent_ || balance_ < 0.0;
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
  // Overdrawn since the balance was last at the level it reopens at.
  bool spent_ = false;
  // The write rate, on the engine thread: what each of the last day's
  // minutes of the clock was charged, a minute's at its number modulo the
  // day's, and the minute of the last charge.
  std::array<uint64_t, kDayMinutes> charged_{};
  uint64_t chargedMinute_ = 0;
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
// A slot is acquired for a payload and takes of the quota what it keeps on
// the volume (quotaBytesFor): the payload's host pages, the rest of the slot
// left a hole, as a freed slot is. The volume does not leave a hole by
// itself: a write past what the file held has the hole before it allocated
// (APFS filled gaps of 1 to 16 MiB, so 256 payloads of 462 KiB in 1.59 MiB
// slots, 1.7.3's KV pages past every window, took 407 MiB until their slots
// were freed and written again, then 116 MiB), so such a write
// returns that hole, and a file ends with its last slot that holds blocks,
// since a hole at its end is allocated whole under a write into it. A
// volume that returns no blocks keeps whole slots, and there a slot counts
// whole.
//
// A temporary file is unlinked at once and gone with the process. A
// persistent file stays, with a record file beside it: one fixed-size record
// per slot, written by the worker after the slot's write has landed, from
// the label its writer gives the slot and the CRC-32C of each segment of the
// payload, which the worker computed while writing it. A freed slot's record
// is cleared before its blocks return. The next process reads the records,
// adopts the slots it keeps, and frees the others. A read of an adopted
// slot fails unless each segment it covers matches its record, and once a
// read covered and matched the whole payload the slot is trusted; so a slot
// that changed after its record was written, or whose blocks never reached
// the drive, is a cache miss and nothing worse. Slots this process wrote are
// trusted as a temporary file's are.
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

  // The host pages a payload of payloadBytes fills, in bytes: what a write of
  // it stores, zeros to the end of its last page, and the size of the slots
  // of a file whose largest payload it is.
  [[nodiscard]] static constexpr uint64_t slotBytesFor(uint64_t payloadBytes) noexcept {
    return alignUp(payloadBytes);
  }
  // What a slot holding a payload of payloadBytes takes of the quota, the
  // blocks it keeps on the volume: the payload's host pages while the volume
  // returns blocks, the whole slot once a punch failed, as on a volume that
  // has no holes (HFS+, exFAT), where the slot keeps what an earlier payload
  // or the volume itself stored around the payload.
  [[nodiscard]] uint64_t quotaBytesFor(uint64_t payloadBytes) const noexcept;

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
    // What the slot takes of the quota while it is held (quotaBytesFor of
    // the payload it was acquired for or adopted with).
    [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

  private:
    friend class SlotFile;
    Slot(std::shared_ptr<Backing> backing, uint32_t index);
    std::shared_ptr<Backing> backing_;
    uint32_t index_;
    uint64_t bytes_ = 0;
    uint64_t id_;
    // Set by the worker, which runs a file's operations in submission
    // order, once a write landed; read beside it too (readAt), which then
    // finds the fields below the write's.
    std::atomic<bool> written_{false};
    // The size of the payload the last complete write stored and, in a
    // persistent file, the CRC-32C of each of its segments. Reads of an
    // adopted slot check the segments they cover against them while verify_
    // holds, which the first read that matched the whole payload clears.
    uint64_t payloadBytes_ = 0;
    std::vector<uint32_t> checksums_;
    mutable std::atomic<bool> verify_{false};
    // Once a read found it failing its record's checksums, which the log
    // says once (reportDamage).
    mutable std::atomic<bool> damaged_{false};
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
  // A slot for a payload of payloadBytes, at most a slot's. Null when the
  // budget's limit for the draw has no room for what it takes
  // (quotaBytesFor, DiskBudget::reserve).
  [[nodiscard]] std::shared_ptr<Slot> acquire(uint64_t payloadBytes,
                                              DiskBudget::Draw draw = DiskBudget::Draw::Cache);
  [[nodiscard]] uint64_t slotBytes() const noexcept;
  // False once a write has failed for another reason than a full volume;
  // complete slots stay readable.
  [[nodiscard]] bool writable() const noexcept;
  // The file takes a state's write now: it is writable, the limit holds a
  // slot, so making room can find one, and the write budget admits a cache
  // write (DiskBudget::admits). Engine thread.
  [[nodiscard]] bool takesWrite() const noexcept;
  [[nodiscard]] bool persistent() const noexcept;
  // The bytes the file's blocks take on its volume, its records' included;
  // zero when they cannot be measured.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;

  // The slots earlier processes left in a persistent file, as read when it
  // opened: empty for a temporary or a new one.
  [[nodiscard]] const std::vector<SlotRecord> &records() const noexcept { return records_; }
  // A persistent file found shorter than its records reach as it opened:
  // its bytes and where its records reach, its records and those that lay
  // past its end, which records() leaves out. None for a whole file.
  struct CutShort final {
    std::string path;
    uint64_t fileBytes = 0;
    uint64_t reachBytes = 0;
    uint32_t records = 0;
    uint32_t cut = 0;
  };
  [[nodiscard]] const std::optional<CutShort> &cutShort() const noexcept { return cutShort_; }
  // Takes back the slot a record names, before finishAdoption(). It counts
  // against the budget as a slot acquired for the record's payload does,
  // even past the capacity, so that the caller can give up what the quota
  // no longer holds; its reads fail unless what they cover matches the
  // record.
  [[nodiscard]] std::shared_ptr<Slot> adopt(const SlotRecord &record);
  // Ends adoption: every slot not adopted is freed and its record cleared,
  // and every block no adopted payload holds goes back, those slots' and
  // what the volume kept around a shorter payload, as the files of a
  // process that returned no such hole hold it. A record that cannot be
  // cleared stays, reported, until its slot is reused; its punched payload
  // fails any later check. acquire() needs it in a persistent file.
  void finishAdoption();
  // Keeps the label in the slot's record once the write submitted before it
  // has landed; a slot whose write failed gets no record. A label is the
  // file's label size. Nothing happens in a temporary file.
  void label(std::shared_ptr<Slot> slot, std::vector<std::byte> label);
  // Flushes the slots and records written so far to the drive.
  [[nodiscard]] std::shared_ptr<Operation> synchronize(std::function<void()> completion);
  // The process is ending: a slot dropped from now on keeps its blocks and
  // its record for the next one, which frees them (finishAdoption), and so
  // does one whose punch has not started by `punchesEnd`, the earliest any
  // seal gave: the worker starts no punch past it, so that an operation
  // after the punches waits past it for one punch at most.
  void seal(AwakeClock::time_point punchesEnd = AwakeClock::time_point::max()) noexcept;
  // The spans total at most one slot, a write's no more host pages than its
  // slot was acquired for and exactly the segments in a file of several, and
  // stay valid until the operation is ready. A write stores them in order
  // from the start of the slot, zeros the rest of their last host page, and
  // takes those pages from the write budget as it is submitted; a read fills
  // them from the start of the slot, at most those pages and exactly the
  // payload while an adopted slot is not yet trusted, and fails otherwise. A
  // write is null once the file accepts no further writes, as acquire() is
  // null at the limit; the write budget never makes it null.
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
  // Once the slot file is open, before any slot counts: whether its volume
  // returns blocks (quotaBytesFor).
  void askVolume();
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
  std::optional<CutShort> cutShort_;
  // Until finishAdoption(): the host pages of the payload each index was
  // adopted with, in bytes; zero for one not adopted.
  std::vector<uint64_t> adopted_;
};

// The damage persistent cache files showed in this process: copies whose
// reads failed their checksums, and records that lay past the end of a file
// found cut short as it opened (/status disk.damage).
struct SlotFileDamage final {
  uint64_t checksumFailures = 0;
  uint64_t cutCopies = 0;
};
[[nodiscard]] SlotFileDamage slotFileDamage() noexcept;

} // namespace splash::model
