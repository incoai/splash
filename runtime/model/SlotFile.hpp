#pragma once

#include "Checked.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace splash::model {

// Bytes one disk quota may hold, shared by every slot file of the cache
// tier. Reservations and releases come from the engine thread and from
// whoever drops the last handle of a slot. The budget bounds live slots. A
// freed slot returns its bytes at once but its blocks only when its file's
// worker punches them, after the transfer that worker is in, so the files'
// blocks can briefly exceed the budget by the slots freed but not yet
// punched; a volume sized exactly to the budget can fill.
class DiskBudget final {
public:
  explicit DiskBudget(uint64_t capacityBytes) noexcept : capacity_(capacityBytes) {}
  [[nodiscard]] uint64_t capacityBytes() const noexcept { return capacity_; }
  [[nodiscard]] uint64_t usedBytes() const noexcept {
    return used_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] bool reserve(uint64_t bytes) noexcept {
    uint64_t used = used_.load(std::memory_order_relaxed);
    do {
      // Slots a persistent file adopted may hold more than the capacity.
      if (used > capacity_ || bytes > capacity_ - used)
        return false;
    } while (!used_.compare_exchange_weak(used, used + bytes, std::memory_order_relaxed));
    return true;
  }
  void release(uint64_t bytes) noexcept { used_.fetch_sub(bytes, std::memory_order_relaxed); }
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

private:
  friend class SlotFile;
  uint64_t capacity_;
  std::atomic<uint64_t> used_{0};
  std::atomic<uint64_t> read_{0};
  std::atomic<uint64_t> written_{0};
  std::atomic<uint64_t> file_{0};
};

// One slot a persistent file keeps from an earlier process, as its record
// says: where it is, the size and CRC-32C of the payload its last complete
// write stored, and the label its writer gave it.
struct SlotRecord final {
  uint32_t index = 0;
  uint64_t payloadBytes = 0;
  uint32_t checksum = 0;
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
// it unreadable, and after a failed write the file accepts no further
// writes. A freed slot returns its quota at once; one that was written
// returns its blocks to the volume (F_PUNCHHOLE) on the worker, after the
// operation in flight and before any later one (DiskBudget). So that a
// file-size limit fails a write rather than killing the process, a file
// ignores SIGXFSZ from its construction on.
//
// A temporary file is unlinked at once and gone with the process. A
// persistent file stays, with a record file beside it: one fixed-size record
// per slot, written by the worker after the slot's write has landed, from
// the label its writer gives the slot and the CRC-32C of the payload the
// worker computed while writing it. A freed slot's record is cleared before
// its blocks return. The next process reads the records, adopts the slots it
// keeps, and frees the others; the first read of an adopted slot fails
// unless its payload matches its record, so a slot that changed after its
// record was written, or whose blocks never reached the drive, is a cache
// miss and nothing worse. Slots this process wrote are trusted as a
// temporary file's are.
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

  class Slot final {
  public:
    ~Slot();
    Slot(const Slot &) = delete;
    Slot &operator=(const Slot &) = delete;

  private:
    friend class SlotFile;
    Slot(std::shared_ptr<Backing> backing, uint32_t index);
    std::shared_ptr<Backing> backing_;
    uint32_t index_;
    // Owned by the worker: operations on one file run in submission order.
    bool written_ = false;
    // In a persistent file, the size and CRC-32C of the payload the last
    // complete write stored; an adopted slot checks its first read against
    // them (verify_). Owned by the worker too.
    uint64_t payloadBytes_ = 0;
    uint32_t checksum_ = 0;
    bool verify_ = false;
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
  SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget);
  // A persistent file: opening it reads the records an earlier process left,
  // and it takes no slot of its own until finishAdoption().
  SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget,
           const Persistence &persistence);
  ~SlotFile();
  SlotFile(const SlotFile &) = delete;
  SlotFile &operator=(const SlotFile &) = delete;
  // Null when the budget is exhausted.
  [[nodiscard]] std::shared_ptr<Slot> acquire();
  [[nodiscard]] uint64_t slotBytes() const noexcept;
  // False once a write has failed; complete slots stay readable.
  [[nodiscard]] bool writable() const noexcept;
  [[nodiscard]] bool persistent() const noexcept;

  // The slots earlier processes left in a persistent file, as read when it
  // opened: empty for a temporary or a new one.
  [[nodiscard]] const std::vector<SlotRecord> &records() const noexcept { return records_; }
  // Takes back the slot a record names, before finishAdoption(). It counts
  // against the budget even past its capacity, so that the caller can give
  // up what the quota no longer holds; its first read fails unless the
  // payload matches the record.
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
  // The spans total at most one slot and stay valid until the operation is
  // ready. A write stores them in order from the start of the slot and zeros
  // the rest; a read fills them from the start of the slot. A write is null
  // once the file accepts no further writes, as acquire() is null when the
  // quota is full.
  [[nodiscard]] std::shared_ptr<Operation> write(
      std::shared_ptr<Slot> slot, std::vector<std::span<const std::byte>> source,
      std::function<void()> completion);
  [[nodiscard]] std::shared_ptr<Operation> read(
      std::shared_ptr<Slot> slot, std::vector<std::span<std::byte>> destination,
      std::function<void()> completion);

private:
  // Runs on the worker, moving the slot straight or through the worker's
  // buffer.
  using Run = std::function<bool(std::span<std::byte>, const std::atomic<bool> &)>;
  struct Work {
    std::shared_ptr<Operation> operation;
    Run run;
    std::function<void()> completion;
  };
  struct Free {
    void operator()(std::byte *memory) const noexcept;
  };
  [[nodiscard]] std::shared_ptr<Operation> submit(Run run,
                                                  std::function<void()> completion);
  // The checks and the setup both constructors share before the file opens;
  // each starts the worker last.
  void prepare(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget);
  void uncache();
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
