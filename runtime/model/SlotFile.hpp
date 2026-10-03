#pragma once

#include "Checked.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
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
      if (bytes > capacity_ - used)
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

// Scratch storage of fixed-size slots in an unlinked temporary file, served
// by one IO worker in submission order and bounded by a disk budget. Callers
// own the memory an operation moves and keep it alive until the operation is
// ready. That memory may have any size and alignment. Whole 1 MiB chunks of
// a span that start at an aligned offset of the slot, in memory aligned
// like slot offsets, move straight between memory and the file; everything
// else, such as the rest of a span short of a chunk, moves through an
// aligned buffer of the worker's own, so the file sees only transfers
// aligned to the host page (kHostPageBytes), as uncached IO wants; a slot is
// a whole number of host pages. A slot is readable only after one complete
// write; a failed or cancelled write leaves it unreadable, and after a
// failed write the file accepts no further writes. A freed slot returns its
// quota at once; one that was written returns its blocks to the volume
// (F_PUNCHHOLE) on the worker, after the operation in flight and before any
// later one (DiskBudget). So that a file-size limit fails a write rather than
// killing the process, a file ignores SIGXFSZ from its construction on.
class SlotFile final {
  struct Backing;

public:
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
  // error. Files sharing a budget compete for its bytes. The file goes to
  // the temporary directory.
  SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget);
  ~SlotFile();
  SlotFile(const SlotFile &) = delete;
  SlotFile &operator=(const SlotFile &) = delete;
  // Null when the budget is exhausted.
  [[nodiscard]] std::shared_ptr<Slot> acquire();
  [[nodiscard]] uint64_t slotBytes() const noexcept;
  // False once a write has failed; complete slots stay readable.
  [[nodiscard]] bool writable() const noexcept;
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
  void run();
  // On the worker: returns a freed slot's blocks to the volume.
  void punchHole(uint32_t index) noexcept;
  // Holds the worker's queue too, so a slot freed on any thread queues the
  // return of its blocks.
  std::shared_ptr<Backing> backing_;
  // The worker's own, aligned for uncached IO: every chunk that does not
  // move straight between memory and the file goes through it.
  std::unique_ptr<std::byte, Free> buffer_;
  std::thread worker_;
};

} // namespace splash::model
