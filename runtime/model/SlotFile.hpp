#pragma once

#include "model/CacheStore.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace splash::model {

// Bytes one disk quota may hold, shared by every slot file of the cache
// tier. Reservations and releases come from the engine thread and from
// whoever drops the last handle of a slot.
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
  // Payload bytes moved by the backend. Scratch IO includes partial work;
  // the transactional backend counts completed slot operations.
  // This is application IO, not physical SSD traffic or filesystem overhead.
  [[nodiscard]] uint64_t readBytes() const noexcept {
    return read_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t writtenBytes() const noexcept {
    return written_.load(std::memory_order_relaxed);
  }

private:
  friend class SlotFile;
  uint64_t capacity_;
  std::atomic<uint64_t> used_{0};
  std::atomic<uint64_t> read_{0};
  std::atomic<uint64_t> written_{0};
};

// A manifest reserves its complete additional footprint before any write.
// Consuming a reservation moves its charge to a slot; destruction returns only
// unused bytes. Slot charges live until the last IO/read handle is released.
class DiskReservation final {
public:
  static std::optional<DiskReservation>
  acquire(std::shared_ptr<DiskBudget> budget, uint64_t bytes);
  DiskReservation(DiskReservation &&other) noexcept;
  DiskReservation &operator=(DiskReservation &&other) noexcept;
  ~DiskReservation();
  DiskReservation(const DiskReservation &) = delete;
  DiskReservation &operator=(const DiskReservation &) = delete;
  [[nodiscard]] uint64_t remainingBytes() const noexcept { return bytes_; }
  [[nodiscard]] const std::shared_ptr<DiskBudget> &budget() const noexcept {
    return budget_;
  }
  // Called only after a slot is ready to take ownership of these bytes.
  void consume(uint64_t bytes);

private:
  DiskReservation(std::shared_ptr<DiskBudget> budget, uint64_t bytes)
      : budget_(std::move(budget)), bytes_(bytes) {}
  std::shared_ptr<DiskBudget> budget_;
  uint64_t bytes_ = 0;
};

// Fixed-size slots in an unlinked temporary file or a shared CacheStore, served
// by one IO worker in submission order and bounded by a disk budget. Callers
// own the memory an operation moves and keep it alive until the operation is
// ready. A slot is readable only after one complete write; a failed or
// cancelled write leaves it unreadable, and after a failed write the file
// accepts no further writes. So that a file-size limit fails a write rather
// than killing the process, a file ignores SIGXFSZ from its construction on.
class SlotFile final {
  struct Backing;

public:
  // Slot offsets stay aligned to this for uncached IO.
  static constexpr uint64_t kAlignmentBytes = 16384;

  class Slot final {
  public:
    ~Slot();
    [[nodiscard]] uint64_t recordId() const noexcept { return recordId_; }
    [[nodiscard]] uint64_t bytes() const noexcept;
    // Ownership changes run on the engine thread. IO captures its accounting
    // budget at submission and never reads this mutable ownership.
    [[nodiscard]] bool durable() const noexcept;
    void transfer(DiskReservation &reservation);
    [[nodiscard]] bool returnTemporary();
    [[nodiscard]] bool reusable() const noexcept { return reusable_; }
    void retire() noexcept { reusable_ = false; }
    Slot(const Slot &) = delete;
    Slot &operator=(const Slot &) = delete;

  private:
    friend class SlotFile;
    Slot(std::shared_ptr<Backing> backing, uint32_t index);
    std::shared_ptr<Backing> backing_;
    uint32_t index_;
    uint64_t recordId_ = 0;
    std::shared_ptr<DiskBudget> budget_;
    bool reusable_ = true;
    // Owned by the worker: operations on one file run in submission order.
    bool written_ = false;
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

  // The default quota belongs only to temporary offload. A zero quota is
  // allowed with a CacheStore: durable writes use an explicit reservation.
  SlotFile(uint64_t slotBytes, std::shared_ptr<DiskBudget> budget,
           const std::filesystem::path &directory =
               std::filesystem::temp_directory_path(),
           std::shared_ptr<CacheStore> store = nullptr);
  // Another payload geometry in the same quota and persistent store.
  [[nodiscard]] std::shared_ptr<SlotFile> sibling(uint64_t slotBytes) const;
  // A file with a budget of its own.
  SlotFile(uint64_t slotBytes, uint64_t capacityBytes,
           const std::filesystem::path &directory = std::filesystem::temp_directory_path());
  ~SlotFile();
  SlotFile(const SlotFile &) = delete;
  SlotFile &operator=(const SlotFile &) = delete;
  // Null when the budget is exhausted.
  [[nodiscard]] std::shared_ptr<Slot> acquire();
  [[nodiscard]] std::shared_ptr<Slot> acquire(DiskReservation &reservation);
  // Startup only: adopts an already committed record, charging it once.
  [[nodiscard]] std::shared_ptr<Slot> reopen(uint64_t id,
                                             DiskReservation &reservation);
  // Metadata commits share the worker with state writes. No engine-thread IO.
  [[nodiscard]] std::shared_ptr<Operation>
  metadata(std::function<void()> work, std::function<void()> completion = {});
  [[nodiscard]] uint64_t slotBytes() const noexcept;
  // The shared budget's capacity and use.
  [[nodiscard]] uint64_t capacityBytes() const noexcept;
  [[nodiscard]] uint64_t usedBytes() const noexcept;
  [[nodiscard]] uint64_t readBytes() const noexcept;
  [[nodiscard]] uint64_t writtenBytes() const noexcept;
  // False once a write has failed; complete slots stay readable.
  [[nodiscard]] bool writable() const noexcept;
  // True when the worker holds nothing: no operation queued and none running.
  // Owners drain their own operations; tests use this to check that none is
  // left.
  [[nodiscard]] bool idle() const;
  // The spans total one slot and stay valid until the operation is ready.
  [[nodiscard]] std::shared_ptr<Operation> write(
      std::shared_ptr<Slot> slot, std::vector<std::span<const std::byte>> source,
      std::function<void()> completion);
  [[nodiscard]] std::shared_ptr<Operation> read(
      std::shared_ptr<Slot> slot, std::vector<std::span<std::byte>> destination,
      std::function<void()> completion);

private:
  std::shared_ptr<Slot> allocate(DiskReservation &reservation);
  struct Work {
    std::shared_ptr<Operation> operation;
    std::function<bool(const std::atomic<bool> &)> run;
    std::function<void()> completion;
  };
  [[nodiscard]] std::shared_ptr<Operation> submit(
      std::function<bool(const std::atomic<bool> &)> work,
      std::function<void()> completion);
  void run();
  std::shared_ptr<Backing> backing_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Work> work_;
  bool running_ = false;
  bool stopping_ = false;
  std::thread worker_;
};

} // namespace splash::model
