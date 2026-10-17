#pragma once

// The reader threads of the weights that stay in their source files and are
// read as steps need them (model::ExpertStore, model::SparseRowStore), which
// a model's stores share: batches of tasks, each reading into memory of its
// own.
//
// A batch either runs now (run, finish), on the calling thread and every
// reader, ahead of every background batch, because a step waits for it; or
// in the background (start), on at most the readers it was given, and only
// while no batch runs now, so a read ahead never delays a step's reads. The
// store that started a background batch ends it: finish() runs the rest of it
// now, and stop() leaves the rest unread and returns once the tasks readers
// took ended, the batch's first taken() tasks; halt() leaves the rest unread
// at once, and its taken tasks end on their own (Batch::ended).
//
// A task that throws is tried once more, for a transient I/O error; a second
// failure is the task's. A batch that runs now throws its first failure once
// every task ended; a background batch keeps its failures for its store
// (failed), which decides whether the read was needed: a read ahead of what
// no step routed is no failure.
//
// Threads: any thread runs, starts, stops and finishes batches; a batch's
// tasks run on the readers and on the threads that run it now, each once.
// A store stops or finishes its background batches before the memory they
// read into goes, and the readers before they go.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace splash::model {

// A model's readers: an SSD reads small pieces at full speed only with many
// in flight (an M5 Pro's reads 460 KiB pieces at 8.1 GB/s from four threads
// and at 12.5 GB/s from sixteen).
inline constexpr unsigned kWeightReaders = 16;

class WeightReaders final {
public:
  using Task = std::function<void(size_t index)>;
  // A batch's tasks tried again after a failure, and those that failed again.
  struct Tally final {
    uint64_t retries = 0, failures = 0;
  };

  class Batch final {
  public:
    // The tasks readers took before stop(): [0, taken()) ran, each once;
    // after finish(), every task.
    [[nodiscard]] size_t taken() const noexcept { return limit_.load(std::memory_order_acquire); }
    // Whether task `index` ended, its last try.
    [[nodiscard]] bool ended(size_t index) const noexcept { return ends_[index].load(std::memory_order_acquire); }
    // Whether task `index` failed twice; call once it ended.
    [[nodiscard]] bool failed(size_t index) const;
    // Its first failure, or null.
    [[nodiscard]] std::exception_ptr failure() const;
    [[nodiscard]] Tally tally() const;

  private:
    friend class WeightReaders;
    Batch(size_t count, Task task, unsigned readers, std::function<void(const Batch &)> ended)
        : count_(count), task_(std::move(task)), readers_(readers), ended_(std::move(ended)),
          ends_(std::make_unique<std::atomic<bool>[]>(count)), limit_(count) {}
    // Takes the next task; false once none is left.
    bool claim(size_t &index) noexcept;
    // Task `index` ended, `failure` if it failed.
    void end(size_t index, std::exception_ptr failure);
    [[nodiscard]] bool exhausted() const noexcept { return next_.load(std::memory_order_relaxed) >= limit_; }
    void wait();

    const size_t count_;
    const Task task_;
    // The most readers on it at once while it runs in the background, and
    // those on it (under WeightReaders::mutex_).
    const unsigned readers_;
    unsigned working_ = 0;
    // Runs on the thread that ended its last task, when every task ran;
    // wait() returns once it has (endedReturned_, under mutex_).
    const std::function<void(const Batch &)> ended_;
    bool endedReturned_ = false;
    // Each task's end (ended).
    const std::unique_ptr<std::atomic<bool>[]> ends_;
    std::atomic<size_t> next_{0}, done_{0};
    // count_, or the tasks taken before stop().
    std::atomic<size_t> limit_;
    std::atomic<uint64_t> retries_{0};
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::pair<size_t, std::exception_ptr>> failures_;
  };

  // Starts `readers` threads.
  explicit WeightReaders(unsigned readers);
  // Ends the readers once the tasks they took ended; every background batch
  // was stopped or finished.
  ~WeightReaders();
  WeightReaders(const WeightReaders &) = delete;
  WeightReaders &operator=(const WeightReaders &) = delete;

  // Runs task(i) for every i below count now (see above); throws the first
  // failure once every task ended, and counts its retries and failures in
  // `tally`, if any.
  void run(size_t count, const Task &task, Tally *tally = nullptr);
  // Starts task(i) for every i below count in the background, in order, on
  // at most `readers` readers at once; `ended`, if any, runs with the batch
  // on the thread that ends its last task once every task ran (never when
  // readers were stopped before taking them all), and stop() and finish()
  // return once it has.
  [[nodiscard]] std::shared_ptr<Batch> start(size_t count, Task task, unsigned readers,
                                             std::function<void(const Batch &)> ended = {});
  // Readers take no more of the batch's tasks; returns once those they took
  // ended (Batch::taken).
  void stop(Batch &batch);
  // Readers take no more of the batch's tasks; returns at once, while those
  // they took may still run (Batch::ended). stop() then waits for them.
  void halt(Batch &batch);
  // Runs the batch's remaining tasks now, and returns once all of them
  // ended (none, once stopped) and its `ended` returned; its failures stay
  // its own (Batch::failed).
  void finish(Batch &batch);

  [[nodiscard]] unsigned readers() const noexcept { return static_cast<unsigned>(threads_.size()); }

private:
  void readerLoop();
  // Runs the batch's tasks on this thread until none is left, or, from the
  // background, until a batch runs now.
  void work(const std::shared_ptr<Batch> &batch, bool background);
  // Runs `batch` now: queued ahead of the background, here and on the
  // readers.
  void runNow(const std::shared_ptr<Batch> &batch);
  void attempt(Batch &batch, size_t index);
  // Wakes up to `count` readers for a batch queued under mutex_.
  void wakeReaders(size_t count);

  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<Batch>> now_, background_;
  // Batches running now, from their start until their last task ended.
  std::atomic<unsigned> running_{0};
  bool stopping_ = false;
  std::vector<std::thread> threads_;
};

} // namespace splash::model
