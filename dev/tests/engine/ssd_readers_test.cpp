// The engine's SSD readers (model::SsdReaders) run every task of a batch
// once: a batch that runs now on the caller and every reader, a background
// batch on at most its readers, the oldest first; a failed task is tried
// once more, and a second failure is the batch's (a batch that runs now
// throws it once every task ended); stop() leaves the tasks no reader took
// unrun and returns once those taken ended, and the batch's end once every
// task ran, halt() returns at once and each taken task tells when it ended,
// finish() runs the rest now and returns once the batch's end has; a batch
// that runs now never waits behind a background batch's queue, and the pool
// is busy for writes exactly while one runs, as a probe that outlives it
// says; and several threads run, start, stop and finish batches at once.
// Built under TSan too.
#include "TestChecks.hpp"
#include "model/SsdReaders.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace splash::model;
using splash::test::require;

namespace {


// Runs of each task of a batch.
struct Runs final {
  explicit Runs(size_t count) : counts(count) {
    for (std::atomic<int> &value : counts) value = 0;
  }
  std::vector<std::atomic<int>> counts;
  [[nodiscard]] bool eachOnce(size_t from, size_t to) const {
    for (size_t index = from; index < to; ++index)
      if (counts[index].load() != 1) return false;
    return true;
  }
  [[nodiscard]] bool noneFrom(size_t from) const {
    for (size_t index = from; index < counts.size(); ++index)
      if (counts[index].load()) return false;
    return true;
  }
};

// A gate tasks wait at until it opens.
class Gate final {
public:
  void wait() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [this] { return open_; });
  }
  void open() {
    {
      const std::lock_guard lock(mutex_);
      open_ = true;
    }
    changed_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool open_ = false;
};

void runsEveryTaskOnce() {
  for (const unsigned readers : {0u, 1u, 8u}) {
    SsdReaders pool(readers);
    Runs runs(1000);
    std::mutex threadsMutex;
    std::vector<std::thread::id> threads;
    pool.run(runs.counts.size(), [&](size_t index) {
      ++runs.counts[index];
      const std::lock_guard lock(threadsMutex);
      if (std::find(threads.begin(), threads.end(), std::this_thread::get_id()) == threads.end())
        threads.push_back(std::this_thread::get_id());
    });
    require(runs.eachOnce(0, runs.counts.size()), "a batch run now did not run every task once");
    require(readers || threads.size() == 1, "a batch run now without readers ran beside its caller");
    pool.run(0, [](size_t) { throw std::logic_error("a task of an empty batch ran"); });
  }
}

void retriesAFailedTaskOnce() {
  SsdReaders pool(4);
  // Task 7 fails once, tasks 3 and 11 every time.
  Runs runs(16);
  const auto task = [&](size_t index) {
    const int run = ++runs.counts[index];
    if (index == 3 || index == 11 || (index == 7 && run == 1))
      throw std::runtime_error("task " + std::to_string(index) + " failed");
  };
  std::string error;
  SsdReaders::Tally tally;
  try {
    pool.run(runs.counts.size(), task, &tally);
  } catch (const std::runtime_error &failure) {
    error = failure.what();
  }
  require(error == "task 3 failed", "a batch run now did not throw its first failure: " + error);
  for (size_t index = 0; index < runs.counts.size(); ++index)
    require(runs.counts[index].load() == (index == 3 || index == 7 || index == 11 ? 2 : 1),
            "a task ran another number of times than once, or twice after a failure");
  require(tally.retries == 3 && tally.failures == 2, "a batch counted other retries or failures");
  // runBatch() keeps the failures for its caller instead.
  for (std::atomic<int> &count : runs.counts) count = 0;
  const std::shared_ptr<SsdReaders::Batch> batch = pool.runBatch(runs.counts.size(), task);
  for (size_t index = 0; index < runs.counts.size(); ++index)
    require(batch->ended(index) && batch->failed(index) == (index == 3 || index == 11),
            "a batch run now did not keep exactly the failures of its tasks");
  require(batch->tally().retries == 3 && batch->tally().failures == 2,
          "a batch run now that keeps its failures counted other retries or failures");
}

void backgroundRunsOnItsReaders() {
  SsdReaders pool(8);
  Runs runs(64);
  std::atomic<int> working = 0, most = 0;
  std::atomic<int> ended = 0;
  const auto batch = pool.start(
      runs.counts.size(),
      [&](size_t index) {
        const int now = ++working;
        for (int seen = most.load(); now > seen && !most.compare_exchange_weak(seen, now);) {
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        ++runs.counts[index];
        --working;
      },
      2, [&](const SsdReaders::Batch &) { ++ended; });
  // In the background it runs on two readers at most, until it ends.
  while (!ended.load()) std::this_thread::yield();
  require(most.load() >= 1 && most.load() <= 2, "a background batch ran on other than its two readers");
  pool.finish(*batch);
  require(runs.eachOnce(0, runs.counts.size()) && batch->taken() == runs.counts.size(),
          "a finished background batch did not run every task once");
  require(ended.load() == 1, "a finished background batch did not report its end once");
  bool emptyEnded = false;
  const auto empty = pool.start(0, [](size_t) {}, 1, [&](const SsdReaders::Batch &) { emptyEnded = true; });
  require(emptyEnded && empty->taken() == 0, "an empty background batch did not end at once");
}

// finish() or stop() of a batch whose tasks all ran returns once its end
// returned, so the caller reads what the end wrote.
void finishWaitsForTheEnd() {
  for (const bool finish : {true, false}) {
    SsdReaders pool(2);
    std::atomic<bool> ran = false, endReturned = false;
    const auto batch = pool.start(
        1, [&](size_t) { ran = true; }, 1,
        [&](const SsdReaders::Batch &) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          endReturned = true;
        });
    while (!ran.load()) std::this_thread::yield();
    if (finish) pool.finish(*batch);
    else pool.stop(*batch);
    require(endReturned.load(), std::string(finish ? "finish()" : "stop()") + " returned before the batch's end did");
  }
}

void stopLeavesTheRestUnread() {
  SsdReaders pool(4);
  Runs runs(100);
  Gate gate;
  std::atomic<int> began = 0;
  const auto batch = pool.start(
      runs.counts.size(),
      [&](size_t index) {
        ++began;
        gate.wait();
        ++runs.counts[index];
        if (index == 1) throw std::runtime_error("task 1 failed");
      },
      3);
  while (began.load() < 3) std::this_thread::yield();
  std::thread opener([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    gate.open();
  });
  pool.stop(*batch);
  opener.join();
  const size_t taken = batch->taken();
  require(taken >= 3 && taken < runs.counts.size(), "a stopped batch reports other tasks taken");
  for (size_t index = 0; index < taken; ++index)
    require(runs.counts[index].load() == (index == 1 ? 2 : 1), "a task a reader took did not end before stop()");
  require(runs.noneFrom(taken), "a task of a stopped batch ran after stop()");
  require(batch->failed(1) && !batch->failed(0) && batch->failure(),
          "a stopped batch does not report the failure of a task it ran");
  // A stopped batch's end never runs; stopping it again changes nothing.
  pool.stop(*batch);
  require(batch->taken() == taken, "stopping a stopped batch changed it");
}

void haltLeavesTheTakenRunning() {
  SsdReaders pool(4);
  Runs runs(100);
  Gate gate;
  std::atomic<int> began = 0;
  const auto batch = pool.start(
      runs.counts.size(),
      [&](size_t index) {
        ++began;
        gate.wait();
        ++runs.counts[index];
      },
      3);
  while (began.load() < 3) std::this_thread::yield();
  pool.halt(*batch);
  const size_t taken = batch->taken();
  require(taken >= 3 && taken < runs.counts.size(), "a halted batch reports other tasks taken");
  for (size_t index = 0; index < taken; ++index)
    require(!batch->ended(index) && !runs.counts[index].load(), "a halted batch waited for a task a reader took");
  gate.open();
  for (size_t index = 0; index < taken; ++index)
    while (!batch->ended(index)) std::this_thread::yield();
  require(runs.eachOnce(0, taken), "a task a reader took did not end after halt()");
  pool.stop(*batch);
  require(batch->taken() == taken && runs.noneFrom(taken), "a halted batch ran more tasks, or stop() changed it");
}

void runNowPassesTheBackground() {
  SsdReaders pool(4);
  Runs background(400), now(32);
  std::atomic<int> backgroundStarted = 0;
  const auto batch = pool.start(
      background.counts.size(),
      [&](size_t index) {
        ++backgroundStarted;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++background.counts[index];
      },
      4);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  const int before = backgroundStarted.load();
  pool.run(now.counts.size(), [&](size_t index) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ++now.counts[index];
  });
  const int during = backgroundStarted.load() - before;
  require(now.eachOnce(0, now.counts.size()), "a batch run beside a background batch did not run every task once");
  // A reader starts at most the background task it was about to take as the
  // batch began and one as it ended; waiting behind the background, the batch
  // would see hundreds start.
  require(during <= 8, "a batch run now waited behind a background batch (" + std::to_string(during) +
                           " background tasks started while it ran)");
  pool.stop(*batch);
  require(background.eachOnce(0, batch->taken()) && background.noneFrom(batch->taken()),
          "a background batch ran a task other than once");
}

// A reader takes the background batches in the order they came: one
// reader runs the tasks of three batches started behind a busy one batch by
// batch.
void backgroundRunsTheOldestFirst() {
  SsdReaders pool(1);
  Gate gate;
  std::atomic<bool> began = false;
  const auto busy = pool.start(
      1,
      [&](size_t) {
        began = true;
        gate.wait();
      },
      1);
  while (!began.load()) std::this_thread::yield();
  std::mutex orderMutex;
  std::vector<size_t> order;
  std::atomic<size_t> ran = 0;
  std::vector<std::shared_ptr<SsdReaders::Batch>> batches;
  for (size_t batch = 0; batch < 3; ++batch)
    batches.push_back(pool.start(
        4,
        [&, batch](size_t index) {
          const std::lock_guard lock(orderMutex);
          order.push_back(batch * 4 + index);
          ++ran;
        },
        1));
  gate.open();
  while (ran.load() < 12) std::this_thread::yield();
  for (const auto &batch : batches) pool.stop(*batch);
  pool.stop(*busy);
  require(std::ranges::is_sorted(order), "a reader took a later background batch first");
}

// Writes yield to the pool exactly while a batch runs now (criticalBusy, as
// its probe says): not while it is idle or only reads in the background. The
// probe outlives the pool and then says nothing runs.
void criticalBusyWhileABatchRunsNow() {
  std::function<bool()> probe;
  {
    SsdReaders pool(2);
    probe = pool.criticalBusyProbe();
    require(!pool.criticalBusy() && !probe(), "an idle pool held writes back");
    Gate gate;
    std::atomic<bool> began = false;
    const auto ahead = pool.start(
        1,
        [&](size_t) {
          began = true;
          gate.wait();
        },
        1);
    while (!began.load()) std::this_thread::yield();
    require(!pool.criticalBusy() && !probe(), "a background batch held writes back");
    std::atomic<int> seen = 0;
    pool.run(8, [&](size_t) { seen += pool.criticalBusy() && probe(); });
    static_cast<void>(pool.runBatch(8, [&](size_t) { seen += pool.criticalBusy() && probe(); }));
    require(seen.load() == 16 && !pool.criticalBusy() && !probe(),
            "a batch run now did not hold writes back exactly while it ran");
    gate.open();
    pool.stop(*ahead);
  }
  require(!probe(), "a pool that is gone held writes back");
}

void threadsShareTheReaders() {
  SsdReaders pool(6);
  std::vector<std::thread> threads;
  std::atomic<int> wrong = 0;
  for (int thread = 0; thread < 4; ++thread)
    threads.emplace_back([&, thread] {
      for (int round = 0; round < 40; ++round) {
        Runs ahead(50), now(30);
        const auto batch = pool.start(ahead.counts.size(), [&](size_t index) { ++ahead.counts[index]; },
                                      1 + (thread + round) % 3);
        pool.run(now.counts.size(), [&](size_t index) { ++now.counts[index]; });
        if ((thread + round) % 2) pool.stop(*batch);
        else pool.finish(*batch);
        if (!now.eachOnce(0, now.counts.size()) || !ahead.eachOnce(0, batch->taken()) ||
            !ahead.noneFrom(batch->taken()))
          ++wrong;
      }
    });
  for (std::thread &thread : threads) thread.join();
  require(!wrong.load(), "batches of several threads ran a task other than once");
}

} // namespace

int main() {
  try {
    runsEveryTaskOnce();
    retriesAFailedTaskOnce();
    backgroundRunsOnItsReaders();
    finishWaitsForTheEnd();
    stopLeavesTheRestUnread();
    haltLeavesTheTakenRunning();
    runNowPassesTheBackground();
    backgroundRunsTheOldestFirst();
    criticalBusyWhileABatchRunsNow();
    threadsShareTheReaders();
  } catch (const std::exception &error) {
    std::cerr << "ssd_readers_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "ssd_readers_test: PASS\n";
  return 0;
}
