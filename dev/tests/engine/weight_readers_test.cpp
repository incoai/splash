// The readers of streamed weights (model::WeightReaders) run every task of a
// batch once: a batch that runs now on the caller and every reader, a
// background batch on at most its readers; a failed task is tried once more,
// and a second failure is the batch's (a batch that runs now throws it once
// every task ended); stop() leaves the tasks no reader took unrun and
// returns once those taken ended, finish() runs the rest now and returns
// once the batch's end has; a batch that runs now never waits behind a
// background batch's queue; and several threads run, start, stop and finish
// batches at once. Built under TSan too.
#include "TestChecks.hpp"
#include "model/WeightReaders.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace splash::model;
using splash::test::require;

namespace {

using Clock = std::chrono::steady_clock;

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
    WeightReaders pool(readers);
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
  WeightReaders pool(4);
  // Task 7 fails once, tasks 3 and 11 every time.
  Runs runs(16);
  const auto task = [&](size_t index) {
    const int run = ++runs.counts[index];
    if (index == 3 || index == 11 || (index == 7 && run == 1))
      throw std::runtime_error("task " + std::to_string(index) + " failed");
  };
  std::string error;
  WeightReaders::Tally tally;
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
}

void backgroundRunsOnItsReaders() {
  WeightReaders pool(8);
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
      2, [&](const WeightReaders::Batch &) { ++ended; });
  pool.finish(*batch);
  require(runs.eachOnce(0, runs.counts.size()) && batch->taken() == runs.counts.size(),
          "a finished background batch did not run every task once");
  require(ended.load() == 1, "a finished background batch did not report its end once");
  // finish() runs the rest on every reader: only before it do two at most.
  require(most.load() >= 1, "no task of a background batch ran");
  bool emptyEnded = false;
  const auto empty = pool.start(0, [](size_t) {}, 1, [&](const WeightReaders::Batch &) { emptyEnded = true; });
  require(emptyEnded && empty->taken() == 0, "an empty background batch did not end at once");
}

// finish() of a batch whose tasks all ran returns once its end returned, so
// the caller reads what the end wrote.
void finishWaitsForTheEnd() {
  WeightReaders pool(2);
  std::atomic<bool> ran = false, endReturned = false;
  const auto batch = pool.start(
      1, [&](size_t) { ran = true; }, 1,
      [&](const WeightReaders::Batch &) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        endReturned = true;
      });
  while (!ran.load()) std::this_thread::yield();
  pool.finish(*batch);
  require(endReturned.load(), "finish() returned before the batch's end did");
}

void stopLeavesTheRestUnread() {
  WeightReaders pool(4);
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

void runNowPassesTheBackground() {
  WeightReaders pool(4);
  Runs background(400), now(32);
  const auto batch = pool.start(
      background.counts.size(),
      [&](size_t index) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++background.counts[index];
      },
      4);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  const auto started = Clock::now();
  pool.run(now.counts.size(), [&](size_t index) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ++now.counts[index];
  });
  const auto waited = Clock::now() - started;
  require(now.eachOnce(0, now.counts.size()), "a batch run beside a background batch did not run every task once");
  // Without passing, it would wait for a second of background tasks.
  require(waited < std::chrono::milliseconds(300), "a batch run now waited behind a background batch");
  pool.stop(*batch);
  require(background.eachOnce(0, batch->taken()) && background.noneFrom(batch->taken()),
          "a background batch ran a task other than once");
}

void threadsShareTheReaders() {
  WeightReaders pool(6);
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
    runNowPassesTheBackground();
    threadsShareTheReaders();
  } catch (const std::exception &error) {
    std::cerr << "weight_readers_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "weight_readers_test: PASS\n";
  return 0;
}
