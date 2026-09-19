#include "model/SlotFile.hpp"

#include <algorithm>
#include <future>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using splash::model::DiskBudget;
using splash::model::SlotFile;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Call>
static bool throws(Call call) {
  try { call(); } catch (const Exception &) { return true; }
  return false;
}

static void testFailedWriteStopsWriting() {
  const pid_t child = fork();
  require(child >= 0, "failed to isolate file limit test");
  if (!child) {
    int result = 0;
    try {
      constexpr size_t size = SlotFile::kAlignmentBytes;
      SlotFile file(size, 2 * size);
      auto complete = file.acquire();
      auto partial = file.acquire();
      std::vector<std::byte> source(size, std::byte{1}), output(size);
      require(file.write(complete, {source}, {})->wait(), "initial write failed");
      struct rlimit original;
      require(getrlimit(RLIMIT_FSIZE, &original) == 0, "file limit unavailable");
      auto limited = original;
      limited.rlim_cur = size / 2;
      signal(SIGXFSZ, SIG_IGN);
      require(setrlimit(RLIMIT_FSIZE, &limited) == 0, "file limit could not be set");
      std::fill(source.begin(), source.end(), std::byte{2});
      require(!file.write(partial, {source}, {})->wait(), "partial write reported success");
      require(!file.read(partial, {output}, {})->wait(), "partially written slot was readable");
      require(setrlimit(RLIMIT_FSIZE, &original) == 0, "file limit restore failed");
      require(!file.writable() &&
                  throws<std::logic_error>([&] { static_cast<void>(file.write(partial, {source}, {})); }),
              "storage failure did not stop further writes");
      require(file.read(complete, {output}, {})->wait() && output.front() == std::byte{1},
              "complete slot became unreadable after a storage failure");
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      result = 1;
    }
    _exit(result);
  }
  int status = 0;
  require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "file failure/integrity test failed");
}

int main() {
  try {
    testFailedWriteStopsWriting();
    constexpr size_t size = 4 * SlotFile::kAlignmentBytes;
    SlotFile file(size, size * 2 + 1);
    require(file.slotBytes() == size && file.capacityBytes() == size * 2 + 1 &&
                file.usedBytes() == 0,
            "file did not report its quota");
    auto first = file.acquire();
    auto second = file.acquire();
    require(first && second && file.usedBytes() == size * 2 && !file.acquire(),
            "a partial slot of the quota was granted");
    require(first && second && !file.acquire(), "disk quota exceeded");
    std::vector<std::byte> source(size, std::byte{0xa5}), restored(size);
    auto write = file.write(first, {std::span(source).first(128), std::span(source).subspan(128)}, {});
    require(write->wait(), "slot write failed");
    require(file.read(first, {restored}, {})->wait() && restored == source,
            "slot did not roundtrip across IO spans");
    require(!file.read(second, {restored}, {})->wait(), "unwritten slot was readable");
    require(file.idle(), "a file with every operation finished called itself busy");

    // Park the worker in a completion so the queue behind it is deterministic:
    // a queued write and a cancelled read behind it.
    std::promise<void> reached, release;
    auto released = release.get_future().share();
    auto hold = file.read(first, {restored}, [&] { reached.set_value(); released.wait(); });
    reached.get_future().wait();
    std::vector<std::byte> later(size, std::byte{0x5a});
    write = file.write(second, {later}, {});
    std::vector<std::byte> untouched(size, std::byte{0});
    auto cancelled = file.read(second, {untouched}, {});
    cancelled->cancel();
    require(!file.idle(), "a file with queued work called itself idle");
    release.set_value();
    require(hold->wait() && write->wait() && !cancelled->wait(), "queued operations misreported");
    require(file.idle(), "the queue drained but the file still held work");
    require(untouched.front() == std::byte{0} && untouched.back() == std::byte{0},
            "cancelled queued read touched destination");
    require(file.read(second, {restored}, {})->wait() && restored == later,
            "queued write did not land in order");

    // A completion is the caller's wake-up: one that throws is not the
    // worker's to die of.
    require(file.write(second, {later}, [] { throw std::runtime_error("completion"); })->wait(),
            "write with a throwing completion failed");
    require(file.read(second, {restored}, {})->wait() && restored == later,
            "the worker did not survive a throwing completion");
    first.reset();
    second.reset();
    auto reused = file.acquire();
    second = file.acquire();
    require(reused && second && !file.acquire(), "released quota was not reusable");
    require(throws<std::invalid_argument>([&] { SlotFile(size, size - 1); }),
            "quota below one slot was accepted");
    require(throws<std::invalid_argument>([&] { SlotFile(size + 1, 4 * size); }),
            "unaligned slot size was accepted");
    require(throws<std::invalid_argument>(
                [&] { static_cast<void>(file.read(reused, {std::span(restored).first(1)}, {})); }),
            "invalid slot shape was accepted");
    {
      // Two files of different slot sizes draw on one budget.
      auto budget = std::make_shared<DiskBudget>(4 * size);
      SlotFile small(size, budget);
      SlotFile large(2 * size, budget);
      require(small.capacityBytes() == 4 * size && large.capacityBytes() == 4 * size &&
                  small.usedBytes() == 0,
              "files did not report the shared budget");
      auto one = large.acquire();
      auto two = small.acquire();
      require(one && two && small.usedBytes() == 3 * size && !large.acquire(),
              "the shared budget did not bound the second file");
      auto three = small.acquire();
      require(three && !small.acquire() && budget->usedBytes() == 4 * size,
              "the last slot of the budget was not granted exactly once");
      one.reset();
      require(budget->usedBytes() == 2 * size && large.acquire() && !small.acquire(),
              "a released slot did not return its bytes to the budget");
      require(throws<std::invalid_argument>(
                  [&] { SlotFile(8 * size, budget); }),
              "a file whose slot exceeds the shared budget was accepted");
    }
    std::cout << "Slot file tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
