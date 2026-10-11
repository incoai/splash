// Processes that open one GpuShare file see the highest priority another
// decodes at, never their own. A process's lock goes when it leaves, ends or
// dies, a file removed meanwhile is opened again, and a failure ends sharing
// with one warning.
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestStderr.hpp"
#include "engine/GpuShare.hpp"

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

using splash::engine::GpuShare;
using splash::engine::RequestPriority;
using splash::test::require;

namespace {

// One byte per priority on the pipes between the processes, '-' for none.
char toByte(std::optional<RequestPriority> priority) {
  return priority ? static_cast<char>('0' + static_cast<int>(*priority)) : '-';
}

std::optional<RequestPriority> fromByte(char value) {
  if (value == '-')
    return std::nullopt;
  return static_cast<RequestPriority>(value - '0');
}

void writeByte(int descriptor, char value) {
  if (write(descriptor, &value, 1) != 1)
    throw std::runtime_error("write to the other process failed");
}

char readByte(int descriptor) {
  char value = 0;
  if (read(descriptor, &value, 1) != 1)
    throw std::runtime_error("read from the other process failed");
  return value;
}

// Another process with its own GpuShare of the file: it decodes at each
// priority it is sent and answers with the priority it sees. Each process
// keeps only its own ends of the pipes, so the child ends when the parent's
// close, and a parent that fails does not wait for it.
class Child final {
public:
  explicit Child(const std::filesystem::path &path) {
    int commands[2], answers[2];
    if (pipe(commands) || pipe(answers))
      throw std::runtime_error("pipe failed");
    pid_ = fork();
    if (pid_ < 0)
      throw std::runtime_error("fork failed");
    if (!pid_) {
      close(commands[1]);
      close(answers[0]);
      int status = 0;
      try {
        GpuShare share(path);
        for (;;)
          writeByte(answers[1], toByte(share.exchange(fromByte(readByte(commands[0])))));
      } catch (const std::exception &) {
        // The parent closed the pipe.
      }
      _exit(status);
    }
    close(commands[0]);
    close(answers[1]);
    commands_ = commands[1];
    answers_ = answers[0];
  }
  ~Child() { end(SIGTERM); }
  Child(const Child &) = delete;
  Child &operator=(const Child &) = delete;

  // What the child sees once it decodes at `priority`.
  std::optional<RequestPriority> decodes(std::optional<RequestPriority> priority) {
    writeByte(commands_, toByte(priority));
    return fromByte(readByte(answers_));
  }

  // Ends the child with `signal`, whatever it holds, and waits for it.
  void end(int signal) {
    if (pid_ <= 0)
      return;
    close(commands_);
    close(answers_);
    kill(pid_, signal);
    int status = 0;
    waitpid(pid_, &status, 0);
    pid_ = -1;
  }

private:
  pid_t pid_ = -1;
  int commands_ = -1;
  int answers_ = -1;
};

void testServersSeeEachOther(const std::filesystem::path &path) {
  Child child(path);
  GpuShare share(path);
  require(share.exchange(std::nullopt) == std::nullopt && child.decodes(std::nullopt) == std::nullopt,
          "a process saw a decoder before any decoded");
  require(child.decodes(RequestPriority::Normal) == std::nullopt &&
              share.exchange(std::nullopt) == RequestPriority::Normal,
          "a process saw its own lock, or not the other's");
  require(share.exchange(RequestPriority::Foreground) == RequestPriority::Normal &&
              child.decodes(RequestPriority::Normal) == RequestPriority::Foreground,
          "a process that decodes saw its own priority instead of the other's");
  require(share.exchange(RequestPriority::Normal) == RequestPriority::Normal &&
              child.decodes(RequestPriority::Normal) == RequestPriority::Normal,
          "processes that decode at one priority did not see each other");
  require(child.decodes(RequestPriority::Background) == RequestPriority::Normal &&
              share.exchange(std::nullopt) == RequestPriority::Background &&
              child.decodes(RequestPriority::Background) == std::nullopt,
          "a changed priority did not replace the one held before");
  require(child.decodes(std::nullopt) == std::nullopt && share.exchange(std::nullopt) == std::nullopt,
          "a process that stopped decoding still held its lock");
}

// The kernel drops the lock of a process that leaves, ends or is killed.
void testLocksGoWithTheirServer(const std::filesystem::path &path) {
  {
    Child child(path);
    GpuShare share(path);
    require(share.exchange(RequestPriority::Normal) == std::nullopt &&
                child.decodes(std::nullopt) == RequestPriority::Normal,
            "the child did not see the parent decode");
    share.leave();
    require(child.decodes(std::nullopt) == std::nullopt &&
                share.exchange(RequestPriority::Normal) == std::nullopt &&
                child.decodes(std::nullopt) == std::nullopt,
            "a process that left kept or took a lock");
  }
  GpuShare share(path);
  for (const int signal : {SIGTERM, SIGKILL}) {
    Child child(path);
    require(child.decodes(RequestPriority::Foreground) == std::nullopt &&
                share.exchange(std::nullopt) == RequestPriority::Foreground,
            "the child's lock did not reach the parent");
    child.end(signal);
    require(share.exchange(std::nullopt) == std::nullopt, "an ended process left its lock behind");
  }
}

// A server that starts after the file was removed opens a new one, and the
// servers that run already open it too.
void testRemovedFileIsOpenedAgain(const std::filesystem::path &path) {
  GpuShare share(path);
  Child child(path);
  require(share.exchange(RequestPriority::Normal) == std::nullopt &&
              child.decodes(RequestPriority::Background) == RequestPriority::Normal,
          "the processes did not see each other before the file was removed");
  std::filesystem::remove_all(path.parent_path());
  Child later(path);
  require(later.decodes(std::nullopt) == std::nullopt,
          "a lock on the removed file reached a new process");
  require(share.exchange(RequestPriority::Normal) == std::nullopt &&
              later.decodes(std::nullopt) == RequestPriority::Normal &&
              child.decodes(RequestPriority::Background) == RequestPriority::Normal &&
              share.exchange(RequestPriority::Normal) == RequestPriority::Background,
          "processes that ran before the file was removed did not open the new one");
}

// A file that cannot be opened again ends sharing with one warning: the
// server drops its lock and sees no other server from then on.
void testFailureEndsSharing(const std::filesystem::path &path) {
  GpuShare share(path);
  require(share.exchange(RequestPriority::Normal) == std::nullopt, "fixture did not decode");
  std::filesystem::remove_all(path.parent_path());
  splash::test::writeFile(path.parent_path(), "not a directory");
  std::optional<RequestPriority> seen = RequestPriority::Foreground;
  const std::string warnings = splash::test::capturedStderr([&] {
    seen = share.exchange(RequestPriority::Normal);
    seen = seen ? seen : share.exchange(RequestPriority::Normal);
  });
  const size_t warning = warnings.find("Warning · --share-gpu: ");
  require(!seen && warning != std::string::npos &&
              warnings.find(path.parent_path().string()) != std::string::npos &&
              warnings.find("Warning", warning + 1) == std::string::npos,
          "a failed exchange did not end sharing with one warning: " + warnings);
  std::filesystem::remove(path.parent_path());
  {
    Child child(path);
    require(child.decodes(RequestPriority::Normal) == std::nullopt &&
                share.exchange(RequestPriority::Normal) == std::nullopt &&
                child.decodes(RequestPriority::Normal) == std::nullopt,
            "a server that stopped sharing still exchanged");
  }
  // A server that cannot make the file's directory, or open the file,
  // refuses to start, naming it.
  std::filesystem::remove_all(path.parent_path());
  splash::test::writeFile(path.parent_path(), "not a directory");
  splash::test::rejects([&] { GpuShare unusable(path); }, path.parent_path().string(),
                        "a directory that cannot be made did not refuse with its path");
  std::filesystem::remove(path.parent_path());
  std::filesystem::create_directories(path);
  splash::test::rejects([&] { GpuShare unusable(path); }, "cannot open " + path.string(),
                        "a file that cannot be opened did not refuse with its path");
  std::filesystem::remove_all(path.parent_path());
}

} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory directory("gpu-share");
    // The directory the file goes in is made with it.
    const std::filesystem::path path = directory.path() / "splash-gpu" / "decoding";
    testServersSeeEachOther(path);
    testLocksGoWithTheirServer(path);
    testRemovedFileIsOpenedAgain(path);
    testFailureEndsSharing(path);
    std::cout << "GPU share tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "GPU share tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
