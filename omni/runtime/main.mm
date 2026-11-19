// The Splash Omni engine process (docs/omni-design.md):
//   splash serve-omni MODEL_DIRECTORY [--lanes N] [--max-memory BYTES] [--idle-release SECONDS|off] [--disable-ane]
//                     [--draft-model DIRECTORY] [--draft-tokens K] [--speculative-lanes N]
//   splash omni-check MODEL_DIRECTORY
//   splash device-check

#include "StderrLine.hpp"
#include "metal/MetalBackend.hpp"
#include "omni/OmniProtocol.hpp"
#include "omni/OmniServer.hpp"

#import <Foundation/Foundation.h>
#include <mach-o/dyld.h>
#include <sysexits.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash {
namespace {

class UsageError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void printUsage(std::string_view executable) {
  const std::string command(executable);
  writeStderrLine("usage: " + command +
                  " serve-omni MODEL_DIRECTORY [--lanes N] [--max-memory BYTES] [--idle-release SECONDS|off] [--disable-ane]"
                  " [--draft-model DIRECTORY] [--draft-tokens K] [--speculative-lanes N]");
  writeStderrLine("       " + command + " omni-check MODEL_DIRECTORY");
  writeStderrLine("       " + command + " device-check");
}

std::filesystem::path executablePath() {
  uint32_t size = PATH_MAX;
  std::vector<char> buffer(size);
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
    buffer.resize(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) throw std::runtime_error("could not resolve executable path");
  }
  std::error_code error;
  std::filesystem::path path = std::filesystem::canonical(buffer.data(), error);
  if (error) throw std::runtime_error("could not canonicalize executable path: " + error.message());
  return path;
}

int checkDevice() {
  const auto message = metal::probeDeviceCapabilities().validationMessage();
  if (!message) return 0;
  writeStderrLine("error: " + *message);
  return EX_SOFTWARE;
}

// Writes all of a frame to stdout; a reader that has gone loses it.
void writeFrame(const std::vector<uint8_t> &frame) {
  size_t written = 0;
  while (written < frame.size()) {
    const ssize_t n = ::write(STDOUT_FILENO, frame.data() + written, frame.size() - written);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return;
    written += static_cast<size_t>(n);
  }
}

std::atomic<bool> shutdownRequested{false};
void onSignal(int) { shutdownRequested = true; }

int serveOmni(int argc, char **argv) {
  omni::ServerOptions options;
  if (argc < 3) throw UsageError("serve-omni takes the model directory");
  options.modelRoot = argv[2];
  options.metallib = executablePath().parent_path() / "splash.metallib";
  for (int i = 3; i < argc; ++i) {
    const std::string_view flag = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw UsageError(std::string(flag) + " takes a value");
      return argv[++i];
    };
    if (flag == "--disable-ane") options.disableAne = true;
    else if (flag == "--lanes") options.lanes = static_cast<uint32_t>(std::stoul(value()));
    else if (flag == "--max-memory") options.maxMemoryBytes = std::stoull(value());
    else if (flag == "--draft-model") options.draftModel = value();
    else if (flag == "--draft-tokens") options.draftTokens = static_cast<uint32_t>(std::stoul(value()));
    else if (flag == "--speculative-lanes") options.speculativeLanes = static_cast<uint32_t>(std::stoul(value()));
    else if (flag == "--idle-release") {
      const std::string v = value();
      options.idleReleaseSeconds = v == "off" ? 0.0 : std::stod(v);
    } else throw UsageError("unknown serve-omni option " + std::string(flag));
  }
  std::optional<omni::LoadedEngine> loaded;
  try {
    if (const auto message = metal::probeDeviceCapabilities().validationMessage()) throw std::runtime_error(*message);
    loaded.emplace(omni::loadEngine(options));
  } catch (const std::exception &error) {
    // A start that fails reports why on stderr, as any failure does, and then
    // in place of its ReadyEvent, as an engine failure the server ends the
    // start with: otherwise the server sees only the protocol end.
    writeStderrLine(std::string("error: ") + error.what());
    writeFrame(omni::protocol::errorFrame(omni::protocol::FailureClass::EngineUnhealthy, false, 0, "start_failed",
                                          error.what()));
    return EX_SOFTWARE;
  }
  omni::LoadedEngine &engine = *loaded;
  std::signal(SIGTERM, onSignal);
  std::signal(SIGINT, onSignal);
  std::signal(SIGHUP, onSignal);
  std::signal(SIGPIPE, SIG_IGN);
  writeStderrLine("Ready");
  return omni::serve(engine, STDIN_FILENO, STDOUT_FILENO, shutdownRequested);
}

} // namespace
} // namespace splash

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "device-check") return splash::checkDevice();
      if (argc == 3 && std::string_view(argv[1]) == "omni-check") {
        std::printf("%s\n", splash::omni::checkModelJson(argv[2]).c_str());
        return 0;
      }
      if (argc > 1 && std::string_view(argv[1]) == "serve-omni") return splash::serveOmni(argc, argv);
      throw splash::UsageError("unknown command");
    } catch (const splash::UsageError &error) {
      splash::writeStderrLine(std::string("error: ") + error.what());
      splash::printUsage(argc > 0 ? argv[0] : "splash");
      return EX_USAGE;
    } catch (const std::system_error &error) {
      splash::writeStderrLine(std::string("error: I/O failed: ") + error.what());
      return EX_IOERR;
    } catch (const std::exception &error) {
      splash::writeStderrLine(std::string("error: ") + error.what());
      return EX_SOFTWARE;
    }
  }
}
