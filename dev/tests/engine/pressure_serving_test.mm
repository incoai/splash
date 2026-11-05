// The memory books under a scripted host on Metal (DESIGN §7.11, V.md §4.2,
// amend/V.md binding 9): a synthetic MoE target whose experts stream, started
// by ServingRuntime::start as serve starts it, its host a script the test
// moves, request frames into its native loop and the production control pass
// between commands. Bytes and counts only, so shader validation and the
// machine's speed do not change the verdicts. Minutes on the Studio or the M5
// Pro, under the machine's GPU lock.
//
// The target is the synthetic Qwen3.6-35B-A3B geometry (SyntheticModels.hpp
// moeConfig, 128 routed experts a layer; HiSparse off): its automatic store
// streams in several chunks under a budget a byte short of the whole model's
// minimum, below the one that holds every expert, which adds the KV of the
// first 64K tokens (a store --expert-memory sizes neither grows nor shrinks).
// The synthetic Flash-Next (flashNextConfig) does not start here: its hidden
// state of 256 values has no hyper-connection norm kernel
// (HyperConnection.cpp, 2560 only), which a start's warm-up runs.

#include "ProtocolPeer.hpp"
#include "SyntheticModels.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestRegistry.hpp"
#include "TestStderr.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryLedger.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/RuntimeResources.hpp"
#include "metal/DeviceCapabilities.hpp"
#include "model/Model.hpp"
#include "model/ModelSizing.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using test::require;

const char *gMetallib = nullptr;

// The Mac around the engine, as the test moves it.
struct Host final {
  uint64_t availableBytes = 64 * kGiB;
  MemoryPressure pressure = MemoryPressure::Normal;
};

// A start of the synthetic MoE target whose experts stream through an
// automatic store, the host read from `host`; its events read as they come.
struct Served final {
  Served(const std::filesystem::path &root, std::shared_ptr<Host> scripted) : host(std::move(scripted)) {
    RuntimeBootstrapConfig config;
    config.resources = test::moeConfig(gMetallib, root);
    uint64_t whole = 0;
    static_cast<void>(test::capturedStderr([&] {
      whole = RuntimeResources::create(config.resources, 0)->memoryPlan().breakdown().minimumRequiredBytes;
    }));
    config.resources.maximumMemoryBytes = whole - 1;
    config.resources.hostMemory = [host = host] {
      return std::optional<HostMemory>(HostMemory{host->availableBytes, host->availableBytes});
    };
    config.resources.memoryPressure = [host = host] { return host->pressure; };
    log = test::capturedStderr([&] {
      bootstrap = ServingRuntime::start(
          std::move(config),
          [this](std::span<const uint8_t> bytes) { output.insert(output.end(), bytes.begin(), bytes.end()); },
          [] { return 0.0; });
      bootstrap->announceReady();
    });
    require(bootstrap && resources().streamedExpertStats() && resources().streamedExpertStats()->capChunks > 1,
            "the target's experts did not stream in chunks");
  }

  [[nodiscard]] RuntimeResources &resources() { return bootstrap->resources(); }
  [[nodiscard]] NativeRuntime &loop() { return bootstrap->nativeLoop(); }
  [[nodiscard]] model::ExpertStoreStats store() { return *resources().streamedExpertStats(); }
  [[nodiscard]] MemoryGovernorSnapshot memory() { return resources().memoryGovernor().snapshot(); }
  [[nodiscard]] std::optional<uint64_t> expertsAccount() {
    for (const MemoryAccountSnapshot &account : memory().accounts)
      if (account.name == "experts")
        return account.currentBytes;
    return std::nullopt;
  }

  void submit(uint64_t id, uint32_t promptTokens, uint32_t outputTokens) {
    protocol::RequestFrame frame;
    frame.requestId = id;
    frame.priority = RequestPriority::Normal;
    frame.absoluteDeadlineUnixMicros = systemUnixMicros() + 600'000'000ULL;
    frame.remainingDeadlineMicros = 600'000'000ULL;
    frame.logicalMaxOutputTokens = outputTokens;
    frame.promptTokens.resize(promptTokens);
    for (uint32_t index = 0; index < promptTokens; ++index)
      frame.promptTokens[index] = 1 + (static_cast<uint32_t>(id) * 7919 + index * 37) % 997;
    submitted.insert(id);
    require(loop().receive(protocol::peer::serialize(frame)), "a request frame was refused");
  }
  // The events written since the last call: each request's end, a
  // failure's code.
  void drain() {
    for (const protocol::EngineEvent &event : reader.feed(output)) {
      if (const auto *done = std::get_if<protocol::DoneEvent>(&event))
        ended[done->requestId] = "done";
      else if (const auto *error = std::get_if<protocol::ErrorEvent>(&event))
        ended[error->requestId] = error->code;
    }
    output.clear();
  }
  // A tick, then the control pass at a command-free point, as the transport
  // runs them; then the books at that sample.
  void step(const std::function<void()> &sample) {
    static_cast<void>(loop().tick());
    if (!loop().commandInFlight()) {
      static_cast<void>(bootstrap->controlPass(host->pressure));
      ++controlPasses;
    }
    drain();
    if (sample)
      sample();
  }
  // Steps for `milliseconds` of wall time, or until done().
  void stepFor(double milliseconds, const std::function<void()> &sample, const std::function<bool()> &done = {}) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double, std::milli>(milliseconds);
    while (std::chrono::steady_clock::now() < end && !(done && done())) {
      step(sample);
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }

  std::shared_ptr<Host> host;
  std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
  uint64_t controlPasses = 0;
  std::vector<uint8_t> output;
  protocol::peer::EventReader reader;
  std::set<uint64_t> submitted;
  std::map<uint64_t, std::string> ended;
  std::string log;
  std::unique_ptr<ServingRuntime> bootstrap;
};

// The books at a sample (G9): the accounts sum to what the governor
// charges, and the experts account is what the store holds.
void requireBooks(Served &served, const std::string &when) {
  const MemoryGovernorSnapshot memory = served.memory();
  uint64_t sum = 0;
  for (const MemoryAccountSnapshot &account : memory.accounts)
    sum += account.currentBytes;
  require(sum == memory.chargedBytes, when + ": the accounts sum to " + std::to_string(sum) +
                                          " bytes, the governor charges " + std::to_string(memory.chargedBytes));
  const std::optional<uint64_t> experts = served.expertsAccount();
  require(experts && *experts == served.store().liveBytes,
          when + ": the experts account holds " + std::to_string(experts.value_or(0)) + " bytes, the store " +
              std::to_string(served.store().liveBytes));
}

// Two host episodes, normal -> warning -> critical -> normal, beside three
// growing prompts and a decoder. At every sample the books hold; within each
// level of the host the experts account moves one way only. Every request
// completes or ends for capacity; Metal stays healthy.
void testEpisodesKeepTheBooks() {
  const test::TemporaryDirectory root("splash-pressure-serving");
  auto host = std::make_shared<Host>();
  Served served(root.path(), host);
  const uint64_t physical = metal::probeDeviceCapabilities().physicalMemoryBytes;
  const uint64_t spareReserve = EngineMemoryPolicy::hostSpareReserveBytes(physical);
  served.submit(1, 64, 4096);
  uint64_t next = 10;
  // Prompts stay within the context the start serves, which refuses a
  // longer one as past its limits (73,721 tokens on the M6).
  const uint64_t lengths = served.resources().memoryPlan().maximumContextTokens() / 512 - 1;
  const auto feed = [&] {
    // Three prompts in flight, each 512 tokens longer than the last, back to
    // 512 past the context.
    while (served.submitted.size() - served.ended.size() < 4) {
      served.submit(next, static_cast<uint32_t>(512 * ((next - 10) % lengths + 1)), 16);
      ++next;
    }
  };
  for (uint32_t episode = 1; episode <= 2; ++episode) {
    const std::string at = "episode " + std::to_string(episode);
    struct Level final {
      MemoryPressure pressure;
      uint64_t available;
      const char *name;
    };
    // Warning beside a host short of its margin, critical, then a roomy host.
    for (const Level &level : {Level{MemoryPressure::Warning, spareReserve + kHostMarginBytes / 2, "warning"},
                               Level{MemoryPressure::Critical, spareReserve, "critical"},
                               Level{MemoryPressure::Normal, 64 * kGiB, "normal"}}) {
      host->pressure = level.pressure;
      host->availableBytes = level.available;
      std::optional<uint64_t> last;
      int direction = 0;
      served.stepFor(4000.0, [&] {
        feed();
        requireBooks(served, at + ", " + level.name);
        const uint64_t current = served.expertsAccount().value_or(0);
        if (last && current != *last) {
          const int moved = current > *last ? 1 : -1;
          require(!direction || moved == direction,
                  at + ", " + level.name + ": the experts account changed direction");
          direction = moved;
        }
        last = current;
      });
    }
  }
  served.stepFor(60'000.0, [] {}, [&] { return served.ended.size() == served.submitted.size(); });
  std::string unended;
  for (const uint64_t id : served.submitted)
    if (!served.ended.contains(id))
      unended += " " + std::to_string(id);
  require(unended.empty(), "requests that neither completed nor failed within 60 s of the episodes:" + unended);
  for (const auto &[id, code] : served.ended)
    require(code == "done" || code == "capacity_exhausted",
            "request " + std::to_string(id) + " ended with " + code);
  require(!served.loop().connectionMustClose(), "the engine failed under the host's episodes");
}

// A start short of host memory (amend/V.md A109): the host leaves less than
// the store's cap at start, the store starts below it; once the host frees,
// free room grows every unit it admits in one pass: the store is at its cap
// within two settled control passes and the host's settling.
void testShortStartRegrowsWhenTheHostFrees() {
  const test::TemporaryDirectory root("splash-pressure-short-start");
  auto host = std::make_shared<Host>();
  const uint64_t physical = metal::probeDeviceCapabilities().physicalMemoryBytes;
  host->availableBytes = EngineMemoryPolicy::hostSpareReserveBytes(physical) + kHostMarginBytes;
  Served served(root.path(), host);
  const model::ExpertStoreStats start = served.store();
  require(start.chunks < start.capChunks, "a start short of host memory held the store at its cap, " +
                                              std::to_string(start.chunks) + " chunks of " +
                                              std::to_string(start.capChunks));
  host->availableBytes = 64 * kGiB;
  served.stepFor(3 * kHostSettleMilliseconds, {}, [&] { return served.store().chunks == served.store().capChunks; });
  const model::ExpertStoreStats grown = served.store();
  require(grown.chunks == grown.capChunks, "the store stayed at " + std::to_string(grown.chunks) + " chunks of " +
                                               std::to_string(grown.capChunks) +
                                               " for two settled passes after the host freed");
  requireBooks(served, "after the host freed");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::cerr << "usage: pressure-serving <metallib> [--list] [--filter substring]\n";
      return EXIT_FAILURE;
    }
    gMetallib = argv[1];
    static constexpr test::TestCase kTests[] = {
        {"testEpisodesKeepTheBooks", testEpisodesKeepTheBooks},
        {"testShortStartRegrowsWhenTheHostFrees", testShortStartRegrowsWhenTheHostFrees},
    };
    const test::TestGroup groups[] = {{"pressure", kTests}};
    argv[1] = argv[0];
    return test::runTests("pressure-serving", groups, argc - 1, argv + 1);
  }
}
